"""Glue between frame input, the network and VRCFT output. Used by GUI and CLI."""

import logging
import os
import threading
import time

import numpy as np
import torch

from . import trainer
from .config import Config
from .datasets import Recorder, convert_legacy_pickles
from .frames import FrameHub, ProxyClient, RateMeter, SRanipalReceiver
from .model import BuddyNet, pick_device
from .vrcft import VRCFTServer, encode_params

log = logging.getLogger(__name__)


def normalize(raw, max_power):
    """Map a raw network output to VRCFT's -1..1 range (same formula as before)."""
    return float(np.clip((raw / max_power) * 2.0 - 1.0, -1.0, 1.0))


class Engine:
    def __init__(self, cfg: Config, config_path=None):
        self.cfg = cfg
        self.config_path = config_path
        self.device = pick_device()
        self.hub = FrameHub()
        self.source = None
        self.vrcft = None
        self.model = None
        self.model_dirty = False  # trained but not saved

        self.busy = None  # "recording" / "training" / None
        self._recorder = None
        self._train_stop = threading.Event()

        self._infer_thread = None
        self._infer_stop = threading.Event()
        self._fastcal_request = threading.Event()
        self.fastcal_state = None  # text shown while fastcal runs
        self.infer_rate = RateMeter()
        self.infer_latency_ms = 0.0
        self.last_raw = None  # np.ndarray of raw outputs
        self.last_out = {}  # class index -> normalized value sent to VRCFT
        self._smoothed = {}

    # ------------------------------------------------------------ lifecycle
    def start(self):
        cfg = self.cfg
        if cfg.source == "proxy":
            self.source = ProxyClient(self.hub, cfg.bind_host, cfg.proxy_port).start()
        else:
            self.source = SRanipalReceiver(self.hub, cfg.bind_host, cfg.face_port, cfg.eye_port,
                                           swapped=cfg.swapped, mode=cfg.input_mode,
                                           stall_timeout=cfg.stall_timeout).start()
        self.vrcft = VRCFTServer(cfg.bind_host, cfg.vrcft_port).start()
        log.info("Compute device: %s", self.device)
        return self

    def stop(self):
        self.stop_inference()
        self._train_stop.set()
        if self._recorder:
            self._recorder.cancel()
        for part in (self.source, self.vrcft):
            if part is not None:
                part.stop()

    def save_config(self):
        if self.config_path:
            self.cfg.save(self.config_path)

    # ------------------------------------------------------------ status
    def mode_hint(self, streams):
        """Suggest an input mode when the connected trackers don't match the selected one.

        Returns "both", "single" (only one tracker streams; which one can't be told
        from the port, so the user picks face/eye using the camera preview) or None."""
        mode = self.cfg.input_mode
        ok = {role: streams.get(role, {}).get("state") == "ok" for role in ("eye", "face")}
        if not any(ok.values()) or self.cfg.source == "proxy":
            return None
        if mode == "both" and not all(ok.values()):
            return "single"
        if mode != "both" and all(ok.values()):
            return "both"
        return None

    def status(self):
        streams = self.source.snapshot(self.cfg.stall_timeout) if self.source else {}
        return {
            "streams": streams,
            "input_mode": self.cfg.input_mode,
            "mode_hint": self.mode_hint(streams),
            "sample_fps": self.hub.sample_rate.rate(),
            "vrcft": self.vrcft.snapshot() if self.vrcft else {"connected": False},
            "inferring": self.inferring,
            "infer_fps": self.infer_rate.rate(),
            "latency_ms": self.infer_latency_ms,
            "busy": self.busy,
            "fastcal": self.fastcal_state,
            "device": str(self.device),
            "model_loaded": self.model is not None,
            "model_dirty": self.model_dirty,
            "swapped": self.cfg.swapped,
        }

    def set_input_mode(self, mode):
        if mode == self.cfg.input_mode:
            return
        if self.busy:
            raise RuntimeError("Wait until %s has finished" % self.busy)
        self.stop_inference()
        self.cfg.input_mode = mode
        if self.source:
            self.source.set_mode(mode)
        if self.model is not None and self.model.input_mode != mode:
            log.info("Unloaded the model: it was trained for input '%s'. Load or train a '%s' model.",
                     self.model.input_mode, mode)
            self.model = None
            self.model_dirty = False
        self.save_config()
        log.info("Input mode: %s", mode)

    def set_swapped(self, value):
        self.cfg.swapped = bool(value)
        if self.source:
            self.source.set_swapped(value)
        self.save_config()

    # ------------------------------------------------------------ model
    def model_path(self):
        return self.cfg.model_path

    def load_model(self, path=None):
        path = path or self.cfg.model_path
        self.model = BuddyNet.load(path, self.device, expected_outputs=self.cfg.num_classes,
                                   expected_mode=self.cfg.input_mode).eval()
        self.model_dirty = False
        log.info("Loaded model %s", path)
        return self.model

    def save_model(self, path=None):
        if self.model is None:
            raise RuntimeError("No model to save - train or load one first")
        path = path or self.cfg.model_path
        self.model.save(path, [c.name for c in self.cfg.classes])
        self.model_dirty = False
        log.info("Saved model to %s", path)

    def ensure_model(self):
        if self.model is None:
            self.load_model()
        return self.model

    # ------------------------------------------------------------ recording
    def record(self, name, on_progress=None, on_done=None, frames=None):
        if self.busy:
            raise RuntimeError("Already %s" % self.busy)
        name = name.strip()
        if not name:
            raise ValueError("Enter a dataset name")
        filename = name if name.endswith(".mmap") else name + "-em.mmap"
        path = self.cfg.dataset_path(filename)
        self.busy = "recording"

        def done(ok, message):
            self.busy = None
            self._recorder = None
            log.info(message)
            if on_done:
                on_done(ok, message, filename)

        self._recorder = Recorder(self.hub, path, frames or self.cfg.record_frames, self.cfg.record_countdown,
                                  on_progress, done).start()
        return filename

    def cancel_record(self):
        if self._recorder:
            self._recorder.cancel()

    def convert_pickles(self):
        return convert_legacy_pickles(self.cfg.dataset_folder)

    # ------------------------------------------------------------ training
    def train_async(self, on_progress=None, on_done=None, resume=False):
        if self.busy:
            raise RuntimeError("Already %s" % self.busy)
        problems = self.cfg.validate()
        if problems:
            raise ValueError("\n".join(problems))
        self.stop_inference()
        self.busy = "training"
        self._train_stop.clear()

        def run():
            ok, message, history = False, "", []
            try:
                init = None
                if resume:
                    init = self.model if self.model is not None else self.load_model()
                    if init.input_mode != self.cfg.input_mode:
                        raise ValueError("The current model was trained for input '%s'" % init.input_mode)
                model, history = trainer.train(self.cfg, on_progress=on_progress, stop_event=self._train_stop,
                                               log_fn=log.info, init_model=init, device=self.device)
                self.model = model
                self.model_dirty = True
                ok, message = True, "Training finished (final loss %.6f). Don't forget to save." % history[-1]
            except trainer.TrainingCancelled:
                message = "Training cancelled"
            except Exception as e:
                log.exception("training failed")
                message = "Training failed: %s" % e
            finally:
                self.busy = None
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
            log.info(message)
            if on_done:
                on_done(ok, message, history)

        threading.Thread(target=run, daemon=True, name="trainer").start()

    def cancel_training(self):
        self._train_stop.set()

    # ------------------------------------------------------------ inference
    @property
    def inferring(self):
        return self._infer_thread is not None and self._infer_thread.is_alive()

    def start_inference(self):
        if self.inferring:
            return
        if self.busy == "training":
            raise RuntimeError("Wait for training to finish")
        self.ensure_model()
        self._infer_stop.clear()
        self._smoothed = {}
        self._infer_thread = threading.Thread(target=self._infer_loop, daemon=True, name="inference")
        self._infer_thread.start()

    def stop_inference(self):
        self._infer_stop.set()
        self._fastcal_request.clear()
        if self._infer_thread is not None:
            self._infer_thread.join(timeout=3)
        self._infer_thread = None

    def request_fastcal(self):
        if not self.inferring:
            self.start_inference()
        self._fastcal_request.set()

    def _predictor(self):
        model = self.model.eval()
        device = self.device
        pin = device.type == "cuda"
        mode = model.input_mode
        host = torch.empty((1, 128 if mode == "both" else 64, 20, 20), dtype=torch.float32, pin_memory=pin)
        host_np = host.numpy()
        # where each tracker's features go in the input tensor (None = not used)
        eye_at = 0 if mode in ("both", "eye") else None
        face_at = 64 if mode == "both" else (0 if mode == "face" else None)

        @torch.inference_mode()
        def predict(sample):
            eye, face = sample
            for data, at in ((eye, eye_at), (face, face_at)):
                if at is None:
                    continue
                if data is None:  # a needed tracker is missing from this sample
                    return None
                host_np[0, at:at + 64] = np.frombuffer(data, dtype=np.float32).reshape(64, 20, 20)
            return model(host.to(device, non_blocking=pin))[0].float().cpu().numpy()

        return predict

    def _infer_loop(self):
        predict = self._predictor()
        seq = 0
        min_interval = 1.0 / self.cfg.max_send_rate if self.cfg.max_send_rate > 0 else 0.0
        last_send = 0.0
        waiting_logged = False
        log.info("Inference started")
        while not self._infer_stop.is_set():
            if self._fastcal_request.is_set():
                self._fastcal_request.clear()
                try:
                    self._run_fastcal(predict)
                except Exception:
                    log.exception("fastcal failed")
                self.fastcal_state = None
                continue
            # Only run the network on *new* frames (the old loop re-ran on the
            # same frame every 10ms and slept even when a new frame was ready).
            seq, sample = self.hub.wait_next(seq, timeout=0.5)
            if sample is None:
                if not waiting_logged:
                    log.info("Waiting for tracker data...")
                    waiting_logged = True
                continue
            waiting_logged = False
            t0 = time.perf_counter()
            raw = predict(sample)
            if raw is None:
                continue
            self.infer_latency_ms = (time.perf_counter() - t0) * 1000
            self.infer_rate.tick()
            self.last_raw = raw

            alpha = self.cfg.smoothing
            out, pairs = {}, []
            for idx, shape, max_power in self.cfg.targets():
                v = normalize(raw[idx], max_power)
                if 0.0 < alpha < 1.0:
                    prev = self._smoothed.get(idx, v)
                    v = prev * alpha + v * (1.0 - alpha)
                    self._smoothed[idx] = v
                out[idx] = v
                pairs.append((shape, v))
            self.last_out = out
            now = time.monotonic()
            if pairs and now - last_send >= min_interval:
                self.vrcft.send_params(pairs)
                last_send = now
        log.info("Inference stopped")

    def _run_fastcal(self, predict):
        """Puppet each target shape on the avatar and record the network output
        while the user copies it; the average becomes that class's max_power."""
        targets = self.cfg.targets()
        if not targets:
            log.warning("No classes have a target shape; nothing to calibrate")
            return
        if not self.vrcft.connected:
            log.warning("VRCFT is not connected - the avatar can't be puppeted, calibrating blind")

        def stopped():
            return self._infer_stop.is_set()

        for t in range(10, 0, -1):
            self.fastcal_state = "Starting in %d s - follow the avatar" % t
            if self._infer_stop.wait(1):
                return
        self.vrcft.send_params([(shape, -1.0) for _, shape, _ in targets])

        results = {}
        for n, (idx, shape, _) in enumerate(targets):
            name = self.cfg.classes[idx].name
            self.fastcal_state = "Calibrating %s (%d/%d)" % (name, n + 1, len(targets))
            log.info(self.fastcal_state)
            for i in range(101):  # ease in over 1s
                self.vrcft.send_packet(encode_params([(shape, i / 50 - 1)]))
                if self._infer_stop.wait(0.01):
                    return
            if self._infer_stop.wait(2):
                return
            total, count, seq = 0.0, 0, self.hub.latest()[0]
            end = time.monotonic() + 1.0
            while time.monotonic() < end and not stopped():
                seq, sample = self.hub.wait_next(seq, timeout=0.2)
                raw = predict(sample) if sample is not None else None
                if raw is not None:
                    total += float(raw[idx])
                    count += 1
            results[idx] = (total / count if count else 0.0) + 1e-9
            for i in range(101):  # ease out
                self.vrcft.send_packet(encode_params([(shape, 1 - i / 50)]))
                if self._infer_stop.wait(0.01):
                    return

        for idx, value in results.items():
            if value < 0.01:
                log.warning("Class '%s' barely activated (%.4f) - keeping previous max_power",
                            self.cfg.classes[idx].name, value)
                continue
            self.cfg.classes[idx].max_power = round(value, 4)
        self.save_config()
        log.info("FastCal finished: %s", {self.cfg.classes[i].name: round(v, 4) for i, v in results.items()})


def dataset_files(cfg):
    folder = cfg.dataset_folder
    if not os.path.isdir(folder):
        return []
    return sorted(f for f in os.listdir(folder) if f.endswith(".mmap"))
