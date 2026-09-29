"""Training loop.

Speedups over the original per-sample Python loop:
  * batches are gathered with one sorted fancy-index read per recording file
    instead of 128 random single-frame reads,
  * a background thread prepares the next batches (pinned memory, async copy)
    while the GPU trains on the current one,
  * optional automatic mixed precision on CUDA,
  * cudnn autotuning.
The sampling distribution is the same: uniform class, uniform file within the
class, uniform frame within the file.
"""

import contextlib
import logging
import queue
import threading
import time

import numpy as np
import torch

from .datasets import open_recording
from .model import BuddyNet, pick_device

log = logging.getLogger(__name__)


class TrainingCancelled(Exception):
    pass


class BatchSampler:
    def __init__(self, recordings_per_class, batch_size, seed=None):
        self.classes = recordings_per_class  # list[list[array (N,128,20,20)]]
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)
        self.num_classes = len(recordings_per_class)
        self.eye = np.eye(self.num_classes, dtype=np.float32)

    def sample(self):
        b = self.batch_size
        cls = self.rng.integers(0, self.num_classes, size=b)
        x = np.empty((b, 128, 20, 20), dtype=np.float32)
        for c in np.unique(cls):
            slots = np.nonzero(cls == c)[0]
            recs = self.classes[c]
            files = self.rng.integers(0, len(recs), size=len(slots))
            for f in np.unique(files):
                s = slots[files == f]
                rec = recs[f]
                idx = self.rng.integers(0, len(rec), size=len(s))
                order = np.argsort(idx)  # sequential-ish disk access for memmaps
                x[s[order]] = rec[idx[order]]
        return x, self.eye[cls]


def load_recordings(cfg, log_fn=print):
    recordings = []
    total = 0
    for c in cfg.classes:
        recs = []
        for name in c.files:
            path = cfg.dataset_path(name)
            try:
                rec = open_recording(path, in_ram=cfg.cache_datasets_in_ram)
            except FileNotFoundError:
                raise FileNotFoundError("Recording '%s' of class '%s' not found" % (path, c.name))
            recs.append(rec)
            total += len(rec)
        if not recs:
            raise ValueError("Class '%s' has no recordings" % c.name)
        recordings.append(recs)
    log_fn("Loaded %d recordings (%d frames) for %d classes" % (sum(map(len, recordings)), total, len(recordings)))
    return recordings, total


def train(cfg, on_progress=None, stop_event=None, log_fn=print, init_model=None, device=None):
    """Train a model on the configured classes and return it (on `device`)."""
    device = device or pick_device()
    stop_event = stop_event or threading.Event()
    on_progress = on_progress or (lambda **kw: None)

    recordings, total_frames = load_recordings(cfg, log_fn)
    model = init_model if init_model is not None else BuddyNet(cfg.num_classes)
    if model.num_outputs != cfg.num_classes:
        raise ValueError("Model output count does not match the class list")
    model.to(device).train()
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    use_amp = cfg.mixed_precision and device.type == "cuda"
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    else:  # torch < 2.3
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    mse = torch.nn.MSELoss()

    # one "epoch" = as many samples as the original: 2048 frames per class
    steps = max(1, (2048 * cfg.num_classes) // cfg.batch_size)
    sampler = BatchSampler(recordings, cfg.batch_size)
    q = queue.Queue(maxsize=6)
    producer_stop = threading.Event()

    def producer():
        pin = device.type == "cuda"
        try:
            while not producer_stop.is_set():
                x, y = sampler.sample()
                x, y = torch.from_numpy(x), torch.from_numpy(y)
                if pin:
                    x, y = x.pin_memory(), y.pin_memory()
                while not producer_stop.is_set():
                    try:
                        q.put((x, y), timeout=0.2)
                        break
                    except queue.Full:
                        pass
        except Exception as e:  # surface loader errors in the training thread
            q.put(e)

    workers = [threading.Thread(target=producer, daemon=True, name="batch-loader-%d" % i) for i in range(2)]
    for w in workers:
        w.start()

    history = []
    start = time.monotonic()
    try:
        for epoch in range(cfg.epochs):
            running = 0.0
            for step in range(steps):
                if stop_event.is_set():
                    raise TrainingCancelled()
                item = q.get()
                if isinstance(item, Exception):
                    raise item
                x = item[0].to(device, non_blocking=True)
                y = item[1].to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with (torch.autocast(device_type="cuda") if use_amp else contextlib.nullcontext()):
                    pred = model(x)
                loss = mse(pred.float(), y) * 100  # same scale as the original
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                value = float(loss.detach()) / 100
                running += value
                if step % 4 == 0 or step == steps - 1:
                    on_progress(epoch=epoch, epochs=cfg.epochs, step=step + 1, steps=steps, loss=value,
                                avg=running / (step + 1), elapsed=time.monotonic() - start)
            avg = running / steps
            history.append(avg)
            log_fn("Epoch %d/%d  avg loss %.6f" % (epoch + 1, cfg.epochs, avg))
            on_progress(epoch=epoch + 1, epochs=cfg.epochs, step=steps, steps=steps, loss=avg, avg=avg,
                        elapsed=time.monotonic() - start, epoch_done=True)
    finally:
        producer_stop.set()
    model.eval()
    if history and history[-1] > 0.001:
        log_fn("Warning: final loss %.6f is above 0.001 - check your recordings / class list" % history[-1])
    return model, history
