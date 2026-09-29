"""Offline tests: fake SRanipal and VRCFT peers talk to the real servers over loopback."""

import os
import socket
import struct
import sys
import tempfile
import threading
import time
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from palbuddy.config import Config, ExpressionClass, default_config  # noqa: E402
from palbuddy.datasets import Recorder, frame_count, open_recording  # noqa: E402
from palbuddy.engine import Engine, normalize  # noqa: E402
from palbuddy.frames import (CAMERA_BYTES, FrameHub, ProxyClient, ProxyServer,  # noqa: E402
                             SRanipalReceiver)
from palbuddy.model import BuddyNet  # noqa: E402
from palbuddy.net import recv_exact  # noqa: E402
from palbuddy.trainer import train  # noqa: E402
from palbuddy.vrcft import VRCFTServer, encode_params  # noqa: E402


def neural(value):
    return np.full(64 * 20 * 20, value, dtype=np.float32).tobytes()


def packet(payload):
    return struct.pack("<iii", 0, len(payload), 1) + payload


def wait_for(cond, timeout=3.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.01)
    return False


def legacy_encode(key, f):
    """Byte-for-byte what the original write_param produced."""
    f = (f + 1) * 32767.0
    f = int(max(0, min(65535, f)))
    return bytes([key]) + bytes([f // 256, f % 256])


class ProtocolTests(unittest.TestCase):
    def test_encode_matches_original(self):
        pairs = [(0, -1.0), (19, 0.25), (2, 1.0), (11, 1.7), (3, -3)]
        expected = bytes([2, len(pairs)]) + b"".join(legacy_encode(k, v) for k, v in pairs)
        self.assertEqual(encode_params(pairs), expected)

    def test_normalize_matches_original(self):
        self.assertAlmostEqual(normalize(0.45, 0.9), 0.0)
        self.assertEqual(normalize(5, 0.9), 1.0)
        self.assertEqual(normalize(0, 0.9), -1.0)


class ModelTests(unittest.TestCase):
    def test_legacy_checkpoint_roundtrip(self):
        m = BuddyNet(4)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "buddyguy.pt")
            # exactly the dict layout the original `save` command wrote
            torch.save({k: getattr(m, k).state_dict() for k in ("conv1", "conv2", "linear1", "linear2")}, path)
            m2 = BuddyNet.load(path, expected_outputs=4).eval()
            x = torch.randn(3, 128, 20, 20)
            m.eval()
            self.assertTrue(torch.allclose(m(x), m2(x)))
            with self.assertRaises(ValueError):
                BuddyNet.load(path, expected_outputs=5)

    def test_any_batch_size(self):
        m = BuddyNet(3).eval()
        for b in (1, 7):
            self.assertEqual(tuple(m(torch.randn(b, 128, 20, 20)).shape), (b, 3))


class ReceiverTests(unittest.TestCase):
    def setUp(self):
        self.hub = FrameHub()
        self.rx = SRanipalReceiver(self.hub, "127.0.0.1", 0, 0).start()

    def tearDown(self):
        self.rx.stop()

    def connect(self, index):
        return socket.create_connection(("127.0.0.1", self.rx.ports[index]))

    def test_pairs_eye_and_face(self):
        face, eye = self.connect(0), self.connect(1)
        eye.sendall(packet(neural(1.0)) + packet(np.zeros(CAMERA_BYTES // 4, np.float32).tobytes()))
        time.sleep(0.1)
        face.sendall(packet(neural(2.0)))
        seq, sample = self.hub.wait_next(0, timeout=2)
        self.assertIsNotNone(sample)
        self.assertEqual(np.frombuffer(sample[0], np.float32)[0], 1.0)
        self.assertEqual(np.frombuffer(sample[1], np.float32)[0], 2.0)
        self.assertIsNotNone(self.hub.cameras["eye"])
        snap = self.rx.snapshot()
        self.assertEqual(snap["face"]["state"], "ok")
        face.close()
        eye.close()

    def test_swap(self):
        self.rx.set_swapped(True)
        a, b = self.connect(0), self.connect(1)
        a.sendall(packet(neural(1.0)))  # port 0 is now the eye
        time.sleep(0.1)
        b.sendall(packet(neural(2.0)))
        _, sample = self.hub.wait_next(0, timeout=2)
        self.assertEqual(np.frombuffer(sample[0], np.float32)[0], 1.0)
        a.close()
        b.close()

    def test_disconnect_and_reconnect(self):
        face = self.connect(0)
        self.assertTrue(wait_for(lambda: self.rx.status["port0"].connected))
        face.close()
        self.assertTrue(wait_for(lambda: not self.rx.status["port0"].connected))
        face2, eye = self.connect(0), self.connect(1)
        eye.sendall(packet(neural(3.0)))
        time.sleep(0.1)
        face2.sendall(packet(neural(4.0)))
        _, sample = self.hub.wait_next(0, timeout=2)
        self.assertIsNotNone(sample)
        face2.close()
        eye.close()

    def test_new_connection_replaces_stale_one(self):
        old = self.connect(0)
        self.assertTrue(wait_for(lambda: self.rx.status["port0"].connected))
        new, eye = self.connect(0), self.connect(1)
        eye.sendall(packet(neural(5.0)))
        time.sleep(0.1)
        new.sendall(packet(neural(6.0)))
        _, sample = self.hub.wait_next(0, timeout=2)
        self.assertEqual(np.frombuffer(sample[1], np.float32)[0], 6.0)
        old.settimeout(2)
        self.assertEqual(old.recv(1), b"")  # server closed the stale socket
        for s in (old, new, eye):
            s.close()

    def test_bad_packet_drops_connection(self):
        face = self.connect(0)
        face.sendall(struct.pack("<iii", 0, 12345, 0))
        face.settimeout(2)
        self.assertEqual(face.recv(1), b"")
        face.close()


class ProxyTests(unittest.TestCase):
    def test_proxy_roundtrip(self):
        src = FrameHub()
        server = ProxyServer(src, "127.0.0.1", 0).start()
        dst = FrameHub()
        client = ProxyClient(dst, "127.0.0.1", server.port).start()
        self.assertTrue(wait_for(lambda: server.clients == 1))
        src.push(neural(7.0), neural(8.0))
        _, sample = dst.wait_next(0, timeout=2)
        self.assertEqual(np.frombuffer(sample[1], np.float32)[0], 8.0)
        client.stop()
        server.stop()


class VRCFTTests(unittest.TestCase):
    def test_send_and_reconnect(self):
        srv = VRCFTServer("127.0.0.1", 0).start()
        c1 = socket.create_connection(("127.0.0.1", srv.port))
        self.assertTrue(wait_for(lambda: srv.connected))
        self.assertTrue(srv.send_params([(3, 1.0)]))
        self.assertEqual(recv_exact(c1, 5), bytearray([2, 1, 3, 255, 254]))
        # VRCFT sends lip data; must be consumed without errors
        c1.sendall(bytes([2]) + b"\x7f\xff" * 60)
        self.assertTrue(wait_for(lambda: 2 in srv.last_received))
        c2 = socket.create_connection(("127.0.0.1", srv.port))
        self.assertTrue(wait_for(lambda: srv.peer and srv.peer.endswith(str(c2.getsockname()[1]))))
        self.assertTrue(srv.send_params([(1, -1.0)]))
        self.assertEqual(recv_exact(c2, 5), bytearray([2, 1, 1, 0, 0]))
        c2.close()
        self.assertTrue(wait_for(lambda: not srv.connected))
        self.assertFalse(srv.send_params([(1, 0.0)]))
        c1.close()
        srv.stop()


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cfg = Config(dataset_folder=self.tmp.name, model_path=os.path.join(self.tmp.name, "m.pt"),
                          face_port=0, eye_port=0, vrcft_port=0, record_countdown=0, record_frames=40,
                          epochs=2, batch_size=16, mixed_precision=False)

    def tearDown(self):
        self.tmp.cleanup()

    def test_record_train_infer(self):
        hub = FrameHub()
        path = os.path.join(self.tmp.name, "a-em.mmap")
        done = threading.Event()
        result = {}
        Recorder(hub, path, 40, countdown=0, on_done=lambda ok, m: (result.update(ok=ok), done.set())).start()
        time.sleep(0.2)
        for i in range(60):
            hub.push(neural(i), neural(-i))
        self.assertTrue(done.wait(5))
        self.assertTrue(result["ok"])
        self.assertEqual(frame_count(path), 40)
        rec = open_recording(path)
        self.assertEqual(rec[3, 0, 0, 0], 3.0)
        self.assertEqual(rec[3, 64, 0, 0], -3.0)

        # tiny synthetic 2-class set
        rng = np.random.default_rng(0)
        for name, offset in (("n", 0.0), ("s", 1.0)):
            mm = np.memmap(os.path.join(self.tmp.name, name + ".mmap"), np.float32, "w+", shape=(32, 128, 20, 20))
            mm[:] = rng.normal(offset, 0.1, size=mm.shape)
            mm.flush()
            del mm
        self.cfg.classes = [ExpressionClass("neutral", ["n.mmap"]), ExpressionClass("smile", ["s.mmap"], "JawOpen")]
        model, history = train(self.cfg, log_fn=lambda *a: None, device=torch.device("cpu"))
        self.assertEqual(len(history), 2)

        engine = Engine(self.cfg)
        engine.device = torch.device("cpu")
        engine.start()
        try:
            engine.model = model
            vr = socket.create_connection(("127.0.0.1", engine.vrcft.port))
            self.assertTrue(wait_for(lambda: engine.vrcft.connected))
            engine.start_inference()
            eye = socket.create_connection(("127.0.0.1", engine.source.ports[1]))
            face = socket.create_connection(("127.0.0.1", engine.source.ports[0]))
            eye.sendall(packet(neural(1.0)))
            time.sleep(0.05)
            face.sendall(packet(neural(1.0)))
            vr.settimeout(3)
            msg = recv_exact(vr, 5)
            self.assertEqual(msg[:3], bytearray([2, 1, 3]))  # JawOpen
            self.assertIsNotNone(engine.last_raw)
            engine.stop_inference()
            self.assertFalse(engine.inferring)
            for s in (vr, eye, face):
                s.close()
        finally:
            engine.stop()


class SingleTrackerTests(unittest.TestCase):
    """Face-only and eye-only setups."""

    def make_rx(self, mode, swapped=False):
        hub = FrameHub()
        rx = SRanipalReceiver(hub, "127.0.0.1", 0, 0, swapped=swapped, mode=mode).start()
        self.addCleanup(rx.stop)
        return hub, rx

    def connect(self, rx, index):
        s = socket.create_connection(("127.0.0.1", rx.ports[index]))
        self.addCleanup(s.close)
        return s

    def test_face_only_on_either_port(self):
        # the single stream lands on the port that is "eye" by default: no swap needed
        hub, rx = self.make_rx("face")
        s = self.connect(rx, 1)
        s.sendall(packet(neural(2.0)))
        _, sample = hub.wait_next(0, timeout=2)
        self.assertIsNone(sample[0])
        self.assertEqual(np.frombuffer(sample[1], np.float32)[0], 2.0)
        self.assertEqual(rx.snapshot()["face"]["state"], "ok")

    def test_eye_only_on_either_port(self):
        hub, rx = self.make_rx("eye")
        s = self.connect(rx, 0)
        s.sendall(packet(neural(3.0)))
        _, sample = hub.wait_next(0, timeout=2)
        self.assertEqual(np.frombuffer(sample[0], np.float32)[0], 3.0)
        self.assertIsNone(sample[1])

    def test_face_mode_with_both_trackers_ignores_eye(self):
        hub, rx = self.make_rx("face")
        face, eye = self.connect(rx, 0), self.connect(rx, 1)
        face.sendall(packet(neural(1.0)))
        eye.sendall(packet(neural(9.0)))
        seq, sample = hub.wait_next(0, timeout=2)
        time.sleep(0.2)
        for _ in range(3):
            eye.sendall(packet(neural(9.0)))
        face.sendall(packet(neural(4.0)))
        deadline = time.time() + 2
        while time.time() < deadline:
            seq, sample = hub.wait_next(seq, timeout=0.5)
            if sample is None:
                continue
            self.assertIsNone(sample[0])
            self.assertNotEqual(np.frombuffer(sample[1], np.float32)[0], 9.0)
            if np.frombuffer(sample[1], np.float32)[0] == 4.0:
                break
        else:
            self.fail("face sample not received")

    def test_stalled_face_never_takes_eye_data(self):
        hub, rx = self.make_rx("face")
        face, eye = self.connect(rx, 0), self.connect(rx, 1)
        face.sendall(packet(neural(1.0)))
        seq, _ = hub.wait_next(0, timeout=2)
        rx.status["port0"].last_frame -= 10  # face tracker stalled for a long time
        eye.sendall(packet(neural(9.0)) * 5)
        seq2, sample = hub.wait_next(seq, timeout=0.5)
        self.assertIsNone(sample)

    def test_proxy_zero_fills_missing_tracker(self):
        src = FrameHub()
        server = ProxyServer(src, "127.0.0.1", 0).start()
        self.addCleanup(server.stop)
        c = socket.create_connection(("127.0.0.1", server.port))
        self.addCleanup(c.close)
        self.assertTrue(wait_for(lambda: server.clients == 1))
        src.push(None, neural(5.0))
        data = np.frombuffer(bytes(recv_exact(c, 2 * 102400)), np.float32)
        self.assertEqual(data[0], 0.0)
        self.assertEqual(data[25600], 5.0)

    def test_model_mode_in_checkpoint(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.pt")
            m = BuddyNet(2, "face")
            self.assertEqual(m(torch.randn(1, 64, 20, 20)).shape[1], 2)
            m.save(path)
            self.assertEqual(BuddyNet.load(path, expected_mode="face").input_mode, "face")
            with self.assertRaises(ValueError):
                BuddyNet.load(path, expected_mode="both")
            legacy = os.path.join(d, "legacy.pt")
            b = BuddyNet(2)
            torch.save({k: getattr(b, k).state_dict() for k in ("conv1", "conv2", "linear1", "linear2")}, legacy)
            self.assertEqual(BuddyNet.load(legacy).input_mode, "both")

    def test_train_and_infer_face_only(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        rng = np.random.default_rng(1)
        for name, offset, eye in (("n", 0.0, True), ("s", 1.0, True), ("faceonly", 1.0, False)):
            mm = np.memmap(os.path.join(tmp.name, name + ".mmap"), np.float32, "w+", shape=(16, 128, 20, 20))
            mm[:] = rng.normal(offset, 0.1, size=mm.shape)
            if not eye:
                mm[:, :64] = 0  # recorded with the facial tracker only
            mm.flush()
            del mm
        cfg = Config(dataset_folder=tmp.name, face_port=0, eye_port=0, vrcft_port=0, epochs=1, batch_size=64,
                     mixed_precision=False, input_mode="face",
                     classes=[ExpressionClass("neutral", ["n.mmap"]),
                              ExpressionClass("smile", ["s.mmap", "faceonly.mmap"], "JawOpen")])
        model, _ = train(cfg, log_fn=lambda *a: None, device=torch.device("cpu"))
        self.assertEqual(model.input_mode, "face")
        self.assertEqual(model.conv1.in_channels, 64)

        cfg.input_mode = "both"  # the face-only recording can't feed a both-model
        with self.assertRaises(ValueError):
            train(cfg, log_fn=lambda *a: None, device=torch.device("cpu"))
        cfg.input_mode = "face"

        engine = Engine(cfg)
        engine.device = torch.device("cpu")
        engine.start()
        self.addCleanup(engine.stop)
        engine.model = model
        vr = socket.create_connection(("127.0.0.1", engine.vrcft.port))
        self.addCleanup(vr.close)
        self.assertTrue(wait_for(lambda: engine.vrcft.connected))
        engine.start_inference()
        face = socket.create_connection(("127.0.0.1", engine.source.ports[1]))
        self.addCleanup(face.close)
        face.sendall(packet(neural(1.0)))
        vr.settimeout(3)
        self.assertEqual(recv_exact(vr, 5)[:3], bytearray([2, 1, 3]))
        self.assertIsNone(engine.status()["mode_hint"])
        engine.set_input_mode("both")  # a face model can't run in both mode
        self.assertIsNone(engine.model)
        self.assertFalse(engine.inferring)

    def test_mode_hint(self):
        e = Engine(Config(input_mode="both"))
        ok = {"state": "ok"}
        off = {"state": "disconnected"}
        self.assertEqual(e.mode_hint({"eye": off, "face": ok}), "single")
        self.assertEqual(e.mode_hint({"eye": ok, "face": off}), "single")
        self.assertIsNone(e.mode_hint({"eye": ok, "face": ok}))
        self.assertIsNone(e.mode_hint({"eye": off, "face": off}))
        e.cfg.input_mode = "face"
        self.assertEqual(e.mode_hint({"eye": ok, "face": ok}), "both")


class ConfigTests(unittest.TestCase):
    def test_roundtrip(self):
        cfg = default_config()
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "config.json")
            cfg.save(p)
            cfg2 = Config.load(p)
        self.assertEqual(cfg2.to_dict(), cfg.to_dict())
        # same mapping as the original to_replace table
        self.assertEqual({i: s for i, s, _ in cfg.targets()}, {2: 0, 1: 1, 4: 20, 5: 19, 7: 2, 3: 11})
        self.assertEqual(cfg.validate(), [])


if __name__ == "__main__":
    unittest.main()
