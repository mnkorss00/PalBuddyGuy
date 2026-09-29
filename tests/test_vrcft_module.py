"""VRCFaceTracking v6 module integration.

ProtocolTwoTests always run (a fake protocol-2 client in Python).

EndToEndTests drive the real chain - Python app -> TCP -> PalBuddyGuy module ->
real VRCFaceTracking module process -> sandbox IPC -> a host emulator - and need
.NET build outputs. Set PALBUDDY_VRCFT_TEST_DIR to a folder containing:
  mp/VRCFaceTracking.ModuleProcess.dll   (VRCFaceTracking.ModuleProcess build)
  harness/Harness.dll                    (vrcft-module/tests/Harness build)
  fake/FakeSRanipal.dll                  (vrcft-module/tests/FakeSRanipal build)
See vrcft-module/README.md for the build commands.
"""

import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from palbuddy import vrcft_install  # noqa: E402
from palbuddy.config import Config, ExpressionClass  # noqa: E402
from palbuddy.net import recv_exact  # noqa: E402
from palbuddy.params import UNIFIED_EXPRESSIONS  # noqa: E402
from palbuddy.vrcft import VRCFTServer  # noqa: E402


def wait_for(cond, timeout=3.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.01)
    return False


def read_message(sock):
    """Decode one app -> module protocol-2 message."""
    kind = recv_exact(sock, 1)[0]
    if kind == 5:
        mode, count = recv_exact(sock, 2)
        names = []
        for _ in range(count):
            n = recv_exact(sock, 1)[0]
            names.append(bytes(recv_exact(sock, n)).decode("ascii"))
        return ("table", mode, names)
    if kind == 3:
        count = recv_exact(sock, 1)[0]
        raw = recv_exact(sock, count * 3)
        return ("weights", [(raw[i * 3], ((raw[i * 3 + 1] << 8) | raw[i * 3 + 2]) / 65535) for i in range(count)])
    raise AssertionError("unknown message %d" % kind)


class ProtocolTwoTests(unittest.TestCase):
    def setUp(self):
        self.srv = VRCFTServer("127.0.0.1", 0).start()
        self.addCleanup(self.srv.stop)

    def connect_v2(self):
        c = socket.create_connection(("127.0.0.1", self.srv.port))
        self.addCleanup(c.close)
        c.sendall(b"PBG2\x02")
        c.settimeout(3)
        self.assertTrue(wait_for(lambda: self.srv.protocol == 2))
        return c

    def test_table_and_weights(self):
        c = self.connect_v2()
        # SRanipal name expands to several Unified shapes; Unified names pass through
        self.assertTrue(self.srv.send_targets([("MouthPout", 1.0), ("BrowLowererLeft", 0.0)]))
        kind, mode, names = read_message(c)
        self.assertEqual((kind, mode), ("table", 0))
        self.assertEqual(names, ["LipPuckerUpperRight", "LipPuckerUpperLeft", "LipPuckerLowerRight",
                                 "LipPuckerLowerLeft", "BrowLowererLeft"])
        self.assertTrue(all(n in UNIFIED_EXPRESSIONS for n in names))
        kind, weights = read_message(c)
        self.assertEqual([s for s, _ in weights], [0, 1, 2, 3, 4])
        self.assertAlmostEqual(weights[0][1], 1.0, places=4)   # v=+1 -> weight 1
        self.assertAlmostEqual(weights[4][1], 0.5, places=4)   # v=0 -> weight 0.5
        # the table is sent once per connection; a subset of targets reuses it
        self.assertTrue(self.srv.send_targets([("BrowLowererLeft", -1.0)]))
        kind, weights = read_message(c)
        self.assertEqual(kind, "weights")
        self.assertEqual(weights, [(4, 0.0)])
        self.srv.clear()
        self.assertEqual(read_message(c), ("weights", []))

    def test_legacy_module_detected_by_silence_or_data(self):
        c = socket.create_connection(("127.0.0.1", self.srv.port))
        self.addCleanup(c.close)
        self.assertTrue(wait_for(lambda: self.srv.connected))
        self.assertFalse(self.srv.send_targets([("JawOpen", 1.0)]))  # nothing sent before we know
        self.assertTrue(wait_for(lambda: self.srv.protocol == 1, timeout=3))
        self.assertTrue(self.srv.send_targets([("JawOpen", 1.0), ("BrowLowererLeft", 1.0)]))
        c.settimeout(3)
        self.assertEqual(recv_exact(c, 5), bytearray([2, 1, 3, 255, 254]))  # only the SRanipal shape

    def test_engine_sends_protocol_two(self):
        import numpy as np
        import torch
        from palbuddy.engine import Engine
        from palbuddy.model import BuddyNet
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        cfg = Config(face_port=0, eye_port=0, vrcft_port=0, infer_engine="pytorch", infer_device="cpu",
                     model_path=os.path.join(tmp.name, "m.pt"), input_mode="face",
                     classes=[ExpressionClass("n"), ExpressionClass("frown", target="BrowLowererLeft")])
        engine = Engine(cfg).start()
        self.addCleanup(engine.stop)
        engine.model = BuddyNet(2, "face")
        c = socket.create_connection(("127.0.0.1", engine.vrcft.port))
        self.addCleanup(c.close)
        c.sendall(b"PBG2\x02")
        c.settimeout(3)
        self.assertTrue(wait_for(lambda: engine.vrcft.protocol == 2))
        engine.start_inference()
        engine.hub.push(None, np.full(25600, 0.5, np.float32).tobytes())
        self.assertEqual(read_message(c)[2], ["BrowLowererLeft"])
        kind, weights = read_message(c)
        self.assertEqual(kind, "weights")
        engine.stop_inference()
        while True:  # the last message after stopping is "clear"
            msg = read_message(c)
            if msg == ("weights", []):
                break
        del torch


class InstallerTests(unittest.TestCase):
    def test_install_uninstall(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        libs = os.path.join(tmp.name, "CustomLibs")
        sran = os.path.join(libs, "some-guid")
        os.makedirs(sran)
        with open(os.path.join(sran, "module.json"), "w") as f:
            json.dump({"ModuleName": "SRanipal", "DllFileName": "SRanipalExtTrackingModule.dll"}, f)
        open(os.path.join(sran, "SRanipalExtTrackingModule.dll"), "wb").close()
        cfg = Config(vrcft_custom_libs=libs, vrcft_port=26499)

        steps = vrcft_install.install(cfg)
        self.assertTrue(any("Disabled" in s for s in steps))
        target = os.path.join(libs, vrcft_install.MODULE_ID)
        self.assertTrue(os.path.exists(os.path.join(target, "PalBuddyGuy.VRCFT.dll")))
        self.assertFalse(os.path.exists(os.path.join(sran, "module.json")))
        with open(os.path.join(target, "palbuddyguy.json")) as f:
            settings = json.load(f)
        self.assertEqual(settings["port"], 26499)
        self.assertTrue(settings["innerModule"].endswith("SRanipalExtTrackingModule.dll"))
        st = vrcft_install.status(cfg)
        self.assertTrue(st["installed"])
        self.assertTrue(st["sranipal"][0]["disabled"])

        vrcft_install.uninstall(cfg)
        self.assertFalse(os.path.exists(target))
        self.assertTrue(os.path.exists(os.path.join(sran, "module.json")))
        self.assertFalse(vrcft_install.status(cfg)["installed"])


TEST_DIR = os.environ.get("PALBUDDY_VRCFT_TEST_DIR")


@unittest.skipUnless(TEST_DIR and os.path.isdir(TEST_DIR), "set PALBUDDY_VRCFT_TEST_DIR to run (needs .NET builds)")
class EndToEndTests(unittest.TestCase):
    def test_overrides_through_real_module_process(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        libs = os.path.join(tmp.name, "CustomLibs")
        sran = os.path.join(libs, "sranipal-guid")
        os.makedirs(sran)
        import shutil
        shutil.copy(os.path.join(TEST_DIR, "fake", "FakeSRanipal.dll"), sran)
        with open(os.path.join(sran, "module.json"), "w") as f:
            json.dump({"ModuleName": "SRanipal (fake)", "DllFileName": "FakeSRanipal.dll"}, f)

        srv = VRCFTServer("127.0.0.1", 0).start()
        self.addCleanup(srv.stop)
        cfg = Config(vrcft_custom_libs=libs, vrcft_port=srv.port)
        vrcft_install.install(cfg)  # installs the committed prebuilt DLL
        module_dll = os.path.join(libs, vrcft_install.MODULE_ID, "PalBuddyGuy.VRCFT.dll")

        proc = subprocess.Popen(
            ["dotnet", os.path.join(TEST_DIR, "harness", "Harness.dll"),
             os.path.join(TEST_DIR, "mp", "VRCFaceTracking.ModuleProcess.dll"), module_dll, "9"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.addCleanup(proc.kill)
        states = []
        threading.Thread(target=lambda: [states.append((time.time(), json.loads(line)))
                                         for line in proc.stdout], daemon=True).start()
        latest = lambda: states[-1][1] if states else {}  # noqa: E731

        self.assertTrue(wait_for(lambda: latest().get("init") and srv.protocol == 2, timeout=20), latest())
        self.assertEqual(latest()["name"], "PalBuddyGuy + SRanipal")
        self.assertTrue(latest()["eye"] and latest()["expr"])
        self.assertTrue(wait_for(lambda: abs(latest().get("JawOpen", 0) - 0.3) < 1e-3))  # plain SRanipal

        stop = threading.Event()

        def drive():
            while not stop.is_set():
                srv.send_targets([("JawOpen", 0.0), ("BrowLowererLeft", 0.6)])  # weights 0.5 and 0.8
                time.sleep(1 / 60)
        driver = threading.Thread(target=drive, daemon=True)
        driver.start()
        ok = wait_for(lambda: abs(latest().get("JawOpen", 0) - 0.5) < 1e-3
                      and abs(latest().get("BrowLowererLeft", 0) - 0.8) < 1e-3, timeout=5)
        reached = time.time()
        time.sleep(1.5)
        stop.set()
        driver.join()
        self.assertTrue(ok, latest())
        # no flicker back to SRanipal's value while overriding (the wrapped module writes its
        # values and then sleeps inside Update, so a naive overlay would lose most samples)
        window = [st for t, st in states if reached + 0.05 < t < reached + 1.4]
        self.assertGreater(len(window), 8)
        self.assertTrue(all(abs(st["JawOpen"] - 0.5) < 1e-3 for st in window),
                        [st["JawOpen"] for st in window])
        s = latest()
        self.assertAlmostEqual(s["EyeOpenLeft"], 0.8, places=3)          # eye data passes through
        self.assertAlmostEqual(s["MouthCornerPullLeft"], 0.25, places=3)  # untouched shape passes through
        self.assertAlmostEqual(s["EyeWideLeft"], 0.4, places=3)

        # no clear message: the module must drop stale overrides by itself
        self.assertTrue(wait_for(lambda: abs(latest().get("JawOpen", 0) - 0.3) < 1e-3
                                 and latest().get("BrowLowererLeft", 1) == 0.0, timeout=3), latest())
        proc.wait(timeout=20)
        stderr = proc.stderr.read()
        self.assertIn("Loaded inner module FakeSRanipal.FakeSRanipalModule", stderr)
        self.assertIn("Connected to Pal Buddy Guy", stderr)


if __name__ == "__main__":
    unittest.main()
