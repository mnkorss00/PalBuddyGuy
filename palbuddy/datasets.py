"""Recording and loading training data.

Recordings are written straight to .mmap files (N x 128 x 20 x 20 float32), the
same format the old `convertmmap` command produced, so the separate
pickle -> mmap conversion step is no longer needed. The frame count is derived
from the file size, so recordings don't have to be exactly 2048 frames.
"""

import logging
import os
import pickle
import threading
import time

import numpy as np

from .frames import SAMPLE_SHAPE, sample_array

log = logging.getLogger(__name__)

SAMPLE_FLOATS = int(np.prod(SAMPLE_SHAPE))
SAMPLE_BYTES = SAMPLE_FLOATS * 4


def frame_count(path):
    return os.path.getsize(path) // SAMPLE_BYTES


def open_recording(path, in_ram=False):
    n = frame_count(path)
    if n == 0:
        raise ValueError("%s is empty or not a recording" % path)
    mm = np.memmap(path, dtype=np.float32, mode="r", shape=(n,) + SAMPLE_SHAPE)
    return np.array(mm) if in_ram else mm


class Recorder:
    """Captures every sample pushed to the FrameHub into a .mmap file."""

    def __init__(self, hub, path, frames, countdown=5.0, on_progress=None, on_done=None):
        self.hub = hub
        self.path = path
        self.frames = frames
        self.countdown = countdown
        self.on_progress = on_progress or (lambda done, total, phase: None)
        self.on_done = on_done or (lambda ok, message: None)
        self._count = 0
        self._mm = None
        self._finished = threading.Event()
        self._cancel = threading.Event()
        self._lock = threading.Lock()

    def start(self):
        threading.Thread(target=self._run, daemon=True, name="recorder").start()
        return self

    def cancel(self):
        self._cancel.set()
        self._finished.set()

    def _on_sample(self, eye, face):
        with self._lock:
            if self._mm is None or self._count >= self.frames:
                return
            self._mm[self._count] = sample_array(eye, face)
            self._count += 1
            if self._count >= self.frames:
                self._finished.set()

    def _run(self):
        tmp = self.path + ".part"
        try:
            end = time.monotonic() + self.countdown
            while not self._cancel.is_set() and time.monotonic() < end:
                self.on_progress(0, self.frames, "countdown %.0f" % max(0.0, end - time.monotonic()))
                self._cancel.wait(0.25)
            if self._cancel.is_set():
                self.on_done(False, "Recording cancelled")
                return
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
            self._mm = np.memmap(tmp, dtype=np.float32, mode="w+", shape=(self.frames,) + SAMPLE_SHAPE)
            self.hub.add_listener(self._on_sample)
            last_seen, last_change = -1, time.monotonic()
            while not self._finished.wait(0.1):
                if self._count != last_seen:
                    last_seen, last_change = self._count, time.monotonic()
                elif time.monotonic() - last_change > 10:
                    log.warning("No frames for 10s while recording - is SRanipal running?")
                    last_change = time.monotonic()
                self.on_progress(self._count, self.frames, "recording")
            self.hub.remove_listener(self._on_sample)
            with self._lock:
                count = self._count
                self._mm.flush()
                self._mm = None
            if self._cancel.is_set() or count < self.frames:
                os.remove(tmp)
                self.on_done(False, "Recording cancelled")
                return
            os.replace(tmp, self.path)
            self.on_progress(count, self.frames, "done")
            self.on_done(True, "Saved %d frames to %s" % (count, self.path))
        except Exception as e:
            log.exception("recording failed")
            self.hub.remove_listener(self._on_sample)
            self.on_done(False, "Recording failed: %s" % e)


def convert_legacy_pickles(folder, on_progress=None):
    """Convert old *-em.pkl recordings to .mmap (the old `convertmmap` command)."""
    converted = []
    files = [f for f in sorted(os.listdir(folder)) if f.endswith(".pkl")]
    for i, name in enumerate(files):
        src = os.path.join(folder, name)
        dst = src[:-3] + "mmap"
        if os.path.exists(dst):
            continue
        with open(src, "rb") as f:
            block = pickle.load(f)
        mm = np.memmap(dst + ".part", dtype=np.float32, mode="w+", shape=(len(block),) + SAMPLE_SHAPE)
        for j, (eye, face) in enumerate(block):
            mm[j, :64] = eye
            mm[j, 64:] = face
        mm.flush()
        del mm
        os.replace(dst + ".part", dst)
        converted.append(dst)
        if on_progress:
            on_progress(i + 1, len(files))
    return converted
