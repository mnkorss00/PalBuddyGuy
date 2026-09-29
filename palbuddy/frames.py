"""Receiving eye/face feature maps from the patched SRanipal tvm_runtime.

The patched tvm_runtime.dll connects to two TCP ports (one per tracker) and
streams packets of

    int32 device | int32 length | int32 thread_id   (little endian)
    payload[length]

where length is 80000 (camera image: 2x100x100 float32) or 102400
(neural feature map: 64x20x20 float32).

`FrameHub` is the in-process replacement for the old list-based
`neural_queue`: it keeps only the newest eye+face sample, wakes waiters with a
Condition instead of 1ms sleep-polling, and lets recorders subscribe to every
sample as it arrives.
"""

import logging
import struct
import threading
import time
from collections import deque

import numpy as np

from .net import Disconnected, close_quietly, make_listener, recv_exact_into, tune_stream

log = logging.getLogger(__name__)

NEURAL_BYTES = 64 * 20 * 20 * 4  # 102400
CAMERA_BYTES = 2 * 100 * 100 * 4  # 80000
SAMPLE_SHAPE = (128, 20, 20)  # eye features (64) + face features (64)
HEADER = struct.Struct("<iii")


def decode_neural(data):
    return np.frombuffer(data, dtype=np.float32).reshape(64, 20, 20)


def sample_array(eye, face):
    """(128, 20, 20) float32, eye first, matching the original training layout."""
    out = np.empty(SAMPLE_SHAPE, dtype=np.float32)
    out[:64] = decode_neural(eye)
    out[64:] = decode_neural(face)
    return out


def decode_camera(data, flipped=False):
    """Return a (100, 200) uint8 grayscale image of both camera views."""
    img = np.frombuffer(data, dtype=np.float32).reshape(2, 100, 100)
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    if flipped:
        return np.hstack((img[1, :, ::-1], img[0, :, ::-1]))
    return np.hstack((img[0], img[1]))


class RateMeter:
    def __init__(self, window=2.0):
        self.window = window
        self.times = deque()
        self.lock = threading.Lock()

    def tick(self, now=None):
        now = now or time.monotonic()
        with self.lock:
            self.times.append(now)
            while self.times and now - self.times[0] > self.window:
                self.times.popleft()

    def rate(self):
        now = time.monotonic()
        with self.lock:
            while self.times and now - self.times[0] > self.window:
                self.times.popleft()
            return len(self.times) / self.window


class StreamStatus:
    def __init__(self, name):
        self.name = name
        self.connected = False
        self.peer = None
        self.last_frame = 0.0
        self.connections = 0
        self.fps = RateMeter()

    def snapshot(self, stall_timeout):
        age = time.monotonic() - self.last_frame if self.last_frame else None
        if not self.connected:
            state = "disconnected"
        elif age is None or age > stall_timeout:
            state = "stalled"
        else:
            state = "ok"
        return {"name": self.name, "state": state, "fps": self.fps.rate(), "peer": self.peer,
                "connections": self.connections}


class FrameHub:
    """Latest-sample slot + fan-out to listeners. Thread safe."""

    def __init__(self):
        self._cond = threading.Condition()
        self._seq = 0
        self._sample = None  # (eye_bytes, face_bytes)
        self._listeners = []
        self._listeners_lock = threading.Lock()
        self.cameras = {"eye": None, "face": None}  # latest raw camera payloads
        self.sample_rate = RateMeter()

    def push(self, eye, face):
        with self._cond:
            self._seq += 1
            self._sample = (eye, face)
            self._cond.notify_all()
        self.sample_rate.tick()
        with self._listeners_lock:
            listeners = list(self._listeners)
        for fn in listeners:
            try:
                fn(eye, face)
            except Exception:  # a broken listener must never kill the receiver
                log.exception("frame listener failed")

    def latest(self):
        with self._cond:
            return self._seq, self._sample

    def wait_next(self, last_seq, timeout=None):
        """Block until a sample newer than `last_seq` exists. Returns (seq, sample)
        or (last_seq, None) on timeout."""
        with self._cond:
            if not self._cond.wait_for(lambda: self._seq != last_seq, timeout):
                return last_seq, None
            return self._seq, self._sample

    def clear(self):
        with self._cond:
            self._sample = None

    def add_listener(self, fn):
        with self._listeners_lock:
            self._listeners.append(fn)

    def remove_listener(self, fn):
        with self._listeners_lock:
            if fn in self._listeners:
                self._listeners.remove(fn)


class SRanipalReceiver:
    """Listens on the two ports the patched tvm_runtime connects to.

    Improvements over the original proxy:
      * a reconnecting SRanipal (runtime restart) replaces the stale connection
        instead of being stuck in the backlog behind it,
      * disconnects are detected (no 100% CPU spin),
      * the eye/face swap can be toggled live and is persisted in the config,
      * per-stream fps / stall status for the GUI.
    """

    def __init__(self, hub, host="127.0.0.1", face_port=18452, eye_port=18453, swapped=False):
        self.hub = hub
        self.host = host
        self.ports = (face_port, eye_port)
        self.swapped = swapped
        self.status = {"port0": StreamStatus("port %d" % face_port), "port1": StreamStatus("port %d" % eye_port)}
        self._last_eye = None
        self._active = [None, None]
        self._stop = threading.Event()
        self._listeners = []
        self._threads = []

    def role(self, port_index):
        """'face' or 'eye' for the given port index, honouring the swap flag."""
        return "face" if (port_index == 0) != self.swapped else "eye"

    def set_swapped(self, value):
        self.swapped = bool(value)
        self._last_eye = None
        self.hub.clear()

    def start(self):
        ports = []
        for i, port in enumerate(self.ports):
            listener = make_listener(self.host, port)
            self._listeners.append(listener)
            port = listener.getsockname()[1]  # resolves port 0 (tests)
            ports.append(port)
            t = threading.Thread(target=self._accept_loop, args=(i, listener), daemon=True,
                                 name="sranipal-accept-%d" % port)
            t.start()
            self._threads.append(t)
        self.ports = tuple(ports)
        log.info("Waiting for SRanipal on %s:%d and %s:%d", self.host, self.ports[0], self.host, self.ports[1])
        return self

    def stop(self):
        self._stop.set()
        for s in self._listeners:
            close_quietly(s)
        for c in self._active:
            close_quietly(c)

    def _accept_loop(self, index, listener):
        while not self._stop.is_set():
            try:
                conn, addr = listener.accept()
            except OSError:
                if self._stop.is_set():
                    return
                time.sleep(0.2)
                continue
            old = self._active[index]
            self._active[index] = conn
            if old is not None:
                log.info("New SRanipal connection on port %d replaces the old one", self.ports[index])
                close_quietly(old)
            threading.Thread(target=self._handle, args=(index, conn, addr), daemon=True,
                             name="sranipal-conn-%d" % self.ports[index]).start()

    def _handle(self, index, conn, addr):
        st = self.status["port%d" % index]
        st.connected = True
        st.peer = "%s:%d" % addr
        st.connections += 1
        tune_stream(conn, low_latency=False)
        log.info("SRanipal connected on port %d from %s", self.ports[index], st.peer)
        header = bytearray(HEADER.size)
        header_view = memoryview(header)
        payload = bytearray(NEURAL_BYTES)
        payload_view = memoryview(payload)
        try:
            while not self._stop.is_set() and self._active[index] is conn:
                recv_exact_into(conn, header_view)
                _device, length, _thread_id = HEADER.unpack(header)
                if length not in (NEURAL_BYTES, CAMERA_BYTES):
                    log.warning("Invalid packet length %d on port %d, dropping connection", length, self.ports[index])
                    break
                recv_exact_into(conn, payload_view[:length])
                data = bytes(payload_view[:length])
                role = self.role(index)
                if length == CAMERA_BYTES:
                    self.hub.cameras[role] = data
                    continue
                st.last_frame = time.monotonic()
                st.fps.tick(st.last_frame)
                if role == "eye":
                    self._last_eye = data
                elif self._last_eye is not None:
                    self.hub.push(self._last_eye, data)
        except (Disconnected, OSError) as e:
            if not self._stop.is_set() and self._active[index] is conn:
                log.info("SRanipal on port %d disconnected: %s", self.ports[index], e)
        finally:
            close_quietly(conn)
            if self._active[index] is conn:
                self._active[index] = None
                st.connected = False

    def snapshot(self, stall_timeout=2.0):
        out = {}
        for i in (0, 1):
            snap = self.status["port%d" % i].snapshot(stall_timeout)
            snap["role"] = self.role(i)
            out[snap["role"]] = snap
        return out


class ProxyClient:
    """Reads samples from a separately running tvm_proxy.py (legacy mode)."""

    SAMPLE_BYTES = NEURAL_BYTES * 2

    def __init__(self, hub, host="127.0.0.1", port=18454):
        self.hub = hub
        self.host = host
        self.port = port
        self.swapped = False  # handled inside the proxy process
        self.status = StreamStatus("proxy %d" % port)
        self._stop = threading.Event()
        self._sock = None

    def set_swapped(self, value):
        log.warning("In proxy mode, use the 'swap' command of tvm_proxy.py")

    def start(self):
        threading.Thread(target=self._loop, daemon=True, name="proxy-client").start()
        return self

    def stop(self):
        self._stop.set()
        close_quietly(self._sock)

    def _loop(self):
        import socket
        backoff = 0.25
        buf = bytearray(self.SAMPLE_BYTES)
        view = memoryview(buf)
        while not self._stop.is_set():
            try:
                self._sock = socket.create_connection((self.host, self.port), timeout=3)
                self._sock.settimeout(None)
                tune_stream(self._sock, low_latency=False)
                self.status.connected = True
                self.status.connections += 1
                self.status.peer = "%s:%d" % (self.host, self.port)
                log.info("Connected to tvm_proxy on %s:%d", self.host, self.port)
                backoff = 0.25
                while not self._stop.is_set():
                    recv_exact_into(self._sock, view)
                    self.status.last_frame = time.monotonic()
                    self.status.fps.tick(self.status.last_frame)
                    self.hub.push(bytes(view[:NEURAL_BYTES]), bytes(view[NEURAL_BYTES:]))
            except (Disconnected, OSError) as e:
                if self.status.connected:
                    log.info("Lost tvm_proxy connection: %s", e)
            finally:
                self.status.connected = False
                close_quietly(self._sock)
            # exponential backoff instead of hammering the port every 100ms
            self._stop.wait(backoff)
            backoff = min(backoff * 2, 3.0)

    def snapshot(self, stall_timeout=2.0):
        snap = self.status.snapshot(stall_timeout)
        return {"eye": dict(snap, role="eye"), "face": dict(snap, role="face")}


class ProxyServer:
    """Serves hub samples to a legacy client on port 18454 (used by tvm_proxy.py).

    Only the newest sample is sent; a slow client never builds up latency.
    """

    def __init__(self, hub, host="127.0.0.1", port=18454):
        self.hub = hub
        self.host = host
        self.port = port
        self._stop = threading.Event()
        self._listener = None
        self.clients = 0

    def start(self):
        self._listener = make_listener(self.host, self.port)
        self.port = self._listener.getsockname()[1]
        threading.Thread(target=self._accept_loop, daemon=True, name="proxy-server").start()
        return self

    def stop(self):
        self._stop.set()
        close_quietly(self._listener)

    def _accept_loop(self):
        while not self._stop.is_set():
            try:
                conn, addr = self._listener.accept()
            except OSError:
                if self._stop.is_set():
                    return
                continue
            threading.Thread(target=self._serve, args=(conn, addr), daemon=True).start()

    def _serve(self, conn, addr):
        log.info("Proxy client connected from %s:%d", *addr)
        tune_stream(conn)
        self.clients += 1
        seq = self.hub.latest()[0]
        try:
            while not self._stop.is_set():
                seq, sample = self.hub.wait_next(seq, timeout=1.0)
                if sample is None:
                    continue
                conn.sendall(sample[0] + sample[1])
        except OSError as e:
            log.info("Proxy client %s:%d left: %s", addr[0], addr[1], e)
        finally:
            self.clients -= 1
            close_quietly(conn)
