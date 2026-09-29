"""Connection to the VRCFaceTracking PalBuddyGuy module (TCP, port 26421).

Wire format (unchanged, so the existing VRCFT module keeps working):

  us -> VRCFT:  0x02 | count | count * (shape_index:u8, value:u16 big endian)
                value = (v + 1) * 32767, v in [-1, 1]
  VRCFT -> us:  0x01 + 37 * u16   (eye data)
                0x02 + 60 * u16   (lip data)
"""

import logging
import socket
import threading
import time

from .net import Disconnected, close_quietly, make_listener, recv_exact, tune_stream

log = logging.getLogger(__name__)

MSG_PARAMS = 2
INCOMING_SIZES = {1: 37, 2: 60}


def encode_value(v):
    v = int((float(v) + 1.0) * 32767.0)
    return min(65535, max(0, v))


def encode_params(pairs):
    """pairs: iterable of (shape_index, value in [-1, 1]) -> one packet."""
    pairs = list(pairs)
    out = bytearray((MSG_PARAMS, len(pairs)))
    for key, value in pairs:
        v = encode_value(value)
        out += bytes((key, v >> 8, v & 0xFF))
    return bytes(out)


class VRCFTServer:
    """Accepts the VRCFT module and sends parameter packets to it.

    Compared to the original:
      * each update is a single sendall() of one packet (was 1 + 3*N tiny sends,
        which with Nagle enabled added latency and could interleave between
        threads), TCP_NODELAY is on,
      * a send timeout means a frozen VRCFT can't stall the inference loop,
      * a reconnecting VRCFT (e.g. after restarting VRChat) replaces the old socket,
      * the reader detects EOF instead of spinning forever on recv() == b"".
    """

    def __init__(self, host="127.0.0.1", port=26421, send_timeout=0.25):
        self.host = host
        self.port = port
        self.send_timeout = send_timeout
        self._conn = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._listener = None
        self.peer = None
        self.connections = 0
        self.packets_sent = 0
        self.last_received = {}

    @property
    def connected(self):
        return self._conn is not None

    def start(self):
        self._listener = make_listener(self.host, self.port)
        self.port = self._listener.getsockname()[1]
        threading.Thread(target=self._accept_loop, daemon=True, name="vrcft-accept").start()
        log.info("Waiting for VRCFaceTracking on %s:%d", self.host, self.port)
        return self

    def stop(self):
        self._stop.set()
        close_quietly(self._listener)
        self._drop()

    def _drop(self, conn=None):
        with self._lock:
            if conn is None or self._conn is conn:
                close_quietly(self._conn)
                self._conn = None
                self.peer = None

    def _accept_loop(self):
        while not self._stop.is_set():
            try:
                conn, addr = self._listener.accept()
            except OSError:
                if self._stop.is_set():
                    return
                time.sleep(0.2)
                continue
            tune_stream(conn)
            conn.settimeout(self.send_timeout)
            with self._lock:
                old = self._conn
                self._conn = conn
                self.peer = "%s:%d" % addr
                self.connections += 1
            if old is not None:
                close_quietly(old)
            log.info("VRCFaceTracking connected from %s", self.peer)
            threading.Thread(target=self._reader, args=(conn,), daemon=True, name="vrcft-reader").start()

    def _reader(self, conn):
        try:
            while not self._stop.is_set():
                try:
                    rid = recv_exact(conn, 1)[0]
                except socket.timeout:
                    continue  # just idle; the timeout exists for sends
                count = INCOMING_SIZES.get(rid)
                if count is None:
                    log.warning("Invalid message id %d from VRCFT", rid)
                    break
                raw = self._read_blocking(conn, count * 2)
                self.last_received[rid] = [
                    ((raw[i] << 8) | raw[i + 1]) / 32767.0 - 1.0 for i in range(0, len(raw), 2)
                ]
        except (Disconnected, OSError) as e:
            if not self._stop.is_set():
                log.info("VRCFaceTracking disconnected: %s", e)
        finally:
            self._drop(conn)

    @staticmethod
    def _read_blocking(conn, length):
        buf = bytearray()
        while len(buf) < length:
            try:
                chunk = conn.recv(length - len(buf))
            except socket.timeout:
                continue
            if not chunk:
                raise Disconnected("peer closed the connection")
            buf += chunk
        return buf

    def send_packet(self, packet):
        with self._lock:
            conn = self._conn
            if conn is None:
                return False
            try:
                conn.sendall(packet)
                self.packets_sent += 1
                return True
            except OSError as e:
                log.info("Send to VRCFT failed (%s); waiting for reconnect", e)
                close_quietly(conn)
                self._conn = None
                self.peer = None
                return False

    def send_params(self, pairs):
        return self.send_packet(encode_params(pairs))

    def snapshot(self):
        return {"connected": self.connected, "peer": self.peer, "connections": self.connections,
                "packets": self.packets_sent}
