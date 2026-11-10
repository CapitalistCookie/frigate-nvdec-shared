#!/usr/bin/env python3
"""frigate-nvdec-shared: every Frigate detect stream decoded on NVDEC inside ONE process.

Why: each `ffmpeg -hwaccel cuda` detect process holds its own CUDA context (about 100 MiB on recent drivers), so N
cameras cost N contexts. This process holds one context for all cameras. It does exactly what the first half of
Frigate's NVIDIA detect chain does (`-hwaccel cuda … -vf fps=N,scale_cuda,hwdownload,format=nv12`): NVDEC decode,
pick frames by the rule of ffmpeg's `fps` filter on wall-clock timestamps, hand out NV12 frames. Frigate's
per-camera ffmpeg then reads those frames from a unix socket and runs the rest of its chain unchanged (whatever
filters followed `hwdownload,format=nv12` in that camera's detect command, and its detect output args).

Nothing is decoded on the CPU: a camera whose decoder does not come up on the GPU is an error, never a fallback.

usage: hub.py <cameras.json>                 serve (cameras.json is written by convert.py: sock_dir, fps,
                                             cameras[name, url, width, height, fps])
       hub.py --clip <file> <out.nv12> [fps]  one deterministic pass over a file, for byte-parity tests
Signals: SIGUSR1 logs one status line per camera. SIGTERM/SIGINT stop.
Exit codes 3 (a camera thread stuck) and 4 (no camera decoding at all) ask the supervisor for a restart.
"""
import ctypes
import json
import logging
import os
import re
import signal
import socket
import sys
import threading
import time
import urllib.parse

import av
from av.codec.hwaccel import HWAccel

log = logging.getLogger("hub")
STOP = threading.Event()
# primary_ctx=1: every stream's device context is the process's one primary CUDA context.
HW_OPTS = {"primary_ctx": "1"}
# Frigate's preset-rtsp-generic input options (tcp, 10 s timeout, wall-clock timestamps).
RTSP_OPTS = {
    "rtsp_transport": "tcp",
    "timeout": "10000000",
    "use_wallclock_as_timestamps": "1",
    "fflags": "+genpts+discardcorrupt",
    "user_agent": "frigate-nvdec-shared",
}
STALL_EXIT_S = 120  # one camera thread stuck this long inside a single call: exit, the supervisor restarts us
DARK_EXIT_S = 90  # no camera at all has produced a frame for this long (dead CUDA context, restreamer gone): likewise
BAD_PACKETS = 100  # consecutive packets one decoder rejected (5 s at 20 fps): reconnect that camera
# A decoder can wedge without raising: after a stream glitch NVDEC kept answering "No decoder surfaces left" and gave
# about 1 frame per 18 packets for hours while every call "succeeded". So: over each window of a session, fewer than
# 1 frame per STARVE_RATIO packets means a wedged decoder, and that camera reconnects (a fresh decoder).
STARVE_WINDOW_S = 20  # the same 20 s Frigate's own watchdog gives a detect ffmpeg without frames
STARVE_FIRST_S = 40  # the first window is longer: a session that joins mid-GOP waits for a keyframe
STARVE_RATIO = 8
STARVE_MIN_PACKETS = 20  # a window with fewer packets says nothing (the stream itself is slow or stalled)
NAME_RE = re.compile(r"^[a-zA-Z0-9_-]+$")
USER_PASS_RE = re.compile(r":\/\/[a-zA-Z0-9_-]+:[\S]+@")  # Frigate's REGEX_RTSP_CAMERA_USER_PASS


def frigate_env(environ=None, secrets_dir=None):
    """The `{FRIGATE_*}` substitutions Frigate itself makes in input paths: env vars, then secret files."""
    environ = os.environ if environ is None else environ
    env = {k: v for k, v in environ.items() if k.startswith("FRIGATE_")}
    secrets_dir = secrets_dir or environ.get("CREDENTIALS_DIRECTORY", "/run/secrets")
    if os.path.isdir(secrets_dir) and os.access(secrets_dir, os.R_OK):
        for name in os.listdir(secrets_dir):
            if name.startswith("FRIGATE_"):
                with open(os.path.join(secrets_dir, name)) as f:
                    env[name] = f.read().strip()
    return env


def stream_url(path, env):
    """The URL Frigate would hand ffmpeg for this input path: env substituted, password URL-encoded."""
    url = path.format(**env)
    m = USER_PASS_RE.search(url)
    if m:
        found = m.group(0)[3:-1]
        pw = found[found.index(":") + 1:]
        url = url.replace(pw, urllib.parse.quote_plus(pw))
    return url


def redact(text, url):
    """`text` with this camera's URL shown without its credentials (PyAV puts the URL in its error messages)."""
    return text.replace(url, re.sub(r"://[^/@]*@", "://***@", url))


def load_config(cfg, env):
    """cameras.json -> (sock_dir, [(name, url, width, height, fps)]). Refuses anything it cannot serve exactly."""
    cams, seen = [], set()
    for c in cfg["cameras"]:
        name, w, h, fps = c["name"], int(c["width"]), int(c["height"]), int(c.get("fps", cfg.get("fps", 0)))
        if not NAME_RE.match(name) or name in seen:
            raise ValueError(f"bad or duplicate camera name {name!r}")
        if w <= 0 or h <= 0 or w % 2 or h % 2 or fps <= 0:
            raise ValueError(f"{name}: need even width/height and fps > 0, got {w}x{h} fps={fps}")
        seen.add(name)
        cams.append((name, stream_url(c["url"], env), w, h, fps))
    if not cams:
        raise ValueError("no cameras")
    return cfg["sock_dir"], cams


def hwaccel():
    # is_hw_owned: decoded frames stay on the GPU; only the picked ones are downloaded, by CudaDownload below.
    return HWAccel(device_type="cuda", allow_software_fallback=False, options=HW_OPTS, is_hw_owned=True)


class CUDA_MEMCPY2D(ctypes.Structure):
    _fields_ = [(f"{side}{name}", typ) for side in ("src", "dst") for name, typ in (
        ("XInBytes", ctypes.c_size_t), ("Y", ctypes.c_size_t), ("MemoryType", ctypes.c_uint),
        ("Host", ctypes.c_void_p), ("Device", ctypes.c_uint64), ("Array", ctypes.c_void_p),
        ("Pitch", ctypes.c_size_t))] + [("WidthInBytes", ctypes.c_size_t), ("Height", ctypes.c_size_t)]


class CudaDownload:
    """GPU NV12 frame -> host bytes, the way ffmpeg's hwdownload does it (one 2D copy per plane), but called through
    ctypes, which releases Python's GIL for the duration of the copy. PyAV's own download keeps the GIL while it waits
    for the GPU; on a busy shared card those waits froze every camera thread at once, long enough for the restreamer
    to drop the hub's connections. Uses the process's primary CUDA context (the one the decoders use)."""

    HOST, DEVICE = 1, 2

    def __init__(self, device=0, lib=None):
        self.cu = lib or ctypes.CDLL("libcuda.so.1")
        self.ctx = ctypes.c_void_p()
        dev = ctypes.c_int()
        self.check(self.cu.cuInit(0), "cuInit")
        self.check(self.cu.cuDeviceGet(ctypes.byref(dev), device), "cuDeviceGet")
        self.check(self.cu.cuDevicePrimaryCtxRetain(ctypes.byref(self.ctx), dev), "cuDevicePrimaryCtxRetain")

    @staticmethod
    def check(res, what):
        if res != 0:
            raise RuntimeError(f"{what} failed: CUDA error {res}")

    def nv12(self, frame, width, height):
        if frame.format.name != "cuda" or frame.width != width or frame.height != height:
            raise RuntimeError(f"unexpected frame {frame.format.name} {frame.width}x{frame.height}")
        out = bytearray(width * height * 3 // 2)
        host = (ctypes.c_char * len(out)).from_buffer(out)
        copies = []
        for plane, rows, offset in ((frame.planes[0], height, 0), (frame.planes[1], height // 2, width * height)):
            d = CUDA_MEMCPY2D()
            d.srcMemoryType, d.srcDevice, d.srcPitch = self.DEVICE, plane.buffer_ptr, plane.line_size
            d.dstMemoryType, d.dstHost, d.dstPitch = self.HOST, ctypes.addressof(host) + offset, width
            d.WidthInBytes, d.Height = width, rows
            copies.append(d)
        self.check(self.cu.cuCtxPushCurrent_v2(self.ctx), "cuCtxPushCurrent")
        try:
            for d in copies:  # synchronous copies on the default stream, after the decoder's own copy into the frame
                self.check(self.cu.cuMemcpy2D_v2(ctypes.byref(d)), "cuMemcpy2D")
        finally:
            self.cu.cuCtxPopCurrent_v2(ctypes.byref(ctypes.c_void_p()))
        del host
        return out


_DL = None


def nv12_bytes(frame, width, height):
    global _DL
    if _DL is None:
        _DL = CudaDownload()
    return _DL.nv12(frame, width, height)


class FpsPicker:
    """Which frame goes out, and how often: the rule of ffmpeg's `fps` filter (vf_fps.c, round=near), as Frigate's
    chain applies it. Output tick n shows the latest frame whose timestamp, rounded to ticks, is <= n; the first
    frame sets tick 0; a hole in the timestamps repeats the last frame. tools/fpstest.py checks this against
    libavfilter itself on random timestamp sequences. (libavfilter is not called here: PyAV 17 leaks memory on
    every pull that returns no frame, about 1.5 MB/min with 14 cameras.)"""

    def __init__(self, time_base, fps):
        self.num, self.den = time_base.numerator * fps, time_base.denominator
        self.prev = self.next = None

    def tick(self, pts):  # av_rescale_q_rnd(pts, time_base, 1/fps, AV_ROUND_NEAR_INF)
        n = (2 * abs(pts) * self.num + self.den) // (2 * self.den)
        return n if pts >= 0 else -n

    def push(self, frame, pts):
        """Returns (frame to send, how many times) now that `frame` has arrived."""
        t = self.tick(pts)
        if self.prev is None:
            self.prev, self.next = frame, t
            return None, 0
        out, times = self.prev, max(0, t - self.next)
        self.prev, self.next = frame, self.next + times
        return out, times


class StarveCheck:
    """Counts packets in and frames out per window; raises at the first packet after a starved window."""

    def __init__(self, now):
        self.until, self.packets, self.frames = now + STARVE_FIRST_S, 0, 0

    def packet(self, now):
        if now >= self.until:
            p, f = self.packets, self.frames
            self.until, self.packets, self.frames = now + STARVE_WINDOW_S, 0, 0
            if p >= STARVE_MIN_PACKETS and f * STARVE_RATIO < p:
                raise RuntimeError(f"decoder starved: {f} frames from {p} packets in one window, reconnecting it")
        self.packets += 1

    def frame(self):
        self.frames += 1


def ts_offset(container, stream):
    """ffmpeg's default input offset: timestamps start at the input's start time (ts_offset = -start_time)."""
    if container.start_time is None:
        return 0
    return round(container.start_time / av.time_base / stream.time_base)


class Camera(threading.Thread):
    def __init__(self, name, url, width, height, fps, sock_dir):
        super().__init__(name=name, daemon=True)
        self.cam, self.url, self.width, self.height, self.fps = name, url, width, height, fps
        self.sock_path = os.path.join(sock_dir, f"{name}.sock")
        self.consumers = []
        self.lock = threading.Lock()
        self.beat = time.monotonic()
        self.last_frame = 0.0
        self.decoded = self.sent = self.sessions = self.barren = 0
        self.state = "new"
        try:
            os.unlink(self.sock_path)
        except FileNotFoundError:
            pass
        self.listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.listener.bind(self.sock_path)
        self.listener.listen(4)
        threading.Thread(target=self.accept_loop, name=f"{name}-accept", daemon=True).start()

    def accept_loop(self):
        while not STOP.is_set():
            try:
                conn, _ = self.listener.accept()
            except OSError:
                return
            conn.settimeout(5)
            with self.lock:
                self.consumers.append(conn)
                n = len(self.consumers)
            log.info("%s: reader connected (%d)", self.cam, n)

    def emit(self, data):
        with self.lock:
            conns = list(self.consumers)
        for conn in conns:
            try:
                conn.sendall(data)
                self.sent += 1
            except OSError as e:
                with self.lock:
                    if conn in self.consumers:
                        self.consumers.remove(conn)
                conn.close()
                log.info("%s: reader gone (%s)", self.cam, e.__class__.__name__)

    def session(self):
        got = 0
        self.state = "connecting"
        with av.open(self.url, options=RTSP_OPTS, timeout=(15, 15), hwaccel=hwaccel()) as c:
            st = c.streams.video[0]
            st.codec_context.thread_count = 1
            picker, off, bad = None, ts_offset(c, st), 0
            starve = StarveCheck(time.monotonic())
            for packet in c.demux(st):
                self.beat = time.monotonic()
                if STOP.is_set():
                    return got
                starve.packet(self.beat)
                try:
                    frames = packet.decode()
                    bad = 0
                except av.FFmpegError:
                    # ffmpeg logs a broken packet and decodes on (a stream joined mid-GOP does this); so do we.
                    bad += 1
                    if bad >= BAD_PACKETS:
                        raise
                    continue
                for frame in frames:
                    starve.frame()
                    if frame.pts is None:
                        continue
                    if picker is None:
                        nv12_bytes(frame, self.width, self.height)  # fail before serving anything wrong
                        picker = FpsPicker(st.time_base, self.fps)
                        self.state = "decoding"
                        log.info("%s: decoding %s %dx%d on GPU", self.cam, st.codec_context.name, frame.width, frame.height)
                    got += 1
                    self.decoded += 1
                    self.last_frame = self.beat
                    out, times = picker.push(frame, frame.pts - off)
                    if times:
                        data = nv12_bytes(out, self.width, self.height)
                        for _ in range(times):
                            self.emit(data)
        return got

    def run(self):
        while not STOP.is_set():
            self.sessions += 1
            self.beat = time.monotonic()
            try:
                got = self.session()
                err = "stream ended"
            except Exception as e:  # noqa: BLE001 - any failure of one camera means: reconnect that camera
                got, err = self.state == "decoding", f"{e.__class__.__name__}: {e}"
            self.barren = 0 if got else self.barren + 1
            self.state = "retry"
            if STOP.is_set():
                return
            log.warning("%s: session over (%s), reconnecting", self.cam, redact(err.split("; last error log")[0], self.url))
            STOP.wait(min(10, 2 * max(1, self.barren)))


def status(cams):
    for c in cams:
        log.info("status %s state=%s decoded=%d sent=%d sessions=%d readers=%d", c.cam, c.state, c.decoded, c.sent,
                 c.sessions, len(c.consumers))


def serve(cfg_path):
    with open(cfg_path) as f:
        sock_dir, specs = load_config(json.load(f), frigate_env())
    os.makedirs(sock_dir, exist_ok=True)
    cams = [Camera(name, url, w, h, fps, sock_dir) for name, url, w, h, fps in specs]
    signal.signal(signal.SIGTERM, lambda *_: STOP.set())
    signal.signal(signal.SIGINT, lambda *_: STOP.set())
    signal.signal(signal.SIGUSR1, lambda *_: status(cams))
    for c in cams:
        c.start()
    log.info("serving %d cameras at %s, fps=%s", len(cams), sock_dir, sorted({c.fps for c in cams}))
    t0 = time.monotonic()
    while not STOP.wait(5):
        now = time.monotonic()
        for c in cams:
            if now - c.beat > STALL_EXIT_S:
                log.error("%s: no progress for %ds, exiting so the supervisor restarts the hub", c.cam, STALL_EXIT_S)
                os._exit(3)
        if now - max(t0, max(c.last_frame for c in cams)) > DARK_EXIT_S:
            log.error("no camera has decoded a frame for %ds, exiting for a clean start", DARK_EXIT_S)
            os._exit(4)
    status(cams)
    log.info("stopped")


def clip(path, out_path, fps):
    """Same decode + pick code path on a file with its own timestamps."""
    n = 0
    with av.open(path, hwaccel=hwaccel()) as c, open(out_path, "wb") as out:
        st = c.streams.video[0]
        st.codec_context.thread_count = 1
        picker, off = FpsPicker(st.time_base, fps), ts_offset(c, st)
        for frame in c.decode(st):
            w, h, end = frame.width, frame.height, frame.pts - off + (frame.duration or 0)
            prev, times = picker.push(frame, frame.pts - off)
            for _ in range(times):
                out.write(nv12_bytes(prev, w, h))
            n += times
        # end of file, as the filter does it: the last frame fills the ticks up to the end timestamp
        for _ in range(max(0, picker.tick(end) - picker.next)):
            out.write(nv12_bytes(picker.prev, w, h))
            n += 1
    print(f"clip {os.path.basename(path)} {w}x{h} frames={n}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    av.logging.set_level(av.logging.ERROR)
    if len(sys.argv) >= 4 and sys.argv[1] == "--clip":
        clip(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 3)
    elif len(sys.argv) == 2:
        serve(sys.argv[1])
    else:
        sys.exit(__doc__)
