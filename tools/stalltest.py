#!/usr/bin/env python3
"""Forced decoder-stall test of the hub's wedged-decoder reconnect, on saved clips only (no network).
Runs hub.serve() on two cameras whose 'streams' are looped saved clips, paced at 20 packets/s. In --stall mode the
first session of camera 'stalled' wedges 30 s in: its decoder still gets every packet (real NVDEC decode) but only
every 18th packet yields frames (a wedge seen in service). Expect: 'decoder starved' for that camera, a reconnect, a
healthy second session, and nothing for the control camera.
Run it in a throwaway container from your Frigate image with the GPU (no network needed); clips: 640x360 and
640x480 H.264, looped to a few minutes with `ffmpeg -stream_loop 12 -i clip.mkv -c copy long.mkv`.
usage: stalltest.py <hub dir> <stalled clip> <control clip> <secs> [--stall]"""
import json, logging, os, sys, threading, time
sys.path.insert(0, sys.argv[1])
import av, hub
stalled_clip, control_clip, secs, stall = sys.argv[2], sys.argv[3], float(sys.argv[4]), "--stall" in sys.argv
hub.RTSP_OPTS = {}  # files: their own timestamps
real_open = av.open
sessions = {}

class Pkt:
    def __init__(self, p, drop):
        self.p, self.drop = p, drop
    def decode(self):
        frames = self.p.decode()  # the real decode always runs on the GPU
        return [] if self.drop() else frames

class Box:
    def __init__(self, url, **kw):
        self.url, self.kw = url, kw
        self.c = real_open(url, **{k: v for k, v in kw.items() if k != "options"})
        self.n = sessions[url] = sessions.get(url, 0) + 1
    def __enter__(self): return self
    def __exit__(self, *a): self.c.close()
    def __getattr__(self, k): return getattr(self.c, k)
    def demux(self, st):
        t0, i = time.monotonic(), 0
        wedge = stall and self.url == stalled_clip and self.n == 1
        state = {"k": 0}
        def drop():
            if not wedge or time.monotonic() - t0 < 30:
                return False
            state["k"] += 1
            return state["k"] % 18 != 0
        for p in self.c.demux(st):
            i += 1
            d = t0 + i * 0.05 - time.monotonic()
            if d > 0:
                time.sleep(d)
            yield Pkt(p, drop)

hub.av.open = lambda url, **kw: Box(url, **kw)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
av.logging.set_level(av.logging.ERROR)
cfg = {"sock_dir": "/tmp/stallrun", "fps": 3, "cameras": [
    {"name": "stalled", "url": stalled_clip, "width": 640, "height": 360},
    {"name": "control", "url": control_clip, "width": 640, "height": 480}]}
json.dump(cfg, open("/tmp/stallcams.json", "w"))
threading.Timer(secs, lambda: os.kill(os.getpid(), 15)).start()
t = time.time()
hub.serve("/tmp/stallcams.json")
print(f"STALLTEST mode={'stall' if stall else 'nostall'} secs={secs} sessions={ {os.path.basename(k): v for k, v in sessions.items()} } wall={time.time() - t:.0f}s")
