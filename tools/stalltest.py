#!/usr/bin/env python3
"""Forced decoder-stall tests of the hub's reconnect checks, on saved clips only (no network).

Runs hub.serve() on two cameras whose "streams" are looped saved clips, paced at 20 packets/s. 30 s in, the first
session of camera "stalled" goes bad in one of two ways seen in service:
  --stall    wedged decoder: every packet still arrives and is really decoded on NVDEC, but only every 18th yields
             frames (the starve check must reconnect it);
  --trickle  trickling stream: only 1 packet every 2 s arrives (under the starve check's 20 packets per window, and
             never 15 s apart, so the read timeout does not fire either) and none yields a frame (the no-frame
             check must reconnect it);
  (neither)  control run: nothing may reconnect.
Expect for --stall/--trickle: "stalled" reconnects once and decodes again, "control" never reconnects. The last
line says RESULT=PASS or RESULT=FAIL.

Run it in a throwaway container from your Frigate image with the GPU; clips: any two short recordings from your
cameras (H.264 or H.265), looped to a few minutes with `ffmpeg -stream_loop 12 -i clip.mp4 -c copy long.mkv`.
usage: stalltest.py <hub dir> <stalled clip> <control clip> <secs> [--stall | --trickle]"""
import json, logging, os, sys, threading, time
sys.path.insert(0, sys.argv[1])
import av, hub
stalled_clip, control_clip, secs = sys.argv[2], sys.argv[3], float(sys.argv[4])
mode = "stall" if "--stall" in sys.argv else "trickle" if "--trickle" in sys.argv else "none"
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
        self.url = url
        self.c = real_open(url, **{k: v for k, v in kw.items() if k != "options"})
        self.n = sessions[url] = sessions.get(url, 0) + 1

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.c.close()

    def __getattr__(self, k):
        return getattr(self.c, k)

    def demux(self, st):
        t0, i, k = time.monotonic(), 0, 0
        bad = mode != "none" and self.url == stalled_clip and self.n == 1
        state = {"k": 0}

        def broken():
            return bad and time.monotonic() - t0 >= 30

        def drop():
            if not broken():
                return False
            if mode == "trickle":
                return True
            state["k"] += 1
            return state["k"] % 18 != 0

        for p in self.c.demux(st):
            i += 1
            d = t0 + i * 0.05 - time.monotonic()
            if d > 0:
                time.sleep(d)
            if mode == "trickle" and broken():
                k += 1
                if k % 40:  # 1 packet in 40 = one every 2 s
                    continue
            yield Pkt(p, drop)


hub.av.open = lambda url, **kw: Box(url, **kw)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
av.logging.set_level(av.logging.ERROR)
def size(path):
    with real_open(path) as c:
        cc = c.streams.video[0].codec_context
        return cc.width, cc.height


(sw, sh), (cw, ch) = size(stalled_clip), size(control_clip)
cfg = {"sock_dir": "/tmp/stallrun", "fps": 3, "cameras": [
    {"name": "stalled", "url": stalled_clip, "width": sw, "height": sh},
    {"name": "control", "url": control_clip, "width": cw, "height": ch}]}
with open("/tmp/stallcams.json", "w") as f:
    json.dump(cfg, f)
threading.Timer(secs, lambda: os.kill(os.getpid(), 15)).start()
t = time.time()
hub.serve("/tmp/stallcams.json")
s, c = sessions.get(stalled_clip, 0), sessions.get(control_clip, 0)
ok = c == 1 and (s == 1 if mode == "none" else s >= 2)
print(f"STALLTEST mode={mode} secs={secs:.0f} sessions stalled={s} control={c} wall={time.time() - t:.0f}s "
      f"RESULT={'PASS' if ok else 'FAIL'}")
