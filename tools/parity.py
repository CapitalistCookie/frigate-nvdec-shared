#!/usr/bin/env python3
"""Byte parity: Frigate's own NVIDIA detect command vs hub.py + the reader command convert.py produces.

Run it in a THROWAWAY container from the same Frigate image as your frigate container, with the GPU, on short
clips saved from your detect streams (see README, "Check byte parity"). It never touches a running Frigate.

usage: python3 parity.py <clip dir> <work dir> [--fps 3]

For every clip (*.mkv, *.mp4) it writes a Frigate config with one camera per clip (detect = the clip's own size,
NVIDIA preset), renders it with Frigate's parser, converts it with convert.py and renders the result again. Then:
  A = Frigate's detect command on the clip                      -> yuv420p frames
  B = hub.py --clip on the clip -> NV12 -> the converted command -> yuv420p frames
and compares A and B byte for byte. Set FFMPEG_DISABLE_GAMMA_EQUALIZER=1 to test Frigate's other NVIDIA chain.
GPU processes run one at a time. Exit 0 when no clip differs.
"""
import argparse
import glob
import hashlib
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import convert  # noqa: E402

# Frigate's preset-rtsp-generic minus the options that only make sense on a live RTSP stream
FILE_INPUT_ARGS = "-avoid_negative_ts make_zero -fflags +genpts+discardcorrupt"


def clip_size(path):
    import av

    with av.open(path) as c:
        cc = c.streams.video[0].codec_context
        return cc.width, cc.height


def config_for(clips, fps):
    lines = ["mqtt:", "  enabled: false", "ffmpeg:", "  hwaccel_args: preset-nvidia",
             f"  input_args: {FILE_INPUT_ARGS}", "detect:", f"  fps: {fps}", "cameras:"]
    for cam, (_, w, h) in clips.items():
        lines += [f"  {cam}:", "    ffmpeg:", "      inputs:",
                  f"        - path: rtsp://127.0.0.1:8554/{cam}", "          roles:", "            - record",
                  f"        - path: rtsp://127.0.0.1:8554/{cam}_sub", "          roles:", "            - detect",
                  "    detect:", f"      width: {w}", f"      height: {h}", "    record:", "      enabled: true"]
    return "\n".join(lines) + "\n"


def swap(cmd, new_input, out):
    """The command with its -i replaced by a file and its pipe: output by `out`."""
    cmd = list(cmd)
    cmd[cmd.index("-i") + 1] = new_input
    assert cmd[-1] == "pipe:"
    return cmd[:-1] + ["-y", out]


def run(cmd, log):
    with open(log, "wb") as f:
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode:
        raise SystemExit(f"failed ({r.returncode}): see {log}")


def md5(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clips")
    ap.add_argument("work")
    ap.add_argument("--fps", type=int, default=3)
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)
    t0 = time.time()
    clips = {}
    for p in sorted(glob.glob(os.path.join(a.clips, "*.mkv")) + glob.glob(os.path.join(a.clips, "*.mp4"))):
        cam = re.sub(r"[^a-zA-Z0-9_]", "_", os.path.splitext(os.path.basename(p))[0])
        clips[cam] = (p, *clip_size(p))
    if not clips:
        raise SystemExit("no clips")
    text = config_for(clips, a.fps)
    res = convert.convert(text, convert.frigate_render, os.path.join(a.work, "nvdec-shared"))
    if res["skipped"]:
        raise SystemExit(f"converter skipped clips: {res['skipped']}")
    old, new = convert.frigate_render(text), convert.frigate_render(res["text"])
    print(f"converter: {len(res['plans'])} cameras, RENDER {res['checks']}")
    print("old detect:", " ".join(old[next(iter(clips))]["cmds"]["detect"]))
    print("new detect:", " ".join(new[next(iter(clips))]["cmds"]["detect"]))
    ok = eof = bad = 0
    for cam, (path, w, h) in clips.items():
        frame = w * h * 3 // 2
        o = os.path.join(a.work, cam)
        run(swap(old[cam]["cmds"]["detect"], path, o + ".A.yuv"), o + ".A.log")
        run([sys.executable, os.path.join(os.path.dirname(HERE), "hub.py"), "--clip", path, o + ".nv12", str(a.fps)],
            o + ".hub.log")
        run(swap(new[cam]["cmds"]["detect"], o + ".nv12", o + ".B.yuv"), o + ".B.log")
        A, B = open(o + ".A.yuv", "rb").read(), open(o + ".B.yuv", "rb").read()
        na, nb = len(A) // frame, len(B) // frame
        if A == B:
            verdict, ok = "IDENTICAL", ok + 1
        elif A.startswith(B) and na - nb == 1:
            # ffmpeg's fps filter can emit one more frame when a FILE ends; a live stream never ends this way
            verdict, eof = f"FIRST-{nb}-FRAMES-IDENTICAL, A has 1 end-of-file frame more", eof + 1
        else:
            verdict, bad = "DIFFERENT", bad + 1
        print(f"{cam} {w}x{h} A={md5(o + '.A.yuv')[:12]} frames={na} B={md5(o + '.B.yuv')[:12]} frames={nb} {verdict}")
        for ext in (".A.yuv", ".B.yuv", ".nv12"):
            os.remove(o + ext)
    print(f"PARITY fps={a.fps} gamma_eq={'off' if os.environ.get('FFMPEG_DISABLE_GAMMA_EQUALIZER') else 'frigate-default'}"
          f" identical={ok} eof_only={eof} different={bad} wall={time.time() - t0:.1f}s")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
