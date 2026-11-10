#!/usr/bin/env python3
"""hub.FpsPicker against libavfilter's own `fps` filter: the same output (which input frame, in which order, how
often) on random timestamp sequences with jitter, holes, bursts, equal and backward timestamps.

Needs PyAV (the Frigate image has it), no GPU. usage: python3 fpstest.py [sequences]
"""
import os
import random
import sys
from fractions import Fraction

import av
import numpy as np
from av.filter import Graph

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import hub  # noqa: E402


def lavfi(pts_list, tb, fps):
    """Input frame indices in the order libavfilter's fps filter outputs them."""
    g = Graph()
    src = g.add_buffer(width=16, height=16, format=av.VideoFormat("gray"), time_base=tb)
    f = g.add("fps", str(fps))
    sink = g.add("buffersink")
    src.link_to(f)
    f.link_to(sink)
    g.configure()
    out = []
    for i, p in enumerate(pts_list):
        a = np.zeros((16, 16), np.uint8)
        a[0, :8].view("<u8")[0] = i  # the frame carries its own index
        fr = av.VideoFrame.from_ndarray(a, format="gray")
        fr.pts, fr.time_base = p, tb
        g.vpush(fr)
        while True:
            try:
                out.append(int(g.vpull().to_ndarray()[0, :8].view("<u8")[0]))
            except (av.BlockingIOError, av.EOFError):
                break
    return out


def picker(pts_list, tb, fps):
    pk, out = hub.FpsPicker(tb, fps), []
    for i, p in enumerate(pts_list):
        fr, k = pk.push(i, p)
        out += [fr] * k
    return out


def main():
    rng = random.Random(52)
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    bad = frames = outs = 0
    for s in range(n):
        tb = rng.choice([Fraction(1, 90000), Fraction(1, 1000), Fraction(1, 12800)])
        fps = rng.choice([3, 3, 3, 5, 10])
        src_fps = rng.choice([20, 12, 25, 15, 2])
        step = tb.denominator / src_fps
        t = rng.choice([0, 0, rng.randint(0, 10 ** 6), rng.randint(10 ** 13, 2 * 10 ** 14)])  # incl. wall-clock size
        pts = []
        for _ in range(rng.randint(40, 260)):
            pts.append(int(t))
            r = rng.random()
            if r < 0.85:
                t += step * rng.uniform(0.2, 1.8)  # jitter
            elif r < 0.90:
                t += step * rng.uniform(20, 120)  # hole
            elif r < 0.95:
                pass  # equal timestamp
            else:
                t += step * rng.uniform(-3, 0.1)  # backward
        a, b = lavfi(pts, tb, fps), picker(pts, tb, fps)
        frames += len(pts)
        outs += len(a)
        if a != b:
            bad += 1
            if bad <= 3:
                first = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
                print("MISMATCH seq", s, "tb", tb, "fps", fps, "first diff at", first, len(a), len(b))
    print(f"FPSTEST sequences={n} input_frames={frames} libavfilter_outputs={outs} mismatching_sequences={bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
