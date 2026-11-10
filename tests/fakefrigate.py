"""A small stand-in for Frigate's command rendering (frigate/config/camera/camera.py `_get_ffmpeg_cmd` and
frigate/ffmpeg_presets.py, Frigate 0.17), enough to test convert.py without Frigate installed. It covers the
presets the tests use; the real parser is used when convert.py runs inside the frigate container."""
import shlex

import yaml

GLOBAL = ["-hide_banner", "-loglevel", "warning", "-threads", "2"]
DETECT_OUT = ["-threads", "2", "-f", "rawvideo", "-pix_fmt", "yuv420p"]
UA = ["-user_agent", "FFmpeg Frigate/0.17.1"]
INPUT = {"preset-rtsp-generic": UA + ["-avoid_negative_ts", "make_zero", "-fflags", "+genpts+discardcorrupt",
                                      "-rtsp_transport", "tcp", "-timeout", "10000000",
                                      "-use_wallclock_as_timestamps", "1"],
         "preset-rtsp-restream": UA + ["-rtsp_transport", "tcp", "-timeout", "10000000"]}
DECODE = {"preset-nvidia": "-hwaccel_device {3} -hwaccel cuda -hwaccel_output_format cuda"}
SCALE = {"preset-nvidia": "-r {0} -vf fps={0},scale_cuda=w={1}:h={2},hwdownload,format=nv12,eq=gamma=1.4:gamma_weight=0.5",
         "preset-vaapi": "-r {0} -vf fps={0},scale_vaapi=w={1}:h={2},hwdownload,format=nv12,eq=gamma=1.4:gamma_weight=0.5",
         "default": "-r {0} -vf fps={0},scale={1}:{2}"}
RECORD = {"preset-record-generic-audio-aac": ["-f", "segment", "-segment_time", "10", "-segment_format", "mp4",
                                              "-reset_timestamps", "1", "-strftime", "1", "-c:v", "copy", "-c:a", "aac"]}


def arglist(a):
    return a if isinstance(a, list) else shlex.split(a)


def scale(arg, detect_args, fps, w, h, gamma_off):
    s = SCALE["default"] if not isinstance(arg, str) or " " in arg else SCALE.get(arg, SCALE["default"])
    if gamma_off and ",hwdownload,format=nv12,eq=gamma=1.4:gamma_weight=0.5" in s:
        s = s.replace(",hwdownload,format=nv12,eq=gamma=1.4:gamma_weight=0.5",
                      ":format=nv12,hwdownload,format=nv12,format=yuv420p")
    return s.format(fps, w, h).split(" ") + detect_args


def render(text, gamma_off=False):
    cfg = yaml.safe_load(text)
    g_ff, g_det = cfg.get("ffmpeg") or {}, cfg.get("detect") or {}
    out = {}
    for name, cam in cfg["cameras"].items():
        ff = {**g_ff, **cam["ffmpeg"]}
        oa = {**(g_ff.get("output_args") or {}), **(cam["ffmpeg"].get("output_args") or {})}
        det = {**g_det, **(cam.get("detect") or {})}
        # Frigate probes the stream when a detect input has no width/height; a fixed size stands in for that here
        w, h, fps = det.get("width", 1920), det.get("height", 1080), det.get("fps", 5)
        cam_hw = ff.get("hwaccel_args", "auto")
        cmds = {}
        di = None
        for inp in cam["ffmpeg"]["inputs"]:
            roles = inp["roles"]
            outs = []
            if "detect" in roles:
                di = inp
                outs = scale(inp.get("hwaccel_args") or cam_hw, arglist(oa.get("detect", DETECT_OUT)), fps, w, h,
                             gamma_off) + ["pipe:"]
            if "record" in roles:
                outs = RECORD["preset-record-generic-audio-aac"] + [f"/tmp/cache/{name}@%Y%m%d%H%M%S%z.mp4"] + outs
            hwi = inp.get("hwaccel_args")
            hw = (DECODE[hwi].format(fps, w, h, 0).split(" ") if hwi in DECODE else arglist(hwi) if hwi else
                  DECODE[cam_hw].format(fps, w, h, 0).split(" ") if cam_hw in DECODE else
                  arglist(cam_hw) if cam_hw != "auto" else [])
            ia = inp.get("input_args") or ff.get("input_args", "preset-rtsp-generic")
            ia = INPUT.get(ia) if isinstance(ia, str) and ia in INPUT else arglist(ia)
            gl = arglist(inp.get("global_args") or ff.get("global_args", GLOBAL))
            cmd = (["/usr/lib/ffmpeg/7.0/bin/ffmpeg"] + gl + (hw if "detect" in roles else []) + ia
                   + ["-i", inp["path"]] + outs)
            cmds["+".join(sorted(roles))] = [p for p in cmd if p != ""]
        out[name] = {"cmds": cmds, "width": w, "height": h, "fps": fps,
                     "global_args": arglist(di.get("global_args") or ff.get("global_args", GLOBAL))}
    return out
