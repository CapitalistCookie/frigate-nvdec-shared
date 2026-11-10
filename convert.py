#!/usr/bin/env python3
"""Point Frigate's NVIDIA detect inputs at frigate-nvdec-shared, validate the result, and roll it back.

Run it inside the frigate container: it renders every camera's ffmpeg commands with Frigate's own config parser.

  docker exec frigate python3 /config/nvdec-shared/convert.py plan       show the change; writes nothing
  docker exec frigate python3 /config/nvdec-shared/convert.py apply      back up config.yml, then write config.yml,
                                                                         cameras.json and applied.json
  docker exec frigate python3 /config/nvdec-shared/convert.py rollback   put the backup back

Options: --config PATH   Frigate config (default: $CONFIG_FILE, else /config/config.yml or /config/config.yaml)
         --base DIR      this tool's directory as the frigate container sees it (default /config/nvdec-shared)
         --cameras a,b   convert only these cameras
         --force         rollback even though config.yml changed after apply

A camera is converted when its detect input has the detect role alone, reads an rtsp:// or rtsps:// stream, sets
detect width and height, and Frigate renders its detect command with the NVIDIA preset chain
`fps=F,scale_cuda=w=W:h=H[:format=nv12],hwdownload,format=nv12[,REST]` on GPU 0. Its detect input then reads NV12
frames from a unix socket served by hub.py, and its detect output args become `-vf REST` (or `-vf null`) followed by
the detect output args it had. Frigate's reader ffmpeg therefore runs the same filters after the download, on the
same frames. Every other camera, and every non-detect command, is left as it is; the tool checks both by rendering
the old and new config with Frigate's parser before it writes anything.

The edit is textual: comments, ordering and formatting of everything else stay byte-for-byte.
"""
import argparse
import collections
import copy
import datetime
import difflib
import hashlib
import json
import os
import re
import shutil
import sys

import yaml

DEFAULT_BASE = "/config/nvdec-shared"
CHAIN_RE = re.compile(r"^fps=(\d+),scale_cuda=w=(\d+):h=(\d+)(?::format=nv12)?,hwdownload,format=nv12(?:,(.+))?$")
READER_HWACCEL = ["-hwaccel", "none"]
FILTER_OPTS = {"-vf", "-filter:v", "-filter_complex", "-lavfi"}
SAFE_TOKEN = re.compile(r"^[^\s\"'\\#]+$")
MARK = "# frigate-nvdec-shared: detect frames decoded by the frigate-nvdec-shared container. Undo: convert.py rollback"


class Skip(Exception):
    """This camera is left as it is, for the reason given."""


def sha256(text):
    return hashlib.sha256(text.encode()).hexdigest()


def sock_path(base, cam):
    return f"unix:{base}/run/{cam}.sock"


def reader_input_args(width, height, fps):
    # -probesize 32 -analyzeduration 0: without them the reader buffers seconds of frames before its first output.
    return ["-probesize", "32", "-analyzeduration", "0", "-f", "rawvideo", "-pix_fmt", "nv12",
            "-video_size", f"{width}x{height}", "-framerate", str(fps)]


def args_value(tokens):
    """The YAML value for an ffmpeg argument list: one string when Frigate's shlex split gives the tokens back
    exactly, otherwise a list."""
    if all(SAFE_TOKEN.match(t) for t in tokens):
        return " ".join(tokens)
    return list(tokens)


def redact(text):
    return re.sub(r"://[^/\s@]*@", "://***@", text)


# ---------------------------------------------------------------- what Frigate renders, and what it must render after


def split_detect(cmd):
    """A detect-only Frigate command -> (args before -i, input path, fps, -vf chain, detect output args)."""
    if cmd.count("-i") != 1:
        raise Skip("detect command has more than one -i")
    i = cmd.index("-i")
    tail = cmd[i + 2:]
    if len(tail) < 5 or tail[0] != "-r" or tail[2] != "-vf" or tail[-1] != "pipe:":
        raise Skip("detect command is not `-r F -vf CHAIN <detect args> pipe:`")
    return cmd[:i], cmd[i + 1], tail[1], tail[3], tail[4:-1]


def plan_camera(name, info, base):
    """What the reader needs for this camera, or Skip. `info` is one camera of a render() result."""
    head, _, fps, chain, out_args = split_detect(info["cmds"]["detect"])
    w, h = info["width"], info["height"]
    m = CHAIN_RE.match(chain)
    if not m:
        raise Skip(f"detect chain is not Frigate's NVIDIA preset: {chain}")
    if (m.group(1), m.group(2), m.group(3)) != (str(info["fps"]), str(w), str(h)) or fps != str(info["fps"]):
        raise Skip(f"detect chain {chain} does not match detect {w}x{h} at {info['fps']} fps")
    if FILTER_OPTS & set(out_args):
        raise Skip("detect output args carry their own filter; convert this camera by hand")
    if "-hwaccel_device" in head and head[head.index("-hwaccel_device") + 1] != "0":
        raise Skip("decodes on a GPU other than 0")
    if w % 2 or h % 2:
        raise Skip(f"odd detect size {w}x{h}")
    rest = m.group(4) or "null"
    hw = READER_HWACCEL
    inp = reader_input_args(w, h, fps)
    out = ["-vf", rest] + out_args
    sock = sock_path(base, name)
    glob = info["global_args"]
    if head[1:1 + len(glob)] != glob:
        raise Skip("cannot find the global args in the detect command")
    expect = (head[:1 + len(glob)] + hw + inp + ["-i", sock, "-r", fps, "-vf", f"fps={fps},scale={w}:{h}"] + out
              + ["pipe:"])
    # hub.py opens every stream with Frigate's preset-rtsp-generic options, which stamp frames with wall-clock time
    note = "" if "-use_wallclock_as_timestamps" in head else (
        "Frigate reads this stream without -use_wallclock_as_timestamps (e.g. preset-rtsp-restream); the hub always "
        "uses wall-clock timestamps, so frame selection can differ from Frigate's own")
    return {"sock": sock, "hwaccel_args": hw, "input_args": inp, "detect_args": out, "expect": expect,
            "rest": rest, "width": w, "height": h, "fps": int(fps), "note": note}


# ---------------------------------------------------------------- where the YAML text has to change


def mget(node, key):
    """(key node, value node) of a key in a block mapping node, or (None, None)."""
    if not isinstance(node, yaml.MappingNode):
        return None, None
    for k, v in node.value:
        if isinstance(k, yaml.ScalarNode) and k.value == key:
            return k, v
    return None, None


def has_merge(node):
    return isinstance(node, yaml.MappingNode) and any(k.value == "<<" for k, _ in node.value)


def block_mapping(node, what):
    if not isinstance(node, yaml.MappingNode) or node.flow_style or has_merge(node):
        raise Skip(f"{what} is not a plain block mapping (flow style, anchor merge or empty)")
    return node


def locate(text, data):
    """camera -> its YAML nodes to edit, or the Skip reason."""
    root = yaml.compose(text)
    refs = [t.start_mark.index for t in yaml.scan(text) if isinstance(t, (yaml.AnchorToken, yaml.AliasToken))]
    _, cams = mget(root, "cameras")
    out = {}
    for knode, cnode in (cams.value if isinstance(cams, yaml.MappingNode) else []):
        name = knode.value
        try:
            # An alias, or an anchor another part of the file may reuse: an edit here would not stay in this camera.
            if cnode.start_mark.index < knode.end_mark.index or any(
                    knode.start_mark.index <= r < cnode.end_mark.index for r in refs):
                raise Skip("camera uses YAML anchors or aliases")
            cam = block_mapping(cnode, "camera")
            _, ff = mget(cam, "ffmpeg")
            ff = block_mapping(ff, "ffmpeg")
            _, inputs = mget(ff, "inputs")
            if not isinstance(inputs, yaml.SequenceNode):
                raise Skip("ffmpeg.inputs is not a list")
            raw = data["cameras"][name]["ffmpeg"]["inputs"]
            idx = [i for i, inp in enumerate(raw) if "detect" in (inp.get("roles") or [])]
            if len(idx) != 1:
                raise Skip("no single detect input")
            raw_in = raw[idx[0]]
            if list(raw_in["roles"]) != ["detect"]:
                raise Skip(f"detect input also has roles {[r for r in raw_in['roles'] if r != 'detect']}; "
                           "give detect its own input first")
            path = str(raw_in.get("path", ""))
            if path.startswith("unix:"):
                raise Skip("already converted")
            if not re.match(r"^rtsps?://", path):
                raise Skip("detect input is not an rtsp:// stream")
            det = {**(data.get("detect") or {}), **(data["cameras"][name].get("detect") or {})}
            if not det.get("width") or not det.get("height"):
                raise Skip("set detect.width and detect.height first")
            item = block_mapping(inputs.value[idx[0]], "detect input")
            pk, pv = mget(item, "path")
            if pv is None or not isinstance(pv, yaml.ScalarNode):
                raise Skip("detect input path is not a plain value")
            _, oa = mget(ff, "output_args")
            if oa is not None:
                block_mapping(oa, "ffmpeg.output_args")
            out[name] = {"camera": cam, "ffmpeg": ff, "item": item, "path": (pk, pv),
                         "ffmpeg_key": mget(cam, "ffmpeg")[0], "hwaccel_args": mget(item, "hwaccel_args"),
                         "input_args": mget(item, "input_args"), "output_args": oa}
        except Skip as e:
            out[name] = e
    return out


def line_start(text, idx):
    return text.rfind("\n", 0, idx) + 1


def line_end(text, idx):
    j = text.find("\n", idx)
    return len(text) if j < 0 else j + 1


def edits_for(text, loc, plan):
    """(start, end, replacement) text edits for one camera."""
    ed = []
    yv = lambda v: json.dumps(v)  # a JSON string or list is a valid YAML flow scalar or sequence

    pk, pv = loc["path"]
    ed.append((pk.end_mark.index, pv.end_mark.index, f": {yv(plan['sock'])}"))
    after_path = line_end(text, pv.end_mark.index)
    col = pk.start_mark.column
    for key in ("hwaccel_args", "input_args"):
        k, v = loc[key]
        val = yv(args_value(plan[key]))
        if k is not None:
            ed.append((k.end_mark.index, v.end_mark.index, f": {val}"))
        else:
            ed.append((after_path, after_path, f"{' ' * col}{key}: {val}\n"))
    item = loc["item"]
    s = line_start(text, item.start_mark.index)
    lead = re.match(r"[ \t]*", text[s:]).group(0)
    ed.append((s, s, f"{lead}{MARK}\n"))

    detect = args_value(plan["detect_args"])
    oa, ff = loc["output_args"], loc["ffmpeg"]
    if oa is not None:
        dk, dv = mget(oa, "detect")
        if dk is not None:
            ed.append((dk.end_mark.index, dv.end_mark.index, f": {yv(detect)}"))
        else:
            first = oa.value[0][0]
            p = line_start(text, first.start_mark.index)
            ed.append((p, p, f"{' ' * first.start_mark.column}detect: {yv(detect)}\n"))
    else:
        first = ff.value[0][0]
        fcol = first.start_mark.column
        step = fcol - loc["ffmpeg_key"].start_mark.column
        step = step if step > 0 else 2
        p = line_start(text, first.start_mark.index)
        if text[p:first.start_mark.index].strip():
            raise Skip("ffmpeg mapping layout not understood")
        ed.append((p, p, f"{' ' * fcol}output_args:\n{' ' * (fcol + step)}detect: {yv(detect)}\n"))
    return ed


def apply_edits(text, edits):
    edits = sorted(edits, key=lambda e: (e[0], e[1]))
    for a, b in zip(edits, edits[1:]):
        if a[1] > b[0]:
            raise AssertionError("overlapping edits")
    for s, e, r in reversed(edits):
        text = text[:s] + r + text[e:]
    return text


# ---------------------------------------------------------------- the whole conversion


def convert(text, render, base=DEFAULT_BASE, only=None):
    """-> dict(new text, cameras.json, per-camera plans, skipped reasons, checks). Raises on any failed check."""
    data = yaml.safe_load(text)
    locs = locate(text, data)
    old = render(text)
    plans, skipped, edits = {}, {}, []
    for name in data["cameras"]:
        loc = locs.get(name, Skip("not found in the YAML"))
        try:
            if only is not None and name not in only:
                raise Skip("not selected")
            if isinstance(loc, Skip):
                raise loc
            plan = plan_camera(name, old[name], base)
            edits += edits_for(text, loc, plan)
            plans[name] = plan
        except Skip as e:
            skipped[name] = str(e)
    if only is not None and set(only) - set(data["cameras"]):
        raise SystemExit(f"unknown camera(s): {sorted(set(only) - set(data['cameras']))}")
    if not plans:
        return {"text": text, "plans": plans, "skipped": skipped, "cameras_json": None, "checks": {}}
    new = apply_edits(text, edits)
    check_yaml(text, new, data, plans)
    checks = check_render(old, render(new), plans)
    cams = {"sock_dir": f"{base}/run", "cameras": []}
    for name, plan in plans.items():
        path = next(i for i in data["cameras"][name]["ffmpeg"]["inputs"] if "detect" in i["roles"])["path"]
        cams["cameras"].append({"name": name, "url": path, "width": plan["width"], "height": plan["height"],
                                "fps": plan["fps"]})
    return {"text": new, "plans": plans, "skipped": skipped, "cameras_json": cams, "checks": checks}


def check_yaml(old, new, data, plans):
    """The parsed new YAML is exactly the old one plus the intended change, and no comment was lost."""
    exp = copy.deepcopy(data)
    for name, plan in plans.items():
        ff = exp["cameras"][name]["ffmpeg"]
        inp = next(i for i in ff["inputs"] if "detect" in i["roles"])
        inp.update(path=plan["sock"], hwaccel_args=args_value(plan["hwaccel_args"]),
                   input_args=args_value(plan["input_args"]))
        ff.setdefault("output_args", {})["detect"] = args_value(plan["detect_args"])
    got = yaml.safe_load(new)
    if got != exp:
        raise AssertionError("the parsed YAML differs from the intended change")
    comments = lambda t: collections.Counter(ln.strip() for ln in t.split("\n") if ln.strip().startswith("#"))
    lost = comments(old) - comments(new)
    if lost:
        raise AssertionError(f"comments would be lost: {[redact(c) for c in lost]}")


def check_render(old, new, plans):
    """With Frigate's parser: every non-detect command and every unconverted camera unchanged, every converted
    detect command exactly the reader command."""
    if set(old) != set(new):
        raise AssertionError("camera set changed")
    same = 0
    for name in old:
        o, n = old[name]["cmds"], new[name]["cmds"]
        if set(o) != set(n):
            raise AssertionError(f"{name}: roles changed {sorted(o)} -> {sorted(n)}")
        for role in o:
            if role == "detect" and name in plans:
                if n[role] != plans[name]["expect"]:
                    raise AssertionError(f"{name}: rendered detect command is not the reader command:\n"
                                         f"  got  {redact(' '.join(n[role]))}\n"
                                         f"  want {redact(' '.join(plans[name]['expect']))}")
            elif o[role] != n[role]:
                raise AssertionError(f"{name}: {role} command changed")
            else:
                same += 1
    return {"other_cmds_unchanged": same, "detect_as_expected": len(plans)}


def frigate_render(text):
    """camera -> {cmds: {roles: cmd}, width, height, fps, global_args}, as Frigate itself renders them."""
    sys.path.insert(0, "/opt/frigate")
    from frigate.config import FrigateConfig
    from frigate.util.builtin import get_ffmpeg_arg_list

    fc = FrigateConfig.parse_yaml(text)
    out = {}
    for name, cam in fc.cameras.items():
        di = next(i for i in cam.ffmpeg.inputs if "detect" in i.roles)
        out[name] = {"cmds": {"+".join(sorted(r.value for r in c["roles"])): c["cmd"] for c in cam.ffmpeg_cmds},
                     "width": cam.detect.width, "height": cam.detect.height, "fps": cam.detect.fps,
                     "global_args": get_ffmpeg_arg_list(di.global_args or cam.ffmpeg.global_args)}
    return out


# ---------------------------------------------------------------- files


def write_inplace(path, text):
    """Rewrite keeping inode, owner and mode (Frigate's config may be a bind-mounted single file)."""
    with open(path, "r+") as f:
        f.write(text)
        f.truncate()


def write_private(path, text):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(text)


def default_config():
    if os.environ.get("CONFIG_FILE"):
        return os.environ["CONFIG_FILE"]
    for p in ("/config/config.yml", "/config/config.yaml"):
        if os.path.exists(p):
            return p
    return "/config/config.yml"


def report(res, old_text):
    for name, plan in res["plans"].items():
        print(f"CONVERT {name} {plan['width']}x{plan['height']} fps={plan['fps']} after-download filters: "
              f"{plan['rest']} -> {plan['sock']}")
        if plan["note"]:
            print(f"  NOTE  {name}: {plan['note']}")
    for name, why in res["skipped"].items():
        print(f"SKIP    {name}: {why}")
    if not res["plans"]:
        return
    diff = difflib.unified_diff(old_text.split("\n"), res["text"].split("\n"), "config (now)", "config (new)",
                                lineterm="", n=1)
    print("\n".join(redact(ln) for ln in diff))
    c = res["checks"]
    print(f"RENDER other_cmds_unchanged={c['other_cmds_unchanged']} detect_as_expected={c['detect_as_expected']}"
          f"/{len(res['plans'])}")
    print(f"old_sha256={sha256(old_text)} candidate_sha256={sha256(res['text'])}")


def main(argv=None, render=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["plan", "apply", "rollback"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--cameras", default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    cfg = a.config or default_config()
    base = a.base.rstrip("/")
    applied = os.path.join(base, "applied.json")
    if a.mode == "rollback":
        return rollback(a.config, applied, a.force)
    if a.mode == "apply" and os.path.exists(applied):
        sys.exit(f"{applied} exists: roll back first (or move it away if that conversion was undone by hand)")
    with open(cfg) as f:
        old = f.read()
    res = convert(old, render or frigate_render, base, a.cameras.split(",") if a.cameras else None)
    report(res, old)
    if not res["plans"]:
        print("NOTHING TO CONVERT")
        return 1
    if a.mode == "plan":
        print("PLAN OK (nothing written)")
        return 0
    os.makedirs(base, exist_ok=True)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = f"{cfg}.pre-nvdec-shared-{stamp}"
    shutil.copy2(cfg, backup)
    with open(backup) as f:
        assert sha256(f.read()) == sha256(old), "backup does not match"
    write_private(os.path.join(base, "cameras.json"), json.dumps(res["cameras_json"], indent=1) + "\n")
    write_private(applied, json.dumps({"config": cfg, "backup": backup, "before_sha256": sha256(old),
                                       "after_sha256": sha256(res["text"]), "utc": stamp,
                                       "cameras": sorted(res["plans"])}, indent=1) + "\n")
    write_inplace(cfg, res["text"])
    with open(cfg) as f:
        got = sha256(f.read())
    assert got == sha256(res["text"]), "config.yml does not read back as written"
    print(f"APPLIED config={cfg} sha256={got} backup={backup}")
    print("next: start-container.sh on the docker host, then restart Frigate")
    return 0


def rollback(cfg, applied, force):
    if not os.path.exists(applied):
        sys.exit(f"no {applied}: nothing to roll back")
    with open(applied) as f:
        rec = json.load(f)
    cfg = rec["config"] if cfg is None else cfg
    with open(rec["backup"]) as f:
        orig = f.read()
    if sha256(orig) != rec["before_sha256"]:
        sys.exit(f"backup {rec['backup']} changed since apply; restore by hand")
    with open(cfg) as f:
        now = sha256(f.read())
    if now != rec["after_sha256"] and not force:
        sys.exit(f"{cfg} changed since apply (sha256 {now}); your later edits would be lost. "
                 "Use --force to restore the backup anyway.")
    write_inplace(cfg, orig)
    with open(cfg) as f:
        got = sha256(f.read())
    assert got == rec["before_sha256"], "config does not read back as the backup"
    os.replace(applied, f"{applied}.rolled-back-{datetime.datetime.now(datetime.timezone.utc):%Y%m%dT%H%M%SZ}")
    print(f"ROLLED BACK config={cfg} sha256={got} (backup kept: {rec['backup']})")
    print("next: restart Frigate; then `docker rm -f frigate-nvdec-shared` on the docker host")
    return 0


if __name__ == "__main__":
    sys.exit(main())
