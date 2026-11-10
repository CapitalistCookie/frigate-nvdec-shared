import difflib
import functools
import json
import os
import shlex

import pytest
import yaml

import convert
import fakefrigate

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE = open(os.path.join(HERE, "sample_config.yml")).read()
BASE = "/config/nvdec-shared"


def run(text=SAMPLE, gamma_off=False, **kw):
    return convert.convert(text, functools.partial(fakefrigate.render, gamma_off=gamma_off), BASE, **kw)


def test_which_cameras_convert_and_why_the_rest_do_not():
    res = run()
    assert sorted(res["plans"]) == ["driveway", "front_door", "side_gate"]
    assert "also has roles ['record']" in res["skipped"]["backyard"]
    assert "not Frigate's NVIDIA preset" in res["skipped"]["garage"]
    assert "detect.width" in res["skipped"]["porch"]


def test_detect_command_is_the_reader_with_the_same_filters_and_output_args():
    res = run()
    new = fakefrigate.render(res["text"])
    cmd = new["front_door"]["cmds"]["detect"]
    assert cmd == ["/usr/lib/ffmpeg/7.0/bin/ffmpeg"] + fakefrigate.GLOBAL + [
        "-hwaccel", "none", "-probesize", "32", "-analyzeduration", "0", "-f", "rawvideo", "-pix_fmt", "nv12",
        "-video_size", "640x360", "-framerate", "5", "-i", "unix:/config/nvdec-shared/run/front_door.sock",
        "-r", "5", "-vf", "fps=5,scale=640:360",
        "-vf", "eq=gamma=1.4:gamma_weight=0.5", "-threads", "2", "-f", "rawvideo", "-pix_fmt", "yuv420p", "pipe:"]
    # the camera's own detect output args survive
    side = new["side_gate"]["cmds"]["detect"]
    assert side[side.index("pipe:") - 6:side.index("pipe:")] == ["-threads", "1", "-f", "rawvideo", "-pix_fmt", "yuv420p"]
    assert "-video_size" in side and side[side.index("-video_size") + 1] == "896x512"
    assert new["driveway"]["cmds"]["detect"][new["driveway"]["cmds"]["detect"].index("-framerate") + 1] == "3"


def test_filters_after_download_are_carried_not_built_in():
    # Frigate drops its gamma eq when FFMPEG_DISABLE_GAMMA_EQUALIZER is set; the reader must follow, not add one.
    res = run(gamma_off=True)
    assert "eq=" not in res["text"]
    assert {p["rest"] for p in res["plans"].values()} == {"format=yuv420p"}
    assert {p["rest"] for p in run()["plans"].values()} == {"eq=gamma=1.4:gamma_weight=0.5"}


def test_note_when_frigate_did_not_use_wall_clock_timestamps():
    res = run()
    assert "wall-clock" in res["plans"]["front_door"]["note"]  # preset-rtsp-restream
    assert res["plans"]["side_gate"]["note"] == ""  # preset-rtsp-generic


def test_no_filters_after_download_gives_null():
    info = {"cmds": {"detect": ["ff", "-g", "-i", "rtsp://x", "-r", "5", "-vf",
                                "fps=5,scale_cuda=w=640:h=360,hwdownload,format=nv12", "-f", "rawvideo", "pipe:"]},
            "width": 640, "height": 360, "fps": 5, "global_args": ["-g"]}
    p = convert.plan_camera("cam", info, BASE)
    assert p["detect_args"] == ["-vf", "null", "-f", "rawvideo"]
    assert p["expect"][:3] == ["ff", "-g", "-hwaccel"]


@pytest.mark.parametrize("chain,out,why", [
    ("fps=5,scale_cuda=w=640:h=480,hwdownload,format=nv12", [], "does not match"),
    ("fps=5,scale=640:360", [], "not Frigate's NVIDIA preset"),
    ("fps=5,scale_cuda=w=640:h=360,hwdownload,format=nv12", ["-vf", "hflip"], "their own filter"),
])
def test_plan_camera_refuses(chain, out, why):
    info = {"cmds": {"detect": ["ff", "-i", "rtsp://x", "-r", "5", "-vf", chain] + out + ["pipe:"]},
            "width": 640, "height": 360, "fps": 5, "global_args": []}
    with pytest.raises(convert.Skip, match=why):
        convert.plan_camera("cam", info, BASE)


def test_other_gpu_is_refused():
    info = {"cmds": {"detect": ["ff", "-hwaccel_device", "1", "-hwaccel", "cuda", "-i", "rtsp://x", "-r", "5", "-vf",
                                "fps=5,scale_cuda=w=640:h=360,hwdownload,format=nv12", "pipe:"]},
            "width": 640, "height": 360, "fps": 5, "global_args": []}
    with pytest.raises(convert.Skip, match="GPU other than 0"):
        convert.plan_camera("cam", info, BASE)


def test_only_the_intended_keys_change_and_comments_stay():
    res = run()
    old, new = SAMPLE.split("\n"), res["text"].split("\n")
    sm = difflib.SequenceMatcher(None, old, new, autojunk=False)
    removed = [ln for tag, i1, i2, _, _ in sm.get_opcodes() if tag in ("replace", "delete") for ln in old[i1:i2]]
    assert sorted(ln.strip().split(":")[0].lstrip("- ").strip() for ln in removed) == ["detect", "hwaccel_args", "input_args", "path", "path", "path"]
    assert "        - path: \"unix:/config/nvdec-shared/run/front_door.sock\"   # 640x360" in new
    for ln in old:
        if ln.strip().startswith("#"):
            assert ln in new
    # every key the converter added sits at its mapping's own indentation (the 4-space camera too)
    assert '                  hwaccel_args: "-hwaccel none"' in new
    assert '              detect: "-vf eq=gamma=1.4:gamma_weight=0.5 -threads 1 -f rawvideo -pix_fmt yuv420p"' in new


def test_parsed_yaml_is_the_old_one_plus_the_change():
    res = run()
    a, b = yaml.safe_load(SAMPLE), yaml.safe_load(res["text"])
    for name in ("backyard", "garage", "porch"):
        assert a["cameras"][name] == b["cameras"][name]
    assert {k: v for k, v in a.items() if k != "cameras"} == {k: v for k, v in b.items() if k != "cameras"}
    fd = b["cameras"]["front_door"]["ffmpeg"]
    assert fd["inputs"][0] == a["cameras"]["front_door"]["ffmpeg"]["inputs"][0]
    assert fd["output_args"] == {"detect": "-vf eq=gamma=1.4:gamma_weight=0.5 -threads 2 -f rawvideo -pix_fmt yuv420p"}
    assert b["cameras"]["driveway"]["ffmpeg"]["output_args"]["record"] == "preset-record-generic-audio-aac"


def test_record_and_unconverted_commands_unchanged():
    res = run()
    old, new = fakefrigate.render(SAMPLE), fakefrigate.render(res["text"])
    for name in old:
        for role in old[name]["cmds"]:
            if not (role == "detect" and name in res["plans"]):
                assert old[name]["cmds"][role] == new[name]["cmds"][role]
    assert res["checks"]["detect_as_expected"] == 3


def test_check_render_catches_a_wrong_reader_command():
    res = run()
    plans = {k: dict(v) for k, v in res["plans"].items()}
    plans["driveway"]["expect"] = plans["driveway"]["expect"][:-2] + ["pipe:"]
    with pytest.raises(AssertionError, match="driveway"):
        convert.check_render(fakefrigate.render(SAMPLE), fakefrigate.render(res["text"]), plans)


def test_check_yaml_catches_an_unintended_change():
    res = run()
    with pytest.raises(AssertionError, match="differs"):
        convert.check_yaml(SAMPLE, res["text"].replace("fps: 3", "fps: 4"), yaml.safe_load(SAMPLE), res["plans"])
    with pytest.raises(AssertionError, match="comments"):
        convert.check_yaml(SAMPLE, res["text"].replace("# substream for detection", ""), yaml.safe_load(SAMPLE),
                           res["plans"])


def test_converting_twice_changes_nothing():
    res2 = run(res_text := run()["text"])
    assert res2["plans"] == {} and res2["text"] == res_text
    assert res2["skipped"]["front_door"] == "already converted"


def test_camera_list_is_generated_from_the_config():
    cams = run()["cameras_json"]
    assert cams["sock_dir"] == "/config/nvdec-shared/run"
    by = {c["name"]: c for c in cams["cameras"]}
    assert by["front_door"] == {"name": "front_door", "url": "rtsp://127.0.0.1:8554/front_door_sub",
                                "width": 640, "height": 360, "fps": 5}
    assert by["driveway"]["height"] == 480 and by["driveway"]["fps"] == 3
    # placeholders stay placeholders: the hub substitutes them like Frigate does, nothing secret is written
    assert by["side_gate"]["url"].startswith("rtsp://{FRIGATE_CAM_USER}:{FRIGATE_CAM_PASS}@")


def test_only_selected_cameras():
    res = run(only=["driveway"])
    assert list(res["plans"]) == ["driveway"] and res["skipped"]["front_door"] == "not selected"
    with pytest.raises(SystemExit):
        run(only=["nope"])


def test_anchored_and_flow_style_cameras_are_left_alone():
    text = """
x-cam: &cam
  ffmpeg:
    inputs:
      - path: rtsp://127.0.0.1:8554/a_sub
        roles: [detect]
  detect: {width: 640, height: 360}
cameras:
  a: *cam
  b: {ffmpeg: {inputs: [{path: "rtsp://127.0.0.1:8554/b_sub", roles: [detect]}]}, detect: {width: 640, height: 360}}
ffmpeg: {hwaccel_args: preset-nvidia}
"""
    res = run(text)
    assert res["plans"] == {}
    assert "anchors or aliases" in res["skipped"]["a"] and "block mapping" in res["skipped"]["b"]


def test_args_value_keeps_tokens_exact():
    assert convert.args_value(["-vf", "eq=gamma=1.4:gamma_weight=0.5"]) == "-vf eq=gamma=1.4:gamma_weight=0.5"
    odd = ["-vf", "drawtext=text='a b'", "-f", "rawvideo"]
    v = convert.args_value(odd)
    assert v == odd
    assert shlex.split(convert.args_value(["-f", "rawvideo"])) == ["-f", "rawvideo"]


def test_reader_args_construction():
    assert convert.reader_input_args(1280, 720, 5) == ["-probesize", "32", "-analyzeduration", "0", "-f", "rawvideo",
                                                       "-pix_fmt", "nv12", "-video_size", "1280x720", "-framerate", "5"]
    assert convert.sock_path("/srv/x", "cam_1") == "unix:/srv/x/run/cam_1.sock"


# ---------------------------------------------------------------- apply / rollback on real files


@pytest.fixture
def setup(tmp_path):
    cfg = tmp_path / "config.yml"
    cfg.write_text(SAMPLE)
    os.chmod(cfg, 0o640)
    base = tmp_path / "nvdec-shared"
    render = functools.partial(fakefrigate.render, gamma_off=False)
    call = lambda *a: convert.main([*a, "--config", str(cfg), "--base", str(base)], render=render)
    return cfg, base, call


def test_plan_writes_nothing(setup, capsys):
    cfg, base, call = setup
    assert call("plan") == 0
    assert cfg.read_text() == SAMPLE and not base.exists()
    out = capsys.readouterr().out
    assert "PLAN OK" in out and "RENDER other_cmds_unchanged=" in out
    assert "FRIGATE_CAM_PASS}@" not in out or "***@" in out


def test_apply_then_rollback_round_trip(setup, capsys):
    cfg, base, call = setup
    ino, mode = cfg.stat().st_ino, cfg.stat().st_mode
    assert call("apply") == 0
    assert cfg.read_text() == convert.convert(SAMPLE, fakefrigate.render, str(base))["text"]
    assert cfg.stat().st_ino == ino and cfg.stat().st_mode == mode
    cams = json.loads((base / "cameras.json").read_text())
    assert len(cams["cameras"]) == 3 and oct((base / "cameras.json").stat().st_mode & 0o777) == "0o600"
    backups = list(cfg.parent.glob("config.yml.pre-nvdec-shared-*"))
    assert len(backups) == 1 and backups[0].read_text() == SAMPLE
    with pytest.raises(SystemExit, match="roll back first"):
        call("apply")
    assert call("rollback") == 0
    assert cfg.read_bytes() == SAMPLE.encode() and cfg.stat().st_ino == ino
    assert not (base / "applied.json").exists()
    assert "ROLLED BACK" in capsys.readouterr().out


def test_rollback_refuses_to_drop_later_edits(setup):
    cfg, base, call = setup
    call("apply")
    cfg.write_text(cfg.read_text() + "\n# edited later\n")
    with pytest.raises(SystemExit, match="changed since apply"):
        call("rollback")
    assert call("rollback", "--force") == 0
    assert cfg.read_text() == SAMPLE


def test_rollback_without_apply(setup):
    _, _, call = setup
    with pytest.raises(SystemExit, match="nothing to roll back"):
        call("rollback")
