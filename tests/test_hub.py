import ctypes
import types
from fractions import Fraction

import pytest

import hub


def test_decoder_options_match_frigates_nvidia_detect_input():
    # Frigate's preset-rtsp-generic: tcp, 10 s timeout, wall-clock timestamps, genpts + discardcorrupt
    assert hub.RTSP_OPTS["rtsp_transport"] == "tcp"
    assert hub.RTSP_OPTS["timeout"] == "10000000"
    assert hub.RTSP_OPTS["use_wallclock_as_timestamps"] == "1"
    assert hub.RTSP_OPTS["fflags"] == "+genpts+discardcorrupt"
    assert hub.HW_OPTS == {"primary_ctx": "1"}


def test_hwaccel_is_cuda_on_the_primary_context_without_software_fallback(monkeypatch):
    monkeypatch.setattr(hub, "HWAccel", lambda **kw: kw)
    assert hub.hwaccel() == {"device_type": "cuda", "allow_software_fallback": False, "options": {"primary_ctx": "1"},
                             "is_hw_owned": True}


def picks(time_base, fps, pts_list):
    p, out = hub.FpsPicker(time_base, fps), []
    for i, pts in enumerate(pts_list):
        frame, times = p.push(i, pts)
        out += [frame] * times
    return out


def test_fps_picker_3_of_20():
    tb = Fraction(1, 90000)
    out = picks(tb, 3, [i * 4500 for i in range(41)])  # 2 s at 20 fps
    # tick n shows the latest frame whose timestamp rounds to tick <= n: frames 0..3 all round to tick 0
    assert out == [3, 9, 16, 23, 29, 36]


def test_fps_picker_hole_repeats_last_frame_and_first_frame_sets_tick_0():
    tb = Fraction(1, 1000)
    out = picks(tb, 3, [5000, 5050, 5400, 6400, 6450])
    assert out == [1, 2, 2, 2]


def test_fps_picker_rounding_matches_av_rescale_near_inf():
    p = hub.FpsPicker(Fraction(1, 1000), 3)
    assert [p.tick(t) for t in (0, 166, 167, 333, 500, -167, -166)] == [0, 0, 1, 1, 2, -1, 0]


def test_ts_offset_is_ffmpegs_default_input_offset():
    st = types.SimpleNamespace(time_base=Fraction(1, 90000))
    assert hub.ts_offset(types.SimpleNamespace(start_time=None), st) == 0
    assert hub.ts_offset(types.SimpleNamespace(start_time=1_500_000), st) == 135000  # 1.5 s in µs -> 90 kHz


def test_load_config_per_camera_fps_and_sizes():
    cfg = {"sock_dir": "/x/run", "fps": 5, "cameras": [
        {"name": "front_door", "url": "rtsp://127.0.0.1:8554/front_door_sub", "width": 640, "height": 360},
        {"name": "driveway", "url": "rtsp://127.0.0.1:8554/driveway_sub", "width": 1280, "height": 720, "fps": 3}]}
    sock_dir, cams = hub.load_config(cfg, {})
    assert sock_dir == "/x/run"
    assert cams == [("front_door", "rtsp://127.0.0.1:8554/front_door_sub", 640, 360, 5),
                    ("driveway", "rtsp://127.0.0.1:8554/driveway_sub", 1280, 720, 3)]


@pytest.mark.parametrize("cam,why", [
    ({"name": "a b", "url": "rtsp://x", "width": 640, "height": 360, "fps": 5}, "name"),
    ({"name": "a", "url": "rtsp://x", "width": 641, "height": 360, "fps": 5}, "even"),
    ({"name": "a", "url": "rtsp://x", "width": 640, "height": 360}, "fps"),
])
def test_load_config_refuses(cam, why):
    with pytest.raises(ValueError, match=why):
        hub.load_config({"sock_dir": "/x", "cameras": [cam]}, {})
    with pytest.raises(ValueError, match="duplicate"):
        good = {"name": "a", "url": "rtsp://x", "width": 640, "height": 360, "fps": 5}
        hub.load_config({"sock_dir": "/x", "cameras": [good, good]}, {})


def test_stream_url_substitutes_like_frigate_and_encodes_the_password():
    env = {"FRIGATE_USER": "viewer", "FRIGATE_PASS": "p@ss/w:rd"}
    url = hub.stream_url("rtsp://{FRIGATE_USER}:{FRIGATE_PASS}@192.0.2.5:554/sub", env)
    assert url == "rtsp://viewer:p%40ss%2Fw%3Ard@192.0.2.5:554/sub"
    assert hub.stream_url("rtsp://127.0.0.1:8554/cam_sub", env) == "rtsp://127.0.0.1:8554/cam_sub"
    with pytest.raises(KeyError):
        hub.stream_url("rtsp://{FRIGATE_MISSING}@x/", env)


def test_frigate_env_reads_env_and_secret_files(tmp_path):
    (tmp_path / "FRIGATE_PASS").write_text("s3cret\n")
    (tmp_path / "OTHER").write_text("x")
    env = hub.frigate_env({"FRIGATE_USER": "u", "PATH": "/bin"}, str(tmp_path))
    assert env == {"FRIGATE_USER": "u", "FRIGATE_PASS": "s3cret"}


def test_errors_never_show_the_password():
    url = "rtsp://viewer:p%40ss@192.0.2.5:554/sub"
    msg = hub.redact(f"HTTPError: [Errno 5] Input/output error: '{url}'", url)
    assert "p%40ss" not in msg and "rtsp://***@192.0.2.5:554/sub" in msg


def test_starve_check_reconnects_a_decoder_that_takes_packets_but_gives_almost_no_frames():
    s = hub.StarveCheck(0.0)
    n = hub.STARVE_FIRST_S * 20
    for i in range(n):  # first window: 20 packets/s, 1 frame per 18 packets (as seen live)
        s.packet(i / 20)
        if i % 18 == 0:
            s.frame()
    with pytest.raises(RuntimeError, match="decoder starved"):
        s.packet(n / 20)


def test_starve_check_leaves_healthy_slow_and_keyframe_waiting_streams_alone():
    s = hub.StarveCheck(0.0)
    # joins mid-GOP: 15 s of packets without a frame, then 1 frame per packet; then 3 windows at 1:1
    for i in range((hub.STARVE_FIRST_S + 3 * hub.STARVE_WINDOW_S) * 20 + 1):
        s.packet(i / 20)
        if i >= 15 * 20:
            s.frame()
    # a stalled stream (2 packets in a whole window) is not a starved decoder
    s2 = hub.StarveCheck(0.0)
    s2.packet(1.0)
    s2.packet(hub.STARVE_FIRST_S - 1)
    s2.packet(hub.STARVE_FIRST_S + 1)


def test_session_counts_packets_and_frames_for_the_starve_check():
    src = open(hub.__file__).read()
    assert "starve.packet(self.beat)" in src and "starve.frame()" in src


def test_cuda_memcpy2d_has_the_driver_api_layout():
    assert ctypes.sizeof(hub.CUDA_MEMCPY2D) == 128  # CUDA_MEMCPY2D_v2 on 64-bit
    assert hub.CUDA_MEMCPY2D.dstHost.offset == 80 and hub.CUDA_MEMCPY2D.WidthInBytes.offset == 112


class FakeCuda:
    """Records the driver calls; 'device memory' is a host buffer, so the copy can be checked byte for byte."""

    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def call(*args):
            self.calls.append(name)
            if name == "cuMemcpy2D_v2":
                d = args[0]._obj
                for row in range(d.Height):
                    ctypes.memmove(d.dstHost + row * d.dstPitch, d.srcDevice + row * d.srcPitch, d.WidthInBytes)
            return 0
        return call


def test_cuda_download_copies_both_nv12_planes_without_the_padding():
    w, h, pitch = 8, 4, 16
    y = bytes(range(pitch * h))
    uv = bytes(200 + i % 50 for i in range(pitch * h // 2))
    ybuf, uvbuf = ctypes.create_string_buffer(y, len(y)), ctypes.create_string_buffer(uv, len(uv))
    plane = lambda buf: types.SimpleNamespace(buffer_ptr=ctypes.addressof(buf), line_size=pitch)
    frame = types.SimpleNamespace(format=types.SimpleNamespace(name="cuda"), width=w, height=h,
                                  planes=[plane(ybuf), plane(uvbuf)])
    cu = FakeCuda()
    out = hub.CudaDownload(lib=cu).nv12(frame, w, h)
    want = b"".join(y[r * pitch:r * pitch + w] for r in range(h)) + b"".join(uv[r * pitch:r * pitch + w] for r in range(h // 2))
    assert bytes(out) == want
    assert cu.calls == ["cuInit", "cuDeviceGet", "cuDevicePrimaryCtxRetain", "cuCtxPushCurrent_v2",
                        "cuMemcpy2D_v2", "cuMemcpy2D_v2", "cuCtxPopCurrent_v2"]


def test_cuda_download_refuses_a_frame_that_is_not_on_the_gpu_or_the_wrong_size():
    dl = hub.CudaDownload(lib=FakeCuda())
    with pytest.raises(RuntimeError, match="unexpected frame"):
        dl.nv12(types.SimpleNamespace(format=types.SimpleNamespace(name="nv12"), width=8, height=4), 8, 4)
    with pytest.raises(RuntimeError, match="unexpected frame"):
        dl.nv12(types.SimpleNamespace(format=types.SimpleNamespace(name="cuda"), width=8, height=6), 8, 4)


def test_noframe_check_reconnects_a_trickling_stream_that_the_starve_check_lets_through():
    # 1 packet every 2 s (10 per 20 s window: below STARVE_MIN_PACKETS) and no frame picked
    starve, noframe = hub.StarveCheck(0.0), hub.NoFrameCheck(0.0)
    t, tripped = 0.0, None
    while t < 120 and tripped is None:
        starve.packet(t)  # never raises: too few packets per window
        try:
            noframe.packet(t)
        except RuntimeError as e:
            tripped = (t, str(e))
        t += 2.0
    assert tripped is not None and hub.NOFRAME_S < tripped[0] <= hub.NOFRAME_S + 2
    assert "no frame picked" in tripped[1]


def test_noframe_check_is_quiet_while_frames_are_picked():
    noframe = hub.NoFrameCheck(0.0)
    for i in range(20 * 300):  # 5 min at 20 packets/s, a pick every 7 packets (3 fps)
        noframe.packet(i / 20)
        if i % 7 == 0:
            noframe.picked(i / 20)


def test_session_wires_the_noframe_check():
    src = open(hub.__file__).read()
    assert "noframe.packet(self.beat)" in src and "noframe.picked(self.beat)" in src
