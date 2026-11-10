import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# hub.py imports PyAV at module level. The unit tests never decode, so they always run against this stand-in, with
# or without PyAV installed; tools/fpstest.py and tools/parity.py check the real thing.
av = types.ModuleType("av")
av.time_base = 1000000
av.FFmpegError = type("FFmpegError", (Exception,), {})
hw = types.ModuleType("av.codec.hwaccel")


class HWAccel:
    def __init__(self, **kw):
        self.kw = kw


hw.HWAccel = HWAccel
codec = types.ModuleType("av.codec")
codec.hwaccel = hw
av.codec = codec
av.logging = types.SimpleNamespace(set_level=lambda *_: None, ERROR=16)
sys.modules.update({"av": av, "av.codec": codec, "av.codec.hwaccel": hw})
