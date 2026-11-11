# frigate-nvdec-shared

Decode all of Frigate's detect streams on an NVIDIA GPU inside **one** process, so they share one CUDA context
instead of paying for one context per camera. Frigate sees byte-identical detect frames; recordings are not touched.

On the one system it was built for, 14 cameras went from **1726 MiB to 225 MiB of VRAM** for detect decoding, CPU
use went down, not up, and GPU load stayed about the same as before (a few percent of the card either way).

## The problem

With an NVIDIA preset (`hwaccel_args: preset-nvidia`), Frigate runs one `ffmpeg -hwaccel cuda` process per
camera's detect stream. Every CUDA process holds its own CUDA context, and on recent drivers that context alone is
about 100 MiB of VRAM, whatever the stream. Measured on the test system:

| Process | VRAM |
|---|---|
| a bare CUDA context (`cuInit` + `cuCtxCreate`, nothing else) | 102 MiB |
| Frigate's NVIDIA detect command on one 640×360 H.264 stream | 117 MiB |
| the same with `CUDA_MODULE_LOADING=LAZY`/`EAGER`, without `scale_cuda`, without `-hwaccel_output_format cuda`, or as `-c:v h264_cuvid` | 113–119 MiB |

So N cameras cost N × ~120 MiB, and almost all of it is context, not decoding. None of the usual knobs shrinks the
context. The only lever is fewer processes holding one.

## How it works

```
 BEFORE: one CUDA context per camera

   restreamer/camera --rtsp--> ffmpeg -hwaccel cuda (context ~100 MiB) --yuv420p--> Frigate detect
   restreamer/camera --rtsp--> ffmpeg -hwaccel cuda (context ~100 MiB) --yuv420p--> Frigate detect
   ... one per camera

 AFTER: one CUDA context for all cameras

   +---------------- container frigate-nvdec-shared (hub.py) ----------------+
   |  one process, one primary CUDA context                                   |
   |                                                                          |
   |  thread front_door: rtsp -> NVDEC -> pick frames like ffmpeg's fps -> NV12 --> run/front_door.sock
   |  thread driveway:   rtsp -> NVDEC -> pick frames like ffmpeg's fps -> NV12 --> run/driveway.sock
   |  ... one thread per camera, each reconnects on its own                   |
   +--------------------------------------------------------------------------+
                                                                                       |
     frigate container: Frigate's own detect ffmpeg per camera, now reading the socket <+
       -hwaccel none -f rawvideo -pix_fmt nv12 -i unix:.../front_door.sock
       -vf <the filters that followed hwdownload,format=nv12 before> <your detect output args> --> Frigate detect

   Recordings: unchanged. Frigate's record ffmpeg still copies the main stream; it never depends on the hub.
```

Frigate's NVIDIA detect chain is `-hwaccel cuda … -vf fps=F,scale_cuda=w=W:h=H,hwdownload,format=nv12,<rest>`.
The hub does the first half (decode on NVDEC, select frames by the rule of ffmpeg's `fps` filter, download NV12);
Frigate's own ffmpeg, still started and supervised by Frigate, does `<rest>` and the output conversion exactly as
before. `<rest>` is whatever Frigate rendered for that camera: with Frigate 0.17's NVIDIA preset it is
`eq=gamma=1.4:gamma_weight=0.5`, or `format=yuv420p` when `FFMPEG_DISABLE_GAMMA_EQUALIZER` is set. The converter
reads it from Frigate's rendered command; nothing is hard-coded.

Details that matter for identical frames:

- one process and one **primary** CUDA context (`primary_ctx=1`) for every NVDEC session;
- `allow_software_fallback=False`: a stream that cannot decode on the GPU is an error, never a silent CPU decode;
- the stream is opened with Frigate's `preset-rtsp-generic` options (TCP, 10 s timeout, wall-clock timestamps);
- frame selection is a 20-line port of ffmpeg's `fps` filter (`round=near`) applied after ffmpeg's default input
  offset (timestamps start at the input's start time). Without that offset the selection lands one source frame off
  on most streams. It is a port rather than a libavfilter graph because PyAV 17 leaks memory on every pull that
  returns no frame (about 1.5 MB/min with 14 cameras). `tools/fpstest.py` checks the port against libavfilter's own
  filter on random timestamp sequences;
- decoded frames stay on the GPU; only the frames the selection picks are downloaded, plane by plane with
  `cuMemcpy2D` (the same 2D copy ffmpeg's `hwdownload` does). The copy is made through `ctypes`, which releases
  Python's GIL while it waits for the GPU. PyAV's own download (used by an earlier version of the hub) holds the GIL
  for every decoded frame; on a busy shared GPU those waits froze all camera threads together for 1–2.4 s at a time;
- the stream must already be at detect size (as a substream usually is, where `scale_cuda` is a same-size pass).
  A stream of another size is refused, never scaled.

Failure handling:

- a camera that fails (refused, timeout, 404, 100 bad packets in a row) reconnects alone, backoff up to 10 s;
- a camera whose decoder keeps taking packets but stops giving frames reconnects too: fewer than 1 frame per 8
  packets over a 20 s window (40 s for a session's first window, which may wait for a keyframe). This happened once
  in service: after a stream glitch, NVDEC answered "No decoder surfaces left" for hours and gave about 1 frame per
  18 packets, while every call returned without an error. `tools/stalltest.py` forces that wedge on saved clips:
  the wedge started 30 s in, the camera reconnected 30 s later (`decoder starved: 22 frames from 401 packets`) and
  decoded normally 2 s after that; the control camera and a run without the forced wedge never reconnected;
- a camera that has had no frame picked for 30 s while packets still arrive reconnects too. This closes a gap the
  check above leaves: a stream that trickles (a few packets now and then, never 15 s apart, so the read timeout does
  not fire, and under 20 packets per window, so the starve check stays out of it). In service one camera did exactly
  that, as far as the logs show, and its detection stopped for about 5 minutes (see "Versions"). `tools/stalltest.py --trickle` reproduces it on
  saved recordings: the previous version never reconnected (`sessions stalled=1 … RESULT=FAIL`), this one
  reconnected 32 s after the trickle began (`no frame picked for 32 s while packets still arrive`) and decoded
  normally 2 s later (`RESULT=PASS`); the starve case and a clean run still pass. 30 s is longer than Frigate's own
  20 s detect watchdog, so Frigate restarts its reader first and the two do not race;
- if one camera thread is stuck in a single call for 120 s, or no camera has produced a frame for 90 s, the hub
  exits (code 3 or 4) and Docker restarts it clean;
- Frigate notices a broken socket like any dead stream and restarts its reader ffmpeg by itself.

## Measured results

One installation; anonymized. 14 cameras, H.264 substreams at 640×360 and 640×480, 20 fps, detect at 3 fps, read
from Frigate's go2rtc restream. NVIDIA GeForce RTX 2070 SUPER (Turing, 8 GB), driver 580.126.18, Frigate 0.17.1.
Every VRAM number is one `nvidia-smi` read of a named process.

**Versions.** Most numbers below come from about five hours of service with the first version of the hub, which let
PyAV download every decoded frame (holding the GIL) and had no wedged-decoder check. The current `hub.py` changes
only the download (GPU-held frames, picked frames only, GIL released) and adds that check. The current version was
checked separately: unit tests, byte parity on fresh clips of all 14 streams (14 of 14 identical), 245 MiB of VRAM
(vs 227 for the first version, same moment), and 13 minutes beside the production hub on the same live streams.
It then replaced the first version in production and ran for 3 hours (checked at the end): all 14 cameras at
3.0-3.1 fps with 0 skipped, detector 11-13 ms, 0 wedged-decoder reconnects, no restarts, RSS 285 MB at 4 minutes
and 312 MB at 3 hours. It needed no restart, but it was not flawless: one camera's detection stalled for about
5 minutes. Frigate's watchdog restarted that camera's reader three times ("No frames received … in 20 seconds"); the
hub's session for it stayed open, logging decoder errors ("No decoder surfaces left"), and sent no frame, yet neither
the read timeout nor the starve check fired, which fits a stream that trickled (not confirmed from the logs); only a
stream error ended the session, and the reconnect brought the camera back. The current `hub.py` adds the 30 s no-frame reconnect for exactly this (see "Failure handling");
that change passed the offline tests above and then went into production; checked after 2 hours there: no restarts,
no error or warning lines, all 14 cameras at 3.0-3.1 fps with 0 skipped, detector 12 ms, RSS 284 MB at 3 minutes and
286 MB at 2 hours. No stream trickled in that time, so the new check never fired: its behaviour on a real trickle is
still shown only by the offline test. In the same minutes as the 5-minute stall the restream refused the hub's RTSP
connections once, and recordings (which do not go through the hub) had gaps on 7 cameras; the same cameras show the
same kind of recording gaps on earlier days, before the hub existed.

Not caused by the hub, but worth knowing if your recordings share a busy hard disk: twice on this system (once before
the hub existed) heavy IO on the single HDD that holds Frigate's clips stalled reads for over 300 s (ZFS deadman
events), and all cameras lost minutes of recordings. Frigate 0.17 writes snapshots on the same thread that hands
detection results to the recorder, so when that disk stalls the recorder drops segments and blames "the detect
stream", even though decoding and detection are fine.

**VRAM**

| | Before | After |
|---|---|---|
| detect decoding | 14 ffmpeg × 119–127 MiB = **1726 MiB** | hub **225 MiB** (231 right after start) |
| cost of one more camera | ~120 MiB | ~14 MiB (one stock ffmpeg with 1 / 2 / 14 inputs on a shared device: 119 / 130 / 298 MiB) |

**CPU** (60 s, one before/after read of `/proc/<pid>/stat`, % of one core): 14 GPU detect ffmpegs **41.7 %** →
hub 23.6 % + 14 reader ffmpegs 4.5 % = **28.1 %**.

**Hub memory** (resident set of the first version's process, read with `docker top`, 14 cameras): 317 MB at 2 min,
318 MB at 9 min, 331 MB at 48 min, 331.5 MB at 68 and 73 min, 329.5 MB at 108 min; then, after about 50 camera
reconnects in 20 minutes, 359 MB at 140 min and 356 MB at 171 min. No steady leak: it was flat for the hour without
reconnects, and 150 clean reconnect cycles of the hub's session code in a test process stayed at 174–180 MB. It
steps up during bursts of errors and reconnects and does not shrink afterwards. The container runs with a 1 GiB cap
and `MALLOC_ARENA_MAX=2`; the cgroup's memory stayed at about 250 MB with no limit events.

**GPU load.** The hub costs about as much GPU time as the stock path it replaces: a few percent of the card. The
test card was shared with other GPU workloads whose load moved between 30 % and 79 % utilization within minutes, so
the cost was measured by alternation: 30-second windows of "production only" and "production + a test load", 10
cycles over two runs, one `nvidia-smi` reading per second, and each load window compared with the production-only
windows on either side of it. The test loads read the same 14 live substreams: 14 copies of Frigate's stock NVIDIA
detect command, or a second hub, or (to lift its signal above the noise) three extra hubs at once. Per 14 streams,
mean ± standard error:

| Test load | GPU utilization | Board power | NVDEC (decoder) utilization |
|---|---|---|---|
| Frigate's stock detect ffmpegs (9 windows) | +3.7 ± 2.3 % | +0.2 ± 2.5 W | +3 % |
| one extra hub (5 windows) | +5.1 ± 3.7 % | +3.1 ± 4.0 W | +3 % |
| three extra hubs, divided by 3 (5 windows) | +2.1 ± 1.2 % | +0.6 ± 1.1 W | +3 % |

No difference between the two paths is measurable. Both decode the same frames on NVDEC (identical decoder load).
The hub downloads every decoded frame to host memory (PyAV transfers each frame as it is decoded) and then picks,
where Frigate's chain picks first and downloads only the picked ones; at 14 × 20 fps of 640×360 that is about
90 MB/s, and it did not show up as GPU cost. These load numbers are for the first version (it downloaded every
decoded frame). A variant that keeps frames on the GPU and downloads only the picked ones, as the current version
does, was measured the same way: +1.1 ± 1.5 % per 14 streams with three running, so it is not measurably cheaper or
dearer. Caveat: GPU
utilization is the share of time any kernel runs, so work that overlaps other workloads shows up smaller than it
would on an idle card; that applies to both paths alike.

**Frigate** (`/api/stats`, about 40 minutes after the change): camera_fps 3.0–3.1 on all 14 cameras before and
after, skipped_fps 0.0 and process_fps equal to camera_fps before and after, no new errors in Frigate's log.
Recordings: one gap per camera at the Frigate restart that applied the config, none at any hub restart.

**Byte parity.** On 12-second clips saved from all 14 detect streams, A = Frigate's own detect command on the clip
and B = `hub.py --clip` → NV12 → the reader command `convert.py` produced, compared by md5 of the yuv420p output.
Re-run with the code in this repository (`tools/parity.py`, Frigate's own parser rendering both commands):

| Run | identical | first N identical, A has 1 end-of-file frame more | different |
|---|---|---|---|
| detect fps 3, Frigate's default chain (`eq` gamma) | 13 | 1 | 0 |
| detect fps 3, `FFMPEG_DISABLE_GAMMA_EQUALIZER=1` (`format=yuv420p`) | 13 | 1 | 0 |
| detect fps 5, Frigate's default chain | 11 | 3 | 0 |
| current version, fresh clips of all 14 streams, detect fps 3, Frigate's default chain | 14 | 0 | 0 |
| current version, the saved clips of the rows above, detect fps 3, Frigate's default chain | 14 | 0 | 0 |
| current version, one 10 s recording per camera (11 × H.265 2560×1440, 3 × H.264 1920×1080), detect = clip size | 14 | 0 | 0 |

The first three rows used the first version's download path (the frame selection code is the same in both). On
the saved clips the current version's A hashes equal those of the first rows, and the one clip whose first-version
output was one end-of-file frame short now matches in full: PyAV's download copies only a frame's timestamp, not
its duration, which the end-of-file fill in `--clip` needs; a GPU-held frame keeps it.

The end-of-file frame: when a *file* ends, ffmpeg's `fps` filter can flush one more frame than the hub. A live
stream never ends that way. The A hashes also equal those of Frigate's stock command recorded before the hub
existed. `tools/fpstest.py`: `FPSTEST sequences=600 input_frames=89820 libavfilter_outputs=284266
mismatching_sequences=0`.

**Faults** (a second hub instance with one good camera beside a refused port, a TCP port that accepts and never
answers, a 404 stream and a wrong-size stream): the good camera delivered 61 frames in 20 s (3 fps) while the other
four retried; the silent one timed out about every 12 s; the wrong-size one was refused before a single frame was
served. SIGTERM stops the hub in under 1 s; a fresh start has all 14 cameras decoding within 3 s. A hub restart
pauses detection for 3–13 s (Frigate restarts its readers; one that tried before the socket existed waits one 10 s
retry). No recording gap.

**Not resolved by measurement: detector speed.** Frigate's detector `inference_speed` read 10.6 and 11.4 ms before
the change and 11.7–16.3 ms in nine reads over the next 80 minutes (plus one read of 20.2 ms seconds after a hub
restart), while other GPU tenants on the same card were active and the card ran at 82–85 °C. Earlier on the same day, a 20-minute mean before any change was 14.9 ms. A
direct A/B (a fixed detector-model loop with nothing / 14 stock ffmpegs / a second hub running on top) spread more
between repeats than between methods. No frame was skipped in any read. So: no regression shown, none excluded.

## Requirements

As observed on the test system (read from the running image):

- NVIDIA GPU with NVDEC; tested on an RTX 2070 SUPER (Turing), driver 580.126.18 (CUDA 13.0).
- Docker with the NVIDIA container toolkit (`--gpus all`), Frigate in Docker with **host networking**.
- Frigate **0.17.1** (`ghcr.io/blakeblackshear/frigate:stable-tensorrt`). The hub runs in the very image your
  frigate container uses; nothing is installed. That image ships **PyAV 17.0.0** built on FFmpeg 8.0 libraries
  (libavcodec 62.11, libavfilter 11.4) for the hub, and Frigate's own **FFmpeg 7.0.2** (`/usr/lib/ffmpeg/7.0`) for
  the reader commands. Check that your image's PyAV has CUDA:
  `docker exec frigate python3 -c "import av; print(av.codec.hwaccel.hwdevices_available())"` must list `cuda`.
- Each camera to convert: a detect input with the `detect` role alone (typically a substream), an `rtsp://` or
  `rtsps://` path, `detect.width`/`detect.height` set and equal to the stream's real size, and Frigate rendering
  its detect command with the NVIDIA preset on GPU 0. `convert.py plan` tells you, per camera, which ones qualify
  and why the rest do not.
- For the tests: Python 3, PyYAML, pytest. No GPU.

## Install

On the docker host, from a checkout of this repository:

```sh
# the host directory mounted at /config in your frigate container
CONFIG=$(docker inspect -f '{{range .Mounts}}{{if eq .Destination "/config"}}{{.Source}}{{end}}{{end}}' frigate)
mkdir -p "$CONFIG/nvdec-shared"
cp hub.py convert.py start-container.sh "$CONFIG/nvdec-shared/"
```

### Convert

```sh
docker exec frigate python3 /config/nvdec-shared/convert.py plan
```

`plan` writes nothing. It edits your config in memory, prints one `CONVERT` or `SKIP <reason>` line per camera,
the diff, and the result of its checks:

- the parsed new YAML equals the old one plus exactly the intended change, and no comment is lost;
- with Frigate's own parser: every record (and other non-detect) command is unchanged, every unconverted camera is
  unchanged, and every converted detect command is exactly the reader command;
- `RENDER other_cmds_unchanged=… detect_as_expected=N/N`, plus the sha256 of the current and candidate config.

The edit is textual: comments, order and formatting of everything else stay byte-for-byte. Per converted camera the
detect input gets `path: "unix:/config/nvdec-shared/run/<camera>.sock"`, `hwaccel_args: "-hwaccel none"` and
`input_args: "-probesize 32 -analyzeduration 0 -f rawvideo -pix_fmt nv12 -video_size WxH -framerate F"`, and
`ffmpeg.output_args.detect` becomes `-vf <rest>` followed by the detect output args the camera had. A marker comment
is placed above each changed input. `--cameras a,b` limits the change to those cameras; `--base` moves the
directory (it must be under `/config`, and `start-container.sh` must be given the same `NVDEC_SHARED_DIR`).

```sh
docker exec frigate python3 /config/nvdec-shared/convert.py apply
```

`apply` repeats every check, then writes `config.yml.pre-nvdec-shared-<UTC>` (the backup), `nvdec-shared/cameras.json`
(the hub's camera list: name, stream path, width, height, fps, generated from your config) and
`nvdec-shared/applied.json` (what rollback needs), and rewrites the config in place (same inode, owner and mode).
`{FRIGATE_…}` placeholders in stream paths stay placeholders: the hub substitutes them from its environment and
`/run/secrets` the way Frigate does. Nothing secret is written that was not already in your config.

### Start the hub, then restart Frigate

```sh
bash "$CONFIG/nvdec-shared/start-container.sh"
docker restart frigate
```

`start-container.sh` (re)creates the `frigate-nvdec-shared` container from the exact image your frigate container
runs, with `--restart unless-stopped`, host networking, a 1 GiB memory cap, the `nvdec-shared` directory mounted
at the same path as in Frigate, and Frigate's `FRIGATE_*` environment variables (passed through a temporary
mode-600 env file). It is not part of your compose file.

### Check

- `docker logs frigate-nvdec-shared`: one `decoding h264 WxH on GPU` line and one `reader connected` per camera.
- `nvidia-smi`: one process for the hub instead of one ffmpeg per camera.
- Frigate's System page: every converted camera at its detect fps, no skipped frames.
- `docker kill -s USR1 frigate-nvdec-shared` logs one status line per camera (state, frames decoded and sent,
  sessions, readers).

## Operate

| | |
|---|---|
| state | `docker ps --filter name=frigate-nvdec-shared` |
| log | `docker logs --tail 50 frigate-nvdec-shared` |
| counters | `docker kill -s USR1 frigate-nvdec-shared` |
| restart / recreate | `bash $CONFIG/nvdec-shared/start-container.sh` (detection pauses 3–13 s; Frigate reconnects by itself) |
| add or change a camera | `convert.py rollback`, edit your config the normal way, `convert.py apply`, recreate the hub, restart Frigate. A camera added the normal way also simply works without the hub; it just costs its own context. |
| after a Frigate upgrade | the hub keeps the image it was created with: recreate it with `start-container.sh`, and re-run the parity check |

## Check byte parity (manual, needs the GPU)

Save a short clip of each detect stream (inside the frigate container, from its restreamer or camera):

```sh
docker exec frigate sh -c 'mkdir -p /tmp/clips && /usr/lib/ffmpeg/7.0/bin/ffmpeg -rtsp_transport tcp \
  -i rtsp://127.0.0.1:8554/front_door_sub -t 12 -c copy /tmp/clips/front_door.mkv'
docker cp frigate:/tmp/clips ./clips
```

Then run the check in a **throwaway** container from the same image (it never touches the running Frigate):

```sh
docker run --rm --gpus all -e NVIDIA_DRIVER_CAPABILITIES=compute,video,utility \
  -v "$PWD/clips:/clips:ro" -v "$PWD:/repo:ro" -v /tmp/nvdec-parity:/work \
  --entrypoint python3 "$(docker inspect -f '{{.Image}}' frigate)" -u /repo/tools/parity.py /clips /work --fps 5
```

Use your detect fps, and add `-e FFMPEG_DISABLE_GAMMA_EQUALIZER=1` if your frigate container sets it. Expect one
line per clip and `PARITY … different=0`; an `A has 1 end-of-file frame more` line is the file-end flush described
above. `tools/fpstest.py` (no GPU) checks the frame selection against libavfilter the same way.

## Roll back

```sh
docker exec frigate python3 /config/nvdec-shared/convert.py rollback
docker restart frigate
docker rm -f frigate-nvdec-shared
```

`rollback` restores the backup byte for byte (and checks its sha256). It refuses if the config changed after
`apply`, because those later edits would be lost; `--force` restores the backup anyway. The backup is kept. By
hand: copy `config.yml.pre-nvdec-shared-<UTC>` over your config and restart Frigate.

## Limitations

- **Tested on one setup**: the system above, 14 cameras, one GPU model, one driver, one Frigate version.
- **The current version has had a few hours in production** (see "Versions"), not weeks; its 30 s no-frame
  reconnect has not yet fired in production (it has been exercised offline only).
- **A shared, busy GPU can still interrupt detection.** On the test system, after a GPU fault caused by another
  workload (`Xid 109`, context-switch timeout), the hub's camera connections timed out more often, one or two
  15-second detect pauses a minute somewhere across 14 cameras, for over an hour. The restream itself kept delivering
  (a demux-only reader saw no gap over 0.2 s). Part of this was the GIL freeze the current version removes; one
  burst that hit the old and the new hub at the same moment is not explained. The stock per-camera ffmpegs had no
  exits over two 5–7 minute windows in the same period.
- **H.264 only tested live.** H.265 decoding on NVDEC passed the byte-parity check offline (11 recordings at
  2560×1440, see the parity table), but no H.265 detect stream has run through the hub in service.
- **Detect streams only.** Recordings, live view, birdseye and any NVENC transcodes are untouched (and keep their
  own CUDA processes, if they had any).
- **One shared failure domain.** Every converted camera's detection depends on one process and one CUDA context. A
  crash or a hang pauses detection on all of them until Docker restarts the hub (3–13 s observed). Recordings do
  not depend on it.
- **Re-run the parity check after Frigate upgrades.** The equivalence rests on Frigate's NVIDIA preset chain and on
  the FFmpeg and PyAV in its image; any of them can change.
- **Detector speed change not resolved by measurement** (see above).
- One GPU (index 0); host networking; RTSP inputs; the stream must already be at detect size; a detect input that
  shares a role with record or audio is skipped. The hub opens every stream with Frigate's `preset-rtsp-generic`
  options (wall-clock timestamps); `plan` warns when a camera used input args without them (for example
  `preset-rtsp-restream`), since its frame selection then can differ.
- Any camera whose detect chain is not exactly Frigate's NVIDIA preset (custom `-vf`, other hwaccel) is skipped.

## FAQ

**Why not one ffmpeg with many inputs, or go2rtc?** One stock ffmpeg with 14 inputs on a shared CUDA device does
fit in 298 MiB, but ffmpeg 7 stalls every input when one live input stalls, and it cannot reconnect a single
input; one bad camera would stop detection everywhere. go2rtc does not decode by itself; when it transcodes it runs
a separate ffmpeg per stream, which is again one context each. The hub is a small purpose-built process: one thread and one reconnect loop per camera.

**Why not CUDA MPS?** Measured: on this GPU each MPS client still keeps its own context (121 MiB for a detect ffmpeg
as an MPS client), so nothing is saved.

**Can't the context itself be made smaller?** Not that we found: `CUDA_MODULE_LOADING`, `CUDA_DEVICE_MAX_CONNECTIONS`,
`CUDA_SCALE_LAUNCH_QUEUES`, `CUDA_DISABLE_UNIFIED_MEMORY`, surface counts and output formats all left it at
102 MiB. Vulkan Video decode used 109 MiB, twice the CPU, and was not byte-identical.

**Why not decode on the CPU?** That saves the VRAM but costs CPU; this keeps decoding on the GPU and uses less CPU
than before.

**Does Frigate see different frames?** Not on any of the tested clips; see the parity table. The reader runs the
same filters, in Frigate's own ffmpeg binary, on the same NV12 frames.

## Development

`ci/fast.sh` runs the unit tests (converter: render, validate, round-trip, rollback; camera-list generation; the
hub's arguments, frame selection and config loading). No GPU, no Frigate, a few seconds. `tools/parity.py` and
`tools/fpstest.py` are the manual checks against the real image.

## License

MIT, see [LICENSE](LICENSE).
