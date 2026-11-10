#!/bin/bash
# (Re)create the frigate-nvdec-shared container next to an existing Frigate container. Run it on the docker host.
#
# It uses the exact image your frigate container runs (so PyAV and FFmpeg are Frigate's own, nothing to install),
# mounts <frigate config dir>/nvdec-shared at /config/nvdec-shared (the same path in both containers, so Frigate's
# reader ffmpeg finds the sockets), and copies the frigate container's FRIGATE_*
# environment variables so {FRIGATE_...} placeholders in camera paths resolve as they do in Frigate.
# Docker restarts it on exit and at boot. Logs: docker logs frigate-nvdec-shared.
# Per-camera counters into the log: docker kill -s USR1 frigate-nvdec-shared.
#
# Environment: FRIGATE_CONTAINER (default frigate), HUB_CONTAINER (default frigate-nvdec-shared),
#              HUB_NETWORK (default: host, required unless Frigate itself uses host networking; see README),
#              HUB_MEMORY (default 1g), NVDEC_SHARED_DIR (default /config/nvdec-shared; must be under /config
#              and match convert.py --base).
set -euo pipefail
FRIGATE=${FRIGATE_CONTAINER:-frigate}
NAME=${HUB_CONTAINER:-frigate-nvdec-shared}
BASE=${NVDEC_SHARED_DIR:-/config/nvdec-shared}
case "$BASE" in /config/?*) ;; *) echo "NVDEC_SHARED_DIR must be under /config/" >&2; exit 1;; esac

IMAGE=$(docker inspect -f '{{.Image}}' "$FRIGATE")
CONFIG_SRC=$(docker inspect -f '{{range .Mounts}}{{if eq .Destination "/config"}}{{.Source}}{{end}}{{end}}' "$FRIGATE")
MODE=$(docker inspect -f '{{.HostConfig.NetworkMode}}' "$FRIGATE")
[ -n "$CONFIG_SRC" ] || { echo "$FRIGATE has no /config mount" >&2; exit 1; }
DIR="$CONFIG_SRC/${BASE#/config/}"
for f in hub.py cameras.json; do
  [ -f "$DIR/$f" ] || { echo "missing $DIR/$f (copy hub.py there; cameras.json comes from convert.py apply)" >&2; exit 1; }
done
NETWORK=${HUB_NETWORK:-}
if [ -z "$NETWORK" ]; then
  if [ "$MODE" = host ]; then NETWORK=host; else
    echo "$FRIGATE uses network '$MODE', not host. Only host networking is tested; to try anyway set" >&2
    echo "HUB_NETWORK=container:$FRIGATE (then recreate this container whenever Frigate's container is recreated)." >&2
    exit 1
  fi
fi

ENVF=$(mktemp)
trap 'rm -f "$ENVF"' EXIT
chmod 600 "$ENVF"
docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' "$FRIGATE" | grep '^FRIGATE_' > "$ENVF" || true

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" --restart unless-stopped --init --gpus all --network "$NETWORK" \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,video,utility --env-file "$ENVF" \
  --memory "${HUB_MEMORY:-1g}" -e MALLOC_ARENA_MAX=2 --no-healthcheck --log-opt max-size=10m --log-opt max-file=3 \
  -v "$DIR:$BASE" --entrypoint python3 "$IMAGE" -u "$BASE/hub.py" "$BASE/cameras.json"
sleep 5
docker ps --filter "name=^${NAME}$" --format '{{.Names}} {{.Status}}'
docker logs --tail 20 "$NAME"
