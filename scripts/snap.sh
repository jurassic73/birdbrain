#!/usr/bin/env bash
# Grab one frame from the camera. Usage: scripts/snap.sh [main|sub] [out.jpg]
set -euo pipefail
cd "$(dirname "$0")/.."

# Read only the keys we need; .env is not shell (values may hold ; # $ ! etc.).
# SNAP_ENV overrides the file (used by tests).
while IFS= read -r -d '' kv; do
  export "$kv"
done < <(python3 -I scripts/readenv.py "${SNAP_ENV:-.env}" \
  RTSP_USER RTSP_PASS CAM_HOST RTSP_PORT RTSP_MAIN_PATH RTSP_SUB_PATH)
: "${RTSP_PASS:?RTSP_PASS is empty in .env}"
: "${CAM_HOST:?CAM_HOST is empty in .env}"

case "${1:-main}" in
  main) path="${RTSP_MAIN_PATH:-ch0}" ;;
  sub)  path="${RTSP_SUB_PATH:-ch1}" ;;
  *)    path="$1" ;;  # any other stream path, as-is
esac
out="${2:-snapshots/snap_${1:-main}_$(date +%Y%m%d_%H%M%S).jpg}"
mkdir -p "$(dirname "$out")"

enc() { python3 -c 'import sys,urllib.parse;print(urllib.parse.quote(sys.argv[1],safe=""))' "$1"; }
url="rtsp://$(enc "${RTSP_USER:-thingino}"):$(enc "$RTSP_PASS")@${CAM_HOST}:${RTSP_PORT:-554}/${path}"

# ffmpeg echoes the input URL on errors; redact credentials from its output
if ! ffmpeg -hide_banner -loglevel error -rtsp_transport tcp -i "$url" -frames:v 1 -y "$out" 2> >(sed -E 's#rtsp://[^@/]*@#rtsp://***@#g' >&2); then
  exit 1
fi
echo "$out"
