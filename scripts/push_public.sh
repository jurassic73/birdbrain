#!/bin/bash
# Push a read-only copy of Birdbrain's data to the public mirror (a server running with BIRDBRAIN_PUBLIC=1).
# The Pi only ever connects out; nothing on the internet can reach it. Run from cron every few minutes:
#   */5 * * * * /opt/birdbrain/scripts/push_public.sh >/dev/null 2>&1
# PUBLIC_DEST (env or .env) is an rsync target, e.g. deploy@droplet:  (the droplet's key is locked to
# rrsync in its data folder, so the path is relative to that). PUBLIC_KEY is the ssh key to use.
set -euo pipefail
cd "$(dirname "$0")/.."
envval() { { grep -E "^$1=" .env 2>/dev/null || true; } | cut -d= -f2- | awk '{print $1}'; }
PUBLIC_DEST="${PUBLIC_DEST:-$(envval PUBLIC_DEST)}"; : "${PUBLIC_DEST:?set PUBLIC_DEST}"
PUBLIC_KEY="${PUBLIC_KEY:-$(envval PUBLIC_KEY)}"; PUBLIC_KEY="${PUBLIC_KEY:-$HOME/.ssh/birdbrain_push}"
DATA="${BIRDBRAIN_DATA:-/var/lib/birdbrain}"

# One push at a time (cron and the watcher both start this) and never longer than 5 minutes, so a hung
# connection can't pile up pushes or block every later one: the script runs itself again under flock and
# timeout. Both are GNU tools; without them (a Mac in development) it just runs once, unguarded.
if [ -z "${PUSH_PUBLIC_GUARDED:-}" ] && command -v flock >/dev/null 2>&1 && command -v timeout >/dev/null 2>&1; then
  exec 9>>"$DATA/push_public.lock"
  flock -n 9 || { echo "push_public: another push is still running" >&2; exit 0; }
  PUSH_PUBLIC_GUARDED=1 exec timeout -k 15 300 bash scripts/push_public.sh "$@"
fi

STAGE="$(mktemp -d)"; trap 'rm -rf "$STAGE"' EXIT
# ssh gives up on a dead connection by itself: 10 s to connect, then three missed keepalives (45 s).
export RSYNC_RSH="ssh -i $PUBLIC_KEY -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=15 -o ServerAliveCountMax=3"
PY="${PUBLIC_PYTHON:-venv/bin/python}"
export BIRDBRAIN_DATA="$DATA"

# The database copy, trimmed to what the public pages read (birdbrain/mirror.py), minus every visit with a
# person in a photo (birdbrain/privacy.py; the Pi's own site is untouched), carrying the Pi's time zone so the
# mirror's "today" matches the birdbath's (birdbrain/clock.py). Next to it, STAGE/images holds links to the only
# photo files that go: the ones that copy refers to. Fails closed: if any of this fails, nothing is pushed this
# round. A visit the owner cleared with Make public on the visit page (visits.public_ok, read from the copy and
# then blanked in it) goes out.
TZNAME="$(timedatectl show -p Timezone --value 2>/dev/null || cat /etc/timezone 2>/dev/null || true)"
"$PY" -m birdbrain.mirror prepare "$DATA/birdbrain.db" "$STAGE" "$TZNAME"

# A fresh live frame from the local site (it grabs one from the camera); keep the last one if that fails.
# No photo of a person goes public: a live frame with a person, or one that can't be checked, isn't sent
# (the mirror keeps its last clean one).
curl -sf --max-time 15 -o "$STAGE/live.jpg" http://127.0.0.1:8000/live.jpg || cp "$DATA/live.jpg" "$STAGE/" 2>/dev/null || true
if [ -f "$STAGE/live.jpg" ] && ! "$PY" -m birdbrain.privacy check "$STAGE/live.jpg"; then rm -f "$STAGE/live.jpg"; fi

# Photos first, so the database never names a photo the mirror lacks. rsync writes each file under a
# temporary name and renames it when done, so the mirror never reads half a database or frame.
# Only the staged links are sent (-L sends the files they point at) and --delete takes everything
# else off the mirror: a visit held back since the last push, an orphan, a leftover temporary file.
# On the mirror the files belong to the push user, with the service reading them through its group:
# directories 2770 (group-owned by setgid), files 660 (the app needs group write on the database), nothing
# for anyone else. The D/F forms of --chmod need rsync 3 (the Pi); older ones and openrsync (a Mac) push
# with plain modes.
OPTS=(-rtpz)
if [ "$(rsync --version 2>/dev/null | sed -nE '1s/^rsync +version +([0-9]+)\..*/\1/p')" -ge 3 ] 2>/dev/null; then
  OPTS+=(--chmod=Dg+rwxs,Do-rwx,Fg+rw,Fo-rwx)
fi
rsync "${OPTS[@]}" -L --delete "$STAGE/images/" "$PUBLIC_DEST/images/"
rsync "${OPTS[@]}" "$STAGE/birdbrain.db" "$PUBLIC_DEST/birdbrain.db"
[ ! -f "$STAGE/live.jpg" ] || rsync "${OPTS[@]}" "$STAGE/live.jpg" "$PUBLIC_DEST/live.jpg"
