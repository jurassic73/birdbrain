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
STAGE="$(mktemp -d)"; trap 'rm -rf "$STAGE"' EXIT
export RSYNC_RSH="ssh -i $PUBLIC_KEY -o IdentitiesOnly=yes -o BatchMode=yes"

# A consistent copy of the database even while the watcher is writing, minus the private alert settings.
# The Pi's time zone travels with the data, so the mirror's "today" matches the birdbath's (birdbrain/clock.py).
TZNAME="$(timedatectl show -p Timezone --value 2>/dev/null || cat /etc/timezone 2>/dev/null || true)"
python3 - "$DATA/birdbrain.db" "$STAGE/birdbrain.db" "$TZNAME" <<'PY'
import sqlite3, sys
src, dst = sqlite3.connect(sys.argv[1]), sqlite3.connect(sys.argv[2])
src.backup(dst)
dst.execute("DELETE FROM settings WHERE key LIKE 'ntfy_%'")
if sys.argv[3]:
    dst.execute("INSERT OR REPLACE INTO settings (key, value) VALUES ('site_timezone', ?)", (sys.argv[3],))
dst.commit()
dst.execute("PRAGMA journal_mode=DELETE")
dst.execute("VACUUM")
dst.close()
PY
# A fresh live frame from the local site (it grabs one from the camera); keep the last one if that fails.
curl -sf --max-time 15 -o "$STAGE/live.jpg" http://127.0.0.1:8000/live.jpg || cp "$DATA/live.jpg" "$STAGE/" 2>/dev/null || true

# No photo of a person goes public (birdbrain/privacy.py; the Pi's own site is untouched). Fails closed:
# a live frame with a person, or one that can't be checked, isn't sent (the mirror keeps its last clean one);
# if the visit filter fails, nothing is pushed this round.
PY="${PUBLIC_PYTHON:-venv/bin/python}"
export BIRDBRAIN_DATA="$DATA"
if [ -f "$STAGE/live.jpg" ] && ! "$PY" -m birdbrain.privacy check "$STAGE/live.jpg"; then rm -f "$STAGE/live.jpg"; fi
"$PY" -m birdbrain.privacy filter "$STAGE/birdbrain.db" "$STAGE/private.txt"

# Photos first, so the database never names a photo the mirror lacks. rsync writes each file under a
# temporary name and renames it when done, so the mirror never reads half a database or frame.
# --delete-excluded also takes a photo off the mirror once it's found to show a person.
OPTS=(-rtpz --chmod=Dg+rwxs,Fg+rw,Fo+r)  # -p: apply these modes; the app needs group write on the db
rsync "${OPTS[@]}" --delete --delete-excluded --exclude-from="$STAGE/private.txt" "$DATA/images/" "$PUBLIC_DEST/images/"
rsync "${OPTS[@]}" "$STAGE/birdbrain.db" "$PUBLIC_DEST/birdbrain.db"
[ ! -f "$STAGE/live.jpg" ] || rsync "${OPTS[@]}" "$STAGE/live.jpg" "$PUBLIC_DEST/live.jpg"
