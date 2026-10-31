#!/bin/bash
# Push a read-only copy of Birdbrain's data to the public mirror (a server running with BIRDBRAIN_PUBLIC=1).
# The Pi only ever connects out; nothing on the internet can reach it. Run from cron every few minutes:
#   */5 * * * * /opt/birdbrain/scripts/push_public.sh >/dev/null 2>&1
# Needs PUBLIC_DEST in the environment or .env, e.g. birdbrain@mirror.somegeek.com:/var/lib/birdbrain
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .env ] && PUBLIC_DEST="${PUBLIC_DEST:-$(grep -E '^PUBLIC_DEST=' .env | cut -d= -f2- | awk '{print $1}')}"
: "${PUBLIC_DEST:?set PUBLIC_DEST}"
DATA="${BIRDBRAIN_DATA:-/var/lib/birdbrain}"
STAGE="$(mktemp -d)"; trap 'rm -rf "$STAGE"' EXIT

# A consistent copy of the database even while the watcher is writing, minus private settings.
sqlite3 "$DATA/birdbrain.db" ".backup '$STAGE/birdbrain.db'"
sqlite3 "$STAGE/birdbrain.db" "DELETE FROM settings WHERE key LIKE 'ntfy_%'; VACUUM;"
# A fresh live frame from the local site (it grabs one from the camera); keep the last one if that fails.
curl -sf --max-time 15 -o "$STAGE/live.jpg" http://127.0.0.1:8000/live.jpg || cp "$DATA/live.jpg" "$STAGE/" 2>/dev/null || true

rsync -az --delete "$DATA/images/" "$PUBLIC_DEST/images/"
rsync -az "$STAGE/birdbrain.db" "$PUBLIC_DEST/birdbrain.db.new"
[ -f "$STAGE/live.jpg" ] && rsync -az "$STAGE/live.jpg" "$PUBLIC_DEST/live.jpg.new"
# Swap the new files in atomically, so the mirror never reads half a database.
ssh "${PUBLIC_DEST%%:*}" "cd '${PUBLIC_DEST#*:}' && mv -f birdbrain.db.new birdbrain.db && { [ ! -f live.jpg.new ] || mv -f live.jpg.new live.jpg; }"
