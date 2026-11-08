#!/bin/bash
# Push a read-only copy of Birdbrain's data to the public mirror (a server running with BIRDBRAIN_PUBLIC=1).
# The Pi only ever connects out; nothing on the internet can reach it. Run from cron every few minutes:
#   */5 * * * * /opt/birdbrain/scripts/push_public.sh >/dev/null 2>&1
# PUBLIC_DEST (env or .env) is an rsync target, e.g. user@mirror-host:  (the mirror's key is locked to
# rrsync in its data folder, so the path is relative to that). PUBLIC_KEY is the ssh key to use.
set -euo pipefail
cd "$(dirname "$0")/.."
# Cron and the watcher throw this script's output away, so anything worth knowing also goes to the system log
# (journalctl -t push_public): a failed step and what the self-check at the end finds.
warn() { echo "push_public: $*" >&2; if command -v logger >/dev/null 2>&1; then logger -t push_public -- "$*" || true; fi; }
trap 'warn "failed (exit $?): $BASH_COMMAND"' ERR
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
#
# Who may read the files on the mirror. They belong to the push user and the service reads them through
# its group, so the group must be the service's and the modes must let the group in:
# - The group is never sent: -rtpz on purpose, not -a, which would add -o and -g. A push user that isn't root
#   can't give files to another owner anyway, and -g would make rsync try the Pi's group names on the mirror.
#   Without them every new file and folder takes the group of the folder it lands in, as Linux does for a
#   folder with the setgid bit; so the data folder and images/ are setgid and group-owned by the service's
#   group once, by hand (README), and the modes below keep the bit on every folder the pushes create under
#   them (rsync makes each folder of the tree itself, images/2026, images/2026/11 and so on, and applies
#   the D mode to each, images/ included).
# - The modes are absolute, not relative to the Pi's: folders 2770 (setgid, group may enter), files 660 (the
#   app needs group write on the database), nothing for anyone else. --chmod is applied here on the Pi, to
#   the modes rsync sends, so the rrsync lock on the mirror never sees the option. Octal and the D/F forms
#   need rsync 3 (the Pi); older ones and openrsync (a Mac) push with plain modes.
OPTS=(-rtpz)
if [ "$(rsync --version 2>/dev/null | sed -nE '1s/^rsync +version +([0-9]+)\..*/\1/p')" -ge 3 ] 2>/dev/null; then
  OPTS+=(--chmod=D2770,F660)
fi
rsync "${OPTS[@]}" -L --delete "$STAGE/images/" "$PUBLIC_DEST/images/"
rsync "${OPTS[@]}" "$STAGE/birdbrain.db" "$PUBLIC_DEST/birdbrain.db"
[ ! -f "$STAGE/live.jpg" ] || rsync "${OPTS[@]}" "$STAGE/live.jpg" "$PUBLIC_DEST/live.jpg"

# Self-check: list the mirror's photo folders (names and modes only, one small listing; rsync's listing has
# no owner or group, and the key is locked to rsync, so the group itself can't be checked from here) and warn
# about any folder the service's group can't enter or that lost its setgid bit, since everything made under
# such a folder takes the push user's own group and the photos in it come up broken. Files aren't listed:
# their mode is forced above, and rsync fails the push (exit 23) when it can't set one. Folders only, through
# rsync's filter rules (they travel inside the connection, not as options the lock would refuse). Never fails
# the push; its words go to stderr and the system log (journalctl -t push_public).
LISTING="$(rsync -r --list-only --include='*/' --exclude='*' "$PUBLIC_DEST/images/" 2>/dev/null)" \
  || { warn "self-check skipped: couldn't list the mirror's photo folders"; LISTING=""; }
BAD="$(printf '%s\n' "$LISTING" | awk '$1 ~ /^d/ && substr($1, 5, 3) != "rws" && n < 5 { print $NF " is " $1; n++ }')"
[ -z "$BAD" ] || warn "mirror folders the service can't read (want drwxrws---, see README, a public read-only copy): $(printf '%s' "$BAD" | tr '\n' ';')"
