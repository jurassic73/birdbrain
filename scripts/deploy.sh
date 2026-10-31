#!/usr/bin/env bash
# Deploy code, .env and services to the Pi.
# Usage: scripts/deploy.sh [user@host]   (default: DEPLOY_USER@DEPLOY_HOST from .env)
set -euo pipefail
cd "$(dirname "$0")/.."

[ -f .env ] || { echo "No .env here. Copy .env.example to .env and fill it in first." >&2; exit 1; }
env_value() { grep -E "^$1=" .env | tail -1 | cut -d= -f2- | sed -E 's/[[:space:]]+#.*$//; s/[[:space:]]+$//' || true; }

TARGET="${1:-}"
if [ -z "$TARGET" ]; then
  DEPLOY_HOST="$(env_value DEPLOY_HOST)"
  DEPLOY_USER="$(env_value DEPLOY_USER)"
  [ -n "$DEPLOY_HOST" ] || { echo "Set DEPLOY_HOST in .env, or pass user@host." >&2; exit 1; }
  TARGET="${DEPLOY_USER:-pi}@${DEPLOY_HOST}"
fi
RUN_AS="${TARGET%@*}"
APP=/opt/birdbrain
DATA=/var/lib/birdbrain

reqs="$(mktemp)"
trap 'rm -f "$reqs"' EXIT
uv export --frozen --no-dev --no-hashes --no-emit-project --quiet -o "$reqs"

ssh "$TARGET" "sudo mkdir -p $APP $DATA && sudo chown $RUN_AS:$RUN_AS $APP $DATA"
parts=(birdbrain deploy pyproject.toml uv.lock)
[ -d models ] && parts+=(models) || echo "No models/ folder: the local detector stays off on the Pi (run scripts/get_model.sh)."
rsync -az --delete --exclude '__pycache__' "${parts[@]}" "$TARGET:$APP/"
ssh "$TARGET" "umask 077 && cat > $APP/.env && chmod 600 $APP/.env" < .env
rsync -a "$reqs" "$TARGET:$APP/requirements.txt"

ssh "$TARGET" bash -s <<EOF
set -euo pipefail
cd $APP
[ -x venv/bin/python ] || python3 -m venv venv
venv/bin/pip install --quiet --upgrade pip
venv/bin/pip install --quiet -r requirements.txt
for unit in birdbrain-web birdbrain-watch; do
  sed "s/__USER__/$RUN_AS/" deploy/\$unit.service | sudo tee /etc/systemd/system/\$unit.service >/dev/null
done
sudo systemctl daemon-reload
sudo systemctl enable --quiet birdbrain-web birdbrain-watch
sudo systemctl restart birdbrain-web birdbrain-watch
sleep 3
systemctl --no-pager --lines=0 status birdbrain-web birdbrain-watch | grep -E "^(●|     Active)"
EOF
echo "Site: http://${TARGET#*@}:8000"
