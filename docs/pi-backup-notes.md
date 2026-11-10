# Pi backup notes (ideas, nothing built yet)

Status: discussion notes from 2026-11-09. No scripts or timers exist. The cloned SD cards (pi-clone) stay as the quick-swap fallback.

## Principle
Treat the card as disposable. Keep what is reproducible in git, and what is irreplaceable in a network backup.

- **Reproducible (git):** setup scripts, service files, `compose.yml`, `.env.example`. A venv is never backed up; rebuild it from the lockfile (`uv sync`).
- **Irreplaceable (network backup):** the data directory (SQLite plus images) and `.env`.
- Recovery: flash a fresh OS, run `deploy.sh`, restore the data.

## Common approaches
1. Config as code (git or Ansible) on a disposable OS. Most common.
2. Docker Compose for everything. Clean (compose file plus volumes), but harder with hardware (Hailo, camera, USB dongles). Only worth it for multi-service stacks.
3. Back up the data, not the machine: `restic` or `borg` nightly to a NAS or cloud.
4. Whole-disk network images (rpi-clone, dd over SSH). Bulky, and live disks can be inconsistent.
5. Network boot or a read-only root. Elegant, fiddly.

## Is Git a backup target?
Good for config, poor for data.

- GitHub free: no hard total cap, but keep repos under about 5 GB (ideally 1 GB). Files over 100 MB are blocked. Git LFS free tier is 1 GB storage and 1 GB bandwidth a month. GitLab free was about 10 GB per project. Limits change; check the current docs.
- A changing SQLite file bloats history forever, images need LFS, and never put secrets in a repo, even a private one.

## Chosen direction: restic to Google Drive via rclone
- `restic -r rclone:<remote>:<folder>`. Restic packs files into a modest number of large, encrypted chunks, which suits Drive. It encrypts before upload and deduplicates, so nightly runs are small.
- Gotchas:
  - Create your own Google API client ID in rclone (the shared default is throttled).
  - Put the OAuth app "In production", or the token expires every 7 days and backups silently stop.
  - On a headless Pi, run `rclone authorize "drive"` on the Mac and paste the token into the Pi's config.
  - Use `drive.file` scope or a dedicated folder so the Pi's credential cannot read the rest of Drive.
  - Drive counts against the shared quota (Gmail, Photos).
  - Keep one other copy (the Mac or a NAS) in case the Google account is locked.
- Alternative: Pis back up to the Mac over SSH, with the repo in a Drive for desktop folder. Simpler, but needs the Mac on.
- Other cheap off-site options: Backblaze B2 (about $6 per TB a month), Cloudflare R2 (10 GB free).

## Backing up a live Pi
Restic is only a reader; services keep running. Run it under `nice` and `ionice`, at night, with `--limit-upload` if needed.

- Static files (images, code, configs): fine live.
- **SQLite in WAL mode: never copy the live file.** First make a consistent copy, then back up the copy and exclude the live `.db`, `-wal` and `-shm`:
  `sqlite3 <data>/birdbrain.db ".backup '<data>/backup/birdbrain.db'"`
- Docker volumes with databases: stop the container briefly or use the database's own dump tool.
- Do not back up `/`. Back up `/etc`, `/opt`, `/var/lib/<service>`, the home directory and a package list (`dpkg --get-selections`). Exclude `/proc`, `/sys`, `/dev`, `/run`, `/tmp` and caches.

## Planned shape (per Pi)
1. A systemd timer fires nightly.
2. Pre-step: `sqlite3 .backup` and the package list dump.
3. `restic backup` on the chosen paths.
4. Weekly `restic forget --keep-daily 7 --keep-weekly 4 --prune`.
5. Monthly `restic check` with a small data sample.
6. A failure sends an ntfy alert.

Restore: flash a fresh OS, install restic, `restic restore latest --target ...`, or `restic mount` to browse snapshots. Test one restore onto a spare folder or card before trusting it.

## Per-host sketch
- **odin:** back up the data directory and `.env`; code and venv come from `deploy.sh`.
- **outpost:** the stack repo plus its Docker volumes.
- **Other Pis:** a setup script in a repo, plus a nightly copy of `/etc` and service data.

## Open
- Check how big the odin data directory is, to size the repo and the first upload.
- Write the script, service and timer on a branch; review before deploying.
