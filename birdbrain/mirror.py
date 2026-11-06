"""What the public mirror gets (scripts/push_public.sh): a trimmed copy of the database and the exact list of
photo files to send with it.

The Pi's database holds the owner's settings, notes and training data; the mirror (BIRDBRAIN_PUBLIC=1) only
needs what its pages read. So the copy keeps an allowlist of settings keys and tables, and the visits lose the
owner's notes, Make public calls and the parts of raw_json nothing public shows. The owner's circle checks (Ask
BioCLIP 2 / Haiku / Sonnet on the visit page, raw_json.calls.owner_checks) go along unless Settings > Public mirror
says not to (OWNER_CHECKS_KEY, read here on the Pi, so the data never leaves it). Only the files the trimmed,
privacy-filtered copy refers to (each visit's photo, thumbnail and burst frames) are pushed, so a photo saved
after the copy was made, an orphan or a temporary file never goes out. They're staged as symlinks in a folder
rsync sends with --copy-links --delete, so whatever else is on the mirror is removed. False triggers and "other"
visits (the admin's pages only) are dropped from the copy, so their photos and bursts are never listed and the next
push removes them.

CLI (exit 0 = ok, 2 = error; the push script stops on a nonzero exit, so nothing goes out half done):
  python -m birdbrain.mirror prepare SRC.db STAGE [TZ]    copy, trim and filter SRC.db into STAGE/birdbrain.db,
                                                           write STAGE/files.txt and stage STAGE/images/
"""
import json
import os
import shutil
import sqlite3
import sys
from pathlib import Path

from . import clock, config, db, images, privacy, zones

# Settings the public pages read (settings table). Everything else (Claude's Notes, alert topics, the camera log,
# calibration status, legacy blobs) stays on the Pi.
SETTINGS_KEYS = frozenset({
    clock.KEY,                          # site_timezone: the mirror's "today" is the birdbath's (clock.py)
    db.SPECIES_PHOTO,                   # the owner's cover photo of each species (species pages)
    "presence_mode", "detector_mode",   # which local checks run, named on Stats and About
    zones.KEY, "zones_auto", "zones_times",  # the zones figure on About, the crop box on visit photos, the recal times
    "false_trigger_max_mb", "false_trigger_photo",  # the storage section on About
})
SETTINGS_PREFIXES = ("retention.",      # photo budget options and flags (About)
                     "refphoto:")       # species reference photos (species cards)
# Visit columns blanked: the owner's note for Claude (a lesson, shown on the Pi's visit page only). public_ok is
# blanked by the privacy filter once it has read it.
PRIVATE_COLUMNS = ("feedback",)
# Visit columns put back to their default: the admin's own marks that only admin pages read (a lesson switched off
# on Claude's Notes).
ADMIN_COLUMNS = {"lesson_off": 0}
# What the visit page, Stats and About read from raw_json: the identification trail with its token counts, the
# zone that fired, the burst's frame info and the zones in use. The rest (the answer again, motion figures, the
# probe and frame counts) is the watcher's own record.
RAW_KEYS = ("calls", "trigger", "burst", "zones")
# Tables emptied: Ask-the-Pi timings (the admin's button) and the animal check's embeddings (its training data).
EMPTIED_TABLES = ("pi_checks", "embeddings")
# Settings > Public mirror: "Show my circle checks on the public mirror". "0" = off; unset or anything else = on
# (the owner finds them helpful there). The key stays on the Pi: the mirror's pages never read it.
OWNER_CHECKS_KEY = "mirror_owner_checks"
OWNER_CHECKS_PATH = "owner_checks"  # under raw_json.calls (circle_check.record)


def owner_checks_shared(conn: sqlite3.Connection) -> bool:
    """Whether the owner's circle checks go to the mirror (the setting, on by default)."""
    row = conn.execute("SELECT value FROM settings WHERE key = ?", (OWNER_CHECKS_KEY,)).fetchone()
    return row is None or row[0] != "0"


def public_raw(text: str | None, owner_checks: bool = True) -> str | None:
    """A visit's raw_json cut down to RAW_KEYS (None when there's nothing readable to keep). With owner_checks off,
    the owner's circle checks are dropped from calls too (Claude's own answer and circle stay)."""
    try:
        raw = json.loads(text or "")
    except ValueError:
        return None
    if not isinstance(raw, dict):
        return None
    kept = {k: raw[k] for k in RAW_KEYS if k in raw}
    if not owner_checks and isinstance(kept.get("calls"), dict) and OWNER_CHECKS_PATH in kept["calls"]:
        kept["calls"] = {k: v for k, v in kept["calls"].items() if k != OWNER_CHECKS_PATH}
    return json.dumps(kept) if kept else None


def trim(conn: sqlite3.Connection) -> None:
    """Strip a copy of the database down to what the public pages read. Never run on the Pi's own database.
    Every push rebuilds the whole copy from the Pi's database (copy_db), so a setting changed on the Pi is in force
    on the next push; nothing stale stays on the mirror."""
    share_checks = owner_checks_shared(conn)  # read before the settings are trimmed: the key itself stays on the Pi
    keys = ", ".join("?" * len(SETTINGS_KEYS))
    like = " OR ".join("key LIKE ?" for _ in SETTINGS_PREFIXES)
    conn.execute(f"DELETE FROM settings WHERE key NOT IN ({keys}) AND NOT ({like})",
                 [*sorted(SETTINGS_KEYS), *(p + "%" for p in SETTINGS_PREFIXES)])
    # False triggers and "other" visits (an animal nobody could name, db.OTHER) are for the admin only: they stay on
    # the Pi, so neither their rows nor their photos, thumbnails and burst frames (files_of lists only the visits left)
    # reach the mirror. Nothing else in the copy points at a visit: embeddings and pi_checks are emptied below, and a
    # species cover pick whose visit is gone falls back to the automatic pick (db.species_summary).
    conn.execute(f"DELETE FROM visits WHERE {db.ADMIN_ONLY}")
    rows = conn.execute("SELECT id, raw_json FROM visits WHERE raw_json IS NOT NULL").fetchall()
    conn.executemany("UPDATE visits SET raw_json = ? WHERE id = ?",
                     [(public_raw(raw, share_checks), vid) for vid, raw in rows])
    for sql in (*(f"UPDATE visits SET {col} = NULL" for col in PRIVATE_COLUMNS),
                *(f"UPDATE visits SET {col} = {value}" for col, value in ADMIN_COLUMNS.items()),
                *(f"DELETE FROM {table}" for table in EMPTIED_TABLES),
                "UPDATE calibrations SET summary = '', image_path = NULL"):  # the zones history stays: older
        try:                                                                   # visit photos draw their zones from it
            conn.execute(sql)
        except sqlite3.OperationalError:
            pass  # a database from before that column or table existed
    conn.commit()


def copy_db(src: Path, dst: Path, timezone: str = "") -> None:
    """A consistent copy of the database even while the watcher is writing, trimmed, as one plain file (no WAL)
    carrying the Pi's time zone so the mirror's "today" matches the birdbath's."""
    source, copy = sqlite3.connect(src), sqlite3.connect(dst)
    source.row_factory = sqlite3.Row  # zones.load reads settings by column name
    try:
        source.backup(copy)
        copy.execute("PRAGMA journal_mode=DELETE")
        trim(copy)
        # The mirror has no zone settings in its .env, and a saved recalibration only counts where .env matches, so
        # send the zones the Pi is using as a snapshot the mirror trusts (zones.stored, "pushed").
        copy.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", (zones.KEY, json.dumps({
            "pushed": True, "adjusted": zones.adjusted(source), "zones": zones.load(source).to_dict()})))
        if timezone:
            copy.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", (clock.KEY, timezone))
        copy.commit()
        # Rebuild the file so what trim deleted (false triggers, the owner's notes, raw_json's private parts, the
        # emptied tables) doesn't linger in its free pages, where anyone with the file could read it back.
        copy.execute("VACUUM")
    finally:
        source.close()
        copy.close()


def inside(rel: str) -> bool:
    """Whether a visit's path stays under the images folder: relative, no '..' step."""
    parts = Path(rel).parts
    return bool(parts) and not Path(rel).is_absolute() and ".." not in parts


def is_temp(name: str) -> bool:
    """A file being written: a thumbnail remade (x.jpg.tmp) or a photo thinned (.x.jpg.thin.tmp), images.py."""
    return name.startswith(".") or name.endswith(".tmp")


def files_of(db_path: Path, images_dir: Path | None = None) -> list[str]:
    """Every photo file the (filtered) copy refers to that is on disk, as paths relative to the images folder:
    each visit's photo, its thumbnail and its burst frames (with a picked frame's thumbnail), never a false trigger's
    or an "other" visit's (trim already dropped them; this is the second lock). Nothing else goes: not
    a photo saved since the copy was made, not an orphan, never a temporary file. Each day folder is listed once
    (images.frame_parts)."""
    images_dir = images_dir or config.IMAGES_DIR
    conn = sqlite3.connect(db_path)
    try:
        visits = conn.execute(f"SELECT image_path, thumb_path FROM visits WHERE {db.PUBLIC} ORDER BY id").fetchall()
    finally:
        conn.close()
    folders: dict[Path, tuple[set[str], dict[str, list[tuple[int, bool, str]]]]] = {}

    def folder(rel: str) -> tuple[set[str], dict[str, list[tuple[int, bool, str]]]]:
        parent = Path(rel).parent
        if parent not in folders:
            names: set[str] = set()
            bursts: dict[str, list[tuple[int, bool, str]]] = {}  # photo stem -> [(n, is thumb, name)] of its burst files
            try:
                with os.scandir(images_dir / parent) as entries:
                    for e in entries:
                        if is_temp(e.name) or not e.is_file():
                            continue
                        names.add(e.name)
                        if parsed := images.burst_file(e.name):
                            bursts.setdefault(parsed[0], []).append((parsed[1], parsed[2], e.name))
            except (FileNotFoundError, NotADirectoryError):
                pass
            folders[parent] = (names, bursts)
        return folders[parent]

    out: list[str] = []
    seen: set[str] = set()
    for image_rel, thumb_rel in visits:
        if not (inside(image_rel) and (not thumb_rel or inside(thumb_rel))):
            continue  # a path that would reach outside the images folder is never staged (rsync -L would follow it)
        wanted = []
        names, bursts = folder(image_rel)
        parent = Path(image_rel).parent
        if Path(image_rel).name in names:
            wanted.append(image_rel)
        wanted += [str(parent / name) for *_, name in sorted(bursts.get(Path(image_rel).name.removesuffix(".jpg"), []))]
        if thumb_rel and Path(thumb_rel).name in folder(thumb_rel)[0]:
            wanted.append(thumb_rel)
        for rel in wanted:
            rel = str(Path(rel))  # no "./": rsync wants the path as it is under the images folder
            if rel not in seen and not is_temp(Path(rel).name):
                seen.add(rel)
                out.append(rel)
    return out


def stage_files(files: list[str], stage: Path, images_dir: Path | None = None) -> Path:
    """Symlinks to the listed files under stage/images, in the same folders, for rsync --copy-links --delete:
    the mirror ends up with exactly these files and nothing else."""
    images_dir = Path(images_dir or config.IMAGES_DIR).absolute()  # the links must work from anywhere
    root = stage / "images"
    shutil.rmtree(root, ignore_errors=True)  # never a link left over from an earlier list
    root.mkdir(parents=True)
    for rel in files:
        link = root / rel
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(images_dir / rel)
    return root


def prepare(src: Path, stage: Path, timezone: str = "") -> tuple[int, int]:
    """Everything the push script sends: STAGE/birdbrain.db (trimmed, minus visits with a person in a photo),
    STAGE/files.txt and STAGE/images/. Returns (files listed, visits kept off the mirror)."""
    stage.mkdir(parents=True, exist_ok=True)
    dst = stage / "birdbrain.db"
    copy_db(src, dst, timezone)
    blocked = privacy.filter_db(dst, stage / "private.txt")
    files = files_of(dst)
    (stage / "files.txt").write_text("".join(f"{rel}\n" for rel in files))
    stage_files(files, stage)
    print(f"mirror: {len(files)} photo files to push")
    return len(files), blocked


def main(argv: list[str]) -> int:
    try:
        if 3 <= len(argv) <= 4 and argv[0] == "prepare":
            prepare(Path(argv[1]), Path(argv[2]), argv[3] if len(argv) > 3 else "")
            return 0
    except Exception as e:  # fail closed: the push script stops on a nonzero exit
        print(f"mirror copy failed: {e}", file=sys.stderr)
        return 2
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
