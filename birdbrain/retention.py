"""Disk space for photos: the false-trigger limit, and the photo budget that thins older real visits.

False triggers: deleting one here deletes only its photo files (photo, thumbnail, burst). The row and its BioCLIP
embedding stay (`visits.photo_deleted_at` is set), so the Pi's animal check keeps learning from it.
False triggers the owner corrected or wrote a note on keep their photos too: they carry the lessons Claude
learns from (db.recent_lessons). So do the ones marked as keepers.

The photo budget (2026-11-02) covers every photo file: real visits, thumbnails, false triggers and their bursts.
Nothing here ever deletes a visit; only photo files change, so counts, first-seen dates, lessons and embeddings all
stay, and every visit keeps its thumbnail. Each day real visits older than `full_days` are thinned once (saved again
at `thin_edge` px, images.thin_photo), except each species' sharpest `full_per_day` photos of each day. When the
photos pass the budget, or the data disk has less free space than `disk_floor_mb`, space is freed in this order:
1) the oldest false-trigger photos, 2) thinning more real visits, oldest first, regardless of age (each species'
best of the day last), 3) the oldest thinned visits keep only their thumbnail. Protected photos are never touched:
keepers, species photos, visits the owner corrected or annotated, the first sighting of each species and every
visit of a species seen fewer than `rare_under` times.

visits.photo_level: NULL = the photo as captured, "thinned", or "thumb" (only the thumbnail is left).
CLI: python -m birdbrain.retention --thin [--dry-run]
"""

import argparse
import logging
import os
import shutil
import sqlite3
import sys
import threading
import time
from collections import defaultdict
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from . import alert, config, db, images

try:
    import fcntl
except ImportError:  # not Linux: the watcher and the web app don't share a machine there
    fcntl = None

log = logging.getLogger("birdbrain.retention")

# Thinning and pruning run in the watcher's retention thread, from the web app's Run now button and from the CLI,
# each in its own process. A thin and a drop of the same photo must never interleave (the photo would be left on
# disk with its row saying 'thumb'), so one run holds this file lock and the others say "already running".
LOCK_FILE = "retention.lock"


class AlreadyRunning(RuntimeError):
    """Another process is thinning or pruning photos right now."""

    def __init__(self) -> None:
        super().__init__("Photo cleanup is already running in another process; try again in a few minutes.")


@contextmanager
def _exclusive() -> Iterator[bool]:
    """Hold DATA_DIR/retention.lock for the block (flock, never waiting). Yields False when another process, or
    another thread of this one, has it."""
    if fcntl is None:
        yield True
        return
    path = config.DATA_DIR / LOCK_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


# The owner's own work on a photo: corrections, notes, circles. These are lessons and training, never deleted.
KEEPS_LESSON = ("(corrected_visitor IS NOT NULL OR corrected_species IS NOT NULL OR corrected_also IS NOT NULL"
                " OR feedback IS NOT NULL OR marks IS NOT NULL)")
# Kept photos: lessons plus the owner's blue keeper mark.
KEPT = f"({KEEPS_LESSON} OR keep IS 1)"

# How false-trigger photos are saved: longest edge in px (None = full size) and JPEG quality.
# Real visits are always saved full size at quality 85.
PHOTO_SIZES = {"small": (640, 70), "medium": (960, 80), "large": (1280, 85), "full": (None, 85)}
LIMIT_CHOICES_MB = (100, 250, 500, 1000, 2000, 0)  # 0 = no limit


def limit_mb(conn: sqlite3.Connection) -> float:
    """The false-trigger limit set on the Settings page, else FALSE_TRIGGER_MAX_MB from .env."""
    value = db.get_value(conn, "false_trigger_max_mb")
    return float(value) if value is not None else config.FALSE_TRIGGER_MAX_MB


def photo_size(conn: sqlite3.Connection) -> str:
    """The false-trigger photo size set on the Settings page, else FALSE_TRIGGER_PHOTO from .env."""
    value = db.get_value(conn, "false_trigger_photo") or config.FALSE_TRIGGER_PHOTO
    return value if value in PHOTO_SIZES else "medium"


def save_options(conn: sqlite3.Connection, limit: float, photo: str) -> int:
    """Store the limit and photo size, then prune to the new limit. Returns how many photos that deleted."""
    if limit < 0 or photo not in PHOTO_SIZES:
        raise ValueError(f"bad storage settings: limit={limit!r}, photo={photo!r}")
    db.set_value(conn, "false_trigger_max_mb", f"{limit:g}")
    db.set_value(conn, "false_trigger_photo", photo)
    return prune(conn)


# Confirmed by the owner as empty and nothing else to keep: the first photos to go.
CONFIRMED = f"(confirmed_at IS NOT NULL AND NOT {KEPT})"


def _rows(conn: sqlite3.Connection, lessons: bool, confirmed_only: bool = False) -> list[sqlite3.Row]:
    """False triggers that still have their photo: deletable ones (confirmed first, then oldest first), or the
    kept ones (lessons and keepers)."""
    keep = KEPT if lessons else f"NOT {KEPT}"
    if confirmed_only:
        keep = CONFIRMED
    return conn.execute(
        f"""SELECT id, image_path, thumb_path, confirmed_at IS NOT NULL AS confirmed FROM visits
            WHERE {db.FALSE_TRIGGER} AND {db.HAS_PHOTO} AND {keep}
            ORDER BY confirmed_at IS NULL, captured_at, id"""
    ).fetchall()


def usage(conn: sqlite3.Connection) -> dict:
    """Space used by false-trigger photos: deletable ones (and how many of those the owner confirmed empty),
    ones kept as lessons or keepers, and how many older false triggers already had their photos deleted."""
    deletable = _rows(conn, lessons=False)
    kept = _rows(conn, lessons=True)
    sizes = images.frames_bytes([(r["image_path"], r["thumb_path"]) for r in deletable + kept])  # one pass over the folders
    confirmed = [size for r, size in zip(deletable, sizes) if r["confirmed"]]
    gone = conn.execute(f"SELECT COUNT(*) FROM visits WHERE {db.FALSE_TRIGGER} AND NOT {db.HAS_PHOTO}").fetchone()[0]
    return {
        "count": len(deletable),
        "bytes": sum(sizes[:len(deletable)]),
        "confirmed_count": len(confirmed),
        "confirmed_bytes": sum(confirmed),
        "lesson_count": len(kept),
        "lesson_bytes": sum(sizes[len(deletable):]),
        "photos_deleted": gone,
        "limit_bytes": int(limit_mb(conn) * 1_000_000),
        "photo": photo_size(conn),
    }


def _delete(conn: sqlite3.Connection, rows: list) -> list:
    """Delete these false triggers' photo files; keep their rows and embeddings for training. Each row is checked
    again as it's marked, under the write lock: one the owner kept, corrected, wrote on or picked as a species
    cover since the rows were chosen (the size scan between is slow) stays, and so does one the other process
    already deleted. Returns the rows whose photos went."""
    if not rows:
        return []
    now = datetime.now().isoformat(timespec="seconds")
    gone = []
    with db.writing(conn):
        covers = set(db.species_photos(conn).values())
        for r in rows:
            if r["id"] in covers:
                continue
            cur = conn.execute(f"UPDATE visits SET photo_deleted_at = ? WHERE id = ? AND NOT {KEPT} AND {db.HAS_PHOTO}",
                               (now, r["id"]))
            if cur.rowcount == 1:
                gone.append(r)
    if gone:
        _note_freed(sum(images.frames_bytes([(r["image_path"], r["thumb_path"]) for r in gone])))
    for r in gone:
        images.delete_frame(r["image_path"], r["thumb_path"])
    return gone


def prune(conn: sqlite3.Connection, limit_bytes: int | None = None) -> int:
    """Delete photos until the rest fit under the limit: ones the owner confirmed empty first, then the oldest.
    Returns how many went (0 when another process is thinning or pruning right now: it prunes too)."""
    limit = int(limit_mb(conn) * 1_000_000) if limit_bytes is None else limit_bytes
    if limit <= 0:
        return 0  # 0 means no limit
    with _exclusive() as mine:
        if not mine:
            log.info("false-trigger prune skipped: %s", AlreadyRunning())
            return 0
        rows = _rows(conn, lessons=False)
        sizes = images.frames_bytes([(r["image_path"], r["thumb_path"]) for r in rows])
        total, cut = sum(sizes), 0
        while total > limit and cut < len(rows):
            total -= sizes[cut]
            cut += 1
        return len(_delete(conn, rows[:cut]))


def clear(conn: sqlite3.Connection) -> int:
    """The Delete button: photos the owner confirmed empty. Unchecked ones stay so they can still be looked at."""
    return len(_delete(conn, _rows(conn, lessons=False, confirmed_only=True)))


# ---------------------------------------------------------------------------------------------------------------
# The photo budget: options

THIN_EDGE_CHOICES = (960, 1280, 1920)  # long edge in px; 1920 is the camera's own size, so it means "don't thin"
FULL_EDGE = max(THIN_EDGE_CHOICES)
OPTIONS = ("budget_mb", "thin_on", "full_days", "full_per_day", "thin_edge", "warn_pct", "rare_under", "disk_floor_mb")
FALLBACK = {"budget_mb": 10000.0, "thin_on": True, "full_days": 90, "full_per_day": 2, "thin_edge": 1280,
            "warn_pct": 80.0, "rare_under": 10, "disk_floor_mb": 5000.0}
# Integer options and their allowed ranges; the MB and percent ones are floats.
_INT_RANGES = {"full_days": (0, 3650), "full_per_day": (0, 100), "rare_under": (0, 100_000)}
_FLOAT_RANGES = {"budget_mb": (0, 100_000_000), "disk_floor_mb": (0, 100_000_000), "warn_pct": (0, 100)}


def _key(name: str) -> str:
    return f"retention.{name}"


def _defaults() -> dict:
    return {"budget_mb": config.PHOTO_BUDGET_MB, "thin_on": config.PHOTO_THIN, "full_days": config.PHOTO_FULL_DAYS,
            "full_per_day": config.PHOTO_FULL_PER_DAY, "thin_edge": config.PHOTO_THIN_EDGE,
            "warn_pct": config.PHOTO_WARN_PCT, "rare_under": config.RARE_UNDER, "disk_floor_mb": config.DISK_FLOOR_MB}


def _valid(name: str, value) -> int | float | bool:
    """The option as it is used (int, float or bool). Raises ValueError (or TypeError) for anything else."""
    if name == "thin_on":
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in ("1", "true", "on", "yes"):
            return True
        if text in ("0", "false", "off", "no"):
            return False
        raise ValueError(f"thin_on must be on or off, not {value!r}")
    if name == "thin_edge":
        edge = int(str(value).strip())
        if edge not in THIN_EDGE_CHOICES:
            raise ValueError(f"thin_edge must be one of {THIN_EDGE_CHOICES}, not {value!r}")
        return edge
    if name in _INT_RANGES:
        n, (lo, hi) = int(str(value).strip()), _INT_RANGES[name]
    elif name in _FLOAT_RANGES:
        n, (lo, hi) = float(str(value).strip()), _FLOAT_RANGES[name]
        if n != n:  # NaN
            raise ValueError(f"{name} must be a number")
    else:
        raise ValueError(f"unknown option {name!r}")
    if not lo <= n <= hi:
        raise ValueError(f"{name} must be between {lo:g} and {hi:g}, not {value!r}")
    return n


def options(conn: sqlite3.Connection) -> dict:
    """Every photo-budget option: the Settings page's value, else the .env default, else the built-in one.

    budget_mb: photo budget (0 = none); thin_on: thin real visits older than full_days; full_per_day: each species'
    sharpest photos per day that stay full size; thin_edge: 960, 1280 or 1920 (= don't thin); warn_pct: one alert
    at this share of the budget (0 = none); rare_under: species with fewer visits keep every photo; disk_floor_mb:
    free disk to keep (0 = don't watch it).
    """
    out, defaults = {}, _defaults()
    for name in OPTIONS:
        for value in (db.get_value(conn, _key(name)), defaults[name], FALLBACK[name]):
            if value is None:
                continue
            try:
                out[name] = _valid(name, value)
                break
            except (TypeError, ValueError):
                continue
    return out


def save_photo_options(conn: sqlite3.Connection, **values) -> dict:
    """Store any of the options (all checked first: one bad value stores nothing). Returns options().

    Raises ValueError naming the bad one. The watcher picks the new values up at its next check (within
    CHECK_EVERY), which thins or frees space if the new budget asks for it.
    """
    clean = {}
    for name, value in values.items():
        if name not in OPTIONS:
            raise ValueError(f"unknown photo budget option {name!r}")
        try:
            clean[name] = _valid(name, value)
        except (TypeError, ValueError) as e:
            raise ValueError(f"bad {name}: {value!r} ({e})") from None
    for name, value in clean.items():
        db.set_value(conn, _key(name), ("1" if value else "0") if isinstance(value, bool) else f"{value:g}")
    forget()
    return options(conn)


# ---------------------------------------------------------------------------------------------------------------
# What's stored, what's protected, and each species' best photos of the day

CONF_RANK = {"high": 2, "medium": 1, "low": 0}
# A thinned photo's size as a share of the original's (measured on real 1920x1080 frames: 0.61 at 1280, 0.38 at
# 960, quality 88 from the stored 85); other sizes scale with (edge / long edge) ** 1.3.
THIN_SIZE_EST = {1280: 0.62, 960: 0.39}
# In the order the breakdown counts them (a photo counts in the first that matches).
REASONS = ("species photo", "keeper", "training", "first sighting", "rare species")


@dataclass
class Photo:
    """One visit's photo files as the budget sees them."""
    id: int
    captured_at: str
    image_path: str
    thumb_path: str
    real: bool
    level: str | None
    sharpness: float | None
    keep: bool
    lesson: bool
    photo: int = 0   # bytes of the photo itself (0 once only the thumbnail is left)
    thumb: int = 0
    burst: int = 0

    @property
    def total(self) -> int:
        return self.photo + self.thumb + self.burst


def _photos(conn: sqlite3.Connection) -> dict[int, Photo]:
    """Every visit that still has photo files, oldest first, with their sizes (one pass over the day folders)."""
    rows = conn.execute(
        f"""SELECT id, captured_at, image_path, thumb_path, {db.REAL} AS real, photo_level, photo_sharpness,
                   keep IS 1 AS keep, {KEEPS_LESSON} AS lesson
            FROM visits WHERE {db.HAS_PHOTO} ORDER BY captured_at, id""").fetchall()
    parts = images.frame_parts([(r["image_path"], r["thumb_path"]) for r in rows])
    out = {}
    for r, (photo, thumb, burst) in zip(rows, parts):
        out[r["id"]] = Photo(r["id"], r["captured_at"], r["image_path"], r["thumb_path"], bool(r["real"]),
                             r["photo_level"], r["photo_sharpness"], bool(r["keep"]), bool(r["lesson"]),
                             photo, thumb, burst)
    return out


def _seen_rows(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Every animal in every real visit (a visit with other kinds in its photo counts for each), as stats count."""
    return conn.execute(f"SELECT id, label, captured_at, confidence FROM ({db._seen(db.REAL)})").fetchall()


def protected(conn: sqlite3.Connection, opts: dict | None = None, seen: list | None = None) -> dict[int, str]:
    """Visits whose photo is never thinned or deleted, with the first reason that applies (REASONS order)."""
    opts = opts or options(conn)
    seen = _seen_rows(conn) if seen is None else seen
    out: dict[int, str] = {}
    for vid in db.species_photos(conn).values():
        out.setdefault(int(vid), "species photo")
    for (vid,) in conn.execute("SELECT id FROM visits WHERE keep = 1"):
        out.setdefault(vid, "keeper")
    for (vid,) in conn.execute(f"SELECT id FROM visits WHERE {KEEPS_LESSON}"):
        out.setdefault(vid, "training")
    first: dict[str, tuple[str, int]] = {}
    ids: dict[str, set[int]] = defaultdict(set)
    for r in seen:  # labels already use the owner's correction, as "first seen" does
        ids[r["label"]].add(r["id"])
        if r["label"] not in first or (r["captured_at"], r["id"]) < first[r["label"]]:
            first[r["label"]] = (r["captured_at"], r["id"])
    for _, vid in first.values():
        out.setdefault(vid, "first sighting")
    for label_ids in ids.values():
        if len(label_ids) < opts["rare_under"]:
            for vid in label_ids:
                out.setdefault(vid, "rare species")
    return out


def best_of_day(photos: dict[int, Photo], seen: list, per_day: int) -> set[int]:
    """Each species' `per_day` best photos of each day: sharpest first (unmeasured ones last), then by confidence,
    then the newest. A visit with several species counts for each and is best if it's best for any of them.
    Visits down to their thumbnail don't compete."""
    if per_day <= 0:
        return set()
    groups: dict[tuple[str, str], dict[int, int]] = defaultdict(dict)  # (label, day) -> {id: confidence rank}
    for r in seen:
        p = photos.get(r["id"])
        if p is None or not p.real or p.level == "thumb":
            continue
        g = groups[(r["label"], r["captured_at"][:10])]
        g[r["id"]] = max(g.get(r["id"], -1), CONF_RANK.get(r["confidence"], -1))
    best: set[int] = set()
    for members in groups.values():
        if len(members) <= per_day:
            best.update(members)
            continue

        def rank(vid: int) -> tuple:
            s = photos[vid].sharpness
            return (s is not None and s >= 0, s or 0.0, members[vid], photos[vid].captured_at, vid)

        best.update(sorted(members, key=rank, reverse=True)[:per_day])
    return best


def _crop_of(raw_json: str | None) -> tuple | None:
    """The crop box a visit's photo was judged in: its saved zones, else today's .env crop."""
    import json

    try:
        raw = json.loads(raw_json or "{}") or {}
    except ValueError:
        raw = {}
    ground = raw.get("trigger") == "ground"
    z = raw.get("zones") or {}
    box = z.get("ground_crop") if ground else z.get("crop")
    if not box:
        box = config.GROUND_CROP if ground else config.CROP_ROI
    return tuple(box) if box and len(box) == 4 else None


def measure_sharpness(image_rel: str, raw_json: str | None = None) -> float | None:
    """Laplacian variance of the stored photo inside its crop box (as watch.sharpness does on the capture), or None
    when the photo can't be read."""
    import cv2  # here, so the web app never loads OpenCV

    gray = cv2.imread(str(config.IMAGES_DIR / image_rel), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return None
    if box := _crop_of(raw_json):
        H, W = gray.shape[:2]
        x, y, w, h = box
        gray = gray[int(y * H):int((y + h) * H), int(x * W):int((x + w) * W)]
    if gray.size == 0:
        return None
    return round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 2)


def measure_new(conn: sqlite3.Connection, busy: threading.Event | None = None,
                say: Callable[[str], None] | None = None) -> int:
    """Measure the sharpness of every real visit's photo that hasn't been (before any thinning). -1 = unreadable.
    ~0.1 s a photo on the Pi; the daily run does the day's new ones. Returns how many."""
    rows = conn.execute(
        f"""SELECT id, image_path, raw_json FROM visits WHERE {db.REAL} AND {db.HAS_PHOTO}
            AND photo_level IS NULL AND photo_sharpness IS NULL ORDER BY id""").fetchall()
    progress = Progress(len(rows), "measured", say) if say and len(rows) > 50 else None
    for r in rows:
        _wait_idle(busy)
        try:
            s = measure_sharpness(r["image_path"], r["raw_json"])
        except Exception:
            log.exception("measuring sharpness of visit %d failed", r["id"])
            s = None
        # One short write each (cheap in WAL mode): the lock is never held while the next photo is read or while
        # waiting for a capture, which would otherwise find the database locked and lose its visit.
        conn.execute("UPDATE visits SET photo_sharpness = ? WHERE id = ?", (-1.0 if s is None else s, r["id"]))
        conn.commit()
        if progress:
            progress.step(failed=s is None)
    return len(rows)


# ---------------------------------------------------------------------------------------------------------------
# Usage, free disk and the breakdown for the Settings card

_lock = threading.Lock()
USAGE_TTL = 3600  # seconds a full count of the photo folders is trusted; stored visits are added in between
_usage: dict = {"bytes": None, "at": 0.0}
BREAKDOWN_TTL = 60
_breakdown: dict = {"at": 0.0, "value": None}


def forget() -> None:
    """Drop the cached usage and breakdown (after the options change, and in tests)."""
    with _lock:
        _usage.update(bytes=None, at=0.0)
        _breakdown.update(at=0.0, value=None)
        _forecast.update(at=0.0, value=None)


def disk_free() -> int:
    """Free bytes on the data volume."""
    try:
        return shutil.disk_usage(config.DATA_DIR).free
    except OSError:
        return 0


def _set_usage(total: int) -> None:
    with _lock:
        _usage.update(bytes=int(total), at=time.monotonic())


def photo_usage(conn: sqlite3.Connection, max_age: float = USAGE_TTL) -> int:
    """Bytes of every photo file the budget counts. A full count at most every `max_age` seconds; note_stored adds
    new visits in between, so the check after each visit is cheap."""
    with _lock:
        cached, at = _usage["bytes"], _usage["at"]
    if cached is not None and time.monotonic() - at < max_age:
        return cached
    total = sum(p.total for p in _photos(conn).values())
    _set_usage(total)
    return total


def note_stored(image_rel: str, thumb_rel: str) -> None:
    """A visit was just stored: add its files to the cached usage and wake the background check."""
    size = images.frame_bytes(image_rel, thumb_rel)
    with _lock:
        if _usage["bytes"] is not None:
            _usage["bytes"] += size
    _wake.set()


def _note_freed(size: int) -> None:
    """Photo files were just deleted: take them off the cached usage, so it doesn't drift up between counts."""
    with _lock:
        if _usage["bytes"] is not None:
            _usage["bytes"] = max(0, _usage["bytes"] - size)


def _tree_bytes(path: Path) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += os.stat(os.path.join(root, name)).st_size
            except OSError:
                pass
    return total


CATEGORIES = (("species_photos", "Species covers"), ("keepers", "Favorites"), ("training", "Training and lessons"),
              ("recent", "Recent visits"), ("older", "Older visits"), ("false_triggers", "False triggers"))


def breakdown(conn: sqlite3.Connection, max_age: float = BREAKDOWN_TTL) -> dict:
    """Where the photo budget goes, for the Settings > Storage card. Cached for a minute.

    categories: [{key, label, count, bytes}] in CATEGORIES order; each visit's photo (and burst) counts once, in the
    first category it fits: species photos, keepers, training (corrected or annotated, real or false), recent real
    visits (younger than full_days), older real visits (thinned or not), false triggers. Thumbnails are summed
    separately (`thumbs`). used_bytes = all of them, which is what the budget counts. levels: how many real visits are
    full, thinned or down to the thumbnail. protected: {count, bytes, by_reason} (photo + thumbnail). disk: free and
    total bytes on the data volume, and the floor. system: models, database and calibration bytes, not in the budget.
    """
    with _lock:
        if _breakdown["value"] is not None and time.monotonic() - _breakdown["at"] < max_age:
            return _breakdown["value"]
    opts = options(conn)
    photos = _photos(conn)
    prot = protected(conn, opts)
    species_ids = set(db.species_photos(conn).values())
    cutoff = (datetime.now() - timedelta(days=opts["full_days"])).isoformat(timespec="seconds")
    cats = {key: {"key": key, "label": label, "count": 0, "bytes": 0} for key, label in CATEGORIES}
    thumbs = {"count": 0, "bytes": 0}
    levels = {"full": 0, "thinned": 0, "thumb": 0}
    reasons = {r: {"count": 0, "bytes": 0} for r in REASONS}
    for p in photos.values():
        if p.id in species_ids and p.real:
            key = "species_photos"
        elif p.keep:
            key = "keepers"
        elif p.lesson:
            key = "training"
        elif p.real:
            key = "recent" if p.captured_at >= cutoff else "older"
        else:
            key = "false_triggers"
        cats[key]["count"] += 1
        cats[key]["bytes"] += p.photo + p.burst
        if p.thumb:
            thumbs["count"] += 1
            thumbs["bytes"] += p.thumb
        if p.real:
            levels[p.level if p.level in ("thinned", "thumb") else "full"] += 1
        if p.id in prot:
            reasons[prot[p.id]]["count"] += 1
            reasons[prot[p.id]]["bytes"] += p.total
    used = sum(p.total for p in photos.values())
    _set_usage(used)
    budget = int(opts["budget_mb"] * 1e6)
    try:
        du = shutil.disk_usage(config.DATA_DIR)
        disk = {"free_bytes": du.free, "total_bytes": du.total}
    except OSError:
        disk = {"free_bytes": 0, "total_bytes": 0}
    db_bytes = sum(Path(f"{config.DB_PATH}{s}").stat().st_size for s in ("", "-wal", "-shm")
                   if Path(f"{config.DB_PATH}{s}").exists())
    value = {
        "categories": list(cats.values()),
        "thumbs": thumbs,
        "used_bytes": used,
        "budget_bytes": budget,
        "used_pct": round(used / budget * 100, 1) if budget else None,
        "warn_pct": opts["warn_pct"],
        "levels": levels,
        "protected": {"count": sum(r["count"] for r in reasons.values()),
                      "bytes": sum(r["bytes"] for r in reasons.values()), "by_reason": reasons},
        "disk": {**disk, "floor_bytes": int(opts["disk_floor_mb"] * 1e6)},
        "system": {"models": _tree_bytes(config.DATA_DIR / "models"), "database": db_bytes,
                   "calibration": _tree_bytes(config.DATA_DIR / "calibration")},
        "false_trigger_limit_bytes": int(limit_mb(conn) * 1e6),
        "options": opts,
        "alerts": {"warned": bool(db.get_value(conn, WARNED)), "over": bool(db.get_value(conn, OVER))},
        "last_run": db.get_value(conn, DAILY_KEY),
        "at": datetime.now().isoformat(timespec="seconds"),
    }
    with _lock:
        _breakdown.update(at=time.monotonic(), value=value)
    return value


# ---------------------------------------------------------------------------------------------------------------
# Forecast: when the photos reach the budget and the disk floor

FORECAST_MIN_DAYS = 7      # less history than this is too little to extrapolate
FORECAST_WINDOW_DAYS = 60  # the pace is measured over this many recent days at most
FORECAST_HORIZON_DAYS = 30 * 366
_forecast: dict = {"at": 0.0, "value": None}


def thin_ratio(edge: int) -> float:
    """A photo's size after thinning to `edge`, as a share of the 1920 px original."""
    return THIN_SIZE_EST.get(edge) or (edge / FULL_EDGE) ** 1.3


def forecast(conn: sqlite3.Connection, now: datetime | None = None, max_age: float = BREAKDOWN_TTL) -> dict:
    """When the photos reach the budget and the disk its floor, at the recent pace.

    The pace is the bytes of every photo file (photo, thumbnail, burst, as stored now) of visits captured in the
    last `window_days` (the days since the first visit, at most 60), per day; it needs at least 7 days of history.
    With thinning on, the same days tell how much of that pace is thinnable (real visits that aren't protected or
    a species' best of the day, `thin_share`); once a photo is `full_days` old it shrinks to `thin_ratio` of its
    size, so the pace after `full_days` is lower, and the photos already stored shrink as they come of age.
    `thin_days` walks that day by day. The disk line counts photo growth only, at the raw pace.

    Returns {ok, reason ("not_enough_data" or None), history_days, window_days, visits_per_day, bytes_per_day,
    budget_days/budget_date, thin (applies?), thin_days/thin_date, thin_ratio, thin_share, disk_days/disk_date,
    disk_floor (bool), at_budget}. A days value is None when it never happens within 30 years (or growth is ~0);
    dates are ISO strings.
    """
    cached = now is None
    if cached:
        with _lock:
            if _forecast["value"] is not None and time.monotonic() - _forecast["at"] < max_age:
                return _forecast["value"]
    now = now or datetime.now()
    opts = options(conn)
    first = conn.execute("SELECT MIN(captured_at) FROM visits").fetchone()[0]
    history = (now - datetime.fromisoformat(first)).total_seconds() / 86400 if first else 0.0
    out: dict = {"ok": False, "reason": "not_enough_data", "history_days": round(history, 1)}
    if history >= FORECAST_MIN_DAYS:
        out = _forecast_from(conn, opts, now, history)
    if cached:
        with _lock:
            _forecast.update(at=time.monotonic(), value=out)
    return out


def _forecast_from(conn: sqlite3.Connection, opts: dict, now: datetime, history: float) -> dict:
    window = max(FORECAST_MIN_DAYS, min(FORECAST_WINDOW_DAYS, int(history)))
    since = (now - timedelta(days=window)).isoformat(timespec="seconds")
    photos = _photos(conn)
    recent = [p for p in photos.values() if p.captured_at >= since]
    per_day = sum(p.total for p in recent) / window
    visits = conn.execute(f"SELECT COUNT(*) FROM visits WHERE captured_at >= ? AND {db.REAL}", (since,)).fetchone()[0]
    budget = int(opts["budget_mb"] * 1e6)
    used = sum(p.total for p in photos.values())
    floor = int(opts["disk_floor_mb"] * 1e6)
    spare = disk_free() - floor
    thin = bool(opts["thin_on"]) and opts["thin_edge"] < FULL_EDGE
    ratio = thin_ratio(opts["thin_edge"]) if thin else 1.0
    no_growth = per_day < 1000  # under a KB a day is no growth

    def date(days):
        return None if days is None else (now + timedelta(days=days)).date().isoformat()

    budget_days = disk_days = thin_days = None
    share = 0.0
    if not no_growth:
        if budget and used < budget:
            budget_days = (budget - used) / per_day
        if spare > 0:
            disk_days = spare / per_day
        if thin and budget and used < budget:
            thin_days, share = _thin_days(conn, opts, photos, recent, per_day, budget - used, ratio, now)
    return {
        "ok": True, "reason": None, "history_days": round(history, 1), "window_days": window,
        "visits_per_day": round(visits / window, 1), "bytes_per_day": int(per_day), "no_growth": no_growth,
        "budget": bool(budget), "at_budget": bool(budget) and used >= budget,
        "budget_days": budget_days, "budget_date": date(budget_days),
        "thin": thin, "thin_ratio": ratio, "thin_share": round(share, 3), "full_days": opts["full_days"],
        "thin_days": thin_days, "thin_date": date(thin_days),
        "disk_days": disk_days, "disk_date": date(disk_days), "disk_floor": floor > 0, "disk_spare": max(spare, 0),
    }


def _thin_days(conn: sqlite3.Connection, opts: dict, photos: dict, recent: list, per_day: float, room: int,
               ratio: float, now: datetime) -> tuple[float | None, float]:
    """Days until `room` more bytes are used when old photos shrink to `ratio`, and the thinnable share of the pace.
    Photos already stored shrink on the day they turn `full_days` old; new ones do the same, at the share of the
    pace that is thinnable. None if that doesn't happen within 30 years."""
    prot = protected(conn, opts)
    best = best_of_day(photos, _seen_rows(conn), opts["full_per_day"])

    def thinnable(p: Photo) -> bool:
        return p.real and p.level not in ("thinned", "thumb") and p.photo > 0 and p.id not in prot and p.id not in best

    total = sum(p.total for p in recent)
    share = sum(p.photo for p in recent if thinnable(p)) / total if total else 0.0
    full_days = opts["full_days"]
    saved_on: dict[int, float] = defaultdict(float)  # day from now -> bytes saved by photos already stored
    for p in photos.values():
        if thinnable(p):
            age = (now - datetime.fromisoformat(p.captured_at)).total_seconds() / 86400
            saved_on[max(0, int(full_days - age))] += p.photo * (1 - ratio)
    future = per_day * share * (1 - ratio)  # saved per day once the first new photos come of age
    grown = -saved_on.pop(0, 0.0)  # photos already old enough shrink at the next daily run
    for day in range(1, FORECAST_HORIZON_DAYS + 1):
        grown += per_day - saved_on.get(day, 0.0) - (future if day > full_days else 0.0)
        if grown >= room:
            return day - (grown - room) / per_day, share
    return None, share


# ---------------------------------------------------------------------------------------------------------------
# Thinning and freeing space

STEPS = (("thin_old", "Thin real visits older than the full-size days"),
         ("false_triggers", "Delete the oldest false-trigger photos"),
         ("thin", "Thin more real visits, oldest first"),
         ("thumb", "Keep only the thumbnail of the oldest visits"))


class Progress:
    """Prints `120/840 thinned, 0 failed, freed 210 MB, ~3 min left` every 50 items or 30 s."""

    def __init__(self, total: int, verb: str, say: Callable[[str], None] | None, every: int = 50,
                 seconds: float = 30):
        self.total, self.verb, self.say, self.every, self.seconds = total, verb, say, every, seconds
        self.done = self.failed = self.freed = 0
        self.start = self.last = time.monotonic()

    def step(self, freed: int = 0, failed: bool = False) -> None:
        self.done += 1
        self.failed += bool(failed)
        self.freed += freed
        now = time.monotonic()
        if self.say and (self.done % self.every == 0 or now - self.last >= self.seconds or self.done == self.total):
            self.last = now
            left = (now - self.start) / self.done * (self.total - self.done)
            eta = f", ~{left / 60:.0f} min left" if left >= 90 else (f", ~{left:.0f} s left" if left >= 1 else "")
            freed_text = f", freed {self.freed / 1e6:.0f} MB" if self.freed else ""
            self.say(f"{self.done}/{self.total} {self.verb}, {self.failed} failed{freed_text}{eta}")


def _wait_idle(busy: threading.Event | None) -> None:
    """Pause while a capture runs, so thinning never slows one down."""
    while busy is not None and busy.is_set():
        time.sleep(0.5)


def _still_unprotected(conn: sqlite3.Connection, vid: int) -> bool:
    """Checked again right before a photo changes: the owner may have marked it since the plan was made."""
    row = conn.execute(f"SELECT keep IS 1 OR {KEEPS_LESSON} FROM visits WHERE id = ?", (vid,)).fetchone()
    return bool(row) and not row[0] and vid not in db.species_photos(conn).values()


def _estimate_thin(p: Photo, edge: int) -> int:
    """What a photo would weigh thinned to `edge`, without doing it."""
    try:
        from PIL import Image

        with Image.open(config.IMAGES_DIR / p.image_path) as im:  # reads the header only
            long = max(im.size)
    except (OSError, ValueError):
        long = FULL_EDGE
    if long <= edge:
        return p.photo
    share = THIN_SIZE_EST.get(edge) if long == FULL_EDGE else None
    return int(p.photo * (share or (edge / long) ** 1.3))


def _need(used: int, free: int, opts: dict) -> int:
    """Bytes to free: what the photos are over the budget, or what the disk is under its floor."""
    over = used - int(opts["budget_mb"] * 1e6) if opts["budget_mb"] > 0 else 0
    short = int(opts["disk_floor_mb"] * 1e6) - free if opts["disk_floor_mb"] > 0 else 0
    return max(over, short, 0)


def _run(conn: sqlite3.Connection, *, apply: bool, age: bool = True, now: datetime | None = None,
         opts: dict | None = None, free: int | None = None, busy: threading.Event | None = None,
         say: Callable[[str], None] | None = None) -> dict:
    """The age rule, then the budget, in order. With apply=False nothing is written: thinned sizes are estimated."""
    opts = opts or options(conn)
    now = now or datetime.now()
    edge = opts["thin_edge"]
    thinning = edge < FULL_EDGE
    photos = _photos(conn)
    seen = _seen_rows(conn)
    prot = protected(conn, opts, seen)
    best = best_of_day(photos, seen, opts["full_per_day"])
    used = sum(p.total for p in photos.values())
    free = disk_free() if free is None else free
    used_before, free_before = used, free
    steps = {key: {"key": key, "label": label, "count": 0, "bytes": 0, "failed": 0} for key, label in STEPS}
    stamp = now.isoformat(timespec="seconds")
    real = [p for p in photos.values() if p.real and p.id not in prot and p.photo > 0]

    def thin(p: Photo, step: dict, progress: Progress | None) -> int:
        if not apply:
            after, failed = _estimate_thin(p, edge), False
        else:
            _wait_idle(busy)  # before the check, so what it saw is still true when the photo changes
            if not _still_unprotected(conn, p.id):
                return 0
            try:
                _, after = images.thin_photo(p.image_path, edge)
                conn.execute("UPDATE visits SET photo_level = 'thinned', photo_thinned_at = ? WHERE id = ?",
                             (stamp, p.id))
                conn.commit()
                failed = False
            except Exception:
                log.exception("thinning visit %d failed", p.id)
                after, failed = p.photo, True
        freed = p.photo - after
        if not failed:
            p.photo, p.level = after, "thinned"
            step["count"] += 1
            step["bytes"] += freed
        step["failed"] += failed
        if progress:
            progress.step(freed, failed)
        return freed

    # 1. The age rule: older visits that aren't protected or among their species' best of the day.
    if age and opts["thin_on"] and thinning:
        cutoff = (now - timedelta(days=opts["full_days"])).isoformat(timespec="seconds")
        old = [p for p in real if p.level is None and p.captured_at < cutoff and p.id not in best]
        progress = Progress(len(old), "thinned", say) if apply and say and old else None
        for p in old:
            freed = thin(p, steps["thin_old"], progress)
            used, free = used - freed, free + freed
    need = _need(used, free, opts)
    # 2. False triggers: the owner's confirmed-empty ones first, then the oldest (as prune does).
    if need > 0:
        chosen, planned = [], need
        for r in _rows(conn, lessons=False):
            if planned <= 0:
                break
            chosen.append(r)
            planned -= photos[r["id"]].total if r["id"] in photos else 0
        # Applied, only the ones still unprotected go (_delete checks again); the rest count as not freed.
        for r in _delete(conn, chosen) if apply and chosen else chosen:
            p = photos.get(r["id"])
            size = p.total if p else 0
            need, used, free = need - size, used - size, free + size
            steps["false_triggers"]["count"] += 1
            steps["false_triggers"]["bytes"] += size
    # 3. Thin more, oldest first, each species' best of the day last.
    if need > 0 and thinning:
        more = sorted((p for p in real if p.level is None), key=lambda p: (p.id in best, p.captured_at, p.id))
        progress = Progress(len(more), "thinned", say) if apply and say and more else None
        for p in more:
            if need <= 0:
                break
            freed = thin(p, steps["thin"], progress)
            need, used, free = need - freed, used - freed, free + freed
    # 4. The oldest thinned visits keep only their thumbnail.
    if need > 0:
        left = sorted((p for p in real if p.level != "thumb" and p.photo > 0),
                      key=lambda p: (p.id in best, p.captured_at, p.id))
        progress = Progress(len(left), "down to the thumbnail", say) if apply and say and left else None
        for p in left:
            if need <= 0:
                break
            if apply:
                if not _still_unprotected(conn, p.id):
                    continue
                conn.execute("UPDATE visits SET photo_level = 'thumb', photo_thinned_at = ? WHERE id = ?",
                             (stamp, p.id))
                conn.commit()
                images.drop_photo(p.image_path)
            size = p.photo + p.burst
            p.photo, p.burst, p.level = 0, 0, "thumb"
            need, used, free = need - size, used - size, free + size
            steps["thumb"]["count"] += 1
            steps["thumb"]["bytes"] += size
            if progress:
                progress.step(size)
    prot_bytes = sum(p.total for p in photos.values() if p.id in prot)
    result = {
        "at": stamp,
        "applied": apply,
        "options": opts,
        "used_before": used_before,
        "used_after": used,
        "budget_bytes": int(opts["budget_mb"] * 1e6),
        "free_before": free_before,
        "free_after": free,
        "floor_bytes": int(opts["disk_floor_mb"] * 1e6),
        "steps": list(steps.values()),
        "freed_bytes": sum(s["bytes"] for s in steps.values()),
        "still_over": need > 0,
        "short_bytes": max(need, 0),
        "protected": {"count": sum(1 for p in photos.values() if p.id in prot), "bytes": prot_bytes},
        "best_count": len(best),
        "unmeasured": sum(1 for p in real if p.level is None and p.sharpness is None),
    }
    if apply:
        _set_usage(used)
        with _lock:
            _breakdown.update(at=0.0, value=None)
        _alerts(conn, result)
    return result


def plan(conn: sqlite3.Connection, now: datetime | None = None, free: int | None = None) -> dict:
    """A dry run of what the daily thinning and the budget would do right now, changing nothing (the Settings
    card's preview). steps: [{key, label, count, bytes, failed}] in the order they run (bytes = freed, thinned sizes
    estimated); used_before/used_after, budget_bytes, free_before/free_after, floor_bytes, still_over (protected
    photos alone are over), protected {count, bytes}, unmeasured (visits whose sharpness isn't known yet, so the
    best-of-day pick for them is by confidence only until the next daily run measures them)."""
    return _run(conn, apply=False, now=now, free=free)


def enforce(conn: sqlite3.Connection, busy: threading.Event | None = None,
            say: Callable[[str], None] | None = None) -> dict:
    """Bring the photos under the budget and the disk above its floor (no age rule). Returns what it did.
    Raises AlreadyRunning when another process is at it."""
    with _exclusive() as mine:
        if not mine:
            raise AlreadyRunning()
        return _run(conn, apply=True, age=False, busy=busy, say=say)


def daily(conn: sqlite3.Connection, now: datetime | None = None, busy: threading.Event | None = None,
          say: Callable[[str], None] | None = None) -> dict:
    """The once-a-day run: measure new photos' sharpness, thin by age, then enforce the budget.
    Raises AlreadyRunning when another process is at it (the watcher's own run, or the web app's Run now)."""
    now = now or datetime.now()
    with _exclusive() as mine:
        if not mine:
            raise AlreadyRunning()
        measure_new(conn, busy, say)
        result = _run(conn, apply=True, now=now, busy=busy, say=say)
        db.set_value(conn, DAILY_KEY, now.date().isoformat())
    return result


# ---------------------------------------------------------------------------------------------------------------
# Alerts: one when the photos reach warn_pct of the budget, one when protected photos alone fill it

WARNED = "retention.warned_at"   # set when the warning went out; cleared once usage drops below warn_pct again
OVER = "retention.over_at"       # the same for "can't get under the budget"


def _flag(conn: sqlite3.Connection, key: str, on: bool, title: str, message: str, send: bool = True) -> None:
    if not on:
        if db.get_value(conn, key):
            conn.execute("DELETE FROM settings WHERE key = ?", (key,))
            conn.commit()
        return
    if db.get_value(conn, key):
        return  # already said
    if send:
        topic = (db.get_value(conn, "ntfy_topic") or "").strip()
        if not topic:
            return  # no phone to tell; it goes out once a topic is set
        qs, qe = db.get_value(conn, "ntfy_quiet_start") or "", db.get_value(conn, "ntfy_quiet_end") or ""
        if qs and qe and alert._in_quiet(datetime.now().time(), qs, qe):
            return  # try again after quiet hours
        click = f"{config.SITE_URL}/settings#storage" if config.SITE_URL else None
        alert.send(topic, title, message, None, click)
    db.set_value(conn, key, datetime.now().isoformat(timespec="seconds"))


def _alerts(conn: sqlite3.Connection, result: dict) -> None:
    opts = result["options"]
    used, budget = result["used_after"], result["budget_bytes"]
    gb = lambda n: f"{n / 1e9:.1f} GB"  # noqa: E731
    if result["still_over"]:
        if budget and used > budget:
            why = (f"Protected photos (favorites, species covers, lessons, first sightings, rare species) and thumbnails "
                   f"alone use {gb(used)}, over the {gb(budget)} photo budget. Raise the budget or unstar some favorites.")
        else:
            why = (f"The data disk has {gb(result['free_after'])} free, under its {gb(result['floor_bytes'])} floor, "
                   "and only protected photos are left to free. Raise the budget, lower the floor or free disk space.")
        _flag(conn, OVER, True, "Birdbrain photo storage is full", why)
    else:
        _flag(conn, OVER, False, "", "")
    pct = used / budget * 100 if budget else 0
    warn = bool(budget and opts["warn_pct"] > 0 and pct >= opts["warn_pct"])
    # When the over alert went out, the warning would only repeat it: mark it said without sending.
    _flag(conn, WARNED, warn, "Birdbrain photos are filling up",
          f"Photos use {gb(used)} of the {gb(budget)} budget ({pct:.0f}%). Past the budget, older photos are thinned "
          "and then cut to their thumbnails; favorites and species covers are always kept.",
          send=not result["still_over"])


# ---------------------------------------------------------------------------------------------------------------
# In the watcher: once a day, plus a cheap check after each stored visit

DAILY_KEY = "retention.daily_on"  # the date of the last daily run
DAILY_HOUR = 3                    # the daily run starts after this hour (few visitors, nothing else running)
CHECK_EVERY = 600                 # seconds between checks even without a new visit (picks up new settings too)
_wake = threading.Event()
_thread: threading.Thread | None = None


def daily_due(conn: sqlite3.Connection, now: datetime | None = None) -> bool:
    now = now or datetime.now()
    return now.hour >= DAILY_HOUR and db.get_value(conn, DAILY_KEY) != now.date().isoformat()


def tick(conn: sqlite3.Connection, now: datetime | None = None, busy: threading.Event | None = None) -> str | None:
    """One check: the daily run when it's due, else free space only if the cached usage or the disk says so.
    Returns "daily", "enforce" or None."""
    try:
        if daily_due(conn, now):
            r = daily(conn, now, busy, log.info)
            log.info("photo budget: daily run freed %.0f MB (%s); photos %.0f MB of %.0f MB",
                     r["freed_bytes"] / 1e6, ", ".join(f"{s['key']} {s['count']}" for s in r["steps"] if s["count"])
                     or "nothing to do", r["used_after"] / 1e6, r["budget_bytes"] / 1e6)
            return "daily"
        opts = options(conn)
        used = photo_usage(conn)
        if _need(used, disk_free(), opts) > 0:
            r = enforce(conn, busy, log.info)
            log.info("photo budget: freed %.0f MB; photos %.0f MB of %.0f MB%s", r["freed_bytes"] / 1e6,
                     r["used_after"] / 1e6, r["budget_bytes"] / 1e6, " (still over: only protected photos left)"
                     if r["still_over"] else "")
            return "enforce"
    except AlreadyRunning as e:
        log.info("photo budget check skipped: %s", e)  # the web app's Run now; the next check carries on
        return None
    budget = int(opts["budget_mb"] * 1e6)
    if budget and opts["warn_pct"] > 0 and used >= budget * opts["warn_pct"] / 100 and not db.get_value(conn, WARNED):
        used = photo_usage(conn, max_age=0)  # a fresh count before warning: photos deleted elsewhere left it high
    _alerts(conn, {"options": opts, "used_after": used, "budget_bytes": budget,
                   "still_over": False, "free_after": disk_free(), "floor_bytes": int(opts["disk_floor_mb"] * 1e6)})
    return None


def _lower_priority() -> None:
    """Linux gives each thread its own nice value: drop this one's to the lowest, so captures always come first."""
    if sys.platform.startswith("linux"):
        try:
            os.setpriority(os.PRIO_PROCESS, threading.get_native_id(), 19)
        except OSError:
            pass


def _loop(busy: threading.Event | None) -> None:
    _lower_priority()
    while True:
        _wake.wait(CHECK_EVERY)
        _wake.clear()
        try:
            with db.session() as conn:
                tick(conn, busy=busy)
        except Exception:
            log.exception("photo budget check failed")


def start_background(busy: threading.Event | None = None) -> threading.Thread:
    """The daily thinning and the budget checks, in a low-priority daemon thread with its own connection."""
    global _thread
    if _thread is None or not _thread.is_alive():
        _thread = threading.Thread(target=_loop, args=(busy,), name="retention", daemon=True)
        _thread.start()
        _wake.set()  # a first check right away
    return _thread


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m birdbrain.retention",
                                     description="Thin older real visits and keep the photos under the budget.")
    parser.add_argument("--thin", action="store_true", required=True,
                        help="measure new photos, thin by age, then free space if over the budget")
    parser.add_argument("--dry-run", action="store_true", help="say what it would do, change nothing")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    mb = lambda n: f"{n / 1e6:,.0f} MB"  # noqa: E731
    with db.session() as conn:
        db.init_db(conn)
        if args.dry_run:
            r = plan(conn)
            print("dry run: nothing is changed")
        else:
            try:
                r = daily(conn, say=print)
            except AlreadyRunning as e:
                sys.exit(str(e))
        o = r["options"]
        print(f"budget {mb(r['budget_bytes']) if r['budget_bytes'] else 'none'}, photos {mb(r['used_before'])}, "
              f"disk free {mb(r['free_before'])} (floor {mb(r['floor_bytes'])}); full size for {o['full_days']} days, "
              f"then {o['thin_edge']} px{'' if o['thin_on'] else ' (age rule off)'}")
        for s in r["steps"]:
            verb = "would free" if args.dry_run else "freed"
            print(f"  {s['label']}: {s['count']} photos, {verb} {mb(s['bytes'])}"
                  + (f", {s['failed']} failed" if s["failed"] else ""))
        print(f"photos after: {mb(r['used_after'])}; protected {r['protected']['count']} ({mb(r['protected']['bytes'])})"
              + ("; still over: only protected photos are left" if r["still_over"] else "")
              + (f"; {r['unmeasured']} not measured yet (picked by confidence)" if args.dry_run and r["unmeasured"]
                 else ""))


if __name__ == "__main__":
    main()
