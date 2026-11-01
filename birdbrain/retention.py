"""Disk space for false-trigger photos: how much they use, an automatic limit, and a manual clean-up.

Deleting a false trigger here deletes only its photo files (photo, thumbnail, burst). The row and its BioCLIP
embedding stay (`visits.photo_deleted_at` is set), so the Pi's animal check keeps learning from it.
False triggers the owner corrected or wrote a note on keep their photos too: they carry the lessons Claude
learns from (db.recent_lessons).
"""

import sqlite3
from datetime import datetime

from . import config, db, images

KEEPS_LESSON = "(corrected_visitor IS NOT NULL OR corrected_species IS NOT NULL OR feedback IS NOT NULL OR marks IS NOT NULL)"

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
CONFIRMED = f"(confirmed_at IS NOT NULL AND NOT {KEEPS_LESSON})"


def _rows(conn: sqlite3.Connection, lessons: bool, confirmed_only: bool = False) -> list[sqlite3.Row]:
    """False triggers that still have their photo: deletable ones (confirmed first, then oldest first), or the
    lessons."""
    keep = KEEPS_LESSON if lessons else f"NOT {KEEPS_LESSON}"
    if confirmed_only:
        keep = CONFIRMED
    return conn.execute(
        f"""SELECT id, image_path, thumb_path, confirmed_at IS NOT NULL AS confirmed FROM visits
            WHERE {db.FALSE_TRIGGER} AND {db.HAS_PHOTO} AND {keep}
            ORDER BY confirmed_at IS NULL, captured_at, id"""
    ).fetchall()


def usage(conn: sqlite3.Connection) -> dict:
    """Space used by false-trigger photos: deletable ones (and how many of those the owner confirmed empty),
    ones kept as lessons, and how many older false triggers already had their photos deleted."""
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


def _delete(conn: sqlite3.Connection, rows: list[sqlite3.Row]) -> int:
    """Delete these false triggers' photo files; keep their rows and embeddings for training."""
    now = datetime.now().isoformat(timespec="seconds")
    conn.executemany("UPDATE visits SET photo_deleted_at = ? WHERE id = ?", [(now, r["id"]) for r in rows])
    conn.commit()
    for r in rows:
        images.delete_frame(r["image_path"], r["thumb_path"])
    return len(rows)


def prune(conn: sqlite3.Connection, limit_bytes: int | None = None) -> int:
    """Delete photos until the rest fit under the limit: ones the owner confirmed empty first, then the oldest.
    Returns how many went."""
    limit = int(limit_mb(conn) * 1_000_000) if limit_bytes is None else limit_bytes
    if limit <= 0:
        return 0  # 0 means no limit
    rows = _rows(conn, lessons=False)
    sizes = images.frames_bytes([(r["image_path"], r["thumb_path"]) for r in rows])
    total, cut = sum(sizes), 0
    while total > limit and cut < len(rows):
        total -= sizes[cut]
        cut += 1
    return _delete(conn, rows[:cut])


def clear(conn: sqlite3.Connection) -> int:
    """The Delete button: photos the owner confirmed empty. Unchecked ones stay so they can still be looked at."""
    return _delete(conn, _rows(conn, lessons=False, confirmed_only=True))
