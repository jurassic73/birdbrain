"""Disk space for false-trigger photos: how much they use, an automatic limit, and a manual clean-up.

False triggers the owner corrected or wrote a note on are never deleted: they carry the lessons Claude
learns from (db.recent_lessons).
"""

import sqlite3

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


def _rows(conn: sqlite3.Connection, lessons: bool) -> list[sqlite3.Row]:
    keep = KEEPS_LESSON if lessons else f"NOT {KEEPS_LESSON}"
    return conn.execute(
        f"SELECT id, image_path, thumb_path FROM visits WHERE {db.FALSE_TRIGGER} AND {keep} ORDER BY captured_at, id"
    ).fetchall()


def usage(conn: sqlite3.Connection) -> dict:
    """Space used by false triggers, split into deletable ones and ones kept as lessons."""
    deletable = _rows(conn, lessons=False)
    kept = _rows(conn, lessons=True)
    size = lambda rows: sum(images.frame_bytes(r["image_path"], r["thumb_path"]) for r in rows)
    return {
        "count": len(deletable),
        "bytes": size(deletable),
        "lesson_count": len(kept),
        "lesson_bytes": size(kept),
        "limit_bytes": int(limit_mb(conn) * 1_000_000),
        "photo": photo_size(conn),
    }


def _delete(conn: sqlite3.Connection, rows: list[sqlite3.Row]) -> int:
    for r in rows:
        conn.execute("DELETE FROM visits WHERE id = ?", (r["id"],))
    conn.commit()
    for r in rows:
        images.delete_frame(r["image_path"], r["thumb_path"])
    return len(rows)


def prune(conn: sqlite3.Connection, limit_bytes: int | None = None) -> int:
    """Delete the oldest deletable false triggers until they fit under the limit. Returns how many went."""
    limit = int(limit_mb(conn) * 1_000_000) if limit_bytes is None else limit_bytes
    if limit <= 0:
        return 0  # 0 means no limit
    rows = _rows(conn, lessons=False)
    sizes = [images.frame_bytes(r["image_path"], r["thumb_path"]) for r in rows]
    total, cut = sum(sizes), 0
    while total > limit and cut < len(rows):
        total -= sizes[cut]
        cut += 1
    return _delete(conn, rows[:cut])


def clear(conn: sqlite3.Connection) -> int:
    """Delete every false trigger except the ones that teach Claude something."""
    return _delete(conn, _rows(conn, lessons=False))
