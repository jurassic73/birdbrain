"""Where the shadow detector keeps its notes: its own SQLite file (events and heartbeats), the event snapshot
folder with its size cap, the heartbeat JSON, and read-only access to the production database.

Nothing here ever opens birdbrain.db for writing: production_db() uses SQLite's mode=ro URI.
"""

import json
import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from .. import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (          -- one row per tracked object, from its first to its last frame
    id INTEGER PRIMARY KEY,
    started_at TEXT NOT NULL,                -- local time of the first frame it was seen in
    ended_at TEXT,                           -- NULL while it is still in view
    label TEXT,                              -- the COCO class seen most often
    score REAL,                              -- the best score of its frames
    labels TEXT,                             -- JSON {class: frames}
    animal INTEGER NOT NULL DEFAULT 0,       -- any animal-like class among them (geometry.ANIMAL_CLASSES)
    zone TEXT,                               -- bath, crop, ground or elsewhere (where it mostly was)
    in_zone INTEGER NOT NULL DEFAULT 0,      -- bath or crop: the area the Motion detector makes visits from
    frames INTEGER NOT NULL DEFAULT 0,       -- frames it was detected in
    seconds REAL,
    box TEXT,                                -- JSON [x0, y0, x1, y1] of the best detection, frame fractions
    snapshots TEXT,                          -- JSON list of paths under HAILO_SNAPSHOT_DIR (pre-trigger shots first)
    model TEXT,                              -- HEF name
    fps REAL,                                -- the detector's frame rate when the event started
    night INTEGER NOT NULL DEFAULT 0,
    concurrent INTEGER NOT NULL DEFAULT 1,   -- most events in view at once while this one lived
    track_id INTEGER
);
CREATE INDEX IF NOT EXISTS events_started_at ON events(started_at);
CREATE TABLE IF NOT EXISTS heartbeats (      -- one row a minute while the service runs
    at TEXT PRIMARY KEY,
    fps REAL,
    infer_ms REAL,
    cpu_pct REAL,
    chip_temp_c REAL,
    frames INTEGER,
    model TEXT
);
"""

EVENT_COLUMNS = ("started_at", "ended_at", "label", "score", "labels", "animal", "zone", "in_zone", "frames", "seconds",
                 "box", "snapshots", "model", "fps", "night", "concurrent", "track_id")


def connect(path: Path | None = None) -> sqlite3.Connection:
    path = path or config.HAILO_SHADOW_DB
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA)
    return conn


@contextmanager
def session(path: Path | None = None) -> Iterator[sqlite3.Connection]:
    conn = connect(path)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def shadow_ro(path: Path | None = None, timeout: float = 1.0) -> sqlite3.Connection:
    """The detector's database for readers that must never block or create it (the Stats page): read-only, a short
    timeout, no schema statements. Raises sqlite3.Error when it is missing, locked or has no tables yet."""
    path = path or config.HAILO_SHADOW_DB
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=timeout)
    conn.row_factory = sqlite3.Row
    return conn


def production_db(path: Path | None = None) -> sqlite3.Connection:
    """The watcher's database, read-only (SQLite's mode=ro URI: no lock it could leave behind, no write possible)."""
    path = path or config.DB_PATH
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def insert_event(conn: sqlite3.Connection, **fields) -> int:
    cols = [c for c in EVENT_COLUMNS if c in fields]
    cur = conn.execute(f"INSERT INTO events ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                       [_sql(fields[c]) for c in cols])
    conn.commit()
    return cur.lastrowid


def update_event(conn: sqlite3.Connection, event_id: int, **fields) -> None:
    cols = [c for c in EVENT_COLUMNS if c in fields]
    conn.execute(f"UPDATE events SET {', '.join(f'{c} = ?' for c in cols)} WHERE id = ?",
                 [_sql(fields[c]) for c in cols] + [event_id])
    conn.commit()


def _sql(value):
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value)
    if isinstance(value, bool):
        return int(value)
    return value


def add_heartbeat(conn: sqlite3.Connection, **fields) -> None:
    cols = ("at", "fps", "infer_ms", "cpu_pct", "chip_temp_c", "frames", "model")
    conn.execute(f"INSERT OR REPLACE INTO heartbeats ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                 [fields.get(c) for c in cols])
    conn.commit()


def events_since(conn: sqlite3.Connection, since: str) -> list[sqlite3.Row]:
    return conn.execute("SELECT * FROM events WHERE started_at >= ? ORDER BY started_at", (since,)).fetchall()


def event_counts(conn: sqlite3.Connection, since: str) -> dict:
    row = conn.execute(
        "SELECT COUNT(*) n, SUM(animal) animals, SUM(animal AND in_zone) in_zone, SUM(ended_at IS NULL) open "
        "FROM events WHERE started_at >= ?", (since,)).fetchone()
    return {"events": row["n"] or 0, "animals": row["animals"] or 0, "in_zone": row["in_zone"] or 0,
            "open": row["open"] or 0}


# --- heartbeat file ----------------------------------------------------------------------------------------------


def write_heartbeat(data: dict, path: Path | None = None) -> None:
    """Atomic: written beside the file and renamed over it, so a reader never sees half a JSON."""
    path = path or config.HAILO_HEARTBEAT
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data))
    os.replace(tmp, path)


def read_heartbeat(path: Path | None = None) -> dict | None:
    path = path or config.HAILO_HEARTBEAT
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


# --- snapshots ---------------------------------------------------------------------------------------------------


def snapshot_rel(when: datetime, event_id: int, tag: str) -> str:
    """YYYY/MM/DD/HHMMSS_<event>_<tag>.jpg under HAILO_SNAPSHOT_DIR."""
    return f"{when:%Y/%m/%d}/{when:%H%M%S}_{event_id}_{tag}.jpg"


def save_jpeg(rel: str, jpeg: bytes, root: Path | None = None) -> Path:
    path = (root or config.HAILO_SNAPSHOT_DIR) / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(jpeg)
    return path


def folder_size(root: Path) -> int:
    return sum(p.stat().st_size for p in root.rglob("*.jpg")) if root.exists() else 0


def prune_snapshots(root: Path | None = None, max_mb: float | None = None) -> int:
    """Delete the oldest snapshot files (by name, which starts with the date and time) until the folder is under
    the cap; empty day folders go too. Returns how many files were removed. 0 or less for max_mb: no cap."""
    root = root or config.HAILO_SNAPSHOT_DIR
    max_mb = config.HAILO_SNAPSHOT_MAX_MB if max_mb is None else max_mb
    if max_mb <= 0 or not root.exists():
        return 0
    cap = int(max_mb * 1024 * 1024)
    files = sorted(root.rglob("*.jpg"), key=lambda p: str(p.relative_to(root)))
    total = sum(p.stat().st_size for p in files)
    removed = 0
    for p in files:
        if total <= cap:
            break
        total -= p.stat().st_size
        p.unlink()
        removed += 1
    for folder in sorted((d for d in root.rglob("*") if d.is_dir()), key=lambda d: len(d.parts), reverse=True):
        if not any(folder.iterdir()):
            folder.rmdir()
    return removed
