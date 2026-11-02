import json
import re
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from pathlib import Path

from . import clock, config

SCHEMA = """
CREATE TABLE IF NOT EXISTS visits (
    id INTEGER PRIMARY KEY,
    captured_at TEXT NOT NULL,
    image_path TEXT NOT NULL,
    thumb_path TEXT NOT NULL,
    visitor TEXT NOT NULL,
    species TEXT,
    count INTEGER,
    confidence TEXT,
    notes TEXT,
    model TEXT,
    escalated INTEGER NOT NULL DEFAULT 0,
    needs_review INTEGER NOT NULL DEFAULT 0,
    corrected_species TEXT,
    corrected_visitor TEXT,
    feedback TEXT,
    marks TEXT,
    also TEXT,
    corrected_also TEXT,
    lesson_off INTEGER NOT NULL DEFAULT 0,
    is_night INTEGER NOT NULL DEFAULT 0,
    raw_json TEXT
);
CREATE INDEX IF NOT EXISTS visits_captured_at ON visits(captured_at);
CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS species_info (
    name TEXT PRIMARY KEY,
    data TEXT NOT NULL,
    fetched_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS counters (
    day TEXT NOT NULL,
    name TEXT NOT NULL,
    n INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (day, name)
);
CREATE TABLE IF NOT EXISTS calibrations (
    id INTEGER PRIMARY KEY,
    ran_at TEXT NOT NULL,
    reason TEXT NOT NULL,   -- scheduled, manual or undo
    status TEXT NOT NULL,   -- moved, unchanged, skipped or undone
    summary TEXT NOT NULL,
    before_json TEXT,       -- the zones before and after, so a change can be undone
    after_json TEXT,
    image_path TEXT         -- the median frame with the old and new zones drawn, under DATA_DIR/calibration
);
CREATE TABLE IF NOT EXISTS embeddings (  -- BioCLIP 2 embedding of each frame's crop box (birdbrain.presence)
    visit_id INTEGER PRIMARY KEY REFERENCES visits(id) ON DELETE CASCADE,
    vec BLOB NOT NULL       -- float16
);
"""
# Approximate Claude prices (USD per million tokens) for the cost and savings figures on the home page.
HAIKU_USD_PER_MTOK = {"input": 0.10, "output": 0.50}
USD_PER_MTOK = {"haiku": HAIKU_USD_PER_MTOK, "sonnet": {"input": 2.00, "output": 10.00}}
# Owner-editable settings (Settings page). Stored in the database so they live with the data, not the code.
SETTINGS = {
    "location": "",      # e.g. "Vancouver, Washington, USA (Clark County, Pacific Northwest lowlands)"
    "scene_notes": "",   # what the camera sees: the bath, rocks, feeders
    "target_notes": "",  # what to look for: expected visitors, their sizes, where they land
    "ignore_notes": "",  # things that move or look like animals but aren't
}
ADDED_COLUMNS = {"corrected_visitor": "TEXT", "feedback": "TEXT", "marks": "TEXT", "lesson_off": "INTEGER NOT NULL DEFAULT 0",
                 "also": "TEXT", "corrected_also": "TEXT", "ended_at": "TEXT",
                 "confirmed_at": "TEXT",  # the owner pressed Confirm: Claude's answer is right (birdbrain.presence)
                 # A false trigger's photo files were deleted to save space (birdbrain.retention); the row and its
                 # embedding stay, so the Pi's animal check keeps learning from it.
                 "photo_deleted_at": "TEXT"}
HAS_PHOTO = "photo_deleted_at IS NULL"

# How many corrections and notes join each identify prompt.
LESSON_LIMIT = 15

VISITORS = ("bird", "squirrel", "raccoon", "possum", "other", "none")
MAMMALS = ("squirrel", "raccoon", "possum")
NO_SPECIES = ("none", "other")


def _cap(col: str) -> str:
    return f"upper(substr({col}, 1, 1)) || substr({col}, 2)"


def label_expr(t: str = "") -> str:
    """Display label: the owner's correction wins, then Claude's species, then the visitor type."""
    p = f"{t}." if t else ""
    return (
        f"COALESCE({p}corrected_species, "
        f"CASE WHEN {p}corrected_visitor IS NOT NULL THEN {_cap(p + 'corrected_visitor')} END, "
        f"{p}species, {_cap(p + 'visitor')})"
    )


LABEL = label_expr()
VISITOR = "COALESCE(corrected_visitor, visitor)"
# Older rows corrected before corrected_visitor existed only set corrected_species.
REAL = f"({VISITOR} != 'none' OR (corrected_visitor IS NULL AND corrected_species IS NOT NULL))"
FALSE_TRIGGER = f"NOT {REAL}"
# Nobody has confirmed or corrected this one yet (the False triggers page lists only these by default).
UNCHECKED = "(confirmed_at IS NULL AND corrected_visitor IS NULL AND corrected_species IS NULL)"
# The other kinds of animal in a visit's photo besides the main one, as JSON [{visitor, species, count, ...}]:
# the owner's list when they corrected it ([] means "nothing else"), else Claude's.
ALSO = "COALESCE(corrected_also, also)"
# Display label of one json_each(ALSO) entry, as LABEL is for the main animal.
ALSO_VISITOR = "json_extract(value, '$.visitor')"
ALSO_LABEL = f"COALESCE(json_extract(value, '$.species'), {_cap(ALSO_VISITOR)})"
CORRECTED = ("(corrected_visitor IS NOT NULL OR corrected_species IS NOT NULL OR feedback IS NOT NULL"
             " OR corrected_also IS NOT NULL)")


def connect(path: Path | None = None) -> sqlite3.Connection:
    path = path or config.DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def session(path: Path | None = None) -> Iterator[sqlite3.Connection]:
    conn = connect(path)
    try:
        yield conn
    finally:
        conn.close()


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    have = {r["name"] for r in conn.execute("PRAGMA table_info(visits)")}
    for col, kind in ADDED_COLUMNS.items():
        if col not in have:
            conn.execute(f"ALTER TABLE visits ADD COLUMN {col} {kind}")
    conn.commit()


def get_settings(conn: sqlite3.Connection) -> dict[str, str]:
    stored = {r["key"]: r["value"] for r in conn.execute("SELECT key, value FROM settings")}
    return {k: stored.get(k, default) for k, default in SETTINGS.items()}


def prompt_site(conn: sqlite3.Connection) -> dict:
    """Claude's Notes (location and the three Your yard boxes): everything identify.system_prompt says about this
    camera. The zones are set on Settings and aren't described to Claude; it sees the crop they give."""
    return get_settings(conn)


def save_settings(conn: sqlite3.Connection, **values: str) -> None:
    unknown = set(values) - set(SETTINGS)
    if unknown:
        raise ValueError(f"unknown settings: {sorted(unknown)}")
    conn.executemany(
        "INSERT INTO settings (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        [(k, (v or "").strip()) for k, v in values.items()],
    )
    conn.commit()


def get_value(conn: sqlite3.Connection, key: str) -> str | None:
    """One raw value from the settings table (used for settings that aren't part of the identify prompt)."""
    row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else None


def set_value(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO settings (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value),
    )
    conn.commit()


def add_calibration(conn: sqlite3.Connection, **fields) -> int:
    cols = ", ".join(fields)
    marks = ", ".join("?" for _ in fields)
    cur = conn.execute(f"INSERT INTO calibrations ({cols}) VALUES ({marks})", tuple(fields.values()))
    conn.commit()
    return cur.lastrowid


def recent_calibrations(conn: sqlite3.Connection, limit: int = 6) -> list[sqlite3.Row]:
    return conn.execute("SELECT * FROM calibrations ORDER BY id DESC LIMIT ?", (limit,)).fetchall()


def get_calibration(conn: sqlite3.Connection, cal_id: int) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM calibrations WHERE id = ?", (cal_id,)).fetchone()


def zones_at(conn: sqlite3.Connection, when: str) -> dict | None:
    """The zones in use at `when` (ISO time) as far as the calibration history tells: the last run's after on or
    before it, else the first later run's before. None without a usable row. Undo rows carry no zones, so a
    photo taken right after an undo may get the undone zones."""
    for sql in ("SELECT after_json FROM calibrations WHERE ran_at <= ? AND after_json IS NOT NULL "
                "ORDER BY ran_at DESC, id DESC LIMIT 1",
                "SELECT before_json FROM calibrations WHERE ran_at > ? AND before_json IS NOT NULL "
                "ORDER BY ran_at, id LIMIT 1"):
        row = conn.execute(sql, (when,)).fetchone()
        if row:
            try:
                data = json.loads(row[0])
            except ValueError:
                return None
            return data if isinstance(data, dict) else None
    return None


def end_stay(conn: sqlite3.Connection, visit_id: int, ended_at: str, reason: str) -> None:
    """The visitor left (or the stay was cut short: reason light or long, see birdbrain/stay.py)."""
    conn.execute("UPDATE visits SET ended_at = ?, raw_json = json_set(COALESCE(raw_json, '{}'), '$.stay_end', ?)"
                 " WHERE id = ?", (ended_at, reason, visit_id))
    conn.commit()


def insert_visit(conn: sqlite3.Connection, **fields) -> int:
    cols = ", ".join(fields)
    marks = ", ".join("?" for _ in fields)
    cur = conn.execute(f"INSERT INTO visits ({cols}) VALUES ({marks})", tuple(fields.values()))
    conn.commit()
    return cur.lastrowid


def _where(
    *,
    species: str | None = None,
    kind: str | None = None,
    tod: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    review: bool = False,
    false_triggers: bool = False,
    has_photo: bool = False,
    unchecked: bool = False,
) -> tuple[str, list]:
    where = [FALSE_TRIGGER if false_triggers else REAL]
    args: list = []
    if has_photo:
        where.append(HAS_PHOTO)
    if unchecked:
        where.append(UNCHECKED)
    if review:
        where.append("needs_review = 1")
    if species:
        where.append(f"({LABEL} = ? OR EXISTS (SELECT 1 FROM json_each({ALSO}) WHERE {ALSO_LABEL} = ?))")
        args.extend([species, species])
    if kind == "bird":
        where.append(f"{VISITOR} = 'bird'")
    elif kind == "mammal":
        where.append(f"{VISITOR} IN ({', '.join('?' for _ in MAMMALS)})")
        args.extend(MAMMALS)
    elif kind == "multiple":
        # Two or more animals: several of one species, or another kind in the photo too.
        where.append(f"(count > 1 OR json_array_length(COALESCE({ALSO}, '[]')) > 0)")
    if tod == "day":
        where.append("is_night = 0")
    elif tod == "night":
        where.append("is_night = 1")
    if date_from:
        where.append("date(captured_at) >= ?")
        args.append(date_from)
    if date_to:
        where.append("date(captured_at) <= ?")
        args.append(date_to)
    return " AND ".join(where), args


def _seen(where: str) -> str:
    """Every animal in the visits matching `where`: a row per visit for its main animal, plus a row per other
    kind in the same photo (extra = 1). Per-species counts use COUNT(DISTINCT id), so a visit counts once.
    `where`'s arguments are needed twice."""
    return f"""
        SELECT id, captured_at, is_night, thumb_path, confidence, corrected_species, corrected_visitor,
               {VISITOR} AS seen_visitor, {LABEL} AS label, 0 AS extra
        FROM visits WHERE {where}
        UNION ALL
        SELECT v.id, v.captured_at, v.is_night, v.thumb_path, json_extract(j.value, '$.confidence'), NULL, NULL,
               json_extract(j.value, '$.visitor'), {ALSO_LABEL.replace("value", "j.value")}, 1
        FROM (SELECT * FROM visits WHERE {where}) v, json_each(COALESCE(v.corrected_also, v.also)) j"""


def also_of(visit) -> list[dict]:
    """A visit row's other animals, as ALSO picks them."""
    for col in ("corrected_also", "also"):
        try:
            text = visit[col]
        except (IndexError, KeyError):
            continue
        if text is not None:
            try:
                rows = json.loads(text)
            except ValueError:
                return []
            return [r for r in rows if isinstance(r, dict) and r.get("visitor")] if isinstance(rows, list) else []
    return []


def also_label(entry: dict) -> str:
    return entry.get("species") or entry["visitor"].capitalize()


def plan_also(visit, names: list[str], counts: list[str], visitor: str | None = None) -> list[dict] | None:
    """Turn the form's "Also here" rows into what to store in corrected_also: None when they match Claude's
    list, else the owner's list. A name like "Squirrel" or "Other" sets that visitor type; a name Claude gave
    keeps its type and circle; any other name is a bird. "No animal" as the main answer means nothing else
    was there either."""
    claude = also_of({"corrected_also": None, "also": visit["also"]})
    by_name = {also_label(c).lower(): c for c in claude}
    if (visitor or visit["visitor"]) == "none":
        return [] if claude else None
    rows, seen = [], set()
    for name, n in zip(names, counts):
        name = re.sub(r"\s+", " ", name).strip()[:60]
        if not name or name.lower() in seen:
            continue
        seen.add(name.lower())
        try:
            count = min(max(int(n), 1), 99)
        except (TypeError, ValueError):
            count = 1
        if name.lower() in VISITORS and name.lower() != "none":
            row = {"visitor": name.lower(), "species": None, "count": count}
        elif (c := by_name.get(name.lower())) is not None:
            row = {"visitor": c["visitor"], "species": c.get("species"), "count": count}
            if c.get("circle"):
                row["circle"] = c["circle"]
        else:
            row = {"visitor": "bird", "species": name, "count": count}
        rows.append(row)
    same = [(also_label(r), r["count"]) for r in rows] == [(also_label(c), c.get("count") or 1) for c in claude]
    return None if same else rows


def list_visits(conn: sqlite3.Connection, *, limit: int = 24, offset: int = 0, **filters) -> list[sqlite3.Row]:
    where, args = _where(**filters)
    order = "ASC" if filters.get("review") else "DESC"
    sql = (
        f"SELECT *, {LABEL} AS label FROM visits WHERE {where} "
        f"ORDER BY captured_at {order}, id {order} LIMIT ? OFFSET ?"
    )
    return conn.execute(sql, (*args, limit, offset)).fetchall()


def count_visits(conn: sqlite3.Connection, **filters) -> int:
    where, args = _where(**filters)
    return conn.execute(f"SELECT COUNT(*) FROM visits WHERE {where}", args).fetchone()[0]


def get_visit(conn: sqlite3.Connection, visit_id: int) -> sqlite3.Row | None:
    return conn.execute(f"SELECT *, {LABEL} AS label FROM visits WHERE id = ?", (visit_id,)).fetchone()


def is_false_trigger(conn: sqlite3.Connection, visit_id: int) -> bool:
    return conn.execute(f"SELECT {FALSE_TRIGGER} FROM visits WHERE id = ?", (visit_id,)).fetchone()[0] == 1


def get_neighbors(conn: sqlite3.Connection, visit_id: int, false_triggers: bool = False,
                  unchecked: bool = False, species: str | None = None) -> tuple[int | None, int | None]:
    """Return (prev_id, next_id) for a visit within the same set (real or false triggers), ordered newest-first.

    unchecked: only false triggers nobody has confirmed or corrected yet (the False triggers list as shown).
    species: only that species' visits, the set its page lists (other animals in the photo count too). The
    visit itself needn't be in it (relabelled): its position by time is used.
    """
    where, args = (f"{FALSE_TRIGGER} AND {HAS_PHOTO}", []) if false_triggers else (REAL, [])
    if species and not false_triggers:
        where, args = _where(species=species)
    if false_triggers and unchecked:
        where += f" AND {UNCHECKED}"
    row = conn.execute("SELECT captured_at FROM visits WHERE id = ?", (visit_id,)).fetchone()
    if not row:
        return None, None
    ts = row["captured_at"]
    prev_row = conn.execute(
        f"SELECT id FROM visits WHERE {where} AND (captured_at > ? OR (captured_at = ? AND id > ?)) "
        f"ORDER BY captured_at ASC, id ASC LIMIT 1",
        (*args, ts, ts, visit_id),
    ).fetchone()
    next_row = conn.execute(
        f"SELECT id FROM visits WHERE {where} AND (captured_at < ? OR (captured_at = ? AND id < ?)) "
        f"ORDER BY captured_at DESC, id DESC LIMIT 1",
        (*args, ts, ts, visit_id),
    ).fetchone()
    return (prev_row["id"] if prev_row else None), (next_row["id"] if next_row else None)


def latest_visit(conn: sqlite3.Connection, real_only: bool = True) -> sqlite3.Row | None:
    where = REAL if real_only else "1"
    return conn.execute(
        f"SELECT *, {LABEL} AS label FROM visits WHERE {where} ORDER BY captured_at DESC, id DESC LIMIT 1"
    ).fetchone()


def newest_id(conn: sqlite3.Connection) -> int:
    """Highest visit id, false triggers included: pages compare it to notice anything new."""
    return conn.execute("SELECT COALESCE(MAX(id), 0) FROM visits").fetchone()[0]


def today_summary(conn: sqlite3.Connection) -> dict:
    today = clock.today().isoformat()
    today_where = f"{REAL} AND date(captured_at) = ?"
    real_today = conn.execute(
        f"SELECT COUNT(DISTINCT id) AS n, COUNT(DISTINCT label) AS species FROM ({_seen(today_where)})",
        (today, today),
    ).fetchone()
    return {
        "visits": real_today["n"],
        "species": real_today["species"],
        "false_triggers": count_visits(conn, false_triggers=True, date_from=today, date_to=today),
        "review": count_visits(conn, review=True),
        "latest": latest_visit(conn),
        "date": today,
        "critters": scoreboard(conn, today, today),
        "last_motion": latest_visit(conn, real_only=False),
    }


def scoreboard(conn: sqlite3.Connection, date_from: str | None = None, date_to: str | None = None) -> list:
    """Each animal seen between two dates (inclusive, None is open): label, sightings n, last_at; most seen first."""
    where, args = REAL, []
    if date_from:
        where += " AND date(captured_at) >= ?"
        args.append(date_from)
    if date_to:
        where += " AND date(captured_at) <= ?"
        args.append(date_to)
    return conn.execute(
        f"SELECT label, COUNT(DISTINCT id) AS n, MAX(captured_at) AS last_at FROM ({_seen(where)})"
        f" GROUP BY label ORDER BY n DESC, last_at DESC",
        args * 2,
    ).fetchall()


def scoreboard_total(conn: sqlite3.Connection, date_from: str | None = None, date_to: str | None = None) -> int:
    """Real visits (photos with an animal) between two dates, same bounds as `scoreboard`; each photo counts once."""
    where, args = REAL, []
    if date_from:
        where += " AND date(captured_at) >= ?"
        args.append(date_from)
    if date_to:
        where += " AND date(captured_at) <= ?"
        args.append(date_to)
    return conn.execute(f"SELECT COUNT(*) FROM visits WHERE {where}", args).fetchone()[0]


def plan_correction(visit: sqlite3.Row, visitor: str, species: str) -> tuple[str | None, str | None]:
    """Turn the review form into what to store: only the parts that differ from Claude's answer.

    Raises ValueError for an unknown visitor type.
    """
    visitor, species = visitor.strip(), species.strip()
    if visitor and visitor not in VISITORS:
        raise ValueError(f"unknown visitor type: {visitor!r}")
    new_visitor = visitor if visitor and visitor != visit["visitor"] else None
    effective = new_visitor or visit["visitor"]
    keep_species = species and effective not in NO_SPECIES and species != (visit["species"] or "")
    return new_visitor, (species if keep_species else None)


def correct_visit(
    conn: sqlite3.Connection,
    visit_id: int,
    visitor: str | None,
    species: str | None,
    feedback: str | None,
    marks: list[dict] | None = None,
    also: list[dict] | None = None,
) -> None:
    """Store the owner's answer. `also` is their list of other animals (None: Claude's list stands)."""
    conn.execute(
        "UPDATE visits SET corrected_visitor = ?, corrected_species = ?, feedback = ?, marks = ?, corrected_also = ?, "
        "needs_review = 0, lesson_off = 0 WHERE id = ?",
        (visitor or None, species or None, (feedback or "").strip() or None, json.dumps(marks) if marks else None,
         json.dumps(also) if also is not None else None, visit_id),
    )
    conn.commit()


def count_unchecked_false_triggers(conn: sqlite3.Connection) -> int:
    """False triggers the owner hasn't confirmed or corrected yet (only ones whose photo is still there to look at)."""
    return count_visits(conn, false_triggers=True, has_photo=True, unchecked=True)


def false_trigger_cards(conn: sqlite3.Connection, ids: list[int]) -> list[sqlite3.Row]:
    """These visits, newest first, if they're still false triggers with a photo."""
    if not ids:
        return []
    return conn.execute(
        f"""SELECT *, {LABEL} AS label FROM visits WHERE id IN ({','.join('?' * len(ids))})
            AND {FALSE_TRIGGER} AND {HAS_PHOTO} ORDER BY captured_at DESC, id DESC""", ids).fetchall()


def false_triggers_after(conn: sqlite3.Connection, after: int, unchecked: bool = True) -> list[sqlite3.Row]:
    """False triggers with a photo newer than this id (the newest the page already shows), newest first.
    unchecked: leave out the confirmed or corrected ones, as the page does without Show confirmed."""
    where = f"id > ? AND {FALSE_TRIGGER} AND {HAS_PHOTO}" + (f" AND {UNCHECKED}" if unchecked else "")
    return conn.execute(
        f"SELECT *, {LABEL} AS label FROM visits WHERE {where} ORDER BY captured_at DESC, id DESC LIMIT 200",
        (after,)).fetchall()


def confirm_visits(conn: sqlite3.Connection, ids: list[int], on: bool = True) -> None:
    """The owner says the stored answer is right (or takes that back): those frames count extra when the
    local animal check retrains."""
    conn.executemany("UPDATE visits SET confirmed_at = ? WHERE id = ?",
                     [(datetime.now().isoformat(timespec="seconds") if on else None, i) for i in ids])
    conn.commit()


def parse_marks(text: str | None) -> list[dict]:
    """The owner's numbered circles on a photo: [{x, y, r}] as fractions of the width, height and width. At most 9."""
    try:
        raw = json.loads(text or "[]")
    except ValueError:
        return []
    marks = []
    for m in raw if isinstance(raw, list) else []:
        try:
            x, y, r = (min(max(float(m[k]), 0.0), 1.0) for k in ("x", "y", "r"))
        except (TypeError, KeyError, ValueError):
            continue
        mark = {"x": round(x, 4), "y": round(y, 4), "r": round(max(r, 0.01), 4)}
        # A circle Claude drew around what it identified, or one the Pi drew where it saw the motion of a visit it
        # named itself (birdbrain/motion_circle.py); either kept by the owner.
        for by in ("claude", "pi"):
            if m.get(by) is True:
                mark[by] = True
        if (mark.get("claude") or mark.get("pi")) and isinstance(m.get("label"), str) and m["label"].strip():
            mark["label"] = m["label"].strip()[:60]  # which animal, when there are several
        marks.append(mark)
    return marks[:9]


def describe_marks(marks: list[dict]) -> str:
    """Where each circle sits, so Claude can match the owner's note ("1 is a rock") to a place in the frame."""
    return "; ".join(
        f"circle {i}{_drew(m)} at {m['x']:.0%} across, {m['y']:.0%} down"
        for i, m in enumerate(marks, 1)
    )


def _drew(mark: dict) -> str:
    if mark.get("pi"):  # Claude didn't draw it: the Pi did, from the motion, on a visit it named itself
        return " (drawn on the Pi where it saw motion)"
    if not mark.get("claude"):
        return ""
    return f" (you drew it around the {mark['label']})" if mark.get("label") else " (the one you drew)"


def _spoken_name(species: str | None, visitor: str) -> str:
    return species or ("no animal" if visitor == "none" else visitor)


def _spoken_also(rows: list[dict]) -> str:
    return ", ".join(f"{r.get('count') or 1} {_spoken_name(r.get('species'), r['visitor'])}" for r in rows)


def all_lessons(conn: sqlite3.Connection) -> list[dict]:
    """Every correction and note, newest first, as the line Claude would read, with whether it's in the prompt now."""
    rows = conn.execute(
        "SELECT id, captured_at, visitor, species, corrected_visitor, corrected_species, feedback, marks, is_night, "
        f"lesson_off, also, corrected_also FROM visits WHERE {CORRECTED} ORDER BY id DESC"
    ).fetchall()
    lessons, used = [], 0
    for r in rows:
        said = _spoken_name(r["species"], r["visitor"])
        actual = r["corrected_species"] or (_spoken_name(None, r["corrected_visitor"]) if r["corrected_visitor"] else None)
        when = "night (IR)" if r["is_night"] else "day"
        if claude_also := also_of({"corrected_also": None, "also": r["also"]}):
            said += f"\" with \"{_spoken_also(claude_also)}"
        prefix = f"- {r['captured_at'][:10]}, {when}: you said \"{said}\""
        line = f"{prefix}; the owner says it was \"{actual}\"." if actual else f"{prefix} and the owner confirmed it."
        if r["corrected_also"] is not None:
            owner_also = also_of(r)
            line += (f" The owner says also there: \"{_spoken_also(owner_also)}\"." if owner_also
                     else " The owner says no other animal was there.")
        if r["feedback"]:
            line += " Owner's note: " + re.sub(r"\s+", " ", r["feedback"]).strip()
        if marks := parse_marks(r["marks"]):
            line += f" The owner circled spots on the full frame: {describe_marks(marks)}."
        in_use = not r["lesson_off"] and used < LESSON_LIMIT
        used += in_use
        lessons.append({"visit_id": r["id"], "captured_at": r["captured_at"], "line": line,
                        "off": bool(r["lesson_off"]), "in_use": in_use,
                        # what the edit form shows: Claude's answer stays fixed, the owner's parts can change
                        "said": said, "when": when, "visitor": r["corrected_visitor"] or r["visitor"],
                        "species": r["corrected_species"] or r["species"] or "", "feedback": r["feedback"] or "",
                        "corrected_also": r["corrected_also"],
                        "circles": len(marks)})
    return lessons


def recent_lessons(conn: sqlite3.Connection, limit: int = LESSON_LIMIT) -> list[str]:
    """The owner's latest corrections and notes not switched off, phrased for the identification prompt."""
    return [l["line"] for l in all_lessons(conn) if not l["off"]][:limit]


def set_lesson_off(conn: sqlite3.Connection, visit_id: int, off: bool) -> None:
    """Leave a correction out of the prompt (or put it back) without touching the visit's label."""
    conn.execute("UPDATE visits SET lesson_off = ? WHERE id = ?", (int(off), visit_id))
    conn.commit()


def mark_reviewed(conn: sqlite3.Connection, visit_id: int) -> None:
    """Confirm (the visit page button). Off the review queue, and confirmed (the local animal check trains on it, counting extra)."""
    conn.execute("UPDATE visits SET needs_review = 0, confirmed_at = COALESCE(confirmed_at, ?) WHERE id = ?",
                 (datetime.now().isoformat(timespec="seconds"), visit_id))
    conn.commit()


def delete_visit(conn: sqlite3.Connection, visit_id: int) -> sqlite3.Row | None:
    row = get_visit(conn, visit_id)
    if row:
        conn.execute("DELETE FROM visits WHERE id = ?", (visit_id,))
        photos = species_photos(conn)
        if visit_id in photos.values():  # a later visit could get the same id
            _save_species_photos(conn, {k: v for k, v in photos.items() if v != visit_id})
        conn.commit()
    return row


SPECIES_PHOTO = "species_photo"  # settings key: JSON {label: visit id}, the owner's pick of each species' photo


def species_photos(conn: sqlite3.Connection) -> dict[str, int]:
    """The visits the owner picked as species photos, by label."""
    try:
        return {str(k): int(v) for k, v in json.loads(get_value(conn, SPECIES_PHOTO) or "{}").items()}
    except (ValueError, TypeError, AttributeError):
        return {}


def _save_species_photos(conn: sqlite3.Connection, photos: dict[str, int]) -> None:
    set_value(conn, SPECIES_PHOTO, json.dumps(photos, sort_keys=True))


def shows_species(conn: sqlite3.Connection, visit_id: int, label: str) -> bool:
    """Whether a real visit has this species in it, as the main animal or one of the others."""
    return conn.execute(f"SELECT 1 FROM ({_seen(REAL + ' AND id = ?')}) WHERE label = ? LIMIT 1",
                        (visit_id, visit_id, label)).fetchone() is not None


def set_species_photo(conn: sqlite3.Connection, label: str, visit_id: int | None) -> None:
    """Use a visit's photo for its species (None: back to the automatic pick). If the visit is later deleted or
    relabelled, species_summary falls back to the automatic pick on its own."""
    photos = species_photos(conn)
    if visit_id is None:
        photos.pop(label, None)
    else:
        photos[label] = int(visit_id)
    _save_species_photos(conn, photos)


def species_without_favorite(conn: sqlite3.Connection, min_photos: int = 2) -> list[str]:
    """Species labels with `min_photos`+ visits that still have a photo and no owner pick that is still valid
    (the picked visit must still show the species and still have its photo), most visited first."""
    where, args = _where()
    rows = conn.execute(f"SELECT label, id FROM ({_seen(where + ' AND ' + HAS_PHOTO)})", [*args, *args]).fetchall()
    ids: dict[str, set[int]] = {}
    for r in rows:
        ids.setdefault(r["label"], set()).add(r["id"])
    picks = species_photos(conn)
    out = [(label, len(v)) for label, v in ids.items()
           if len(v) >= min_photos and label.lower() not in ("none", "other") and picks.get(label) not in v]
    return [label for label, _ in sorted(out, key=lambda t: (-t[1], t[0]))]


SPECIES_SORTS = {
    "visits": "visits DESC, label",
    "newest": "first_seen DESC",
    "recent": "last_seen DESC",
    "name": "label",
}


def species_summary(
    conn: sqlite3.Connection, sort: str = "visits", limit: int | None = None, name: str | None = None
) -> list[sqlite3.Row]:
    """One row per species with visit counts, first/last seen and a representative photo.

    The photo is the one the owner picked (species_photos) while that visit still shows the species; otherwise
    it prefers visits where it's the main animal, then owner-corrected ones, then high-confidence ones, then the
    newest. A window function picks it in one pass instead of a per-species subquery. Visits where it was one of
    the other animals in the photo count too. photo_id is the visit the photo comes from, picked whether the
    owner chose it.
    """
    order = SPECIES_SORTS.get(sort, SPECIES_SORTS["visits"])
    where, args = _where()
    args = [*args, *args, json.dumps(species_photos(conn))]
    if name:
        args.append(name)
    sql = f"""
        WITH v AS ({_seen(where)}
        ), ranked AS (
            SELECT label, seen_visitor, id, captured_at, thumb_path, picked,
                   ROW_NUMBER() OVER (
                       PARTITION BY label
                       ORDER BY picked DESC, extra,
                                (corrected_species IS NOT NULL OR corrected_visitor IS NOT NULL) DESC,
                                (confidence = 'high') DESC, captured_at DESC
                   ) AS rank
            FROM (SELECT v.*, COALESCE(fav.value = v.id, 0) AS picked
                  FROM v LEFT JOIN json_each(?) fav ON fav.key = v.label)
            {"WHERE label = ?" if name else ""}
        )
        SELECT label, MAX(seen_visitor) AS visitor, COUNT(DISTINCT id) AS visits,
               MIN(captured_at) AS first_seen, MAX(captured_at) AS last_seen,
               MAX(CASE WHEN rank = 1 THEN thumb_path END) AS thumb_path,
               MAX(CASE WHEN rank = 1 THEN id END) AS photo_id,
               MAX(CASE WHEN rank = 1 THEN picked END) AS photo_picked
        FROM ranked GROUP BY label ORDER BY {order}
    """
    if limit:
        sql += f" LIMIT {int(limit)}"
    return conn.execute(sql, args).fetchall()


def species_row(conn: sqlite3.Connection, name: str) -> sqlite3.Row | None:
    rows = species_summary(conn, name=name)
    return rows[0] if rows else None


def species_names(conn: sqlite3.Connection) -> list[str]:
    return [r[0] for r in conn.execute(f"SELECT DISTINCT label FROM ({_seen(REAL)}) ORDER BY label")]


def species_kinds(conn: sqlite3.Connection) -> list[tuple[str, str]]:
    """Every name seen so far with its visitor type (the most common one, if it was ever labelled two ways)."""
    rows = conn.execute(
        f"""SELECT label, seen_visitor, COUNT(DISTINCT id) AS n FROM ({_seen(REAL)})
            GROUP BY label, seen_visitor ORDER BY label, n DESC"""
    ).fetchall()
    kinds: dict[str, str] = {}
    for r in rows:
        kinds.setdefault(r["label"], r["seen_visitor"])
    return list(kinds.items())


def bump(conn: sqlite3.Connection, name: str, day: str | None = None) -> None:
    """Count events that leave no visit row behind (e.g. discarded rain probes)."""
    conn.execute(
        "INSERT INTO counters (day, name, n) VALUES (?, ?, 1) ON CONFLICT(day, name) DO UPDATE SET n = n + 1",
        (day or clock.today().isoformat(), name),
    )
    conn.commit()


def avg_haiku_call_usd(conn: sqlite3.Connection, sample: int = 200) -> float:
    """Average cost of one Haiku call, from the token counts saved with recent visits."""
    row = conn.execute(
        """SELECT AVG(json_extract(raw_json, '$.calls.haiku.input_tokens')) AS i,
                  AVG(json_extract(raw_json, '$.calls.haiku.output_tokens')) AS o
           FROM (SELECT raw_json FROM visits
                 WHERE json_extract(raw_json, '$.calls.haiku.input_tokens') IS NOT NULL
                 ORDER BY id DESC LIMIT ?)""",
        (sample,),
    ).fetchone()
    if row["i"] is None:
        return 0.0
    return (row["i"] * HAIKU_USD_PER_MTOK["input"] + row["o"] * HAIKU_USD_PER_MTOK["output"]) / 1_000_000


def activity(conn: sqlite3.Connection, day: str | None = None) -> dict:
    """One day's motion events, broken down so the rows add up:

    visitors + false_alarms = all motion events
    false_alarms = kept_on_pi (discarded probes + visits the Pi settled) + false alarms Claude checked
    sent_to_claude = every visit Claude was asked about (visitors and false alarms), with what it cost
    would_keep = Claude calls the animal check would have skipped in gate mode (it judged them empty)
    """
    day = day or clock.today().isoformat()
    r = conn.execute(
        f"""SELECT
              SUM({REAL}) AS visitors,
              (SELECT COUNT(DISTINCT label) FROM ({_seen(f"{REAL} AND date(captured_at) = :day")})) AS species,
              SUM({FALSE_TRIGGER}) AS stored_false,
              SUM(model IN ('yolox-tiny', 'bioclip-presence', 'bioclip-2')) AS settled_on_pi,
              SUM(model LIKE 'claude%') AS sent,
              SUM(model LIKE 'claude%'
                  AND json_extract(raw_json, '$.calls.presence.empty') = 1) AS would_keep
            FROM visits WHERE date(captured_at) = :day""",
        {"day": day},
    ).fetchone()
    probes = conn.execute(
        "SELECT COALESCE(SUM(n), 0) FROM counters WHERE name = 'probe_empty' AND day = ?", (day,)
    ).fetchone()[0]
    spend = 0.0
    for raw in conn.execute("SELECT raw_json FROM visits WHERE date(captured_at) = ? AND model LIKE 'claude%'", (day,)):
        calls = (json.loads(raw[0] or "{}") or {}).get("calls") or {}
        for model, price in USD_PER_MTOK.items():
            c = calls.get(model) or {}
            spend += (c.get("input_tokens", 0) * price["input"] + c.get("output_tokens", 0) * price["output"]) / 1e6
    n = lambda k: r[k] or 0
    return {
        "visitors": n("visitors"),
        "species": n("species"),
        "false_alarms": n("stored_false") + probes,
        "kept_on_pi": probes + n("settled_on_pi"),
        "sent_to_claude": n("sent"),
        "claude_usd": spend,
        "would_keep": n("would_keep"),
        "would_keep_usd": n("would_keep") * avg_haiku_call_usd(conn),
    }


def _spend_sql(model: str) -> str:
    price = USD_PER_MTOK[model]
    return (f"COALESCE(SUM(json_extract(raw_json, '$.calls.{model}.input_tokens')), 0) * {price['input']} + "
            f"COALESCE(SUM(json_extract(raw_json, '$.calls.{model}.output_tokens')), 0) * {price['output']}")


def totals(conn: sqlite3.Connection) -> dict:
    """All-time counts for the About page: how far motion events got down the pipeline, and what Claude cost."""
    r = conn.execute(
        f"""SELECT
              MIN(captured_at) AS since,
              SUM({REAL}) AS visitors,
              (SELECT COUNT(DISTINCT label) FROM ({_seen(REAL)}) WHERE seen_visitor != 'other') AS species,
              SUM({FALSE_TRIGGER}) AS stored_false,
              SUM(model IN ('yolox-tiny', 'bioclip-presence', 'bioclip-2')) AS settled_on_pi,
              SUM(model LIKE 'claude%') AS sent,
              SUM(escalated) AS escalated,
              SUM(needs_review) AS waiting_review,
              SUM({CORRECTED}) AS corrected,
              ({_spend_sql('haiku')} + {_spend_sql('sonnet')}) / 1e6 AS claude_usd
            FROM visits"""
    ).fetchone()
    probes = conn.execute("SELECT COALESCE(SUM(n), 0) FROM counters WHERE name = 'probe_empty'").fetchone()[0]
    checks = owner_checks_spend(conn)
    n = lambda k: r[k] or 0
    return {
        "since": r["since"],
        "events": n("visitors") + n("stored_false") + probes,
        "visitors": n("visitors"),
        "species": n("species"),
        "false_alarms": n("stored_false") + probes,
        "kept_on_pi": probes + n("settled_on_pi"),
        "sent_to_claude": n("sent"),
        "haiku_only": n("sent") - n("escalated"),
        "escalated": n("escalated"),
        "waiting_review": n("waiting_review"),
        "corrected": n("corrected"),
        "claude_usd": (r["claude_usd"] or 0.0) + checks["usd"],
        "owner_checks": checks["n"],
        "owner_checks_usd": checks["usd"],
    }


def owner_checks_spend(conn: sqlite3.Connection) -> dict:
    """The Claude calls the owner asked for on a circle (birdbrain.circle_check): how many and what they cost."""
    r = conn.execute(
        """SELECT COUNT(*) AS n, COALESCE(SUM(json_extract(c.value, '$.usd')), 0) AS usd
           FROM visits v,
                json_each(CASE WHEN json_valid(v.raw_json) THEN v.raw_json END, '$.calls.owner_checks') c
           WHERE json_extract(c.value, '$.source') IN ('haiku', 'sonnet')"""
    ).fetchone()
    return {"n": r["n"] or 0, "usd": r["usd"] or 0.0}


def capture_rate(conn: sqlite3.Connection, days: int = 7) -> dict:
    """How often photos are being captured, for the About page: recent and all-time daily rates, the busiest
    hour and the typical wait between real visitors. Days with no rows still count, so a quiet week reads low."""
    today = clock.today()
    since = (today - timedelta(days=days - 1)).isoformat()
    r = conn.execute(
        f"""SELECT COUNT(*) AS photos, COALESCE(SUM({REAL}), 0) AS visitors,
                   SUM(date(captured_at) >= ?) AS recent_photos,
                   COALESCE(SUM({REAL} AND date(captured_at) >= ?), 0) AS recent_visitors,
                   MIN(date(captured_at)) AS first_day, MAX(captured_at) AS last
            FROM visits""",
        (since, since),
    ).fetchone()
    if not r["photos"]:
        return {"photos": 0}
    span = max(1, (today - date.fromisoformat(r["first_day"])).days + 1)
    recent_days = min(days, span)
    hour = conn.execute(
        f"SELECT CAST(strftime('%H', captured_at) AS INTEGER) AS h, COUNT(*) AS n FROM visits"
        f" WHERE {REAL} AND date(captured_at) >= ? GROUP BY h ORDER BY n DESC, h LIMIT 1",
        (since,),
    ).fetchone()
    times = [datetime.fromisoformat(t) for (t,) in conn.execute(
        f"SELECT captured_at FROM visits WHERE {REAL} AND date(captured_at) >= ? ORDER BY captured_at", (since,))]
    gaps = sorted((b - a).total_seconds() / 60 for a, b in zip(times, times[1:]))
    return {
        "photos": r["photos"],
        "days": days,
        "recent_days": recent_days,
        "photos_per_day": r["recent_photos"] / recent_days,
        "visitors_per_day": r["recent_visitors"] / recent_days,
        "all_photos_per_day": r["photos"] / span,
        "all_visitors_per_day": r["visitors"] / span,
        "busiest_hour": hour["h"] if hour else None,
        "median_gap_min": gaps[len(gaps) // 2] if gaps else None,
        "last": r["last"],
    }


def detector_report(conn: sqlite3.Connection, days: int = 14) -> dict:
    """How gate mode would have done, judged against the final answer (owner corrections win over Claude).

    Only daytime visits where Claude was asked count; gate mode always asks Claude at night.
    """
    since = (clock.today() - timedelta(days=days - 1)).isoformat()
    rows = conn.execute(
        f"""SELECT id, {LABEL} AS label, {VISITOR} AS final_visitor,
                   json_extract(raw_json, '$.calls.detector.animal') AS det_animal
            FROM visits
            WHERE date(captured_at) >= ? AND is_night = 0 AND model LIKE 'claude%'
              AND json_extract(raw_json, '$.calls.detector') IS NOT NULL""",
        (since,),
    ).fetchall()
    saved = sum(1 for r in rows if not r["det_animal"] and r["final_visitor"] == "none")
    missed = [r for r in rows if not r["det_animal"] and r["final_visitor"] != "none"]
    return {
        "days": days,
        "checked": len(rows),
        "would_skip": sum(1 for r in rows if not r["det_animal"]),
        "would_save": saved,
        "would_miss": [{"id": r["id"], "label": r["label"]} for r in missed],
        "caught": sum(1 for r in rows if r["det_animal"] and r["final_visitor"] != "none"),
        "animals": sum(1 for r in rows if r["final_visitor"] != "none"),
    }


def empty_objects(conn: sqlite3.Connection, ground: bool, night: bool, limit: int,
                  exclude: int | None = None) -> list[list]:
    """The detector's object boxes from the newest empty frames (final answer: no animal) of one zone
    (ground or the bath) and time of day: the scenery check's memory of what the empty scene holds."""
    rows = conn.execute(
        f"""SELECT json_extract(raw_json, '$.calls.detector.objects') AS objs FROM visits
            WHERE {FALSE_TRIGGER} AND is_night = ? AND id != ?
              AND (COALESCE(json_extract(raw_json, '$.trigger'), '') = 'ground') = ?
              AND json_extract(raw_json, '$.calls.detector.objects') IS NOT NULL
            ORDER BY id DESC LIMIT ?""",
        (int(night), exclude or -1, int(ground), limit),
    ).fetchall()
    return [json.loads(r["objs"]) for r in rows]


def set_raw_json(conn: sqlite3.Connection, visit_id: int, raw: dict) -> None:
    conn.execute("UPDATE visits SET raw_json = ? WHERE id = ?", (json.dumps(raw), visit_id))
    conn.commit()


CONFIRMED_MIN = 3  # visits before a species counts as confirmed here


CLASSIFIER_MODEL = "bioclip-2"  # visits named by the local classifier in gate mode (classifier.MODEL_NAME)


def confirmed_species(conn: sqlite3.Connection, minimum: int = CONFIRMED_MIN) -> set[str]:
    """Species with at least `minimum` visits that are settled: answered with high confidence and not
    corrected to something else, or labelled by the owner. Haiku naming one of these at medium confidence
    doesn't need Sonnet's second look."""
    rows = conn.execute(
        f"""SELECT {LABEL} AS label, COUNT(*) AS n FROM visits
            WHERE {REAL} AND {LABEL} IS NOT NULL
              AND (corrected_species IS NOT NULL OR (confidence = 'high' AND corrected_visitor IS NULL))
            GROUP BY label HAVING n >= ?""",
        (minimum,),
    ).fetchall()
    return {r["label"] for r in rows}


def settled_species(conn: sqlite3.Connection, minimum: int = CONFIRMED_MIN) -> dict[str, str]:
    """confirmed_species with each one's visitor type (label -> bird/squirrel/...), for the classifier gate.
    Visits the gate named itself only count once the owner has checked them, so it can't confirm itself."""
    rows = conn.execute(
        f"""SELECT {LABEL} AS label, {VISITOR} AS visitor, COUNT(*) AS n FROM visits
            WHERE {REAL} AND {LABEL} IS NOT NULL
              AND (model != '{CLASSIFIER_MODEL}' OR confirmed_at IS NOT NULL OR corrected_species IS NOT NULL)
              AND (corrected_species IS NOT NULL OR (confidence = 'high' AND corrected_visitor IS NULL))
            GROUP BY label, visitor HAVING n >= ?""",
        (minimum,),
    ).fetchall()
    return {r["label"]: r["visitor"] for r in rows}


def scenery_report(conn: sqlite3.Connection, days: int = 14) -> dict:
    """How the scenery check (birdbrain.scenery) would have done if it decided which frames Claude sees,
    day and night, judged against the final answer (owner corrections win)."""
    since = (clock.today() - timedelta(days=days - 1)).isoformat()
    rows = conn.execute(
        f"""SELECT id, is_night, {LABEL} AS label, {REAL} AS real,
                   json_extract(raw_json, '$.calls.detector.scenery.empty') AS empty
            FROM visits
            WHERE date(captured_at) >= ? AND model LIKE 'claude%'
              AND json_extract(raw_json, '$.calls.detector.scenery') IS NOT NULL""",
        (since,),
    ).fetchall()
    return {
        "days": days,
        "checked": len(rows),
        "would_save": sum(1 for r in rows if r["empty"] and not r["real"]),
        "empties": sum(1 for r in rows if not r["real"]),
        "animals": sum(1 for r in rows if r["real"]),
        "night_animals": sum(1 for r in rows if r["real"] and r["is_night"]),
        "would_miss": [{"id": r["id"], "label": r["label"]} for r in rows if r["empty"] and r["real"]],
    }


def _name_key(name: str | None) -> str:
    return re.sub(r"[^a-z]", "", (name or "").lower())


def classifier_report(conn: sqlite3.Connection, days: int | None = None) -> dict:
    """How the local species classifier (birdbrain.classifier, shadow mode) agrees with the final label
    (owner corrections win, else Claude's) on visits named to species (birds and mammals): overall, top-1 and
    top-3, by day and night, and per species. A generic mammal label matches a species ending in it ("Raccoon":
    "Common Raccoon", "Possum": "Virginia Opossum"). All time unless `days` is given."""
    where, args = "", ()
    if days:
        where, args = "AND date(captured_at) >= ?", ((clock.today() - timedelta(days=days - 1)).isoformat(),)
    rows = conn.execute(
        f"""SELECT id, is_night, {LABEL} AS label,
                   json_extract(raw_json, '$.calls.classifier') AS guess
            FROM visits
            WHERE {VISITOR} != 'none' AND json_extract(raw_json, '$.calls.classifier.top') IS NOT NULL {where}
              -- a visit it named itself only scores it once the owner has checked or corrected it
              AND (model != '{CLASSIFIER_MODEL}' OR confirmed_at IS NOT NULL
                   OR corrected_species IS NOT NULL OR corrected_visitor IS NOT NULL)
            ORDER BY id DESC""",
        args,
    ).fetchall()
    per: dict[str, dict] = {}
    checked = top1 = top3 = night = night_top1 = 0
    seconds, misses = [], []
    for r in rows:
        k = _name_key(r["label"])
        if k in ("", "bird", "squirrel", "other"):  # no species to compare with
            continue
        guess = json.loads(r["guess"])
        names = [_name_key(g.get("name")) for g in guess.get("top") or []]
        same = (lambda n: n == k) if k not in ("raccoon", "possum") else (lambda n: n.endswith(k))
        hit1, hit3 = bool(names) and same(names[0]), any(same(n) for n in names)
        checked += 1
        top1 += hit1
        top3 += hit3
        if r["is_night"]:
            night += 1
            night_top1 += hit1
        if guess.get("seconds") is not None:
            seconds.append(guess["seconds"])
        s = per.setdefault(r["label"], {"label": r["label"], "visits": 0, "top1": 0, "top3": 0})
        s["visits"] += 1
        s["top1"] += hit1
        s["top3"] += hit3
        if not hit1 and len(misses) < 8:
            top = (guess.get("top") or [{}])[0]
            misses.append({"id": r["id"], "label": r["label"], "guess": top.get("name") or "nothing"})
    seconds.sort()
    return {
        "days": days,
        "checked": checked,
        "top1": top1,
        "top3": top3,
        "night": night,
        "night_top1": night_top1,
        "seconds": seconds[len(seconds) // 2] if seconds else None,
        "species": sorted(per.values(), key=lambda s: (-s["visits"], s["label"])),
        "misses": misses,
    }


def presence_report(conn: sqlite3.Connection, days: int = 14) -> dict:
    """The local animal check (birdbrain.presence) over the last `days`: frames it judged live (not backfilled),
    against the final answer. skipped = Claude wasn't asked (gate mode); missed = it said empty but the owner
    (or, in shadow mode, Claude) found an animal; would_skip = empty frames it would have kept from Claude."""
    since = (clock.today() - timedelta(days=days - 1)).isoformat()
    rows = conn.execute(
        f"""SELECT id, {LABEL} AS label, {REAL} AS real, is_night, confirmed_at IS NOT NULL AS confirmed,
                   (corrected_visitor IS NOT NULL OR corrected_species IS NOT NULL) AS corrected,
                   json_extract(raw_json, '$.calls.presence.empty') AS empty,
                   json_extract(raw_json, '$.calls.skipped_claude') AS skipped, model
            FROM visits WHERE json_extract(raw_json, '$.calls.presence.score') IS NOT NULL AND date(captured_at) >= ?
            ORDER BY id DESC""",
        (since,),
    ).fetchall()
    out = {"days": days, "frames": len(rows), "visitors": 0, "empty": 0, "skipped": 0, "would_skip": 0,
           "unchecked": 0, "missed": []}
    for r in rows:
        if r["real"]:
            out["visitors"] += 1
            if r["empty"]:
                out["missed"].append({"id": r["id"], "label": r["label"], "night": bool(r["is_night"])})
        else:
            out["empty"] += 1
            out["would_skip"] += bool(r["empty"])
        if r["model"] == "bioclip-presence":
            out["skipped"] += 1
            out["unchecked"] += not (r["confirmed"] or r["corrected"])
    return out


DETECTOR_MODES = ("off", "shadow", "gate")
GATE_MIN_ANIMALS = 20  # real daytime visitors needed before judging whether gate mode is safe


def detector_mode(conn: sqlite3.Connection) -> str:
    """The detector mode picked on the Settings page, else DETECTOR from .env."""
    value = get_value(conn, "detector_mode")
    return value if value in DETECTOR_MODES else config.DETECTOR


def gate_advice(report: dict) -> tuple[str, str]:
    """Recommended detector mode and why, from detector_report."""
    n, missed = report["animals"], len(report["would_miss"])
    if missed:
        return "shadow", f"It would have missed {missed} of {n} visitors, so keep Claude checking everything."
    if n < GATE_MIN_ANIMALS:
        return "shadow", f"Not enough data yet: {n} of {GATE_MIN_ANIMALS} daytime visitors needed to judge it."
    return "gate", f"It spotted all {n} visitors, so skipping Claude when it sees nothing looks safe."


def daily_counts(conn: sqlite3.Connection, days: int = 30, species: str | None = None) -> list[tuple[str, int]]:
    start = clock.today() - timedelta(days=days - 1)
    where, args = _where(species=species, date_from=start.isoformat())
    rows = conn.execute(f"SELECT date(captured_at) AS d, COUNT(*) AS n FROM visits WHERE {where} GROUP BY d", args)
    counts = {r["d"]: r["n"] for r in rows}
    days_list = [(start + timedelta(days=i)).isoformat() for i in range(days)]
    return [(d, counts.get(d, 0)) for d in days_list]


def month_counts(conn: sqlite3.Connection, species: str) -> list[int]:
    """Our own visits of one species per calendar month (Jan first), all years together."""
    where, args = _where(species=species)
    rows = conn.execute(
        f"SELECT CAST(strftime('%m', captured_at) AS INTEGER) AS m, COUNT(*) AS n FROM visits WHERE {where} GROUP BY m",
        args,
    )
    counts = {r["m"]: r["n"] for r in rows}
    return [counts.get(m, 0) for m in range(1, 13)]


def get_species_info(conn: sqlite3.Connection, name: str) -> tuple[dict, str] | None:
    row = conn.execute("SELECT data, fetched_at FROM species_info WHERE name = ?", (name,)).fetchone()
    return (json.loads(row["data"]), row["fetched_at"]) if row else None


def save_species_info(conn: sqlite3.Connection, name: str, data: dict, fetched_at: str) -> None:
    conn.execute(
        "INSERT INTO species_info (name, data, fetched_at) VALUES (?, ?, ?) "
        "ON CONFLICT(name) DO UPDATE SET data = excluded.data, fetched_at = excluded.fetched_at",
        (name, json.dumps(data), fetched_at),
    )
    conn.commit()


def hourly_counts(conn: sqlite3.Connection, species: str | None = None, false_triggers: bool = False) -> list[int]:
    where, args = _where(species=species, false_triggers=false_triggers)
    rows = conn.execute(
        f"SELECT CAST(strftime('%H', captured_at) AS INTEGER) AS h, COUNT(*) AS n FROM visits WHERE {where} GROUP BY h",
        args,
    )
    counts = {r["h"]: r["n"] for r in rows}
    return [counts.get(h, 0) for h in range(24)]
