import json
import re
import sqlite3
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from pathlib import Path

from . import clock, config, images

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
    vec BLOB NOT NULL,      -- float16
    -- For a false trigger later corrected to an animal: the saved burst frame whose crop box looks most like one
    -- (birdbrain.presence.pick_frames). animal_frame is 1-based; 0 = looked, no burst frames to look at.
    -- animal_by_owner = 1: the frame the owner's first circle is on, embedded because they circled it there.
    animal_vec BLOB,
    animal_frame INTEGER,
    animal_score REAL,
    animal_by_owner INTEGER
);
CREATE TABLE IF NOT EXISTS pi_checks (  -- one row per Ask the Pi click; no foreign key, so deleting visits keeps them
    id INTEGER PRIMARY KEY,
    at TEXT,
    seconds REAL,           -- what the owner waited (queue, model load and compute)
    ok INTEGER,
    visit_id INTEGER,
    model_seconds REAL      -- the model's own compute time, when it answered
);
CREATE TABLE IF NOT EXISTS weather_hours (  -- local hourly weather from Open-Meteo (birdbrain.weather), metric
    hour TEXT PRIMARY KEY,  -- 'YYYY-MM-DDTHH:00' in the site's local time, like visits.captured_at
    temp_c REAL, precip_mm REAL, rain_mm REAL, cloud_pct REAL, wind_kmh REAL,
    humidity_pct REAL,      -- relative humidity at 2 m
    code INTEGER,           -- WMO weather code (weather.describe turns it into words)
    is_day INTEGER,
    fetched_at TEXT
);
CREATE TABLE IF NOT EXISTS weather_days (
    day TEXT PRIMARY KEY,   -- 'YYYY-MM-DD'
    temp_max_c REAL, temp_min_c REAL, precip_mm REAL, precip_hours REAL,
    cloud_pct REAL,         -- mean of the day's hourly cloud cover
    humidity_pct REAL,      -- mean of the day's hourly relative humidity
    wind_kmh REAL,          -- mean wind speed over the daytime hours (is_day)
    wind_max_kmh REAL,      -- the windiest hour
    code INTEGER, sunrise TEXT, sunset TEXT,
    final INTEGER NOT NULL DEFAULT 0,  -- 1 once the day is long over; final days are never fetched again
    fetched_at TEXT
);
"""
# Approximate Claude prices (USD per million tokens) for the cost and savings figures on the home page. The system
# prompt is cached between calls (identify.ask): the API bills those tokens apart from input_tokens, reads at a
# tenth of the input price and writes (a 5-minute cache) at 1.25 times it, as listed for both models on 2026-11-03.
HAIKU_USD_PER_MTOK = {"input": 0.10, "output": 0.50, "cache_read": 0.01, "cache_write": 0.125}
USD_PER_MTOK = {"haiku": HAIKU_USD_PER_MTOK,
                "sonnet": {"input": 2.00, "output": 10.00, "cache_read": 0.20, "cache_write": 2.50}}
# The usage field each price applies to, as saved with a visit (raw_json.calls.haiku / .sonnet).
TOKEN_KEYS = {"input": "input_tokens", "output": "output_tokens",
              "cache_read": "cache_read_input_tokens", "cache_write": "cache_creation_input_tokens"}


def call_usd(price: dict, usage: dict) -> float:
    """What one call cost, from its token counts (cached ones counted at their own price; missing ones as 0)."""
    return sum((usage.get(key) or 0) * price[kind] for kind, key in TOKEN_KEYS.items()) / 1e6
# Owner-editable settings (Settings page). Stored in the database so they live with the data, not the code.
SETTINGS = {
    "location": "",      # e.g. "Vancouver, Washington, USA (Clark County, Pacific Northwest lowlands)"
    "scene_notes": "",   # what the camera sees: the bath, rocks, feeders
    "target_notes": "",  # what to look for: expected visitors, their sizes, where they land
    "ignore_notes": "",  # things that move or look like animals but aren't
    # Claude's distillation of every correction and note (birdbrain.lessons), accepted by the owner. In SETTINGS
    # so prompt_site carries it to every identify prompt; written only through lessons.accept.
    "lessons_summary": "",
}
# Running count and cost of the Lessons summary distills (JSON {n, usd}), for the Claude spend in totals.
DISTILL_SPEND = "lessons_distill_spend"
ADDED_COLUMNS = {"corrected_visitor": "TEXT", "feedback": "TEXT", "marks": "TEXT", "lesson_off": "INTEGER NOT NULL DEFAULT 0",
                 "also": "TEXT", "corrected_also": "TEXT", "ended_at": "TEXT",
                 "confirmed_at": "TEXT",  # the owner pressed Confirm: Claude's answer is right (birdbrain.presence)
                 # A false trigger's photo files were deleted to save space (birdbrain.retention); the row and its
                 # embedding stay, so the Pi's animal check keeps learning from it.
                 "photo_deleted_at": "TEXT",
                 # The owner saw no person in a photo the Pi's person check flagged ("Make public"):
                 # JSON {"at", "files": {photo: score}} pinning the scores they cleared (birdbrain.privacy).
                 "public_ok": "TEXT",
                 # The owner's blue keeper mark: 1 = keep this photo forever, never thinned or deleted to save space.
                 "keep": "INTEGER",
                 # Photo budget (birdbrain.retention) for real visits: NULL = the photo as captured, 'thinned' = saved
                 # once smaller, 'thumb' = only the thumbnail is left. The visit itself always stays listed.
                 "photo_level": "TEXT",
                 "photo_thinned_at": "TEXT",
                 # Laplacian variance of the stored photo inside its crop box, measured before any thinning: picks
                 # each species' sharpest photos of the day, which stay full size.
                 "photo_sharpness": "REAL",
                 # A false trigger's photo as the owner picked it: the burst frame (1-based) the big photo and the
                 # card show instead of the one Claude saw (NULL). image_path stays Claude's frame: its answer,
                 # circles and the animal check's training all go by that one (photo_rel, thumb_rel).
                 "photo_frame": "INTEGER"}
# The visit's photo files are there (false triggers past the storage limit lose them; see retention.prune).
HAS_PHOTO = "photo_deleted_at IS NULL"
# The photo itself is there, full size or thinned, not just the thumbnail: for checks that look at the frame.
FULL_FRAME = f"({HAS_PHOTO} AND photo_level IS NOT 'thumb')"
EMBEDDING_ADDED_COLUMNS = {"animal_vec": "BLOB", "animal_frame": "INTEGER", "animal_score": "REAL",
                           "animal_by_owner": "INTEGER"}

# How many corrections and notes join each identify prompt.
LESSON_LIMIT = 15
NOTE_LIMIT = 300  # characters of the owner's free-text note that go into a lesson line

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
# A row of _seen that counts as a species: not "other" (an animal nobody could name) and not unlabelled. Every
# species count (today, the day's activity, all-time totals, the weather page) uses this one filter, so they agree.
SPECIES_ONLY = "(seen_visitor != 'other' AND label IS NOT NULL)"
# Partial indexes for the hot filters. Each starts with captured_at (a day's range and newest-first order come from
# the index) and carries the columns its filter and the label need, so counts and _seen never read the wide visit
# rows (raw_json). SQLite uses a partial index only when the query's WHERE holds the index's WHERE word for word,
# so these are built from the same fragments the queries use; test_db_speed checks the plans. Created in init_db,
# after the columns they name exist.
INDEXES = (
    f"CREATE INDEX IF NOT EXISTS visits_real ON visits(captured_at, visitor, corrected_visitor, corrected_species, "
    f"species, confidence, is_night, thumb_path, also, corrected_also) WHERE {REAL}",
    "CREATE INDEX IF NOT EXISTS visits_review ON visits(captured_at, visitor, corrected_visitor, corrected_species) "
    "WHERE needs_review = 1",
    f"CREATE INDEX IF NOT EXISTS visits_false_photo ON visits(captured_at, visitor, corrected_visitor, "
    f"corrected_species, confirmed_at) WHERE {FALSE_TRIGGER} AND {HAS_PHOTO}",
)
# Every column the indexes name; a visits table without them all (a test's stub) gets no indexes.
INDEX_COLUMNS = {"captured_at", "visitor", "corrected_visitor", "corrected_species", "species", "confidence", "is_night",
                 "thumb_path", "also", "corrected_also", "needs_review", "confirmed_at", "photo_deleted_at"}
MAX_ID = "visits.max_id"  # settings: the highest visit id ever handed out (insert_visit), so no id is used twice


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


@contextmanager
def writing(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """One short read-then-write transaction that takes the write lock first (BEGIN IMMEDIATE), so two processes
    that read the same row or setting and write it back can't lose each other's change. Commits at the end, rolls
    back on an error. Nothing slow belongs inside: a capture waits for the lock (and gives up after 10 s)."""
    if not conn.in_transaction:  # else a write of this connection is pending: it holds the lock, and commits here
        conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except BaseException:
        conn.rollback()
        raise
    conn.commit()


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}


def _add_column(conn: sqlite3.Connection, table: str, col: str, kind: str) -> None:
    """ALTER TABLE ADD COLUMN, unless another process (the watcher and the web app start together after a
    deploy) added it between our look at the table and now."""
    try:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {kind}")
    except sqlite3.OperationalError as e:
        if "duplicate column name" not in str(e):
            raise


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    have = _columns(conn, "visits")
    for col, kind in ADDED_COLUMNS.items():
        if col not in have:
            _add_column(conn, "visits", col, kind)
    have = _columns(conn, "embeddings")
    for col, kind in EMBEDDING_ADDED_COLUMNS.items():
        if col not in have:
            _add_column(conn, "embeddings", col, kind)
    for table, cols in WEATHER_ADDED_COLUMNS.items():
        have = _columns(conn, table)
        for col in cols:
            if col not in have:
                _add_column(conn, table, col, "REAL")
    if INDEX_COLUMNS <= _columns(conn, "visits"):
        for sql in INDEXES:
            try:
                conn.execute(sql)
            except sqlite3.OperationalError as e:  # the other process built it between IF NOT EXISTS' look and now
                if "already exists" not in str(e):
                    raise
    # A species cover (yellow star) is always kept; covers picked before that rule get their keep mark here.
    picks = list(species_photos(conn).values())
    if picks:
        conn.execute(f"UPDATE visits SET keep = 1 WHERE keep IS NOT 1 AND id IN ({','.join('?' * len(picks))})", picks)
    conn.commit()


WEATHER_HOUR_COLS = ("temp_c", "precip_mm", "rain_mm", "cloud_pct", "wind_kmh", "humidity_pct", "code", "is_day",
                     "fetched_at")
WEATHER_DAY_COLS = ("temp_max_c", "temp_min_c", "precip_mm", "precip_hours", "cloud_pct", "humidity_pct", "wind_kmh",
                    "wind_max_kmh", "code", "sunrise", "sunset", "final", "fetched_at")
# Weather columns added after the tables first shipped; init_db adds them to older databases (all REAL, NULL until
# the day is fetched again: `python -m birdbrain.weather --backfill`).
WEATHER_ADDED_COLUMNS = {"weather_hours": ("humidity_pct",), "weather_days": ("humidity_pct", "wind_kmh", "wind_max_kmh")}


def _upsert(conn: sqlite3.Connection, table: str, key: str, cols: tuple[str, ...], rows: list[dict]) -> None:
    names = (key, *cols)
    conn.executemany(
        f"INSERT INTO {table} ({', '.join(names)}) VALUES ({', '.join('?' * len(names))}) "
        f"ON CONFLICT({key}) DO UPDATE SET {', '.join(f'{c} = excluded.{c}' for c in cols)}",
        [tuple(r.get(n) for n in names) for r in rows],
    )


def save_weather(conn: sqlite3.Connection, days: list[dict], hours: list[dict]) -> None:
    """Insert or update weather rows (dicts keyed by column name; see birdbrain.weather.parse)."""
    _upsert(conn, "weather_hours", "hour", WEATHER_HOUR_COLS, hours)
    _upsert(conn, "weather_days", "day", WEATHER_DAY_COLS, days)
    conn.commit()


def weather_days(conn: sqlite3.Connection, start: str, end: str) -> list[sqlite3.Row]:
    """Daily weather from day `start` through `end` ('YYYY-MM-DD', inclusive), oldest first."""
    return conn.execute("SELECT * FROM weather_days WHERE day BETWEEN ? AND ? ORDER BY day", (start, end)).fetchall()


def weather_hours(conn: sqlite3.Connection, start: str, end: str) -> list[sqlite3.Row]:
    """Hourly weather for the days `start` through `end` ('YYYY-MM-DD', inclusive), oldest first."""
    stop = (date.fromisoformat(end) + timedelta(days=1)).isoformat()
    return conn.execute("SELECT * FROM weather_hours WHERE hour >= ? AND hour < ? ORDER BY hour", (start, stop)).fetchall()


def weather_missing_days(conn: sqlite3.Connection, start: str, end: str) -> list[str]:
    """Days from `start` through `end` that have no weather yet or aren't final, oldest first."""
    done = {r["day"] for r in conn.execute("SELECT day FROM weather_days WHERE final = 1 AND day BETWEEN ? AND ?",
                                           (start, end))}
    d, last, out = date.fromisoformat(start), date.fromisoformat(end), []
    while d <= last:
        if d.isoformat() not in done:
            out.append(d.isoformat())
        d += timedelta(days=1)
    return out


def first_visit_day(conn: sqlite3.Connection) -> str | None:
    """'YYYY-MM-DD' of the oldest visit, or None when there are none."""
    row = conn.execute("SELECT MIN(captured_at) AS t FROM visits").fetchone()
    return row["t"][:10] if row and row["t"] else None


# The Weather & visits page (/weather). Thresholds live here so the page, its findings and the tests agree.
RAIN_HOUR_MM = 0.2      # an hour with at least this much rain or snow (mm) is a rainy hour
WET_DAY_MM = 1.0        # a day with at least this much is a wet day
OVERCAST_PCT = 70       # cloud cover at or above this is overcast
TEMP_BANDS_C = (8, 12)  # hour temperature bands: below 8, 8 to 12, 12 and up (degrees C)
# Windy from 15 km/h (about 9 mph): the top of a gentle breeze, where leaves and twigs keep moving, perches sway and
# floating leaves drift across the bath (more false alarms). Below it, branches are mostly still.
WINDY_KMH = 15
HUMIDITY_BANDS_PCT = (80, 95)  # hour humidity bands: below 80, 80 to 95, 95 and up (fall air here is mostly damp)
MIN_HOURS = 10         # each side of a comparison needs this many hours before it counts as a finding
MIN_DAYS = 2            # each side of the wet vs dry days comparison needs this many days
MIN_DAYLIGHT = 1.0      # a day needs this many watched daylight hours for a per-hour rate
# "Today at the bath" (birdbrain.outlook): photos with an animal taken in daylight hours with weather like today's
# (wet or dry, same temperature band) before the box names the regulars for that weather; fewer, and it shows the
# most seen of the last week instead.
MIN_LIKE_TODAY = 15
HOT_C = 27              # a high at or above this (about 81°F) makes a hot day
FREEZING_C = 0          # a low at or below this makes a freezing day


def _hours_between(a: datetime, b: datetime) -> float:
    return max(0.0, (b - a).total_seconds() / 3600)


def _rate_group(key: str, hours: list[dict], **extra) -> dict:
    n, v = len(hours), sum(h["visitors"] for h in hours)
    return {"key": key, "hours": n, "visitors": v, "rate": v / n if n else None, "enough": n >= MIN_HOURS, **extra}


def weather_report(conn: sqlite3.Connection, now: datetime | None = None) -> dict:
    """Weather and visits side by side, from the first stored photo through now (site local time). Reads the database
    only (the public mirror runs this too); birdbrain.weather keeps the weather tables filled.

    days: one dict per calendar day, oldest first: its weather (None when not fetched yet), real visitors (photos with
    an animal), daytime visitors per watched daylight hour (sunrise to sunset, clipped to when the camera started and to
    now), distinct species (each counted once a day, other animals in the same photo included), and false alarms
    (stored false triggers plus the probes the Pi threw away).

    rain / cloud / temp: daylight hours compared, each with its hours, visitors and visitors an hour. An hour H takes
    its rain from weather row H+1 (Open-Meteo's precipitation is the sum of the hour before it) and its cloud and
    temperature from the mean of rows H and H+1. Only whole hours after the first photo count; cloud compares dry
    hours only, so rain doesn't decide it. by_hour: those hours themselves ({"hour": 'YYYY-MM-DDTHH:00', "rain",
    "cloud", "temp", "wind", "humidity", "visitors"}), oldest first."""
    now = now or clock.now()
    report = {"days": [], "first": None, "now": now, "has_weather": False, "hours": 0, "by_hour": [], "rain": [], "cloud": [],
              "temp": [], "wind": [], "humidity": [], "points": [], "max_rain": 0.0, "max_visitors": 0}
    first = conn.execute("SELECT MIN(captured_at) FROM visits").fetchone()[0]
    if not first:
        return report
    start = datetime.fromisoformat(first)
    a, b = start.date().isoformat(), now.date().isoformat()
    wdays = {r["day"]: r for r in weather_days(conn, a, b)}
    whours = {r["hour"]: r for r in weather_hours(conn, a, b)}
    days_where, days_args = _days(a, b)
    seen = [datetime.fromisoformat(t) for (t,) in conn.execute(
        f"SELECT captured_at FROM visits WHERE {REAL} AND {days_where} ORDER BY captured_at", days_args)]
    false = {r[0]: r[1] for r in conn.execute(
        f"SELECT date(captured_at), COUNT(*) FROM visits WHERE {FALSE_TRIGGER} AND {days_where} GROUP BY 1", days_args)}
    for d, n in conn.execute("SELECT day, SUM(n) FROM counters WHERE name = 'probe_empty' AND day BETWEEN ? AND ?"
                             " GROUP BY day", (a, b)):
        false[d] = false.get(d, 0) + n
    species: dict[str, list[tuple[str, int]]] = {}
    for r in conn.execute(
            f"""SELECT date(captured_at) AS d, label, COUNT(DISTINCT id) AS n
                FROM ({_seen(f"{REAL} AND {days_where}")}) WHERE {SPECIES_ONLY}
                GROUP BY d, label ORDER BY d, n DESC, label""", days_args * 2):
        species.setdefault(r["d"], []).append((r["label"], r["n"]))
    seen_by_day: dict[date, list[datetime]] = {}
    for t in seen:
        seen_by_day.setdefault(t.date(), []).append(t)

    days = []
    d = start.date()
    while d <= now.date():
        key = d.isoformat()
        w = wdays.get(key)
        mine = seen_by_day.get(d, [])
        row = {"day": key, "date": d, "today": d == now.date(), "first": d == start.date(),
               "since": start if d == start.date() else None, "weather": dict(w) if w else None,
               "visitors": len(mine), "species": len(species.get(key, [])), "top": species.get(key, []),
               "false_alarms": false.get(key, 0), "daylight": None, "day_visitors": None, "per_hour": None}
        if w and w["sunrise"] and w["sunset"]:
            lo = max(datetime.fromisoformat(w["sunrise"]), start)
            hi = min(datetime.fromisoformat(w["sunset"]), now)
            row["daylight"] = _hours_between(lo, hi)
            row["day_visitors"] = sum(1 for t in mine if lo <= t < hi)
            if row["daylight"] >= MIN_DAYLIGHT:
                row["per_hour"] = row["day_visitors"] / row["daylight"]
        days.append(row)
        d += timedelta(days=1)

    # Whole daylight hours with weather for them.
    by_hour: dict[str, int] = {}
    for t in seen:
        k = t.strftime("%Y-%m-%dT%H:00")
        by_hour[k] = by_hour.get(k, 0) + 1
    hours = []
    h = start.replace(minute=0, second=0, microsecond=0)
    if h < start:
        h += timedelta(hours=1)
    while h + timedelta(hours=1) <= now:
        k, k1 = h.strftime("%Y-%m-%dT%H:00"), (h + timedelta(hours=1)).strftime("%Y-%m-%dT%H:00")
        w1, wd, mid = whours.get(k1), wdays.get(h.date().isoformat()), h + timedelta(minutes=30)
        if (w1 is not None and w1["precip_mm"] is not None and wd and wd["sunrise"] and wd["sunset"]
                and datetime.fromisoformat(wd["sunrise"]) <= mid < datetime.fromisoformat(wd["sunset"])):
            both = [x for x in (whours.get(k), w1) if x is not None]

            def mean(col: str) -> float | None:
                vals = [x[col] for x in both if x[col] is not None]
                return sum(vals) / len(vals) if vals else None

            hours.append({"hour": k, "rain": w1["precip_mm"], "cloud": mean("cloud_pct"), "temp": mean("temp_c"),
                          "wind": mean("wind_kmh"), "humidity": mean("humidity_pct"), "visitors": by_hour.get(k, 0)})
        h += timedelta(hours=1)
    wet = [x for x in hours if x["rain"] >= RAIN_HOUR_MM]
    dry = [x for x in hours if x["rain"] < RAIN_HOUR_MM]
    lo_t, hi_t = TEMP_BANDS_C
    temps = [x for x in hours if x["temp"] is not None]
    clouded = [x for x in dry if x["cloud"] is not None]
    winds = [x for x in hours if x["wind"] is not None]
    damp = [x for x in hours if x["humidity"] is not None]
    lo_h, hi_h = HUMIDITY_BANDS_PCT
    humidity = [_rate_group("dryish", [x for x in damp if x["humidity"] < lo_h], high=lo_h),
                _rate_group("damp", [x for x in damp if lo_h <= x["humidity"] < hi_h], low=lo_h, high=hi_h),
                _rate_group("soaked", [x for x in damp if x["humidity"] >= hi_h], low=hi_h)]
    report.update({
        "days": days, "first": start, "has_weather": bool(wdays), "hours": len(hours), "by_hour": hours,
        "rain": [_rate_group("rain", wet), _rate_group("dry", dry)],
        "cloud": [_rate_group("clearer", [x for x in clouded if x["cloud"] < OVERCAST_PCT]),
                  _rate_group("overcast", [x for x in clouded if x["cloud"] >= OVERCAST_PCT])],
        "temp": [_rate_group("cold", [x for x in temps if x["temp"] < lo_t], high=lo_t),
                 _rate_group("mild", [x for x in temps if lo_t <= x["temp"] < hi_t], low=lo_t, high=hi_t),
                 _rate_group("warm", [x for x in temps if x["temp"] >= hi_t], low=hi_t)],
        "wind": [_rate_group("calm", [x for x in winds if x["wind"] < WINDY_KMH]),
                 _rate_group("windy", [x for x in winds if x["wind"] >= WINDY_KMH])],
        # Humidity only once two of its bands have enough hours to compare (a damp fall can be all one band).
        "humidity": humidity if sum(g["enough"] for g in humidity) >= 2 else [],
        "points": [{"day": r["day"], "rain": r["weather"]["precip_mm"] or 0.0, "rate": round(r["per_hour"], 2),
                    "visitors": r["day_visitors"], "daylight": round(r["daylight"], 1),
                    "partial": r["today"] or r["first"]}
                   for r in days if r["weather"] and r["per_hour"] is not None],
        "max_rain": max([r["weather"]["precip_mm"] or 0.0 for r in days if r["weather"]] or [0.0]),
        "max_visitors": max([r["visitors"] for r in days] or [0]),
    })
    return report


def seen_labels(conn: sqlite3.Connection, start: str, end: str) -> list[sqlite3.Row]:
    """Every animal seen in a real visit from day `start` through `end` ('YYYY-MM-DD', inclusive): (id, captured_at,
    label), one row per kind per photo, other animals in the photo included, "other" left out; oldest first."""
    where, args = _days(start, end)
    return conn.execute(
        f"""SELECT DISTINCT id, captured_at, label FROM ({_seen(f"{REAL} AND {where}")})
            WHERE {SPECIES_ONLY} ORDER BY captured_at, label""", args * 2).fetchall()


def weather_findings(report: dict) -> list[str]:
    """Plain sentences from a weather_report's numbers. They say what was seen, never why."""
    per = lambda x: f"{x:.1f}"
    out: list[str] = []
    wet, dry = report["rain"] or ({"enough": False, "hours": 0}, {"enough": False, "hours": 0})
    if wet["enough"] and dry["enough"] and dry["rate"]:
        change = wet["rate"] / dry["rate"] - 1
        how = ("about the same as" if abs(change) < 0.15 else
               f"{round(-100 * change)}% fewer than" if change < 0 else f"{round(100 * change)}% more than")
        out.append(f"Rainy daylight hours had {per(wet['rate'])} visitors an hour ({wet['visitors']} in {wet['hours']} "
                   f"hours), {how} dry hours at {per(dry['rate'])} ({dry['visitors']} in {dry['hours']} hours).")
    elif report["hours"]:
        few = wet if wet["hours"] < dry["hours"] else dry
        out.append(f"Not enough {'rainy' if few is wet else 'dry'} daylight hours yet to compare rain with dry weather "
                   f"({few['hours']} so far, {MIN_HOURS} needed).")
    rated = [r for r in report["days"] if r["per_hour"] is not None and r["weather"]]
    wet_days = [r for r in rated if (r["weather"]["precip_mm"] or 0) >= WET_DAY_MM]
    dry_days = [r for r in rated if (r["weather"]["precip_mm"] or 0) < WET_DAY_MM]
    if len(wet_days) >= MIN_DAYS and len(dry_days) >= MIN_DAYS:
        pooled = lambda rs: sum(r["day_visitors"] for r in rs) / sum(r["daylight"] for r in rs)
        out.append(f"On the {len(wet_days)} wet days the bath averaged {per(pooled(wet_days))} visitors per daylight "
                   f"hour; on the {len(dry_days)} dry days, {per(pooled(dry_days))}.")
        order = {r["day"]: i for i, r in enumerate(report["days"])}
        mean_pos = lambda rs: sum(order[r["day"]] for r in rs) / len(rs)
        if mean_pos(wet_days) < mean_pos(dry_days) - 1:
            out.append("The wet days mostly came first, when the camera was new (and birds may still have been finding "
                       "the bath), so rain and newness are mixed together here.")
    whole = [r for r in rated if not r["today"] and not r["first"]]
    if len(whole) >= 3:
        best, worst = max(whole, key=lambda r: r["per_hour"]), min(whole, key=lambda r: r["per_hour"])
        rain = lambda r: (f"{r['weather']['precip_mm']:.0f} mm of rain" if (r["weather"]["precip_mm"] or 0) >= 0.5
                          else "no rain")
        name = lambda r: r["date"].strftime("%a %b %-d")
        out.append(f"Busiest whole day: {name(best)}, {best['visitors']} visitors with {rain(best)}. "
                   f"Quietest: {name(worst)}, {worst['visitors']} with {rain(worst)}.")
    if report["wind"]:
        calm, windy = report["wind"]
        if calm["enough"] and windy["enough"] and calm["rate"]:
            change = windy["rate"] / calm["rate"] - 1
            if abs(change) >= 0.15:
                out.append(f"Windy hours ({WINDY_KMH} km/h or more) had {per(windy['rate'])} visitors an hour "
                           f"({windy['hours']} hours), {round(abs(100 * change))}% {'fewer' if change < 0 else 'more'} "
                           f"than calm hours at {per(calm['rate'])} ({calm['hours']} hours).")
    if report["cloud"]:
        clear, grey = report["cloud"]
        if clear["enough"] and grey["enough"]:
            if abs(grey["rate"] - clear["rate"]) < 0.15 * max(grey["rate"], clear["rate"], 0.01):
                out.append(f"Among dry hours, overcast and clearer skies drew about the same ({per(grey['rate'])} and "
                           f"{per(clear['rate'])} visitors an hour).")
            else:
                more = "more" if grey["rate"] > clear["rate"] else "fewer"
                out.append(f"Among dry hours, overcast ones had {more} visitors ({per(grey['rate'])} an hour in "
                           f"{grey['hours']} hours) than clearer ones ({per(clear['rate'])} in {clear['hours']} hours).")
    return out


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


def delete_value(conn: sqlite3.Connection, key: str) -> None:
    conn.execute("DELETE FROM settings WHERE key = ?", (key,))
    conn.commit()


def get_json(conn: sqlite3.Connection, key: str) -> dict | None:
    """A settings value stored as a JSON object, or None when missing or unreadable."""
    raw = get_value(conn, key)
    if not raw:
        return None
    try:
        value = json.loads(raw)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def add_spend(conn: sqlite3.Connection, key: str, usd: float) -> dict:
    """Count one more Claude call under `key` (JSON {n, usd}) and return the running total."""
    with writing(conn):
        row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
        try:
            spend = json.loads(row["value"]) if row else {}
        except ValueError:
            spend = {}
        spend = {"n": int(spend.get("n") or 0) + 1, "usd": float(spend.get("usd") or 0.0) + usd}
        conn.execute("INSERT INTO settings (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                     (key, json.dumps(spend)))
    return spend


PI_CHECK_TIMES = "pi_check_times"  # legacy settings blob {n, total, recent}; still counted, never written
PI_CHECK_RECENT = 20               # the "usually" figure averages this many latest successful checks


def record_pi_check(conn: sqlite3.Connection, seconds: float, ok: bool = True, visit_id: int | None = None,
                    model_seconds: float | None = None) -> dict:
    """One row for every Ask the Pi click, failed ones too. Returns the updated pi_check_average."""
    conn.execute("INSERT INTO pi_checks (at, seconds, ok, visit_id, model_seconds) VALUES (?, ?, ?, ?, ?)",
                 (datetime.now().isoformat(timespec="seconds"), round(float(seconds), 2), int(bool(ok)),
                  visit_id, model_seconds))
    conn.commit()
    return pi_check_average(conn)


def pi_check_average(conn: sqlite3.Connection) -> dict:
    """{n, avg, recent_avg} over successful Ask-the-Pi checks, in seconds (None before the first). avg is all-time,
    including the old settings blob's count and total; recent_avg is the last PI_CHECK_RECENT."""
    try:
        old = json.loads(get_value(conn, PI_CHECK_TIMES) or "{}")
    except ValueError:
        old = {}
    n0, total0 = int(old.get("n", 0)), float(old.get("total", 0))
    n, total = conn.execute("SELECT COUNT(*), COALESCE(SUM(seconds), 0) FROM pi_checks WHERE ok = 1").fetchone()
    recent = [r[0] for r in conn.execute(
        "SELECT seconds FROM pi_checks WHERE ok = 1 ORDER BY id DESC LIMIT ?", (PI_CHECK_RECENT,))]
    if len(recent) < PI_CHECK_RECENT:  # top up from the legacy blob's last times
        recent += [float(x) for x in old.get("recent", [])][::-1][:PI_CHECK_RECENT - len(recent)]
    n, total = n + n0, total + total0
    if not n:
        return {"n": 0, "avg": None, "recent_avg": None}
    return {"n": n, "avg": round(total / n, 1), "recent_avg": round(sum(recent) / len(recent), 1) if recent else None}


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
    """Store a visit and return its id, always above every id ever used. SQLite alone would hand a deleted newest
    visit's id to the next one (the table isn't AUTOINCREMENT), and open pages compare ids to notice a new arrival;
    so the highest id given out is kept in settings (MAX_ID) and read, with the write lock held, before each insert."""
    with writing(conn):
        if "id" not in fields:
            top, mark = conn.execute(
                "SELECT COALESCE((SELECT MAX(id) FROM visits), 0), "
                "COALESCE((SELECT CAST(value AS INTEGER) FROM settings WHERE key = ?), 0)", (MAX_ID,)).fetchone()
            fields = {"id": max(top, mark) + 1, **fields}
        cols = ", ".join(fields)
        marks = ", ".join("?" for _ in fields)
        cur = conn.execute(f"INSERT INTO visits ({cols}) VALUES ({marks})", tuple(fields.values()))
        conn.execute("INSERT INTO settings (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value"
                     " WHERE CAST(value AS INTEGER) < CAST(excluded.value AS INTEGER)", (MAX_ID, str(cur.lastrowid)))
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
    # The review queue holds every visit waiting for the owner, false triggers included: "no animal" at low
    # confidence means something might have been one.
    where = [FALSE_TRIGGER] if false_triggers else ([] if review else [REAL])
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
    if date_from or date_to:
        days_where, days_args = _days(date_from, date_to)
        where.append(days_where)
        args.extend(days_args)
    return " AND ".join(where), args


def _day_after(day: str) -> str | None:
    """The 'YYYY-MM-DD' after `day`, or None when `day` isn't a date in that form (a hand-edited URL)."""
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", day):
        return None
    try:
        return (date.fromisoformat(day) + timedelta(days=1)).isoformat()
    except ValueError:
        return None


def _days(date_from: str | None = None, date_to: str | None = None) -> tuple[str, list]:
    """WHERE terms (and their arguments) for the visits of the days `date_from` through `date_to` ('YYYY-MM-DD',
    inclusive; None leaves that end open) as range compares on captured_at itself, which its indexes serve:
    'YYYY-MM-DDTHH:MM:SS' text sorts by time, and 'YYYY-MM-DD' sorts just before that day's first second. A bound
    that isn't a date gets the old date(captured_at) compare, so it filters as it always did (slowly)."""
    terms, args = [], []
    if date_from:
        terms.append("captured_at >= ?" if _day_after(date_from) else "date(captured_at) >= ?")
        args.append(date_from)
    if date_to:
        stop = _day_after(date_to)
        terms.append("captured_at < ?" if stop else "date(captured_at) <= ?")
        args.append(stop or date_to)
    return " AND ".join(terms) or "1", args


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


def photo_frame(visit) -> int | None:
    """The burst frame (1-based) the owner picked as a visit's photo, or None for the one Claude saw."""
    try:
        n = visit["photo_frame"]
    except (IndexError, KeyError):
        return None
    return n if isinstance(n, int) and n > 0 else None


def thumb_rel(visit) -> str:
    """A visit's thumbnail for the cards: the picked burst frame's (images.pick_thumb), else the photo's."""
    n = photo_frame(visit)
    return images.burst_thumb_path(visit["image_path"], n) if n else visit["thumb_path"]


def photo_rel(visit) -> str:
    """The biggest photo of a visit still on disk: its photo (the burst frame the owner picked, else the one
    Claude saw), or its thumbnail once only that is left."""
    try:
        thumb_only = visit["photo_level"] == "thumb"
    except (IndexError, KeyError):
        thumb_only = False
    if thumb_only:
        return thumb_rel(visit)
    n = photo_frame(visit)
    return images.burst_path(visit["image_path"], n) if n else visit["image_path"]


def set_photo_frame(conn: sqlite3.Connection, visit_id: int, frame: int | None) -> None:
    """Record which burst frame is the visit's photo (None: back to the one Claude saw)."""
    with writing(conn):
        conn.execute("UPDATE visits SET photo_frame = ? WHERE id = ?", (frame, visit_id))


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
    day_where, day_args = _days(today, today)
    real_today = conn.execute(
        f"""SELECT COUNT(DISTINCT id) AS n, COUNT(DISTINCT CASE WHEN {SPECIES_ONLY} THEN label END) AS species
            FROM ({_seen(f"{REAL} AND {day_where}")})""",
        day_args * 2,
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
    days_where, args = _days(date_from, date_to)
    where = f"{REAL} AND {days_where}"
    return conn.execute(
        f"SELECT label, COUNT(DISTINCT id) AS n, MAX(captured_at) AS last_at FROM ({_seen(where)})"
        f" GROUP BY label ORDER BY n DESC, last_at DESC",
        args * 2,
    ).fetchall()


def scoreboard_total(conn: sqlite3.Connection, date_from: str | None = None, date_to: str | None = None) -> int:
    """Real visits (photos with an animal) between two dates, same bounds as `scoreboard`; each photo counts once."""
    days_where, args = _days(date_from, date_to)
    return conn.execute(f"SELECT COUNT(*) FROM visits WHERE {REAL} AND {days_where}", args).fetchone()[0]


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


def set_keep(conn: sqlite3.Connection, visit_id: int, on: bool) -> None:
    """The blue keeper mark: this photo is never thinned or deleted to save space."""
    conn.execute("UPDATE visits SET keep = ? WHERE id = ?", (1 if on else None, visit_id))
    conn.commit()


def set_public_ok(conn: sqlite3.Connection, visit_id: int, override: dict | None) -> None:
    """Record (or with None, take back) the owner's Make public for a visit's flagged photos."""
    conn.execute("UPDATE visits SET public_ok = ? WHERE id = ?",
                 (json.dumps(override) if override else None, visit_id))
    conn.commit()


MAX_FRAME = 60  # a burst frame number past this isn't one (bursts are 6 frames)


def parse_marks(text: str | None, claude_frame: int | None = None) -> list[dict]:
    """The owner's numbered circles on a photo: [{x, y, r}] as fractions of the width, height and width. At most 9.
    A circle drawn on another frame of the burst carries `frame` (1-based, as the visit page's filmstrip counts);
    none means the photo Claude saw. `claude_frame` (that photo's place in the burst) folds `frame` back into none."""
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
        frame = m.get("frame")
        if (isinstance(frame, int) and not isinstance(frame, bool) and 1 <= frame <= MAX_FRAME
                and frame != claude_frame and not (mark.get("claude") or mark.get("pi"))):
            mark["frame"] = frame  # Claude's and the Pi's circles belong to the photo they were drawn on
        marks.append(mark)
    return marks[:9]


def describe_marks(marks: list[dict]) -> str:
    """Where each circle sits, so Claude can match the owner's note ("1 is a rock") to a place in the frame."""
    return "; ".join(
        f"circle {i}{_drew(m)} at {m['x']:.0%} across, {m['y']:.0%} down"
        + (f" (on burst frame {m['frame']}, not the photo you saw)" if m.get("frame") else "")
        for i, m in enumerate(marks, 1)
    )


def mark_frame(mark: dict) -> int | None:
    """The burst frame a circle was drawn on (1-based), or None for the photo Claude saw."""
    return mark.get("frame") or None


def claude_frame(raw_json: str | dict | None) -> int | None:
    """Where the photo Claude saw sits in the visit's saved burst (1-based), or None without a burst."""
    try:
        raw = json.loads(raw_json or "{}") if isinstance(raw_json, (str, bytes)) or raw_json is None else raw_json
        best = (raw.get("burst") or {}).get("best")
    except (ValueError, AttributeError):
        return None
    return best + 1 if isinstance(best, int) and not isinstance(best, bool) else None


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


# What settled a visit on the Pi without asking Claude (identify.settled_*), as a lesson line tells Claude.
SKIPPED_BY = {"presence": "the Pi's animal check judged this frame empty",
              "detector": "the Pi's local detector found no animal"}


def skipped_by(raw_json: str | None) -> str | None:
    """How the Pi settled a visit without asking Claude: "presence" (the local animal check judged it empty),
    "detector" (the local detector found nothing) or "classifier" (the local classifier named it). None when
    Claude was asked, or the saved trail is missing or unreadable."""
    try:
        calls = (json.loads(raw_json or "{}") or {}).get("calls")
    except (ValueError, TypeError, AttributeError):
        return None
    if not isinstance(calls, dict) or not calls.get("skipped_claude"):
        return None
    if calls.get("settled_by") == "classifier":
        return "classifier"
    presence = calls.get("presence")
    return "presence" if isinstance(presence, dict) and presence.get("empty") else "detector"


def all_lessons(conn: sqlite3.Connection) -> list[dict]:
    """Every correction and note, newest first, as the line Claude would read, with whether it's in the prompt now.
    A visit the Pi settled without asking Claude reads as such (Claude never answered it). One where the owner
    only agreed with the Pi, with no note, circles or other animals, tells Claude nothing and is left out."""
    rows = conn.execute(
        "SELECT id, captured_at, visitor, species, corrected_visitor, corrected_species, feedback, marks, is_night, "
        f"lesson_off, also, corrected_also, raw_json FROM visits WHERE {CORRECTED} ORDER BY id DESC"
    ).fetchall()
    lessons, used = [], 0
    for r in rows:
        said = _spoken_name(r["species"], r["visitor"])
        actual = r["corrected_species"] or (_spoken_name(None, r["corrected_visitor"]) if r["corrected_visitor"] else None)
        when = "night (IR)" if r["is_night"] else "day"
        marks = parse_marks(r["marks"])
        owner_also = also_of(r) if r["corrected_also"] is not None else None
        if by := skipped_by(r["raw_json"]):
            changed = actual is not None and actual != said
            if not (changed or r["feedback"] or marks or owner_also):
                continue  # the owner agreed with the Pi and said nothing more: no lesson for Claude
            head = (f"the Pi's species classifier named it \"{said}\"" if by == "classifier" else SKIPPED_BY[by])
            prefix = f"- {r['captured_at'][:10]}, {when}: {head}, so you weren't asked"
            line = (f"{prefix}; the owner says it was \"{actual}\"." if changed
                    else f"{prefix}; the owner confirmed {'no animal was there' if r['visitor'] == 'none' else 'it'}.")
            said_by = "the Pi"
        else:
            if claude_also := also_of({"corrected_also": None, "also": r["also"]}):
                said += f"\" with \"{_spoken_also(claude_also)}"
            prefix = f"- {r['captured_at'][:10]}, {when}: you said \"{said}\""
            line = f"{prefix}; the owner says it was \"{actual}\"." if actual else f"{prefix} and the owner confirmed it."
            said_by = "Claude"
        if owner_also is not None:
            line += (f" The owner says also there: \"{_spoken_also(owner_also)}\"." if owner_also
                     else " The owner says no other animal was there.")
        if r["feedback"]:
            note = re.sub(r"\s+", " ", r["feedback"]).strip()
            if len(note) > NOTE_LIMIT:  # a long note is cut for the prompt; the visit keeps all of it
                note = note[:NOTE_LIMIT - 1].rstrip() + "…"
            line += " Owner's note: " + note
        if marks:
            line += f" The owner circled spots on the full frame: {describe_marks(marks)}."
        in_use = not r["lesson_off"] and used < LESSON_LIMIT
        used += in_use
        lessons.append({"visit_id": r["id"], "captured_at": r["captured_at"], "line": line,
                        "off": bool(r["lesson_off"]), "in_use": in_use,
                        # what the edit form shows: the first answer stays fixed, the owner's parts can change
                        "said": said, "said_by": said_by, "when": when,
                        "visitor": r["corrected_visitor"] or r["visitor"],
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
        with writing(conn):
            conn.execute("DELETE FROM visits WHERE id = ?", (visit_id,))
            photos = species_photos(conn)
            if visit_id in photos.values():  # a pick of a visit that is gone
                _save_species_photos(conn, {k: v for k, v in photos.items() if v != visit_id})
    return row


SPECIES_PHOTO = "species_photo"  # settings key: JSON {label: visit id}, the owner's pick of each species' photo


def species_photos(conn: sqlite3.Connection) -> dict[str, int]:
    """The visits the owner picked as species photos, by label."""
    try:
        return {str(k): int(v) for k, v in json.loads(get_value(conn, SPECIES_PHOTO) or "{}").items()}
    except (ValueError, TypeError, AttributeError):
        return {}


def _save_species_photos(conn: sqlite3.Connection, photos: dict[str, int]) -> None:
    """Writes the picks without committing: call it inside writing(), after species_photos() read them there, so a
    pick made in the other process meanwhile isn't written over."""
    conn.execute(
        "INSERT INTO settings (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (SPECIES_PHOTO, json.dumps(photos, sort_keys=True)),
    )


def shows_species(conn: sqlite3.Connection, visit_id: int, label: str) -> bool:
    """Whether a real visit has this species in it, as the main animal or one of the others."""
    return conn.execute(f"SELECT 1 FROM ({_seen(REAL + ' AND id = ?')}) WHERE label = ? LIMIT 1",
                        (visit_id, visit_id, label)).fetchone() is not None


def set_species_photo(conn: sqlite3.Connection, label: str, visit_id: int | None) -> None:
    """Use a visit's photo for its species (None: back to the automatic pick). If the visit is later deleted or
    relabelled, species_summary falls back to the automatic pick on its own."""
    with writing(conn):
        photos = species_photos(conn)
        if visit_id is None:
            photos.pop(label, None)
        else:
            photos[label] = int(visit_id)
        _save_species_photos(conn, photos)


STAR_STATES = ("unset", "favorite", "cover")


def set_star(conn: sqlite3.Connection, label: str, visit_id: int, state: str | None) -> tuple[str, int | None]:
    """The star on a photo: unset -> favorite (blue, kept) -> species cover (yellow, one per species, also kept)
    -> unset (no longer kept). `state` sets one outright; None moves one step along. A new cover leaves the old
    one as a kept favorite. One transaction, so two quick clicks (or the other process) can't lose a cover; and
    the keep mark only goes when the visit is no cover of any species any more. Returns (the state set, the
    species' cover now)."""
    if state is not None and state not in STAR_STATES:
        raise ValueError(f"unknown star state {state!r}")
    with writing(conn):
        photos = species_photos(conn)
        if state is None:
            kept = conn.execute("SELECT keep IS 1 FROM visits WHERE id = ?", (visit_id,)).fetchone()
            now = "cover" if photos.get(label) == visit_id else "favorite" if kept and kept[0] else "unset"
            state = STAR_STATES[(STAR_STATES.index(now) + 1) % len(STAR_STATES)]
        if state == "cover":
            photos[label] = int(visit_id)
        elif photos.get(label) == visit_id:
            photos.pop(label)
        _save_species_photos(conn, photos)
        keep = state != "unset" or visit_id in photos.values()
        conn.execute("UPDATE visits SET keep = ? WHERE id = ?", (1 if keep else None, visit_id))
    return state, photos.get(label)


def species_without_favorite(conn: sqlite3.Connection, min_photos: int = 2) -> list[str]:
    """Species labels with `min_photos`+ visits that still have a photo and no owner pick that is still valid
    (the picked visit must still show the species and still have its photo), most visited first."""
    where, args = _where()
    rows = conn.execute(f"SELECT label, id FROM ({_seen(where + ' AND ' + FULL_FRAME)})", [*args, *args]).fetchall()
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


def known_species(conn: sqlite3.Connection) -> set[str]:
    """Every name the visits have carried: the labels shown now (main and other animals), plus what Claude said
    before a correction. Cards and reference photos are only fetched for these, never for a name made up in a URL."""
    rows = conn.execute(
        """SELECT species AS n FROM visits WHERE species IS NOT NULL
           UNION SELECT corrected_species FROM visits WHERE corrected_species IS NOT NULL
           UNION SELECT json_extract(j.value, '$.species') FROM visits v, json_each(v.also) j
                 WHERE json_extract(j.value, '$.species') IS NOT NULL
           UNION SELECT json_extract(j.value, '$.species') FROM visits v, json_each(v.corrected_also) j
                 WHERE json_extract(j.value, '$.species') IS NOT NULL"""
    ).fetchall()
    return {r["n"] for r in rows} | set(species_names(conn))


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
    cols = ", ".join(f"AVG(COALESCE(json_extract(raw_json, '$.calls.haiku.{key}'), 0)) AS {kind}"
                     for kind, key in TOKEN_KEYS.items())
    row = conn.execute(
        f"""SELECT {cols}
            FROM (SELECT raw_json FROM visits
                  WHERE json_extract(raw_json, '$.calls.haiku.input_tokens') IS NOT NULL
                  ORDER BY id DESC LIMIT ?)""",
        (sample,),
    ).fetchone()
    if row["input"] is None:
        return 0.0
    return call_usd(HAIKU_USD_PER_MTOK, {key: row[kind] for kind, key in TOKEN_KEYS.items()})


def activity(conn: sqlite3.Connection, day: str | None = None) -> dict:
    """One day's motion events, broken down so the rows add up:

    visitors + false_alarms = all motion events
    false_alarms = kept_on_pi (discarded probes + visits the Pi settled) + false alarms Claude checked
    sent_to_claude = every visit Claude was asked about (visitors and false alarms), with what it cost
    would_keep = Claude calls the animal check would have skipped in gate mode (it judged them empty)
    """
    day = day or clock.today().isoformat()
    day_where, day_args = _days(day, day)
    r = conn.execute(
        f"""SELECT
              SUM({REAL}) AS visitors,
              (SELECT COUNT(DISTINCT label) FROM ({_seen(f"{REAL} AND {day_where}")}) WHERE {SPECIES_ONLY}) AS species,
              SUM({FALSE_TRIGGER}) AS stored_false,
              SUM(model IN ('yolox-tiny', 'bioclip-presence', 'bioclip-2')) AS settled_on_pi,
              SUM(model LIKE 'claude%') AS sent,
              SUM(model LIKE 'claude%'
                  AND json_extract(raw_json, '$.calls.presence.empty') = 1) AS would_keep
            FROM visits WHERE {day_where}""",
        day_args * 3,
    ).fetchone()
    probes = conn.execute(
        "SELECT COALESCE(SUM(n), 0) FROM counters WHERE name = 'probe_empty' AND day = ?", (day,)
    ).fetchone()[0]
    spend = 0.0
    for raw in conn.execute(f"SELECT raw_json FROM visits WHERE {day_where} AND model LIKE 'claude%'", day_args):
        calls = (json.loads(raw[0] or "{}") or {}).get("calls") or {}
        for model, price in USD_PER_MTOK.items():
            spend += call_usd(price, calls.get(model) or {})
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
    return " + ".join(f"COALESCE(SUM(json_extract(raw_json, '$.calls.{model}.{key}')), 0) * {price[kind]}"
                      for kind, key in TOKEN_KEYS.items())


TOTALS_TTL = 60  # seconds the About page's all-time figures are reused: they read every visit's raw_json
_totals_cache: dict[tuple[str, str], tuple[float, dict]] = {}  # (what, database file) -> (when, value)


def _cached(conn: sqlite3.Connection, what: str, max_age: float, build: Callable[[], dict]) -> dict:
    """build()'s result, kept for max_age seconds per database file (the watcher and the web app each keep their
    own; a test's fresh database never sees another's). Nothing invalidates it but time."""
    key = (what, conn.execute("PRAGMA database_list").fetchone()["file"])
    hit = _totals_cache.get(key)
    if hit and time.monotonic() - hit[0] < max_age:
        return hit[1]
    value = build()
    _totals_cache[key] = (time.monotonic(), value)
    return value


def forget_totals() -> None:
    """Drop the cached totals (tests)."""
    _totals_cache.clear()


def totals(conn: sqlite3.Connection, max_age: float = TOTALS_TTL) -> dict:
    """All-time counts for the About page: how far motion events got down the pipeline, and what Claude cost.
    Cached for a minute (TOTALS_TTL): the token sums parse every row's raw_json."""
    return _cached(conn, "totals", max_age, lambda: _totals(conn, max_age))


def _totals(conn: sqlite3.Connection, max_age: float) -> dict:
    r = conn.execute(
        f"""SELECT
              MIN(captured_at) AS since,
              SUM({REAL}) AS visitors,
              (SELECT COUNT(DISTINCT label) FROM ({_seen(REAL)}) WHERE {SPECIES_ONLY}) AS species,
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
    checks = owner_checks_spend(conn, max_age)
    distills = get_json(conn, DISTILL_SPEND) or {}
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
        "claude_usd": (r["claude_usd"] or 0.0) + checks["usd"] + float(distills.get("usd") or 0.0),
        "owner_checks": checks["n"],
        "owner_checks_usd": checks["usd"],
        "distills": int(distills.get("n") or 0),
        "distill_usd": float(distills.get("usd") or 0.0),
    }


def owner_checks_spend(conn: sqlite3.Connection, max_age: float = TOTALS_TTL) -> dict:
    """The Claude calls the owner asked for on a circle (birdbrain.circle_check): how many and what they cost.
    Cached like totals (it reads every row's raw_json too)."""
    def build() -> dict:
        r = conn.execute(
            """SELECT COUNT(*) AS n, COALESCE(SUM(json_extract(c.value, '$.usd')), 0) AS usd
               FROM visits v,
                    json_each(CASE WHEN json_valid(v.raw_json) THEN v.raw_json END, '$.calls.owner_checks') c
               WHERE json_extract(c.value, '$.source') IN ('haiku', 'sonnet')"""
        ).fetchone()
        return {"n": r["n"] or 0, "usd": r["usd"] or 0.0}
    return _cached(conn, "owner_checks", max_age, build)


def capture_rate(conn: sqlite3.Connection, days: int = 7) -> dict:
    """How often photos are being captured, for the About page: recent and all-time daily rates, the busiest
    hour and the typical wait between real visitors. Days with no rows still count, so a quiet week reads low."""
    today = clock.today()
    since = (today - timedelta(days=days - 1)).isoformat()
    # Each count on its own, so the all-time ones come from the indexes and the recent ones from a range on them.
    r = conn.execute(
        f"""SELECT (SELECT COUNT(*) FROM visits) AS photos,
                   (SELECT COUNT(*) FROM visits WHERE {REAL}) AS visitors,
                   (SELECT COUNT(*) FROM visits WHERE captured_at >= ?) AS recent_photos,
                   (SELECT COUNT(*) FROM visits WHERE {REAL} AND captured_at >= ?) AS recent_visitors,
                   (SELECT substr(MIN(captured_at), 1, 10) FROM visits) AS first_day,
                   (SELECT MAX(captured_at) FROM visits) AS last""",
        (since, since),
    ).fetchone()
    if not r["photos"]:
        return {"photos": 0}
    span = max(1, (today - date.fromisoformat(r["first_day"])).days + 1)
    recent_days = min(days, span)
    hour = conn.execute(
        f"SELECT CAST(strftime('%H', captured_at) AS INTEGER) AS h, COUNT(*) AS n FROM visits"
        f" WHERE {REAL} AND captured_at >= ? GROUP BY h ORDER BY n DESC, h LIMIT 1",
        (since,),
    ).fetchone()
    times = [datetime.fromisoformat(t) for (t,) in conn.execute(
        f"SELECT captured_at FROM visits WHERE {REAL} AND captured_at >= ? ORDER BY captured_at", (since,))]
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
            WHERE captured_at >= ? AND is_night = 0 AND model LIKE 'claude%'
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
    """Replace a visit's whole raw_json. Only for a value built just now, in one go: a backfill that read the row,
    ran a model and writes the dict back wipes what landed in between (set_raw_entry is for that)."""
    conn.execute("UPDATE visits SET raw_json = ? WHERE id = ?", (json.dumps(raw), visit_id))
    conn.commit()


# Parameters: (path, JSON text, visit id). raw_json that isn't a JSON object (NULL, or damaged) starts over.
SET_RAW_ENTRY = ("UPDATE visits SET raw_json = json_set(CASE WHEN json_valid(raw_json) AND json_type(raw_json) = 'object'"
                 " THEN raw_json ELSE '{}' END, ?, json(?)) WHERE id = ?")


def set_raw_entry(conn: sqlite3.Connection, visit_id: int, path: str, value) -> None:
    """Write one entry of a visit's raw_json ('$.calls.classifier', say) and leave the rest as it is now, so the
    watcher's stay end or the owner's check stored since the row was read stays."""
    conn.execute(SET_RAW_ENTRY, (path, json.dumps(value), visit_id))
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
            WHERE captured_at >= ? AND model LIKE 'claude%'
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
        where, args = "AND captured_at >= ?", ((clock.today() - timedelta(days=days - 1)).isoformat(),)
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
            FROM visits WHERE json_extract(raw_json, '$.calls.presence.score') IS NOT NULL AND captured_at >= ?
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
