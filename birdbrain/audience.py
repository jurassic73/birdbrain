"""Guests: anonymous counts of who looks at the public site and what they use.

Counting happens only on the public mirror (BIRDBRAIN_PUBLIC=1), where outside visitors are and where Cloudflare
names their country (CF-IPCountry). The admin's site on the Pi never counts anything; it pulls the mirror's counts
(pull()) and shows them on the Guests page, admin only.

What is counted, per 5-minute slot, in a `counts` table of (slot, metric, n):
- page views per named page ("page:home"), and people per page ("page_people:home")
- visitors: people once a UTC day ("visitors"), the returning ones among them ("returning"), and their
  country ("country:US", XX when Cloudflare gives none)
- clicks per control from the CONTROLS allowlist ("click:theme") and people per control ("click_people:theme")

People are told apart only by a hash of their address and browser with a random salt that is drawn anew every
UTC day, so nobody can be followed from one day to the next, and no address is ever stored. "Returning" is a
cookie that holds no identifier, only "this browser has been here before". The owner's own browsers are left out
by opening any page once with ?count=no (?count=yes undoes it), which sets a cookie; it needs no secret, since a
stranger who sets it only takes their own visits out. Obvious bots and link previewers are skipped.

Counts are kept in memory and written about once a minute (a write per request would be wasteful), to a SQLite
file of their own (AUDIENCE_DB) outside the data the Pi pushes to the mirror, which every push overwrites.

The Pi gets them with GET FEED_PATH?since=SLOT and the shared secret AUDIENCE_TOKEN in the X-Audience-Token header.
Without a token configured the mirror answers that path with a 404 like any missing page, and so it does for a
wrong token. The world map's outlines (web/world-110m.json) are Natural Earth's 1:110m countries, public domain.
"""
import hashlib
import hmac
import json
import logging
import math
import os
import re
import secrets
import sqlite3
import threading
import time
from contextlib import closing
from datetime import datetime, timedelta
from pathlib import Path

from . import config

log = logging.getLogger("birdbrain.audience")

# The page's name and address, in one place: a rename is a change here (and nowhere else in the code).
NAME = "Guests"
PATH = "/guests"
FEED_PATH = "/api/audience-counts"
TRACK_PATH = "/api/track"
TOKEN_HEADER = "x-audience-token"

SLOT = 300            # seconds per counts row
FLUSH_EVERY = 60      # seconds between writes of the pending counts
PULL_EVERY = 120      # the Pi asks the mirror at most this often (when the page is opened)
PULL_OVERLAP = 86400  # each pull asks again for the last day, so a slot still filling is brought up to date
MIRROR_KEEP = 400 * 86400  # the mirror forgets rows older than this; the Pi keeps everything it pulled

SEEN_COOKIE = "bb_seen"       # "1": this browser has been here before (no identifier)
NOCOUNT_COOKIE = "bb_nocount"  # "1": the owner's own browser, never counted
COOKIE_AGE = 400 * 86400

# The public pages, by the route they come from: (metric name, label on the Guests page).
PAGES = {
    "/": ("home", "Visits (home)"),
    "/live": ("live", "Live"),
    "/visit/{visit_id}": ("visit", "A visit"),
    "/species": ("species_list", "Species list"),
    "/species/{name}": ("species", "A species page"),
    "/stats": ("stats", "Stats"),
    "/weather": ("weather", "Weather"),
    "/about": ("about", "About"),
}
PAGE_LABELS = dict(PAGES.values())

# The controls whose clicks count (data-track="name" in the templates, or bbTrack("name") in a script). Only these
# names count, so a forged one counts nothing.
CONTROLS = {
    "theme": "Light/dark toggle",
    "palette": "Palette switch",
    "overlays": "Hide circles (header eye)",
    "species_card": "Species info card opened",
    "bird_song": "Played a bird song",
    "photo_zoom": "Photo enlarged",
    "live_zoom": "Live view enlarged",
    "species_photos": "Species photos full size",
    "visit_arrows": "Next/previous visit arrows",
    "filter_chip": "Filter chips",
    "more_filters": "More filters",
    "scoreboard": "Scoreboard period",
    "load_more": "Load more",
    "species_info_tab": "Species Info tab",
    "burst_play": "Burst play",
    "burst_auto": "Burst auto-play toggle",
    "burst_frames": "Burst frame steps",
    "photo_circles": "Circles on the visit photo",
    "photo_zones": "Zones on the visit photo",
    "keys": "Keyboard shortcuts",
    "tour": "Run the tour",
}

COUNTRY_RE = re.compile(r"[A-Z][A-Z0-9]")  # ISO codes, plus Cloudflare's T1 (Tor) and XX
BOT_RE = re.compile(r"bot|crawl|spider|slurp|curl|wget|python|httpx|aiohttp|go-http|java/|okhttp|headless|"
                    r"lighthouse|pingdom|uptime|monitor|preview|scan|facebookexternalhit|whatsapp|telegram|"
                    r"discord|slack|embedly|skype|bingpreview", re.I)

PERIODS = (("today", "Today"), ("7d", "7 days"), ("30d", "30 days"), ("year", "Year"))
PERIOD_WORDS = {"today": "today", "7d": "the last 7 days", "30d": "the last 30 days", "year": "the last year"}


# ---- settings (read when used, so tests and the .env change them without a reload) --------------------------------

def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def counting_on(public: bool) -> bool:
    """Counting runs on the public mirror only, unless AUDIENCE_COUNT=0 turns it off there too."""
    return public and _env("AUDIENCE_COUNT", "1").lower() not in ("0", "off", "no", "false")


def db_path() -> Path:
    """The counts file: AUDIENCE_DB, else audience.db in the data folder. The mirror must name one outside the
    pushed data (the Docker image uses /state/audience.db)."""
    return Path(_env("AUDIENCE_DB") or config.DATA_DIR / "audience.db")


def token() -> str:
    return _env("AUDIENCE_TOKEN")


def source_url() -> str:
    return _env("AUDIENCE_SOURCE_URL")


def token_ok(given: str | None) -> bool:
    """Whether a request may read the counts: a token is configured and the given one is the same."""
    want = token()
    return bool(want and given) and hmac.compare_digest(given.encode(), want.encode())


# ---- what counts ---------------------------------------------------------------------------------------------------

def country(header: str | None) -> str:
    """Cloudflare's two-letter country for the client, or XX when absent or malformed."""
    code = (header or "").strip().upper()
    return code if COUNTRY_RE.fullmatch(code) else "XX"


def client(headers) -> str:
    """The visitor's address as Cloudflare gives it. Only requests carrying it are counted at all (skipped()), and it
    is only ever hashed, never kept. Not checked against a trusted proxy: the mirror's port is reachable only through
    the proxy, and a forged header could at worst move a count."""
    return headers.get("cf-connecting-ip", "")


def is_bot(user_agent: str | None) -> bool:
    ua = (user_agent or "").strip()
    return not ua or bool(BOT_RE.search(ua))


def page_name(route: str | None) -> str | None:
    """The page metric for a route template, None for anything that isn't a public page."""
    entry = PAGES.get(route or "")
    return entry[0] if entry else None


def owner_switch(query_count: str | None) -> str | None:
    """'off' for ?count=no (set the owner's cookie), 'on' for ?count=yes (clear it), else None."""
    return {"no": "off", "yes": "on"}.get((query_count or "").lower())


def skipped(headers, cookies, query_count: str | None = None) -> bool:
    """Requests that never count: not through Cloudflare (no CF-Connecting-IP: a check on the server itself), a bot,
    or the owner's own browser (its cookie, or the ?count=no that sets it; ?count=yes counts again)."""
    if not headers.get("cf-connecting-ip"):
        return True
    if is_bot(headers.get("user-agent")):
        return True
    switch = owner_switch(query_count)
    if switch == "on":
        return False
    return switch == "off" or cookies.get(NOCOUNT_COOKIE) == "1"


def is_page_view(method: str, status: int, content_type: str, headers) -> bool:
    """A real look at a page: a GET answered 200 with HTML, asked for as a document (not an htmx swap or a script's
    fetch, which say otherwise in Sec-Fetch-Dest) and not a prefetch or prerender. A prerendered page that is then
    shown counts through static/track.js instead."""
    if method != "GET" or status != 200 or not content_type.startswith("text/html"):
        return False
    if headers.get("hx-request"):
        return False
    dest = headers.get("sec-fetch-dest")
    if dest is not None and dest != "document":
        return False
    purpose = (headers.get("sec-purpose") or headers.get("purpose") or "").lower()
    return "prefetch" not in purpose and "prerender" not in purpose


class Counter:
    """Pending counts in memory and today's anonymous names. Times are Unix seconds (passed in, so tests can step
    the clock without sleeping)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.pending: dict[tuple[int, str], int] = {}
        self.day = ""
        self.salt = ""
        self.seen: set[str] = set()
        self.flushed_at = 0.0
        self.loaded = False

    def bump(self, metric: str, now: float, n: int = 1) -> None:
        key = (int(now) // SLOT * SLOT, metric)
        self.pending[key] = self.pending.get(key, 0) + n

    def roll(self, now: float) -> None:
        """A new salt and no remembered names when the UTC date changes."""
        day = time.strftime("%Y-%m-%d", time.gmtime(now))
        if self.day != day:
            self.day, self.salt = day, secrets.token_hex(16)
            self.seen.clear()

    def person(self, ip: str, user_agent: str) -> str:
        """Today's anonymous name for a client: its address and browser with the day's salt, hashed and cut short."""
        return hashlib.sha256(f"{self.salt}|{ip}|{user_agent}".encode()).hexdigest()[:16]

    def first_today(self, key: str) -> bool:
        if key in self.seen:
            return False
        self.seen.add(key)
        return True

    def view(self, page: str, ip: str, user_agent: str, where: str, been_before: bool, now: float) -> None:
        """A page view; the person's first today also counts them as a visitor, returning or not, and their country."""
        if page not in PAGE_LABELS:
            return
        with self.lock:
            self.roll(now)
            who = self.person(ip, user_agent)
            self.bump("page:" + page, now)
            if self.first_today(who + "|page:" + page):
                self.bump("page_people:" + page, now)
            if self.first_today(who):
                self.bump("visitors", now)
                self.bump("country:" + where, now)
                if been_before:
                    self.bump("returning", now)

    def click(self, control: str, ip: str, user_agent: str, now: float) -> bool:
        """A click on an allowlisted control (False, and nothing counted, for any other name)."""
        if control not in CONTROLS:
            return False
        with self.lock:
            self.roll(now)
            who = self.person(ip, user_agent)
            self.bump("click:" + control, now)
            if self.first_today(who + "|click:" + control):
                self.bump("click_people:" + control, now)
        return True

    def take(self) -> list[tuple[int, str, int]]:
        """The pending counts as rows, forgotten here."""
        with self.lock:
            rows = [(slot, metric, n) for (slot, metric), n in self.pending.items()]
            self.pending.clear()
        return rows

    def state(self) -> dict:
        with self.lock:
            return {"day": self.day, "salt": self.salt, "seen": sorted(self.seen)}

    def restore(self, saved: dict) -> None:
        """Today's salt and names from before a restart, so nobody is counted twice; another day's are ignored."""
        with self.lock:
            if saved.get("day") and saved.get("salt") and not self.day:
                self.day, self.salt = saved["day"], saved["salt"]
                self.seen = set(saved.get("seen") or [])


counter = Counter()


# ---- storage -------------------------------------------------------------------------------------------------------

SCHEMA = ("CREATE TABLE IF NOT EXISTS counts (slot INTEGER NOT NULL, metric TEXT NOT NULL, n INTEGER NOT NULL, "
          "PRIMARY KEY (slot, metric)) WITHOUT ROWID;"
          "CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT NOT NULL);")
ADD = "INSERT INTO counts (slot, metric, n) VALUES (?, ?, ?) ON CONFLICT (slot, metric) DO UPDATE SET n = n + excluded.n"
REPLACE = "INSERT OR REPLACE INTO counts (slot, metric, n) VALUES (?, ?, ?)"


def connect() -> sqlite3.Connection:
    path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA)
    return conn


def _get_state(conn, key: str) -> dict:
    row = conn.execute("SELECT value FROM state WHERE key = ?", (key,)).fetchone()
    try:
        return json.loads(row[0]) if row else {}
    except ValueError:
        return {}


def _put_state(conn, key: str, value: dict) -> None:
    conn.execute("INSERT OR REPLACE INTO state (key, value) VALUES (?, ?)", (key, json.dumps(value)))


def load(c: Counter | None = None) -> None:
    """Once per process: pick up today's salt and names saved before a restart."""
    c = c or counter
    if c.loaded:
        return
    c.loaded = True
    try:
        with closing(connect()) as conn:
            c.restore(_get_state(conn, "visitors"))
    except sqlite3.Error as e:
        log.warning("guests: couldn't read the saved counter state: %s", e)


def flush(c: Counter | None = None, now: float | None = None) -> None:
    """Add the pending counts to the file and save today's salt and names; forget rows past MIRROR_KEEP."""
    c = c or counter
    now = time.time() if now is None else now
    c.flushed_at = now
    rows = c.take()
    try:
        with closing(connect()) as conn, conn:
            conn.executemany(ADD, rows)
            _put_state(conn, "visitors", c.state())
            conn.execute("DELETE FROM counts WHERE slot < ?", (int(now) - MIRROR_KEEP,))
    except sqlite3.Error as e:
        log.warning("guests: couldn't save %d counts: %s", len(rows), e)
        with c.lock:  # keep them for the next try
            for slot, metric, n in rows:
                c.pending[(slot, metric)] = c.pending.get((slot, metric), 0) + n


def maybe_flush(c: Counter | None = None, now: float | None = None) -> None:
    c = c or counter
    now = time.time() if now is None else now
    if now - c.flushed_at >= FLUSH_EVERY:
        flush(c, now)


def rows_since(since: int) -> list[list]:
    """Every counts row from slot `since` on, for the Pi's pull (pending ones written first)."""
    flush()
    with closing(connect()) as conn:
        return [list(r) for r in conn.execute("SELECT slot, metric, n FROM counts WHERE slot >= ? ORDER BY slot",
                                              (max(0, since),))]


def merge(rows) -> int:
    """Store pulled rows, each replacing its (slot, metric): pulling the same rows twice changes nothing.
    Malformed rows are skipped. Returns how many were stored."""
    good = []
    for r in rows:
        try:
            slot, metric, n = int(r[0]), str(r[1]), int(r[2])
        except (TypeError, ValueError, IndexError):
            continue
        if len(metric) <= 64 and n >= 0:
            good.append((slot, metric, n))
    with closing(connect()) as conn, conn:
        conn.executemany(REPLACE, good)
    return len(good)


# ---- the Pi's pull -------------------------------------------------------------------------------------------------

def pull(now: float | None = None, force: bool = False) -> None:
    """Fetch new counts from the mirror (AUDIENCE_SOURCE_URL with AUDIENCE_TOKEN), at most every PULL_EVERY seconds.
    Called when the Guests page opens, so the Pi does nothing while nobody looks. A failure is logged and noted
    for the page; the counts already pulled stay."""
    url, key = source_url(), token()
    if not url or not key:
        return
    now = time.time() if now is None else now
    with closing(connect()) as conn:
        st = _get_state(conn, "pull")
        if not force and now - st.get("tried", 0) < PULL_EVERY:
            return
        newest = conn.execute("SELECT MAX(slot) FROM counts").fetchone()[0]
    since = max(0, (newest or 0) - PULL_OVERLAP)
    st["tried"] = now
    try:
        import httpx
        r = httpx.get(url, params={"since": since}, timeout=6,
                      headers={TOKEN_HEADER: key, "User-Agent": "Birdbrain guests pull"})
        r.raise_for_status()
        stored = merge(r.json().get("rows") or [])
        st.update(ok=now, error="")
        log.info("guests: pulled %d count rows", stored)
    except Exception as e:  # network, HTTP or JSON trouble: say so on the page, try again later
        st["error"] = f"{type(e).__name__}: {e}"[:200]
        log.warning("guests: pull failed: %s", st["error"])
    with closing(connect()) as conn, conn:
        _put_state(conn, "pull", st)


# ---- the page ------------------------------------------------------------------------------------------------------

def period_bounds(key: str, now: datetime) -> dict:
    """The period's start and end (local, naive), its chart buckets [(start, label)], the bucket length and the
    previous period of the same length. Today goes by hour, 7 and 30 days by day, the year by week."""
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    if key == "today":
        start, step, n, fmt = midnight, timedelta(hours=1), 24, "hour"
        back = timedelta(days=1)
    elif key == "year":
        start, step, n, fmt = midnight - timedelta(days=7 * 52 - 1), timedelta(days=7), 52, "week"
        back = timedelta(days=7 * 52)
    else:
        days = 30 if key == "30d" else 7
        start, step, n, fmt = midnight - timedelta(days=days - 1), timedelta(days=1), days, "day"
        back = timedelta(days=days)
    buckets = []
    for i in range(n):
        b = start + i * step
        if fmt == "hour":
            label = f"{b.hour % 12 or 12}{'a' if b.hour < 12 else 'p'}"
        else:
            label = f"{b:%b} {b.day}"
        buckets.append((b, label))
    return {"key": key, "start": start, "end": now, "step": step, "buckets": buckets, "unit": fmt,
            "prev_start": start - back, "prev_end": now - back}


def bucket_values(buckets, step: timedelta, slots: dict[int, int]) -> list[int]:
    """Sum {slot: n} into the buckets (each [start, start + step)), slots read as local time."""
    edges = [b.timestamp() for b, _ in buckets]
    end = (buckets[-1][0] + step).timestamp()
    out = [0] * len(buckets)
    for slot, n in slots.items():
        if slot < edges[0] or slot >= end:
            continue
        lo, hi = 0, len(edges) - 1
        while lo < hi:  # the last edge at or before the slot
            mid = (lo + hi + 1) // 2
            if edges[mid] <= slot:
                lo = mid
            else:
                hi = mid - 1
        out[lo] += n
    return out


def change(now_n: int, prev_n: int) -> int | None:
    """Percent change from the previous period, None when there was nothing then to compare with."""
    if not prev_n:
        return None
    return round((now_n - prev_n) * 100 / prev_n)


def flag(code: str) -> str:
    """The flag emoji for a two-letter country code (a globe for unknown)."""
    if not re.fullmatch(r"[A-Z]{2}", code) or code == "XX":
        return "\U0001F310"
    return "".join(chr(0x1F1E6 + ord(ch) - 65) for ch in code)


_world: dict = {}
WORLD_FILE = Path(__file__).parent / "web" / "world-110m.json"
# Places too small for the 110m map, so the tiles still name them.
SMALL_NAMES = {
    "AD": "Andorra", "AG": "Antigua and Barbuda", "AW": "Aruba", "BB": "Barbados", "BH": "Bahrain", "BM": "Bermuda",
    "CV": "Cape Verde", "CW": "Curacao", "DM": "Dominica", "FO": "Faroe Islands", "GD": "Grenada", "GF": "French Guiana",
    "GG": "Guernsey", "GI": "Gibraltar", "GP": "Guadeloupe", "GU": "Guam", "HK": "Hong Kong", "IM": "Isle of Man",
    "JE": "Jersey", "KY": "Cayman Islands", "LC": "Saint Lucia", "LI": "Liechtenstein", "MC": "Monaco", "MO": "Macau",
    "MQ": "Martinique", "MT": "Malta", "MU": "Mauritius", "MV": "Maldives", "PF": "French Polynesia", "RE": "Reunion",
    "SC": "Seychelles", "SG": "Singapore", "SM": "San Marino", "T1": "Tor network", "TO": "Tonga", "VA": "Vatican City",
    "VI": "U.S. Virgin Islands", "WS": "Samoa", "XX": "Unknown",
}


def world() -> dict:
    if not _world:
        try:
            _world.update(json.loads(WORLD_FILE.read_text()))
        except (OSError, ValueError) as e:
            log.warning("guests: no world map: %s", e)
    return _world


def country_name(code: str) -> str:
    if code == "US":
        return "United States"
    return world().get("names", {}).get(code) or SMALL_NAMES.get(code) or code


MAP_STEPS = 5


def map_step(n: int, top: int) -> int:
    """Shade 1..MAP_STEPS for a count of at least 1, on a log scale: a few countries hold most visitors."""
    if top <= 1:
        return MAP_STEPS
    return min(int(math.log(n) / math.log(top) * MAP_STEPS), MAP_STEPS - 1) + 1


def world_map(by_country: dict[str, int]) -> dict | None:
    """The map's paths, each with its shade (0 = nobody) and tip, and the legend's steps."""
    w = world()
    if not w:
        return None
    top = max(by_country.values(), default=1)
    paths = []
    for code, d in w["countries"].items():
        n = by_country.get(code, 0)
        tip = f"{country_name(code)}: {n:,} visitor{'s' if n != 1 else ''}" if n else ""
        paths.append({"code": code, "d": d, "step": map_step(n, top) if n else 0, "tip": tip})
    legend = sorted({max(math.ceil(top ** (i / MAP_STEPS)), 1) for i in range(MAP_STEPS)})
    return {"w": w["w"], "h": w["h"], "paths": paths, "legend": [(n, map_step(n, top)) for n in legend]}


def _sums(conn, start: datetime, end: datetime) -> dict[str, int]:
    return dict(conn.execute("SELECT metric, SUM(n) FROM counts WHERE slot >= ? AND slot < ? GROUP BY metric",
                             (int(start.timestamp()), int(end.timestamp()) + 1)))


def _ranked(sums: dict, prefix: str, people_prefix: str, labels: dict) -> list[dict]:
    rows = [{"name": m[len(prefix):], "label": labels[m[len(prefix):]], "n": n,
             "people": sums.get(people_prefix + m[len(prefix):], 0)}
            for m, n in sums.items() if m.startswith(prefix) and m[len(prefix):] in labels and n]
    rows.sort(key=lambda r: (-r["n"], r["label"]))
    top = rows[0]["n"] if rows else 1
    for r in rows:
        r["frac"] = r["n"] / top
    return rows


def report(key: str, now: datetime) -> dict:
    """Everything the Guests page shows for one period."""
    key = key if key in dict(PERIODS) else "7d"
    p = period_bounds(key, now)
    with closing(connect()) as conn:
        cur, prev = _sums(conn, p["start"], p["end"]), _sums(conn, p["prev_start"], p["prev_end"])
        lo, hi = int(p["start"].timestamp()), int((p["buckets"][-1][0] + p["step"]).timestamp())
        visitors_by_slot = dict(conn.execute("SELECT slot, n FROM counts WHERE metric = 'visitors' AND slot >= ? "
                                             "AND slot < ?", (lo, hi)))
        any_counts = conn.execute("SELECT 1 FROM counts LIMIT 1").fetchone() is not None
        first = conn.execute("SELECT MIN(slot) FROM counts").fetchone()[0]
        pulled = _get_state(conn, "pull")

    def views(s: dict) -> int:
        return sum(n for m, n in s.items() if m.startswith("page:") and m[5:] in PAGE_LABELS)

    # No change shown against a previous period that started before counting did: half empty, it would say +1000%.
    prev_whole = first is not None and first <= p["prev_start"].timestamp()
    words = PERIOD_WORDS[key]
    prev_words = {"today": "yesterday at this time", "7d": "the 7 days before", "30d": "the 30 days before",
                  "year": "the year before"}[key]
    head = []
    for name, label, a, b, tip in (
        ("visitors", "Visitors", cur.get("visitors", 0), prev.get("visitors", 0),
         "People, each counted once a day (UTC) however many pages they open."),
        ("views", "Page views", views(cur), views(prev), "Every page opened, reloads included."),
        ("returning", "Returning", cur.get("returning", 0), prev.get("returning", 0),
         "Visitors whose browser had been here on an earlier day."),
    ):
        head.append({"key": name, "label": label, "n": a, "prev": b, "change": change(a, b) if prev_whole else None,
                     "tip": f"{tip} {words.capitalize()}: {a:,}; {prev_words}: {b:,}."})
    countries = sorted(((c[8:], n) for c, n in cur.items() if c.startswith("country:") and n),
                       key=lambda cn: (cn[0] == "XX", -cn[1], cn[0]))
    top = max((n for _, n in countries), default=1)
    tiles = [{"code": c, "name": country_name(c), "flag": flag(c), "n": n, "frac": n / top} for c, n in countries]
    known = [t for t in tiles if t["code"] != "XX"]
    return {
        "period": key, "periods": PERIODS, "words": words, "head": head,
        "chart": {"labels": [label for _, label in p["buckets"]],
                  # Hours still to come today are gaps, not zeros.
                  "data": [None if b > now else v for (b, _), v in
                           zip(p["buckets"], bucket_values(p["buckets"], p["step"], visitors_by_slot))],
                  "unit": {"hour": "per hour", "day": "per day", "week": "per week"}[p["unit"]]},
        "pages": _ranked(cur, "page:", "page_people:", PAGE_LABELS),
        "controls": _ranked(cur, "click:", "click_people:", CONTROLS),
        "countries": tiles, "known_countries": len(known), "unknown": len(tiles) - len(known),
        "map": world_map({t["code"]: t["n"] for t in known}) if known else None,
        "any_counts": any_counts, "counting_since": datetime.fromtimestamp(first) if first else None,
        "pulled_at": datetime.fromtimestamp(pulled["ok"]) if pulled.get("ok") else None,
        "pull_error": pulled.get("error", ""), "source_set": bool(source_url() and token()),
    }
