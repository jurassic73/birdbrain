"""Local weather for the site, hour by hour, from Open-Meteo (free, no key), so visits can be compared with the weather.

Needs SITE_LAT and SITE_LON in .env (without them nothing happens). Recent days (and today) come from the forecast API's
`past_days`, older ones from the archive API. Hours are in the site's local time (`timezone=auto`), the same naive
local-time strings as visits.captured_at. Everything is stored metric in weather_hours and weather_days (see db.py).

The watcher runs `start_background()`: one refresh shortly after start (which backfills to the first visit) and then
hourly. A failed fetch is logged, never raised, and not retried for RETRY_AFTER.
    uv run python -m birdbrain.weather [--backfill] [--days N]
Weather data by Open-Meteo.com (CC BY 4.0), https://open-meteo.com
"""

import argparse
import logging
import sqlite3
import threading
import time
from datetime import date, datetime, timedelta

import httpx

from . import config, db

log = logging.getLogger("birdbrain.weather")

HEADERS = {"User-Agent": "Birdbrain/0.1 (personal backyard bird camera)", "Accept-Encoding": "gzip"}
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
HOURLY = "temperature_2m,precipitation,rain,cloud_cover,wind_speed_10m,weather_code,is_day"
DAILY = "temperature_2m_max,temperature_2m_min,precipitation_sum,precipitation_hours,weather_code,sunrise,sunset"
FORECAST_DAYS = 90      # the forecast API reaches back 92 days; older days come from the archive
FINAL_AFTER = 2         # a day is final once it is this many days old (yesterday is still refreshed)
RETRY_AFTER = 3600      # seconds to wait after a failed fetch
EVERY = 3600            # the watcher's refresh interval
FIRST_DELAY = 45        # the watcher's first refresh, so the camera loop starts first

_failed_at = 0.0        # time.monotonic() of the last failure, 0 when none
_lock = threading.Lock()


def located() -> bool:
    return config.WEATHER and config.SITE_LAT is not None and config.SITE_LON is not None


# WMO weather interpretation codes, as documented by Open-Meteo: code -> (kind, words).
_CODES = {
    0: ("clear", "Clear sky"), 1: ("clear", "Mainly clear"), 2: ("partly", "Partly cloudy"), 3: ("cloudy", "Overcast"),
    45: ("fog", "Fog"), 48: ("fog", "Freezing fog"),
    51: ("drizzle", "Light drizzle"), 53: ("drizzle", "Drizzle"), 55: ("drizzle", "Heavy drizzle"),
    56: ("drizzle", "Light freezing drizzle"), 57: ("drizzle", "Freezing drizzle"),
    61: ("rain", "Light rain"), 63: ("rain", "Rain"), 65: ("rain", "Heavy rain"),
    66: ("rain", "Light freezing rain"), 67: ("rain", "Freezing rain"),
    71: ("snow", "Light snow"), 73: ("snow", "Snow"), 75: ("snow", "Heavy snow"), 77: ("snow", "Snow grains"),
    80: ("rain", "Light showers"), 81: ("rain", "Showers"), 82: ("rain", "Heavy showers"),
    85: ("snow", "Light snow showers"), 86: ("snow", "Heavy snow showers"),
    95: ("storm", "Thunderstorm"), 96: ("storm", "Thunderstorm with hail"), 99: ("storm", "Thunderstorm with heavy hail"),
}


def describe(code: int | None) -> tuple[str, str]:
    """(kind, words) for a WMO weather code. kind is one of clear, partly, cloudy, fog, drizzle, rain, snow, storm."""
    if code is None or int(code) not in _CODES:
        return "cloudy", "Unknown"
    return _CODES[int(code)]


def _get(url: str, **params) -> dict:
    r = httpx.get(url, params=params, headers=HEADERS, timeout=15)
    r.raise_for_status()
    return r.json()


def _num(values: list | None, i: int) -> float | None:
    try:
        v = values[i]
    except (TypeError, IndexError):
        return None
    return None if v is None else float(v)


def _int(values: list | None, i: int) -> int | None:
    v = _num(values, i)
    return None if v is None else int(v)


def parse(data: dict, fetched_at: str) -> tuple[list[dict], list[dict]]:
    """(days, hours) rows, as dicts keyed by column name, from one Open-Meteo response. Rows with nothing in them
    (the archive's latest hours are null) are left out. `final` is 0; refresh() sets it."""
    h = data.get("hourly") or {}
    hours = []
    for i, t in enumerate(h.get("time") or []):
        row = {"hour": t, "temp_c": _num(h.get("temperature_2m"), i), "precip_mm": _num(h.get("precipitation"), i),
               "rain_mm": _num(h.get("rain"), i), "cloud_pct": _num(h.get("cloud_cover"), i),
               "wind_kmh": _num(h.get("wind_speed_10m"), i), "code": _int(h.get("weather_code"), i),
               "is_day": _int(h.get("is_day"), i), "fetched_at": fetched_at}
        if row["temp_c"] is not None or row["precip_mm"] is not None or row["code"] is not None:
            hours.append(row)
    clouds: dict[str, list[float]] = {}
    for r in hours:
        if r["cloud_pct"] is not None:
            clouds.setdefault(r["hour"][:10], []).append(r["cloud_pct"])
    d = data.get("daily") or {}
    days = []
    for i, day in enumerate(d.get("time") or []):
        cloud = clouds.get(day)
        row = {"day": day, "temp_max_c": _num(d.get("temperature_2m_max"), i),
               "temp_min_c": _num(d.get("temperature_2m_min"), i), "precip_mm": _num(d.get("precipitation_sum"), i),
               "precip_hours": _num(d.get("precipitation_hours"), i),
               "cloud_pct": round(sum(cloud) / len(cloud), 1) if cloud else None, "code": _int(d.get("weather_code"), i),
               "sunrise": (d.get("sunrise") or [None] * (i + 1))[i], "sunset": (d.get("sunset") or [None] * (i + 1))[i],
               "final": 0, "fetched_at": fetched_at}
        if row["temp_max_c"] is not None or row["precip_mm"] is not None or row["code"] is not None:
            days.append(row)
    return days, hours


def _partial_today(day: dict, hours: list[dict]) -> None:
    """Today's daily figures from the API include the forecast for the rest of the day; use the hours so far."""
    mine = [r for r in hours if r["hour"][:10] == day["day"]]
    temps = [r["temp_c"] for r in mine if r["temp_c"] is not None]
    precip = [r["precip_mm"] for r in mine if r["precip_mm"] is not None]
    if temps:
        day["temp_max_c"], day["temp_min_c"] = max(temps), min(temps)
    if precip:
        day["precip_mm"] = round(sum(precip), 2)
        day["precip_hours"] = float(sum(1 for p in precip if p > 0))


def refresh(conn: sqlite3.Connection, now: datetime | None = None, days: int | None = None, force: bool = False) -> dict | None:
    """Fetch the weather days we are missing, from the day of the first visit through today. Past days are final and
    never fetched again; today and yesterday are refreshed. `days` limits it to the last N days (refetching them),
    `force` refetches everything in range and ignores the wait after a failure. Returns what it saved
    ({"days", "hours", "first", "last"}), or None when it did nothing or failed. Never raises."""
    global _failed_at
    try:
        if not located():
            return None
        now = now or datetime.now()
        today = now.date()
        first = db.first_visit_day(conn)
        if first is None:
            return None
        if days is not None:
            first = max(first, (today - timedelta(days=days - 1)).isoformat())
        if first > today.isoformat():
            return None
        if _failed_at and time.monotonic() - _failed_at < RETRY_AFTER and not force:
            return None
        if days is not None or force:
            need = [(date.fromisoformat(first) + timedelta(days=i)).isoformat()
                    for i in range((today - date.fromisoformat(first)).days + 1)]
        else:
            need = db.weather_missing_days(conn, first, today.isoformat())
        if not need:
            return None
        stamp = now.replace(microsecond=0).isoformat()
        got_days: list[dict] = []
        got_hours: list[dict] = []
        old = [d for d in need if (today - date.fromisoformat(d)).days > FORECAST_DAYS]
        recent = [d for d in need if d not in old]
        place = {"latitude": config.SITE_LAT, "longitude": config.SITE_LON, "timezone": "auto", "hourly": HOURLY, "daily": DAILY}
        if old:
            data = _get(ARCHIVE_URL, start_date=old[0], end_date=old[-1], **place)
            d, h = parse(data, stamp)
            got_days += d
            got_hours += h
        if recent:
            data = _get(FORECAST_URL, past_days=(today - date.fromisoformat(recent[0])).days, forecast_days=1, **place)
            d, h = parse(data, stamp)
            got_days += d
            got_hours += h
        want = set(need)
        now_hour = now.strftime("%Y-%m-%dT%H:00")
        got_hours = [r for r in got_hours if r["hour"][:10] in want and r["hour"] <= now_hour]
        got_days = [r for r in got_days if r["day"] in want]
        for r in got_days:
            if r["day"] == today.isoformat():
                _partial_today(r, got_hours)
            r["final"] = 1 if (today - date.fromisoformat(r["day"])).days >= FINAL_AFTER else 0
        db.save_weather(conn, got_days, got_hours)
        _failed_at = 0.0
        if not got_days:
            return None
        return {"days": len(got_days), "hours": len(got_hours), "first": got_days[0]["day"], "last": got_days[-1]["day"]}
    except Exception as e:
        _failed_at = time.monotonic()
        log.warning("weather fetch failed (%s: %s); trying again in %d min", type(e).__name__, e, RETRY_AFTER // 60)
        return None


def _loop() -> None:
    time.sleep(FIRST_DELAY)
    while True:
        try:
            with db.session() as conn:
                done = refresh(conn)
            if done:
                log.info("weather: %d days, %d hours (%s..%s)", done["days"], done["hours"], done["first"], done["last"])
        except Exception:
            log.exception("weather refresh crashed")
        time.sleep(EVERY)


def start_background() -> threading.Thread | None:
    """Refresh the weather shortly from now and then hourly, in a daemon thread. None when weather isn't set up."""
    if not located():
        return None
    t = threading.Thread(target=_loop, name="weather", daemon=True)
    t.start()
    return t


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch local weather from Open-Meteo into the database.")
    parser.add_argument("--backfill", action="store_true", help="refetch every day from the first visit, even final ones")
    parser.add_argument("--days", type=int, help="refetch just the last N days")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not located():
        print("weather is off: set SITE_LAT and SITE_LON in .env (and don't set WEATHER=off)")
        return
    with db.session() as conn:
        db.init_db(conn)
        done = refresh(conn, days=args.days, force=args.backfill or args.days is not None)
    if done:
        print(f"fetched {done['days']} days, {done['hours']} hours ({done['first']}..{done['last']})")
    else:
        print("nothing to fetch (no visits yet, everything is up to date, or the fetch failed: see the log above)")


if __name__ == "__main__":
    main()
