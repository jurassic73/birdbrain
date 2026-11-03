"""Today at the bath: a few plain lines at the top of the Weather & visits page, so a casual visitor gets the gist in
seconds. Rules over data already stored, worked out on each page load: no Claude call and no network, so the
read-only public mirror runs it too.

1. today's weather (the hours saved so far; the Pi stores no forecast hours),
2. what that means for visitors: this bath's own rainy/dry, windy/calm, cold/warm comparison when the
   weather report has enough hours behind it, else an honestly hedged general line,
3. what's in season around here this month, from the species cards' local monthly data,
4. the regulars in weather like today's, learned from this bath's visits once there are enough of them
   (db.MIN_LIKE_TODAY), else the most seen of the last week.

Each line is a list of parts: plain strings and {"sp": name} for a species (the template links those to its card),
plus a tip for anything that needs explaining.
"""

from datetime import datetime, timedelta

from . import db, weather

FRESH_HOURS = 3         # the newest saved hour counts as "now" when it is at most this old
RECENT_RAIN_HOURS = 3   # rain in any of the last this many saved hours means it is raining now
SAME = 0.15             # yard rates within this share of each other are "about the same" (as weather_findings)
REGULARS_SHOWN = 3
SEASON_SHOWN = 3
COULD_SHOW_UP = 2
GOOD_MONTH = 1.0        # a species has a good month when it gets at least its average month's local reports
LATELY_DAYS = 7
WET_KINDS = {"drizzle", "rain", "snow", "storm"}


def _celsius(c: float) -> str:
    return f"{round(c)}°C"


def _names(names: list[str]) -> list:
    """Parts for 'A', 'A and B' or 'A, B and C', each name linked."""
    out: list = []
    for i, n in enumerate(names):
        if i:
            out.append(" and " if i == len(names) - 1 else ", ")
        out.append({"sp": n})
    return out


def _merge(key: str, groups: list[dict]) -> dict:
    n, v = sum(g["hours"] for g in groups), sum(g["visitors"] for g in groups)
    return {"key": key, "hours": n, "visitors": v, "rate": v / n if n else None, "enough": n >= db.MIN_HOURS}


def _group(report: dict, section: str, key: str) -> dict | None:
    return next((g for g in report.get(section) or [] if g["key"] == key), None)


def _band(temp_c: float) -> str:
    lo, hi = db.TEMP_BANDS_C
    return "cold" if temp_c < lo else "mild" if temp_c < hi else "warm"


def _band_words(band: str, temp) -> str:
    lo, hi = db.TEMP_BANDS_C
    return {"cold": f"below {temp(lo)}", "mild": f"{temp(lo)} to {temp(hi)}", "warm": f"{temp(hi)} and up"}[band]


def today_weather(conn, now: datetime) -> dict | None:
    """Today's weather from what's saved: the newest hour (if fresh) and the day so far. None when neither exists."""
    day = now.date().isoformat()
    rows = db.weather_days(conn, day, day)
    hours = [dict(h) for h in db.weather_hours(conn, day, day) if h["hour"] <= now.strftime("%Y-%m-%dT%H:00")]
    latest = hours[-1] if hours else None
    if latest and datetime.fromisoformat(latest["hour"]) < now - timedelta(hours=FRESH_HOURS):
        latest = None
    d = dict(rows[0]) if rows else None
    if not d and not latest:
        return None
    code = latest["code"] if latest and latest["code"] is not None else (d["code"] if d else None)
    kind, words = weather.describe(code)
    recent = hours[-RECENT_RAIN_HOURS:] if latest else []
    return {
        "kind": kind, "words": words, "now": bool(latest), "at": datetime.fromisoformat(latest["hour"]) if latest else None,
        "temp_c": latest["temp_c"] if latest else None,
        "wind_kmh": latest["wind_kmh"] if latest and latest["wind_kmh"] is not None else (d or {}).get("wind_kmh"),
        "high_c": d["temp_max_c"] if d else None, "low_c": d["temp_min_c"] if d else None,
        "rain_mm": (d["precip_mm"] or 0.0) if d else None,
        "day_words": weather.describe(d["code"])[1] if d and d["code"] is not None else None,
        "since": datetime.fromisoformat(hours[-1]["hour"]) if hours else None,  # the newest saved hour
        "raining": kind in WET_KINDS or any((h["precip_mm"] or 0) >= db.RAIN_HOUR_MM for h in recent),
    }


def _condition(t: dict) -> str:
    """The one thing about today's weather that matters most to the bath."""
    temp = t["temp_c"] if t["temp_c"] is not None else t["high_c"]
    if t["kind"] == "storm":
        return "storm"
    if t["raining"]:
        return "snow" if t["kind"] == "snow" else "rain"
    if (t["low_c"] is not None and t["low_c"] <= db.FREEZING_C) or (t["temp_c"] is not None and t["temp_c"] <= db.FREEZING_C):
        return "freezing"
    if t["high_c"] is not None and t["high_c"] >= db.HOT_C:
        return "hot"
    if t["wind_kmh"] is not None and t["wind_kmh"] >= db.WINDY_KMH:
        return "windy"
    if (t["rain_mm"] or 0) >= db.WET_DAY_MM:
        return "after_rain"
    if temp is not None and _band(temp) == "cold":
        return "chilly"
    return "fine"


# condition: (lead, general line when this bath has too few hours to say, which way that general line leans)
LEADS = {
    "storm": ("Stormy today", "birds shelter in storms, so expect a quiet bath until it passes."),
    "rain": ("Rainy today", "birds often lie low in steady rain, so expect fewer visitors."),
    "snow": ("Snowy today", "with food and puddles buried, open water can draw birds in, if the bath isn't frozen."),
    "freezing": ("Freezing today", "when other water freezes, an open bath can draw more birds than usual."),
    "hot": ("Hot today", "birds drink and bathe more in the heat, so the bath may be busier."),
    "windy": ("Windy today", "small birds often keep to shelter in wind, so expect fewer visitors."),
    "after_rain": ("Rain earlier, drier now", "birds often come out to feed and bathe after rain."),
    "chilly": ("Chilly but dry", "birds still need to drink and bathe in the cold, so expect the usual regulars."),
    "fine": ("Dry and calm today", "good bath weather, so the regulars should be about."),
}


def _comparison(cond: str, report: dict, temp) -> tuple[dict, dict, str, str, str] | None:
    """(this, other, what this is, what other is, tip about the groups) from the weather report for the condition."""
    lo, hi = db.TEMP_BANDS_C
    rain, dry = _group(report, "rain", "rain"), _group(report, "rain", "dry")
    if cond in ("storm", "rain", "snow"):
        if rain and dry:
            what = "Rainy hours" if cond != "snow" else "Wet hours (rain or snow)"
            return rain, dry, what, "dry ones", f"at least {db.RAIN_HOUR_MM} mm of rain or snow in the hour"
    elif cond in ("after_rain", "fine"):
        if rain and dry:
            return dry, rain, "Dry hours", "rainy ones", f"rainy means at least {db.RAIN_HOUR_MM} mm of rain or snow in the hour"
    elif cond == "windy":
        calm, windy = _group(report, "wind", "calm"), _group(report, "wind", "windy")
        if calm and windy:
            return windy, calm, "Windy hours", "calm ones", f"windy means {db.WINDY_KMH} km/h of wind or more"
    elif cond in ("freezing", "chilly"):
        cold = _group(report, "temp", "cold")
        rest = [g for g in report.get("temp") or [] if g["key"] != "cold"]
        if cold and rest:
            return cold, _merge("warmer", rest), "Cold hours", "warmer ones", f"cold means below {temp(lo)}"
    elif cond == "hot":
        warm = _group(report, "temp", "warm")
        rest = [g for g in report.get("temp") or [] if g["key"] != "warm"]
        if warm and rest:
            return warm, _merge("cooler", rest), "Warm hours", "cooler ones", f"warm means {temp(hi)} and up"
    return None


def visitors_line(t: dict, report: dict, temp=_celsius) -> dict:
    """What today's weather means for visitors: this bath's own numbers when they're there, else a general line."""
    cond = _condition(t)
    lead, general = LEADS[cond]
    caveat = (" Rain makes the camera wait longer between photos, so wet hours may miss a few visitors."
              if cond in ("storm", "rain", "snow", "after_rain", "fine") else "")
    cmp = _comparison(cond, report, temp)
    if cmp:
        this, other, what, other_what, means = cmp
        if this["enough"] and other["enough"] and other["rate"]:
            change = this["rate"] / other["rate"] - 1
            tip = (f"{what} at this bath: {this['rate']:.1f} visitors an hour ({this['visitors']} in {this['hours']} "
                   f"daylight hours); {other_what}: {other['rate']:.1f} ({other['visitors']} in {other['hours']}). "
                   f"{means[0].upper()}{means[1:]}.{caveat}")
            if abs(change) < SAME:
                text = f"{lead}: about the usual. {what} here have drawn about as many visitors as {other_what}."
            else:
                more = change > 0
                text = (f"{lead}: expect {'more' if more else 'fewer'} visitors. {what} here have had "
                        f"{round(abs(change) * 100)}% {'more' if more else 'fewer'} than {other_what}.")
            return {"key": "visitors", "cond": cond, "yard": True, "parts": [text], "tip": tip}
        few = this if this["hours"] < other["hours"] else other
        tip = (f"A general rule, not this bath's numbers: only {few['hours']} {'such' if few is this else 'comparable'} "
               f"daylight hour{'s' if few['hours'] != 1 else ''} here so far, {db.MIN_HOURS} needed to compare "
               f"({means}).{caveat}")
    else:
        tip = f"A general rule, not this bath's numbers: no hours to compare yet.{caveat}"
    return {"key": "visitors", "cond": cond, "yard": False, "parts": [f"{lead}: {general}"], "tip": tip}


def season_line(cards: dict, yard: dict[str, int], now: datetime) -> dict | None:
    """What's in season around here this month, from the species cards' local monthly reports (iNaturalist).
    Birds this bath has had come first: the ones with a good month now (at least their average month's reports,
    busiest first), else simply those in season, most visits first; then a couple that could show up (in season
    and having a good month, never seen here). `yard` is visits per species at this bath. None without cards that
    have local monthly data."""
    m = now.month
    month = now.strftime("%B")
    birds = {n: c for n, c in cards.items()
             if c and c.get("expected_months") and c.get("local_counts") and c.get("is_bird") is not False}
    now_in = [n for n, c in birds.items() if m in c["expected_months"]]
    if not now_in:
        return None

    def lift(n: str) -> float:
        """This month's local reports against the species' average month (1 = an average month)."""
        counts = birds[n]["local_counts"]
        avg = sum(counts) / 12
        return counts[m - 1] / avg if avg else 0.0

    here = [n for n in now_in if yard.get(n)]
    good = sorted((n for n in here if lift(n) >= GOOD_MONTH), key=lambda n: (-lift(n), -yard[n], n))
    maybe = sorted((n for n in now_in if not yard.get(n) and lift(n) >= GOOD_MONTH),
                   key=lambda n: (-lift(n), n))[:COULD_SHOW_UP]
    if good:
        shown = good[:SEASON_SHOWN]
        parts: list = [f"It's {month}, a good month around here for ", *_names(shown), "."]
    elif here:
        shown = sorted(here, key=lambda n: (-yard[n], n))[:SEASON_SHOWN]
        parts = [f"It's {month}. In season around here now: ", *_names(shown), "."]
    else:
        shown, parts = [], [f"It's {month}."]
    if maybe:
        parts += [" Could show up: ", *_names(maybe), "."]
    if not shown and not maybe:
        return None

    def about(n: str) -> str:
        pct = round((lift(n) - 1) * 100)
        trend = (f"{pct}% more reports than in an average month" if pct > 0 else
                 f"{-pct}% fewer reports than in an average month" if pct < 0 else "an average month's reports")
        seen = f"{yard[n]} visit{'s' if yard[n] != 1 else ''} here" if yard.get(n) else "not seen here yet"
        return f"{n}: {(birds[n].get('status') or 'in season').lower()}, {trend}, {seen}"

    tip = (f"From iNaturalist reports in the area, month by month (in season: {month} has at least a tenth of the "
           f"busiest month's reports). " + "; ".join(about(n) for n in shown + maybe) + ".")
    return {"key": "season", "parts": parts, "tip": tip}


def regulars_line(rows: list, report: dict, t: dict | None, now: datetime, temp=_celsius) -> dict | None:
    """The species seen most at this bath in daylight hours with weather like today's (wet or dry, and the same
    temperature band, else wet or dry alone), once there are db.MIN_LIKE_TODAY such visits; else the most seen
    of the last week. `rows` are db.seen_labels since the first photo. None with no visits at all."""
    first = report.get("first")
    if not first or not rows:
        return None

    def top(sel) -> tuple[list[tuple[str, int]], int]:
        counts: dict[str, int] = {}
        ids = set()
        for r in sel:
            counts[r["label"]] = counts.get(r["label"], 0) + 1
            ids.add(r["id"])
        return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:REGULARS_SHOWN], len(ids)

    counts_tip = lambda best: ", ".join(f"{n} {c}" for n, c in best)
    short = 0
    temp_now = None if not t else t["temp_c"] if t["temp_c"] is not None else t["high_c"]
    if t and report.get("by_hour"):
        wet = t["raining"]
        band = _band(temp_now) if temp_now is not None else None
        tries = ([(True, f"{'rainy' if wet else 'dry'}, {_band_words(band, temp)}")] if band else []) + \
                [(False, "rainy" if wet else "dry")]
        for use_temp, words in tries:
            like = {h["hour"] for h in report["by_hour"]
                    if ((h["rain"] or 0) >= db.RAIN_HOUR_MM) == wet
                    and (not use_temp or (h["temp"] is not None and _band(h["temp"]) == band))}
            best, n = top(r for r in rows if r["captured_at"][:13] + ":00" in like)
            if n >= db.MIN_LIKE_TODAY:
                return {"key": "regulars", "learned": True,
                        "parts": [f"In weather like today's ({words}), the regulars here are ", *_names([b[0] for b in best]), "."],
                        "tip": f"Learned from this bath: {n} visits in {len(like)} daylight hours like this ({words}) "
                               f"since {first.strftime('%b %-d')}. Most seen: {counts_tip(best)}. It sharpens as visits "
                               f"pile up."}
            short = max(short, n)
    since = (now - timedelta(days=LATELY_DAYS)).isoformat()
    best, n = top(r for r in rows if r["captured_at"] >= since)
    if not best:
        return None
    why = (f"Not enough visits in weather like today's yet ({short} of {db.MIN_LIKE_TODAY} needed), so these are "
           if t else "No weather saved for today, so these are ")
    return {"key": "regulars", "learned": False,
            "parts": ["Most seen this week: ", *_names([b[0] for b in best]), "."],
            "tip": f"{why}the last {LATELY_DAYS} days' most frequent visitors: {counts_tip(best)}."}


def outlook(conn, report: dict, cards: dict, now: datetime, temp=_celsius) -> dict:
    """Everything for the box: {"weather": today_weather or None, "lines": [line, ...]}, lines in reading order
    (visitors, season, regulars), each {"key", "parts", "tip"}. `temp` formats a Celsius temperature in the page's
    unit (default '13°C'); `cards` are the cached species cards by name (species_info.cached_cards())."""
    t = today_weather(conn, now) if report.get("has_weather") else None
    lines = []
    if t:
        lines.append(visitors_line(t, report, temp))
    first = report.get("first")
    rows = db.seen_labels(conn, first.date().isoformat(), now.date().isoformat()) if first else []
    yard: dict[str, int] = {}
    for r in rows:
        yard[r["label"]] = yard.get(r["label"], 0) + 1
    season = season_line(cards or {}, yard, now)
    if season:
        lines.append(season)
    regulars = regulars_line(rows, report, t, now, temp)
    if regulars:
        lines.append(regulars)
    return {"weather": t, "lines": lines}
