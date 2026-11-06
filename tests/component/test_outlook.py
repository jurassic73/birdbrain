"""Today at the bath (birdbrain.outlook): the quick-read box at the top of the Weather & visits page."""

from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from birdbrain import clock, db, outlook
from birdbrain.web import app as webapp


NOW = datetime(2026, 11, 2, 14, 30)


TODAY = "2026-11-02"


PAST = ["2026-10-29", "2026-10-30", "2026-10-31", "2026-11-01"]


def hour_row(hour: str, precip: float = 0.0, temp: float = 5.0, wind: float = 5.0) -> dict:
    return {"hour": hour, "temp_c": temp, "precip_mm": precip, "rain_mm": precip, "cloud_pct": 90.0, "wind_kmh": wind,
            "humidity_pct": 90.0, "code": 61 if precip else 3, "is_day": 1 if "07" <= hour[11:13] < "17" else 0,
            "fetched_at": "s"}


def day_row(day: str, hours: list[dict], code: int | None = None) -> dict:
    temps = [h["temp_c"] for h in hours]
    precip = sum(h["precip_mm"] for h in hours)
    return {"day": day, "temp_max_c": max(temps), "temp_min_c": min(temps), "precip_mm": precip,
            "precip_hours": float(sum(1 for h in hours if h["precip_mm"])), "cloud_pct": 90.0, "humidity_pct": 90.0,
            "wind_kmh": sum(h["wind_kmh"] for h in hours) / len(hours), "wind_max_kmh": max(h["wind_kmh"] for h in hours),
            "code": code if code is not None else (61 if precip else 3),
            "sunrise": f"{day}T07:00", "sunset": f"{day}T17:00", "final": 0, "fetched_at": "s"}


def make_yard(conn, add_visit, wet_days=("2026-10-30", "2026-11-01"), today=None, today_code=None):
    """Four past days (wet ones rain all day) and today up to 14:00, all at 5°C unless `today` says otherwise:
    a function of the hour (0 to 14) giving hour_row keyword arguments. Dry daylight hours get a Dark-eyed Junco and
    an American Robin, wet ones a Song Sparrow. Today has no visits. The first photo is a false trigger at 06:30
    on Oct 29, so whole daylight hours run 07:00 to 16:00 each day."""
    days, hours = [], []
    for d in PAST:
        mine = [hour_row(f"{d}T{h:02d}:00", precip=1.0 if d in wet_days else 0.0) for h in range(24)]
        hours += mine
        days.append(day_row(d, mine))
        for h in range(7, 17):
            if d in wet_days:
                add_visit(captured_at=f"{d}T{h:02d}:20:00", species="Song Sparrow")
            else:
                add_visit(captured_at=f"{d}T{h:02d}:10:00", species="Dark-eyed Junco")
                add_visit(captured_at=f"{d}T{h:02d}:40:00", species="American Robin")
    mine = [hour_row(f"{TODAY}T{h:02d}:00", **(today(h) if today else {})) for h in range(15)]
    hours += mine
    days.append(day_row(TODAY, mine, today_code))
    db.save_weather(conn, days, hours)
    add_visit(captured_at="2026-10-29T06:30:00", visitor="none", species=None)
    return conn


def run(conn, cards=None):
    report = db.weather_report(conn, now=NOW)
    return outlook.outlook(conn, report, cards or {}, NOW)


def text(line: dict) -> str:
    return "".join(p if isinstance(p, str) else p["sp"] for p in line["parts"])


def lines(out: dict) -> dict[str, dict]:
    return {line["key"]: line for line in out["lines"]}


def raining_now(h: int) -> dict:
    return {"precip": 1.0 if h >= 12 else 0.0}


def test_rainy_day_with_enough_data_uses_this_baths_numbers(conn, add_visit):
    out = run(make_yard(conn, add_visit, today=raining_now))
    t = out["weather"]
    assert t["kind"] == "rain" and t["now"] and t["temp_c"] == 5.0 and t["rain_mm"] == 3.0 and t["since"] == datetime(2026, 11, 2, 14)
    v = lines(out)["visitors"]
    assert v["yard"] and v["cond"] == "rain"
    # Rainy hours: 20 sparrows in 20 past wet hours plus today's 3 (no visits); dry: 40 in 20 plus today's 4.
    assert text(v) == "Rainy today: expect fewer visitors. Rainy hours here have had 48% fewer than dry ones."
    assert "0.9 visitors an hour (20 in 23 daylight hours)" in v["tip"] and "back off" not in v["tip"]
    assert "wait longer between photos" in v["tip"]


def test_rainy_day_with_thin_data_hedges_and_claims_no_numbers(conn, add_visit):
    out = run(make_yard(conn, add_visit, wet_days=(), today=raining_now))
    v = lines(out)["visitors"]
    assert not v["yard"]
    assert text(v) == "Rainy today: birds often lie low in steady rain, so expect fewer visitors."
    assert "%" not in text(v) and v["tip"].startswith("A general rule, not this bath's numbers: only 3 such")


def test_more_visitors_in_rain_here_says_so(conn, add_visit):
    """When this bath's numbers go against the general rule, the line follows the numbers."""
    make_yard(conn, add_visit, today=raining_now)
    for d in ("2026-10-30", "2026-11-01"):
        for h in range(7, 17):
            for m in (30, 45, 50):
                add_visit(captured_at=f"{d}T{h:02d}:{m}:00", species="Song Sparrow")
    v = lines(run(conn))["visitors"]
    assert text(v).startswith("Rainy today: expect more visitors. Rainy hours here have had ")
    assert "% more than dry ones." in text(v)


def test_hot_day(conn, add_visit):
    out = run(make_yard(conn, add_visit, today=lambda h: {"temp": 30.0}))
    v = lines(out)["visitors"]
    assert v["cond"] == "hot" and not v["yard"]  # 7 warm hours, 10 needed
    assert text(v) == "Hot today: birds drink and bathe more in the heat, so the bath may be busier."
    assert "warm means 12°C and up" in v["tip"]


def test_freezing_day_compares_cold_hours_once_there_are_enough(conn, add_visit):
    make_yard(conn, add_visit, today=lambda h: {"temp": -2.0 if h < 9 else 4.0})
    # Some warmer past hours to compare against: Oct 31 afternoon at 14°C.
    conn.execute("UPDATE weather_hours SET temp_c = 14 WHERE hour >= '2026-10-31T06:00' AND hour < '2026-11-01T00:00'")
    conn.commit()
    v = lines(run(conn))["visitors"]
    assert v["cond"] == "freezing" and v["yard"]
    assert text(v).startswith("Freezing today: ") and "Cold hours here have had" in text(v)
    assert "than warmer ones." in text(v) and "Cold means below 8°C" in v["tip"]


def test_freezing_day_without_enough_warm_hours_is_general(conn, add_visit):
    out = run(make_yard(conn, add_visit, today=lambda h: {"temp": -2.0}))
    v = lines(out)["visitors"]
    assert v["cond"] == "freezing" and not v["yard"]
    assert text(v) == "Freezing today: when other water freezes, an open bath can draw more birds than usual."


def test_windy_and_after_rain_and_chilly(conn, add_visit):
    make_yard(conn, add_visit, today=lambda h: {"wind": 25.0})
    assert lines(run(conn))["visitors"]["cond"] == "windy"
    conn.execute("UPDATE weather_hours SET wind_kmh = 5, precip_mm = CASE WHEN hour < '2026-11-02T05:00' "
                 "AND hour >= '2026-11-02T00:00' THEN 1 ELSE precip_mm END")
    conn.execute(f"UPDATE weather_days SET precip_mm = 5, wind_kmh = 5 WHERE day = '{TODAY}'")
    conn.commit()
    v = lines(run(conn))["visitors"]
    assert v["cond"] == "after_rain" and text(v).startswith("Rain earlier, drier now: expect more visitors. Dry hours")
    conn.execute(f"UPDATE weather_days SET precip_mm = 0 WHERE day = '{TODAY}'")
    conn.commit()
    assert lines(run(conn))["visitors"]["cond"] == "chilly"


def test_stale_weather_is_not_now(conn, add_visit):
    make_yard(conn, add_visit)
    conn.execute(f"DELETE FROM weather_hours WHERE hour > '{TODAY}T10:00'")
    conn.commit()
    t = run(conn)["weather"]
    assert not t["now"] and t["temp_c"] is None and t["high_c"] == 5.0


def test_day_outlook_comes_from_the_days_code(conn, add_visit):
    t = run(make_yard(conn, add_visit, today_code=63))["weather"]
    assert t["words"] == "Overcast" and t["day_words"] == "Rain"


def test_no_weather_saved(conn, add_visit):
    add_visit(captured_at="2026-11-02T09:10:00", species="Dark-eyed Junco")
    add_visit(captured_at="2026-11-01T09:10:00", species="Bushtit")
    add_visit(captured_at="2026-11-01T09:40:00", species="Dark-eyed Junco")
    out = run(conn)
    assert out["weather"] is None and [line["key"] for line in out["lines"]] == ["regulars"]
    r = lines(out)["regulars"]
    assert text(r) == "Most seen this week: Dark-eyed Junco and Bushtit." and "No weather saved" in r["tip"]


def test_no_visits(conn):
    assert run(conn) == {"weather": None, "lines": []}


CARDS = {
    "Dark-eyed Junco": {"expected_months": [1, 2, 3, 10, 11, 12], "status": "Winter visitor", "is_bird": True,
                        "local_counts": [50, 40, 30, 5, 1, 1, 1, 1, 2, 30, 60, 70]},
    "Song Sparrow": {"expected_months": list(range(1, 13)), "status": "Year-round resident", "is_bird": True,
                     "local_counts": [20] * 12},
    "American Robin": {"expected_months": list(range(1, 13)), "status": "Year-round resident", "is_bird": True,
                       "local_counts": [30] * 12},
    "Rufous Hummingbird": {"expected_months": [4, 5, 6, 7], "status": "Summer visitor", "is_bird": True,
                           "local_counts": [0, 0, 1, 20, 30, 25, 10, 1, 0, 0, 0, 0]},
    "Varied Thrush": {"expected_months": [1, 2, 3, 10, 11, 12], "status": "Winter visitor", "is_bird": True,
                      "local_counts": [10, 10, 8, 2, 0, 0, 0, 0, 1, 8, 12, 12]},
    "Eastern Gray Squirrel": {"expected_months": list(range(1, 13)), "is_bird": False, "local_counts": [9] * 12},
    "Bushtit": {"name": "Bushtit", "links": []},  # a card without local data
}


def test_season_line_with_cards(conn, add_visit):
    s = lines(run(make_yard(conn, add_visit), CARDS))["season"]
    # Birds seen here with at least an average month now, busiest first (ties: visits, then name); then one that
    # could show up. The hummingbird is out of season, the squirrel isn't a bird, the bushtit has no local data.
    assert text(s) == ("It's November, a good month around here for Dark-eyed Junco, American Robin and Song Sparrow. "
                       "Could show up: Varied Thrush.")
    assert "Dark-eyed Junco: winter visitor, 147% more reports than in an average month, 20 visits here" in s["tip"]
    assert "Varied Thrush: winter visitor, 129% more reports than in an average month, not seen here yet" in s["tip"]
    assert {"sp": "Varied Thrush"} in s["parts"]


def test_season_line_when_no_bird_here_has_a_good_month(conn, add_visit):
    cards = {n: {**c, "local_counts": [40] * 10 + [5, 40]} for n, c in CARDS.items() if "local_counts" in c}
    s = lines(run(make_yard(conn, add_visit), cards))["season"]
    assert text(s) == "It's November. In season around here now: American Robin, Dark-eyed Junco and Song Sparrow."
    assert "87% fewer reports than in an average month" in s["tip"]


def test_season_line_without_cards(conn, add_visit):
    out = run(make_yard(conn, add_visit))
    assert "season" not in lines(out)
    assert outlook.season_line({"Bushtit": CARDS["Bushtit"]}, {"Bushtit": 3}, NOW) is None


def test_regulars_learned_from_weather_like_today(conn, add_visit):
    r = lines(run(make_yard(conn, add_visit, today=raining_now)))["regulars"]
    assert r["learned"]
    assert text(r) == "In weather like today's (rainy, below 8°C), the regulars here are Song Sparrow."
    assert "20 visits in 23 daylight hours like this (rainy, below 8°C) since Oct 29" in r["tip"]
    dry = lines(run(conn))["regulars"]  # same data, judged as a dry day: juncos and robins
    assert dry["learned"]


def test_regulars_fall_back_to_temperature_free_match_then_to_the_last_week(conn, add_visit):
    # Today is 10°C (the mild band): no past hour matches on temperature, so it matches on dry alone.
    r = lines(run(make_yard(conn, add_visit, today=lambda h: {"temp": 10.0})))["regulars"]
    assert text(r) == "In weather like today's (dry), the regulars here are American Robin and Dark-eyed Junco."
    # Raining and 10°C: rainy alone, 20 sparrows.
    conn.execute(f"UPDATE weather_hours SET precip_mm = 1 WHERE hour >= '{TODAY}T12:00'")
    conn.commit()
    assert text(lines(run(conn))["regulars"]) == "In weather like today's (rainy), the regulars here are Song Sparrow."
    # Too few rainy visits even without the temperature: the last week's most seen instead.
    conn.execute("DELETE FROM visits WHERE species = 'Song Sparrow' AND captured_at > '2026-10-30T12'")
    conn.commit()
    r = lines(run(conn))["regulars"]
    assert not r["learned"] and "(5 of 15 needed)" in r["tip"]


def test_regulars_without_enough_data(conn, add_visit):
    r = lines(run(make_yard(conn, add_visit, wet_days=(), today=raining_now)))["regulars"]
    assert not r["learned"]
    assert text(r) == "Most seen this week: American Robin and Dark-eyed Junco."
    assert r["tip"].startswith("Not enough visits in weather like today's yet (0 of 15 needed)")


@pytest.fixture
def client(conn, monkeypatch):
    monkeypatch.setattr(clock, "now", lambda: NOW)
    with TestClient(webapp.app) as c:
        yield c


def test_page_shows_the_box(client, conn, add_visit):
    make_yard(conn, add_visit, today=raining_now)
    db.save_species_info(conn, "Dark-eyed Junco", CARDS["Dark-eyed Junco"], "2026-11-01T00:00:00")
    html = client.get("/weather").text
    assert "Today at the bath" in html and "Mon Nov 2" in html
    assert "cover midnight to 2 PM" in html and "Rainy today: expect fewer visitors." in html and "41° now" in html and "3 mm of rain" in html
    assert "a good month around here for " in html and 'data-species="Dark-eyed Junco"' in html
    assert "In weather like today&#39;s (rainy, below 46°)" in html
    assert html.index("Today at the bath") < html.index("wx-diary")


def test_page_box_without_weather_and_without_visits(client, add_visit):
    assert "Today at the bath" not in client.get("/weather").text
    add_visit(captured_at="2026-11-02T09:10:00", species="Dark-eyed Junco")
    html = client.get("/weather").text
    assert "Today at the bath" in html and "Most seen this week: " in html and "wx-now-head" not in html
