"""Day-range queries, their indexes and counts that must agree (db.py's speed work).

The seeded database crosses a month boundary (Sep 30 / Oct 1, Oct 31 / Nov 1) and the night the clocks go back
(2026-11-01: 01:00 to 02:00 local happens twice, so two visits share a captured_at an hour apart). "Today" is
Nov 1, 12:30. Every count here was pinned on the date(captured_at) = ? queries before they became range compares on
captured_at itself, and must stay the same."""
import json
import re
from datetime import date, datetime, timedelta

import pytest

from conftest import FROZEN_NOW
from birdbrain import clock, db, watch

DAY = "2026-11-01"
NOW = datetime(2026, 11, 1, 12, 30)
TOKENS = {"input_tokens": 1000, "output_tokens": 100}


def raw(visitor: str, species: str | None = None, **more) -> str:
    calls = {"haiku": TOKENS, "presence": {"score": 0.5, "empty": visitor == "none"},
             "detector": {"animal": visitor != "none", "scenery": {"empty": visitor == "none"}}, **more}
    if species:
        calls["classifier"] = {"top": [{"name": species}], "seconds": 1.0}
    return json.dumps({"calls": calls})


@pytest.fixture
def boundary(conn, add_visit, monkeypatch):
    """Fourteen visits (ids 1 to 14) around the month ends and the fall-back night; now is 2026-11-01 12:30."""
    monkeypatch.setattr(clock, "today", lambda: date(2026, 11, 1))
    monkeypatch.setattr(clock, "now", lambda: NOW)
    add_visit(captured_at="2026-09-30T23:59:59")                                                         # 1
    add_visit(captured_at="2026-10-01T00:00:00", species="Bushtit")                                      # 2
    add_visit(captured_at="2026-10-02T12:00:00", visitor="none", species=None, confidence=None,
              confirmed_at="2026-10-02T18:00:00", photo_deleted_at="2026-11-01T03:00:00")                # 3
    add_visit(captured_at="2026-10-03T00:00:00", species="Song Sparrow")                                 # 4
    add_visit(captured_at="2026-10-25T08:00:00", species="Bushtit", needs_review=1)                      # 5
    add_visit(captured_at="2026-10-31T00:00:00", visitor="none", species=None, confidence=None,
              photo_deleted_at="2026-11-01T03:00:00")                                                    # 6
    add_visit(captured_at="2026-10-31T23:59:59", species="Bushtit", raw_json=raw("bird", "Bushtit"),
              also=json.dumps([{"visitor": "bird", "species": "Song Sparrow", "count": 2}]))             # 7
    add_visit(captured_at="2026-11-01T00:00:00", visitor="squirrel", species=None, raw_json=raw("squirrel"))  # 8
    add_visit(captured_at="2026-11-01T01:30:00", raw_json=raw("bird", "Dark-eyed Junco"))               # 9 (PDT)
    add_visit(captured_at="2026-11-01T01:30:00", raw_json=raw("bird", "Dark-eyed Junco"), needs_review=1)  # 10 (PST)
    add_visit(captured_at="2026-11-01T01:59:59", visitor="none", species=None, confidence=None,
              raw_json=raw("none"))                                                                      # 11
    add_visit(captured_at="2026-11-01T02:00:00", visitor="other", species=None, confidence="low",
              raw_json=raw("other"))                                                                     # 12
    add_visit(captured_at="2026-11-01T09:00:00", species="Bushtit", model="claude-sonnet-5-5", escalated=1,
              raw_json=raw("bird", "Bushtit", sonnet={"input_tokens": 3000, "output_tokens": 300}))      # 13
    add_visit(captured_at="2026-11-01T12:00:00", visitor="none", species=None, confidence=None,
              confirmed_at="2026-11-01T12:10:00", raw_json=raw("none"))                                  # 14
    db.bump(conn, "probe_empty", day="2026-10-31")
    db.bump(conn, "probe_empty", day=DAY)
    db.bump(conn, "probe_empty", day=DAY)
    return conn


def test_date_filters_are_inclusive_days(boundary):
    conn = boundary
    assert db.count_visits(conn, date_from=DAY, date_to=DAY) == 5
    assert db.count_visits(conn, date_from="2026-10-31", date_to="2026-10-31") == 1
    assert db.count_visits(conn, date_from="2026-10-01", date_to="2026-10-31") == 4
    assert db.count_visits(conn, date_to="2026-09-30") == 1
    assert db.count_visits(conn, date_from="2026-11-01") == 5
    assert db.count_visits(conn, date_from="2026-11-02") == 0
    assert db.count_visits(conn, false_triggers=True, date_from=DAY, date_to=DAY) == 2
    assert db.count_visits(conn, false_triggers=True, has_photo=True, unchecked=True) == 1
    assert [r["id"] for r in db.list_visits(conn, date_from="2026-10-31", date_to=DAY)] == [13, 12, 10, 9, 8, 7]
    assert [r["id"] for r in db.list_visits(conn, review=True)] == [5, 10]
    assert [r["id"] for r in db.list_visits(conn, species="Song Sparrow", date_to="2026-10-31")] == [7, 4]


def test_a_date_that_is_not_one_still_filters_rather_than_failing(boundary):
    """A hand-edited URL: no 500, and the same (odd) string compare as before."""
    assert db.count_visits(boundary, date_from="yesterday") == 0
    assert db.count_visits(boundary, date_to="yesterday") == 10
    assert db.count_visits(boundary, date_from="2026-10", date_to="2026-11") == 4


def test_today_summary_at_the_boundaries(boundary):
    s = db.today_summary(boundary)
    assert (s["visits"], s["false_triggers"], s["review"], s["date"]) == (5, 2, 2, DAY)
    assert s["latest"]["id"] == 13 and s["last_motion"]["id"] == 14
    assert [(c["label"], c["n"]) for c in s["critters"]] == [("Dark-eyed Junco", 2), ("Bushtit", 1), ("Other", 1),
                                                             ("Squirrel", 1)]


def test_scoreboard_at_the_boundaries(boundary):
    conn = boundary
    rows = lambda *a: [(r["label"], r["n"]) for r in db.scoreboard(conn, *a)]
    assert sorted(rows("2026-10-31", "2026-10-31")) == [("Bushtit", 1), ("Song Sparrow", 1)]
    assert rows(None, "2026-10-31") == [("Bushtit", 3), ("Song Sparrow", 2), ("Dark-eyed Junco", 1)]
    assert rows(DAY, None) == [("Dark-eyed Junco", 2), ("Bushtit", 1), ("Other", 1), ("Squirrel", 1)]
    assert rows("2026-10-01", "2026-10-03") == [("Song Sparrow", 1), ("Bushtit", 1)]
    assert db.scoreboard_total(conn, "2026-10-31", "2026-10-31") == 1
    assert db.scoreboard_total(conn, None, "2026-10-31") == 5
    assert db.scoreboard_total(conn, DAY, None) == 5
    assert db.scoreboard_total(conn) == 10


def test_activity_at_the_boundaries(boundary):
    a = db.activity(boundary)
    assert (a["visitors"], a["false_alarms"], a["kept_on_pi"], a["sent_to_claude"], a["would_keep"]) == (5, 4, 2, 7, 2)
    assert a["claude_usd"] == pytest.approx(7 * 0.00015 + 0.009)
    b = db.activity(boundary, day="2026-10-31")
    assert (b["visitors"], b["false_alarms"], b["kept_on_pi"], b["sent_to_claude"], b["would_keep"]) == (1, 2, 1, 2, 0)
    assert db.activity(boundary, day="2026-09-30")["visitors"] == 1
    assert db.activity(boundary, day="2026-10-02")["false_alarms"] == 1


def test_weather_report_days_at_the_boundaries(boundary):
    r = db.weather_report(boundary, NOW)
    by_day = {d["day"]: d for d in r["days"]}
    assert [d["day"] for d in r["days"]][:2] == ["2026-09-30", "2026-10-01"] and r["days"][-1]["day"] == DAY
    assert len(r["days"]) == 33 and r["max_visitors"] == 5 and r["first"] == datetime(2026, 9, 30, 23, 59, 59)
    pick = lambda day: (by_day[day]["visitors"], by_day[day]["species"], by_day[day]["false_alarms"])
    assert pick("2026-09-30") == (1, 1, 0)
    assert pick("2026-10-01") == (1, 1, 0)
    assert pick("2026-10-02") == (0, 0, 1)
    assert pick("2026-10-03") == (1, 1, 0)
    assert pick("2026-10-31") == (1, 2, 2)
    assert pick(DAY) == (5, 3, 4)
    assert by_day[DAY]["top"] == [("Dark-eyed Junco", 2), ("Bushtit", 1), ("Squirrel", 1)]
    assert by_day["2026-10-31"]["top"] == [("Bushtit", 1), ("Song Sparrow", 1)]
    assert sorted(tuple(r) for r in db.seen_labels(boundary, "2026-10-31", DAY)) == [
        (7, "2026-10-31T23:59:59", "Bushtit"), (7, "2026-10-31T23:59:59", "Song Sparrow"),
        (8, "2026-11-01T00:00:00", "Squirrel"), (9, "2026-11-01T01:30:00", "Dark-eyed Junco"),
        (10, "2026-11-01T01:30:00", "Dark-eyed Junco"), (13, "2026-11-01T09:00:00", "Bushtit")]


def test_capture_rate_at_the_boundaries(boundary):
    r = db.capture_rate(boundary)
    assert (r["photos"], r["days"], r["recent_days"]) == (14, 7, 7)
    assert r["photos_per_day"] == pytest.approx(9 / 7) and r["visitors_per_day"] == pytest.approx(6 / 7)
    assert r["all_photos_per_day"] == pytest.approx(14 / 33) and r["all_visitors_per_day"] == pytest.approx(10 / 33)
    assert (r["busiest_hour"], r["median_gap_min"], r["last"]) == (1, 30.0, "2026-11-01T12:00:00")
    assert db.capture_rate(boundary, days=1)["photos_per_day"] == 7


def test_daily_hourly_and_month_counts_at_the_boundaries(boundary):
    conn = boundary
    daily = dict(db.daily_counts(conn))
    assert len(daily) == 30 and list(daily)[0] == "2026-10-03" and list(daily)[-1] == DAY
    assert {d: n for d, n in daily.items() if n} == {"2026-10-03": 1, "2026-10-25": 1, "2026-10-31": 1, DAY: 5}
    assert db.daily_counts(conn, days=3, species="Dark-eyed Junco") == [("2026-10-30", 0), ("2026-10-31", 0), (DAY, 2)]
    assert db.daily_counts(conn, days=33)[0] == ("2026-09-30", 1)
    assert db.daily_counts(conn, days=2, species="Song Sparrow") == [("2026-10-31", 1), (DAY, 0)]
    hourly = db.hourly_counts(conn)
    assert {h: n for h, n in enumerate(hourly) if n} == {0: 3, 1: 2, 2: 1, 8: 1, 9: 1, 23: 2}
    assert {h: n for h, n in enumerate(db.hourly_counts(conn, false_triggers=True)) if n} == {0: 1, 1: 1, 12: 2}
    assert db.month_counts(conn, "Bushtit") == [0] * 9 + [3, 1, 0]


def test_reports_over_recent_days_at_the_boundaries(boundary):
    conn = boundary
    d = db.detector_report(conn, days=14)
    assert (d["checked"], d["would_skip"], d["would_save"], d["caught"], d["animals"], d["would_miss"]) == (8, 2, 2, 6, 6, [])
    assert db.detector_report(conn, days=1)["checked"] == 7
    s = db.scenery_report(conn, days=2)
    assert (s["checked"], s["would_save"], s["empties"], s["animals"]) == (8, 2, 2, 6)
    p = db.presence_report(conn, days=1)
    assert (p["frames"], p["visitors"], p["empty"], p["would_skip"]) == (7, 5, 2, 2)
    assert db.classifier_report(conn, days=2)["checked"] == 4
    assert db.classifier_report(conn, days=1)["checked"] == 3
    assert db.classifier_report(conn)["checked"] == 4


def test_watcher_counts_todays_claude_calls_by_day(conn, add_visit):
    today = FROZEN_NOW.date()
    add_visit(captured_at=f"{today}T00:00:00")
    add_visit(captured_at=f"{today}T23:59:59", raw_json=json.dumps({"trigger": "ground"}))
    add_visit(captured_at=f"{today - timedelta(days=1)}T23:59:59")
    add_visit(captured_at=f"{today}T12:00:00", model="bioclip-presence")
    assert watch.calls_today() == 1 and watch.calls_today(ground=True) == 1


def plans(conn, run) -> dict[str, list[str]]:
    """EXPLAIN QUERY PLAN of every SELECT that run(conn) executes, by statement."""
    seen: list[str] = []
    conn.set_trace_callback(seen.append)
    try:
        run(conn)
    finally:
        conn.set_trace_callback(None)
    return {sql: [r[3] for r in conn.execute("EXPLAIN QUERY PLAN " + sql)]
            for sql in seen if sql.lstrip().upper().startswith(("SELECT", "WITH"))}


def test_todays_summary_and_the_nav_counts_come_from_indexes(boundary):
    """No query of the home page's Today panel or the nav bubbles reads the whole visits table."""
    conn = boundary
    table_scan = re.compile(r"SCAN visits(?! USING)")
    today = plans(conn, db.today_summary)
    for sql, plan in today.items():
        assert not any(table_scan.search(p) for p in plan), (sql, plan)
    main = next(plan for sql, plan in today.items() if "COUNT(DISTINCT id)" in sql)
    searched = [p for p in main if "visits" in p]
    assert searched and all(re.match(r"SEARCH visits USING (COVERING )?INDEX visits_\w+ \(captured_at>\? AND captured_at<\?\)", p)
                            for p in searched), main
    for run, index in ((lambda c: db.count_visits(c, review=True), "visits_review"),
                       (db.count_unchecked_false_triggers, "visits_false_photo")):
        (plan,) = plans(conn, run).values()
        assert plan == [f"SCAN visits USING COVERING INDEX {index}"] or plan == [f"SCAN visits USING INDEX {index}"], plan
    # the species page, the home list and the week's scoreboard read real visits from their index too
    for run in (db.species_summary, lambda c: db.species_row(c, "Bushtit"), lambda c: db.list_visits(c, limit=24),
                lambda c: db.scoreboard(c, "2026-10-26", DAY), db.count_visits, db.hourly_counts, db.daily_counts):
        for sql, plan in plans(conn, run).items():
            assert not any(table_scan.search(p) for p in plan), (sql, plan)


def test_init_db_adds_the_indexes_to_an_old_database(tmp_path):
    path = tmp_path / "old.db"
    with db.session(path) as conn:
        conn.execute(
            "CREATE TABLE visits (id INTEGER PRIMARY KEY, captured_at TEXT NOT NULL, image_path TEXT NOT NULL, "
            "thumb_path TEXT NOT NULL, visitor TEXT NOT NULL, species TEXT, count INTEGER, confidence TEXT, notes TEXT, "
            "model TEXT, escalated INTEGER NOT NULL DEFAULT 0, needs_review INTEGER NOT NULL DEFAULT 0, "
            "corrected_species TEXT, is_night INTEGER NOT NULL DEFAULT 0, raw_json TEXT)")
        conn.execute("INSERT INTO visits (captured_at, image_path, thumb_path, visitor) VALUES ('2026-01-01T09:00:00', 'a', 'b', 'bird')")
        conn.commit()
        db.init_db(conn)
        db.init_db(conn)
        names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'index' AND tbl_name = 'visits'")}
        assert {"visits_captured_at", "visits_real", "visits_review", "visits_false_photo"} <= names
        assert db.count_visits(conn) == 1 and db.count_visits(conn, review=True) == 0
        assert db.insert_visit(conn, captured_at="2026-01-02T09:00:00", image_path="a", thumb_path="b", visitor="bird") == 2


def test_species_counts_agree_everywhere_and_leave_out_other(boundary, add_visit):
    conn = boundary
    add_visit(captured_at=f"{DAY}T10:00:00", species="Bushtit", also='[{"count": 1}]')  # an entry with no name
    counts = {"today": db.today_summary(conn)["species"], "activity": db.activity(conn)["species"],
              "weather": {d["day"]: d["species"] for d in db.weather_report(conn, NOW)["days"]}[DAY]}
    assert counts == {"today": 3, "activity": 3, "weather": 3}  # Squirrel, Dark-eyed Junco, Bushtit; not the "other"
    assert db.totals(conn)["species"] == 4  # plus Song Sparrow, seen in October


def test_totals_are_reused_for_a_minute_per_database(conn, add_visit, tmp_path, monkeypatch):
    add_visit()
    assert db.totals(conn)["visitors"] == 1
    add_visit(raw_json=json.dumps({"calls": {"owner_checks": [{"source": "haiku", "usd": 0.002}]}}))
    assert db.totals(conn)["visitors"] == 1 and db.owner_checks_spend(conn)["n"] == 0  # a minute old at most
    assert db.totals(conn, max_age=0)["visitors"] == 2 and db.totals(conn)["owner_checks"] == 1
    with db.session(tmp_path / "other.db") as other:  # another database's figures are its own
        db.init_db(other)
        assert db.totals(other)["visitors"] == 0
    add_visit()
    assert db.totals(conn)["visitors"] == 2
    later = db.time.monotonic() + db.TOTALS_TTL + 1
    monkeypatch.setattr(db.time, "monotonic", lambda: later)
    assert db.totals(conn)["visitors"] == 3
    db.forget_totals()
    assert db.totals(conn)["visitors"] == 3


def test_a_deleted_newest_visits_id_is_never_used_again(conn, add_visit):
    """Open pages notice a new visit by its id being higher than the last one they saw."""
    first, last = add_visit(), add_visit()
    db.delete_visit(conn, last)
    assert db.newest_id(conn) == first
    again = add_visit()
    assert again > last
    with db.session() as other:  # the other process: its own connection
        db.delete_visit(other, again)
        theirs = db.insert_visit(other, captured_at=FROZEN_NOW.isoformat(), image_path="a", thumb_path="b", visitor="bird")
    assert theirs > again and add_visit() > theirs
    assert db.get_value(conn, db.MAX_ID) == str(db.newest_id(conn))
    conn.execute("DELETE FROM settings WHERE key = ?", (db.MAX_ID,))  # a database from before the mark
    conn.commit()
    newest = db.newest_id(conn)
    assert add_visit() == newest + 1 and db.get_value(conn, db.MAX_ID) == str(newest + 1)
