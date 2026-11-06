"""Component tests of birdbrain/circle.py (check): one real local dependency, a temp database or files on disk."""

import json

from helpers.circle_check import checks_of, photo_visit
from birdbrain import circle_check, db


def test_checks_kept_per_visit_are_capped(conn, photo_visit):
    vid = photo_visit()
    for i in range(circle_check.KEEP + 5):
        circle_check.record(conn, vid, {"source": "pi", "say": str(i)}, {"x": 0.5, "y": 0.5, "r": 0.1})
    checks = checks_of(conn, vid)
    assert len(checks) == circle_check.KEEP and checks[-1]["say"] == str(circle_check.KEEP + 4)


def test_record_writes_only_the_checks_and_keeps_the_rest(conn, photo_visit):
    vid = photo_visit(raw_json=json.dumps({"trigger": "rim", "calls": {"circle": {"x": 0.5, "y": 0.5, "r": 0.1}}}))
    db.end_stay(conn, vid, "2026-11-01T09:05:00", "left")  # what the watcher writes to the same row
    circle_check.record(conn, vid, {"source": "pi", "say": "a bird"}, {"x": 0.5, "y": 0.5, "r": 0.1, "frame": 2})
    raw = json.loads(db.get_visit(conn, vid)["raw_json"])
    assert raw["trigger"] == "rim" and raw["stay_end"] == "left" and raw["calls"]["circle"]["r"] == 0.1
    assert checks_of(conn, vid)[0]["frame"] == 2 and not conn.in_transaction
    conn.execute("UPDATE visits SET raw_json = 'not json' WHERE id = ?", (vid,))
    conn.commit()
    circle_check.record(conn, vid, {"source": "pi", "say": "again"}, {"x": 0.5, "y": 0.5, "r": 0.1})
    assert [c["say"] for c in checks_of(conn, vid)] == ["again"]


def test_owner_check_spend_skips_bad_raw_json(conn, add_visit):
    vid = add_visit()
    conn.execute("UPDATE visits SET raw_json = 'not json' WHERE id = ?", (vid,))
    conn.commit()
    assert db.owner_checks_spend(conn) == {"n": 0, "usd": 0.0}


def test_pi_average_recent_vs_all_time_and_legacy_blob(conn):
    db.set_value(conn, db.PI_CHECK_TIMES, json.dumps({"n": 10, "total": 100.0, "recent": [10.0] * 10}))
    assert db.pi_check_average(conn) == {"n": 10, "avg": 10.0, "recent_avg": 10.0}
    for _ in range(db.PI_CHECK_RECENT):
        db.record_pi_check(conn, 30)
    a = db.pi_check_average(conn)
    assert a["n"] == 10 + 20 and a["avg"] == round((100 + 600) / 30, 1) and a["recent_avg"] == 30.0
    db.record_pi_check(conn, 99, ok=False)
    assert db.pi_check_average(conn) == a
    assert json.loads(db.get_value(conn, db.PI_CHECK_TIMES))["n"] == 10  # the blob is never written again
