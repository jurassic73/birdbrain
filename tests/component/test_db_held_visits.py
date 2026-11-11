"""Component tests of birdbrain/db.py (held visits): a visit whose video came through choppy is held for review.
It shows in the review queue and nowhere else, to the admin either, until the owner confirms or corrects it."""

import re
import sqlite3

from helpers.sql import plans
from birdbrain import config, db, mirror


def held_and_shown(conn, add_visit) -> tuple[int, int]:
    """An older junco that shows and a newer squirrel that is held."""
    bird = add_visit(captured_at="2026-11-01T08:00:00")
    squirrel = add_visit(captured_at="2026-11-01T09:00:00", visitor="squirrel", species="Eastern Gray Squirrel")
    db.hold_for_review(conn, squirrel)
    return bird, squirrel


def test_a_held_visit_is_in_the_review_queue_and_nowhere_else(conn, add_visit):
    bird, squirrel = held_and_shown(conn, add_visit)
    for visitor in (False, True):
        with db.visitor_view(visitor):
            assert [r["id"] for r in db.list_visits(conn)] == [bird] and db.count_visits(conn) == 1
            assert db.latest_visit(conn)["id"] == bird and db.latest_visit(conn, real_only=False)["id"] == bird
            assert "Eastern Gray Squirrel" not in db.species_names(conn)
            assert db.get_neighbors(conn, bird) == (None, None)
    assert [r["id"] for r in db.list_visits(conn, review=True)] == [squirrel]  # the admin's queue
    with db.visitor_view():
        assert db.list_visits(conn, review=True) == []
    assert db.is_admin_only(conn, squirrel) and not db.is_admin_only(conn, bird)


def test_confirming_or_correcting_a_held_visit_shows_it_again(conn, add_visit):
    bird, squirrel = held_and_shown(conn, add_visit)
    db.mark_reviewed(conn, squirrel)
    assert [r["id"] for r in db.list_visits(conn)] == [squirrel, bird] and db.list_visits(conn, review=True) == []
    assert not db.is_admin_only(conn, squirrel)
    db.hold_for_review(conn, bird)
    db.correct_visit(conn, bird, "bird", "Bushtit", None)
    assert [r["id"] for r in db.list_visits(conn)] == [squirrel, bird]


def test_lists_and_counts_still_come_from_the_index_with_held_in_it(conn, add_visit):
    held_and_shown(conn, add_visit)
    table_scan = re.compile(r"SCAN visits(?! USING)")
    for visitor in (False, True):
        with db.visitor_view(visitor):
            for run in (db.today_summary, lambda c: db.list_visits(c, limit=24), db.count_visits, db.daily_counts):
                for sql, plan in plans(conn, run).items():
                    assert not any(table_scan.search(p) for p in plan), (sql, plan)


def test_init_db_rebuilds_an_index_whose_definition_changed(conn, add_visit):
    held_and_shown(conn, add_visit)
    conn.execute("DROP INDEX visits_real")
    conn.execute(f"CREATE INDEX visits_real ON visits(captured_at) WHERE {db.REAL}")  # as an older release built it
    conn.commit()
    db.init_db(conn)
    (sql,) = conn.execute("SELECT sql FROM sqlite_master WHERE name = 'visits_real'").fetchone()
    assert "held" in sql and db.count_visits(conn) == 1


def test_the_mirror_copy_drops_held_visits(conn, add_visit, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "IMAGES_DIR", tmp_path / "images")
    bird, squirrel = held_and_shown(conn, add_visit)
    copy = tmp_path / "mirror.db"
    mirror.copy_db(config.DB_PATH, copy)
    assert [r[0] for r in sqlite3.connect(copy).execute("SELECT id FROM visits")] == [bird]
    assert db.get_visit(conn, squirrel) is not None  # the Pi keeps it
