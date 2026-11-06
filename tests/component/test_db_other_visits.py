"""Component tests of birdbrain/db.py (other visits): one real local dependency, a temp database or files on disk."""

import re
import sqlite3

from helpers.other_visits import other, yard
from helpers.sql import plans
from birdbrain import config, db, mirror


def rerun_migration(conn) -> None:
    conn.execute("DELETE FROM settings WHERE key = ?", (db.OTHER_REVIEW_DONE,))
    conn.commit()
    db.init_db(conn)


def flagged(conn, vid: int) -> int:
    return conn.execute("SELECT needs_review FROM visits WHERE id = ?", (vid,)).fetchone()[0]


def test_init_db_sends_old_unchecked_other_visits_to_review_once(conn, add_visit):
    plain = other(add_visit)
    confirmed = other(add_visit, confirmed_at="2026-11-01T12:00:00")
    noted = other(add_visit, feedback="the neighbour's cat")
    relabelled = other(add_visit, corrected_visitor="squirrel")  # the owner's label wins: not an "other" now
    to_other = add_visit(species="Bushtit", corrected_visitor="other")  # the owner's own label: reviewed already
    bird = add_visit(species="Bushtit")
    rerun_migration(conn)
    assert [flagged(conn, v) for v in (plain, confirmed, noted, relabelled, to_other, bird)] == [1, 0, 0, 0, 0, 0]
    # The owner clears it with a save that leaves no trace (nothing changed): a restart doesn't put it back.
    db.correct_visit(conn, plain, None, None, None)
    db.init_db(conn)
    assert flagged(conn, plain) == 0


def test_the_review_queue_lists_other_visits_first_then_oldest_first(conn, add_visit):
    old_bird = add_visit(captured_at="2026-10-01T08:00:00", needs_review=1, confidence="low")
    cat = other(add_visit, captured_at="2026-11-02T09:00:00", needs_review=1)
    empty = add_visit(captured_at="2026-10-15T08:00:00", visitor="none", species=None, needs_review=1)
    deer = other(add_visit, captured_at="2026-10-20T09:00:00", species=None, needs_review=1)
    fixed = add_visit(captured_at="2026-09-01T08:00:00", visitor="other", corrected_visitor="bird", needs_review=1)
    assert [r["id"] for r in db.list_visits(conn, review=True)] == [deer, cat, fixed, old_bird, empty]
    assert db.count_visits(conn, review=True) == 5


def test_visitor_queries_still_come_from_the_indexes(conn, add_visit):
    """PUBLIC keeps REAL as a term of its own, so visits_real still serves a visitor's lists and counts."""
    other(add_visit)
    add_visit(species="Bushtit")
    table_scan = re.compile(r"SCAN visits(?! USING)")
    with db.visitor_view():
        for run in (db.today_summary, db.species_summary, lambda c: db.list_visits(c, limit=24), db.count_visits,
                    db.hourly_counts, db.daily_counts, lambda c: db.scoreboard(c, "2026-10-26", "2026-11-01")):
            for sql, plan in plans(conn, run).items():
                assert not any(table_scan.search(p) for p in plan), (sql, plan)
        (plan,) = plans(conn, lambda c: db.count_visits(c, review=True)).values()
        assert "INDEX visits_review" in plan[0], plan


def test_db_leaves_other_out_only_for_a_visitor(conn, yard):
    bird, cat = yard
    assert db.latest_visit(conn)["id"] == cat and db.count_visits(conn) == 2  # the pipeline and the admin see all
    with db.visitor_view():
        assert db.latest_visit(conn)["id"] == bird
        assert db.latest_visit(conn, real_only=False)["id"] == bird
        assert db.count_visits(conn) == 1 and [r["id"] for r in db.list_visits(conn)] == [bird]
        assert db.today_summary(conn)["visits"] == 1 and db.activity(conn)["visitors"] == 1
        assert "Domestic Cat" not in db.species_names(conn)
        assert db.species_row(conn, "Domestic Cat") is None
        assert db.get_neighbors(conn, bird) == (None, None)
        assert [r["label"] for r in db.scoreboard(conn)] == ["Bushtit"] and db.scoreboard_total(conn) == 1
    assert not db.hiding_other()
    assert db.get_neighbors(conn, bird) == (cat, None)
    assert db.is_admin_only(conn, cat) and not db.is_admin_only(conn, bird)


def test_the_mirror_copy_drops_other_visits_and_their_files(conn, add_visit, tmp_path, monkeypatch):
    root = tmp_path / "images"
    monkeypatch.setattr(config, "IMAGES_DIR", root)
    day = root / "2026" / "11" / "01"
    day.mkdir(parents=True)
    for name in ("cat.jpg", "cat_t.jpg", "cat_b1.jpg", "fixed.jpg", "fixed_t.jpg", "a.jpg", "a_t.jpg"):
        (day / name).write_bytes(b"x")
    bird = add_visit(image_path="2026/11/01/a.jpg", thumb_path="2026/11/01/a_t.jpg")
    cat = other(add_visit)
    fixed = other(add_visit, corrected_visitor="squirrel", image_path="2026/11/01/fixed.jpg",
                  thumb_path="2026/11/01/fixed_t.jpg")
    assert not any("cat" in f for f in mirror.files_of(config.DB_PATH))  # even an untrimmed copy lists none
    copy = tmp_path / "mirror.db"
    mirror.copy_db(config.DB_PATH, copy)
    assert [r[0] for r in sqlite3.connect(copy).execute("SELECT id FROM visits ORDER BY id")] == [bird, fixed]
    files = mirror.files_of(copy)
    assert "2026/11/01/fixed.jpg" in files and not any("cat" in f for f in files)
    assert db.get_visit(conn, cat) is not None  # the Pi keeps it
