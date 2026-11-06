""""Other" visits (an animal Claude couldn't put in a visitor type, or its placeholder when it couldn't answer):
they always wait for review, come first in the review queue, and are the admin's only, like false triggers."""
import re
import sqlite3

import pytest
from fastapi.testclient import TestClient

from birdbrain import config, db, mirror
from birdbrain.web import auth

from helpers.clock import FROZEN_NOW
from helpers.sql import plans

TODAY = FROZEN_NOW.date().isoformat()


@pytest.fixture
def client(conn):
    from birdbrain.web import app as webapp

    with TestClient(webapp.app) as c:
        yield c


@pytest.fixture
def locked(monkeypatch):
    """Read-only mode: an admin password is set, so a client without the cookie is a visitor."""
    monkeypatch.setenv("ADMIN_PASSWORD", "s3cret!")
    auth._fails.clear()


def card_ids(html: str) -> list[int]:
    return [int(i) for i in re.findall(r'class="card[^"]*" href="/visit/(\d+)', html)]


def other(add_visit, **fields) -> int:
    return add_visit(**{"visitor": "other", "species": "Domestic Cat", "confidence": "high",
                        "captured_at": f"{TODAY}T11:00:00", "image_path": "2026/11/01/cat.jpg",
                        "thumb_path": "2026/11/01/cat_t.jpg", **fields})


@pytest.fixture
def yard(add_visit):
    """A bird, then a newer "other" (the newest visit of all)."""
    bird = add_visit(captured_at=f"{TODAY}T10:00:00", species="Bushtit")
    cat = other(add_visit)
    return bird, cat


def rerun_migration(conn) -> None:
    conn.execute("DELETE FROM settings WHERE key = ?", (db.OTHER_REVIEW_DONE,))
    conn.commit()
    db.init_db(conn)


def flagged(conn, vid: int) -> int:
    return conn.execute("SELECT needs_review FROM visits WHERE id = ?", (vid,)).fetchone()[0]


# ---------- needs review ----------

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


# ---------- admin only ----------

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


def test_visitors_never_see_an_other_visit(client, yard, locked):
    bird, cat = yard
    home = client.get("/").text
    assert card_ids(home) == [bird] and f"/visit/{cat}" not in home and "Domestic Cat" not in home
    assert client.get(f"/visits/more?before={TODAY}T23:59:59,999").text.count(f"/visit/{cat}") == 0
    live = client.get("/live").text
    assert f"/visit/{bird}" in live and f"/visit/{cat}" not in live
    api = client.get("/api/latest-visit").json()
    assert (api["id"], api["any_id"]) == (bird, bird)
    assert client.get(f"/visit/{cat}").status_code == 404
    assert f"/visit/{cat}" not in client.get(f"/visit/{bird}").text  # no arrow to it
    assert "Domestic Cat" not in client.get("/species").text
    assert client.get("/species/Domestic Cat").status_code == 404
    assert "All time: 1 visit since " in client.get("/stats").text


def test_the_admin_sees_other_visits_everywhere(client, yard, locked):
    bird, cat = yard
    assert client.post("/login", data={"password": "s3cret!"}, follow_redirects=False).status_code == 303
    assert card_ids(client.get("/").text) == [cat, bird]
    assert client.get("/api/latest-visit").json()["id"] == cat
    assert client.get(f"/visit/{cat}").status_code == 200
    assert f"/visit/{cat}" in client.get(f"/visit/{bird}").text
    assert "All time: 2 visits since " in client.get("/stats").text


def test_the_admins_review_page_starts_with_the_other_visits(client, conn, add_visit, locked):
    unsure = add_visit(captured_at="2026-10-01T08:00:00", confidence="low", needs_review=1)
    cat = other(add_visit, needs_review=1)
    assert client.post("/login", data={"password": "s3cret!"}, follow_redirects=False).status_code == 303
    assert card_ids(client.get("/review").text) == [cat, unsure]


def test_no_password_means_everyone_sees_other_visits(client, yard):
    bird, cat = yard
    assert client.get(f"/visit/{cat}").status_code == 200 and card_ids(client.get("/").text) == [cat, bird]


def test_relabelled_to_an_animal_is_public_again_confirmed_other_stays_hidden(client, conn, yard, locked):
    bird, cat = yard
    db.mark_reviewed(conn, cat)  # Confirm: it really was something nobody can name
    assert client.get(f"/visit/{cat}").status_code == 404
    db.correct_visit(conn, cat, "squirrel", None, None)
    assert client.get(f"/visit/{cat}").status_code == 200 and card_ids(client.get("/").text) == [cat, bird]
    db.correct_visit(conn, bird, "other", None, None)  # the owner's own "other" is hidden too
    assert client.get(f"/visit/{bird}").status_code == 404


def test_the_public_mirror_has_no_other_pages(client, yard, monkeypatch):
    monkeypatch.setenv("BIRDBRAIN_PUBLIC", "1")
    bird, cat = yard
    assert client.get(f"/visit/{cat}").status_code == 404 and card_ids(client.get("/").text) == [bird]


def test_visitors_get_404_for_an_other_visits_burst_frames(client, add_visit, locked):
    day = config.IMAGES_DIR / "2026" / "11" / "01"
    day.mkdir(parents=True, exist_ok=True)
    names = ("cat.jpg", "cat_t.jpg", "cat_b1.jpg", "cat_b1_t.jpg", "real.jpg", "real_b1.jpg")
    for name in names:
        (day / name).write_bytes(b"jpeg")
    try:
        other(add_visit)
        add_visit(image_path="2026/11/01/real.jpg", thumb_path="2026/11/01/real_t.jpg")
        for url in ("/images/2026/11/01/cat_b1.jpg", "/images/2026/11/01/cat_b1_t.jpg"):
            assert client.get(url).status_code == 404, url
        assert client.get("/images/2026/11/01/real_b1.jpg").status_code == 200
        assert client.post("/login", data={"password": "s3cret!"}, follow_redirects=False).status_code == 303
        assert client.get("/images/2026/11/01/cat_b1.jpg").status_code == 200
    finally:
        for name in names:
            (day / name).unlink(missing_ok=True)


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


def test_also_in_the_photo_other_entries_stay_on_a_public_visit(client, add_visit, locked):
    """Only the main animal decides: an "other" among a public visit's other animals is still shown (for now)."""
    vid = add_visit(species="Bushtit", also='[{"visitor": "other", "species": null, "count": 1}]')
    assert client.get(f"/visit/{vid}").status_code == 200 and card_ids(client.get("/").text) == [vid]
