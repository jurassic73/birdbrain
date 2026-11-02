import sqlite3

import numpy as np
import pytest

from birdbrain import config, privacy
from birdbrain.detector import Detection


class FakeDetector:
    """Finds a person only in views whose top-left pixel is 255 (so a test can put one in a quarter)."""

    def __init__(self, score=0.6):
        self.score, self.calls = score, 0

    def detect(self, view, min_score):
        self.calls += 1
        return [Detection("person", self.score, (0, 0, 1, 1))] if view[0, 0, 0] == 255 else []


@pytest.fixture
def setup(tmp_path, monkeypatch, conn, add_visit):
    import cv2

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "IMAGES_DIR", tmp_path / "images")
    (tmp_path / "images").mkdir()
    fake = FakeDetector()
    monkeypatch.setattr(privacy, "_det", fake)

    def photo(name, person=False):
        img = np.zeros((40, 60, 3), np.uint8)
        if person:
            img[0, 0] = 255
        cv2.imwrite(str(tmp_path / "images" / name), img)
        return name

    clean = add_visit(image_path=photo("a.jpg"), thumb_path="a_t.jpg")
    person = add_visit(image_path=photo("b.jpg", person=True), thumb_path="b_t.jpg")
    photo("c.jpg")
    photo("c_b2.jpg", person=True)  # a person in a burst frame, not the best one
    burst = add_visit(image_path="c.jpg", thumb_path="c_t.jpg")
    conn.commit()
    return fake, clean, person, burst


def test_views_cover_the_frame_with_overlapping_quarters():
    img = np.zeros((100, 200, 3), np.uint8)
    v = privacy.views(img)
    assert len(v) == 5 and v[0].shape == (100, 200, 3)
    assert all(q.shape[0] > 50 and q.shape[1] > 100 for q in v[1:])


def test_filter_drops_people_from_the_mirror_copy_only(setup, tmp_path):
    fake, clean, person, burst = setup
    excl = tmp_path / "private.txt"
    assert privacy.filter_db(config.DB_PATH, excl) == 2
    left = [r[0] for r in sqlite3.connect(config.DB_PATH).execute("SELECT id FROM visits")]
    assert left == [clean]
    lines = excl.read_text().split()
    assert {"/b.jpg", "/b_t.jpg", "/c.jpg", "/c_b2.jpg", "/c_t.jpg"} <= set(lines) and "/a.jpg" not in lines
    # Cached: the second run checks nothing again.
    calls = fake.calls
    privacy.filter_db(config.DB_PATH, excl)
    assert fake.calls == calls
    assert privacy.visit_status("b.jpg") == "person" and privacy.visit_status("a.jpg") is None


def test_unchecked_visits_stay_off_when_out_of_time(setup, tmp_path):
    _, clean, *_ = setup
    rows = sqlite3.connect(config.DB_PATH).execute("SELECT id, image_path, thumb_path FROM visits").fetchall()
    blocked, checked = privacy.blocked_visits(rows, {}, budget=-1)
    assert checked == 0 and clean in blocked and len(blocked) == 3


def test_check_cli_fails_closed(setup, tmp_path, monkeypatch):
    assert privacy.main(["check", str(tmp_path / "images" / "a.jpg")]) == 0
    assert privacy.main(["check", str(tmp_path / "images" / "b.jpg")]) == 1
    assert privacy.main(["check", str(tmp_path / "missing.jpg")]) == 2
    monkeypatch.setattr(privacy, "_det", None)
    monkeypatch.setattr(config, "DETECTOR_MODEL", tmp_path / "nope.onnx")
    assert privacy.main(["check", str(tmp_path / "images" / "a.jpg")]) == 2


def mirror_copy(tmp_path):
    """What push_public.sh hands the filter: a copy of the Pi's database."""
    dst = tmp_path / "mirror.db"
    dst.unlink(missing_ok=True)
    src, out = sqlite3.connect(config.DB_PATH), sqlite3.connect(dst)
    src.backup(out)
    src.close()
    out.close()
    return dst


def pushed(tmp_path) -> set[int]:
    """Run the filter on a fresh copy; the visit ids that would go to the mirror."""
    copy = mirror_copy(tmp_path)
    privacy.filter_db(copy, tmp_path / "excl.txt")
    return {r[0] for r in sqlite3.connect(copy).execute("SELECT id FROM visits")}


@pytest.fixture
def fresh_status(monkeypatch):
    monkeypatch.setattr(privacy, "_status", (0.0, {}))


def test_public_anyway_lets_a_flagged_visit_through(setup, tmp_path, conn, fresh_status):
    from birdbrain import db

    _, clean, person, burst = setup
    assert pushed(tmp_path) == {clean}  # also scores every photo
    cache = privacy.load_cache()
    assert privacy.make_override("a.jpg", cache) is None  # nothing to clear on a clean visit
    assert privacy.make_override("b.jpg", {}) is None  # nor before it's checked
    override = privacy.make_override("b.jpg", cache)
    assert override["files"] == {"b.jpg": 0.6}
    db.set_public_ok(conn, person, override)
    assert privacy.visit_status("b.jpg", db.get_visit(conn, person)["public_ok"]) == "cleared"
    assert privacy.visit_status("c.jpg", None) == "person"

    assert pushed(tmp_path) == {clean, person}
    lines = (tmp_path / "excl.txt").read_text().split()
    assert "/b.jpg" not in lines and "/c_b2.jpg" in lines
    # The owner's call stays on the Pi: the copy has no overrides left.
    copy = mirror_copy(tmp_path)
    privacy.filter_db(copy, tmp_path / "excl.txt")
    assert sqlite3.connect(copy).execute("SELECT COUNT(*) FROM visits WHERE public_ok IS NOT NULL").fetchone()[0] == 0

    # Undo holds it back again.
    db.set_public_ok(conn, person, None)
    assert db.get_visit(conn, person)["public_ok"] is None
    assert pushed(tmp_path) == {clean}


def test_public_anyway_covers_only_the_photos_and_scores_cleared(setup, tmp_path, conn):
    import cv2
    from birdbrain import db

    _, clean, person, burst = setup
    pushed(tmp_path)
    db.set_public_ok(conn, burst, privacy.make_override("c.jpg", privacy.load_cache()))
    assert burst in pushed(tmp_path)

    # A higher score on a re-check holds it back.
    score = privacy.load_cache()["c_b2.jpg"]
    privacy.save_cache({"c_b2.jpg": score + 0.1})
    assert burst not in pushed(tmp_path)
    privacy.save_cache({"c_b2.jpg": score})
    assert burst in pushed(tmp_path)

    # So does a new photo for the visit that shows a person.
    img = np.zeros((40, 60, 3), np.uint8)
    img[0, 0] = 255
    cv2.imwrite(str(tmp_path / "images" / "c_b3.jpg"), img)
    assert burst not in pushed(tmp_path)


def test_unreadable_override_keeps_the_visit_private(setup, tmp_path, conn):
    _, clean, person, burst = setup
    pushed(tmp_path)
    for bad in ("not json", '{"files": ["b.jpg"]}', '{"files": {"b.jpg": "high"}}', "[]"):
        conn.execute("UPDATE visits SET public_ok = ? WHERE id = ?", (bad, person))
        conn.commit()
        assert privacy.parse_override(bad) == {}
        assert person not in pushed(tmp_path), bad


def test_filter_reads_an_old_database_without_overrides(setup, tmp_path):
    copy = mirror_copy(tmp_path)
    old = sqlite3.connect(copy)
    old.execute("ALTER TABLE visits DROP COLUMN public_ok")
    old.commit()
    old.close()
    assert privacy.filter_db(copy, tmp_path / "excl.txt") == 2


def test_public_anyway_route_is_admin_only(setup, tmp_path, conn, monkeypatch, fresh_status):
    from fastapi.testclient import TestClient

    from birdbrain import db
    from birdbrain.web import app as webapp
    from birdbrain.web import auth

    _, clean, person, burst = setup
    pushed(tmp_path)
    with TestClient(webapp.app) as client:
        page = client.get(f"/visit/{person}").text
        assert f'hx-post="/visit/{person}/public"' in page and 'data-confirm-title="No person' in page
        assert client.post(f"/visit/{clean}/public").status_code == 409
        assert client.post(f"/visit/{person}/public").status_code == 200
        assert db.get_visit(conn, person)["public_ok"]
        page = client.get(f"/visit/{person}").text
        assert ">public (checked)<" in page and f'hx-post="/visit/{person}/private"' in page
        assert client.post(f"/visit/{person}/private").status_code == 200
        assert db.get_visit(conn, person)["public_ok"] is None

        monkeypatch.setenv("ADMIN_PASSWORD", "s3cret!")
        auth._fails.clear()
        for path in (f"/visit/{person}/public", f"/visit/{person}/private"):
            assert client.post(path).status_code == 403
        assert db.get_visit(conn, person)["public_ok"] is None
        assert "Public anyway" not in client.get(f"/visit/{person}").text
