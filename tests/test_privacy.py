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
