import io
import json
from datetime import datetime

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from birdbrain import circle_check, classifier, db, identify, images
from birdbrain.web import app as webapp

LABELS = [("Junco hyemalis", "Dark-eyed Junco", True), ("Haemorhous mexicanus", "House Finch", True),
          ("Sciurus carolinensis", "Eastern Gray Squirrel", True), ("background", "", True)]
LOCAL = {"Junco hyemalis": "Dark-eyed Junco", "Haemorhous mexicanus": "House Finch",
         "Sciurus carolinensis": "Eastern Gray Squirrel"}
MARKS = json.dumps([{"x": 0.2, "y": 0.2, "r": 0.1, "claude": True}, {"x": 0.5, "y": 0.5, "r": 0.05}])


class FakeClassifier:
    """Embeddings as one-hot vectors over LABELS, so the scores are known exactly."""
    labels = LABELS
    text = np.eye(len(LABELS), dtype=np.float32)
    mammals = {"Sciurus carolinensis"}

    def __init__(self, i=0, k=1.0):
        self.vec, self.patches = np.zeros(len(LABELS), np.float32), []
        self.vec[i] = k

    def embed(self, bgr):
        self.patches.append(bgr.shape)
        return self.vec


@pytest.fixture
def client(conn):
    with TestClient(webapp.app) as c:
        yield c


@pytest.fixture
def photo_visit(add_visit):
    """A false trigger with a real 640 x 360 photo on disk."""
    def add(**fields) -> int:
        buf = io.BytesIO()
        Image.new("RGB", (640, 360), (90, 120, 200)).save(buf, "JPEG")
        image_rel, thumb_rel = images.store_frame(buf.getvalue(), datetime.now())
        row = {"visitor": "none", "species": None, "model": "bioclip-presence", "confidence": "medium"}
        return add_visit(image_path=image_rel, thumb_path=thumb_rel, **{**row, **fields})
    return add


@pytest.fixture
def fake_pi(monkeypatch):
    def use(clf):
        monkeypatch.setattr(classifier, "load", lambda: clf)
        monkeypatch.setattr(classifier, "local_species", lambda fetch=True: LOCAL)
        return clf
    return use


def checks_of(conn, vid):
    return json.loads(db.get_visit(conn, vid)["raw_json"] or "{}")["calls"]["owner_checks"]


def test_owner_circle_is_the_first_one_the_owner_drew():
    assert circle_check.owner_circle(MARKS) == (2, {"x": 0.5, "y": 0.5, "r": 0.05})
    assert circle_check.owner_circle(json.dumps([{"x": 0.2, "y": 0.2, "r": 0.1, "claude": True}])) is None
    assert circle_check.owner_circle("") is None and circle_check.owner_circle("not json") is None


def test_square_is_the_circles_box_with_a_margin_inside_the_frame():
    x, y, w, h = circle_check.square({"x": 0.5, "y": 0.5, "r": 0.05}, 1920, 1080)
    side = 2 * 0.05 * 1920 * circle_check.MARGIN
    assert w == pytest.approx(side / 1920, abs=1e-3) and h == pytest.approx(side / 1080, abs=1e-3)
    assert x == pytest.approx(0.5 - w / 2, abs=1e-3)
    x, y, w, h = circle_check.square({"x": 0.99, "y": 0.99, "r": 0.05}, 1920, 1080)  # near a corner: slides in
    assert x + w <= 1 + 1e-9 and y + h <= 1 + 1e-9


def test_pi_verdicts():
    sure = [{"name": "Dark-eyed Junco", "score": 0.9}]
    weak = [{"name": "Dark-eyed Junco", "score": 0.3}]
    animal, empty = {"score": 0.9, "empty": False}, {"score": 0.01, "empty": True}
    assert circle_check.pi_verdict(sure, 0.9, animal) == "sure"
    assert circle_check.pi_verdict(sure, 0.9, None) == "sure"           # the animal check not trained yet
    assert circle_check.pi_verdict(sure, 0.1, animal) == "unsure"       # mostly on species that aren't local
    assert circle_check.pi_verdict(sure, 0.9, empty) == "unsure"        # the two disagree
    assert circle_check.pi_verdict(weak, 0.9, animal) == "unsure"
    assert circle_check.pi_verdict(weak, 0.9, empty) == "empty"
    assert circle_check.pi_verdict([], None, None) == "empty"
    assert circle_check.pi_say("sure", sure) == "Looks like Dark-eyed Junco."
    assert circle_check.pi_say("empty", []) == "The Pi didn't find an animal here."


def test_visitor_of_names_the_form_type():
    m = {"Sciurus carolinensis", "Procyon lotor", "Didelphis virginiana", "Felis catus"}
    assert circle_check.visitor_of("Dark-eyed Junco", "Junco hyemalis", m) == "bird"
    assert circle_check.visitor_of("Eastern Gray Squirrel", "Sciurus carolinensis", m) == "squirrel"
    assert circle_check.visitor_of("Common Raccoon", "Procyon lotor", m) == "raccoon"
    assert circle_check.visitor_of("Virginia Opossum", "Didelphis virginiana", m) == "possum"
    assert circle_check.visitor_of("Domestic Cat", "Felis catus", m) == "other"


def test_ask_the_pi_names_whats_in_the_circle_and_changes_nothing(client, conn, photo_visit, fake_pi):
    clf = fake_pi(FakeClassifier(i=0))
    vid = photo_visit()
    r = client.post(f"/visit/{vid}/check-circle/pi", data={"marks": MARKS}).json()
    assert r["verdict"] == "sure" and r["say"] == "Looks like Dark-eyed Junco." and r["circle_n"] == 2
    assert r["top"][0]["name"] == "Dark-eyed Junco" and r["top"][0]["visitor"] == "bird"
    side = round(max(2 * 0.05 * 640 * circle_check.MARGIN, classifier.MIN_SIDE))
    assert clf.patches == [(side, side, 3)]  # just the circle's square, not the crop box
    v = db.get_visit(conn, vid)
    assert v["visitor"] == "none" and v["corrected_visitor"] is None and v["corrected_species"] is None
    (stored,) = checks_of(conn, vid)
    assert stored["source"] == "pi" and stored["circle"] == {"x": 0.5, "y": 0.5, "r": 0.05} and stored["at"]
    assert "usd" not in stored
    html = client.get(f"/visit/{vid}?ft=1").text
    assert "Your circle checks" in html and "Looks like Dark-eyed Junco." in html
    assert db.totals(conn)["claude_usd"] == 0  # the Pi is free


def test_ask_the_pi_weak_answer_and_squirrels(client, photo_visit, fake_pi):
    vid = photo_visit()
    fake_pi(FakeClassifier(i=2, k=0.01))  # spread over every species
    r = client.post(f"/visit/{vid}/check-circle/pi", data={"marks": MARKS}).json()
    assert r["verdict"] == "unsure" and r["say"].startswith("Maybe ")
    fake_pi(FakeClassifier(i=2))
    r = client.post(f"/visit/{vid}/check-circle/pi", data={"marks": MARKS}).json()
    assert r["top"][0] == {**r["top"][0], "name": "Eastern Gray Squirrel", "visitor": "squirrel"}


def test_checks_need_a_circle_a_photo_and_a_model(client, conn, photo_visit, add_visit):
    vid = photo_visit()
    only_claude = json.dumps([{"x": 0.2, "y": 0.2, "r": 0.1, "claude": True}])
    for marks in ("", "[]", only_claude):
        assert "Circle the animal first" in client.post(f"/visit/{vid}/check-circle/pi", data={"marks": marks}).json()["error"]
    # No model on this machine (conftest: no downloads): said plainly, nothing stored.
    assert "isn't available" in client.post(f"/visit/{vid}/check-circle/pi", data={"marks": MARKS}).json()["error"]
    assert "owner_checks" not in (json.loads(db.get_visit(conn, vid)["raw_json"] or "{}").get("calls") or {})
    gone = add_visit(visitor="none", species=None)
    conn.execute("UPDATE visits SET photo_deleted_at = '2026-10-31' WHERE id = ?", (gone,))
    conn.commit()
    assert "deleted" in client.post(f"/visit/{gone}/check-circle/pi", data={"marks": MARKS}).json()["error"]
    assert client.post(f"/visit/{vid}/check-circle/opus", data={"marks": MARKS}).status_code == 404
    assert client.post("/visit/99999/check-circle/pi", data={"marks": MARKS}).status_code == 404


def test_ask_haiku_then_sonnet_on_just_the_circle(client, conn, photo_visit, monkeypatch):
    vid = photo_visit()
    seen = []

    def ask(model, images_b64, captured_at, lessons=None, site=None, note=""):
        import base64
        seen.append((model, Image.open(io.BytesIO(base64.b64decode(images_b64[0]))).size, note))
        if model == identify.HAIKU:
            return identify.Sighting(visitor="bird", species="house finch", count=1, confidence="medium",
                                     notes="Red wash on the head."), \
                {"model": model, "input_tokens": 1500, "output_tokens": 100}
        return identify.Sighting(visitor="none", species=None, count=0, confidence="high", notes="A leaf."), \
            {"model": model, "input_tokens": 1500, "output_tokens": 100}
    monkeypatch.setattr(identify, "ask", ask)
    r = client.post(f"/visit/{vid}/check-circle/haiku", data={"marks": MARKS}).json()
    assert r["verdict"] == "unsure" and r["name"] == "House Finch" and r["visitor"] == "bird"  # eBird spelling
    assert r["usd"] == pytest.approx((1500 * 0.10 + 100 * 0.50) / 1e6)
    model, size, note = seen[0]
    side = round(max(2 * 0.05 * 640 * circle_check.MARGIN, classifier.MIN_SIDE))
    assert model == identify.HAIKU and abs(size[0] - side) <= 1 and abs(size[1] - side) <= 1 and "circled" in note
    r = client.post(f"/visit/{vid}/check-circle/sonnet", data={"marks": MARKS}).json()
    assert r["verdict"] == "empty" and r["say"] == "Sonnet didn't find an animal here."
    checks = checks_of(conn, vid)
    assert [c["source"] for c in checks] == ["haiku", "sonnet"]
    assert db.get_visit(conn, vid)["corrected_species"] is None  # only a suggestion until saved
    t = db.totals(conn)
    assert t["owner_checks"] == 2 and t["claude_usd"] == pytest.approx(sum(c["usd"] for c in checks))
    html = client.get(f"/visit/{vid}").text
    assert "Haiku says House Finch (medium)." in html and "Red wash on the head." in html


def test_claude_errors_are_said_plainly(client, conn, photo_visit, monkeypatch):
    vid = photo_visit()

    def broken(*a, **k):
        raise ConnectionError("offline")
    monkeypatch.setattr(identify, "ask", broken)
    assert "couldn't be reached" in client.post(f"/visit/{vid}/check-circle/haiku", data={"marks": MARKS}).json()["error"]
    monkeypatch.setattr(identify, "ask", lambda *a, **k: (None, {"input_tokens": 10, "output_tokens": 0}))
    r = client.post(f"/visit/{vid}/check-circle/haiku", data={"marks": MARKS}).json()
    assert r["verdict"] == "refused" and "declined" in r["say"]


def test_checks_kept_per_visit_are_capped(conn, photo_visit):
    vid = photo_visit()
    for i in range(circle_check.KEEP + 5):
        circle_check.record(conn, vid, {"source": "pi", "say": str(i)}, {"x": 0.5, "y": 0.5, "r": 0.1})
    checks = checks_of(conn, vid)
    assert len(checks) == circle_check.KEEP and checks[-1]["say"] == str(circle_check.KEEP + 4)


def test_owner_check_spend_skips_bad_raw_json(conn, add_visit):
    vid = add_visit()
    conn.execute("UPDATE visits SET raw_json = 'not json' WHERE id = ?", (vid,))
    conn.commit()
    assert db.owner_checks_spend(conn) == {"n": 0, "usd": 0.0}


def test_visit_page_offers_the_circle_check(client, photo_visit):
    vid = photo_visit()
    html = client.get(f"/visit/{vid}?ft=1").text
    assert 'id="circle-check"' in html and "Ask the Pi" in html and 'data-source="haiku"' in html
    assert "Ask Sonnet" in html


def test_release_frees_the_model_only_when_idle(monkeypatch):
    monkeypatch.setattr(classifier, "_classifier", object())
    monkeypatch.setattr(circle_check, "_last_used", 0.0)
    circle_check.release_if_idle()
    assert classifier._classifier is None
    monkeypatch.setattr(classifier, "_classifier", object())
    import time
    monkeypatch.setattr(circle_check, "_last_used", time.monotonic())
    circle_check.release_if_idle()
    assert classifier._classifier is not None


def test_identify_note_goes_in_the_users_text():
    content = identify.user_content(["x"], datetime(2026, 11, 1, 9, 0), note="A close crop.")
    assert content[-1]["text"].endswith("A close crop.")
    assert "crop" not in identify.user_content(["x"], datetime(2026, 11, 1, 9, 0))[-1]["text"]
