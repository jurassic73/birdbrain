import io
import json
import time
from datetime import datetime

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from helpers.clock import FROZEN_NOW
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
        image_rel, thumb_rel = images.store_frame(buf.getvalue(), FROZEN_NOW)
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
    assert circle_check.pi_say("unsure", sure) == "Maybe Dark-eyed Junco, but BioCLIP 2 isn't sure."
    assert circle_check.pi_say("empty", []) == "BioCLIP 2 didn't find an animal here."


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


def test_a_circle_on_a_burst_frame_is_checked_on_that_frame(client, conn, photo_visit, fake_pi, monkeypatch):
    """The photo is blue; burst frame 4 is red and smaller (frames can be another size: marks are fractions)."""
    seen = []

    class Clf(FakeClassifier):
        def embed(self, bgr):
            seen.append((bgr.shape, tuple(int(c) for c in bgr.reshape(-1, 3).mean(0))))
            return self.vec

    fake_pi(Clf(i=0))
    vid = photo_visit(raw_json=json.dumps({"burst": {"count": 6, "best": 0, "fps": 4}}))
    frames = []
    for i in range(6):
        buf = io.BytesIO()
        Image.new("RGB", (320, 180), (220, 30, 30) if i == 3 else (30, 220, 30)).save(buf, "JPEG")
        frames.append(buf.getvalue())
    images.store_burst(frames, db.get_visit(conn, vid)["image_path"])
    on4 = json.dumps([{"x": 0.2, "y": 0.2, "r": 0.1, "claude": True}, {"x": 0.5, "y": 0.5, "r": 0.05, "frame": 4}])
    r = client.post(f"/visit/{vid}/check-circle/pi", data={"marks": on4}).json()
    assert r["circle_n"] == 2 and r["frame"] == 4
    shape, (b, g, red) = seen[-1]
    assert red > 150 and b < 100  # cropped from frame 4 (OpenCV's BGR), not the blue photo
    stored = checks_of(conn, vid)[-1]
    assert stored["frame"] == 4 and stored["circle"] == {"x": 0.5, "y": 0.5, "r": 0.05}
    assert "burst frame 4" in client.get(f"/visit/{vid}?ft=1").text  # the check's line under Your circle checks

    # Haiku crops the same frame, using that frame's own size.
    sizes = []

    def ask(model, images_b64, captured_at, lessons=None, site=None, note="", night=False):
        return None, {"model": model, "input_tokens": 1, "output_tokens": 0}
    monkeypatch.setattr(identify, "ask", ask)
    monkeypatch.setattr(circle_check, "square", lambda c, w, h, margin=circle_check.MARGIN: sizes.append((w, h)) or (0, 0, 1, 1))
    client.post(f"/visit/{vid}/check-circle/haiku", data={"marks": on4})
    assert sizes == [(320, 180)]

    # Frame 1 is the photo Claude saw (best 0): that circle is checked on the photo itself.
    r = client.post(f"/visit/{vid}/check-circle/pi", data={"marks": json.dumps([{"x": 0.5, "y": 0.5, "r": 0.05, "frame": 1}])}).json()
    assert r["frame"] is None and seen[-1][1][0] > 150  # blue
    # A frame that's gone: the photo instead, and the check says so by having no frame.
    r = client.post(f"/visit/{vid}/check-circle/pi", data={"marks": json.dumps([{"x": 0.5, "y": 0.5, "r": 0.05, "frame": 9}])}).json()
    assert r["frame"] is None and seen[-1][1][0] > 150 and "frame" not in checks_of(conn, vid)[-1]


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

    def ask(model, images_b64, captured_at, lessons=None, site=None, note="", night=False):
        import base64
        seen.append((model, Image.open(io.BytesIO(base64.b64decode(images_b64[0]))).size, note))
        if model == identify.HAIKU:
            return identify.Sighting(visitor="bird", species="house finch", count=1, confidence="medium",
                                     notes="Red wash on the head.", circle=None, also=[]), \
                {"model": model, "input_tokens": 1500, "output_tokens": 100}
        return identify.Sighting(visitor="none", species=None, count=0, confidence="high", notes="A leaf.", circle=None, also=[]), \
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


def test_visit_page_offers_the_circle_check(client, photo_visit):
    vid = photo_visit()
    html = client.get(f"/visit/{vid}?ft=1").text
    assert 'id="circle-check"' in html and "Ask BioCLIP 2" in html and 'data-source="haiku"' in html
    assert "Ask Sonnet" in html and "Ask the Pi" not in html
    # the section's name, and the buttons' tip while no circle of the owner's is drawn yet
    assert "Manual circle and ask" in html and "What's in your circle" not in html
    assert 'data-no-circle-tip="Draw a circle on the photo first' in html
    assert "Free, runs on the Pi" in html  # the free local check still says so


def test_visit_page_has_the_unsaved_changes_guard_for_admins_only(client, photo_visit):
    vid = photo_visit()
    html = client.get(f"/visit/{vid}?ft=1").text
    assert "You have unsaved changes" in html and "window.bbUnsaved" in html
    assert 'alt: "Discard", cancel: "Keep editing"' in html
    # the pop-up asks the page before closing; the page answers through the guard
    assert '"birdbrain:closing"' in client.get(f"/visit/{vid}?ft=1&embed=1").text
    assert '"birdbrain:closing"' in client.get("/false-triggers").text


def test_release_frees_the_model_only_when_idle(monkeypatch):
    monkeypatch.setattr(classifier, "_classifier", object())
    # monotonic() counts from boot, so "0.0" is only long ago on a machine that's been up for IDLE seconds
    monkeypatch.setattr(circle_check, "_last_used", time.monotonic() - circle_check.IDLE - 1)
    circle_check.release_if_idle()
    assert classifier._classifier is None
    monkeypatch.setattr(classifier, "_classifier", object())
    monkeypatch.setattr(circle_check, "_last_used", time.monotonic())
    circle_check.release_if_idle()
    assert classifier._classifier is not None


def test_identify_note_goes_in_the_users_text():
    content = identify.user_content(["x"], datetime(2026, 11, 1, 9, 0), note="A close crop.")
    assert content[-1]["text"].endswith("A close crop.")
    assert "crop" not in identify.user_content(["x"], datetime(2026, 11, 1, 9, 0))[-1]["text"]


def test_every_pi_click_is_a_row_and_averages_use_the_good_ones(client, conn, photo_visit, fake_pi):
    assert db.pi_check_average(conn) == {"n": 0, "avg": None, "recent_avg": None}
    vid = photo_visit()
    # a click with no model on this machine fails: a row with ok=0, not in the average
    assert "isn't available" in client.post(f"/visit/{vid}/check-circle/pi", data={"marks": MARKS}).json()["error"]
    (row,) = conn.execute("SELECT * FROM pi_checks").fetchall()
    assert row["ok"] == 0 and row["visit_id"] == vid and row["seconds"] >= 0 and row["at"]
    assert db.pi_check_average(conn)["n"] == 0
    fake_pi(FakeClassifier(i=0))
    r = client.post(f"/visit/{vid}/check-circle/pi", data={"marks": MARKS}).json()
    assert r["wall_seconds"] >= 0 and r["pi_average"]["n"] == 1
    ok = conn.execute("SELECT * FROM pi_checks WHERE ok = 1").fetchone()
    assert ok["seconds"] == r["wall_seconds"] and ok["model_seconds"] == r["seconds"]
    assert conn.execute("SELECT COUNT(*) FROM pi_checks").fetchone()[0] == 2
    # rows survive deleting the visit
    db.delete_visit(conn, vid)
    assert conn.execute("SELECT COUNT(*) FROM pi_checks").fetchone()[0] == 2
    assert db.pi_check_average(conn)["n"] == 1


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


def test_visit_page_shows_the_usual_time_beside_the_timer(client, conn, photo_visit):
    vid = photo_visit()
    for sec in (10, 20, 30):
        db.record_pi_check(conn, sec)
    html = client.get(f"/visit/{vid}?ft=1").text
    assert "(usually ~20 s)" in html
    assert "Average of the last 3 Ask BioCLIP 2 checks on this Pi (all-time 20 s over 3 checks)." in html


def test_claude_checks_say_when_the_frame_is_infrared(client, conn, photo_visit, monkeypatch):
    """A night visit's crop goes to Haiku or Sonnet with the same night note every identification sends."""
    seen = []

    def ask(model, images_b64, captured_at, lessons=None, site=None, note="", night=False):
        seen.append(night)
        return None, {"model": model, "input_tokens": 1, "output_tokens": 0}
    monkeypatch.setattr(identify, "ask", ask)
    client.post(f"/visit/{photo_visit(is_night=1)}/check-circle/haiku", data={"marks": MARKS})
    client.post(f"/visit/{photo_visit(is_night=0)}/check-circle/sonnet", data={"marks": MARKS})
    assert seen == [True, False]


def test_a_visit_cut_down_to_its_thumbnail_is_checked_on_the_thumbnail(client, conn, photo_visit, monkeypatch):
    """The photo budget may leave a real visit with only its thumbnail (photo_level 'thumb', not deleted): the page
    still shows it with the Ask buttons, so the check crops the thumbnail, as Ask (ask-species) does."""
    import base64

    from birdbrain import config

    vid = photo_visit(visitor="bird", species="Bushtit", model="claude-haiku-5-5", confidence="medium")
    v = db.get_visit(conn, vid)
    (config.IMAGES_DIR / v["image_path"]).unlink()
    conn.execute("UPDATE visits SET photo_level = 'thumb' WHERE id = ?", (vid,))
    conn.commit()
    v = db.get_visit(conn, vid)
    jpeg, frame = circle_check.jpeg_of(v)
    assert jpeg is not None and frame is None
    sizes = []

    def ask(model, images_b64, captured_at, lessons=None, site=None, note="", night=False):
        sizes.append(Image.open(io.BytesIO(base64.b64decode(images_b64[0]))).size)
        return None, {"model": model, "input_tokens": 1, "output_tokens": 0}
    monkeypatch.setattr(identify, "ask", ask)
    r = client.post(f"/visit/{vid}/check-circle/haiku", data={"marks": MARKS}).json()
    assert "error" not in r and r["frame"] is None and sizes and sizes[0][0] >= classifier.MIN_SIDE
    # A photo that really is gone still says so.
    conn.execute("UPDATE visits SET photo_deleted_at = '2026-11-01' WHERE id = ?", (vid,))
    conn.commit()
    assert "deleted" in client.post(f"/visit/{vid}/check-circle/haiku", data={"marks": MARKS}).json()["error"]
