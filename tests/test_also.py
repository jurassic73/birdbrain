"""Several kinds of animal in one photo: Claude's `also` list, the owner's corrections, and the counts."""

import io
import json
from datetime import date, datetime

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from conftest import FROZEN_NOW
from birdbrain import alert, db, identify
from birdbrain.identify import HAIKU, SONNET, Also, Circle, Sighting
from birdbrain.web import app as webapp

WHEN = datetime(2026, 10, 25, 9, 30)


def jpeg() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (1920, 1080), (90, 120, 200)).save(buf, "JPEG")
    return buf.getvalue()


def frame() -> np.ndarray:
    return np.full((360, 640, 3), (90, 110, 80), np.uint8)

BUSHTITS = [{"visitor": "bird", "species": "Bushtit", "count": 2, "confidence": "high",
             "circle": {"x": 0.6, "y": 0.4, "r": 0.03, "claude": True}}]


def junco(confidence="high", also=()) -> Sighting:
    return Sighting(visitor="bird", species="Dark-eyed Junco", count=1, confidence=confidence, notes="",
                    circle=Circle(x=0.5, y=0.5, r=0.1), also=list(also))


def bushtit(confidence="high", **kw) -> Also:
    return Also(**{"visitor": "bird", "species": "Bushtit", "count": 2, "confidence": confidence,
                   "circle": Circle(x=0.2, y=0.2, r=0.05), **kw})


@pytest.fixture
def fake_ask(monkeypatch):
    calls = []

    def install(answers):
        def ask(model, images_b64, captured_at, lessons=None, site=None, night=False):
            calls.append(model)
            return answers[model], {"model": model}
        monkeypatch.setattr(identify, "ask", ask)
        return calls
    return install


@pytest.fixture
def client(conn):
    with TestClient(webapp.app) as c:
        yield c


# --- identify ---

def test_other_animals_are_kept_with_circles_on_the_full_frame(fake_ask):
    fake_ask({HAIKU: junco(also=[bushtit()])})
    r = identify.identify(jpeg(), WHEN, crop=(0.2, 0.4, 0.5, 0.5))
    assert r.raw["also"] == [{"visitor": "bird", "species": "Bushtit", "count": 2, "confidence": "high",
                              "circle": {"x": 0.3, "y": 0.5, "r": 0.025, "claude": True}}]
    assert (r.escalated, r.needs_review) == (False, False)


def test_also_drops_repeats_of_the_main_animal_and_of_each_other():
    s = junco(also=[bushtit(), bushtit(count=1), bushtit(species="dark-eyed junco")])
    assert [a["species"] for a in identify.also_rows(s)] == ["Bushtit"]
    assert identify.also_rows(Sighting(visitor="none", species=None, count=0, confidence="high", notes="",
                                       circle=None, also=[bushtit()])) == []


def test_a_low_guess_at_the_second_animal_gets_sonnets_look(fake_ask):
    calls = fake_ask({HAIKU: junco(also=[bushtit("low")]), SONNET: junco(also=[bushtit("medium")])})
    r = identify.identify(jpeg(), WHEN)
    assert calls == [HAIKU, SONNET] and (r.model, r.needs_review) == (SONNET, False)
    assert r.raw["also"][0]["confidence"] == "medium"


def test_still_low_on_the_second_animal_goes_to_review(fake_ask):
    fake_ask({HAIKU: junco(also=[bushtit("low")]), SONNET: junco(also=[bushtit("low")])})
    assert identify.identify(jpeg(), WHEN).needs_review is True


def test_the_prompt_asks_for_every_kind():
    assert "also:" in identify.SYSTEM and "list the others here" not in identify.SYSTEM


# --- counting ---

def test_other_animals_count_for_their_species_everywhere(conn, add_visit):
    vid = add_visit(also=json.dumps(BUSHTITS))
    add_visit(species="Bushtit")
    today = FROZEN_NOW.date().isoformat()
    assert [r["id"] for r in db.list_visits(conn, species="Bushtit")][-1] == vid
    assert db.count_visits(conn, species="Bushtit") == 2 and db.count_visits(conn) == 2
    rows = {r["label"]: r for r in db.species_summary(conn)}
    assert (rows["Bushtit"]["visits"], rows["Dark-eyed Junco"]["visits"]) == (2, 1)
    assert db.species_row(conn, "Bushtit")["visits"] == 2
    assert db.species_names(conn) == ["Bushtit", "Dark-eyed Junco"]
    t = db.today_summary(conn)
    assert (t["visits"], t["species"]) == (2, 2)
    assert {r["label"]: r["n"] for r in t["critters"]} == {"Bushtit": 2, "Dark-eyed Junco": 1}
    assert db.activity(conn, today)["species"] == 2 and db.totals(conn)["species"] == 2
    assert sum(n for _, n in db.daily_counts(conn, species="Bushtit")) == 2


def test_species_photo_prefers_visits_where_it_was_the_main_animal(conn, add_visit):
    add_visit(also=json.dumps(BUSHTITS), thumb_path="mixed_t.jpg", captured_at=FROZEN_NOW.isoformat())
    add_visit(species="Bushtit", confidence="medium", thumb_path="own_t.jpg", captured_at="2026-01-01T09:00:00")
    assert db.species_row(conn, "Bushtit")["thumb_path"] == "own_t.jpg"


def test_false_triggers_never_count_their_other_animals(conn, add_visit):
    add_visit(visitor="none", species=None, also=json.dumps(BUSHTITS))
    assert db.species_names(conn) == [] and db.count_visits(conn, species="Bushtit") == 0


# --- corrections ---

def form(visit, names, counts, visitor=None):
    return db.plan_also(visit, names, [str(c) for c in counts], visitor)


def test_plan_also(conn, add_visit):
    v = db.get_visit(conn, add_visit(also=json.dumps(BUSHTITS)))
    assert form(v, ["Bushtit"], [2]) is None                  # same as Claude's
    assert form(v, [], []) == []                              # nothing else was there
    assert form(v, ["Bushtit", " Spotted  Towhee ", "squirrel", "", "bushtit"], [3, 1, "x", 1, 1]) == [
        {"visitor": "bird", "species": "Bushtit", "count": 3, "circle": BUSHTITS[0]["circle"]},
        {"visitor": "bird", "species": "Spotted Towhee", "count": 1},
        {"visitor": "squirrel", "species": None, "count": 1},
    ]
    assert form(v, ["Bushtit"], [2], visitor="none") == []
    plain = db.get_visit(conn, add_visit())
    assert form(plain, [], []) is None and form(plain, ["x"], [1], visitor="none") is None


def test_owner_list_wins_and_undo_brings_back_claudes(client, conn, add_visit):
    vid = add_visit(also=json.dumps(BUSHTITS))
    client.post(f"/visit/{vid}/correct", data={"visitor": "bird", "species": "Dark-eyed Junco",
                                               "also_species": ["Spotted Towhee"], "also_count": ["1"]})
    v = db.get_visit(conn, vid)
    assert [db.also_label(a) for a in db.also_of(v)] == ["Spotted Towhee"]
    assert db.count_visits(conn, species="Bushtit") == 0 and db.count_visits(conn, species="Spotted Towhee") == 1
    line = db.recent_lessons(conn)[0]
    assert 'you said "Dark-eyed Junco" with "2 Bushtit"' in line
    assert 'The owner says also there: "1 Spotted Towhee".' in line
    page = client.get(f"/visit/{vid}").text
    assert "Corrected by you. Claude said: Dark-eyed Junco with Bushtit x2." in page
    client.post(f"/visit/{vid}/clear-correction")
    assert db.count_visits(conn, species="Bushtit") == 1 and db.recent_lessons(conn) == []


def test_removing_every_other_animal_is_a_lesson(client, conn, add_visit):
    vid = add_visit(also=json.dumps(BUSHTITS))
    client.post(f"/visit/{vid}/correct", data={"visitor": "bird", "species": "Dark-eyed Junco"})
    assert db.recent_lessons(conn)[0].endswith("The owner says no other animal was there.")
    assert db.totals(conn)["corrected"] == 1


def test_editing_a_lesson_keeps_the_owners_other_animals(client, conn, add_visit):
    vid = add_visit(also=json.dumps(BUSHTITS))
    client.post(f"/visit/{vid}/correct", data={"also_species": ["Bushtit"], "also_count": ["5"]})
    client.post(f"/lessons/{vid}/edit", data={"visitor": "bird", "species": "Song Sparrow", "feedback": "x"})
    assert db.also_of(db.get_visit(conn, vid))[0]["count"] == 5


# --- pages ---

def test_cards_and_visit_page_show_the_other_animals(client, conn, add_visit):
    circle = {"x": 0.45, "y": 0.65, "r": 0.05, "claude": True}
    vid = add_visit(also=json.dumps(BUSHTITS), raw_json=json.dumps({"calls": {"circle": circle}}))
    assert '<span class="also-name">+ <span class="sp-name" data-species="Bushtit"' in client.get("/").text
    page = client.get(f"/visit/{vid}").text
    assert '<p class="also-line">With <span class="sp-name" data-species="Bushtit"' in page
    assert 'name="also_species"' in page and 'value="Bushtit"' in page
    marks = json.loads(page.split('data-marks="')[1].split('"')[0].replace("&#34;", '"'))
    assert [m.get("label") for m in marks] == ["Dark-eyed Junco", "Bushtit"]
    client.post(f"/visit/{vid}/correct", data={"feedback": "ok", "marks": json.dumps(marks),
                                               "also_species": ["Bushtit"], "also_count": ["2"]})
    assert "circle 2 (you drew it around the Bushtit)" in db.recent_lessons(conn)[0]
    assert "Bushtit" in client.get("/species/Bushtit").text


def test_one_circle_needs_no_label(client, add_visit):
    vid = add_visit(raw_json=json.dumps({"calls": {"circle": {"x": 0.4, "y": 0.4, "r": 0.05, "claude": True}}}))
    assert webapp.start_marks(db.get_visit(webapp.db.connect(), vid)) == [
        {"x": 0.4, "y": 0.4, "r": 0.05, "claude": True}]


# --- alerts and the watcher ---

def test_a_new_species_seen_only_beside_another_is_not_new_again(conn, add_visit):
    db.set_value(conn, "ntfy_topic", "t")
    add_visit(also=json.dumps(BUSHTITS))
    assert alert.reason(conn, "bird", "Bushtit", datetime(2026, 10, 25, 14, 0)) is None
    assert alert.reason(conn, "bird", "Spotted Towhee", datetime(2026, 10, 25, 14, 0)) == "New species: Spotted Towhee!"


def test_watcher_stores_the_other_animals_and_alerts_on_a_new_one(monkeypatch, conn, add_visit):
    import cv2
    from birdbrain import watch
    from birdbrain.identify import Result

    add_visit()  # the junco is old news
    db.set_value(conn, "ntfy_topic", "t")
    burst = [cv2.imencode(".jpg", frame())[1].tobytes()] * 3
    monkeypatch.setattr(watch, "capture_burst", lambda: burst)
    monkeypatch.setattr(watch, "presence_mode", lambda: "off")
    monkeypatch.setattr(watch.identify, "identify", lambda *a, **k: Result(
        junco(), HAIKU, False, False, {"also": [{k: v for k, v in BUSHTITS[0].items()}]}))
    sent = []
    monkeypatch.setattr(watch.alert, "send", lambda topic, title, *a: sent.append(title))
    assert watch.handle_event() == "bird"
    v = db.latest_visit(conn)
    assert [db.also_label(a) for a in db.also_of(v)] == ["Bushtit"]
    assert sent == ["New species: Bushtit!"]
