"""Several kinds of animal in one photo: Claude's `also` list, the owner's corrections, and the counts."""

import json
from datetime import datetime

import numpy as np

from helpers.also import BUSHTITS, junco
from helpers.clock import FROZEN_NOW
from birdbrain import alert, db
from birdbrain.identify import HAIKU


def frame() -> np.ndarray:
    return np.full((360, 640, 3), (90, 110, 80), np.uint8)


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
