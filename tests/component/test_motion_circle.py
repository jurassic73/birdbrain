"""Component tests of birdbrain/motion.py (circle): one real local dependency, a temp database or files on disk."""

import json

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from helpers.motion_circle import RIM_TOP, ZONES, animal, circle, jpeg, near, scene
from birdbrain import config, db, motion_circle as mc, watch
from birdbrain.web import app as webapp


def test_an_exposure_change_alone_is_not_an_animal():
    empty = scene()
    brighter = np.clip(empty.astype(np.float32) * 1.25 + 10, 0, 255).astype(np.uint8)
    assert circle(brighter, empty)[0] is None
    c, _ = circle(animal(brighter, RIM_TOP), empty)
    assert c and near(c, RIM_TOP)


def test_change_away_from_the_zone_that_fired_is_ignored():
    empty = scene()
    corner = (0.3, 0.3)  # inside the crop box, but far from the rim
    assert circle(animal(empty, corner), empty)[0] is None
    assert circle(animal(empty, corner), empty, trigger="ground")[0] is None  # not on the ground zone either
    ground = (0.3, 0.8)
    c, _ = circle(animal(empty, ground, size=(90, 60)), empty, trigger="ground")
    assert c and near(c, ground)


def store(img: np.ndarray, name: str) -> str:
    rel = f"mc/{name}.jpg"
    path = config.IMAGES_DIR / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(jpeg(img))
    return rel


def raw(**calls) -> str:
    return json.dumps({"calls": calls, "trigger": "rim", "zones": ZONES.to_dict()})


def test_backfill_circles_pi_named_visits_only(conn, add_visit, capsys):
    empty = scene()
    # Neighbors of the same view: two empty photos and one with a bird elsewhere (the median outvotes it).
    for i, img in enumerate((empty, empty, animal(empty, (0.62, 0.43)))):
        add_visit(captured_at=f"2026-11-01T10:0{i}:00", image_path=store(img, f"n{i}"), visitor="none",
                  species=None, raw_json=raw())
    named = add_visit(captured_at="2026-11-01T10:05:00", image_path=store(animal(empty, RIM_TOP), "pi"),
                      model="bioclip-2", raw_json=raw(settled_by="classifier", skipped_claude=True))
    claude_circle = {"x": 0.1, "y": 0.1, "r": 0.05, "claude": True}
    claude = add_visit(captured_at="2026-11-01T10:06:00", image_path=store(animal(empty, RIM_TOP), "cl"),
                       raw_json=raw(circle=claude_circle))
    before = conn.execute("SELECT raw_json FROM visits WHERE id = ?", (claude,)).fetchone()[0]

    mc.backfill(dry_run=True)
    assert "1/1 done, 0 no blob" in capsys.readouterr().out
    assert "motion_circle" not in json.loads(db.get_visit(conn, named)["raw_json"])["calls"]

    mc.backfill()
    calls = json.loads(db.get_visit(conn, named)["raw_json"])["calls"]
    assert calls["circle"]["pi"] and near(calls["circle"], RIM_TOP) and calls["motion_circle"]["source"] == "stored"
    v = db.get_visit(conn, named)
    assert v["species"] == "Dark-eyed Junco" and v["model"] == "bioclip-2"  # labels never change
    assert conn.execute("SELECT raw_json FROM visits WHERE id = ?", (claude,)).fetchone()[0] == before
    assert mc.pending(conn) == []  # tried once; --redo tries again
    assert [r["id"] for r in mc.pending(conn, redo=True)] == [named]


def test_backfill_without_an_empty_view_records_the_try(conn, add_visit):
    named = add_visit(image_path=store(animal(scene(), RIM_TOP), "alone"), model="bioclip-2",
                      raw_json=raw(settled_by="classifier", skipped_claude=True))
    mc.backfill()
    calls = json.loads(db.get_visit(conn, named)["raw_json"])["calls"]
    assert "circle" not in calls and calls["motion_circle"]["why"] == "no empty view"


def test_a_retry_that_finds_nothing_drops_only_the_pis_circle(conn, add_visit):
    named = add_visit(raw_json=raw(settled_by="classifier", circle={"x": 0.5, "y": 0.5, "r": 0.05, "pi": True}))
    mc.save(conn, named, None, {"found": False, "why": "nothing"})
    assert "circle" not in json.loads(db.get_visit(conn, named)["raw_json"])["calls"]
    claude = add_visit(raw_json=raw(circle={"x": 0.5, "y": 0.5, "r": 0.05, "claude": True}))
    mc.save(conn, claude, None, {"found": False})
    assert json.loads(db.get_visit(conn, claude)["raw_json"])["calls"]["circle"]["claude"]


def test_watcher_circles_a_visit_the_pi_named(monkeypatch, conn):
    empty = scene()
    monkeypatch.setattr(watch, "capture_burst", lambda: [jpeg(cv2.cvtColor(animal(empty, RIM_TOP), cv2.COLOR_GRAY2BGR))])
    monkeypatch.setattr(watch.zones, "current", lambda: ZONES)
    monkeypatch.setattr(watch, "presence_mode", lambda: "off")
    monkeypatch.setattr(watch.classifier, "gate_check", lambda emb, night, settled: (
        {"top": [{"name": "Dark-eyed Junco", "score": 0.95}], "model": "bioclip-2"}, ("Dark-eyed Junco", "bird")))
    monkeypatch.setattr(watch, "count_animals", lambda *a: 1)
    monkeypatch.setattr(watch.identify, "identify", lambda *a, **k: pytest.fail("Claude was asked"))
    background = cv2.GaussianBlur(cv2.resize(empty, mc.WORK, interpolation=cv2.INTER_AREA), (21, 21), 0)
    sc = watch.Scene(background, np.full(background.shape, 255, np.uint8), None)
    assert watch.handle_event(trigger="rim", scene=sc) == "bird"
    calls = json.loads(conn.execute("SELECT raw_json FROM visits ORDER BY id DESC").fetchone()[0])["calls"]
    assert calls["circle"]["pi"] and near(calls["circle"], RIM_TOP) and calls["motion_circle"]["source"] == "empty"

    monkeypatch.setattr(config, "CIRCLE_FROM_MOTION", False)  # switched off: no try at all
    assert watch.handle_event(trigger="rim", scene=sc) == "bird"
    calls = json.loads(conn.execute("SELECT raw_json FROM visits ORDER BY id DESC").fetchone()[0])["calls"]
    assert "circle" not in calls and "motion_circle" not in calls


@pytest.fixture
def client(conn):
    return TestClient(webapp.app)


def test_pages_show_the_pis_circle_or_say_there_is_none(client, add_visit):
    pi_circle = {"x": 0.5, "y": 0.43, "r": 0.02, "pi": True}
    found = add_visit(model="bioclip-2", raw_json=json.dumps({"calls": {
        "settled_by": "classifier", "skipped_claude": True, "circle": pi_circle,
        "motion_circle": {"found": True, "source": "empty", "seconds": 0.02}}}))
    html = client.get(f"/visit/{found}").text
    assert "Circle from the motion" in html and "Circle drawn on the Pi from the motion" in html
    assert 'data-tip="Pi circle:' in html and "&#34;pi&#34;: true" in html
    assert "Named on the Pi, no circle" not in html

    none = add_visit(model="bioclip-2", raw_json=json.dumps({"calls": {
        "settled_by": "classifier", "skipped_claude": True,
        "motion_circle": {"found": False, "why": "two alike", "source": "empty"}}}))
    html = client.get(f"/visit/{none}").text
    assert "Named on the Pi, no circle" in html and "two patches of change of a similar size" in html
    assert "Named on the Pi, no circle" in client.get("/").text  # the home page's latest visitor
    claude = add_visit(raw_json=json.dumps({"calls": {"circle": {"x": 0.5, "y": 0.5, "r": 0.04, "claude": True}}}))
    assert "Named on the Pi, no circle" not in client.get(f"/visit/{claude}").text
    assert "Named on the Pi, no circle" not in client.get("/").text
