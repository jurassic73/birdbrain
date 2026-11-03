"""Circle from the motion: the Pi circles the animal on visits it named itself (birdbrain/motion_circle.py)."""
import json

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from birdbrain import circle_check, config, db, motion_circle as mc, watch, zones
from birdbrain.web import app as webapp

W, H = 1920, 1080
ZONES = zones.Zones(ring=(0.5, 0.55, 0.16, 0.12, 0.13, 0.095), crop=(0.25, 0.25, 0.5, 0.55),
                    ground=((0.05, 0.6), (0.95, 0.6), (0.95, 0.98), (0.05, 0.98)), ground_crop=(0.0, 0.5, 1.0, 0.5))
RIM_TOP = (0.5, 0.55 - 0.12)  # the far rim, where a small bird perches


def scene(seed: int = 0) -> np.ndarray:
    """An empty yard in gray: busy mulch everywhere, a pale rim and dark water."""
    rng = np.random.default_rng(seed)
    img = cv2.resize(rng.integers(50, 190, (135, 240)).astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    outer, inner = config.ring_ellipses(ZONES.ring)
    config.draw_ellipse(img, outer, 200)
    config.draw_ellipse(img, inner, 70)
    return img


def animal(img: np.ndarray, at, size=(60, 42), shade: int = 25) -> np.ndarray:
    out = img.copy()
    cv2.ellipse(out, (int(at[0] * W), int(at[1] * H)), size, 0, 0, 360, shade, -1)
    return out


def jpeg(img: np.ndarray) -> bytes:
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes()


def circle(img: np.ndarray, empty: np.ndarray, trigger: str = "rim"):
    return mc.circle_for(jpeg(img), empty, ZONES, trigger, "stored")


def near(c: dict, at, tol: float = 0.02) -> bool:
    return abs(c["x"] - at[0]) < tol and abs(c["y"] - at[1]) < tol


@pytest.mark.parametrize("shade", [25, 250])  # a dark bird by day, a bright animal in infrared: either direction
def test_circles_the_animal_on_the_rim(shade):
    empty = scene()
    c, info = circle(animal(empty, RIM_TOP, shade=shade), empty)
    assert c and near(c, RIM_TOP) and 0.015 < c["r"] < 0.05 and c["pi"] is True
    assert info["found"] and info["source"] == "stored" and info["seconds"] < 1


def test_same_form_as_claudes_circle():
    empty = scene()
    c, _ = circle(animal(empty, RIM_TOP), empty)
    assert set(c) == {"x", "y", "r", "pi"} and db.parse_marks(json.dumps([c])) == [c]


def test_an_exposure_change_alone_is_not_an_animal():
    empty = scene()
    brighter = np.clip(empty.astype(np.float32) * 1.25 + 10, 0, 255).astype(np.uint8)
    assert circle(brighter, empty)[0] is None
    c, _ = circle(animal(brighter, RIM_TOP), empty)
    assert c and near(c, RIM_TOP)


def test_a_sun_patch_is_only_light():
    """A patch of sun keeps the mulch's pattern, so it doesn't count; the bird elsewhere still does."""
    empty = scene()
    sunny = empty.copy()
    patch = sunny[270:445, 576:1344].astype(np.float32)  # the mulch above the bath, up to the rim
    sunny[270:445, 576:1344] = np.clip(patch * 1.6 + 20, 0, 255).astype(np.uint8)
    assert circle(sunny, empty)[0] is None
    bird = (0.64, 0.52)
    c, _ = circle(animal(sunny, bird), empty)
    assert c and near(c, bird)


def test_two_animals_of_a_size_give_no_circle():
    empty = scene()
    c, info = circle(animal(animal(empty, (0.4, 0.45)), (0.6, 0.45)), empty)
    assert c is None and info["why"] == "two alike" and not info["found"]


def test_a_much_smaller_second_change_doesnt_matter():
    empty = scene()
    c, _ = circle(animal(animal(empty, (0.4, 0.45), size=(90, 60)), (0.6, 0.45), size=(25, 18)), empty)
    assert c and near(c, (0.4, 0.45))


def test_change_away_from_the_zone_that_fired_is_ignored():
    empty = scene()
    corner = (0.3, 0.3)  # inside the crop box, but far from the rim
    assert circle(animal(empty, corner), empty)[0] is None
    assert circle(animal(empty, corner), empty, trigger="ground")[0] is None  # not on the ground zone either
    ground = (0.3, 0.8)
    c, _ = circle(animal(empty, ground, size=(90, 60)), empty, trigger="ground")
    assert c and near(c, ground)


def test_too_big_a_change_is_the_light():
    empty = scene()
    c, info = circle(animal(empty, (0.5, 0.42), size=(400, 260)), empty)
    assert c is None and info["why"] in ("too big", "light")


def test_the_water_counts_only_for_one_compact_patch():
    empty = scene()
    sky = empty.copy()  # the reflection changed across the water (clouds): no animal
    outer, inner = config.ring_ellipses(ZONES.ring)
    config.draw_ellipse(sky, (inner[0] - 0.03, inner[1], inner[2] * 0.6, inner[3] * 0.6), 160)
    assert circle(sky, empty)[0] is None
    bather = (0.47, 0.56)
    c, info = circle(animal(empty, bather, size=(40, 30), shade=200), empty)
    assert c and near(c, bather) and info["where"] == "water"


def test_the_watchers_empty_scene_is_used_as_it_is():
    """The motion detector's background is already substream-sized, gray and blurred: it isn't blurred again."""
    empty = scene()
    background = cv2.GaussianBlur(cv2.resize(empty, mc.WORK, interpolation=cv2.INTER_AREA), (21, 21), 0)
    c, info = mc.live(jpeg(animal(empty, RIM_TOP)), background, ZONES, "rim", "2026-11-01T10:00:00", False)
    assert c and near(c, RIM_TOP) and info["source"] == "empty"


def test_fails_soft(monkeypatch):
    empty = scene()
    assert circle(empty, None)[1] == {"found": False, "why": "no empty view", "source": None}
    assert mc.circle_for(b"not a jpeg", empty, ZONES, "rim", "stored")[1]["why"] == "unreadable photo"
    monkeypatch.setattr(mc, "find", lambda *a: 1 / 0)
    c, info = circle(animal(empty, RIM_TOP), empty)
    assert c is None and info["why"] == "error"


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


def test_marks_keep_the_pi_flag_and_lessons_say_who_drew_it():
    marks = db.parse_marks(json.dumps([{"x": 0.5, "y": 0.4, "r": 0.03, "pi": True, "label": "Junco"},
                                       {"x": 0.2, "y": 0.2, "r": 0.05}]))
    assert marks[0] == {"x": 0.5, "y": 0.4, "r": 0.03, "pi": True, "label": "Junco"}
    said = db.describe_marks(marks)
    assert "drawn on the Pi where it saw motion" in said and "you drew" not in said
    assert circle_check.owner_circle(json.dumps(marks)) == (2, marks[1])  # the owner's own circle, not the Pi's


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
