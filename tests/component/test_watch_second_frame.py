"""Component tests of the watcher's shadow second-frame look (handle_event, on a temp database): when the tight
look falls just short, the next sharpest burst frame gets one more BioCLIP 2 look that is stored but never used."""

import json

import cv2
import numpy as np
import pytest

from helpers.classifier import MOTION_CIRCLE, gate_yard, sure  # noqa: F401  (gate_yard is a fixture)
from birdbrain import config, db, watch

# A burst whose frames can be told apart by colour and whose sharpness order is fixed: A (noisy block) is the
# sharpest, B (mild checker) next, C (flat) last, so ranked == [A, B, C].
H, W = 360, 640


def framed(color, texture=None) -> bytes:
    img = np.full((H, W, 3), color, np.uint8)
    if texture is not None:
        img[120:240, 220:420] = texture
    return cv2.imencode(".jpg", img)[1].tobytes()


_rng = np.random.default_rng(7)
_checker = np.where(np.indices((120, 200)).sum(axis=0) // 4 % 2 == 0, 100, 140).astype(np.uint8)
FRAME_A = framed((40, 120, 200), _rng.integers(0, 256, (120, 200, 3), dtype=np.uint8))  # red-ish, very sharp
FRAME_B = framed((200, 120, 40), np.stack([_checker] * 3, axis=-1))                      # blue-ish, mildly sharp
FRAME_C = framed((120, 200, 40))                                                          # green-ish, flat
KEYS = {2: "A", 0: "B", 1: "C"}  # dominant BGR channel -> frame


def frame_key(img: np.ndarray) -> str:
    return KEYS[int(img.mean(axis=(0, 1)).argmax())]


@pytest.fixture
def yard(gate_yard, monkeypatch):
    """gate_yard with a three-frame burst of distinguishable frames and a BioCLIP 2 fake whose answer depends on the
    frame it looks at. Knobs: `answers` (frame -> guess, None for no guess), `seen` (every look with a circle:
    (frame, circle, crop, night)), `burst`, plus gate_yard's own."""
    knobs = gate_yard
    knobs["burst"] = [FRAME_A, FRAME_B, FRAME_C]
    knobs["answers"] = {"A": sure("Dark-eyed Junco", 0.85), "B": sure("House Finch", 0.61), "C": sure("Bushtit", 0.3)}
    knobs["seen"] = []
    monkeypatch.setattr(watch, "capture_burst", lambda: list(knobs["burst"]))

    def fake_classify(img, circle, box, night, local=None, clf=None):
        key = frame_key(img)
        if circle:
            knobs["seen"].append((key, circle, box, night))
        answer = knobs["answers"].get(key)
        if isinstance(answer, Exception):
            raise answer
        return None if answer is None else {**answer, "crop": "circle", "seconds": 5.9}

    monkeypatch.setattr(watch.classifier, "classify", fake_classify)
    return knobs


def latest(conn):
    row = conn.execute("SELECT visitor, species, model, raw_json FROM visits ORDER BY id DESC").fetchone()
    return row, json.loads(row["raw_json"])["calls"]


def test_shadow_look_runs_on_the_next_sharpest_frame_when_the_tight_look_falls_just_short(yard, conn):
    assert watch.best_frames(yard["burst"], None) == [FRAME_A, FRAME_B, FRAME_C]  # the premise of every test here
    assert watch.handle_event() == "bird" and yard["asked"] == [False]  # Claude still decides
    row, calls = latest(conn)
    assert row["species"] == "Bushtit" and row["model"] == "claude-haiku-5-5" and "settled_by" not in calls
    assert calls["classifier_tight"]["top"][0]["score"] == 0.85 and calls["classifier_tight"]["why"] == "score"
    second = calls["classifier_second"]
    assert second["crop"] == "motion" and second["frame"] == "second"
    assert second["top"][0] == {"name": "House Finch", "scientific": "", "score": 0.61}  # frame B's answer
    assert "combined" in second and second["combined"] is None  # the fakes give no embeddings to average
    assert isinstance(second["seconds"], (int, float))
    assert yard["seen"] == [("A", MOTION_CIRCLE, config.CROP_ROI, False), ("B", MOTION_CIRCLE, config.CROP_ROI, False)]
    assert db.valid_json(json.dumps(calls))


@pytest.mark.parametrize("score, runs", [(0.85, True), (0.8, True), (0.79, False), (0.5, False)])
def test_shadow_look_needs_the_tight_looks_top_score_at_or_over_second_frame_min(yard, conn, score, runs):
    yard["answers"]["A"] = sure("Dark-eyed Junco", score)
    assert watch.handle_event() == "bird" and yard["asked"] == [False]
    row, calls = latest(conn)
    assert row["species"] == "Bushtit" and calls["classifier_tight"]["top"][0]["score"] == score
    assert ("classifier_second" in calls) is runs
    assert len(yard["seen"]) == (2 if runs else 1)


def test_no_shadow_look_when_the_tight_look_named_the_visit(yard, conn):
    yard["answers"]["A"] = sure("Dark-eyed Junco", 0.97)
    assert watch.handle_event() == "bird" and not yard["asked"]
    row, calls = latest(conn)
    assert row["model"] == "bioclip-2" and calls["classifier"]["crop"] == "motion"
    assert "classifier_second" not in calls and len(yard["seen"]) == 1


def test_no_shadow_look_at_night(yard, conn, monkeypatch):
    monkeypatch.setattr(watch, "is_night", lambda jpeg: True)
    assert watch.handle_event() == "bird" and yard["asked"] == [True]
    assert "classifier_second" not in latest(conn)[1] and not yard["seen"]


def test_no_shadow_look_with_a_single_frame(yard, conn):
    yard["burst"] = [FRAME_A]
    assert watch.handle_event() == "bird" and yard["asked"] == [False]
    row, calls = latest(conn)
    assert calls["classifier_tight"]["top"][0]["score"] == 0.85  # the tight look was made and fell short
    assert "classifier_second" not in calls and len(yard["seen"]) == 1


@pytest.mark.parametrize("off", [0, 0.0])
def test_no_shadow_look_when_disabled(yard, conn, monkeypatch, off):
    monkeypatch.setattr(config, "SECOND_FRAME_MIN", off)
    assert watch.handle_event() == "bird" and yard["asked"] == [False]
    row, calls = latest(conn)
    assert calls["classifier_tight"]["top"][0]["score"] == 0.85
    assert "classifier_second" not in calls and len(yard["seen"]) == 1


def test_shadow_look_never_changes_the_decision_or_claudes_call(yard, conn, monkeypatch):
    calls_made = []

    def recording_identify(*args, **kwargs):
        from birdbrain.identify import Result, Sighting
        calls_made.append((args, kwargs))
        return Result(Sighting(visitor="bird", species="Bushtit", count=1, confidence="high", notes="", circle=None,
                               also=[]), "claude-haiku-5-5", False, False, {})

    monkeypatch.setattr(watch.identify, "identify", recording_identify)
    monkeypatch.setattr(config, "SECOND_FRAME_MIN", 0)  # without the shadow look
    assert watch.handle_event() == "bird"
    without = latest(conn)
    monkeypatch.setattr(config, "SECOND_FRAME_MIN", 0.8)  # with it
    assert watch.handle_event() == "bird"
    with_look = latest(conn)
    assert "classifier_second" not in without[1] and "classifier_second" in with_look[1]
    assert len(calls_made) == 2
    (args0, kw0), (args1, kw1) = calls_made
    assert args0 == args1  # the same photo, time, crop and lessons
    assert callable(kw0.pop("local_check")) and callable(kw1.pop("local_check"))
    assert kw0 == kw1  # the same extra frames, site, confirmed species and night flag
    assert kw0["extra_frames"] == [FRAME_B, FRAME_C]
    for key in ("visitor", "species", "model"):
        assert without[0][key] == with_look[0][key]
    assert without[1]["classifier_tight"] == with_look[1]["classifier_tight"]


def test_a_failing_shadow_look_still_asks_claude_as_before(yard, conn):
    yard["answers"]["B"] = RuntimeError("model broke on the second frame")
    assert watch.handle_event() == "bird" and yard["asked"] == [False]
    row, calls = latest(conn)
    assert row["species"] == "Bushtit" and row["model"] == "claude-haiku-5-5"
    assert calls["classifier_tight"]["top"][0]["score"] == 0.85 and calls["classifier_tight"]["why"] == "score"
    assert "classifier_second" not in calls
    assert [s[0] for s in yard["seen"]] == ["A", "B"]  # it was tried, and the visit went on
