"""Component tests of the shadow second-frame look in the watcher (watch.second_frame_look in handle_event): when
it runs, what it stores as raw_json.calls.classifier_second, and that it changes neither the decision nor the Claude
call. A temp database (gate_yard); the camera, BioCLIP 2, the motion circle and Claude are fakes."""
import json

import numpy as np
import pytest

from helpers.classifier import MOTION_CIRCLE, gate_yard, one_hot, sure  # noqa: F401  (a fixture)
from birdbrain import classifier, config, watch


def latest(conn):
    row = conn.execute("SELECT species, model, raw_json FROM visits ORDER BY id DESC").fetchone()
    return row, json.loads(row["raw_json"])["calls"]


@pytest.fixture
def shadow(gate_yard):
    """gate_yard with a tight look that falls just short of the bar (0.85): the second-frame case."""
    gate_yard["tight"] = sure("Dark-eyed Junco", 0.85)
    return gate_yard


def test_a_tight_miss_just_short_of_the_bar_gets_a_second_frame_look(shadow, conn):
    assert watch.handle_event() == "bird" and shadow["asked"] == [False]
    # the tight look on the sharpest frame, then the second-frame look on the same circle
    assert shadow["tight_calls"] == [(MOTION_CIRCLE, config.CROP_ROI, False)] * 2
    row, calls = latest(conn)
    assert row["species"] == "Bushtit" and row["model"] == "claude-haiku-5-5"  # Claude decided, as before
    second = calls["classifier_second"]
    assert second["crop"] == classifier.TIGHT_CROP and second["frame"] == "second"
    assert second["top"][0] == {"name": "Dark-eyed Junco", "scientific": "", "score": 0.85}
    assert second["combined"] is None  # the fake BioCLIP 2 leaves no embedding, so there is nothing to average
    assert isinstance(second["seconds"], float)
    assert calls["classifier_tight"]["why"] == "score" and "settled_by" not in calls and "circle" not in calls
    assert "vec" not in json.dumps(calls)


def test_the_combined_guess_averages_the_two_embeddings(shadow, conn, monkeypatch):
    embeddings = iter([one_hot(0, 1.0), np.array([0.8, 0.0, 0.6, 0.0, 0.0, 0.0, 0.0], np.float32)])

    def fake_classify(img, circle, box, night, local=None, clf=None):
        shadow["tight_calls"].append((circle, box, night))
        classifier._mine.embedding = next(embeddings)
        return {**shadow["tight"], "crop": "circle", "seconds": 5.9}

    monkeypatch.setattr(classifier, "classify", fake_classify)
    assert watch.handle_event() == "bird" and shadow["asked"] == [False]
    row, calls = latest(conn)
    combined = calls["classifier_second"]["combined"]
    assert set(combined) == {"top", "local_share"}
    assert combined["top"][0]["name"] == "Dark-eyed Junco"  # the junco direction dominates the averaged vector
    assert combined["local_share"] is not None and row["species"] == "Bushtit"


@pytest.mark.parametrize("change", ["score", "night", "named", "one frame", "disabled", "unsettled"])
def test_no_second_frame_look_outside_the_stated_conditions(shadow, conn, monkeypatch, change):
    if change == "score":
        shadow["tight"] = sure("Dark-eyed Junco", 0.5)
    elif change == "night":
        monkeypatch.setattr(watch, "is_night", lambda jpeg: True)
    elif change == "named":
        shadow["tight"] = sure("Dark-eyed Junco", 0.97)
    elif change == "one frame":
        burst = watch.capture_burst()[:1]
        monkeypatch.setattr(watch, "capture_burst", lambda: burst)
    elif change == "disabled":
        monkeypatch.setattr(config, "SECOND_FRAME_MIN", 0.0)
    elif change == "unsettled":
        shadow["tight"] = sure("House Finch", 0.99)  # over the bar: the miss is "unsettled", not the score
    assert watch.handle_event() == "bird"
    row, calls = latest(conn)
    assert "classifier_second" not in calls
    assert len(shadow["tight_calls"]) <= 1  # at most the tight look itself
    if change == "named":
        assert row["model"] == classifier.MODEL_NAME and not shadow["asked"]
    else:
        assert shadow["asked"]


def test_the_claude_call_is_the_same_with_and_without_the_look(shadow, conn, monkeypatch):
    seen = []
    real = watch.identify.identify

    def spy(*args, **kwargs):
        seen.append((args[0], args[2:], kwargs))  # the frame, the crop and lessons, every keyword (not the clock)
        return real(*args, **kwargs)

    monkeypatch.setattr(watch.identify, "identify", spy)
    assert watch.handle_event() == "bird"
    assert "classifier_second" in latest(conn)[1]
    monkeypatch.setattr(config, "SECOND_FRAME_MIN", 0.0)
    assert watch.handle_event() == "bird"
    assert "classifier_second" not in latest(conn)[1]
    with_look, without = seen
    assert with_look[0] == without[0] and with_look[1] == without[1]
    assert set(with_look[2]) == set(without[2])
    for key in with_look[2]:
        if key != "local_check":  # a closure made per event
            assert with_look[2][key] == without[2][key], key


def test_a_failing_second_frame_look_loses_nothing(shadow, conn, monkeypatch):
    calls_made = []
    real = classifier.classify

    def flaky(img, circle, box, night, local=None, clf=None):
        calls_made.append(circle)
        if len(calls_made) == 2:  # the second-frame look
            raise RuntimeError("model broke")
        return real(img, circle, box, night, local, clf)

    monkeypatch.setattr(classifier, "classify", flaky)
    assert watch.handle_event() == "bird" and shadow["asked"] == [False]
    row, calls = latest(conn)
    assert row["species"] == "Bushtit" and "classifier_second" not in calls
    assert calls["classifier_tight"]["why"] == "score"
