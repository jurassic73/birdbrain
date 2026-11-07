"""Component tests of the shadow second-frame look in the watcher (handle_event) that test_watch_second_frame.py
does not cover: the averaged guess stored end to end when both looks leave an embedding, and the "unsettled" miss
that gets no second frame. A temp database (gate_yard); the camera, BioCLIP 2, the motion circle and Claude are fakes."""
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


def test_no_second_frame_look_when_the_miss_is_unsettled_rather_than_the_score(shadow, conn):
    """Over the bar but not a species settled here: gate_miss says "unsettled", not "score", so no second frame.
    (The score, night, named, single-frame and disabled cases live in test_watch_second_frame.py.)"""
    shadow["tight"] = sure("House Finch", 0.99)
    assert watch.handle_event() == "bird" and shadow["asked"]
    row, calls = latest(conn)
    assert calls["classifier_tight"]["why"] == "unsettled" and "classifier_second" not in calls
    assert len(shadow["tight_calls"]) == 1  # the tight look itself, nothing more

