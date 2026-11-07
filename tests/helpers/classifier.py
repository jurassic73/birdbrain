"""Fakes for the BioCLIP 2 species check: a fixed label list, classifiers with known answers, and `gate_yard`,
a whole yard in gate mode for watcher tests (a fixture; import it into a test module to use it)."""
import json

import cv2
import numpy as np
import pytest

from birdbrain import classifier, config, watch

LABELS = [("Junco hyemalis", "Dark-eyed Junco", True), ("Junco hyemalis", "Oregon Junco", False),
          ("Haemorhous mexicanus", "house finch", True), ("Haemorhous purpureus", "Purple Finch", True),
          ("Picoides pubescens", "Downy Woodpecker", True), ("Garrulus glandarius", "Eurasian Jay", True),
          ("background", "", True)]
LOCAL = {"Junco hyemalis": "Dark-eyed Junco", "Haemorhous mexicanus": "House Finch",
         "Dryobates pubescens": "Downy Woodpecker"}


class FakeClassifier:
    labels = LABELS

    def __init__(self, probs=(0.2, 0.3, 0.1, 0.05, 0.05, 0.3, 0.0), fail=False):
        self._probs, self.fail, self.shapes = np.array(probs, np.float32), fail, []

    def probs(self, bgr):
        if self.fail:
            raise RuntimeError("model broke")
        self.shapes.append(bgr.shape)
        return self._probs


class FakeTextClassifier:
    """Embeddings as one-hot vectors over LABELS, so the species scores are known exactly."""
    labels = LABELS
    text = np.eye(len(LABELS), dtype=np.float32)


def one_hot(i, k=1.0):
    v = np.zeros(len(LABELS), np.float32)
    v[i] = k
    return v


def sure(name, score):
    """A one-species guess with the given score (test_classifier's guess() scores every name 0.5)."""
    return {"model": classifier.MODEL_NAME, "crop": "circle", "night": False, "seconds": 0.05, "local_share": 0.9,
            "top": [{"name": name, "scientific": "", "score": score}]}


MOTION_CIRCLE = {"x": 0.48, "y": 0.5, "r": 0.04, "pi": True}
MOTION_FOUND = {"found": True, "source": "empty", "where": "land", "seconds": 0.02}


@pytest.fixture
def gate_yard(monkeypatch, conn, add_visit):
    """A yard in gate mode with Dark-eyed Junco settled, a fake BioCLIP 2 for both looks and a stubbed motion circle.
    Returns a dict of knobs: the embedding for the first look, the tight look's answer, the motion circle, the
    animals counted, whether Claude was asked, and what the tight look and the circle finder were called with."""
    for _ in range(3):
        add_visit(species="Dark-eyed Junco")
    frame = np.full((1080, 1920, 3), (40, 120, 200), np.uint8)
    burst = [cv2.imencode(".jpg", frame)[1].tobytes()] * 3
    knobs = {"emb": one_hot(0, 0.01), "tight": sure("Dark-eyed Junco", 0.97), "circle": (MOTION_CIRCLE, MOTION_FOUND),
             "animals": 1, "boxes": [], "yolox": (None, {"why": "nothing", "seconds": 0.3}),
             "asked": [], "tight_calls": [], "circle_calls": [], "yolox_calls": []}
    monkeypatch.setattr(watch, "capture_burst", lambda: burst)
    monkeypatch.setattr(watch, "presence_mode", lambda: "gate")
    monkeypatch.setattr(config, "CLASSIFIER", "gate")
    monkeypatch.setattr(watch.classifier, "box_embedding", lambda jpeg, zones, trigger: knobs["emb"])
    monkeypatch.setattr(watch.presence, "check", lambda conn, e, view=None: {"model": "bioclip-presence", "score": 0.9,
                                                                             "threshold": 0.4, "empty": False})
    monkeypatch.setattr(watch.presence, "store", lambda conn, vid, e: None)

    def fake_count(scene, jpeg, crop, detail=None):
        if detail is not None:  # the crowd counter's box per counted animal, for the blob finder
            detail.extend(knobs["boxes"])
        return knobs["animals"]

    monkeypatch.setattr(watch, "count_animals", fake_count)

    def fake_animal_box(jpeg, crop, min_score):
        knobs["yolox_calls"].append((crop, min_score))
        return knobs["yolox"]

    monkeypatch.setattr(watch.finder.detector, "animal_box", fake_animal_box)
    monkeypatch.setattr(classifier, "load", lambda: FakeTextClassifier())
    monkeypatch.setattr(classifier, "local_species", lambda: LOCAL)

    def fake_classify(img, circle, box, night, local=None, clf=None):
        if circle:  # the tight look; the scorecard look after Claude has no circle here (fake_identify draws none)
            knobs["tight_calls"].append((circle, box, night))
        return None if knobs["tight"] is None else {**knobs["tight"], "crop": "circle", "seconds": 5.9}

    monkeypatch.setattr(classifier, "classify", fake_classify)

    def fake_live(jpeg, empty, z, trigger, captured_at, night):
        knobs["circle_calls"].append((trigger, night))
        return knobs["circle"]

    monkeypatch.setattr(watch.motion_circle, "live", fake_live)

    def fake_identify(*args, **kwargs):
        from birdbrain.identify import Result, Sighting
        knobs["asked"].append(kwargs.get("night"))
        return Result(Sighting(visitor="bird", species="Bushtit", count=1, confidence="high", notes="", circle=None,
                               also=[]), "claude-haiku-5-5", False, False, {})

    monkeypatch.setattr(watch.identify, "identify", fake_identify)
    return knobs


def guess(*names, night=False):
    return {"model": classifier.MODEL_NAME, "crop": "circle", "night": night, "seconds": 0.05,
            "top": [{"name": n, "scientific": "", "score": 0.5} for n in names]}


def raw(*names, night=False):
    return json.dumps({"calls": {"classifier": guess(*names, night=night)}})
