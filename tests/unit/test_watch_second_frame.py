"""Unit tests of birdbrain/watch.py's shadow second-frame look (watch.second_frame_look): pure functions, no files,
database or network. The BioCLIP 2 model is a fake that hands out a known embedding and known probabilities."""

import json

import cv2
import numpy as np
import pytest

from helpers.classifier import LABELS, LOCAL, one_hot
from birdbrain import classifier, config, db, watch

H, W = 180, 320
CIRCLE = {"x": 0.5, "y": 0.5, "r": 0.1, "pi": True}
# Two burst frames told apart by their dominant colour: the sharpest (blue-ish, "first") and the next one ("second").
FIRST = cv2.imencode(".jpg", np.full((H, W, 3), (200, 60, 60), np.uint8))[1].tobytes()
SECOND = cv2.imencode(".jpg", np.full((H, W, 3), (60, 60, 200), np.uint8))[1].tobytes()
JUNCO = (0.9, 0.05, 0.02, 0.0, 0.0, 0.03, 0.0)  # probabilities over LABELS: Dark-eyed Junco on top


class LookTextClassifier:
    """A BioCLIP 2 stand-in with both halves the look needs: look() gives (embedding, probs) for a patch and
    `text` (one-hot per label) scores an embedding, so from_embedding's answer is known exactly."""
    labels = LABELS
    text = np.eye(len(LABELS), dtype=np.float32)

    def __init__(self, vec, probs=JUNCO, fail=False):
        self.vec, self._probs, self.fail = vec, np.array(probs, np.float32), fail
        self.seen: list[str] = []  # which frame each look was of
        self.shapes: list[tuple] = []

    def look(self, bgr):
        if self.fail:
            raise RuntimeError("model broke")
        self.seen.append("first" if bgr.mean(axis=(0, 1)).argmax() == 0 else "second")
        self.shapes.append(bgr.shape)
        return self.vec, self._probs

    def probs(self, bgr):
        return self.look(bgr)[1]


@pytest.fixture
def yard(monkeypatch):
    monkeypatch.setattr(config, "CLASSIFIER", "gate")
    monkeypatch.setattr(classifier, "local_species", lambda: LOCAL)

    def use(clf):
        monkeypatch.setattr(classifier, "load", lambda: clf)
        return clf

    return use


def normalize(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def test_second_frame_look_embeds_the_next_sharpest_frame_on_the_same_circle(yard):
    clf = yard(LookTextClassifier(one_hot(0)))
    record, vec = watch.second_frame_look([FIRST, SECOND], CIRCLE, None, False, None)
    assert clf.seen == ["second"]  # one look, of ranked[1], never of the frame the tight look already saw
    side = classifier.circle_square(CIRCLE, W, H)[2]
    assert clf.shapes == [(side, side, 3)]  # the same square around the same circle as the tight look
    assert record["crop"] == classifier.TIGHT_CROP and record["frame"] == "second"
    assert record["top"][0]["name"] == "Dark-eyed Junco" and record["model"] == classifier.MODEL_NAME
    assert isinstance(record["seconds"], (int, float))
    assert record["combined"] is None  # no tight embedding to average with: the second guess alone is kept
    assert np.array_equal(vec, one_hot(0))
    assert db.valid_json(json.dumps(record))  # nothing numpy leaks into the stored record


def test_combined_is_from_embedding_of_the_mean_of_both_embeddings(yard):
    # Both looks lean junco over house finch by the same small margin; their large "background" parts point opposite
    # ways and cancel in the mean, so the normalized average is far surer than either look alone or than the
    # unnormalized mean (the softmax runs at LOGIT_SCALE): only the contract's formula gives `expected`.
    tight_vec = normalize(np.array([0.3, 0, 0.29, 0, 0, 0, 0.9], np.float32))
    second_vec = normalize(np.array([0.3, 0, 0.29, 0, 0, 0, -0.9], np.float32))
    clf = yard(LookTextClassifier(second_vec, probs=(0.3, 0.05, 0.6, 0.0, 0.0, 0.05, 0.0)))
    expected = classifier.from_embedding(normalize((tight_vec + second_vec) / 2), False, local=LOCAL, clf=clf)
    alone = classifier.from_embedding(tight_vec, False, local=LOCAL, clf=clf)
    unnormalized = classifier.from_embedding((tight_vec + second_vec) / 2, False, local=LOCAL, clf=clf)
    assert expected["top"][0]["score"] not in (alone["top"][0]["score"], unnormalized["top"][0]["score"])  # premise
    record, vec = watch.second_frame_look([FIRST, SECOND], CIRCLE, None, False, tight_vec)
    assert record["combined"]["top"] == expected["top"]
    assert record["combined"]["local_share"] == expected["local_share"]
    assert record["combined"]["top"][0]["name"] == "Dark-eyed Junco"
    # the second frame's own guess is still the record's guess (House Finch on top here), not the average
    assert record["top"][0]["name"] == "House Finch" and record["frame"] == "second"
    assert np.array_equal(vec, second_vec)
    assert db.valid_json(json.dumps(record))


def test_combined_is_none_when_there_is_no_tight_embedding_but_the_guess_is_kept(yard):
    yard(LookTextClassifier(one_hot(2)))
    record, vec = watch.second_frame_look([FIRST, SECOND], CIRCLE, None, False, None)
    assert record is not None and record["combined"] is None
    assert record["top"][0]["name"] == "Dark-eyed Junco" and record["crop"] == classifier.TIGHT_CROP
    assert vec is not None


def test_second_frame_look_fails_soft(yard, monkeypatch):
    yard(LookTextClassifier(one_hot(0), fail=True))  # the model breaks: no guess, no embedding
    assert watch.second_frame_look([FIRST, SECOND], CIRCLE, None, False, one_hot(0)) == (None, None)
    yard(LookTextClassifier(one_hot(0)))
    assert watch.second_frame_look([FIRST, b"not a jpeg"], CIRCLE, None, False, one_hot(0)) == (None, None)

    def broken(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(classifier, "run", broken)  # an exception anywhere inside: (None, None), never a raise
    assert watch.second_frame_look([FIRST, SECOND], CIRCLE, None, False, one_hot(0)) == (None, None)
