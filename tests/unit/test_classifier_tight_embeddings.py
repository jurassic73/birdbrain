"""Unit tests of birdbrain/classifier: pure functions, no files, database or network."""

import json

import cv2
import numpy as np

from helpers.classifier import LOCAL, MOTION_CIRCLE, FakeClassifier
from helpers.tight_embeddings import vec
from birdbrain import classifier, config, db


class LookClassifier(FakeClassifier):
    def look(self, bgr):
        if self.fail:
            raise RuntimeError("model broke")
        self.shapes.append(bgr.shape)
        return vec(3), self._probs


def test_with_embedding_returns_the_tight_squares_embedding_and_leaves_the_guess_alone():
    img = np.full((1080, 1920, 3), 100, np.uint8)
    clf = LookClassifier()
    circle = {"x": 0.5, "y": 0.5, "r": 0.05}
    guess, emb = classifier.with_embedding(classifier.classify, img, circle, None, False, local=LOCAL, clf=clf)
    assert np.array_equal(emb, vec(3))
    assert not any(isinstance(v, np.ndarray) for v in guess.values())
    assert guess["crop"] == "circle" and db.valid_json(json.dumps(guess))
    # the crop box or the whole frame is not a tight view; a probs-only classifier has no embedding
    assert classifier.with_embedding(classifier.classify, img, None, (0.25, 0.25, 0.5, 0.5), False,
                                     local=LOCAL, clf=clf)[1] is None
    assert classifier.with_embedding(classifier.classify, img, circle, None, False, local=LOCAL,
                                     clf=FakeClassifier())[1] is None
    # nothing leaks to the next caller
    assert classifier.with_embedding(lambda: None)[1] is None
    clf.fail = True
    assert classifier.with_embedding(classifier.run, cv2.imencode(".jpg", img)[1].tobytes(), circle, None, False,
                                     )[1] is None


def test_tight_check_hands_its_embedding_out(monkeypatch):
    monkeypatch.setattr(config, "CLASSIFIER", "gate")
    img = np.full((1080, 1920, 3), 100, np.uint8)
    jpeg = cv2.imencode(".jpg", img)[1].tobytes()
    monkeypatch.setattr(classifier, "load", lambda: LookClassifier(probs=(0.9, 0.05, 0.02, 0, 0, 0.03, 0)))
    monkeypatch.setattr(classifier, "local_species", lambda: LOCAL)
    (guess, pick), emb = classifier.with_embedding(classifier.tight_check, jpeg, MOTION_CIRCLE, None, False,
                                                   {"Dark-eyed Junco": "bird"})
    assert pick == ("Dark-eyed Junco", "bird") and guess["crop"] == "motion" and np.array_equal(emb, vec(3))
