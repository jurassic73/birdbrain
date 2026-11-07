"""Unit tests of the shadow second-frame look (watch.second_frame_wanted, watch.second_frame_look and
classifier.normalize): pure, with the BioCLIP 2 species check faked at classifier.run / from_embedding."""
import numpy as np
import pytest

from helpers.classifier import MOTION_CIRCLE, sure
from birdbrain import classifier, config, watch

SETTLED = {"Dark-eyed Junco": "bird", "House Finch": "bird"}
CROP = (0.1, 0.1, 0.8, 0.8)


def test_normalize_scales_to_unit_length_and_leaves_a_zero_vector_alone():
    v = classifier.normalize([3.0, 4.0])
    assert v.dtype == np.float32 and np.allclose(v, [0.6, 0.8])
    assert np.allclose(classifier.normalize(np.zeros(3)), np.zeros(3))


@pytest.mark.parametrize("score, pick, night, frames, bar, wanted", [
    (0.85, None, False, 3, 0.8, True),    # a score miss over the bar, by day, with a second frame: runs
    (0.8, None, False, 2, 0.8, True),     # exactly the bar, exactly two frames: runs
    (0.5, None, False, 3, 0.8, False),    # too far short
    (0.85, None, True, 3, 0.8, False),    # night: never
    (0.85, ("Dark-eyed Junco", "bird"), False, 3, 0.8, False),  # the tight look named the visit
    (0.85, None, False, 1, 0.8, False),   # one frame: nothing to look at
    (0.85, None, False, 3, 0.0, False),   # disabled
])
def test_second_frame_wanted_only_in_the_stated_conditions(monkeypatch, score, pick, night, frames, bar, wanted):
    monkeypatch.setattr(config, "SECOND_FRAME_MIN", bar)
    monkeypatch.setattr(classifier, "GATE_MIN_SCORE_TIGHT", 0.95)
    tight = {**sure("Dark-eyed Junco", score), "crop": classifier.TIGHT_CROP}
    assert watch.second_frame_wanted(tight, pick, SETTLED, night, frames) is wanted


def test_second_frame_wanted_needs_a_score_miss_not_a_local_or_unsettled_one(monkeypatch):
    monkeypatch.setattr(config, "SECOND_FRAME_MIN", 0.8)
    monkeypatch.setattr(classifier, "GATE_MIN_SCORE_TIGHT", 0.95)
    assert watch.second_frame_wanted(None, None, SETTLED, False, 3) is False
    # over the bar, so the miss is "unsettled" or "local" (gate_miss tests the score first): not a second-frame case
    unsettled = {**sure("Eurasian Jay", 0.99), "crop": classifier.TIGHT_CROP}
    assert watch.second_frame_wanted(unsettled, None, SETTLED, False, 3) is False
    local = {**sure("Dark-eyed Junco", 0.99), "local_share": 0.2}
    assert watch.second_frame_wanted(local, None, SETTLED, False, 3) is False


def _fake_run(monkeypatch, guess, vec):
    """classifier.run answers `guess` and leaves `vec` as the embedding with_embedding collects."""
    calls = []

    def run(jpeg, circle, box, night):
        calls.append((jpeg, circle, box, night))
        if vec is not None:
            classifier._mine.embedding = vec
        return guess

    monkeypatch.setattr(classifier, "run", run)
    return calls


def test_second_frame_look_embeds_the_second_frame_and_averages_with_the_tight_look(monkeypatch):
    second_guess = {**sure("House Finch", 0.6), "crop": "circle", "seconds": 5.9}
    calls = _fake_run(monkeypatch, second_guess, np.array([0.0, 1.0, 0.0], np.float32))
    asked = []

    def from_embedding(emb, night, local=None, clf=None):
        asked.append((emb, night))
        return {"model": classifier.MODEL_NAME, "crop": "box", "night": night, "seconds": 0.0, "local_share": 0.8,
                "top": [{"name": "Dark-eyed Junco", "scientific": "", "score": 0.91}], "background": 0.0}

    monkeypatch.setattr(classifier, "from_embedding", from_embedding)
    frames = [b"sharpest", b"second", b"third"]
    record, vec = watch.second_frame_look(frames, MOTION_CIRCLE, CROP, False, np.array([2.0, 0.0, 0.0], np.float32))
    assert calls == [(b"second", MOTION_CIRCLE, CROP, False)]  # the next sharpest frame, on the same circle
    assert np.allclose(vec, [0.0, 1.0, 0.0])
    assert record["crop"] == classifier.TIGHT_CROP and record["frame"] == "second"
    assert record["top"][0]["name"] == "House Finch" and record["top"][0]["score"] == 0.6
    assert isinstance(record["seconds"], float)
    # the combined guess comes from the normalized mean of the two unit vectors
    (emb, night), = asked
    assert night is False and np.allclose(emb, [1 / np.sqrt(2), 1 / np.sqrt(2), 0.0])
    assert record["combined"] == {"top": [{"name": "Dark-eyed Junco", "scientific": "", "score": 0.91}], "local_share": 0.8}


def test_second_frame_look_without_a_tight_embedding_stores_the_guess_and_no_combined(monkeypatch):
    calls = _fake_run(monkeypatch, {**sure("House Finch", 0.6), "crop": "circle"}, None)
    monkeypatch.setattr(classifier, "from_embedding", lambda *a, **k: pytest.fail("no embeddings to combine"))
    record, vec = watch.second_frame_look([b"a", b"b"], MOTION_CIRCLE, CROP, False, None)
    assert len(calls) == 1 and vec is None
    assert record["combined"] is None and record["frame"] == "second" and record["top"][0]["name"] == "House Finch"


def test_second_frame_look_is_fail_soft(monkeypatch):
    def broken(*a, **k):
        raise RuntimeError("model broke")

    monkeypatch.setattr(classifier, "run", broken)
    assert watch.second_frame_look([b"a", b"b"], MOTION_CIRCLE, CROP, False, None) == (None, None)
    _fake_run(monkeypatch, None, None)  # no guess at all
    assert watch.second_frame_look([b"a", b"b"], MOTION_CIRCLE, CROP, False, None) == (None, None)
    assert watch.second_frame_look([b"a"], MOTION_CIRCLE, CROP, False, None) == (None, None)  # no second frame
