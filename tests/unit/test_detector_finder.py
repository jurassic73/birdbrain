"""Unit tests of the YOLOX finder's entry points in birdbrain/detector.py (to_frame_box, animal_box): the builder's own
regression tests, beside the test smith's in test_detector.py. Pure: the model file is only checked for existence and
the ONNX session is a fake."""

import numpy as np
import pytest

from birdbrain import detector


def test_to_frame_box_puts_a_crop_detection_back_on_the_frame():
    # A 1920x1080 frame; the crop box runs from (192, 216) to (1152, 864). A detection at (96, 108)-(288, 324) in the
    # crop is at (288, 324)-(480, 540) on the frame: 0.15, 0.3, width 0.1, height 0.2 of it.
    assert detector.to_frame_box((96, 108, 288, 324), (192, 216, 1152, 864), (1080, 1920, 3)) == [0.15, 0.3, 0.1, 0.2]
    # Without a crop the detection is already in frame pixels.
    assert detector.to_frame_box((192, 216, 384, 432), None, (1080, 1920)) == [0.1, 0.2, 0.1, 0.2]
    # A box poking outside the crop (YOLOX can) is held inside it.
    assert detector.to_frame_box((-50, -20, 2000, 900), (192, 216, 1152, 864), (1080, 1920)) == [0.1, 0.2, 0.5, 0.6]


class FakeDetector:
    def __init__(self, found):
        self.found, self.calls = found, []

    def detect(self, bgr, min_score):
        self.calls.append((bgr.shape, min_score))
        return self.found


@pytest.fixture
def small_jpeg():
    import cv2
    ok, buf = cv2.imencode(".jpg", np.full((36, 64, 3), 90, np.uint8))
    assert ok
    return buf.tobytes()


@pytest.fixture
def model_present(monkeypatch):
    """The model file "exists" (this test file stands in for it: only its existence is checked) and no session is
    built: `animal_box` gets the detector from the module's cache."""
    from pathlib import Path
    monkeypatch.setattr(detector.config, "DETECTOR_MODEL", Path(__file__))
    monkeypatch.setattr(detector.config, "DETECTOR", "off")  # the finder must not care about the retired knob


def test_animal_box_returns_the_best_animal_box_and_skips_people(monkeypatch, model_present, small_jpeg):
    D = detector.Detection
    fake = FakeDetector([D("person", 0.9, (0, 0, 10, 10)), D("bird", 0.5, (8, 4, 24, 16)), D("cat", 0.3, (0, 0, 4, 4)),
                         D("teddy_bear", 0.8, (0, 0, 4, 4))])
    monkeypatch.setattr(detector, "_detector", fake)
    box, info = detector.animal_box(small_jpeg, [0.25, 0.25, 0.5, 0.5], min_score=0.2)
    # The crop is (16, 9)-(48, 27) of the 64x36 frame: the bird at (8, 4)-(24, 16) in it is (24, 13)-(40, 25) on the frame.
    assert box == [round(24 / 64, 4), round(13 / 36, 4), 0.25, round(12 / 36, 4)]
    assert info["label"] == "bird" and info["score"] == 0.5 and info["seconds"] >= 0
    assert fake.calls == [((18, 32, 3), 0.2)]  # one inference, on the crop, at the finder's own bar


def test_animal_box_says_why_when_it_finds_nothing(monkeypatch, model_present, small_jpeg):
    fake = FakeDetector([detector.Detection("person", 0.9, (0, 0, 10, 10)), detector.Detection("bench", 0.9, (0, 0, 10, 10))])
    monkeypatch.setattr(detector, "_detector", fake)
    box, info = detector.animal_box(small_jpeg, None, min_score=0.15)
    assert box is None and info["why"] == "nothing" and "seconds" in info
    assert fake.calls == [((36, 64, 3), 0.15)]  # no crop: the whole frame
    assert detector.animal_box(small_jpeg, [2.0, 2.0, 0.1, 0.1], min_score=0.15)[1]["why"] == "nothing"  # crop off the frame
    assert len(fake.calls) == 1


def test_animal_box_without_a_model_or_with_a_broken_one_never_raises(monkeypatch, small_jpeg, model_present):
    from pathlib import Path
    monkeypatch.setattr(detector.config, "DETECTOR_MODEL", Path("/nonexistent/yolox_tiny.onnx"))
    assert detector.animal_box(small_jpeg, None, min_score=0.15) == (None, {"why": "no model"})
    monkeypatch.setattr(detector.config, "DETECTOR_MODEL", Path(__file__))

    class Broken:
        def detect(self, bgr, min_score):
            raise RuntimeError("onnx broke")
    monkeypatch.setattr(detector, "_detector", Broken())
    box, info = detector.animal_box(small_jpeg, None, min_score=0.15)
    assert box is None and info["why"] == "error"
    monkeypatch.setattr(detector, "_detector", FakeDetector([]))
    # a JPEG that can't be decoded is its own reason (the Motion detector's wording), not "saw no animal"
    box, info = detector.animal_box(b"not a jpeg", None, min_score=0.15)
    assert box is None and info["why"] == "unreadable photo" and "seconds" in info
