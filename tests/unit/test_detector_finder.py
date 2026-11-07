"""Unit tests of the YOLOX finder's entry points in birdbrain/detector.py (to_frame_box, animal_box): the builder's own
regression tests for what the test smith's in test_detector.py do not pin (a None crop, a box poking outside the crop,
a crop off the frame, a broken session, an unreadable JPEG). Pure: the model file is only checked for existence and
the ONNX session is a fake."""

import numpy as np
import pytest

from birdbrain import detector


def test_to_frame_box_without_a_crop_and_with_a_box_poking_outside_the_crop():
    # Without a crop the detection is already in frame pixels (a 1920x1080 frame).
    assert detector.to_frame_box((192, 216, 384, 432), None, (1080, 1920)) == [0.1, 0.2, 0.1, 0.2]
    # A box poking outside the crop (YOLOX can) is held inside it: the crop runs from (192, 216) to (1152, 864).
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


def test_animal_box_with_a_crop_off_the_frame_sees_nothing_without_an_inference(monkeypatch, model_present, small_jpeg):
    fake = FakeDetector([detector.Detection("bird", 0.9, (0, 0, 10, 10))])
    monkeypatch.setattr(detector, "_detector", fake)
    box, info = detector.animal_box(small_jpeg, [2.0, 2.0, 0.1, 0.1], min_score=0.15)
    assert box is None and info["why"] == "nothing" and fake.calls == []


def test_animal_box_with_a_broken_session_or_an_unreadable_jpeg_never_raises(monkeypatch, small_jpeg, model_present):
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
