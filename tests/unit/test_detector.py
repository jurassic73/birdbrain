"""Unit tests of birdbrain/detector: pure functions, no files, database or network."""

import numpy as np
import pytest

from birdbrain import detector
from birdbrain.detector import COCO, INPUT


def raw_output(cell: int, cls: str, obj: float = 0.9, cls_score: float = 0.9, wh: float = 2.0) -> np.ndarray:
    """A fake YOLOX output with one confident box at grid cell `cell` (stride 8 grid), everything else empty."""
    out = np.zeros((1, 3549, 85), np.float32)
    out[0, :, 4] = 0.0
    out[0, cell, 0:2] = 0.5             # offset within the cell
    out[0, cell, 2:4] = np.log(wh)      # width/height in stride units, before exp
    out[0, cell, 4] = obj
    out[0, cell, 5 + COCO.index(cls)] = cls_score
    return out


def test_coco_has_80_classes_in_order():
    assert len(COCO) == 80 and COCO[0] == "person" and COCO[14] == "bird" and COCO[21] == "bear"


def test_preprocess_letterboxes_on_gray():
    img = np.full((360, 640, 3), 200, np.uint8)
    tensor, ratio = detector.preprocess(img)
    assert tensor.shape == (1, 3, INPUT, INPUT) and tensor.dtype == np.float32
    assert ratio == INPUT / 640
    assert tensor[0, 0, -1, 0] == 114 and tensor[0, 0, 0, 0] == 200  # padding below, image on top


def test_decode_maps_a_box_back_to_source_pixels():
    found = detector.decode(raw_output(cell=52 * 10 + 20, cls="bird"), ratio=0.5, min_score=0.15)
    assert len(found) == 1 and found[0].label == "bird" and found[0].score == pytest.approx(0.81)
    x0, y0, x1, y1 = found[0].box
    # cell (20, 10) at stride 8, +0.5 offset, 16 px wide; scaled back by 1/ratio = 2
    assert (x0, y0, x1, y1) == (312, 152, 344, 184)


def test_decode_drops_low_scores():
    assert detector.decode(raw_output(cell=100, cls="bird", obj=0.3, cls_score=0.3), ratio=1.0, min_score=0.15) == []


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
    assert detector.animal_box(b"not a jpeg", None, min_score=0.15)[1]["why"] == "nothing"


def test_verdict_counts_only_visitor_classes():
    found = [detector.Detection("bicycle", 0.9, (0, 0, 1, 1)), detector.Detection("bird", 0.4, (0, 0, 1, 1))]
    v = detector.verdict(found, 0.123)
    assert v["animal"] is True and v["best"] == {"label": "bird", "score": 0.4} and v["seconds"] == 0.12
    assert detector.verdict([detector.Detection("sink", 0.5, (0, 0, 1, 1))], 0.1)["animal"] is False
