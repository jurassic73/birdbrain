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


def test_verdict_counts_only_visitor_classes():
    found = [detector.Detection("bicycle", 0.9, (0, 0, 1, 1)), detector.Detection("bird", 0.4, (0, 0, 1, 1))]
    v = detector.verdict(found, 0.123)
    assert v["animal"] is True and v["best"] == {"label": "bird", "score": 0.4} and v["seconds"] == 0.12
    assert detector.verdict([detector.Detection("sink", 0.5, (0, 0, 1, 1))], 0.1)["animal"] is False


# --- the finder chain's YOLOX finder (birdbrain/finder.py): to_frame_box and animal_box ---------------------------

def test_to_frame_box_maps_a_crop_detection_back_to_frame_fractions():
    # A 1920x1080 frame; the crop box (0.25, 0.25, 0.5, 0.5) is pixels (480, 270, 1440, 810), as zones.pixel_box
    # gives it; a detection at 96..288 x 54..162 inside the crop is 576..768 x 324..432 in the frame.
    box = detector.to_frame_box((96, 54, 288, 162), (480, 270, 1440, 810), (1080, 1920))
    assert box == pytest.approx([0.3, 0.3, 0.1, 0.1], abs=1e-3)
    # The whole frame as the crop: plain fractions, no offset.
    box = detector.to_frame_box((192, 108, 384, 216), (0, 0, 1920, 1080), (1080, 1920))
    assert box == pytest.approx([0.1, 0.1, 0.1, 0.1], abs=1e-3)
    # A substream-sized frame with a crop that starts away from the origin: the offset is in crop pixels, the
    # fractions in frame size.
    box = detector.to_frame_box((32, 18, 96, 54), (160, 90, 480, 270), (360, 640))
    assert box == pytest.approx([0.3, 0.3, 0.1, 0.1], abs=1e-3)
    assert isinstance(box, list) and len(box) == 4


class FakeDetector:
    """Stands in for the ONNX session: hands back the detections given, above min_score, plus the class-agnostic
    objects (which must never become a finder's box)."""

    def __init__(self, found, objs=()):
        self.found, self.objs, self.calls = found, list(objs), []

    def detect(self, bgr, min_score):
        self.calls.append((bgr.shape, min_score))
        return sorted([d for d in self.found if d.score >= min_score], key=lambda d: d.score, reverse=True)

    def detect_all(self, bgr, min_score):
        return self.detect(bgr, min_score), self.objs


def _fake_model(monkeypatch, fake):
    from pathlib import Path
    from birdbrain import config
    monkeypatch.setattr(config, "DETECTOR_MODEL", Path(detector.__file__))  # a file that exists: the model is there
    monkeypatch.setattr(config, "DETECTOR", "off")  # the retired watcher knob must not switch the finder off
    monkeypatch.setattr(detector, "_detector", fake)
    monkeypatch.setattr(detector, "Detector", lambda path: fake)


def test_animal_box_returns_the_best_visitor_box_and_ignores_people_and_objects(monkeypatch):
    from helpers.images import flat_jpeg
    D = detector.Detection
    fake = FakeDetector([D("person", 0.95, (0, 0, 50, 50)), D("bicycle", 0.99, (10, 10, 60, 60)),
                         D("teddy_bear", 0.9, (5, 5, 40, 40)), D("cat", 0.6, (32, 18, 96, 54)),
                         D("bird", 0.4, (100, 100, 120, 120)), D("dog", 0.2, (0, 0, 10, 10))],
                        objs=[[0.99, 0.1, 0.1, 0.2, 0.2]])  # a rock, with high objectness: never a finder's box
    _fake_model(monkeypatch, fake)
    # a 640x360 frame; the crop (0.25, 0.25, 0.5, 0.5) is pixels (160, 90, 480, 270): the cat's crop box
    # (32, 18, 96, 54) is (192, 108, 256, 144) in the frame
    box, info = detector.animal_box(flat_jpeg(360, 640), (0.25, 0.25, 0.5, 0.5), min_score=0.3)
    assert box == pytest.approx([0.3, 0.3, 0.1, 0.1], abs=1e-3)
    assert info["label"] == "cat" and info["score"] == pytest.approx(0.6) and info["seconds"] >= 0
    assert fake.calls == [((180, 320, 3), 0.3)]  # one inference, on the crop, at the finder's own bar


def test_animal_box_says_nothing_when_only_people_or_non_visitors_are_seen(monkeypatch):
    from helpers.images import flat_jpeg
    D = detector.Detection
    fake = FakeDetector([D("person", 0.95, (0, 0, 50, 50)), D("bench", 0.9, (0, 0, 50, 50))],
                        objs=[[0.99, 0.1, 0.1, 0.2, 0.2], [0.8, 0.5, 0.5, 0.1, 0.1]])
    _fake_model(monkeypatch, fake)
    box, info = detector.animal_box(flat_jpeg(360, 640), None, min_score=0.15)
    assert box is None and info["why"] == "nothing" and "seconds" in info
    assert fake.calls == [((360, 640, 3), 0.15)]  # no crop box: the whole frame


def test_animal_box_without_a_model_file_is_a_soft_miss(monkeypatch):
    from pathlib import Path
    from helpers.images import flat_jpeg
    from birdbrain import config
    monkeypatch.setattr(config, "DETECTOR_MODEL", Path("/nonexistent/birdbrain-unit-tests/yolox_tiny.onnx"))
    monkeypatch.setattr(detector, "_detector", None)
    monkeypatch.setattr(detector, "Detector", lambda path: pytest.fail("tried to load a model that isn't there"))
    box, info = detector.animal_box(flat_jpeg(360, 640), (0.25, 0.25, 0.5, 0.5), min_score=0.15)
    assert box is None and info["why"] == "no model"
