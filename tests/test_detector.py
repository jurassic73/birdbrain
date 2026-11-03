from pathlib import Path

import cv2
import numpy as np
import pytest

from birdbrain import config, detector, watch
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


def test_run_returns_none_when_off_or_model_missing(monkeypatch, tmp_path):
    jpeg = cv2.imencode(".jpg", np.zeros((10, 10, 3), np.uint8))[1].tobytes()
    monkeypatch.setattr(config, "DETECTOR", "off")
    assert detector.run(jpeg) is None
    monkeypatch.setattr(config, "DETECTOR", "shadow")
    monkeypatch.setattr(config, "DETECTOR_MODEL", tmp_path / "missing.onnx")
    assert detector.run(jpeg) is None


MODEL = Path(config.ROOT / "models" / "yolox_tiny.onnx")


@pytest.mark.skipif(not MODEL.exists(), reason="model not downloaded (scripts/get_model.sh)")
def test_real_model_runs_and_finds_nothing_in_a_blank_frame():
    found = detector.Detector(MODEL).detect(np.full((360, 640, 3), 120, np.uint8), 0.15)
    assert [d for d in found if d.label in detector.VISITOR_CLASSES] == []


@pytest.mark.parametrize(
    "visitor, streak, expected",
    [
        ("bird", 5, (60, 0)),      # a real visitor resets the backoff
        ("none", 0, (20, 1)),
        ("none", 1, (40, 2)),
        ("none", 3, (160, 4)),
        ("none", 9, (300, 10)),    # capped
        (None, 0, (20, 1)),        # nothing captured counts as a false trigger
    ],
)
def test_false_trigger_backoff(monkeypatch, visitor, streak, expected):
    monkeypatch.setattr(config, "COOLDOWN_VISIT", 60)
    monkeypatch.setattr(config, "COOLDOWN_NONE", 20)
    monkeypatch.setattr(config, "COOLDOWN_NONE_MAX", 300)
    assert watch.next_cooldown(visitor, streak) == expected
