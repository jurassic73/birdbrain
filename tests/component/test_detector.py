"""Component tests of birdbrain/detector.py: one real local dependency, a temp database or files on disk."""

import cv2
import numpy as np

from birdbrain import config, detector


def test_run_returns_none_when_off_or_model_missing(monkeypatch, tmp_path):
    jpeg = cv2.imencode(".jpg", np.zeros((10, 10, 3), np.uint8))[1].tobytes()
    monkeypatch.setattr(config, "DETECTOR", "off")
    assert detector.run(jpeg) is None
    monkeypatch.setattr(config, "DETECTOR", "shadow")
    monkeypatch.setattr(config, "DETECTOR_MODEL", tmp_path / "missing.onnx")
    assert detector.run(jpeg) is None
