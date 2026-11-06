"""Integration tests: detector."""

from pathlib import Path

import numpy as np
import pytest

from birdbrain import config, detector


MODEL = Path(config.ROOT / "models" / "yolox_tiny.onnx")


@pytest.mark.skipif(not MODEL.exists(), reason="model not downloaded (scripts/get_model.sh)")
def test_real_model_runs_and_finds_nothing_in_a_blank_frame():
    found = detector.Detector(MODEL).detect(np.full((360, 640, 3), 120, np.uint8), 0.15)
    assert [d for d in found if d.label in detector.VISITOR_CLASSES] == []
