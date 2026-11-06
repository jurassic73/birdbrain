"""Shared helpers for the circle check tests (used by more than one test module)."""
import io
import json

import pytest
from PIL import Image

from helpers.clock import FROZEN_NOW
from birdbrain import db, images


MARKS = json.dumps([{"x": 0.2, "y": 0.2, "r": 0.1, "claude": True}, {"x": 0.5, "y": 0.5, "r": 0.05}])


@pytest.fixture
def photo_visit(add_visit):
    """A false trigger with a real 640 x 360 photo on disk."""
    def add(**fields) -> int:
        buf = io.BytesIO()
        Image.new("RGB", (640, 360), (90, 120, 200)).save(buf, "JPEG")
        image_rel, thumb_rel = images.store_frame(buf.getvalue(), FROZEN_NOW)
        row = {"visitor": "none", "species": None, "model": "bioclip-presence", "confidence": "medium"}
        return add_visit(image_path=image_rel, thumb_path=thumb_rel, **{**row, **fields})
    return add


def checks_of(conn, vid):
    return json.loads(db.get_visit(conn, vid)["raw_json"] or "{}")["calls"]["owner_checks"]
