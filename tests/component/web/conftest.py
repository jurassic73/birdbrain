"""Fixtures shared by the web app's component tests: visits with real photo files, and a scripted live camera."""
import io
import json

import pytest
from PIL import Image

from helpers.clock import FROZEN_NOW
from birdbrain import db, images
from birdbrain.web import app as webapp


@pytest.fixture
def stored_visit(add_visit):
    """A visit with real image files on disk, so delete can be checked."""
    def add(**fields) -> int:
        buf = io.BytesIO()
        Image.new("RGB", (640, 360), (90, 120, 200)).save(buf, "JPEG")
        image_rel, thumb_rel = images.store_frame(buf.getvalue(), FROZEN_NOW)
        return add_visit(image_path=image_rel, thumb_path=thumb_rel, **fields)
    return add


@pytest.fixture
def burst_visit(stored_visit):
    """A false trigger with a 4-frame burst on disk; Claude saw frame 2. Returns (id, image_rel, burst paths)."""
    def add(**fields):
        vid = stored_visit(visitor="none", species=None, raw_json=json.dumps({"burst": {"count": 4, "best": 1, "fps": 4}}),
                           **fields)
        with db.session() as conn:
            image_rel = db.get_visit(conn, vid)["image_path"]
        frames = []
        for shade in (40, 90, 140, 190):  # each frame its own gray, so the thumbnail says which frame it came from
            buf = io.BytesIO()
            Image.new("RGB", (640, 360), (shade, shade, shade)).save(buf, "JPEG")
            frames.append(buf.getvalue())
        return vid, image_rel, images.store_burst(frames, image_rel)
    return add


@pytest.fixture
def live(monkeypatch):
    """Fake camera grabs: each call pops the next scripted result."""
    state = {"jpeg": None, "at": 0.0, "tried": 0.0}
    monkeypatch.setattr(webapp, "_live", state)
    script: list[bytes] = []
    grabs = []

    async def grab():
        grabs.append(1)
        return script.pop(0) if script else b""

    monkeypatch.setattr(webapp, "grab_live_frame", grab)
    return state, script, grabs


@pytest.fixture
def frame_file(tmp_path, monkeypatch):
    monkeypatch.setattr(webapp, "live_file", lambda: tmp_path / "live.jpg")
    return tmp_path / "live.jpg"
