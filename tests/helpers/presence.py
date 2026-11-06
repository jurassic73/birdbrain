"""Shared helpers for the presence tests (used by more than one test module)."""
import json

import cv2
import numpy as np
import pytest

from helpers.clock import FROZEN_NOW
from birdbrain import db, presence


DIM = 16


def vec(animal: bool, rng) -> np.ndarray:
    """Fake embeddings: animals lean one way, empty frames the other."""
    v = rng.normal(0, 0.3, DIM)
    v[0] += 1.0 if animal else -1.0
    return v / np.linalg.norm(v)


CROP = (0.25, 0.25, 0.5, 0.5)


INSIDE = {"x": 0.5, "y": 0.5, "r": 0.05}  # a circle in the middle of the crop box


ABOVE = {"x": 0.5, "y": 0.12, "r": 0.05}  # one above it (the dove of visit 693: only its tail was in the box)


def left_out(conn) -> dict:
    return dict(presence.examples(conn)["left_out"])


def found_animal(conn, vid, circle=INSIDE, visitor="squirrel"):
    """The owner corrects a false trigger to an animal and circles it on the photo."""
    raw = json.loads(db.get_visit(conn, vid)["raw_json"] or "{}")
    db.set_raw_json(conn, vid, {**raw, "zones": {"crop": list(CROP)}, "trigger": "rim"})
    db.correct_visit(conn, vid, visitor, None, None, [circle] if circle else None)


@pytest.fixture
def yard(conn, add_visit):
    """12 visitors and 40 empty frames with embeddings."""
    rng = np.random.default_rng(1)
    for i in range(52):
        animal = i < 12
        vid = add_visit(**({} if animal else {"visitor": "none", "species": None}))
        presence.store(conn, vid, vec(animal, rng))
    return rng


class FakeClf:
    """Embeds a crop as an animal when it's bright (a bird landed), else as empty."""

    def __init__(self, rng):
        self.rng, self.calls = rng, 0

    def embed(self, bgr):
        self.calls += 1
        return vec(bgr.mean() > 128, self.rng)


def burst_visit(conn, add_visit, brightness):
    from birdbrain import images

    frames = [cv2.imencode(".jpg", np.full((90, 160, 3), b, np.uint8))[1].tobytes() for b in brightness]
    image_rel, thumb_rel = images.store_frame(frames[0], FROZEN_NOW)
    images.store_burst(frames, image_rel)
    vid = add_visit(visitor="none", species=None, model=presence.MODEL_NAME, image_path=image_rel,
                    thumb_path=thumb_rel, raw_json=json.dumps({"zones": {"crop": list(CROP)}, "burst": {"best": 0}}))
    presence.store(conn, vid, vec(False, np.random.default_rng(5)))  # the sharpest frame looked empty
    return vid


def on_frame(circle: dict, frame: int) -> dict:
    return {**circle, "frame": frame}
