"""Unit tests of birdbrain/classifier: pure functions, no files, database or network."""

import json

import numpy as np
import pytest

from helpers.presence_bowl import CROP, GROUND_CROP, LIVE, frame
from birdbrain import classifier, zones
from birdbrain.web import app as webapp


def test_the_bowl_square_holds_the_whole_rim_and_a_bird_above_it():
    x0, y0, side = classifier.presence_square((1080, 1920, 3), LIVE)
    W, H = 1920, 1080
    assert x0 <= 0.302 * W and x0 + side >= 0.634 * W                      # the outer rim, both ends
    assert y0 <= (0.5347 - 0.1175 - classifier.BIRD_BAND) * H and y0 + side >= (0.5347 + 0.1175) * H
    assert side <= H and x0 >= 0 and y0 >= 0 and x0 + side <= W and y0 + side <= H
    # The old view kept only the crop box's middle square: 0.335..0.615 of the width.
    assert x0 < 0.335 * W - 50 and x0 + side > 0.615 * W + 30


def test_the_square_slides_inside_the_frame_and_clips_to_its_height():
    near = {"ring": [0.05, 0.9, 0.1, 0.1, 0.08, 0.08]}
    x0, y0, side = classifier.presence_square((1080, 1920), near)
    assert x0 == 0 and y0 + side == 1080 and y0 >= 0
    huge = {"ring": [0.5, 0.5, 0.45, 0.6, 0.4, 0.5]}
    x0, y0, side = classifier.presence_square((1080, 1920), huge)
    assert side == 1080 and y0 == 0 and 0 <= x0 <= 1920 - 1080
    turned = {"ring": [0.5, 0.5, 0.2, 0.05, 0.15, 0.04, 0.5, 0.5, 90.0]}  # turned upright: taller than wide
    x0, y0, side = classifier.presence_square((1080, 1920), turned)
    assert y0 <= 0.5 * 1080 - 0.2 * 1920 + 1 and side >= 0.4 * 1920
    assert classifier.presence_square((1080, 1920), {"crop": list(CROP)}) is None


def test_the_square_is_the_same_on_a_smaller_copy_of_the_frame():
    big = classifier.presence_square((1080, 1920), LIVE)
    small = classifier.presence_square((540, 960), LIVE)
    assert all(abs(a / 2 - b) <= 1 for a, b in zip(big, small))
    x, y, w, h = classifier.presence_view_box(LIVE, "rim")
    assert abs(x * 1920 - big[0]) <= 1 and abs(y * 1080 - big[1]) <= 1
    assert abs(w * 1920 - big[2]) <= 1 and abs(h * 1080 - big[2]) <= 1


def test_padding_keeps_the_box_unchanged_and_fills_with_the_clip_mean():
    img = frame(90, 160)
    box = (0.1, 0.2, 0.5, 0.4)
    patch, kind = classifier.presence_patch(img, {"crop": list(box)}, "rim")
    x0, y0, x1, y1 = zones.pixel_box(img.shape, box)
    w, h = x1 - x0, y1 - y0
    assert kind == "padded" and patch.shape[:2] == (w, w)
    top = (w - h) // 2
    assert np.array_equal(patch[top:top + h], img[y0:y1, x0:x1])
    assert (patch[:top] == classifier.PAD_FILL).all() and (patch[top + h:] == classifier.PAD_FILL).all()
    # Padding normalizes to about nothing.
    rgb = np.array(classifier.PAD_FILL[::-1], np.float32) / 255
    assert np.abs((rgb - classifier.MEAN) / classifier.STD).max() < 0.02


def test_nothing_of_the_patch_is_cut_off_on_the_way_into_the_model():
    img = frame()
    for z, trigger in ((LIVE, "rim"), (LIVE, "ground"), ({"crop": list(CROP)}, "rim")):
        patch, _ = classifier.presence_patch(img, z, trigger)
        patch = patch.copy()
        k = max(4, patch.shape[1] // 40)
        patch[:, :k] = (0, 0, 255)   # red down the far left
        patch[:, -k:] = (255, 0, 0)  # blue down the far right
        x = classifier.to_input(patch)[0]
        assert x.shape == (3, 224, 224)
        red = (x[0] > 1.5) & (x[2] < -1)
        blue = (x[2] > 1.5) & (x[0] < -1)
        assert red[:, 0].mean() > 0.9 and blue[:, -1].mean() > 0.9


def test_a_bird_on_the_rims_left_end_is_now_in_view():
    img = frame()
    W, H = 1920, 1080
    bx, by = int(0.31 * W), int((0.5347 - 0.03) * H)
    img[by - 20:by + 20, bx - 20:bx + 20] = 250  # a bright bird on the rim, in the strip the old view cut off
    old, _ = classifier.crop_for(img, None, CROP)
    h, w = old.shape[:2]
    cut = old[:, (w - h) // 2:(w - h) // 2 + h]  # to_input's old center crop
    assert (old >= 250).all(axis=2).sum() > 1000 and (cut >= 250).all(axis=2).sum() == 0
    patch, kind = classifier.presence_patch(img, LIVE, "rim")
    assert kind == "bowl" and (patch >= 250).all(axis=2).sum() >= 1600


def test_the_view_box_for_drawing_and_coverage():
    assert classifier.presence_view_box(LIVE, "ground") == GROUND_CROP
    assert classifier.presence_view_box({"crop": list(CROP)}, "rim") == CROP
    assert classifier.presence_view_box({}, "rim") is None
    x, y, w, h = classifier.presence_view_box(LIVE, None)  # watch --once: a birdbath view
    assert abs(w * 16 / 9 - h) < 0.01


def test_the_species_check_step_names_the_padded_view_by_trigger():
    def where(trigger, crop):
        raw = {"trigger": trigger, "calls": {"classifier": {"top": [{"name": "Bushtit", "score": 0.9}], "crop": crop}}}
        return webapp.decision_steps(json.dumps(raw))["steps"][-1]["crop"]
    assert where("ground", "padded") == "on the ground crop"
    assert where("rim", "padded") == "on the crop box"
    assert where("rim", "frame") == "on the whole frame"
    assert where("rim", "box") == "on the crop box"            # older visits and classifier.run keep theirs
    assert where("rim", "circle") == "around Claude's circle"


OFF_FRAME = [[1.2, 0.1, 0.3, 0.3], [0.2, 0.1, 0, 0.3], [0.2, 0.1, 0.3, -0.1], [0.2, 0.1, 0.3], None, "box",
             [0.2, float("nan"), 0.3, 0.3]]


@pytest.mark.parametrize("box", OFF_FRAME)
def test_an_unusable_box_is_the_whole_frame_everywhere(box):
    img = frame(90, 160)
    for z, trigger in (({"crop": box}, "rim"), ({"ground_crop": box, "crop": list(CROP)}, "ground")):
        assert classifier.view_kind(z, trigger) == "frame"
        assert classifier.presence_view_box(z, trigger) is None
        patch, kind = classifier.presence_patch(img, z, trigger)
        assert kind == "frame" and patch.shape[:2] == (160, 160)


def test_a_box_partly_off_the_frame_is_its_part_in_view():
    z = {"crop": [0.8, 0.5, 0.4, 0.6]}
    assert classifier.view_kind(z, "rim") == "padded"
    assert classifier.presence_view_box(z, "rim") == (0.8, 0.5, 0.2, 0.5)
    patch, kind = classifier.presence_patch(frame(90, 160), z, "rim")
    assert kind == "padded" and patch.shape[:2] == (45, 45)  # 32 x 45 px of it is in the frame
