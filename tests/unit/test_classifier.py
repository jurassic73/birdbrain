"""Unit tests of birdbrain/classifier: pure functions, no files, database or network."""

import json

import cv2
import numpy as np
import pytest

from helpers.classifier import LABELS, LOCAL, FakeClassifier, FakeTextClassifier, guess, one_hot, sure
from helpers.presence_view import BIG, GROUND_CROP, LIVE_CROP, SMALL, centered, fill_color, jpeg, live, need, noise
from birdbrain import classifier, config, zones


def test_rank_keeps_local_species_and_renormalizes():
    r = classifier.rank(np.array([0.2, 0.3, 0.1, 0.05, 0.05, 0.3, 0.0]), LABELS, LOCAL)
    # Junco 0.2 + Oregon 0.3; the Eurasian Jay and Purple Finch aren't local; Downy matched by its common name
    # although its genus was renamed; eBird spelling for "house finch".
    assert [g["name"] for g in r["top"]] == ["Dark-eyed Junco", "House Finch", "Downy Woodpecker"]
    assert [g["score"] for g in r["top"]] == [pytest.approx(0.5 / 0.65, abs=1e-3),
                                              pytest.approx(0.1 / 0.65, abs=1e-3),
                                              pytest.approx(0.05 / 0.65, abs=1e-3)]
    assert r["top"][2]["scientific"] == "Picoides pubescens"
    assert r["masked"] and r["species"] == 3 and r["local_share"] == pytest.approx(0.65, abs=1e-3)


def test_rank_without_a_local_list_uses_every_species():
    r = classifier.rank(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.1]), LABELS, None)
    assert r["top"][0]["name"] == "Eurasian Jay" and r["top"][0]["score"] == 1.0
    assert not r["masked"] and r["local_share"] is None and r["background"] == pytest.approx(0.1)


def test_crop_is_a_square_around_the_circle_else_the_box():
    img = np.zeros((1080, 1920, 3), np.uint8)
    patch, used = classifier.crop_for(img, {"x": 0.5, "y": 0.5, "r": 0.05}, (0.25, 0.25, 0.5, 0.5))
    side = round(2 * 0.05 * 1920 * classifier.CIRCLE_MARGIN)
    assert used == "circle" and patch.shape[:2] == (side, side)
    patch, used = classifier.crop_for(img, {"x": 0.99, "y": 0.01, "r": 0.05}, None)  # near a corner: slides in
    assert used == "circle" and patch.shape[:2] == (side, side)
    patch, used = classifier.crop_for(img, None, (0.25, 0.25, 0.5, 0.5))
    assert used == "box" and patch.shape[:2] == (540, 960)
    assert classifier.crop_for(img, None, None)[1] == "frame"


def test_off_means_no_guess(monkeypatch):
    monkeypatch.setattr(config, "CLASSIFIER", "off")
    monkeypatch.setattr(classifier, "load", lambda: 1 / 0)
    assert classifier.run(b"not even a jpeg", None, None, False) is None


def test_a_circle_is_classified_at_each_margin_and_averaged(monkeypatch):
    monkeypatch.setattr(classifier, "CIRCLE_MARGINS", (1.5, 3.0))
    img = np.zeros((1080, 1920, 3), np.uint8)
    fake = FakeClassifier()
    r = classifier.classify(img, {"x": 0.5, "y": 0.5, "r": 0.03}, None, False, local=LOCAL, clf=fake)
    assert [s[0] for s in fake.shapes] == [round(2 * 0.03 * 1920 * m) for m in classifier.CIRCLE_MARGINS]
    assert r["crop"] == "circle" and r["top"]
    fake.shapes.clear()
    classifier.classify(img, None, (0.25, 0.25, 0.5, 0.5), False, local=LOCAL, clf=fake)
    assert len(fake.shapes) == 1  # the crop box is classified once


def test_input_is_a_normalized_center_crop():
    x = classifier.to_input(np.full((300, 500, 3), 255, np.uint8))
    assert x.shape == (1, 3, 224, 224) and x.dtype == np.float32
    assert x[0, 0, 0, 0] == pytest.approx((1 - classifier.MEAN[0]) / classifier.STD[0], abs=1e-4)


def test_gate_names_only_sure_daytime_guesses_of_settled_species():
    clf = FakeTextClassifier()
    sure = classifier.from_embedding(one_hot(0), False, local=LOCAL, clf=clf)
    assert sure["top"][0]["name"] == "Dark-eyed Junco" and sure["top"][0]["score"] > 0.99 and sure["crop"] == "box"
    settled = {"Dark-eyed Junco": "bird"}
    assert classifier.gate_pick(sure, settled, night=False) == ("Dark-eyed Junco", "bird")
    assert classifier.gate_pick(sure, settled, night=True) is None            # never at night (IR unseen)
    assert classifier.gate_pick(sure, {}, night=False) is None                # not settled in the yard yet
    unsure = classifier.from_embedding(one_hot(0, 0.01), False, local=LOCAL, clf=clf)
    assert classifier.gate_pick(unsure, settled, night=False) is None         # spread over every species
    foreign = dict(sure, local_share=0.2)
    assert classifier.gate_pick(foreign, settled, night=False) is None        # mostly a species from elsewhere


def test_gate_miss_gives_the_reason_in_the_gates_order():
    sure = {"top": [{"name": "Dark-eyed Junco", "score": 0.99}], "local_share": 0.9}
    settled = {"Dark-eyed Junco": "bird"}
    assert classifier.gate_miss(sure, settled, False) is None
    assert classifier.gate_miss(sure, settled, True) == "night"
    assert classifier.gate_miss(None, settled, False) == "none"
    assert classifier.gate_miss(dict(sure, top=[{"name": "Bushtit", "score": 0.5}]), settled, False) == "score"
    assert classifier.gate_miss(dict(sure, local_share=0.1), settled, False) == "local"
    assert classifier.gate_miss(sure, {}, False) == "unsettled"
    assert classifier.gate_miss(sure, settled, False, crowd="Dark-eyed Junco") == "crowd"
    # an unsure guess of an unsettled species says "score" first, like gate_pick's tests
    assert classifier.gate_miss(dict(sure, top=[{"name": "Bushtit", "score": 0.5}]), {}, False) == "score"
    assert classifier.gate_pick(sure, settled, False) == ("Dark-eyed Junco", "bird")


def test_gate_miss_and_pick_judge_each_look_against_its_own_bar(monkeypatch):
    monkeypatch.setattr(classifier, "GATE_MIN_SCORE_BOWL", 0.8)
    monkeypatch.setattr(classifier, "GATE_MIN_SCORE_TIGHT", 0.95)
    settled = {"Dark-eyed Junco": "bird"}
    middling = {"top": [{"name": "Dark-eyed Junco", "score": 0.9}], "local_share": 0.9}
    assert classifier.gate_bar() == 0.8 and classifier.gate_bar("tight") == 0.95 and classifier.gate_bar("bowl") == 0.8
    # 0.9 clears the bowl's bar but not the tight look's
    assert classifier.gate_miss(middling, settled, False) is None
    assert classifier.gate_miss(middling, settled, False, view="bowl") is None
    assert classifier.gate_miss(middling, settled, False, view="tight") == "score"
    assert classifier.gate_pick(middling, settled, False) == ("Dark-eyed Junco", "bird")
    assert classifier.gate_pick(middling, settled, False, view="tight") is None
    sure = dict(middling, top=[{"name": "Dark-eyed Junco", "score": 0.96}])
    assert classifier.gate_pick(sure, settled, False, view="tight") == ("Dark-eyed Junco", "bird")
    assert classifier.gate_miss(dict(sure, local_share=0.1), settled, False, view="tight") == "local"
    assert classifier.gate_miss(sure, {}, False, view="tight") == "unsettled"
    assert classifier.gate_miss(sure, settled, True, view="tight") == "night"
    assert classifier.gate_miss(sure, settled, False, crowd="Dark-eyed Junco", view="tight") == "crowd"
    # the stored record says why against the look's own bar
    assert classifier.first_look(middling, settled, False)["why"] is None
    assert classifier.first_look(middling, settled, False, view="tight")["why"] == "score"
    # the bars come from .env through config, like the other tunables
    assert config.GATE_MIN_SCORE_BOWL == 0.8 and config.GATE_MIN_SCORE_TIGHT == 0.95


def test_tight_check_classifies_a_square_around_the_motion_circle(monkeypatch):
    monkeypatch.setattr(config, "CLASSIFIER", "gate")
    img = np.full((1080, 1920, 3), (40, 120, 200), np.uint8)
    jpeg = cv2.imencode(".jpg", img)[1].tobytes()
    fake = FakeClassifier(probs=(0.9, 0.05, 0.02, 0.0, 0.0, 0.03, 0.0))
    monkeypatch.setattr(classifier, "load", lambda: fake)
    monkeypatch.setattr(classifier, "local_species", lambda: LOCAL)
    circle = {"x": 0.5, "y": 0.5, "r": 0.05, "pi": True}
    settled = {"Dark-eyed Junco": "bird"}
    guess, pick = classifier.tight_check(jpeg, circle, (0.25, 0.25, 0.5, 0.5), False, settled)
    assert classifier.CIRCLE_MARGIN == 2.0 and classifier.CIRCLE_MARGINS == (2.0,)
    side = round(2 * 0.05 * 1920 * 2.0)  # one square, 2.0 radii from the center to each side
    assert fake.shapes == [(side, side, 3)]
    assert guess["crop"] == "motion" and guess["top"][0]["name"] == "Dark-eyed Junco" and guess["top"][0]["score"] > 0.95
    assert pick == ("Dark-eyed Junco", "bird")
    # under the tight bar: a guess, no pick
    fake._probs = np.array((0.5, 0.05, 0.3, 0.0, 0.0, 0.15, 0.0), np.float32)
    guess, pick = classifier.tight_check(jpeg, circle, None, False, settled)
    assert pick is None and guess["crop"] == "motion"
    # never at night, without a circle, in shadow mode, or when the model breaks: Claude is asked
    assert classifier.tight_check(jpeg, circle, None, True, settled) == (None, None)
    assert classifier.tight_check(jpeg, None, None, False, settled) == (None, None)
    assert classifier.tight_check(jpeg, {"x": 0.5, "y": 0.5}, None, False, settled) == (None, None)
    monkeypatch.setattr(config, "CLASSIFIER", "shadow")
    assert classifier.tight_check(jpeg, circle, None, False, settled) == (None, None)
    monkeypatch.setattr(config, "CLASSIFIER", "gate")
    fake.fail = True
    assert classifier.tight_check(jpeg, circle, None, False, settled) == (None, None)


def test_the_view_constants():
    assert need("PRESENCE_VIEW") == 2
    assert need("BIRD_BAND") == 0.06 and need("VIEW_MARGIN") == 0.04


def test_padding_is_the_clip_mean_color_so_it_normalizes_to_zero():
    mean_bgr = classifier.MEAN[::-1] * 255
    assert np.allclose(np.asarray(need("PAD_FILL"), float)[:3], mean_bgr, atol=1.5)
    flat = classifier.to_input(np.full((224, 224, 3), fill_color(), np.uint8))
    assert np.abs(flat).max() < 0.03


def test_the_square_holds_the_whole_outer_rim_and_a_birds_height_above_it():
    x0, y0, side = need("presence_square")(BIG, live())
    H, W = BIG[:2]
    rim_left, rim_right = 0.302 * W, 0.634 * W
    top = (0.5347 - 0.1175 - need("BIRD_BAND")) * H
    bottom = (0.5347 + 0.1175) * H
    assert x0 <= rim_left + 1 and x0 + side >= rim_right - 1
    assert y0 <= top + 1 and y0 + side >= bottom - 1
    assert side <= 1080 and x0 >= 0 and y0 >= 0 and x0 + side <= W and y0 + side <= H
    assert side > rim_right - rim_left  # some room around the bowl, not a bare fit


def test_the_square_grows_by_the_margin_on_each_side(monkeypatch):
    bare = 0.332 * BIG[1]
    side = need("presence_square")(BIG, live())[2]
    assert side >= bare * (1 + need("VIEW_MARGIN"))  # at least the margin on the sides
    monkeypatch.setattr(classifier, "VIEW_MARGIN", 0.0)
    no_margin = classifier.presence_square(BIG, live())[2]
    assert no_margin < side - 10


def test_a_tall_ring_is_limited_by_the_bird_band_above_it():
    ring = (0.5, 0.6, 0.04, 0.2, 0.03, 0.15)  # taller than wide: the band decides where the square starts
    x0, y0, side = need("presence_square")(BIG, zones.Zones(ring=ring))
    H, W = BIG[:2]
    assert y0 <= (0.6 - 0.2 - need("BIRD_BAND")) * H + 1 and y0 + side >= (0.6 + 0.2) * H - 1
    assert x0 <= 0.46 * W and x0 + side >= 0.54 * W


@pytest.mark.parametrize("ring", [(0.08, 0.5, 0.1, 0.1, 0.08, 0.08),    # left edge
                                  (0.95, 0.5, 0.1, 0.1, 0.08, 0.08),    # right edge
                                  (0.5, 0.1, 0.1, 0.08, 0.08, 0.06),    # top edge (the bird band leaves the frame)
                                  (0.5, 0.95, 0.1, 0.08, 0.08, 0.06)])  # bottom edge
def test_a_ring_near_the_edge_slides_the_square_inside_the_frame(ring):
    H, W = BIG[:2]
    x0, y0, side = need("presence_square")(BIG, zones.Zones(ring=ring))
    assert x0 >= 0 and y0 >= 0 and x0 + side <= W and y0 + side <= H
    cx, cy, rx, ry = ring[:4]
    if cx - rx >= 0 and cx + rx <= 1:
        assert x0 <= (cx - rx) * W + 1 and x0 + side >= (cx + rx) * W - 1
    if cy - ry >= 0 and cy + ry <= 1:
        assert y0 <= (cy - ry) * H + 1 and y0 + side >= (cy + ry) * H - 1


def test_a_ring_taller_than_the_frame_clips_the_side_to_the_height():
    ring = (0.5, 0.5, 0.3, 0.6, 0.25, 0.5)
    x0, y0, side = need("presence_square")(BIG, zones.Zones(ring=ring))
    assert side == 1080 and y0 == 0 and 0 <= x0 <= 1920 - 1080
    x0, y0, side = classifier.presence_square((1920, 1080, 3), zones.Zones(ring=ring))  # portrait: min(H, W) is W
    assert side <= 1080 and x0 >= 0 and y0 >= 0 and y0 + side <= 1920


def test_a_rotated_ring_is_held_by_its_rotated_extent():
    # 90 degrees turns the pixel ellipse (rx*W by ry*H) on its side: 108 px wide, 384 px tall each way.
    ring = (0.5, 0.5, 0.2, 0.1, 0.15, 0.08, 0.5, 0.5, 90.0)
    x0, y0, side = need("presence_square")(BIG, zones.Zones(ring=ring))
    assert y0 <= 540 - 384 + 1 and y0 + side >= 540 + 384 - 1
    assert x0 <= 960 - 108 + 1 and x0 + side >= 960 + 108 - 1


def test_no_ring_no_square():
    assert need("presence_square")(BIG, zones.Zones(crop=LIVE_CROP)) is None
    assert classifier.presence_square(BIG, {}) is None


def test_zones_may_be_a_dict_or_a_zones_object():
    z = live()
    from_obj = need("presence_square")(BIG, z)
    assert need("presence_square")(BIG, z.to_dict()) == from_obj
    assert classifier.presence_square(BIG, json.loads(json.dumps(z.to_dict()))) == from_obj  # raw_json.zones


def test_the_same_zones_on_a_half_size_image_give_the_same_square():
    big = need("presence_square")(BIG, live())
    for shape in (SMALL, (720, 1280, 3)):
        k = BIG[1] / shape[1]
        small = classifier.presence_square(shape, live())
        assert all(abs(a * k - b) <= 2 * k for a, b in zip(small, big)), (shape, small, big)


def test_a_bath_trigger_with_a_ring_gets_the_bowl_square():
    img = noise(BIG)
    patch, kind = need("presence_patch")(img, live(), "rim")
    x0, y0, side = classifier.presence_square(img.shape, live())
    assert kind == "bowl" and patch.shape == (side, side, 3)
    assert np.array_equal(patch, img[y0:y0 + side, x0:x0 + side])
    assert classifier.presence_patch(img, live(), "water")[1] == "bowl"
    assert classifier.presence_patch(img, live().to_dict(), "rim")[1] == "bowl"


def test_a_tall_box_is_padded_on_the_sides():
    img = noise(BIG, 2)
    box = (0.4, 0.0, 0.1, 1.0)
    patch, kind = need("presence_patch")(img, live(ground_crop=box), "ground")
    x0, y0, x1, y1 = zones.pixel_box(img.shape, box)
    assert kind == "padded" and abs(patch.shape[0] - patch.shape[1]) <= 1 and patch.shape[0] == 1080
    oy, ox = centered(patch, img[y0:y1, x0:x1])
    assert (patch[:, :ox - 1] == fill_color()).all() and (patch[:, ox + (x1 - x0) + 1:] == fill_color()).all()


def test_a_bath_trigger_without_a_ring_pads_the_crop():
    img = noise(BIG, 3)
    patch, kind = need("presence_patch")(img, zones.Zones(crop=LIVE_CROP), "rim")
    x0, y0, x1, y1 = zones.pixel_box(img.shape, LIVE_CROP)
    assert kind == "padded" and abs(patch.shape[0] - patch.shape[1]) <= 1 and patch.shape[1] == x1 - x0
    centered(patch, img[y0:y1, x0:x1])


def test_a_wide_box_markers_survive_once_padded():
    img = np.full(BIG, 90, np.uint8)
    x0, y0, x1, y1 = zones.pixel_box(img.shape, GROUND_CROP)
    img[y0 + 200:y0 + 340, x0:x0 + 40] = 255
    img[y0 + 200:y0 + 340, x1 - 40:x1] = 255
    patch = need("presence_patch")(img, live(), "ground")[0]
    x = classifier.to_input(patch)
    assert x[0, :, :, 0:4].max() > 1.0 and x[0, :, :, 220:224].max() > 1.0


def test_a_bird_on_the_left_rim_is_in_the_new_view_and_was_cut_from_the_old_one():
    img = np.full(BIG, 90, np.uint8)
    cv2.circle(img, (int(0.31 * 1920), int(0.5347 * 1080)), 20, (255, 255, 255), -1)
    old = classifier.to_input(classifier.crop_for(img, None, LIVE_CROP)[0])
    new = classifier.to_input(need("presence_patch")(img, live(), "rim")[0])
    assert old.max() < 1.0   # the center cut dropped it
    assert new.max() > 1.5   # the bowl square keeps it


def test_the_view_box_of_a_rim_visit_is_the_bowl_square_as_frame_fractions():
    x, y, w, h = need("presence_view_box")(live(), "rim")
    x0, y0, side = classifier.presence_square(BIG, live())
    assert (x, y, w, h) == pytest.approx((x0 / 1920, y0 / 1080, side / 1920, side / 1080), abs=0.003)
    assert w != pytest.approx(h, abs=0.01)  # a square on a 16:9 frame
    assert tuple(classifier.presence_view_box(live().to_dict(), "water")) == pytest.approx((x, y, w, h))


def test_the_view_box_follows_the_aspect():
    x, y, w, h = need("presence_view_box")(live(), "rim", aspect=1.0)
    x0, y0, side = classifier.presence_square((1000, 1000, 3), live())
    assert (x, y, w, h) == pytest.approx((x0 / 1000, y0 / 1000, side / 1000, side / 1000), abs=0.003)


def test_the_view_box_of_a_padded_view_is_the_original_box():
    assert tuple(need("presence_view_box")(live(), "ground")) == pytest.approx(GROUND_CROP)
    assert tuple(classifier.presence_view_box(zones.Zones(crop=LIVE_CROP), "rim")) == pytest.approx(LIVE_CROP)
    box = classifier.presence_view_box(zones.Zones(), "rim")
    assert box is None or tuple(box) == pytest.approx((0, 0, 1, 1))


def test_box_embedding_fails_soft(monkeypatch):
    class Broken:
        def embed(self, bgr):
            raise RuntimeError("out of memory")

    monkeypatch.setattr(classifier, "load", lambda: Broken())
    monkeypatch.setattr(classifier.config, "CLASSIFIER", "shadow")
    assert need("box_embedding")(jpeg(noise(SMALL)), live(), "rim") is None
    assert classifier.box_embedding(b"not a jpeg", live(), "rim") is None
    monkeypatch.setattr(classifier.config, "CLASSIFIER", "off")
    assert classifier.box_embedding(jpeg(noise(SMALL)), live(), "rim") is None


class ShapeClf:
    labels = [("Junco hyemalis", "Dark-eyed Junco", True)]

    def __init__(self):
        self.shapes = []

    def probs(self, bgr):
        self.shapes.append(bgr.shape)
        return np.array([1.0], np.float32)


def test_the_species_check_still_looks_at_the_circle_or_the_unpadded_crop_box():
    img = noise(BIG, 7)
    clf = ShapeClf()
    classifier.classify(img, {"x": 0.5, "y": 0.5, "r": 0.03}, LIVE_CROP, False, local={}, clf=clf)
    x0, y0, side = classifier.circle_square({"x": 0.5, "y": 0.5, "r": 0.03}, 1920, 1080)
    assert clf.shapes[-1][:2] == (side, side)
    classifier.classify(img, None, LIVE_CROP, False, local={}, clf=clf)
    px = zones.pixel_box(img.shape, LIVE_CROP)
    assert clf.shapes[-1][:2] == (px[3] - px[1], px[2] - px[0])  # wide, not padded and not the bowl square


def test_crop_for_is_unchanged():
    img = noise(BIG, 8)
    patch, used = classifier.crop_for(img, None, LIVE_CROP)
    x0, y0, x1, y1 = zones.pixel_box(img.shape, LIVE_CROP)
    assert used == "box" and np.array_equal(patch, img[y0:y1, x0:x1])
    assert classifier.crop_for(img, None, None)[1] == "frame"
    assert classifier.crop_for(img, {"x": 0.5, "y": 0.5, "r": 0.02}, LIVE_CROP)[1] == "circle"


def test_to_input_is_unchanged_for_square_patches_and_still_center_cuts_wide_ones():
    img = noise((300, 300, 3), 9)
    rgb = cv2.cvtColor(cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2RGB)
    want = ((rgb.astype(np.float32) / 255 - classifier.MEAN) / classifier.STD).transpose(2, 0, 1)[None]
    assert np.allclose(classifier.to_input(img), want, atol=1e-5)
    wide = np.full((300, 900, 3), 90, np.uint8)
    wide[:, :100] = 255
    assert classifier.to_input(wide).max() < 1.0  # the far left strip is still cut off for non-square input
