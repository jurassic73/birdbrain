"""The Pi animal check sees the whole bowl (bath triggers) or the whole padded ground crop (ground triggers).

Written from the contract only (classifier.presence_square / presence_patch / presence_view_box / box_embedding,
embeddings.view, presence.store(view=), backfill(redo=), check(view=)). New names are looked up inside the tests so a
missing one fails that test, never the collection of the file.
"""
import json
import re

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from helpers.clock import FROZEN_NOW
from birdbrain import classifier, db, images, presence, watch, zones
from birdbrain.web import app as webapp

DIM = 16

# The live birdbath (2026-11): the outer rim spans x 0.302..0.634 of the width.
LIVE_RING = (0.4683, 0.5347, 0.1659, 0.1175, 0.1345, 0.0966, 0.5224, 0.4681)
LIVE_CROP = (0.248, 0.30, 0.4547, 0.4991)       # 873 x 539 px on 1920x1080; the old center cut kept x 0.335..0.615
NARROW_CROP = (0.335, 0.30, 0.28, 0.50)         # just that kept strip
GROUND_CROP = (0.0, 0.5, 1.0, 0.5)              # 1920 x 540 px: very wide
GROUND_POLY = ((0.1, 0.5), (0.9, 0.5), (0.9, 1.0), (0.1, 1.0))
BIG, SMALL = (1080, 1920, 3), (540, 960, 3)
BOWL_CIRCLE = {"x": 0.4683, "y": 0.5347, "r": 0.03}
RIM_LEFT = {"x": 0.31, "y": 0.5347, "r": 0.015}  # on the rim's left edge: inside the bowl square, in the old cut strip


def need(name):
    """A name the contract promises on classifier; a missing one is an AttributeError in the test that uses it."""
    return getattr(classifier, name)


def live(**kw) -> "zones.Zones":
    return zones.Zones(**{"ring": LIVE_RING, "crop": LIVE_CROP, "ground": GROUND_POLY, "ground_crop": GROUND_CROP, **kw})


def noise(shape, seed=0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, shape, dtype=np.uint8)


def fill_color() -> np.ndarray:
    return np.rint(np.asarray(need("PAD_FILL"), float)).astype(np.uint8)[:3]


def jpeg(img) -> bytes:
    return cv2.imencode(".jpg", img)[1].tobytes()


def vec(animal: bool, rng) -> np.ndarray:
    v = rng.normal(0, 0.3, DIM)
    v[0] += 1.0 if animal else -1.0
    return v / np.linalg.norm(v)


class FakeClf:
    """Embeds a patch as an animal when it's bright, else as empty; remembers every patch it was given."""

    def __init__(self, rng=None):
        self.rng = rng or np.random.default_rng(3)
        self.means, self.shapes = [], []

    def embed(self, bgr):
        self.means.append(float(bgr.mean()))
        self.shapes.append(bgr.shape)
        return vec(bgr.mean() > 128, self.rng)


@pytest.fixture
def yard(conn, add_visit):
    """12 visitors and 40 empty frames with embeddings made the current way."""
    rng = np.random.default_rng(1)
    for i in range(52):
        animal = i < 12
        vid = add_visit(**({} if animal else {"visitor": "none", "species": None}))
        presence.store(conn, vid, vec(animal, rng))
    return rng


def stored_visit(add_visit, img, raw=None, **fields) -> int:
    rel, thumb = images.store_frame(jpeg(img), FROZEN_NOW)
    return add_visit(image_path=rel, thumb_path=thumb, raw_json=None if raw is None else json.dumps(raw), **fields)


def view_of(conn, vid):
    return conn.execute("SELECT view FROM embeddings WHERE visit_id = ?", (vid,)).fetchone()[0]


def left_out(conn) -> dict:
    return dict(presence.examples(conn)["left_out"])


def centered(patch, box_img) -> tuple[int, int]:
    """Where the box's pixels sit in a padded patch (within a pixel or two of the middle): (top, left)."""
    bh, bw = box_img.shape[:2]
    dh, dw = patch.shape[0] - bh, patch.shape[1] - bw
    inner = box_img[3:-3, 3:-3]
    for oy in range(max(dh // 2 - 2, 0), dh // 2 + 3):
        for ox in range(max(dw // 2 - 2, 0), dw // 2 + 3):
            if np.array_equal(patch[oy + 3:oy + bh - 3, ox + 3:ox + bw - 3], inner):
                return oy, ox
    raise AssertionError("the box's pixels are not unchanged in the middle of the patch")


# --- the constants --------------------------------------------------------------------------------------------

def test_the_view_constants():
    assert need("PRESENCE_VIEW") == 2
    assert need("BIRD_BAND") == 0.06 and need("VIEW_MARGIN") == 0.04


def test_padding_is_the_clip_mean_color_so_it_normalizes_to_zero():
    mean_bgr = classifier.MEAN[::-1] * 255
    assert np.allclose(np.asarray(need("PAD_FILL"), float)[:3], mean_bgr, atol=1.5)
    flat = classifier.to_input(np.full((224, 224, 3), fill_color(), np.uint8))
    assert np.abs(flat).max() < 0.03


# --- 1 to 3: presence_square -----------------------------------------------------------------------------------

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


# --- 4 and 5: presence_patch -----------------------------------------------------------------------------------

def test_a_bath_trigger_with_a_ring_gets_the_bowl_square():
    img = noise(BIG)
    patch, kind = need("presence_patch")(img, live(), "rim")
    x0, y0, side = classifier.presence_square(img.shape, live())
    assert kind == "bowl" and patch.shape == (side, side, 3)
    assert np.array_equal(patch, img[y0:y0 + side, x0:x0 + side])
    assert classifier.presence_patch(img, live(), "water")[1] == "bowl"
    assert classifier.presence_patch(img, live().to_dict(), "rim")[1] == "bowl"


def test_a_ground_trigger_gets_the_ground_crop_padded_to_a_square():
    img = noise(BIG, 1)
    patch, kind = need("presence_patch")(img, live(), "ground")
    assert kind == "padded" and abs(patch.shape[0] - patch.shape[1]) <= 1 and patch.shape[1] == 1920
    x0, y0, x1, y1 = zones.pixel_box(img.shape, GROUND_CROP)
    oy, ox = centered(patch, img[y0:y1, x0:x1])
    fill = fill_color()
    assert (patch[:oy - 1] == fill).all() and (patch[oy + (y1 - y0) + 1:] == fill).all()
    assert oy > 600  # padded above and below, nothing of the bowl added


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


def test_nothing_usable_pads_the_whole_frame():
    img = noise(BIG, 4)
    patch, kind = need("presence_patch")(img, zones.Zones(), "rim")
    assert kind == "frame" and abs(patch.shape[0] - patch.shape[1]) <= 1 and patch.shape[1] == 1920
    oy, ox = centered(patch, img)
    assert 400 < oy < 440 and (patch[:oy - 1] == fill_color()).all()
    assert classifier.presence_patch(img, zones.Zones(), "ground")[1] == "frame"  # no ground crop either


def test_the_patch_is_square_on_any_image_size():
    for shape in (SMALL, (720, 1280, 3), (480, 960, 3)):
        img = noise(shape, 5)
        for trigger in ("rim", "water", "ground"):
            patch, kind = need("presence_patch")(img, live(), trigger)
            assert abs(patch.shape[0] - patch.shape[1]) <= 1 and patch.shape[2] == 3
            assert kind == ("padded" if trigger == "ground" else "bowl")
        for z in (zones.Zones(crop=LIVE_CROP), zones.Zones()):
            patch, kind = classifier.presence_patch(img, z, "rim")
            assert abs(patch.shape[0] - patch.shape[1]) <= 1 and kind in ("padded", "frame")


def test_the_model_input_keeps_the_whole_patch():
    img = np.full(BIG, 90, np.uint8)
    for z, trigger in ((live(), "rim"), (live(), "ground"), (zones.Zones(), "rim")):
        patch = need("presence_patch")(img, z, trigger)[0].copy()
        mid = patch.shape[0] // 2
        patch[mid - 30:mid + 30, :40] = 255          # a marker block at the far left column
        patch[mid - 30:mid + 30, -40:] = 255         # and the far right
        x = classifier.to_input(patch)
        assert x.shape == (1, 3, 224, 224)
        assert x[0, :, 100:124, 0:4].max() > 1.0 and x[0, :, 100:124, 220:224].max() > 1.0


def test_a_wide_box_markers_survive_once_padded():
    img = np.full(BIG, 90, np.uint8)
    x0, y0, x1, y1 = zones.pixel_box(img.shape, GROUND_CROP)
    img[y0 + 200:y0 + 340, x0:x0 + 40] = 255
    img[y0 + 200:y0 + 340, x1 - 40:x1] = 255
    patch = need("presence_patch")(img, live(), "ground")[0]
    x = classifier.to_input(patch)
    assert x[0, :, :, 0:4].max() > 1.0 and x[0, :, :, 220:224].max() > 1.0


# --- 6: a bird on the rim's left edge ---------------------------------------------------------------------------

def test_a_bird_on_the_left_rim_is_in_the_new_view_and_was_cut_from_the_old_one():
    img = np.full(BIG, 90, np.uint8)
    cv2.circle(img, (int(0.31 * 1920), int(0.5347 * 1080)), 20, (255, 255, 255), -1)
    old = classifier.to_input(classifier.crop_for(img, None, LIVE_CROP)[0])
    new = classifier.to_input(need("presence_patch")(img, live(), "rim")[0])
    assert old.max() < 1.0   # the center cut dropped it
    assert new.max() > 1.5   # the bowl square keeps it


# --- presence_view_box ------------------------------------------------------------------------------------------

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


# --- box_embedding ----------------------------------------------------------------------------------------------

def test_box_embedding_embeds_the_presence_patch(monkeypatch):
    clf = FakeClf()
    monkeypatch.setattr(classifier, "load", lambda: clf)
    monkeypatch.setattr(classifier.config, "CLASSIFIER", "shadow")
    img = noise(BIG, 6)
    emb = need("box_embedding")(jpeg(img), live(), "rim")
    assert emb is not None and len(clf.shapes) == 1
    side = classifier.presence_square(BIG, live())[2]
    assert clf.shapes[0][:2] == (side, side)
    classifier.box_embedding(jpeg(img), live(), "ground")
    assert clf.shapes[1][0] == clf.shapes[1][1] == 1920


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


# --- 7: the schema ----------------------------------------------------------------------------------------------

def test_init_db_adds_the_view_column_to_old_databases_and_old_rows_read_null(tmp_path):
    import sqlite3

    old = sqlite3.connect(tmp_path / "old.db")
    old.row_factory = sqlite3.Row
    old.execute("CREATE TABLE embeddings (visit_id INTEGER PRIMARY KEY, vec BLOB NOT NULL, animal_vec BLOB, "
                "animal_frame INTEGER, animal_score REAL, animal_by_owner INTEGER)")
    old.execute("INSERT INTO embeddings (visit_id, vec) VALUES (1, x'0000')")
    db.init_db(old)
    cols = {r["name"] for r in old.execute("PRAGMA table_info(embeddings)")}
    assert "view" in cols
    assert old.execute("SELECT view FROM embeddings WHERE visit_id = 1").fetchone()[0] is None
    db.init_db(old)  # and again: nothing to add


def test_a_fresh_database_has_the_view_column(conn):
    assert "view" in {r["name"] for r in conn.execute("PRAGMA table_info(embeddings)")}


def test_store_writes_the_current_view_unless_told_otherwise(conn, add_visit):
    vid = add_visit()
    presence.store(conn, vid, np.ones(DIM))
    assert view_of(conn, vid) == need("PRESENCE_VIEW")
    presence.store(conn, vid, np.ones(DIM), view=1)
    assert view_of(conn, vid) == 1
    presence.store(conn, vid, np.zeros(DIM))  # stored again the current way: the view follows
    assert view_of(conn, vid) == classifier.PRESENCE_VIEW


# --- 8: training uses only current-view rows --------------------------------------------------------------------

def old_yard(conn, add_visit, view=1, visitors=12, empties=40, seed=1):
    rng = np.random.default_rng(seed)
    ids = []
    for i in range(visitors + empties):
        animal = i < visitors
        vid = add_visit(**({} if animal else {"visitor": "none", "species": None}))
        presence.store(conn, vid, vec(animal, rng), view=view)
        ids.append(vid)
    return ids, rng


def test_old_view_rows_alone_train_nothing_and_gate_nothing(conn, add_visit):
    ids, rng = old_yard(conn, add_visit, view=1)
    assert presence.model(conn) is None
    assert presence.labelled(conn)[0].shape[0] == 0
    emb = vec(False, rng)
    expected = {"model": presence.MODEL_NAME, "score": None, "empty": False, "note": "not enough labelled frames yet"}
    assert presence.check(conn, emb) == expected
    got = presence.check(conn, emb, view="bowl")
    assert got["score"] is None and got["empty"] is False


def test_rows_with_a_null_view_are_old_too(conn, add_visit):
    ids, rng = old_yard(conn, add_visit, view=need("PRESENCE_VIEW"))
    assert presence.model(conn) is not None
    conn.execute("UPDATE embeddings SET view = NULL")
    conn.commit()
    assert presence.model(conn) is None and presence.examples(conn)["ids"].size == 0


def test_enough_current_view_rows_train_even_beside_old_ones(conn, add_visit):
    old_ids, _ = old_yard(conn, add_visit, view=1, visitors=6, empties=6, seed=9)
    # The old rows lean the opposite way: if they trained, the model would be confused.
    for i, vid in enumerate(old_ids):
        presence.store(conn, vid, vec(i >= 6, np.random.default_rng(i)), view=1)
    new_ids, rng = old_yard(conn, add_visit, view=need("PRESENCE_VIEW"))
    m = presence.model(conn)
    assert m is not None and m["visitors"] == 12 and m["empty"] == 40
    assert set(presence.labelled(conn)[3]) == set(new_ids) and not set(old_ids) & set(presence.examples(conn)["ids"])
    assert presence.check(conn, vec(False, rng), view="bowl")["empty"] is True
    assert presence.check(conn, vec(True, rng), view="bowl")["empty"] is False


def test_check_reports_the_view_it_was_given(conn, yard):
    emb = vec(True, yard)
    assert presence.check(conn, emb, view="bowl")["view"] == "bowl"
    assert presence.check(conn, emb, view="padded")["view"] == "padded"
    assert presence.check(conn, emb).get("view") is None
    assert presence.check(conn, None, view="bowl") is None


def test_the_label_key_changes_when_a_row_is_re_embedded_to_the_current_view(conn, add_visit):
    ids, rng = old_yard(conn, add_visit, view=1)
    before = presence.label_key(conn)
    presence.store(conn, ids[0], vec(True, rng))  # now the current view
    after = presence.label_key(conn)
    assert after != before
    presence.store(conn, ids[0], vec(True, rng), view=1)  # and back: the key follows
    assert presence.label_key(conn) == before


# --- 9: re-embedding clears the old burst pick ------------------------------------------------------------------

def burst_visit(conn, add_visit, brightness, owner_circle=None):
    """A false trigger whose burst is on disk, zones with a ring, embedded the current way."""
    frames = [jpeg(np.full((90, 160, 3), b, np.uint8)) for b in brightness]
    rel, thumb = images.store_frame(frames[0], FROZEN_NOW)
    images.store_burst(frames, rel)
    raw = {"zones": {"ring": list(LIVE_RING), "crop": list(LIVE_CROP)}, "trigger": "rim", "burst": {"best": 0}}
    vid = add_visit(visitor="none", species=None, model=presence.MODEL_NAME, image_path=rel, thumb_path=thumb,
                    raw_json=json.dumps(raw))
    presence.store(conn, vid, vec(False, np.random.default_rng(5)))
    db.correct_visit(conn, vid, "squirrel", None, None, [owner_circle or BOWL_CIRCLE])
    return vid


def pick_of(conn, vid) -> dict:
    return dict(conn.execute("SELECT animal_frame, animal_score, animal_by_owner, view, animal_vec IS NOT NULL AS has_vec "
                             "FROM embeddings WHERE visit_id = ?", (vid,)).fetchone())


def test_re_embedding_a_row_clears_its_old_burst_pick_and_it_is_picked_again(monkeypatch, conn, yard, add_visit):
    vid = burst_visit(conn, add_visit, [60, 60, 60, 60, 200, 60])
    clf = FakeClf(yard)
    assert presence.pick_frames(conn, clf=clf) == 1
    assert pick_of(conn, vid)["animal_frame"] == 5 and pick_of(conn, vid)["has_vec"]
    assert all(s[0] == s[1] for s in clf.shapes), "the burst frames are embedded as square patches"
    # The row was made the old way: its pick is from the old view too.
    conn.execute("UPDATE embeddings SET view = 1 WHERE visit_id = ?", (vid,))
    conn.commit()
    monkeypatch.setattr(classifier, "load", lambda: FakeClf(yard))
    presence.backfill(redo=True)
    row = pick_of(conn, vid)
    assert row["view"] == need("PRESENCE_VIEW")
    assert row["animal_frame"] is None and row["animal_score"] is None and not row["has_vec"]
    assert [p["id"] for p in presence.pending_picks(conn)] == [vid]  # needs looking through again
    clf = FakeClf(yard)
    assert presence.pick_frames(conn, clf=clf) == 1
    assert len(clf.shapes) == 6 and all(s[0] == s[1] for s in clf.shapes)
    row = pick_of(conn, vid)
    assert row["animal_frame"] == 5 and row["has_vec"] and row["view"] == classifier.PRESENCE_VIEW


def test_an_owner_circled_visit_is_re_picked_after_a_re_embed(monkeypatch, conn, yard, add_visit):
    vid = burst_visit(conn, add_visit, [60, 200, 60, 60], owner_circle={**BOWL_CIRCLE, "frame": 2})
    assert presence.pick_frames(conn, clf=FakeClf(yard)) == 1
    pick = pick_of(conn, vid)
    assert pick["animal_frame"] == 2 and pick["animal_by_owner"] == 1 and pick["has_vec"]
    conn.execute("UPDATE embeddings SET view = 1 WHERE visit_id = ?", (vid,))
    conn.commit()
    monkeypatch.setattr(classifier, "load", lambda: FakeClf(yard))
    presence.backfill(redo=True)
    assert pick_of(conn, vid)["animal_frame"] is None and not pick_of(conn, vid)["has_vec"]
    pending = presence.pending_picks(conn)
    assert [(p["id"], p["owner_frame"]) for p in pending] == [(vid, 2)]
    clf = FakeClf(yard)
    assert presence.pick_frames(conn, clf=clf) == 1 and len(clf.shapes) == 1  # just the owner's frame
    pick = pick_of(conn, vid)
    assert pick["animal_frame"] == 2 and pick["animal_by_owner"] == 1 and pick["has_vec"] and pick["view"] == 2
    assert presence.pending_picks(conn) == []


# --- 10: backfill --redo and its progress lines -----------------------------------------------------------------

BRIGHT = (40, 100, 160, 220)
PROGRESS = re.compile(r"\d+/\d+ done, \d+ failed, ~\d+ min left")


def uniform_visit(add_visit, brightness, raw=None, shape=(90, 160)):
    return stored_visit(add_visit, np.full((*shape, 3), brightness, np.uint8),
                        raw if raw is not None else {"zones": {"ring": list(LIVE_RING), "crop": list(LIVE_CROP)}, "trigger": "rim"})


def order_of(clf) -> list[int]:
    return [min(BRIGHT, key=lambda b: abs(b - m)) for m in clf.means]


def test_redo_re_embeds_old_rows_newest_first_and_leaves_current_ones_alone(monkeypatch, conn, add_visit, capsys):
    a, b, c = (uniform_visit(add_visit, x) for x in BRIGHT[:3])
    d = uniform_visit(add_visit, BRIGHT[3])
    mine = np.full(DIM, 0.25, np.float16)
    presence.store(conn, a, np.ones(DIM), view=1)
    presence.store(conn, b, np.ones(DIM), view=1)
    presence.store(conn, c, mine)                  # current view
    clf = FakeClf()
    monkeypatch.setattr(classifier, "load", lambda: clf)
    presence.backfill(redo=True)
    assert order_of(clf) == [BRIGHT[3], BRIGHT[1], BRIGHT[0]]  # d (no embedding yet) and b, a: newest first
    assert [view_of(conn, i) for i in (a, b, c, d)] == [need("PRESENCE_VIEW")] * 4
    assert conn.execute("SELECT vec FROM embeddings WHERE visit_id = ?", (c,)).fetchone()[0] == mine.tobytes()
    out = capsys.readouterr().out
    assert PROGRESS.search(out) and re.search(r"3/3 done, 0 failed", out)


def test_redo_leaves_current_rows_and_a_plain_backfill_leaves_old_ones(monkeypatch, conn, add_visit):
    old = uniform_visit(add_visit, BRIGHT[0])
    fresh = uniform_visit(add_visit, BRIGHT[1])
    new = uniform_visit(add_visit, BRIGHT[2])  # no embedding at all
    presence.store(conn, old, np.ones(DIM), view=1)
    presence.store(conn, fresh, np.ones(DIM))
    clf = FakeClf()
    monkeypatch.setattr(classifier, "load", lambda: clf)
    presence.backfill()
    assert order_of(clf) == [BRIGHT[2]]                       # only the frame with no embedding
    assert view_of(conn, old) == 1 and view_of(conn, new) == need("PRESENCE_VIEW")
    clf.means.clear()
    presence.backfill(redo=True)
    assert order_of(clf) == [BRIGHT[0]]                       # now the old one, and not the two current ones
    assert view_of(conn, old) == classifier.PRESENCE_VIEW


def test_redo_with_a_limit_does_the_newest_old_rows_first(monkeypatch, conn, add_visit):
    ids = [uniform_visit(add_visit, x) for x in BRIGHT[:3]]
    for i in ids:
        presence.store(conn, i, np.ones(DIM), view=1)
    clf = FakeClf()
    monkeypatch.setattr(classifier, "load", lambda: clf)
    presence.backfill(limit=1, redo=True)
    assert order_of(clf) == [BRIGHT[2]]
    assert [view_of(conn, i) for i in ids] == [1, 1, need("PRESENCE_VIEW")]


def test_progress_lines_every_25_frames_and_a_final_line(monkeypatch, conn, add_visit, capsys):
    ids = [uniform_visit(add_visit, 100) for _ in range(27)]
    for i in ids:
        presence.store(conn, i, np.ones(DIM), view=1)
    monkeypatch.setattr(classifier, "load", lambda: FakeClf())
    presence.backfill(redo=True)
    lines = [ln for ln in capsys.readouterr().out.splitlines() if PROGRESS.search(ln)]
    assert len(lines) >= 2
    assert "25/27 done, 0 failed" in lines[0] and "27/27 done, 0 failed" in lines[-1]


def test_a_frame_that_cannot_be_read_is_counted_as_failed(monkeypatch, conn, add_visit, capsys):
    good = uniform_visit(add_visit, 100)
    gone = add_visit(image_path="2000/01/01/gone.jpg", raw_json=json.dumps({"zones": {"ring": list(LIVE_RING)}}))
    for i in (good, gone):
        presence.store(conn, i, np.ones(DIM), view=1)
    monkeypatch.setattr(classifier, "load", lambda: FakeClf())
    presence.backfill(redo=True)
    out = capsys.readouterr().out
    assert re.search(r"\d+/2 done, 1 failed", out)
    assert view_of(conn, good) == need("PRESENCE_VIEW") and view_of(conn, gone) == 1


def test_backfill_embeds_each_visit_with_its_own_zones_and_trigger(monkeypatch, conn, add_visit):
    ring_zones = {"ring": list(LIVE_RING), "crop": list(LIVE_CROP)}
    rim = uniform_visit(add_visit, 100, {"zones": ring_zones, "trigger": "rim"}, shape=(540, 960))
    ground = uniform_visit(add_visit, 100, {"zones": {**ring_zones, "ground_crop": list(GROUND_CROP)},
                                            "trigger": "ground"}, shape=(540, 960))
    older = uniform_visit(add_visit, 100, {}, shape=(540, 960))  # no zones saved: the calibration history
    db.add_calibration(conn, ran_at="2000-01-01T00:00:00", reason="manual", status="moved", summary="",
                       after_json=json.dumps(ring_zones))
    clf = FakeClf()
    monkeypatch.setattr(classifier, "load", lambda: clf)
    presence.backfill()
    bowl_side = need("presence_square")(SMALL, zones.Zones.from_dict(ring_zones))[2]
    assert clf.shapes == [(bowl_side, bowl_side, 3), (960, 960, 3), (bowl_side, bowl_side, 3)]  # newest first
    assert {view_of(conn, i) for i in (rim, ground, older)} == {need("PRESENCE_VIEW")}


def test_the_cli_takes_backfill_redo_and_limit(monkeypatch, conn):
    calls = []
    monkeypatch.setattr(presence, "backfill", lambda limit=None, redo=False: calls.append((limit, redo)) or 0)
    for argv in (["--backfill", "--redo"], ["--backfill", "--redo", "--limit", "5"], ["--backfill"]):
        monkeypatch.setattr("sys.argv", ["presence", *argv])
        presence.main()
    assert calls == [(None, True), (5, True), (None, False)]


# --- 11: the watcher's trigger path -----------------------------------------------------------------------------

@pytest.fixture
def trigger(monkeypatch, conn, yard):
    """handle_event with the camera, the classifier and Claude faked; the real presence.store and presence.check."""
    from birdbrain.identify import Result, Sighting

    z = live()
    monkeypatch.setattr(watch.zones, "current", lambda: z)
    burst = [jpeg(np.full(BIG, 90, np.uint8))] * 3
    monkeypatch.setattr(watch, "capture_burst", lambda: burst)
    monkeypatch.setattr(watch, "presence_mode", lambda: "shadow")
    seen = []

    def box_embedding(jpeg_bytes, zones_in_use, trigger_kind):
        seen.append((zones_in_use, trigger_kind))
        return vec(True, yard)

    monkeypatch.setattr(watch.classifier, "box_embedding", box_embedding)
    monkeypatch.setattr(watch.classifier, "gate_check", lambda emb, night, settled: (None, None))
    monkeypatch.setattr(watch.classifier, "run", lambda *a, **k: None)
    monkeypatch.setattr(watch.identify, "identify", lambda *a, **k: Result(
        Sighting(visitor="none", species=None, count=0, confidence="high", notes="", circle=None, also=[]),
        "claude-haiku-5-5", False, False, {}))
    return z, seen


def as_zones(z) -> "zones.Zones":
    return z if isinstance(z, zones.Zones) else zones.Zones.from_dict(z)


def test_a_rim_trigger_embeds_the_bowl_and_stores_the_current_view(trigger, conn):
    z, seen = trigger
    watch.handle_event(trigger="rim")
    assert len(seen) >= 1 and all(t == "rim" and as_zones(zz) == z for zz, t in seen)
    v = db.latest_visit(conn, real_only=False)
    assert view_of(conn, v["id"]) == need("PRESENCE_VIEW")
    assert json.loads(v["raw_json"])["calls"]["presence"]["view"] == "bowl"


def test_a_ground_trigger_embeds_the_padded_ground_crop(trigger, conn):
    z, seen = trigger
    watch.handle_event(trigger="ground")
    assert seen and all(t == "ground" and as_zones(zz) == z for zz, t in seen)
    v = db.latest_visit(conn, real_only=False)
    assert view_of(conn, v["id"]) == need("PRESENCE_VIEW")
    assert json.loads(v["raw_json"])["calls"]["presence"]["view"] == "padded"


def test_a_water_trigger_is_a_bath_trigger(trigger, conn):
    z, seen = trigger
    watch.handle_event(trigger="water")
    assert seen[0][1] == "water"
    assert json.loads(db.latest_visit(conn, real_only=False)["raw_json"])["calls"]["presence"]["view"] == "bowl"


# --- 12: coverage is judged against the view, not the crop box --------------------------------------------------

def false_trigger_to_animal(conn, add_visit, circle, trigger="rim", crop=NARROW_CROP):
    raw = {"zones": {"ring": list(LIVE_RING), "crop": list(crop), "ground_crop": list(GROUND_CROP)}, "trigger": trigger}
    vid = add_visit(visitor="none", species=None, raw_json=json.dumps(raw))
    presence.store(conn, vid, vec(False, np.random.default_rng(vid)))
    db.correct_visit(conn, vid, "squirrel", None, None, [circle])
    return vid


def test_a_circle_on_the_rim_in_the_old_cut_strip_counts_for_a_rim_visit(conn, add_visit):
    vid = false_trigger_to_animal(conn, add_visit, RIM_LEFT)
    assert vid not in left_out(conn)
    assert presence.teaches(conn, vid)["used"] is True


def test_a_circle_far_outside_the_bowl_still_leaves_a_rim_visit_out(conn, add_visit):
    vid = false_trigger_to_animal(conn, add_visit, {"x": 0.05, "y": 0.5347, "r": 0.015})
    assert vid in left_out(conn)


def test_coverage_against_the_view_box_meets_the_minimum_for_the_rim_circle():
    box = need("presence_view_box")(live(), "rim")
    assert presence.coverage(RIM_LEFT, box) >= presence.MIN_COVERAGE
    assert presence.coverage(RIM_LEFT, NARROW_CROP) < presence.MIN_COVERAGE  # the old box missed it


def test_a_ground_visit_is_judged_against_its_ground_crop(conn, add_visit):
    low = {"x": 0.5, "y": 0.75, "r": 0.03}   # inside the ground crop, nowhere near the bowl
    assert false_trigger_to_animal(conn, add_visit, low, trigger="ground") not in left_out(conn)
    high = {"x": 0.5, "y": 0.2, "r": 0.03}   # above the ground crop
    assert false_trigger_to_animal(conn, add_visit, high, trigger="ground") in left_out(conn)


# --- 13: the visit page's dashed box ----------------------------------------------------------------------------

@pytest.fixture
def client(conn):
    with TestClient(webapp.app) as c:
        yield c


def dashed_box(html) -> tuple[float, float, float, float]:
    rects = re.findall(r'<rect class="vz-crop"[^>]*? x="([\d.]+)" y="([\d.]+)" width="([\d.]+)" height="([\d.]+)"', html)
    assert len(rects) == 1, "one dashed box on the photo"
    return tuple(float(v) for v in rects[0])


def test_the_dashed_box_of_a_rim_visit_is_the_view_box_not_the_crop_box(client, add_visit):
    z = live(crop=NARROW_CROP)
    vid = add_visit(raw_json=json.dumps({"trigger": "rim", "zones": z.to_dict()}))
    box = dashed_box(client.get(f"/visit/{vid}").text)
    x, y, w, h = need("presence_view_box")(z, "rim")
    assert box == pytest.approx((x * 160, y * 90, w * 160, h * 90), abs=0.02)
    assert box != pytest.approx((NARROW_CROP[0] * 160, NARROW_CROP[1] * 90, NARROW_CROP[2] * 160, NARROW_CROP[3] * 90),
                                abs=0.5)


def test_the_dashed_box_of_a_water_visit_is_the_bowl_view_too(client, add_visit):
    vid = add_visit(raw_json=json.dumps({"trigger": "water", "zones": live(crop=NARROW_CROP).to_dict()}))
    x, y, w, h = need("presence_view_box")(live(crop=NARROW_CROP), "water")
    assert dashed_box(client.get(f"/visit/{vid}").text) == pytest.approx((x * 160, y * 90, w * 160, h * 90), abs=0.02)


def test_the_dashed_box_of_a_ground_visit_is_the_ground_crop(client, add_visit):
    vid = add_visit(visitor="none", species=None, raw_json=json.dumps({"trigger": "ground", "zones": live().to_dict()}))
    assert dashed_box(client.get(f"/visit/{vid}").text) == pytest.approx((0.0, 45.0, 160.0, 45.0), abs=0.02)  # GROUND_CROP


def test_the_dashed_box_without_a_ring_is_the_crop_the_pi_pads(client, add_visit):
    z = zones.Zones(crop=LIVE_CROP)
    vid = add_visit(raw_json=json.dumps({"trigger": "rim", "zones": z.to_dict()}))
    assert dashed_box(client.get(f"/visit/{vid}").text) == pytest.approx(
        (LIVE_CROP[0] * 160, LIVE_CROP[1] * 90, LIVE_CROP[2] * 160, LIVE_CROP[3] * 90), abs=0.02)


# --- 14: what must not change -----------------------------------------------------------------------------------

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
