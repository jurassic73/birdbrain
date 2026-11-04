"""The Pi animal check's view: a square around the whole bowl for birdbath triggers, the ground crop padded to a
square for ground ones (classifier.presence_patch), and embeddings versioned by it (embeddings.view)."""

import json
import re
import sqlite3

import cv2
import numpy as np
import pytest

from conftest import FROZEN_NOW
from birdbrain import classifier, config, db, presence, watch, zones
from birdbrain.web import app as webapp

# The live zones on atmos (2026-11-04): the crop box is 873 x 539 px on a 1920 x 1080 frame.
RING = (0.4683, 0.5347, 0.1659, 0.1175, 0.1345, 0.0966, 0.5224, 0.4681)
CROP = (0.2477, 0.2236, 0.4543, 0.4982)
GROUND_CROP = (0.003, 0.0, 0.96, 0.9304)
LIVE = {"ring": list(RING), "spots": [], "roi": None, "crop": list(CROP), "ground": [], "ground_crop": list(GROUND_CROP)}
DIM = 16


def vec(animal: bool, rng) -> np.ndarray:
    v = rng.normal(0, 0.3, DIM)
    v[0] += 1.0 if animal else -1.0
    return v / np.linalg.norm(v)


# --- geometry ---------------------------------------------------------------------------------------------

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


def frame(h=1080, w=1920, value=60):
    rng = np.random.default_rng(0)
    return np.clip(rng.normal(value, 20, (h, w, 3)), 0, 255).astype(np.uint8)


def test_the_patch_is_square_whatever_the_view():
    img = frame()
    patch, kind = classifier.presence_patch(img, LIVE, "rim")
    assert kind == "bowl" and patch.shape[0] == patch.shape[1]
    patch, kind = classifier.presence_patch(img, zones.Zones.from_dict(LIVE), "water")
    assert kind == "bowl"
    patch, kind = classifier.presence_patch(img, LIVE, "ground")
    x0, y0, x1, y1 = zones.pixel_box(img.shape, GROUND_CROP)
    assert kind == "padded" and patch.shape[0] == patch.shape[1] == max(x1 - x0, y1 - y0)
    patch, kind = classifier.presence_patch(img, {"crop": list(CROP)}, "rim")
    assert kind == "padded" and patch.shape[0] == patch.shape[1]
    patch, kind = classifier.presence_patch(img, {}, "rim")
    assert kind == "frame" and patch.shape[:2] == (1920, 1920)
    patch, kind = classifier.presence_patch(img, {"ring": list(RING)}, "ground")  # no ground crop
    assert kind == "frame"


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


# --- the stored embeddings ----------------------------------------------------------------------------------

OLD_EMBEDDINGS = """CREATE TABLE embeddings (visit_id INTEGER PRIMARY KEY REFERENCES visits(id) ON DELETE CASCADE,
    vec BLOB NOT NULL, animal_vec BLOB, animal_frame INTEGER, animal_score REAL, animal_by_owner INTEGER)"""


def test_init_db_adds_the_view_column_to_an_old_database(tmp_path):
    c = sqlite3.connect(tmp_path / "old.db")
    c.row_factory = sqlite3.Row
    c.execute("CREATE TABLE visits (id INTEGER PRIMARY KEY, captured_at TEXT)")
    c.execute(OLD_EMBEDDINGS)
    c.execute("INSERT INTO visits (id) VALUES (1)")
    c.execute("INSERT INTO embeddings (visit_id, vec) VALUES (1, X'0000')")
    c.commit()
    db.init_db(c)
    assert c.execute("SELECT view FROM embeddings WHERE visit_id = 1").fetchone()["view"] is None


def yard(conn, add_visit, view, rng, n_animal=12, n_empty=40):
    ids = []
    for i in range(n_animal + n_empty):
        animal = i < n_animal
        vid = add_visit(**({} if animal else {"visitor": "none", "species": None}))
        presence.store(conn, vid, vec(animal, rng), view=view)
        ids.append(vid)
    return ids


def test_only_embeddings_of_the_current_view_train(conn, add_visit):
    rng = np.random.default_rng(1)
    old = yard(conn, add_visit, None, rng)
    assert presence.model(conn) is None and len(presence.examples(conn)["y"]) == 0
    assert presence.check(conn, vec(False, rng)) == {"model": presence.MODEL_NAME, "score": None, "empty": False,
                                                    "note": "not enough labelled frames yet"}
    assert presence.teaches(conn, old[0]) is None
    key = presence.label_key(conn)
    presence.store(conn, old[0], vec(True, rng))  # re-embedded in the current view
    assert presence.label_key(conn) != key
    for vid in old[1:]:
        presence.store(conn, vid, vec(vid <= old[11], rng))
    m = presence.model(conn)
    assert m and m["visitors"] == 12 and m["empty"] == 40
    check = presence.check(conn, vec(False, rng), view="bowl")
    assert check["empty"] is True and check["view"] == "bowl"


def test_re_embedding_forgets_the_old_burst_pick(conn, add_visit):
    rng = np.random.default_rng(2)
    vid = add_visit(visitor="none", species=None)
    presence.store(conn, vid, vec(False, rng), view=None)
    conn.execute("UPDATE embeddings SET animal_vec = ?, animal_frame = 4, animal_score = 0.9, animal_by_owner = 1 "
                 "WHERE visit_id = ?", (vec(True, rng).astype(np.float16).tobytes(), vid))
    conn.commit()
    presence.store(conn, vid, vec(False, rng))
    row = conn.execute("SELECT * FROM embeddings WHERE visit_id = ?", (vid,)).fetchone()
    assert row["view"] == classifier.PRESENCE_VIEW
    assert (row["animal_vec"], row["animal_frame"], row["animal_score"], row["animal_by_owner"]) == (None,) * 4
    # Storing the current view again keeps a pick made in it.
    conn.execute("UPDATE embeddings SET animal_frame = 2, animal_score = 0.8 WHERE visit_id = ?", (vid,))
    presence.store(conn, vid, vec(False, rng))
    assert conn.execute("SELECT animal_frame FROM embeddings WHERE visit_id = ?", (vid,)).fetchone()[0] == 2


class FakeClf:
    def __init__(self):
        self.shapes = []

    def embed(self, bgr):
        self.shapes.append(bgr.shape)
        return np.full(DIM, 0.25)


def test_the_owners_circled_frame_is_picked_again_after_a_redo(conn, add_visit):
    from birdbrain import images

    burst = [cv2.imencode(".jpg", frame(540, 960))[1].tobytes() for _ in range(3)]
    image_rel, thumb_rel = images.store_frame(burst[0], FROZEN_NOW)
    images.store_burst(burst, image_rel)
    vid = add_visit(visitor="none", species=None, image_path=image_rel, thumb_path=thumb_rel,
                    raw_json=json.dumps({"zones": LIVE, "trigger": "rim", "burst": {"best": 0}}))
    db.correct_visit(conn, vid, "bird", "Dark-eyed Junco", None, [{"x": 0.31, "y": 0.5, "r": 0.03, "frame": 2}])
    presence.store(conn, vid, vec(False, np.random.default_rng(3)), view=None)
    conn.execute("UPDATE embeddings SET animal_vec = X'0000', animal_frame = 2, animal_by_owner = 1 WHERE visit_id = ?",
                 (vid,))
    conn.commit()
    assert presence.pending_picks(conn) == []  # an old-view row isn't looked at
    presence.store(conn, vid, vec(False, np.random.default_rng(3)))
    assert [r["id"] for r in presence.pending_picks(conn)] == [vid]
    clf = FakeClf()
    assert presence.pick_frames(conn, clf=clf) == 1
    row = conn.execute("SELECT animal_frame, animal_by_owner FROM embeddings WHERE visit_id = ?", (vid,)).fetchone()
    assert (row["animal_frame"], row["animal_by_owner"]) == (2, 1)
    side = classifier.presence_square((540, 960), LIVE)[2]
    assert clf.shapes == [(side, side, 3)]  # the bowl square of the burst frame


def stored_visits(conn, add_visit, n, raw=None):
    from birdbrain import images

    ids = []
    for _ in range(n):
        image_rel, thumb_rel = images.store_frame(cv2.imencode(".jpg", frame(540, 960))[1].tobytes(), FROZEN_NOW)
        ids.append(add_visit(image_path=image_rel, thumb_path=thumb_rel,
                             raw_json=json.dumps(raw or {"zones": LIVE, "trigger": "rim"})))
    return ids


def test_backfill_redo_re_embeds_old_views_newest_first(conn, add_visit, capsys, monkeypatch):
    ids = stored_visits(conn, add_visit, 4)
    presence.store(conn, ids[0], np.zeros(DIM), view=None)
    presence.store(conn, ids[1], np.zeros(DIM), view=1)
    presence.store(conn, ids[2], np.zeros(DIM))  # already the current view
    missing = add_visit(image_path="gone.jpg")  # no photo on disk: counted as failed
    clf = FakeClf()
    assert presence.backfill(clf=clf) == 1 and len(clf.shapes) == 1  # only ids[3]; old views left alone
    views = dict(conn.execute("SELECT visit_id, view FROM embeddings").fetchall())
    assert views[ids[0]] is None and views[ids[1]] == 1
    assert "1 failed" in capsys.readouterr().out

    order = []
    monkeypatch.setattr(presence, "store", lambda c, vid, emb, view=classifier.PRESENCE_VIEW: (
        order.append(vid), db_store(c, vid, emb, view)))
    assert presence.backfill(redo=True, clf=clf) == 2
    assert order == [ids[1], ids[0]]  # newest first; ids[2] and ids[3] untouched
    out = capsys.readouterr().out
    assert re.search(r"^3/3 done, 1 failed, ~\d+ min left$", out, re.M)
    assert missing not in order


db_store = presence.store


def test_backfill_prints_progress_every_25(conn, add_visit, capsys, monkeypatch):
    stored_visits(conn, add_visit, 26)
    presence.backfill(limit=26, clf=FakeClf())
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("25/26 done, 0 failed, ~") and lines[-1].startswith("26/26 done, 0 failed")


def test_backfill_cli_takes_redo(monkeypatch, conn):
    seen = []
    monkeypatch.setattr(presence, "backfill", lambda limit=None, redo=False: seen.append((limit, redo)) or 0)
    monkeypatch.setattr("sys.argv", ["presence", "--backfill", "--redo", "--limit", "5"])
    presence.main()
    assert seen == [(5, True)]


# --- the watcher and the pages ----------------------------------------------------------------------------

def test_the_watcher_embeds_the_bowl_view_and_records_it(monkeypatch, conn):
    from birdbrain.identify import Result, Sighting

    monkeypatch.setattr(config, "MOTION_RING", RING)
    monkeypatch.setattr(config, "CROP_ROI", CROP)
    burst = [cv2.imencode(".jpg", np.full((1080, 1920, 3), 90, np.uint8))[1].tobytes()] * 3
    monkeypatch.setattr(watch, "capture_burst", lambda: burst)
    monkeypatch.setattr(watch, "presence_mode", lambda: "shadow")
    seen = []
    monkeypatch.setattr(classifier, "box_embedding", lambda jpeg, z, trigger: seen.append((z, trigger)) or np.ones(DIM))
    views = []
    monkeypatch.setattr(presence, "check", lambda conn, emb, view=None: views.append(view) or {
        "model": presence.MODEL_NAME, "score": 0.9, "threshold": 0.4, "empty": False})
    monkeypatch.setattr(watch.identify, "identify", lambda *a, **k: Result(
        Sighting(visitor="bird", species="Bushtit", count=1, confidence="high", notes="", circle=None, also=[]),
        "claude-haiku-5-5", False, False, {}))
    watch.handle_event(trigger="rim")
    assert seen and seen[0][1] == "rim" and seen[0][0].ring == RING and views[0] == "bowl"
    v = db.latest_visit(conn)
    assert json.loads(v["raw_json"])["calls"]["presence"]["view"] == "bowl"
    row = conn.execute("SELECT view FROM embeddings WHERE visit_id = ?", (v["id"],)).fetchone()
    assert row["view"] == classifier.PRESENCE_VIEW


def test_a_circle_on_the_rims_left_end_counts_as_in_view(conn, add_visit):
    raw = {"zones": LIVE, "trigger": "rim", "calls": {"circle": {"x": 0.31, "y": 0.5, "r": 0.03}}}
    box = presence.view_box(raw)
    assert presence.photo_coverage(None, raw, box) >= presence.MIN_COVERAGE
    assert presence.coverage({"x": 0.31, "y": 0.5, "r": 0.03}, CROP) >= presence.MIN_COVERAGE  # in the crop box too
    vid = add_visit(raw_json=json.dumps(raw))
    presence.store(conn, vid, np.ones(DIM))
    assert presence.teaches(conn, vid)["used"]
    above = {**raw, "calls": {"circle": {"x": 0.47, "y": 0.1, "r": 0.03}}}  # high over the bowl: out of view
    assert presence.photo_coverage(None, above, presence.view_box(above)) < presence.MIN_COVERAGE


def test_the_visit_photo_draws_the_pi_checks_view(client, add_visit):
    vid = add_visit(raw_json=json.dumps({"trigger": "rim", "zones": LIVE}))
    html = client.get(f"/visit/{vid}").text
    x, y, w, h = classifier.presence_view_box(LIVE, "rim")
    want = f'x="{round(x * 160, 2)}" y="{round(y * 90, 2)}" width="{round(w * 160, 2)}" height="{round(h * 90, 2)}"'
    assert f'class="vz-crop"' in html and want in html
    cx, cy, cw, ch = CROP
    assert f'x="{round(cx * 160, 2)}" y="{round(cy * 90, 2)}"' not in html
    assert "a square around the whole bowl" in html
    vid = add_visit(visitor="none", species=None, raw_json=json.dumps({"trigger": "ground", "zones": LIVE}))
    html = client.get(f"/visit/{vid}").text
    gx, gy, gw, gh = GROUND_CROP
    assert f'x="{round(gx * 160, 2)}" y="{round(gy * 90, 2)}" width="{round(gw * 160, 2)}"' in html


@pytest.fixture
def client(conn):
    from fastapi.testclient import TestClient
    return TestClient(webapp.app)


# --- round 2: the species gate's view, deleted photos, the gate during a redo, progress, bad zones -----------

def test_the_species_gate_says_which_view_it_saw(monkeypatch, conn):
    monkeypatch.setattr(config, "MOTION_RING", RING)
    monkeypatch.setattr(config, "CROP_ROI", CROP)
    monkeypatch.setattr(config, "CIRCLE_FROM_MOTION", False)
    burst = [cv2.imencode(".jpg", np.full((1080, 1920, 3), 90, np.uint8))[1].tobytes()] * 3
    monkeypatch.setattr(watch, "capture_burst", lambda: burst)
    monkeypatch.setattr(watch, "presence_mode", lambda: "shadow")
    monkeypatch.setattr(watch, "count_animals", lambda *a: 1)
    monkeypatch.setattr(classifier, "box_embedding", lambda jpeg, z, trigger: np.ones(DIM))
    monkeypatch.setattr(presence, "check", lambda conn, emb, view=None: {
        "model": presence.MODEL_NAME, "score": 0.9, "threshold": 0.4, "empty": False})
    monkeypatch.setattr(classifier, "gate_check", lambda emb, night, settled: (
        {"model": classifier.MODEL_NAME, "crop": "box", "top": [{"name": "Dark-eyed Junco", "score": 0.97}]},
        ("Dark-eyed Junco", "bird")))
    monkeypatch.setattr(watch.identify, "identify", lambda *a, **k: pytest.fail("Claude was asked"))
    assert watch.handle_event(trigger="rim") == "bird"
    raw = json.loads(db.latest_visit(conn)["raw_json"])
    assert raw["calls"]["classifier"]["crop"] == "bowl"
    step = webapp.decision_steps(json.dumps(raw))["steps"][-1]
    assert step["model"] == "BioCLIP 2 species check" and step["crop"] == "on a square around the whole bowl"


def test_the_species_check_step_names_the_padded_view_by_trigger():
    def where(trigger, crop):
        raw = {"trigger": trigger, "calls": {"classifier": {"top": [{"name": "Bushtit", "score": 0.9}], "crop": crop}}}
        return webapp.decision_steps(json.dumps(raw))["steps"][-1]["crop"]
    assert where("ground", "padded") == "on the ground crop"
    assert where("rim", "padded") == "on the crop box"
    assert where("rim", "frame") == "on the whole frame"
    assert where("rim", "box") == "on the crop box"            # older visits and classifier.run keep theirs
    assert where("rim", "circle") == "around Claude's circle"


def test_redo_leaves_deleted_photos_alone_and_out_of_the_limit(conn, add_visit, capsys):
    keep = stored_visits(conn, add_visit, 2)
    deleted = stored_visits(conn, add_visit, 3)
    thumb = stored_visits(conn, add_visit, 1)[0]
    for vid in keep + deleted + [thumb]:
        presence.store(conn, vid, np.zeros(DIM), view=None)
    conn.executemany("UPDATE visits SET photo_deleted_at = '2026-11-01T00:00:00' WHERE id = ?", [(v,) for v in deleted])
    conn.execute("UPDATE visits SET photo_level = 'thumb' WHERE id = ?", (thumb,))
    conn.commit()
    clf = FakeClf()
    assert presence.backfill(limit=1, redo=True, clf=clf) == 1  # the newest with a photo, not a deleted one
    views = dict(conn.execute("SELECT visit_id, view FROM embeddings").fetchall())
    assert views[keep[1]] == classifier.PRESENCE_VIEW and views[keep[0]] is None
    assert all(views[v] is None for v in deleted + [thumb])
    out = capsys.readouterr().out.splitlines()
    assert out[-1] == "1/1 done, 0 failed, ~0 min left, 4 photos already deleted (left as they were)"
    assert presence.backfill(redo=True, clf=clf) == 1  # the next run does the last one, still not the deleted
    assert "1/1 done, 0 failed" in capsys.readouterr().out


def gate_seen(conn):
    """A FakeClf that notes the Pi animal check's mode each time it embeds."""
    seen = []

    class Clf(FakeClf):
        def embed(self, bgr):
            with db.session() as c:
                seen.append(presence.mode(c))
            return super().embed(bgr)
    return Clf(), seen


def test_redo_holds_gate_mode_at_shadow_and_puts_it_back(conn, add_visit, capsys):
    ids = stored_visits(conn, add_visit, 2)
    for vid in ids:
        presence.store(conn, vid, np.zeros(DIM), view=None)
    db.set_value(conn, "presence_mode", "gate")
    clf, seen = gate_seen(conn)
    presence.backfill(redo=True, clf=clf)
    assert seen == ["shadow", "shadow"] and presence.mode(conn) == "gate"
    out = capsys.readouterr().out
    assert "switched to shadow" in out and "back in gate mode" in out


def test_redo_holds_gate_from_env_and_removes_the_setting_again(conn, add_visit, monkeypatch, capsys):
    monkeypatch.setattr(config, "PRESENCE", "gate")
    vid = stored_visits(conn, add_visit, 1)[0]
    presence.store(conn, vid, np.zeros(DIM), view=None)
    clf, seen = gate_seen(conn)
    presence.backfill(redo=True, clf=clf)
    assert seen == ["shadow"] and db.get_value(conn, "presence_mode") is None and presence.mode(conn) == "gate"


def test_redo_puts_gate_back_even_when_it_fails_and_leaves_shadow_and_plain_backfills_alone(conn, add_visit, capsys):
    vid = stored_visits(conn, add_visit, 1)[0]
    presence.store(conn, vid, np.zeros(DIM), view=None)
    db.set_value(conn, "presence_mode", "gate")

    class Stop(FakeClf):
        def embed(self, bgr):
            raise KeyboardInterrupt  # Ctrl-C halfway

    with pytest.raises(KeyboardInterrupt):
        presence.backfill(redo=True, clf=Stop())
    assert presence.mode(conn) == "gate" and "back in gate mode" in capsys.readouterr().out
    # Not in gate mode, or no redo: the mode isn't touched.
    db.set_value(conn, "presence_mode", "shadow")
    clf, seen = gate_seen(conn)
    presence.backfill(redo=True, clf=clf)
    assert seen == ["shadow"] and db.get_value(conn, "presence_mode") == "shadow"
    db.set_value(conn, "presence_mode", "gate")
    stored_visits(conn, add_visit, 1)
    clf, seen = gate_seen(conn)
    presence.backfill(clf=clf)
    assert seen == ["gate"]
    assert "switched to shadow" not in capsys.readouterr().out


def test_redo_leaves_the_mode_alone_if_the_owner_changed_it_meanwhile(conn, add_visit, capsys):
    vid = stored_visits(conn, add_visit, 1)[0]
    presence.store(conn, vid, np.zeros(DIM), view=None)
    db.set_value(conn, "presence_mode", "gate")

    class Owner(FakeClf):
        def embed(self, bgr):
            with db.session() as c:
                db.set_value(c, "presence_mode", "off")
            return super().embed(bgr)

    presence.backfill(redo=True, clf=Owner())
    assert presence.mode(conn) == "off" and "left as it is now" in capsys.readouterr().out


def test_progress_every_30_seconds_and_the_last_line_once(conn, add_visit, capsys, monkeypatch):
    stored_visits(conn, add_visit, 4)
    now = [1000.0]
    monkeypatch.setattr(presence, "_clock", lambda: now[0])

    class Slow(FakeClf):
        def embed(self, bgr):
            now[0] += 20  # 20 s a frame: a line after frames 2 and 4 (30 s or more since the last)
            return super().embed(bgr)

    presence.backfill(clf=Slow())
    assert capsys.readouterr().out.splitlines() == ["2/4 done, 0 failed, ~1 min left", "4/4 done, 0 failed, ~0 min left"]
    stored_visits(conn, add_visit, 25)  # exactly 25: the 25th frame's line is the last line, printed once
    presence.backfill(clf=FakeClf())
    assert capsys.readouterr().out.splitlines() == ["25/25 done, 0 failed, ~0 min left"]


def test_the_cli_refuses_redo_without_backfill(monkeypatch, conn, capsys):
    monkeypatch.setattr(presence, "backfill", lambda *a, **k: pytest.fail("backfill ran"))
    monkeypatch.setattr("sys.argv", ["presence", "--redo"])
    with pytest.raises(SystemExit) as e:
        presence.main()
    assert e.value.code == 2 and "--redo goes with --backfill" in capsys.readouterr().err


OFF_FRAME = [[1.2, 0.1, 0.3, 0.3], [0.2, 0.1, 0, 0.3], [0.2, 0.1, 0.3, -0.1], [0.2, 0.1, 0.3], None, "box",
             [0.2, float("nan"), 0.3, 0.3]]
BAD_RINGS = [RING[:5], [0.5, None, 0.1, 0.1, 0.08, 0.08], [0.5, float("nan"), 0.1, 0.1, 0.08, 0.08], "ring",
             [0.5, 0.5, 0, 0.1, 0.08, 0.08], [1.5, 0.5, 0.1, 0.1, 0.08, 0.08], list(RING) + [0] * 5]


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


@pytest.mark.parametrize("ring", BAD_RINGS)
def test_a_malformed_ring_is_no_ring_everywhere(ring, conn, add_visit, client):
    z = {"ring": ring, "crop": list(CROP)}
    assert classifier.usable_ring(ring) is None
    assert classifier.view_kind(z, "rim") == "padded" and classifier.presence_square((90, 160), z) is None
    assert classifier.presence_view_box(z, "rim") == CROP
    assert classifier.presence_patch(frame(90, 160), z, "rim")[1] == "padded"
    # Stored with a visit, it doesn't break retraining for everything else, or the visit's page.
    raw = {"zones": z, "trigger": "rim", "calls": {"circle": {"x": 0.47, "y": 0.45, "r": 0.03}}}
    vid = add_visit(raw_json=json.dumps(raw))
    presence.store(conn, vid, np.ones(DIM))
    assert presence.view_box(raw) == CROP
    assert vid in presence.examples(conn)["ids"]
    page = client.get(f"/visit/{vid}")
    assert page.status_code == 200
    x, y, w, h = CROP
    assert f'x="{round(x * 160, 2)}" y="{round(y * 90, 2)}"' in page.text and "Dashed box" in page.text


def test_the_visit_page_draws_the_view_its_embedding_was_made_of(conn, add_visit, client):
    vid = add_visit(raw_json=json.dumps({"trigger": "rim", "zones": LIVE}))

    def rect(box):
        x, y, w, h = box
        return f'x="{round(x * 160, 2)}" y="{round(y * 90, 2)}" width="{round(w * 160, 2)}" height="{round(h * 90, 2)}"'

    old = classifier.old_view_box(LIVE, "rim")
    assert abs(old[0] - 0.335) < 0.001 and abs(old[0] + old[2] - 0.615) < 0.001  # what the center cut kept
    presence.store(conn, vid, np.ones(DIM), view=None)
    html = client.get(f"/visit/{vid}").text
    assert rect(old) in html and "the middle square of the crop box" in html
    presence.store(conn, vid, np.ones(DIM))  # re-embedded
    html = client.get(f"/visit/{vid}").text
    assert rect(classifier.presence_view_box(LIVE, "rim")) in html and rect(old) not in html
    assert "a square around the whole bowl" in html and "Claude sees the crop box" in html
    # A ground visit of the old view: the middle of the ground crop; of the current view the ground crop itself.
    gid = add_visit(visitor="none", species=None, raw_json=json.dumps({"trigger": "ground", "zones": LIVE}))
    presence.store(conn, gid, np.ones(DIM), view=1)
    html = client.get(f"/visit/{gid}").text
    assert rect(classifier.old_view_box(LIVE, "ground")) in html and "the middle square of the ground crop" in html
    assert "Claude sees the ground crop" in html


def test_the_tip_names_the_view_for_no_ring_and_no_trigger(conn, add_visit, client):
    vid = add_visit(raw_json=json.dumps({"trigger": "rim", "zones": {"crop": list(CROP)}}))
    html = client.get(f"/visit/{vid}").text
    assert "looked at, the crop box, so" in html and "whole bowl" not in html
    vid = add_visit(raw_json=json.dumps({"zones": LIVE}))  # no trigger saved: a birdbath view
    assert "looked at, a square around the whole bowl, so" in client.get(f"/visit/{vid}").text


def test_backfill_falls_back_to_the_current_zones(conn, add_visit, monkeypatch):
    monkeypatch.setattr(config, "MOTION_RING", RING)
    monkeypatch.setattr(config, "CROP_ROI", CROP)
    zones.forget()
    stored_visits(conn, add_visit, 1, raw={"trigger": "rim"})  # no zones saved, no calibration history
    clf = FakeClf()
    assert presence.backfill(clf=clf) == 1
    side = classifier.presence_square((540, 960), LIVE)[2]
    assert clf.shapes == [(side, side, 3)]


def test_pick_frames_on_a_ground_visit_embeds_the_padded_ground_crop(conn, add_visit):
    from birdbrain import images

    burst = [cv2.imencode(".jpg", frame(540, 960))[1].tobytes() for _ in range(3)]
    image_rel, thumb_rel = images.store_frame(burst[0], FROZEN_NOW)
    images.store_burst(burst, image_rel)
    vid = add_visit(visitor="none", species=None, image_path=image_rel, thumb_path=thumb_rel,
                    raw_json=json.dumps({"zones": LIVE, "trigger": "ground", "burst": {"best": 0}}))
    presence.store(conn, vid, vec(False, np.random.default_rng(4)))
    db.correct_visit(conn, vid, "squirrel", None, None, [{"x": 0.5, "y": 0.7, "r": 0.05, "frame": 2}])
    clf = FakeClf()
    assert presence.pick_frames(conn, clf=clf) == 1
    x0, y0, x1, y1 = zones.pixel_box((540, 960), GROUND_CROP)
    side = max(x1 - x0, y1 - y0)
    assert clf.shapes == [(side, side, 3)]  # the owner's frame, as the padded ground crop
    row = conn.execute("SELECT animal_frame, animal_by_owner FROM embeddings WHERE visit_id = ?", (vid,)).fetchone()
    assert (row["animal_frame"], row["animal_by_owner"]) == (2, 1)
