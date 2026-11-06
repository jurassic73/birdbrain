"""The Pi animal check's view: a square around the whole bowl for birdbath triggers, the ground crop padded to a
square for ground ones (classifier.presence_patch), and embeddings versioned by it (embeddings.view)."""

import json
import re

import cv2
import numpy as np
import pytest

from helpers.clock import FROZEN_NOW
from helpers.images import flat_jpeg
from helpers.presence_bowl import CROP, DIM, GROUND_CROP, LIVE, RING, frame
from birdbrain import classifier, config, db, presence, watch, zones
from birdbrain.web import app as webapp


def vec(animal: bool, rng) -> np.ndarray:
    v = rng.normal(0, 0.3, DIM)
    v[0] += 1.0 if animal else -1.0
    return v / np.linalg.norm(v)


# The photo a stored visit or burst frame is made of when the pixels never matter: substream-sized, flat, encoded once.
SMALL_JPEG = flat_jpeg(540, 960)


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

    burst = [SMALL_JPEG] * 3
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
        image_rel, thumb_rel = images.store_frame(SMALL_JPEG, FROZEN_NOW)
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
    assert out[-2] == "1/1 done, 0 failed, ~0 min left, 4 photos gone (left as they were)"
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


def test_redo_stopped_halfway_stays_in_shadow_and_leaves_shadow_and_plain_backfills_alone(conn, add_visit, capsys):
    vid = stored_visits(conn, add_visit, 1)[0]
    presence.store(conn, vid, np.zeros(DIM), view=None)
    db.set_value(conn, "presence_mode", "gate")

    class Stop(FakeClf):
        def embed(self, bgr):
            raise KeyboardInterrupt  # Ctrl-C halfway

    with pytest.raises(KeyboardInterrupt):
        presence.backfill(redo=True, clf=Stop())
    # A12/A18: a partial redo leaves gate in shadow (the frame is still to redo) and keeps the hold.
    out = capsys.readouterr().out
    assert presence.mode(conn) == "shadow" and "back in gate mode" not in out
    assert "Not finished: 1 frame still to redo. The Pi animal check stays in shadow" in out
    assert db.get_value(conn, presence.HOLD_KEY) is not None
    for key in (presence.HOLD_KEY, presence.RESTORE_KEY):
        db.delete_value(conn, key)
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

    burst = [SMALL_JPEG] * 3
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


@pytest.fixture
def hold_unread(monkeypatch):
    """The watcher's cached hold forgotten, so the next probe_ready reads the database."""
    def forget():
        monkeypatch.setattr(watch, "_hold_cache", (None, False))
    forget()
    return forget


def test_a_redo_holds_rain_probes_and_a_stale_hold_runs_out(monkeypatch, conn, hold_unread):
    monkeypatch.setattr(watch.presence, "trained", lambda: True)
    monkeypatch.setattr(watch.classifier, "usable", lambda: True)
    monkeypatch.setattr(watch, "presence_mode", lambda: "shadow")
    assert watch.probe_ready() is True
    presence.hold_gate()
    hold_unread()
    assert presence.held(conn) and watch.probe_ready() is False
    now = [float(db.get_value(conn, presence.HOLD_KEY))]
    monkeypatch.setattr(presence, "_now", lambda: now[0])
    now[0] += presence.HOLD_MAX - 60
    presence.check_in()  # a progress line: the hold's time moves on
    now[0] += presence.HOLD_MAX - 60
    hold_unread()
    assert watch.probe_ready() is False
    now[0] += 120  # a killed redo: no check-in for HOLD_MAX
    hold_unread()
    assert not presence.held(conn) and watch.probe_ready() is True
    db.set_value(conn, presence.HOLD_KEY, "garbage")
    assert not presence.held(conn)


def test_the_hold_is_read_at_most_every_15_seconds(monkeypatch, conn, hold_unread):
    clock = [5000.0]
    monkeypatch.setattr(watch.time, "monotonic", lambda: clock[0])
    assert watch.presence_held() is False
    db.set_value(conn, presence.HOLD_KEY, str(presence._now()))
    clock[0] += 10
    assert watch.presence_held() is False  # still the cached answer
    clock[0] += 10
    assert watch.presence_held() is True


def test_the_hold_is_read_on_a_freshly_booted_machine(monkeypatch, conn, hold_unread):
    """time.monotonic() counts from boot on Linux: with 0.0 as the "never read" mark the watcher's first 15 s
    answered "not held" without looking (a CI runner VM is that young when the tests start)."""
    db.set_value(conn, presence.HOLD_KEY, str(presence._now()))
    monkeypatch.setattr(watch.time, "monotonic", lambda: 5.0)
    assert watch.presence_held() is True


def test_the_hold_is_on_while_the_redo_runs_and_gone_after(conn, add_visit, hold_unread):
    vid = stored_visits(conn, add_visit, 1)[0]
    presence.store(conn, vid, np.zeros(DIM), view=None)
    seen = []

    class Clf(FakeClf):
        def embed(self, bgr):
            with db.session() as c:
                seen.append(presence.held(c))
            return super().embed(bgr)

    presence.backfill(redo=True, clf=Clf())
    assert seen == [True] and db.get_value(conn, presence.HOLD_KEY) is None
    stored_visits(conn, add_visit, 1)
    seen.clear()
    presence.backfill(clf=Clf())  # a plain backfill holds nothing
    assert seen == [False]


def test_a_probe_during_a_redo_is_unchecked_not_dropped_as_empty(monkeypatch, conn, hold_unread):
    monkeypatch.setattr(config, "MOTION_RING", RING)
    monkeypatch.setattr(config, "CROP_ROI", CROP)
    burst = [cv2.imencode(".jpg", np.full((1080, 1920, 3), 90, np.uint8))[1].tobytes()] * 3
    monkeypatch.setattr(watch, "capture_burst", lambda: burst)
    monkeypatch.setattr(watch, "presence_mode", lambda: "shadow")
    monkeypatch.setattr(classifier, "box_embedding", lambda jpeg, z, trigger: np.ones(DIM))
    monkeypatch.setattr(presence, "check", lambda conn, emb, view=None: {
        "model": presence.MODEL_NAME, "score": 0.01, "threshold": 0.4, "empty": True})
    monkeypatch.setattr(watch.identify, "identify", lambda *a, **k: pytest.fail("Claude was asked"))
    db.set_value(conn, presence.HOLD_KEY, str(presence._now()))
    assert watch.handle_event(probe=True, trigger="rim") == watch.PROBE_UNCHECKED
    assert conn.execute("SELECT COUNT(*) FROM counters WHERE name = 'probe_empty'").fetchone()[0] == 0
    db.delete_value(conn, presence.HOLD_KEY)
    hold_unread()
    assert watch.handle_event(probe=True, trigger="rim") == watch.PROBE_EMPTY


def test_the_watchers_loop_picks_no_burst_frames_during_a_redo(monkeypatch, conn):
    class Stop(Exception):
        pass

    def stop(seconds):
        raise Stop

    picks = []
    monkeypatch.setattr(config, "CLASSIFIER", "gate")
    monkeypatch.setattr(presence, "pick_frames", lambda conn, limit=None, busy=None: picks.append(limit))
    monkeypatch.setattr(presence.time, "sleep", stop)
    db.set_value(conn, "presence_mode", "shadow")
    db.set_value(conn, presence.HOLD_KEY, str(presence._now()))
    with pytest.raises(Stop):
        presence._keep_fresh(60)
    assert picks == []
    db.delete_value(conn, presence.HOLD_KEY)  # released: picks go on
    with pytest.raises(Stop):
        presence._keep_fresh(60)
    assert picks == [1]


def test_a_redo_with_a_limit_stays_in_shadow_until_a_later_one_finishes(conn, add_visit, capsys):
    ids = stored_visits(conn, add_visit, 3)
    for vid in ids:
        presence.store(conn, vid, np.zeros(DIM), view=None)
    db.set_value(conn, "presence_mode", "gate")
    presence.backfill(limit=1, redo=True, clf=FakeClf())
    out = capsys.readouterr().out
    assert presence.mode(conn) == "shadow" and db.get_value(conn, presence.HOLD_KEY) is not None
    assert out.splitlines()[-1] == (
        "Not finished: 2 frames still to redo. The Pi animal check stays in shadow until a redo finishes, and the "
        "watcher sends no rain probes and picks no burst frames for 3 hours from now, when the hold lapses "
        "(shadow stays). Run --backfill --redo again to finish and put gate back.")
    assert "back in gate mode" not in out
    # The next run starts in the shadow the first one left, and puts gate back when it finishes.
    presence.backfill(redo=True, clf=FakeClf())
    out = capsys.readouterr().out
    assert presence.mode(conn) == "gate" and presence.still_to_redo(conn) == 0
    assert "left the Pi animal check in shadow" in out and out.splitlines()[-1] == "The Pi animal check is back in gate mode."
    assert db.get_value(conn, presence.HOLD_KEY) is None and db.get_value(conn, presence.RESTORE_KEY) is None


def test_a_complete_redo_with_an_unreadable_file_still_puts_gate_back(conn, add_visit, capsys):
    good = stored_visits(conn, add_visit, 1)[0]
    missing = add_visit(image_path="2026/11/01/120000_gone.jpg", raw_json=json.dumps({"zones": LIVE, "trigger": "rim"}))
    deleted = stored_visits(conn, add_visit, 1)[0]
    for vid in (good, missing, deleted):
        presence.store(conn, vid, np.zeros(DIM), view=None)
    conn.execute("UPDATE visits SET photo_deleted_at = '2026-11-01T00:00:00' WHERE id = ?", (deleted,))
    conn.commit()
    db.set_value(conn, "presence_mode", "gate")
    assert presence.backfill(redo=True, clf=FakeClf()) == 1
    out = capsys.readouterr().out.splitlines()
    assert out[-2] == "2/2 done, 1 failed, ~0 min left, 1 photos gone (left as they were)"
    assert out[-1] == "The Pi animal check is back in gate mode." and presence.mode(conn) == "gate"
    assert presence.still_to_redo(conn) == 1 and presence.still_to_redo(conn, {missing}) == 0


def burst_false_trigger(conn, add_visit, owner_frame=None, view=None):
    """A false trigger corrected to a bird, its burst on disk, embedded in `view` (None: the old way)."""
    from birdbrain import images

    burst = [SMALL_JPEG] * 3
    image_rel, thumb_rel = images.store_frame(burst[0], FROZEN_NOW)
    images.store_burst(burst, image_rel)
    vid = add_visit(visitor="none", species=None, image_path=image_rel, thumb_path=thumb_rel,
                    raw_json=json.dumps({"zones": LIVE, "trigger": "rim", "burst": {"best": 0}}))
    circle = {"x": 0.31, "y": 0.5, "r": 0.03, **({"frame": owner_frame} if owner_frame else {})}
    db.correct_visit(conn, vid, "bird", "Dark-eyed Junco", None, [circle])
    presence.store(conn, vid, vec(False, np.random.default_rng(vid)), view=view)
    return vid


def test_a_redo_picks_the_bursts_it_forgot_before_gate_comes_back(conn, add_visit, capsys):
    rng = np.random.default_rng(7)
    yard(conn, add_visit, classifier.PRESENCE_VIEW, rng)  # a model can be trained
    corrected = burst_false_trigger(conn, add_visit)
    circled = burst_false_trigger(conn, add_visit, owner_frame=2)
    db.set_value(conn, "presence_mode", "gate")
    order = []

    class Clf(FakeClf):
        def embed(self, bgr):
            with db.session() as c:
                order.append((presence.mode(c), presence.held(c),
                              c.execute("SELECT COUNT(*) FROM embeddings WHERE view IS NOT ?",
                                        (classifier.PRESENCE_VIEW,)).fetchone()[0]))
            return super().embed(bgr)

    clf = Clf()
    presence.backfill(redo=True, clf=clf)
    picks = dict(conn.execute("SELECT visit_id, animal_frame FROM embeddings WHERE visit_id IN (?, ?)",
                              (corrected, circled)).fetchall())
    assert picks[circled] == 2 and picks[corrected] in (1, 2, 3)  # both picked by the redo itself
    assert presence.pending_picks(conn) == []
    # 2 re-embeds, then 1 owner frame and 3 burst frames, all in shadow under the hold, none of it old by then.
    assert len(order) == 6 and all(m == "shadow" and h for m, h, _ in order) and [o[2] for o in order[2:]] == [0] * 4
    out = capsys.readouterr().out.splitlines()
    assert out[-2] == "bursts: 2/2 visits looked through, ~0 min left"
    assert out[-1] == "The Pi animal check is back in gate mode." and presence.mode(conn) == "gate"
    assert presence.label_key(conn) == presence._model["key"]  # retrained with the picks before gate came back


def test_a_redo_stopped_during_the_picks_keeps_the_hold_and_shadow(conn, add_visit, capsys):
    yard(conn, add_visit, classifier.PRESENCE_VIEW, np.random.default_rng(8))
    burst_false_trigger(conn, add_visit, owner_frame=2)
    burst_false_trigger(conn, add_visit, owner_frame=3)
    db.set_value(conn, "presence_mode", "gate")

    class Stop(FakeClf):
        def embed(self, bgr):
            if len(self.shapes) == 3:
                raise KeyboardInterrupt  # Ctrl-C on the second pick, after both re-embeds and one pick
            return super().embed(bgr)

    with pytest.raises(KeyboardInterrupt):
        presence.backfill(redo=True, clf=Stop())
    out = capsys.readouterr().out
    assert "Not finished: 1 burst still to look through. The Pi animal check stays in shadow" in out
    assert presence.mode(conn) == "shadow" and presence.held(conn) and presence.still_to_redo(conn) == 0
    # Nothing left to re-embed, but the next redo still finishes the picks and puts gate back.
    clf = FakeClf()
    presence.backfill(redo=True, clf=clf)
    assert len(clf.shapes) == 1 and presence.pending_picks(conn) == []
    assert presence.mode(conn) == "gate" and not presence.held(conn)


def test_a_redo_in_shadow_leaves_the_picks_to_the_watcher(conn, add_visit, capsys):
    yard(conn, add_visit, classifier.PRESENCE_VIEW, np.random.default_rng(9))
    vid = burst_false_trigger(conn, add_visit)
    clf = FakeClf()
    presence.backfill(redo=True, clf=clf)
    assert len(clf.shapes) == 1 and [r["id"] for r in presence.pending_picks(conn)] == [vid]
    assert not presence.held(conn)  # released: the watcher picks it within a few minutes
    assert capsys.readouterr().out.splitlines()[-1] == "The Pi animal check is now in shadow mode."


def test_gate_from_env_survives_a_partial_run(conn, add_visit, monkeypatch, capsys):
    monkeypatch.setattr(config, "PRESENCE", "gate")
    for vid in stored_visits(conn, add_visit, 2):
        presence.store(conn, vid, np.zeros(DIM), view=None)
    presence.backfill(limit=1, redo=True, clf=FakeClf())
    assert presence.mode(conn) == "shadow" and db.get_value(conn, presence.RESTORE_KEY) == "env"
    presence.backfill(redo=True, clf=FakeClf())
    assert db.get_value(conn, "presence_mode") is None and presence.mode(conn) == "gate"
    assert capsys.readouterr().out.splitlines()[-1] == "The Pi animal check is back in gate mode."


@pytest.mark.parametrize("between", ["off", "gate"])
def test_the_owners_change_between_runs_wins(conn, add_visit, capsys, between):
    for vid in stored_visits(conn, add_visit, 3):
        presence.store(conn, vid, np.zeros(DIM), view=None)
    db.set_value(conn, "presence_mode", "gate")
    presence.backfill(limit=1, redo=True, clf=FakeClf())
    db.set_value(conn, "presence_mode", between)  # Settings > Pi checks
    presence.backfill(limit=1, redo=True, clf=FakeClf())
    out = capsys.readouterr().out
    if between == "off":
        assert "mode was changed to off since an earlier redo; left as it is" in out
        assert db.get_value(conn, presence.RESTORE_KEY) is None and presence.mode(conn) == "off"
        presence.backfill(redo=True, clf=FakeClf())
        assert presence.mode(conn) == "off"
        assert capsys.readouterr().out.splitlines()[-1] == "The Pi animal check is now in off mode."
    else:  # gate again: held at shadow again, gate back at the end
        assert "switched to shadow" in out and presence.mode(conn) == "shadow"
        presence.backfill(redo=True, clf=FakeClf())
        assert presence.mode(conn) == "gate"


def test_the_owners_change_during_a_partial_run_wins(conn, add_visit, capsys):
    for vid in stored_visits(conn, add_visit, 3):
        presence.store(conn, vid, np.zeros(DIM), view=None)
    db.set_value(conn, "presence_mode", "gate")

    class Owner(FakeClf):
        def embed(self, bgr):
            with db.session() as c:
                db.set_value(c, "presence_mode", "off")
            return super().embed(bgr)

    presence.backfill(limit=1, redo=True, clf=Owner())
    out = capsys.readouterr().out
    assert "left as it is now" in out and "stays in shadow" not in out and presence.mode(conn) == "off"
    assert db.get_value(conn, presence.RESTORE_KEY) is None


def test_a_partial_run_keeps_probes_and_picks_paused_until_the_hold_lapses(monkeypatch, conn, add_visit, hold_unread):
    for vid in stored_visits(conn, add_visit, 2):
        presence.store(conn, vid, np.zeros(DIM), view=None)
    now = [1_000_000.0]
    monkeypatch.setattr(presence, "_now", lambda: now[0])
    presence.backfill(limit=1, redo=True, clf=FakeClf())
    monkeypatch.setattr(watch.presence, "trained", lambda: True)
    monkeypatch.setattr(watch.classifier, "usable", lambda: True)
    monkeypatch.setattr(watch, "presence_mode", lambda: "shadow")
    now[0] += presence.HOLD_MAX - 60
    hold_unread()
    assert presence.held(conn) and watch.probe_ready() is False  # still paused, nearly 3 h on
    now[0] += 120
    hold_unread()
    assert not presence.held(conn) and watch.probe_ready() is True  # lapsed
