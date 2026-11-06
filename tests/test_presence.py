import json

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from helpers.clock import FROZEN_NOW
from birdbrain import classifier, db, presence, watch
from birdbrain.web import app as webapp

DIM = 16


def vec(animal: bool, rng) -> np.ndarray:
    """Fake embeddings: animals lean one way, empty frames the other."""
    v = rng.normal(0, 0.3, DIM)
    v[0] += 1.0 if animal else -1.0
    return v / np.linalg.norm(v)


CROP = (0.25, 0.25, 0.5, 0.5)
INSIDE = {"x": 0.5, "y": 0.5, "r": 0.05}  # a circle in the middle of the crop box
ABOVE = {"x": 0.5, "y": 0.12, "r": 0.05}  # one above it (the dove of visit 693: only its tail was in the box)


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


def test_too_few_labels_means_no_verdict(conn, add_visit):
    presence.store(conn, add_visit(), np.ones(DIM))
    assert presence.check(conn, np.ones(DIM)) == {"model": presence.MODEL_NAME, "score": None, "empty": False,
                                                 "note": "not enough labelled frames yet"}


def test_it_learns_empty_from_animal_and_keeps_every_visitor(conn, yard):
    m = presence.model(conn)
    assert m["visitors"] == 12 and m["empty"] == 40 and 0 < m["threshold"] <= presence.MAX_THRESHOLD
    assert presence.check(conn, vec(False, yard))["empty"] is True
    assert presence.check(conn, vec(True, yard))["empty"] is False


def test_it_retrains_when_a_label_changes(conn, yard, add_visit):
    first = presence.model(conn)
    assert presence.model(conn) is first  # cached while nothing changed
    empty = conn.execute("SELECT id FROM visits WHERE visitor = 'none' LIMIT 1").fetchone()[0]
    found_animal(conn, empty)  # the owner found an animal in a false trigger
    second = presence.model(conn)
    assert second is not first and second["visitors"] == 13
    db.confirm_visits(conn, [empty + 1])
    assert presence.model(conn) is not second  # a confirm retrains too (it counts extra)


def test_the_batched_fit_matches_fitting_each_fold_alone(yard, conn):
    X, y, w, ids = presence.labelled(conn)
    w = w.copy()
    w[::7] = presence.CONFIRMED_WEIGHT
    rows = np.stack([ids % 3 != 0, np.ones(len(y), bool)], axis=1)
    coef, bias = presence.fit(X, y, w, rows)
    keep = rows[:, 0]
    alone, alone_bias = presence.fit(X[keep], y[keep], w[keep], np.ones((keep.sum(), 1), bool))
    assert np.allclose(coef[:, 0], alone[:, 0], atol=1e-4) and abs(bias[0] - alone_bias[0]) < 1e-4
    assert not np.allclose(coef[:, 0], coef[:, 1], atol=1e-3)  # the other column trained on more frames


def test_a_trigger_never_waits_for_a_retrain(monkeypatch, conn, yard, add_visit):
    monkeypatch.setattr(presence, "BACKGROUND", True)
    first = presence.model(conn)  # nothing trained yet in this process: the first call trains right away
    assert first is not None
    vid = add_visit()
    presence.store(conn, vid, vec(True, yard))
    with presence._train_lock:  # hold any training back: the trigger still gets an answer at once
        assert presence.model(conn) is first
        thread = presence._thread
        assert thread is not None and thread.is_alive()
        presence.model(conn)
        assert presence._thread is thread  # one training at a time
    thread.join(timeout=10)
    assert not thread.is_alive()
    second = presence.model(conn)
    assert second is not first and second["visitors"] == 13


def test_the_watchers_loop_picks_up_a_correction_without_a_trigger(monkeypatch, conn, yard):
    class Stop(Exception):
        pass

    def stop(seconds):
        raise Stop

    first = presence.model(conn)
    empty = conn.execute("SELECT id FROM visits WHERE visitor = 'none' LIMIT 1").fetchone()[0]
    found_animal(conn, empty)
    monkeypatch.setattr(presence.time, "sleep", stop)
    with pytest.raises(Stop):
        presence._keep_fresh(60)  # one round of the loop start_background() runs
    assert presence._model["model"] is not first and presence._model["model"]["visitors"] == 13


def test_gate_skips_claude_and_keeps_the_frame_for_review(monkeypatch, conn, yard):
    burst = [cv2.imencode(".jpg", np.full((1080, 1920, 3), 90, np.uint8))[1].tobytes()] * 3
    monkeypatch.setattr(watch, "capture_burst", lambda: burst)
    monkeypatch.setattr(watch, "presence_mode", lambda: "gate")
    emb = vec(False, yard)
    monkeypatch.setattr(classifier, "box_embedding", lambda jpeg, zones, trigger: emb)
    asked = []
    monkeypatch.setattr(watch.identify, "identify", lambda *a, **k: asked.append(1))
    assert watch.handle_event() == "none"
    assert asked == []
    v = db.latest_visit(conn, real_only=False)
    raw = json.loads(v["raw_json"])
    assert v["model"] == presence.MODEL_NAME and raw["calls"]["skipped_claude"] and raw["calls"]["presence"]["empty"]
    stored = conn.execute("SELECT vec FROM embeddings WHERE visit_id = ?", (v["id"],)).fetchone()
    assert stored is not None  # it becomes training data once checked


def test_shadow_records_the_score_and_still_asks_claude(monkeypatch, conn, yard):
    from birdbrain.identify import Result, Sighting
    burst = [cv2.imencode(".jpg", np.full((1080, 1920, 3), 90, np.uint8))[1].tobytes()] * 3
    monkeypatch.setattr(watch, "capture_burst", lambda: burst)
    monkeypatch.setattr(watch, "presence_mode", lambda: "shadow")
    monkeypatch.setattr(classifier, "box_embedding", lambda jpeg, zones, trigger: vec(False, yard))
    monkeypatch.setattr(watch.identify, "identify", lambda *a, **k: Result(
        Sighting(visitor="none", species=None, count=0, confidence="high", notes="", circle=None, also=[]), "claude-haiku-5-5", False, False, {}))
    assert watch.handle_event() == "none"
    raw = json.loads(db.latest_visit(conn, real_only=False)["raw_json"])
    assert raw["calls"]["presence"]["empty"] and "skipped_claude" not in raw["calls"]
    r = db.presence_report(conn)
    assert (r["frames"], r["empty"], r["would_skip"], r["skipped"]) == (1, 1, 1, 0)


@pytest.fixture
def client(conn):
    with TestClient(webapp.app) as c:
        yield c


def test_looks_right_confirms_and_undo_takes_it_back(client, conn, add_visit):
    vid = add_visit()
    assert ">Confirm</button>" in client.get(f"/visit/{vid}").text
    client.post(f"/visit/{vid}/reviewed")
    assert db.get_visit(conn, vid)["confirmed_at"]
    page = client.get(f"/visit/{vid}").text
    assert "confirmed by human" in page and ">Confirm</button>" not in page
    client.post(f"/visit/{vid}/unconfirm")
    assert db.get_visit(conn, vid)["confirmed_at"] is None


def test_confirm_says_no_animal_only_on_empty_frames(client, add_visit):
    assert "No animal in image:" not in client.get(f"/visit/{add_visit()}").text
    empty = add_visit(visitor="none", species=None, count=0)
    assert "No animal in image:</span><button" in client.get(f"/visit/{empty}").text


def test_false_triggers_confirm_all_shown_skips_real_visits(client, conn, add_visit):
    empty = [add_visit(visitor="none", species=None) for _ in range(3)]
    real = add_visit()
    page = client.get("/false-triggers").text
    assert ">3</span>" in page and "Confirm all shown false triggers have no animals</button>" in page
    assert "not contain an animal" in page  # the pop-up before it confirms
    assert 'href="#ft-storage"' in page and 'id="ft-storage"' in page  # the intro links to the Delete button
    assert "Motion, but no animal found. Open each photo to check it." in page and 'id="ft-visit"' in page
    r = client.post("/false-triggers/confirm", data={"ids": ",".join(map(str, [*empty[:2], real]))},
                    follow_redirects=False)
    assert r.headers["location"] == "/false-triggers?confirmed=2"
    assert [bool(db.get_visit(conn, i)["confirmed_at"]) for i in [*empty, real]] == [True, True, False, False]
    assert db.count_unchecked_false_triggers(conn) == 1


def test_false_triggers_list_only_unconfirmed_unless_show_all(client, conn, add_visit):
    done, todo = add_visit(visitor="none", species=None), add_visit(visitor="none", species=None)
    db.confirm_visits(conn, [done])
    page = client.get("/false-triggers").text
    assert f'data-id="{todo}"' in page and f'data-id="{done}"' not in page
    assert "not yet confirmed" not in page and "Show confirmed (1)" in page
    page = client.get("/false-triggers?show=all").text
    assert f'data-id="{todo}"' in page and f'data-id="{done}"' in page and "Hide confirmed" in page
    assert "data-show-all" in page and "have no animals</button>" not in page  # bulk confirm only on the to-check list
    db.confirm_visits(conn, [todo])
    assert "Nothing left to check" in client.get("/false-triggers").text


def test_settings_switches_the_animal_check(client, conn):
    assert "Pi checks" in client.get("/settings").text
    client.post("/settings/presence", data={"mode": "gate"})
    assert presence.mode(conn) == "gate"
    assert client.post("/settings/presence", data={"mode": "nope"}).status_code == 400


# --- which frame teaches "animal" (visit 693: a corrected false trigger whose crop only held a tail) -----------

def left_out(conn) -> dict:
    return dict(presence.examples(conn)["left_out"])


def test_coverage_of_a_circle_in_the_crop_box():
    assert presence.coverage(INSIDE, CROP) == 1.0
    assert presence.coverage(ABOVE, CROP) == 0.0
    half = presence.coverage({"x": 0.25, "y": 0.5, "r": 0.05}, CROP)  # centred on the box's left edge
    assert 0.45 < half < 0.55
    # r is a fraction of the width: on a 16:9 frame the circle reaches 16/9 as far up and down, so one just above
    # the box still dips into it
    assert 0 < presence.coverage({"x": 0.5, "y": 0.23, "r": 0.02}, CROP) < 0.2


def test_a_corrected_false_trigger_teaches_animal_only_when_its_crop_shows_one(conn, yard):
    empty = [r[0] for r in conn.execute("SELECT id FROM visits WHERE visitor = 'none' LIMIT 3")]
    found_animal(conn, empty[0], circle=None)  # no circle, no burst: nothing shows it was in the crop box
    found_animal(conn, empty[1], circle=ABOVE)  # circled above the crop box
    found_animal(conn, empty[2], circle=INSIDE)  # circled inside it: a hard example worth learning
    out = left_out(conn)
    assert out == {empty[0]: "no frame shows the animal in the dashed box", empty[1]: "the circle is outside the dashed box"}
    assert presence.model(conn)["visitors"] == 13 and presence.model(conn)["left_out"] == 2


def test_claudes_own_visits_still_teach_animal(conn, yard):
    real = conn.execute("SELECT id FROM visits WHERE visitor = 'bird' LIMIT 2").fetchall()
    # Claude's circle inside the crop box, and an older visit with no circle at all: both count.
    db.set_raw_json(conn, real[0][0], {"zones": {"crop": list(CROP)}, "calls": {"circle": {**INSIDE, "claude": True}}})
    assert left_out(conn) == {}
    # The owner circling the animal outside the crop box takes it out (the crop Claude saw didn't show it).
    db.correct_visit(conn, real[1][0], None, None, None, [ABOVE])
    db.set_raw_json(conn, real[1][0], {"zones": {"crop": list(CROP)}})
    assert left_out(conn) == {real[1][0]: "the circle is outside the dashed box"}


def test_older_visits_find_their_crop_box_in_the_calibration_history(conn, yard):
    empty = conn.execute("SELECT id FROM visits WHERE visitor = 'none' LIMIT 1").fetchone()[0]
    db.correct_visit(conn, empty, "squirrel", None, None, [INSIDE])  # no zones saved with the visit
    # No crop box known, so the circle can't say anything, and nothing else shows the animal was in the crop box.
    assert left_out(conn) == {empty: "no frame shows the animal in the dashed box"}
    db.add_calibration(conn, ran_at="2000-01-01T00:00:00", reason="manual", status="moved", summary="",
                       after_json=json.dumps({"crop": list(CROP)}))
    assert left_out(conn) == {}
    db.correct_visit(conn, empty, "squirrel", None, None, [ABOVE])
    assert left_out(conn) == {empty: "the circle is outside the dashed box"}


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


def test_the_burst_frame_with_the_animal_becomes_its_training_example(conn, yard, add_visit):
    vid = burst_visit(conn, add_visit, [60, 60, 60, 60, 200, 60])  # the bird is in frame 5 only
    clf = FakeClf(yard)
    assert presence.pick_frames(conn, clf=clf) == 0  # still a false trigger: nothing to look through
    found_animal(conn, vid, circle=ABOVE)
    assert left_out(conn) == {vid: "the circle is outside the dashed box"}  # left out until its burst is looked at
    key = presence.label_key(conn)
    assert presence.pick_frames(conn, clf=clf) == 1 and clf.calls == 6
    row = conn.execute("SELECT animal_frame, animal_score FROM embeddings WHERE visit_id = ?", (vid,)).fetchone()
    assert row["animal_frame"] == 5 and row["animal_score"] >= presence.PICK_MIN
    assert left_out(conn) == {} and presence.label_key(conn) != key  # retrains with it
    assert presence.pending_picks(conn) == []  # looked through once
    db.correct_visit(conn, vid, None, None, None)  # back to empty: the sharpest frame's crop is the example again
    ex = presence.examples(conn)
    i = list(ex["ids"]).index(vid)
    assert not ex["y"][i] and ex["X"][i][0] < 0


def test_a_burst_with_no_animal_in_the_crop_box_is_left_out(conn, yard, add_visit):
    vid = burst_visit(conn, add_visit, [60] * 6)  # the dove stayed above the crop box the whole time
    found_animal(conn, vid, circle=None)
    assert left_out(conn) == {vid: "waiting for its burst to be looked through"}
    presence.pick_frames(conn, clf=FakeClf(yard))
    assert left_out(conn) == {vid: "no frame shows the animal in the dashed box"}


def test_looking_through_a_burst_waits_for_a_capture(conn, yard, add_visit):
    import threading

    found_animal(conn, burst_visit(conn, add_visit, [60, 200]), circle=None)
    busy = threading.Event()
    busy.set()
    clf = FakeClf(yard)
    assert presence.pick_frames(conn, busy=busy, clf=clf) == 0 and clf.calls == 0
    assert len(presence.pending_picks(conn)) == 1  # tried again on the next round


def on_frame(circle: dict, frame: int) -> dict:
    return {**circle, "frame": frame}


def pick_of(conn, vid) -> dict:
    return dict(conn.execute("SELECT animal_frame, animal_score, animal_by_owner, animal_vec IS NOT NULL AS has_vec "
                             "FROM embeddings WHERE visit_id = ?", (vid,)).fetchone())


def test_the_frame_the_owner_circled_teaches_animal(conn, yard, add_visit):
    # The bird is bright in frame 5, but the owner circled it on frame 2 (dark here, so the model scores it low):
    # their pick wins over the automatic one, whatever it scores.
    vid = burst_visit(conn, add_visit, [60, 60, 60, 60, 200, 60])
    found_animal(conn, vid, circle=on_frame(INSIDE, 2))
    assert left_out(conn) == {vid: "waiting for burst frame 2 (your circle) to be looked at"}
    assert [r["owner_frame"] for r in presence.pending_picks(conn)] == [2]
    key = presence.label_key(conn)
    clf = FakeClf(yard)
    assert presence.pick_frames(conn, clf=clf) == 1 and clf.calls == 1  # just that frame, not all six
    pick = pick_of(conn, vid)
    assert pick["animal_frame"] == 2 and pick["animal_by_owner"] == 1 and pick["has_vec"]
    assert pick["animal_score"] < presence.PICK_MIN
    assert left_out(conn) == {} and presence.label_key(conn) != key and presence.pending_picks(conn) == []
    ex = presence.examples(conn)
    i = list(ex["ids"]).index(vid)
    assert ex["y"][i] and ex["X"][i][0] < 0  # frame 2's (dark) crop, not frame 5's
    assert presence.teaches(conn, vid) == {"animal": True, "used": True, "why": "burst frame 2 (your circle)",
                                           "checked": True}

    # Moving the circle to frame 5 and saving again: picked again, and the labels change so it retrains.
    key = presence.label_key(conn)
    db.correct_visit(conn, vid, "squirrel", None, None, [on_frame(INSIDE, 5)])
    assert presence.label_key(conn) != key
    assert left_out(conn) == {vid: "waiting for burst frame 5 (your circle) to be looked at"}
    assert presence.pick_frames(conn, clf=clf) == 1 and clf.calls == 2
    assert pick_of(conn, vid)["animal_frame"] == 5 and left_out(conn) == {}
    i = list(presence.examples(conn)["ids"]).index(vid)
    assert presence.examples(conn)["X"][i][0] > 0

    # Only the frame changes (same spot, same text length): still a new label key.
    key = presence.label_key(conn)
    db.correct_visit(conn, vid, "squirrel", None, None, [on_frame(INSIDE, 4)])
    assert presence.label_key(conn) != key

    # Back on the photo Claude saw: the owner's pick is forgotten and the burst is looked through as before.
    db.correct_visit(conn, vid, "squirrel", None, None, [INSIDE])
    assert presence.pick_frames(conn, clf=clf) == 1 and pick_of(conn, vid)["animal_frame"] is None
    assert presence.pick_frames(conn, clf=clf) == 1
    assert pick_of(conn, vid)["animal_frame"] == 5 and not pick_of(conn, vid)["animal_by_owner"]
    assert presence.teaches(conn, vid)["why"] == "burst frame 5 (the one it scored most like an animal)"


def test_a_circle_outside_the_crop_box_on_its_frame_leaves_the_visit_out(conn, yard, add_visit):
    vid = burst_visit(conn, add_visit, [60, 200, 60])
    found_animal(conn, vid, circle=on_frame(ABOVE, 2))  # visit 693: only the tail was in the box
    presence.pick_frames(conn, clf=FakeClf(yard))
    assert pick_of(conn, vid)["animal_frame"] == 2
    assert left_out(conn) == {vid: "your circle on burst frame 2 is outside the dashed box"}
    # A circle on the photo doesn't count for frame 2: only the circles on that frame do.
    db.correct_visit(conn, vid, "squirrel", None, None, [on_frame(ABOVE, 2), INSIDE])
    assert left_out(conn) == {vid: "your circle on burst frame 2 is outside the dashed box"}
    db.correct_visit(conn, vid, "squirrel", None, None, [on_frame(ABOVE, 2), on_frame(INSIDE, 2)])
    assert left_out(conn) == {}


def test_an_owners_pick_needs_no_trained_model_and_a_gone_frame_is_left_out(conn, add_visit):
    vid = burst_visit(conn, add_visit, [60, 200])
    found_animal(conn, vid, circle=on_frame(INSIDE, 2))
    clf = FakeClf(np.random.default_rng(2))
    assert presence.model(conn) is None and presence.pick_frames(conn, clf=clf) == 1
    assert pick_of(conn, vid) == {"animal_frame": 2, "animal_score": None, "animal_by_owner": 1, "has_vec": 1}
    db.correct_visit(conn, vid, "squirrel", None, None, [on_frame(INSIDE, 7)])  # no frame 7 on disk
    presence.pick_frames(conn, clf=clf)
    assert left_out(conn) == {vid: "burst frame 7 (your circle) is no longer on disk"}
    assert presence.pending_picks(conn) == []  # not tried again until the circle moves


def test_the_visit_page_says_which_frame_teaches_the_check(conn, yard, add_visit):
    vid = burst_visit(conn, add_visit, [60, 200])
    found_animal(conn, vid, circle=on_frame(INSIDE, 2))
    presence.pick_frames(conn, clf=FakeClf(yard))
    with TestClient(webapp.app) as c:
        html = c.get(f"/visit/{vid}").text
    assert "burst frame 2 (your circle)" in html and "An animal example" in html
    assert 'class="save-btn"' in html and "your first circle is on another burst frame" in html


def test_one_outlier_visitor_cant_drag_the_threshold_to_zero():
    scores = np.array([0.0006, 0.2, 0.3, 0.5, 0.9, 0.95] + [0.97] * 10)
    birds = np.array(["bird/day"] * 16, object)
    threshold, low, ignored = presence.threshold_from(scores, birds)
    assert (ignored, low) == (0, 1) and threshold == pytest.approx(0.8 * 0.2)
    # A first-of-its-kind visitor (the first possum at night) is never set aside: it lowers the threshold.
    first = birds.copy()
    first[0] = "possum/night"
    assert presence.threshold_from(scores, first)[2] is None
    assert presence.threshold_from(scores, first)[0] == pytest.approx(0.8 * 0.0006)
    # Nor is one that is merely the lowest of a smooth spread.
    assert presence.threshold_from(np.array([0.05] + [0.2] * 15), birds)[2] is None
    assert presence.threshold_from(scores)[2] is None  # no groups: the plain minimum


def test_training_sets_the_outlier_aside_and_says_which_visit(conn, yard, add_visit):
    m = presence.model(conn)
    assert m["set_aside"] is None and m["set_by"] in [r[0] for r in conn.execute("SELECT id FROM visits")]
    odd = add_visit()  # Claude said bird, but its crop looks like every empty frame
    presence.store(conn, odd, vec(False, yard))
    m = presence.model(conn)
    assert m["set_aside"] == odd and m["threshold"] > 0.05


# --- the burst's last frame -------------------------------------------------------------------------------

def scored_frames(monkeypatch, scores: dict[bytes, float], threshold=0.3):
    """box_embedding returns the frame itself; presence.check scores it from the table."""
    looked = []
    monkeypatch.setattr(classifier, "box_embedding", lambda jpeg, zones, trigger: looked.append(jpeg) or np.array([scores[jpeg]]))
    monkeypatch.setattr(presence, "check", lambda conn, emb, view=None: {
        "model": presence.MODEL_NAME, "score": float(emb[0]), "threshold": threshold, "empty": bool(emb[0] < threshold)})
    return looked


def test_the_last_frame_counts_when_the_sharpest_looks_empty(monkeypatch, conn):
    frames = [b"ripples", b"b", b"bird in the bowl"]
    looked = scored_frames(monkeypatch, {b"ripples": 0.01, b"b": 0.02, b"bird in the bowl": 0.9})
    emb, check, ranked = watch.animal_check(frames, [b"ripples", b"b"], CROP)
    assert looked == [b"ripples", b"bird in the bowl"] and emb[0] == 0.9  # its embedding is stored for training
    assert (check["empty"], check["frame"], check["frames"]) == (False, "last", {"sharpest": 0.01, "last": 0.9})
    assert ranked == [b"bird in the bowl", b"ripples"]  # it becomes the photo Claude sees


def test_the_last_frame_isnt_looked_at_when_the_sharpest_already_shows_an_animal(monkeypatch, conn):
    looked = scored_frames(monkeypatch, {b"a": 0.9, b"z": 0.1})
    emb, check, ranked = watch.animal_check([b"a", b"z"], [b"a", b"z"], CROP)
    assert looked == [b"a"] and not check["empty"] and "frames" not in check and ranked == [b"a", b"z"]


def test_both_empty_keeps_the_sharpest_photo_and_the_higher_score(monkeypatch, conn):
    looked = scored_frames(monkeypatch, {b"a": 0.01, b"z": 0.2})
    emb, check, ranked = watch.animal_check([b"a", b"z"], [b"a", b"z"], CROP)
    assert len(looked) == 2 and check["empty"] and check["score"] == 0.2 and emb[0] == 0.2 and ranked[0] == b"a"


def test_the_last_frame_check_can_be_switched_off_and_fails_soft(monkeypatch, conn):
    looked = scored_frames(monkeypatch, {b"a": 0.01, b"z": 0.9})
    monkeypatch.setattr(watch.config, "PRESENCE_LAST_FRAME", False)
    assert watch.animal_check([b"a", b"z"], [b"a", b"z"], CROP)[1]["empty"] and looked == [b"a"]
    monkeypatch.setattr(watch.config, "PRESENCE_LAST_FRAME", True)

    def broken(jpeg, zones, trigger):
        if jpeg == b"z":
            raise RuntimeError("out of memory")
        return np.array([0.01])

    monkeypatch.setattr(classifier, "box_embedding", broken)
    emb, check, ranked = watch.animal_check([b"a", b"z"], [b"a", b"z"], CROP)
    assert emb[0] == 0.01 and check["empty"] and "frames" not in check


def test_gate_asks_claude_about_the_last_frame_when_only_it_shows_a_bird(monkeypatch, conn):
    from birdbrain.identify import Result, Sighting
    sharp = np.full((360, 640, 3), 90, np.uint8)
    cv2.putText(sharp, "RIPPLES", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 6)
    burst = [cv2.imencode(".jpg", sharp)[1].tobytes(),
             cv2.imencode(".jpg", np.full((360, 640, 3), 80, np.uint8))[1].tobytes(),
             cv2.imencode(".jpg", np.full((360, 640, 3), 70, np.uint8))[1].tobytes()]
    monkeypatch.setattr(watch, "capture_burst", lambda: burst)
    monkeypatch.setattr(watch, "presence_mode", lambda: "gate")
    scored_frames(monkeypatch, {burst[0]: 0.01, burst[1]: 0.01, burst[2]: 0.8})
    seen = []

    def identify(jpeg, *a, **k):
        seen.append(jpeg)
        return Result(Sighting(visitor="bird", species="Bushtit", count=1, confidence="high", notes="", circle=None, also=[]),
                      "claude-haiku-5-5", False, False, {})

    monkeypatch.setattr(watch.identify, "identify", identify)
    monkeypatch.setattr(watch.classifier, "gate_check", lambda emb, night, settled: (None, None))
    monkeypatch.setattr(watch.classifier, "run", lambda *a, **k: None)
    assert watch.handle_event() == "bird"
    assert seen == [burst[2]]
    raw = json.loads(db.latest_visit(conn)["raw_json"])
    assert raw["calls"]["presence"]["frame"] == "last"


def test_the_visit_page_shows_both_frames_scores(client, conn, add_visit):
    vid = add_visit(raw_json=json.dumps({"calls": {"presence": {
        "model": presence.MODEL_NAME, "score": 0.8, "threshold": 0.1, "empty": False, "trained_on": 700,
        "frames": {"sharpest": 0.01, "last": 0.8}, "frame": "last"}}}))
    page = client.get(f"/visit/{vid}").text
    assert "the burst's last frame 0.800" in page and "so the last frame became the photo" in page


def test_init_db_adds_the_burst_pick_columns_to_old_databases(tmp_path):
    import sqlite3

    old = sqlite3.connect(tmp_path / "old.db")
    old.row_factory = sqlite3.Row
    old.execute("CREATE TABLE embeddings (visit_id INTEGER PRIMARY KEY, vec BLOB NOT NULL)")
    db.init_db(old)
    cols = {r["name"] for r in old.execute("PRAGMA table_info(embeddings)")}
    assert {"animal_vec", "animal_frame", "animal_score", "animal_by_owner"} <= cols


def test_the_note_box_says_claude_wasnt_asked_on_a_skipped_visit(client, add_visit):
    skipped = add_visit(visitor="none", species=None, count=0, model=presence.MODEL_NAME, raw_json=json.dumps(
        {"calls": {"skipped_claude": True, "presence": {"score": 0.01, "threshold": 0.1, "empty": True, "trained_on": 700}}}))
    page = client.get(f"/visit/{skipped}").text
    assert "Claude didn't look at this one. Describe each numbered circle, e.g. 1: spiderweb moving in the wind" in page
    assert "Claude wasn't asked about this visit" in page
    asked = client.get(f"/visit/{add_visit(visitor='none', species=None, count=0)}").text
    assert 'placeholder="Describe each numbered circle, e.g. 1: spiderweb moving in the wind"' in asked
    assert "Claude wasn't asked about this visit" not in asked


# --- "What this photo trains" (web/steps.py): the four components, in this photo's real state --------------------

def trains_box(client, vid) -> str:
    html = client.get(f"/visit/{vid}").text
    start = html.index('id="trains"')
    return html[start:html.index("</section>", start)]


def pi_settled_empty(add_visit, conn, yard):
    vid = add_visit(visitor="none", species=None, count=0, model=presence.MODEL_NAME, raw_json=json.dumps(
        {"calls": {"skipped_claude": True, "presence": {"score": 0.01, "threshold": 0.1, "empty": True, "trained_on": 700}},
         "trigger": "rim"}))
    presence.store(conn, vid, vec(False, yard))
    return vid


def test_the_box_for_an_empty_photo_the_pi_settled_and_nobody_checked(client, conn, yard, add_visit):
    box = trains_box(client, pi_settled_empty(add_visit, conn, yard))
    for name in ("Motion detector", "Pi animal check", "BioCLIP 2 species check", "Claude"):
        assert name in box
    assert "An empty frame." in box and "Counts 3x once you Confirm" in box and "counts 3x on Confirm" in box
    assert "Not reached" in box and "not reached" in box
    assert "Not asked" in box and "no lesson yet" in box and "Confirm alone is not a lesson" in box
    assert "It spotted movement on the rim" in box


def test_the_box_after_confirming_says_counted_3x(client, conn, yard, add_visit):
    vid = pi_settled_empty(add_visit, conn, yard)
    db.confirm_visits(conn, [vid])
    box = trains_box(client, vid)
    assert "An empty frame, counted 3x" in box and ">counted 3x<" in box
    assert "no lesson yet" in box  # Confirm alone is not a lesson


def test_the_box_for_a_false_trigger_corrected_to_an_animal(client, conn, yard, add_visit):
    vid = pi_settled_empty(add_visit, conn, yard)
    found_animal(conn, vid)
    box = trains_box(client, vid)
    assert "An animal example" in box and "counted 3x: you checked it" in box
    assert "An empty frame" not in box
    assert "Not reached" in box and "Not asked" in box
    assert ">lesson<" in box and "A lesson on Claude" in box  # a correction is a lesson, sent word for word


def test_the_box_for_a_visit_claude_answered(client, conn, yard, add_visit):
    vid = add_visit(raw_json=json.dumps({"calls": {"haiku": {"frames": 1, "answer": {
        "visitor": "bird", "species": "Dark-eyed Junco", "confidence": "high"}}}}))
    presence.store(conn, vid, vec(True, yard))
    box = trains_box(client, vid)
    assert "Answered:" in box and "Dark-eyed Junco" in box and "An animal example" in box
    assert "No guess" in box  # no BioCLIP 2 species check step was recorded


def test_the_visit_box_is_for_admins_only(client, conn, yard, add_visit, monkeypatch):
    monkeypatch.setenv("ADMIN_PASSWORD", "s3cret!")
    assert 'id="trains"' not in client.get(f"/visit/{pi_settled_empty(add_visit, conn, yard)}").text


def test_the_box_for_a_pi_named_visit_the_owner_changed_or_confirmed(client, conn, yard, add_visit):
    """A visit the BioCLIP 2 species check named counts toward that species being settled only once the owner
    confirms it or names a species (db.settled_species); "No animal" or a bare type does not count."""
    raw = {"calls": {"skipped_claude": True, "settled_by": "classifier", "classifier": {
        "top": [{"name": "Dark-eyed Junco", "score": 0.97, "scientific": "Junco hyemalis"}], "local_share": 0.9}}}
    vid = add_visit(model="bioclip-2", species="Dark-eyed Junco", raw_json=json.dumps(raw))
    presence.store(conn, vid, vec(True, yard))
    box = trains_box(client, vid)
    assert "Named the animal on the Pi" in box and "Press Looks right" in box
    db.correct_visit(conn, vid, "none", None, None)  # the owner: nothing was there
    box = trains_box(client, vid)
    assert "counts toward Dark-eyed Junco being settled" not in box and "does not count" in box
    db.correct_visit(conn, vid, "squirrel", None, None)  # a type with no name counts toward nothing either
    assert "does not count" in trains_box(client, vid)
    db.correct_visit(conn, vid, None, "Bushtit", None)
    assert "counts toward Bushtit being settled" in trains_box(client, vid)
    db.correct_visit(conn, vid, None, None, None)
    db.confirm_visits(conn, [vid])
    assert "counts toward Dark-eyed Junco being settled" in trains_box(client, vid)
