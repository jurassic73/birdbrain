"""Component tests of birdbrain/presence.py: one real local dependency, a temp database or files on disk."""

import json

import cv2
import numpy as np
import pytest

from helpers.presence import ABOVE, CROP, DIM, INSIDE, FakeClf, burst_visit, found_animal, on_frame, vec, yard
from birdbrain import classifier, db, presence, watch


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


def left_out(conn) -> dict:
    return dict(presence.examples(conn)["left_out"])


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


def test_training_sets_the_outlier_aside_and_says_which_visit(conn, yard, add_visit):
    m = presence.model(conn)
    assert m["set_aside"] is None and m["set_by"] in [r[0] for r in conn.execute("SELECT id FROM visits")]
    odd = add_visit()  # Claude said bird, but its crop looks like every empty frame
    presence.store(conn, odd, vec(False, yard))
    m = presence.model(conn)
    assert m["set_aside"] == odd and m["threshold"] > 0.05


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


def test_init_db_adds_the_burst_pick_columns_to_old_databases(tmp_path):
    import sqlite3

    old = sqlite3.connect(tmp_path / "old.db")
    old.row_factory = sqlite3.Row
    old.execute("CREATE TABLE embeddings (visit_id INTEGER PRIMARY KEY, vec BLOB NOT NULL)")
    db.init_db(old)
    cols = {r["name"] for r in old.execute("PRAGMA table_info(embeddings)")}
    assert {"animal_vec", "animal_frame", "animal_score", "animal_by_owner"} <= cols
