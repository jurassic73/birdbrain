"""Component tests of the finder chain in the watcher (birdbrain/finder.py in watch.handle_event): the tight look
runs on whichever finder's circle, what is stored with the visit, and FINDERS=motion keeps today's JSON. A temp
database (gate_yard); the camera, BioCLIP 2, the motion circle, the crowd counter and YOLOX are fakes."""
import json

import pytest

from helpers.classifier import MOTION_CIRCLE, MOTION_FOUND, gate_yard, one_hot, sure  # noqa: F401  (a fixture)
from helpers.finder import BLOB_CIRCLE, BOX, MISS, YOLOX_BOX, YOLOX_CIRCLE
from birdbrain import classifier, config, db, detector, watch

TODAY_NAMED = {"animals_seen", "circle", "classifier", "classifier_gate", "motion_circle", "presence", "settled_by",
               "skipped_claude"}  # calls keys of a visit the tight look named, on the base this branch grew from
TODAY_GUESS = {"crop", "local_share", "model", "night", "seconds", "top"}  # keys of the tight look's stored guess
TODAY_NO_PATCH = {"animals_seen", "classifier", "classifier_gate", "motion_circle", "presence"}  # motion missed


@pytest.fixture
def finders(gate_yard, monkeypatch):
    """gate_yard plus the other two finders' collaborators: the crowd counter appends knobs["boxes"] to its detail
    list (the watcher's animal_boxes), YOLOX (detector.animal_box) answers knobs["yolox"] and logs each call."""
    knobs = {"boxes": [BOX], "yolox": (None, {"why": "nothing", "seconds": 0.2}), "yolox_calls": []}

    def count(scene, jpeg, crop, detail=None):
        if detail is not None:
            detail.extend(knobs["boxes"])
        return gate_yard["animals"]

    def animal_box(jpeg, crop, min_score=None):
        knobs["yolox_calls"].append((jpeg, crop, min_score))
        return knobs["yolox"]

    monkeypatch.setattr(watch, "count_animals", count)
    monkeypatch.setattr(detector, "animal_box", animal_box, raising=False)
    gate_yard.update(knobs)
    return gate_yard


def latest(conn):
    row = conn.execute("SELECT species, model, raw_json FROM visits ORDER BY id DESC").fetchone()
    return row, json.loads(row["raw_json"])["calls"]


def names(record):
    return [t["finder"] for t in record["tried"]]


def test_the_tight_look_runs_on_the_blob_circle_when_motion_finds_nothing(finders, conn):
    finders["circle"] = (None, MISS)
    assert watch.handle_event() == "bird"
    assert finders["tight_calls"] == [(BLOB_CIRCLE, config.CROP_ROI, False)]  # the look is around the blob's circle
    assert not finders["asked"]
    row, calls = latest(conn)
    assert row["model"] == classifier.MODEL_NAME and row["species"] == "Dark-eyed Junco"
    assert finders["yolox_calls"] == []  # found before yolox
    assert calls["circle"] == BLOB_CIRCLE and calls["motion_circle"] == MISS  # the motion attempt kept as before
    record = calls["finder"]
    assert record["finder"] == "blob" and record["motion"] == MISS and names(record) == ["motion", "blob"]
    assert record["tried"][0]["found"] is False and record["tried"][0]["why"] == "nothing"
    assert record["tried"][1]["found"] is True
    assert calls["classifier"]["finder"] == "blob" and calls["classifier"]["crop"] == classifier.TIGHT_CROP
    assert calls["settled_by"] == "classifier" and "classifier_tight" not in calls
    assert db.valid_json(json.dumps(calls))


def test_a_tight_miss_on_the_blob_circle_records_its_finder(finders, conn):
    finders["circle"] = (None, MISS)
    finders["tight"] = sure("Dark-eyed Junco", 0.9)  # under the tight look's bar: Claude is asked
    assert watch.handle_event() == "bird"
    assert finders["tight_calls"] == [(BLOB_CIRCLE, config.CROP_ROI, False)]
    assert finders["asked"] == [False]
    row, calls = latest(conn)
    assert row["species"] == "Bushtit"
    tight = calls["classifier_tight"]
    assert tight["finder"] == "blob" and tight["circle"] == BLOB_CIRCLE and tight["why"] == "score"
    assert calls["finder"]["finder"] == "blob" and names(calls["finder"]) == ["motion", "blob"]
    assert calls["motion_circle"] == MISS and "circle" not in calls and "settled_by" not in calls


def test_yolox_runs_once_and_only_when_motion_and_blob_both_fail(finders, conn):
    finders["circle"], finders["boxes"] = (None, MISS), []
    finders["yolox"] = (YOLOX_BOX, {"label": "bird", "score": 0.42, "seconds": 0.2})
    assert watch.handle_event() == "bird"
    assert len(finders["yolox_calls"]) == 1  # one inference per visit
    assert not finders["asked"]
    row, calls = latest(conn)
    assert row["model"] == classifier.MODEL_NAME
    jpeg, crop, min_score = finders["yolox_calls"][0]
    assert crop == config.CROP_ROI and min_score == config.FINDER_YOLOX_MIN_SCORE and isinstance(jpeg, bytes)
    assert finders["tight_calls"] == [(YOLOX_CIRCLE, config.CROP_ROI, False)]
    assert calls["circle"] == YOLOX_CIRCLE and calls["classifier"]["finder"] == "yolox"
    record = calls["finder"]
    assert record["finder"] == "yolox" and names(record) == ["motion", "blob", "yolox"]
    assert [t["found"] for t in record["tried"]] == [False, False, True] and record["tried"][1]["why"] == "no box"
    assert calls["motion_circle"] == MISS


def test_no_finder_finding_a_patch_means_claude_as_before(finders, conn):
    finders["circle"], finders["boxes"] = (None, MISS), []
    assert watch.handle_event() == "bird"
    assert len(finders["yolox_calls"]) == 1 and not finders["tight_calls"]
    assert finders["asked"] == [False]
    row, calls = latest(conn)
    assert row["species"] == "Bushtit"
    record = calls["finder"]
    assert record["finder"] is None and names(record) == ["motion", "blob", "yolox"]
    assert not any(t["found"] for t in record["tried"])
    assert calls["motion_circle"]["why"] == "nothing" and "classifier_tight" not in calls and "circle" not in calls


def test_motion_finding_the_patch_never_wakes_yolox(finders, conn):
    assert watch.handle_event() == "bird" and not finders["asked"]
    row, calls = latest(conn)
    assert finders["yolox_calls"] == [] and finders["tight_calls"] == [(MOTION_CIRCLE, config.CROP_ROI, False)]
    assert calls["finder"]["finder"] == "motion" and names(calls["finder"]) == ["motion"]
    assert calls["classifier"]["finder"] == "motion"
    assert calls["circle"] == MOTION_CIRCLE and calls["motion_circle"] == MOTION_FOUND


def test_finders_motion_only_stores_todays_json_plus_the_finder_key(finders, conn, monkeypatch):
    monkeypatch.setattr(config, "FINDERS", ("motion",), raising=False)
    assert watch.handle_event() == "bird" and not finders["asked"]
    row, calls = latest(conn)
    assert set(calls) == TODAY_NAMED | {"finder"}
    assert set(calls["classifier"]) == TODAY_GUESS | {"finder"} and calls["classifier"]["finder"] == "motion"
    assert calls["circle"] == MOTION_CIRCLE and calls["motion_circle"] == MOTION_FOUND  # as today
    assert calls["finder"]["finder"] == "motion" and calls["finder"]["motion"] == MOTION_FOUND
    assert names(calls["finder"]) == ["motion"] and calls["finder"]["tried"][0]["found"] is True
    # motion misses: a box and a detector are there, but only motion may run, so Claude is asked as today
    finders["circle"], finders["asked"][:], finders["tight_calls"][:] = (None, MISS), [], []
    assert watch.handle_event() == "bird" and finders["asked"] == [False]
    row, calls = latest(conn)
    assert set(calls) == TODAY_NO_PATCH | {"finder"}
    assert not finders["tight_calls"] and finders["yolox_calls"] == []
    assert calls["motion_circle"] == MISS and calls["finder"]["finder"] is None and names(calls["finder"]) == ["motion"]


def test_a_bowl_named_visit_gets_its_circle_from_the_chain(finders, conn, monkeypatch):
    monkeypatch.setattr(config, "CIRCLE_FROM_MOTION", True)
    finders["emb"], finders["circle"] = one_hot(0), (None, MISS)  # the first look is sure; motion finds no patch
    assert watch.handle_event() == "bird" and not finders["asked"]
    row, calls = latest(conn)
    assert row["model"] == classifier.MODEL_NAME and not finders["tight_calls"]
    assert calls["classifier"]["crop"] == "frame"  # named on the bowl look, not the tight one
    assert calls["circle"] == BLOB_CIRCLE and calls["motion_circle"] == MISS
    assert calls["finder"]["finder"] == "blob" and names(calls["finder"]) == ["motion", "blob"]
    assert finders["yolox_calls"] == []


def test_a_bowl_named_visit_at_night_never_calls_yolox(finders, conn, monkeypatch):
    monkeypatch.setattr(config, "CIRCLE_FROM_MOTION", True)
    monkeypatch.setattr(watch, "is_night", lambda jpeg: True)
    monkeypatch.setattr(classifier, "gate_check", lambda emb, night, settled: (
        {"model": classifier.MODEL_NAME, "crop": "box", "top": [{"name": "Dark-eyed Junco", "score": 0.97}]},
        ("Dark-eyed Junco", "bird")))
    finders["circle"], finders["boxes"] = (None, MISS), []
    finders["yolox"] = (YOLOX_BOX, {"label": "cat", "score": 0.5, "seconds": 0.2})  # it would find one
    assert watch.handle_event() == "bird" and not finders["asked"]
    row, calls = latest(conn)
    assert finders["yolox_calls"] == [] and "circle" not in calls
    record = calls["finder"]
    assert record["finder"] is None
    yolox = next(t for t in record["tried"] if t["finder"] == "yolox")
    assert yolox["found"] is False and yolox["why"] == "night"
    assert calls["motion_circle"] == MISS
