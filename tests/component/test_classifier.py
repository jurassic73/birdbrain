"""Component tests of birdbrain/classifier.py: one real local dependency, a temp database or files on disk."""

import json
from datetime import datetime

import cv2
import numpy as np
import pytest

from helpers.classifier import (LOCAL, MOTION_CIRCLE, MOTION_FOUND, FakeClassifier, FakeTextClassifier, gate_yard,
                                guess, one_hot, raw, sure)
from helpers.clock import FROZEN_NOW
from helpers.images import flat_jpeg
from birdbrain import classifier, config, db, images, watch
from birdbrain.classifier import fetch_place_species as real_fetch_place_species


def test_place_species_are_fetched_page_by_page_and_cached(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(classifier.time, "sleep", lambda s: None)
    pages = []

    def get(params):
        pages.append(params["page"])
        assert params["place_id"] == "502" and params["iconic_taxa"] == "Aves"
        taxa = [{"count": 9, "taxon": {"name": "Junco hyemalis", "preferred_common_name": "Dark-eyed Junco"}}]
        if params["page"] == 2:
            taxa = [{"count": 1, "taxon": {"name": "Junco hyemalis oreganus", "preferred_common_name": "Oregon Junco"}},
                    {"count": 1, "taxon": {"name": "Psaltriparus minimus", "preferred_common_name": "Bushtit"}}]
        return {"total_results": 3, "per_page": 2, "results": taxa}

    species = real_fetch_place_species("502", get=get)
    assert pages == [1, 2] and [s["name"] for s in species] == ["Junco hyemalis", "Junco hyemalis", "Psaltriparus minimus"]

    monkeypatch.setattr(classifier, "fetch_place_species", lambda place, taxon="Aves": species)
    local = classifier.place_species("502")
    assert local["Psaltriparus minimus"] == "Bushtit" and (tmp_path / "inat_birds_502.json").exists()
    # Offline later: the cached list still works, stale or not.
    monkeypatch.setattr(classifier, "PLACE_MAX_AGE", classifier.timedelta(seconds=-1))
    monkeypatch.setattr(classifier, "fetch_place_species", lambda place, taxon="Aves": 1 / 0)
    assert classifier.place_species("502") == local
    assert classifier.place_species("") is None
    assert classifier.place_species("999") is None  # never fetched and offline: no limit


# The watcher tests' burst: substream-sized and flat, since nothing in them looks at the pixels or the frame's size.
BURST = [flat_jpeg(360, 640, (40, 120, 200))] * 3


def test_watcher_stores_the_guess_without_changing_claudes_label(monkeypatch, conn):
    burst = BURST
    monkeypatch.setattr(watch, "capture_burst", lambda: burst)
    monkeypatch.setattr(watch, "presence_mode", lambda: "off")

    def fake_identify(*args, **kwargs):
        from birdbrain.identify import Result, Sighting
        return Result(Sighting(visitor="bird", species="Bushtit", count=1, confidence="high", notes="", circle=None, also=[]),
                      "claude-haiku-5-5", False, False, {"circle": {"x": 0.5, "y": 0.5, "r": 0.03, "claude": True}})

    monkeypatch.setattr(watch.identify, "identify", fake_identify)
    fake = FakeClassifier()
    monkeypatch.setattr(classifier, "load", lambda: fake)
    monkeypatch.setattr(classifier, "local_species", lambda: LOCAL)
    assert watch.handle_event() == "bird"
    row = conn.execute("SELECT species, raw_json FROM visits").fetchone()
    guess = json.loads(row["raw_json"])["calls"]["classifier"]
    assert row["species"] == "Bushtit"  # Claude's label stands
    assert guess["crop"] == "circle" and guess["top"][0]["name"] == "Dark-eyed Junco" and len(guess["top"]) == 3
    assert guess["model"] == classifier.MODEL_NAME and "seconds" in guess and guess["night"] is False

    fake.fail = True  # fail soft: the visit is stored without a guess
    assert watch.handle_event() == "bird"
    raws = [json.loads(r[0]) for r in conn.execute("SELECT raw_json FROM visits ORDER BY id")]
    assert "classifier" not in raws[1]["calls"]


def test_a_failed_download_is_not_retried_on_every_visit(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    tries = []
    monkeypatch.setattr(classifier, "download_model", lambda path: tries.append(path) or 1 / 0)
    for _ in range(3):
        assert classifier.run(cv2.imencode(".jpg", np.zeros((8, 8, 3), np.uint8))[1].tobytes(), None, None, False) is None
    assert len(tries) == 1


def test_report_scores_against_the_final_label(conn, add_visit):
    add_visit(species="Dark-eyed Junco", raw_json=raw("Dark-eyed Junco", "Bushtit"))            # top 1
    add_visit(species="Bushtit", raw_json=raw("Dark-eyed Junco", "Bushtit"))                     # top 3
    v = add_visit(species="Bushtit", raw_json=raw("House Finch", "Dark-eyed Junco"))
    db.correct_visit(conn, v, "bird", "house finch", None)  # the owner's label wins (key ignores case)
    add_visit(species="Bushtit", is_night=1, raw_json=raw("Song Sparrow", night=True))           # miss
    add_visit(species=None, raw_json=raw("Bushtit"))                                             # "Bird": no species
    add_visit(visitor="squirrel", species="Eastern Gray Squirrel", raw_json=raw("Eastern Gray Squirrel"))  # mammals too
    add_visit(visitor="raccoon", species=None, raw_json=raw("Common Raccoon"))                   # "Raccoon" matches
    add_visit(visitor="squirrel", species=None, raw_json=raw("Douglas's Squirrel"))              # "Squirrel": no species
    add_visit(species="Bushtit")                                                                 # never scored
    r = db.classifier_report(conn)
    assert (r["checked"], r["top1"], r["top3"], r["night"], r["night_top1"]) == (6, 4, 5, 1, 0)
    assert r["seconds"] == 0.05
    bushtit = next(s for s in r["species"] if s["label"] == "Bushtit")
    assert (bushtit["visits"], bushtit["top1"], bushtit["top3"]) == (2, 0, 1)
    assert [m["guess"] for m in r["misses"]] == ["Song Sparrow", "Dark-eyed Junco"]


def test_backfill_scores_stored_visits(conn, add_visit):
    def stored(**fields):
        image_rel, thumb_rel = images.store_frame(flat_jpeg(1080, 1920), FROZEN_NOW)  # main-stream size: the circle's
        return add_visit(image_path=image_rel, thumb_path=thumb_rel, **fields)      # square below is in its pixels

    old = stored(raw_json=json.dumps({"calls": {"circle": {"x": 0.5, "y": 0.5, "r": 0.02}}}))
    owner = stored()
    db.correct_visit(conn, owner, "bird", "Bushtit", None, marks=[{"x": 0.2, "y": 0.3, "r": 0.04}])
    stored(visitor="none", species=None)  # false triggers aren't scored
    done = stored(raw_json=raw("Bushtit"))
    fake = FakeClassifier()
    assert classifier.backfill(limit=1, clf=fake, local=LOCAL) == 1  # newest unscored first
    assert classifier.backfill(clf=fake, local=LOCAL) == 1
    assert classifier.backfill(clf=fake, local=LOCAL) == 0
    calls = {r["id"]: json.loads(r["raw_json"] or "{}").get("calls", {}) for r in conn.execute("SELECT id, raw_json FROM visits")}
    assert calls[old]["classifier"]["crop"] == "circle" and calls[old]["classifier"]["backfilled"]
    assert calls[owner]["classifier"]["crop"] == "circle"
    assert "backfilled" not in calls[done]["classifier"]  # left alone without --redo
    assert fake.shapes[0][:2] == (round(2 * 0.04 * 1920 * classifier.CIRCLE_MARGIN),) * 2  # the owner's circle
    assert classifier.backfill(redo=True, clf=fake, local=LOCAL) == 3


def test_backfill_keeps_an_owner_check_stored_while_it_ran(conn, add_visit, monkeypatch):
    image_rel, thumb_rel = images.store_frame(flat_jpeg(1080, 1920), datetime.now())
    vid = add_visit(image_path=image_rel, thumb_path=thumb_rel, raw_json=json.dumps({"trigger": "rim"}))

    def slow_classify(*args, **kwargs):  # the owner asks the Pi about the visit while the backfill is on it
        with db.session() as other:
            db.set_raw_entry(other, vid, "$.calls.owner_checks", [{"source": "pi", "verdict": "sure"}])
            db.end_stay(other, vid, "2026-11-01T09:05:00", "left")
        return {"crop": "box", "top": []}

    monkeypatch.setattr(classifier, "classify", slow_classify)
    assert classifier.backfill(clf=object(), local=LOCAL) == 1
    raw = json.loads(db.get_visit(conn, vid)["raw_json"])
    assert raw["calls"]["classifier"]["backfilled"] and raw["calls"]["classifier"]["crop"] == "box"
    assert raw["calls"]["owner_checks"][0]["verdict"] == "sure" and raw["stay_end"] == "left" and raw["trigger"] == "rim"


def test_probs_are_a_softmax_over_the_species_embeddings(monkeypatch, tmp_path):
    names = [["Junco hyemalis", "Dark-eyed Junco"], ["Haemorhous mexicanus", "House Finch"]]
    np.save(tmp_path / "bioclip2_ebird_text.npy", np.array([[1, 0], [0, 1]], np.float16))
    (tmp_path / "bioclip2_ebird_names.json").write_text(json.dumps(names))
    np.save(tmp_path / "bioclip2_mammals_text.npy", np.array([[-1, 0]], np.float16))
    (tmp_path / "bioclip2_mammals_names.json").write_text(json.dumps([["Sciurus carolinensis", "Eastern Gray Squirrel"]]))

    class Session:
        def __init__(self, *a):
            pass

        def run(self, out, feed):
            assert feed["image"].shape == (1, 3, 224, 224)
            return [np.array([[0.6, 0.8]], np.float32)]

    import onnxruntime
    monkeypatch.setattr(onnxruntime, "InferenceSession", Session)
    clf = classifier.Classifier(tmp_path)
    p = clf.probs(np.zeros((100, 100, 3), np.uint8))
    assert p.shape == (3,) and p.sum() == pytest.approx(1) and p[1] > 0.99  # 100 x (0.8 - 0.6) apart
    assert clf.labels[2] == ("Sciurus carolinensis", "Eastern Gray Squirrel", True)
    r = classifier.rank(p, clf.labels, {"Junco hyemalis": "Dark-eyed Junco", "Haemorhous mexicanus": "House Finch"})
    assert r["top"][0]["name"] == "House Finch" and r["background"] == 0


def test_settled_species_ignores_visits_the_gate_named_itself(conn, add_visit):
    for _ in range(3):
        add_visit(species="Dark-eyed Junco")
        add_visit(visitor="squirrel", species="Eastern Gray Squirrel")
        add_visit(species="House Finch", model="bioclip-2")
    add_visit(species="Bushtit", confidence="medium")
    assert db.settled_species(conn) == {"Dark-eyed Junco": "bird", "Eastern Gray Squirrel": "squirrel"}


def test_watcher_skips_claude_when_the_gate_is_sure(monkeypatch, conn, add_visit):
    for _ in range(3):
        add_visit(species="Dark-eyed Junco")
    burst = BURST
    monkeypatch.setattr(watch, "capture_burst", lambda: burst)
    monkeypatch.setattr(watch, "presence_mode", lambda: "gate")
    monkeypatch.setattr(config, "CLASSIFIER", "gate")
    emb = {"v": one_hot(0)}
    monkeypatch.setattr(watch.classifier, "box_embedding", lambda jpeg, zones, trigger: emb["v"])
    monkeypatch.setattr(watch.presence, "check", lambda conn, e, view=None: {"model": "bioclip-presence", "score": 0.9,
                                                                  "threshold": 0.4, "empty": False})
    monkeypatch.setattr(watch.presence, "store", lambda conn, vid, e: None)
    monkeypatch.setattr(classifier, "load", lambda: FakeTextClassifier())
    monkeypatch.setattr(classifier, "local_species", lambda: LOCAL)
    asked = []

    def fake_identify(*args, **kwargs):
        from birdbrain.identify import Result, Sighting
        asked.append(1)
        return Result(Sighting(visitor="bird", species="Bushtit", count=2, confidence="high", notes="", circle=None, also=[]),
                      "claude-haiku-5-5", False, False, {})

    monkeypatch.setattr(watch.identify, "identify", fake_identify)
    monkeypatch.setattr(classifier, "run", lambda *a: None)
    assert watch.handle_event() == "bird"
    row = conn.execute("SELECT species, model, confidence, raw_json FROM visits ORDER BY id DESC").fetchone()
    calls = json.loads(row["raw_json"])["calls"]
    assert not asked and row["species"] == "Dark-eyed Junco" and row["model"] == "bioclip-2"
    assert calls["skipped_claude"] and calls["settled_by"] == "classifier" and calls["presence"]["score"] == 0.9

    emb["v"] = one_hot(0, 0.01)  # unsure: Claude names it
    assert watch.handle_event() == "bird" and asked
    row = conn.execute("SELECT species, raw_json FROM visits ORDER BY id DESC").fetchone()
    assert row["species"] == "Bushtit"
    first = json.loads(row["raw_json"])["calls"]["classifier_gate"]  # the look that didn't name it is kept
    assert first["why"] == "score" and first["top"][0]["name"] == "Dark-eyed Junco" and first["crop"] != "circle"

    # Sure, but two animals in view: one guess names one animal, so Claude counts them.
    emb["v"], asked[:] = one_hot(0), []
    monkeypatch.setattr(watch, "count_animals", lambda scene, jpeg, crop, detail=None: 2)
    assert watch.handle_event() == "bird" and asked
    row = conn.execute("SELECT species, raw_json FROM visits ORDER BY id DESC").fetchone()
    calls = json.loads(row["raw_json"])["calls"]
    assert row["species"] == "Bushtit" and calls["animals_seen"] == 2
    assert calls["gate_crowd"] == {"species": "Dark-eyed Junco", "animals": 2}
    assert calls["classifier_gate"]["why"] == "crowd"

    monkeypatch.setattr(watch, "count_animals", lambda scene, jpeg, crop, detail=None: 1)  # one: the Pi names it
    asked[:] = []
    assert watch.handle_event() == "bird" and not asked
    calls = json.loads(conn.execute("SELECT raw_json FROM visits ORDER BY id DESC").fetchone()[0])["calls"]
    assert calls["settled_by"] == "classifier" and calls["animals_seen"] == 1 and "gate_crowd" not in calls
    assert "classifier_gate" not in calls  # the guess that named it is calls.classifier


def test_missing_model_without_release_url_fails_soft(monkeypatch, tmp_path):
    monkeypatch.undo()  # drop the suite's no-network stub to reach the real download_model
    assert not config.CLASSIFIER_RELEASE_URL
    with pytest.raises(RuntimeError, match="CLASSIFIER_RELEASE_URL"):
        classifier.download_model(tmp_path)


def latest_calls(conn):
    row = conn.execute("SELECT species, model, raw_json FROM visits ORDER BY id DESC").fetchone()
    return row, json.loads(row["raw_json"])["calls"]


def test_watcher_names_the_visit_on_the_tight_look(gate_yard, conn):
    # The first look (one_hot(0, 0.01): a junco at about 65%) is under the bowl's bar; the tight look is sure.
    assert watch.handle_event() == "bird"
    row, calls = latest_calls(conn)
    assert not gate_yard["asked"] and row["species"] == "Dark-eyed Junco" and row["model"] == "bioclip-2"
    assert gate_yard["circle_calls"] == [("rim", False)]
    assert gate_yard["tight_calls"] == [(MOTION_CIRCLE, config.CROP_ROI, False)]
    assert calls["settled_by"] == "classifier" and calls["skipped_claude"]
    assert calls["classifier"]["crop"] == "motion" and calls["classifier"]["top"][0]["score"] == 0.97
    assert calls["circle"] == MOTION_CIRCLE and calls["motion_circle"] == MOTION_FOUND
    first = calls["classifier_gate"]  # the bowl look is kept, with why, so the page shows both
    assert first["why"] == "score" and first["top"][0]["name"] == "Dark-eyed Junco" and 0.6 < first["top"][0]["score"] < 0.8
    assert first["crop"] == "frame"  # the test yard has no ring or crop box
    assert "classifier_tight" not in calls and calls["animals_seen"] == 1
    assert db.valid_json(json.dumps(calls))


def test_tight_look_that_is_not_sure_stores_itself_and_asks_claude(gate_yard, conn):
    gate_yard["tight"] = sure("Dark-eyed Junco", 0.9)  # over the bowl's 80% but under the tight look's 95%
    assert watch.handle_event() == "bird" and gate_yard["asked"] == [False]
    row, calls = latest_calls(conn)
    assert row["species"] == "Bushtit" and row["model"] == "claude-haiku-5-5"
    assert calls["classifier_gate"]["why"] == "score"
    tight = calls["classifier_tight"]
    assert tight["why"] == "score" and tight["crop"] == "motion" and tight["top"][0]["score"] == 0.9
    assert tight["circle"] == MOTION_CIRCLE and tight["seconds"] == 5.9
    assert "settled_by" not in calls and "circle" not in calls and "motion_circle" not in calls
    # unsettled on the tight look: the same, with its reason
    gate_yard["tight"], gate_yard["asked"][:] = sure("House Finch", 0.99), []
    assert watch.handle_event() == "bird" and gate_yard["asked"]
    assert latest_calls(conn)[1]["classifier_tight"]["why"] == "unsettled"


def test_no_patch_of_motion_means_the_old_behaviour(gate_yard, conn):
    gate_yard["circle"] = (None, {"found": False, "why": "nothing", "source": "empty"})
    assert watch.handle_event() == "bird" and gate_yard["asked"] == [False]
    row, calls = latest_calls(conn)
    assert row["species"] == "Bushtit" and not gate_yard["tight_calls"]
    assert calls["classifier_gate"]["why"] == "score"
    assert "classifier_tight" not in calls and "circle" not in calls
    assert calls["motion_circle"]["why"] == "nothing"  # kept so a miss like visit 1024's can be explained


def test_no_tight_look_at_night(gate_yard, conn, monkeypatch):
    monkeypatch.setattr(watch, "is_night", lambda jpeg: True)
    assert watch.handle_event() == "bird" and gate_yard["asked"] == [True]
    row, calls = latest_calls(conn)
    assert not gate_yard["circle_calls"] and not gate_yard["tight_calls"]
    assert "classifier_gate" not in calls and "classifier_tight" not in calls


def test_no_tight_look_for_a_crowd(gate_yard, conn):
    gate_yard["animals"] = 2
    assert watch.handle_event() == "bird" and gate_yard["asked"]  # the first look unsure, two animals: Claude
    row, calls = latest_calls(conn)
    assert not gate_yard["circle_calls"] and not gate_yard["tight_calls"] and "classifier_tight" not in calls
    assert calls["classifier_gate"]["why"] == "score" and calls["animals_seen"] == 2
    gate_yard["emb"], gate_yard["asked"][:] = one_hot(0), []  # the first look sure of one of a crowd: Claude counts
    assert watch.handle_event() == "bird" and gate_yard["asked"]
    row, calls = latest_calls(conn)
    assert not gate_yard["circle_calls"] and calls["classifier_gate"]["why"] == "crowd"
    assert calls["gate_crowd"] == {"species": "Dark-eyed Junco", "animals": 2}


def test_a_sure_first_look_still_names_the_visit_without_a_tight_look(gate_yard, conn, monkeypatch):
    monkeypatch.setattr(config, "CIRCLE_FROM_MOTION", True)
    gate_yard["emb"] = one_hot(0)
    assert watch.handle_event() == "bird" and not gate_yard["asked"]
    row, calls = latest_calls(conn)
    assert row["model"] == "bioclip-2" and not gate_yard["tight_calls"]
    assert calls["classifier"]["crop"] == "frame" and "classifier_gate" not in calls
    assert gate_yard["circle_calls"] == [("rim", False)] and calls["circle"] == MOTION_CIRCLE  # circled, as before


NO_PATCH = (None, {"found": False, "why": "nothing", "source": "empty", "seconds": 0.02})
BLOB_BOX = [0.4, 0.5, 0.1, 0.2]


def test_the_crowd_counters_box_gives_the_tight_look_its_patch_when_motion_misses(gate_yard, conn):
    from birdbrain import finder
    gate_yard["circle"], gate_yard["boxes"] = NO_PATCH, [BLOB_BOX]
    assert watch.handle_event() == "bird" and not gate_yard["asked"]
    row, calls = latest_calls(conn)
    assert row["model"] == "bioclip-2" and row["species"] == "Dark-eyed Junco"
    blob = finder.box_circle(BLOB_BOX)
    assert gate_yard["tight_calls"] == [(blob, config.CROP_ROI, False)] and not gate_yard["yolox_calls"]
    assert calls["circle"] == blob and calls["classifier"]["crop"] == "motion" and calls["classifier"]["finder"] == "blob"
    assert calls["motion_circle"] == NO_PATCH[1]  # the Motion detector's attempt, kept in the old shape
    assert calls["finder"]["finder"] == "blob" and calls["finder"]["motion"] == NO_PATCH[1]
    assert [(t["finder"], t["found"], t["why"]) for t in calls["finder"]["tried"]] == [("motion", False, "nothing"), ("blob", True, None)]
    assert db.valid_json(json.dumps(calls))
    # The same, when the tight look is not sure: the miss records the finder too, and Claude is asked.
    gate_yard["tight"] = sure("Dark-eyed Junco", 0.9)
    assert watch.handle_event() == "bird" and gate_yard["asked"] == [False]
    row, calls = latest_calls(conn)
    assert calls["classifier_tight"]["finder"] == "blob" and calls["classifier_tight"]["circle"] == blob
    assert calls["finder"]["finder"] == "blob" and "circle" not in calls and calls["motion_circle"]["why"] == "nothing"


def test_the_object_detector_is_the_last_finder_and_runs_once(gate_yard, conn):
    from birdbrain import finder
    gate_yard["circle"] = NO_PATCH
    gate_yard["yolox"] = ([0.6, 0.3, 0.05, 0.08], {"label": "bird", "score": 0.4, "seconds": 0.3})
    assert watch.handle_event() == "bird" and not gate_yard["asked"]
    row, calls = latest_calls(conn)
    assert gate_yard["yolox_calls"] == [(config.CROP_ROI, config.FINDER_YOLOX_MIN_SCORE)]
    yolo = finder.box_circle([0.6, 0.3, 0.05, 0.08])
    assert gate_yard["tight_calls"] == [(yolo, config.CROP_ROI, False)] and calls["circle"] == yolo
    assert calls["classifier"]["finder"] == "yolox" and calls["finder"]["finder"] == "yolox"
    tried = calls["finder"]["tried"]
    assert [(t["finder"], t["why"]) for t in tried] == [("motion", "nothing"), ("blob", "no box"), ("yolox", None)]
    assert tried[2]["label"] == "bird" and tried[2]["score"] == 0.4
    # Two boxes: the crowd counter can't say which, and the detector finds nothing: Claude, with the tries recorded.
    gate_yard["boxes"], gate_yard["yolox"], gate_yard["yolox_calls"][:] = [BLOB_BOX, [0.1, 0.1, 0.05, 0.05]], (None, {"why": "nothing", "seconds": 0.3}), []
    assert watch.handle_event() == "bird" and gate_yard["asked"] == [False]
    row, calls = latest_calls(conn)
    assert len(gate_yard["yolox_calls"]) == 1 and len(gate_yard["tight_calls"]) == 1  # no second tight look
    assert calls["finder"]["finder"] is None and [t["why"] for t in calls["finder"]["tried"]] == ["nothing", "several", "nothing"]
    assert "classifier_tight" not in calls and "circle" not in calls and calls["motion_circle"]["why"] == "nothing"


def test_a_count_the_crowd_counter_could_not_make_is_no_box_for_the_blob_finder(gate_yard, conn):
    gate_yard["circle"], gate_yard["animals"] = NO_PATCH, None  # the light changed: count_animals gave up
    assert watch.handle_event() == "bird" and gate_yard["asked"] == [False]
    calls = latest_calls(conn)[1]
    assert [(t["finder"], t["why"]) for t in calls["finder"]["tried"]] == [("motion", "nothing"), ("blob", "no count"), ("yolox", "nothing")]


def test_finders_motion_alone_stores_what_it_did_before_plus_the_finder_record(gate_yard, conn, monkeypatch):
    monkeypatch.setattr(config, "FINDERS", ("motion",))
    gate_yard["circle"], gate_yard["boxes"] = NO_PATCH, [BLOB_BOX]  # a box the blob finder would have used
    assert watch.handle_event() == "bird" and gate_yard["asked"] == [False]
    row, calls = latest_calls(conn)
    assert not gate_yard["tight_calls"] and not gate_yard["yolox_calls"] and "circle" not in calls
    assert calls["motion_circle"] == NO_PATCH[1] and calls["classifier_gate"]["why"] == "score"
    assert calls["finder"] == {"finder": None, "motion": NO_PATCH[1],
                               "tried": [{"finder": "motion", "found": False, "why": "nothing", "seconds": calls["finder"]["tried"][0]["seconds"]}]}
    # And when the Motion detector finds the patch: the same visit as before, with the finder named.
    gate_yard["circle"], gate_yard["asked"][:] = (MOTION_CIRCLE, MOTION_FOUND), []
    assert watch.handle_event() == "bird" and not gate_yard["asked"]
    row, calls = latest_calls(conn)
    assert calls["circle"] == MOTION_CIRCLE and calls["motion_circle"] == MOTION_FOUND
    assert calls["classifier"]["finder"] == "motion" and calls["finder"]["finder"] == "motion"
    assert set(calls) == {"classifier", "skipped_claude", "settled_by", "presence", "circle", "motion_circle",
                          "classifier_gate", "animals_seen", "finder"}


def test_a_bowl_named_visit_gets_its_circle_from_the_finder_chain(gate_yard, conn, monkeypatch):
    from birdbrain import finder
    monkeypatch.setattr(config, "CIRCLE_FROM_MOTION", True)
    gate_yard["emb"], gate_yard["circle"], gate_yard["boxes"] = one_hot(0), NO_PATCH, [BLOB_BOX]
    assert watch.handle_event() == "bird" and not gate_yard["asked"]
    row, calls = latest_calls(conn)
    assert row["model"] == "bioclip-2" and not gate_yard["tight_calls"] and calls["classifier"]["crop"] == "frame"
    assert calls["circle"] == finder.box_circle(BLOB_BOX) and calls["motion_circle"] == NO_PATCH[1]
    assert calls["finder"]["finder"] == "blob" and "finder" not in calls["classifier"]  # no tight look was made


def test_tight_look_failing_asks_claude(gate_yard, conn, monkeypatch):
    def broken(*a, **k):
        raise RuntimeError("model broke")
    monkeypatch.setattr(classifier, "classify", broken)
    assert watch.handle_event() == "bird" and gate_yard["asked"] == [False]
    calls = latest_calls(conn)[1]
    assert "classifier_tight" not in calls and calls["classifier_gate"]["why"] == "score"


def test_backfill_gate_writes_first_looks_from_stored_embeddings(conn, add_visit, monkeypatch):
    from birdbrain import presence

    monkeypatch.setattr(classifier, "local_species", lambda: LOCAL)
    asked = {"haiku": {"answer": {"visitor": "bird", "species": "Bushtit", "confidence": "high"}}}

    def visit(emb, calls=None, view=classifier.PRESENCE_VIEW, **fields):
        vid = add_visit(raw_json=json.dumps({"calls": calls or asked}), **fields)
        presence.store(conn, vid, emb, view)
        return vid

    unsure = visit(one_hot(0, 0.01))
    sure = visit(one_hot(0))
    old_view = visit(one_hot(0, 0.01), view=None)
    named = visit(one_hot(0), {"settled_by": "classifier", "skipped_claude": True})
    night = visit(one_hot(0, 0.01), is_night=1)
    done = visit(one_hot(0, 0.01), {**asked, "classifier_gate": {"why": "score", "top": []}})
    no_embedding = add_visit(raw_json=json.dumps({"calls": asked}))
    settled = {"Dark-eyed Junco": "bird"}
    clf = FakeTextClassifier()

    def gate(vid):
        raw = json.loads(conn.execute("SELECT raw_json FROM visits WHERE id = ?", (vid,)).fetchone()[0])
        return raw["calls"].get("classifier_gate")

    lines = []
    c = classifier.backfill_gate(dry_run=True, clf=clf, local=LOCAL, settled=settled, say=lines.append)
    assert c["done"] == 3 and gate(unsure) is None and lines[-1].startswith("5/5")  # nothing written
    c = classifier.backfill_gate(clf=clf, local=LOCAL, settled=settled, say=lambda m: None)
    assert (c["done"], c["named_now"], c["old_view"]) == (3, 1, 1)
    assert gate(unsure)["why"] == "score" and gate(unsure)["backfilled"] and gate(unsure)["crop"] == "frame"  # the test yard has no ring or crop
    assert gate(sure)["why"] == "changed"  # it would name this one today, but Claude was asked then
    assert gate(old_view)["crop"] == "box" and "older" in gate(old_view)["view_note"]
    assert gate(named) is None and gate(night) is None and gate(no_embedding) is None
    assert gate(done) == {"why": "score", "top": []}  # already has one
    classifier.backfill_gate(redo=True, limit=1, clf=clf, local=LOCAL, settled=settled, say=lambda m: None)
    assert gate(done)["backfilled"]  # newest first, redone
