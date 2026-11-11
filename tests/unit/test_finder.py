"""Unit tests of birdbrain/finder: the finder chain for the tight look, with its collaborators (motion_circle.live,
detector.animal_box) faked at the boundary. Pure: no files, database or network.

The module is imported inside each test (not at the top) so this file collects on a tree that doesn't have it yet."""
import importlib

import pytest

from helpers.finder import BLOB_CIRCLE, BOX, MISS, YOLOX_BOX, YOLOX_CIRCLE
from helpers.classifier import MOTION_CIRCLE, MOTION_FOUND
from helpers.images import flat_jpeg
from helpers.motion_circle import ZONES
from birdbrain import config, detector, motion_circle

JPEG = flat_jpeg()
EMPTY = object()  # motion_circle.live's empty view: passed through untouched
CROP = (0.25, 0.25, 0.5, 0.55)
AT = "2026-11-06T12:00:00"
NOTHING = (None, {"why": "nothing", "seconds": 0.21})  # detector.animal_box with no visitor-class detection
ALL = ("motion", "blob", "yolox")


def finder():
    return importlib.import_module("birdbrain.finder")


class Fail(Exception):
    pass


@pytest.fixture
def chain(monkeypatch):
    """The finders' collaborators, faked: motion_circle.live answers knobs["motion"], detector.animal_box answers
    knobs["yolox"] (an Exception in either is raised), and every call is logged under calls[name].
    config.FINDERS starts at the default order."""
    knobs = {"motion": (None, MISS), "yolox": NOTHING, "calls": {"motion": [], "yolox": []}}

    def live(jpeg, empty, z, trigger, captured_at, night):
        knobs["calls"]["motion"].append((jpeg, empty, z, trigger, captured_at, night))
        if isinstance(knobs["motion"], Exception):
            raise knobs["motion"]
        return knobs["motion"]

    def animal_box(jpeg, crop, min_score=None):
        knobs["calls"]["yolox"].append((jpeg, crop, min_score))
        if isinstance(knobs["yolox"], Exception):
            raise knobs["yolox"]
        return knobs["yolox"]

    monkeypatch.setattr(motion_circle, "live", live)
    monkeypatch.setattr(detector, "animal_box", animal_box, raising=False)
    monkeypatch.setattr(config, "FINDERS", ALL, raising=False)
    return knobs


def locate(boxes=None, night=False, trigger="rim"):
    return finder().locate(JPEG, EMPTY, ZONES, trigger, AT, night, boxes=boxes, crop=CROP)


def names(record):
    return [t["finder"] for t in record["tried"]]


def test_the_finders_run_motion_then_blob_then_yolox():
    assert finder().FINDERS == ALL


def test_box_circle_is_centred_on_the_box_with_r_from_its_bigger_side():
    box_circle = finder().box_circle
    # behaviour 4: r = max(0.1, 0.2 * 9/16) / 2 = 0.05625, rounded to 4 places
    assert box_circle([0.4, 0.5, 0.1, 0.2]) == {"x": 0.45, "y": 0.6, "r": 0.0562, "pi": True}
    assert box_circle(BOX) == BLOB_CIRCLE and box_circle(YOLOX_BOX) == YOLOX_CIRCLE
    assert box_circle([0.4, 0.5, 0.2, 0.1]) == {"x": 0.5, "y": 0.55, "r": 0.1, "pi": True}  # the width wins here
    assert box_circle([0.4, 0.5, 0.1, 0.2], aspect=1.0)["r"] == 0.1  # a square frame: the height counts in full
    assert box_circle([0.5, 0.5, 0.002, 0.002])["r"] == 0.01  # never smaller than 0.01 of the width
    assert box_circle([0.123456, 0.2, 0.1, 0.1])["x"] == 0.1735  # 4 places
    assert box_circle((0.4, 0.5, 0.1, 0.2)) == BLOB_CIRCLE  # a tuple box works too


def test_every_tried_entry_has_the_same_four_keys(chain):
    circle, record = locate(boxes=None)
    assert circle is None and record["finder"] is None
    for t in record["tried"]:
        assert set(t) == {"finder", "found", "why", "seconds"}
        assert t["found"] is False and isinstance(t["why"], str) and isinstance(t["seconds"], float)


def test_motion_finding_the_patch_ends_the_chain(chain):
    chain["motion"] = (MOTION_CIRCLE, MOTION_FOUND)
    circle, record = locate(boxes=[BOX])  # a blob box is there, but motion came first
    assert circle == MOTION_CIRCLE and record["finder"] == "motion"
    assert record["motion"] == MOTION_FOUND and names(record) == ["motion"]
    assert record["tried"][0]["found"] is True and not record["tried"][0]["why"]
    assert chain["calls"]["motion"] == [(JPEG, EMPTY, ZONES, "rim", AT, False)]  # live is called as today
    assert chain["calls"]["yolox"] == []


def test_one_blob_box_gives_its_circle_when_motion_misses(chain):
    circle, record = locate(boxes=[BOX])
    assert circle == BLOB_CIRCLE and record["finder"] == "blob"
    assert record["motion"] == MISS and names(record) == ["motion", "blob"]
    assert record["tried"][0] == {"finder": "motion", "found": False, "why": "nothing",
                                  "seconds": record["tried"][0]["seconds"]}
    assert record["tried"][1]["found"] is True and not record["tried"][1]["why"]
    assert chain["calls"]["yolox"] == []  # found before yolox: no inference


def test_yolox_runs_once_when_motion_and_blob_both_miss(chain):
    chain["yolox"] = (YOLOX_BOX, {"label": "bird", "score": 0.42, "seconds": 0.21})
    circle, record = locate(boxes=[])
    assert circle == YOLOX_CIRCLE and record["finder"] == "yolox"
    assert names(record) == ["motion", "blob", "yolox"]
    assert [t["found"] for t in record["tried"]] == [False, False, True]
    assert record["tried"][1]["why"] == "no box"
    assert chain["calls"]["yolox"] == [(JPEG, CROP, config.FINDER_YOLOX_MIN_SCORE)]


@pytest.mark.parametrize("boxes, why", [(None, "no count"), ([], "no box"), ([BOX, YOLOX_BOX], "several")])
def test_all_three_missing_gives_no_circle_and_three_tried_entries(chain, boxes, why):
    circle, record = locate(boxes=boxes)
    assert circle is None and record["finder"] is None
    assert names(record) == ["motion", "blob", "yolox"]
    assert [t["why"] for t in record["tried"]] == ["nothing", why, "nothing"]
    assert not any(t["found"] for t in record["tried"])
    assert record["motion"] == MISS
    assert len(chain["calls"]["motion"]) == 1 and len(chain["calls"]["yolox"]) == 1


def test_a_finder_that_raises_is_an_error_entry_and_the_chain_goes_on(chain):
    chain["motion"], chain["yolox"] = Fail("decode broke"), Fail("onnx broke")
    circle, record = locate(boxes=None)  # never raises
    assert circle is None and record["finder"] is None
    assert [(t["finder"], t["found"], t["why"]) for t in record["tried"]] == [
        ("motion", False, "error"), ("blob", False, "no count"), ("yolox", False, "error")]
    # motion broke, but the crowd counter's box still gives the tight look its circle
    circle, record = locate(boxes=[BOX])
    assert circle == BLOB_CIRCLE and record["finder"] == "blob"
    assert record["tried"][0]["why"] == "error"


def test_config_finders_order_is_honoured(chain, monkeypatch):
    monkeypatch.setattr(config, "FINDERS", ("yolox", "motion"), raising=False)
    chain["yolox"] = (YOLOX_BOX, {"label": "cat", "score": 0.5, "seconds": 0.2})
    circle, record = locate(boxes=[BOX])
    assert circle == YOLOX_CIRCLE and record["finder"] == "yolox" and names(record) == ["yolox"]
    assert chain["calls"]["motion"] == [] and record.get("motion") is None  # motion never ran, blob isn't listed
    chain["yolox"] = NOTHING
    chain["motion"] = (MOTION_CIRCLE, MOTION_FOUND)
    circle, record = locate(boxes=[BOX])
    assert circle == MOTION_CIRCLE and record["finder"] == "motion" and names(record) == ["yolox", "motion"]
    assert record["motion"] == MOTION_FOUND
    monkeypatch.setattr(config, "FINDERS", ("blob",), raising=False)
    circle, record = locate(boxes=[BOX])
    assert circle == BLOB_CIRCLE and names(record) == ["blob"] and len(chain["calls"]["motion"]) == 1


def test_finders_motion_only_behaves_like_motion_circle_live(chain, monkeypatch):
    monkeypatch.setattr(config, "FINDERS", ("motion",), raising=False)
    chain["motion"] = (MOTION_CIRCLE, MOTION_FOUND)
    circle, record = locate(boxes=[BOX])
    assert (circle, record["motion"]) == motion_circle.live(JPEG, EMPTY, ZONES, "rim", AT, False)
    assert record["finder"] == "motion" and names(record) == ["motion"]
    chain["motion"] = (None, MISS)
    circle, record = locate(boxes=[BOX])  # a box and a detector are there, but only motion may run
    assert circle is None and record["finder"] is None and record["motion"] == MISS
    assert names(record) == ["motion"] and record["tried"][0]["why"] == "nothing"
    assert chain["calls"]["yolox"] == []


def test_at_night_yolox_is_skipped_and_says_so(chain):
    chain["yolox"] = (YOLOX_BOX, {"label": "cat", "score": 0.5, "seconds": 0.2})  # it would find one
    circle, record = locate(boxes=None, night=True)
    assert circle is None and record["finder"] is None
    assert chain["calls"]["yolox"] == []
    assert chain["calls"]["motion"][0][-1] is True  # motion still runs, told it is night
    yolox = next(t for t in record["tried"] if t["finder"] == "yolox")
    assert yolox["found"] is False and yolox["why"] == "night"


def test_the_trigger_is_passed_through_to_motion(chain):
    chain["motion"] = (MOTION_CIRCLE, MOTION_FOUND)
    locate(trigger="ground")
    assert chain["calls"]["motion"][0][3] == "ground"


def clear_record(**changes):
    """The chain's record when every finder looked and saw nothing (the rain frames of 2026-11-10), with one
    finder's entry changed: clear_record(motion="light")."""
    whys = {"motion": "nothing", "blob": "no box", "yolox": "nothing", **changes}
    return {"finder": None, "motion": {"found": False, "why": whys["motion"], "source": "empty"},
            "tried": [{"finder": n, "found": False, "why": whys[n], "seconds": 0.1} for n in ALL]}


def test_nothing_on_land_needs_every_finder_to_have_looked_and_seen_nothing():
    f = finder()
    assert f.nothing_on_land(clear_record())
    # Something did change, a finder couldn't look, or it failed: nobody can say the land was empty.
    for name, why in (("motion", "light"), ("motion", "too big"), ("motion", "two alike"), ("motion", "no empty view"),
                      ("motion", "error"), ("blob", "no count"), ("blob", "several"), ("blob", "too big"),
                      ("yolox", "no model"), ("yolox", "error"), ("yolox", "night")):
        assert not f.nothing_on_land(clear_record(**{name: why})), (name, why)
    found = clear_record()
    found["finder"], found["tried"][1]["found"] = "blob", True
    assert not f.nothing_on_land(found)
    short = clear_record()
    short["tried"] = short["tried"][:1]  # FINDERS=motion: the other two never looked
    assert not f.nothing_on_land(short)
    stale = clear_record()
    stale["motion"]["source"] = "calibration"  # not the trigger's own empty scene
    assert not f.nothing_on_land(stale)
    assert not f.nothing_on_land(None) and not f.nothing_on_land({})
