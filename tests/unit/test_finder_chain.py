"""Unit tests of birdbrain/finder.py: the chain of finders that points at the animal for the BioCLIP 2 tight look.

The builder's own regression tests; the test smith's blind tests are in test_finder.py, and the three cases both
files covered the same way (motion finds first, one blob box after a motion miss, FINDERS=motion alone) live there."""

import pytest

from birdbrain import config, finder

CIRCLE = {"x": 0.48, "y": 0.5, "r": 0.04, "pi": True}
FOUND = {"found": True, "source": "empty", "where": "land", "seconds": 0.02}
MISSED = {"found": False, "source": "empty", "why": "nothing", "seconds": 0.02}
BOX = [0.4, 0.5, 0.1, 0.2]
ARGS = (b"jpeg", None, None, "rim", "2026-11-06T10:00:00", False)


@pytest.fixture
def chain(monkeypatch):
    """Fakes for the three finders with a log of calls; knobs: "motion" (the live() answer), "yolox" (animal_box's)."""
    knobs = {"motion": (CIRCLE, FOUND), "yolox": ([0.6, 0.3, 0.05, 0.08], {"label": "bird", "score": 0.4, "seconds": 0.3}),
             "calls": []}

    def live(jpeg, empty, z, trigger, captured_at, night):
        knobs["calls"].append(("motion", trigger, night))
        return knobs["motion"]

    def animal_box(jpeg, crop, min_score):
        knobs["calls"].append(("yolox", crop, min_score))
        return knobs["yolox"]

    monkeypatch.setattr(finder.motion_circle, "live", live)
    monkeypatch.setattr(finder.detector, "animal_box", animal_box)
    monkeypatch.setattr(config, "FINDERS", ("motion", "blob", "yolox"))
    monkeypatch.setattr(config, "FINDER_YOLOX_MIN_SCORE", 0.15)
    return knobs


def test_the_order_is_motion_blob_yolox():
    assert finder.FINDERS == ("motion", "blob", "yolox") == config.FINDER_NAMES


def test_box_circle_is_centred_on_the_box_and_reaches_its_longer_side():
    assert finder.box_circle(BOX) == {"x": 0.45, "y": 0.6, "r": 0.0562, "pi": True}  # max(0.1, 0.2 * 9/16) / 2
    assert finder.box_circle([0.5, 0.5, 0.3, 0.01])["r"] == 0.15  # the width wins
    assert finder.box_circle([0.5, 0.5, 0.002, 0.002])["r"] == 0.01  # never smaller than the Motion detector's least
    assert finder.box_circle([0.4, 0.5, 0.1, 0.2], aspect=1.0)["r"] == 0.1


@pytest.mark.parametrize("boxes, why", [(None, "no count"), ([], "no box"), ([BOX, [0.1, 0.1, 0.05, 0.05]], "several")])
def test_yolox_runs_once_when_motion_and_blob_miss(chain, boxes, why):
    chain["motion"] = (None, MISSED)
    circle, record = finder.locate(*ARGS, boxes=boxes, crop=[0.1, 0.1, 0.8, 0.8])
    assert circle == finder.box_circle([0.6, 0.3, 0.05, 0.08]) and record["finder"] == "yolox"
    assert chain["calls"] == [("motion", "rim", False), ("yolox", [0.1, 0.1, 0.8, 0.8], 0.15)]
    tried = record["tried"]
    assert [(t["finder"], t["found"], t["why"]) for t in tried] == [("motion", False, "nothing"), ("blob", False, why), ("yolox", True, None)]
    assert tried[2]["label"] == "bird" and tried[2]["score"] == 0.4  # what the object detector saw, for the record


def test_all_miss_gives_no_circle_and_three_tries(chain):
    chain["motion"] = (None, MISSED)
    chain["yolox"] = (None, {"why": "nothing", "seconds": 0.3})
    circle, record = finder.locate(*ARGS, boxes=[])
    assert circle is None and record["finder"] is None and len(record["tried"]) == 3
    assert [t["why"] for t in record["tried"]] == ["nothing", "no box", "nothing"]
    assert finder.tried_text(record) == "motion: nothing, blob: no box, yolox: nothing"
    chain["yolox"] = (None, {"why": "no model"})
    assert finder.locate(*ARGS, boxes=[])[1]["tried"][2]["why"] == "no model"


def test_a_finder_that_raises_is_recorded_and_the_chain_goes_on(chain, monkeypatch):
    def broken(*a, **k):
        raise RuntimeError("boom")
    monkeypatch.setattr(finder.motion_circle, "live", broken)
    circle, record = finder.locate(*ARGS, boxes=[BOX])
    assert circle == finder.box_circle(BOX) and record["finder"] == "blob" and "motion" not in record
    assert record["tried"][0] == {"finder": "motion", "found": False, "why": "error", "seconds": record["tried"][0]["seconds"]}
    monkeypatch.setattr(finder.detector, "animal_box", broken)
    circle, record = finder.locate(*ARGS, boxes=[])
    assert circle is None and [t["why"] for t in record["tried"]] == ["error", "no box", "error"]
    assert chain["calls"] == []


def test_config_finders_order_is_honoured(chain, monkeypatch):
    monkeypatch.setattr(config, "FINDERS", ("yolox", "motion"))
    circle, record = finder.locate(*ARGS, boxes=[BOX])
    assert record["finder"] == "yolox" and [c[0] for c in chain["calls"]] == ["yolox"] and "motion" not in record
    monkeypatch.setattr(config, "FINDERS", ("blob",))
    circle, record = finder.locate(*ARGS, boxes=[BOX])
    assert record["finder"] == "blob" and len(chain["calls"]) == 1  # neither motion nor yolox ran
    monkeypatch.setattr(config, "FINDERS", ())
    assert finder.locate(*ARGS, boxes=[BOX]) == (None, {"finder": None, "tried": []})


def test_at_night_only_the_motion_detector_runs(chain):
    chain["motion"] = (None, MISSED)
    circle, record = finder.locate(b"jpeg", None, None, "rim", "2026-11-06T22:00:00", True, boxes=[BOX])
    assert circle is None and record["finder"] is None
    assert [(t["finder"], t["why"], t["seconds"]) for t in record["tried"]] == [("motion", "nothing", record["tried"][0]["seconds"]),
                                                                              ("blob", "night", 0.0), ("yolox", "night", 0.0)]
    assert chain["calls"] == [("motion", "rim", True)]  # yolox never called at night


def test_motion_miss_reason_comes_from_its_record(chain):
    chain["motion"] = (None, {"found": False, "why": "two alike", "patches": 2})
    chain["yolox"] = (None, {"why": "nothing"})
    _, record = finder.locate(*ARGS, boxes=[])
    assert record["tried"][0]["why"] == "two alike" and record["motion"]["patches"] == 2
    chain["motion"] = (None, {})  # a bare miss with no reason
    assert finder.locate(*ARGS, boxes=[])[1]["tried"][0]["why"] == "nothing"


def test_tried_text_without_a_record():
    assert finder.tried_text(None) == "none" and finder.tried_text({}) == "none"


def test_config_parses_the_finder_list_and_rejects_unknown_names():
    assert config.parse_finders("motion,blob,yolox") == ("motion", "blob", "yolox")
    assert config.parse_finders(" yolox, Motion ") == ("yolox", "motion")
    assert config.parse_finders("motion") == ("motion",) and config.parse_finders("") == ()
    with pytest.raises(ValueError, match="FINDERS: 'haiku' is not a finder"):
        config.parse_finders("motion,haiku")
