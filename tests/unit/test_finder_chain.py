"""Unit tests of birdbrain/finder.py: the chain of finders that points at the animal for the BioCLIP 2 tight look.

The builder's own regression tests for what the test smith's blind tests in test_finder.py do not pin: the blob
finder's size cap, a Motion detector that fails being recorded as failed, what the yolox entry keeps, the blob finder's
night skip, tried_text and config.parse_finders. The chain itself (order, misses, a raising finder, FINDERS=motion)
is pinned there."""

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
    monkeypatch.setattr(config, "CIRCLE_MAX_R", 0.13)
    return knobs


def test_the_order_is_motion_blob_yolox():
    assert finder.FINDERS == ("motion", "blob", "yolox") == config.FINDER_NAMES


def test_a_blob_box_too_big_for_an_animal_is_a_miss_by_the_motion_detectors_bar(chain):
    # A cloud or a glint changes a third of the crop: the Motion detector says "too big", but the crowd counter still
    # outlines one box over most of the crop. Its circle (r 0.3 of the width) is over CIRCLE_MAX_R (0.13), so the blob
    # finder misses the same way and the chain goes on, instead of a 6 s tight look on nearly the whole frame.
    chain["motion"] = (None, {**MISSED, "why": "too big"})
    circle, record = finder.locate(*ARGS, boxes=[[0.1, 0.1, 0.6, 0.5]])
    assert record["finder"] == "yolox" and [(t["finder"], t["why"]) for t in record["tried"]] == [
        ("motion", "too big"), ("blob", "too big"), ("yolox", None)]
    chain["yolox"] = (None, {"why": "nothing", "seconds": 0.3})
    circle, record = finder.locate(*ARGS, boxes=[[0.1, 0.1, 0.6, 0.5]])
    assert circle is None and record["finder"] is None
    assert [t["why"] for t in record["tried"]] == ["too big", "too big", "nothing"]
    # Right at the bar it is still an animal: r = max(0.26, 0.1 * 9/16) / 2 = 0.13.
    circle, record = finder.locate(*ARGS, boxes=[[0.4, 0.5, 0.26, 0.1]])
    assert record["finder"] == "blob" and circle["r"] == 0.13


def test_a_motion_detector_that_raises_is_recorded_as_failed_not_as_not_run(chain, monkeypatch):
    # watch.find_animal stores record["motion"] as calls.motion_circle, or "not run" when the key is absent: a Motion
    # detector that raised did run, so the visit page must say it failed, not that FINDERS left it out.
    def broken(*a, **k):
        raise RuntimeError("decode broke")
    monkeypatch.setattr(finder.motion_circle, "live", broken)
    circle, record = finder.locate(*ARGS, boxes=[BOX])
    assert circle == finder.box_circle(BOX) and record["finder"] == "blob"
    assert record["motion"] == {"found": False, "why": "error", "source": None}
    assert record["tried"][0] == {"finder": "motion", "found": False, "why": "error", "seconds": record["tried"][0]["seconds"]}


def test_the_yolox_entry_keeps_what_the_object_detector_saw(chain):
    chain["motion"] = (None, MISSED)
    _, record = finder.locate(*ARGS, boxes=[])
    assert record["finder"] == "yolox" and record["tried"][2]["label"] == "bird" and record["tried"][2]["score"] == 0.4
    chain["yolox"] = (None, {"why": "no model"})  # no model file on the Pi: its own reason, nothing seen to keep
    entry = finder.locate(*ARGS, boxes=[])[1]["tried"][2]
    assert entry["why"] == "no model" and "label" not in entry and "score" not in entry


def test_at_night_the_blob_finder_is_skipped_too_and_a_skipped_finder_has_no_timing(chain):
    chain["motion"] = (None, MISSED)
    circle, record = finder.locate(b"jpeg", None, None, "rim", "2026-11-06T22:00:00", True, boxes=[BOX])
    assert circle is None and record["finder"] is None
    assert [(t["finder"], t["why"], t["seconds"]) for t in record["tried"]][1:] == [("blob", "night", 0.0), ("yolox", "night", 0.0)]
    assert chain["calls"] == [("motion", "rim", True)]


def test_motion_miss_reason_comes_from_its_record(chain):
    chain["motion"] = (None, {"found": False, "why": "two alike", "patches": 2})
    chain["yolox"] = (None, {"why": "nothing"})
    _, record = finder.locate(*ARGS, boxes=[])
    assert record["tried"][0]["why"] == "two alike" and record["motion"]["patches"] == 2
    chain["motion"] = (None, {})  # a bare miss with no reason
    assert finder.locate(*ARGS, boxes=[])[1]["tried"][0]["why"] == "nothing"


def test_tried_text_lists_each_finder_and_why():
    assert finder.tried_text(None) == "none" and finder.tried_text({}) == "none"
    record = {"finder": "yolox", "tried": [{"finder": "motion", "found": False, "why": "nothing"},
                                           {"finder": "blob", "found": False, "why": "no box"},
                                           {"finder": "yolox", "found": True, "why": None}]}
    assert finder.tried_text(record) == "motion: nothing, blob: no box, yolox: found"


def test_config_parses_the_finder_list_and_rejects_unknown_names():
    assert config.parse_finders("motion,blob,yolox") == ("motion", "blob", "yolox")
    assert config.parse_finders(" yolox, Motion ") == ("yolox", "motion")
    assert config.parse_finders("motion") == ("motion",)
    with pytest.raises(ValueError, match="FINDERS: 'haiku' is not a finder"):
        config.parse_finders("motion,haiku")


def test_config_rejects_an_empty_finder_list_and_a_finder_listed_twice():
    # Empty, rule H would silently never take a tight look; twice, the Motion detector would run twice and overwrite
    # its own record. Both are refused at import like an unknown name, naming FINDERS.
    for raw in ("", " , "):
        with pytest.raises(ValueError, match="FINDERS is empty"):
            config.parse_finders(raw)
    for raw in ("motion,motion", "motion, blob, Motion"):
        with pytest.raises(ValueError, match="FINDERS: .* lists a finder twice"):
            config.parse_finders(raw)
