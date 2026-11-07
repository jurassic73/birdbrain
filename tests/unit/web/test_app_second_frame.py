"""Unit tests of birdbrain/web/app for the shadow second-frame look and the finder wording carried over from the
finder chain: second_frame_words, tight_where_at and the decision trail's circle step. Pure: JSON in, dicts out."""
import json

from helpers.classifier import MOTION_CIRCLE, MOTION_FOUND, guess
from helpers.finder import finder_raw
from birdbrain import classifier
from birdbrain.web import app as webapp, steps

SECOND = {"model": classifier.MODEL_NAME, "crop": classifier.TIGHT_CROP, "night": False, "frame": "second", "seconds": 6.1,
          "local_share": 0.9, "top": [{"name": "Dark-eyed Junco", "scientific": "", "score": 0.874}],
          "combined": {"top": [{"name": "Dark-eyed Junco", "scientific": "", "score": 0.961}], "local_share": 0.92}}


def test_second_frame_words_is_the_one_sentence_with_two_decimals():
    assert webapp.second_frame_words(SECOND) == "A second frame said Dark-eyed Junco (0.87); averaged Dark-eyed Junco (0.96). Shadow only: not used."
    assert webapp.second_frame_words({**SECOND, "combined": None}) == "A second frame said Dark-eyed Junco (0.87). Shadow only: not used."
    assert webapp.second_frame_words(None) is None and webapp.second_frame_words({"top": []}) is None


def test_the_tight_step_carries_the_second_frame_sentence_only_when_there_is_one():
    raw = json.loads(finder_raw("blob", score=0.9))
    raw["calls"]["classifier_second"] = SECOND
    tight = [s for s in webapp.decision_steps(json.dumps(raw))["steps"] if s.get("tight")]
    assert len(tight) == 1 and tight[0]["second_frame"].startswith("A second frame said Dark-eyed Junco (0.87); averaged")
    tight = [s for s in webapp.decision_steps(finder_raw("blob", score=0.9))["steps"] if s.get("tight")]
    assert tight[0]["second_frame"] is None


def test_tight_where_at_follows_the_finder_recorded_with_the_look():
    assert steps.tight_at(None) == "at the patch that moved" and steps.tight_at("motion") == "at the patch that moved"
    assert steps.tight_at("blob") == "at the crowd counter's outline"
    assert steps.tight_at("yolox") == "at the box a small object detector (YOLOX-tiny) drew"
    assert webapp.tight_where_at(json.loads(finder_raw("blob", score=0.9))["calls"]) == "at the crowd counter's outline"
    assert webapp.tight_where_at(json.loads(finder_raw("yolox", named=True))["calls"]).endswith("(YOLOX-tiny) drew")
    assert webapp.tight_where_at({}) == "at the patch that moved"  # visits from before the chain
    first = [s for s in webapp.decision_steps(finder_raw("yolox", score=0.9))["steps"] if s.get("first_look")]
    assert first[0]["tight_at"] == "at the box a small object detector (YOLOX-tiny) drew"


def test_the_circle_step_is_named_after_the_pi_and_knows_whether_a_tight_look_was_made():
    circle = [s for s in webapp.decision_steps(finder_raw("blob", score=0.9))["steps"] if s.get("motion_circle")]
    assert circle[0]["model"] == "Circle from the Pi" and circle[0]["for_tight"] is True
    # a bowl-named visit: the finder only drew the circle, there was no tight look
    bowl = json.dumps({"calls": {"classifier": {**guess("Dark-eyed Junco"), "crop": "bowl"}, "skipped_claude": True,
                                 "settled_by": "classifier", "circle": {**MOTION_CIRCLE, "pi": False},
                                 "motion_circle": {"found": False, "why": "nothing", "source": "empty"},
                                 "finder": {"finder": "blob", "tried": [], "motion": {"found": False, "why": "nothing"}}}})
    circle = [s for s in webapp.decision_steps(bowl)["steps"] if s.get("motion_circle")]
    assert circle[0]["for_tight"] is False and circle[0]["pointed"] == "the crowd counter's outline"
    assert all(s["model"] != "Circle from the motion" for s in webapp.decision_steps(bowl)["steps"])
    assert MOTION_FOUND["found"]  # the helper is the found attempt: the step for it keeps its words
