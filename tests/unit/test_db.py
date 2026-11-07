"""Unit tests of birdbrain/db: pure functions, no files, database or network."""

import json

import pytest

from birdbrain import circle_check, db


def test_who_answered_a_visit():
    from birdbrain import db
    assert db.answerer("bioclip-2") == "BioCLIP 2 species check"
    assert db.answerer("bioclip-presence") == "Pi animal check"
    assert db.answerer("yolox-tiny") == "Local detector"
    assert [db.answerer(m) for m in ("claude-haiku-5-5", "claude-sonnet-5-5", None, "")] == ["Claude"] * 4


@pytest.mark.parametrize("animals, missed, mode, words", [
    (5, 0, "shadow", "5 of 20"),
    (25, 1, "shadow", "missed 1 of 25"),
    (25, 0, "gate", "all 25"),
])
def test_gate_advice(animals, missed, mode, words):
    report = {"animals": animals, "would_miss": [{"id": i, "label": "Bushtit"} for i in range(missed)]}
    got, why = db.gate_advice(report)
    assert got == mode and words in why


def test_marks_keep_the_pi_flag_and_lessons_say_who_drew_it():
    marks = db.parse_marks(json.dumps([{"x": 0.5, "y": 0.4, "r": 0.03, "pi": True, "label": "Junco"},
                                       {"x": 0.2, "y": 0.2, "r": 0.05}]))
    assert marks[0] == {"x": 0.5, "y": 0.4, "r": 0.03, "pi": True, "label": "Junco"}
    said = db.describe_marks(marks)
    assert "drawn on the Pi where it saw motion" in said and "you drew" not in said
    assert circle_check.owner_circle(json.dumps(marks)) == (2, marks[1])  # the owner's own circle, not the Pi's


def test_circles_keep_the_burst_frame_they_were_drawn_on():
    marks = json.dumps([
        {"x": 0.5, "y": 0.5, "r": 0.05},                          # on the photo Claude saw
        {"x": 0.2, "y": 0.3, "r": 0.05, "frame": 4},              # on burst frame 4
        {"x": 0.2, "y": 0.3, "r": 0.05, "frame": 2},              # frame 2 is the photo Claude saw: no frame
        {"x": 0.1, "y": 0.1, "r": 0.05, "frame": 0},              # not a frame
        {"x": 0.1, "y": 0.1, "r": 0.05, "frame": "3"},            # not a number
        {"x": 0.1, "y": 0.1, "r": 0.05, "frame": True},           # not a number either
        {"x": 0.1, "y": 0.1, "r": 0.05, "frame": 999},            # no burst is that long
        {"x": 0.3, "y": 0.3, "r": 0.05, "frame": 5, "claude": True},  # Claude's circles stay on its photo
    ] + [{"x": 0.9, "y": 0.9, "r": 0.02, "frame": 6}] * 4)
    got = db.parse_marks(marks, claude_frame=2)
    assert len(got) == 9  # still at most 9, frames or not
    assert [m.get("frame") for m in got[:8]] == [None, 4, None, None, None, None, None, None]
    assert got[7] == {"x": 0.3, "y": 0.3, "r": 0.05, "claude": True} and got[8]["frame"] == 6
    assert db.parse_marks(marks)[2]["frame"] == 2  # without knowing Claude's frame it stays
    assert db.claude_frame(json.dumps({"burst": {"best": 1}})) == 2 and db.claude_frame("{}") is None
    assert "circle 2 at 20% across, 30% down (on burst frame 4, not the photo you saw)" in db.describe_marks(got)


def test_the_tight_look_crop_name_matches_the_classifier():
    from birdbrain import classifier
    assert db.TIGHT_CROP == classifier.TIGHT_CROP  # agreement_report reads the stored guess by this crop name
