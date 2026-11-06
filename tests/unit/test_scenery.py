"""Unit tests of birdbrain/scenery: pure functions, no files, database or network."""

from helpers.scenery import BIRD, ROCK
from birdbrain import scenery


def test_boxes_seen_in_empty_frames_are_scenery():
    background = [[ROCK], [ROCK], [ROCK]]
    assert scenery.novel_score([ROCK], background) == 0.0
    assert scenery.novel_score([ROCK, BIRD], background) == 0.8


def test_a_box_in_only_one_empty_frame_is_still_new():
    assert scenery.novel_score([BIRD], [[BIRD], [ROCK]]) == 0.8
