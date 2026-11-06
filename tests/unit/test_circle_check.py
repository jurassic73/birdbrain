"""Unit tests of birdbrain/circle: pure functions, no files, database or network."""

import json
import time

import pytest

from helpers.circle_check import MARKS
from birdbrain import circle_check, classifier


def test_owner_circle_is_the_first_one_the_owner_drew():
    assert circle_check.owner_circle(MARKS) == (2, {"x": 0.5, "y": 0.5, "r": 0.05})
    assert circle_check.owner_circle(json.dumps([{"x": 0.2, "y": 0.2, "r": 0.1, "claude": True}])) is None
    assert circle_check.owner_circle("") is None and circle_check.owner_circle("not json") is None


def test_square_is_the_circles_box_with_a_margin_inside_the_frame():
    x, y, w, h = circle_check.square({"x": 0.5, "y": 0.5, "r": 0.05}, 1920, 1080)
    side = 2 * 0.05 * 1920 * circle_check.MARGIN
    assert w == pytest.approx(side / 1920, abs=1e-3) and h == pytest.approx(side / 1080, abs=1e-3)
    assert x == pytest.approx(0.5 - w / 2, abs=1e-3)
    x, y, w, h = circle_check.square({"x": 0.99, "y": 0.99, "r": 0.05}, 1920, 1080)  # near a corner: slides in
    assert x + w <= 1 + 1e-9 and y + h <= 1 + 1e-9


def test_pi_verdicts():
    sure = [{"name": "Dark-eyed Junco", "score": 0.9}]
    weak = [{"name": "Dark-eyed Junco", "score": 0.3}]
    animal, empty = {"score": 0.9, "empty": False}, {"score": 0.01, "empty": True}
    assert circle_check.pi_verdict(sure, 0.9, animal) == "sure"
    assert circle_check.pi_verdict(sure, 0.9, None) == "sure"           # the animal check not trained yet
    assert circle_check.pi_verdict(sure, 0.1, animal) == "unsure"       # mostly on species that aren't local
    assert circle_check.pi_verdict(sure, 0.9, empty) == "unsure"        # the two disagree
    assert circle_check.pi_verdict(weak, 0.9, animal) == "unsure"
    assert circle_check.pi_verdict(weak, 0.9, empty) == "empty"
    assert circle_check.pi_verdict([], None, None) == "empty"
    assert circle_check.pi_say("sure", sure) == "Looks like Dark-eyed Junco."
    assert circle_check.pi_say("unsure", sure) == "Maybe Dark-eyed Junco, but BioCLIP 2 isn't sure."
    assert circle_check.pi_say("empty", []) == "BioCLIP 2 didn't find an animal here."


def test_visitor_of_names_the_form_type():
    m = {"Sciurus carolinensis", "Procyon lotor", "Didelphis virginiana", "Felis catus"}
    assert circle_check.visitor_of("Dark-eyed Junco", "Junco hyemalis", m) == "bird"
    assert circle_check.visitor_of("Eastern Gray Squirrel", "Sciurus carolinensis", m) == "squirrel"
    assert circle_check.visitor_of("Common Raccoon", "Procyon lotor", m) == "raccoon"
    assert circle_check.visitor_of("Virginia Opossum", "Didelphis virginiana", m) == "possum"
    assert circle_check.visitor_of("Domestic Cat", "Felis catus", m) == "other"


def test_release_frees_the_model_only_when_idle(monkeypatch):
    monkeypatch.setattr(classifier, "_classifier", object())
    # monotonic() counts from boot, so "0.0" is only long ago on a machine that's been up for IDLE seconds
    monkeypatch.setattr(circle_check, "_last_used", time.monotonic() - circle_check.IDLE - 1)
    circle_check.release_if_idle()
    assert classifier._classifier is None
    monkeypatch.setattr(classifier, "_classifier", object())
    monkeypatch.setattr(circle_check, "_last_used", time.monotonic())
    circle_check.release_if_idle()
    assert classifier._classifier is not None
