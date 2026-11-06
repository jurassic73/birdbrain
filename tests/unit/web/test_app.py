"""Unit tests of birdbrain/web/app: pure functions, no files, database or network."""

import json
from datetime import timedelta

import pytest

from helpers.clock import FROZEN_NOW
from helpers.storage_ui import NOW
from helpers.web import calibrated
from helpers.web_security import ROUTES
from birdbrain import config
from birdbrain.web import app as webapp


def test_stayed_label():
    assert webapp.stayed("2026-10-28T09:00:00", "2026-10-28T09:03:40") == "stayed 4 min"
    assert webapp.stayed("2026-10-28T09:00:00", "2026-10-28T09:00:40") == ""
    assert webapp.stayed("2026-10-28T09:00:00", None) == ""


def test_spans_and_dates_are_rounded_to_words():
    assert [webapp._span(d) for d in (1, 45, 245, 700, 800, 1000)] == [
        "1 day", "45 days", "8 months", "23 months", "2 years", "2.5 years"]
    assert webapp._span(40 * 365) == "more than 30 years"
    assert webapp._when(30, NOW) == "December 2" and webapp._when(245, NOW) == "July 2027"


def test_decision_steps_handles_refusals_and_old_or_broken_records():
    refused = json.dumps({"calls": {"haiku": {"answer": None}, "sonnet": {"frames": 3, "answer": None}}})
    steps = webapp.decision_steps(refused)["steps"]
    assert [s["refused"] for s in steps] == [True, True]
    assert [s["error"] for s in steps] == [None, None] and not webapp.decision_steps(refused)["failed"]


@pytest.mark.parametrize("value, text", [(0, "0.0 MB"), (420_500_000, "420 MB"), (1_500_000_000, "1.5 GB")])
def test_mb_formatting(value, text):
    assert webapp.mb(value) == text


@pytest.mark.parametrize("days, text", [(0, "today"), (1, "yesterday"), (5, "~5 days ago"), (21, "~3 weeks ago"),
                                         (100, "~3 months ago"), (400, "~13 months ago"), (900, "~2 years ago")])
def test_since_ago(days, text):
    assert webapp.since_ago((webapp.clock.now() - timedelta(days=days)).isoformat(timespec="seconds")) == text


@pytest.mark.parametrize("value, text", [(0, "$0"), (0.0004, "under $0.001"), (0.031, "$0.031"), (0.42, "$0.42")])
def test_usd_formatting(value, text):
    assert webapp.usd(value) == text


def test_zone_shapes_match_the_motion_masks(calibrated, monkeypatch):
    z = webapp.zone_shapes()
    assert z["rim"].startswith("M40.00,45.00 a40.00,27.00")                     # outer: cx .5, rx .25 of 160x90
    assert "M48.00,43.20 a32.00,19.80" in z["rim"]                              # inner ellipse uses inner_cy
    assert z["water"].count("M") == 2 and len(z["spots"]) == 1
    assert z["crop"] == {"x": 32.0, "y": 9.0, "w": 96.0, "h": 72.0}
    assert z["ground"] is None
    monkeypatch.setattr(config, "GROUND_ZONE", ((0.1, 0.5), (0.9, 0.5), (0.9, 1)))
    ground = webapp.zone_shapes()["ground"]
    assert ground.startswith("M16.00,45.00 L144.00,45.00 L144.00,90.00 Z")      # polygon in 160x90
    assert ground.endswith(webapp.ellipse_path(0.5, 0.5, 0.25, 0.3))            # minus the bath, like ground_mask


@pytest.mark.parametrize("secs,want", [
    (0, "(just now)"), (59, "(just now)"), (60, "(1 min ago)"), (119, "(1 min ago)"), (59 * 60, "(59 min ago)"),
    (3600, "(~1 hour ago)"), (2 * 3600, "(~2 hours ago)"), (23 * 3600, "(~23 hours ago)"),
    (86400, "(1 day ago)"), (6 * 86400, "(6 days ago)"), (7 * 86400, "(1 week ago)"),
    (13 * 86400, "(1 week ago)"), (14 * 86400, "(2 weeks ago)"), (30 * 86400, "(1 month ago)"),
    (75 * 86400, "(2 months ago)"), (400 * 86400, "(1 year ago)"),
])
def test_ago_rough_wording(secs, want):
    assert webapp.ago((FROZEN_NOW - timedelta(seconds=secs)).isoformat()) == want


def test_ago_bare_form_has_no_brackets():
    for secs, want in ((30, "just now"), (300, "5 min ago"), (7200, "~2 hours ago"), (2 * 86400, "2 days ago")):
        assert webapp.ago((FROZEN_NOW - timedelta(seconds=secs)).isoformat(), bare=True) == want


def test_route_list_is_real():
    assert ("GET", "/calibration/{name}", "/calibration/x") in ROUTES
    assert ("POST", "/settings/camera/reboot", "/settings/camera/reboot") in ROUTES
