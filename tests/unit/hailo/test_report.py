"""Joining detector events to visits by time (pure)."""
from datetime import datetime, timedelta

import pytest

from birdbrain.hailo import report

T0 = datetime(2026, 11, 8, 14, 0, 0)


def event(i, start, end=None, animal=True, in_zone=True, label="bird", score=0.8, frames=10):
    return {"id": i, "started_at": T0 + timedelta(seconds=start),
            "ended_at": T0 + timedelta(seconds=end) if end is not None else None, "animal": animal,
            "in_zone": in_zone, "label": label, "score": score, "frames": frames, "snapshots": [f"e{i}.jpg"]}


def visit(i, at, real=True, label="bird", ended=None):
    return {"id": i, "captured_at": T0 + timedelta(seconds=at), "real": real, "label": label,
            "ended_at": T0 + timedelta(seconds=ended) if ended is not None else None}


def test_parse_since():
    assert report.parse_since("24h") == timedelta(hours=24)
    assert report.parse_since(" 3d ") == timedelta(days=3)
    assert report.parse_since("90m") == timedelta(minutes=90)
    with pytest.raises(ValueError):
        report.parse_since("soon")


def test_correlate_matches_leads_misses_and_suppressions():
    events = [event(1, 100, 130),               # 20 s before visit 10's trigger: a lead
              event(2, 500, 520, in_zone=True),  # nobody triggered: detector-only
              event(3, 700, 701, frames=2),      # too short to bother the owner
              event(4, 900, 905, label="chair", animal=False),  # not an animal: ignored
              event(5, 1000, 1010, in_zone=False)]  # animal elsewhere: not a motion miss
    visits = [visit(10, 120), visit(11, 300), visit(12, 1200, real=False, label="none"),
              visit(13, 1005, real=False, label="none")]
    r = report.correlate(events, visits)
    assert [(m["visit"]["id"], m["event"]["id"], m["lead_s"]) for m in r["matched"]] == [(10, 1, 20.0)]
    assert [v["id"] for v in r["missed"]] == [11]
    assert [e["id"] for e in r["detector_only"]] == [2]
    assert [v["id"] for v in r["suppressed"]] == [12]
    assert [(a["visit"]["id"], a["event"]["id"]) for a in r["agreed"]] == [(13, 5)]
    assert (r["events"], r["animal_events"], r["visits"], r["real"]) == (5, 4, 4, 2)


def test_an_event_during_a_long_stay_belongs_to_that_visit():
    events = [event(1, 200, 260)]
    visits = [visit(10, 100, ended=250)]
    r = report.correlate(events, visits)
    assert len(r["matched"]) == 1 and r["matched"][0]["lead_s"] == -100.0
    assert r["detector_only"] == []


def test_render_lists_everything_with_snapshot_paths():
    r = report.correlate([event(1, 100, 130), event(2, 500, 520)], [visit(10, 120), visit(11, 300)])
    text = report.render(r, "24h")
    assert "Real visits the detector also saw: 1 of 2" in text
    assert "lead +20 s" in text
    assert "visit 11" in text and "never saw: 1" in text
    assert "event 2" in text and "e2.jpg" in text
    assert "left alone: 0 of 0" in text
