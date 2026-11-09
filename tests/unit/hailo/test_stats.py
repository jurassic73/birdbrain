"""The Stats card's pure parts: windows, health, the scorecard, the review list and snapshot name checks."""
from datetime import datetime, timedelta

from birdbrain.hailo import report, stats

NOW = datetime(2026, 11, 10, 12, 0, 0)


def test_window_since_knows_its_keys_and_falls_back_to_the_default():
    assert stats.window_since("24h", NOW) == "2026-11-09T12:00:00"
    assert stats.window_since("3d", NOW) == "2026-11-07T12:00:00"
    assert stats.window_since("7d", NOW) == "2026-11-03T12:00:00"
    assert stats.window_since("all", NOW) == stats.ALL_SINCE
    assert stats.window_since("bogus", NOW) == stats.window_since(stats.DEFAULT_WINDOW, NOW)


def test_health_states():
    assert stats.health(None, NOW)["state"] == "missing"
    assert stats.health({"fps": 15}, NOW)["state"] == "missing"
    assert stats.health({"at": "not a time"}, NOW)["state"] == "missing"
    fresh = stats.health({"at": "2026-11-10T11:59:50", "fps": 15.0, "infer_ms": 11.8, "cpu_pct": 21.0,
                          "chip_temp_c": 43.1, "model": "yolov8s_h8", "view": "crop", "frames": 27690,
                          "uptime_s": 1830, "night": True, "errors": 0, "reconnects": 1}, NOW)
    assert fresh["state"] == "running" and fresh["age_s"] == 10 and fresh["uptime_min"] == 30
    assert fresh["view"] == "crop" and fresh["reconnects"] == 1 and fresh["model"] == "yolov8s_h8"
    stale = stats.health({"at": "2026-11-10T11:00:00"}, NOW)
    assert stale["state"] == "stale" and stale["age_s"] == 3600
    assert stale["view"] in ("crop", "frame")  # the config default stands in when the heartbeat has none


def _event(i, start, end=None, label="bird", score=0.8, in_zone=True, frames=10, snapshots=()):
    return {"id": i, "started_at": NOW + timedelta(seconds=start),
            "ended_at": NOW + timedelta(seconds=end) if end is not None else None, "animal": True,
            "in_zone": in_zone, "label": label, "score": score, "frames": frames, "zone": "bath", "seconds": 3.0,
            "night": 0, "snapshots": list(snapshots)}


def _visit(i, at, real=True, label="bird"):
    return {"id": i, "captured_at": NOW + timedelta(seconds=at), "ended_at": None, "real": real, "label": label}


def test_scorecard_aggregates_the_join():
    events = [_event(1, 100, 130), _event(2, 300, 320, label="cat"), _event(3, 600, 610), _event(4, 900, 905),
              _event(5, 1200, 1210, label="dog")]
    visits = [_visit(10, 120), _visit(11, 330, label="squirrel"), _visit(12, 650), _visit(13, 2000),
              _visit(14, 1205, real=False, label="none"), _visit(15, 3000, real=False, label="none")]
    s = stats.scorecard(report.correlate(events, visits))
    assert (s["events"], s["animal_events"], s["visits"], s["real"], s["false"]) == (5, 5, 6, 4, 2)
    assert s["matched"] == 3 and s["agreement"] == 75 and s["missed"] == 1
    assert s["lead_median"] == 30.0 and s["lead_p90"] == 50.0 and s["ahead"] == 3
    assert s["suppressed"] == 1 and s["suppressed_pct"] == 50 and s["agreed"] == 1
    assert s["detector_only"] == 1
    assert s["classes"] == [{"visit": "bird", "coco": "bird", "n": 2}, {"visit": "squirrel", "coco": "cat", "n": 1},
                            {"visit": "false trigger", "coco": "dog", "n": 1}]


def test_scorecard_with_nothing_has_no_divisions_by_zero():
    s = stats.scorecard(report.correlate([], []))
    assert s["agreement"] is None and s["lead_median"] is None and s["suppressed_pct"] is None
    assert s["classes"] == [] and s["events"] == 0


def test_review_list_is_newest_first_capped_and_keeps_only_servable_shots():
    snaps = ["2026/11/10/120000_1_pre2s.jpg", "2026/11/10/120000_1_pre1s.jpg", "2026/11/10/120000_1_start.jpg",
             "2026/11/10/120000_1_best.jpg", "2026/11/10/120000_1_end.jpg", "../../etc/passwd", "odd_name.jpg"]
    events = [_event(i, 100 * i, 100 * i + 5, snapshots=snaps) for i in range(1, 6)]
    review = stats.review_list(report.correlate(events, []), limit=3)
    assert [e["id"] for e in review] == [5, 4, 3]
    assert [s["tag"] for s in review[0]["shots"]] == ["pre2s", "pre1s", "start", "best"]
    assert review[0]["shots"][0]["word"] == "2 s before"


def test_snapshot_names_must_match_what_the_detector_writes():
    ok = "2026/11/10/120000_12_start.jpg"
    assert stats.SNAPSHOT_RE.fullmatch(ok)
    for bad in ("../2026/11/10/120000_12_start.jpg", "2026/11/10/120000_12_start.png", "2026/11/10/x.jpg",
                "/etc/passwd", "2026/11/10/120000_12_start.jpg/../../x.jpg", "2026/11/10/120000_12_Start.jpg"):
        assert not stats.SNAPSHOT_RE.fullmatch(bad), bad
