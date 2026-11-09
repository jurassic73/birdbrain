"""The live loop's event path with a fake engine: detections become an event row with its arrival snapshots."""
import json

import numpy as np
import pytest

from birdbrain import config, zones
from birdbrain.hailo import geometry, service, store


class FakeEngine:
    model, infer_ms = "fake_h8", 1.0

    def __init__(self, script):
        self.script, self.calls, self.seen = script, 0, []

    def infer(self, bgr):
        self.seen.append(bgr.shape)
        dets = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        return dets

    def chip_temperature(self):
        return 45.5

    def close(self):
        pass


@pytest.fixture
def shadow(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "HAILO_SHADOW_DB", tmp_path / "hailo_shadow.db")
    monkeypatch.setattr(config, "HAILO_SNAPSHOT_DIR", tmp_path / "snaps")
    monkeypatch.setattr(config, "HAILO_HEARTBEAT", tmp_path / "hb.json")
    monkeypatch.setattr(config, "HAILO_MIN_HITS", 3)
    monkeypatch.setattr(config, "HAILO_EVENT_GAP", 1.0)
    monkeypatch.setattr(config, "HAILO_VIEW", "frame")
    monkeypatch.setattr(service, "ZONES_EVERY", 10 ** 9)  # never re-read the zones from a database
    return tmp_path


def frame(value=60):
    f = np.full((360, 640, 3), value, np.uint8)
    f[..., 2] = 200  # colour, so it is not night
    return f


def run_script(svc, script, dt=0.1, t0=100.0):
    """One step per entry; an entry sets what the fake engine answers from then on (None keeps the last)."""
    for i, dets in enumerate(script):
        if dets is not None:
            svc.engine.script, svc.engine.calls = [dets], 0
        svc.step(frame(), t0 + i * dt)


def test_a_persisting_bird_becomes_an_event_with_arrival_and_best_snapshots(shadow):
    svc = service.Service()
    svc.zones = zones.Zones(ring=(0.5, 0.5, 0.2, 0.15, 0.15, 0.1), crop=(0.2, 0.2, 0.6, 0.6))
    svc.engine = FakeEngine([[]])
    bird = geometry.Detection("bird", 0.6, (0.45, 0.4, 0.5, 0.45))
    better = geometry.Detection("bird", 0.9, (0.46, 0.4, 0.51, 0.45))
    # 30 empty frames (3 s of ring buffer), then the bird for 6 frames, then 1.5 s of nothing
    run_script(svc, [[]] * 30 + [[bird]] * 3 + [[better]] + [[bird]] * 2 + [[]] * 16)
    with store.session() as conn:
        rows = conn.execute("SELECT * FROM events").fetchall()
    assert len(rows) == 1
    ev = rows[0]
    assert ev["label"] == "bird" and ev["score"] == 0.9 and ev["animal"] == 1
    assert ev["zone"] == "bath" and ev["in_zone"] == 1 and ev["frames"] == 6 and ev["ended_at"] is not None
    assert json.loads(ev["labels"]) == {"bird": 6} and json.loads(ev["box"]) == list(better.box)
    assert ev["model"] == "fake_h8" and ev["night"] == 0 and ev["concurrent"] == 1
    snaps = json.loads(ev["snapshots"])
    assert [s.rsplit("_", 1)[1] for s in snaps] == ["pre2s.jpg", "pre1s.jpg", "start.jpg", "best.jpg", "end.jpg"]
    for s in snaps:
        assert (config.HAILO_SNAPSHOT_DIR / s).stat().st_size > 0
    assert svc.open == {} and svc.tracker.tracks == []


def test_non_animal_detections_are_tallied_not_tracked(shadow):
    svc = service.Service()
    svc.zones = zones.from_env()
    svc.engine = FakeEngine([[]])
    toilet = geometry.Detection("toilet", 0.65, (0.4, 0.4, 0.6, 0.7))
    run_script(svc, [[toilet]] * 10)
    with store.session() as conn:
        assert conn.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 0
    assert svc.other_classes == {"toilet": 10}
    svc.heartbeat(101.0, force=True)  # the file is written every few seconds, not every frame
    hb = store.read_heartbeat()
    assert hb["other_classes"] == {"toilet": 10} and hb["chip_temp_c"] == 45.5 and hb["model"] == "fake_h8"
    assert hb["frames"] == 10 and hb["fps"] == pytest.approx(10.0, abs=0.1)


def test_crop_view_feeds_the_engine_the_crop_box_and_maps_boxes_back(shadow, monkeypatch):
    monkeypatch.setattr(config, "HAILO_VIEW", "crop")
    svc = service.Service()
    svc.zones = zones.Zones(crop=(0.25, 0.25, 0.5, 0.5))
    # the engine sees the cut-out and reports a bird filling its lower-right quarter
    svc.engine = FakeEngine([[geometry.Detection("bird", 0.7, (0.5, 0.5, 1.0, 1.0))]])
    run_script(svc, [None] * 3)
    assert svc.engine.seen[0] == (180, 320, 3)
    (track,) = svc.tracker.tracks
    assert track.best.box == (0.5, 0.5, 0.75, 0.75)


def test_a_lost_stream_closes_open_events(shadow):
    svc = service.Service()
    svc.zones = zones.from_env()
    svc.engine = FakeEngine([[geometry.Detection("cat", 0.5, (0.1, 0.1, 0.3, 0.3))]])
    run_script(svc, [None] * 4)
    with store.session() as conn:
        assert conn.execute("SELECT ended_at FROM events").fetchone()[0] is None
    svc.lost_stream()
    with store.session() as conn:
        row = conn.execute("SELECT ended_at, snapshots FROM events").fetchone()
    assert row[0] is not None
    assert [s.rsplit("_", 1)[1] for s in json.loads(row[1])] == ["start.jpg", "best.jpg"]  # no ring buffer yet, no end frame


def test_an_engine_error_drops_the_device_and_the_loop_carries_on(shadow, monkeypatch):
    class Broken(FakeEngine):
        def infer(self, bgr):
            raise RuntimeError("HailoRT says no")

    svc = service.Service()
    svc.zones = zones.from_env()
    svc.engine = Broken([[]])
    monkeypatch.setattr(service.Service, "ensure_engine", lambda self: self.engine is not None)
    svc.step(frame(), 1.0)
    assert svc.engine is None and svc.engine_failed_at > 0
    svc.step(frame(), 1.1)  # no engine: the frame is counted, nothing else
    assert svc.frames == 2
