"""The shadow detector's own database, snapshot folder, heartbeat file and read-only view of the production db."""
import sqlite3
from datetime import datetime

import pytest

from birdbrain import config, db
from birdbrain.hailo import store


@pytest.fixture
def shadow(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "HAILO_SHADOW_DB", tmp_path / "hailo_shadow.db")
    monkeypatch.setattr(config, "HAILO_SNAPSHOT_DIR", tmp_path / "snaps")
    monkeypatch.setattr(config, "HAILO_HEARTBEAT", tmp_path / "hb.json")
    return tmp_path


def test_events_are_inserted_then_closed(shadow):
    with store.session() as conn:
        eid = store.insert_event(conn, started_at="2026-11-08T12:00:00", label="bird", score=0.8,
                                 labels={"bird": 3}, animal=True, zone="bath", in_zone=True, frames=3,
                                 box=[0.1, 0.1, 0.2, 0.2], model="yolov8s_h8", fps=14.9, night=False)
        store.update_event(conn, eid, ended_at="2026-11-08T12:00:09", frames=40, seconds=9.0,
                           snapshots=["2026/11/08/120000_1_start.jpg"])
        row = conn.execute("SELECT * FROM events WHERE id = ?", (eid,)).fetchone()
        assert row["labels"] == '{"bird": 3}' and row["animal"] == 1 and row["in_zone"] == 1
        assert row["ended_at"] == "2026-11-08T12:00:09" and row["frames"] == 40
        assert row["snapshots"] == '["2026/11/08/120000_1_start.jpg"]'
        counts = store.event_counts(conn, "2026-11-08T00:00:00")
        assert counts == {"events": 1, "animals": 1, "in_zone": 1, "open": 0}
        assert store.event_counts(conn, "2026-11-09T00:00:00")["events"] == 0
        assert [r["id"] for r in store.events_since(conn, "2026-11-08T00:00:00")] == [eid]


def test_the_shadow_database_is_its_own_file_not_birdbrain_db(shadow, conn):
    with store.session() as shadow_conn:
        store.insert_event(shadow_conn, started_at="2026-11-08T12:00:00", frames=1)
    assert (shadow / "hailo_shadow.db").exists()
    with pytest.raises(sqlite3.OperationalError):  # the production schema has no events table
        conn.execute("SELECT COUNT(*) FROM events")


def test_heartbeat_file_round_trips_and_a_missing_one_is_none(shadow):
    assert store.read_heartbeat() is None
    store.write_heartbeat({"fps": 14.9, "chip_temp_c": 47.2})
    assert store.read_heartbeat() == {"fps": 14.9, "chip_temp_c": 47.2}
    assert not (shadow / "hb.tmp").exists()
    with store.session() as conn:
        store.add_heartbeat(conn, at="2026-11-08T12:00:00", fps=14.9, infer_ms=11.0, cpu_pct=30.0, chip_temp_c=47.0,
                            frames=1000, model="yolov8s_h8")
        store.add_heartbeat(conn, at="2026-11-08T12:00:00", fps=15.0)  # same minute: replaced, not doubled
        assert conn.execute("SELECT COUNT(*), MAX(fps) FROM heartbeats").fetchone()[:] == (1, 15.0)


def test_snapshots_are_saved_by_day_and_the_oldest_pruned_first(shadow):
    when = datetime(2026, 11, 8, 12, 0, 0)
    rel = store.snapshot_rel(when, 7, "start")
    assert rel == "2026/11/08/120000_7_start.jpg"
    for day, tag in ((6, "a"), (7, "b"), (8, "c")):
        store.save_jpeg(store.snapshot_rel(when.replace(day=day), 1, tag), b"x" * 1000)
    assert store.folder_size(config.HAILO_SNAPSHOT_DIR) == 3000
    removed = store.prune_snapshots(max_mb=2500 / 1024 / 1024)
    assert removed == 1
    left = sorted(p.name for p in config.HAILO_SNAPSHOT_DIR.rglob("*.jpg"))
    assert left == ["120000_1_b.jpg", "120000_1_c.jpg"]
    assert not (config.HAILO_SNAPSHOT_DIR / "2026/11/06").exists()  # the emptied day folder is gone too
    assert store.prune_snapshots(max_mb=0) == 0  # no cap
    assert store.prune_snapshots(root=shadow / "nowhere", max_mb=1) == 0


def test_production_db_is_read_only(conn, add_visit):
    add_visit()
    ro = store.production_db()
    try:
        assert ro.execute("SELECT COUNT(*) FROM visits").fetchone()[0] == 1
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            ro.execute("UPDATE visits SET visitor = 'none'")
    finally:
        ro.close()
    assert db.get_visit(conn, 1)["visitor"] == "bird"
