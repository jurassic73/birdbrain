"""Unit tests of birdbrain/web/viewer: pure functions, no files, database or network."""

import time

import pytest

from birdbrain.web import viewer


def test_old_tries_are_forgotten_and_the_table_stays_small(monkeypatch):
    viewer._fails.clear()
    monkeypatch.setattr(viewer, "MAX_TRACKED", 5)
    now = time.time()
    viewer._fails["stale"] = [now - viewer.LOCKOUT - 1]
    viewer._fails["mixed"] = [now - viewer.LOCKOUT - 1, now - 1]
    viewer.failed("fresh")
    assert "stale" not in viewer._fails and viewer._fails["mixed"] == [pytest.approx(now - 1)]
    assert viewer.locked("stale") == 0
    for n in range(20):
        viewer.failed(f"10.0.0.{n}")
    assert len(viewer._fails) <= 6 and "*" in viewer._fails  # MAX_TRACKED addresses plus everyone's count
    assert len(viewer._fails["*"]) <= viewer.ALL_FAILS
    for _ in range(10):
        viewer.failed("10.0.0.1")
    assert len(viewer._fails["10.0.0.1"]) == viewer.PER_IP_FAILS and viewer.locked("10.0.0.1") > 0
    viewer._fails.clear()


def test_viewer_files_never_resolve_under_etc_in_tests():
    """Guard: the autouse fixture keeps the gate off the host's real /etc/birdbrain files."""
    for p in (viewer.password_file(), viewer.invite_file()):
        assert not str(p).startswith("/etc"), p
