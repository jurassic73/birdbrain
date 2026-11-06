"""Shared helpers for the storage ui tests (used by more than one test module)."""
from datetime import datetime, timedelta

from birdbrain import retention


NOW = datetime(2026, 11, 2, 12, 0, 0)


MB = 1_000_000


def seed(conn, add_visit, monkeypatch, days=30, size=MB, thumb=0, **opts):
    """One real visit a day at noon for `days` days (today first), each photo `size` bytes."""
    photos = {}
    for d in range(days):
        when = (NOW - timedelta(days=d)).isoformat()
        vid = add_visit(captured_at=when)
        photos[vid] = retention.Photo(vid, when, "i.jpg", "t.jpg", True, None, None, False, False, size, thumb, 0)
    monkeypatch.setattr(retention, "_photos", lambda c: photos)
    monkeypatch.setattr(retention, "disk_free", lambda: 100_000_000_000)
    retention.save_photo_options(conn, **{"rare_under": 0, "full_per_day": 0, "full_days": 20, **opts})
    return photos
