"""Component tests of birdbrain/retention.py (storage ui): one real local dependency, a temp database or files on disk."""

from datetime import timedelta

import pytest

from helpers.storage_ui import MB, NOW, seed
from birdbrain import retention
from birdbrain.web import app as webapp


def test_forecast_raw_pace_and_thinning_stretch(conn, add_visit, monkeypatch):
    seed(conn, add_visit, monkeypatch, budget_mb=200, thin_edge=1280)
    fc = retention.forecast(conn, NOW)
    pace = 30 * MB / 29  # 30 photos fall in the 29-day window (both ends)
    assert fc["ok"] and fc["window_days"] == 29 and fc["bytes_per_day"] == int(pace)
    assert fc["visits_per_day"] == round(30 / 29, 1)
    assert fc["budget_days"] == pytest.approx((200 * MB - 30 * MB) / pace)
    assert fc["budget_date"] == (NOW + timedelta(days=fc["budget_days"])).date().isoformat()
    # thinning: all but the first sighting are thinnable; the 29 stored ones shrink by day 20, new ones from day 21
    share, saved = 29 / 30, 0.38
    steady = pace * (1 - share * saved)
    expect = 20 + (170 * MB - (20 * pace - 29 * MB * saved)) / steady
    assert fc["thin"] and fc["thin_share"] == pytest.approx(share, abs=.001)
    assert fc["thin_days"] == pytest.approx(expect, abs=1.5) and fc["thin_days"] > fc["budget_days"] * 1.4


def test_forecast_without_thinning_has_only_the_raw_pace(conn, add_visit, monkeypatch):
    seed(conn, add_visit, monkeypatch, budget_mb=200, thin_on="0")
    fc = retention.forecast(conn, NOW)
    assert not fc["thin"] and fc["thin_days"] is None and fc["budget_days"] > 0
    retention.save_photo_options(conn, thin_on="1", thin_edge=1920)  # 1920 means don't thin
    assert not retention.forecast(conn, NOW)["thin"]


def test_forecast_needs_a_week_of_visits(conn, add_visit, monkeypatch):
    assert retention.forecast(conn, NOW) == {"ok": False, "reason": "not_enough_data", "history_days": 0.0}
    seed(conn, add_visit, monkeypatch, days=6)
    fc = retention.forecast(conn, NOW)
    assert not fc["ok"] and fc["reason"] == "not_enough_data" and fc["history_days"] == 5.0


def test_forecast_window_is_capped_at_60_days(conn, add_visit, monkeypatch):
    seed(conn, add_visit, monkeypatch, days=100)
    assert retention.forecast(conn, NOW)["window_days"] == 60


def test_forecast_with_no_budget_or_no_growth(conn, add_visit, monkeypatch):
    seed(conn, add_visit, monkeypatch, budget_mb=0)
    fc = retention.forecast(conn, NOW)
    assert fc["ok"] and not fc["budget"] and fc["budget_days"] is None and fc["thin_days"] is None
    assert fc["disk_days"] is not None  # the disk line doesn't need a budget
    seed(conn, add_visit, monkeypatch, days=30, size=0, budget_mb=200)
    fc = retention.forecast(conn, NOW)
    assert fc["no_growth"] and fc["budget_days"] is None and fc["thin_days"] is None and fc["disk_days"] is None


def test_forecast_disk_floor_counts_photos_at_the_raw_pace(conn, add_visit, monkeypatch):
    seed(conn, add_visit, monkeypatch, budget_mb=200, disk_floor_mb=5000)
    monkeypatch.setattr(retention, "disk_free", lambda: 5_100_000_000)
    fc = retention.forecast(conn, NOW)
    assert fc["disk_days"] == pytest.approx(100 * MB / (30 * MB / 29)) and fc["disk_floor"]
    monkeypatch.setattr(retention, "disk_free", lambda: 4_000_000_000)
    assert retention.forecast(conn, NOW)["disk_days"] is None


def blurb(conn, add_visit, monkeypatch, free=100_000_000_000, **kw):
    seed(conn, add_visit, monkeypatch, **kw)
    monkeypatch.setattr(retention, "disk_free", lambda: free)
    retention.forget()
    bd = retention.breakdown(conn)
    return webapp.forecast_blurb(retention.forecast(conn, NOW), bd, NOW)


def test_blurb_text_early_estimate_and_method_tip(conn, add_visit, monkeypatch):
    b = blurb(conn, add_visit, monkeypatch, budget_mb=200, disk_floor_mb=5000, free=5_300_000_000)
    t = b["text"]
    assert t.startswith("Early estimate: At the last 29 days' pace (about 1.0 MB a day, 1 visit a day), photos reach "
                        "the 200 MB budget in about 5 months (around April 2027); with thinning after 20 days, about ")
    assert t.endswith(" Disk floor: about 10 months.")
    assert "29 days" in b["tip"] and "62%" in b["tip"] and "photo growth only" in b["tip"]


def test_blurb_without_prefix_after_a_month_and_with_too_little_history(conn, add_visit, monkeypatch):
    assert not blurb(conn, add_visit, monkeypatch, days=45, budget_mb=200)["text"].startswith("Early estimate")


def test_blurb_with_too_little_history(conn, add_visit, monkeypatch):
    assert blurb(conn, add_visit, monkeypatch, days=3)["text"] == "Not enough history yet for a forecast."
