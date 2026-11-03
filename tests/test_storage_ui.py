"""Settings > Storage: the usage bar, the limits form, preview and run now."""
import time
from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from birdbrain import db, retention
from birdbrain.web import app as webapp
from birdbrain.web import auth

FORM = {"budget_gb": "5", "budget_custom": "", "full_days": "90", "full_per_day": "2", "thin_edge": "1280",
        "warn_pct": "80", "rare_under": "10", "disk_floor_gb": "5", "thin_on": "1"}


@pytest.fixture
def client(conn):
    retention.forget()
    with TestClient(webapp.app) as c:
        yield c
    retention.forget()


def test_settings_shows_the_bar_and_limits_form(client, add_visit):
    add_visit()
    html = client.get("/settings").text
    assert 'id="storage-bar"' in html and 'name="full_days"' in html and "Run now" in html
    assert "Not counted: models" in html and "False triggers" in html


def test_bar_renders_with_no_data_and_over_budget(client, conn):
    assert 'class="sbar' in client.get("/settings").text
    assert 'class="sbar' in client.get("/false-triggers").text
    retention.save_photo_options(conn, budget_mb=0.001)
    retention.forget()
    # fake some use: a tiny budget is exceeded by thumbnails alone once any photo exists
    bd = retention.breakdown(conn)
    bd = {**bd, "used_bytes": 5_000_000, "budget_bytes": 1000, "thumbs": {"count": 1, "bytes": 5_000_000}}
    bar = webapp.storage_bar(bd)
    assert bar["over"] and bar["budget_at"] is not None and bar["warn_at"] is None
    html = webapp.templates.get_template("_storage_bar.html").render(bd=bd, bar=bar)
    assert "sbar over" in html and "over by" in html


def test_save_limits_and_validation_error(client, conn):
    r = client.post("/settings/storage/limits", data={**FORM, "budget_gb": "custom", "budget_custom": "7.5",
                                                      "thin_edge": "960"}, follow_redirects=False)
    assert r.status_code == 303 and "limits_saved=1" in r.headers["location"]
    o = retention.options(conn)
    assert o["budget_mb"] == 7500 and o["thin_edge"] == 960
    r = client.post("/settings/storage/limits", data={**FORM, "warn_pct": "250"}, follow_redirects=False)
    assert "limits_error=" in r.headers["location"]
    assert retention.options(conn)["warn_pct"] == 80  # nothing stored
    page = client.get(r.headers["location"]).text
    assert "Not saved" in page and "warn_pct" in page
    client.post("/settings/storage/limits", data={k: v for k, v in FORM.items() if k != "thin_on"})
    assert retention.options(conn)["thin_on"] is False
    client.post("/settings/storage/limits", data={**FORM, "budget_gb": "none"})
    assert retention.options(conn)["budget_mb"] == 0


def test_preview_and_run_now(client, conn, add_visit):
    add_visit()
    assert "Nothing to do" in client.get("/settings/storage/preview").text
    r = client.post("/settings/storage/run")
    assert r.status_code == 200
    for _ in range(50):
        if not webapp._run_state["running"]:
            break
        time.sleep(0.1)
    assert not webapp._run_state["running"] and not webapp._run_state["error"]
    assert "Done." in client.get("/settings/storage/run").text


def test_run_now_says_so_when_the_watcher_is_already_running(client, conn, add_visit):
    import fcntl

    add_visit()
    with open(retention.config.DATA_DIR / retention.LOCK_FILE, "a") as watcher:  # the watcher's daily run, mid-way
        fcntl.flock(watcher, fcntl.LOCK_EX | fcntl.LOCK_NB)
        client.post("/settings/storage/run")
        for _ in range(50):
            if not webapp._run_state["running"]:
                break
            time.sleep(0.1)
    assert "already running" in webapp._run_state["error"]
    assert "already running" in client.get("/settings/storage/run").text


def test_run_now_is_post_and_admin_only(client, monkeypatch):
    assert client.get("/settings/storage/limits").status_code == 405
    monkeypatch.setenv("ADMIN_PASSWORD", "s3cret!")
    auth._fails.clear()
    for path in ("/settings/storage/run", "/settings/storage/limits"):
        assert client.post(path).status_code == 403
    r = client.get("/settings/storage/preview", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"].startswith("/login")


def test_about_has_storage_section(client):
    html = client.get("/about").text
    assert 'id="storage"' in html and "Never thinned" in html


def _bd(used, budget, warn=80):
    cats = [{"key": k, "label": k, "count": 1, "bytes": b} for k, b in
            (("species_photos", 10), ("keepers", 0), ("training", 500), ("recent", 9000), ("older", 100), ("false_triggers", 40))]
    return {"categories": cats, "thumbs": {"count": 3, "bytes": 200}, "used_bytes": used, "budget_bytes": budget,
            "warn_pct": warn, "options": {"full_days": 90, "full_per_day": 2}, "false_trigger_limit_bytes": 500_000_000,
            "disk": {"free_bytes": 5_000_000_000, "floor_bytes": 0},
            "system": {"models": 1, "database": 1, "calibration": 1}}


def _render(bd):
    bar = webapp.storage_bar(bd)
    return bar, webapp.templates.get_template("_storage_bar.html").render(bd=bd, bar=bar)


def test_table_sorted_by_size_with_scale_labels():
    bar, html = _render(_bd(147_000_000, 10_000_000_000))
    sizes = [r["bytes"] for r in bar["table"]]
    assert sizes == sorted(sizes, reverse=True) and bar["table"][0]["key"] == "recent"
    assert html.index("Free in budget") > html.index("Thumbnails") and "Total" in html
    assert ">0</span>" in html and ">10 GB</span>" in html and ">147 MB</span>" in html
    assert "80% alert" in html and "Phone alert when photos reach 80% of the budget (8 GB)" in html
    assert "Thinned after 90 days" in html and "own 500 MB cap" in html


def test_no_alert_label_when_warn_is_off_and_over_budget_scale():
    _, html = _render(_bd(147_000_000, 10_000_000_000, warn=0))
    assert "alert</span>" not in html
    bar, html = _render(_bd(5_000_000, 1_000_000))
    assert "sbar-end end" in html and "sc-limit at" in html and bar["budget_at"] == 20


# --- the caret sits on the bar's visible end ---

def test_caret_marker_follows_the_last_used_segment_before_the_free_part():
    bar, html = _render(_bd(147_000_000, 10_000_000_000))
    marker = html.index('class="sbar-end')
    last_used = html.rindex('<span class="sbar-seg sb-', 0, html.index('sbar-seg sb-free'))
    assert last_used < marker < html.index('sbar-seg sb-free')
    assert "--at: 1.47" in html and "sc-caret" not in html
    assert 'class="sbar-end end"' in _render(_bd(5_000_000, 1_000_000))[1]


# --- the forecast ---

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


def test_spans_and_dates_are_rounded_to_words():
    assert [webapp._span(d) for d in (1, 45, 245, 700, 800, 1000)] == [
        "1 day", "45 days", "8 months", "23 months", "2 years", "2.5 years"]
    assert webapp._span(40 * 365) == "more than 30 years"
    assert webapp._when(30, NOW) == "December 2" and webapp._when(245, NOW) == "July 2027"


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


def test_forecast_shows_on_settings_and_about_but_not_false_triggers(client, conn, add_visit, monkeypatch):
    seed(conn, add_visit, monkeypatch, budget_mb=200)
    retention.forget()
    assert 'class="sbar-forecast"' in client.get("/settings").text
    assert 'class="sbar-forecast"' in client.get("/about").text
    assert "sbar-forecast" not in client.get("/false-triggers").text
