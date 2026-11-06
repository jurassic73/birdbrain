"""Settings > Storage: the usage bar, the limits form, preview and run now."""

import threading
import types

import pytest
from fastapi.testclient import TestClient

from helpers.storage_ui import seed
from birdbrain import retention
from birdbrain.web import app as webapp, auth


FORM = {"budget_gb": "5", "budget_custom": "", "full_days": "90", "full_per_day": "2", "thin_edge": "1280",
        "warn_pct": "80", "rare_under": "10", "disk_floor_gb": "5", "thin_on": "1"}


@pytest.fixture
def client(conn):
    retention.forget()
    with TestClient(webapp.app) as c:
        yield c
    retention.forget()


@pytest.fixture
def worker(monkeypatch):
    """Run now's background thread, so a test joins it instead of sleep-polling _run_state. The route reaches the
    thread class through the module's `threading` name, which is swapped for a copy whose Thread remembers every
    thread it starts (the module's lock was made at import; nothing else in the module uses the name)."""
    started = []

    class Tracked(threading.Thread):
        def start(self):
            started.append(self)
            super().start()

    monkeypatch.setattr(webapp, "threading", types.SimpleNamespace(Thread=Tracked, Lock=threading.Lock))

    def join() -> list:
        for t in started:
            t.join(timeout=10)
        return started

    return join


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


def test_preview_and_run_now(client, conn, add_visit, worker):
    add_visit()
    assert "Nothing to do" in client.get("/settings/storage/preview").text
    r = client.post("/settings/storage/run")
    assert r.status_code == 200
    assert len(worker()) == 1
    assert not webapp._run_state["running"] and not webapp._run_state["error"]
    assert "Done." in client.get("/settings/storage/run").text


def test_run_now_says_so_when_the_watcher_is_already_running(client, conn, add_visit, worker):
    import fcntl

    add_visit()
    with open(retention.config.DATA_DIR / retention.LOCK_FILE, "a") as watcher:  # the watcher's daily run, mid-way
        fcntl.flock(watcher, fcntl.LOCK_EX | fcntl.LOCK_NB)
        client.post("/settings/storage/run")
        assert len(worker()) == 1  # joined while the watcher still holds the lock
    assert not webapp._run_state["running"] and "already running" in webapp._run_state["error"]
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


def test_caret_marker_follows_the_last_used_segment_before_the_free_part():
    bar, html = _render(_bd(147_000_000, 10_000_000_000))
    marker = html.index('class="sbar-end')
    last_used = html.rindex('<span class="sbar-seg sb-', 0, html.index('sbar-seg sb-free'))
    assert last_used < marker < html.index('sbar-seg sb-free')
    assert "--at: 1.47" in html and "sc-caret" not in html
    assert 'class="sbar-end end"' in _render(_bd(5_000_000, 1_000_000))[1]


def test_forecast_shows_on_settings_and_about_but_not_false_triggers(client, conn, add_visit, monkeypatch):
    seed(conn, add_visit, monkeypatch, budget_mb=200)
    retention.forget()
    assert 'class="sbar-forecast"' in client.get("/settings").text
    assert 'class="sbar-forecast"' in client.get("/about").text
    assert "sbar-forecast" not in client.get("/false-triggers").text
