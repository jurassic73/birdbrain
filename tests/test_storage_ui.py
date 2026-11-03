"""Settings > Storage: the usage bar, the limits form, preview and run now."""
import time

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
    assert o["budget_mb"] == 7680 and o["thin_edge"] == 960
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
