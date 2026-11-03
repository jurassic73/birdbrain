import pytest
from fastapi.testclient import TestClient

from conftest import FROZEN_NOW
from birdbrain import db, species_info
from birdbrain.web import app as webapp
from birdbrain.web import auth


@pytest.fixture
def client(conn):
    with TestClient(webapp.app) as c:
        yield c


@pytest.fixture
def locked(monkeypatch):
    """Read-only mode: an admin password is set."""
    monkeypatch.setenv("ADMIN_PASSWORD", "s3cret!")
    auth._fails.clear()


def test_visitors_browse_but_change_nothing(client, add_visit, conn, locked):
    vid = add_visit(visitor="bird", species="Dark-eyed Junco")
    page = client.get(f"/visit/{vid}")
    assert page.status_code == 200 and "Dark-eyed Junco" in page.text
    assert "/correct" not in page.text and "Delete visit" not in page.text and 'href="/settings"' not in page.text
    assert 'href="/login' in page.text
    for path in (f"/visit/{vid}/correct", f"/visit/{vid}/delete", f"/visit/{vid}/check-circle/pi",
                 f"/visit/{vid}/check-circle/haiku", "/notes/refine", "/settings/alerts", "/settings/mirror",
                 "/settings/zones/manual"):
        assert client.post(path).status_code == 403
    assert db.get_visit(conn, vid) is not None
    for path in ("/settings", "/notes", "/review", "/false-triggers", "/false-triggers/cards?after=0",
                 "/settings/camera", "/lessons"):
        r = client.get(path, follow_redirects=False)
        assert r.status_code == 303 and r.headers["location"].startswith("/login")
    for path in ("/", "/species", "/stats", "/about"):
        assert client.get(path).status_code == 200


def test_visitors_species_card_never_fetches(client, locked, monkeypatch):
    seen = []
    monkeypatch.setattr(species_info, "get", lambda conn, name, fetch=True: seen.append(fetch))
    client.get("/species-info/Bushtit")
    assert seen == [False]


def test_admin_login_and_logout(client, add_visit, locked):
    vid = add_visit(visitor="bird", species="Bushtit")
    assert client.post("/login", data={"password": "wrong"}).status_code == 401
    r = client.post("/login", data={"password": "s3cret!", "next": "//evil.example"}, follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/"
    assert "httponly" in r.headers["set-cookie"].lower()
    assert client.get("/settings").status_code == 200
    assert f'hx-post="/visit/{vid}/correct"' in client.get(f"/visit/{vid}").text
    client.post("/logout")
    assert client.get("/settings", follow_redirects=False).status_code == 303


def test_changing_the_password_signs_everyone_out(client, locked, monkeypatch):
    client.post("/login", data={"password": "s3cret!"})
    monkeypatch.setenv("ADMIN_PASSWORD", "new-one")
    assert client.get("/settings", follow_redirects=False).status_code == 303


def test_login_locks_out_after_repeated_wrong_passwords(client, locked):
    for _ in range(5):
        client.post("/login", data={"password": "nope"})
    assert client.post("/login", data={"password": "s3cret!"}).status_code == 429


def test_no_admin_password_means_everyone_is_admin(client):
    assert client.get("/settings").status_code == 200
    assert "Log out" not in client.get("/").text and 'href="/login' not in client.get("/").text


def test_public_mirror_has_no_admin_and_never_touches_camera(client, add_visit, conn, monkeypatch):
    monkeypatch.setenv("BIRDBRAIN_PUBLIC", "1")
    monkeypatch.setenv("ADMIN_PASSWORD", "s3cret!")  # even with a password set, nobody can log in
    async def no_camera():
        raise AssertionError("public mirror contacted the camera")
    monkeypatch.setattr(webapp, "grab_live_frame", no_camera)
    vid = add_visit(visitor="bird", species="Dark-eyed Junco")
    page = client.get(f"/visit/{vid}")
    assert page.status_code == 200 and "Dark-eyed Junco" in page.text
    assert 'href="/login' not in page.text and "Log out" not in page.text and "/correct" not in page.text
    assert client.get("/login").status_code == 404
    assert client.post("/login", data={"password": "s3cret!"}).status_code == 403
    assert client.post(f"/visit/{vid}/delete").status_code == 403
    for path in ("/settings", "/notes", "/review", "/false-triggers", "/settings/camera", "/lessons"):
        assert client.get(path, follow_redirects=False).status_code == 404
    for path in ("/", "/species", "/stats", "/about"):
        assert client.get(path).status_code == 200
    assert client.get("/live.jpg").status_code == 503  # no synced frame; must not reach the camera either way
    assert db.get_visit(conn, vid) is not None


def test_base_path_prefixes_links_redirects_and_fetch(client, add_visit, monkeypatch):
    monkeypatch.setattr(webapp, "BASE", "/birdbrain")
    vid = add_visit(visitor="bird", species="Dark-eyed Junco")
    page = client.get("/").text
    assert f'href="/birdbrain/visit/{vid}"' in page and 'href="/visit/' not in page
    assert 'src="/birdbrain/static/' in page and 'src="/static/' not in page
    assert "window.BB_BASE=\"/birdbrain\"" in page
    r = client.post("/logout", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/birdbrain/"


def test_mirror_today_follows_site_timezone(client, add_visit, conn, monkeypatch):
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo
    from birdbrain import clock
    # A zone whose date differs from this machine's for most of the day, so "today" visibly moves.
    def in_zone(name):  # the frozen "now" as the wall clock in that zone
        return FROZEN_NOW.astimezone().astimezone(ZoneInfo(name)).replace(tzinfo=None)
    zone = "Pacific/Kiritimati" if in_zone("Pacific/Kiritimati").date() != FROZEN_NOW.date() else "Etc/GMT+12"
    there = in_zone(zone)
    assert there.date() != FROZEN_NOW.date()
    try:
        db.set_value(conn, clock.KEY, zone)
        clock.set_zone(None); clock._zone["checked"] = 0.0  # make the next request read the setting
        assert client.get("/").status_code == 200
        assert clock.today() == there.date()
        db.set_value(conn, clock.KEY, "")
        clock._zone["checked"] = 0.0
        assert client.get("/").status_code == 200
        assert clock.today() == FROZEN_NOW.date()
    finally:
        clock.set_zone(None); clock._zone["checked"] = 0.0


def test_somegeek_home_logo_only_on_public_mirror(client, monkeypatch):
    assert "somegeek.com" not in client.get("/").text
    monkeypatch.setenv("BIRDBRAIN_PUBLIC", "1")
    monkeypatch.setattr(webapp, "BASE", "/birdbrain")
    page = client.get("/").text
    assert 'href="https://somegeek.com/"' in page and 'aria-label="somegeek.com home"' in page
    assert 'src="/birdbrain/static/somegeek-logo.png' in page


def test_latest_visit_api_waiting_counts_only_for_admin(client, add_visit, locked, monkeypatch):
    add_visit(needs_review=1)
    add_visit(visitor="none", species=None)
    add_visit(visitor="none", species=None)
    zero = {"review": 0, "false": 0}
    assert client.get("/api/latest-visit").json()["waiting"] == zero  # visitor
    client.post("/login", data={"password": "s3cret!"})
    assert client.get("/api/latest-visit").json()["waiting"] == {"review": 1, "false": 2}
    monkeypatch.setenv("BIRDBRAIN_PUBLIC", "1")  # the public mirror never shows counts
    assert client.get("/api/latest-visit").json()["waiting"] == zero


def test_false_trigger_does_not_start_drawing_for_visitors(client, add_visit, locked):
    ft = add_visit(visitor="none", species=None)
    assert "if (false) setEditing(true);" in client.get(f"/visit/{ft}?ft=1").text
