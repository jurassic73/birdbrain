import pytest
from fastapi.testclient import TestClient

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
    for path in (f"/visit/{vid}/correct", f"/visit/{vid}/delete", "/notes/refine", "/settings/alerts", "/notes/scene/outline"):
        assert client.post(path).status_code == 403
    assert db.get_visit(conn, vid) is not None
    for path in ("/settings", "/notes", "/review", "/false-triggers", "/settings/camera", "/lessons"):
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
