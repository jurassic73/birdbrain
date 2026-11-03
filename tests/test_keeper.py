import re
import sqlite3

import pytest
from fastapi.testclient import TestClient

from birdbrain import db
from birdbrain.web import app as webapp
from birdbrain.web import auth


@pytest.fixture
def client(conn):
    with TestClient(webapp.app) as c:
        yield c


@pytest.fixture
def locked(monkeypatch):
    monkeypatch.setenv("ADMIN_PASSWORD", "s3cret!")
    auth._fails.clear()


def test_init_db_adds_keep_to_an_old_database():
    old = sqlite3.connect(":memory:")
    old.row_factory = sqlite3.Row
    old.execute("CREATE TABLE visits (id INTEGER PRIMARY KEY, captured_at TEXT NOT NULL)")
    db.init_db(old)
    assert "keep" in {r["name"] for r in old.execute("PRAGMA table_info(visits)")}


def test_set_keep_toggles(conn, add_visit):
    vid = add_visit()
    assert not db.get_visit(conn, vid)["keep"]
    db.set_keep(conn, vid, True)
    assert db.get_visit(conn, vid)["keep"] == 1
    db.set_keep(conn, vid, False)
    assert not db.get_visit(conn, vid)["keep"]


def test_keep_route_returns_the_updated_button(client, conn, add_visit):
    vid = add_visit()
    r = client.post(f"/visit/{vid}/keep", data={"on": 1})
    assert r.status_code == 200 and 'aria-pressed="true"' in r.text and '"on": 0' in r.text
    assert db.get_visit(conn, vid)["keep"] == 1
    r = client.post(f"/visit/{vid}/keep", data={"on": 0})
    assert 'aria-pressed="false"' in r.text and not db.get_visit(conn, vid)["keep"]
    assert client.post("/visit/9999/keep", data={"on": 1}).status_code == 404


def test_visitors_cannot_keep_or_see_the_mark(client, conn, add_visit, locked):
    vid = add_visit()
    db.set_keep(conn, vid, True)
    assert client.post(f"/visit/{vid}/keep", data={"on": 0}).status_code == 403
    assert db.get_visit(conn, vid)["keep"] == 1
    for path in ("/", f"/visit/{vid}", "/visits"):
        text = client.get(path).text
        assert "fav-star" not in text and "Favorite</button>" not in text, path


def test_admin_sees_the_star_on_cards_and_the_visit_photo(client, conn, add_visit):
    # Cards carry one cycling star (static/star.js); the old separate keep-mark button is gone from cards.
    vid = add_visit()
    for path in ("/", "/visits", "/species/Dark-eyed%20Junco"):
        text = client.get(path).text
        assert 'class="fav-star"' in text and 'class="keep-mark"' not in text, path
    page = client.get(f"/visit/{vid}").text
    assert page.count('class="fav-star"') == 1 and 'data-star-species="Dark-eyed Junco"' in page
    assert "Favorite</button>" not in page and 'class="badge keeper"' not in page and "keep-btn" not in page
    db.set_keep(conn, vid, True)
    page = client.get(f"/visit/{vid}").text
    assert 'class="fav-star" data-state="favorite"' in page and 'class="badge keeper"' not in page
    assert 'data-state="favorite"' in client.get("/visits").text


def test_admin_login_sees_the_star_when_locked(client, add_visit, locked):
    add_visit()
    client.post("/login", data={"password": "s3cret!"})
    assert 'class="fav-star"' in client.get("/").text


def test_visit_star_without_a_species_only_toggles_favorite(client, conn, add_visit):
    vid = add_visit(species="Other", visitor="other")
    star = re.search(r'<button type="button" class="fav-star"[^>]*>', client.get(f"/visit/{vid}").text).group(0)
    assert "data-star-species" not in star and "data-tip-cover" not in star
