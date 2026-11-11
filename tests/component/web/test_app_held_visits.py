"""A visit held for review (its video came through choppy): out of the gallery for everyone, in the admin's review
queue with a pill that says why, and back in the gallery once confirmed."""

import re

import pytest
from fastapi.testclient import TestClient

from birdbrain import db


@pytest.fixture
def client(conn):
    from birdbrain.web import app as webapp

    with TestClient(webapp.app) as c:
        yield c


@pytest.fixture
def held(conn, add_visit) -> tuple[int, int]:
    bird = add_visit(captured_at="2026-11-01T08:00:00")
    squirrel = add_visit(captured_at="2026-11-01T09:00:00", visitor="squirrel", species="Eastern Gray Squirrel")
    db.set_clip(conn, squirrel, None, note="the camera's stream stalled for 4.0 s while it recorded")
    db.hold_for_review(conn, squirrel)
    return bird, squirrel


def card_ids(html: str) -> list[int]:
    return [int(i) for i in re.findall(r'class="card[^"]*" href="/visit/(\d+)', html)]


def test_visitors_never_see_a_held_visit(client, held, locked):
    bird, squirrel = held
    home = client.get("/").text
    assert card_ids(home) == [bird] and f"/visit/{squirrel}" not in home and "Eastern Gray Squirrel" not in home
    assert client.get("/api/latest-visit").json()["id"] == bird
    assert client.get(f"/visit/{squirrel}").status_code == 404


def test_the_admin_finds_it_under_review_only_and_confirm_shows_it_again(client, held, locked):
    bird, squirrel = held
    assert client.post("/login", data={"password": "s3cret!"}, follow_redirects=False).status_code == 303
    home = client.get("/").text
    assert card_ids(home) == [bird] and "Eastern Gray Squirrel" not in home  # not the latest visitor either
    assert client.get("/api/latest-visit").json()["id"] == bird
    review = client.get("/review").text
    assert card_ids(review) == [squirrel] and "held for review" in review
    page = client.get(f"/visit/{squirrel}")
    assert page.status_code == 200 and "held for review" in page.text and "Video not captured" in page.text
    assert client.post(f"/visit/{squirrel}/reviewed").status_code == 200
    assert card_ids(client.get("/").text) == [squirrel, bird] and card_ids(client.get("/review").text) == []
    assert "held for review" not in client.get(f"/visit/{squirrel}").text
