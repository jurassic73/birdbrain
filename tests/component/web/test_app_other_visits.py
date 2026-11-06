""""Other" visits (an animal Claude couldn't put in a visitor type, or its placeholder when it couldn't answer):
they always wait for review, come first in the review queue, and are the admin's only, like false triggers."""

import re

import pytest
from fastapi.testclient import TestClient

from helpers.other_visits import TODAY, other, yard
from birdbrain import config, db


@pytest.fixture
def client(conn):
    from birdbrain.web import app as webapp

    with TestClient(webapp.app) as c:
        yield c


def card_ids(html: str) -> list[int]:
    return [int(i) for i in re.findall(r'class="card[^"]*" href="/visit/(\d+)', html)]


def test_visitors_never_see_an_other_visit(client, yard, locked):
    bird, cat = yard
    home = client.get("/").text
    assert card_ids(home) == [bird] and f"/visit/{cat}" not in home and "Domestic Cat" not in home
    assert client.get(f"/visits/more?before={TODAY}T23:59:59,999").text.count(f"/visit/{cat}") == 0
    live = client.get("/live").text
    assert f"/visit/{bird}" in live and f"/visit/{cat}" not in live
    api = client.get("/api/latest-visit").json()
    assert (api["id"], api["any_id"]) == (bird, bird)
    assert client.get(f"/visit/{cat}").status_code == 404
    assert f"/visit/{cat}" not in client.get(f"/visit/{bird}").text  # no arrow to it
    assert "Domestic Cat" not in client.get("/species").text
    assert client.get("/species/Domestic Cat").status_code == 404
    assert "All time: 1 visit since " in client.get("/stats").text


def test_the_admin_sees_other_visits_everywhere(client, yard, locked):
    bird, cat = yard
    assert client.post("/login", data={"password": "s3cret!"}, follow_redirects=False).status_code == 303
    assert card_ids(client.get("/").text) == [cat, bird]
    assert client.get("/api/latest-visit").json()["id"] == cat
    assert client.get(f"/visit/{cat}").status_code == 200
    assert f"/visit/{cat}" in client.get(f"/visit/{bird}").text
    assert "All time: 2 visits since " in client.get("/stats").text


def test_the_admins_review_page_starts_with_the_other_visits(client, conn, add_visit, locked):
    unsure = add_visit(captured_at="2026-10-01T08:00:00", confidence="low", needs_review=1)
    cat = other(add_visit, needs_review=1)
    assert client.post("/login", data={"password": "s3cret!"}, follow_redirects=False).status_code == 303
    assert card_ids(client.get("/review").text) == [cat, unsure]


def test_no_password_means_everyone_sees_other_visits(client, yard):
    bird, cat = yard
    assert client.get(f"/visit/{cat}").status_code == 200 and card_ids(client.get("/").text) == [cat, bird]


def test_relabelled_to_an_animal_is_public_again_confirmed_other_stays_hidden(client, conn, yard, locked):
    bird, cat = yard
    db.mark_reviewed(conn, cat)  # Confirm: it really was something nobody can name
    assert client.get(f"/visit/{cat}").status_code == 404
    db.correct_visit(conn, cat, "squirrel", None, None)
    assert client.get(f"/visit/{cat}").status_code == 200 and card_ids(client.get("/").text) == [cat, bird]
    db.correct_visit(conn, bird, "other", None, None)  # the owner's own "other" is hidden too
    assert client.get(f"/visit/{bird}").status_code == 404


def test_the_public_mirror_has_no_other_pages(client, yard, monkeypatch):
    monkeypatch.setenv("BIRDBRAIN_PUBLIC", "1")
    bird, cat = yard
    assert client.get(f"/visit/{cat}").status_code == 404 and card_ids(client.get("/").text) == [bird]


def test_visitors_get_404_for_an_other_visits_burst_frames(client, add_visit, locked):
    day = config.IMAGES_DIR / "2026" / "11" / "01"
    day.mkdir(parents=True, exist_ok=True)
    names = ("cat.jpg", "cat_t.jpg", "cat_b1.jpg", "cat_b1_t.jpg", "real.jpg", "real_b1.jpg")
    for name in names:
        (day / name).write_bytes(b"jpeg")
    try:
        other(add_visit)
        add_visit(image_path="2026/11/01/real.jpg", thumb_path="2026/11/01/real_t.jpg")
        for url in ("/images/2026/11/01/cat_b1.jpg", "/images/2026/11/01/cat_b1_t.jpg"):
            assert client.get(url).status_code == 404, url
        assert client.get("/images/2026/11/01/real_b1.jpg").status_code == 200
        assert client.post("/login", data={"password": "s3cret!"}, follow_redirects=False).status_code == 303
        assert client.get("/images/2026/11/01/cat_b1.jpg").status_code == 200
    finally:
        for name in names:
            (day / name).unlink(missing_ok=True)


def test_also_in_the_photo_other_entries_stay_on_a_public_visit(client, add_visit, locked):
    """Only the main animal decides: an "other" among a public visit's other animals is still shown (for now)."""
    vid = add_visit(species="Bushtit", also='[{"visitor": "other", "species": null, "count": 1}]')
    assert client.get(f"/visit/{vid}").status_code == 200 and card_ids(client.get("/").text) == [vid]
