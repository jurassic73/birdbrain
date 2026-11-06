"""Component tests of the web app through TestClient on a temp database: keeper."""

import re

from birdbrain import db


def test_keep_route_sets_and_clears_the_mark(client, conn, add_visit):
    vid = add_visit()
    assert client.post(f"/visit/{vid}/keep", data={"on": 1}).status_code == 204
    assert db.get_visit(conn, vid)["keep"] == 1
    assert client.post(f"/visit/{vid}/keep", data={"on": 0}).status_code == 204
    assert not db.get_visit(conn, vid)["keep"]
    assert client.post("/visit/9999/keep", data={"on": 1}).status_code == 404


def test_visitors_cannot_keep_or_see_the_mark(client, conn, add_visit, locked):
    vid = add_visit()
    db.set_keep(conn, vid, True)
    assert client.post(f"/visit/{vid}/keep", data={"on": 0}).status_code == 403
    assert db.get_visit(conn, vid)["keep"] == 1
    for path in ("/", f"/visit/{vid}"):
        text = client.get(path).text
        assert "fav-star" not in text and "Favorite</button>" not in text, path


def test_admin_sees_the_star_on_cards_and_the_visit_photo(client, conn, add_visit):
    # Cards carry one cycling star (static/star.js); the old separate keep-mark button is gone from cards.
    vid = add_visit()
    for path in ("/", "/species/Dark-eyed%20Junco"):
        text = client.get(path).text
        assert 'class="fav-star"' in text and 'class="keep-mark"' not in text, path
    page = client.get(f"/visit/{vid}").text
    assert page.count('class="fav-star"') == 1 and 'data-star-species="Dark-eyed Junco"' in page
    assert "Favorite</button>" not in page and 'class="badge keeper"' not in page and "keep-btn" not in page
    db.set_keep(conn, vid, True)
    page = client.get(f"/visit/{vid}").text
    assert 'class="fav-star" data-state="favorite"' in page and 'class="badge keeper"' not in page
    assert 'data-state="favorite"' in client.get("/").text


def test_admin_login_sees_the_star_when_locked(client, add_visit, locked):
    add_visit()
    client.post("/login", data={"password": "s3cret!"})
    assert 'class="fav-star"' in client.get("/").text


def test_visit_star_without_a_species_only_toggles_favorite(client, conn, add_visit):
    vid = add_visit(species="Other", visitor="other")
    star = re.search(r'<button type="button" class="fav-star"[^>]*>', client.get(f"/visit/{vid}").text).group(0)
    assert "data-star-species" not in star and "data-tip-cover" not in star
