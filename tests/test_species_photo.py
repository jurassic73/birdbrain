import json

import pytest
from fastapi.testclient import TestClient

from birdbrain import db
from birdbrain.web import app as webapp
from birdbrain.web import auth


@pytest.fixture
def client(conn):
    with TestClient(webapp.app) as c:
        yield c


def photo(conn, label):
    return db.species_row(conn, label)["thumb_path"]


def test_picked_photo_wins_until_the_visit_stops_showing_the_species(conn, add_visit):
    auto = add_visit(species="Bushtit", thumb_path="auto_t.jpg", corrected_species="Bushtit")
    pick = add_visit(species="Bushtit", thumb_path="pick_t.jpg", confidence="low")
    assert photo(conn, "Bushtit") == "auto_t.jpg"

    db.set_species_photo(conn, "Bushtit", pick)
    row = db.species_row(conn, "Bushtit")
    assert row["thumb_path"] == "pick_t.jpg" and row["photo_id"] == pick and row["photo_picked"]
    assert [r["thumb_path"] for r in db.species_summary(conn) if r["label"] == "Bushtit"] == ["pick_t.jpg"]

    # relabelled to another species: back to the automatic pick (and the choice doesn't leak to the new species)
    conn.execute("UPDATE visits SET corrected_species = 'Dark-eyed Junco' WHERE id = ?", (pick,))
    assert photo(conn, "Bushtit") == "auto_t.jpg" and not db.species_row(conn, "Bushtit")["photo_picked"]
    conn.execute("UPDATE visits SET corrected_species = NULL WHERE id = ?", (pick,))
    assert photo(conn, "Bushtit") == "pick_t.jpg"

    # cleared: automatic again
    db.set_species_photo(conn, "Bushtit", None)
    assert photo(conn, "Bushtit") == "auto_t.jpg" and db.species_photos(conn) == {}
    assert auto


def test_deleting_the_picked_visit_forgets_it(conn, add_visit):
    add_visit(species="Bushtit", thumb_path="auto_t.jpg")
    pick = add_visit(species="Bushtit", thumb_path="pick_t.jpg", confidence="low")
    db.set_species_photo(conn, "Bushtit", pick)
    db.delete_visit(conn, pick)
    assert photo(conn, "Bushtit") == "auto_t.jpg" and db.species_photos(conn) == {}


def test_a_visit_with_the_species_only_among_the_others_can_be_picked(conn, add_visit):
    add_visit(species="Bushtit", thumb_path="auto_t.jpg")
    extra = add_visit(species="Dark-eyed Junco", thumb_path="both_t.jpg",
                      also=json.dumps([{"visitor": "bird", "species": "Bushtit", "count": 2, "confidence": "high"}]))
    assert db.shows_species(conn, extra, "Bushtit") and not db.shows_species(conn, extra, "Steller's Jay")
    db.set_species_photo(conn, "Bushtit", extra)
    assert photo(conn, "Bushtit") == "both_t.jpg"
    assert photo(conn, "Dark-eyed Junco") == "both_t.jpg"  # its own species is untouched by the pick


def test_bad_stored_value_means_no_picks(conn, add_visit):
    add_visit(species="Bushtit")
    db.set_value(conn, db.SPECIES_PHOTO, "not json")
    assert db.species_photos(conn) == {} and db.species_row(conn, "Bushtit")["visits"] == 1


def click(client, vid, name="Bushtit"):
    return client.post(f"/species/{name}/photo", data={"visit_id": vid}).json()


def test_star_cycles_cover_favorite_unset(client, conn, add_visit):
    add_visit(species="Bushtit", thumb_path="auto_t.jpg")
    pick = add_visit(species="Bushtit", thumb_path="pick_t.jpg", confidence="low")
    page = client.get("/species/Bushtit").text
    assert 'class="fav-star" data-state="unset"' in page and 'class="fav-star" data-state="cover"' not in page

    # yellow: the species photo, and kept like a blue one
    assert click(client, pick) == {"visit_id": pick, "species": "Bushtit", "state": "cover", "cover_id": pick}
    assert photo(conn, "Bushtit") == "pick_t.jpg" and db.get_visit(conn, pick)["keep"] == 1
    assert client.get("/species/Bushtit").text.count('class="fav-star" data-state="cover"') == 1
    assert 'src="/images/pick_t.jpg"' in client.get("/species").text

    # blue: the cover pick is cleared, keep stays
    assert click(client, pick)["state"] == "favorite"
    assert photo(conn, "Bushtit") == "auto_t.jpg" and db.get_visit(conn, pick)["keep"] == 1
    assert 'class="fav-star" data-state="favorite"' in client.get("/species/Bushtit").text

    # unset: no longer kept
    assert click(client, pick) == {"visit_id": pick, "species": "Bushtit", "state": "unset", "cover_id": None}
    assert not db.get_visit(conn, pick)["keep"] and db.species_photos(conn) == {}

    # a visit that isn't this species can't be starred for it
    other = add_visit(species="Dark-eyed Junco")
    assert client.post("/species/Bushtit/photo", data={"visit_id": other}).status_code == 404
    assert not db.get_visit(conn, other)["keep"]


def test_only_one_cover_and_the_old_one_stays_kept(client, conn, add_visit):
    a = add_visit(species="Bushtit")
    b = add_visit(species="Bushtit")
    click(client, a)
    r = click(client, b)
    assert r["state"] == "cover" and r["cover_id"] == b and db.species_photos(conn) == {"Bushtit": b}
    assert db.get_visit(conn, a)["keep"] == 1  # the old cover is a blue favorite now
    text = client.get("/species/Bushtit").text
    assert text.count('class="fav-star" data-state="cover"') == 1 and text.count('class="fav-star" data-state="favorite"') == 1
    assert click(client, a)["state"] == "unset"  # blue -> unset
    assert db.species_photos(conn) == {"Bushtit": b}


def test_cover_that_is_not_kept_goes_blue_on_the_next_click(client, conn, add_visit):
    vid = add_visit(species="Bushtit")
    db.set_species_photo(conn, "Bushtit", vid)  # old data: a cover without keep
    r = click(client, vid)
    assert r["state"] == "favorite" and r["cover_id"] is None
    assert db.get_visit(conn, vid)["keep"] == 1 and db.species_photos(conn) == {}


def test_star_shows_on_other_pages_with_the_cover_yellow(client, conn, add_visit):
    cover = add_visit(species="Bushtit")
    plain = add_visit(species="Dark-eyed Junco")
    nothing = add_visit(visitor="none", species=None, confidence=None)
    db.set_species_photo(conn, "Bushtit", cover)
    for path in ("/", "/visits"):
        text = client.get(path).text
        assert "class=\"fav-star\" data-state=\"cover\"" in text, path
        assert f'data-id="{cover}" data-species="Bushtit"' in text and f'data-id="{plain}" data-species="Dark-eyed Junco"' in text
        assert f'data-id="{nothing}" data-species' not in text
    assert 'class="fav-star"' not in client.get("/false-triggers?show=all").text
    # a star on the home page posts under the card's own species
    assert click(client, plain, "Dark-eyed Junco")["state"] == "cover"


def test_visitors_get_no_star_and_cannot_pick(client, conn, add_visit, monkeypatch):
    monkeypatch.setenv("ADMIN_PASSWORD", "s3cret!")
    auth._fails.clear()
    pick = add_visit(species="Bushtit")
    assert 'class="fav-star"' not in client.get("/species/Bushtit").text
    assert client.post("/species/Bushtit/photo", data={"visit_id": pick}).status_code == 403
    assert db.species_photos(conn) == {}


def test_species_without_favorite(conn, add_visit):
    b1 = add_visit(species="Bushtit")
    add_visit(species="Bushtit")
    add_visit(species="Dark-eyed Junco")  # one photo only
    j1 = add_visit(species="Steller's Jay")
    add_visit(species="Steller's Jay")
    assert db.species_without_favorite(conn) == ["Bushtit", "Steller's Jay"]
    db.set_species_photo(conn, "Steller's Jay", j1)  # picked: not listed
    assert db.species_without_favorite(conn) == ["Bushtit"]
    conn.execute("UPDATE visits SET photo_deleted_at = '2026-01-01T00:00:00' WHERE id = ?", (b1,))
    assert db.species_without_favorite(conn) == []  # only one photo left


def test_stale_pick_counts_as_no_pick(conn, add_visit):
    add_visit(species="Bushtit")
    pick = add_visit(species="Bushtit")
    db.set_species_photo(conn, "Bushtit", pick)
    assert db.species_without_favorite(conn) == []
    conn.execute("UPDATE visits SET photo_deleted_at = '2026-01-01T00:00:00' WHERE id = ?", (pick,))
    add_visit(species="Bushtit")
    assert db.species_without_favorite(conn) == ["Bushtit"]


def test_photo_reminder_is_admin_only(client, add_visit, monkeypatch):
    add_visit(species="Bushtit")
    add_visit(species="Bushtit")
    assert client.get("/api/photo-reminder").json() == {"species": ["Bushtit"]}  # no password: everyone is admin
    monkeypatch.setenv("ADMIN_PASSWORD", "s3cret!")
    assert client.get("/api/photo-reminder").json() == {"species": []}


def test_covers_picked_before_the_keep_rule_get_kept_at_startup(conn, add_visit):
    old = add_visit(species="Bushtit", thumb_path="old_t.jpg")
    db.set_species_photo(conn, "Bushtit", old)  # picked the old way: cover only, no keep mark
    assert not db.get_visit(conn, old)["keep"]
    db.init_db(conn)
    assert db.get_visit(conn, old)["keep"] == 1
    db.init_db(conn)  # runs every start; harmless the second time
    assert db.get_visit(conn, old)["keep"] == 1
