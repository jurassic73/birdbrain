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


def test_star_route_toggles(client, conn, add_visit):
    add_visit(species="Bushtit", thumb_path="auto_t.jpg")
    pick = add_visit(species="Bushtit", thumb_path="pick_t.jpg", confidence="low")
    page = client.get("/species/Bushtit").text
    assert 'class="fav-star"' in page and 'aria-pressed="true"' not in page

    r = client.post("/species/Bushtit/photo", data={"visit_id": pick})
    assert r.status_code == 200 and r.json() == {"visit_id": pick}
    assert photo(conn, "Bushtit") == "pick_t.jpg"
    assert client.get("/species/Bushtit").text.count('aria-pressed="true"') == 1
    assert 'src="/images/pick_t.jpg"' in client.get("/species").text

    assert client.post("/species/Bushtit/photo", data={"visit_id": pick}).json() == {"visit_id": None}
    assert photo(conn, "Bushtit") == "auto_t.jpg"

    # a visit that isn't this species can't be picked for it
    other = add_visit(species="Dark-eyed Junco")
    assert client.post("/species/Bushtit/photo", data={"visit_id": other}).status_code == 404


def test_visitors_get_no_star_and_cannot_pick(client, conn, add_visit, monkeypatch):
    monkeypatch.setenv("ADMIN_PASSWORD", "s3cret!")
    auth._fails.clear()
    pick = add_visit(species="Bushtit")
    assert 'class="fav-star"' not in client.get("/species/Bushtit").text
    assert client.post("/species/Bushtit/photo", data={"visit_id": pick}).status_code == 403
    assert db.species_photos(conn) == {}
