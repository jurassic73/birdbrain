import json

import pytest
from fastapi.testclient import TestClient

from birdbrain import config, db
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


def click(client, vid, name="Bushtit", state=None):
    data = {"visit_id": vid, **({"state": state} if state else {})}
    return client.post(f"/species/{name}/photo", data=data).json()


def test_star_cycles_favorite_cover_unset(client, conn, add_visit):
    add_visit(species="Bushtit", thumb_path="auto_t.jpg")
    pick = add_visit(species="Bushtit", thumb_path="pick_t.jpg", confidence="low")
    page = client.get("/species/Bushtit").text
    assert 'class="fav-star" data-state="unset"' in page and 'class="fav-star" data-state="cover"' not in page

    # blue first: kept, the cover untouched
    assert click(client, pick) == {"visit_id": pick, "species": "Bushtit", "state": "favorite", "cover_id": None}
    assert photo(conn, "Bushtit") == "auto_t.jpg" and db.get_visit(conn, pick)["keep"] == 1
    assert 'class="fav-star" data-state="favorite"' in client.get("/species/Bushtit").text

    # yellow: the species photo, and kept like a blue one
    assert click(client, pick) == {"visit_id": pick, "species": "Bushtit", "state": "cover", "cover_id": pick}
    assert photo(conn, "Bushtit") == "pick_t.jpg" and db.get_visit(conn, pick)["keep"] == 1
    assert client.get("/species/Bushtit").text.count('class="fav-star" data-state="cover"') == 1
    assert 'src="/images/pick_t.jpg"' in client.get("/species").text

    # unset: not the cover, no longer kept
    assert click(client, pick) == {"visit_id": pick, "species": "Bushtit", "state": "unset", "cover_id": None}
    assert not db.get_visit(conn, pick)["keep"] and db.species_photos(conn) == {}
    assert photo(conn, "Bushtit") == "auto_t.jpg"

    # a visit that isn't this species can't be starred for it
    other = add_visit(species="Dark-eyed Junco")
    assert client.post("/species/Bushtit/photo", data={"visit_id": other}).status_code == 404
    assert not db.get_visit(conn, other)["keep"]


def test_star_sets_the_state_it_is_sent(client, conn, add_visit):
    """static/star.js always says which state: yellow is held a moment, and blue -> unset skips the cover."""
    vid = add_visit(species="Bushtit")
    assert click(client, vid, state="favorite")["state"] == "favorite"
    assert click(client, vid, state="unset") == {"visit_id": vid, "species": "Bushtit", "state": "unset", "cover_id": None}
    assert db.species_photos(conn) == {} and not db.get_visit(conn, vid)["keep"]
    assert click(client, vid, state="cover")["cover_id"] == vid and db.get_visit(conn, vid)["keep"] == 1
    assert click(client, vid, state="favorite") == {"visit_id": vid, "species": "Bushtit", "state": "favorite", "cover_id": None}
    assert db.species_photos(conn) == {} and db.get_visit(conn, vid)["keep"] == 1
    assert client.post("/species/Bushtit/photo", data={"visit_id": vid, "state": "gold"}).status_code == 400


def test_only_one_cover_and_the_old_one_stays_kept(client, conn, add_visit):
    a = add_visit(species="Bushtit")
    b = add_visit(species="Bushtit")
    click(client, a, state="cover")
    r = click(client, b, state="cover")
    assert r["state"] == "cover" and r["cover_id"] == b and db.species_photos(conn) == {"Bushtit": b}
    assert db.get_visit(conn, a)["keep"] == 1  # the old cover is a blue favorite now
    text = client.get("/species/Bushtit").text
    assert text.count('class="fav-star" data-state="cover"') == 1 and text.count('class="fav-star" data-state="favorite"') == 1
    assert click(client, a)["state"] == "cover"  # blue -> yellow takes the cover back
    assert db.species_photos(conn) == {"Bushtit": a} and db.get_visit(conn, b)["keep"] == 1


def test_unstarring_one_species_keeps_a_visit_that_still_covers_another(client, conn, add_visit):
    both = add_visit(species="Dark-eyed Junco", thumb_path="both_t.jpg",
                     also=json.dumps([{"visitor": "bird", "species": "Bushtit", "count": 2, "confidence": "high"}]))
    assert click(client, both, "Bushtit", state="cover")["cover_id"] == both
    assert click(client, both, "Dark-eyed Junco", state="cover")["cover_id"] == both
    assert db.species_photos(conn) == {"Bushtit": both, "Dark-eyed Junco": both}
    r = click(client, both, "Bushtit", state="unset")
    assert (r["state"], r["cover_id"]) == ("unset", None)
    assert db.species_photos(conn) == {"Dark-eyed Junco": both}
    assert db.get_visit(conn, both)["keep"] == 1  # still the junco's cover: never thinned or deleted
    assert click(client, both, "Dark-eyed Junco", state="unset")["cover_id"] is None
    assert not db.get_visit(conn, both)["keep"] and db.species_photos(conn) == {}


def test_a_pick_is_one_transaction(conn, add_visit, monkeypatch):
    """Two quick stars (two requests, or the watcher's keep-the-covers at startup) read the picks and write them
    back: the write lock is taken before the read, so neither loses the other's cover."""
    import sqlite3

    vid = add_visit(species="Bushtit")
    other = sqlite3.connect(config.DB_PATH, timeout=0.2)
    with pytest.raises(ValueError, match="star state"):
        db.set_star(conn, "Bushtit", vid, "gold")
    assert not conn.in_transaction
    save, saved = db._save_species_photos, []

    def save_while_the_other_waits(c, photos):
        with pytest.raises(sqlite3.OperationalError, match="locked"):  # it can't get in between read and write
            other.execute("BEGIN IMMEDIATE")
        save(c, photos)
        saved.append(photos)

    monkeypatch.setattr(db, "_save_species_photos", save_while_the_other_waits)
    assert db.set_star(conn, "Bushtit", vid, "cover") == ("cover", vid)
    db.set_species_photo(conn, "Bushtit", None)
    assert saved == [{"Bushtit": vid}, {}] and not conn.in_transaction
    assert db.species_photos(conn) == {} and db.get_visit(conn, vid)["keep"] == 1
    other.close()


def test_cover_that_is_not_kept_unsets_on_the_next_click(client, conn, add_visit):
    vid = add_visit(species="Bushtit")
    db.set_species_photo(conn, "Bushtit", vid)  # old data: a cover without keep
    r = click(client, vid)
    assert r["state"] == "unset" and r["cover_id"] is None
    assert not db.get_visit(conn, vid)["keep"] and db.species_photos(conn) == {}


def test_cover_compare_gives_both_photos_with_the_species_circle(client, conn, add_visit):
    old = add_visit(species="Bushtit", image_path="old.jpg",
                    raw_json=json.dumps({"calls": {"circle": {"x": 0.25, "y": 0.5, "r": 0.05}}}))
    new = add_visit(species="Dark-eyed Junco", image_path="new.jpg",
                    also=json.dumps([{"visitor": "bird", "species": "Bushtit", "count": 1, "confidence": "high",
                                      "circle": {"x": 0.7, "y": 0.4, "r": 0.03}}]),
                    raw_json=json.dumps({"calls": {"circle": {"x": 0.1, "y": 0.1, "r": 0.1}}}))
    url = f"/species/Bushtit/cover-compare?visit_id={new}"
    assert client.get(url).json()["cover"] is None  # no cover yet: the star just sets it
    db.set_species_photo(conn, "Bushtit", old)
    r = client.get(url).json()
    assert r["species"] == "Bushtit"
    assert r["cover"]["id"] == old and r["cover"]["src"].endswith("old.jpg") and r["cover"]["mark"] == {"x": 0.25, "y": 0.5, "r": 0.05}
    # the Bushtit's circle, not the Junco's
    assert r["candidate"]["id"] == new and r["candidate"]["mark"] == {"x": 0.7, "y": 0.4, "r": 0.03}
    assert r["candidate"]["when"]
    # asking about the cover itself: nothing to compare
    assert client.get(f"/species/Bushtit/cover-compare?visit_id={old}").json()["cover"] is None
    assert client.get(f"/species/Steller's Jay/cover-compare?visit_id={new}").status_code == 404


def test_star_shows_on_other_pages_with_the_cover_yellow(client, conn, add_visit):
    cover = add_visit(species="Bushtit")
    plain = add_visit(species="Dark-eyed Junco")
    nothing = add_visit(visitor="none", species=None, confidence=None)
    db.set_species_photo(conn, "Bushtit", cover)
    for path in ("/",):
        text = client.get(path).text
        assert "class=\"fav-star\" data-state=\"cover\"" in text, path
        assert f'data-id="{cover}" data-star-species="Bushtit"' in text and f'data-id="{plain}" data-star-species="Dark-eyed Junco"' in text
        assert f'data-id="{nothing}" data-star-species' not in text
    assert 'class="fav-star"' not in client.get("/false-triggers?show=all").text
    # a star on the home page posts under the card's own species
    assert click(client, plain, "Dark-eyed Junco")["state"] == "favorite"


def test_big_latest_photo_has_the_star_too(client, conn, add_visit):
    add_visit(species="Bushtit")
    latest = add_visit(species="Dark-eyed Junco")
    db.set_species_photo(conn, "Dark-eyed Junco", latest)
    text = client.get("/").text
    hero = text[text.index('class="hero-latest"'):text.index('<aside class="today">')]
    assert f'class="fav-star" data-state="cover"' in hero and f'data-id="{latest}" data-star-species="Dark-eyed Junco"' in hero
    # the first card is the same visit and carries its own star
    assert text.count(f'data-id="{latest}" data-star-species="Dark-eyed Junco"') == 2


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


def test_the_star_is_not_a_species_link(client, add_visit):
    # species-card.js opens the species pop-up for any [data-species] click (and hover), so the star must not carry it.
    import re
    add_visit(species="Bushtit")
    stars = re.findall(r'<button type="button" class="fav-star"[^>]*>', client.get("/").text)
    assert stars and all(" data-species=" not in s for s in stars)
