"""Component tests of birdbrain/db.py (species photo): one real local dependency, a temp database or files on disk."""

import json

import pytest

from helpers.species_photo import photo
from birdbrain import config, db


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


def test_a_pick_is_one_transaction(conn, add_visit, monkeypatch):
    """Two quick stars (two requests, or the watcher's keep-the-covers at startup) read the picks and write them
    back: the write lock is taken before the read, so neither loses the other's cover."""
    import sqlite3

    vid = add_visit(species="Bushtit")
    other = sqlite3.connect(config.DB_PATH, timeout=0)  # no wait: the lock is either held or it is not
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


def test_covers_picked_before_the_keep_rule_get_kept_at_startup(conn, add_visit):
    old = add_visit(species="Bushtit", thumb_path="old_t.jpg")
    db.set_species_photo(conn, "Bushtit", old)  # picked the old way: cover only, no keep mark
    assert not db.get_visit(conn, old)["keep"]
    db.init_db(conn)
    assert db.get_visit(conn, old)["keep"] == 1
    db.init_db(conn)  # runs every start; harmless the second time
    assert db.get_visit(conn, old)["keep"] == 1
