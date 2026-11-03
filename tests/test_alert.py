"""Tests for birdbrain.alert: quiet-hour logic and alert trigger rules."""

from datetime import datetime, time

import pytest

from birdbrain import alert, db


# --- _in_quiet ---

def test_in_quiet_inside():
    assert alert._in_quiet(time(10, 0), "09:00", "11:00") is True


def test_in_quiet_before():
    assert alert._in_quiet(time(8, 0), "09:00", "11:00") is False


def test_in_quiet_at_end():
    # end is exclusive
    assert alert._in_quiet(time(11, 0), "09:00", "11:00") is False


def test_in_quiet_midnight_crossing_inside():
    assert alert._in_quiet(time(23, 30), "22:00", "07:00") is True


def test_in_quiet_midnight_crossing_early_morning():
    assert alert._in_quiet(time(3, 0), "22:00", "07:00") is True


def test_in_quiet_midnight_crossing_outside():
    assert alert._in_quiet(time(12, 0), "22:00", "07:00") is False


def test_in_quiet_bad_values():
    assert alert._in_quiet(time(10, 0), "bad", "11:00") is False
    assert alert._in_quiet(time(10, 0), "09:00", "") is False


# --- reason() ---

NOW = datetime(2026, 10, 25, 14, 30, 0)


def _set_ntfy(conn, topic="test-topic", new_species="1", mammals="1", qs="", qe=""):
    db.set_value(conn, "ntfy_topic", topic)
    db.set_value(conn, "ntfy_new_species", new_species)
    db.set_value(conn, "ntfy_mammals", mammals)
    db.set_value(conn, "ntfy_quiet_start", qs)
    db.set_value(conn, "ntfy_quiet_end", qe)


def test_reason_no_topic(conn):
    assert alert.reason(conn, "raccoon", "Raccoon", NOW) is None


def test_reason_raccoon(conn):
    _set_ntfy(conn)
    r = alert.reason(conn, "raccoon", "Raccoon", NOW)
    assert r == "Raccoon at the birdbath!"


def test_reason_possum(conn):
    _set_ntfy(conn)
    r = alert.reason(conn, "possum", "Possum", NOW)
    assert r == "Possum at the birdbath!"


def test_reason_mammals_disabled(conn):
    _set_ntfy(conn, mammals="0")
    assert alert.reason(conn, "raccoon", "Raccoon", NOW) is None


def test_reason_new_species(conn):
    _set_ntfy(conn)
    # No visits yet, so this bird is new
    r = alert.reason(conn, "bird", "Dark-eyed Junco", NOW)
    assert r == "New species: Dark-eyed Junco!"


def test_reason_known_species(conn, add_visit):
    _set_ntfy(conn)
    add_visit(species="Dark-eyed Junco")
    r = alert.reason(conn, "bird", "Dark-eyed Junco", NOW)
    assert r is None


def test_reason_new_species_disabled(conn):
    _set_ntfy(conn, new_species="0")
    r = alert.reason(conn, "bird", "American Robin", NOW)
    assert r is None


def test_reason_quiet_hours(conn):
    _set_ntfy(conn, qs="14:00", qe="15:00")  # 14:30 is inside
    assert alert.reason(conn, "raccoon", "Raccoon", NOW) is None


def test_reason_outside_quiet_hours(conn):
    _set_ntfy(conn, qs="20:00", qe="22:00")  # 14:30 is outside
    r = alert.reason(conn, "raccoon", "Raccoon", NOW)
    assert r is not None


def test_reason_no_alert_for_none_visitor(conn):
    _set_ntfy(conn)
    assert alert.reason(conn, "none", "None", NOW) is None


# --- get_settings / save_settings ---

def test_get_settings_defaults(conn):
    s = alert.get_settings(conn)
    assert s["topic"] == ""
    assert s["new_species"] is True   # default on
    assert s["mammals"] is True       # default on
    assert s["quiet_start"] == ""
    assert s["quiet_end"] == ""


def test_save_and_get_settings(conn):
    alert.save_settings(conn, topic="my-topic", new_species=True, mammals=False,
                        quiet_start="22:00", quiet_end="07:00")
    s = alert.get_settings(conn)
    assert s["topic"] == "my-topic"
    assert s["new_species"] is True
    assert s["mammals"] is False
    assert s["quiet_start"] == "22:00"
    assert s["quiet_end"] == "07:00"


def _card(conn, name, expected):
    db.save_species_info(conn, name, {"name": name, "local_counts": [10] * 12, "expected_months": expected},
                         datetime.now().isoformat())


def test_reason_out_of_season(conn, add_visit):
    _set_ntfy(conn)
    add_visit(species="Rufous Hummingbird")  # seen before, so not a new species
    _card(conn, "Rufous Hummingbird", [4, 5, 6, 7, 8])
    march = datetime(2027, 3, 20, 9, 0)
    assert alert.reason(conn, "bird", "Rufous Hummingbird", march) == "Early arrival: Rufous Hummingbird"
    july = datetime(2027, 7, 20, 9, 0)
    assert alert.reason(conn, "bird", "Rufous Hummingbird", july) is None


def test_reason_out_of_season_toggle_off(conn, add_visit):
    _set_ntfy(conn)
    db.set_value(conn, "ntfy_unusual", "0")
    add_visit(species="Rufous Hummingbird")
    _card(conn, "Rufous Hummingbird", [4, 5, 6, 7, 8])
    assert alert.reason(conn, "bird", "Rufous Hummingbird", datetime(2027, 12, 1, 9, 0)) is None


def test_reason_new_species_beats_out_of_season(conn):
    _set_ntfy(conn)
    _card(conn, "Rufous Hummingbird", [4, 5, 6, 7, 8])
    r = alert.reason(conn, "bird", "Rufous Hummingbird", datetime(2027, 3, 20, 9, 0))
    assert r == "New species: Rufous Hummingbird!"


def test_rarely_reported_does_not_alert(conn, add_visit):
    _set_ntfy(conn)
    add_visit(species="Mystery Warbler")
    _card(conn, "Mystery Warbler", [])
    assert alert.reason(conn, "bird", "Mystery Warbler", NOW) is None


def test_settings_page_saves_unusual_toggle(conn):
    from fastapi.testclient import TestClient
    from birdbrain.web import app as webapp
    with TestClient(webapp.app) as c:
        c.post("/settings/alerts", data={"ntfy_topic": "t", "ntfy_new_species": "1"})
        s = alert.get_settings(conn)
        assert s["unusual"] is False and s["new_species"] is True and s["mammals"] is False
        assert "Out of season" in c.get("/settings").text
