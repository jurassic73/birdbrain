"""Reference photos beside Ask-the-Pi guesses: cached card first, else a photo-only iNaturalist lookup."""
import pytest
from fastapi.testclient import TestClient

from birdbrain import db, species_info
from birdbrain.web import app as webapp, auth


@pytest.fixture
def client(conn):
    with TestClient(webapp.app) as c:
        yield c


@pytest.fixture
def inat(monkeypatch):
    """Stub the network: counts lookups; build() must never run."""
    calls = []
    monkeypatch.setattr(species_info, "find_taxon", lambda http, q: calls.append(q) or {"id": 1, "name": q})
    monkeypatch.setattr(species_info, "open_photo", lambda http, t: {
        "medium_url": f"https://inat.test/{t['name'].replace(' ', '_')}.jpg", "license_code": "cc-by",
        "attribution": "(c) Sam, some rights reserved (CC BY), uploaded by sam"})

    def boom(name):
        raise AssertionError("build() must not run for a reference photo")
    monkeypatch.setattr(species_info, "build", boom)
    return calls


def test_photo_comes_from_the_cached_card(client, conn, inat):
    db.save_species_info(conn, "Dark-eyed Junco", {"photo": {"url": "https://x/j.jpg", "credit": "A", "license": "CC BY"}},
                         "2099-01-01T00:00:00")
    r = client.get("/species-photo", params={"name": "Dark-eyed Junco"}).json()
    assert r["photo"] == {"url": "https://x/j.jpg", "credit": "A", "license": "CC BY"}
    assert inat == []


def test_photo_only_lookup_is_cached_and_never_builds_a_card(client, conn, inat, add_visit):
    add_visit(species="Red-breasted Nuthatch")
    for _ in range(2):
        p = client.get("/species-photo", params={"name": "Red-breasted Nuthatch"}).json()["photo"]
    assert p["url"] == "https://inat.test/Red-breasted_Nuthatch.jpg"
    assert p["credit"] == "(c) Sam, some rights reserved (CC BY)" and p["license"] == "CC BY"
    assert inat == ["Red-breasted Nuthatch"]  # looked up once
    assert db.get_species_info(conn, "Red-breasted Nuthatch") is None  # no card was built


def test_generic_labels_use_aliases_and_misses_are_cached(client, conn, inat, monkeypatch, add_visit):
    add_visit(visitor="raccoon", species=None)
    add_visit(species="Nobody")
    client.get("/species-photo", params={"name": "Raccoon"})
    assert inat == ["Common Raccoon"]
    monkeypatch.setattr(species_info, "find_taxon", lambda http, q: inat.append(q))  # nothing found
    assert client.get("/species-photo", params={"name": "Nobody"}).json()["photo"] is None
    client.get("/species-photo", params={"name": "Nobody"})
    assert inat.count("Nobody") == 1
    assert client.get("/species-photo", params={"name": "None"}).json()["photo"] is None


def test_lookup_failure_is_soft(client, inat, monkeypatch, add_visit):
    add_visit(species="Bushtit")

    def fail(http, q):
        raise RuntimeError("offline")
    monkeypatch.setattr(species_info, "find_taxon", fail)
    assert client.get("/species-photo", params={"name": "Bushtit"}).json() == {"photo": None}


def test_unknown_names_are_never_looked_up(client, conn, inat):
    """Only an animal the visits know gets a lookup (and a cached row): a name made up in a URL gets nothing."""
    assert client.get("/species-photo", params={"name": "Red-breasted Nuthatch"}).json() == {"photo": None}
    assert inat == [] and db.get_value(conn, "refphoto:Red-breasted Nuthatch") is None


def test_visitors_only_get_cached_photos(client, conn, inat, monkeypatch, add_visit):
    add_visit(species="Bushtit")
    monkeypatch.setenv("ADMIN_PASSWORD", "s3cret!")
    auth._fails.clear()
    assert client.get("/species-photo", params={"name": "Bushtit"}).json()["photo"] is None
    assert inat == []  # a visitor can't make the site call iNaturalist
    db.save_species_info(conn, "Bushtit", {"photo": {"url": "https://x/b.jpg"}}, "2099-01-01T00:00:00")
    assert client.get("/species-photo", params={"name": "Bushtit"}).json()["photo"]["url"] == "https://x/b.jpg"
