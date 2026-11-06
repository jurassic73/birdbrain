"""Outside data fails in awkward ways: partial answers, truncated downloads, odd types. None of it may touch the network."""

import time
from datetime import datetime, timedelta

import httpx
import pytest
from fastapi.testclient import TestClient

from helpers.cards import CARD
from birdbrain import config, db, names, species_info
from birdbrain.web import app as webapp


def _stale(conn, data=CARD):
    old = (datetime.now() - timedelta(days=45)).replace(microsecond=0).isoformat()
    db.save_species_info(conn, "Bushtit", data, old)
    return old


def test_partial_rebuild_keeps_old_fields_and_retries_later(conn, monkeypatch):
    old = _stale(conn)
    partial = {"name": "Bushtit", "common": "Bushtit", "inat_id": 14886, "is_bird": True, "status": "Summer visitor",
               "links": []}  # Wikipedia, Haiku and the photo all failed
    monkeypatch.setattr(species_info, "build", lambda name: partial)
    card = species_info.get(conn, "Bushtit")
    assert card["description"] == CARD["description"] and card["notes"] == CARD["notes"]
    assert card["photo"] == CARD["photo"] and card["map_center"] == CARD["map_center"]
    assert card["status"] == "Summer visitor"  # what the rebuild did get is used
    saved, stamp = db.get_species_info(conn, "Bushtit")
    assert saved["description"] == CARD["description"]
    assert stamp == old  # still stale: rebuilt again later


def test_complete_rebuild_refreshes_timestamp(conn, monkeypatch):
    old = _stale(conn)
    monkeypatch.setattr(species_info, "build", lambda name: {**CARD, "description": "New text."})
    assert species_info.get(conn, "Bushtit")["description"] == "New text."
    assert db.get_species_info(conn, "Bushtit")[1] != old


def test_failed_fetch_backs_off(conn, monkeypatch):
    _stale(conn)
    calls = []

    def build(name):
        calls.append(name)
        raise RuntimeError("down")
    monkeypatch.setattr(species_info, "build", build)
    assert species_info.get(conn, "Bushtit")["common"] == "Bushtit"
    species_info.get(conn, "Bushtit")
    assert calls == ["Bushtit"]
    species_info._failed_at["Bushtit"] = time.monotonic() - species_info.RETRY_AFTER - 1
    species_info.get(conn, "Bushtit")
    assert len(calls) == 2


def test_concurrent_build_is_skipped(conn, monkeypatch):
    _stale(conn)
    calls = []
    monkeypatch.setattr(species_info, "build", lambda name: calls.append(name) or CARD)
    species_info._building.add("Bushtit")
    assert species_info.get(conn, "Bushtit")["common"] == "Bushtit"  # old card, no second build
    assert calls == []
    species_info._building.clear()
    species_info.get(conn, "Bushtit")
    assert calls == ["Bushtit"] and not species_info._building


def test_mini_never_fetches(conn, monkeypatch, add_visit):
    _stale(conn)
    add_visit(species="Bushtit")  # cards are only built for animals the visits know
    calls = []
    monkeypatch.setattr(species_info, "build", lambda name: calls.append(name) or CARD)
    with TestClient(webapp.app) as c:
        assert c.get("/species-info/Bushtit?mini=1").status_code == 200
        assert c.get("/species-info/Mystery%20Wren?mini=1").status_code == 200
        assert calls == []
        c.get("/species-info/Bushtit")  # the full card (for the owner) may rebuild
        assert calls == ["Bushtit"]


class _Taxonomy:
    """Counts downloads; `body` is what httpx.get returns."""

    def __init__(self, fail=False):
        self.calls, self.fail = 0, fail

    def get(self, url, **kw):
        self.calls += 1
        if self.fail:
            raise httpx.ConnectError("offline")
        return httpx.Response(200, json=[{"comName": "House Finch", "speciesCode": "houfin"}],
                              request=httpx.Request("GET", url))


@pytest.fixture
def real_names(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(names, "_names", None)
    return tmp_path


def test_names_failure_retried_at_most_hourly(real_names, monkeypatch):
    fake = _Taxonomy(fail=True)
    monkeypatch.setattr(httpx, "get", fake.get)
    assert names.bird_names() == [] and names.bird_names() == [] and names.search("fin", []) == []
    assert fake.calls == 1
    names._failed_at = time.monotonic() - 3601
    names.bird_names()
    assert fake.calls == 2
    assert not list(real_names.glob("*.part")) and not (real_names / "ebird_names.json").exists()


def test_names_truncated_file_is_missing(real_names, monkeypatch):
    (real_names / "ebird_names.json").write_text('["House Fin')
    fake = _Taxonomy()
    monkeypatch.setattr(httpx, "get", fake.get)
    assert names.bird_names() == ["House Finch"]  # re-downloaded
    assert fake.calls == 1
    assert (real_names / "ebird_names.json").read_text() == '["House Finch"]'


def test_names_truncated_file_and_offline_retries_later(real_names, monkeypatch):
    (real_names / "ebird_names.json").write_text("[")
    fake = _Taxonomy(fail=True)
    monkeypatch.setattr(httpx, "get", fake.get)
    assert names.bird_names() == [] and names.bird_names() == []
    assert fake.calls == 1


def test_atomic_write_leaves_no_partial_file(tmp_path, monkeypatch):
    path = tmp_path / "x.json"
    species_info.write_json_atomic(path, ["a"])
    assert path.read_text() == '["a"]'

    def boom(src, dst):
        raise OSError("disk full")
    monkeypatch.setattr(species_info.os, "replace", boom)
    with pytest.raises(OSError):
        species_info.write_json_atomic(path, ["b"])
    assert path.read_text() == '["a"]' and list(tmp_path.iterdir()) == [path]


def test_ebird_code_download_atomic_and_throttled(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(500)
        return httpx.Response(200, json=[{"comName": "House Finch", "speciesCode": "houfin"}])
    client = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(httpx.HTTPStatusError):
        species_info.ebird_code(client, "House Finch")
    assert species_info.ebird_code(client, "House Finch") is None and len(calls) == 1  # within the hour
    species_info._failed_at["ebird_codes"] = time.monotonic() - 3601
    assert species_info.ebird_code(client, "House Finch") == "houfin"
    (tmp_path / "ebird_codes.json").write_text("{")  # truncated: treated as missing
    assert species_info.ebird_code(client, "House Finch") == "houfin" and len(calls) == 3
    assert not list(tmp_path.glob("*.part"))
