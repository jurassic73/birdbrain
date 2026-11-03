import os
import tempfile
from datetime import datetime

# Before birdbrain is imported: never read the real .env (camera password, API key), and keep images in a temp dir.
os.environ["BIRDBRAIN_NO_DOTENV"] = "1"
os.environ["BIRDBRAIN_DATA"] = tempfile.mkdtemp(prefix="birdbrain-test-")
os.environ["ANTHROPIC_API_KEY"] = "test-key-never-sent"
os.environ["RTSP_PASS"] = "p@ss#word!"

import pytest  # noqa: E402

from birdbrain import config, db  # noqa: E402


@pytest.fixture
def conn(tmp_path, monkeypatch):
    """A fresh, initialized database per test (the web app picks it up through config.DB_PATH too)."""
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "test.db")
    with db.session() as c:
        db.init_db(c)
        yield c


@pytest.fixture
def add_visit(conn):
    def add(**fields) -> int:
        row = {
            "captured_at": datetime.now().replace(microsecond=0).isoformat(),
            "image_path": "img.jpg",
            "thumb_path": "img_t.jpg",
            "visitor": "bird",
            "species": "Dark-eyed Junco",
            "count": 1,
            "confidence": "high",
            "model": "claude-haiku-5-5",
        }
        row.update(fields)
        return db.insert_visit(conn, **row)

    return add


@pytest.fixture(autouse=True)
def offline_species_info(monkeypatch):
    """Species cards are built from the internet; tests that need one patch build() themselves."""
    from birdbrain import species_info
    monkeypatch.setattr(species_info, "build", lambda name: {"name": name, "links": []})
    # prefetch() builds in a background thread that can outlive the test: once this patch is undone it ran the real build()
    # (Wikipedia, iNaturalist, then Haiku with the fake key, a stray 401 after "100%" in about 1 run in 3). Tests never need it.
    monkeypatch.setattr(species_info, "prefetch", lambda name: None)
    monkeypatch.setattr(species_info, "_failed_at", {})
    monkeypatch.setattr(species_info, "_building", set())
    species_info.forget_cached_cards()


@pytest.fixture(autouse=True)
def fresh_detector_mode(monkeypatch):
    """The watcher caches the detector mode for 15 s; tests must not see one another's."""
    from birdbrain import watch
    monkeypatch.setattr(watch, "_presence_cache", (0.0, "", ""))


@pytest.fixture(autouse=True)
def synchronous_presence(monkeypatch):
    """The animal check retrains in a background thread; tests train it right away (no thread outlives a test)
    and never see another test's model."""
    from birdbrain import presence
    monkeypatch.setattr(presence, "BACKGROUND", False)
    monkeypatch.setattr(presence, "_model", None)
    monkeypatch.setattr(presence, "_thread", None)


@pytest.fixture(autouse=True)
def fresh_zones():
    """The zones in use are cached for 30 s; tests that change them in config must not see stale ones."""
    from birdbrain import zones
    zones.forget()
    yield
    zones.forget()


@pytest.fixture(autouse=True)
def offline_bird_names(monkeypatch):
    """The eBird name list is downloaded on first use; tests get a small fixed one."""
    from birdbrain import names
    monkeypatch.setattr(names, "_failed_at", 0.0)
    monkeypatch.setattr(names, "_names", ["House Finch", "Purple Finch", "Cassin's Finch", "House Sparrow",
                                          "Dark-eyed Junco", "Black-capped Chickadee", "Bushtit"])


@pytest.fixture(autouse=True)
def offline_classifier(monkeypatch):
    """The species classifier downloads its model and the place's bird list on first use; tests never do.
    With no model the watcher's fail-soft call just skips it; tests that need one pass a fake."""
    from birdbrain import classifier

    def no_network(*args, **kwargs):
        raise RuntimeError("tests never touch the network")

    monkeypatch.setattr(classifier, "_classifier", None)
    monkeypatch.setattr(classifier, "_failed_at", 0.0)
    monkeypatch.setattr(classifier, "_place_failed_at", 0.0)
    monkeypatch.setattr(classifier, "download_model", no_network)
    monkeypatch.setattr(classifier, "fetch_place_species", no_network)
