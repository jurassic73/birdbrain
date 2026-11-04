import os
import socket
import sys
import tempfile
from datetime import date as _real_date, datetime as _real_datetime
from pathlib import Path

# Before birdbrain is imported: never read the real .env (camera password, API key), and keep images in a temp dir.
os.environ["BIRDBRAIN_NO_DOTENV"] = "1"
os.environ["BIRDBRAIN_DATA"] = tempfile.mkdtemp(prefix="birdbrain-test-")
os.environ["ANTHROPIC_API_KEY"] = "test-key-never-sent"
os.environ["RTSP_PASS"] = "p@ss#word!"

import pytest  # noqa: E402

from birdbrain import config, db  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent

# "Now" for every test: midday on the day the run started, so a run that crosses midnight (or starts just past it)
# sees one fixed day. Tests that need a time relative to now use the `now` fixture or `from conftest import FROZEN_NOW`.
FROZEN_NOW = _real_datetime.now().replace(hour=12, minute=0, second=0, microsecond=0)


class _Meta(type):
    def __instancecheck__(cls, obj):
        return isinstance(obj, cls.__mro__[1])


class FrozenDatetime(_real_datetime, metaclass=_Meta):
    @classmethod
    def now(cls, tz=None):
        if tz is None:
            return FROZEN_NOW
        return FROZEN_NOW.astimezone().astimezone(tz)

    @classmethod
    def today(cls):
        return FROZEN_NOW


class FrozenDate(_real_date, metaclass=_Meta):
    @classmethod
    def today(cls):
        return FROZEN_NOW.date()


@pytest.fixture(autouse=True)
def frozen_clock(monkeypatch):
    """Freeze "now" at FROZEN_NOW in every birdbrain module that imported datetime/date by name.
    (time.time()/monotonic() are left alone: they only drive cache ages and staleness.)"""
    import birdbrain.web.app  # noqa: F401  (make sure every module is loaded)
    for name, mod in list(sys.modules.items()):
        if not name.startswith("birdbrain"):
            continue
        if getattr(mod, "datetime", None) is _real_datetime:
            monkeypatch.setattr(mod, "datetime", FrozenDatetime)
        if getattr(mod, "date", None) is _real_date:
            monkeypatch.setattr(mod, "date", FrozenDate)


def image_size(path) -> tuple[int, int]:
    """An image file's (width, height), closing the file again."""
    from PIL import Image
    with Image.open(path) as im:
        return im.size


@pytest.fixture
def now():
    return FROZEN_NOW


@pytest.fixture(scope="session", autouse=True)
def repo_root_untouched():
    """Fail the run if any test leaves files in the repo root (images/, birdbrain.db, ...)."""
    def listing():
        skip = {".git", ".venv", ".pytest_cache", "__pycache__", ".ruff_cache"}
        return {p.name for p in REPO_ROOT.iterdir() if p.name not in skip}
    before = listing()
    yield
    created = sorted(listing() - before)
    assert not created, f"tests created files in the repo root: {created}"


_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "testserver", ""}


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """A test that reaches for a real host fails loudly instead of going out; localhost stays open."""
    real_connect, real_connect_ex = socket.socket.connect, socket.socket.connect_ex

    def check(address):
        if isinstance(address, (tuple, list)) and address:
            host = address[0]
            if isinstance(host, bytes):
                host = host.decode()
            if host not in _LOCAL_HOSTS and not str(host).startswith("127."):
                raise RuntimeError(f"tests must not touch the network (tried to connect to {address!r})")

    def connect(self, address):
        check(address)
        return real_connect(self, address)

    def connect_ex(self, address):
        check(address)
        return real_connect_ex(self, address)

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", connect_ex)


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
            "captured_at": FROZEN_NOW.isoformat(),
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


@pytest.fixture(autouse=True)
def viewer_files_in_tmp(monkeypatch, tmp_path_factory):
    """The guest viewer gate reads its password and invite files from /etc/birdbrain by default. A host that has
    them (the public mirror) must not leak into tests, so every test starts with both pointed at an empty temp dir.
    Tests that need a password or invite file set the variables again."""
    d = tmp_path_factory.mktemp("viewer_etc")
    monkeypatch.setenv("VIEWER_PASSWORD_FILE", str(d / "viewer.pw"))
    monkeypatch.setenv("VIEWER_INVITE_FILE", str(d / "viewer.invite"))
