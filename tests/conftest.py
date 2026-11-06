import atexit
import os
import shutil
import socket
import sys
import tempfile
import threading
from datetime import date as _real_date, datetime as _real_datetime
from pathlib import Path

# Before birdbrain is imported: never read the real .env (camera password, API key), and keep data in a temp dir.
# The folder is this process's fallback for anything that runs before the per-test `data_dirs` fixture (every
# xdist worker and the controller make their own); `_remove_session_dir` takes it away when the session ends.
os.environ["BIRDBRAIN_NO_DOTENV"] = "1"
_SESSION_DIR = tempfile.mkdtemp(prefix="birdbrain-test-")
os.environ["BIRDBRAIN_DATA"] = _SESSION_DIR
os.environ["ANTHROPIC_API_KEY"] = "test-key-never-sent"
os.environ["RTSP_PASS"] = "p@ss#word!"

import pytest  # noqa: E402

from birdbrain import config, db  # noqa: E402
from helpers.clock import FROZEN_NOW, FrozenDate, FrozenDatetime  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent


def _remove_session_dir() -> None:
    shutil.rmtree(_SESSION_DIR, ignore_errors=True)


atexit.register(_remove_session_dir)  # a backstop for a process that never reaches pytest_unconfigure


def pytest_unconfigure(config):  # noqa: ARG001  (pytest's config, not birdbrain's)
    _remove_session_dir()


# --- markers and tiers -------------------------------------------------------------------------------------------

TIERS = ("unit", "component", "integration")
GUARD_FILES = {"test_no_personal_info.py"}
TEST_TIMEOUT = 60  # seconds per test when pytest-timeout is installed (plain pytest runs without a limit)


def _timeout_given(config) -> bool:
    """True when the run already has a limit of its own: `--timeout`, `PYTEST_TIMEOUT` or an ini `timeout` key.
    pytest-timeout reads a `timeout` marker before all three, so the default marker below must step aside for them
    (`--timeout=0` counts too: it switches the limit off). Only called when the plugin is loaded."""
    if config.getoption("timeout", default=None) is not None:
        return True
    if os.environ.get("PYTEST_TIMEOUT"):
        return True
    try:
        return bool(config.getini("timeout"))
    except ValueError:  # the key is only registered by the plugin
        return False


def pytest_configure(config):
    config.addinivalue_line("markers", "real_scrypt: hash the guest password with the production scrypt cost "
                                       "(every other test gets the cheap one from the cheap_scrypt fixture)")


def pytest_collection_modifyitems(config, items):
    """Mark each test with its tier from its folder (tests/unit/** -> unit, ...), the guard files `guard`, and give
    every test a TEST_TIMEOUT limit when pytest-timeout is present and the run was not given a timeout itself."""
    tests_dir = Path(str(config.rootpath)) / "tests"
    timeout = config.pluginmanager.hasplugin("timeout") and not _timeout_given(config)
    for item in items:
        path = Path(str(item.path))
        try:
            rel = path.relative_to(tests_dir)
        except ValueError:
            rel = None
        if rel is not None and rel.parts and rel.parts[0] in TIERS:
            item.add_marker(getattr(pytest.mark, rel.parts[0]))
        if path.name in GUARD_FILES:
            item.add_marker(pytest.mark.guard)
        if timeout and item.get_closest_marker("timeout") is None:
            item.add_marker(pytest.mark.timeout(TEST_TIMEOUT))


# --- the clock ---------------------------------------------------------------------------------------------------


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


@pytest.fixture
def now():
    return FROZEN_NOW


# --- isolation: folders, database, module state ------------------------------------------------------------------


@pytest.fixture(autouse=True)
def data_dirs(tmp_path_factory, monkeypatch):
    """Every test gets its own data folder: config.DATA_DIR, IMAGES_DIR and DB_PATH point into a fresh folder next
    to tmp_path (not inside it: some tests list tmp_path), and so does BIRDBRAIN_DATA for anything a test starts in
    a subprocess. Every path birdbrain derives from them (the web app's live.jpg, the retention lock, the
    calibration/reference pictures, the admin secret, the privacy cache, the name lists, the models folder) is
    computed from config at call time, so this covers them all but one: the /images mount, whose folder StaticFiles
    fixed when the app was imported, is pointed at the new IMAGES_DIR too. A file one test writes is never seen by
    the next; `conn` still puts the test database at tmp_path/test.db."""
    data = tmp_path_factory.mktemp("data")
    images = data / "images"
    monkeypatch.setattr(config, "DATA_DIR", data)
    monkeypatch.setattr(config, "IMAGES_DIR", images)
    monkeypatch.setattr(config, "DB_PATH", data / "birdbrain.db")
    monkeypatch.setenv("BIRDBRAIN_DATA", str(data))
    from starlette.routing import Mount
    from birdbrain.web import app as webapp
    for route in webapp.app.routes:
        if isinstance(route, Mount) and route.name == "images":
            monkeypatch.setattr(route.app, "directory", images)
            monkeypatch.setattr(route.app, "all_directories", [images])


@pytest.fixture(autouse=True)
def fresh_module_state(monkeypatch):
    """Module globals that would otherwise carry over from one test to the next (caches, lockout counters, the
    live frame, background flags) start every test at their import-time values. Tests that want another value
    set it themselves on top of this."""
    from birdbrain import circle_check, clock, identify, lessons, privacy, retention, watch, weather
    from birdbrain.web import app as webapp, auth, viewer
    monkeypatch.setattr(clock, "_zone", {"tz": None, "checked": 0.0})  # set_zone() would move every later test's clock
    monkeypatch.setattr(retention, "_thread", None)
    monkeypatch.setattr(retention, "_wake", threading.Event())
    monkeypatch.setattr(auth, "_fails", {})
    monkeypatch.setattr(auth, "_secrets", {})
    monkeypatch.setattr(viewer, "_fails", {})
    monkeypatch.setattr(viewer, "_cache", {})
    monkeypatch.setattr(webapp, "_live", {"jpeg": None, "at": 0.0, "tried": 0.0})
    monkeypatch.setattr(webapp, "_run_state", {"running": False, "result": None, "error": ""})
    monkeypatch.setattr(watch, "_hold_cache", (0.0, False))
    monkeypatch.setattr(watch, "last_visit_id", None)
    monkeypatch.setattr(watch, "_last_logged", {})
    monkeypatch.setattr(watch, "_push", None)
    monkeypatch.setattr(watch, "_push_started", 0.0)
    monkeypatch.setattr(circle_check, "_last_used", 0.0)
    monkeypatch.setattr(lessons, "_running", False)
    monkeypatch.setattr(lessons, "BACKGROUND", False)  # the automatic distill runs in the request, never a thread
    monkeypatch.setattr(db, "_totals_cache", {})
    monkeypatch.setattr(identify, "_client", None)
    monkeypatch.setattr(privacy, "_det", None)
    monkeypatch.setattr(privacy, "_status", (0.0, {}))
    monkeypatch.setattr(weather, "_failed_at", 0.0)
    retention.forget()  # cached usage, breakdown and forecast; the flock file itself lives under this test's DATA_DIR
    yield
    retention.forget()


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
def cheap_scrypt(request, monkeypatch):
    """The guest password is hashed with scrypt at n=2**14 (~50 ms a hash, two per login): every test that makes
    or checks one gets the same hashlib.scrypt with n=2**4 instead, so the function's logic (salt, base64, the
    constant-time compare in check_password) is unchanged and only the work factor drops. A test marked
    `real_scrypt` keeps the production parameters, so one real hash is still made and checked every run."""
    import hashlib
    from birdbrain.web import viewer
    if request.node.get_closest_marker("real_scrypt"):
        return

    def scrypt(password: str, salt: bytes) -> bytes:
        return hashlib.scrypt(password.encode(), salt=salt, n=2**4, r=8, p=1, dklen=32)

    monkeypatch.setattr(viewer, "_scrypt", scrypt)


@pytest.fixture(autouse=True)
def viewer_files_in_tmp(monkeypatch, tmp_path_factory):
    """The guest viewer gate reads its password and invite files from /etc/birdbrain by default. A host that has
    them (the public mirror) must not leak into tests, so every test starts with both pointed at an empty temp dir.
    Tests that need a password or invite file set the variables again."""
    d = tmp_path_factory.mktemp("viewer_etc")
    monkeypatch.setenv("VIEWER_PASSWORD_FILE", str(d / "viewer.pw"))
    monkeypatch.setenv("VIEWER_INVITE_FILE", str(d / "viewer.invite"))
