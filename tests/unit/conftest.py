"""The unit tier's guard: unit tests are pure (no files, database, sleeps, subprocesses, threads or network), and
this conftest makes that a rule pytest enforces rather than a note. The per-test data folder the base conftest makes
is not made here (config.DATA_DIR, IMAGES_DIR and DB_PATH point at a path that does not exist), the tmp_path and
capfd fixtures refuse, and the plain I/O a test might reach for by accident fails with a message saying where the test belongs.
Component and integration tests are not affected: this file applies to tests/unit/** only.

Every unit file, and every single unit test, must run on its own (`pytest tests/unit/test_config.py`), so nothing
here may depend on another file having imported the web app first."""
import builtins
import io
import os
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import pytest

from birdbrain import config

NOWHERE = Path("/nonexistent/birdbrain-unit-tests")  # nothing is ever written under it
MESSAGE = "unit tests are pure; move this test to tests/component (it tried {what})"
WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC


def _refuse(what: str):
    def refused(*args, **kwargs):
        raise RuntimeError(MESSAGE.format(what=what))
    return refused


def _load_app():
    """Import the web app while config still points at a real folder: its import makes IMAGES_DIR (app.images_dir),
    which would be an OSError once the paths point at NOWHERE. The base conftest's frozen_clock does the same import,
    but the autouse fixtures of a conftest run in alphabetical order, so `data_dirs` below runs before it."""
    from birdbrain.web import app as webapp
    return webapp


@pytest.fixture(autouse=True)
def data_dirs(monkeypatch):
    """Overrides the base conftest's data_dirs: no folder is made, and every config path points at NOWHERE (the
    /images mount too, so no unit test can see the session folder either)."""
    webapp = _load_app()
    monkeypatch.setattr(config, "DATA_DIR", NOWHERE)
    monkeypatch.setattr(config, "IMAGES_DIR", NOWHERE / "images")
    monkeypatch.setattr(config, "DB_PATH", NOWHERE / "birdbrain.db")
    monkeypatch.setenv("BIRDBRAIN_DATA", str(NOWHERE))
    from starlette.routing import Mount
    for route in webapp.app.routes:
        if isinstance(route, Mount) and route.name == "images":
            monkeypatch.setattr(route.app, "directory", NOWHERE / "images")
            monkeypatch.setattr(route.app, "all_directories", [NOWHERE / "images"])


@pytest.fixture(autouse=True)
def viewer_files_in_tmp(monkeypatch):
    """Overrides the base fixture (which makes a temp folder per test): the viewer files point at NOWHERE."""
    monkeypatch.setenv("VIEWER_PASSWORD_FILE", str(NOWHERE / "viewer.pw"))
    monkeypatch.setenv("VIEWER_INVITE_FILE", str(NOWHERE / "viewer.invite"))


@pytest.fixture
def tmp_path():
    pytest.fail(MESSAGE.format(what="the tmp_path fixture"))


@pytest.fixture
def tmp_path_factory():
    pytest.fail(MESSAGE.format(what="the tmp_path_factory fixture"))


@pytest.fixture
def tmpdir():
    pytest.fail(MESSAGE.format(what="the tmpdir fixture"))


@pytest.fixture
def tmpdir_factory():
    pytest.fail(MESSAGE.format(what="the tmpdir_factory fixture"))


# pytest's capfd and capfdbinary capture file descriptors through a temp file (tempfile.TemporaryFile, an open for
# writing). Left alone, that open hits the guard inside pytest's own capture code, where the error breaks the capture
# manager for the rest of the session: every later test errors and the patches stay applied. Refusing in fixture code
# instead fails just the test that asked. capsys and capsysbinary capture in memory and stay allowed.
@pytest.fixture
def capfd():
    pytest.fail(MESSAGE.format(what="the capfd fixture; capsys captures without a file"))


@pytest.fixture
def capfdbinary():
    pytest.fail(MESSAGE.format(what="the capfdbinary fixture; capsysbinary captures without a file"))


# The real callables, taken when this conftest is imported, before any test's guard is in place.
_TARGETS = {"time.sleep": time.sleep, "sqlite3.connect": sqlite3.connect}
_ALIASES: dict[str, list[tuple[object, str]]] = {name: [] for name in _TARGETS}
_SCANNED: set[str] = set()  # module names already looked through


def _aliases() -> dict[str, list[tuple[object, str]]]:
    """Module-level names in birdbrain.* bound to a guarded callable at import time (species_info._sleep is
    time.sleep), which patching time.sleep alone would miss: "time.sleep" / "sqlite3.connect" -> [(module, attr)].
    Importing the web app loads most of birdbrain but not all of it (autocal, calibrate, recircle and others are
    imported by tests or lazily), so every call looks at the birdbrain modules in sys.modules that have not been
    scanned yet: a module that arrives later is seen the next time a unit test starts, and again when each unit test
    ends, for the cost of one set difference each. A name bound to a guard (the module was first imported inside a
    unit test's body, while the guard was in place) is put back to the real callable by the scan at the end of that
    same test, so no stale guard outlives the unit test that made it: a component test that runs next, in random
    order or as the next tier, finds the real time.sleep behind the alias (test_tier_rules.py probes this)."""
    _load_app()
    new = [name for name in sys.modules if name.startswith("birdbrain") and name not in _SCANNED]
    for mod_name in new:
        mod = sys.modules.get(mod_name)
        _SCANNED.add(mod_name)
        if mod is None:
            continue
        for attr, value in list(vars(mod).items()):
            for name, target in _TARGETS.items():
                if value is target or getattr(value, "_unit_guard_for", None) == name:
                    setattr(mod, attr, target)
                    _ALIASES[name].append((mod, attr))
    return _ALIASES


@pytest.fixture(autouse=True)
def pure(monkeypatch):
    """Runs after the base conftest's autouse fixtures and before the test's own: a database connection, a sleep,
    a subprocess, a thread, a new folder, a file opened for writing (builtins.open, io.open, os.open with a write or
    create flag, Path.write_text/write_bytes, cv2.imwrite), or a file removed, renamed or replaced fails the test.
    Reading files stays allowed. Module-level aliases of time.sleep and sqlite3.connect inside birdbrain are
    guarded too (see `_aliases`), and a module the test body imported for the first time is scanned at teardown so
    an alias it bound to a guard is repaired before the next test."""
    aliases = _aliases()  # before time.sleep is patched below: the scan compares against the real one
    real_open = builtins.open

    def guarded_open(file, mode="r", *args, **kwargs):
        if isinstance(mode, str) and any(c in mode for c in "wax+"):
            raise RuntimeError(MESSAGE.format(what=f"open({str(file)!r}, {mode!r})"))
        return real_open(file, mode, *args, **kwargs)

    real_os_open = os.open

    def guarded_os_open(path, flags, *args, **kwargs):
        if flags & WRITE_FLAGS:
            raise RuntimeError(MESSAGE.format(what=f"os.open({str(path)!r}, flags={flags:#o})"))
        return real_os_open(path, flags, *args, **kwargs)

    real_sleep = time.sleep

    def sleep(seconds):
        if seconds > 0:
            raise RuntimeError(MESSAGE.format(what=f"time.sleep({seconds})"))
        real_sleep(seconds)

    connect = _refuse("sqlite3.connect")
    sleep._unit_guard_for, connect._unit_guard_for = "time.sleep", "sqlite3.connect"  # `_aliases` spots stale copies

    monkeypatch.setattr(builtins, "open", guarded_open)
    monkeypatch.setattr(io, "open", guarded_open)  # Path.open / write_text / write_bytes go through io.open
    monkeypatch.setattr(os, "open", guarded_os_open)  # Path.touch, tempfile.mkstemp, the admin secret file
    monkeypatch.setattr(Path, "write_text", _refuse("Path.write_text"))
    monkeypatch.setattr(Path, "write_bytes", _refuse("Path.write_bytes"))
    monkeypatch.setattr(Path, "mkdir", _refuse("Path.mkdir"))
    monkeypatch.setattr(os, "mkdir", _refuse("os.mkdir"))
    monkeypatch.setattr(os, "makedirs", _refuse("os.makedirs"))
    for name in ("remove", "unlink", "rename", "replace", "rmdir", "system"):  # Path.unlink/rename go through these
        monkeypatch.setattr(os, name, _refuse(f"os.{name}"))
    monkeypatch.setattr(shutil, "rmtree", _refuse("shutil.rmtree"))  # it stats the tree before it reaches os.unlink
    monkeypatch.setattr(cv2, "imwrite", _refuse("cv2.imwrite"))
    monkeypatch.setattr(sqlite3, "connect", connect)
    monkeypatch.setattr(time, "sleep", sleep)
    monkeypatch.setattr(subprocess, "Popen", _refuse("subprocess.Popen"))
    monkeypatch.setattr(subprocess, "run", _refuse("subprocess.run"))
    monkeypatch.setattr(threading.Thread, "start", _refuse("threading.Thread.start"))
    guards = {"time.sleep": sleep, "sqlite3.connect": connect}
    for name, places in aliases.items():
        for mod, attr in places:
            monkeypatch.setattr(mod, attr, guards[name])
    yield
    _aliases()  # a module first imported during the test bound the guard, not the real callable: put it back now
