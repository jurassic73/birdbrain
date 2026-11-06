"""The unit tier's guard: unit tests are pure (no files, database, sleeps, subprocesses or network), and this
conftest makes that a rule pytest enforces rather than a note. The per-test data folder the base conftest makes is
not made here (config.DATA_DIR, IMAGES_DIR and DB_PATH point at a path that does not exist), the tmp_path fixtures
refuse, and the plain I/O a test might reach for by accident fails with a message saying where the test belongs.
Component and integration tests are not affected: this file applies to tests/unit/** only."""
import builtins
import io
import os
import sqlite3
import subprocess
import time
from pathlib import Path

import pytest

from birdbrain import config

NOWHERE = Path("/nonexistent/birdbrain-unit-tests")  # nothing is ever written under it
MESSAGE = "unit tests are pure; move this test to tests/component (it tried {what})"


def _refuse(what: str):
    def refused(*args, **kwargs):
        raise RuntimeError(MESSAGE.format(what=what))
    return refused


@pytest.fixture(autouse=True)
def data_dirs(monkeypatch):
    """Overrides the base conftest's data_dirs: no folder is made, and every config path points at NOWHERE (the
    /images mount too, so no unit test can see the session folder either)."""
    monkeypatch.setattr(config, "DATA_DIR", NOWHERE)
    monkeypatch.setattr(config, "IMAGES_DIR", NOWHERE / "images")
    monkeypatch.setattr(config, "DB_PATH", NOWHERE / "birdbrain.db")
    monkeypatch.setenv("BIRDBRAIN_DATA", str(NOWHERE))
    from starlette.routing import Mount
    from birdbrain.web import app as webapp
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


@pytest.fixture(autouse=True)
def pure(monkeypatch):
    """Runs after the base conftest's autouse fixtures and before the test's own: a database connection, a sleep,
    a subprocess, a new folder or a file opened for writing (builtins.open, io.open, Path.write_text/write_bytes)
    fails the test. Reading files stays allowed."""
    real_open = builtins.open

    def guarded_open(file, mode="r", *args, **kwargs):
        if isinstance(mode, str) and any(c in mode for c in "wax+"):
            raise RuntimeError(MESSAGE.format(what=f"open({str(file)!r}, {mode!r})"))
        return real_open(file, mode, *args, **kwargs)

    real_sleep = time.sleep

    def sleep(seconds):
        if seconds > 0:
            raise RuntimeError(MESSAGE.format(what=f"time.sleep({seconds})"))
        real_sleep(seconds)

    monkeypatch.setattr(builtins, "open", guarded_open)
    monkeypatch.setattr(io, "open", guarded_open)  # Path.open / write_text / write_bytes go through io.open
    monkeypatch.setattr(Path, "write_text", _refuse("Path.write_text"))
    monkeypatch.setattr(Path, "write_bytes", _refuse("Path.write_bytes"))
    monkeypatch.setattr(Path, "mkdir", _refuse("Path.mkdir"))
    monkeypatch.setattr(os, "mkdir", _refuse("os.mkdir"))
    monkeypatch.setattr(os, "makedirs", _refuse("os.makedirs"))
    monkeypatch.setattr(sqlite3, "connect", _refuse("sqlite3.connect"))
    monkeypatch.setattr(time, "sleep", sleep)
    monkeypatch.setattr(subprocess, "Popen", _refuse("subprocess.Popen"))
    monkeypatch.setattr(subprocess, "run", _refuse("subprocess.run"))
