"""Rules the test suite itself must follow (contract #2a Groundwork, G1-G9).

Written blind, before the groundwork landed: these tests pin down the behaviour the contract promises, using the
names it gives. Paired tests (``_1_`` then ``_2_``) rely on file order: the first dirties something, the second
checks that the suite cleaned it up in between.
"""
import importlib
import os
import re
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

import pytest

from birdbrain import config, db

REPO_ROOT = Path(__file__).resolve().parent.parent
TESTS_DIR = REPO_ROOT / "tests"
PYPROJECT = REPO_ROOT / "pyproject.toml"
TIERS = ("unit", "component", "integration", "smoke", "guard")


def _pyproject() -> dict:
    return tomllib.loads(PYPROJECT.read_text())


def _ini() -> dict:
    return _pyproject()["tool"]["pytest"]["ini_options"]


def _addopts() -> list[str]:
    opts = _ini().get("addopts", "")
    return opts.split() if isinstance(opts, str) else list(opts)


def _test_sources():
    """Every test module under tests/ (any depth), path and text."""
    for path in sorted(TESTS_DIR.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        yield path, path.read_text()


def _run_pytest(*args, cwd=REPO_ROOT, timeout=60) -> subprocess.CompletedProcess:
    """A pytest session in a subprocess, with the repo's pyproject and no cache dir left behind."""
    cmd = [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "--rootdir", str(REPO_ROOT),
           "-c", str(PYPROJECT), *args]
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)


# --- G1: per-test data folders -------------------------------------------------------------------------------------

_written: list[Path] = []  # paths the first test of each pair wrote, for the second to check
_G4_DIRTY = object()  # a sentinel the G4 pairs plant in client/detector globals


def test_g1_1_a_test_writes_under_data_dir_and_images_dir():
    """First of a pair: leave files under DATA_DIR and IMAGES_DIR and remember where they went."""
    config.IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    data_file = config.DATA_DIR / "g1_leak.txt"
    image_file = config.IMAGES_DIR / "g1_leak.jpg"
    data_file.write_text("left behind by test_g1_1")
    image_file.write_bytes(b"\xff\xd8not really a jpeg")
    _written.extend([data_file, image_file])
    assert data_file.exists() and image_file.exists()


def test_g1_2_files_from_the_previous_test_are_not_visible():
    """Second of the pair: a later test never sees the files, under its own paths or the old ones."""
    assert _written, "test_g1_1 must run first (file order)"
    leaked = [old for old in _written if old.exists()]
    for old in leaked:
        old.unlink()  # never let the leak reach the rest of the suite, whatever this test says
    assert not leaked, f"files written by an earlier test are still there: {leaked}"
    assert not (config.DATA_DIR / "g1_leak.txt").exists()
    assert not (config.IMAGES_DIR / "g1_leak.jpg").exists()


def test_g1_data_dir_is_fresh_per_test_not_the_session_folder(tmp_path_factory):
    """DATA_DIR and IMAGES_DIR point under pytest's temp tree for this test, not at the session BIRDBRAIN_DATA."""
    session_dir = Path(os.environ["BIRDBRAIN_DATA"]).resolve()
    base = Path(tmp_path_factory.getbasetemp()).resolve()
    data_dir = config.DATA_DIR.resolve()
    assert data_dir != session_dir, "config.DATA_DIR is still the shared session folder"
    assert data_dir.is_relative_to(base), f"{data_dir} is not under pytest's tmp_path tree {base}"
    assert config.IMAGES_DIR.resolve().is_relative_to(data_dir), "IMAGES_DIR does not follow DATA_DIR"
    assert config.IMAGES_DIR.resolve() != (session_dir / "images")


def test_g1_db_path_follows_the_per_test_data_dir():
    assert config.DB_PATH.resolve().is_relative_to(config.DATA_DIR.resolve())
    assert not config.DB_PATH.resolve().is_relative_to(Path(os.environ["BIRDBRAIN_DATA"]).resolve())


def test_g1_web_app_live_frame_file_is_under_the_test_data_dir(tmp_path_factory):
    from birdbrain.web import app as webapp
    live = webapp.live_file().resolve()
    base = Path(tmp_path_factory.getbasetemp()).resolve()
    assert live.is_relative_to(config.DATA_DIR.resolve())
    assert live.is_relative_to(base), f"live frame file {live} is outside the test's tmp tree"
    assert not live.exists(), "a live frame from an earlier test is still there"


def test_g1_retention_lock_file_is_under_the_test_data_dir(tmp_path_factory):
    from birdbrain import retention
    base = Path(tmp_path_factory.getbasetemp()).resolve()
    lock = (config.DATA_DIR / retention.LOCK_FILE).resolve()
    assert lock.is_relative_to(base), f"retention lock {lock} is outside the test's tmp tree"
    assert not lock.exists(), "a retention lock from an earlier test is still there"


def test_g1_admin_secret_file_is_under_the_test_data_dir(tmp_path_factory):
    from birdbrain.web import auth
    base = Path(tmp_path_factory.getbasetemp()).resolve()
    secret = (config.DATA_DIR / auth.SECRET_FILE).resolve()
    assert secret.is_relative_to(base)
    assert not secret.exists(), "an admin secret from an earlier test is still there"


def test_g1_3_live_frame_written_here_is_gone_in_the_next_test():
    from birdbrain.web import app as webapp
    path = webapp.live_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\xff\xd8frame")
    _written.append(path)
    assert path.exists()


def test_g1_4_live_frame_from_the_previous_test_is_gone():
    from birdbrain.web import app as webapp
    assert _written and _written[-1].name == "live.jpg", "test_g1_3 must run first (file order)"
    leaked = _written[-1].exists()
    if leaked:
        _written[-1].unlink()  # never let the leak reach the rest of the suite
    assert not leaked, "live.jpg leaks from one test into the next"
    assert not webapp.live_file().exists()


# --- G2: the session folder is removed when the session ends --------------------------------------------------------

def test_g2_session_data_folder_is_removed_when_the_session_ends(tmp_path):
    """A tiny pytest session using the repo's conftest leaves no birdbrain-test-* folder in $TMPDIR."""
    probe = tmp_path / "test_probe_session.py"
    probe.write_text(
        "import os\n"
        "def test_where_the_session_folder_is():\n"
        "    print('SESSION_DATA=' + os.environ['BIRDBRAIN_DATA'])\n"
    )
    tmp = Path(tempfile.gettempdir())
    before = {p for p in tmp.glob("birdbrain-test-*")}
    result = _run_pytest("-p", "tests.conftest", "-s", str(probe))
    assert result.returncode == 0, result.stdout + result.stderr
    match = re.search(r"SESSION_DATA=(\S+)", result.stdout)
    assert match, result.stdout
    session_dir = Path(match.group(1))
    assert session_dir.name.startswith("birdbrain-test-"), session_dir
    assert not session_dir.exists(), f"session folder {session_dir} was left behind"
    left = {p for p in tmp.glob("birdbrain-test-*")} - before
    assert not left, f"new leftover folders: {sorted(p.name for p in left)}"


# --- G4: module globals are reset between tests ---------------------------------------------------------------------

def test_g4_1_dirty_module_globals():
    """First of a pair: leave state in the globals the contract names."""
    from birdbrain import lessons
    from birdbrain.web import app as webapp, auth, viewer
    auth._fails["203.0.113.9"] = [1.0, 2.0, 3.0]
    auth._secrets["g4"] = b"not-a-real-secret"
    viewer._fails["203.0.113.9"] = [1.0]
    viewer._cache["g4"] = (0.0, "dirty")
    webapp._live["jpeg"] = b"\xff\xd8dirty"
    webapp._live["at"] = 12345.0
    webapp._run_state["running"] = True
    db._totals_cache[("g4", "nowhere.db")] = (0.0, {"dirty": True})
    lessons._running = True
    assert auth._fails and webapp._live["jpeg"] is not None and db._totals_cache


def test_g4_2_module_globals_are_clean_again():
    """Second of the pair: every global the previous test dirtied starts clean."""
    from birdbrain import lessons
    from birdbrain.web import app as webapp, auth, viewer
    leaks = []
    if "203.0.113.9" in auth._fails:
        leaks.append("auth._fails")
        auth._fails.pop("203.0.113.9")
    if "g4" in auth._secrets:
        leaks.append("auth._secrets")
        auth._secrets.pop("g4")
    if "203.0.113.9" in viewer._fails:
        leaks.append("viewer._fails")
        viewer._fails.pop("203.0.113.9")
    if "g4" in viewer._cache:
        leaks.append("viewer._cache")
        viewer._cache.pop("g4")
    if webapp._live["jpeg"] is not None or webapp._live["at"] != 0.0:
        leaks.append("webapp._live")
        webapp._live.update(jpeg=None, at=0.0, tried=0.0)
    if webapp._run_state["running"] is not False:
        leaks.append("webapp._run_state")
        webapp._run_state["running"] = False
    if ("g4", "nowhere.db") in db._totals_cache:
        leaks.append("db._totals_cache")
        db._totals_cache.pop(("g4", "nowhere.db"))
    if lessons._running is not False:
        leaks.append("lessons._running")
        lessons._running = False
    # The leaks above were undone first so a failure here never spoils the rest of the suite.
    assert not leaks, f"module globals leak between tests: {leaks}"


def test_g4_lessons_background_is_off_for_every_test():
    from birdbrain import lessons
    assert lessons.BACKGROUND is False, "lessons.BACKGROUND must be False in tests (no distill thread outlives a test)"


def test_g4_3_dirty_the_other_named_globals():
    """First of a pair: the globals that hold a client, a detector or the watcher's memory of its last visit."""
    from birdbrain import circle_check, identify, privacy, watch
    watch.last_visit_id = 999_999
    watch._last_logged["g4 dirty"] = 1.0
    watch._hold_cache = (999_999.0, True)
    circle_check._last_used = 999_999.0
    identify._client = _G4_DIRTY
    privacy._det = _G4_DIRTY


def test_g4_4_the_other_named_globals_are_clean_again():
    from birdbrain import circle_check, identify, privacy, watch
    leaks = []
    if watch.last_visit_id == 999_999:
        leaks.append("watch.last_visit_id")
        watch.last_visit_id = None
    if "g4 dirty" in watch._last_logged:
        leaks.append("watch._last_logged")
        watch._last_logged.pop("g4 dirty")
    if watch._hold_cache == (999_999.0, True):
        leaks.append("watch._hold_cache")
        watch._hold_cache = (0.0, False)
    if circle_check._last_used == 999_999.0:
        leaks.append("circle_check._last_used")
        circle_check._last_used = 0.0
    if identify._client is _G4_DIRTY:
        leaks.append("identify._client")
        identify._client = None
    if privacy._det is _G4_DIRTY:
        leaks.append("privacy._det")
        privacy._det = None
    # Undone first so a failure here never spoils the rest of the suite.
    assert not leaks, f"module globals leak between tests: {leaks}"
    assert identify._client is None and privacy._det is None


# --- G5: the species card helper stamps the frozen clock ----------------------------------------------------------

def test_g5_save_card_helper_uses_frozen_now_not_the_real_clock():
    """Wherever `_save_card` lives now (test_species_info.py or tests/helpers/), it must not read the real clock."""
    found = []
    for path, text in _test_sources():
        m = re.search(r"^def _save_card\(.*?(?=^\S)", text, re.S | re.M)
        if m:
            found.append((path, m.group(0)))
    assert found, "no _save_card helper found under tests/"
    for path, body in found:
        assert "FROZEN_NOW" in body, f"{path.name}: _save_card does not stamp fetched_at with FROZEN_NOW"
        assert "datetime.now(" not in body and "time.time(" not in body, f"{path.name}: _save_card reads the real clock"


# --- G6: no test imports another test module or conftest; helpers live in tests/helpers/ ----------------------------

_FORBIDDEN_IMPORT = re.compile(r"^\s*(?:from\s+(?:conftest|test_\w+)\s+import\b|import\s+(?:conftest|test_\w+)\b)", re.M)


def test_g6_no_test_module_imports_conftest_or_another_test_module():
    offenders = []
    for path, text in _test_sources():
        for m in _FORBIDDEN_IMPORT.finditer(text):
            line_no = text.count("\n", 0, m.start()) + 1
            offenders.append(f"{path.relative_to(REPO_ROOT)}:{line_no}: {m.group(0).strip()}")
    assert not offenders, "test modules importing conftest or other test modules:\n" + "\n".join(offenders)


def test_g6_helpers_package_exists():
    assert (TESTS_DIR / "helpers").is_dir(), "tests/helpers/ package is missing"
    assert (TESTS_DIR / "helpers" / "__init__.py").is_file(), "tests/helpers/__init__.py is missing"


def test_g6_helpers_package_is_importable_as_helpers():
    helpers = importlib.import_module("helpers")
    assert Path(helpers.__file__).resolve() == (TESTS_DIR / "helpers" / "__init__.py").resolve()


def test_g6_helpers_carry_the_shared_fakes():
    """The shared names the contract lists are importable from somewhere in helpers.*."""
    wanted = {"FROZEN_NOW", "image_size", "FakeClient", "CARD"}
    found = set()
    for path in sorted((TESTS_DIR / "helpers").glob("*.py")):
        mod = importlib.import_module(f"helpers.{path.stem}" if path.stem != "__init__" else "helpers")
        found |= {name for name in wanted if hasattr(mod, name)}
    assert found == wanted, f"not found in tests/helpers/: {sorted(wanted - found)}"


def test_g6_pythonpath_is_repo_root_and_tests():
    assert _ini().get("pythonpath") == [".", "tests"]


# --- G7: tier markers -----------------------------------------------------------------------------------------------

def test_g7_the_five_tier_markers_are_registered(request):
    registered = {line.split(":")[0].split("(")[0].strip() for line in request.config.getini("markers")}
    missing = [m for m in TIERS if m not in registered]
    assert not missing, f"markers not registered in pyproject: {missing}"


def test_g7_pyproject_registers_the_markers_itself():
    names = {line.split(":")[0].split("(")[0].strip() for line in _ini().get("markers", [])}
    assert set(TIERS) <= names, f"pyproject markers: {sorted(names)}"


def test_g7_strict_markers_is_in_addopts():
    assert "--strict-markers" in _addopts(), f"addopts = {_addopts()}"


def test_g7_tier_marker_follows_the_folder(request, tmp_path):
    """tests/unit/** -> unit, tests/component/** -> component, tests/integration/** -> integration, via the
    collection hook. Checked by collecting a throwaway file in each folder with `-m <tier>`."""
    results = {}
    for tier in ("unit", "component", "integration"):
        folder = TESTS_DIR / tier / "_g7_probe"
        folder.mkdir(parents=True, exist_ok=True)
        probe = folder / "test_g7_probe.py"
        try:
            probe.write_text("def test_probe():\n    pass\n")
            res = _run_pytest("--collect-only", "-m", tier, str(probe))  # addopts already has one -q: node ids
            results[tier] = (res.returncode, res.stdout + res.stderr)
        finally:
            probe.unlink(missing_ok=True)
            for p in folder.rglob("*"):
                if p.is_file():
                    p.unlink()
            for p in sorted(folder.rglob("*"), reverse=True):
                p.rmdir()
            folder.rmdir()
            if not any((TESTS_DIR / tier).iterdir()):
                (TESTS_DIR / tier).rmdir()
    for tier, (code, out) in results.items():
        assert code == 0, f"{tier}: {out}"
        assert "test_g7_probe.py::test_probe" in out, f"tests/{tier}/ item did not get the {tier} marker:\n{out}"


def test_g7_no_personal_info_tests_are_marked_guard():
    """Deselecting `guard` leaves nothing of tests/test_no_personal_info.py."""
    guard_file = str(TESTS_DIR / "test_no_personal_info.py")
    res = _run_pytest("--collect-only", "-m", "not guard", guard_file)  # addopts already has one -q: node ids
    out = res.stdout + res.stderr
    assert "test_no_personal_info.py::" not in out, "test_no_personal_info.py items are not marked guard:\n" + out
    res = _run_pytest("--collect-only", "-m", "guard", guard_file)
    assert res.returncode == 0 and "test_no_personal_info.py::" in res.stdout, res.stdout + res.stderr


def test_g7_guard_marker_on_the_items_in_this_session(request):
    """When the privacy guard is part of this run, each of its items carries the guard marker."""
    items = [i for i in request.session.items if Path(str(i.fspath)).name == "test_no_personal_info.py"]
    if not items:
        pytest.skip("test_no_personal_info.py is not part of this run")
    unmarked = [i.nodeid for i in items if i.get_closest_marker("guard") is None]
    assert not unmarked, f"not marked guard: {unmarked}"


# --- G8: dev dependencies -------------------------------------------------------------------------------------------

def _dev_group_names() -> set[str]:
    dev = _pyproject().get("dependency-groups", {}).get("dev", [])
    return {re.split(r"[<>=!~\[; ]", d, maxsplit=1)[0].strip().lower() for d in dev if isinstance(d, str)}


def test_g8_pytest_xdist_is_a_dev_dependency():
    assert "pytest-xdist" in _dev_group_names(), f"dev group: {sorted(_dev_group_names())}"


def test_g8_pytest_timeout_is_a_dev_dependency():
    assert "pytest-timeout" in _dev_group_names(), f"dev group: {sorted(_dev_group_names())}"


def test_g8_lock_file_knows_the_new_dev_dependencies():
    lock = (REPO_ROOT / "uv.lock").read_text()
    assert 'name = "pytest-xdist"' in lock, "uv.lock has no pytest-xdist"
    assert 'name = "pytest-timeout"' in lock, "uv.lock has no pytest-timeout"


def test_g8_addopts_never_forces_xdist_workers():
    opts = _addopts()
    assert "-n" not in opts and not any(o.startswith("--numprocesses") or o.startswith("-n") for o in opts), opts


def test_g8_timeout_applies_when_the_plugin_is_installed(request):
    """With pytest-timeout installed every test gets a 60 s limit; without it the ini must not break the run."""
    if request.config.pluginmanager.hasplugin("timeout"):
        timeout = request.config.getoption("timeout", default=None) or request.config.getini("timeout")
        assert float(timeout or 0) == 60, f"pytest-timeout limit is {timeout!r}, not 60 s"
    else:
        ini = _ini()
        assert "timeout" not in ini, "a timeout ini key without pytest-timeout installed warns on every run"
        assert not any(o.startswith("--timeout") for o in _addopts()), "--timeout in addopts breaks plain pytest"
        pytest.skip("pytest-timeout is not installed in this environment")


# --- G3: the live.jpg pair in both orders (the known bug) -----------------------------------------------------------

@pytest.mark.skipif(os.environ.get("BIRDBRAIN_SUITE_RULES_SLOW") != "1",
                    reason="about 4 s: run with BIRDBRAIN_SUITE_RULES_SLOW=1; the full-suite gates cover it")
def test_g3_live_snapshot_then_public_mirror_pass_in_the_failing_order():
    res = _run_pytest("tests/test_web.py::test_live_snapshot_is_shared_within_the_refresh_window",
                      "tests/test_read_only.py::test_public_mirror_has_no_admin_and_never_touches_camera")
    assert res.returncode == 0, res.stdout + res.stderr
