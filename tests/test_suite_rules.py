"""Rules the test suite itself must follow (contract #2a Groundwork, G1-G9).

Written blind, before the groundwork landed: these tests pin down the behaviour the contract promises, using the
names it gives. The "one test dirties, the next must be clean" checks run as a tiny two-test probe file in a pytest
subprocess with the repo's conftest (``LEAK_PROBE``, one run per module, shared by the tests that read it), so they
pass in any order, on any xdist worker, and still catch a leak.
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


# --- the leak probe: one test dirties files and globals, the next must start clean -----------------------------------

LEAK_PROBE = r'''"""Probe run by tests/test_suite_rules.py in a subprocess with the repo's conftest (`-p tests.conftest`).
test_1 leaves files under DATA_DIR / IMAGES_DIR / live_file() and dirties every module global the contract names;
test_2 reports which of them are still visible as `LEAKS=a,b,c` (empty when the suite cleaned up) and fails on any.
test_3 prints the per-test time limit pytest-timeout will enforce on it (`none` without the plugin)."""
from pathlib import Path

from birdbrain import circle_check, clock, config, db, identify, lessons, privacy, retention, watch
from birdbrain.web import app as webapp, auth, viewer

_seen = {}
DIRTY = object()
IP = "203.0.113.9"


def _session_folder(path):
    return any(part.startswith("birdbrain-test-") for part in Path(path).resolve().parts)


def test_1_dirty_files_and_globals(tmp_path_factory):
    _seen["base"] = Path(tmp_path_factory.getbasetemp()).resolve()
    _seen["data"] = config.DATA_DIR.resolve()
    _seen["images"] = config.IMAGES_DIR.resolve()
    config.IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    (config.DATA_DIR / "leak.txt").write_text("left behind by test_1")
    (config.IMAGES_DIR / "leak.jpg").write_bytes(b"\xff\xd8not really a jpeg")
    live = webapp.live_file()
    live.parent.mkdir(parents=True, exist_ok=True)
    live.write_bytes(b"\xff\xd8frame")
    _seen["live"] = live.resolve()
    auth._fails[IP] = [1.0, 2.0, 3.0]
    auth._secrets["probe"] = b"not-a-real-secret"
    viewer._fails[IP] = [1.0]
    viewer._cache["probe"] = (0.0, "dirty")
    webapp._live["jpeg"] = b"\xff\xd8dirty"
    webapp._live["at"] = 12345.0
    webapp._run_state["running"] = True
    db._totals_cache[("probe", "nowhere.db")] = (0.0, {"dirty": True})
    lessons._running = True
    watch.last_visit_id = 999_999
    watch._last_logged["probe dirty"] = 1.0
    watch._hold_cache = (999_999.0, True)
    circle_check._last_used = 999_999.0
    identify._client = DIRTY
    privacy._det = DIRTY
    clock.set_zone("Pacific/Kiritimati")
    retention._thread = DIRTY
    retention._wake.set()


def test_2_everything_is_clean_again():
    assert _seen, "test_1 must run first (the probe is run with -p no:randomly)"
    data = config.DATA_DIR.resolve()
    leaks = []
    if not data.is_relative_to(_seen["base"]):
        leaks.append("data_dir_outside_pytest_tmp")
    if _session_folder(data) or _session_folder(config.IMAGES_DIR):
        leaks.append("data_dir_is_session_folder")
    if not config.IMAGES_DIR.resolve().is_relative_to(data):
        leaks.append("images_dir_not_under_data_dir")
    if data == _seen["data"]:
        leaks.append("data_dir_shared")
    if (config.DATA_DIR / "leak.txt").exists():
        leaks.append("data_file_visible")
    if (config.IMAGES_DIR / "leak.jpg").exists():
        leaks.append("image_file_visible")
    if (_seen["data"] / "leak.txt").is_relative_to(data) or (_seen["images"] / "leak.jpg").is_relative_to(data):
        leaks.append("old_files_under_new_data_dir")
    live = webapp.live_file()
    if live.exists():
        leaks.append("live_file_visible")
    if live.resolve() == _seen["live"]:
        leaks.append("live_file_shared")
    if _seen["live"].is_relative_to(data):
        leaks.append("old_live_under_new_data_dir")
    if IP in auth._fails:
        leaks.append("auth._fails")
    if "probe" in auth._secrets:
        leaks.append("auth._secrets")
    if IP in viewer._fails:
        leaks.append("viewer._fails")
    if "probe" in viewer._cache:
        leaks.append("viewer._cache")
    if webapp._live["jpeg"] is not None or webapp._live["at"] != 0.0:
        leaks.append("webapp._live")
    if webapp._run_state["running"] is not False:
        leaks.append("webapp._run_state")
    if ("probe", "nowhere.db") in db._totals_cache:
        leaks.append("db._totals_cache")
    if lessons._running is not False:
        leaks.append("lessons._running")
    if lessons.BACKGROUND is not False:
        leaks.append("lessons.BACKGROUND")
    if watch.last_visit_id == 999_999:
        leaks.append("watch.last_visit_id")
    if "probe dirty" in watch._last_logged:
        leaks.append("watch._last_logged")
    if watch._hold_cache == (999_999.0, True):
        leaks.append("watch._hold_cache")
    if circle_check._last_used == 999_999.0:
        leaks.append("circle_check._last_used")
    if identify._client is not None:
        leaks.append("identify._client")
    if privacy._det is not None:
        leaks.append("privacy._det")
    if clock._zone["tz"] is not None or clock._zone["checked"] != 0.0:
        leaks.append("clock._zone")
    if retention._thread is not None:
        leaks.append("retention._thread")
    if retention._wake.is_set():
        leaks.append("retention._wake")
    print("LEAKS=" + ",".join(leaks))
    assert not leaks, f"leaks from one test into the next: {leaks}"


def test_3_effective_timeout(request):
    try:
        import pytest_timeout
    except ImportError:
        print("EFFECTIVE_TIMEOUT=none")
        return
    print(f"EFFECTIVE_TIMEOUT={pytest_timeout._get_item_settings(request.node).timeout}")
'''

G1_FILE_LEAKS = {"data_dir_shared", "data_file_visible", "image_file_visible", "old_files_under_new_data_dir"}
G1_FRESH_DIR_LEAKS = {"data_dir_outside_pytest_tmp", "data_dir_is_session_folder", "images_dir_not_under_data_dir",
                      "data_dir_shared"}
G1_LIVE_LEAKS = {"live_file_visible", "live_file_shared", "old_live_under_new_data_dir"}
G4_WEB_LEAKS = {"auth._fails", "auth._secrets", "viewer._fails", "viewer._cache", "webapp._live", "webapp._run_state",
                "db._totals_cache", "lessons._running", "lessons.BACKGROUND"}
G4_OTHER_LEAKS = {"watch.last_visit_id", "watch._last_logged", "watch._hold_cache", "circle_check._last_used",
                  "identify._client", "privacy._det"}
G4_CLOCK_LEAKS = {"clock._zone", "retention._thread", "retention._wake"}


def _run_probe(folder: Path, *options, select: str = "", env_extra: dict | None = None) -> subprocess.CompletedProcess:
    """Run LEAK_PROBE (all of it, or the `select` node id such as "::test_3_effective_timeout") in a pytest
    subprocess that loads the repo's conftest, with a private TMPDIR (the session folder lands there) and none of
    this run's own timeout settings leaking in; `options` go on the pytest command line."""
    private_tmp = folder / "t"
    private_tmp.mkdir(exist_ok=True)
    probe = folder / "test_leak_probe.py"
    if not probe.exists():
        probe.write_text(LEAK_PROBE)
    env = dict(os.environ, TMPDIR=str(private_tmp), TMP=str(private_tmp), TEMP=str(private_tmp))
    for key in ("PYTEST_TIMEOUT", "PYTEST_ADDOPTS"):
        env.pop(key, None)
    env.update(env_extra or {})
    cmd = [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-p", "no:randomly", "-p", "no:xdist",
           "--rootdir", str(REPO_ROOT), "-c", str(PYPROJECT), "-p", "tests.conftest", "-s", *options,
           str(probe) + select]
    return subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=60, env=env)


@pytest.fixture(scope="module")
def leak_probe(tmp_path_factory):
    """One run of LEAK_PROBE for this module: the process result and the set of leaks test_2 reported. Missing
    report line = the probe did not get that far; the tests below then fail with its whole output."""
    res = _run_probe(tmp_path_factory.mktemp("leak_probe"))
    out = res.stdout + res.stderr
    m = re.search(r"LEAKS=(.*)$", res.stdout, re.M)  # the progress dot precedes it
    leaks = set(filter(None, m.group(1).strip().split(","))) if m else None
    return res, out, leaks


def _assert_no_leak_in(leak_probe, names: set[str]) -> None:
    res, out, leaks = leak_probe
    assert leaks is not None, "the probe did not report LEAKS=:\n" + out
    found = sorted(leaks & names)
    assert not found, f"leaks from one test into the next: {found}\n{out}"


# --- G1: per-test data folders -------------------------------------------------------------------------------------

def _is_session_folder(path: Path) -> bool:
    """True when `path` is, or lies under, a shared `birdbrain-test-*` session folder (the mkdtemp prefix)."""
    return any(part.startswith("birdbrain-test-") for part in path.resolve().parts)


def test_g1_files_written_by_one_test_are_not_visible_in_the_next(leak_probe):
    """The probe's first test writes under DATA_DIR and IMAGES_DIR; its second finds the same relative names absent,
    a different DATA_DIR, and the old files not under it. The old absolute paths may well still be on disk (pytest
    keeps a few old tmp folders); the contract only promises they are not visible from the next test."""
    _assert_no_leak_in(leak_probe, G1_FILE_LEAKS)


def test_g1_data_dir_is_fresh_per_test_not_the_session_folder(tmp_path_factory, leak_probe):
    """DATA_DIR and IMAGES_DIR point under pytest's temp tree for this test, never at a shared birdbrain-test-*
    session folder, and consecutive tests get different folders (checked across the probe's pair)."""
    base = Path(tmp_path_factory.getbasetemp()).resolve()
    data_dir = config.DATA_DIR.resolve()
    assert data_dir.is_relative_to(base), f"{data_dir} is not under pytest's tmp_path tree {base}"
    assert not _is_session_folder(data_dir), f"config.DATA_DIR {data_dir} is a shared session folder"
    assert not data_dir.name.startswith("birdbrain-test-"), data_dir
    assert config.IMAGES_DIR.resolve().is_relative_to(data_dir), "IMAGES_DIR does not follow DATA_DIR"
    assert not _is_session_folder(config.IMAGES_DIR), f"IMAGES_DIR {config.IMAGES_DIR} is in a session folder"
    _assert_no_leak_in(leak_probe, G1_FRESH_DIR_LEAKS)


def test_g1_db_path_follows_the_per_test_data_dir(tmp_path, tmp_path_factory):
    """DB_PATH lives under this test's DATA_DIR (or its tmp tree), never under a shared session folder."""
    db_path = config.DB_PATH.resolve()
    base = Path(tmp_path_factory.getbasetemp()).resolve()
    assert db_path.is_relative_to(config.DATA_DIR.resolve()) or db_path.is_relative_to(tmp_path.resolve()), \
        f"DB_PATH {db_path} does not follow DATA_DIR {config.DATA_DIR} or tmp_path {tmp_path}"
    assert db_path.is_relative_to(base), f"DB_PATH {db_path} is outside pytest's tmp tree {base}"
    assert not _is_session_folder(db_path), f"DB_PATH {db_path} is in a shared session folder"


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


def test_g1_live_frame_written_by_one_test_is_gone_in_the_next(leak_probe):
    """The live frame the probe's first test saved is not visible to its second: live_file() does not exist, is not
    the same file, and the DATA_DIR it derives from is another folder (the known live.jpg bug, G3)."""
    _assert_no_leak_in(leak_probe, G1_LIVE_LEAKS)


# --- G2: the session folder is removed when the session ends --------------------------------------------------------

def test_g2_session_data_folder_is_removed_when_the_session_ends(tmp_path):
    """A tiny pytest session using the repo's conftest creates a birdbrain-test-* folder in $TMPDIR while it runs
    and leaves none behind. The session runs with a private TMPDIR so no other process can create or remove
    folders in the dir being watched."""
    private_tmp = tmp_path / "t"
    private_tmp.mkdir()
    probe = tmp_path / "test_probe_session.py"
    probe.write_text(
        "import tempfile\n"
        "from pathlib import Path\n"
        "def test_a_session_folder_exists_while_the_session_runs():\n"
        "    tmp = Path(tempfile.gettempdir())\n"
        "    found = sorted(p.name for p in tmp.glob('birdbrain-test-*') if p.is_dir())\n"
        "    print('SESSION_FOLDERS=' + ','.join(found))\n"
        "    assert found, f'no birdbrain-test-* folder in {tmp}'\n"
    )
    env = dict(os.environ, TMPDIR=str(private_tmp), TMP=str(private_tmp), TEMP=str(private_tmp))
    cmd = [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "--rootdir", str(REPO_ROOT),
           "-c", str(PYPROJECT), "-p", "tests.conftest", "-s", str(probe)]
    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=60, env=env)
    assert result.returncode == 0, result.stdout + result.stderr
    match = re.search(r"SESSION_FOLDERS=(\S+)", result.stdout)
    assert match, "the probe did not report a session folder:\n" + result.stdout
    assert all(name.startswith("birdbrain-test-") for name in match.group(1).split(",")), match.group(1)
    left = sorted(p.name for p in private_tmp.glob("birdbrain-test-*"))
    assert not left, f"session folders left behind in {private_tmp}: {left}"


# --- G4: module globals are reset between tests ---------------------------------------------------------------------

def test_g4_module_globals_are_clean_in_the_next_test(leak_probe):
    """The probe dirties auth._fails/_secrets, viewer._fails/_cache, webapp._live/_run_state, db._totals_cache and
    lessons._running; the next test finds every one of them at its import-time value."""
    _assert_no_leak_in(leak_probe, G4_WEB_LEAKS)


def test_g4_lessons_background_is_off_for_every_test():
    from birdbrain import lessons
    assert lessons.BACKGROUND is False, "lessons.BACKGROUND must be False in tests (no distill thread outlives a test)"


def test_g4_the_other_named_globals_are_clean_in_the_next_test(leak_probe):
    """The globals that hold a client, a detector or the watcher's memory of its last visit (watch.last_visit_id,
    _last_logged, _hold_cache, circle_check._last_used, identify._client, privacy._det) start clean too."""
    _assert_no_leak_in(leak_probe, G4_OTHER_LEAKS)


def test_g4_clock_zone_and_retention_thread_are_clean_in_the_next_test(leak_probe):
    """clock.set_zone() in one test must not move every later test's clock (until the 60 s recheck), and the
    retention loop's _thread/_wake start at their defined values."""
    _assert_no_leak_in(leak_probe, G4_CLOCK_LEAKS)


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


def _given_timeout(config):
    """The limit this run was started with (--timeout or PYTEST_TIMEOUT), else None."""
    given = config.getoption("timeout", default=None)
    if given is None and os.environ.get("PYTEST_TIMEOUT"):
        given = float(os.environ["PYTEST_TIMEOUT"])
    return given


def test_g8_timeout_applies_when_the_plugin_is_installed(request):
    """With pytest-timeout installed every test gets a 60 s limit; without it the ini must not break the run."""
    if request.config.pluginmanager.hasplugin("timeout"):
        # Any way of applying the limit counts: a `timeout` marker on this item (closest wins), the --timeout
        # option, or the ini key. The contract asks for the 60 s limit, not for one mechanism.
        marker = request.node.get_closest_marker("timeout")
        given = _given_timeout(request.config)
        if given is not None:
            # This run was started with its own limit, which must win over the 60 s default (see the probes below):
            # no default marker may sit on this item, so the plugin reads the given one.
            assert marker is None, f"a timeout marker {marker.args} overrides the run's own --timeout={given}"
            pytest.skip(f"this run sets its own limit ({given} s); the 60 s default is checked by the probes")
        if marker is not None:
            limit = marker.args[0] if marker.args else marker.kwargs.get("timeout")
            source = "timeout marker"
        else:
            limit = request.config.getini("timeout")
            source = "ini"
        assert float(limit or 0) == 60, f"pytest-timeout limit ({source}) is {limit!r}, not 60 s"
    else:
        ini = _ini()
        assert "timeout" not in ini, "a timeout ini key without pytest-timeout installed warns on every run"
        assert not any(o.startswith("--timeout") for o in _addopts()), "--timeout in addopts breaks plain pytest"
        pytest.skip("pytest-timeout is not installed in this environment")


def _effective_timeout(res: subprocess.CompletedProcess) -> str:
    out = res.stdout + res.stderr
    assert res.returncode == 0, out
    m = re.search(r"EFFECTIVE_TIMEOUT=(\S+)$", res.stdout, re.M)
    assert m, "the probe did not report its effective timeout:\n" + out
    return m.group(1)


def test_g8_default_limit_is_60_s_when_the_run_sets_none(request, leak_probe):
    """A test in a run without --timeout, PYTEST_TIMEOUT or an ini key is given 60 s (the limit pytest-timeout
    itself resolves for the item, not just the presence of a marker)."""
    if not request.config.pluginmanager.hasplugin("timeout"):
        pytest.skip("pytest-timeout is not installed in this environment")
    res, out, _ = leak_probe
    value = _effective_timeout(res)
    assert value != "none", "the probe could not import pytest_timeout:\n" + out
    assert float(value) == 60, f"effective limit in a plain run is {value}, not 60 s\n{out}"


@pytest.mark.parametrize("how", ["--timeout=300", "PYTEST_TIMEOUT=300"])
def test_g8_an_explicit_timeout_wins_over_the_60_s_default(request, tmp_path, how):
    """pytest-timeout reads a `timeout` marker before --timeout, PYTEST_TIMEOUT and the ini key, so a 60 s default
    marker on every test would silently defeat `pytest --timeout=300`. A run started with 300 must give its tests
    300, not 60."""
    if not request.config.pluginmanager.hasplugin("timeout"):
        pytest.skip("pytest-timeout is not installed in this environment")
    if how.startswith("--"):
        res = _run_probe(tmp_path, how, select="::test_3_effective_timeout")
    else:
        key, _, value = how.partition("=")
        res = _run_probe(tmp_path, select="::test_3_effective_timeout", env_extra={key: value})
    value = _effective_timeout(res)
    assert value != "none", "the probe could not import pytest_timeout:\n" + res.stdout + res.stderr
    assert float(value) == 300, f"{how} was ignored: the effective limit is {value}, not 300\n{res.stdout}{res.stderr}"


# --- G3: the live.jpg pair in both orders (the known bug) -----------------------------------------------------------

@pytest.mark.skipif(os.environ.get("BIRDBRAIN_SUITE_RULES_SLOW") != "1",
                    reason="about 4 s: run with BIRDBRAIN_SUITE_RULES_SLOW=1; the full-suite gates cover it")
def test_g3_live_snapshot_then_public_mirror_pass_in_the_failing_order():
    res = _run_pytest("tests/test_web.py::test_live_snapshot_is_shared_within_the_refresh_window",
                      "tests/test_read_only.py::test_public_mirror_has_no_admin_and_never_touches_camera")
    assert res.returncode == 0, res.stdout + res.stderr
