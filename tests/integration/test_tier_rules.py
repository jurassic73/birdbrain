"""Rules the three-tier test layout must follow (contract #7 Tiers, T1, T4, T5).

Written blind, before the tests were moved: these tests pin down what the contract promises about the tree
(T1), the tier markers and `-m` selections (T4) and the unit guard in tests/unit/conftest.py (T5, plus the C1
and C2 probes carried into #3: the capfd fixtures and a sleep alias in a module imported late), without knowing how
the builder did it. T4 collects the real repo in pytest subprocesses (one shared run per selection,
all started together); T5 copies the repo's conftests into a throwaway tree under tmp_path, drops probe tests into
its unit and component folders and reads the junit report by test name, so a probe that died for another reason
never counts as "the guard works". The probe tree is run twice, together: whole (both tiers' conftests loaded) and
tests/unit alone, because a unit conftest that only works once some other file has imported the web app passed the
whole-tree run and broke `pytest tests/unit/test_config.py`; one real unit test is also run by itself. Every
subprocess runs with a private TMPDIR and none of this run's own pytest settings (addopts, timeout, xdist) in its
environment. Timing (T8) and coverage (T7) are gates, not tests here.
"""
import os
import re
import shutil
import subprocess
import sys
import tomllib
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TESTS_DIR = REPO_ROOT / "tests"
PYPROJECT = REPO_ROOT / "pyproject.toml"
TIERS = ("unit", "component", "integration")
GUARD_FILE = "test_no_personal_info.py"
NODE_ID = re.compile(r"^(tests/[^\s:]+\.py::\S.*)$", re.M)


# --- helpers --------------------------------------------------------------------------------------------------------

def _ini() -> dict:
    return tomllib.loads(PYPROJECT.read_text())["tool"]["pytest"]["ini_options"]


def _addopts_without_verbosity() -> list[str]:
    """The project's addopts minus its -q/-v flags: `-o addopts=""` drops them all, so anything that changes how the
    suite collects (--strict-markers, an --import-mode) is passed again by hand and only the output shape changes."""
    opts = _ini().get("addopts", "")
    opts = opts.split() if isinstance(opts, str) else list(opts)
    return [o for o in opts if not re.fullmatch(r"-[qv]+|--quiet|--verbose", o)]


def _clean_env(folder: Path) -> dict:
    """A private TMPDIR under `folder` and none of this run's own pytest settings (addopts, a timeout, xdist's worker
    variables, the current test id)."""
    private_tmp = folder / "t"
    private_tmp.mkdir(exist_ok=True)
    env = dict(os.environ, TMPDIR=str(private_tmp), TMP=str(private_tmp), TEMP=str(private_tmp))
    for key in list(env):
        if key in ("PYTEST_ADDOPTS", "PYTEST_TIMEOUT", "PYTEST_CURRENT_TEST") or key.startswith("PYTEST_XDIST_"):
            del env[key]
    return env


def _pytest(folder: Path, *args, rootdir: Path = REPO_ROOT, timeout=120) -> subprocess.CompletedProcess:
    """A pytest session in a subprocess: the repo's pyproject, the project's addopts re-passed without -q (see
    `_addopts_without_verbosity`), no cache dir, no random order, no xdist, a clean environment."""
    cmd = [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-p", "no:randomly", "-p", "no:xdist",
           "--rootdir", str(rootdir), "-c", str(PYPROJECT), "-o", "addopts=", *_addopts_without_verbosity(),
           "--strict-markers", *args]
    return subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=timeout, env=_clean_env(folder))


def _test_files() -> list[Path]:
    """Every test_*.py under tests/ (any depth), relative to the repo."""
    return sorted(p.relative_to(REPO_ROOT) for p in TESTS_DIR.rglob("test_*.py") if "__pycache__" not in p.parts)


def _tier_of(rel: Path) -> str | None:
    """The tier folder a path under tests/ lives in, or None (tests/x.py, tests/helpers/...)."""
    parts = rel.parts
    return parts[1] if len(parts) > 2 and parts[0] == "tests" and parts[1] in TIERS else None


# --- T1: layout -----------------------------------------------------------------------------------------------------

def test_t1_every_tier_folder_exists_and_holds_tests():
    for tier in TIERS:
        folder = TESTS_DIR / tier
        assert folder.is_dir(), f"tests/{tier}/ is missing"
        files = [p for p in _test_files() if _tier_of(p) == tier]
        assert files, f"tests/{tier}/ holds no test files"


def test_t1_every_test_file_lives_under_a_tier_folder():
    """Also the "no test_*.py directly in tests/" rule: a stray tests/test_x.py is named in the message."""
    files = _test_files()
    assert files, "no test files found under tests/"
    outside = [str(p) for p in files if _tier_of(p) is None]
    assert not outside, ("test files outside tests/unit, tests/component and tests/integration (move them into a "
                         "tier folder):\n" + "\n".join(outside))


def _source_for_unit_test(rel: Path, root: Path = REPO_ROOT) -> list[Path]:
    """The source file(s) a unit test file mirrors, derived from its path: tests/unit/test_<m>.py -> birdbrain/<m>.py,
    tests/unit/<sub>/test_<m>.py -> birdbrain/<sub>/<m>.py (so tests/unit/web/ -> birdbrain/web/), except
    tests/unit/scripts/test_<m>.py -> scripts/<m>.* (any extension). A file may also be named test_<m>_<topic>.py
    (T1 as amended): <m> is the longest existing module name the file name starts with, cut at a `_` boundary, so
    test_db_speed.py mirrors db.py while test_dbx.py mirrors nothing. Relative to `root`; [] when no module matches."""
    sub = Path(*rel.parts[2:-1])  # the folders between tests/unit and the file
    parts = rel.stem.removeprefix("test_").split("_")
    for n in range(len(parts), 0, -1):  # longest candidate first
        module = "_".join(parts[:n])
        if sub.parts[:1] == ("scripts",):
            found = sorted(p.relative_to(root) for p in (root / "scripts" / Path(*sub.parts[1:])).glob(f"{module}.*"))
        else:
            candidate = Path("birdbrain") / sub / f"{module}.py"
            found = [candidate] if (root / candidate).is_file() else []
        if found:
            return found
    return []


def test_t1_unit_tests_mirror_the_source_tree():
    """Every tests/unit/**/test_<m>[_<topic>].py has the source file its path names (see `_source_for_unit_test`),
    and the mirror is checked against the real tree, not a list: a unit file whose name starts with no module in
    its folder, or put in the wrong subfolder, shows up here."""
    unit_files = [p for p in _test_files() if _tier_of(p) == "unit"]
    assert unit_files, "tests/unit/ holds no test files"
    orphans = [f"{p}: no {_expected_source_folder(p)}<m>.* that {p.stem.removeprefix('test_')} starts with"
               for p in unit_files if not _source_for_unit_test(p)]
    assert not orphans, "unit test files with no matching source module:\n" + "\n".join(orphans)


@pytest.mark.parametrize("rel, source", [
    ("tests/unit/test_db.py", "birdbrain/db.py"),
    ("tests/unit/web/test_app.py", "birdbrain/web/app.py"),
    ("tests/unit/scripts/test_readenv.py", "scripts/readenv.py"),
])
def test_t1_mirror_rule_maps_each_source_folder(rel, source):
    """The mapping used above, on paths whose sources exist on every branch."""
    assert _source_for_unit_test(Path(rel)) == [Path(source)]


@pytest.mark.parametrize("rel, source", [
    ("tests/unit/test_db_speed.py", "birdbrain/db.py"),
    ("tests/unit/web/test_app_routes.py", "birdbrain/web/app.py"),
    ("tests/unit/scripts/test_readenv_quoting.py", "scripts/readenv.py"),
])
def test_t1_mirror_rule_maps_a_topic_file_to_its_module(rel, source):
    """T1 as amended: test_<m>_<topic>.py mirrors <m> in the same folder."""
    assert _source_for_unit_test(Path(rel)) == [Path(source)]


def test_t1_mirror_rule_takes_the_longest_module_name(tmp_path):
    """Where two modules share a prefix the longer match wins; the real tree has no such pair, so a made-up one."""
    (tmp_path / "birdbrain").mkdir()
    (tmp_path / "birdbrain" / "db.py").write_text("")
    (tmp_path / "birdbrain" / "db_speed.py").write_text("")
    assert _source_for_unit_test(Path("tests/unit/test_db_speed.py"), tmp_path) == [Path("birdbrain/db_speed.py")]
    assert _source_for_unit_test(Path("tests/unit/test_db_speed_rows.py"), tmp_path) == [Path("birdbrain/db_speed.py")]
    assert _source_for_unit_test(Path("tests/unit/test_db_rows.py"), tmp_path) == [Path("birdbrain/db.py")]


@pytest.mark.parametrize("rel", [
    "tests/unit/test_nosuchmodule.py",
    "tests/unit/test_nosuchmodule_topic.py",
    "tests/unit/test_app.py",  # web module in the wrong folder
    "tests/unit/test_dbx.py",  # the prefix must end at a `_` boundary
    "tests/unit/test_dbx_speed.py",
    "tests/unit/scripts/test_readenvx.py",
])
def test_t1_mirror_rule_rejects_a_name_matching_no_module(rel):
    assert _source_for_unit_test(Path(rel)) == []


def test_t1_there_is_exactly_one_guard_file():
    """(That it sits in a tier folder is `test_t1_every_test_file_lives_under_a_tier_folder`'s job.)"""
    found = [p for p in _test_files() if p.name == GUARD_FILE]
    assert len(found) == 1, f"expected exactly one {GUARD_FILE} under tests/, found {found}"


# --- T4: markers and -m selections ----------------------------------------------------------------------------------

def _collect(folder: Path, *args) -> tuple[int, set[str], str]:
    """`pytest --collect-only -q` on the real repo: (exit code, node ids, whole output). Exit 5 means nothing was
    selected, which is a legitimate answer the tests below judge for themselves."""
    res = _pytest(folder, "--collect-only", "-q", *args)
    out = res.stdout + res.stderr
    return res.returncode, set(NODE_ID.findall(res.stdout)), out


SELECTIONS = {"all": (), "unit": ("-m", "unit"), "component": ("-m", "component"),
              "integration": ("-m", "integration"), "unit_or_guard": ("-m", "unit or guard"),
              "guard": ("-m", "guard")}


@pytest.fixture(scope="module")
def collected(tmp_path_factory):
    """One collect-only run per selection in SELECTIONS, started together (each ~1.5 s): name -> node ids. A run
    that failed to collect (an import error, a duplicate basename clash, an unknown marker) fails here with its
    output, so nothing below reads a broken collection as "no items"."""
    base = tmp_path_factory.mktemp("collect")
    folders = {name: base / name for name in SELECTIONS}
    for f in folders.values():
        f.mkdir()
    with ThreadPoolExecutor(max_workers=len(SELECTIONS)) as pool:
        futures = {name: pool.submit(_collect, folders[name], *args) for name, args in SELECTIONS.items()}
        results = {name: fut.result() for name, fut in futures.items()}
    for name, (code, ids, out) in results.items():
        tail = "\n".join(out.splitlines()[-40:])  # the errors come after the node ids
        assert code in (0, 5), f"collection with {SELECTIONS[name]} failed (exit {code}), output ends:\n{tail}"
        assert (code == 0) == bool(ids), f"exit {code} but {len(ids)} node ids parsed for {name}:\n{tail}"
    return {name: ids for name, (code, ids, out) in results.items()}


def _under(ids: set[str], tier: str) -> set[str]:
    return {i for i in ids if i.startswith(f"tests/{tier}/")}


def test_t4_m_unit_or_guard_adds_exactly_the_guard_tests(collected):
    """Also the guard-marker rule: every item of the guard file carries `guard` (so `-m "not guard"` would leave
    none of it) and nothing else does."""
    guard = {i for i in collected["all"] if Path(i.split("::")[0]).name == GUARD_FILE}
    assert guard, f"no collected items from {GUARD_FILE}"
    assert collected["unit"], "no items selected by -m unit"
    assert collected["unit_or_guard"] - collected["unit"] == guard - collected["unit"], (
        f"-m 'unit or guard' adds {sorted(collected['unit_or_guard'] - collected['unit'] - guard)[:10]} beyond the "
        f"guard tests, or misses {sorted(guard - collected['unit_or_guard'])[:10]}")
    assert collected["guard"] == guard, f"-m guard selected {sorted(collected['guard'] ^ guard)[:10]} unexpectedly"


def test_t4_every_collected_item_has_exactly_one_tier_marker(collected):
    """The three `-m <tier>` selections partition the whole collection: together they are every item, and no item
    is in two of them (an item with no tier marker is missing from all three; one with two is in two)."""
    everything = collected["all"]
    assert everything, "nothing collected"
    tiers = {t: collected[t] for t in TIERS}
    unmarked = everything - set().union(*tiers.values())
    assert not unmarked, f"{len(unmarked)} items carry no tier marker, e.g. {sorted(unmarked)[:10]}"
    for a in TIERS:
        for b in TIERS:
            if a < b:
                both = tiers[a] & tiers[b]
                assert not both, f"items marked both {a} and {b}: {sorted(both)[:10]}"
    for tier, ids in tiers.items():
        assert ids, f"no collected items under tests/{tier}/"
        assert ids == _under(everything, tier), (
            f"-m {tier}: {sorted(ids - _under(everything, tier))[:10]} selected but not under tests/{tier}; "
            f"{sorted(_under(everything, tier) - ids)[:10]} under tests/{tier} but not selected")


# --- T5: the unit guard ---------------------------------------------------------------------------------------------

# The unit probe must not contain the word "comp" + "onent" anywhere (its source shows in tracebacks), so that word
# in a failure message can only have come from the guard. Paths for the write probes come from the private TMPDIR,
# not tmp_path, so a guard that also refuses mkdir cannot error the probes at setup.
UNIT_PROBE = '''"""Probe tests dropped into the throwaway tree's unit folder by tests/integration/test_tier_rules.py."""
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from birdbrain import config, species_info


def _target(name):
    return Path(tempfile.gettempdir()) / f"unit-probe-{os.getpid()}-{name}"


def test_u_sqlite_connect():
    c = sqlite3.connect(":memory:")
    c.close()


def test_u_sleep_positive():
    time.sleep(0.01)


def test_u_sleep_alias_in_a_birdbrain_module():
    species_info._sleep(0.01)  # bound to time.sleep at import, so patching time.sleep alone misses it


def test_u_subprocess_run():
    subprocess.run([sys.executable, "-c", "pass"], check=True)


def test_u_thread_start():
    t = threading.Thread(target=lambda: None)
    t.start()
    t.join()


def test_u_open_for_writing():
    with open(_target("open.txt"), "w") as f:
        f.write("x")


def test_u_os_open_for_writing():
    os.close(os.open(_target("osopen.txt"), os.O_WRONLY | os.O_CREAT))


def test_u_path_touch():
    _target("touch.txt").touch()


def test_u_tempfile_mkstemp():
    fd, _ = tempfile.mkstemp(prefix="unit-probe-")
    os.close(fd)


def test_u_path_write_text():
    _target("text.txt").write_text("x")


def test_u_path_write_bytes():
    _target("bytes.bin").write_bytes(b"x")


def test_u_cv2_imwrite():
    assert cv2.imwrite(str(_target("frame.jpg")), np.zeros((4, 4, 3), np.uint8))


def test_u_path_mkdir():
    _target("dir1").mkdir()


def test_u_os_mkdir():
    os.mkdir(_target("dir2"))


def test_u_os_makedirs():
    os.makedirs(_target("dir3") / "deep")


def test_u_os_remove():
    os.remove(_target("gone1"))


def test_u_os_unlink():
    os.unlink(_target("gone2"))


def test_u_os_rename():
    os.rename(_target("from1"), _target("to1"))


def test_u_os_replace():
    os.replace(_target("from2"), _target("to2"))


def test_u_os_rmdir():
    os.rmdir(_target("gone3"))


def test_u_shutil_rmtree():
    shutil.rmtree(_target("gone4"))


def test_u_os_system():
    os.system("true")


def test_u_tmp_path_fixture(tmp_path):
    pass


def test_u_tmp_path_factory_fixture(tmp_path_factory):
    pass


def test_u_tmpdir_fixture(tmpdir):
    pass


def test_u_tmpdir_factory_fixture(tmpdir_factory):
    pass


def test_u_capfd_fixture(capfd):
    pass


def test_u_capfdbinary_fixture(capfdbinary):
    pass


def test_u_pure_work_after_the_capture_fixtures_is_fine():
    # Collected right after the capfd probes: a refused fixture fails that one test, not every later one.
    assert sorted({3, 1, 2}) == [1, 2, 3]


def test_u_sleep_zero_is_fine():
    time.sleep(0)


def test_u_pure_work_is_fine():
    assert sorted({3, 1, 2}) == [1, 2, 3]
    assert sum(range(10)) == 45


def test_u_reading_a_file_is_fine():
    assert "test_u_reading_a_file_is_fine" in Path(__file__).read_text()


def test_u_os_open_read_only_is_fine():
    fd = os.open(__file__, os.O_RDONLY)
    assert b"test_u_os_open_read_only_is_fine" in os.read(fd, 1 << 20)
    os.close(fd)


def test_u_data_dir_does_not_exist():
    missing = [str(p) for p in (config.DATA_DIR, config.IMAGES_DIR, config.DB_PATH) if p.exists()]
    assert not missing, f"a unit test has a data folder: {missing}"
'''

# A birdbrain module the probe tree imports only after the first unit test has run (see LATE_PROBE): like
# species_info._sleep it binds time.sleep at import, so a guard that scanned birdbrain's modules once, at the first
# unit test, never sees it.
LATE_MODULE = '''"""Written into the throwaway tree by tests/integration/test_tier_rules.py and imported late."""
import time

_sleep = time.sleep
'''

# Sorts after test_probe_unit_rules.py, so its first test runs after the guard has already done its first scan.
LATE_PROBE = '''"""Probe dropped into the throwaway tree's unit folder by tests/integration/test_tier_rules.py, collected after
test_probe_unit_rules.py. Its module-scoped fixture runs before this file's first test's own guard fixtures (higher
scope first), when time.sleep is still the real one, and imports a birdbrain module that binds `_sleep = time.sleep`:
a module first imported after the first unit test of the session."""
import importlib
import sys
from pathlib import Path

import pytest

import birdbrain

LATE_SOURCE = Path(__file__).resolve().parents[2] / "late_modules"


@pytest.fixture(scope="module")
def late_module():
    assert "birdbrain.probe_late_sleep" not in sys.modules, "the late module must not be imported before this fixture"
    birdbrain.__path__.append(str(LATE_SOURCE))
    return importlib.import_module("birdbrain.probe_late_sleep")


def test_u_sleep_alias_in_a_module_imported_late(late_module):
    late_module._sleep(0.01)


def test_u_pure_work_after_the_late_import_is_fine():
    assert sum(range(10)) == 45
'''

COMPONENT_PROBE = '''"""Probe tests dropped into the throwaway tree's component folder by tests/integration/test_tier_rules.py."""
import sqlite3
import subprocess
import sys
import time

from birdbrain import config


def test_c_sqlite_connect_to_a_tmp_file(tmp_path):
    c = sqlite3.connect(tmp_path / "probe.db")
    c.execute("create table t (x)")
    c.close()


def test_c_writes_under_tmp_path(tmp_path):
    with open(tmp_path / "a.txt", "w") as f:
        f.write("a")
    (tmp_path / "b.txt").write_text("b")
    (tmp_path / "c.bin").write_bytes(b"c")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a.txt", "b.txt", "c.bin"]


def test_c_sleep_and_subprocess():
    time.sleep(0.001)
    subprocess.run([sys.executable, "-c", "pass"], check=True)


def test_c_data_dir_is_a_fresh_folder():
    assert config.DATA_DIR.is_dir(), f"no data folder: {config.DATA_DIR}"
    (config.DATA_DIR / "probe.txt").write_text("x")
'''

UNIT_MUST_FAIL = ("test_u_sqlite_connect", "test_u_sleep_positive", "test_u_sleep_alias_in_a_birdbrain_module",
                  "test_u_subprocess_run", "test_u_thread_start",
                  "test_u_open_for_writing", "test_u_os_open_for_writing", "test_u_path_touch",
                  "test_u_tempfile_mkstemp", "test_u_path_write_text", "test_u_path_write_bytes", "test_u_cv2_imwrite",
                  "test_u_path_mkdir", "test_u_os_mkdir", "test_u_os_makedirs",
                  "test_u_os_remove", "test_u_os_unlink", "test_u_os_rename", "test_u_os_replace", "test_u_os_rmdir",
                  "test_u_shutil_rmtree", "test_u_os_system",
                  "test_u_tmp_path_fixture", "test_u_tmp_path_factory_fixture", "test_u_tmpdir_fixture",
                  "test_u_tmpdir_factory_fixture",
                  "test_u_capfd_fixture", "test_u_capfdbinary_fixture",  # C1
                  "test_u_sleep_alias_in_a_module_imported_late")  # C2, from LATE_PROBE
CAPTURE_PROBES = ("test_u_capfd_fixture", "test_u_capfdbinary_fixture")
AFTER_CAPTURE = "test_u_pure_work_after_the_capture_fixtures_is_fine"
LATE_ALIAS = "test_u_sleep_alias_in_a_module_imported_late"
AFTER_LATE = "test_u_pure_work_after_the_late_import_is_fine"
UNIT_MUST_PASS = ("test_u_sleep_zero_is_fine", "test_u_pure_work_is_fine", "test_u_reading_a_file_is_fine",
                  "test_u_os_open_read_only_is_fine", "test_u_data_dir_does_not_exist", AFTER_CAPTURE, AFTER_LATE)
COMPONENT_MUST_PASS = ("test_c_sqlite_connect_to_a_tmp_file", "test_c_writes_under_tmp_path",
                       "test_c_sleep_and_subprocess", "test_c_data_dir_is_a_fresh_folder")
UNIT_PROBES = UNIT_MUST_FAIL + UNIT_MUST_PASS
ALL_PROBES = UNIT_PROBES + COMPONENT_MUST_PASS


def _build_probe_tree(tree: Path) -> None:
    """tests/ under `tree` with the repo's real conftest.py files and __init__.py files (if the suite uses packages)
    at the same relative places, the unit probes in tests/unit/ (the late-alias probe sorts after the main one), the
    module the late-alias probe imports in late_modules/ and the component probe in tests/component/."""
    for src in sorted(TESTS_DIR.rglob("*.py")):
        rel = src.relative_to(REPO_ROOT)
        if "__pycache__" in rel.parts or rel.parts[1:2] == ("helpers",):
            continue
        if src.name in ("conftest.py", "__init__.py"):
            dst = tree / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, dst)
    for tier in ("unit", "component"):
        (tree / "tests" / tier).mkdir(parents=True, exist_ok=True)
        if (TESTS_DIR / "__init__.py").exists():
            (tree / "tests" / tier / "__init__.py").touch()
    (tree / "tests" / "unit" / "test_probe_unit_rules.py").write_text(UNIT_PROBE)
    (tree / "tests" / "unit" / "test_probe_unit_z_late_alias.py").write_text(LATE_PROBE)
    (tree / "late_modules").mkdir()
    (tree / "late_modules" / "probe_late_sleep.py").write_text(LATE_MODULE)
    (tree / "tests" / "component" / "test_probe_in_place.py").write_text(COMPONENT_PROBE)


def _junit_outcomes(report: Path) -> dict[str, tuple[str, str]]:
    """Test name -> (outcome, message) from a junit report, in the order the tests ran; the message is the failure
    or error's short message plus its full text."""
    outcomes = {}
    for case in ET.parse(report).getroot().iter("testcase"):
        outcome, message = "passed", ""
        for kind in ("error", "failure", "skipped"):
            node = case.find(kind)
            if node is not None:
                outcome = kind
                message = (node.get("message") or "") + "\n" + (node.text or "")
                break
        outcomes[case.get("name")] = (outcome, message)
    return outcomes


def _run_probe_tree(folder: Path, tree: Path, *paths: str, expected: tuple[str, ...]) -> tuple[dict, str]:
    """One pytest run of the probe tree on `paths` (relative to `tree`): name -> (outcome, message) for every probe
    it ran, plus the run's output. The run must have collected and run exactly `expected` (collection errors, a
    conftest that fails to import, or a missing probe fail here with the whole output)."""
    folder.mkdir(exist_ok=True)
    report = folder / "junit.xml"
    res = _pytest(folder, "-q", "--junitxml", str(report), *(str(tree / p) for p in paths), rootdir=tree)
    out = res.stdout + res.stderr
    leaked = ("" if "unit tests are pure" not in out else
              " (the guard's refusal is in pytest's own traceback: a patch leaked past the test that was refused, "
              "which is what C1 fixes for capfd)")
    assert report.exists(), f"the probe run on {paths} wrote no junit report (exit {res.returncode}){leaked}:\n{out}"
    outcomes = _junit_outcomes(report)
    assert set(outcomes) == set(expected), (
        f"the probe run on {paths} did not run exactly the expected probes: "
        f"missing {sorted(set(expected) - set(outcomes))}, unexpected {sorted(set(outcomes) - set(expected))}\n{out}")
    # Sanity: pytest's own summary is in the output. An all-errors run says "error(s)" and nothing else, and must
    # still get through to the per-probe messages below (C3).
    assert any(word in out for word in ("collected", "passed", "failed", "error")), out
    return outcomes, out


@pytest.fixture(scope="module")
def unit_guard_probe(tmp_path_factory):
    """The probe tree run twice, together: "whole" is `pytest tests` (both tiers' conftests loaded, the component
    probes included) and "unit_alone" is `pytest tests/unit` (only the base and unit conftests, as when one unit
    file is run by itself). Each is (name -> (outcome, message), the run's output)."""
    folder = tmp_path_factory.mktemp("unit_guard")
    tree = folder / "repo"
    _build_probe_tree(tree)
    with ThreadPoolExecutor(max_workers=2) as pool:
        whole = pool.submit(_run_probe_tree, folder / "whole", tree, "tests", expected=ALL_PROBES)
        alone = pool.submit(_run_probe_tree, folder / "alone", tree, "tests/unit", expected=UNIT_PROBES)
        results = {"whole": whole.result(), "unit_alone": alone.result()}
    assert not any((TESTS_DIR / tier / "test_probe_unit_rules.py").exists() for tier in TIERS), \
        "the probe tree must never be written inside the repo"
    return results


def _guard_message(message: str) -> bool:
    text = message.lower()
    return "unit" in text and "component" in text


def _not_stopped_by_the_guard(outcomes: dict, probe: str) -> str:
    """"" when the probe failed or errored with the guard's message, else what went wrong instead."""
    outcome, message = outcomes[probe]
    if outcome not in ("failure", "error"):
        return f"{outcome} (the guard let it through)"
    if not _guard_message(message):
        return f"failed, but not with the guard's message (it should say the test is not pure and belongs in the " \
               f"component tier):\n{message}"
    return ""


@pytest.mark.parametrize("probe", UNIT_MUST_FAIL)
def test_t5_impure_call_fails_a_unit_test(unit_guard_probe, probe):
    """sqlite3.connect, time.sleep(0.01) (also through a birdbrain module's import-time alias of it), subprocess.run,
    Thread.start, a file opened for writing (open, os.open with a write or create flag, Path.touch, tempfile.mkstemp,
    Path.write_text/write_bytes, cv2.imwrite), a new folder (Path.mkdir, os.mkdir, os.makedirs), a file removed,
    renamed or replaced (os.remove/unlink/rename/replace/rmdir, shutil.rmtree), os.system, the tmp_path family of
    fixtures, the capfd and capfdbinary fixtures (C1) and a sleep through the alias of a birdbrain module imported
    after the first unit test (C2) each fail the unit test that uses them, with a message that says the test is not a
    unit test and where it belongs ("unit" and "component" both in it)."""
    outcomes, out = unit_guard_probe["whole"]
    problem = _not_stopped_by_the_guard(outcomes, probe)
    assert not problem, f"{probe} in tests/unit: {problem}\n{out}"


def test_t5_pure_unit_probes_pass(unit_guard_probe):
    """time.sleep(0), pure work and reading a file (open and os.open read-only) are allowed in a unit test."""
    outcomes, out = unit_guard_probe["whole"]
    failed = {p: outcomes[p] for p in ("test_u_sleep_zero_is_fine", "test_u_pure_work_is_fine",
                                       "test_u_reading_a_file_is_fine", "test_u_os_open_read_only_is_fine",
                                       AFTER_CAPTURE, AFTER_LATE)
              if outcomes[p][0] != "passed"}
    assert not failed, "pure probes did not pass in tests/unit:\n" + "\n".join(f"{k}: {v}" for k, v in failed.items())


def test_t5_unit_tests_get_no_data_folder(unit_guard_probe):
    """config.DATA_DIR, IMAGES_DIR and DB_PATH point at paths that do not exist during a unit test."""
    outcomes, out = unit_guard_probe["whole"]
    outcome, message = outcomes["test_u_data_dir_does_not_exist"]
    assert outcome == "passed", f"a unit test saw a data folder:\n{message}"


def _ran_after(outcomes: dict, later: str, earlier: str) -> bool:
    order = list(outcomes)
    return order.index(later) > order.index(earlier)


@pytest.mark.parametrize("run", ["whole", "unit_alone"])
def test_t5_capture_fixtures_are_refused_without_breaking_the_tests_after_them(unit_guard_probe, run):
    """C1: a unit test asking for capfd or capfdbinary fails alone, with the guard's message, and the plain pure probe
    collected right after the two (plus every other probe that runs later in the session) still passes. Before the
    fix pytest's capfd hit the guarded open inside a capture hook, which errored every later test and left the
    guard's patches applied."""
    outcomes, out = unit_guard_probe[run]
    wrong = {p: _not_stopped_by_the_guard(outcomes, p) for p in CAPTURE_PROBES}
    wrong = {p: why for p, why in wrong.items() if why}
    assert not wrong, f"capture fixtures in tests/unit ({run}):\n" + "\n".join(f"{k}: {v}" for k, v in wrong.items())
    for probe in CAPTURE_PROBES:
        assert _ran_after(outcomes, AFTER_CAPTURE, probe), f"{AFTER_CAPTURE} must be collected after {probe}"
    later = [p for p in list(outcomes)[list(outcomes).index(AFTER_CAPTURE):] if p in UNIT_MUST_PASS]
    assert AFTER_CAPTURE in later
    broken = {p: outcomes[p] for p in later if outcomes[p][0] != "passed"}
    assert not broken, (f"pure probes after the capfd probes did not pass ({run}): the refused fixture broke the "
                        f"tests after it:\n" + "\n".join(f"{k}: {v}" for k, v in broken.items()) + f"\n{out}")


@pytest.mark.parametrize("run", ["whole", "unit_alone"])
def test_t5_a_sleep_alias_in_a_module_imported_after_the_first_unit_test_is_refused(unit_guard_probe, run):
    """C2: the guard's scan for import-time aliases of time.sleep (species_info._sleep) must also cover a birdbrain
    module first imported after the first unit test has run (LATE_PROBE imports one in a module-scoped fixture, so
    nothing has scanned it yet); a scan made once per session let `_sleep(0.01)` through. The pure probe after it
    still passes."""
    outcomes, out = unit_guard_probe[run]
    assert _ran_after(outcomes, LATE_ALIAS, "test_u_sleep_positive"), \
        f"the late-alias probe must run after the first unit probes, order was {list(outcomes)}"
    problem = _not_stopped_by_the_guard(outcomes, LATE_ALIAS)
    assert not problem, f"{LATE_ALIAS} in tests/unit ({run}): {problem}\n{out}"
    assert outcomes[AFTER_LATE][0] == "passed", f"{AFTER_LATE} ({run}): {outcomes[AFTER_LATE]}"


def test_t5_the_unit_folder_alone_behaves_the_same(unit_guard_probe):
    """`pytest tests/unit` with nothing else collected (no component conftest, so nothing has imported the web app
    before the unit conftest's fixtures run) gives every unit probe the same verdict as the whole-tree run: the
    impure ones are stopped by the guard, the pure ones pass, the data folder does not exist. A unit conftest that
    only worked after some other file had imported the app errored every probe here (OSError on a path that does
    not exist) while the whole-tree run looked fine."""
    outcomes, out = unit_guard_probe["unit_alone"]
    wrong = {p: _not_stopped_by_the_guard(outcomes, p) for p in UNIT_MUST_FAIL}
    wrong = {p: why for p, why in wrong.items() if why}
    wrong.update({p: outcomes[p] for p in UNIT_MUST_PASS if outcomes[p][0] != "passed"})
    assert not wrong, ("probes behave differently when tests/unit runs alone:\n"
                       + "\n".join(f"{k}: {v}" for k, v in wrong.items()) + f"\n{out}")


def test_t5_one_real_unit_test_runs_on_its_own(collected, tmp_path):
    """The first unit test of the real repo, run as the only item of a session, passes: no unit test may need another
    file's import or fixture to have run before it."""
    node = min(collected["unit"])
    res = _pytest(tmp_path, "-q", node)
    out = res.stdout + res.stderr
    assert res.returncode == 0, f"{node} fails when run on its own (exit {res.returncode}):\n{out}"
    assert re.search(r"\b1 (passed|skipped)\b", out), f"{node} did not run as the one item of the session:\n{out}"


def test_t5_component_tests_are_unaffected(unit_guard_probe):
    """In tests/component the same calls work: sqlite to a tmp file, writes under tmp_path, sleep and subprocess,
    and the per-test data folder is there to write into."""
    outcomes, out = unit_guard_probe["whole"]
    failed = {p: outcomes[p] for p in COMPONENT_MUST_PASS if outcomes[p][0] != "passed"}
    assert not failed, "component probes did not pass:\n" + "\n".join(f"{k}: {v}" for k, v in failed.items())
