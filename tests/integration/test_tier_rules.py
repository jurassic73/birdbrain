"""Rules the three-tier test layout must follow (contract #7 Tiers, T1, T4, T5).

Written blind, before the tests were moved: these tests pin down what the contract promises about the tree
(T1), the tier markers and `-m` selections (T4) and the unit guard in tests/unit/conftest.py (T5), without
knowing how the builder did it. T4 collects the real repo in pytest subprocesses (one shared run per selection,
all started together); T5 copies the repo's conftests into a throwaway tree under tmp_path, drops probe tests into
its unit and component folders and reads the junit report by test name, so a probe that died for another reason
never counts as "the guard works". Every subprocess runs with a private TMPDIR and none of this run's own pytest
settings (addopts, timeout, xdist) in its environment. Timing (T8) and coverage (T7) are gates, not tests here.
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

def test_t1_no_test_file_sits_directly_in_tests():
    stray = sorted(p.name for p in TESTS_DIR.glob("test_*.py"))
    assert not stray, f"test files still directly in tests/ (move them into a tier folder): {stray}"


def test_t1_every_tier_folder_exists_and_holds_tests():
    for tier in TIERS:
        folder = TESTS_DIR / tier
        assert folder.is_dir(), f"tests/{tier}/ is missing"
        files = [p for p in _test_files() if _tier_of(p) == tier]
        assert files, f"tests/{tier}/ holds no test files"


def test_t1_every_test_file_lives_under_a_tier_folder():
    files = _test_files()
    assert files, "no test files found under tests/"
    outside = [str(p) for p in files if _tier_of(p) is None]
    assert not outside, "test files outside tests/unit, tests/component and tests/integration:\n" + "\n".join(outside)


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


def test_t1_guard_file_lives_in_a_tier_folder():
    found = [p for p in _test_files() if p.name == GUARD_FILE]
    assert len(found) == 1, f"expected exactly one {GUARD_FILE} under tests/, found {found}"
    assert _tier_of(found[0]) is not None, f"{found[0]} is not in a tier folder"


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


def test_t4_m_unit_selects_exactly_the_items_under_tests_unit(collected):
    expected = _under(collected["all"], "unit")
    assert expected, "no collected items under tests/unit/"
    assert collected["unit"] == expected, (
        f"-m unit: {sorted(collected['unit'] - expected)[:10]} selected but not under tests/unit; "
        f"{sorted(expected - collected['unit'])[:10]} under tests/unit but not selected")


def test_t4_m_unit_or_guard_adds_exactly_the_guard_tests(collected):
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
        assert ids == _under(everything, tier), f"-m {tier} is not exactly the items under tests/{tier}/"


def test_t4_tier_markers_in_this_session_follow_the_folder(request):
    """Cheap in-process check of the same rule on whatever this run collected: each item's tier marker is the one
    its folder names, and there is exactly one."""
    tests_dir = TESTS_DIR.resolve()
    wrong = []
    for item in request.session.items:
        marks = {m.name for m in item.iter_markers() if m.name in TIERS}
        try:
            rel = Path(str(item.path)).resolve().relative_to(tests_dir)
        except ValueError:
            rel = None
        want = rel.parts[0] if rel is not None and rel.parts and rel.parts[0] in TIERS else None
        if marks != ({want} if want else set()):
            wrong.append(f"{item.nodeid}: folder says {want}, markers {sorted(marks)}")
    assert not wrong, "items whose tier markers do not match their folder:\n" + "\n".join(wrong[:20])


# --- T5: the unit guard ---------------------------------------------------------------------------------------------

# The unit probe must not contain the word "comp" + "onent" anywhere (its source shows in tracebacks), so that word
# in a failure message can only have come from the guard. Paths for the write probes come from the private TMPDIR,
# not tmp_path, so a guard that also refuses mkdir cannot error the probes at setup.
UNIT_PROBE = '''"""Probe tests dropped into the throwaway tree's unit folder by tests/integration/test_tier_rules.py."""
import os
import sqlite3
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from birdbrain import config


def _target(name):
    return Path(tempfile.gettempdir()) / f"unit-probe-{os.getpid()}-{name}"


def test_u_sqlite_connect():
    c = sqlite3.connect(":memory:")
    c.close()


def test_u_sleep_positive():
    time.sleep(0.01)


def test_u_subprocess_run():
    subprocess.run([sys.executable, "-c", "pass"], check=True)


def test_u_open_for_writing():
    with open(_target("open.txt"), "w") as f:
        f.write("x")


def test_u_path_write_text():
    _target("text.txt").write_text("x")


def test_u_path_write_bytes():
    _target("bytes.bin").write_bytes(b"x")


def test_u_sleep_zero_is_fine():
    time.sleep(0)


def test_u_pure_work_is_fine():
    assert sorted({3, 1, 2}) == [1, 2, 3]
    assert sum(range(10)) == 45


def test_u_reading_a_file_is_fine():
    assert "test_u_reading_a_file_is_fine" in Path(__file__).read_text()


def test_u_data_dir_does_not_exist():
    missing = [str(p) for p in (config.DATA_DIR, config.IMAGES_DIR, config.DB_PATH) if p.exists()]
    assert not missing, f"a unit test has a data folder: {missing}"
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

UNIT_MUST_FAIL = ("test_u_sqlite_connect", "test_u_sleep_positive", "test_u_subprocess_run",
                  "test_u_open_for_writing", "test_u_path_write_text", "test_u_path_write_bytes")
UNIT_MUST_PASS = ("test_u_sleep_zero_is_fine", "test_u_pure_work_is_fine", "test_u_reading_a_file_is_fine",
                  "test_u_data_dir_does_not_exist")
COMPONENT_MUST_PASS = ("test_c_sqlite_connect_to_a_tmp_file", "test_c_writes_under_tmp_path",
                       "test_c_sleep_and_subprocess", "test_c_data_dir_is_a_fresh_folder")
ALL_PROBES = UNIT_MUST_FAIL + UNIT_MUST_PASS + COMPONENT_MUST_PASS


def _build_probe_tree(tree: Path) -> None:
    """tests/ under `tree` with the repo's real conftest.py files and __init__.py files (if the suite uses packages)
    at the same relative places, the unit probe in tests/unit/ and the component probe in tests/component/."""
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
    (tree / "tests" / "component" / "test_probe_in_place.py").write_text(COMPONENT_PROBE)


def _junit_outcomes(report: Path) -> dict[str, tuple[str, str]]:
    """Test name -> (outcome, message) from a junit report; the message is the failure or error's short message
    plus its full text."""
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


@pytest.fixture(scope="module")
def unit_guard_probe(tmp_path_factory):
    """One run of the probe tree for this module: name -> (outcome, message) for every probe, plus the run's output.
    The run must have collected and run exactly the probes listed in ALL_PROBES (collection errors, a conftest that
    fails to import, or a missing probe fail here with the whole output)."""
    folder = tmp_path_factory.mktemp("unit_guard")
    tree = folder / "repo"
    _build_probe_tree(tree)
    report = folder / "junit.xml"
    res = _pytest(folder, "-q", "--junitxml", str(report), str(tree / "tests"), rootdir=tree)
    out = res.stdout + res.stderr
    assert report.exists(), f"the probe run wrote no junit report (exit {res.returncode}):\n{out}"
    outcomes = _junit_outcomes(report)
    assert set(outcomes) == set(ALL_PROBES), (
        f"the probe run did not run exactly the expected probes: missing {sorted(set(ALL_PROBES) - set(outcomes))}, "
        f"unexpected {sorted(set(outcomes) - set(ALL_PROBES))}\n{out}")
    assert "collected" in out or "passed" in out or "failed" in out, out
    assert not any((TESTS_DIR / tier / "test_probe_unit_rules.py").exists() for tier in TIERS), \
        "the probe tree must never be written inside the repo"
    return outcomes, out


def test_t5_unit_conftest_exists():
    assert (TESTS_DIR / "unit" / "conftest.py").is_file(), "tests/unit/conftest.py (the unit guard) is missing"


def _guard_message(message: str) -> bool:
    text = message.lower()
    return "unit" in text and "component" in text


@pytest.mark.parametrize("probe", UNIT_MUST_FAIL)
def test_t5_impure_call_fails_a_unit_test(unit_guard_probe, probe):
    """sqlite3.connect, time.sleep(0.01), subprocess.run, open(..., "w"), Path.write_text and Path.write_bytes
    each fail the unit test that makes them, with a message that says the test is not a unit test and where it
    belongs ("unit" and "component" both in it)."""
    outcomes, out = unit_guard_probe
    outcome, message = outcomes[probe]
    assert outcome in ("failure", "error"), f"{probe} {outcome} in tests/unit (the guard let it through):\n{out}"
    assert _guard_message(message), (f"{probe} failed, but not with the guard's message (it should say the test "
                                     f"is not pure and belongs in the component tier):\n{message}")


def test_t5_pure_unit_probes_pass(unit_guard_probe):
    """time.sleep(0), pure work and reading a file are allowed in a unit test."""
    outcomes, out = unit_guard_probe
    failed = {p: outcomes[p] for p in ("test_u_sleep_zero_is_fine", "test_u_pure_work_is_fine",
                                       "test_u_reading_a_file_is_fine") if outcomes[p][0] != "passed"}
    assert not failed, "pure probes did not pass in tests/unit:\n" + "\n".join(f"{k}: {v}" for k, v in failed.items())


def test_t5_unit_tests_get_no_data_folder(unit_guard_probe):
    """config.DATA_DIR, IMAGES_DIR and DB_PATH point at paths that do not exist during a unit test."""
    outcomes, out = unit_guard_probe
    outcome, message = outcomes["test_u_data_dir_does_not_exist"]
    assert outcome == "passed", f"a unit test saw a data folder:\n{message}"


def test_t5_component_tests_are_unaffected(unit_guard_probe):
    """In tests/component the same calls work: sqlite to a tmp file, writes under tmp_path, sleep and subprocess,
    and the per-test data folder is there to write into."""
    outcomes, out = unit_guard_probe
    failed = {p: outcomes[p] for p in COMPONENT_MUST_PASS if outcomes[p][0] != "passed"}
    assert not failed, "component probes did not pass:\n" + "\n".join(f"{k}: {v}" for k, v in failed.items())
