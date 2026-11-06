"""Component tests of scripts/check_test_budget.py: `read_junit(path)` on a junit file under tmp_path, and the CLI
(`python scripts/check_test_budget.py JUNIT.xml`: exit 1 on any error, 0 otherwise; TEST_BUDGET_TOTAL and
TEST_BUDGET_TEST override the limits). The pure `check` is covered in tests/unit/scripts/."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "check_test_budget.py"
sys.path.insert(0, str(ROOT / "scripts"))
from check_test_budget import read_junit  # noqa: E402


def _junit(*cases: tuple[str, str, float]) -> str:
    """A junit report the way pytest writes one: (classname, name, seconds) per test case."""
    body = "".join(f'<testcase classname="{cls}" name="{name}" time="{secs:.6f}" />' for cls, name, secs in cases)
    n = len(cases)
    return ('<?xml version="1.0" encoding="utf-8"?><testsuites>'
            f'<testsuite name="pytest" errors="0" failures="0" skipped="0" tests="{n}" time="1.0">{body}'
            "</testsuite></testsuites>")


def _write(tmp_path: Path, xml: str) -> Path:
    report = tmp_path / "junit.xml"
    report.write_text(xml)
    return report


def test_read_junit_returns_one_id_and_seconds_per_test_case(tmp_path):
    report = _write(tmp_path, _junit(("unit.test_a", "test_one", 0.0012), ("unit.web.test_app", "test_two", 0.3)))
    cases = read_junit(report)
    assert len(cases) == 2
    ids = [case[0] for case in cases]
    times = [case[1] for case in cases]
    assert "test_one" in ids[0] and "test_two" in ids[1], ids
    assert times == pytest.approx([0.0012, 0.3])
    assert all(isinstance(secs, float) for secs in times)


def test_read_junit_ids_tell_two_tests_of_the_same_name_apart(tmp_path):
    report = _write(tmp_path, _junit(("unit.test_a", "test_x", 0.01), ("unit.test_b", "test_x", 0.02)))
    ids = [case[0] for case in read_junit(report)]
    assert len(set(ids)) == 2, ids


def test_read_junit_takes_a_string_path_too(tmp_path):
    report = _write(tmp_path, _junit(("unit.test_a", "test_one", 0.5)))
    assert [secs for _, secs in read_junit(str(report))] == pytest.approx([0.5])


def test_read_junit_of_an_empty_report_is_an_empty_list(tmp_path):
    report = _write(tmp_path, _junit())
    assert read_junit(report) == []


def _cli(report: Path, **env: str) -> subprocess.CompletedProcess:
    """The script run on `report` with the given TEST_BUDGET_* variables (and no stray ones from this shell)."""
    clean = {k: v for k, v in os.environ.items() if not k.startswith("TEST_BUDGET_")}
    return subprocess.run([sys.executable, str(SCRIPT), str(report)], cwd=ROOT, capture_output=True, text=True,
                          timeout=60, env={**clean, **env})


def test_cli_exits_0_when_everything_is_within_budget(tmp_path):
    report = _write(tmp_path, _junit(("unit.test_a", "test_one", 0.001), ("unit.test_a", "test_two", 0.01)))
    res = _cli(report)
    assert res.returncode == 0, res.stdout + res.stderr


def test_cli_exits_1_when_one_test_is_over_the_per_test_limit_and_prints_it(tmp_path):
    report = _write(tmp_path, _junit(("unit.test_a", "test_fast", 0.001), ("unit.test_a", "test_slow", 0.3)))
    res = _cli(report)
    assert res.returncode == 1, res.stdout + res.stderr
    assert "test_slow" in res.stdout + res.stderr
    assert "test_fast" not in res.stdout + res.stderr


def test_cli_prints_a_warning_for_a_test_over_50_ms_but_still_exits_0(tmp_path):
    report = _write(tmp_path, _junit(("unit.test_a", "test_warm", 0.1)))
    res = _cli(report)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "test_warm" in res.stdout + res.stderr


def test_cli_exits_1_when_the_total_is_over_the_limit(tmp_path):
    report = _write(tmp_path, _junit(*((f"unit.test_{i}", f"test_{i}", 0.2) for i in range(5))))  # 1.0 s in all
    res = _cli(report, TEST_BUDGET_TOTAL="0.5")
    assert res.returncode == 1, res.stdout + res.stderr
    assert "total" in (res.stdout + res.stderr).lower()


def test_env_overrides_raise_the_per_test_limit_on_a_slow_machine(tmp_path):
    report = _write(tmp_path, _junit(("unit.test_a", "test_slow", 0.3)))
    assert _cli(report).returncode == 1
    res = _cli(report, TEST_BUDGET_TEST="1.0")
    assert res.returncode == 0, res.stdout + res.stderr


def test_env_overrides_lower_the_total_limit(tmp_path):
    report = _write(tmp_path, _junit(("unit.test_a", "test_a", 0.01), ("unit.test_a", "test_b", 0.01)))
    assert _cli(report).returncode == 0
    res = _cli(report, TEST_BUDGET_TOTAL="0.015")
    assert res.returncode == 1, res.stdout + res.stderr


def test_cli_on_a_report_with_no_test_cases_exits_0(tmp_path):
    report = _write(tmp_path, _junit())
    res = _cli(report)
    assert res.returncode == 0, res.stdout + res.stderr
