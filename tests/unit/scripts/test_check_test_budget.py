"""Unit tests of scripts/check_test_budget.py `check(cases, total_limit, test_limit, warn_at)` (contract #3, CI3): the
pure core of the runtime budget the fast CI job applies to the unit tier's junit report. `cases` is what `read_junit`
returns, a list of (test id, seconds). Over the limit fails, equal to the limit passes; a test over `warn_at` but
within `test_limit` is a warning, not an error. `read_junit` and the CLI need a file, so they are tested in
tests/component/scripts/."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from check_test_budget import check  # noqa: E402

FAST = [("tests/unit/test_a.py::test_one", 0.001), ("tests/unit/test_a.py::test_two", 0.002),
        ("tests/unit/test_b.py::test_three", 0.010)]


def _mentions(lines: list[str], test_id: str) -> bool:
    """A line names the test when it carries the test id, or at least its function name."""
    name = test_id.rsplit("::", 1)[-1]
    return any(test_id in line or name in line for line in lines)


def test_no_cases_is_clean():
    assert check([]) == ([], [])


def test_fast_tests_within_the_total_are_clean():
    assert check(FAST) == ([], [])


def test_the_defaults_are_30_s_total_a_quarter_second_per_test_and_a_50_ms_warning():
    """The defaults as the contract states them: 30.0 / 0.25 / 0.05. Equal passes, just over does not. (A test at
    the 0.25 s limit is still over the 50 ms warning line, so only the errors are read there.)"""
    assert check([("t::ok", 0.25)])[0] == []
    errors, _ = check([("t::slow", 0.251)])
    assert errors and _mentions(errors, "t::slow")
    assert check([("t::quick", 0.05)]) == ([], [])
    _, warnings = check([("t::warm", 0.051)])
    assert warnings and _mentions(warnings, "t::warm")
    exactly_30_s = [(f"t::{i}", 0.25) for i in range(120)]  # 0.25 is exact in binary, so the sum is exactly 30.0
    assert check(exactly_30_s)[0] == []
    errors, _ = check(exactly_30_s + [("t::more", 0.001)])
    assert errors


def test_a_total_over_the_limit_is_an_error():
    cases = [(f"tests/unit/test_x.py::test_{i}", 0.04) for i in range(10)]  # 0.4 s, every test under warn_at
    errors, warnings = check(cases, total_limit=0.3)
    assert len(errors) == 1
    assert "total" in errors[0].lower()
    assert warnings == []


def test_a_total_equal_to_the_limit_passes():
    cases = [(f"t::test_{i}", 0.125) for i in range(3)]  # 0.125 is exact in binary: the sum is exactly 0.375
    assert check(cases, total_limit=0.375, warn_at=1.0) == ([], [])


def test_a_total_just_over_the_limit_fails():
    cases = [(f"t::test_{i}", 0.125) for i in range(3)] + [("t::tail", 0.0001)]
    errors, _ = check(cases, total_limit=0.375)
    assert errors


def test_one_slow_test_is_an_error_that_names_it():
    cases = FAST + [("tests/unit/test_c.py::test_slow", 0.3)]
    errors, warnings = check(cases)
    assert len(errors) == 1
    assert _mentions(errors, "tests/unit/test_c.py::test_slow")
    assert not _mentions(warnings, "tests/unit/test_c.py::test_slow"), "over the limit is an error, not a warning"


def test_every_slow_test_gets_its_own_error():
    cases = FAST + [("t::slow_one", 0.5), ("t::slow_two", 0.26), ("t::fine", 0.2)]
    errors, _ = check(cases)
    slow = [e for e in errors if _mentions([e], "t::slow_one") or _mentions([e], "t::slow_two")]
    assert len(slow) == 2
    assert not _mentions(errors, "t::fine")


def test_a_test_at_the_limit_passes_and_just_over_fails():
    assert check([("t::edge", 0.25)])[0] == []
    assert check([("t::edge", 0.25)], warn_at=0.25) == ([], [])
    errors, _ = check([("t::edge", 0.2501)])
    assert errors and _mentions(errors, "t::edge")


def test_a_test_over_warn_at_but_within_the_limit_is_a_warning_only():
    cases = FAST + [("tests/unit/test_d.py::test_warm", 0.12)]
    errors, warnings = check(cases)
    assert errors == []
    assert len(warnings) == 1
    assert _mentions(warnings, "tests/unit/test_d.py::test_warm")


def test_a_test_at_warn_at_is_not_warned_and_just_over_is():
    assert check([("t::w", 0.05)]) == ([], [])
    _, warnings = check([("t::w", 0.0501)])
    assert warnings and _mentions(warnings, "t::w")


def test_warnings_and_errors_can_both_be_reported():
    cases = [("t::warm", 0.1), ("t::slow", 0.4)]
    errors, warnings = check(cases)
    assert _mentions(errors, "t::slow")
    assert _mentions(warnings, "t::warm")
    assert not _mentions(errors, "t::warm")


def test_the_limits_are_parameters():
    cases = [("t::a", 0.02), ("t::b", 0.03)]
    assert check(cases, total_limit=1.0, test_limit=0.1, warn_at=0.05) == ([], [])
    errors, warnings = check(cases, total_limit=1.0, test_limit=0.025, warn_at=0.01)
    assert _mentions(errors, "t::b") and not _mentions(errors, "t::a")
    assert _mentions(warnings, "t::a")
    errors, _ = check(cases, total_limit=0.04, test_limit=0.1, warn_at=0.05)
    assert errors and "total" in " ".join(errors).lower()


def test_a_zero_time_test_is_fine():
    assert check([("t::instant", 0.0)]) == ([], [])


def test_the_input_is_not_changed():
    cases = [("t::slow", 0.5), ("t::fast", 0.001)]
    before = list(cases)
    check(cases)
    assert cases == before


def test_the_result_is_a_pair_of_lists_of_strings():
    errors, warnings = check([("t::slow", 0.5), ("t::warm", 0.1)])
    assert isinstance(errors, list) and isinstance(warnings, list)
    assert all(isinstance(e, str) for e in errors + warnings)
