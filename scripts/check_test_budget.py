#!/usr/bin/env python3
"""The unit tier's runtime budget: read the junit report of a `pytest tests/unit --junitxml=...` run and fail when
the tier as a whole, or any single test, took longer than it is allowed to.

Limits (the one place they live; the CLI's env variables override the first two for a slow machine):
  TOTAL_LIMIT  summed time of every unit test, seconds           TEST_BUDGET_TOTAL
  TEST_LIMIT   one unit test's time, seconds (an error above it)  TEST_BUDGET_TEST
  WARN_AT      a unit test above this is listed as a warning (the 50 ms rule; TEST_LIMIT leaves headroom for a slow
               CI runner, so only TEST_LIMIT fails the run)

Usage: python scripts/check_test_budget.py JUNIT.xml
Locally: uv run pytest tests/unit -p no:cacheprovider --junitxml=/tmp/unit.xml && python scripts/check_test_budget.py /tmp/unit.xml
Prints a summary, the warnings and the errors; exit 1 on any error, 2 when the report cannot be read or holds no
test cases at all (an empty report means the unit step produced nothing, so "within budget" would be a false pass:
the step is broken, not fast). The time of a test is what pytest wrote for it (setup, call and teardown together),
so the bars are a little stricter than a call-time-only rule. Stdlib only, nothing here is imported by the service
code.
"""
from __future__ import annotations

import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

TOTAL_LIMIT = 30.0
TEST_LIMIT = 0.25
WARN_AT = 0.05
ENV_TOTAL = "TEST_BUDGET_TOTAL"
ENV_TEST = "TEST_BUDGET_TEST"


def read_junit(path) -> list[tuple[str, float]]:
    """(test id, seconds) for every testcase in a junit report, in file order; the id is `classname::name`
    (tests.unit.test_config::test_x[param]) or just the name when the report has no classname. A testcase without a
    time, or an unreadable one, counts as 0 s."""
    root = ET.parse(str(path)).getroot()
    cases = []
    for case in root.iter("testcase"):
        name = case.get("name") or ""
        classname = case.get("classname") or ""
        test_id = f"{classname}::{name}" if classname else name
        try:
            seconds = float(case.get("time") or 0.0)
        except ValueError:
            seconds = 0.0
        cases.append((test_id, seconds))
    return cases


def check(cases, total_limit=TOTAL_LIMIT, test_limit=TEST_LIMIT, warn_at=WARN_AT) -> tuple[list[str], list[str]]:
    """(errors, warnings) for `cases` as read_junit returns them. Errors: the summed time is over `total_limit`, or a
    test took more than `test_limit`. Warnings: a test over `warn_at` that stayed within `test_limit`. Both lists
    name the slowest first."""
    errors: list[str] = []
    warnings: list[str] = []
    total = sum(seconds for _, seconds in cases)
    if total > total_limit:
        errors.append(f"the unit tier took {total:.2f} s in total, over its {total_limit:g} s budget")
    for test_id, seconds in sorted(cases, key=lambda c: -c[1]):
        if seconds > test_limit:
            errors.append(f"{test_id} took {seconds:.3f} s, over the {test_limit:g} s limit for one unit test")
        elif seconds > warn_at:
            warnings.append(f"{test_id} took {seconds:.3f} s, over the {warn_at * 1000:g} ms rule for a unit test")
    return errors, warnings


def _limit(env_name: str, default: float) -> float:
    raw = os.environ.get(env_name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        raise SystemExit(f"{env_name} must be a number of seconds, not {raw!r}")


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__.strip().splitlines()[0], file=sys.stderr)
        print("usage: python scripts/check_test_budget.py JUNIT.xml", file=sys.stderr)
        return 2
    path = Path(argv[1])
    try:
        cases = read_junit(path)
    except (OSError, ET.ParseError) as e:
        print(f"cannot read the junit report {path}: {e}", file=sys.stderr)
        return 2
    if not cases:
        print(f"the junit report {path} holds no tests: the unit step produced nothing", file=sys.stderr)
        return 2
    total_limit = _limit(ENV_TOTAL, TOTAL_LIMIT)
    test_limit = _limit(ENV_TEST, TEST_LIMIT)
    errors, warnings = check(cases, total_limit=total_limit, test_limit=test_limit)
    total = sum(seconds for _, seconds in cases)
    slowest_id, slowest = max(cases, key=lambda c: c[1])
    print(f"{len(cases)} unit tests, {total:.2f} s in total (budget {total_limit:g} s); "
          f"slowest {slowest:.3f} s (limit {test_limit:g} s): {slowest_id}")
    for line in warnings:
        print(f"warning: {line}")
    for line in errors:
        print(f"error: {line}")
    if errors:
        print(f"{len(errors)} error(s): the unit tier is over its budget")
        return 1
    print(f"within budget ({len(warnings)} warning(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
