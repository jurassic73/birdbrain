"""Rules for the CI workflow (.github/workflows/tests.yml) and the two scripts it runs, checked on the real files.

Plain text checks rather than a YAML parser (pyyaml is not a dependency): the workflow is small and its shape is
what matters here, that the fast job measures the unit tier's runtime budget on the junit report it just wrote, that
pull requests add the component tier, that the nightly run repeats the suite in random order with a printed seed,
that every action is pinned to a major version and that the file names nobody and no repo.
"""
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "tests.yml"
SELECT = REPO_ROOT / "scripts" / "select_tests.py"
BUDGET = REPO_ROOT / "scripts" / "check_test_budget.py"


@pytest.fixture(scope="module")
def text() -> str:
    assert WORKFLOW.is_file(), f"{WORKFLOW.relative_to(REPO_ROOT)} is missing"
    return WORKFLOW.read_text()


def _job(text: str, name: str) -> str:
    """The lines of one job (from `  <name>:` under jobs: to the next job at the same indent)."""
    m = re.search(rf"^  {name}:\n(.*?)(?=^  \w+:\n|\Z)", text, re.S | re.M)
    assert m, f"no job named {name} in the workflow"
    return m.group(1)


def test_triggers_are_push_pull_request_nightly_and_manual(text):
    on = re.search(r"^on:\n(.*?)^\S", text, re.S | re.M).group(1)
    for trigger in ("push:", "pull_request:", "schedule:", "workflow_dispatch:"):
        assert re.search(rf"^  {re.escape(trigger)}", on, re.M), f"{trigger} is not a trigger"


def test_nightly_cron_avoids_the_busy_minutes(text):
    crons = re.findall(r'cron:\s*"([^"]+)"', text)
    assert len(crons) == 1, crons
    minute = crons[0].split()[0]
    assert minute.isdigit() and int(minute) not in (0, 30), f"cron minute {minute!r} is :00 or :30 (the busy minutes)"


def test_every_action_is_pinned_to_a_major_version(text):
    uses = re.findall(r"uses:\s*(\S+)", text)
    assert uses, "no actions used"
    unpinned = [u for u in uses if not re.search(r"@v\d+$", u)]
    assert not unpinned, f"actions without a major version pin: {unpinned}"


def test_fast_job_runs_the_unit_tier_and_checks_its_budget_on_that_junit(text):
    fast = _job(text, "fast")
    assert "github.event_name == 'push' || github.event_name == 'pull_request'" in fast
    junit = re.search(r"pytest tests/unit \S+( \S+)* --junitxml=(\S+)", fast)
    assert junit, "the fast job does not run `pytest tests/unit ... --junitxml=...`"
    assert f"scripts/check_test_budget.py {junit.group(2)}" in fast, "the budget check does not read that junit"
    assert "-m guard" in fast, "the guard tests do not run in the fast job"
    assert "scripts/select_tests.py" in fast, "the changed-path selection does not run in the fast job"
    assert "github.event.before" in fast and "github.event.pull_request.base.sha" in fast, "BASE for push and PR"
    assert "fetch-depth: 0" in fast, "the changed-path diff needs history"


def test_component_job_runs_the_whole_tier_on_pull_requests_and_main(text):
    component = _job(text, "component")
    assert "github.event_name == 'pull_request'" in component
    assert "github.ref == 'refs/heads/main'" in component
    assert re.search(r"pytest tests/component\b.*-n \d", component), "the component tier does not run with xdist"


def test_full_job_is_nightly_and_manual_and_repeats_in_random_order(text):
    full = _job(text, "full")
    assert "github.event_name == 'schedule'" in full and "github.event_name == 'workflow_dispatch'" in full
    assert "--with pytest-randomly" in full, "random order needs pytest-randomly (not a dev dependency)"
    assert "--randomly-seed=${{ github.run_number }}" in full
    assert re.search(r"echo .*seed.*github\.run_number", full), "the seed is not printed"


def test_workflow_uses_no_secrets_and_names_no_repo_or_person(text):
    assert "secrets." not in text
    assert "github.com/" not in text
    assert "\u2014" not in text, "no em dashes"
    assert "uv sync --frozen" in text
    assert 'PYTHON_VERSION: "3.11"' in text


def test_the_scripts_are_stdlib_only_and_runnable_without_the_venv():
    """Both scripts run under a bare interpreter (CI could call them before `uv sync`, and nothing in the service
    code imports them): the budget check prints its usage and exits 2, the selector prints the whole tier when BASE
    is all zeros."""
    res = subprocess.run([sys.executable, "-I", str(BUDGET)], capture_output=True, text=True)
    assert res.returncode == 2 and "usage" in res.stderr, (res.returncode, res.stderr)
    res = subprocess.run([sys.executable, "-I", str(SELECT), "0" * 40], capture_output=True, text=True, cwd=REPO_ROOT)
    assert res.returncode == 0 and res.stdout.strip() == "tests/component", (res.returncode, res.stdout, res.stderr)
    for script in (SELECT, BUDGET):
        imports = re.findall(r"^(?:from|import) (\w+)", script.read_text(), re.M)
        assert "birdbrain" not in imports, f"{script.name} imports the service code"
