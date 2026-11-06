"""Component tests of the scripts/select_tests.py CLI (`python scripts/select_tests.py BASE [HEAD]`): it runs git on
the real repo, read-only, so it is not a unit test. The selection itself is covered in tests/unit/scripts/. Few and
fast: a BASE that is missing, all zeros or unknown prints `tests/component`; an empty diff prints nothing."""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "select_tests.py"
ZEROS = "0" * 40


def _cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPT), *args], cwd=ROOT, capture_output=True, text=True, timeout=60)


@pytest.mark.parametrize("args", [(), (ZEROS,), (ZEROS, "HEAD"), ("no-such-ref-for-select-tests",)])
def test_a_missing_unknown_or_all_zeros_base_means_the_whole_component_tier(args):
    res = _cli(*args)
    assert res.returncode == 0, res.stdout + res.stderr
    assert res.stdout.split() == ["tests/component"], res.stdout + res.stderr


def test_an_empty_diff_prints_nothing():
    res = _cli("HEAD", "HEAD")
    assert res.returncode == 0, res.stdout + res.stderr
    assert res.stdout.strip() == "", res.stdout + res.stderr


def test_the_paths_are_printed_space_separated_on_one_line():
    """Whatever the last commit changed, the output is one line of space-separated paths (or nothing), fit for
    `pytest $(python scripts/select_tests.py BASE)`."""
    res = _cli("HEAD~1", "HEAD")
    assert res.returncode == 0, res.stdout + res.stderr
    lines = [line for line in res.stdout.splitlines() if line.strip()]
    assert len(lines) <= 1, res.stdout
    for path in res.stdout.split():
        assert path.startswith("tests/component"), path
        assert (ROOT / path).exists(), path
