"""Unit tests of scripts/select_tests.py `select(changed, root)` (contract #3, CI2): the pure core of the changed-path
selection the fast CI job uses to pick component test files. It only looks at which files exist under `root`, so the
real repo root is passed read-only and the expectations come from real paths. The CLI (git) is not tested here."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from select_tests import select  # noqa: E402

COMPONENT = ROOT / "tests" / "component"
WHOLE_TIER = ["tests/component"]


def _db_component_files() -> list[str]:
    """tests/component/test_db.py and its test_db_*.py siblings, as they exist on this branch."""
    files = {"tests/component/test_db.py"} | {f"tests/component/{p.name}" for p in COMPONENT.glob("test_db_*.py")}
    return sorted(files)


def test_the_db_module_selects_its_component_file_and_the_topic_siblings():
    """birdbrain/<m>.py -> tests/component/test_<m>.py and tests/component/test_<m>_*.py."""
    got = select(["birdbrain/db.py"], ROOT)
    assert "tests/component/test_db.py" in got
    assert "tests/component/test_db_speed.py" in got
    assert got == _db_component_files()


def test_a_topic_sibling_is_not_confused_with_a_longer_module_name():
    """test_db_speed.py belongs to db, so a change to a module whose name merely starts with db selects nothing."""
    assert select(["birdbrain/dbx.py"], ROOT) == []


def test_a_module_with_only_unit_tests_selects_nothing():
    """birdbrain/config.py has no tests/component/test_config*.py: only existing files are returned."""
    assert not (COMPONENT / "test_config.py").exists()
    assert select(["birdbrain/config.py"], ROOT) == []


@pytest.mark.parametrize("path", [
    "birdbrain/web/app.py",
    "birdbrain/web/steps.py",
    "birdbrain/web/templates/visit.html",
    "birdbrain/web/static/style.css",
])
def test_anything_under_the_web_package_selects_the_web_component_folder(path):
    assert (ROOT / path).is_file(), f"{path} should exist on every branch"
    assert select([path], ROOT) == ["tests/component/web"]


@pytest.mark.parametrize("path", [
    "tests/component/test_db.py",
    "tests/component/web/test_app_home.py",
])
def test_a_changed_component_test_file_selects_itself(path):
    assert (ROOT / path).is_file(), f"{path} should exist on every branch"
    assert select([path], ROOT) == [path]


def test_a_deleted_test_file_is_not_returned():
    assert not (COMPONENT / "test_gone_forever.py").exists()
    assert select(["tests/component/test_gone_forever.py"], ROOT) == []


@pytest.mark.parametrize("path", [
    "tests/conftest.py",
    "tests/unit/conftest.py",
    "tests/component/conftest.py",
    "tests/component/web/conftest.py",
    "tests/helpers/__init__.py",
    "tests/helpers/claude.py",
    "pyproject.toml",
    "uv.lock",
])
def test_shared_test_infrastructure_selects_the_whole_component_tier(path):
    assert (ROOT / path).is_file(), f"{path} should exist on every branch"
    assert select([path], ROOT) == WHOLE_TIER


def test_the_whole_tier_subsumes_every_narrower_match():
    """Once the whole tier is needed the answer is exactly ["tests/component"]: no file listed twice over."""
    assert select(["birdbrain/db.py", "tests/conftest.py", "birdbrain/web/app.py"], ROOT) == WHOLE_TIER


@pytest.mark.parametrize("path", [
    "README.md",
    ".env.example",
    ".gitignore",
    "deploy/birdbrain-web.service",
])
def test_docs_and_other_files_select_nothing(path):
    assert select([path], ROOT) == []


def test_a_deleted_or_unknown_source_path_selects_nothing():
    assert not (ROOT / "birdbrain" / "nosuchmodule.py").exists()
    assert select(["birdbrain/nosuchmodule.py"], ROOT) == []


def test_nothing_changed_selects_nothing():
    assert select([], ROOT) == []


def test_a_mixed_list_is_the_sorted_union_without_duplicates():
    changed = ["README.md", "birdbrain/web/app.py", "birdbrain/db.py", "birdbrain/db.py",
               "birdbrain/nosuchmodule.py", "tests/component/test_db.py"]
    got = select(changed, ROOT)
    expected = sorted(set(_db_component_files()) | {"tests/component/web"})
    assert got == expected
    assert got == sorted(got)
    assert len(got) == len(set(got))


def test_the_result_is_sorted_whatever_the_input_order():
    forward = select(["birdbrain/db.py", "birdbrain/web/app.py"], ROOT)
    backward = select(["birdbrain/web/app.py", "birdbrain/db.py"], ROOT)
    assert forward == backward == sorted(forward)


def test_paths_are_repo_relative_strings_with_forward_slashes():
    for path in select(["birdbrain/db.py", "birdbrain/web/app.py"], ROOT):
        assert isinstance(path, str)
        assert path.startswith("tests/component/")
        assert "\\" not in path and not path.startswith("/")
