"""Unit tests of scripts/select_tests.py `select(changed, root)` (contract #3, CI2 as amended): the pure core of the
changed-path selection the fast CI job uses to pick component test files. It only looks at which files exist under
`root` and reads the component tests' import lines, so the real repo root is passed read-only and the expectations
come from real paths and real imports. The CLI (git) is not tested here."""

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from select_tests import imported_modules, select  # noqa: E402

COMPONENT = ROOT / "tests" / "component"
WHOLE_TIER = ["tests/component"]
WHOLE_WEB = ["tests/component/web"]


def _db_component_files() -> list[str]:
    """tests/component/test_db.py and its test_db_*.py siblings, as they exist on this branch."""
    files = {"tests/component/test_db.py"} | {f"tests/component/{p.name}" for p in COMPONENT.glob("test_db_*.py")}
    return sorted(files)


def _imports(rel: str, module: str) -> bool:
    """Whether the component test file `rel` has an import line naming birdbrain module `module` (checked with a
    plain pattern of its own, so the expectation does not come from the code under test)."""
    text = (ROOT / rel).read_text()
    return bool(re.search(rf"^\s*from birdbrain import .*\b{module}\b|^\s*from birdbrain\.{module} import", text, re.M))


def test_the_db_module_selects_its_component_file_and_the_topic_siblings():
    """birdbrain/<m>.py -> tests/component/test_<m>.py and tests/component/test_<m>_*.py (plus its importers)."""
    got = select(["birdbrain/db.py"], ROOT)
    assert "tests/component/test_db.py" in got
    assert "tests/component/test_db_speed.py" in got
    assert set(_db_component_files()) <= set(got)


def test_a_topic_sibling_is_not_confused_with_a_longer_module_name():
    """test_db_speed.py belongs to db, so a change to a module whose name merely starts with db selects nothing."""
    assert select(["birdbrain/dbx.py"], ROOT) == []


def test_a_module_with_no_file_of_its_own_selects_the_tests_that_import_it():
    """birdbrain/config.py has no tests/component/test_config*.py, but many component tests import it: those run."""
    assert not (COMPONENT / "test_config.py").exists()
    got = select(["birdbrain/config.py"], ROOT)
    assert "tests/component/test_zones.py" in got and _imports("tests/component/test_zones.py", "config")
    assert "tests/component/test_weather.py" in got and _imports("tests/component/test_weather.py", "config")
    assert "tests/component/test_watch.py" not in got, "test_watch.py has no import line naming config"
    assert all(p.startswith("tests/component/") for p in got)


def test_a_module_selects_an_importer_whose_name_does_not_mention_it():
    """db.py is imported by test_presence.py (among others): a change to db runs it although the file is not named
    after db."""
    got = select(["birdbrain/db.py"], ROOT)
    assert _imports("tests/component/test_presence.py", "db")
    assert "tests/component/test_presence.py" in got
    assert "tests/component/test_alert.py" in got and _imports("tests/component/test_alert.py", "db")


def test_a_module_nobody_else_imports_selects_only_its_own_file():
    """birdbrain/outlook.py has its own component file and no other component test imports it."""
    importers = [p for p in COMPONENT.rglob("test_*.py")
                 if _imports(str(p.relative_to(ROOT)), "outlook") and p.name != "test_outlook.py"]
    assert importers == [], f"the expectation of this test needs a module nobody else imports: {importers}"
    assert select(["birdbrain/outlook.py"], ROOT) == ["tests/component/test_outlook.py"]


def test_a_web_module_selects_the_web_folder_and_its_importers_elsewhere():
    """birdbrain/web/auth.py: tests/component/web/ as a whole (every web file is under it) plus a component test
    outside that folder that has `from birdbrain.web import auth`."""
    got = select(["birdbrain/web/auth.py"], ROOT)
    assert "tests/component/web" in got
    text = (ROOT / "tests/component/test_privacy.py").read_text()
    assert re.search(r"^\s*from birdbrain\.web import .*\bauth\b", text, re.M), "an import inside a test function"
    assert "tests/component/test_privacy.py" in got
    assert not any(p.startswith("tests/component/web/") for p in got)


def test_a_file_under_an_already_selected_folder_is_not_listed_separately():
    """birdbrain/web/app.py selects tests/component/web (and the component tests outside it that import the app);
    a changed test file inside that folder adds nothing and is never listed next to the folder."""
    assert (COMPONENT / "web" / "test_app_home.py").is_file()
    alone = select(["birdbrain/web/app.py"], ROOT)
    got = select(["birdbrain/web/app.py", "tests/component/web/test_app_home.py"], ROOT)
    assert "tests/component/web" in got
    assert not any(p.startswith("tests/component/web/") for p in got), got
    assert got == alone


def test_a_folder_and_a_file_inside_it_from_two_changes_collapse_to_the_folder():
    """A changed web template selects the folder and nothing else; the changed test file inside it is absorbed."""
    assert select(["birdbrain/web/templates/visit.html", "tests/component/web/test_app_home.py"], ROOT) == WHOLE_WEB


@pytest.mark.parametrize("text, expected", [
    ("from birdbrain import db\n", {"birdbrain", "birdbrain.db"}),
    ("from birdbrain import config, db, motion_circle as mc, watch\n",
     {"birdbrain", "birdbrain.config", "birdbrain.db", "birdbrain.motion_circle", "birdbrain.watch"}),
    ("from birdbrain.web import app as webapp, auth\n", {"birdbrain", "birdbrain.web", "birdbrain.web.app",
                                                          "birdbrain.web.auth"}),
    ("from birdbrain.identify import Result, Sighting\n", {"birdbrain", "birdbrain.identify",
                                                            "birdbrain.identify.Result", "birdbrain.identify.Sighting"}),
    ("    from birdbrain import species_info\n", {"birdbrain", "birdbrain.species_info"}),
    ("from birdbrain import (\n    config,\n    db,  # a comment\n)\n", {"birdbrain", "birdbrain.config", "birdbrain.db"}),
    ("import birdbrain.stay\n", {"birdbrain", "birdbrain.stay"}),
    ("import birdbrain.web.app as webapp\n", {"birdbrain", "birdbrain.web", "birdbrain.web.app"}),
    ("import os\nimport pytest\nfrom helpers.clock import FROZEN_NOW\n", set()),
    ("# from birdbrain import db\nx = 'from birdbrain import db'\n", set()),
])
def test_imported_modules_reads_the_import_lines(text, expected):
    assert imported_modules(text) == expected


@pytest.mark.parametrize("path", [
    "birdbrain/web/templates/visit.html",
    "birdbrain/web/static/style.css",
])
def test_a_template_or_static_file_selects_exactly_the_web_component_folder(path):
    assert (ROOT / path).is_file(), f"{path} should exist on every branch"
    assert select([path], ROOT) == WHOLE_WEB


@pytest.mark.parametrize("path", ["birdbrain/web/app.py", "birdbrain/web/steps.py"])
def test_a_web_module_selects_the_web_component_folder_whole(path):
    """The folder as one entry, never its files one by one; component tests outside it that import the module may
    come along (test_alert.py has `from birdbrain.web import app as webapp`)."""
    assert (ROOT / path).is_file(), f"{path} should exist on every branch"
    got = select([path], ROOT)
    assert "tests/component/web" in got
    assert not any(p.startswith("tests/component/web/") for p in got), got
    assert all(p == "tests/component/web" or p.startswith("tests/component/test_") for p in got), got


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
    parts = set(select(["birdbrain/db.py"], ROOT)) | set(select(["birdbrain/web/app.py"], ROOT))
    expected = sorted({p for p in parts if not p.startswith("tests/component/web/")} | {"tests/component/web"})
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
