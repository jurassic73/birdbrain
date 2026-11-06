#!/usr/bin/env python3
"""Pick the component tests worth running for a set of changed files (the CI fast job's changed-path step).

The unit tier always runs whole; this chooses which component tests to add from `git diff --name-only BASE...HEAD`:
  birdbrain/<m>.py            -> tests/component/test_<m>.py and tests/component/test_<m>_*.py
  birdbrain/<sub>/<m>.py      -> tests/component/<sub>/test_<m>.py and test_<m>_*.py (birdbrain/web/** is special:
                                 app, templates and static all map to the whole tests/component/web/ folder)
  scripts/<m>.*               -> tests/component/scripts/test_<m>.py and test_<m>_*.py
  tests/component/**/test_*.py -> itself
  tests/conftest.py, tests/helpers/**, any tier conftest.py, pyproject.toml, uv.lock -> the whole component tier
Only paths that exist are returned; nothing matched means nothing to add (the unit tier still ran).

Usage: python scripts/select_tests.py BASE [HEAD]   (HEAD defaults to HEAD)
Prints the paths space-separated, nothing when there is nothing to add, and `tests/component` when the whole tier
is needed, including when BASE is missing, unknown to git or all zeros (a new branch or a force push), so a doubtful
range always runs everything. Stdlib only, nothing here is imported by the service code.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path, PurePosixPath

COMPONENT = "tests/component"
WHOLE_TIER_FILES = {"pyproject.toml", "uv.lock", "tests/conftest.py"}
WHOLE_TIER_PREFIXES = ("tests/helpers/",)
TIERS = ("unit", "component", "integration")
ALL_ZEROS = re.compile(r"^0+$")


def _normal(path: str) -> str:
    """A repo-relative posix path: no leading ./ and no backslashes."""
    text = path.strip().replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    return text.lstrip("/")


def _needs_whole_tier(path: str) -> bool:
    if path in WHOLE_TIER_FILES or path.startswith(WHOLE_TIER_PREFIXES):
        return True
    parts = PurePosixPath(path).parts
    # a tier conftest at any depth: tests/<tier>/conftest.py, tests/<tier>/web/conftest.py, ...
    return len(parts) >= 3 and parts[0] == "tests" and parts[1] in TIERS and parts[-1] == "conftest.py"


def _module_tests(folder: Path, module: str) -> list[Path]:
    """tests/component/<folder>/test_<module>.py plus test_<module>_*.py, those that exist."""
    if not folder.is_dir():
        return []
    found = [folder / f"test_{module}.py", *folder.glob(f"test_{module}_*.py")]
    return [p for p in found if p.is_file()]


def _tests_for(path: str, root: Path) -> list[Path]:
    """The component test paths one changed file calls for (not the whole-tier case)."""
    parts = PurePosixPath(path).parts
    component = root / COMPONENT
    if not parts:
        return []
    if parts[0] == "birdbrain" and len(parts) >= 2:
        if parts[1] == "web":
            web = component / "web"
            return [web] if web.is_dir() else []
        if parts[-1].endswith(".py"):
            return _module_tests(component.joinpath(*parts[1:-1]), parts[-1][:-3])
        return []
    if parts[0] == "scripts" and len(parts) == 2 and "." in parts[1]:
        return _module_tests(component / "scripts", parts[1].rsplit(".", 1)[0])
    if parts[:2] == ("tests", "component") and parts[-1].startswith("test_") and parts[-1].endswith(".py"):
        candidate = root.joinpath(*parts)
        return [candidate] if candidate.is_file() else []
    return []


def select(changed: list[str], root: Path) -> list[str]:
    """Sorted repo-relative component test paths to run for `changed` (repo-relative paths): `["tests/component"]`
    when the whole tier is needed, `[]` when nothing matched. Only paths that exist under `root` are returned."""
    root = Path(root)
    paths = [_normal(p) for p in changed if p and p.strip()]
    if any(_needs_whole_tier(p) for p in paths):
        return [COMPONENT]
    found: set[str] = set()
    for path in paths:
        for test in _tests_for(path, root):
            found.add(test.relative_to(root).as_posix())
    folders = [f for f in found if (root / f).is_dir()]
    return sorted(p for p in found if not any(p != f and p.startswith(f + "/") for f in folders))


# --- the git wrapper ---------------------------------------------------------------------------------------------

def _git(root: Path, *args: str) -> str | None:
    try:
        res = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return res.stdout


def changed_files(base: str | None, head: str, root: Path) -> list[str] | None:
    """`git diff --name-only BASE...HEAD`, or None when the range cannot be trusted (no BASE, an all-zero BASE from a
    new branch or force push, a commit git does not have), which means run everything."""
    if not base or ALL_ZEROS.match(base):
        return None
    if _git(root, "rev-parse", "--verify", "--quiet", f"{base}^{{commit}}") is None:
        return None
    out = _git(root, "diff", "--name-only", f"{base}...{head}")
    if out is None:
        return None
    return [line for line in out.splitlines() if line.strip()]


def main(argv: list[str]) -> int:
    root = Path(__file__).resolve().parent.parent
    base = argv[1] if len(argv) > 1 else None
    head = argv[2] if len(argv) > 2 else "HEAD"
    changed = changed_files(base, head, root)
    paths = [COMPONENT] if changed is None else select(changed, root)
    if paths:
        print(" ".join(paths))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
