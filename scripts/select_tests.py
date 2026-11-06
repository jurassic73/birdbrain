#!/usr/bin/env python3
"""Pick the component tests worth running for a set of changed files (the CI fast job's changed-path step).

The unit tier always runs whole; this chooses which component tests to add from `git diff --name-only BASE...HEAD`:
  birdbrain/<m>.py            -> tests/component/test_<m>.py and tests/component/test_<m>_*.py, plus every
                                 tests/component/**/test_*.py whose import lines name the module (`import birdbrain.<m>`,
                                 `from birdbrain import ..., <m>`, `from birdbrain.<m> import ...`); a module with no
                                 file of its own name (config) is still covered by the tests that import it
  birdbrain/<sub>/<m>.py      -> tests/component/<sub>/test_<m>.py and test_<m>_*.py, plus its importers the same way
                                 (`from birdbrain.<sub> import <m>` too); birdbrain/web/** is special: app, templates
                                 and static all map to the whole tests/component/web/ folder
  scripts/<m>.*               -> tests/component/scripts/test_<m>.py and test_<m>_*.py
  tests/component/**/test_*.py -> itself
  tests/conftest.py, tests/helpers/**, any tier conftest.py, pyproject.toml, uv.lock -> the whole component tier
Only paths that exist are returned, a file inside a folder already listed is not listed again, and nothing matched
means nothing to add (the unit tier still ran).

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


# Import lines of a test file: `from birdbrain[.sub...] import a, b as c` (the names may continue inside parentheses)
# and `import birdbrain.x[, birdbrain.y as z]`, at any indentation (a few tests import inside a function).
FROM_IMPORT = re.compile(r"^[ \t]*from[ \t]+(birdbrain(?:\.\w+)*)[ \t]+import[ \t]+(\([^)]*\)|[^\n]*)", re.M)
IMPORT = re.compile(r"^[ \t]*import[ \t]+((?:birdbrain(?:\.\w+)*(?:[ \t]+as[ \t]+\w+)?[ \t]*,?[ \t]*)+)", re.M)


def _with_parents(dotted: str) -> set[str]:
    """birdbrain.web.app -> {birdbrain, birdbrain.web, birdbrain.web.app}: importing a module imports its packages."""
    parts = dotted.split(".")
    return {".".join(parts[:n]) for n in range(1, len(parts) + 1)}


def imported_modules(text: str) -> set[str]:
    """The dotted birdbrain modules a test file's import lines name, with their parent packages. A name imported from
    a package (`from birdbrain import db`) counts as the module `birdbrain.db`; a name imported from a module
    (`from birdbrain.db import session`) adds `birdbrain.db.session`, which no changed file is ever named like, so
    it does no harm. Pure: a regex over the text, no import is executed."""
    found: set[str] = set()
    for package, names in FROM_IMPORT.findall(text):
        found |= _with_parents(package)
        for name in names.strip("()").replace("\n", " ").split(","):
            name = name.split("#", 1)[0].strip().split(" ")[0]
            if name.isidentifier():
                found |= _with_parents(f"{package}.{name}")
    for group in IMPORT.findall(text):
        for name in group.split(","):
            name = name.strip().split(" ")[0]
            if name.startswith("birdbrain"):
                found |= _with_parents(name)
    return found


def _component_imports(root: Path) -> dict[Path, set[str]]:
    """Every tests/component/**/test_*.py under `root` -> the birdbrain modules its import lines name (read once per
    call of `select`)."""
    files = sorted(p for p in (root / COMPONENT).rglob("test_*.py") if "__pycache__" not in p.parts)
    return {p: imported_modules(p.read_text(encoding="utf-8", errors="replace")) for p in files}


def _importers(dotted: str, imports: dict[Path, set[str]]) -> list[Path]:
    """The component test files whose import lines name the module `dotted`."""
    return [p for p, names in imports.items() if dotted in names]


def _tests_for(path: str, root: Path, imports: dict[Path, set[str]]) -> list[Path]:
    """The component test paths one changed file calls for (not the whole-tier case). `imports` is what the component
    tier's test files import (see `_component_imports`), looked up when `path` is a birdbrain module."""
    parts = PurePosixPath(path).parts
    component = root / COMPONENT
    if not parts:
        return []
    if parts[0] == "birdbrain" and len(parts) >= 2:
        found = []
        if parts[1] == "web":
            web = component / "web"
            found = [web] if web.is_dir() else []
        if parts[-1].endswith(".py"):
            module = parts[-1][:-3]
            found += _module_tests(component.joinpath(*parts[1:-1]), module)
            dotted = ".".join(parts[:-1] if module == "__init__" else parts[:-1] + (module,))
            found += _importers(dotted, imports)
        return found
    if parts[0] == "scripts" and len(parts) == 2 and "." in parts[1]:
        return _module_tests(component / "scripts", parts[1].rsplit(".", 1)[0])
    if parts[:2] == ("tests", "component") and parts[-1].startswith("test_") and parts[-1].endswith(".py"):
        candidate = root.joinpath(*parts)
        return [candidate] if candidate.is_file() else []
    return []


def select(changed: list[str], root: Path) -> list[str]:
    """Sorted repo-relative component test paths to run for `changed` (repo-relative paths): `["tests/component"]`
    when the whole tier is needed, `[]` when nothing matched. Only paths that exist under `root` are returned, and a
    file inside a folder that is returned (tests/component/web) is not returned on its own as well."""
    root = Path(root)
    paths = [_normal(p) for p in changed if p and p.strip()]
    if any(_needs_whole_tier(p) for p in paths):
        return [COMPONENT]
    imports = _component_imports(root) if any(p.startswith("birdbrain/") for p in paths) else {}
    found: set[str] = set()
    for path in paths:
        for test in _tests_for(path, root, imports):
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
