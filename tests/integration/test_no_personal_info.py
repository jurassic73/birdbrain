"""Guard: tracked files must not contain the repo owner's account name, email addresses or repo URLs.

The owner name is read from the git remote at runtime so this file never spells it.
"""

import re
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")


# Tiny allowlist: obvious placeholder domains, and the Anthropic no-reply address used in commit trailers.
OK_EMAIL = re.compile(r"@(example\.(com|org|net)|noreply\.anthropic\.com)$|^noreply@anthropic\.com$", re.I)


def _git(*args):
    try:
        r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return r.stdout


def _files():
    out = _git("ls-files")
    if out is None:
        pytest.skip("not a git checkout")
    for rel in out.splitlines():
        if rel == "uv.lock" or "/static/vendor/" in rel:
            continue
        path = ROOT / rel
        try:
            data = path.read_bytes()
        except OSError:
            continue
        if b"\0" in data:
            continue
        yield rel, data.decode("utf-8", "replace").splitlines()


def _owner():
    url = (_git("config", "--get", "remote.origin.url") or "").strip()
    m = re.match(r"(?:https?://(?:[^@/]+@)?github\.com/|git@github\.com:|ssh://git@github\.com/)([^/]+)/", url)
    return m.group(1) if m else None


def test_no_owner_name_or_repo_url():
    owner = _owner()
    if owner is None:
        pytest.skip("no GitHub remote")
    name = re.compile(re.escape(owner), re.I)
    hits = [f"{rel}:{n}" for rel, lines in _files() for n, line in enumerate(lines, 1) if name.search(line)]
    assert not hits, "repo owner's account name (or a repo URL) found in: " + ", ".join(hits)


def test_no_email_addresses():
    hits = [f"{rel}:{n} ({m.group(0)})" for rel, lines in _files() for n, line in enumerate(lines, 1)
            for m in EMAIL.finditer(line) if not OK_EMAIL.search(m.group(0))]
    assert not hits, "email addresses found in: " + ", ".join(hits)
