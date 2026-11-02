"""Friends & family gate for the public mirror: one shared password in front of the whole site.

The password lives only as a scrypt hash in VIEWER_PASSWORD_FILE (default /etc/birdbrain/viewer.pw, readable
by the service, never in the repo or the synced data). No file = no gate, so the Pi's own site is unchanged.
Set or change it on the server with:  cd /opt/birdbrain && sudo venv/bin/python -m birdbrain.web.viewer
Changing it signs everyone out.

Share links: `cd /opt/birdbrain && sudo venv/bin/python -m birdbrain.web.viewer link` makes a long random key and prints a link like
https://somegeek.com/birdbrain/guest-login#k=KEY. The key sits after the '#', so browsers never send it to any
server or log; the sign-in page reads it and posts it like a password. Only its SHA-256 is stored, in
VIEWER_INVITE_FILE (default /etc/birdbrain/viewer.invite). Making a new link retires the old one and signs
everyone out.

Guessing is throttled: 3 wrong tries from one address locks that address out for 15 minutes, and 20 wrong
tries from everyone together locks the login for everyone for 15 minutes (so rotating addresses doesn't help).
"""
import base64
import getpass
import hashlib
import hmac
import os
import secrets
import shutil
import sys
import time
from pathlib import Path

from fastapi import Request

COOKIE = "bb_guest"
MAX_AGE = 30 * 24 * 3600
PER_IP_FAILS = 3
ALL_FAILS = 20
LOCKOUT = 15 * 60
# Exact paths only (compared with ==, never a prefix). The manifest holds just the app name, colors and icon URLs,
# and browsers fetch it without cookies, so behind the gate it would 303 and Android couldn't read it.
OPEN_PATHS = ("/guest-login", "/manifest.webmanifest")
OPEN_PREFIXES = ("/static/",)
_fails: dict[str, list[float]] = {}
_cache: dict = {}


def password_file() -> Path:
    return Path(os.environ.get("VIEWER_PASSWORD_FILE", "/etc/birdbrain/viewer.pw"))


def invite_file() -> Path:
    return Path(os.environ.get("VIEWER_INVITE_FILE", "/etc/birdbrain/viewer.invite"))


def _read(path: Path) -> str:
    """A file's stripped text, re-read only when it changes; '' if it's missing."""
    try:
        mtime = path.stat().st_mtime
    except OSError:
        _cache.pop(str(path), None)
        return ""
    cached = _cache.get(str(path))
    if not cached or cached[0] != mtime:
        cached = _cache[str(path)] = (mtime, path.read_text().strip())
    return cached[1]


def stored_hash() -> str:
    """The 'scrypt$salt$hash' line; '' when there is no gate."""
    return _read(password_file())


def invite_hash() -> str:
    return _read(invite_file())


def check_invite(key: str) -> bool:
    want = invite_hash()
    return bool(want and key) and hmac.compare_digest(hashlib.sha256(key.encode()).hexdigest(), want)


def enabled() -> bool:
    return bool(stored_hash())


def _scrypt(password: str, salt: bytes) -> bytes:
    return hashlib.scrypt(password.encode(), salt=salt, n=2**14, r=8, p=1, dklen=32)


def make_hash(password: str) -> str:
    salt = secrets.token_bytes(16)
    return "scrypt$" + base64.b64encode(salt).decode() + "$" + base64.b64encode(_scrypt(password, salt)).decode()


def check_password(given: str) -> bool:
    try:
        kind, salt, want = stored_hash().split("$")
    except ValueError:
        return False
    return kind == "scrypt" and hmac.compare_digest(_scrypt(given, base64.b64decode(salt)), base64.b64decode(want))


def _key() -> bytes:
    # From the stored hash, so a new password signs everyone out.
    return hashlib.sha256(b"birdbrain-guest:" + stored_hash().encode() + b":" + invite_hash().encode()).digest()


def make_cookie() -> str:
    expires = int(time.time()) + MAX_AGE
    return f"{expires}.{hmac.new(_key(), str(expires).encode(), hashlib.sha256).hexdigest()}"


def signed_in(request: Request) -> bool:
    expires, _, sig = request.cookies.get(COOKIE, "").partition(".")
    if not expires.isdigit() or int(expires) < time.time():
        return False
    return hmac.compare_digest(sig, hmac.new(_key(), expires.encode(), hashlib.sha256).hexdigest())


def is_open(path: str) -> bool:
    return path in OPEN_PATHS or path.startswith(OPEN_PREFIXES)


def client_ip(request: Request) -> str:
    """The visitor's address. Behind the Cloudflare tunnel every request reaches us from 127.0.0.1, so use the
    header Cloudflare sets (it overwrites any copy a visitor sends); without it, the direct peer."""
    return request.headers.get("cf-connecting-ip") or (request.client.host if request.client else "?")


def _recent(key: str) -> list[float]:
    now = time.time()
    _fails[key] = [t for t in _fails.get(key, []) if now - t < LOCKOUT]
    return _fails[key]


def locked(ip: str) -> int:
    """Seconds until this address may try again (0 = may try now)."""
    for key, limit in ((ip, PER_IP_FAILS), ("*", ALL_FAILS)):
        tries = _recent(key)
        if len(tries) >= limit:
            return int(tries[-limit] + LOCKOUT - time.time()) + 1
    return 0


def failed(ip: str) -> None:
    now = time.time()
    _fails.setdefault(ip, []).append(now)
    _fails.setdefault("*", []).append(now)


def succeeded(ip: str) -> None:
    _fails.pop(ip, None)


def _write(path: Path, text: str) -> None:
    """Owner-writable, readable by the service's group (birdbrain, when it exists), hidden from everyone else."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(text + "\n")
    tmp.chmod(0o640)
    try:
        shutil.chown(tmp, group="birdbrain")
        shutil.chown(path.parent, group="birdbrain")
        path.parent.chmod(0o750)
    except (LookupError, PermissionError):
        pass  # no birdbrain group here (a dev machine): the file stays the caller's
    tmp.replace(path)


def make_link(site: str) -> str:
    key = secrets.token_urlsafe(24)
    _write(invite_file(), hashlib.sha256(key.encode()).hexdigest())
    return f"{site.rstrip('/')}/guest-login#k={key}"


def main() -> None:
    """Set the password (prompts twice, writes only the hash), or with `link`, make a new share link."""
    if sys.argv[1:2] == ["link"]:
        site = sys.argv[2] if len(sys.argv) > 2 else "https://somegeek.com/birdbrain"
        print(make_link(site))
        print("Old links stop working, and everyone signed in before is now signed out.")
        return
    path = password_file()
    first = getpass.getpass("New friends & family password: ")
    if len(first) < 8:
        sys.exit("Use at least 8 characters.")
    if getpass.getpass("Again: ") != first:
        sys.exit("Those didn't match. Nothing changed.")
    _write(path, make_hash(first))
    print(f"Saved to {path}. Everyone signed in before is now signed out.")


if __name__ == "__main__":
    main()
