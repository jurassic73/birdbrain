"""Read-only mode: visitors browse, only the admin (one password in .env) changes anything or calls Claude.

With no ADMIN_PASSWORD set everyone is admin, so a LAN-only install works as before."""
import hashlib
import hmac
import os
import secrets
import time

from fastapi import Request

COOKIE = "bb_admin"
MAX_AGE = 30 * 24 * 3600
# Pages a visitor never sees (prefixes); every POST is blocked for visitors too.
ADMIN_PATHS = ("/notes", "/lessons", "/review", "/false-triggers", "/settings", "/calibration")
MAX_FAILS = 5
FAIL_WINDOW = 15 * 60
_fails: dict[str, list[float]] = {}


def password() -> str:
    return os.environ.get("ADMIN_PASSWORD", "")


def _key() -> bytes:
    # Derived from the password, so changing it signs every device out.
    return hashlib.sha256(b"birdbrain-admin:" + password().encode()).digest()


def _sign(expires: int) -> str:
    return hmac.new(_key(), str(expires).encode(), hashlib.sha256).hexdigest()


def make_cookie() -> str:
    expires = int(time.time()) + MAX_AGE
    return f"{expires}.{_sign(expires)}"


def is_admin(request: Request) -> bool:
    if not password():
        return True
    expires, _, sig = request.cookies.get(COOKIE, "").partition(".")
    if not expires.isdigit() or int(expires) < time.time():
        return False
    return secrets.compare_digest(sig, _sign(int(expires)))


def check_password(given: str) -> bool:
    return bool(password()) and secrets.compare_digest(given.encode(), password().encode())


def blocked(ip: str) -> bool:
    """Too many wrong passwords from this address lately."""
    now = time.time()
    _fails[ip] = [t for t in _fails.get(ip, []) if now - t < FAIL_WINDOW]
    return len(_fails[ip]) >= MAX_FAILS


def failed(ip: str) -> None:
    _fails.setdefault(ip, []).append(time.time())


def admin_only(path: str) -> bool:
    return any(path == p or path.startswith(p + "/") for p in ADMIN_PATHS)
