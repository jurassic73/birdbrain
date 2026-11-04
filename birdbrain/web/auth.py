"""Read-only mode: visitors browse, only the admin (one password in .env) changes anything or calls Claude.

With no ADMIN_PASSWORD set everyone is admin, so a LAN-only install works as before."""
import hashlib
import hmac
import os
import re
import secrets
import time
from urllib.parse import urlsplit

from fastapi import Request

from .. import config

COOKIE = "bb_admin"
MAX_AGE = 30 * 24 * 3600
# Pages a visitor never sees (prefixes); every POST is blocked for visitors too.
ADMIN_PATHS = ("/notes", "/lessons", "/review", "/false-triggers", "/settings", "/calibration")
# The public mirror (public()) serves only these routes, as app.py declares them, and only to GET/HEAD; the one
# thing it takes a POST for is the friends & family sign-in. Everything else (the admin pages, every change, Claude
# and camera calls, the admin's JSON) is a 404 there, as if it didn't exist. Deny by default: a new route stays off
# the mirror until it is added here, and tests/test_public_no_admin.py makes every route be one or the other.
MIRROR_ROUTES = frozenset({
    "/", "/live", "/scoreboard", "/visits", "/visits/archive-more", "/visits/more", "/visit/{visit_id}",
    "/species", "/species/{name}", "/species/{name}/more", "/species-info/{name}",
    "/stats", "/weather", "/about",
    "/api/latest-visit", "/live/latest.jpg", "/live.jpg", "/manifest.webmanifest", "/favicon.ico",
    "/guest-login", "/static", "/images",
})
MIRROR_POSTS = frozenset({"/guest-login"})
MAX_FAILS = 5
FAIL_WINDOW = 15 * 60
_fails: dict[str, list[float]] = {}
# A random secret the cookies are signed with, made once per install (DATA_DIR/admin_secret, mode 0600).
SECRET_FILE = "admin_secret"
_secrets: dict[str, bytes] = {}
_SIG = re.compile(r"[0-9a-f]{64}")


def public() -> bool:
    """Public mirror (BIRDBRAIN_PUBLIC=1): a copy fed by the home Pi. Nobody is admin, there is no login,
    and the camera is never contacted; the live frame is whatever data/live.jpg the Pi last sent."""
    return os.environ.get("BIRDBRAIN_PUBLIC", "").strip().lower() in ("1", "true", "yes")


def password() -> str:
    return os.environ.get("ADMIN_PASSWORD", "")


def _secret() -> bytes:
    """The install's random signing secret, created on first use. Readable only by the service's user, so a
    cookie seen on the wire (the Pi speaks plain http) gives no way to test password guesses offline."""
    path = config.DATA_DIR / SECRET_FILE
    key = str(path)
    if key not in _secrets:
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            pass
        except FileNotFoundError:
            path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        else:
            with os.fdopen(fd, "w") as f:
                f.write(secrets.token_hex(32) + "\n")
        _secrets[key] = path.read_bytes().strip()
    return _secrets[key]


def _key() -> bytes:
    # The secret keeps the key unguessable; the password in it means changing it signs every device out.
    return hmac.new(_secret(), b"birdbrain-admin:" + password().encode(), hashlib.sha256).digest()


def _sign(expires: int) -> str:
    return hmac.new(_key(), str(expires).encode(), hashlib.sha256).hexdigest()


def make_cookie() -> str:
    expires = int(time.time()) + MAX_AGE
    return f"{expires}.{_sign(expires)}"


def is_admin(request: Request) -> bool:
    if public():
        return False
    if not password():
        return True
    expires, _, sig = request.cookies.get(COOKIE, "").partition(".")
    # Anything but "<digits>.<64 hex>" is no cookie at all (a stray non-ASCII value must not raise).
    if not expires.isdigit() or not _SIG.fullmatch(sig) or int(expires) < time.time():
        return False
    return secrets.compare_digest(sig, _sign(int(expires)))


def check_password(given: str) -> bool:
    return not public() and bool(password()) and secrets.compare_digest(given.encode(), password().encode())


def _host_port(netloc: str, scheme: str) -> str:
    """'host:port' with the scheme's default port dropped, lowercased, so Origin and Host compare alike."""
    netloc = netloc.lower()
    if (scheme == "http" and netloc.endswith(":80")) or (scheme == "https" and netloc.endswith(":443")):
        netloc = netloc.rsplit(":", 1)[0]
    return netloc


def cross_site(request: Request) -> bool:
    """Whether a browser says this request was started by another site (a page that auto-submits a form to
    the Pi, say). Sec-Fetch-Site is the browser's own word; failing that, the Origin must be this site's
    host. Requests with neither header (curl, scripts) are let through: they carry no cookie to abuse."""
    site = request.headers.get("sec-fetch-site")
    if site is not None:
        return site != "same-origin"
    origin = request.headers.get("origin")
    if origin is None:
        return False
    parts = urlsplit(origin)
    if not parts.netloc:
        return True  # "null" or malformed
    host = request.headers.get("host", "")
    return _host_port(parts.netloc, parts.scheme) != _host_port(host, parts.scheme)


def blocked(ip: str) -> bool:
    """Too many wrong passwords from this address lately."""
    now = time.time()
    _fails[ip] = [t for t in _fails.get(ip, []) if now - t < FAIL_WINDOW]
    return len(_fails[ip]) >= MAX_FAILS


def failed(ip: str) -> None:
    _fails.setdefault(ip, []).append(time.time())


def admin_only(path: str) -> bool:
    return any(path == p or path.startswith(p + "/") for p in ADMIN_PATHS)


def on_mirror(method: str, route: str | None) -> bool:
    """Whether the public mirror answers this request: `route` is the path template it goes to (None: no route)."""
    if route in MIRROR_POSTS and method == "POST":
        return True
    return method in ("GET", "HEAD") and route in MIRROR_ROUTES
