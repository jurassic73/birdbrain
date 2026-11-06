"""Unit tests of birdbrain/web/auth: pure functions, no files, database or network."""

from starlette.routing import Mount

from helpers.public_no_admin import ADMIN, PUBLIC
from birdbrain.web import app as webapp, auth


def routes() -> set[tuple[str, str]]:
    """Every (method, template) the app answers; the mounts count as GET."""
    out = set()
    for r in webapp.app.routes:
        if isinstance(r, Mount):
            out.add(("GET", r.path))
        else:
            out |= {(m, r.path) for m in r.methods if m != "HEAD"}
    return out


def test_every_route_is_classified():
    have = routes()
    assert not PUBLIC & ADMIN
    assert not have - PUBLIC - ADMIN, "new route: may the public mirror serve it? Add it to PUBLIC (and " \
        f"auth.MIRROR_ROUTES) or ADMIN here: {sorted(have - PUBLIC - ADMIN)}"
    assert not (PUBLIC | ADMIN) - have, f"routes that are gone: {sorted((PUBLIC | ADMIN) - have)}"


def test_the_apps_allowlist_is_exactly_the_public_routes():
    assert auth.MIRROR_ROUTES == {path for method, path in PUBLIC if method == "GET"}
    assert auth.MIRROR_POSTS == {path for method, path in PUBLIC if method == "POST"}


def test_https_origin_behind_the_proxy():
    """nginx passes Host without the port; the browser's Origin for an https page has none either."""
    assert auth._host_port("somegeek.com", "https") == auth._host_port("somegeek.com:443", "https") == "somegeek.com"
    assert auth._host_port("192.168.1.20:8000", "http") == "192.168.1.20:8000"
