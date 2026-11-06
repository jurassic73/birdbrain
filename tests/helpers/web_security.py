"""Shared helpers for the web security tests (used by more than one test module)."""
from starlette.routing import Route

from birdbrain.web import app as webapp


FILL = {"visit_id": "1", "name": "x", "source": "pi", "state": "x", "change_id": "1", "cal_id": "1"}


def _routes() -> list[tuple[str, str, str]]:
    """(method, template, concrete path) for every app route, e.g. ('GET', '/calibration/{name}', '/calibration/x')."""
    out = []
    for r in webapp.app.routes:
        if not isinstance(r, Route):
            continue  # the /static and /images mounts
        path = r.path
        for k, v in FILL.items():
            path = path.replace("{" + k + "}", v)
        assert "{" not in path, f"add a FILL value for {r.path}"
        for m in sorted(r.methods or []):
            if m != "HEAD":
                out.append((m, r.path, path))
    return out


ROUTES = _routes()
