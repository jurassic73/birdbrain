import asyncio
import io
import json
import logging
import math
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta
from pathlib import Path
from urllib.parse import quote, urlencode

from fastapi import Depends, FastAPI, Form, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image, ImageStat
from starlette.routing import Match

from . import auth, steps, viewer
from .. import alert, camera, circle_check, classifier, clock, config, db, identify, images, lessons, mirror, names, outlook, presence, privacy, refine, retention, species_info, weather, zones

HERE = Path(__file__).parent
log = logging.getLogger("birdbrain.web")
PAGE_SIZE = 24
REVIEW_SHOWN = 200  # the review queue is oldest first and short; it shows this many and says so
SPECIES_PAGE = 48  # a species page loads this many at a time (a common bird has thousands)
FT_PAGE = 48  # false triggers come in pages too, so a rainy week isn't one 20,000 px scroll on a phone


@asynccontextmanager
async def lifespan(app: FastAPI):
    with db.session() as conn:
        db.init_db(conn)
    yield


def images_dir() -> Path:
    """The photo folder served as /images. The Pi's site makes it; the public mirror never does. There the push
    user owns it and every push gives folders and files the group the service reads through (scripts/push_public.sh,
    README "a public read-only copy"); a folder the service made itself would belong to the service's user and
    group, with no write for the push user, and stop every push. Until the first push the mirror answers 404 for
    photos, which is why the mount below doesn't insist on the folder existing."""
    if not auth.public():
        config.IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    return config.IMAGES_DIR


class PhotoFiles(StaticFiles):
    """Static files from a folder that may not exist yet (the mirror before its first push): a photo asked for
    then is a 404, not an error, and the folder is used as soon as it appears, without a restart."""

    async def check_config(self) -> None:
        try:
            await super().check_config()
        except RuntimeError:  # "directory does not exist": not yet
            pass


app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
app.mount("/images", PhotoFiles(directory=images_dir(), check_dir=False), name="images")
templates = Jinja2Templates(directory=HERE / "templates")


# The one change a visitor may make: telling the site a recording on a species card wouldn't play. It only
# sets a flag (checked against the card's own saved sounds); the public mirror still refuses every POST.
_SOUND_BROKEN = re.compile(r"^/species-info/[^/]+/sound-broken$")


def not_found() -> Response:
    """The very answer FastAPI gives for a path no route has."""
    return JSONResponse({"detail": "Not Found"}, status_code=404)


def route_of(scope) -> str | None:
    """The path template of the route (or mount) a request goes to, e.g. "/visit/{visit_id}", picked as the router
    picks it: the first route that fits path and method, else the first that fits the path; None when none does."""
    partial = None
    for route in app.router.routes:
        match = route.matches(scope)[0]
        if match is Match.FULL:
            return route.path
        if match is Match.PARTIAL and partial is None:
            partial = route.path
    return partial


def admin_only_burst(rel: str) -> bool:
    """Whether a file under /images is a burst frame (or a burst frame's thumbnail) of a visit only the admin may open:
    a false trigger or an "other" (db.ADMIN_ONLY). Only file names shaped like a burst frame (images.burst_file) are
    looked up, with one indexed query, so the gallery's photos and thumbnails never touch the database. A real
    visit's burst (a false trigger the owner corrected to an animal) stays public."""
    rel = os.path.normpath(rel.lstrip("/"))  # as StaticFiles reads it: "//", "./" and ".." steps resolved
    parsed = images.burst_file(os.path.basename(rel))
    if not parsed:
        return False
    with db.session() as conn:
        return db.is_admin_only_photo(conn, os.path.join(os.path.dirname(rel), parsed[0] + ".jpg"))


@app.middleware("http")
async def read_only(request: Request, call_next):
    """Visitors can browse; changes, Claude calls and the admin pages need the admin login."""
    if request.method not in ("GET", "HEAD") and auth.cross_site(request):
        # Another site's page can't change anything here with the browser's cookies (or, with no password,
        # with nothing at all): every change is a POST, and the browser tells us where a POST came from.
        return Response("Cross-site requests aren't accepted.", status_code=403)
    try:
        gate_on = viewer.enabled()
        gated = gate_on and not viewer.is_open(request.url.path) and not viewer.signed_in(request)
    except viewer.Unreadable as e:
        # The gate's file is there but can't be read: shut, not open, until someone fixes the permissions.
        log.error("friends & family gate can't be read, refusing everyone: %s", e)
        return Response("Birdbrain is closed for a moment: its sign-in settings can't be read.", status_code=503)
    if gated:
        # Friends & family gate: nothing past this point for anyone without the guest cookie. The redirect is
        # never cached: a CDN in front (Cloudflare caches .jpg URLs by default, whatever cookie the request had)
        # would otherwise keep one cookieless fetch of a photo URL and hand the redirect to every signed-in guest
        # as that photo, for minutes, which looks like broken images.
        if request.method in ("GET", "HEAD"):
            return RedirectResponse("/guest-login?" + urlencode({"next": request.url.path}), status_code=303,
                                    headers={"Cache-Control": "no-store"})
        return Response("Sign in first.", status_code=401, headers={"Cache-Control": "no-store"})
    admin = request.state.admin = auth.is_admin(request)
    if clock.stale():
        with db.session() as conn:
            clock.set_zone(db.get_value(conn, clock.KEY))
    path = request.url.path
    if auth.public():
        # The mirror has no admin at all: anything but its public pages (auth.MIRROR_ROUTES) is the same 404 as a
        # page that doesn't exist, every change and admin page included, never a login or a "read-only" answer.
        if not auth.on_mirror(request.method, route_of(request.scope)):
            return not_found()
    elif not admin and path not in ("/login", "/logout"):
        if request.method not in ("GET", "HEAD") and not _SOUND_BROKEN.match(path):
            return Response("Read-only: log in as admin to change things.", status_code=403)
        if auth.admin_only(path):
            return RedirectResponse("/login?" + urlencode({"next": path}), status_code=303)
        if path.startswith("/images/") and admin_only_burst(path.removeprefix("/images/")):
            return not_found()  # false triggers and "other" visits are the admin's: their burst frames too, by exact URL
    # Everything a visitor's request reads leaves out the "other" visits (db.shown): lists, counts, the latest
    # visitor, arrows, species and stats. The admin sees them all.
    with db.visitor_view(not admin):
        response = await call_next(request)
    if gate_on and not viewer.is_open(path):
        # Behind the gate, photos and pages are for signed-in guests only: a shared cache in front must not keep
        # them (or serve a guest's photo to someone without the cookie). The browser's own cache still may.
        # Static files stay open and versioned, so they keep whatever caching they had.
        response.headers.setdefault("Cache-Control", "private")
    return response


# Serving under a path such as somegeek.com/birdbrain/: the proxy strips the prefix on the way in, and this
# puts it back on the way out (links, forms, redirects), so the templates can keep their plain "/..." URLs.
BASE = os.environ.get("BIRDBRAIN_BASE", "").strip().rstrip("/")
_URL_ATTR = re.compile(r'(\s(?:href|src|action|formaction|hx-get|hx-post|data-[a-z-]+)=")/(?!/)')
# fetch() calls in the page and in static/*.js use "/..." too; this shim prefixes those.
_FETCH_SHIM = ('<script>window.BB_BASE={base};(()=>{{const f=window.fetch;window.fetch=(u,o)=>f('
               'typeof u==="string"&&u.startsWith("/")&&!u.startsWith("//")?{base}+u:u,o)}})();</script>')


def site_url(path: str) -> str:
    """A site URL as the browser must ask for it: `path` ("/images/...") with the base path in front. The
    base_path middleware rewrites HTML, but JSON isn't touched, so any route that answers a URL in JSON (for a
    script to put in an img src or a fetch) builds it with this. BASE is read at call time (tests set it)."""
    return BASE + path if path.startswith("/") and not path.startswith("//") else path


@app.middleware("http")
async def base_path(request: Request, call_next):
    response = await call_next(request)
    if not BASE:
        return response
    loc = response.headers.get("location", "")
    if loc.startswith("/") and not loc.startswith("//"):
        response.headers["location"] = BASE + loc
    if not response.headers.get("content-type", "").startswith("text/html"):
        return response
    body = b"".join([chunk async for chunk in response.body_iterator]).decode()
    body = _URL_ATTR.sub(lambda m: m.group(1) + BASE + "/", body)
    body = body.replace('"href_matches": "/', '"href_matches": "' + BASE + "/")  # prerender hints
    body = body.replace("<head>", "<head>" + _FETCH_SHIM.format(base=json.dumps(BASE)), 1)
    headers = {k: v for k, v in response.headers.items() if k.lower() != "content-length"}
    return Response(body, status_code=response.status_code, headers=headers)


def nice_time(value: str) -> str:
    return datetime.fromisoformat(value).strftime("%b %-d, %-I:%M %p")


def nice_date(value: str) -> str:
    return datetime.fromisoformat(value).strftime("%b %-d, %Y")


def short_time(value: str) -> str:
    """'9:58 AM' for today, 'Oct 24, 9:58 AM' otherwise."""
    dt = datetime.fromisoformat(value)
    return dt.strftime("%-I:%M %p") if dt.date() == clock.today() else dt.strftime("%b %-d, %-I:%M %p")


def _plural(n: int, unit: str) -> str:
    return f"{n} {unit}{'' if n == 1 else 's'} ago"


def _days_ago(days: int) -> str:
    """'1 day ago' .. '6 days ago', then weeks, months, years."""
    if days < 7:
        return _plural(days, "day")
    if days < 30:
        return _plural(days // 7, "week")
    if days < 365:
        return _plural(days // 30, "month")
    return _plural(days // 365, "year")


def since_ago(value: str) -> str:
    """A rough 'how long since' for a date: 'today', 'yesterday', '~5 days ago', '~3 weeks ago', '~4 months ago',
    '~2 years ago' (counted in calendar days, rounded to the nearest unit)."""
    days = max(0, (clock.today() - datetime.fromisoformat(value).date()).days)
    if days == 0:
        return "today"
    if days == 1:
        return "yesterday"
    if days < 14:
        return f"~{days} days ago"
    if days < 60:
        return f"~{round(days / 7)} weeks ago"
    if days < 730:
        return f"~{round(days / 30.4)} months ago"
    return f"~{round(days / 365)} years ago"


def age_seconds(value: str) -> int:
    """Seconds since an ISO time (0 for the future); the home page's script ticks from this."""
    return max(0, int((clock.now() - datetime.fromisoformat(value)).total_seconds()))


def ago(value: str, bare: bool = False) -> str:
    """'(just now)', '(5 min ago)', '(~2 hours ago)', then days, weeks, months, years, in brackets for use right
    after a clock time. bare=True drops the brackets ('2 days ago') for a line with no time before it.
    static/ago.js mirrors this wording and keeps the page's copies current."""
    secs = age_seconds(value)
    if secs < 60:
        text = "just now"
    elif secs < 3600:
        text = f"{secs // 60} min ago"
    elif secs < 86400:
        text = f"~{_plural(secs // 3600, 'hour')}"
    else:
        text = _days_ago(secs // 86400)
    return text if bare else f"({text})"


def stayed(captured_at: str, ended_at: str | None) -> str:
    """How long a visitor stayed, from a minute up ('stayed 3 min'); shorter stays say nothing."""
    if not ended_at:
        return ""
    seconds = (datetime.fromisoformat(ended_at) - datetime.fromisoformat(captured_at)).total_seconds()
    return f"stayed {round(seconds / 60)} min" if seconds >= 60 else ""


def usd(value: float) -> str:
    """Small dollar amounts readably: $0.42, $0.031, or under a cent."""
    if not value:
        return "$0"
    if value < 0.001:
        return "under $0.001"
    return f"${value:.2f}" if value >= 0.1 else f"${value:.3f}"


def mb(value: int) -> str:
    """Bytes as MB (or GB past 1000 MB)."""
    return f"{value / 1e9:.1f} GB" if value >= 1e9 else f"{value / 1e6:.0f} MB" if value >= 1e6 else f"{value / 1e6:.1f} MB"


# The Weather & visits page's units: "us" (°F, mph) or "metric" (°C, km/h); the other one shows in a tip.
UNITS = "us"
TEMP_UNIT = "F" if UNITS == "us" else "C"
WIND_UNIT = "mph" if UNITS == "us" else "km/h"


def _wind(kmh: float | None, unit: str) -> str:
    if kmh is None:
        return "n/a"
    return f"{round(kmh / 1.609344)} mph" if unit == "mph" else f"{round(kmh)} km/h"


def wind(kmh: float | None) -> str:
    """A wind speed (stored km/h) in the page's unit: '6 mph', or 'n/a'."""
    return _wind(kmh, WIND_UNIT)


def wind_alt(kmh: float | None) -> str:
    return _wind(kmh, "km/h" if WIND_UNIT == "mph" else "mph")


def _temp(c: float | None, unit: str) -> str:
    if c is None:
        return "?"
    return f"{round(c * 9 / 5 + 32)}°" if unit == "F" else f"{round(c)}°"


def temp(c: float | None) -> str:
    """A Celsius temperature in the page's unit: '56°'."""
    return _temp(c, TEMP_UNIT)


def temp_alt(c: float | None) -> str:
    """The same temperature in the other unit, for a tip: '13°C'."""
    other = "C" if TEMP_UNIT == "F" else "F"
    return _temp(c, other) + other


templates.env.globals["temp_unit"] = TEMP_UNIT
templates.env.globals["wind_unit"] = WIND_UNIT
templates.env.filters.update(temp=temp, temp_alt=temp_alt, wind=wind, wind_alt=wind_alt)
templates.env.filters.update(nice_time=nice_time, nice_date=nice_date, since_ago=since_ago, short_time=short_time, ago=ago, age_seconds=age_seconds, usd=usd, mb=mb,
                             month_name=lambda v: datetime.fromisoformat(v).strftime("%B"))
templates.env.filters["tile_fact"] = species_info.fact_item
templates.env.globals["stayed"] = lambda v: stayed(v["captured_at"], v["ended_at"]) if "ended_at" in v.keys() else ""
templates.env.globals["mirror_status"] = privacy.visit_status
templates.env.globals["has_card"] = species_info.has_card
templates.env.globals["steps_bottom"] = steps.bottom_line
templates.env.globals["season_badge"] = species_info.visit_badge
templates.env.globals["short_status"] = species_info.short_status
# Cards show the visit's photo (the burst frame the owner picked, else the one Claude saw) with the circles drawn
# on that very frame; circles on other burst frames stay on the visit page.
templates.env.globals["photo_marks"] = lambda visit: [m for m in start_marks(visit)
                                                     if db.mark_frame(m) == db.photo_frame(visit)]
templates.env.globals["no_circle_note"] = lambda visit: no_circle_note(visit)
templates.env.globals["also_of"] = db.also_of
templates.env.globals["answerer"] = db.answerer
# The biggest photo of a visit still on disk (just the thumbnail once the photo budget cut it down).
templates.env.globals["photo_src"] = db.photo_rel
templates.env.globals["thumb_src"] = db.thumb_rel
templates.env.globals["also_label"] = db.also_label
# Templates check `request.state.admin` (set by the read_only middleware); this says whether a login exists at all.
templates.env.globals["public"] = auth.public
templates.env.globals["login_on"] = lambda: bool(auth.password()) and not auth.public()
# Changes whenever a static file changes, so browsers fetch fresh CSS/JS after a deploy.
templates.env.globals["asset_version"] = int(max(p.stat().st_mtime for p in (HERE / "static").iterdir()))


def get_conn(request: Request):
    with db.session() as conn:
        request.state.conn = conn  # the nav's waiting counts reuse it instead of opening another
        yield conn


def nav_waiting(request: Request) -> dict:
    """Review and False triggers counts for the nav bubbles. Admins only, and only when base.html renders
    (full pages; the partials never extend it)."""
    conn = getattr(request.state, "conn", None)
    if conn is None or not request.state.admin or auth.public():
        return {"review": 0, "false": 0}
    return {"review": db.count_visits(conn, review=True), "false": db.count_unchecked_false_triggers(conn)}


def count_label(n: int) -> str:
    return "99+" if n > 99 else str(n)


templates.env.globals["nav_waiting"] = nav_waiting


def star_covers(request: Request) -> dict:
    """Species -> visit id of the owner's cover picks, for the stars on cards (admins only; one query per page)."""
    conn = getattr(request.state, "conn", None)
    if conn is None or not request.state.admin:
        return {}
    return db.species_photos(conn)


templates.env.globals["star_covers"] = star_covers
templates.env.globals["count_label"] = count_label


def lessons_popup(request: Request) -> dict | None:
    """The Lessons summary draft an automatic distill left, for the review pop-up in base.html. Admins only (never
    on the public mirror), and not on Claude's Notes itself, where the draft is already on the page."""
    conn = getattr(request.state, "conn", None)
    if conn is None or not request.state.admin or auth.public() or request.url.path == "/notes":
        return None
    return lessons.pending_auto_draft(conn)


templates.env.globals["lessons_popup"] = lessons_popup


def filters_from(request: Request) -> dict:
    q = request.query_params
    return {
        "species": q.get("species") or None,
        "kind": q.get("kind") or None,
        "tod": q.get("tod") or None,
        "date_from": q.get("date_from") or None,
        "date_to": q.get("date_to") or None,
    }


def query_string(filters: dict, **extra) -> str:
    return urlencode({k: v for k, v in {**filters, **extra}.items() if v})


def page(request: Request, name: str, status_code: int = 200, **ctx) -> HTMLResponse:
    return templates.TemplateResponse(request, name, ctx, status_code=status_code)


def safe_next(target: str) -> str:
    """Only same-site paths, so the login can't be used to bounce people elsewhere."""
    return target if target.startswith("/") and not target.startswith("//") else "/"


def _guest_reply(request: Request, ok: bool, next: str, error: str = "", wait: int = 0, status: int = 200):
    """JSON for the pop-up on the somegeek.com tiles page, the plain sign-in page otherwise."""
    if "application/json" in request.headers.get("accept", ""):
        return JSONResponse({"ok": ok, "error": error, "wait": wait}, status_code=status)
    if ok:
        return RedirectResponse(safe_next(next), status_code=303)
    return templates.TemplateResponse(request, "guest_login.html", {"next": safe_next(next), "error": error,
                                                                    "wait": wait}, status_code=status)


def _wait_text(seconds: int) -> str:
    minutes = max(1, round(seconds / 60))
    return f"Too many tries. Wait {minutes} minute{'s' if minutes != 1 else ''} and try again."


@app.get("/guest-login", response_class=HTMLResponse)
def guest_login_page(request: Request, next: str = "/"):
    if not viewer.enabled() or viewer.signed_in(request):
        return RedirectResponse(safe_next(next), status_code=303)
    wait = viewer.locked(viewer.client_ip(request))
    return _guest_reply(request, False, next, _wait_text(wait) if wait else "", wait)


@app.post("/guest-login")
def guest_login(request: Request, password: str = Form(""), key: str = Form(""), next: str = Form("/")):
    if not viewer.enabled() or viewer.signed_in(request):  # already in: no need to check, or to throttle
        return _guest_reply(request, True, next)
    ip = viewer.client_ip(request)
    if wait := viewer.locked(ip):
        return _guest_reply(request, False, next, _wait_text(wait), wait, 429)
    if key:
        ok = viewer.check_invite(key)  # a share link is unguessable, so it's never slowed down
    else:
        if delay := viewer.slow_down():  # everyone together has been guessing a lot: each try waits a little
            time.sleep(delay)
        ok = viewer.check_password(password)
    if not ok:
        viewer.failed(ip)
        wait = viewer.locked(ip)
        return _guest_reply(request, False, next, _wait_text(wait) if wait else
                            ("That link isn't valid anymore. Ask for a new one." if key else "That password isn't right."),
                            wait,
                            429 if wait else 401)
    viewer.succeeded(ip)
    resp = _guest_reply(request, True, next)
    # Path covers the whole site whether it's served at / or under a prefix such as /birdbrain.
    resp.set_cookie(viewer.COOKIE, viewer.make_cookie(), max_age=viewer.MAX_AGE, httponly=True, samesite="lax",
                    path=(BASE or "") + "/",
                    secure=https(request))
    return resp


def https(request: Request) -> bool:
    """Whether the visitor's connection is HTTPS, also behind Cloudflare's tunnel and nginx, which talk plain HTTP."""
    return (request.url.scheme == "https" or request.headers.get("x-forwarded-proto") == "https"
            or '"https"' in request.headers.get("cf-visitor", ""))


@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request, next: str = "/"):
    if request.state.admin:
        return RedirectResponse(safe_next(next), status_code=303)
    return page(request, "login.html", next=safe_next(next), error=None)


@app.post("/login")
def login(request: Request, password: str = Form(""), next: str = Form("/")):
    ip = request.client.host if request.client else "?"
    if auth.blocked(ip):
        return page(request, "login.html", 429, next=safe_next(next), error="Too many tries. Wait 15 minutes and try again.")
    if not auth.check_password(password):
        auth.failed(ip)
        return page(request, "login.html", 401, next=safe_next(next), error="That password isn't right.")
    resp = RedirectResponse(safe_next(next), status_code=303)
    resp.set_cookie(auth.COOKIE, auth.make_cookie(), max_age=auth.MAX_AGE, httponly=True, samesite="lax",
                    secure=https(request))
    return resp


@app.post("/logout")
def logout():
    resp = RedirectResponse("/", status_code=303)
    resp.delete_cookie(auth.COOKIE)
    return resp


def quick_filters() -> list[tuple[str, dict]]:
    today = clock.today()
    return [
        ("All", {}),
        ("Birds", {"kind": "bird"}),
        ("Mammals", {"kind": "mammal"}),
        ("Multiple", {"kind": "multiple"}),
        ("Night", {"tod": "night"}),
        ("Today", {"date_from": today.isoformat(), "date_to": today.isoformat()}),
        ("This week", {"date_from": (today - timedelta(days=6)).isoformat()}),
    ]


def species_filters(request: Request) -> dict:
    """The filters a species page takes from its query string: the home page's, minus the Species box and the
    Birds / Mammals kinds, which mean nothing for one species. Multiple animals stays."""
    filters = filters_from(request)
    del filters["species"]
    if filters["kind"] != "multiple":
        filters["kind"] = None
    return filters


def species_chips(name: str, filters: dict, sort: str) -> list[tuple[str, str, bool]]:
    """The quick filter chips for a species' Visits tab: the home page's list minus Birds and Mammals. Links keep
    the species list's sort (the phone swipe order) and end in #visits so a shared link opens on the Visits tab."""
    active = {k: v for k, v in filters.items() if v}
    keep = {"sort": sort} if sort != "visits" else {}
    base = f"/species/{quote(name)}"
    chips = []
    for label, f in quick_filters():
        if f.get("kind") in ("bird", "mammal"):
            continue
        q = query_string(f, **keep)
        chips.append((label, f"{base}?{q}#visits" if q else f"{base}#visits", f == active))
    return chips


# The Today panel's scoreboard periods: key, button, heading words, days back (None: all time). Yesterday is special.
SCORE_RANGES = [
    ("today", "Today", "today", 0),
    ("yesterday", "Yesterday", "yesterday", 1),
    ("week", "Week", "this week", 6),
    ("month", "Month", "this month", 29),
    ("year", "Year", "this year", 364),
    ("all", "All time", "of all time", None),
]


def score_range(key: str | None) -> dict:
    """One scoreboard period with its date bounds; an unknown key is today."""
    key, button, words, back = next((r for r in SCORE_RANGES if r[0] == key), SCORE_RANGES[0])
    today = clock.today()
    start = None if back is None else (today - timedelta(days=back)).isoformat()
    end = (today - timedelta(days=1)).isoformat() if key == "yesterday" else None
    return {"key": key, "words": words, "date_from": start, "date_to": end or (start and today.isoformat())}


def scoreboard_ctx(conn, key: str | None) -> dict:
    r = score_range(key)
    return {"score": r, "score_rows": db.scoreboard(conn, r["date_from"], r["date_to"]),
            "score_total": db.scoreboard_total(conn, r["date_from"], r["date_to"]), "score_ranges": SCORE_RANGES}


@app.get("/", response_class=HTMLResponse)
def home(request: Request, conn=Depends(get_conn)):
    filters = filters_from(request)
    active = {k: v for k, v in filters.items() if v}
    visits = db.list_visits(conn, **filters, limit=PAGE_SIZE)
    chips = [(name, f"/?{query_string(f)}" if f else "/", f == active) for name, f in quick_filters()]
    today = db.today_summary(conn)
    return page(
        request,
        "home.html",
        # Facts about the latest visitor's species for the bottom of the Today panel; the cached card only.
        latest_facts=species_info.cached_facts(conn, today["latest"]["label"]) if today["latest"] else [],
        visits=visits,
        filters=filters,
        filtered=bool(active),
        more_filters_open=bool(filters["species"] or (filters["date_to"] and filters["date_from"] != filters["date_to"])),
        chips=chips,
        total=db.count_visits(conn, **filters),
        any_visits=db.latest_visit(conn) is not None,
        today=today,
        presence_mode=presence.mode(conn),
        activity=db.activity(conn),
        species_names=db.species_names(conn),
        next_url=more_url("/visits/more", visits, PAGE_SIZE, filters),
        split_day=None if active else clock.today().isoformat(),
        hero_id=today["latest"]["id"] if today["latest"] else None,
        # The Today and This week chips show the same period on the scoreboard.
        **scoreboard_ctx(conn, next(({"Today": "today", "This week": "week"}.get(n) for n, _, on in chips if on), None)),
    )


@app.get("/live", response_class=HTMLResponse)
def live_page(request: Request, conn=Depends(get_conn)):
    """The live view on its own tab: the frame, plus a slim panel (light, frame age, today, the last visitor)."""
    load_saved_frame()
    return page(
        request,
        "live.html",
        today=db.today_summary(conn),
        light=frame_light(),
        live_refresh=config.LIVE_REFRESH,
        live_stale=LIVE_STALE,
    )


@app.get("/scoreboard", response_class=HTMLResponse)
def scoreboard(request: Request, range: str = "today", conn=Depends(get_conn)):
    return page(request, "_scoreboard.html", **scoreboard_ctx(conn, range))


def _with_query(path: str, request: Request) -> str:
    """`path` plus the request's query string, so a redirect keeps its filters."""
    q = request.url.query
    return f"{path}?{q}" if q else path


@app.get("/visits")
def visits_archive(request: Request):
    """The old Visits archive (removed 2026-11-03: the home page is the Visits tab now). Old links and bookmarks
    land on the home page with the same filters, which use the same names (see filters_from)."""
    return RedirectResponse(_with_query("/", request), status_code=307)


def keyset(visit) -> str:
    """The `before` cursor for the page after this card: its time and id, which fix its place in the newest-first order."""
    return f"{visit['captured_at']},{visit['id']}"


def visits_before(conn, before: str = "", *, limit: int, **filters) -> list:
    """Visits newest first, only those older than the `before` cursor (the last card already shown, see keyset()).
    Unlike an offset this can't skip or repeat a card when the list changes between pages, e.g. when a false trigger
    is confirmed and leaves it. A bad cursor is the first page."""
    where, args = db._where(**filters)
    ts, _, vid = before.rpartition(",")
    if ts and vid.isdigit():
        where += " AND (captured_at < ? OR (captured_at = ? AND id < ?))"
        args = [*args, ts, ts, int(vid)]
    return conn.execute(
        f"SELECT *, {db.LABEL} AS label FROM visits WHERE {where} ORDER BY captured_at DESC, id DESC LIMIT ?",
        (*args, limit)).fetchall()


def more_url(path: str, visits: list, limit: int, filters: dict | None = None, **extra) -> str | None:
    """The Load more link after a full page of `visits`, or None when that was the last page."""
    if len(visits) < limit:
        return None
    return f"{path}?{query_string(filters or {}, before=keyset(visits[-1]), **extra)}"


@app.get("/visits/archive-more")
def visits_archive_more(request: Request):
    """The archive's Load more: a page still open from before the archive went keeps paging through the home list."""
    return RedirectResponse(_with_query("/visits/more", request), status_code=307)


@app.get("/visits/more", response_class=HTMLResponse)
def more_visits(request: Request, before: str = "", conn=Depends(get_conn)):
    filters = filters_from(request)
    visits = visits_before(conn, before, limit=PAGE_SIZE, **filters)
    # Unfiltered, the cards are split into Today and Earlier; the card just before this page (the cursor's) says which side it was on.
    split_day = prev_day = None
    if not any(filters.values()) and before:
        split_day = clock.today().isoformat()
        prev_day = before[:10]
    return page(request, "_cards.html", visits=visits, split_day=split_day, prev_day=prev_day,
                next_url=more_url("/visits/more", visits, PAGE_SIZE, filters))


@app.get("/visit/{visit_id}", response_class=HTMLResponse)
def visit_detail(request: Request, visit_id: int, ft: bool = False, todo: bool = False,
                 species: str = "", conn=Depends(get_conn)):
    visit = db.get_visit(conn, visit_id)
    if not visit:
        raise HTTPException(404)
    is_ft = db.is_false_trigger(conn, visit_id)
    if not request.state.admin:
        # False triggers and "other" visits are for the admin only: a visitor gets the same 404 as for an id that
        # doesn't exist (the mirror has none of them either), and the ft flag can't switch a real visit's arrows
        # over to them.
        if db.is_admin_only(conn, visit_id):
            raise HTTPException(404)
        ft = False
    is_ft = ft or is_ft
    # todo: opened from the False triggers list of unconfirmed photos, so the arrows (and C) skip confirmed ones.
    todo = todo and is_ft
    # species: opened from that species' page, so the arrows step through only its visits.
    scope = "" if is_ft else species
    prev_id, next_id = db.get_neighbors(conn, visit_id, false_triggers=is_ft, unchecked=todo, species=scope or None)
    # The arrow targets' photos, prefetched so stepping through visits paints without a blank frame.
    neighbor_images = [db.photo_rel(v) for v in (db.get_visit(conn, i) for i in (prev_id, next_id) if i) if v]
    decision = decision_steps(visit["raw_json"])
    teaches = presence.teaches(conn, visit_id) if request.state.admin else None
    return page(
        request,
        "visit.html",
        visit=visit,
        visitors=db.VISITORS,
        species_kinds=db.species_kinds(conn),
        decision=decision,
        classifier_mode=config.CLASSIFIER,
        marks=start_marks(visit),
        prev_id=prev_id,
        next_id=next_id,
        neighbor_images=neighbor_images,
        is_ft=is_ft,
        todo=todo,
        scope=scope,
        confirmed=db.confirmed_species(conn),
        burst=burst_frames(visit),
        claude_frame=db.claude_frame(visit["raw_json"]),
        photo_frame=db.photo_frame(visit),
        teaches=teaches,
        weight=f"{presence.CONFIRMED_WEIGHT:g}",
        lesson_limit=db.LESSON_LIMIT,
        trains=photo_trains(conn, visit, decision, teaches) if request.state.admin else None,
        vzones=visit_zones(visit, conn),
        pi_average=db.pi_check_average(conn),
    )


def photo_trains(conn, visit, decision: dict, teaches: dict | None) -> list[dict]:
    """The visit page's "What this photo trains": the four components and what this photo does to each (steps.py)."""
    lesson = None
    if any(visit[k] is not None for k in ("corrected_visitor", "corrected_species", "feedback", "corrected_also")):
        lesson = next((l for l in db.all_lessons(conn) if l["visit_id"] == visit["id"]), None)
    return steps.for_visit(visit, decision, teaches, lesson, db.settled_species(conn),
                           visit["corrected_visitor"] or visit["visitor"], config.CLASSIFIER)


def burst_frames(visit) -> list[dict]:
    """A false trigger's whole burst for the visit page's filmstrip, in capture order: each frame's path,
    seconds after the first, whether it's the one Claude saw (`best`) and whether it's the visit's photo (`on`:
    the frame the owner picked, else Claude's). Empty when only the best frame was kept."""
    paths = images.burst_paths(visit["image_path"])
    if not paths:
        return []
    try:
        info = (json.loads(visit["raw_json"] or "{}") or {}).get("burst") or {}
    except (ValueError, AttributeError):
        info = {}
    fps = info.get("fps") or config.BURST_FPS
    shown = db.photo_frame(visit) or db.claude_frame(visit["raw_json"])
    return [{"path": p, "at": i / fps, "best": i == info.get("best"), "on": i + 1 == shown}
            for i, p in enumerate(paths)]


@app.post("/visit/{visit_id}/photo-frame")
def set_photo_frame(visit_id: int, frame: str = Form(""), conn=Depends(get_conn)):
    """The owner picks a burst frame as a false trigger's photo (the big photo and its card), or takes the pick
    back (frame 0, empty or Claude's own frame). Nothing is moved or deleted: image_path stays the frame Claude
    saw, the pick is a column, and the picked frame gets a thumbnail of its own for the cards (images.pick_thumb)."""
    v = db.get_visit(conn, visit_id)
    if not v:
        raise HTTPException(404)
    if not db.is_false_trigger(conn, visit_id):
        raise HTTPException(400, "Only a false trigger keeps its burst, so only there can the photo be picked.")
    if v["photo_deleted_at"] or v["photo_level"] == "thumb":
        raise HTTPException(400, "This visit's photos were deleted to save space.")
    try:
        n = int(frame.strip() or 0)
    except ValueError:
        raise HTTPException(400, "The frame must be a number.") from None
    if n == db.claude_frame(v["raw_json"]):
        n = 0  # Claude's own frame: the same as no pick
    if n:
        if n < 0 or n > len(images.burst_paths(v["image_path"])):
            raise HTTPException(400, "No such burst frame.")
        try:
            images.pick_thumb(v["image_path"], n)
        except FileNotFoundError:
            raise HTTPException(400, "That burst frame is no longer on disk.") from None
    else:
        images.drop_burst_thumbs(v["image_path"])
    db.set_photo_frame(conn, visit_id, n or None)
    return Response(headers=REFRESH)


FIRED = {"rim": "the rim and rocks", "water": "the open water", "ground": "the ground"}


def old_view(raw: dict, visit_id: int, conn) -> bool:
    """Whether the dashed box is the old view (the crop box's middle square), in this order: (1) its verdict names
    a view (raw_json.calls.presence.view, newer visits): no; (2) a verdict without a view (made before views had
    names): yes, the verdict shown was made on the old square, even if the frame was re-embedded since; (3) no
    verdict, an embedding: old view yes, current view no; (4) nothing about it at all: no. The verdict comes
    first so the public mirror, which has no embeddings, gives the same answer whenever there is one."""
    calls = raw.get("calls")
    verdict = calls.get("presence") if isinstance(calls, dict) else None
    if isinstance(verdict, dict):
        return not verdict.get("view")
    row = conn.execute("SELECT view FROM embeddings WHERE visit_id = ?", (visit_id,)).fetchone()
    return row is not None and row["view"] != classifier.PRESENCE_VIEW


def visit_zones(visit, conn) -> dict | None:
    """What the visit photo draws over itself: the dashed box the Pi animal check looked at, and the zone whose
    motion fired the capture, each in its Settings color. The dashed box is the current view
    (classifier.presence_view_box: a square around the bowl, the ground crop, or the crop box when there's no ring)
    or the old one, the middle square of the crop box (classifier.old_view_box), decided by old_view(). `view` and
    `claude_sees` say so in words for the tip. `source` says whose
    zones: stored with the photo (since 2026-11-01), rebuilt from the calibration history at the time, or today's.
    Bad stored zones never break the page: a ring that can't be drawn counts as none."""
    try:
        raw = json.loads(visit["raw_json"] or "{}") or {}
    except (ValueError, AttributeError):
        raw = {}
    if not isinstance(raw, dict):
        raw = {}
    z = None
    for source, data in (("photo", lambda: raw.get("zones")),
                         ("history", lambda: db.zones_at(conn, visit["captured_at"]))):
        d = data()
        if isinstance(d, dict):
            if d.get("ring") is not None and classifier.usable_ring(d["ring"]) is None:
                d = {**d, "ring": None}
            try:
                z = zones.Zones.from_dict(d)
                s = zone_shapes(z)
                break
            except (TypeError, ValueError, IndexError):
                z = None
    if z is None:
        source, z = "today", zones.load(conn)
        s = zone_shapes(z)
    colors, trigger = s["colors"], raw.get("trigger")
    shapes = []
    if trigger == "rim":
        if s["rim"]:
            shapes = [{"d": s["rim"], "color": colors["ring"]}]
            shapes += [{"d": d, "color": c} for d, c in zip(s["spots"], s["spot_colors"])]
        elif s["roi"]:
            r = s["roi"]
            shapes = [{"d": f"M{r['x']:.2f},{r['y']:.2f} h{r['w']:.2f} v{r['h']:.2f} h{-r['w']:.2f} Z",
                       "color": colors["ring"]}]
    elif trigger == "water" and s["water"]:
        shapes = [{"d": s["water"], "color": colors.get("water", colors["ring"])}]
    elif trigger == "ground" and s["ground"]:
        shapes = [{"d": s["ground"], "color": colors.get("ground", SCENE_COLORS[0])}]
    ground = trigger == "ground"
    claude_box = "the ground crop" if ground else "the crop box"
    if old_view(raw, visit["id"], conn):  # judged the old way: what it saw then
        crop = box_rect(classifier.old_view_box(z, trigger, ZONE_W / ZONE_H))
        had_box = classifier.usable_box(z.ground_crop if ground else z.crop)
        view = (f"the middle square of {claude_box if had_box else 'the photo'}, "
                "all it looked at before it saw the whole bowl")
    else:
        kind = classifier.view_kind(z, trigger)
        crop = box_rect(classifier.presence_view_box(z, trigger, ZONE_W / ZONE_H))
        view = {"bowl": "a square around the whole bowl", "padded": claude_box}.get(kind)
    # The ground crop isn't a Settings shape: it takes the ground's color.
    crop_color = colors.get("ground" if ground else "crop", SCENE_COLORS[0])
    if not crop and not shapes:
        return None
    claude_sees = (f"Claude sees {claude_box} set on Settings > Zones."
                   if (z.ground_crop if ground else z.crop) else "Claude sees the whole photo.")
    return {"w": ZONE_W, "h": ZONE_H, "shapes": shapes, "crop": crop, "crop_color": crop_color,
            "view": view if crop else None, "claude_sees": claude_sees, "fired": FIRED.get(trigger), "source": source}


def start_marks(visit) -> list[dict]:
    """The circles on the visit photo: the owner's saved ones, else the circles Claude drew around what it saw
    (the main animal, then each other kind in the photo, labelled when there's more than one)."""
    if visit["marks"]:
        return db.parse_marks(visit["marks"])
    try:
        circle = ((json.loads(visit["raw_json"] or "{}") or {}).get("calls") or {}).get("circle")
    except (ValueError, AttributeError):
        circle = None
    also = [a for a in db.also_of(visit) if a.get("circle")]
    circles = [{**circle, "label": visit["label"]}] if circle else []
    circles += [{**a["circle"], "claude": True, "label": db.also_label(a)} for a in also]
    if len(circles) == 1:
        circles[0].pop("label")
    return db.parse_marks(json.dumps(circles))


def no_circle_note(visit) -> str:
    """"Named on the Pi, no circle" for a visit the Pi named itself where the circle from the motion found nothing
    sure (birdbrain/motion_circle.py), so the photo doesn't look like the circle just failed to load."""
    if visit["marks"]:
        return ""
    try:
        calls = (json.loads(visit["raw_json"] or "{}") or {}).get("calls") or {}
    except (ValueError, AttributeError):
        return ""
    if calls.get("settled_by") != "classifier" or calls.get("circle"):
        return ""
    return "Named on the Pi, no circle"


TRIGGERS = {"rim": "on the rim", "water": "in the water", "ground": "on the ground"}

# Circle from the motion (birdbrain/motion_circle.py, not imported here: the web process stays free of OpenCV).
CIRCLE_SOURCES = {"empty": "the empty scene from just before the motion",
                  "stored": "nearby photos of the empty scene",
                  "reference": "the zone calibration's view of the empty bath"}
CIRCLE_WHY = {"nothing": "no clear patch of change near the zone that fired",
              "too big": "the only change was too big for an animal (the light)",
              "two alike": "two patches of change of a similar size, so it couldn't tell which",
              "light": "most of the view changed (the light)",
              "no empty view": "no empty view of the scene to compare with",
              "unreadable photo": "the photo couldn't be read",
              "no photo or zones": "the photo or its zones are gone",
              "error": "something went wrong"}


# Where the species check looked: classifier.run's crops, and for a gate guess made from the Pi animal check's
# embedding, that view's kind (classifier.view_kind; "padded" is the ground crop or the crop box, by trigger).
CLASSIFIER_CROPS = {"circle": "around Claude's circle", "box": "on the crop box", "frame": "on the whole frame",
                    "bowl": "on a square around the whole bowl",
                    classifier.TIGHT_CROP: "on a tight square around the patch that moved"}


miss_text = steps.miss_text  # decision_steps has a local called steps


def bioclip_step(guess: dict, raw: dict) -> dict:
    """One BioCLIP 2 species check look for the visit page: its top guesses and where it looked."""
    top = guess["top"][0] if guess.get("top") else None
    return {
        "model": "BioCLIP 2 species check",
        "bioclip": True,
        "guesses": [f"{g['name']} ({g['score']:.0%})" for g in guess["top"]],
        "said": f"{top['name']} ({top['score']:.0%})" if top else "no local species",
        "seconds": guess.get("seconds"),
        "crop": (("on the ground crop" if raw.get("trigger") == "ground" else "on the crop box")
                 if guess.get("crop") == "padded" else CLASSIFIER_CROPS.get(guess.get("crop"), "")),
        "night": bool(guess.get("night")),
    }


def decision_steps(raw_json: str | None) -> dict:
    """The identification trail saved with a visit: each model's answer, in order, for the visit page."""
    try:
        raw = json.loads(raw_json or "{}") or {}
        calls = raw.get("calls") or {}
    except (ValueError, AttributeError):
        return {"steps": [], "lessons": 0, "trigger": None}
    steps = []
    pres = calls.get("presence")
    if pres and pres.get("score") is not None:
        steps.append({
            "model": "Pi animal check",
            "presence": True,
            "said": "nothing there" if pres.get("empty") else "an animal",
            "score": pres["score"],
            "threshold": pres.get("threshold"),
            "trained_on": pres.get("trained_on"),
            "frames": pres.get("frames"),  # the sharpest and the last frame's scores, when it looked at both
            "frame": pres.get("frame"),
            "skipped_claude": bool(calls.get("skipped_claude") and pres.get("empty")),
        })
    det = calls.get("detector")
    if det:
        best = det.get("best")
        steps.append({
            "model": "Local detector",
            "local": True,
            "said": f"{best['label']} ({best['score']:.0%})" if best else "no animal",
            "seconds": det.get("seconds"),
            "skipped_claude": bool(calls.get("skipped_claude")),
            "scenery": det.get("scenery"),
        })
    if calls.get("error"):
        # Claude couldn't be asked at all (identify.unanswered): the visit was kept for review with no answer.
        calls = {**calls, "haiku": {"frames": 1, "answer": None, "error": calls["error"]}}
    for key, name in (("haiku", "Haiku"), ("sonnet", "Sonnet")):
        call = calls.get(key)
        if not call:
            continue
        answer = call.get("answer") or {}
        steps.append({
            "model": name,
            "frames": call.get("frames", 1),
            "refused": not answer,
            "error": call.get("error") if not answer else None,  # the call failed, rather than declining
            "said": answer.get("species") or ("no animal" if answer.get("visitor") == "none" else answer.get("visitor")),
            "also": [db.also_label(a) + (f" x{a['count']}" if (a.get("count") or 1) > 1 else "")
                     for a in answer.get("also") or [] if answer.get("visitor") != "none" and a.get("visitor")],
            "confidence": answer.get("confidence"),
            "notes": answer.get("notes"),
        })
    guess = calls.get("classifier")
    named = calls.get("settled_by") == "classifier"  # the Pi named the visit: that guess was its last look
    if guess and guess.get("top") is not None:
        # Named by the tight look (rule H): the guess is of a square around the patch that moved, after a first look
        # on the bowl (kept as classifier_gate) fell short.
        tight = named and guess.get("crop") == classifier.TIGHT_CROP
        steps.append({**bioclip_step(guess, raw), "gated": named, "first_look": named and not tight, "tight": tight,
                      "crowd": calls.get("gate_crowd"), "scorecard": not named})
    # Where a look made before Claude goes: ahead of Claude's answers, or of the look that named the visit.
    before_claude = lambda: next((i for i, st in enumerate(steps) if st["model"] in ("Haiku", "Sonnet") or st.get("gated")),
                                 len(steps))
    first = calls.get("classifier_gate")
    if first and first.get("top") is not None:
        # The first look, before Claude: made from the Pi animal check's embedding; it did not name the visit.
        steps.insert(before_claude(), {**bioclip_step(first, raw), "gated": False, "first_look": True, "scorecard": False,
                                       "why": first.get("why"), "reason": miss_text(first.get("why"), first),
                                       "crowd": calls.get("gate_crowd"), "backfilled": bool(first.get("backfilled")),
                                       "view_note": first.get("view_note")})
        for st in steps:
            if st.get("scorecard"):
                st["second_look"] = True
    second = calls.get("classifier_tight")
    if second and second.get("top") is not None:
        # The tight look (rule H), after the first look and before Claude: it did not name the visit either.
        steps.insert(before_claude(), {**bioclip_step(second, raw), "gated": False, "first_look": False, "tight": True,
                                       "scorecard": False, "why": second.get("why"),
                                       "reason": miss_text(second.get("why"), second, view="tight")})
    found = calls.get("motion_circle")
    if isinstance(found, dict):
        steps.append({
            "model": "Circle from the motion",
            "motion_circle": True,
            "found": bool(found.get("found") and calls.get("circle")),
            "said": "circled" if found.get("found") and calls.get("circle") else "no circle",
            "source": CIRCLE_SOURCES.get(found.get("source"), ""),
            "why": CIRCLE_WHY.get(found.get("why"), ""),
            "seconds": found.get("seconds"),
        })
    return {"steps": steps, "lessons": calls.get("lessons", 0), "trigger": TRIGGERS.get(raw.get("trigger")),
            "owner_checks": owner_checks(calls), "failed": any(s.get("error") for s in steps),
            "has_first_look": any(st.get("first_look") and not st.get("gated") for st in steps),
            "has_tight_look": any(st.get("tight") for st in steps),
            "skipped_sonnet": bool(calls.get("skipped_sonnet")), "asked_sonnet": bool(calls.get("sonnet")),
            "local_disagreed": calls.get("local_disagreed"), "local_agreed": bool(calls.get("local_agreed"))}


CHECK_BY = {"pi": "BioCLIP 2", "haiku": "Haiku", "sonnet": "Sonnet"}  # the visit page's Ask buttons


def owner_checks(calls: dict) -> list[dict]:
    """The owner's circle checks (circle_check), oldest first, for the visit page's decision trail."""
    out = []
    for c in calls.get("owner_checks") or []:
        if not isinstance(c, dict) or c.get("source") not in CHECK_BY:
            continue
        if c["source"] == "pi":
            detail = ", ".join(f"{g['name']} ({g['score']:.0%})" for g in c.get("top") or []) or "no local species"
        else:
            detail = (c.get("answer") or {}).get("notes") or ""
        out.append({"by": CHECK_BY[c["source"]], "pi": c["source"] == "pi", "say": c.get("say") or "",
                    "detail": detail, "at": c.get("at") or "", "usd": c.get("usd"), "seconds": c.get("seconds"),
                    "frame": c.get("frame")})
    return out


REFRESH = {"HX-Refresh": "true"}


@app.post("/visit/{visit_id}/correct")
def correct(
    visit_id: int,
    visitor: str = Form(""),
    species: str = Form(""),
    feedback: str = Form(""),
    marks: str = Form(""),
    also_species: list[str] = Form([]),
    also_count: list[str] = Form([]),
    conn=Depends(get_conn),
):
    v = db.get_visit(conn, visit_id)
    if not v:
        raise HTTPException(404)
    try:
        new_visitor, new_species = db.plan_correction(v, visitor, species)
    except ValueError as e:
        raise HTTPException(400, str(e)) from None
    also = db.plan_also(v, also_species, also_count, visitor.strip() or None)
    db.correct_visit(conn, visit_id, new_visitor, new_species, feedback,
                     db.parse_marks(marks, db.claude_frame(v["raw_json"])), also)
    lessons.maybe_auto(conn)  # every AUTO_EVERY new notes, Claude distills them in the background
    return Response(headers=REFRESH)


@app.get("/species-search")
def species_search(q: str = "", known: list[str] = Query([])):
    """eBird species names matching q, for the correction form's species boxes (the page lists the yard's own)."""
    if auth.public():
        raise HTTPException(404)  # only the correction form uses it; the mirror shouldn't download eBird's list
    return JSONResponse(names.search(q, known))


@app.post("/visit/{visit_id}/ask-species")
async def ask_species(visit_id: int, hint: str = Form(""), conn=Depends(get_conn)):
    """Sonnet names the animal the owner's clue points to, in the official spelling. Nothing is saved."""
    v = db.get_visit(conn, visit_id)
    if not v:
        raise HTTPException(404)
    hint = hint.strip()[:200]
    if not hint:
        return JSONResponse({"error": "Type a clue first, e.g. red head, finch."})
    if v["photo_deleted_at"]:
        return JSONResponse({"error": "This photo was deleted to save space, so Claude has nothing to look at."})
    jpeg = (config.IMAGES_DIR / db.photo_rel(v)).read_bytes()  # the thumbnail once only that is left
    try:
        result, usage = await asyncio.to_thread(
            names.ask, identify.prepare(jpeg), hint, datetime.fromisoformat(v["captured_at"]),
            db.prompt_site(conn).get("location", ""))
    except Exception as e:
        return JSONResponse({"error": f"Claude couldn't be reached ({type(e).__name__})."})
    if result is None:
        return JSONResponse({"error": "Claude didn't give a name this time. Try again."})
    if result.visitor == "none":
        return JSONResponse({"error": f"Claude didn't find that in the photo: {result.reason.strip()}"})
    visitor = result.visitor if result.visitor in db.VISITORS else "other"
    name = result.name.strip()
    if visitor == "bird":  # only an eBird name: keeps one spelling per species
        official = await asyncio.to_thread(names.canonical, name, [n for n, _ in db.species_kinds(conn)])
        if not official and await asyncio.to_thread(names.bird_names):
            return JSONResponse({"error": f"Claude said \"{name}\", which isn't an eBird species name. Try another clue."})
        name = official or name
    return JSONResponse({"name": name, "visitor": visitor, "confidence": result.confidence,
                         "reason": result.reason.strip(), "usd": round(names.cost(usage), 4)})


@app.post("/visit/{visit_id}/check-circle/{source}")
async def check_circle(visit_id: int, source: str, marks: str = Form(""), conn=Depends(get_conn)):
    """What's in the owner's circle: the Pi's local model (free), else Haiku or Sonnet on just that square.
    Kept with the visit (raw_json.calls.owner_checks); the label only changes when the owner saves."""
    if source not in circle_check.SOURCES:
        raise HTTPException(404)
    v = db.get_visit(conn, visit_id)
    if not v:
        raise HTTPException(404)
    picked = circle_check.owner_circle(marks, db.claude_frame(v["raw_json"]))
    if not picked:
        return JSONResponse({"error": "Circle the animal first: drag on the photo."})
    number, circle = picked
    # The frame the circle was drawn on (a burst frame), else the photo when that frame is gone.
    jpeg, frame = circle_check.jpeg_of(v, db.mark_frame(circle))
    circle = {k: circle[k] for k in ("x", "y", "r")} | ({"frame": frame} if frame else {})
    if jpeg is None:
        return JSONResponse({"error": "This photo was deleted to save space, so there's nothing to look at."})
    loop = asyncio.get_running_loop()
    started = time.monotonic()
    try:
        if source == "pi":
            # Its own low-priority thread, one check at a time: ~6 to 22 s of the Pi's cores.
            entry = await loop.run_in_executor(circle_check.POOL, circle_check.ask_pi, conn, jpeg, circle,
                                               bool(v["is_night"]))
            loop.call_later(circle_check.IDLE + 1, circle_check.release_if_idle)
        else:
            entry = await asyncio.to_thread(circle_check.ask_claude, conn, source, jpeg, circle,
                                            datetime.fromisoformat(v["captured_at"]), circle_check.frame_size(jpeg),
                                            bool(v["is_night"]))
    except Exception as e:
        if source == "pi":
            db.record_pi_check(conn, time.monotonic() - started, ok=False, visit_id=visit_id)
            return JSONResponse({"error": f"The Pi's model isn't available right now ({e})."})
        return JSONResponse({"error": f"Claude couldn't be reached ({type(e).__name__})."})
    extra = {}
    if source == "pi":
        # What the owner waited: queue, model load and compute. Every click is a row; averages use the good ones.
        entry["wall_seconds"] = round(time.monotonic() - started, 1)
        extra["pi_average"] = db.record_pi_check(conn, entry["wall_seconds"], True, visit_id, entry.get("seconds"))
    circle_check.record(conn, visit_id, entry, circle)
    return JSONResponse({**entry, "circle_n": number, "frame": frame, **extra})


@app.get("/species-photo")
async def species_photo(request: Request, name: str = Query(""), conn=Depends(get_conn)):
    """A reference photo for the Ask-the-Pi guesses. Only admins can make the site look one up, and only for an
    animal the visits know; visitors (and any other name) get what is already cached."""
    name = name.strip()
    fetch = bool(request.state.admin) and name in db.known_species(conn)
    photo = await asyncio.to_thread(species_info.reference_photo, conn, name, fetch)
    return JSONResponse({"photo": photo})


@app.post("/visit/{visit_id}/clear-correction")
def clear_correction(visit_id: int, conn=Depends(get_conn)):
    db.correct_visit(conn, visit_id, None, None, None)
    return Response(headers=REFRESH)


@app.post("/visit/{visit_id}/reviewed")
def reviewed(visit_id: int, conn=Depends(get_conn)):
    db.mark_reviewed(conn, visit_id)
    return Response(headers=REFRESH)


@app.post("/visit/{visit_id}/unconfirm")
def unconfirm(visit_id: int, conn=Depends(get_conn)):
    db.confirm_visits(conn, [visit_id], on=False)
    return Response(headers=REFRESH)


@app.post("/visit/{visit_id}/keep")
def keep(request: Request, visit_id: int, on: int = Form(...), conn=Depends(get_conn)):
    """Favorite on or off for a visit with no species (static/star.js calls it; the star repaints itself)."""
    visit = db.get_visit(conn, visit_id)
    if visit is None:
        raise HTTPException(404)
    db.set_keep(conn, visit_id, bool(on))
    return Response(status_code=204)


@app.post("/visit/{visit_id}/public")
def public_anyway(visit_id: int, conn=Depends(get_conn)):
    """The owner saw no person in a photo the Pi's person check flagged: pin today's scores so the next push
    to the public mirror includes the visit (birdbrain.privacy)."""
    visit = db.get_visit(conn, visit_id)
    if visit is None:
        raise HTTPException(404)
    override = privacy.make_override(visit["image_path"], privacy.load_cache(), visit["thumb_path"])
    if override is None:
        raise HTTPException(409, "Nothing to clear: this visit isn't held back for a person, or isn't fully checked yet.")
    db.set_public_ok(conn, visit_id, override)
    return Response(headers=REFRESH)


@app.post("/visit/{visit_id}/private")
def public_undo(visit_id: int, conn=Depends(get_conn)):
    """Undo Make public: the visit is held back again from the next push (and taken off the mirror)."""
    db.set_public_ok(conn, visit_id, None)
    return Response(headers=REFRESH)


@app.post("/false-triggers/confirm")
def confirm_false_triggers(ids: str = Form(""), conn=Depends(get_conn)):
    """Confirm all on the page: the false triggers the owner looked at had no animal."""
    wanted = [int(i) for i in ids.split(",") if i.strip().isdigit()]
    keep = {r[0] for r in conn.execute(
        f"SELECT id FROM visits WHERE id IN ({','.join('?' * len(wanted))}) AND NOT {db.REAL}", wanted)} if wanted else set()
    db.confirm_visits(conn, sorted(keep))
    return RedirectResponse(f"/false-triggers?confirmed={len(keep)}", status_code=303)


@app.post("/visit/{visit_id}/delete")
def delete(visit_id: int, conn=Depends(get_conn)):
    row = db.delete_visit(conn, visit_id)
    if row:
        images.delete_frame(row["image_path"], row["thumb_path"])
    return Response(headers={"HX-Redirect": "/"})


@app.get("/species", response_class=HTMLResponse)
def species_list(request: Request, sort: str = "visits", conn=Depends(get_conn)):
    return page(request, "species_list.html", species=db.species_summary(conn, sort=sort), sort=sort,
                # tiles keep the list's sort on their links so the species page's swipe follows the same order
                link_sort=sort if sort in db.SPECIES_SORTS and sort != "visits" else "",
                confirmed=db.confirmed_species(conn))


@app.get("/api/photo-reminder")
def photo_reminder(request: Request, conn=Depends(get_conn)):
    """Species with 2+ photos and no favourite yet, for the weekly reminder pop-up. Admins only (visitors get none)."""
    names = db.species_without_favorite(conn) if request.state.admin else []
    return {"species": names}


@app.get("/species/{name}", response_class=HTMLResponse)
def species_page(request: Request, name: str, sort: str = "visits", conn=Depends(get_conn)):
    row = db.species_row(conn, name)
    if not row:
        raise HTTPException(404)
    # Phone swipe steps along the species list in the sort the owner came from (templates: <link rel=prev/next>).
    sort = sort if sort in db.SPECIES_SORTS else "visits"
    order = db.species_order(conn, sort)
    at = order.index(name) if name in order else -1
    suffix = f"?sort={sort}" if sort != "visits" else ""

    def neighbour(i: int) -> str | None:
        return f"/species/{quote(order[i])}{suffix}" if at >= 0 and 0 <= i < len(order) else None
    filters = species_filters(request)
    active = {k: v for k, v in filters.items() if v}
    visits = db.list_visits(conn, species=name, **filters, limit=SPECIES_PAGE)
    return page(
        request,
        "species.html",
        sp=row,
        visits=visits,
        swipe_prev=neighbour(at - 1),
        swipe_next=neighbour(at + 1),
        # desktop arrows in the top bar: position and the neighbours' names for their tooltips
        nav_at=at, nav_total=len(order),
        nav_prev_name=order[at - 1] if at > 0 else "", nav_next_name=order[at + 1] if 0 <= at < len(order) - 1 else "",
        next_url=more_url(f"/species/{quote(name)}/more", visits, SPECIES_PAGE, filters),
        filters=filters,
        filtered=bool(active),
        matching=db.count_visits(conn, species=name, **filters) if active else row["visits"],
        more_filters_open=bool(filters["tod"] == "day" or (filters["date_to"] and filters["date_from"] != filters["date_to"])),
        chips=species_chips(name, filters, sort),
        sort_kept=sort if sort != "visits" else "",
        daily=db.daily_counts(conn, days=60, species=name),
        hourly=db.hourly_counts(conn, species=name),
        # A cached info card renders with the page; an uncached one still loads afterward over htmx.
        info=species_info.get(conn, name, fetch=False),
        ours=db.month_counts(conn, name),
        months=species_info.MONTHS,
        this_month=clock.today().month,
        confirmed=db.confirmed_species(conn),
        fav_label=name,
        scope_species=name,
    )


@app.get("/species/{name}/more", response_class=HTMLResponse)
def species_more(request: Request, name: str, before: str = "", conn=Depends(get_conn)):
    """The next page of a species' grid; its cards carry the same star and ?species= scope as the page's."""
    filters = species_filters(request)
    visits = visits_before(conn, before, limit=SPECIES_PAGE, species=name, **filters)
    return page(request, "_cards.html", visits=visits, fav_label=name, scope_species=name,
                next_url=more_url(f"/species/{quote(name)}/more", visits, SPECIES_PAGE, filters))


STAR_STATES = db.STAR_STATES


@app.post("/species/{name}/photo")
def set_species_cover(name: str, visit_id: int = Form(...), state: str = Form(""), conn=Depends(get_conn)):
    """The star on a photo: unset -> favorite (blue, kept) -> species cover (yellow, one per species, also kept)
    -> unset (no longer kept). `state` sets one outright (static/star.js always sends it: it holds yellow for a
    moment and may ask before replacing a cover); without it the click moves one step along. A new cover leaves
    the old one as a kept favorite (db.set_star, in one transaction). Answers with the visit's new state and the
    species' cover now."""
    if not db.shows_species(conn, visit_id, name):
        raise HTTPException(404)
    if state and state not in STAR_STATES:
        raise HTTPException(400)
    state, cover = db.set_star(conn, name, visit_id, state or None)
    return {"visit_id": visit_id, "species": name, "state": state, "cover_id": cover}


def _compare_photo(v, name: str) -> dict:
    """One side of the star's "Replace the cover?" question: the photo, when, and the circle around the species
    (fractions; null when there's none, then the whole frame shows)."""
    marks = [m for m in start_marks(v) if not db.mark_frame(m)]
    mark = next((m for m in marks if m.get("label") == name), None) or next((m for m in marks if not m.get("label")), None) \
        or (marks[0] if marks else None)
    return {"id": v["id"], "src": site_url("/images/" + db.photo_rel(v)), "when": nice_time(v["captured_at"]),
            "mark": {k: mark[k] for k in ("x", "y", "r")} if mark else None}


@app.get("/species/{name}/cover-compare")
def species_cover_compare(name: str, visit_id: int, conn=Depends(get_conn)):
    """The species' current cover and a candidate, for the side-by-side question (cover null when there is none,
    or it is this visit, or its photo is gone: then the star just sets it)."""
    v = db.get_visit(conn, visit_id)
    if v is None or not db.shows_species(conn, visit_id, name):
        raise HTTPException(404)
    cid = db.species_photos(conn).get(name)
    old = db.get_visit(conn, cid) if cid and cid != visit_id else None
    if old is not None and (old["photo_deleted_at"] or not db.shows_species(conn, cid, name)):
        old = None
    return {"species": name, "cover": _compare_photo(old, name) if old else None, "candidate": _compare_photo(v, name)}


@app.get("/species-info/{name}", response_class=HTMLResponse)
def species_card(request: Request, name: str, mini: bool = False, fold: bool = False, conn=Depends(get_conn)):
    """The species info card (or its hover preview), built from outside sources on first use."""
    # Visitors only see cards already built, so they can't make the site fetch or call Claude; and a card is
    # only built for an animal the visits know, so a made-up name in the URL can't start fetches or Haiku calls.
    # The hover preview (mini) never fetches.
    info = species_info.get(conn, name, fetch=bool(request.state.admin) and not mini and name in db.known_species(conn))
    if not info:
        return HTMLResponse('<p class="muted sp-empty">No info found for this one yet.</p>')
    return page(
        request,
        "_species_tip.html" if mini else "_species_fold.html" if fold else "_species_card.html",
        info=info,
        sp=db.species_row(conn, name),
        ours=db.month_counts(conn, name),
        months=species_info.MONTHS,
        this_month=clock.today().month,
        confirmed=db.confirmed_species(conn),
        fold=fold,  # the species page folds the long part; the pop-up stays open
    )


@app.post("/species-info/{name}/sound-broken")
async def species_sound_broken(request: Request, name: str, conn=Depends(get_conn)):
    """A card's recording wouldn't load in someone's browser. Flags the card so its sounds are fetched again
    next time an admin opens it or a visit of the species arrives; never fetches or calls Claude here."""
    try:
        url = (await request.json()).get("url")
    except Exception:
        url = None
    if not isinstance(url, str) or name not in db.known_species(conn) or not species_info.flag_broken_sound(conn, name, url):
        return Response(status_code=400)
    return Response(status_code=204)


@app.get("/stats", response_class=HTMLResponse)
def stats(request: Request, conn=Depends(get_conn)):
    return page(
        request,
        "stats.html",
        daily=db.daily_counts(conn, days=30),
        hourly=db.hourly_counts(conn),
        id_split=db.id_split_by_day(conn),
        leaders=db.species_summary(conn, sort="visits", limit=10),
        newest=db.species_summary(conn, sort="newest", limit=5),
        total=db.count_visits(conn),
        first_visit=db.first_visit_at(conn),
        today_label=clock.today().strftime("%a, %b %-d"),
        activity=db.activity(conn),
        classifier=db.classifier_report(conn),
        presence_mode=presence.mode(conn),
        classifier_on=config.CLASSIFIER != "off",
        classifier_mode=config.CLASSIFIER,
    )


@app.get("/weather", response_class=HTMLResponse)
def weather_page(request: Request, conn=Depends(get_conn)):
    """Weather & visits: each day's weather beside its visitors. Reads the database only; the watcher fetches the
    weather (birdbrain.weather), so the public mirror shows whatever the Pi last sent."""
    report = db.weather_report(conn)
    return page(
        request,
        "weather.html",
        w=report,
        # Today at the bath: rules over stored data, cached species cards only (never the network).
        today=outlook.outlook(conn, report, species_info.cached_cards(), report["now"], temp=temp),
        findings=db.weather_findings(report) if report["has_weather"] else [],
        located=weather.located(),
        rain_hour_mm=db.RAIN_HOUR_MM,
        wet_day_mm=db.WET_DAY_MM,
        overcast_pct=db.OVERCAST_PCT,
        min_hours=db.MIN_HOURS,
        windy_kmh=db.WINDY_KMH,
        describe=weather.describe,
    )


# The About page draws zones in a 16:9 box, the shape of both camera streams.
ZONE_W, ZONE_H = 160, 90
# The About figure's fixed daytime photo was taken from a slightly different spot than the zones were set from, so
# the zones are drawn nudged by this much (fractions of the frame: right, down). Picked by eye with the owner.
ZONE_SHIFT = (0.01, 0.05)


def ellipse_path(cx: float, cy: float, rx: float, ry: float, angle: float = 0.0) -> str:
    """SVG path for an ellipse given as frame fractions (angle: degrees clockwise), so several can be combined into
    one even-odd shape."""
    x, y, a, b = cx * ZONE_W, cy * ZONE_H, rx * ZONE_W, ry * ZONE_H
    ux, uy = a * math.cos(math.radians(angle)), a * math.sin(math.radians(angle))
    return (f"M{x - ux:.2f},{y - uy:.2f} a{a:.2f},{b:.2f} {angle:g} 1,0 {2 * ux:.2f},{2 * uy:.2f} "
            f"a{a:.2f},{b:.2f} {angle:g} 1,0 {-2 * ux:.2f},{-2 * uy:.2f} Z")


def box_rect(box) -> dict | None:
    if not box:
        return None
    x, y, w, h = box
    return {"x": x * ZONE_W, "y": y * ZONE_H, "w": w * ZONE_W, "h": h * ZONE_H}


def zone_shapes(z: zones.Zones | None = None) -> dict:
    """The zones in use as SVG shapes, mirroring watch.motion_mask, water_mask and ground_mask."""
    z = z or zones.from_env()
    shapes = {"w": ZONE_W, "h": ZONE_H, "crop": box_rect(z.crop), "roi": None, "rim": None,
              "water": None, "spots": [], "ground": None}
    outer = None
    if z.ring:
        rim, pool = config.ring_ellipses(z.ring)
        inner = ellipse_path(*pool)
        outer = ellipse_path(*rim)
        shapes["spots"] = [ellipse_path(*s) for s in z.spots]  # spots only apply with a ring
        shapes["rim"] = outer + " " + inner
        shapes["water"] = " ".join([inner, *shapes["spots"]])
    else:
        shapes["roi"] = box_rect(z.roi)
    if z.ground:  # the polygon minus the bath's outer ellipse, as an even-odd hole
        points = " L".join(f"{x * ZONE_W:.2f},{y * ZONE_H:.2f}" for x, y in z.ground)
        shapes["ground"] = " ".join(filter(None, [f"M{points} Z", outer]))
    shapes.update(zone_colors(z))
    shapes.update(zone_labels(z, shapes["colors"], shapes["spot_colors"]))
    return shapes


def zone_labels(z: zones.Zones, colors: dict, spot_colors: list) -> dict:
    """The number of each zone, as on Settings > Zones (its place in zones.objects), and where to write it on the
    About figure: `labels` (n, x, y in figure units, color) and `nums` (role -> number, rocks -> a list)."""
    labels, nums, n = [], {}, 0

    def add(role, x, y, color):
        nonlocal n
        n += 1
        labels.append({"n": n, "x": x * ZONE_W, "y": y * ZONE_H, "color": color})
        return n

    if z.ring:
        rim, pool = config.ring_ellipses(z.ring)
        nums["ring"] = add("ring", rim[0] - rim[2], rim[1], colors["ring"])  # the rim's left end
        nums["water"] = add("water", pool[0], pool[1] - pool[3], colors.get("water", colors["ring"]))  # water's top
        nums["spots"] = [add("spot", s[0], s[1], spot_colors[i]) for i, s in enumerate(z.spots)]
    if z.ground:
        nums["ground"] = add("ground", *z.ground[0], colors["ground"])
    if z.crop:
        nums["crop"] = add("crop", z.crop[0], z.crop[1], colors["crop"])
    return {"labels": labels, "nums": nums}


def zone_colors(z: zones.Zones) -> dict:
    """Each zone's color, the same as on Settings > Zones (by its place in zones.objects): `colors` by role
    (ring, water, ground, crop) and `spot_colors`, one per rock spot."""
    colors: dict = {}
    spots = []
    for i, o in enumerate(zones.objects(z)):
        color = SCENE_COLORS[i % len(SCENE_COLORS)]
        if o["role"] == "spot":
            spots.append(color)
        else:
            colors[o["role"]] = color
    colors.setdefault("ring", SCENE_COLORS[0])  # no ring yet: the watched rectangle or whole frame
    return {"colors": colors, "spot_colors": spots}


@app.get("/about", response_class=HTMLResponse)
def about(request: Request, conn=Depends(get_conn)):
    """The 10,000-foot view: live values only where the page states a rule or a setting."""
    return page(
        request,
        "about.html",
        cfg=config,
        zones={**zone_shapes(zones.load(conn)), "dx": ZONE_SHIFT[0] * ZONE_W, "dy": ZONE_SHIFT[1] * ZONE_H},
        recal=zones.schedule(conn),
        recal_stored=zones.adjusted(conn),
        classifier_mode=config.CLASSIFIER,
        presence_mode=presence.mode(conn),
        haiku=identify.HAIKU,
        sonnet=identify.SONNET,
        four=steps.components(),
        weight=f"{presence.CONFIRMED_WEIGHT:g}",
        lesson_limit=db.LESSON_LIMIT,
        auto_every=lessons.AUTO_EVERY,
        confirmed_min=db.CONFIRMED_MIN,
        site=db.prompt_site(conn),
        avg_call=db.avg_haiku_call_usd(conn),
        alerts=alert.get_settings(conn),
        card_days=species_info.REFRESH_AFTER.days,
        **storage_context(conn),
    )


@app.get("/review", response_class=HTMLResponse)
def review(request: Request, conn=Depends(get_conn)):
    return page(request, "review.html", visits=db.list_visits(conn, review=True, limit=REVIEW_SHOWN),
                total=db.count_visits(conn, review=True), shown=REVIEW_SHOWN)


@app.get("/notes", response_class=HTMLResponse)
def notes(request: Request, saved: bool = False, accepted: bool = False, conn=Depends(get_conn)):
    """Claude's Notes: everything Claude is told about this yard, in one place."""
    site = db.prompt_site(conn)
    rows = db.all_lessons(conn)
    return page(
        request,
        "notes.html",
        site=site,
        saved=saved,
        accepted=accepted,
        lessons=rows,
        in_use=sum(l["in_use"] for l in rows),
        limit=db.LESSON_LIMIT,
        visitors=db.VISITORS,
        summary=lessons.status(conn, rows),
        in_summary=lessons.in_summary(conn, rows),
        prompt_preview=identify_prompt_preview(site, db.recent_lessons(conn)),
        **zone_drawing(zones.load(conn)),
    )


@app.post("/notes/summary/distill", response_class=HTMLResponse)
async def distill_lessons(request: Request, conn=Depends(get_conn)):
    """Distill now: Sonnet folds every lesson into a draft summary; the section re-renders with it (or the reason
    nothing was saved). Nothing reaches the prompt until the owner accepts."""
    result = await asyncio.to_thread(lessons.distill, conn)
    return page(request, "_lessons_summary.html", summary=lessons.status(conn), limit=db.LESSON_LIMIT,
                summary_error=None if result["ok"] else result["error"], distilled=result["ok"])


@app.post("/notes/summary/accept")
def accept_lessons_summary(text: str = Form(""), conn=Depends(get_conn)):
    """The draft, as edited in the textarea, becomes the summary Claude reads from the next photo on."""
    if not lessons.accept(conn, text):
        raise HTTPException(404, "There is no draft to accept.")
    return RedirectResponse("/notes?accepted=1#summary", status_code=303)


@app.post("/notes/summary/discard")
def discard_lessons_summary(conn=Depends(get_conn)):
    lessons.discard(conn)
    return RedirectResponse("/notes#summary", status_code=303)


@app.post("/notes")
def save_notes(
    location: str = Form(""),
    scene_notes: str = Form(""),
    target_notes: str = Form(""),
    ignore_notes: str = Form(""),
    conn=Depends(get_conn),
):
    db.save_settings(conn, location=location, scene_notes=scene_notes, target_notes=target_notes,
                     ignore_notes=ignore_notes)
    return RedirectResponse("/notes?saved=1#camera", status_code=303)


@app.post("/notes/refine", response_class=HTMLResponse)
async def refine_notes(
    request: Request,
    location: str = Form(""),
    scene_notes: str = Form(""),
    target_notes: str = Form(""),
    ignore_notes: str = Form(""),
):
    """Haiku's tidier version of the boxes, put back in the form unsaved for the owner to check."""
    form = {"location": location, "scene_notes": scene_notes, "target_notes": target_notes, "ignore_notes": ignore_notes}
    if not (scene_notes or target_notes or ignore_notes).strip():
        return page(request, "_yard_form.html", form=form, refine_error="Write something first.")
    try:
        result, usage = await asyncio.to_thread(refine.ask, location, scene_notes, target_notes, ignore_notes)
    except Exception as e:  # the API being down shouldn't break the page
        return page(request, "_yard_form.html", form=form, refine_error=f"Claude couldn't be reached ({type(e).__name__}).")
    if result is None:
        return page(request, "_yard_form.html", form=form, refine_error="Claude didn't give a suggestion this time. Try again.")
    words = lambda *t: sum(len(x.split()) for x in t)  # noqa: E731
    new = {**form, "scene_notes": result.scene_notes.strip(), "target_notes": result.target_notes.strip(),
           "ignore_notes": result.ignore_notes.strip()}
    refined = {"changes": result.changes.strip(), "usd": refine.cost(usage),
               "before": words(scene_notes, target_notes, ignore_notes),
               "after": words(new["scene_notes"], new["target_notes"], new["ignore_notes"])}
    if refined["after"] >= refined["before"]:  # a longer "refinement" costs tokens on every photo: keep the owner's
        refined["kept"] = True
        new = form
    return page(request, "_yard_form.html", form=new, refined=refined)


@app.get("/lessons")
def lessons_redirect():
    return RedirectResponse("/notes#lessons", status_code=301)


@app.post("/lessons/{visit_id}/edit")
def lesson_edit(
    visit_id: int, visitor: str = Form(""), species: str = Form(""), feedback: str = Form(""), conn=Depends(get_conn)
):
    """Change what it was and the note; the date, Claude's answer, the circles and the other animals are kept."""
    v = db.get_visit(conn, visit_id)
    if not v:
        raise HTTPException(404)
    try:
        new_visitor, new_species = db.plan_correction(v, visitor, species)
    except ValueError as e:
        raise HTTPException(400, str(e)) from None
    also = json.loads(v["corrected_also"]) if v["corrected_also"] is not None else None
    db.correct_visit(conn, visit_id, new_visitor, new_species, feedback, db.parse_marks(v["marks"]), also)
    lessons.maybe_auto(conn)
    return Response(headers=REFRESH)


@app.post("/lessons/{visit_id}/{state}")
def lesson_switch(visit_id: int, state: str, conn=Depends(get_conn)):
    if state not in ("on", "off"):
        raise HTTPException(404)
    db.set_lesson_off(conn, visit_id, state == "off")
    return Response(headers=REFRESH)


_TIPS = {
    "species_photos": "Species covers (yellow star): the photo shown for each species. Never thinned or deleted.",
    "keepers": "Favorites (blue star): photos you chose to keep. Never thinned or deleted.",
    "training": "Corrected or annotated visits: Claude learns from them, so they are never thinned or deleted.",
    "recent": "Recent real visits: kept full size for {full_days} days, then thinned, except each species' best {full_per_day} per day.",
    "older": "Older real visits: thinned once to save space. If the budget runs short, the oldest keep only their thumbnail.",
    "false_triggers": "False-trigger photos and their bursts: the oldest go first when space runs short, and they have their own cap too.",
    "thumbs": "Thumbnails: every visit keeps one, always.",
}


def _short(n: int) -> str:
    """Bytes for a scale label: 10 GB, 147 MB, 0."""
    return "0" if not n else mb(n).replace(".0 ", " ")


def storage_bar(bd: dict) -> dict:
    """The photo budget as bar segments (widths in % of the bar), a table of kinds (biggest first), scale labels and ticks."""
    o = bd["options"]
    budget, used = bd["budget_bytes"], bd["used_bytes"]
    over = bool(budget) and used > budget
    scale = max(used, budget) or 1
    tips = {k: v.format(**o) for k, v in _TIPS.items()}
    rule = {"species_photos": "Protected", "keepers": "Protected", "training": "Protected",
            "recent": f"Thinned after {o['full_days']} days", "older": "Cut to thumbnail, oldest first",
            "false_triggers": f"Oldest first, own {_short(bd['false_trigger_limit_bytes'])} cap", "thumbs": "Never removed"}
    rows = [{**c, "tip": f"{tips[c['key']]} {c['count']} photo{'s' if c['count'] != 1 else ''}, {mb(c['bytes'])}."}
            for c in bd["categories"]]
    rows.append({"key": "thumbs", "label": "Thumbnails", **bd["thumbs"],
                 "tip": f"{tips['thumbs']} {bd['thumbs']['count']} thumbnails, {mb(bd['thumbs']['bytes'])}."})
    for r in rows:
        r["rule"] = rule.get(r["key"], "")
        r["tip"] += f" When space runs short: {r['rule']}."
    table = sorted(rows, key=lambda r: -r["bytes"])
    legend = [r for r in rows if r["bytes"]]
    segments = [{**r, "pct": round(100 * r["bytes"] / scale, 3)} for r in legend]
    free = max(budget - used, 0) if budget else 0
    warn_at = round(100 * budget * bd["warn_pct"] / 100 / scale, 2) if budget and bd["warn_pct"] and not over else None
    use_at = round(100 * used / scale, 2)
    pct = lambda v: "{:g}".format(bd["warn_pct"])
    return {
        "segments": segments,
        "legend": legend,
        "table": table,
        "over": over,
        "free_bytes": free,
        "free_pct": round(100 * free / scale, 3) if free else 0,
        "free_tip": f"Room left in the budget: {mb(free)}.",
        "warn_at": warn_at,
        "warn_low": warn_at is not None and abs(use_at - warn_at) < 18,
        "warn_side": "r" if use_at < (warn_at or 0) else "l",
        "warn_tip": (f"Phone alert when photos reach {pct(0)}% of the budget ({_short(int(budget * bd['warn_pct'] / 100))})"
                     if warn_at is not None else ""),
        "budget_at": round(100 * budget / scale, 2) if over else None,
        "use_at": use_at,
        "use_label": _short(used) if used else "",
        "limit_label": _short(budget) if budget else "",
        "total_count": sum(r["count"] for r in rows),
    }


def _span(days: float) -> str:
    """A length of time to round words: days under 60, months under 24, else years (halves)."""
    if days < 60:
        n = max(1, round(days))
        return f"{n} day{'s' if n != 1 else ''}"
    if days < 24 * 30.44:
        n = round(days / 30.44)
        return f"{n} months"
    if days >= 30 * 365.25:
        return "more than 30 years"
    years = round(days / 365.25 * 2) / 2
    return f"{years:g} years"


def _about(days: float) -> str:
    span = _span(days)
    return span if span.startswith("more") else f"about {span}"


def _when(days: float, now: datetime) -> str:
    d = now + timedelta(days=days)
    return d.strftime("%B %-d") if days < 60 else d.strftime("%B %Y")


def forecast_blurb(fc: dict, bd: dict, now: datetime | None = None) -> dict:
    """The storage forecast as one sentence group (`text`) and the method for its tip (`tip`)."""
    now = now or datetime.now()
    if not fc["ok"]:
        return {"text": "Not enough history yet for a forecast.",
                "tip": f"A forecast needs at least {retention.FORECAST_MIN_DAYS} days of visits; there are "
                       f"{fc['history_days']:g}."}
    o = bd["options"]
    w = fc["window_days"]
    rate = f"{fc['bytes_per_day'] / 1e6:.1f} MB" if fc["bytes_per_day"] < 1e7 else mb(fc["bytes_per_day"])
    n = fc["visits_per_day"]
    pace = f"At the last {w} days' pace (about {rate} a day, {n:g} visit{'s' if n != 1 else ''} a day)"
    if fc["no_growth"]:
        text = f"Photos haven't grown over the last {w} days, so there is nothing to forecast."
    elif fc["at_budget"]:
        text = f"{pace}, photos have reached the budget; thinning and trimming keep them at or under it."
    elif not fc["budget"]:
        text = f"{pace}, no budget is set."
    else:
        text = f"{pace}, photos reach the {_short(bd['budget_bytes'])} budget in {_about(fc['budget_days'])} " \
               f"(around {_when(fc['budget_days'], now)})"
        if fc["thin"]:
            td = fc["thin_days"]
            if td is None:
                text += f"; with thinning after {o['full_days']} days, not within 30 years"
            elif _when(td, now) != _when(fc["budget_days"], now):
                text += f"; with thinning after {o['full_days']} days, {_about(td)} ({_when(td, now)})"
        text += "."
    if not fc["no_growth"]:
        label = "Disk floor" if fc["disk_floor"] else "Disk full"
        if fc["disk_days"] is not None:
            text += f" {label}: {_about(fc['disk_days'])}."
        else:
            text += f" {label}: already reached."
    if fc["history_days"] < 30:
        text = "Early estimate: " + text
    tip = (f"Pace: the bytes of every photo file (photo, thumbnail, burst) of visits captured in the last {w} days "
           f"(the days since the first visit, at most 60), divided by {w}. ")
    if fc["thin"]:
        tip += (f"Thinning: photos older than {o['full_days']} days shrink to about {round(fc['thin_ratio'] * 100)}% "
                f"({o['thin_edge']} px), except protected photos and each species' best {o['full_per_day']} a day; "
                f"{round(fc['thin_share'] * 100)}% of the recent bytes would shrink, and photos already stored shrink as "
                f"they age. ")
    tip += "The disk line counts photo growth only, at the full pace, down to the floor; nothing else on the disk."
    return {"text": text, "tip": tip}


def storage_context(conn) -> dict:
    bd = retention.breakdown(conn)
    return {"bd": bd, "bar": storage_bar(bd), "forecast": forecast_blurb(retention.forecast(conn), bd)}


@app.get("/false-triggers", response_class=HTMLResponse)
def false_triggers(request: Request, cleared: int | None = None, confirmed: int | None = None, show: str = "",
                   conn=Depends(get_conn)):
    """Only the photos nobody has confirmed or corrected yet, unless ?show=all (then the confirmed ones show dimmed)."""
    show_all = show == "all"
    visits = db.list_visits(conn, false_triggers=True, has_photo=True, unchecked=not show_all, limit=FT_PAGE)
    return page(
        request,
        "false_triggers.html",
        visits=visits,
        show_all=show_all,
        next_url=_ft_more(visits, show_all),
        hourly=db.hourly_counts(conn, false_triggers=True),
        total=db.count_visits(conn, false_triggers=True, has_photo=True),
        storage=retention.usage(conn),
        **storage_context(conn),
        cleared=cleared,
        confirmed=confirmed,
        unchecked=db.count_unchecked_false_triggers(conn),
        ft_cards=True,
        four=steps.components(),
    )


@app.get("/false-triggers/cards", response_class=HTMLResponse)
def false_trigger_cards(request: Request, ids: str = "", after: int | None = None, show: str = "",
                        conn=Depends(get_conn)):
    """Fresh cards for visits opened in the pop-up: ones that stopped being false triggers are left out.
    ?after=<id>: instead, the false triggers newer than that id, filtered like the page (?show=all keeps
    confirmed ones), which the open page slides in at the top."""
    if after is not None:
        visits = db.false_triggers_after(conn, after, unchecked=show != "all")
    else:
        wanted = [int(i) for i in ids.split(",") if i.strip().isdigit()][:200]
        visits = db.false_trigger_cards(conn, wanted)
    resp = page(request, "_cards.html", visits=visits, ft_cards=True)
    resp.headers["X-Unchecked"] = str(db.count_unchecked_false_triggers(conn))
    resp.headers["X-Total"] = str(db.count_visits(conn, false_triggers=True, has_photo=True))
    return resp


def _ft_more(visits: list, show_all: bool) -> str | None:
    return more_url("/false-triggers/more", visits, FT_PAGE, show=("all" if show_all else ""))


@app.get("/false-triggers/more", response_class=HTMLResponse)
def more_false_triggers(request: Request, before: str = "", show: str = "", conn=Depends(get_conn)):
    """The next page of the list, after the card with the `before` cursor. Confirming photos while paging removes
    them from the list, so a position count would skip unseen ones; the cursor does not."""
    show_all = show == "all"
    visits = visits_before(conn, before, limit=FT_PAGE, false_triggers=True, has_photo=True, unchecked=not show_all)
    return page(request, "_cards.html", visits=visits, next_url=_ft_more(visits, show_all), ft_cards=True)


@app.post("/false-triggers/clean")
def clean_false_triggers(conn=Depends(get_conn)):
    return RedirectResponse(f"/false-triggers?cleared={retention.clear(conn)}", status_code=303)


@app.get("/settings", response_class=HTMLResponse)
def settings_page(request: Request, pruned: int | None = None, limits_error: str = "", limits_saved: bool = False,
                  alerts_saved: bool = False, alerts_test: str = "", zones_saved: bool = False, presence_saved: bool = False,
                  mirror_saved: bool = False, conn=Depends(get_conn)):
    storage = retention.usage(conn)
    limit = storage["limit_bytes"] / 1e6
    return page(
        request,
        "settings.html",
        pruned=pruned,
        mirror_saved=mirror_saved,
        mirror_owner_checks=mirror.owner_checks_shared(conn),
        presence_saved=presence_saved,
        presence_mode=presence.mode(conn),
        presence_report=db.presence_report(conn),
        alerts_saved=alerts_saved,
        alerts_test=alerts_test,
        alerts=alert.get_settings(conn),
        storage=storage,
        **storage_context(conn),
        limits_error=limits_error,
        limits_saved=limits_saved,
        thin_choices=retention.THIN_EDGE_CHOICES,
        budget_choices=BUDGET_CHOICES_GB,
        on_settings=True,
        activity=db.activity(conn),
        classifier=db.classifier_report(conn),
        classifier_on=config.CLASSIFIER != "off",
        classifier_mode=config.CLASSIFIER,
        current_limit=limit,
        # a limit set in .env that isn't one of the buttons still shows up as one, selected
        limit_choices=sorted({*retention.LIMIT_CHOICES_MB, limit}, key=lambda mb: mb or float("inf")),
        zones_saved=zones_saved,
        **zones_context(conn),
    )


@app.post("/settings/presence")
def save_presence(mode: str = Form(...), conn=Depends(get_conn)):
    if mode not in presence.MODES:
        raise HTTPException(400, "Unknown animal check mode")
    db.set_value(conn, "presence_mode", mode)
    return RedirectResponse("/settings?presence_saved=1#local", status_code=303)


@app.post("/settings/alerts/test")
def test_alert(conn=Depends(get_conn)):
    s = alert.get_settings(conn)
    if not s["topic"]:
        return RedirectResponse("/settings?alerts_test=no_topic#alerts", status_code=303)
    alert.send(s["topic"], "Birdbrain test", "Notifications are working!", None, None)
    return RedirectResponse("/settings?alerts_test=sent#alerts", status_code=303)


@app.post("/settings/alerts")
def save_alerts(
    ntfy_topic: str = Form(""),
    ntfy_new_species: str | None = Form(None),
    ntfy_mammals: str | None = Form(None),
    ntfy_unusual: str | None = Form(None),
    ntfy_quiet_start: str = Form(""),
    ntfy_quiet_end: str = Form(""),
    conn=Depends(get_conn),
):
    alert.save_settings(
        conn,
        topic=ntfy_topic,
        new_species=ntfy_new_species is not None,
        mammals=ntfy_mammals is not None,
        unusual=ntfy_unusual is not None,
        quiet_start=ntfy_quiet_start,
        quiet_end=ntfy_quiet_end,
    )
    return RedirectResponse("/settings?alerts_saved=1#alerts", status_code=303)


@app.post("/settings/mirror")
def save_mirror(owner_checks: str | None = Form(None), conn=Depends(get_conn)):
    """Settings > Public mirror: whether the owner's circle checks go with the next push (birdbrain.mirror)."""
    db.set_value(conn, mirror.OWNER_CHECKS_KEY, "1" if owner_checks is not None else "0")
    return RedirectResponse("/settings?mirror_saved=1#mirror", status_code=303)


@app.post("/settings/storage")
def save_storage(limit_mb: float = Form(...), photo: str = Form(...), conn=Depends(get_conn)):
    try:
        pruned = retention.save_options(conn, limit_mb, photo)
    except ValueError:
        raise HTTPException(400, "Unknown storage setting")
    return RedirectResponse(f"/settings?pruned={pruned}#storage", status_code=303)


BUDGET_CHOICES_GB = (2, 5, 10, 20, 50)
_run_state = {"running": False, "result": None, "error": ""}
_run_lock = threading.Lock()


def _step_lines(plan: dict) -> list[str]:
    lines = []
    for st in plan["steps"]:
        if st["count"]:
            lines.append(f"{st['label']}: {st['count']} photo{'s' if st['count'] != 1 else ''}, frees {mb(st['bytes'])}"
                         + (f" ({st['failed']} would fail)" if st.get("failed") else ""))
    return lines


@app.post("/settings/storage/limits")
def save_storage_limits(budget_gb: str = Form(...), budget_custom: str = Form(""), full_days: str = Form(...),
                        full_per_day: str = Form(...), thin_edge: str = Form(...), warn_pct: str = Form(...),
                        rare_under: str = Form(...), disk_floor_gb: str = Form(...),
                        thin_on: str | None = Form(None), conn=Depends(get_conn)):
    try:
        gb = budget_custom.strip() if budget_gb == "custom" else budget_gb
        values = {"budget_mb": float(gb) * 1000 if gb not in ("", "none") else 0,
                  "disk_floor_mb": float(disk_floor_gb) * 1000, "full_days": full_days, "full_per_day": full_per_day,
                  "thin_edge": thin_edge, "warn_pct": warn_pct, "rare_under": rare_under,
                  "thin_on": thin_on is not None}
        retention.save_photo_options(conn, **values)
    except ValueError as e:
        return RedirectResponse(f"/settings?limits_error={quote(str(e))}#storage", status_code=303)
    return RedirectResponse("/settings?limits_saved=1#storage", status_code=303)


@app.get("/settings/storage/preview", response_class=HTMLResponse)
def storage_preview(request: Request, conn=Depends(get_conn)):
    p = retention.plan(conn)
    return page(request, "_storage_preview.html", plan=p, lines=_step_lines(p))


@app.post("/settings/storage/run", response_class=HTMLResponse)
def storage_run(request: Request):
    """Thin and free space now, in a background thread so the request returns at once."""
    def work():
        try:
            with db.session() as c:
                result = retention.daily(c)
            _run_state.update(result=result, error="")
        except Exception as e:  # shown on the card, not raised into a dead thread
            _run_state.update(result=None, error=str(e) or type(e).__name__)
        finally:
            retention.forget()
            _run_state["running"] = False

    with _run_lock:
        if not _run_state["running"]:
            _run_state.update(running=True, result=None, error="")
            threading.Thread(target=work, daemon=True).start()
    return page(request, "_storage_run.html", state=_run_state)


@app.get("/settings/storage/run", response_class=HTMLResponse)
def storage_run_status(request: Request):
    return page(request, "_storage_run.html", state=_run_state)


def zones_context(conn) -> dict:
    """The Zones section of the Settings page."""
    runs = db.recent_calibrations(conn)
    current = zones.load(conn)
    # Undo applies to the change that made the current zones, as long as nothing has changed them since.
    undo = next((r for r in runs if r["status"] == "moved"), None)
    if undo and zones.Zones.from_dict(json.loads(undo["after_json"])) != current:
        undo = None
    return {"recal": zones.schedule(conn), "recal_runs": runs, "recal_pending": zones.pending(conn), "recal_status": zones.status(conn),
            "recal_undo": undo["id"] if undo else None,
            "recal_spots": len(current.spots), "recal_ring": bool(current.ring), "zones_now": zone_shapes(current),
            "zones_raw": current.to_dict(), "live_refresh": config.LIVE_REFRESH,
            **zone_drawing(current)}


def zone_drawing(z: zones.Zones) -> dict:
    """The zones as colored, numbered shapes over the live photo: editable on Settings, shown on Claude's Notes."""
    return {"zone_objs": [{**o, **scene_shape(o), "color": SCENE_COLORS[i % len(SCENE_COLORS)],
                           "water": ellipse_path(*o["inner"]) if "inner" in o else None}
                          for i, o in enumerate(zones.objects(z))],
            "zone_w": ZONE_W, "zone_h": ZONE_H}


def camera_context(conn, saved: str = "", error: str = "", tab: str = "") -> dict:
    """Settings > Camera: what the camera reports now, or why it couldn't be read."""
    ctx = dict(cam_configured=camera.configured(), cam_saved=saved, cam_error=error,
               cam_tabs=camera.TABS, cam_tab=tab if tab in camera.TABS or tab == "changes" else "picture",
               cam_image_fields=camera.IMAGE, cam_daynight_fields=camera.DAYNIGHT,
               cam_log=camera.history(conn), cam_fields=camera.FIELDS, cam_show=camera.show,
               cam_stamp=int(time.time()))
    if not ctx["cam_configured"]:
        return ctx
    try:
        # Asked side by side: one after another took ~0.8 s on the Pi, the slowest alone ~0.45 s.
        with ThreadPoolExecutor(3) as pool:
            reads = {k: pool.submit(f) for k, f in
                     (("cam_picture", camera.picture), ("cam_daynight", camera.daynight), ("cam_status", camera.status))}
        ctx.update({k: f.result() for k, f in reads.items()})
    except camera.CameraError as e:
        ctx.update(cam_down=str(e))
    return ctx


@app.get("/settings/camera", response_class=HTMLResponse)
def camera_panel(request: Request, conn=Depends(get_conn)):
    return page(request, "_camera.html", **camera_context(conn))


async def _camera_form(request: Request) -> dict:
    return {k: v for k, v in (await request.form()).items() if isinstance(v, str)}


def _camera_change(request: Request, conn, action, saved: str, tab: str) -> HTMLResponse:
    try:
        action()
    except (camera.CameraError, ValueError) as e:
        return page(request, "_camera.html", **camera_context(conn, error=str(e), tab=tab))
    return page(request, "_camera.html", **camera_context(conn, saved=saved, tab=tab))


@app.post("/settings/camera/picture", response_class=HTMLResponse)
async def camera_picture(request: Request, conn=Depends(get_conn)):
    form = await _camera_form(request)
    tab = form.get("tab", "")

    def save():
        values = camera.defaults(tab) if form.get("defaults") else camera.clean_picture(form, tab)
        camera.set_picture(conn, values)
    return await asyncio.to_thread(_camera_change, request, conn, save, "picture", tab)


@app.post("/settings/camera/daynight", response_class=HTMLResponse)
async def camera_daynight(request: Request, conn=Depends(get_conn)):
    form = await _camera_form(request)
    return await asyncio.to_thread(_camera_change, request, conn,
                                   lambda: camera.set_daynight(conn, camera.clean_daynight(form)), "daynight", "daynight")


@app.post("/settings/camera/undo/{change_id}", response_class=HTMLResponse)
def camera_undo(request: Request, change_id: int, conn=Depends(get_conn)):
    return _camera_change(request, conn, lambda: camera.undo(conn, change_id), "undone", "changes")


@app.post("/settings/camera/reboot", response_class=HTMLResponse)
def camera_reboot(request: Request, conn=Depends(get_conn)):
    try:
        camera.reboot()
    except camera.CameraError as e:
        return page(request, "_camera.html", **camera_context(conn, error=str(e)))
    return page(request, "_camera.html", cam_configured=True, cam_rebooting=True)


@app.get("/settings/zones", response_class=HTMLResponse)
def zones_panel(request: Request, conn=Depends(get_conn)):
    """The Zones section's status alone; it polls itself while a recalibration is running."""
    return page(request, "_zones.html", **zones_context(conn))


@app.get("/settings/zones/status")
def zones_status(conn=Depends(get_conn)):
    """Polled by the Zones panel's timer while a run is pending, so the panel itself isn't redrawn."""
    return zones.status(conn)


@app.post("/settings/zones/now")
def recalibrate_now(conn=Depends(get_conn)):
    zones.request(conn)
    return RedirectResponse("/settings#zones", status_code=303)


@app.post("/settings/zones")
def save_zone_schedule(auto: str | None = Form(None), time1: str = Form(""), time2: str = Form(""),
                       sun1: str | None = Form(None), sun2: str | None = Form(None), conn=Depends(get_conn)):
    try:
        zones.save_schedule(conn, auto is not None, ["sunrise" if sun1 else time1, "sunset" if sun2 else time2])
    except ValueError:
        raise HTTPException(400, "Times must look like 14:30") from None
    return RedirectResponse("/settings?zones_saved=1#zones", status_code=303)


def _record_undo(conn, summary: str) -> None:
    db.add_calibration(conn, ran_at=datetime.now().isoformat(timespec="seconds"), reason="undo",
                       status="undone", summary=summary)


@app.post("/settings/zones/undo/{cal_id}")
def undo_recalibration(cal_id: int, conn=Depends(get_conn)):
    row = db.get_calibration(conn, cal_id)
    if not row or not row["before_json"]:
        raise HTTPException(404)
    before = zones.Zones.from_dict(json.loads(row["before_json"]))
    if before == zones.from_env():
        zones.reset(conn)
    else:
        zones.save(conn, before, manual=False if row["reason"] == "hand" else None)
    zones.clear_reference()  # it may show the camera where the undone zones put it
    _record_undo(conn, f"Put the zones back as they were before {nice_time(row['ran_at'])}.")
    return RedirectResponse("/settings#zones", status_code=303)


@app.post("/settings/zones/manual")
async def save_manual_zones(request: Request, conn=Depends(get_conn)):
    """Adjust zones: the owner dragged the shapes. They become the anchor for later automatic runs, and the
    reference views are retaken on the next run so camera moves are measured from this view."""
    before = zones.load(conn)
    try:
        data = await request.json()
        if "objects" in data:  # from the shape editor
            data = zones.from_objects(data["objects"])
        after = zones.edited(before, data)
    except (ValueError, TypeError, KeyError):
        raise HTTPException(400, "Bad zones")
    zones.save(conn, after, manual=True)
    zones.clear_reference()
    db.add_calibration(conn, ran_at=datetime.now().isoformat(timespec="seconds"), reason="hand", status="moved",
                       summary="You adjusted the zones by hand. Later automatic checks start from these.",
                       before_json=json.dumps(before.to_dict()), after_json=json.dumps(after.to_dict()))
    return {"ok": True}


# --- Shapes drawn over photos (the zones on Settings and Claude's Notes) ---

def scene_shape(obj: dict) -> dict:
    """One shape (a zone from zones.objects) as an SVG path in the ZONE_W x ZONE_H box, with where to put its label."""
    if "ellipse" in obj:
        cx, cy, rx, ry = obj["ellipse"][:4]
        d = ellipse_path(*obj["ellipse"]) + (" " + ellipse_path(*obj["inner"]) if "inner" in obj else "")  # a ring
        # the ring's water is a hole, so clicks there need their own (invisible) target
        hit = ellipse_path(*obj["ellipse"]) if "inner" in obj else None
        return {"d": d, "hit": hit, "lx": cx * ZONE_W, "ly": max(cy - ry, 0.04) * ZONE_H}
    pts = obj["points"]
    poly = lambda ps: "M" + " L".join(f"{x * ZONE_W:.2f},{y * ZONE_H:.2f}" for x, y in ps) + " Z"  # noqa: E731
    d = " ".join(poly(ps) for ps in [pts, *obj.get("pieces", [])])
    top = min(pts, key=lambda p: p[1])
    return {"d": d, "lx": top[0] * ZONE_W, "ly": max(top[1], 0.04) * ZONE_H}


# One color per shape (not per kind), so two shapes of a kind, or overlapping ones, can be told apart.
SCENE_COLORS = ["#4fb3ff", "#7be08a", "#ff6b6b", "#e5d36b", "#c99bff", "#ffb347", "#5fe0d0", "#ff7ab8", "#a0a8ff",
                "#d0e86b", "#ff9f6b", "#8fd3ff"]
WATER_COLOR = "#9ff3ff"  # the water line inside a bath ring, dashed


@app.get("/calibration/{name}")
def calibration_picture(name: str):
    path = config.DATA_DIR / "calibration" / name
    if not re.fullmatch(r"cal_\d{8}_\d{6}\.jpg", name) or not path.exists():
        raise HTTPException(404)
    return FileResponse(path, media_type="image/jpeg")


def identify_prompt_preview(site: dict, lessons: list[str]) -> str:
    from ..identify import system_prompt  # imported lazily: keeps the web app from loading the SDK at startup

    return system_prompt(lessons, site)


@app.get("/api/latest-visit")
def latest_visit(request: Request, conn=Depends(get_conn)):
    """Newest real visitor (not false triggers), plus the newest visit of any kind (`any_id`, the admin's False
    triggers page refreshes on it). Visitors and the public mirror never learn of a false trigger: their any_id is
    the real visitor's id, so it only ever moves along with a real visit (static/tab-alert.js).

    Pages poll this to flag new arrivals in the tab and to pull in new cards, numbers and arrows."""
    row = db.latest_visit(conn)
    newest = {"id": row["id"], "label": row["label"], "captured_at": row["captured_at"]} if row else \
        {"id": 0, "label": None, "captured_at": None}
    any_id = db.newest_id(conn) if request.state.admin else newest["id"]
    # Nav bubble counts: zeros for visitors and the public mirror (nav_waiting checks both).
    return {**newest, "any_id": any_id, "waiting": nav_waiting(request)}


@app.get("/manifest.webmanifest", include_in_schema=False)
def manifest():
    """Web app manifest: lets Android use a maskable icon (a full-bleed square it can crop to a circle) when
    the site is added to the home screen. JSON isn't rewritten by the base_path middleware, so site_url()."""
    st = site_url("/static")
    v = templates.env.globals["asset_version"]
    icons = [{"src": f"{st}/icon-{n}.png?v={v}", "sizes": f"{n}x{n}", "type": "image/png", "purpose": "any"}
             for n in (192, 512)]
    icons += [{"src": f"{st}/icon-maskable-{n}.png?v={v}", "sizes": f"{n}x{n}", "type": "image/png",
               "purpose": "maskable"} for n in (192, 512)]
    data = {"name": "Birdbrain", "short_name": "Birdbrain", "start_url": site_url("/"), "scope": site_url("/"),
            "display": "standalone", "background_color": "#f6f5f0", "icons": icons}
    # No theme_color, on purpose. In the installed Android app Chrome paints the navigation bar with the
    # manifest's theme_color, fixed when the app is installed and never from the page, and it has no dark
    # variant in the manifest (only desktop experiments), so a light one gave a white bar on a dark phone.
    # Without it Chrome uses its own default bar, which follows the phone's dark mode. In a browser tab and
    # in the app's status bar the page's <meta name="theme-color"> (set from the saved theme) still rules.
    return Response(json.dumps(data), media_type="application/manifest+json")


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return FileResponse(HERE / "static" / "favicon.ico", media_type="image/x-icon")


# Live snapshots: at most one camera grab per refresh interval, shared by every viewer.
LIVE_MAX_AGE = max(config.LIVE_REFRESH - 1, 1)
# A frame older than this means the camera is down; say so instead of serving it as live.
LIVE_STALE = max(3 * config.LIVE_REFRESH, 30)
_live = {"jpeg": None, "at": 0.0, "tried": 0.0}
_live_lock = asyncio.Lock()


def live_file() -> Path:
    return config.DATA_DIR / "live.jpg"


def load_saved_frame() -> None:
    """Start with the frame saved before the last restart so the page has something to show at once."""
    path = live_file()
    if not path.exists():
        return
    # The public mirror gets a new file from the Pi every few minutes, so reload whenever it changes.
    if _live["jpeg"] is None or (auth.public() and path.stat().st_mtime > _live["at"]):
        _live.update(jpeg=path.read_bytes(), at=path.stat().st_mtime)


def save_frame(jpeg: bytes) -> None:
    path = live_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(jpeg)
    tmp.replace(path)  # atomic, so a reader never sees half a file


NIGHT_SATURATION = 12  # the watcher's rule (watch.frame_is_night): IR night frames are nearly grayscale


def frame_light() -> str | None:
    """'day' or 'night' for the cached live frame, judged from its color (IR frames are gray); None without a frame.

    Read from the picture, never from the camera, so the page costs the camera nothing and works the same on the
    public mirror. Cached per frame."""
    jpeg = _live["jpeg"]
    if jpeg is None:
        return None
    if _live.get("light_for") != _live["at"]:
        try:
            with Image.open(io.BytesIO(jpeg)) as im:
                im.draft("RGB", (160, 90))  # JPEG decodes small for free
                sat = ImageStat.Stat(im.convert("RGB").resize((64, 36)).convert("HSV")).mean[1]
            _live["light"] = "night" if sat < NIGHT_SATURATION else "day"
        except (OSError, ValueError):
            _live["light"] = None
        _live["light_for"] = _live["at"]
    return _live["light"]


def frame_response() -> Response:
    """The cached frame plus its age and light, so the page can label stale frames honestly."""
    age = max(0.0, time.time() - _live["at"])
    headers = {"Cache-Control": "no-store", "X-Frame-Age": f"{age:.1f}"}
    if light := frame_light():
        headers["X-Frame-Light"] = light
    return Response(_live["jpeg"], media_type="image/jpeg", headers=headers)


async def grab_live_frame() -> bytes:
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-rtsp_transport", "tcp",
        "-i", config.rtsp_url(config.RTSP_SUB), "-frames:v", "1", "-f", "image2", "-c:v", "mjpeg", "pipe:1",
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
    )
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=10)
        return out
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return b""


@app.api_route("/live/latest.jpg", methods=["GET", "HEAD"])
def live_latest():
    """Whatever frame we already have, however old, without touching the camera: paints instantly."""
    load_saved_frame()
    if _live["jpeg"] is None:
        raise HTTPException(404, "no frame yet")
    return frame_response()


async def fresh_frame() -> bytes | None:
    """The live frame, grabbed anew if older than LIVE_MAX_AGE; None if the camera is down."""
    if auth.public():
        load_saved_frame()
        return _live["jpeg"]  # never stale-checked: the page labels its age from X-Frame-Age
    async with _live_lock:
        if time.time() - _live["tried"] >= LIVE_MAX_AGE:
            _live["tried"] = time.time()
            if out := await grab_live_frame():
                _live.update(jpeg=out, at=time.time())
                save_frame(out)
        if _live["jpeg"] is None or time.time() - _live["at"] > LIVE_STALE:
            return None
        return _live["jpeg"]


@app.get("/live.jpg")
async def live_jpg():
    """A fresh frame (at most LIVE_MAX_AGE old), or 503 if the camera is down."""
    if await fresh_frame() is None:
        raise HTTPException(503, "camera unavailable")
    return frame_response()
