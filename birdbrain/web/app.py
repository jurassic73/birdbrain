import asyncio
import json
import math
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta
from pathlib import Path
from urllib.parse import urlencode

from fastapi import Depends, FastAPI, Form, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from . import auth, viewer
from .. import alert, camera, circle_check, clock, config, db, identify, images, names, presence, privacy, refine, retention, species_info, zones

HERE = Path(__file__).parent
PAGE_SIZE = 24
FT_PAGE = 48  # false triggers come in pages too, so a rainy week isn't one 20,000 px scroll on a phone


@asynccontextmanager
async def lifespan(app: FastAPI):
    with db.session() as conn:
        db.init_db(conn)
    yield


app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None)
config.IMAGES_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
app.mount("/images", StaticFiles(directory=config.IMAGES_DIR), name="images")
templates = Jinja2Templates(directory=HERE / "templates")


@app.middleware("http")
async def read_only(request: Request, call_next):
    """Visitors can browse; changes, Claude calls and the admin pages need the admin login."""
    if viewer.enabled() and not viewer.is_open(request.url.path) and not viewer.signed_in(request):
        # Friends & family gate: nothing past this point for anyone without the guest cookie.
        if request.method in ("GET", "HEAD"):
            return RedirectResponse("/guest-login?" + urlencode({"next": request.url.path}), status_code=303)
        return Response("Sign in first.", status_code=401)
    admin = request.state.admin = auth.is_admin(request)
    if clock.stale():
        with db.session() as conn:
            clock.set_zone(db.get_value(conn, clock.KEY))
    path = request.url.path
    if auth.public():
        if request.method not in ("GET", "HEAD") and path != "/guest-login":
            return Response("This is a read-only copy.", status_code=403)
        if auth.admin_only(path) or path == "/login":
            return Response("Not found", status_code=404)
    elif not admin and path not in ("/login", "/logout"):
        if request.method not in ("GET", "HEAD"):
            return Response("Read-only: log in as admin to change things.", status_code=403)
        if auth.admin_only(path):
            return RedirectResponse("/login?" + urlencode({"next": path}), status_code=303)
    return await call_next(request)


# Serving under a path such as somegeek.com/birdbrain/: the proxy strips the prefix on the way in, and this
# puts it back on the way out (links, forms, redirects), so the templates can keep their plain "/..." URLs.
BASE = os.environ.get("BIRDBRAIN_BASE", "").strip().rstrip("/")
_URL_ATTR = re.compile(r'(\s(?:href|src|action|formaction|hx-get|hx-post|data-[a-z-]+)=")/(?!/)')
# fetch() calls in the page and in static/*.js use "/..." too; this shim prefixes those.
_FETCH_SHIM = ('<script>window.BB_BASE={base};(()=>{{const f=window.fetch;window.fetch=(u,o)=>f('
               'typeof u==="string"&&u.startsWith("/")&&!u.startsWith("//")?{base}+u:u,o)}})();</script>')


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


templates.env.filters.update(nice_time=nice_time, nice_date=nice_date, short_time=short_time, usd=usd, mb=mb,
                             month_name=lambda v: datetime.fromisoformat(v).strftime("%B"))
templates.env.globals["stayed"] = lambda v: stayed(v["captured_at"], v["ended_at"]) if "ended_at" in v.keys() else ""
templates.env.globals["mirror_status"] = privacy.visit_status
templates.env.globals["has_card"] = species_info.has_card
templates.env.globals["season_badge"] = species_info.visit_badge
templates.env.globals["short_status"] = species_info.short_status
templates.env.globals["photo_marks"] = lambda visit: start_marks(visit)
templates.env.globals["no_circle_note"] = lambda visit: no_circle_note(visit)
templates.env.globals["also_of"] = db.also_of
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
templates.env.globals["count_label"] = count_label


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
    if not viewer.enabled():
        return _guest_reply(request, True, next)
    ip = viewer.client_ip(request)
    if wait := viewer.locked(ip):
        return _guest_reply(request, False, next, _wait_text(wait), wait, 429)
    if not (viewer.check_invite(key) if key else viewer.check_password(password)):
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
                    secure=request.url.scheme == "https" or request.headers.get("x-forwarded-proto") == "https")
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
        ("Night", {"tod": "night"}),
        ("Today", {"date_from": today.isoformat(), "date_to": today.isoformat()}),
        ("This week", {"date_from": (today - timedelta(days=6)).isoformat()}),
    ]


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
    return {"score": r, "score_rows": db.scoreboard(conn, r["date_from"], r["date_to"]), "score_ranges": SCORE_RANGES}


@app.get("/", response_class=HTMLResponse)
def home(request: Request, conn=Depends(get_conn)):
    filters = filters_from(request)
    active = {k: v for k, v in filters.items() if v}
    visits = db.list_visits(conn, **filters, limit=PAGE_SIZE)
    chips = [(name, f"/?{query_string(f)}" if f else "/", f == active) for name, f in quick_filters()]
    return page(
        request,
        "home.html",
        visits=visits,
        filters=filters,
        filtered=bool(active),
        more_filters_open=bool(filters["species"] or (filters["date_to"] and filters["date_from"] != filters["date_to"])),
        chips=chips,
        total=db.count_visits(conn, **filters),
        any_visits=db.latest_visit(conn) is not None,
        today=db.today_summary(conn),
        presence_mode=presence.mode(conn),
        activity=db.activity(conn),
        species_names=db.species_names(conn),
        next_url=f"/visits/more?{query_string(filters, offset=PAGE_SIZE)}" if len(visits) == PAGE_SIZE else None,
        split_day=None if active else clock.today().isoformat(),
        live_refresh=config.LIVE_REFRESH,
        live_stale=LIVE_STALE,
        # The Today and This week chips show the same period on the scoreboard.
        **scoreboard_ctx(conn, next(({"Today": "today", "This week": "week"}.get(n) for n, _, on in chips if on), None)),
    )


@app.get("/scoreboard", response_class=HTMLResponse)
def scoreboard(request: Request, range: str = "today", conn=Depends(get_conn)):
    return page(request, "_scoreboard.html", **scoreboard_ctx(conn, range))


@app.get("/visits", response_class=HTMLResponse)
def visits_archive(request: Request, conn=Depends(get_conn)):
    """The plain archive: every real visit, newest first, one grid with the home page's filters."""
    filters = filters_from(request)
    active = {k: v for k, v in filters.items() if v}
    visits = db.list_visits(conn, **filters, limit=PAGE_SIZE)
    return page(
        request,
        "visits.html",
        visits=visits,
        filters=filters,
        filtered=bool(active),
        total=db.count_visits(conn, **filters),
        species_names=db.species_names(conn),
        next_url=f"/visits/archive-more?{query_string(filters, offset=PAGE_SIZE)}" if len(visits) == PAGE_SIZE else None,
    )


@app.get("/visits/archive-more", response_class=HTMLResponse)
def visits_archive_more(request: Request, offset: int = 0, conn=Depends(get_conn)):
    filters = filters_from(request)
    visits = db.list_visits(conn, **filters, limit=PAGE_SIZE, offset=offset)
    more = f"/visits/archive-more?{query_string(filters, offset=offset + PAGE_SIZE)}" if len(visits) == PAGE_SIZE else None
    return page(request, "_cards.html", visits=visits, next_url=more)


@app.get("/visits/more", response_class=HTMLResponse)
def more_visits(request: Request, offset: int = 0, conn=Depends(get_conn)):
    filters = filters_from(request)
    visits = db.list_visits(conn, **filters, limit=PAGE_SIZE, offset=offset)
    next_offset = offset + PAGE_SIZE
    # Unfiltered, the cards are split into Today and Earlier; the card just before this page says which side it was on.
    split_day = prev_day = None
    if not any(filters.values()) and offset > 0:
        split_day = clock.today().isoformat()
        prev = db.list_visits(conn, limit=1, offset=offset - 1)
        prev_day = prev[0]["captured_at"][:10] if prev else None
    return page(
        request,
        "_cards.html",
        visits=visits,
        split_day=split_day,
        prev_day=prev_day,
        next_url=f"/visits/more?{query_string(filters, offset=next_offset)}" if len(visits) == PAGE_SIZE else None,
    )


@app.get("/visit/{visit_id}", response_class=HTMLResponse)
def visit_detail(request: Request, visit_id: int, ft: bool = False, todo: bool = False,
                 species: str = "", conn=Depends(get_conn)):
    visit = db.get_visit(conn, visit_id)
    if not visit:
        raise HTTPException(404)
    is_ft = ft or db.is_false_trigger(conn, visit_id)
    # todo: opened from the False triggers list of unconfirmed photos, so the arrows (and C) skip confirmed ones.
    todo = todo and is_ft
    # species: opened from that species' page, so the arrows step through only its visits.
    scope = "" if is_ft else species
    prev_id, next_id = db.get_neighbors(conn, visit_id, false_triggers=is_ft, unchecked=todo, species=scope or None)
    # The arrow targets' photos, prefetched so stepping through visits paints without a blank frame.
    neighbor_images = [v["image_path"] for v in (db.get_visit(conn, i) for i in (prev_id, next_id) if i) if v]
    return page(
        request,
        "visit.html",
        visit=visit,
        visitors=db.VISITORS,
        species_kinds=db.species_kinds(conn),
        decision=decision_steps(visit["raw_json"]),
        marks=start_marks(visit),
        prev_id=prev_id,
        next_id=next_id,
        neighbor_images=neighbor_images,
        is_ft=is_ft,
        todo=todo,
        scope=scope,
        confirmed=db.confirmed_species(conn),
        burst=burst_frames(visit),
        vzones=visit_zones(visit, conn),
    )


def burst_frames(visit) -> list[dict]:
    """A false trigger's whole burst for the visit page's filmstrip, in capture order: each frame's path,
    seconds after the first and whether it's the one Claude saw. Empty when only the best frame was kept."""
    paths = images.burst_paths(visit["image_path"])
    if not paths:
        return []
    try:
        info = (json.loads(visit["raw_json"] or "{}") or {}).get("burst") or {}
    except (ValueError, AttributeError):
        info = {}
    fps = info.get("fps") or config.BURST_FPS
    return [{"path": p, "at": i / fps, "best": i == info.get("best")} for i, p in enumerate(paths)]


FIRED = {"rim": "the rim and rocks", "water": "the open water", "ground": "the ground"}


def visit_zones(visit, conn) -> dict | None:
    """What the visit photo draws over itself: the crop box the Pi's animal check and Claude looked at, and the
    zone whose motion fired the capture, each in its Settings color. `source` says whose zones: stored with the
    photo (since 2026-11-01), rebuilt from the calibration history at the time, or today's."""
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
            try:
                z = zones.Zones.from_dict(d)
                break
            except (TypeError, ValueError):
                pass
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
    if trigger == "ground":  # the ground crop isn't a Settings shape: it takes the ground's color
        crop, crop_color = box_rect(z.ground_crop), colors.get("ground", SCENE_COLORS[0])
    else:
        crop, crop_color = box_rect(z.crop), colors.get("crop", SCENE_COLORS[0])
    if not crop and not shapes:
        return None
    return {"w": ZONE_W, "h": ZONE_H, "shapes": shapes, "crop": crop, "crop_color": crop_color,
            "fired": FIRED.get(trigger), "source": source}


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


CLASSIFIER_CROPS = {"circle": "around Claude's circle", "box": "on the crop box", "frame": "on the whole frame"}


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
            "model": "Local animal check",
            "presence": True,
            "said": "nothing there" if pres.get("empty") else "an animal",
            "score": pres["score"],
            "threshold": pres.get("threshold"),
            "trained_on": pres.get("trained_on"),
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
    for key, name in (("haiku", "Haiku"), ("sonnet", "Sonnet")):
        call = calls.get(key)
        if not call:
            continue
        answer = call.get("answer") or {}
        steps.append({
            "model": name,
            "frames": call.get("frames", 1),
            "refused": not answer,
            "said": answer.get("species") or ("no animal" if answer.get("visitor") == "none" else answer.get("visitor")),
            "also": [db.also_label(a) + (f" x{a['count']}" if (a.get("count") or 1) > 1 else "")
                     for a in answer.get("also") or [] if answer.get("visitor") != "none" and a.get("visitor")],
            "confidence": answer.get("confidence"),
            "notes": answer.get("notes"),
        })
    guess = calls.get("classifier")
    if guess and guess.get("top") is not None:
        steps.append({
            "model": "Local classifier",
            "shadow": True,
            "gated": calls.get("settled_by") == "classifier",
            "crowd": calls.get("gate_crowd"),
            "guesses": [f"{g['name']} ({g['score']:.0%})" for g in guess["top"]],
            "seconds": guess.get("seconds"),
            "crop": CLASSIFIER_CROPS.get(guess.get("crop"), ""),
            "night": bool(guess.get("night")),
        })
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
            "owner_checks": owner_checks(calls),
            "skipped_sonnet": bool(calls.get("skipped_sonnet")), "asked_sonnet": bool(calls.get("sonnet"))}


CHECK_BY = {"pi": "The Pi", "haiku": "Haiku", "sonnet": "Sonnet"}


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
                    "detail": detail, "at": c.get("at") or "", "usd": c.get("usd"), "seconds": c.get("seconds")})
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
    db.correct_visit(conn, visit_id, new_visitor, new_species, feedback, db.parse_marks(marks), also)
    return Response(headers=REFRESH)


@app.get("/species-search")
def species_search(q: str = "", known: list[str] = Query([])):
    """eBird species names matching q, for the correction form's species boxes (the page lists the yard's own)."""
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
    jpeg = (config.IMAGES_DIR / v["image_path"]).read_bytes()
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
        official = names.canonical(name, [n for n, _ in db.species_kinds(conn)])
        if not official and names.bird_names():
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
    picked = circle_check.owner_circle(marks)
    if not picked:
        return JSONResponse({"error": "Circle the animal first: press Edit circles and drag on the photo."})
    number, circle = picked
    jpeg = circle_check.jpeg_of(v)
    if jpeg is None:
        return JSONResponse({"error": "This photo was deleted to save space, so there's nothing to look at."})
    loop = asyncio.get_running_loop()
    try:
        if source == "pi":
            # Its own low-priority thread, one check at a time: ~6 to 22 s of the Pi's cores.
            entry = await loop.run_in_executor(circle_check.POOL, circle_check.ask_pi, conn, jpeg, circle,
                                               bool(v["is_night"]))
            loop.call_later(circle_check.IDLE + 1, circle_check.release_if_idle)
        else:
            entry = await asyncio.to_thread(circle_check.ask_claude, conn, source, jpeg, circle,
                                            datetime.fromisoformat(v["captured_at"]), circle_check.frame_size(jpeg))
    except Exception as e:
        if source == "pi":
            return JSONResponse({"error": f"The Pi's model isn't available right now ({e})."})
        return JSONResponse({"error": f"Claude couldn't be reached ({type(e).__name__})."})
    circle_check.record(conn, visit_id, entry, circle)
    return JSONResponse({**entry, "circle_n": number})


@app.get("/species-photo")
async def species_photo(request: Request, name: str = Query(""), conn=Depends(get_conn)):
    """A reference photo for the Ask-the-Pi guesses. Only admins can make the site look one up; visitors get
    what is already cached."""
    photo = await asyncio.to_thread(species_info.reference_photo, conn, name.strip(), bool(request.state.admin))
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
                confirmed=db.confirmed_species(conn))


@app.get("/api/photo-reminder")
def photo_reminder(request: Request, conn=Depends(get_conn)):
    """Species with 2+ photos and no favourite yet, for the weekly reminder pop-up. Admins only (visitors get none)."""
    names = db.species_without_favorite(conn) if request.state.admin else []
    return {"species": names}


@app.get("/species/{name}", response_class=HTMLResponse)
def species_page(request: Request, name: str, conn=Depends(get_conn)):
    row = db.species_row(conn, name)
    if not row:
        raise HTTPException(404)
    return page(
        request,
        "species.html",
        sp=row,
        visits=db.list_visits(conn, species=name, limit=200),
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
        fav_pick=row["photo_id"] if row["photo_picked"] else None,
    )


@app.post("/species/{name}/photo")
def species_photo(name: str, visit_id: int = Form(...), conn=Depends(get_conn)):
    """The star on a species page's cards: use this visit's photo for the species, or (the current pick again)
    go back to the automatic pick. Answers with the visit now picked, or null."""
    if not db.shows_species(conn, visit_id, name):
        raise HTTPException(404)
    picked = None if db.species_photos(conn).get(name) == visit_id else visit_id
    db.set_species_photo(conn, name, picked)
    return {"visit_id": picked}


@app.get("/species-info/{name}", response_class=HTMLResponse)
def species_card(request: Request, name: str, mini: bool = False, fold: bool = False, conn=Depends(get_conn)):
    """The species info card (or its hover preview), built from outside sources on first use."""
    # Visitors only see cards already built, so they can't make the site fetch or call Claude.
    info = species_info.get(conn, name, fetch=request.state.admin)
    if not info:
        return HTMLResponse('<p class="muted sp-empty">No info found for this one yet.</p>')
    return page(
        request,
        "_species_tip.html" if mini else "_species_card.html",
        info=info,
        sp=db.species_row(conn, name),
        ours=db.month_counts(conn, name),
        months=species_info.MONTHS,
        this_month=clock.today().month,
        confirmed=db.confirmed_species(conn),
        fold=fold,  # the species page folds the long part; the pop-up stays open
    )


@app.get("/stats", response_class=HTMLResponse)
def stats(request: Request, conn=Depends(get_conn)):
    return page(
        request,
        "stats.html",
        daily=db.daily_counts(conn, days=30),
        hourly=db.hourly_counts(conn),
        leaders=db.species_summary(conn, sort="visits", limit=10),
        newest=db.species_summary(conn, sort="newest", limit=5),
        total=db.count_visits(conn),
        activity=db.activity(conn),
        classifier=db.classifier_report(conn),
        presence=db.presence_report(conn),
        presence_mode=presence.mode(conn),
        classifier_on=config.CLASSIFIER != "off",
        classifier_mode=config.CLASSIFIER,
    )


# The About page draws zones in a 16:9 box, the shape of both camera streams.
ZONE_W, ZONE_H = 160, 90


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
    return shapes


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


def backoff_steps() -> list[int]:
    """Seconds waited after each false trigger in a row, until the cap."""
    steps = []
    while len(steps) < 8 and (not steps or steps[-1] < config.COOLDOWN_NONE_MAX):
        steps.append(config.false_trigger_wait(len(steps) + 1))
    return steps


# Illustration for the season badges when no cached card has a clear season yet: a summer visitor's reports.
EXAMPLE_COUNTS = [0, 0, 1, 6, 48, 90, 100, 74, 28, 4, 0, 0]


def season_example(cards: dict[str, dict]) -> dict:
    """A species card with a distinct season (or the illustration) and the badge each month would get."""
    name, counts, expected = None, EXAMPLE_COUNTS, species_info.local_status(EXAMPLE_COUNTS)[1]
    for label in sorted(cards):
        card = cards[label]
        if card.get("local_counts") and 3 <= len(card.get("expected_months") or []) <= 9:
            name, counts, expected = label, card["local_counts"], card["expected_months"]
            break
    info = {"local_counts": counts, "expected_months": expected}
    peak = max(counts) or 1
    months = [{"name": species_info.MONTHS[m - 1], "pct": round(100 * n / peak), "expected": m in expected,
               "badge": species_info.season_badge(info, m)} for m, n in enumerate(counts, 1)]
    return {"name": name, "months": months, "status": species_info.local_status(counts)[0]}


@app.get("/about", response_class=HTMLResponse)
def about(request: Request, conn=Depends(get_conn)):
    cards = species_info.cached_cards()
    return page(
        request,
        "about.html",
        cfg=config,
        totals=db.totals(conn),
        rate=db.capture_rate(conn),
        zones=zone_shapes(zones.load(conn)),
        recal=zones.schedule(conn),
        recal_stored=zones.stored(conn) is not None,
        has_frame=_live["jpeg"] is not None or live_file().exists(),
        backoff=backoff_steps(),
        classifier_mode=config.CLASSIFIER,
        classifier=db.classifier_report(conn),
        presence=db.presence_report(conn),
        presence_mode=presence.mode(conn),
        haiku=identify.HAIKU,
        sonnet=identify.SONNET,
        lessons=len(db.recent_lessons(conn)),
        confirmed_min=db.CONFIRMED_MIN,
        site=db.prompt_site(conn),
        avg_call=db.avg_haiku_call_usd(conn),
        alerts=alert.get_settings(conn),
        cards=len(cards),
        season=season_example(cards),
        expected_pct=round(species_info.EXPECTED_SHARE * 100),
        card_days=species_info.REFRESH_AFTER.days,
        local_km=species_info.LOCAL_KM,
        nearby_km=species_info.NEARBY_KM,
    )


@app.get("/review", response_class=HTMLResponse)
def review(request: Request, conn=Depends(get_conn)):
    return page(request, "review.html", visits=db.list_visits(conn, review=True, limit=200))


@app.get("/notes", response_class=HTMLResponse)
def notes(request: Request, saved: bool = False, conn=Depends(get_conn)):
    """Claude's Notes: everything Claude is told about this yard, in one place."""
    site = db.prompt_site(conn)
    rows = db.all_lessons(conn)
    return page(
        request,
        "notes.html",
        site=site,
        saved=saved,
        lessons=rows,
        in_use=sum(l["in_use"] for l in rows),
        limit=db.LESSON_LIMIT,
        visitors=db.VISITORS,
        prompt_preview=identify_prompt_preview(site, db.recent_lessons(conn)),
        **zone_drawing(zones.load(conn)),
    )


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
def lessons():
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
    return Response(headers=REFRESH)


@app.post("/lessons/{visit_id}/{state}")
def lesson_switch(visit_id: int, state: str, conn=Depends(get_conn)):
    if state not in ("on", "off"):
        raise HTTPException(404)
    db.set_lesson_off(conn, visit_id, state == "off")
    return Response(headers=REFRESH)


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
        next_url=_ft_more(FT_PAGE, show_all) if len(visits) == FT_PAGE else None,
        hourly=db.hourly_counts(conn, false_triggers=True),
        total=db.count_visits(conn, false_triggers=True, has_photo=True),
        storage=retention.usage(conn),
        cleared=cleared,
        confirmed=confirmed,
        unchecked=db.count_unchecked_false_triggers(conn),
        ft_cards=True,
    )


@app.get("/false-triggers/cards", response_class=HTMLResponse)
def false_trigger_cards(request: Request, ids: str = "", conn=Depends(get_conn)):
    """Fresh cards for visits opened in the pop-up: ones that stopped being false triggers are left out."""
    wanted = [int(i) for i in ids.split(",") if i.strip().isdigit()][:200]
    resp = page(request, "_cards.html", visits=db.false_trigger_cards(conn, wanted), ft_cards=True)
    resp.headers["X-Unchecked"] = str(db.count_unchecked_false_triggers(conn))
    return resp


def _ft_more(offset: int, show_all: bool) -> str:
    return f"/false-triggers/more?offset={offset}" + ("&show=all" if show_all else "")


@app.get("/false-triggers/more", response_class=HTMLResponse)
def more_false_triggers(request: Request, offset: int = 0, show: str = "", conn=Depends(get_conn)):
    show_all = show == "all"
    visits = db.list_visits(conn, false_triggers=True, has_photo=True, unchecked=not show_all, limit=FT_PAGE,
                            offset=offset)
    next_url = _ft_more(offset + FT_PAGE, show_all) if len(visits) == FT_PAGE else None
    return page(request, "_cards.html", visits=visits, next_url=next_url, ft_cards=True)


@app.post("/false-triggers/clean")
def clean_false_triggers(conn=Depends(get_conn)):
    return RedirectResponse(f"/false-triggers?cleared={retention.clear(conn)}", status_code=303)


@app.get("/settings", response_class=HTMLResponse)
def settings_page(request: Request, pruned: int | None = None,
                  alerts_saved: bool = False, alerts_test: str = "", zones_saved: bool = False, presence_saved: bool = False,
                  conn=Depends(get_conn)):
    storage = retention.usage(conn)
    limit = storage["limit_bytes"] / 1e6
    return page(
        request,
        "settings.html",
        pruned=pruned,
        presence_saved=presence_saved,
        presence_mode=presence.mode(conn),
        presence_report=db.presence_report(conn),
        alerts_saved=alerts_saved,
        alerts_test=alerts_test,
        alerts=alert.get_settings(conn),
        storage=storage,
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


@app.post("/settings/storage")
def save_storage(limit_mb: float = Form(...), photo: str = Form(...), conn=Depends(get_conn)):
    try:
        pruned = retention.save_options(conn, limit_mb, photo)
    except ValueError:
        raise HTTPException(400, "Unknown storage setting")
    return RedirectResponse(f"/settings?pruned={pruned}#storage", status_code=303)


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
def latest_visit(conn=Depends(get_conn)):
    """Newest real visitor (not false triggers), plus the newest visit of any kind (`any_id`).

    Pages poll this to flag new arrivals in the tab and to pull in new cards, numbers and arrows."""
    row = db.latest_visit(conn)
    newest = {"id": row["id"], "label": row["label"], "captured_at": row["captured_at"]} if row else \
        {"id": 0, "label": None, "captured_at": None}
    return {**newest, "any_id": db.newest_id(conn)}


@app.get("/manifest.webmanifest", include_in_schema=False)
def manifest():
    """Web app manifest: lets Android use a maskable icon (a full-bleed square it can crop to a circle) when
    the site is added to the home screen. JSON isn't rewritten by the base_path middleware, so add BASE here."""
    st = f"{BASE}/static"
    v = templates.env.globals["asset_version"]
    icons = [{"src": f"{st}/icon-{n}.png?v={v}", "sizes": f"{n}x{n}", "type": "image/png", "purpose": "any"}
             for n in (192, 512)]
    icons += [{"src": f"{st}/icon-maskable-{n}.png?v={v}", "sizes": f"{n}x{n}", "type": "image/png",
               "purpose": "maskable"} for n in (192, 512)]
    data = {"name": "Birdbrain", "short_name": "Birdbrain", "start_url": f"{BASE}/", "scope": f"{BASE}/",
            "display": "standalone", "background_color": "#f6f5f0", "theme_color": "#f6f5f0", "icons": icons}
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


def frame_response() -> Response:
    """The cached frame plus its age, so the page can label stale frames honestly."""
    age = max(0.0, time.time() - _live["at"])
    return Response(
        _live["jpeg"],
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store", "X-Frame-Age": f"{age:.1f}"},
    )


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
