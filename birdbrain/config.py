import os
from pathlib import Path
from urllib.parse import quote

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
if not os.environ.get("BIRDBRAIN_NO_DOTENV"):  # tests set this to stay hermetic
    load_dotenv(ROOT / ".env")


def _numbers(name: str, raw: str, counts: tuple[int, ...], shape: str) -> tuple[float, ...]:
    try:
        values = tuple(float(v) for v in raw.split(","))
    except ValueError:
        raise ValueError(f"{name}: {raw!r} is not a list of numbers; expected {shape}") from None
    if len(values) not in counts:
        raise ValueError(f"{name}: expected {shape}, got {len(values)} numbers")
    return values


def parse_box(raw: str, name: str = "box") -> tuple[float, float, float, float] | None:
    """A region as fractions of the frame: "x,y,w,h", e.g. "0.25,0.3,0.5,0.6"."""
    raw = raw.strip()
    return _numbers(name, raw, (4,), "x,y,w,h") if raw else None


RING_FORM = "cx,cy,outer_rx,outer_ry,inner_rx,inner_ry[,inner_cy[,inner_cx[,angle[,inner_angle]]]]"


def parse_ring(raw: str, name: str = "ring") -> tuple[float, ...] | None:
    """An elliptical ring as fractions of the frame: the rim's ellipse and the water's inside it.

    Seen from above at an angle the water's ellipse sits higher than the rim's, hence the optional inner_cy; inner_cx
    lets it sit off to one side, and the angles (degrees, clockwise on the picture) turn either ellipse.
    """
    raw = raw.strip()
    return _numbers(name, raw, (6, 7, 8, 9, 10), RING_FORM) if raw else None


def ring_ellipses(ring) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """The rim's and the water's ellipses of a ring, each (cx, cy, rx, ry, angle in degrees)."""
    cx, cy, orx, ory, irx, iry = ring[:6]
    icy = ring[6] if len(ring) > 6 else cy
    icx = ring[7] if len(ring) > 7 else cx
    angle = ring[8] if len(ring) > 8 else 0.0
    iangle = ring[9] if len(ring) > 9 else angle
    return (cx, cy, orx, ory, angle), (icx, icy, irx, iry, iangle)


def make_ring(outer, inner) -> tuple[float, ...]:
    """A ring from its two ellipses (cx, cy, rx, ry[, angle]), in the shortest form that holds them."""
    cx, cy, orx, ory, angle = ellipse5(outer)
    icx, icy, irx, iry, iangle = ellipse5(inner)
    ring = [cx, cy, orx, ory, irx, iry, icy, icx, angle, iangle]
    for default in (angle, 0, cx):  # drop trailing values that equal what they default to
        if ring[-1] != default:
            break
        ring.pop()
    return tuple(ring)


def ellipse5(e) -> tuple[float, ...]:
    """A spot or ellipse as (cx, cy, rx, ry, angle): ones saved before rotation have no angle."""
    return (*e[:4], e[4] if len(e) > 4 else 0.0)


def parse_spots(raw: str, name: str = "spots") -> tuple[tuple[float, ...], ...]:
    """Extra watched ellipses: "cx,cy,rx,ry[,angle];cx,cy,rx,ry"."""
    return tuple(_numbers(name, s.strip(), (4, 5), "cx,cy,rx,ry[,angle]") for s in raw.split(";") if s.strip())


def draw_ellipse(mask, e, value, thickness: int = -1, scale=(1.0, 1.0)) -> None:
    """cv2.ellipse for a frame-fraction ellipse (cx, cy, rx, ry[, angle]) on an image of any size."""
    import cv2  # only the watcher and calibration need OpenCV, not the web process
    H, W = mask.shape[:2]
    cx, cy, rx, ry, angle = ellipse5(e)
    cv2.ellipse(mask, (int(cx * W), int(cy * H)), (int(rx * W * scale[0]), int(ry * H * scale[1])), angle, 0, 360,
                value, thickness)


def parse_polygon(raw: str, name: str = "polygon") -> tuple[tuple[float, float], ...]:
    """A watched area as corner points, fractions of the frame: "x,y;x,y;x,y;..." (at least 3)."""
    points = tuple(_numbers(name, p.strip(), (2,), "x,y") for p in raw.split(";") if p.strip())
    if points and len(points) < 3:
        raise ValueError(f"{name}: a polygon needs at least 3 points, got {len(points)}")
    return points


def polygon_box(points) -> tuple[float, float, float, float] | None:
    """The x,y,w,h box around a polygon."""
    if not points:
        return None
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    return min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


DATA_DIR = Path(_env("BIRDBRAIN_DATA", str(ROOT / "data")))
DB_PATH = DATA_DIR / "birdbrain.db"
IMAGES_DIR = DATA_DIR / "images"

CAM_HOST = _env("CAM_HOST", "birdbrain-cam.home")
RTSP_PORT = int(_env("RTSP_PORT", "554"))
RTSP_USER = _env("RTSP_USER", "thingino")
RTSP_PASS = _env("RTSP_PASS")
# Thingino web API key (/etc/thingino-api.key on the camera) for Settings > Camera; unset hides it.
CAM_API_KEY = _env("CAM_API_KEY")
# Stream paths differ by camera; these are Thingino's. Main = full resolution for ID shots, sub = small for motion.
RTSP_MAIN = _env("RTSP_MAIN_PATH", "ch0")
RTSP_SUB = _env("RTSP_SUB_PATH", "ch1")

# Where to look for motion (substream) and what to send to Claude (main stream). Unset = whole frame.
# MOTION_RING (the birdbath rim minus the water) takes precedence over MOTION_ROI.
MOTION_RING = parse_ring(_env("MOTION_RING"), "MOTION_RING")
MOTION_SPOTS = parse_spots(_env("MOTION_SPOTS"), "MOTION_SPOTS")  # rocks birds land on inside the water
MOTION_ROI = parse_box(_env("MOTION_ROI"), "MOTION_ROI")
CROP_ROI = parse_box(_env("CROP_ROI"), "CROP_ROI")
MOTION_MIN_AREA = float(_env("MOTION_MIN_AREA", "0.004"))  # fraction of the watched area that must change
COOLDOWN_VISIT = int(_env("COOLDOWN_VISIT", "60"))
COOLDOWN_NONE = int(_env("COOLDOWN_NONE", "20"))
MAX_CALLS_PER_DAY = int(_env("MAX_CALLS_PER_DAY", "500"))
LIVE_REFRESH = int(_env("LIVE_REFRESH", "5"))  # seconds between live snapshots on the home page
# Defaults for false-trigger storage; the Settings page overrides both (retention.limit_mb, retention.photo_size).
# Disk space for false-trigger photos; the oldest are deleted beyond this (0 = no limit). Ones you corrected or
# left a note on are always kept, since Claude learns from them.
FALSE_TRIGGER_MAX_MB = float(_env("FALSE_TRIGGER_MAX_MB", "500"))
FALSE_TRIGGER_PHOTO = _env("FALSE_TRIGGER_PHOTO", "medium").strip().lower()  # small, medium, large or full

# Repeated false triggers (rain, wind) double the wait each time, up to this many seconds.
COOLDOWN_NONE_MAX = int(_env("COOLDOWN_NONE_MAX", "300"))
# After this many false triggers in a row, every trigger is a probe (local check first) until a visitor
# shows up, with one full Claude check at most every FULL_CHECK_EVERY seconds in case the local check misses.
PROBE_AFTER = int(_env("PROBE_AFTER", "3"))
FULL_CHECK_EVERY = int(_env("FULL_CHECK_EVERY", "1800"))

# One visit per stay: after a real visit, new triggers wait until the visitor has left (birdbrain/stay.py).
STAY = _env("STAY", "1").strip() not in ("0", "false", "off", "no")
STAY_GAP = int(_env("STAY_GAP", "30"))          # seconds with nothing visitor-sized in view: it left
STAY_MAX = int(_env("STAY_MAX", "900"))         # close the visit after this long regardless
STAY_STILL = int(_env("STAY_STILL", "60"))      # changes that haven't moved for this long are scenery (leaves)
STAY_DIFF = int(_env("STAY_DIFF", "30"))        # brightness change versus the empty scene
STAY_MIN_BLOB = float(_env("STAY_MIN_BLOB", "0.0015"))  # fraction of the frame: about a bushtit on the far rim
STAY_LIGHT = float(_env("STAY_LIGHT", "0.4"))   # more of the watched area changed than this: the light changed
STAY_REARM = int(_env("STAY_REARM", "10"))      # seconds after a stay ends before the next trigger

TRIGGER_FRAMES = 3  # consecutive substream frames of rim motion needed to count as a visit
BURST_FRAMES = 6    # main-stream frames captured per visit, at BURST_FPS; the sharpest goes to Claude
BURST_FPS = 4


def false_trigger_wait(streak: int) -> int:
    """Seconds to wait after the streak-th false trigger in a row: COOLDOWN_NONE, doubling up to COOLDOWN_NONE_MAX."""
    return min(COOLDOWN_NONE * 2 ** (max(streak, 1) - 1), COOLDOWN_NONE_MAX)

# Open-water zone (inside MOTION_RING): triggers only on one solid, bird-sized blob that persists, never on ripples.
WATER_WATCH = _env("WATER_WATCH", "1").strip() not in ("0", "false", "off", "no")
WATER_DIFF = int(_env("WATER_DIFF", "40"))                 # stricter than the rim's 25
WATER_MIN_BLOB = float(_env("WATER_MIN_BLOB", "0.01"))     # blob size as a fraction of the water area
WATER_MAX_BLOB = float(_env("WATER_MAX_BLOB", "0.35"))
WATER_MIN_SOLIDITY = float(_env("WATER_MIN_SOLIDITY", "0.6"))  # area / convex hull: compact shapes only
# Consecutive substream frames (~0.6 s). Raindrop impact rings persisted up to 6 frames in heavy rain.
WATER_PERSIST_FRAMES = int(_env("WATER_PERSIST_FRAMES", "10"))

# Ground zone: critters passing by on the ground around the bath (the bath itself stays the rim's and water's).
# Triggers on one animal-sized blob that persists; by day the local detector looks first and empty frames are
# dropped, at night (IR) the blob must be cat-sized and Claude is asked, under its own daily limit.
GROUND_ZONE = parse_polygon(_env("GROUND_ZONE"), "GROUND_ZONE")   # unset = no ground watching
GROUND_CROP = parse_box(_env("GROUND_CROP"), "GROUND_CROP") or polygon_box(GROUND_ZONE)
GROUND_DIFF = int(_env("GROUND_DIFF", "20"))  # night lawn is dim, so animals there are low contrast
GROUND_MIN_BLOB = float(_env("GROUND_MIN_BLOB", "0.003"))              # fraction of the frame: a ground bird
GROUND_MIN_BLOB_NIGHT = float(_env("GROUND_MIN_BLOB_NIGHT", "0.006"))  # a cat or possum, even on the far lawn
GROUND_MAX_BLOB = float(_env("GROUND_MAX_BLOB", "0.35"))               # bigger = light changing, not an animal
GROUND_PERSIST_FRAMES = int(_env("GROUND_PERSIST_FRAMES", "8"))        # ~0.5 s on a 15 fps substream
GROUND_MAX_STEP = float(_env("GROUND_MAX_STEP", "0.08"))               # per frame, fraction of the frame width
GROUND_BG_RATE = float(_env("GROUND_BG_RATE", "0.02"))  # background update per frame; slow so walkers don't fade
GROUND_MAX_CALLS_PER_DAY = int(_env("GROUND_MAX_CALLS_PER_DAY", "100"))  # separate from MAX_CALLS_PER_DAY

# Local detector before Claude: "off", "shadow" (run and record it, Claude still sees everything),
# or "gate" (frames where it finds nothing skip Claude, except at night).
DETECTOR = _env("DETECTOR", "shadow").strip().lower()
if DETECTOR not in ("off", "shadow", "gate"):
    raise ValueError(f"DETECTOR must be off, shadow or gate, not {DETECTOR!r}")
DETECTOR_MODEL = Path(_env("DETECTOR_MODEL", str(ROOT / "models" / "yolox_tiny.onnx")))
DETECTOR_MIN_SCORE = float(_env("DETECTOR_MIN_SCORE", "0.15"))  # low on purpose: missing a bird costs more than a call

# Local species classifier (birdbrain/classifier.py): "shadow" records its guess with each visit for the Stats
# scorecard (Claude still decides every label), "off" skips it. The model downloads into DATA_DIR on first use.
CLASSIFIER = _env("CLASSIFIER", "shadow").strip().lower()
if CLASSIFIER not in ("off", "shadow"):
    raise ValueError(f"CLASSIFIER must be off or shadow, not {CLASSIFIER!r}")
PRESENCE = _env("PRESENCE", "shadow").strip().lower()  # local animal check: off, shadow or gate (Settings wins)
if PRESENCE not in ("off", "shadow", "gate"):
    raise ValueError(f"PRESENCE must be off, shadow or gate, not {PRESENCE!r}")
CLASSIFIER_MODEL = Path(_env("CLASSIFIER_MODEL")) if _env("CLASSIFIER_MODEL").strip() else None


REPO_URL = _env("REPO_URL", "https://github.com/jurassic73/birdbrain").strip()  # source code link on the About page; blank hides it
SITE_URL = _env("SITE_URL").rstrip("/")  # e.g. http://192.168.1.205:8000; used in ntfy alert links
# iNaturalist place for local seasonality on species cards (e.g. a county); unset hides that chart.
INAT_PLACE_ID = _env("INAT_PLACE_ID").strip()
# Where the bath is, for local sunrise and sunset (Settings > Zones); city-level precision is plenty
SITE_LAT = float(_env("SITE_LAT")) if _env("SITE_LAT").strip() else None
SITE_LON = float(_env("SITE_LON")) if _env("SITE_LON").strip() else None
XC_API_KEY = _env("XC_API_KEY").strip()  # xeno-canto, for bird sounds on species cards; unset hides them


def rtsp_url(path: str | None = None) -> str:
    """URL for a camera stream; defaults to the main stream. Credentials are URL-encoded."""
    path = (path or RTSP_MAIN).lstrip("/")
    return f"rtsp://{quote(RTSP_USER, safe='')}:{quote(RTSP_PASS, safe='')}@{CAM_HOST}:{RTSP_PORT}/{path}"
