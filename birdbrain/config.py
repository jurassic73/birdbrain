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
# The photo budget (birdbrain.retention): every stored photo, thumbnail and false-trigger burst together. Defaults
# only; the Settings page overrides them. Over the budget, or with less free disk than DISK_FLOOR_MB, space is freed:
# false triggers first, then older real visits are thinned, then the oldest thinned ones keep only their thumbnail.
# Keepers, species photos, corrected or annotated visits, first sightings and rare species are never touched.
PHOTO_BUDGET_MB = float(_env("PHOTO_BUDGET_MB", "10000"))   # 0 = no budget
PHOTO_THIN = _env("PHOTO_THIN", "1").strip() not in ("0", "false", "off", "no")  # thin real visits after PHOTO_FULL_DAYS
PHOTO_FULL_DAYS = int(_env("PHOTO_FULL_DAYS", "90"))       # real visits younger than this stay full size
PHOTO_FULL_PER_DAY = int(_env("PHOTO_FULL_PER_DAY", "2"))  # sharpest photos per species per day that stay full size
PHOTO_THIN_EDGE = int(_env("PHOTO_THIN_EDGE", "1280"))     # long edge of a thinned photo: 960, 1280 or 1920 (= full)
PHOTO_WARN_PCT = float(_env("PHOTO_WARN_PCT", "80"))       # one phone alert when photos reach this % of the budget
RARE_UNDER = int(_env("RARE_UNDER", "10"))                 # every visit of a species seen fewer times stays full size
DISK_FLOOR_MB = float(_env("DISK_FLOOR_MB", "5000"))       # free space to keep on the data disk (0 = don't watch it)

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
GROUP_GAP = int(_env("GROUP_GAP", "8"))  # substream px: changes closer than this count as one animal
# Count in colour: the watcher keeps a colour copy of the empty scene (updated every COLOR_BG_EVERY substream frames,
# fading at the same rate as the grey one) and a change is the channel that changed most. In grey a grey squirrel or
# a grey-blue jay standing on a pale rim shows only as a head and a tail, two "animals". COUNT_COLOR=0 counts in grey.
COUNT_COLOR = _env("COUNT_COLOR", "1").strip() not in ("0", "false", "off", "no")
COLOR_BG_EVERY = max(int(_env("COLOR_BG_EVERY", "3")), 1)

# Circle from the motion (birdbrain/motion_circle.py): where the animal is in a photo the Pi named itself.
CIRCLE_FROM_MOTION = _env("CIRCLE_FROM_MOTION", "1").strip() not in ("0", "false", "off", "no")
CIRCLE_DIFF = int(_env("CIRCLE_DIFF", "25"))  # least brightness change versus the empty view (more on a noisy pair)
CIRCLE_MIN_BLOB = float(_env("CIRCLE_MIN_BLOB", "0.0005"))  # fraction of the frame: smaller patches are ignored
CIRCLE_MAX_R = float(_env("CIRCLE_MAX_R", "0.13"))  # of the width: a bigger patch is the light, not an animal
CIRCLE_AMBIGUOUS = float(_env("CIRCLE_AMBIGUOUS", "0.6"))  # a second patch this big (of the first): no circle

SUB_FPS = int(_env("SUB_FPS", "15"))  # the substream's frame rate; the watcher checks every frame
TRIGGER_FRAMES = 3  # consecutive substream frames of rim motion needed to count as a visit
BURST_FRAMES = 6    # main-stream frames captured per visit, at BURST_FPS; the sharpest goes to Claude
BURST_FPS = 4
# Stored photos get the capture time drawn in their lower-left corner (a dark glass pill). The camera's own
# burn-in is turned off, so this is the only timestamp. 0 turns it off.
PHOTO_STAMP = _env("PHOTO_STAMP", "1") != "0"
# Font for the stamp; the first of these that exists is used, else Pillow's built-in font.
STAMP_FONTS = [f for f in _env("STAMP_FONT").split(",") if f] + [
    "/usr/share/fonts/opentype/urw-base35/URWGothic-Demi.otf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/System/Library/Fonts/Supplemental/Futura.ttc",
]


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

# Finders for the BioCLIP 2 species check's tight look (birdbrain/finder.py), tried in this order until one points
# at the animal: motion (the Motion detector's circle), blob (the crowd counter's outline), yolox (YOLOX-tiny on the
# crop box, by day). FINDERS=motion is the old behaviour.
FINDER_NAMES = ("motion", "blob", "yolox")


def parse_finders(raw: str) -> tuple[str, ...]:
    """A comma list of finder names, in order, each at most once and at least one; an unknown name, an empty list
    or a repeated name raises ValueError naming FINDERS."""
    names = tuple(n.strip().lower() for n in raw.split(",") if n.strip())
    expected = f"expected a comma list of {', '.join(FINDER_NAMES)}"
    if not names:
        raise ValueError(f"FINDERS is empty; {expected} (FINDERS=motion is the old behaviour)")
    for n in names:
        if n not in FINDER_NAMES:
            raise ValueError(f"FINDERS: {n!r} is not a finder; {expected}")
    if len(set(names)) != len(names):
        raise ValueError(f"FINDERS: {raw.strip()!r} lists a finder twice; {expected}, each once")
    return names


FINDERS = parse_finders(_env("FINDERS", "motion,blob,yolox"))
FINDER_YOLOX_MIN_SCORE = float(_env("FINDER_YOLOX_MIN_SCORE", "0.15"))  # YOLOX's least class score for the finder
if not 0 <= FINDER_YOLOX_MIN_SCORE <= 1:
    raise ValueError(f"FINDER_YOLOX_MIN_SCORE must be between 0 and 1, not {FINDER_YOLOX_MIN_SCORE!r}")
# Rain on the water (watch.rain_on_water, gate mode of the Pi animal check only): by day a rim trigger is taken for rain
# or ripples, and Claude isn't asked, when no finder saw anything new on land and the Pi animal check scored the frame
# under this. 0 turns the rule off.
RAIN_SCORE_MAX = float(_env("RAIN_SCORE_MAX", "0.3"))
if not 0 <= RAIN_SCORE_MAX <= 1:
    raise ValueError(f"RAIN_SCORE_MAX must be between 0 and 1, not {RAIN_SCORE_MAX!r}")

# Local species classifier (birdbrain/classifier.py): "shadow" records its guess with each visit for the Stats
# scorecard (Claude still decides every label), "off" skips it. The model downloads into DATA_DIR on first use.
# gate: a confident daytime guess of a species already settled in this yard names the visit without Claude.
CLASSIFIER = _env("CLASSIFIER", "gate").strip().lower()
if CLASSIFIER not in ("off", "shadow", "gate"):
    raise ValueError(f"CLASSIFIER must be off, shadow or gate, not {CLASSIFIER!r}")
# Gate mode's two bars for the top guess (birdbrain/classifier.py, rule H): the first look on the Pi animal check's
# wide view of the bowl, and the tight look around the motion circle made when the first falls short.
GATE_MIN_SCORE_BOWL = float(_env("GATE_MIN_SCORE_BOWL", "0.8"))
GATE_MIN_SCORE_TIGHT = float(_env("GATE_MIN_SCORE_TIGHT", "0.95"))
for _name, _bar in (("GATE_MIN_SCORE_BOWL", GATE_MIN_SCORE_BOWL), ("GATE_MIN_SCORE_TIGHT", GATE_MIN_SCORE_TIGHT)):
    if not 0 < _bar <= 1:
        raise ValueError(f"{_name} must be between 0 and 1, not {_bar!r}")
# Shadow second-frame look (watch.second_frame_look): when the tight look falls short of its bar by score alone with
# its top guess at least this sure, by day, the next sharpest burst frame is embedded on the same circle and the two
# guesses are stored to keep score on Stats. It names nothing. 0 or off: never.
_second = _env("SECOND_FRAME_MIN").strip().lower() or "0.8"  # blank means the default, like the other knobs
SECOND_FRAME_MIN = 0.0 if _second in ("0", "off", "no", "false") else float(_second)
if not 0 <= SECOND_FRAME_MIN <= 1:
    raise ValueError(f"SECOND_FRAME_MIN must be between 0 and 1 (or off), not {_second!r}")
PRESENCE = _env("PRESENCE", "shadow").strip().lower()  # local animal check: off, shadow or gate (Settings wins)
if PRESENCE not in ("off", "shadow", "gate"):
    raise ValueError(f"PRESENCE must be off, shadow or gate, not {PRESENCE!r}")
# The animal check also scores the burst's last frame when the sharpest one looks empty (a landing bird is usually in
# the bowl by then; sharpness can favour ripples or mulch). ~6 s more per empty-looking trigger on a Pi 4.
PRESENCE_LAST_FRAME = _env("PRESENCE_LAST_FRAME", "1").strip() not in ("0", "false", "off", "no")
CLASSIFIER_MODEL = Path(_env("CLASSIFIER_MODEL")) if _env("CLASSIFIER_MODEL").strip() else None
# Folder URL of the BioCLIP 2 model release assets; only used for the first download. Blank: no download.
_release = _env("CLASSIFIER_RELEASE_URL").strip()
CLASSIFIER_RELEASE_URL = _release if not _release or _release.endswith("/") else _release + "/"


# Hailo-8 shadow detector (birdbrain/hailo/, its own service and database; it never changes what the watcher does).
# HAILO_DETECT=off makes the service exit at once; "shadow" runs it. Everything else has a default.
HAILO_DETECT = _env("HAILO_DETECT", "shadow").strip().lower()
if HAILO_DETECT not in ("off", "shadow"):
    raise ValueError(f"HAILO_DETECT must be off or shadow, not {HAILO_DETECT!r}")
HAILO_HEF = Path(_env("HAILO_HEF", "/usr/share/hailo-models/yolov8s_h8.hef"))
# What the detector looks at on each substream frame: "crop" cuts the crop box out first (the same pixels, upscaled to
# the model's input, finds about twice the visitors on the stored photos), "frame" is the whole frame (sees the ground).
HAILO_VIEW = _env("HAILO_VIEW", "crop").strip().lower()
if HAILO_VIEW not in ("crop", "frame"):
    raise ValueError(f"HAILO_VIEW must be crop or frame, not {HAILO_VIEW!r}")
HAILO_MIN_SCORE = float(_env("HAILO_MIN_SCORE", "0.3"))        # the HEFs' own post-process already drops under 0.2
HAILO_EVERY = max(int(_env("HAILO_EVERY", "1")), 1)             # look at every Nth substream frame (1 = all of them)
HAILO_RING_SECONDS = float(_env("HAILO_RING_SECONDS", "3"))      # frames kept before an event, for the arrival shots
HAILO_EVENT_GAP = float(_env("HAILO_EVENT_GAP", "2"))            # seconds unseen before an event ends
HAILO_MIN_HITS = int(_env("HAILO_MIN_HITS", "3"))                # frames a detection must persist to become an event
HAILO_SHADOW_DB = Path(_env("HAILO_SHADOW_DB", str(DATA_DIR / "hailo_shadow.db")))
HAILO_SNAPSHOT_DIR = Path(_env("HAILO_SNAPSHOT_DIR", str(DATA_DIR / "hailo_snapshots")))
HAILO_SNAPSHOT_MAX_MB = float(_env("HAILO_SNAPSHOT_MAX_MB", "500"))  # oldest event snapshots go beyond this
HAILO_HEARTBEAT = Path(_env("HAILO_HEARTBEAT", str(DATA_DIR / "hailo_heartbeat.json")))

SITE_URL = _env("SITE_URL").rstrip("/")  # e.g. http://192.168.1.20:8000; used in ntfy alert links
# iNaturalist place for local seasonality on species cards (e.g. a county); unset hides that chart.
INAT_PLACE_ID = _env("INAT_PLACE_ID").strip()
# Where the bath is, for local sunrise and sunset (Settings > Zones); city-level precision is plenty
SITE_LAT = float(_env("SITE_LAT")) if _env("SITE_LAT").strip() else None
SITE_LON = float(_env("SITE_LON")) if _env("SITE_LON").strip() else None
# Hourly weather from Open-Meteo (no key; birdbrain/weather.py) needs SITE_LAT and SITE_LON; WEATHER=off skips it.
WEATHER = _env("WEATHER", "on").strip().lower() not in ("off", "0", "no", "false")
XC_API_KEY = _env("XC_API_KEY").strip()  # xeno-canto, for bird sounds on species cards; unset hides them


def rtsp_url(path: str | None = None) -> str:
    """URL for a camera stream; defaults to the main stream. Credentials are URL-encoded."""
    path = (path or RTSP_MAIN).lstrip("/")
    return f"rtsp://{quote(RTSP_USER, safe='')}:{quote(RTSP_PASS, safe='')}@{CAM_HOST}:{RTSP_PORT}/{path}"
