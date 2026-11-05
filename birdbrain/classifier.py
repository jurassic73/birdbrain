"""Local bird species classifier. In shadow mode it guesses the species on the Pi and records the guess, while
Claude stays in charge of every label; the Stats page scores the guesses against the final labels (owner
corrections win, else Claude's). In gate mode (rule H, below gate_miss) a sure daytime guess of a species already
settled in the yard names the visit without Claude: a first look on the Pi animal check's wide view, then, when
that falls short, one tight look around the circle from the motion.

Model: BioCLIP 2 (Imageomics Institute, MIT), a CLIP model trained on the Tree of Life (incl. iNaturalist).
Its image encoder is exported to ONNX with 8-bit weights (300 MB, run by onnxruntime, ~6 s per crop on a Pi 4,
~400 MB of RAM); the text side is precomputed: one embedding per eBird species (11k, float16, made on a Mac
with open_clip, "a photo of {scientific} with common name {common}."). All three files are assets of this
repo's GitHub release bioclip2-v1, downloaded on first use into DATA_DIR/models and checked by SHA-256.
On the first 28 real bird visits (2026-10-31) it named 25 right first time; Google Coral's MobileNet V2 (iNat
2017 birds), used before, named 11 (it called most small birds on mulch "background").

The scores are a softmax over every eBird species; outputs are then limited to birds reported in the
configured iNaturalist place (INAT_PLACE_ID; species_counts cached in DATA_DIR for 30 days) and renormalized,
which is the same as asking the model to choose among the local birds only. Names are eBird's.
Mammals too (2026-10-31): every iNaturalist mammal species (6.9k) is in the list as well, limited the same way to
the place's mammals, so squirrels get a species (26 of the first 27 squirrel visits were called a squirrel).

The same image embedding also feeds birdbrain.presence (is anything there at all?). That one is made of
presence_patch, not the crop box: a square around the whole bowl for birdbath triggers, the ground crop
padded to a square for ground ones, since to_input's center crop cut the rim's left and right ends off the
wide crop box (visits 759, 760, 761, 794: birds on the rim settled as empty).
IR night frames are grayscale; it hasn't been checked on them yet.

CLI: python -m birdbrain.classifier photo.jpg            (top 3 for whole photos)
     python -m birdbrain.classifier --backfill [--limit N] [--redo]   (score stored visits)
     python -m birdbrain.classifier --backfill-gate [--limit N] [--redo] [--dry-run]   (first looks, no images)
"""

import argparse
import hashlib
import json
import logging
import re
import sys
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from . import config, db, zones

log = logging.getLogger("birdbrain.classifier")

MODEL_NAME = "bioclip-2"
MODEL_FILES = {  # file: SHA-256
    "bioclip2_int8.onnx": "18cc1f73af2ce3fa3dd34a3b7a104cb2c5d9d45d95f2613cc9624368b514dfae",
    "bioclip2_ebird_text.npy": "a7fe1e93a5bb0994009adc3076fe8521f530c37a85339ced3c5e52e237d6339b",
    "bioclip2_ebird_names.json": "2e346b2105c463bfa119ec6dd98591c8a1f2ab719ea62df417bd574cba205f92",
    "bioclip2_mammals_text.npy": "833e545e0143c277772bdc8eec40263a010473f4d9375f9a8d726f410f19b0bc",
    "bioclip2_mammals_names.json": "ff644196404319ed571b43e7fe0479f30d3ee0f936d338eacff405e1659973a6",
}
TAXA = {"Aves": "birds", "Mammalia": "mammals"}  # iNaturalist iconic taxon: cache file word
INPUT = 224
MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)  # CLIP's normalization
STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)
LOGIT_SCALE = 100.0
THREADS = 3  # of the Pi's 4 cores; the watcher runs at Nice=10, below the ADS-B decoder
TOP = 3
# A square around Claude's circle, this many radii from the center to each side (1.0: the circle's bounding
# square). On the first 28 real bird visits BioCLIP 2 named 25 right with 1.0 or 1.5, and averaging 1.5 and 3
# didn't help; each extra square costs another ~6 s on the Pi. (MobileNet needed 1.5 + 3 to see a bird at all.)
# 2.0 since 2026-11-02: Claude's circles are often too small (a junco's became a 96 px square cutting the bird,
# Junco 0.29); on 105 daytime visits 2.0 named 96 right vs 94 at 1.0, and would gate 72 vs 61.
CIRCLE_MARGINS = (2.0,)
CIRCLE_MARGIN = CIRCLE_MARGINS[0]
MIN_SIDE = 96  # px; never classify a smaller patch than this (it's upscaled to 224 anyway)

SPECIES_URL = "https://api.inaturalist.org/v1/observations/species_counts"
PLACE_MAX_AGE = timedelta(days=30)
RETRY_AFTER = 3600  # seconds before trying a failed download (model or bird list) again
_place_failed_at = 0.0


def model_dir() -> Path:
    return config.CLASSIFIER_MODEL or config.DATA_DIR / "models"


def download_model(folder: Path) -> None:
    """Fetch the model files that are missing and check them; a bad download is never kept. Streamed to
    disk: the image encoder is 300 MB."""
    import httpx

    from .species_info import HEADERS

    folder.mkdir(parents=True, exist_ok=True)
    base = config.CLASSIFIER_RELEASE_URL
    if not base and any(not (folder / n).exists() for n in MODEL_FILES):
        raise RuntimeError(f"classifier model files are missing from {folder} and CLASSIFIER_RELEASE_URL is not "
                           "set in .env, so they can't be downloaded; set it or copy the files there")
    for name, sha in MODEL_FILES.items():
        path = folder / name
        if path.exists():
            continue
        tmp, digest = path.with_suffix(".part"), hashlib.sha256()
        with httpx.stream("GET", base + name, headers=HEADERS, timeout=60, follow_redirects=True) as r:
            r.raise_for_status()
            with tmp.open("wb") as f:
                for chunk in r.iter_bytes(1 << 20):
                    digest.update(chunk)
                    f.write(chunk)
        if digest.hexdigest() != sha:
            tmp.unlink()
            raise RuntimeError(f"downloaded {name} doesn't match its checksum; not using it")
        tmp.replace(path)


def _key(name: str) -> str:
    return re.sub(r"[^a-z]", "", (name or "").lower())


# --- local species list --------------------------------------------------------------------------------

def local_species(fetch: bool = True) -> dict[str, str] | None:
    """The place's birds and mammals together; None (no limit) when neither list can be had."""
    lists = [place_species(fetch=fetch, taxon=t) for t in TAXA]
    if all(lst is None for lst in lists):
        return None
    return {k: v for lst in lists if lst for k, v in lst.items()}


def place_species(place_id: str | None = None, fetch: bool = True, taxon: str = "Aves") -> dict[str, str] | None:
    """{scientific name: common name} of every species of the taxon (Aves or Mammalia) reported in the
    iNaturalist place, cached in DATA_DIR for PLACE_MAX_AGE. None without a place or when it was never
    fetched; a failed refresh keeps the old list."""
    place_id = (place_id if place_id is not None else config.INAT_PLACE_ID).strip()
    if not place_id:
        return None
    path = config.DATA_DIR / f"inat_{TAXA[taxon]}_{place_id}.json"
    cached = None
    if path.exists():
        try:
            cached = json.loads(path.read_text())
        except ValueError:
            cached = None
    global _place_failed_at
    fresh = cached and datetime.fromisoformat(cached["fetched"]) > datetime.now() - PLACE_MAX_AGE
    if not fresh and fetch and not (_place_failed_at and time.monotonic() - _place_failed_at < RETRY_AFTER):
        try:
            species = fetch_place_species(place_id, taxon=taxon)
            if species:
                cached = {"fetched": datetime.now().isoformat(timespec="seconds"), "species": species}
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(cached))
        except Exception:
            _place_failed_at = time.monotonic()  # offline: don't hold up every visit retrying
            log.warning("couldn't fetch the place's %s list from iNaturalist; %s", TAXA[taxon],
                        "using the cached one" if cached else "not limiting guesses")
    if not cached:
        return None
    return {s["name"]: s.get("common") or "" for s in cached["species"]}


def fetch_place_species(place_id: str, get=None, taxon: str = "Aves") -> list[dict]:
    """Every page of iNaturalist's species_counts for the place's birds or mammals (verifiable observations)."""
    import httpx

    from .species_info import HEADERS

    get = get or (lambda params: httpx.get(SPECIES_URL, params=params, headers=HEADERS, timeout=30).json())
    species, page = [], 1
    while True:
        data = get({"place_id": place_id, "iconic_taxa": taxon, "verifiable": "true", "per_page": 500,
                    "page": page})
        results = data.get("results") or []
        for r in results:
            t = r.get("taxon") or {}
            if t.get("name") and t.get("rank") in (None, "species", "subspecies"):
                species.append({"name": " ".join(t["name"].split()[:2]), "common": t.get("preferred_common_name"),
                                "count": r.get("count", 0)})
        if not results or page * int(data.get("per_page") or 500) >= int(data.get("total_results") or 0):
            return species
        page += 1
        time.sleep(1)  # iNaturalist asks for about one request a second


# --- scoring --------------------------------------------------------------------------------------------

def group_species(probs: np.ndarray, labels: list[tuple[str, str, bool]]) -> tuple[dict[str, float], dict[str, str], float]:
    """Per-species probability (subspecies summed into their species), each species' common name (the
    species' own label wins over a subspecies'), and the background probability."""
    scores: dict[str, float] = {}
    commons: dict[str, str] = {}
    background = 0.0
    for p, (sci, common, own) in zip(probs, labels):
        if sci == "background":
            background += float(p)
            continue
        scores[sci] = scores.get(sci, 0.0) + float(p)
        if own or sci not in commons:
            commons[sci] = common
    return scores, commons, background


def local_mask(scores: dict[str, float], commons: dict[str, str], local: dict[str, str] | None):
    """The species allowed here: matched on the scientific name, else on the common name (genera get
    renamed: the 2017 labels say Picoides pubescens, iNaturalist now Dryobates). Returns {species: local
    common name or None}; None (no limit) without a local list."""
    if local is None:
        return None
    by_common = {_key(c): c for c in local.values() if c}
    allowed = {}
    for sci in scores:
        if sci in local:
            allowed[sci] = local[sci] or None
        elif _key(commons.get(sci, "")) in by_common:
            allowed[sci] = by_common[_key(commons[sci])]
    return allowed


def ebird_name(*candidates: str | None) -> str | None:
    """The first candidate that is an eBird name (official spelling), else the first one given."""
    from . import names

    given = [c for c in candidates if c]
    # Only once the eBird list is on disk (the site fetches it): the watcher never waits on that download.
    if names._names is not None or (config.DATA_DIR / "ebird_names.json").exists():
        for c in given:
            if n := names.canonical(c):
                return n
    return given[0] if given else None


def rank(probs: np.ndarray, labels: list[tuple[str, str, bool]], local: dict[str, str] | None, top: int = TOP) -> dict:
    """Top guesses from the raw outputs, limited to the local species and renormalized among them."""
    scores, commons, background = group_species(probs, labels)
    allowed = local_mask(scores, commons, local)
    pool = scores if allowed is None else {s: scores[s] for s in allowed}
    total = sum(pool.values())
    best = sorted(pool.items(), key=lambda kv: kv[1], reverse=True)[:top]
    guesses = []
    for sci, p in best:
        if p <= 0:
            break
        local_common = allowed.get(sci) if allowed else None
        guesses.append({"name": ebird_name(local_common, commons.get(sci)) or sci, "scientific": sci,
                        "score": round(p / total, 3) if total else 0.0})
    return {
        "top": guesses,
        # How much of the model's belief went to local species: low means it saw something it doesn't know.
        "local_share": round(total / max(sum(scores.values()), 1e-9), 3) if allowed is not None else None,
        "background": round(background, 3),
        "masked": allowed is not None,
        "species": len(pool),
    }


# --- cropping -------------------------------------------------------------------------------------------

# The Pi animal check's view (birdbrain.presence). Version 1 (stored as NULL) was the crop box center-cut to a
# square by to_input, which lost the rim's left and right ends; version 2 is presence_patch. Only embeddings of the
# current view train or are scored, so a change of view starts training afresh (backfill --redo re-embeds).
PRESENCE_VIEW = 2
# Of the frame's height, kept in the bowl square above the rim's top and beyond its left and right ends: a perched
# bird's body. Checked on real frames (2026-11-04): the junco of visit 759 hung its tail ~0.04 of the width left of
# the rim and the nuthatch of 760 its head ~0.03 right of it, outside a square with the band above the rim only.
BIRD_BAND = 0.06
VIEW_MARGIN = 0.04  # extra space around the bowl, as a fraction of the square's side, on each side
PAD_FILL = tuple(int(round(v)) for v in MEAN[::-1] * 255)  # BGR; CLIP's mean color, so padding normalizes to ~0


def _zone_dict(z) -> dict:
    """A zones.Zones or a dict like Zones.to_dict() (raw_json.zones) as a dict; {} for anything else."""
    if isinstance(z, zones.Zones):
        return z.to_dict()
    return z if isinstance(z, dict) else {}


MIN_BOX = 0.01  # a box narrower or shorter than this (of the frame) inside the frame is no box: the whole frame


def usable_box(v) -> tuple[float, float, float, float] | None:
    """An x, y, w, h box of frame fractions as the part of it inside the frame, or None when it can't be one:
    not 4 finite numbers, or less than MIN_BOX of it in the frame (off the frame, zero size). view_kind,
    presence_view_box and presence_patch all go by it, so they agree on what the Pi animal check sees."""
    try:
        x, y, w, h = (float(t) for t in v)
    except (TypeError, ValueError):
        return None
    if not all(np.isfinite((x, y, w, h))):
        return None
    x0, y0, x1, y1 = max(x, 0.0), max(y, 0.0), min(x + w, 1.0), min(y + h, 1.0)
    if x1 - x0 < MIN_BOX or y1 - y0 < MIN_BOX:
        return None
    if (x0, y0, x1, y1) == (x, y, x + w, y + h):
        return x, y, w, h
    return round(x0, 4), round(y0, 4), round(x1 - x0, 4), round(y1 - y0, 4)


def usable_ring(v) -> tuple[float, ...] | None:
    """A ring (config.ring_ellipses' 6 to 10 numbers) that can be drawn: finite numbers, the rim's center on the
    frame and its radii above 0. Anything else (too short, None or NaN in it) counts as no ring everywhere."""
    try:
        ring = tuple(float(t) for t in v)
    except (TypeError, ValueError):
        return None
    if not 6 <= len(ring) <= 10 or not all(np.isfinite(ring)):
        return None
    cx, cy, rx, ry = ring[:4]
    return ring if 0 <= cx <= 1 and 0 <= cy <= 1 and rx > 0 and ry > 0 else None


def _bowl_square(W: float, H: float, ring) -> tuple[float, float, float]:
    """(left, top, side) of the bowl square on a W x H frame, unrounded (see presence_square); `ring` is usable."""
    cx, cy, rx, ry, angle = config.ring_ellipses(ring)[0]
    a, b, t = rx * W, ry * H, np.radians(angle)
    hx = float(np.hypot(a * np.cos(t), b * np.sin(t)))  # half the rotated ellipse's bounding box
    hy = float(np.hypot(a * np.sin(t), b * np.cos(t)))
    x0, x1 = cx * W - hx - BIRD_BAND * H, cx * W + hx + BIRD_BAND * H
    y0, y1 = cy * H - hy - BIRD_BAND * H, cy * H + hy
    side = min(max(x1 - x0, y1 - y0) / (1 - 2 * VIEW_MARGIN), W, H)
    left = min(max((x0 + x1) / 2 - side / 2, 0.0), W - side)
    top = min(max((y0 + y1) / 2 - side / 2, 0.0), H - side)
    return left, top, side


def view_kind(z, trigger: str | None) -> str:
    """Which view presence_patch takes for these zones and trigger: "bowl", "padded" or "frame"."""
    d = _zone_dict(z)
    if trigger == "ground":
        return "padded" if usable_box(d.get("ground_crop")) else "frame"
    if usable_ring(d.get("ring")):
        return "bowl"
    return "padded" if usable_box(d.get("crop")) else "frame"


def _padded_box(d: dict, trigger: str | None):
    return usable_box(d.get("ground_crop") if trigger == "ground" else d.get("crop"))


def presence_square(shape, z) -> tuple[int, int, int] | None:
    """The square (left, top, side in pixels) the Pi animal check sees on a birdbath trigger: the whole outer rim
    (a turned one by its bounding box) plus BIRD_BAND of the frame's height above it and beyond its left and right
    ends, VIEW_MARGIN more on each side, centered on that, slid inside the frame and at most the frame's height.
    None without a (usable) ring."""
    ring = usable_ring(_zone_dict(z).get("ring"))
    if ring is None:
        return None
    H, W = shape[:2]
    left, top, side = _bowl_square(W, H, ring)
    side_px = min(int(round(side)), H, W)
    x0 = min(max(int(round(left)), 0), W - side_px)
    y0 = min(max(int(round(top)), 0), H - side_px)
    return x0, y0, side_px


def pad_square(patch: np.ndarray) -> np.ndarray:
    """The patch unchanged in the middle of a square of side max(w, h), the rest PAD_FILL."""
    h, w = patch.shape[:2]
    side = max(h, w)
    if h == w:
        return patch
    out = np.empty((side, side, *patch.shape[2:]), patch.dtype)
    out[...] = PAD_FILL if patch.ndim == 3 and patch.shape[2] == 3 else int(np.mean(PAD_FILL))
    y, x = (side - h) // 2, (side - w) // 2
    out[y:y + h, x:x + w] = patch
    return out


def presence_patch(img: np.ndarray, z, trigger: str | None) -> tuple[np.ndarray, str]:
    """The square patch the Pi animal check embeds, and its kind: "bowl" (a birdbath trigger with a ring: the
    presence_square), "padded" (a ground trigger's ground crop, or a birdbath trigger's crop box when there is
    no ring, padded to a square) or "frame" (no usable box: the whole frame padded). Square, so to_input's
    center crop keeps all of it. Geometry is in fractions, so any size of the same frame gives the same view."""
    kind = view_kind(z, trigger)
    if kind == "bowl":
        x0, y0, side = presence_square(img.shape, z)
        return img[y0:y0 + side, x0:x0 + side], kind
    if kind == "padded":
        if px := zones.pixel_box(img.shape, _padded_box(_zone_dict(z), trigger)):
            x0, y0, x1, y1 = px
            return pad_square(img[y0:y1, x0:x1]), kind
    return pad_square(img), "frame"


def presence_view_box(z, trigger: str | None, aspect: float = 16 / 9) -> tuple[float, float, float, float] | None:
    """The Pi animal check's view as frame fractions (x, y, w, h), for drawing it and judging circles: the bowl
    square (w and h differ as fractions on a wide frame), or for "padded" the box itself, the part inside the
    frame (padding adds nothing to see). None for "frame"."""
    kind = view_kind(z, trigger)
    d = _zone_dict(z)
    if kind == "bowl":
        W, H = float(aspect), 1.0
        left, top, side = _bowl_square(W, H, usable_ring(d["ring"]))
        return (round(left / W, 4), round(top / H, 4), round(side / W, 4), round(side / H, 4))
    if kind == "padded":
        return _padded_box(d, trigger)
    return None


def old_view_box(z, trigger: str | None, aspect: float = 16 / 9) -> tuple[float, float, float, float]:
    """What the Pi animal check saw before PRESENCE_VIEW 2, for embeddings still of that view: the middle square of
    the crop box (ground triggers: the ground crop; no box: the whole frame) that to_input's center cut kept, as
    frame fractions."""
    box = _padded_box(_zone_dict(z), trigger) or (0.0, 0.0, 1.0, 1.0)
    x, y, w, h = box
    side = min(w * aspect, h)  # in frame heights
    cx, cy = x + w / 2, y + h / 2
    return (round(cx - side / aspect / 2, 4), round(cy - side / 2, 4), round(side / aspect, 4), round(side, 4))


def crop_for(img: np.ndarray, circle: dict | None, box, margin: float = CIRCLE_MARGIN) -> tuple[np.ndarray, str]:
    """A square around the circle (fractions of the frame; r is a fraction of the width) when there is one,
    else the crop box. Returns the patch and which crop was used."""
    H, W = img.shape[:2]
    if circle and circle.get("r"):
        x0, y0, side = circle_square(circle, W, H, margin)
        return img[y0:y0 + side, x0:x0 + side], "circle"
    if box and (px := zones.pixel_box(img.shape, box)):
        x0, y0, x1, y1 = px
        return img[y0:y1, x0:x1], "box"
    return img, "frame"


def circle_square(circle: dict, W: int, H: int, margin: float = CIRCLE_MARGIN) -> tuple[int, int, int]:
    """The square (left, top, side in pixels) `margin` radii from the circle's center to each side, at least
    MIN_SIDE, slid inside the frame when the circle is near an edge."""
    cx, cy = circle["x"] * W, circle["y"] * H
    half = max(circle["r"] * W * margin, MIN_SIDE / 2)
    half = min(half, min(W, H) / 2)
    x0 = int(round(min(max(cx - half, 0), W - 2 * half)))
    y0 = int(round(min(max(cy - half, 0), H - 2 * half)))
    return x0, y0, int(round(2 * half))


def to_input(bgr: np.ndarray) -> np.ndarray:
    """BGR patch to the model's 1 x 3 x 224 x 224 input, like open_clip: shorter side to 224 (bicubic),
    center crop, CLIP normalization."""
    import cv2

    h, w = bgr.shape[:2]
    k = INPUT / min(h, w)
    im = cv2.resize(bgr, (max(INPUT, round(w * k)), max(INPUT, round(h * k))), interpolation=cv2.INTER_CUBIC)
    y, x = (im.shape[0] - INPUT) // 2, (im.shape[1] - INPUT) // 2
    rgb = cv2.cvtColor(im[y:y + INPUT, x:x + INPUT], cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    return ((rgb - MEAN) / STD).transpose(2, 0, 1)[None]


class Classifier:
    def __init__(self, folder: Path):
        import onnxruntime as ort  # imported here so the web app never pays for it

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = THREADS
        self.session = ort.InferenceSession(str(folder / "bioclip2_int8.onnx"), opts)
        texts, self.labels, self.mammals = [], [], set()
        for kind in ("ebird", "mammals"):
            texts.append(np.load(folder / f"bioclip2_{kind}_text.npy").astype(np.float32))
            names = json.loads((folder / f"bioclip2_{kind}_names.json").read_text())
            self.labels += [(sci, common, True) for sci, common in names]
            if kind == "mammals":
                self.mammals = {sci for sci, _ in names}
        self.text = np.concatenate(texts)

    def embed(self, bgr: np.ndarray) -> np.ndarray:
        """The image embedding (normalized, 768 floats)."""
        return self.session.run(None, {"image": to_input(bgr)})[0][0]

    def probs(self, bgr: np.ndarray) -> np.ndarray:
        """Softmax over every eBird bird and iNaturalist mammal (rank() narrows it to the local ones)."""
        z = LOGIT_SCALE * (self.text @ self.embed(bgr))
        e = np.exp(z - z.max())
        return e / e.sum()


_classifier: Classifier | None = None
_failed_at = 0.0
_load_lock = threading.Lock()  # captures and the animal check's background thread may both ask for it first


def load() -> Classifier | None:
    """The classifier, downloading the model the first time. None when it's off; raises when it can't be
    had (and doesn't try to download again for RETRY_AFTER)."""
    if config.CLASSIFIER == "off":
        return None
    loaded = _classifier  # read once: circle_check may let go of it meanwhile
    if loaded is None:
        with _load_lock:
            return _load()
    return loaded


def usable() -> bool:
    """Whether an embedding can be expected right now, without loading anything: on, and the model loaded or at
    least not known to be missing (a failed download isn't tried again for RETRY_AFTER)."""
    if config.CLASSIFIER == "off":
        return False
    return _classifier is not None or not (_failed_at and time.monotonic() - _failed_at < RETRY_AFTER)


def _load() -> Classifier:
    global _classifier, _failed_at
    loaded = _classifier
    if loaded is None:
        folder = model_dir()
        if not all((folder / f).exists() for f in MODEL_FILES):
            if time.monotonic() - _failed_at < RETRY_AFTER and _failed_at:
                raise RuntimeError("classifier model download failed recently; trying again later")
            try:
                download_model(folder)
            except Exception:
                _failed_at = time.monotonic()
                raise
        _classifier = loaded = Classifier(folder)
    return loaded


def classify(img: np.ndarray, circle: dict | None, box, night: bool, local=None, clf=None) -> dict | None:
    """The classifier's stored verdict for one frame (raw_json.calls.classifier), or None when it's off."""
    clf = clf or load()
    if clf is None:
        return None
    margins = CIRCLE_MARGINS if circle and circle.get("r") else (CIRCLE_MARGIN,)
    start = time.perf_counter()
    crops = [crop_for(img, circle, box, m) for m in margins]
    probs = np.mean([clf.probs(patch) for patch, _ in crops], axis=0)
    used = crops[0][1]
    seconds = time.perf_counter() - start
    local = local_species() if local is None else local
    return {"model": MODEL_NAME, "crop": used, "night": bool(night), **rank(probs, clf.labels, local),
            "seconds": round(seconds, 3)}


# Gate mode, rule H: two looks, each with its own bar. The first look reuses the Pi animal check's embedding of the
# wide view (the bowl square), which costs nothing extra but is the bottleneck: on 197 real visits its top guess was
# right 63 to 70% of the time, against 92 to 97% on a tight square around the bird. So when the bowl look falls
# short, one more look is made on a square CIRCLE_MARGIN radii around the motion circle (about 6 s on a Pi 4), with
# a higher bar. Before 2026-11-05 the one bar was 0.93 (checked 2026-10-31 on 66 visits re-scored from the crop box:
# 30 named, no mistakes). The one confident miss then (a junco called a California Scrub-Jay at 0.98) is stopped by
# requiring a species already settled in the yard. Both bars can be overridden from .env (config.py).
GATE_MIN_SCORE_BOWL = config.GATE_MIN_SCORE_BOWL    # the first look, on the Pi animal check's wide view
GATE_MIN_SCORE_TIGHT = config.GATE_MIN_SCORE_TIGHT  # the tight look, around the motion circle
GATE_MIN_LOCAL = 0.5  # share of its belief on local species; low means it saw something it doesn't know
GATE_VIEWS = ("bowl", "tight")  # the two looks, in the order the gate makes them


def gate_bar(view: str = "bowl") -> float:
    """The top-guess bar for a look: GATE_MIN_SCORE_TIGHT for the tight look, GATE_MIN_SCORE_BOWL for the first."""
    return GATE_MIN_SCORE_TIGHT if view == "tight" else GATE_MIN_SCORE_BOWL


def from_embedding(emb: np.ndarray, night: bool, local=None, clf=None) -> dict | None:
    """The stored verdict from an embedding already made (the Pi animal check's), so gate mode costs nothing
    extra. None when off. Its "crop" says "box"; the watcher replaces it with the view the embedding was of
    (view_kind: "bowl", "padded" or "frame")."""
    clf = clf or load()
    if clf is None:
        return None
    z = LOGIT_SCALE * (clf.text @ np.asarray(emb, np.float32))
    e = np.exp(z - z.max())
    local = local_species() if local is None else local
    return {"model": MODEL_NAME, "crop": "box", "night": bool(night), **rank(e / e.sum(), clf.labels, local),
            "seconds": 0.0}


def gate_miss(guess: dict | None, settled: dict[str, str], night: bool, crowd=None, view: str = "bowl") -> str | None:
    """Why a guess would not name the visit without Claude, in the order the gate tests it: "night", "none" (no
    local species at all), "score" (top guess under the look's bar, gate_bar(view)), "local" (under GATE_MIN_LOCAL
    of its belief on local species), "unsettled" (the species isn't settled here yet) or "crowd" (`crowd` is
    truthy: several animals in view). None when it would name it. `view` is which look this is: "bowl" (the first
    look, GATE_MIN_SCORE_BOWL) or "tight" (the look around the motion circle, GATE_MIN_SCORE_TIGHT). One place for
    the order: gate_pick, the watcher's stored looks and the visit page all read it."""
    if night:
        return "night"
    if not guess or not guess.get("top"):
        return "none"
    top = guess["top"][0]
    if top["score"] < gate_bar(view):
        return "score"
    if (guess.get("local_share") or 0) < GATE_MIN_LOCAL:
        return "local"
    visitor = settled.get(top["name"])
    if not visitor or visitor in ("none", "other"):
        return "unsettled"
    return "crowd" if crowd else None


def gate_pick(guess: dict | None, settled: dict[str, str], night: bool, view: str = "bowl") -> tuple[str, str] | None:
    """(species, visitor) when the guess is trusted to name the visit without Claude: daytime only (it hasn't
    seen infrared), a strong first guess (the look's bar, gate_bar(view)) mostly on local species, and a species
    with enough settled visits here (`settled`: label -> visitor type, from db.settled_species)."""
    if gate_miss(guess, settled, night, view=view):
        return None
    name = guess["top"][0]["name"]
    return name, settled[name]


def first_look(guess: dict, settled: dict[str, str], night: bool, crowd=None, view: str = "bowl") -> dict:
    """A gate look as the watcher stores it when it did not name the visit (raw_json.calls.classifier_gate for the
    first look on the bowl, calls.classifier_tight for the tight look): the guess plus why (gate_miss, judged
    against the look's own bar). The guess made on Claude's circle afterwards stays calls.classifier."""
    return {**guess, "why": gate_miss(guess, settled, night, crowd, view)}


def gate_check(emb, night: bool, settled: dict[str, str]) -> tuple[dict | None, tuple[str, str] | None]:
    """Fail-soft first gate look for the watcher, from the Pi animal check's embedding of its wide view, against
    GATE_MIN_SCORE_BOWL: (the guess, the pick or None)."""
    if config.CLASSIFIER != "gate" or emb is None or night:
        return None, None
    try:
        guess = from_embedding(emb, night)
    except Exception:
        log.exception("local classifier gate failed; asking Claude")
        return None, None
    return guess, gate_pick(guess, settled, night)


TIGHT_CROP = "motion"  # the stored guess's "crop" for the tight look: a square around the motion circle


def tight_check(jpeg: bytes, circle: dict | None, box, night: bool, settled: dict[str, str]
                ) -> tuple[dict | None, tuple[str, str] | None]:
    """Fail-soft second gate look for the watcher (rule H), made only when the first look did not name the visit:
    the photo is classified on a square CIRCLE_MARGIN radii around the motion circle (classify's crop of a circle),
    judged against GATE_MIN_SCORE_TIGHT. (the guess with crop TIGHT_CROP, the pick or None); (None, None) when
    not in gate mode, at night, without a circle or when anything fails (the watcher then asks Claude)."""
    if config.CLASSIFIER != "gate" or night or not circle or not circle.get("r"):
        return None, None
    try:
        guess = run(jpeg, circle, box, night)
        if not guess:
            return None, None
        guess = {**guess, "crop": TIGHT_CROP}
        return guess, gate_pick(guess, settled, night, view="tight")
    except Exception:
        log.exception("the tight look failed; asking Claude")
        return None, None


def box_embedding(jpeg: bytes, z, trigger: str | None) -> np.ndarray | None:
    """Fail-soft embedding of the Pi animal check's view (presence_patch of these zones for this trigger, for
    birdbrain.presence); None when off or anything fails."""
    if config.CLASSIFIER == "off":
        return None
    try:
        import cv2

        clf = load()
        img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        return clf.embed(presence_patch(img, z, trigger)[0]) if clf else None
    except Exception:
        log.exception("local embedding failed; skipped for this trigger")
        return None


def run(jpeg: bytes, circle: dict | None, box, night: bool) -> dict | None:
    """Fail-soft entry point for the watcher: any problem just means no guess for this visit."""
    if config.CLASSIFIER == "off":
        return None
    try:
        import cv2

        img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        return classify(img, circle, box, night)
    except Exception:
        log.exception("local classifier failed; skipped for this visit")
        return None


# --- backfill -------------------------------------------------------------------------------------------

def visit_circle(marks: str | None, raw: dict) -> dict | None:
    """The owner's first circle on the photo if they drew one (circles on other burst frames don't count), else
    Claude's."""
    owner = [m for m in db.parse_marks(marks) if not db.mark_frame(m)]
    if owner:
        return owner[0]
    return (raw.get("calls") or {}).get("circle")


def backfill(limit: int | None = None, redo: bool = False, clf=None, local=None) -> int:
    """Score stored real visits (newest first) from their saved photos. Returns how many were scored."""
    import cv2

    clf = clf or load()
    if clf is None:
        raise SystemExit("the classifier is off (CLASSIFIER=off)")
    local = local_species() if local is None else local
    with db.session() as conn:
        rows = conn.execute(
            f"SELECT id, image_path, is_night, marks, raw_json FROM visits WHERE {db.REAL} ORDER BY id DESC"
        ).fetchall()
    done = 0
    for r in rows:
        if limit is not None and done >= limit:
            break
        raw = json.loads(r["raw_json"] or "{}") or {}
        calls = raw.setdefault("calls", {})
        if calls.get("classifier") and not redo:
            continue
        img = cv2.imread(str(config.IMAGES_DIR / r["image_path"]))
        if img is None:
            continue
        box = config.GROUND_CROP if raw.get("trigger") == "ground" else config.CROP_ROI
        result = classify(img, visit_circle(r["marks"], raw), box, bool(r["is_night"]), local=local, clf=clf)
        with db.session() as conn:  # only this entry: the row may have gained a stay end or an owner check since
            db.set_raw_entry(conn, r["id"], "$.calls.classifier", {**result, "backfilled": True})
        done += 1
    return done


def backfill_gate(limit: int | None = None, redo: bool = False, dry_run: bool = False, clf=None, local=None,
                  settled: dict | None = None, every: int = 25, say=print) -> dict:
    """Write the first look (raw_json.calls.classifier_gate) for older visits that went to Claude, newest first,
    from the Pi animal check's stored embedding: only the text side is used, so no image is read or encoded.
    Done for daytime real visits Claude answered, not ones the Pi named or called empty, with no first look yet
    (`redo` makes them again). The record says `backfilled` and is made with today's settled list, which may
    differ from the one at the time; an embedding of an older view says which (`view_note`). A guess that would
    name the visit today gets why = "changed". Returns the counts."""
    clf = clf or load()
    if clf is None:
        raise SystemExit("the classifier is off (CLASSIFIER=off)")
    local = local_species() if local is None else local
    with db.session() as conn:
        settled = db.settled_species(conn) if settled is None else settled
        now_zones = zones.load(conn).to_dict()  # for visits that saved none
        rows = conn.execute(
            f"""SELECT v.id, v.is_night, v.raw_json, e.vec, e.view FROM visits v
                JOIN embeddings e ON e.visit_id = v.id
                WHERE {db.REAL} AND v.is_night = 0
                ORDER BY v.id DESC""").fetchall()
    counts = {"done": 0, "skipped": 0, "named_now": 0, "old_view": 0}
    total = len(rows)
    for n, r in enumerate(rows, 1):
        if limit is not None and counts["done"] >= limit:
            break
        try:
            raw = json.loads(r["raw_json"] or "{}") or {}
        except (ValueError, TypeError):
            raw = {}
        calls = raw.get("calls") if isinstance(raw, dict) else None
        calls = calls if isinstance(calls, dict) else {}
        asked = calls.get("haiku") or calls.get("sonnet") or calls.get("error")
        if (not asked or calls.get("settled_by") or calls.get("skipped_claude")
                or (calls.get("classifier_gate") and not redo)):
            counts["skipped"] += 1
        else:
            emb = np.frombuffer(r["vec"], np.float16).astype(np.float32)
            crowd = calls.get("gate_crowd") or ((raw.get("animals_seen") or 0) > 1 or None)
            guess = from_embedding(emb, False, local=local, clf=clf)
            view = r["view"]
            guess["crop"] = view_kind(raw.get("zones") or now_zones, raw.get("trigger")) if view == PRESENCE_VIEW else "box"
            record = first_look(guess, settled, False, crowd)
            if record["why"] is None:
                record["why"] = "changed"
                counts["named_now"] += 1
            record["backfilled"] = True
            if view != PRESENCE_VIEW:
                record["view_note"] = "its fingerprint is of the older, center-cut view of the crop box"
                counts["old_view"] += 1
            if not dry_run:
                with db.session() as conn:
                    db.set_raw_entry(conn, r["id"], "$.calls.classifier_gate", record)
            counts["done"] += 1
        if n % every == 0 or n == total:
            say(f"{n}/{total} looked at, {counts['done']} {'would be ' if dry_run else ''}written")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m birdbrain.classifier")
    parser.add_argument("photos", nargs="*", help="classify whole photos and print the top 3")
    parser.add_argument("--backfill", action="store_true", help="score stored visits that have no guess yet")
    parser.add_argument("--limit", type=int, help="with --backfill or --backfill-gate: at most this many visits, newest first")
    parser.add_argument("--redo", action="store_true", help="with --backfill or --backfill-gate: do visits again")
    parser.add_argument("--backfill-gate", action="store_true",
                        help="write the first look (BioCLIP 2 guess before Claude) for older visits, from stored embeddings")
    parser.add_argument("--dry-run", action="store_true", help="with --backfill-gate: count, write nothing")
    args = parser.parse_args()
    if args.backfill:
        with db.session() as conn:
            db.init_db(conn)
        print(f"classifier scored {backfill(args.limit, args.redo)} visits")
    elif args.backfill_gate:
        with db.session() as conn:
            db.init_db(conn)
        c = backfill_gate(args.limit, args.redo, args.dry_run)
        print(f"first look {'would be written' if args.dry_run else 'written'} for {c['done']} visits "
              f"({c['named_now']} that it would name today, {c['old_view']} from an older view, {c['skipped']} skipped)")
    elif args.photos:
        import cv2

        for p in args.photos:
            print(p, json.dumps(classify(cv2.imread(p), None, None, False)))
    else:
        parser.error("nothing to do (give photos, --backfill or --backfill-gate)")
    sys.exit(0)


if __name__ == "__main__":
    main()
