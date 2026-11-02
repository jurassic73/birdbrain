"""Local bird species classifier, in shadow mode: it guesses the species on the Pi and records the guess,
while Claude stays in charge of every label. The Stats page scores the guesses against the final labels
(owner corrections win, else Claude's), the first step toward identifying birds without Claude.

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

The same image embedding of the crop box also feeds birdbrain.presence (is anything there at all?).
IR night frames are grayscale; it hasn't been checked on them yet.

CLI: python -m birdbrain.classifier photo.jpg            (top 3 for whole photos)
     python -m birdbrain.classifier --backfill [--limit N] [--redo]   (score stored visits)
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

from . import config, db

log = logging.getLogger("birdbrain.classifier")

MODEL_NAME = "bioclip-2"
RELEASE_URL = "https://github.com/jurassic73/birdbrain/releases/download/bioclip2-v1/"
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
    for name, sha in MODEL_FILES.items():
        path = folder / name
        if path.exists():
            continue
        tmp, digest = path.with_suffix(".part"), hashlib.sha256()
        with httpx.stream("GET", RELEASE_URL + name, headers=HEADERS, timeout=60, follow_redirects=True) as r:
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

def crop_for(img: np.ndarray, circle: dict | None, box, margin: float = CIRCLE_MARGIN) -> tuple[np.ndarray, str]:
    """A square around the circle (fractions of the frame; r is a fraction of the width) when there is one,
    else the crop box. Returns the patch and which crop was used."""
    H, W = img.shape[:2]
    if circle and circle.get("r"):
        x0, y0, side = circle_square(circle, W, H, margin)
        return img[y0:y0 + side, x0:x0 + side], "circle"
    if box:
        x, y, w, h = box
        return img[int(y * H):int((y + h) * H), int(x * W):int((x + w) * W)], "box"
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


# Gate mode. Checked 2026-10-31 on 66 real visits re-scored from the crop box (what the gate sees, since Claude's
# circle doesn't exist yet): these limits named 30 with no mistakes. The one confident miss (a junco called a
# California Scrub-Jay at 0.98) is stopped by requiring a species already settled in the yard.
GATE_MIN_SCORE = 0.93
GATE_MIN_LOCAL = 0.5  # share of its belief on local species; low means it saw something it doesn't know


def from_embedding(emb: np.ndarray, night: bool, local=None, clf=None) -> dict | None:
    """The stored verdict from an embedding already made of the crop box (presence's), so gate mode costs
    nothing extra. None when off."""
    clf = clf or load()
    if clf is None:
        return None
    z = LOGIT_SCALE * (clf.text @ np.asarray(emb, np.float32))
    e = np.exp(z - z.max())
    local = local_species() if local is None else local
    return {"model": MODEL_NAME, "crop": "box", "night": bool(night), **rank(e / e.sum(), clf.labels, local),
            "seconds": 0.0}


def gate_pick(guess: dict | None, settled: dict[str, str], night: bool) -> tuple[str, str] | None:
    """(species, visitor) when the guess is trusted to name the visit without Claude: daytime only (it hasn't
    seen infrared), a strong first guess mostly on local species, and a species with enough settled visits
    here (`settled`: label -> visitor type, from db.settled_species)."""
    if night or not guess or not guess.get("top"):
        return None
    top = guess["top"][0]
    if top["score"] < GATE_MIN_SCORE or (guess.get("local_share") or 0) < GATE_MIN_LOCAL:
        return None
    visitor = settled.get(top["name"])
    return (top["name"], visitor) if visitor and visitor not in ("none", "other") else None


def gate_check(emb, night: bool, settled: dict[str, str]) -> tuple[dict | None, tuple[str, str] | None]:
    """Fail-soft gate step for the watcher: (the guess, the pick or None)."""
    if config.CLASSIFIER != "gate" or emb is None or night:
        return None, None
    try:
        guess = from_embedding(emb, night)
    except Exception:
        log.exception("local classifier gate failed; asking Claude")
        return None, None
    return guess, gate_pick(guess, settled, night)


def box_embedding(jpeg: bytes, box) -> np.ndarray | None:
    """Fail-soft embedding of the crop box (for birdbrain.presence); None when off or anything fails."""
    if config.CLASSIFIER == "off":
        return None
    try:
        import cv2

        clf = load()
        img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        return clf.embed(crop_for(img, None, box)[0]) if clf else None
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
    """The owner's first circle if they drew one, else Claude's."""
    owner = db.parse_marks(marks)
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
        calls["classifier"] = {**result, "backfilled": True}
        with db.session() as conn:
            db.set_raw_json(conn, r["id"], raw)
        done += 1
    return done


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m birdbrain.classifier")
    parser.add_argument("photos", nargs="*", help="classify whole photos and print the top 3")
    parser.add_argument("--backfill", action="store_true", help="score stored visits that have no guess yet")
    parser.add_argument("--limit", type=int, help="with --backfill: at most this many visits, newest first")
    parser.add_argument("--redo", action="store_true", help="with --backfill: score visits again")
    args = parser.parse_args()
    if args.backfill:
        with db.session() as conn:
            db.init_db(conn)
        print(f"classifier scored {backfill(args.limit, args.redo)} visits")
    elif args.photos:
        import cv2

        for p in args.photos:
            print(p, json.dumps(classify(cv2.imread(p), None, None, False)))
    else:
        parser.error("nothing to do (give photos or --backfill)")
    sys.exit(0)


if __name__ == "__main__":
    main()
