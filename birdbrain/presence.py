"""Is anything there? A small check on BioCLIP 2's image embedding of the crop box (the same model and file
as birdbrain.classifier), trained on this yard's own labelled frames: every stored trigger with its final
answer (the owner's correction wins, else Claude's), frames the owner confirmed or corrected counting
CONFIRMED_WEIGHT times.

Describing "nothing" in words didn't work (BioCLIP called almost every empty frame an animal), but a logistic
regression on the embeddings did: on the first 683 frames, 5-fold cross-validated, keeping all 55 visitors it
would have dropped 616 of 628 empty frames (255/265 by day, 361/363 at night; no night visitor yet).

The threshold comes from the same cross-validation: MARGIN times the lowest score any visitor got, so a frame
has to look emptier than every animal so far (and at most MAX_THRESHOLD). Retrained whenever the labels change (it takes milliseconds).

Modes (Settings, else PRESENCE in .env): off; shadow records the score; gate skips Claude when it says empty.
Skipped frames are stored as false triggers for the owner's daily review; correcting one teaches it.

CLI: python -m birdbrain.presence --backfill [--limit N]   (embeddings for stored frames that have none)
"""

import argparse
import json
import logging
import time

import numpy as np

from . import classifier, config, db

log = logging.getLogger("birdbrain.presence")

MODEL_NAME = "bioclip-presence"
MODES = ("off", "shadow", "gate")
MIN_VISITORS, MIN_EMPTY = 10, 30  # labelled frames needed before it judges anything
CONFIRMED_WEIGHT = 3.0
MARGIN = 0.8
MAX_THRESHOLD = 0.5  # never call a frame empty when it's more likely an animal than not
FOLDS = 5
L2 = 1e-3
_model: dict | None = None
# Checked by the owner: confirmed with the button, or corrected.
CHECKED = "(v.confirmed_at IS NOT NULL OR v.corrected_visitor IS NOT NULL OR v.corrected_species IS NOT NULL)"


def mode(conn) -> str:
    value = db.get_value(conn, "presence_mode")
    return value if value in MODES else config.PRESENCE


def store(conn, visit_id: int, emb: np.ndarray) -> None:
    conn.execute("INSERT OR REPLACE INTO embeddings (visit_id, vec) VALUES (?, ?)",
                 (visit_id, np.asarray(emb, np.float16).tobytes()))
    conn.commit()


def labelled(conn) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Embeddings, labels (an animal was there), weights and visit ids of every stored frame."""
    rows = conn.execute(
        f"""SELECT e.visit_id, e.vec, {db.REAL} AS real, {CHECKED} AS confirmed
            FROM embeddings e JOIN visits v ON v.id = e.visit_id ORDER BY e.visit_id"""
    ).fetchall()
    if not rows:
        return np.zeros((0, 0), np.float32), np.zeros(0, bool), np.zeros(0), np.zeros(0, int)
    X = np.stack([np.frombuffer(r["vec"], np.float16).astype(np.float32) for r in rows])
    y = np.array([bool(r["real"]) for r in rows])
    w = np.array([CONFIRMED_WEIGHT if r["confirmed"] else 1.0 for r in rows])
    return X, y, w, np.array([r["visit_id"] for r in rows])


def fit(X: np.ndarray, y: np.ndarray, w: np.ndarray, steps: int = 400, lr: float = 0.5) -> tuple[np.ndarray, float]:
    """Logistic regression, classes balanced (visitors are rare), by plain gradient descent."""
    X = X * 10  # embeddings are unit length: spread them out so it converges in a few hundred steps
    balance = np.where(y, (~y).sum() / max(y.sum(), 1), 1.0) * w
    coef, bias = np.zeros(X.shape[1]), 0.0
    for _ in range(steps):
        p = 1 / (1 + np.exp(-(X @ coef + bias)))
        g = (p - y) * balance
        coef -= lr * (X.T @ g / len(y) + L2 * coef)
        bias -= lr * g.mean()
    return coef * 10, bias


def score(model: dict, emb: np.ndarray) -> float:
    return float(1 / (1 + np.exp(-(np.asarray(emb, np.float32) @ model["coef"] + model["bias"]))))


def train(X, y, w, ids) -> dict | None:
    """The model plus its threshold from cross-validation, or None while there are too few labels."""
    if y.sum() < MIN_VISITORS or (~y).sum() < MIN_EMPTY:
        return None
    cv = np.zeros(len(y))
    for f in range(FOLDS):
        test = ids % FOLDS == f
        coef, bias = fit(X[~test], y[~test], w[~test])
        cv[test] = 1 / (1 + np.exp(-(X[test] @ coef + bias)))
    threshold = float(min(MARGIN * cv[y].min(), MAX_THRESHOLD))
    coef, bias = fit(X, y, w)
    return {"coef": coef, "bias": bias, "threshold": threshold, "visitors": int(y.sum()), "empty": int((~y).sum()),
            "would_drop": round(float((cv[~y] < threshold).mean()), 3)}


def model(conn) -> dict | None:
    """The trained model, retrained when the labels changed (a new frame, a correction, a confirm)."""
    global _model
    key = tuple(conn.execute(
        f"""SELECT COUNT(*), SUM({db.REAL}), SUM({CHECKED})
            FROM embeddings e JOIN visits v ON v.id = e.visit_id""").fetchone())
    if _model is None or _model["key"] != key:
        _model = {"key": key, "model": train(*labelled(conn))}
    return _model["model"]


def check(conn, emb: np.ndarray | None) -> dict | None:
    """The stored verdict for one trigger (raw_json.calls.presence), or None without an embedding."""
    if emb is None:
        return None
    start = time.perf_counter()
    m = model(conn)
    if m is None:
        return {"model": MODEL_NAME, "score": None, "empty": False, "note": "not enough labelled frames yet"}
    s = score(m, emb)
    return {"model": MODEL_NAME, "score": round(s, 4), "threshold": round(m["threshold"], 4), "empty": s < m["threshold"],
            "trained_on": m["visitors"] + m["empty"], "seconds": round(time.perf_counter() - start, 3)}


def backfill(limit: int | None = None) -> int:
    """Embeddings for stored frames that have none (~6 s each on a Pi 4), newest first."""
    import cv2

    from . import zones

    clf = classifier.load()
    if clf is None:
        raise SystemExit("the classifier is off (CLASSIFIER=off)")
    z = zones.current()
    with db.session() as conn:
        rows = conn.execute("""SELECT id, image_path, raw_json FROM visits
                               WHERE id NOT IN (SELECT visit_id FROM embeddings) ORDER BY id DESC""").fetchall()
    done = 0
    for r in rows[:limit]:
        img = cv2.imread(str(config.IMAGES_DIR / r["image_path"]))
        if img is None:
            continue
        raw = json.loads(r["raw_json"] or "{}") or {}
        box = z.ground_crop if raw.get("trigger") == "ground" else z.crop
        emb = clf.embed(classifier.crop_for(img, None, box)[0])
        with db.session() as conn:
            store(conn, r["id"], emb)
        done += 1
    return done


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m birdbrain.presence")
    parser.add_argument("--backfill", action="store_true", help="embed stored frames that have no embedding")
    parser.add_argument("--limit", type=int, help="with --backfill: at most this many, newest first")
    args = parser.parse_args()
    with db.session() as conn:
        db.init_db(conn)
    if args.backfill:
        print(f"embedded {backfill(args.limit)} frames")
    with db.session() as conn:
        m = model(conn)
    print("not enough labelled frames yet" if m is None else
          f"trained on {m['visitors']} visitors and {m['empty']} empty frames; threshold {m['threshold']:.3f}, "
          f"cross-validated it drops {m['would_drop']:.0%} of empty frames and keeps every visitor")


if __name__ == "__main__":
    main()
