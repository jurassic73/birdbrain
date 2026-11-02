"""Is anything there? A small check on BioCLIP 2's image embedding of the crop box (the same model and file
as birdbrain.classifier), trained on this yard's own labelled frames: every stored trigger with its final
answer (the owner's correction wins, else Claude's), frames the owner confirmed or corrected counting
CONFIRMED_WEIGHT times.

Describing "nothing" in words didn't work (BioCLIP called almost every empty frame an animal), but a logistic
regression on the embeddings did: on the first 683 frames, 5-fold cross-validated, keeping all 55 visitors it
would have dropped 616 of 628 empty frames (255/265 by day, 361/363 at night; no night visitor yet).

The threshold comes from the same cross-validation: MARGIN times the lowest score any visitor got, so a frame
has to look emptier than every animal so far (and at most MAX_THRESHOLD).

Retrained whenever the labels change (a new frame, a correction, a confirm), but never on a trigger's path: a
trigger scores with the newest model already trained, and the retrain runs in a background thread with its own
database connection (one at a time). The watcher also checks the labels every RETRAIN_CHECK seconds, so a
correction or confirm reaches its model within about a minute even with no trigger. Training fits the 5
cross-validation folds and the final model together in one batched gradient descent in float32: ~4 s on a Pi 4
and ~0.05 s on a Mac for ~800 frames (fitted one by one in float64 it took ~30 s on the Pi, same results). Only
the first check after a process starts, with no model at all, waits for a training.

Modes (Settings, else PRESENCE in .env): off; shadow records the score; gate skips Claude when it says empty.
Skipped frames are stored as false triggers for the owner's daily review; correcting one teaches it.

CLI: python -m birdbrain.presence --backfill [--limit N]   (embeddings for stored frames that have none)
"""

import argparse
import json
import logging
import threading
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
STEPS = 400
RETRAIN_CHECK = 60  # seconds between the watcher's label checks (birdbrain.watch runs start_background())
BACKGROUND = True  # retrain in a thread; tests set False to train synchronously (no stray threads)
# {"key": label key, "model": train()'s result}. Replaced whole (an atomic swap), never changed in place.
_model: dict | None = None
_train_lock = threading.Lock()  # one training at a time in this process
_thread_lock = threading.Lock()
_thread: threading.Thread | None = None
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


def sigmoid(z):
    return 0.5 * (1 + np.tanh(0.5 * z))  # the logistic function, without exp()'s overflow warnings in float32


def fit(X: np.ndarray, y: np.ndarray, w: np.ndarray, rows: np.ndarray, steps: int = STEPS,
        lr: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Logistic regressions, classes balanced (visitors are rare), by plain gradient descent: one per column of
    `rows` (n x k, True where that model trains on the frame), all k in one pass over X per step. Each column
    matches fitting on X[rows[:, j]] alone; batching reads X once per step instead of k times, which is what
    costs on a Pi (memory bound). Returns coefficients (d x k) and biases (k)."""
    X = np.asarray(X, np.float32) * np.float32(10)  # embeddings are unit length: spread them out to converge sooner
    on = rows.astype(np.float32)
    target = y.astype(np.float32)[:, None]
    n = on.sum(0)
    visitors = (on * target).sum(0)
    balance = (np.where(y[:, None], (n - visitors) / np.maximum(visitors, 1), 1.0) * w[:, None] * on).astype(np.float32)
    coef = np.zeros((X.shape[1], rows.shape[1]), np.float32)
    bias = np.zeros(rows.shape[1], np.float32)
    lr, l2 = np.float32(lr), np.float32(L2)
    for _ in range(steps):
        g = (sigmoid(X @ coef + bias) - target) * balance  # zero for frames a model doesn't train on
        coef -= lr * ((X.T @ g) / n + l2 * coef)
        bias -= lr * g.sum(0) / n
    return coef * np.float32(10), bias


def score(model: dict, emb: np.ndarray) -> float:
    return float(sigmoid(np.asarray(emb, np.float32) @ model["coef"] + model["bias"]))


def train(X, y, w, ids) -> dict | None:
    """The model plus its threshold from cross-validation, or None while there are too few labels. The 5 folds
    and the final model (all frames) are fitted together."""
    if y.sum() < MIN_VISITORS or (~y).sum() < MIN_EMPTY:
        return None
    rows = np.stack([ids % FOLDS != f for f in range(FOLDS)] + [np.ones(len(y), bool)], axis=1)
    coef, bias = fit(X, y, w, rows)
    cv = np.zeros(len(y))
    for f in range(FOLDS):
        test = ~rows[:, f]
        cv[test] = sigmoid(np.asarray(X[test], np.float32) @ coef[:, f] + bias[f])
    threshold = float(min(MARGIN * cv[y].min(), MAX_THRESHOLD))
    return {"coef": np.ascontiguousarray(coef[:, -1]), "bias": float(bias[-1]), "threshold": threshold,
            "visitors": int(y.sum()), "empty": int((~y).sum()),
            "would_drop": round(float((cv[~y] < threshold).mean()), 3)}


def label_key(conn) -> tuple:
    """Changes with every new frame, correction or confirm: (frames, visitors, frames the owner checked)."""
    return tuple(conn.execute(
        f"""SELECT COUNT(*), SUM({db.REAL}), SUM({CHECKED})
            FROM embeddings e JOIN visits v ON v.id = e.visit_id""").fetchone())


def retrain(conn) -> dict:
    """Train on the current labels unless the model in use already has them. Waits for a training under way
    (then usually finds it current). Returns {"key", "model"}."""
    global _model
    with _train_lock:
        key = label_key(conn)
        current = _model
        if current is not None and current["key"] == key:
            return current
        start = time.perf_counter()
        fresh = {"key": key, "model": train(*labelled(conn))}
        _model = fresh  # one assignment: other threads see the old model or the new one, never a mix
        log.info("animal check trained on %d frames in %.1f s", key[0], time.perf_counter() - start)
        return fresh


def _retrain_quietly() -> None:
    try:
        with db.session() as conn:  # a connection of its own: sqlite connections can't cross threads
            retrain(conn)
    except Exception:
        log.exception("could not retrain the animal check")


def retrain_soon() -> threading.Thread | None:
    """Retrain in a daemon thread unless one is already running (that one, or the next check, catches up).
    With BACKGROUND off (tests) it trains right here and returns None."""
    global _thread
    if not BACKGROUND:
        _retrain_quietly()
        return None
    with _thread_lock:
        if _thread is None or not _thread.is_alive():
            _thread = threading.Thread(target=_retrain_quietly, name="presence-train", daemon=True)
            _thread.start()
        return _thread


def model(conn) -> dict | None:
    """The newest trained model. When the labels changed since (a new frame, a correction, a confirm) a retrain
    starts in the background and this returns the model in use, a few labels behind, without waiting. Only the
    first call in a process, with no model yet, trains here (or waits for the training under way)."""
    current = _model
    if current is None:
        return retrain(conn)["model"]
    if current["key"] != label_key(conn):
        if not BACKGROUND:
            return retrain(conn)["model"]
        retrain_soon()
    return current["model"]


def _keep_fresh(every: float) -> None:
    while True:
        try:
            with db.session() as conn:
                if mode(conn) != "off":
                    retrain(conn)  # trains only when the labels changed
        except Exception:
            log.exception("could not retrain the animal check")
        time.sleep(every)


def start_background(every: float = RETRAIN_CHECK) -> threading.Thread:
    """Train now, then retrain whenever the labels change (checked every `every` seconds), in a daemon thread with
    its own database connection. The watcher runs this so its first trigger finds a model ready and the owner's
    corrections and confirms reach the check without waiting for a trigger."""
    t = threading.Thread(target=_keep_fresh, args=(every,), name="presence", daemon=True)
    t.start()
    return t


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
        start = time.perf_counter()
        m = retrain(conn)["model"]
        took = time.perf_counter() - start
    print(f"trained in {took:.1f} s")
    print("not enough labelled frames yet" if m is None else
          f"trained on {m['visitors']} visitors and {m['empty']} empty frames; threshold {m['threshold']:.3f}, "
          f"cross-validated it drops {m['would_drop']:.0%} of empty frames and keeps every visitor")


if __name__ == "__main__":
    main()
