"""Is anything there? A small check on BioCLIP 2's image embedding of the crop box (the same model and file
as birdbrain.classifier), trained on this yard's own labelled frames: every stored trigger with its final
answer (the owner's correction wins, else Claude's), frames the owner confirmed or corrected counting
CONFIRMED_WEIGHT times.

Describing "nothing" in words didn't work (BioCLIP called almost every empty frame an animal), but a logistic
regression on the embeddings did: on the first 683 frames, 5-fold cross-validated, keeping all 55 visitors it
would have dropped 616 of 628 empty frames (255/265 by day, 361/363 at night; no night visitor yet).

The threshold comes from the same cross-validation: MARGIN times the lowest score any visitor got, so a frame
has to look emptier than every animal so far (and at most MAX_THRESHOLD).

One visitor whose crop box doesn't actually show it would drag that threshold to ~0 (visit 693: a dove whose sharpest
frame held only its tail; labelled a dove it took the threshold from 0.11 to 0.006 on 2026-11-02). So a visitor
only teaches "animal" with a frame that shows it (animal_example):
  - a false trigger corrected to an animal keeps its whole burst on disk: pick_frames embeds every burst frame and
    keeps the one that scores highest; it's used when it scores at least PICK_MIN (more likely an animal than not);
  - else the photo's own crop, when the owner's circle (or Claude's) is mostly (MIN_COVERAGE) inside the crop box,
    or, with no circle, when Claude or the Pi found the animal in that crop at the time;
  - else it's left out of training (a corrected false trigger with no burst and no circle: nothing shows the
    animal was in the crop box).
And as a backstop (threshold_from): when the lowest visitor sits more than OUTLIER_GAP below the next one (in
log-odds) and at least OUTLIER_PEERS visitors of its kind (visitor type, day or night) are in training, the
threshold comes from the next one. A first-of-its-kind visitor (the first possum at night) is never set aside.
On atmos's 116 visitors (2026-11-02) the gaps between the lowest scores were at most 1.1; visit 693 labelled a
dove sat 2.95 below the next.

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
     python -m birdbrain.presence --pick                   (burst frames for corrected false triggers, now)
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
PICK_MIN = 0.5  # a burst frame shows the animal when it scores at least this (more likely an animal than not)
MIN_COVERAGE = 0.5  # share of a circle inside the crop box for the photo's crop to count as showing the animal
ASPECT = 16 / 9  # the camera's frame (circles' r is a fraction of the width)
OUTLIER_GAP = 2.0  # log-odds below the next visitor (7x less likely) for the lowest to be set aside...
OUTLIER_PEERS = 10  # ...and only with this many others of its kind (visitor type, day or night) in training
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
    conn.execute("""INSERT INTO embeddings (visit_id, vec) VALUES (?, ?)
                    ON CONFLICT(visit_id) DO UPDATE SET vec = excluded.vec""",
                 (visit_id, np.asarray(emb, np.float16).tobytes()))
    conn.commit()


def crop_box(raw: dict, zones: dict | None = None) -> tuple | None:
    """The crop box the visit's frames were judged in: raw_json.zones (saved with newer visits), else `zones`."""
    z = raw.get("zones") or zones or {}
    box = z.get("ground_crop") if raw.get("trigger") == "ground" else z.get("crop")
    return tuple(box) if box and len(box) == 4 else None


def zones_then(conn, captured_at: str | None) -> dict | None:
    """The zones in use when a visit without saved zones was captured: the calibration history, else today's."""
    from . import zones

    return db.zones_at(conn, captured_at or "") or zones.load(conn).to_dict()


def coverage(circle: dict, box, aspect: float = ASPECT) -> float:
    """Share of a circle ({x, y, r} fractions; r of the width) inside a crop box (x, y, w, h fractions), counting
    only the part of the circle inside the frame."""
    t = np.linspace(-1, 1, 41)
    dx, dy = np.meshgrid(t, t)
    disc = dx ** 2 + dy ** 2 <= 1
    px = circle["x"] + circle["r"] * dx[disc]
    py = circle["y"] + circle["r"] * aspect * dy[disc]
    seen = (px >= 0) & (px <= 1) & (py >= 0) & (py <= 1)
    if not seen.any():
        return 0.0
    x, y, w, h = box
    inside = seen & (px >= x) & (px <= x + w) & (py >= y) & (py <= y + h)
    return float(inside.sum() / seen.sum())


def photo_coverage(marks: str | None, raw: dict, box) -> float | None:
    """How much of the animal is inside the crop box on the photo: the owner's circles (the best of them), else
    Claude's (or the Pi's) circle. None when there's no circle or no crop box."""
    circle = (raw.get("calls") or {}).get("circle")
    circles = db.parse_marks(marks) or ([circle] if isinstance(circle, dict) and circle.get("r") else [])
    if not box or not circles:
        return None
    return max(coverage(c, box) for c in circles)


def needs_pick(row) -> bool:
    """A false trigger corrected to an animal whose saved burst hasn't been looked through yet."""
    return bool(row["real"] and row["visitor"] == "none" and row["has_burst"] and row["photo_deleted_at"] is None
                and row["animal_frame"] is None)


def animal_example(row, conn=None) -> tuple[bytes | None, str]:
    """The embedding that teaches "animal" for a visitor (see the module notes), or None to leave it out, with why.
    `conn` finds the crop box of older visits that didn't save their zones."""
    if row["animal_vec"] is not None and (row["animal_score"] or 0) >= PICK_MIN:
        return row["animal_vec"], f"burst frame {row['animal_frame']}"
    try:
        raw = json.loads(row["raw_json"] or "{}") or {}
    except ValueError:
        raw = {}
    box = crop_box(raw)
    if box is None and conn is not None:
        box = crop_box(raw, zones_then(conn, row["captured_at"]))
    cover = photo_coverage(row["marks"], raw, box)
    if cover is not None:
        return (row["vec"], "photo") if cover >= MIN_COVERAGE else (None, "the circle is outside the crop box")
    if row["visitor"] != "none":
        return row["vec"], "photo"  # Claude (or the Pi) found the animal in this very crop
    if needs_pick(row):
        return None, "waiting for its burst to be looked through"
    return None, "no frame shows the animal in the crop box"


def examples(conn) -> dict:
    """Every stored frame as training data: embeddings X, labels y (an animal was there), weights w, visit ids,
    groups (visitor type and day/night, for threshold_from) and the visitors left out [(id, why)]."""
    rows = conn.execute(
        f"""SELECT e.visit_id, e.vec, e.animal_vec, e.animal_frame, e.animal_score, {db.REAL} AS real,
                   {CHECKED} AS confirmed, v.visitor, v.marks, v.raw_json, v.photo_deleted_at, v.is_night,
                   v.captured_at,
                   COALESCE(v.corrected_visitor, v.visitor) AS final,
                   json_extract(v.raw_json, '$.burst') IS NOT NULL AS has_burst
            FROM embeddings e JOIN visits v ON v.id = e.visit_id ORDER BY e.visit_id"""
    ).fetchall()
    vecs, y, w, ids, groups, left_out = [], [], [], [], [], []
    for r in rows:
        vec, why = animal_example(r, conn) if r["real"] else (r["vec"], "")
        if vec is None:
            left_out.append((r["visit_id"], why))
            continue
        vecs.append(np.frombuffer(vec, np.float16).astype(np.float32))
        y.append(bool(r["real"]))
        w.append(CONFIRMED_WEIGHT if r["confirmed"] else 1.0)
        ids.append(r["visit_id"])
        groups.append(f"{r['final']}/{'night' if r['is_night'] else 'day'}")
    X = np.stack(vecs) if vecs else np.zeros((0, 0), np.float32)
    return {"X": X, "y": np.array(y, bool), "w": np.array(w, float), "ids": np.array(ids, int),
            "groups": np.array(groups, object), "left_out": left_out}


def labelled(conn) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Embeddings, labels (an animal was there), weights and visit ids of every frame it trains on."""
    ex = examples(conn)
    return ex["X"], ex["y"], ex["w"], ex["ids"]


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def threshold_from(scores: np.ndarray, groups: np.ndarray | None = None) -> tuple[float, int, int | None]:
    """(threshold, index of the visitor it comes from, index of a visitor set aside or None) from the visitors'
    cross-validated scores. The lowest is set aside only when it is far below the rest (OUTLIER_GAP) and its kind
    is well known here (OUTLIER_PEERS), so a crop that doesn't really show its animal can't drag the threshold to
    ~0 while a first-of-its-kind visitor still lowers it."""
    order = np.argsort(scores, kind="stable")
    low, ignored = int(order[0]), None
    if groups is not None and len(order) > 1:
        nxt = int(order[1])
        peers = int((groups == groups[low]).sum()) - 1
        if logit(scores[nxt]) - logit(scores[low]) >= OUTLIER_GAP and peers >= OUTLIER_PEERS:
            low, ignored = nxt, low
    return float(min(MARGIN * scores[low], MAX_THRESHOLD)), low, ignored


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


def train(X, y, w, ids, groups=None) -> dict | None:
    """The model plus its threshold from cross-validation, or None while there are too few labels. The 5 folds
    and the final model (all frames) are fitted together. `groups` (visitor type and day/night per frame) lets
    threshold_from set one outlier aside."""
    if y.sum() < MIN_VISITORS or (~y).sum() < MIN_EMPTY:
        return None
    rows = np.stack([ids % FOLDS != f for f in range(FOLDS)] + [np.ones(len(y), bool)], axis=1)
    coef, bias = fit(X, y, w, rows)
    cv = np.zeros(len(y))
    for f in range(FOLDS):
        test = ~rows[:, f]
        cv[test] = sigmoid(np.asarray(X[test], np.float32) @ coef[:, f] + bias[f])
    visitor_ids = ids[y]
    threshold, low, ignored = threshold_from(cv[y], None if groups is None else np.asarray(groups)[y])
    return {"coef": np.ascontiguousarray(coef[:, -1]), "bias": float(bias[-1]), "threshold": threshold,
            "visitors": int(y.sum()), "empty": int((~y).sum()),
            "would_drop": round(float((cv[~y] < threshold).mean()), 3),
            "set_by": int(visitor_ids[low]), "set_aside": None if ignored is None else int(visitor_ids[ignored])}


def label_key(conn) -> tuple:
    """Changes with every new frame, correction or confirm, burst frame picked or circle drawn: (frames, visitors,
    frames the owner checked, burst frames picked, length of the owner's circles)."""
    return tuple(conn.execute(
        f"""SELECT COUNT(*), SUM({db.REAL}), SUM({CHECKED}), COUNT(e.animal_frame), SUM(length(v.marks))
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
        ex = examples(conn)
        trained = train(ex["X"], ex["y"], ex["w"], ex["ids"], ex["groups"])
        if trained is not None:
            trained["left_out"] = len(ex["left_out"])
        fresh = {"key": key, "model": trained}
        _model = fresh  # one assignment: other threads see the old model or the new one, never a mix
        log.info("animal check trained on %d frames in %.1f s", len(ex["y"]), time.perf_counter() - start)
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


def _keep_fresh(every: float, busy: threading.Event | None = None) -> None:
    while True:
        try:
            with db.session() as conn:
                if mode(conn) != "off":
                    if config.CLASSIFIER != "off":
                        pick_frames(conn, limit=1, busy=busy)  # one corrected false trigger per round, never mid-capture
                    retrain(conn)  # trains only when the labels changed
        except Exception:
            log.exception("could not retrain the animal check")
        time.sleep(every)


def start_background(every: float = RETRAIN_CHECK, busy: threading.Event | None = None) -> threading.Thread:
    """Train now, then retrain whenever the labels change (checked every `every` seconds), in a daemon thread with
    its own database connection. The watcher runs this so its first trigger finds a model ready and the owner's
    corrections and confirms reach the check without waiting for a trigger. Each round also looks through the burst
    of one false trigger corrected to an animal (pick_frames), pausing while `busy` (a capture) is set."""
    t = threading.Thread(target=_keep_fresh, args=(every, busy), name="presence", daemon=True)
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


def pending_picks(conn) -> list:
    """False triggers corrected to an animal whose saved burst hasn't been looked through, oldest first."""
    return conn.execute(
        f"""SELECT v.id, v.image_path, v.raw_json, v.captured_at FROM embeddings e JOIN visits v ON v.id = e.visit_id
            WHERE {db.REAL} AND v.visitor = 'none' AND e.animal_frame IS NULL AND v.photo_deleted_at IS NULL
              AND json_extract(v.raw_json, '$.burst') IS NOT NULL
            ORDER BY v.id""").fetchall()


def pick_frames(conn, limit: int | None = None, busy: threading.Event | None = None, clf=None) -> int:
    """For false triggers the owner corrected to an animal: embed the crop box of every saved burst frame and keep
    the one the current model scores highest (animal_vec, animal_frame, animal_score), so training sees the frame
    where the animal is in the crop box rather than the sharpest one. ~6 s a frame on a Pi 4. Stops (and tries again
    later) while `busy` is set. Returns how many visits were looked through."""
    import cv2

    from . import images

    rows = pending_picks(conn)[:limit]
    if not rows or (busy is not None and busy.is_set()):
        return 0
    m = model(conn)
    clf = clf or classifier.load()
    if m is None or clf is None:
        return 0
    done = 0
    for r in rows:
        raw = json.loads(r["raw_json"] or "{}") or {}
        box = crop_box(raw) or crop_box(raw, zones_then(conn, r["captured_at"]))
        emb, best, n = None, -1.0, 0
        for i, rel in enumerate(images.burst_paths(r["image_path"]), 1):
            if busy is not None and busy.is_set():
                return done
            img = cv2.imread(str(config.IMAGES_DIR / rel))
            if img is None:
                continue
            e = clf.embed(classifier.crop_for(img, None, box)[0])
            s = score(m, e)
            if s > best:
                emb, best, n = e, s, i
        conn.execute("UPDATE embeddings SET animal_vec = ?, animal_frame = ?, animal_score = ? WHERE visit_id = ?",
                     (None if emb is None else np.asarray(emb, np.float16).tobytes(), n,
                      None if emb is None else round(best, 4), r["id"]))
        conn.commit()
        if n:
            log.info("visit %d: burst frame %d shows the animal best (score %.3f)", r["id"], n, best)
        else:
            log.info("visit %d: no burst frames to look through", r["id"])
        done += 1
    return done


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
    parser.add_argument("--pick", action="store_true",
                        help="look through the bursts of false triggers corrected to an animal")
    args = parser.parse_args()
    with db.session() as conn:
        db.init_db(conn)
    if args.backfill:
        print(f"embedded {backfill(args.limit)} frames")
    if args.pick:
        logging.basicConfig(level=logging.INFO, format="%(message)s")  # one line per visit as it goes
        with db.session() as conn:
            todo = len(pending_picks(conn))
            print(f"{todo} corrected false triggers to look through (~{todo * 36} s on a Pi 4)")
            print(f"looked through {pick_frames(conn)}")
    with db.session() as conn:
        start = time.perf_counter()
        m = retrain(conn)["model"]
        took = time.perf_counter() - start
    print(f"trained in {took:.1f} s")
    print("not enough labelled frames yet" if m is None else
          f"trained on {m['visitors']} visitors and {m['empty']} empty frames; threshold {m['threshold']:.3f}, "
          f"cross-validated it drops {m['would_drop']:.0%} of empty frames; threshold set by visit {m['set_by']}"
          + (f", visit {m['set_aside']} set aside as an outlier" if m["set_aside"] else "")
          + (f"; {m['left_out']} visitors left out (no frame shows them in the crop box)" if m["left_out"] else ""))


if __name__ == "__main__":
    main()
