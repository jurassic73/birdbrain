"""Is anything there? A small check on BioCLIP 2's image embedding of its view (the same model and file as
birdbrain.classifier), trained on this yard's own labelled frames: every stored trigger with its final
answer (the owner's correction wins, else Claude's), frames the owner confirmed or corrected counting
CONFIRMED_WEIGHT times.

Its view (classifier.presence_patch, version classifier.PRESENCE_VIEW, saved in embeddings.view): a square around
the whole bowl, rim and a bird's height above it, for birdbath triggers; the ground crop padded to a square for
ground ones. It used to be the crop box center-cut to a square, which cut the rim's left and right ends off (birds
perched there were called empty). Only embeddings of the current view train or are scored; older ones wait for
`--backfill --redo`, which holds the check while it runs (HOLD_KEY: gate mode at shadow, no rain probes, no burst
picks; a model trained on the first few redone frames mustn't skip Claude or drop a probe). Only a complete redo
lets go: every frame with a photo redone and, when gate is to come back, the bursts waiting to be looked through
picked too (redoing a row forgets its pick), retrained, and then gate restored (RESTORE_KEY remembers it across
partial runs). A partial run leaves the hold on until a later redo finishes or it lapses (HOLD_MAX). "Inside the
crop box" below means inside that view (classifier.presence_view_box), the dashed box on the visit page.

Describing "nothing" in words didn't work (BioCLIP called almost every empty frame an animal), but a logistic
regression on the embeddings did: on the first 683 frames, 5-fold cross-validated, keeping all 55 visitors it
would have dropped 616 of 628 empty frames (255/265 by day, 361/363 at night; no night visitor yet).

The threshold comes from the same cross-validation: MARGIN times the lowest score any visitor got, so a frame
has to look emptier than every animal so far (and at most MAX_THRESHOLD).

One visitor whose crop box doesn't actually show it would drag that threshold to ~0 (visit 693: a dove whose sharpest
frame held only its tail; labelled a dove it took the threshold from 0.11 to 0.006 on 2026-11-02). So a visitor
only teaches "animal" with a frame that shows it (animal_example):
  - the owner's pick: when their first circle sits on another frame of the burst (the visit page lets them draw on
    any frame), that frame's crop box is the example, if the circle is mostly (MIN_COVERAGE) inside it, else the
    visit is left out. pick_frames embeds it in the background (~6 s on a Pi 4), and again whenever the circle
    moves to another frame; until then the visit waits, left out;
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
     python -m birdbrain.presence --backfill --redo [--limit N]   (also re-embed frames of an older view)
     python -m birdbrain.presence --pick                   (burst frames for corrected false triggers and the
                                                           owner's circled frames, now)
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
# Set while `--backfill --redo` runs and kept after a partial one (the time it last checked in, epoch seconds): the
# model in hand then is trained on part of the frames, so the watcher sends no rain probes (watch.probe_ready) and
# picks no burst frames. A redo checks in with every progress line, and a partial run once more as it stops; a hold
# older than HOLD_MAX (no complete redo since) counts as gone.
HOLD_KEY = "presence_hold"
HOLD_MAX = 3 * 3600
# The mode a redo put at shadow, to restore once a redo completes (kept across partial runs): "gate" (the Settings
# value) or "env" (gate came from .env: the setting is removed again).
RESTORE_KEY = "presence_hold_restore"
RESTORES = ("gate", "env")
_now = time.time  # the hold's clock (tests move it on)
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


def held(conn) -> bool:
    """Whether a redo is holding the check (HOLD_KEY set and checked in within HOLD_MAX)."""
    try:
        since = float(db.get_value(conn, HOLD_KEY) or "nan")
    except ValueError:
        return False
    return 0 <= _now() - since < HOLD_MAX


def store(conn, visit_id: int, emb: np.ndarray, view: int = classifier.PRESENCE_VIEW) -> None:
    """Save a frame's embedding of `view`. Re-embedding a row of another view also forgets its burst frame
    pick (made in the old view), so pick_frames looks again."""
    conn.execute("""INSERT INTO embeddings (visit_id, vec, view) VALUES (?, ?, ?)
                    ON CONFLICT(visit_id) DO UPDATE SET vec = excluded.vec, view = excluded.view,
                      animal_vec = CASE WHEN embeddings.view IS excluded.view THEN animal_vec END,
                      animal_frame = CASE WHEN embeddings.view IS excluded.view THEN animal_frame END,
                      animal_score = CASE WHEN embeddings.view IS excluded.view THEN animal_score END,
                      animal_by_owner = CASE WHEN embeddings.view IS excluded.view THEN animal_by_owner END""",
                 (visit_id, np.asarray(emb, np.float16).tobytes(), view))
    conn.commit()


def visit_zones(raw: dict, conn=None, captured_at: str | None = None) -> dict | None:
    """The zones a visit's frames were judged in: raw_json.zones, else (with `conn`) the zones at the time."""
    z = raw.get("zones")
    if isinstance(z, dict) and z:
        return z
    return zones_then(conn, captured_at) if conn is not None else None


def view_box(raw: dict, zones: dict | None = None) -> tuple | None:
    """The Pi animal check's view (x, y, w, h fractions) for the visit's trigger: from raw_json.zones (saved with
    newer visits), else `zones`. Circles are judged against it."""
    z = raw.get("zones") or zones or {}
    return classifier.presence_view_box(z, raw.get("trigger"), ASPECT)


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


def photo_coverage(marks: str | None, raw: dict, box, frame: int | None = None) -> float | None:
    """How much of the animal is inside the crop box on the photo (or on burst frame `frame`): the owner's circles
    on that frame (the best of them), else, on the photo, Claude's (or the Pi's) circle. None when there's no circle
    or no crop box."""
    circle = (raw.get("calls") or {}).get("circle")
    circles = [m for m in db.parse_marks(marks, db.claude_frame(raw)) if db.mark_frame(m) == frame]
    if not circles and frame is None and isinstance(circle, dict) and circle.get("r"):
        circles = [circle]
    if not box or not circles:
        return None
    return max(coverage(c, box) for c in circles)


def owner_frame(marks: str | None, raw: dict | None = None) -> int | None:
    """The burst frame the owner's first circle is on (1-based), or None when it's on the photo Claude saw (or
    there's no circle of theirs). Claude's and the Pi's circles are always on the photo."""
    first = next((m for m in db.parse_marks(marks, db.claude_frame(raw or {})) if not (m.get("claude") or m.get("pi"))),
                 None)
    return db.mark_frame(first) if first else None


def _raw(row) -> dict:
    try:
        raw = json.loads(row["raw_json"] or "{}") or {}
    except ValueError:
        raw = {}
    return raw if isinstance(raw, dict) else {}


def _by_owner(row) -> bool:
    return bool("animal_by_owner" in row.keys() and row["animal_by_owner"])


def needs_pick(row) -> bool:
    """A false trigger corrected to an animal whose saved burst hasn't been looked through yet."""
    return bool(row["real"] and row["visitor"] == "none" and row["has_burst"] and row["photo_deleted_at"] is None
                and row["animal_frame"] is None)


def animal_example(row, conn=None) -> tuple[bytes | None, str]:
    """The embedding that teaches "animal" for a visitor (see the module notes), or None to leave it out, with why.
    `conn` finds the crop box of older visits that didn't save their zones."""
    raw = _raw(row)
    box = view_box(raw)
    if box is None and conn is not None:
        box = view_box(raw, zones_then(conn, row["captured_at"]))
    own = owner_frame(row["marks"], raw)
    if own is not None:  # the owner circled it on another burst frame: that frame or nothing
        picked = _by_owner(row) and row["animal_frame"] == own
        if not picked and row["has_burst"] and row["photo_deleted_at"] is None:
            return None, f"waiting for burst frame {own} (your circle) to be looked at"
        if not picked or row["animal_vec"] is None:
            return None, f"burst frame {own} (your circle) is no longer on disk"
        cover = photo_coverage(row["marks"], raw, box, own)
        if cover is not None and cover < MIN_COVERAGE:
            return None, f"your circle on burst frame {own} is outside the dashed box"
        return row["animal_vec"], f"burst frame {own} (your circle)"
    if (row["animal_vec"] is not None and not _by_owner(row)  # an owner's pick from a circle since moved: stale
            and (row["animal_score"] or 0) >= PICK_MIN):
        return row["animal_vec"], f"burst frame {row['animal_frame']}"
    cover = photo_coverage(row["marks"], raw, box)
    if cover is not None:
        return (row["vec"], "photo") if cover >= MIN_COVERAGE else (None, "the circle is outside the dashed box")
    if row["visitor"] != "none":
        return row["vec"], "photo"  # Claude (or the Pi) found the animal in this very crop
    if needs_pick(row):
        return None, "waiting for its burst to be looked through"
    return None, "no frame shows the animal in the dashed box"


EXAMPLE_ROWS = f"""SELECT e.visit_id, e.vec, e.animal_vec, e.animal_frame, e.animal_score, e.animal_by_owner,
                          {db.REAL} AS real, {CHECKED} AS confirmed, v.visitor, v.marks, v.raw_json,
                          v.photo_deleted_at, v.is_night, v.captured_at,
                          COALESCE(v.corrected_visitor, v.visitor) AS final,
                          json_extract(v.raw_json, '$.burst') IS NOT NULL AS has_burst
                   FROM embeddings e JOIN visits v ON v.id = e.visit_id
                   WHERE e.view = {classifier.PRESENCE_VIEW}"""


def teaches(conn, visit_id: int) -> dict | None:
    """What one visit teaches the check, for its page: {"animal", "used", "why", "checked"}, or None without an
    embedding. `why` names the frame it learns from ("burst frame 4 (your circle)") or why it's left out."""
    row = conn.execute(EXAMPLE_ROWS + " AND e.visit_id = ?", (visit_id,)).fetchone()
    if row is None:
        return None
    if not row["real"]:
        return {"animal": False, "used": True, "why": "", "checked": bool(row["confirmed"])}
    vec, why = animal_example(row, conn)
    if vec is not None and why.startswith("burst frame") and "(" not in why:
        why += " (the one it scored most like an animal)"
    return {"animal": True, "used": vec is not None, "why": "the photo" if why == "photo" else why,
            "checked": bool(row["confirmed"])}


def examples(conn) -> dict:
    """Every stored frame as training data: embeddings X, labels y (an animal was there), weights w, visit ids,
    groups (visitor type and day/night, for threshold_from) and the visitors left out [(id, why)]."""
    rows = conn.execute(EXAMPLE_ROWS + " ORDER BY e.visit_id").fetchall()
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
    frames the owner checked, burst frames picked, which ones and whose, length of the owner's circles, the burst
    frames their circles are on)."""
    return tuple(conn.execute(
        f"""SELECT COUNT(*), SUM({db.REAL}), SUM({CHECKED}), COUNT(e.animal_frame), SUM(e.animal_frame),
                   SUM(e.animal_by_owner), SUM(length(v.marks)),
                   SUM(CASE WHEN v.marks LIKE '%"frame"%' AND json_valid(v.marks) THEN
                       (SELECT TOTAL(json_extract(j.value, '$.frame') * (j.key + 1)) FROM json_each(v.marks) j)
                       END)
            FROM embeddings e JOIN visits v ON v.id = e.visit_id
            WHERE e.view = {classifier.PRESENCE_VIEW}""").fetchone())


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


def trained() -> bool:
    """Whether a trained model is in hand right now (no database, no training): the watcher's rain probes need
    one, since without a verdict nothing would filter them."""
    current = _model
    return current is not None and current["model"] is not None


def _keep_fresh(every: float, busy: threading.Event | None = None) -> None:
    while True:
        try:
            with db.session() as conn:
                if mode(conn) != "off":
                    # One corrected false trigger per round, never mid-capture, and none during a redo: the model
                    # then is trained on part of the frames, and a low score would leave a visitor out for good.
                    if config.CLASSIFIER != "off" and not held(conn):
                        pick_frames(conn, limit=1, busy=busy)
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


def check(conn, emb: np.ndarray | None, view: str | None = None) -> dict | None:
    """The stored verdict for one trigger (raw_json.calls.presence), or None without an embedding. `view` (the
    kind of classifier.presence_patch the embedding was made of) is recorded with it when given."""
    if emb is None:
        return None
    start = time.perf_counter()
    m = model(conn)
    seen = {"view": view} if view else {}
    if m is None:
        return {"model": MODEL_NAME, "score": None, "empty": False, "note": "not enough labelled frames yet", **seen}
    s = score(m, emb)
    return {"model": MODEL_NAME, "score": round(s, 4), "threshold": round(m["threshold"], 4), "empty": s < m["threshold"],
            "trained_on": m["visitors"] + m["empty"], "seconds": round(time.perf_counter() - start, 3), **seen}


def pending_picks(conn) -> list:
    """Visits whose burst needs looking at, the owner's picks first, then oldest first: false triggers corrected to an animal whose saved burst
    hasn't been looked through, and visitors whose owner's first circle sits on a burst frame other than the one
    embedded for it (`owner_frame` set on the row) or no longer does (an owner's pick to undo)."""
    rows = conn.execute(
        f"""SELECT v.id, v.image_path, v.raw_json, v.captured_at, v.marks, v.visitor, e.animal_frame, e.animal_by_owner
            FROM embeddings e JOIN visits v ON v.id = e.visit_id
            WHERE {db.REAL} AND v.photo_deleted_at IS NULL AND json_extract(v.raw_json, '$.burst') IS NOT NULL
              AND e.view = {classifier.PRESENCE_VIEW}
              AND ((v.visitor = 'none' AND e.animal_frame IS NULL) OR e.animal_by_owner
                   OR v.marks LIKE '%"frame"%')
            ORDER BY v.id""").fetchall()
    out = []
    for r in rows:
        own = owner_frame(r["marks"], _raw(r))
        if own is not None:
            if r["animal_by_owner"] and r["animal_frame"] == own:
                continue  # already embedded from that frame
        elif r["animal_by_owner"]:
            pass  # the circle left its burst frame: undo the owner's pick
        elif not (r["visitor"] == "none" and r["animal_frame"] is None):
            continue
        out.append({**dict(r), "owner_frame": own})
    return sorted(out, key=lambda r: r["owner_frame"] is None)  # the owner's picks first: one frame each, not six


def _save_pick(conn, visit_id: int, emb, frame: int | None, score: float | None, by_owner: bool) -> None:
    conn.execute("""UPDATE embeddings SET animal_vec = ?, animal_frame = ?, animal_score = ?, animal_by_owner = ?
                    WHERE visit_id = ?""",
                 (None if emb is None else np.asarray(emb, np.float16).tobytes(), frame,
                  None if emb is None or score is None else round(score, 4), 1 if by_owner else None, visit_id))
    conn.commit()


def pick_frames(conn, limit: int | None = None, busy: threading.Event | None = None, clf=None) -> int:
    """Embed burst frames for training (see pending_picks), ~6 s a frame on a Pi 4:
    - the owner circled the animal on burst frame k: embed k's view (classifier.presence_patch, animal_by_owner),
      score it if a model is trained (shown, not used to judge it);
    - a false trigger the owner corrected to an animal: embed the view of every saved burst frame and keep the
      one the current model scores highest (animal_vec, animal_frame, animal_score), so training sees the frame
      where the animal is in the crop box rather than the sharpest one;
    - an owner's pick whose circle went back to the photo: forget it (a false trigger then gets looked through
      as above on a later round).
    Stops (and tries again later) while `busy` is set. Returns how many visits were done."""
    rows = pending_picks(conn)
    if not rows or (busy is not None and busy.is_set()):
        return 0
    clf = clf or classifier.load()
    if clf is None:
        return 0
    m = model(conn)
    if m is None:  # nothing to score with: only the owner's picks (and undoing them) can go ahead
        rows = [r for r in rows if r["owner_frame"] is not None or r["animal_by_owner"]]
    done = 0
    for r in rows[:limit]:
        if not _pick_one(conn, r, m, clf, busy):
            return done
        done += 1
    return done


def _pick_one(conn, r: dict, m: dict | None, clf, busy: threading.Event | None = None) -> bool:
    """pick_frames for one pending visit (a pending_picks row), scored with model `m` (or None). False when `busy`
    stopped it before anything was saved."""
    import cv2

    from . import images

    if r["owner_frame"] is None and r["animal_by_owner"]:  # the circle left its burst frame: back to the automatic pick
        _save_pick(conn, r["id"], None, None, None, False)
        log.info("visit %d: your circle is back on the photo, so its burst frame is forgotten", r["id"])
        return True
    raw = json.loads(r["raw_json"] or "{}") or {}
    z = visit_zones(raw, conn, r["captured_at"])
    paths = images.burst_paths(r["image_path"])
    want = [r["owner_frame"]] if r["owner_frame"] is not None else range(1, len(paths) + 1)
    emb, best, n = None, -1.0, 0
    for i in want:
        if busy is not None and busy.is_set():
            return False
        img = cv2.imread(str(config.IMAGES_DIR / paths[i - 1])) if i <= len(paths) else None
        if img is None:
            continue
        e = clf.embed(classifier.presence_patch(img, z, raw.get("trigger"))[0])
        s = score(m, e) if m is not None else None
        if s is None or s > best:
            emb, best, n = e, (-1.0 if s is None else s), i
    if r["owner_frame"] is not None:
        # Recorded even when the frame is gone (vec None), so it isn't tried again until the circle moves.
        _save_pick(conn, r["id"], emb, r["owner_frame"], best if emb is not None and m is not None else None, True)
        log.info("visit %d: burst frame %d (the owner's circle) %s", r["id"], r["owner_frame"],
                 "embedded" if emb is not None else "is gone")
    else:
        _save_pick(conn, r["id"], emb, n, None if emb is None else best, False)
        if n:
            log.info("visit %d: burst frame %d shows the animal best (score %.3f)", r["id"], n, best)
        else:
            log.info("visit %d: no burst frames to look through", r["id"])
    return True


PROGRESS_EVERY, PROGRESS_SECONDS = 25, 30  # backfill prints a progress line this often (whichever comes first)
_clock = time.monotonic  # backfill's clock (tests move it on)


def progress_line(n: int, total: int, failed: int, elapsed: float) -> str:
    """"142/683 done, 3 failed, ~4 min left"."""
    left = elapsed / n * (total - n) if n else 0.0
    return f"{n}/{total} done, {failed} failed, ~{round(left / 60)} min left"


def pick_line(n: int, total: int, elapsed: float) -> str:
    """"bursts: 3/20 visits looked through, ~2 min left" (a redo's picks before gate comes back)."""
    left = elapsed / n * (total - n) if n else 0.0
    return f"bursts: {n}/{total} visits looked through, ~{round(left / 60)} min left"


def _restore(conn) -> str | None:
    value = db.get_value(conn, RESTORE_KEY)
    return value if value in RESTORES else None


def hold_gate() -> str | None:
    """Hold the check for a redo: HOLD_KEY set (no rain probes, no burst picks) and gate mode switched to shadow,
    RESTORE_KEY saying what to put back ("gate", or "env" when gate came from .env). A redo that starts in the
    shadow an earlier partial one left keeps that earlier record; one whose mode the owner changed since (not
    shadow any more) drops it. Returns what a complete redo will restore, or None. Left at shadow if the process
    dies: the safe way, shown on Settings; the hold runs out after HOLD_MAX."""
    with db.session() as conn:
        db.set_value(conn, HOLD_KEY, str(_now()))
        restore, now = _restore(conn), mode(conn)
        if now == "gate":
            restore = "gate" if db.get_value(conn, "presence_mode") == "gate" else "env"
            db.set_value(conn, RESTORE_KEY, restore)
            db.set_value(conn, "presence_mode", "shadow")
            print("The Pi animal check was in gate mode: switched to shadow while frames are re-embedded, so a model "
                  "trained on the first few can't skip Claude. It goes back to gate once every frame is redone "
                  "and the bursts waiting to be looked through are picked.", flush=True)
            return restore
        if restore is None:
            db.delete_value(conn, RESTORE_KEY)
            return None
        if now == "shadow":
            print("An earlier redo that didn't finish left the Pi animal check in shadow; it goes back to gate once "
                  "this one finishes.", flush=True)
            return restore
        db.delete_value(conn, RESTORE_KEY)
    print(f"The Pi animal check's mode was changed to {now} since an earlier redo; left as it is.", flush=True)
    return None


def check_in() -> None:
    """A redo still running: the hold's time moved on, so it doesn't run out mid-redo."""
    with db.session() as conn:
        if db.get_value(conn, HOLD_KEY) is not None:
            db.set_value(conn, HOLD_KEY, str(_now()))


def still_to_redo(conn, failed: set[int] = frozenset()) -> int:
    """Frames embedded in an older view that a redo could still do (their photo is there), apart from `failed`
    (tried in this run and unreadable: a complete run with failures is still complete)."""
    rows = conn.execute(f"""SELECT id FROM visits WHERE {db.FULL_FRAME} AND id IN
                              (SELECT visit_id FROM embeddings WHERE view IS NOT {classifier.PRESENCE_VIEW})""")
    return sum(1 for r in rows if r["id"] not in failed)


def gate_due(conn) -> bool:
    """Whether a complete redo would put gate back: a mode to restore and the shadow the hold set still in place."""
    return _restore(conn) is not None and db.get_value(conn, "presence_mode") == "shadow"


def pick_all(conn, clf, beat=None) -> set[int]:
    """A redo's picks before gate comes back: retrain on the redone frames, then pick_frames each pending visit
    once (new ones that turn up meanwhile too), with progress lines (calling `beat`), then retrain again, so no
    visitor whose pick the redo forgot is missing from the model gate mode starts with. With no model to score a
    false trigger's burst it's looked at and left for the watcher. Returns the ids looked at."""
    m = retrain(conn)["model"]
    tried: set[int] = set()
    start = last = _clock()
    while True:
        todo = [r for r in pending_picks(conn) if r["id"] not in tried]
        if not todo:
            break
        r = todo[0]
        tried.add(r["id"])
        if m is not None or r["owner_frame"] is not None or r["animal_by_owner"]:
            _pick_one(conn, r, m, clf)
        n, total, now = len(tried), len(tried) + len(todo) - 1, _clock()
        if n < total and (n % PROGRESS_EVERY == 0 or now - last >= PROGRESS_SECONDS):
            print(pick_line(n, total, now - start), flush=True)
            if beat is not None:
                beat()
            last = now
    if tried:
        print(pick_line(len(tried), len(tried), _clock() - start), flush=True)
        retrain(conn)
    return tried


def release_gate(left: int = 0, picks_left: int = 0) -> None:
    """The end of a redo. Complete (`left` frames still to redo and `picks_left` bursts still to look through
    both 0): the hold removed, the mode in RESTORE_KEY put back unless someone changed it meanwhile (the setting
    is no longer shadow), and the mode it's now in printed. Partial (Ctrl-C, --limit, an error): the hold kept
    (from now, for HOLD_MAX) and the record too, and a line says what stays paused and how to finish."""
    with db.session() as conn:
        restore = _restore(conn)
        changed = restore is not None and db.get_value(conn, "presence_mode") != "shadow"
        if changed:
            db.delete_value(conn, RESTORE_KEY)
            print("The Pi animal check's mode was changed during the redo; left as it is now.", flush=True)
            restore = None
        if left or picks_left:
            db.set_value(conn, HOLD_KEY, str(_now()))
            todo = []
            if left:
                todo.append(f"{left} frame{'s' * (left != 1)} still to redo")
            if picks_left:
                todo.append(f"{picks_left} burst{'s' * (picks_left != 1)} still to look through")
            hours = f"{HOLD_MAX / 3600:g} hour{'s' * (HOLD_MAX != 3600)}"
            paused = (f"the watcher sends no rain probes and picks no burst frames for {hours} from now, "
                      "when the hold lapses")
            if restore:
                print(f"Not finished: {' and '.join(todo)}. The Pi animal check stays in shadow until a redo "
                      f"finishes, and {paused} (shadow stays). Run --backfill --redo again to finish and put gate "
                      "back.", flush=True)
            else:
                print(f"Not finished: {' and '.join(todo)}. Until a redo finishes, {paused}. Run --backfill --redo "
                      "again to finish.", flush=True)
            return
        db.delete_value(conn, HOLD_KEY)
        db.delete_value(conn, RESTORE_KEY)
        if restore == "gate":
            db.set_value(conn, "presence_mode", "gate")
        elif restore == "env":
            db.delete_value(conn, "presence_mode")
        now = mode(conn)
    print(f"The Pi animal check is {'back' if restore else 'now'} in {now} mode.", flush=True)


def backfill(limit: int | None = None, redo: bool = False, clf=None) -> int:
    """Embeddings of the current view (~6 s each on a Pi 4), newest first, for stored frames that have none and,
    with `redo`, for frames embedded in an older view (the check is held meanwhile, see hold_gate; gate mode comes
    back only from a complete redo, see release_gate, after pick_all). Visits whose photo is gone (deleted to save
    space, or only the thumbnail left) are left as they were and counted apart, never in `limit`. Each visit is
    seen with its own zones (raw_json.zones, else the zones at the time, else today's) and trigger. A redo with
    nothing left to embed still finishes what an earlier partial one left (the picks, the hold). Prints progress
    as it goes; returns how many were embedded."""
    clf = clf or classifier.load()
    if clf is None:
        raise SystemExit("the classifier is off (CLASSIFIER=off)")
    old = f" OR id IN (SELECT visit_id FROM embeddings WHERE view IS NOT {classifier.PRESENCE_VIEW})" if redo else ""
    with db.session() as conn:
        rows = conn.execute(f"""SELECT id, image_path, raw_json, captured_at, {db.FULL_FRAME} AS has_photo
                                FROM visits WHERE id NOT IN (SELECT visit_id FROM embeddings){old}
                                ORDER BY id DESC""").fetchall()
        earlier = db.get_value(conn, HOLD_KEY) is not None or db.get_value(conn, RESTORE_KEY) is not None
    gone = sum(1 for r in rows if not r["has_photo"])
    rows = [r for r in rows if r["has_photo"]][:limit]
    holding = bool(redo and (rows or earlier))
    if holding:
        hold_gate()
    failed: set[int] = set()
    picks_done = False
    try:
        done = _embed_rows(rows, clf, gone, failed, check_in if holding else None) if rows or not holding else 0
        if holding:
            with db.session() as conn:
                if not still_to_redo(conn, failed) and gate_due(conn):
                    pick_all(conn, clf, check_in)
            picks_done = True
    finally:
        if holding:
            with db.session() as conn:
                left = still_to_redo(conn, failed)
                picks_left = len(pending_picks(conn)) if not left and not picks_done and gate_due(conn) else 0
            release_gate(left, picks_left)
    return done


def _embed_rows(rows: list, clf, gone: int = 0, failed_ids: set[int] | None = None, beat=None) -> int:
    """backfill's loop: how many were embedded (the ids that failed go in `failed_ids`). Prints progress lines,
    calling `beat` with each; the last one always, once, with the `gone` visits whose photos are gone."""
    import cv2

    total, done, failed = len(rows), 0, 0
    start = last = _clock()
    for i, r in enumerate(rows, 1):
        try:
            img = cv2.imread(str(config.IMAGES_DIR / r["image_path"])) if r["image_path"] else None
            if img is None:
                raise FileNotFoundError(r["image_path"])
            raw = json.loads(r["raw_json"] or "{}") or {}
            with db.session() as conn:
                z = visit_zones(raw, conn, r["captured_at"])
            emb = clf.embed(classifier.presence_patch(img, z, raw.get("trigger"))[0])
            with db.session() as conn:
                store(conn, r["id"], emb)
            done += 1
        except Exception as e:
            failed += 1
            if failed_ids is not None:
                failed_ids.add(r["id"])
            log.warning("visit %d: not embedded (%s)", r["id"], e)
        now = _clock()
        if i < total and (i % PROGRESS_EVERY == 0 or now - last >= PROGRESS_SECONDS):
            print(progress_line(i, total, failed, now - start), flush=True)
            if beat is not None:
                beat()
            last = now
    photos_gone = f", {gone} photos gone (left as they were)" if gone else ""
    print(progress_line(total, total, failed, _clock() - start) + photos_gone, flush=True)
    return done


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m birdbrain.presence")
    parser.add_argument("--backfill", action="store_true", help="embed stored frames that have no embedding")
    parser.add_argument("--limit", type=int, help="with --backfill: at most this many, newest first")
    parser.add_argument("--redo", action="store_true",
                        help="with --backfill: also re-embed frames embedded in an older view (the old center cut)")
    parser.add_argument("--pick", action="store_true",
                        help="look through the bursts of false triggers corrected to an animal, and embed the frames "
                             "the owner circled")
    args = parser.parse_args()
    if args.redo and not args.backfill:
        parser.error("--redo goes with --backfill (python -m birdbrain.presence --backfill --redo)")
    with db.session() as conn:
        db.init_db(conn)
    if args.backfill:
        print(f"embedded {backfill(args.limit, args.redo)} frames")
    if args.pick:
        logging.basicConfig(level=logging.INFO, format="%(message)s")  # one line per visit as it goes
        with db.session() as conn:
            todo = len(pending_picks(conn))
            print(f"{todo} visits whose burst needs looking at (up to ~36 s each on a Pi 4)")
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
          + (f"; {m['left_out']} visitors left out (no frame shows them in the Pi's view)" if m["left_out"] else ""))


if __name__ == "__main__":
    main()
