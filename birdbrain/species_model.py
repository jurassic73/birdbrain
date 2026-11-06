"""Offline study of a yard-trained species model: a small softmax regression on BioCLIP 2's tight embeddings.

BioCLIP 2 itself is frozen; only a layer on top can learn this yard's birds (mostly junco, House Finch,
chickadee). The input is `embeddings.circle_vec`, the square around Claude's circle (the tight view: the wide
bowl view was the bottleneck in the 2026-11-04 study). Nothing here runs in the watcher, and nothing is written:
it reads a database and prints a report.

    uv run python -m birdbrain.species_model [--min-per-class 5] [--blocks 5] [--owner-only]

Walk-forward: visits are sorted by time and the later ones are scored in blocks by a model fitted only on
everything before the block, the way it would run live. The report gives, per confidence bar, how many visits
the model would name (coverage) and how many of those it got wrong, next to BioCLIP's own top guess on the same
visits (`raw_json.calls.classifier`, also made on Claude's circle).

Labels: the owner's correction, else Claude's species; a visit Claude was unsure of (low confidence, needs
review) and not checked by the owner is left out, so the labels are as clean as the data allows.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter

import numpy as np

from . import classifier, db

MIN_PER_CLASS = 5      # a species with fewer training visits is not a class (the model abstains by never naming it)
BARS = (0.5, 0.7, 0.8, 0.9, 0.95, 0.98)
L2 = 1e-3
STEPS = 300
LR = 0.5
SCALE = 10.0           # embeddings are unit length: spread them out so gradient descent converges sooner


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit(X: np.ndarray, y: np.ndarray, k: int, steps: int = STEPS, lr: float = LR, l2: float = L2
        ) -> tuple[np.ndarray, np.ndarray]:
    """Multinomial logistic regression by plain gradient descent, classes balanced (juncos outnumber the rest).
    X is n x d, y holds class numbers 0..k-1. Returns coefficients (d x k) and biases (k)."""
    X = np.asarray(X, np.float32) * np.float32(SCALE)
    n = len(y)
    onehot = np.zeros((n, k), np.float32)
    onehot[np.arange(n), y] = 1.0
    counts = np.maximum(onehot.sum(0), 1.0)
    weight = (n / (k * counts))[y].astype(np.float32)[:, None]  # rarer class, bigger weight
    coef = np.zeros((X.shape[1], k), np.float32)
    bias = np.zeros(k, np.float32)
    for _ in range(steps):
        g = (softmax(X @ coef + bias) - onehot) * weight
        coef -= np.float32(lr) * ((X.T @ g) / n + np.float32(l2) * coef)
        bias -= np.float32(lr) * g.sum(0) / n
    return coef * np.float32(SCALE), bias


def predict(X: np.ndarray, coef: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return softmax(np.asarray(X, np.float32) @ coef + bias)


def walk_forward(X: np.ndarray, labels: list[str], blocks: int = 5, min_per_class: int = MIN_PER_CLASS,
                 first: float = 0.4, steps: int = STEPS) -> list[dict]:
    """Score visits (already in time order) from the `first` fraction on, in `blocks` chunks, each by a model
    fitted on everything before it. A species with fewer than `min_per_class` visits before the chunk isn't a
    class then. Returns one dict per scored visit: i, truth, guess (None when its class wasn't known) and p."""
    n = len(labels)
    start = max(int(n * first), 1)
    edges = np.linspace(start, n, blocks + 1).astype(int)
    out: list[dict] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi <= lo:
            continue
        seen = Counter(labels[:lo])
        classes = sorted(c for c, m in seen.items() if m >= min_per_class)
        if len(classes) < 2:
            continue
        index = {c: j for j, c in enumerate(classes)}
        rows = [i for i in range(lo) if labels[i] in index]
        y = np.array([index[labels[i]] for i in rows])
        coef, bias = fit(X[rows], y, len(classes), steps=steps)
        probs = predict(X[lo:hi], coef, bias)
        for i, p in zip(range(lo, hi), probs):
            j = int(p.argmax())
            out.append({"i": i, "truth": labels[i], "guess": classes[j], "p": float(p[j])})
    return out


def at_bar(scored: list[dict], bar: float) -> dict:
    """What a bar would have done: visits named (p at or over the bar), how many of those were wrong, coverage."""
    named = [s for s in scored if s["p"] >= bar]
    wrong = [s for s in named if s["guess"] != s["truth"]]
    return {"bar": bar, "scored": len(scored), "named": len(named), "wrong": len(wrong),
            "coverage": len(named) / len(scored) if scored else 0.0}


def per_species(scored: list[dict]) -> dict[str, dict]:
    """Per true species: visits, and how many the model's top guess got right (at any confidence)."""
    out: dict[str, dict] = {}
    for s in scored:
        row = out.setdefault(s["truth"], {"visits": 0, "right": 0})
        row["visits"] += 1
        row["right"] += s["guess"] == s["truth"]
    return out


def baseline(rows: list[dict], scored: list[dict]) -> list[dict]:
    """BioCLIP's own top guess on Claude's circle for the same visits, as `scored` entries (p = its top score)."""
    out = []
    for s in scored:
        top = (rows[s["i"]].get("clf_top") or None)
        if top:
            out.append({"i": s["i"], "truth": s["truth"], "guess": top["name"], "p": float(top["score"])})
    return out


def load_rows(conn, owner_only: bool = False) -> list[dict]:
    """Bird visits with a tight embedding of Claude's circle and a trustworthy label, oldest first."""
    sql = f"""SELECT v.id, v.captured_at, v.species, v.corrected_species, v.confidence, v.needs_review,
                     v.confirmed_at, v.raw_json, e.circle_vec
              FROM visits v JOIN embeddings e ON e.visit_id = v.id
              WHERE e.circle_vec IS NOT NULL AND e.circle_view IS {classifier.TIGHT_VIEW}
                AND COALESCE(v.corrected_visitor, v.visitor) = 'bird'
                AND COALESCE(v.corrected_species, v.species) IS NOT NULL
                AND {db.REAL} ORDER BY v.captured_at"""
    rows = []
    for r in conn.execute(sql):
        checked = bool(r["corrected_species"] or r["confirmed_at"])
        if owner_only and not checked:
            continue
        if not checked and (r["needs_review"] or r["confidence"] == "low"):
            continue
        try:
            calls = (json.loads(r["raw_json"] or "{}") or {}).get("calls") or {}
        except ValueError:
            calls = {}
        top = ((calls.get("classifier") or {}).get("top") or [None])[0]
        vec = np.frombuffer(r["circle_vec"], np.float16).astype(np.float32)
        norm = float(np.linalg.norm(vec))
        if not vec.size or not np.isfinite(norm) or norm == 0:
            continue
        rows.append({"id": r["id"], "label": r["corrected_species"] or r["species"], "vec": vec / norm,
                     "checked": checked, "clf_top": top})
    return rows


def format_report(rows: list[dict], scored: list[dict], base: list[dict]) -> str:
    lines = [f"{len(rows)} labelled bird visits with a tight embedding "
             f"({sum(r['checked'] for r in rows)} checked by the owner); {len(scored)} scored walk-forward.",
             "", "species in the data: " + ", ".join(f"{c} {m}" for c, m in
                                                     Counter(r["label"] for r in rows).most_common()),
             "", f"{'bar':>5} | {'yard model: named / wrong':>26} | {'BioCLIP 2 alone: named / wrong':>31}"]
    for bar in BARS:
        a, b = at_bar(scored, bar), at_bar(base, bar)
        lines.append(f"{bar:>5} | {a['named']:>5} ({a['coverage']:>4.0%}) / {a['wrong']:<3}"
                     f"{'':>10} | {b['named']:>5} ({b['coverage']:>4.0%}) / {b['wrong']:<3}")
    lines += ["", "top guess right, per true species (yard model):"]
    for name, row in sorted(per_species(scored).items(), key=lambda kv: -kv[1]["visits"]):
        lines.append(f"  {name:<28} {row['right']:>3}/{row['visits']:<3}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m birdbrain.species_model", description=__doc__.split("\n")[0])
    parser.add_argument("--min-per-class", type=int, default=MIN_PER_CLASS)
    parser.add_argument("--blocks", type=int, default=5)
    parser.add_argument("--owner-only", action="store_true", help="only visits the owner corrected or confirmed")
    args = parser.parse_args()
    with db.session() as conn:
        rows = load_rows(conn, args.owner_only)
    if len(rows) < 20:
        print(f"only {len(rows)} usable visits: run python -m birdbrain.classifier --backfill-circle-embeddings first")
        return 1
    X = np.stack([r["vec"] for r in rows])
    scored = walk_forward(X, [r["label"] for r in rows], args.blocks, args.min_per_class)
    print(format_report(rows, scored, baseline(rows, scored)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
