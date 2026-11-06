"""Unit tests of birdbrain/species_model: pure numpy, tiny data, no files or database."""

import numpy as np

from birdbrain import species_model as sm


def blobs(n_per=30, seed=0):
    """Three well-separated species in 8 dimensions, interleaved in time order."""
    rng = np.random.default_rng(seed)
    centers = np.eye(8, dtype=np.float32)[:3]
    X, labels = [], []
    for i in range(n_per):
        for j, name in enumerate(("junco", "finch", "chickadee")):
            v = centers[j] + 0.05 * rng.standard_normal(8).astype(np.float32)
            X.append(v / np.linalg.norm(v))
            labels.append(name)
    return np.stack(X), labels


def test_walk_forward_learns_separable_species_and_scores_only_later_visits():
    X, labels = blobs()
    scored = sm.walk_forward(X, labels, blocks=3, steps=150)
    assert scored and min(s["i"] for s in scored) >= int(len(labels) * 0.4)
    assert all(s["guess"] == s["truth"] for s in scored)
    assert all(0 < s["p"] <= 1 for s in scored)


def test_a_species_with_too_few_earlier_visits_is_not_a_class():
    X, labels = blobs(n_per=20)
    labels = ["rare" if i == 55 else name for i, name in enumerate(labels)]  # one visit, after the first fold
    scored = sm.walk_forward(X, labels, blocks=2, steps=100)
    rare = [s for s in scored if s["truth"] == "rare"]
    assert rare and rare[0]["guess"] != "rare"


def test_at_bar_counts_named_and_wrong():
    scored = [{"truth": "a", "guess": "a", "p": 0.99}, {"truth": "a", "guess": "b", "p": 0.96},
              {"truth": "b", "guess": "b", "p": 0.6}, {"truth": "b", "guess": "b", "p": 0.4}]
    r = sm.at_bar(scored, 0.95)
    assert (r["named"], r["wrong"], r["coverage"]) == (2, 1, 0.5)
    assert sm.at_bar([], 0.9)["coverage"] == 0.0


def test_per_species_counts_right_guesses():
    scored = [{"truth": "a", "guess": "a", "p": 1}, {"truth": "a", "guess": "b", "p": 1}, {"truth": "b", "guess": "b", "p": 1}]
    assert sm.per_species(scored) == {"a": {"visits": 2, "right": 1}, "b": {"visits": 1, "right": 1}}


def test_baseline_uses_biocips_top_guess_and_skips_visits_without_one():
    rows = [{"clf_top": {"name": "junco", "score": 0.9}}, {"clf_top": None}]
    scored = [{"i": 0, "truth": "junco", "guess": "x", "p": 0.5}, {"i": 1, "truth": "finch", "guess": "x", "p": 0.5}]
    assert sm.baseline(rows, scored) == [{"i": 0, "truth": "junco", "guess": "junco", "p": 0.9}]


def test_format_report_has_a_row_per_bar():
    rows = [{"label": "junco", "checked": True}] * 3
    scored = [{"i": 0, "truth": "junco", "guess": "junco", "p": 0.97}]
    text = sm.format_report(rows, scored, scored)
    assert text.count("\n") > len(sm.BARS) and "junco" in text and "0.95" in text
