"""Unit tests of birdbrain/web/steps: pure functions, no files, database or network."""


def test_reasons_use_whole_percents_like_the_pills():
    from birdbrain.web import steps
    assert steps._pct(0.528) == "53%" and steps._pct(0.8) == "80%" and steps._pct(0.95) == "95%" and steps._pct(1) == "100%"
    look = {"top": [{"name": "Bushtit", "score": 0.528}]}
    assert steps.miss_text("score", look) == "its top guess was 53%, under the 80% it needs"
    assert steps.miss_text("score", look, view="tight") == "its top guess was 53%, under the 95% it needs"
    assert steps._share(0.5) == "half"
