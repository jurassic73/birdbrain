"""Repo-wide check: the .env.example template documents every knob the finder chain reads."""
import re
from pathlib import Path

from birdbrain import config

ENV_EXAMPLE = Path(config.ROOT) / ".env.example"


def test_env_example_lists_the_finder_knobs():
    text = ENV_EXAMPLE.read_text()
    for name in ("FINDERS", "FINDER_YOLOX_MIN_SCORE"):
        assert re.search(rf"^#?\s*{name}=", text, re.M), f"{name} is not in .env.example"
