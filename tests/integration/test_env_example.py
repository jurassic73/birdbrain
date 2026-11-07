"""Repo-wide check: the .env.example template documents every knob the finder chain reads."""
import re
from pathlib import Path

from birdbrain import config

ENV_EXAMPLE = Path(config.ROOT) / ".env.example"


def test_env_example_lists_the_finder_knobs():
    text = ENV_EXAMPLE.read_text()
    for name in ("FINDERS", "FINDER_YOLOX_MIN_SCORE"):
        assert re.search(rf"^#?\s*{name}=", text, re.M), f"{name} is not in .env.example"


def test_env_example_lists_the_guests_keys():
    text = ENV_EXAMPLE.read_text()
    for name in ("AUDIENCE_TOKEN", "AUDIENCE_SOURCE_URL", "AUDIENCE_DB", "AUDIENCE_COUNT"):
        assert re.search(rf"^#?\s*{name}=", text, re.M), f"{name} is not in .env.example"
