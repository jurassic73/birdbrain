"""Shared helpers for the identify tests (used by more than one test module)."""
import io
from datetime import datetime

import pytest
from PIL import Image

from birdbrain import identify
from birdbrain.identify import Sighting


WHEN = datetime(2026, 10, 25, 9, 30)


def jpeg(size=(1920, 1080)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, (90, 120, 200)).save(buf, "JPEG")
    return buf.getvalue()


def sighting(visitor="bird", species="Dark-eyed Junco", confidence="high", notes="", also=()) -> Sighting:
    return Sighting(visitor=visitor, species=species, count=1, confidence=confidence, notes=notes, circle=None,
                    also=list(also))


@pytest.fixture
def fake_ask(monkeypatch):
    """Replace the API call with scripted answers per model; records which models were asked. An answer can be a
    (sighting, meta) pair when the test needs to script the call details too."""
    calls = []

    def install(answers: dict):
        def ask(model, images_b64, captured_at, lessons=None, site=None, night=False):
            calls.append((model, lessons, len(images_b64)))
            answer = answers[model]
            if isinstance(answer, tuple):
                return answer
            return answer, {"model": model}
        monkeypatch.setattr(identify, "ask", ask)
        return calls

    return install
