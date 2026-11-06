"""Unit tests of birdbrain/identify: pure functions, no files, database or network."""

import io
from datetime import datetime

import pytest
from PIL import Image

from helpers.also import junco
from birdbrain import identify
from birdbrain.identify import HAIKU, SONNET, Also, Circle, Sighting


WHEN = datetime(2026, 10, 25, 9, 30)


def jpeg() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (1920, 1080), (90, 120, 200)).save(buf, "JPEG")
    return buf.getvalue()


def bushtit(confidence="high", **kw) -> Also:
    return Also(**{"visitor": "bird", "species": "Bushtit", "count": 2, "confidence": confidence,
                   "circle": Circle(x=0.2, y=0.2, r=0.05), **kw})


@pytest.fixture
def fake_ask(monkeypatch):
    calls = []

    def install(answers):
        def ask(model, images_b64, captured_at, lessons=None, site=None, night=False):
            calls.append(model)
            return answers[model], {"model": model}
        monkeypatch.setattr(identify, "ask", ask)
        return calls
    return install


def test_other_animals_are_kept_with_circles_on_the_full_frame(fake_ask):
    fake_ask({HAIKU: junco(also=[bushtit()])})
    r = identify.identify(jpeg(), WHEN, crop=(0.2, 0.4, 0.5, 0.5))
    assert r.raw["also"] == [{"visitor": "bird", "species": "Bushtit", "count": 2, "confidence": "high",
                              "circle": {"x": 0.3, "y": 0.5, "r": 0.025, "claude": True}}]
    assert (r.escalated, r.needs_review) == (False, False)


def test_also_drops_repeats_of_the_main_animal_and_of_each_other():
    s = junco(also=[bushtit(), bushtit(count=1), bushtit(species="dark-eyed junco")])
    assert [a["species"] for a in identify.also_rows(s)] == ["Bushtit"]
    assert identify.also_rows(Sighting(visitor="none", species=None, count=0, confidence="high", notes="",
                                       circle=None, also=[bushtit()])) == []


def test_a_low_guess_at_the_second_animal_gets_sonnets_look(fake_ask):
    calls = fake_ask({HAIKU: junco(also=[bushtit("low")]), SONNET: junco(also=[bushtit("medium")])})
    r = identify.identify(jpeg(), WHEN)
    assert calls == [HAIKU, SONNET] and (r.model, r.needs_review) == (SONNET, False)
    assert r.raw["also"][0]["confidence"] == "medium"


def test_still_low_on_the_second_animal_goes_to_review(fake_ask):
    fake_ask({HAIKU: junco(also=[bushtit("low")]), SONNET: junco(also=[bushtit("low")])})
    assert identify.identify(jpeg(), WHEN).needs_review is True


def test_the_prompt_asks_for_every_kind():
    assert "also:" in identify.SYSTEM and "list the others here" not in identify.SYSTEM
