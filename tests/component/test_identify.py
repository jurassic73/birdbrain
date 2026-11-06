"""Component tests of birdbrain/identify.py: one real local dependency, a temp database or files on disk."""

import base64
import io

import anthropic
import httpx
import pytest
from PIL import Image

from helpers.identify import WHEN, fake_ask, jpeg, sighting
from birdbrain import identify
from birdbrain.identify import HAIKU, SONNET


def decoded_size(b64: str) -> tuple[int, int]:
    return Image.open(io.BytesIO(base64.standard_b64decode(b64))).size


def test_prepare_shrinks_to_the_api_limit():
    assert max(decoded_size(identify.prepare(jpeg()))) == identify.MAX_EDGE


def test_prepare_crops_before_shrinking():
    w, h = decoded_size(identify.prepare(jpeg(), crop=(0.25, 0.25, 0.5, 0.5)))
    assert (w, h) == (960, 540)  # half of 1920x1080, already under the limit


@pytest.mark.parametrize("confidence", ["low", "medium"])
@pytest.mark.smoke
def test_unsure_bird_escalates_to_sonnet_with_all_burst_frames(fake_ask, confidence):
    calls = fake_ask({HAIKU: sighting(confidence=confidence), SONNET: sighting(species="Song Sparrow")})
    r = identify.identify(jpeg(), WHEN, extra_frames=[jpeg(), jpeg()])
    assert (r.sighting.species, r.model, r.escalated, r.needs_review) == ("Song Sparrow", SONNET, True, False)
    assert [(m, n) for m, _, n in calls] == [(HAIKU, 1), (SONNET, 3)]
    assert r.raw["sonnet"]["frames"] == 3


def test_a_doubtful_empty_frame_gets_sonnets_look_and_waits_for_review_if_still_doubtful(fake_ask):
    doubt = sighting(visitor="none", species=None, confidence="low")
    calls = fake_ask({HAIKU: doubt, SONNET: doubt})
    r = identify.identify(jpeg(), WHEN, extra_frames=[jpeg(), jpeg()])
    assert [(m, n) for m, _, n in calls] == [(HAIKU, 1), (SONNET, 3)]
    assert (r.sighting.visitor, r.model, r.escalated, r.needs_review) == ("none", SONNET, True, True)


def test_sonnet_settling_a_doubtful_empty_frame_either_way_needs_no_review(fake_ask):
    fake_ask({HAIKU: sighting(visitor="none", species=None, confidence="low"), SONNET: sighting()})
    r = identify.identify(jpeg(), WHEN)
    assert (r.sighting.species, r.needs_review) == ("Dark-eyed Junco", False)
    fake_ask({HAIKU: sighting(visitor="none", species=None, confidence="low"),
              SONNET: sighting(visitor="none", species=None, confidence="high")})
    r = identify.identify(jpeg(), WHEN)
    assert (r.sighting.visitor, r.needs_review) == ("none", False)


def test_names_are_never_downloaded_while_identifying(fake_ask, monkeypatch, tmp_path):
    from birdbrain import config, names
    monkeypatch.setattr(names, "_names", None)       # nothing loaded...
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)  # ...and no list on disk (a download would trip no_network)
    fake_ask({HAIKU: sighting(species="black-capped chickadee")})
    r = identify.identify(jpeg(), WHEN, confirmed={"Black-capped Chickadee"})
    assert r.sighting.species == "Black-capped Chickadee"      # the yard's own names still work
    fake_ask({HAIKU: sighting(species="black-capped chickadee")})  # a fresh answer: the last one was renamed in place
    assert identify.identify(jpeg(), WHEN).sighting.species == "black-capped chickadee"
    assert names._names is None and names.canonical("house finch", fetch=False) is None


def test_the_visit_records_the_model_that_actually_answered(fake_ask):
    fake_ask({HAIKU: sighting(confidence="low"), SONNET: (sighting(), {"model": "claude-opus-4-8"})})
    r = identify.identify(jpeg(), WHEN)
    assert (r.model, r.escalated) == ("claude-opus-4-8", True) and r.raw["sonnet"]["model"] == "claude-opus-4-8"
    fake_ask({HAIKU: (sighting(), {"model": "claude-haiku-5-5-20270101"})})
    assert identify.identify(jpeg(), WHEN).model == "claude-haiku-5-5-20270101"


def test_sonnet_error_keeps_the_haiku_answer_for_review(monkeypatch):
    def ask(model, images_b64, captured_at, lessons=None, site=None, night=False):
        if model == SONNET:
            raise anthropic.APIConnectionError(request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"))
        return sighting(confidence="medium"), {"model": model}

    monkeypatch.setattr(identify, "ask", ask)
    r = identify.identify(jpeg(), WHEN, extra_frames=[jpeg()])
    assert (r.sighting.species, r.model, r.escalated, r.needs_review) == ("Dark-eyed Junco", HAIKU, False, True)
    assert r.raw["sonnet"]["answer"] is None and r.raw["sonnet"]["error"].startswith("APIConnectionError")
    assert r.raw["haiku"]["answer"]["species"] == "Dark-eyed Junco"


def test_local_check_only_runs_when_sonnet_would_be_skipped(fake_ask):
    asked = []
    fake_ask({HAIKU: sighting(confidence="high"), SONNET: sighting()})
    identify.identify(jpeg(), WHEN, confirmed={"Dark-eyed Junco"}, local_check=lambda c: asked.append(c))
    fake_ask({HAIKU: sighting(species="Song Sparrow", confidence="medium"), SONNET: sighting()})
    identify.identify(jpeg(), WHEN, confirmed={"Dark-eyed Junco"}, local_check=lambda c: asked.append(c))
    assert asked == []
