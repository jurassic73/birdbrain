import base64
import io
from datetime import datetime

import anthropic
import httpx
import pydantic
import pytest
from PIL import Image

from birdbrain import identify
from birdbrain.identify import HAIKU, SONNET, Sighting

WHEN = datetime(2026, 10, 25, 9, 30)


def jpeg(size=(1920, 1080)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, (90, 120, 200)).save(buf, "JPEG")
    return buf.getvalue()


def decoded_size(b64: str) -> tuple[int, int]:
    return Image.open(io.BytesIO(base64.standard_b64decode(b64))).size


def test_prepare_shrinks_to_the_api_limit():
    assert max(decoded_size(identify.prepare(jpeg()))) == identify.MAX_EDGE


def test_prepare_crops_before_shrinking():
    w, h = decoded_size(identify.prepare(jpeg(), crop=(0.25, 0.25, 0.5, 0.5)))
    assert (w, h) == (960, 540)  # half of 1920x1080, already under the limit


def test_system_prompt_appends_lessons_only_when_there_are_some():
    assert identify.system_prompt(None) == identify.SYSTEM
    assert identify.system_prompt([]) == identify.SYSTEM
    prompt = identify.system_prompt(["- lesson one", "- lesson two"])
    assert prompt.startswith(identify.SYSTEM) and prompt.endswith("- lesson one\n- lesson two")


def test_generic_instructions_name_no_particular_place():
    assert "Vancouver" not in identify.SYSTEM and "rocks" not in identify.SYSTEM


def test_system_prompt_includes_the_owners_settings_in_order():
    site = {"location": "Vancouver, Washington, USA", "scene_notes": "Rocks in the middle of the water.",
            "ignore_notes": "A branch on the right sways."}
    prompt = identify.system_prompt(["- a lesson"], site)
    assert "The camera is in Vancouver, Washington, USA." in prompt
    order = [prompt.index(s) for s in ("Vancouver", "Rocks in the middle", "branch on the right", "- a lesson")]
    assert order == sorted(order)
    assert identify.system_prompt(None, {"location": "", "scene_notes": "", "ignore_notes": ""}) == identify.SYSTEM


def test_identify_passes_settings_to_every_model(monkeypatch):
    seen = []

    def ask(model, images_b64, captured_at, lessons=None, site=None):
        seen.append(site)
        return (sighting(confidence="low") if model == HAIKU else sighting()), {"model": model}

    monkeypatch.setattr(identify, "ask", ask)
    identify.identify(jpeg(), WHEN, site={"location": "Here"})
    assert seen == [{"location": "Here"}, {"location": "Here"}]


def sighting(visitor="bird", species="Dark-eyed Junco", confidence="high") -> Sighting:
    return Sighting(visitor=visitor, species=species, count=1, confidence=confidence, notes="")


@pytest.fixture
def fake_ask(monkeypatch):
    """Replace the API call with scripted answers per model; records which models were asked."""
    calls = []

    def install(answers: dict):
        def ask(model, images_b64, captured_at, lessons=None, site=None):
            calls.append((model, lessons, len(images_b64)))
            return answers[model], {"model": model}
        monkeypatch.setattr(identify, "ask", ask)
        return calls

    return install


def test_confident_haiku_answer_is_used_without_escalating(fake_ask):
    calls = fake_ask({HAIKU: sighting()})
    r = identify.identify(jpeg(), WHEN, lessons=["- a lesson"], extra_frames=[jpeg(), jpeg()])
    assert (r.model, r.escalated, r.needs_review) == (HAIKU, False, False)
    assert calls == [(HAIKU, ["- a lesson"], 1)]


def test_claudes_circle_is_moved_from_the_crop_onto_the_full_frame(fake_ask):
    s = sighting()
    s.circle = identify.Circle(x=0.5, y=0.5, r=0.1)
    fake_ask({HAIKU: s})
    r = identify.identify(jpeg(), WHEN, crop=(0.2, 0.4, 0.5, 0.5))
    assert r.raw["circle"] == {"x": 0.45, "y": 0.65, "r": 0.05, "claude": True}


@pytest.mark.parametrize("visitor, circle", [("none", identify.Circle(x=0.5, y=0.5, r=0.1)),
                                             ("bird", identify.Circle(x=1.5, y=0.5, r=0.1)),
                                             ("bird", None)])
def test_no_circle_for_empty_frames_or_bad_coordinates(visitor, circle):
    s = sighting(visitor=visitor)
    s.circle = circle
    assert identify.frame_circle(s) is None


@pytest.mark.parametrize("confidence", ["low", "medium"])
def test_unsure_bird_escalates_to_sonnet_with_all_burst_frames(fake_ask, confidence):
    calls = fake_ask({HAIKU: sighting(confidence=confidence), SONNET: sighting(species="Song Sparrow")})
    r = identify.identify(jpeg(), WHEN, extra_frames=[jpeg(), jpeg()])
    assert (r.sighting.species, r.model, r.escalated, r.needs_review) == ("Song Sparrow", SONNET, True, False)
    assert [(m, n) for m, _, n in calls] == [(HAIKU, 1), (SONNET, 3)]
    assert r.raw["sonnet"]["frames"] == 3


def test_multi_frame_prompt_explains_the_frames_belong_together():
    one = identify.user_content(["a"], WHEN)
    three = identify.user_content(["a", "b", "c"], WHEN)
    assert [b["type"] for b in three] == ["image", "image", "image", "text"]
    assert "same visit" in three[-1]["text"] and "same visit" not in one[-1]["text"]


def test_medium_after_escalation_is_kept_without_review(fake_ask):
    fake_ask({HAIKU: sighting(confidence="low"), SONNET: sighting(confidence="medium")})
    r = identify.identify(jpeg(), WHEN)
    assert (r.model, r.needs_review) == (SONNET, False)


def test_still_low_after_escalation_goes_to_review(fake_ask):
    fake_ask({HAIKU: sighting(confidence="low"), SONNET: sighting(confidence="low")})
    assert identify.identify(jpeg(), WHEN).needs_review is True


def test_low_confidence_empty_frame_does_not_escalate(fake_ask):
    calls = fake_ask({HAIKU: sighting(visitor="none", species=None, confidence="low")})
    r = identify.identify(jpeg(), WHEN)
    assert (r.model, r.needs_review) == (HAIKU, False) and len(calls) == 1


def test_haiku_refusal_falls_back_to_sonnet(fake_ask):
    fake_ask({HAIKU: None, SONNET: sighting()})
    assert identify.identify(jpeg(), WHEN).model == SONNET


def test_both_refusing_records_an_other_visit_for_review(fake_ask):
    fake_ask({HAIKU: None, SONNET: None})
    r = identify.identify(jpeg(), WHEN)
    assert (r.sighting.visitor, r.needs_review) == ("other", True)


def test_sonnet_refusal_keeps_the_haiku_answer(fake_ask):
    fake_ask({HAIKU: sighting(confidence="low"), SONNET: None})
    r = identify.identify(jpeg(), WHEN)
    assert (r.sighting.species, r.model, r.needs_review) == ("Dark-eyed Junco", HAIKU, True)


def test_sonnet_refusal_after_a_medium_answer_goes_to_review(fake_ask):
    fake_ask({HAIKU: sighting(species="Song Sparrow", confidence="medium"), SONNET: None})
    r = identify.identify(jpeg(), WHEN, confirmed={"Dark-eyed Junco"})
    assert (r.sighting.species, r.model, r.escalated, r.needs_review) == ("Song Sparrow", HAIKU, False, True)


def test_sonnet_error_keeps_the_haiku_answer_for_review(monkeypatch):
    def ask(model, images_b64, captured_at, lessons=None, site=None):
        if model == SONNET:
            raise anthropic.APIConnectionError(request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"))
        return sighting(confidence="medium"), {"model": model}

    monkeypatch.setattr(identify, "ask", ask)
    r = identify.identify(jpeg(), WHEN, extra_frames=[jpeg()])
    assert (r.sighting.species, r.model, r.escalated, r.needs_review) == ("Dark-eyed Junco", HAIKU, False, True)
    assert r.raw["sonnet"]["answer"] is None and r.raw["sonnet"]["error"].startswith("APIConnectionError")
    assert r.raw["haiku"]["answer"]["species"] == "Dark-eyed Junco"


class FakeClient:
    """client() stand-in whose parse() returns a scripted response or raises."""

    def __init__(self, response=None, raises=None):
        self.beta = self
        self.messages = self
        self.response, self.raises = response, raises

    def parse(self, **kwargs):
        if self.raises:
            raise self.raises
        return self.response


class FakeResponse:
    def __init__(self, stop_reason, parsed_output=None):
        self.model, self.stop_reason, self.parsed_output = HAIKU, stop_reason, parsed_output
        self.usage = type("Usage", (), {"input_tokens": 1200, "output_tokens": 40})()


def truncated_output_error() -> pydantic.ValidationError:
    """What the SDK raises when the answer was cut off at max_tokens (or a refusal came mid-output)."""
    try:
        pydantic.TypeAdapter(Sighting).validate_json('{"visitor": "bi')
    except pydantic.ValidationError as e:
        return e
    raise AssertionError("expected a ValidationError")


def test_ask_treats_an_unparsable_answer_like_a_refusal(monkeypatch):
    monkeypatch.setattr(identify, "client", lambda: FakeClient(raises=truncated_output_error()))
    answer, meta = identify.ask(HAIKU, ["abc"], WHEN)
    assert answer is None and meta["stop_reason"] == "unparsable" and meta["model"] == HAIKU
    assert "ValidationError" in meta["error"]


@pytest.mark.parametrize("stop_reason, parsed", [("refusal", None), ("max_tokens", None)])
def test_ask_returns_no_answer_when_the_model_stopped_short(monkeypatch, stop_reason, parsed):
    monkeypatch.setattr(identify, "client", lambda: FakeClient(FakeResponse(stop_reason, parsed)))
    answer, meta = identify.ask(HAIKU, ["abc"], WHEN)
    assert answer is None and meta["stop_reason"] == stop_reason and meta["input_tokens"] == 1200


def test_ask_cut_off_at_max_tokens_is_no_answer_even_if_it_parsed(monkeypatch):
    monkeypatch.setattr(identify, "client", lambda: FakeClient(FakeResponse("max_tokens", sighting())))
    assert identify.ask(HAIKU, ["abc"], WHEN)[0] is None


def test_ask_lets_api_errors_through(monkeypatch):
    err = anthropic.RateLimitError("slow down", response=httpx.Response(429, request=httpx.Request("POST", "x")), body=None)
    monkeypatch.setattr(identify, "client", lambda: FakeClient(raises=err))
    with pytest.raises(anthropic.RateLimitError):
        identify.ask(HAIKU, ["abc"], WHEN)


def test_unanswered_result_lands_in_the_review_queue_with_the_error():
    r = identify.unanswered(TimeoutError("read timed out"))
    assert (r.sighting.visitor, r.sighting.species, r.needs_review, r.escalated) == ("other", None, True, False)
    assert r.model == HAIKU and r.raw == {"error": "TimeoutError: read timed out"}


def test_system_prompt_gives_the_owners_three_boxes_in_order():
    site = {"scene_notes": "A 17.5 inch bath.", "target_notes": "Chickadees to ravens.", "ignore_notes": "A power cord."}
    prompt = identify.system_prompt(None, site)
    assert "What to look for here, in the owner's words:\nChickadees to ravens." in prompt
    assert "Things here that move or look like animals but are not:\nA power cord." in prompt
    order = [prompt.index(s) for s in ("17.5 inch", "Chickadees", "A power cord")]
    assert order == sorted(order)
    assert "Objects in view" not in prompt


def test_medium_answer_naming_a_confirmed_species_skips_sonnet(fake_ask):
    calls = fake_ask({HAIKU: sighting(confidence="medium"), SONNET: sighting(species="Song Sparrow")})
    r = identify.identify(jpeg(), WHEN, confirmed={"Dark-eyed Junco"})
    assert (r.sighting.species, r.model, r.escalated, r.needs_review) == ("Dark-eyed Junco", HAIKU, False, False)
    assert [m for m, _, _ in calls] == [HAIKU] and r.raw["skipped_sonnet"] == "confirmed species"


def guess_of(name):
    return {"model": "bioclip-2", "top": [{"name": name, "scientific": "", "score": 0.9}], "local_share": 0.9}


def test_local_classifier_disagreeing_sends_a_confirmed_species_to_sonnet(fake_ask):
    calls = fake_ask({HAIKU: sighting(species="House Finch", confidence="medium"), SONNET: sighting()})
    circles = []
    r = identify.identify(jpeg(), WHEN, confirmed={"House Finch"},
                          local_check=lambda c: circles.append(c) or guess_of("Dark-eyed Junco"))
    assert [m for m, _, _ in calls] == [HAIKU, SONNET] and r.escalated and r.sighting.species == "Dark-eyed Junco"
    assert r.raw["local_disagreed"] == "Dark-eyed Junco" and "skipped_sonnet" not in r.raw
    assert r.raw["classifier"]["top"][0]["name"] == "Dark-eyed Junco"
    assert len(circles) == 1  # asked once, on Haiku's circle


@pytest.mark.parametrize("said, local", [("Dark-eyed Junco", "Dark-eyed Junco"), ("Raccoon", "Common Raccoon")])
def test_local_classifier_agreeing_still_skips_sonnet(fake_ask, said, local):
    calls = fake_ask({HAIKU: sighting(species=said, confidence="medium"), SONNET: sighting(species="Song Sparrow")})
    r = identify.identify(jpeg(), WHEN, confirmed={said}, local_check=lambda c: guess_of(local))
    assert [m for m, _, _ in calls] == [HAIKU] and r.raw["skipped_sonnet"] == "confirmed species"
    assert r.raw["local_agreed"] and "local_disagreed" not in r.raw


def test_local_check_only_runs_when_sonnet_would_be_skipped(fake_ask):
    asked = []
    fake_ask({HAIKU: sighting(confidence="high"), SONNET: sighting()})
    identify.identify(jpeg(), WHEN, confirmed={"Dark-eyed Junco"}, local_check=lambda c: asked.append(c))
    fake_ask({HAIKU: sighting(species="Song Sparrow", confidence="medium"), SONNET: sighting()})
    identify.identify(jpeg(), WHEN, confirmed={"Dark-eyed Junco"}, local_check=lambda c: asked.append(c))
    assert asked == []


def test_no_local_guess_keeps_the_skip(fake_ask):
    calls = fake_ask({HAIKU: sighting(confidence="medium"), SONNET: sighting(species="Song Sparrow")})
    r = identify.identify(jpeg(), WHEN, confirmed={"Dark-eyed Junco"}, local_check=lambda c: None)
    assert [m for m, _, _ in calls] == [HAIKU] and "local_agreed" not in r.raw and "classifier" not in r.raw


@pytest.mark.parametrize("confidence, species", [("low", "Dark-eyed Junco"), ("medium", "Song Sparrow")])
def test_low_answers_and_new_names_still_get_sonnet(fake_ask, confidence, species):
    calls = fake_ask({HAIKU: sighting(species=species, confidence=confidence), SONNET: sighting()})
    identify.identify(jpeg(), WHEN, confirmed={"Dark-eyed Junco"})
    assert [m for m, _, _ in calls] == [HAIKU, SONNET]
