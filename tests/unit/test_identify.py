"""Unit tests of birdbrain/identify: pure functions, no files, database or network."""

from datetime import datetime

import anthropic
import httpx
import pydantic
import pytest

from helpers.claude import FakeClient, FakeResponse
from helpers.identify import WHEN, fake_ask, jpeg, sighting
from birdbrain import identify
from birdbrain.identify import HAIKU, SONNET, Sighting


def test_identify_note_goes_in_the_users_text():
    content = identify.user_content(["x"], datetime(2026, 11, 1, 9, 0), note="A close crop.")
    assert content[-1]["text"].endswith("A close crop.")
    assert "crop" not in identify.user_content(["x"], datetime(2026, 11, 1, 9, 0))[-1]["text"]


def test_system_prompt_appends_lessons_only_when_there_are_some():
    assert identify.system_prompt(None) == identify.SYSTEM
    assert identify.system_prompt([]) == identify.SYSTEM
    prompt = identify.system_prompt(["- lesson one", "- lesson two"])
    assert prompt.startswith(identify.SYSTEM) and prompt.endswith("- lesson one\n- lesson two\n</corrections>")


def test_owner_notes_and_lessons_sit_inside_their_tags_after_one_line_about_them():
    site = {"location": "Vancouver, Washington, USA", "scene_notes": "Rocks in the water.", "target_notes": "Juncos.",
            "ignore_notes": "A cord."}
    prompt = identify.system_prompt(["- a lesson"], site)
    notes = prompt.split("<yard_notes>\n")[1].split("\n</yard_notes>")[0]
    assert all(s in notes for s in ("The camera is in Vancouver", "Rocks in the water.", "Juncos.", "A cord."))
    lessons = prompt.split("<corrections>\n")[1].split("\n</corrections>")[0]
    assert lessons.endswith("- a lesson") and "newest first" in lessons
    intro = prompt.index(identify.OWNER_INTRO)
    assert prompt.index(identify.SYSTEM) < intro < prompt.index("<yard_notes>") < prompt.index("<corrections>")
    assert "never change" in identify.OWNER_INTRO and "answer format" in identify.OWNER_INTRO
    # Only the parts the owner filled in get a tag.
    assert "<yard_notes>" not in identify.system_prompt(["- a lesson"]) and "<corrections>" in identify.system_prompt(["- a lesson"])
    assert "<corrections>" not in identify.system_prompt(None, site) and "<yard_notes>" in identify.system_prompt(None, site)


def test_the_prompt_defines_confidence_for_an_empty_photo():
    bullet = next(l for l in identify.SYSTEM.splitlines() if l.startswith("- confidence:"))
    assert 'For "none"' in bullet and "clearly empty" in bullet and "might be an animal" in bullet


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

    def ask(model, images_b64, captured_at, lessons=None, site=None, night=False):
        seen.append((site, night))
        return (sighting(confidence="low") if model == HAIKU else sighting()), {"model": model}

    monkeypatch.setattr(identify, "ask", ask)
    identify.identify(jpeg(), WHEN, site={"location": "Here"}, night=True)
    assert seen == [({"location": "Here"}, True), ({"location": "Here"}, True)]


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


def test_multi_frame_prompt_explains_the_frames_belong_together():
    one = identify.user_content(["a"], WHEN)
    three = identify.user_content(["a", "b", "c"], WHEN)
    assert [b["type"] for b in three] == ["image", "image", "image", "text"]
    assert "same visit" in three[-1]["text"] and "same visit" not in one[-1]["text"]


def test_medium_after_escalation_is_kept_without_review(fake_ask):
    fake_ask({HAIKU: sighting(confidence="low"), SONNET: sighting(confidence="medium")})
    r = identify.identify(jpeg(), WHEN)
    assert (r.model, r.needs_review) == (SONNET, False)


def test_an_other_animal_always_waits_for_review_however_sure(fake_ask):
    """"Other" (a cat, a person, a deer) is the admin's to look at: it is flagged even at high confidence, and so is
    a visit settled on the Pi that names one."""
    fake_ask({HAIKU: sighting(visitor="other", species="Domestic Cat")})
    r = identify.identify(jpeg(), WHEN)
    assert (r.sighting.visitor, r.needs_review) == ("other", True)
    guess = {"top": [{"name": "Domestic Cat", "score": 0.97}], "model": "bioclip-2"}
    assert identify.settled_by_classifier(guess, ("Domestic Cat", "other"), None).needs_review is True
    assert identify.settled_by_classifier(guess, ("Bushtit", "bird"), None).needs_review is False


def test_still_low_after_escalation_goes_to_review(fake_ask):
    fake_ask({HAIKU: sighting(confidence="low"), SONNET: sighting(confidence="low")})
    assert identify.identify(jpeg(), WHEN).needs_review is True


@pytest.mark.parametrize("confidence", ["high", "medium"])
def test_a_confident_empty_frame_does_not_escalate(fake_ask, confidence):
    calls = fake_ask({HAIKU: sighting(visitor="none", species=None, confidence=confidence)})
    r = identify.identify(jpeg(), WHEN)
    assert (r.model, r.needs_review) == (HAIKU, False) and len(calls) == 1


def test_sonnet_failing_on_a_doubtful_empty_frame_leaves_it_in_review(fake_ask):
    fake_ask({HAIKU: sighting(visitor="none", species=None, confidence="low"), SONNET: None})
    r = identify.identify(jpeg(), WHEN)
    assert (r.sighting.visitor, r.model, r.needs_review) == ("none", HAIKU, True)


@pytest.mark.parametrize("said, official", [
    ("black-capped chickadee", "Black-capped Chickadee"),
    ("Steller’s Jay", "Steller's Jay"),      # curly apostrophe
    ("Stellers Jay", "Steller's Jay"),
    ("Dark eyed Junco", "Dark-eyed Junco"),
    ("House Finch ", "House Finch"),
])
def test_bird_names_are_stored_in_their_official_spelling(fake_ask, said, official):
    fake_ask({HAIKU: sighting(species=said)})
    r = identify.identify(jpeg(), WHEN, confirmed={"Steller's Jay"})
    assert r.sighting.species == official and r.raw["haiku"]["answer"]["species"] == official


def test_a_race_in_parentheses_moves_to_the_notes(fake_ask):
    fake_ask({HAIKU: sighting(species="Dark-eyed Junco (Oregon)", notes="Bathing.")})
    r = identify.identify(jpeg(), WHEN)
    assert (r.sighting.species, r.sighting.notes) == ("Dark-eyed Junco", "Bathing. (Oregon)")


def test_an_unknown_bird_name_is_kept_as_said(fake_ask):
    fake_ask({HAIKU: sighting(species="Mystery Warbler (eastern)")})
    assert identify.identify(jpeg(), WHEN).sighting.species == "Mystery Warbler (eastern)"


def test_other_animals_get_official_names_too_and_mammals_are_left_alone(fake_ask):
    from birdbrain.identify import Also
    s = sighting(also=[Also(visitor="bird", species="bushtit", count=2, confidence="high", circle=None),
                       Also(visitor="squirrel", species="douglas squirrel", count=1, confidence="high", circle=None)])
    fake_ask({HAIKU: s})
    r = identify.identify(jpeg(), WHEN)
    assert [a["species"] for a in r.raw["also"]] == ["Bushtit", "douglas squirrel"]


def test_official_spelling_counts_as_the_confirmed_species_so_sonnet_is_skipped(fake_ask):
    calls = fake_ask({HAIKU: sighting(species="dark-eyed junco", confidence="medium"), SONNET: sighting()})
    r = identify.identify(jpeg(), WHEN, confirmed={"Dark-eyed Junco"})
    assert [m for m, _, _ in calls] == [HAIKU] and r.sighting.species == "Dark-eyed Junco"


def test_sonnets_answer_is_canonical_as_well(fake_ask):
    fake_ask({HAIKU: sighting(confidence="low"), SONNET: sighting(species="house finch")})
    assert identify.identify(jpeg(), WHEN).sighting.species == "House Finch"


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


def request_of(monkeypatch, model, night=False) -> dict:
    """The keyword arguments ask() sends to messages.parse for one call."""
    fake = FakeClient(FakeResponse("end_turn", sighting()))
    monkeypatch.setattr(identify, "client", lambda: fake)
    identify.ask(model, ["abc"], WHEN, ["- a lesson"], {"location": "Here"}, night=night)
    return fake.request


def test_fallbacks_are_only_sent_for_sonnet(monkeypatch):
    haiku = request_of(monkeypatch, HAIKU)
    assert "fallbacks" not in haiku and "betas" not in haiku
    sonnet = request_of(monkeypatch, SONNET)
    assert sonnet["fallbacks"] == "default" and sonnet["betas"] == ["server-side-fallback-2026-07-01"]


def test_the_system_prompt_is_sent_as_one_cached_block(monkeypatch):
    for model in (HAIKU, SONNET):
        system = request_of(monkeypatch, model)["system"]
        assert len(system) == 1 and system[0]["cache_control"] == {"type": "ephemeral"}
        assert system[0]["text"] == identify.system_prompt(["- a lesson"], {"location": "Here"})


def test_night_frames_say_so_in_the_user_text(monkeypatch):
    text = request_of(monkeypatch, HAIKU, night=True)["messages"][0]["content"][-1]["text"]
    assert "This is a night infrared frame: grayscale, no color." in text
    assert "infrared" not in request_of(monkeypatch, HAIKU)["messages"][0]["content"][-1]["text"]
    three = identify.user_content(["a", "b", "c"], WHEN, night=True)[-1]["text"]
    assert "These are night infrared frames" in three and "same visit" in three


def test_the_schema_requires_the_circle_and_the_other_animals():
    from anthropic.lib._parse._transform import to_json_output_format
    schema = to_json_output_format(Sighting)["schema"]
    assert set(schema["required"]) == {"visitor", "species", "count", "confidence", "notes", "circle", "also"}
    assert set(schema["$defs"]["Also"]["required"]) == {"visitor", "species", "count", "confidence", "circle"}


def test_ask_records_the_served_model_and_cached_tokens(monkeypatch):
    fake = FakeClient(FakeResponse("end_turn", sighting(), model="claude-opus-4-8",
                                   cache_read_input_tokens=900, cache_creation_input_tokens=0))
    monkeypatch.setattr(identify, "client", lambda: fake)
    answer, meta = identify.ask(SONNET, ["abc"], WHEN)
    assert answer is not None and meta == {"model": "claude-opus-4-8", "input_tokens": 1200, "output_tokens": 40,
                                           "cache_read_input_tokens": 900, "cache_creation_input_tokens": 0}
    monkeypatch.setattr(identify, "client", lambda: FakeClient(FakeResponse("end_turn", sighting())))
    assert "cache_read_input_tokens" not in identify.ask(HAIKU, ["abc"], WHEN)[1]  # an SDK without the fields


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


def test_no_local_guess_keeps_the_skip(fake_ask):
    calls = fake_ask({HAIKU: sighting(confidence="medium"), SONNET: sighting(species="Song Sparrow")})
    r = identify.identify(jpeg(), WHEN, confirmed={"Dark-eyed Junco"}, local_check=lambda c: None)
    assert [m for m, _, _ in calls] == [HAIKU] and "local_agreed" not in r.raw and "classifier" not in r.raw


@pytest.mark.parametrize("confidence, species", [("low", "Dark-eyed Junco"), ("medium", "Song Sparrow")])
def test_low_answers_and_new_names_still_get_sonnet(fake_ask, confidence, species):
    calls = fake_ask({HAIKU: sighting(species=species, confidence=confidence), SONNET: sighting()})
    identify.identify(jpeg(), WHEN, confirmed={"Dark-eyed Junco"})
    assert [m for m, _, _ in calls] == [HAIKU, SONNET]


def with_circle(s):
    s.circle = identify.Circle(x=0.3, y=0.5, r=0.08)
    return s


def strong(name, score=0.99, share=0.95):
    return {"model": "bioclip-2", "top": [{"name": name, "scientific": "", "score": score}], "local_share": share}


def test_low_haiku_answer_backed_by_the_local_look_on_its_circle_skips_sonnet(fake_ask):
    calls = fake_ask({HAIKU: with_circle(sighting(confidence="low")), SONNET: sighting(species="Song Sparrow")})
    r = identify.identify(jpeg(), WHEN, confirmed={"Dark-eyed Junco"}, local_check=lambda c: strong("Dark-eyed Junco"))
    assert [m for m, _, _ in calls] == [HAIKU] and not r.escalated and r.model == HAIKU
    assert (r.sighting.confidence, r.needs_review) == ("high", False)
    assert r.raw["skipped_sonnet"] == "confirmed species" and r.raw["local_confirmed"]
    assert r.raw["haiku"]["answer"]["confidence"] == "low"  # Haiku's own answer is kept as it was


@pytest.mark.parametrize("guess, confirmed, circle", [
    (strong("Dark-eyed Junco", score=0.9), {"Dark-eyed Junco"}, True),   # under the tight look's bar
    (strong("Dark-eyed Junco", share=0.3), {"Dark-eyed Junco"}, True),   # mostly species not seen here
    (strong("House Finch"), {"Dark-eyed Junco"}, True),                  # a different species
    (strong("Dark-eyed Junco"), set(), True),                            # species not confirmed here yet
    (strong("Dark-eyed Junco"), {"Dark-eyed Junco"}, False),             # no circle to look at
    (None, {"Dark-eyed Junco"}, True),
])
def test_a_low_haiku_answer_still_gets_sonnet_unless_the_local_look_is_strong(fake_ask, guess, confirmed, circle):
    s = sighting(confidence="low")
    calls = fake_ask({HAIKU: with_circle(s) if circle else s, SONNET: sighting()})
    identify.identify(jpeg(), WHEN, confirmed=confirmed, local_check=lambda c: guess)
    assert [m for m, _, _ in calls] == [HAIKU, SONNET]


def test_several_animals_in_haikus_answer_never_skip_sonnet_on_the_local_look(fake_ask):
    other = identify.Also(visitor="squirrel", species="Eastern Gray Squirrel", count=1, confidence="high", circle=None)
    calls = fake_ask({HAIKU: with_circle(sighting(confidence="low", also=[other])), SONNET: sighting()})
    identify.identify(jpeg(), WHEN, confirmed={"Dark-eyed Junco"}, local_check=lambda c: strong("Dark-eyed Junco"))
    assert [m for m, _, _ in calls] == [HAIKU, SONNET]


def test_medium_answer_with_a_strong_agreeing_local_look_shows_high(fake_ask):
    fake_ask({HAIKU: with_circle(sighting(confidence="medium")), SONNET: sighting()})
    r = identify.identify(jpeg(), WHEN, confirmed={"Dark-eyed Junco"}, local_check=lambda c: strong("Dark-eyed Junco"))
    assert r.sighting.confidence == "high" and r.raw["skipped_sonnet"] and "local_confirmed" not in r.raw
