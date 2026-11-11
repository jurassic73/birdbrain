"""Unit tests of birdbrain/species: pure functions, no files, database or network."""

import httpx
import pytest

from helpers.species_info import F, HOME, SUMMER, FakeHaiku
from birdbrain import species_info


def test_local_status_classes():
    assert species_info.local_status([10] * 12) == ("Year-round resident", list(range(1, 13)))
    summer = [0, 0, 0, 20, 40, 50, 50, 40, 20, 0, 0, 0]
    assert species_info.local_status(summer)[0] == "Summer visitor"
    winter = [50, 40, 20, 2, 0, 0, 0, 0, 0, 20, 40, 50]
    assert species_info.local_status(winter)[0] == "Winter visitor"
    passage = [0, 0, 0, 30, 40, 2, 0, 1, 30, 40, 0, 0]
    assert species_info.local_status(passage)[0] == "Passes through in spring and fall"
    assert species_info.local_status([1] * 12)[0] == "Rarely reported here"


def test_expected_months_use_ten_percent_of_peak():
    counts = [100, 9, 10, 0, 0, 0, 0, 0, 0, 0, 50, 80]
    assert species_info.local_status(counts)[1] == [1, 3, 11, 12]


def test_links_for_birds_and_mammals():
    bird = species_info.links("Anna's Hummingbird", True, "annhum", 6317, "https://en.wikipedia.org/wiki/X")
    urls = {l["label"]: l["url"] for l in bird}
    assert urls["Audubon"].endswith("/annas-hummingbird")
    assert urls["All About Birds"].endswith("/Anna%27s_Hummingbird")
    assert urls["eBird"] == "https://ebird.org/species/annhum"
    mammal = species_info.links("Common Raccoon", False, None, 41663, None)
    assert [l["label"] for l in mammal] == ["iNaturalist"]


def rec(id, type, q="A", lat=45.5, lon=-122.6, **extra):
    return {"id": str(id), "type": type, "q": q, "lat": str(lat), "lon": str(lon), "rec": f"Rec {id}",
            "lic": "//creativecommons.org/licenses/by-nc-sa/4.0/", "file": f"//xeno-canto.org/{id}/download",
            "loc": "Somewhere", "cnt": "United States", "length": "0:20", **extra}


def test_license_name():
    assert species_info.license_name("//creativecommons.org/licenses/by-nc-sa/4.0/") == "CC BY-NC-SA 4.0"
    assert species_info.license_name("") == ""


def test_pick_sounds_prefers_nearby_song_and_call():
    recs = [
        rec(1, "song", q="A", lat=40.7, lon=-74.0),   # New York, best quality but far
        rec(2, "song", q="B"),                        # near Portland
        rec(3, "call", q="C"),
        rec(4, "call, flight call", q="A", lat=47.6, lon=-122.3),  # Seattle, ~210 km
        rec(5, "alarm call", q="A", lat=None),        # no location
        {"id": "6", "type": "song"},                  # no file: skipped
    ]
    picked = species_info.pick_sounds(recs, HOME)
    assert [p["id"] for p in picked] == ["2", "3", "4"]  # the close call beats the better one 210 km away
    assert picked[0]["label"] == "Song" and picked[1]["label"] == "Call"
    assert picked[0]["file"] == "https://xeno-canto.org/2/download"
    assert picked[0]["page"] == "https://xeno-canto.org/2"
    assert picked[1]["km"] < 150 and 150 < picked[2]["km"] < 300 and picked[0]["license"] == "CC BY-NC-SA 4.0"


def test_pick_sounds_without_home_uses_quality():
    picked = species_info.pick_sounds([rec(1, "song", q="C"), rec(2, "song", q="A")], None, n=1)
    assert picked[0]["id"] == "2" and picked[0]["km"] is None


class FakeXC:
    """Stands in for httpx.Client: answers xeno-canto searches from a dict keyed by whether a box was used."""

    def __init__(self, boxed, everywhere):
        self.boxed, self.everywhere, self.queries = boxed, everywhere, []

    def get(self, url, params=None, headers=None, timeout=None):
        self.queries.append(params["query"])
        recs = self.boxed if "box:" in params["query"] else self.everywhere
        return httpx.Response(200, json={"recordings": recs}, request=httpx.Request("GET", url))


def test_fetch_sounds_searches_near_home_first(monkeypatch):
    monkeypatch.setattr(species_info.config, "XC_API_KEY", "k")
    fake = FakeXC([rec(1, "song"), rec(2, "call"), rec(3, "call")], [rec(9, "song")])
    picked = species_info.fetch_sounds(fake, "Junco hyemalis", HOME)
    assert [p["id"] for p in picked] == ["1", "2", "3"]
    assert len(fake.queries) == 1 and "box:41.28,-128.94,50.28,-116.02" in fake.queries[0]


def test_fetch_sounds_falls_back_when_few_nearby(monkeypatch):
    monkeypatch.setattr(species_info.config, "XC_API_KEY", "k")
    fake = FakeXC([rec(1, "song")], [rec(1, "song"), rec(7, "call", lat=40.7, lon=-74.0)])
    picked = species_info.fetch_sounds(fake, "Junco hyemalis", HOME)
    assert [p["id"] for p in picked] == ["1", "7"] and len(fake.queries) == 2


def test_pick_sounds_closer_band_beats_quality_and_call_slot_is_a_call():
    recs = [
        rec(1, "song", q="A", lat=49.3, lon=-123.2),   # Vancouver BC, ~400 km
        rec(2, "song", q="B"),                         # Portland, ~30 km
        rec(3, "call, song", q="A"),
        rec(4, "call", q="C"),
    ]
    picked = species_info.pick_sounds(recs, HOME)
    assert [(p["id"], p["label"]) for p in picked] == [("2", "Song"), ("4", "Call"), ("3", "Song and call")]


WINTER = {"local_counts": [50, 40, 20, 0, 0, 0, 0, 0, 0, 20, 40, 50], "expected_months": [1, 2, 3, 10, 11, 12]}


@pytest.mark.parametrize("info,month,badge", [
    (SUMMER, 6, None),
    (SUMMER, 3, "Early arrival"),
    (SUMMER, 10, "Late departure"),
    (SUMMER, 12, "Unusual this time of year"),
    (WINTER, 9, "Early arrival"),          # wraps: season runs Oct to Mar
    (WINTER, 4, "Late departure"),
    (WINTER, 7, "Unusual this time of year"),
    (WINTER, 1, None),
    ({"local_counts": [1] * 12, "expected_months": []}, 5, "Rarely reported here"),
    ({"name": "x"}, 5, None),              # no local data
    (None, 5, None),
])
def test_season_badge(info, month, badge):
    assert species_info.season_badge(info, month) == badge


def test_short_status_for_tight_lists(monkeypatch):
    from birdbrain import species_info
    monkeypatch.setattr(species_info, "cached_cards", lambda: {"Bushtit": {"status": "Year-round resident"}, "Raccoon": {}})
    assert species_info.short_status("Bushtit") == ("Year-round", "Year-round resident")
    assert species_info.short_status("Raccoon") is None
    assert species_info.short_status("None") is None


WORDS = ["Beetles", "Granite", "Violins", "Marathons", "Volcanoes", "Teapots", "Glaciers", "Lanterns", "Pebbles",
         "Thunder", "Orchards", "Compass", "Harbors", "Meadows", "Lichens", "Embers"]


def T(topic, word):
    """A fact on a topic whose words no other fact shares (so the repeat check leaves it alone)."""
    return {"lead": word, "text": f"{word} {word}zo {word}zu.", "topic": topic}


def test_clean_facts_guard():
    notes = ["Eats seeds on the ground.", "Travels in flocks in winter."]
    facts = [
        F("x", ""),                                                           # blank
        F("x", "   "),                                                        # blank
        F("Quarters", "Weighs about as much as three quarters."),             # kept
        F("", "Eats seeds on the ground."),                                   # repeats a note
        F("Flocks", "Flocks travel together in winter."),                     # mostly the same words as a note
        F("", " ".join(["word"] * 21)),                                       # too long
        F("Quarters", "Weighs about as much as three quarters!"),             # repeats an earlier fact
        F("Males", "Males sing from the tops of tall trees."),
        F("10 years", "Some live over 10 years in the wild."),
        F("Snow bird", "Its name means snow bird."),
        F("Pairs", "Pairs stay together all season."),
    ] + [T("other", w) for w in WORDS]  # more than the pool holds
    got = species_info.clean_facts(facts, notes)
    assert got[0] == F("Quarters", "Weighs about as much as three quarters.")
    texts = [f["text"] for f in got]
    assert "Eats seeds on the ground." not in texts and "Flocks travel together in winter." not in texts
    assert len(got) == species_info.MAX_FACTS == 15
    assert not any(len(t.split()) > species_info.MAX_FACT_WORDS or not t.strip() for t in texts)
    assert species_info.clean_facts(None, notes) == []


def test_clean_facts_accepts_models_dicts_and_old_strings():
    out = species_info.clean_facts([species_info.Fact(lead="Blizzard", text="A flock is called a blizzard."),
                                    "Plain old string fact lives to 11 years."], [])
    assert out == [F("Blizzard", "A flock is called a blizzard."), F("11 years", "Plain old string fact lives to 11 years.")]


@pytest.mark.parametrize("lead, text, want", [
    ("Blizzard", "A flock is called a blizzard.", "Blizzard"),                      # valid, case-insensitive
    ("blizzard", "A flock is called a blizzard.", "Blizzard"),                      # it heads a tile: capital first letter
    ("75%", "Seeds make up 75% of the diet.", "75%"),                               # number as written
    ("75%", "Seeds make up 75 percent of the diet.", "75%"),                        # % vs the word
    ("12 years", "Lives up to 12-year stretches.", "12 years"),                     # unit dropped in the text
    ("40 million", "About 40 million live in North America.", "40 million"),
    ("1,741", "Named in 1741 by a Swede.", "1,741"),                                # same number, comma or not
    ("1,800", "Named in 1741 by a Swede.", "1741"),                                 # a different number: the text's own
    ("", "Weighs about 20 grams.", "20 grams"),                                     # missing: number fallback with unit
    ("", "Lives in flocks near feeders.", ""),                                      # missing, no number: no lead
    ("Hummingbird", "A flock is called a blizzard.", ""),                           # not in the text, no number
    ("Warbler", "Weighs 9 ounces and sings.", "9 ounces"),                          # not in text: number fallback
    ("Seven word long lead here", "Seven word long lead here is odd.", ""),         # over 3 words
    ("Extraordinarily rare", "Extraordinarily rare bird.", ""),                          # over the character cap
    ("Eight", "About 8% of the diet is bugs.", "8%"),                               # word lead absent: number fallback
])
def test_clean_lead(lead, text, want):
    assert species_info.clean_lead(lead, text) == want


def test_clean_notes_drops_a_note_that_repeats_an_earlier_one():
    notes = ["In winter it often forages in flocks, and it is common at feeders.",
             "Juncos forage in mixed flocks in winter and are common at feeders.",
             "", "Eats seeds and small insects.", "Eats seeds and small insects."]
    assert species_info.clean_notes(notes) == [notes[0], "Eats seeds and small insects."]
    many = ["Eats beetles.", "Nests in hedges.", "Sings at dawn.", "Drinks often.", "Migrates south."]
    assert len(species_info.clean_notes(many)) == species_info.MAX_NOTES


def test_card_text_parses_both_lists_and_guards_them(monkeypatch):
    from birdbrain import identify
    Fact = species_info.Fact
    fake = FakeHaiku(["Eats seeds.", "", "Lives in flocks."] + [f"Note about {w}." for w in ("red", "blue", "green", "pink")],
                     [Fact(lead="Deck of cards", text="A tiny bird that weighs less than a deck of cards."),
                      Fact(lead="x", text="Eats seeds."), Fact(lead="", text="")])
    monkeypatch.setattr(identify, "client", lambda: fake)
    out = species_info.card_text("Bushtit", "Article text " * 10)
    assert len(out["notes"]) == species_info.MAX_NOTES and out["notes"][:2] == ["Eats seeds.", "Lives in flocks."]
    assert out["facts"] == [F("Deck of cards", "A tiny bird that weighs less than a deck of cards.")]
    sent = fake.requests[0]
    assert sent["output_format"] is species_info.CardText and "facts" in sent["system"] and "lead" in sent["system"]
    assert "Animal: Bushtit" in sent["messages"][0]["content"]
    # Sonnet picks the facts (Haiku chose genetics and study details), with the refusal fallback every Sonnet call sends
    assert sent["model"] == identify.SONNET and sent["fallbacks"] == "default" and sent["output_config"] == {"effort": "medium"}


def test_card_text_sends_the_whole_article_up_to_the_cap(monkeypatch):
    """The song, call and migration sections sit far down a Wikipedia article: the old 12,000 character cut lost them."""
    from birdbrain import identify
    fake = FakeHaiku([], [])
    monkeypatch.setattr(identify, "client", lambda: fake)
    species_info.card_text("Bushtit", "a" * 30000 + " MIGRATION " + "b" * 100000)
    sent = fake.requests[0]["messages"][0]["content"]
    assert "MIGRATION" in sent and len(sent) < species_info.ARTICLE_CHARS + 100


def test_card_text_without_a_parsed_answer(monkeypatch):
    from birdbrain import identify
    fake = FakeHaiku([], [])
    fake.parsed = None
    monkeypatch.setattr(identify, "client", lambda: fake)
    assert species_info.card_text("Bushtit", "x") == {"notes": [], "facts": []}


def test_clean_facts_drops_a_leadless_fact_unless_too_few_remain():
    notes = []
    mixed = [F("Blizzard", "A flock of juncos is called a blizzard."), F("", "Juncos forage on the ground in winter flocks."),
             F("Arctic", "Some juncos breed as far north as the Arctic.")]
    assert [f["lead"] for f in species_info.clean_facts(mixed, notes)] == ["Blizzard", "Arctic"]
    one_led = [F("Blizzard", "A flock of juncos is called a blizzard."), F("", "Juncos forage on the ground in winter flocks.")]
    assert len(species_info.clean_facts(one_led, notes)) == 2  # dropping would leave 1: keep both
    none_led = [F("", "Juncos forage on the ground in winter flocks."), F("", "Males sing from the tops of tall trees.")]
    assert len(species_info.clean_facts(none_led, notes)) == 2
    many = [T("other", w) for w in WORDS]
    assert len(species_info.clean_facts(many, notes)) == species_info.MAX_FACTS


def test_prompt_asks_for_whole_single_claim_sentences_with_a_lead():
    p = species_info.NOTES_PROMPT
    assert "complete sentence" in p and "exactly one claim" in p and "Every fact must have a lead" in p


def test_prompt_asks_for_a_pool_spread_over_every_topic():
    p = species_info.NOTES_PROMPT
    assert "10 to 15" in p and f"at most {species_info.MAX_PER_TOPIC} facts on any one topic" in p
    assert all(topic in p for topic in species_info.TOPICS)  # every topic the schema allows is explained
    assert species_info.FACTS_WANTED == 10 and species_info.MAX_FACTS == 15


def test_clean_facts_caps_a_topic_and_deals_the_pool_out_by_topic():
    raw = [T("song", w) for w in WORDS[:5]] + [T("size", WORDS[5]), T("other", WORDS[6]), T("made-up", WORDS[7]),
                                               T("size", WORDS[8])]
    got = species_info.clean_facts(raw, [])
    assert [f.get("topic") for f in got].count("song") == species_info.MAX_PER_TOPIC  # the 4th and 5th song facts go
    # dealt out one topic at a time, so neighbours differ; "other" and an unknown topic are stored without one
    assert [(f["lead"], f.get("topic")) for f in got] == [
        (WORDS[0], "song"), (WORDS[5], "size"), (WORDS[6], None), (WORDS[7], None), (WORDS[1], "song"), (WORDS[8], "size"),
        (WORDS[2], "song")]
    # the model's own Fact objects carry the topic the same way
    assert species_info.clean_facts([species_info.Fact(lead="Trill", text="Its song is a trill.", topic="song")], []) == [
        {"lead": "Trill", "text": "Its song is a trill.", "topic": "song"}]


def test_rotate_shows_a_new_window_each_turn_and_wraps():
    pool = list("abcdefgh")
    assert species_info.rotate(pool, 0, 3) == ["a", "b", "c"]
    assert species_info.rotate(pool, 1, 3) == ["d", "e", "f"]
    assert species_info.rotate(pool, 2, 3) == ["g", "h", "a"]  # wraps around
    assert species_info.rotate(pool, 8, 3) == ["a", "b", "c"]  # and comes back
    shown = {x for turn in range(3) for x in species_info.rotate(pool, turn, 3)}
    assert shown == set(pool)  # every fact gets its turn
    assert species_info.rotate(pool[:3], 7, 3) == ["a", "b", "c"]  # a pool that fits is shown whole, in order
    assert species_info.rotate([], 7, 3) == []


def test_rotate_passes_over_a_topic_already_on_show():
    pool = [T("song", "A"), T("song", "B"), T("size", "C"), T("song", "D"), T("food", "E"), {"lead": "", "text": "F."}]
    assert [f["lead"] for f in species_info.rotate(pool, 0, 3)] == ["A", "C", "E"]
    # not enough other topics: the passed-over ones fill in, so it is always n facts
    songs = [T("song", w) for w in "ABCD"]
    assert [f["lead"] for f in species_info.rotate(songs, 0, 3)] == ["A", "B", "C"]
    # facts without a topic never crowd each other out
    plain = [{"lead": "", "text": t} for t in "ABCD"]
    assert [f["text"] for f in species_info.rotate(plain, 1, 3)] == ["D", "A", "B"]


def test_fact_turn_moves_on_with_time_and_with_the_extra():
    from datetime import datetime, timedelta
    at = datetime(2026, 11, 10, 9, 0, 0)
    turn = species_info.fact_turn(at)
    assert species_info.fact_turn(at + timedelta(seconds=species_info.ROTATE_SECONDS)) == turn + 1
    assert species_info.fact_turn(at + timedelta(seconds=5)) in (turn, turn + 1)  # steady inside a window
    assert species_info.fact_turn(at, extra=7) == turn + 7  # a new visit (a higher id) moves it on too


def test_tile_facts_shows_four_of_the_pool_and_reads_old_formats():
    pool = [T("song", "A"), T("size", "B"), T("food", "C"), T("name", "D"), T("calls", "E"), T("colors", "F")]
    assert [f["lead"] for f in species_info.tile_facts(pool, turn=0)] == ["A", "B", "C", "D"]
    assert [f["lead"] for f in species_info.tile_facts(pool, turn=1)] == ["E", "F", "A", "B"]
    assert species_info.tile_facts(["Old plain fact.", {"text": "  "}, 5, None], turn=3) == [{"lead": "", "text": "Old plain fact."}]
    assert species_info.tile_facts(None, turn=0) == []


def test_prompt_wants_a_named_us_coin():
    assert "common US coin" in species_info.NOTES_PROMPT and "quarter 5.67 g" in species_info.NOTES_PROMPT
    assert "Never a half dollar" in species_info.NOTES_PROMPT
    assert "just state the weight" in species_info.NOTES_PROMPT and "Facts, not folklore" in species_info.NOTES_PROMPT


def test_vague_coin_flags_only_unnamed_coins():
    F = species_info.Fact
    vague = "Both sexes of the black-capped chickadee weigh 10-14 g, about the weight of a coin."
    assert species_info.vague_coin([F(lead="10-14 g", text=vague)])
    assert species_info.vague_coin([{"lead": "x", "text": vague}, "It sings."])
    assert not species_info.vague_coin([F(lead="11 g", text="It weighs 11 g, about as much as a US half dollar.")])
    assert not species_info.vague_coin([F(lead="6 g", text="It weighs about the same as a quarter.")])
    assert not species_info.vague_coin(None)


def test_http_client_offers_http2_so_wikimedia_does_not_403():
    client = species_info.http_client()
    try:
        assert client.follow_redirects
        assert client._transport._pool._http2
    finally:
        client.close()
