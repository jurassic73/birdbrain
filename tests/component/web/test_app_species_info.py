"""Component tests of the web app through TestClient on a temp database: species info."""

import re
from datetime import timedelta

from helpers.cards import CARD, SOUNDS
from helpers.clock import FROZEN_NOW
from helpers.species_info import F, SUMMER, fake_build, _save_card
from birdbrain import db, species_info


def test_card_route_renders_full_card(client, add_visit, fake_build):
    add_visit(species="Bushtit", captured_at="2026-01-05T09:00:00")
    html = client.get("/species-info/Bushtit").text
    assert "Psaltriparus minimus" in html and "Year-round resident" in html
    assert "AI summary based on Wikipedia" in html and "1</span> visit" in html
    assert 'data-taxon="14886"' in html and "ebird.org/species/bushti" in html


def test_card_route_mini(client, add_visit, fake_build):
    add_visit(species="Bushtit")
    client.get("/species-info/Bushtit")  # builds the card; the hover preview only reads it
    html = client.get("/species-info/Bushtit?mini=1").text
    assert "Click for more" in html and "<h2>" not in html


def test_card_fold_splits_into_the_bird_and_the_info_body(client, add_visit, fake_build):
    add_visit(species="Bushtit")
    assert "<details" not in client.get("/species-info/Bushtit").text
    assert "<h2>" in client.get("/species-info/Bushtit").text
    folded = client.get("/species-info/Bushtit?fold=1").text
    # The title and photo swap into the page; the Info tab's body rides along out of band.
    assert "sp-hero" in folded and folded.count("<h1") == 1 and 'id="sp-info-body" hx-swap-oob="innerHTML"' in folded
    assert folded.index('id="sp-info-body"') > folded.index("sp-photo")
    assert folded.index('class="sp-map"') > folded.index('id="sp-info-body"')
    assert "Click for more" in client.get("/species-info/Bushtit?mini=1&fold=1").text
    assert 'id="daily"' in folded and 'id="daily"' not in client.get("/species-info/Bushtit").text  # not in the pop-up
    assert "sp-more" not in folded and "Jump to visits" not in folded


def test_species_page_has_info_and_visits_tabs(client, add_visit, fake_build):
    add_visit(species="Bushtit")
    client.get("/species-info/Bushtit")  # builds and caches the card
    html = client.get("/species/Bushtit").text
    assert 'role="tablist"' in html and html.count('role="tab"') == 2 and html.count('role="tabpanel"') == 2
    assert html.index('data-sp-tab="visits"') < html.index('data-sp-tab="info"')
    assert re.search(r'id="tab-visits"[^>]*aria-selected="true"', html) and re.search(r'id="tab-info"[^>]*aria-selected="false" tabindex="-1"', html)  # Visits is the default tab
    assert re.search(r'id="panel-info"[^>]*hidden>', html) and not re.search(r'id="panel-visits"[^>]*hidden>', html) and "Jump to visits" not in html
    # Facts, charts and the card's sections live in the Info panel; the grid and its tools in the Visits panel.
    visits, info = html.index('id="panel-visits"'), html.index('id="panel-info"')
    assert "First seen" not in html
    assert "grid-count" not in html and html.index('class="tab-count"') < visits < html.index('class="lb-open"', visits) < info  # the count is in the Visits tab only
    assert re.search(r'data-sp-tab="info">Info</button>\s*<button[^>]*lb-tab', html)  # phones: count in the tab, expand button after it
    lb_tab = re.search(r'<button[^>]*lb-tab[^>]*>', html).group(0)
    assert "aria-disabled" not in lb_tab and "tabindex" not in lb_tab  # Visits is the default tab: the expand button starts enabled
    assert html.index("Year-round resident") < html.index('role="tablist"')  # the status pill sits with the bird
    assert info < html.index('id="daily"') and info < html.index('class="sp-map"')
    assert visits < html.index('class="grid-tools"') < html.index('id="visit-grid"') < info
    assert html.index("<h1") < html.index('role="tablist"')  # the bird comes first
    assert "species-tabs.js" in html and html.count("<h1") == 1 and "<h2>" not in html
    assert "?fold=1" not in html  # rendered inline, no htmx load
    assert html.count('id="daily"') == 1 and html.count('id="hourly"') == 1


def test_species_page_loading_card_asks_for_fold_and_keeps_a_fallback(client, add_visit):
    add_visit(species="Bushtit")
    html = client.get("/species/Bushtit").text  # no cached card yet: loads over htmx
    assert "/species-info/Bushtit?fold=1" in html
    # Title, facts and charts stand in until the card arrives (the page drops them then), or stay if it never does.
    assert 'id="sp-fallback"' in html and 'id="daily"' in html


def test_species_page_without_card_keeps_title(client, add_visit):
    add_visit(species="Other")
    html = client.get("/species/Other").text
    assert html.count("<h1>Other</h1>") == 1 and 'id="daily"' in html and 'role="tablist"' in html
    assert html.index("<h1>Other</h1>") < html.index('role="tablist"') < html.index('id="visit-grid"') < html.index('id="daily"')


def test_card_route_unknown(client, monkeypatch):
    monkeypatch.setattr(species_info, "build", lambda name: {"name": name, "links": []})
    assert "No info found" in client.get("/species-info/Nothing").text


def test_names_open_cards_but_not_false_triggers(client, add_visit):
    add_visit(species="Bushtit")
    add_visit(visitor="none", species=None)
    assert 'data-species="Bushtit"' in client.get("/").text
    assert "data-species" not in client.get("/false-triggers").text


def test_card_shows_sounds_with_credits(client, add_visit, monkeypatch):
    add_visit(species="Bushtit")
    sounds = [{"id": "9", "label": "Song", "file": "https://xeno-canto.org/9/download", "page": "https://xeno-canto.org/9",
               "recordist": "A Recordist", "license": "CC BY 4.0", "where": "Portland, United States", "km": 1234,
               "length": "0:30"}]
    monkeypatch.setattr(species_info, "build", lambda name: {**CARD, "name": name, "sounds": sounds})
    html = client.get("/species-info/Bushtit").text
    assert '<audio controls preload="none" src="https://xeno-canto.org/9/download" data-card="Bushtit">' in html
    assert "A Recordist, CC BY 4.0" in html and "(1,234 km away)" in html and "XC9" in html
    assert "Recordings streamed from xeno-canto" in html


def test_cards_show_season_badge_from_cached_card(client, add_visit, conn):
    db.save_species_info(conn, "Rufous Hummingbird", {**CARD, **SUMMER, "name": "Rufous Hummingbird"},
                         FROZEN_NOW.isoformat())
    add_visit(species="Rufous Hummingbird", captured_at="2026-03-28T09:00:00")
    add_visit(species="Bushtit", captured_at="2026-03-28T10:00:00")  # no cached card: no badge
    home = client.get("/").text
    assert home.count('class="badge out-of-season"') == 1 and "Early arrival" in home
    visit_id = db.list_visits(conn, species="Rufous Hummingbird")[0]["id"]
    page = client.get(f"/visit/{visit_id}").text
    assert "Early arrival" in page and "March is outside its usual months here." in page


def test_false_trigger_cards_get_no_season_badge(client, add_visit, conn):
    db.save_species_info(conn, "Rufous Hummingbird", {**CARD, **SUMMER}, FROZEN_NOW.isoformat())
    add_visit(visitor="none", species=None, corrected_species=None, captured_at="2026-03-28T09:00:00")
    assert 'badge out-of-season' not in client.get("/false-triggers").text


def _post(client, url, name="Bushtit", **kw):
    return client.post(f"/species-info/{name}/sound-broken", json={"url": url}, **kw)


def test_sound_broken_route_flags_only_a_real_sound_of_a_known_species(client, conn, add_visit):
    add_visit(species="Bushtit")
    _save_card(conn)
    assert _post(client, SOUNDS[0]["file"], name="Made Up Bird").status_code == 400
    assert _post(client, "https://elsewhere.example/x.mp3").status_code == 400
    assert client.post("/species-info/Bushtit/sound-broken", content="nope").status_code == 400
    assert not db.get_species_info(conn, "Bushtit")[0].get("sounds_broken")
    assert _post(client, SOUNDS[0]["file"]).status_code == 204
    card = db.get_species_info(conn, "Bushtit")[0]
    assert card["sounds_broken"] is True and len(card["sounds"]) == 2  # flagged, nothing removed


def test_sound_broken_allowed_for_visitors_on_the_pi_but_not_the_mirror(client, conn, add_visit, monkeypatch):
    add_visit(species="Bushtit")
    _save_card(conn)
    monkeypatch.setenv("ADMIN_PASSWORD", "s3cret!")
    assert _post(client, SOUNDS[1]["file"]).status_code == 204
    assert client.post("/species-info/Bushtit/delete").status_code == 403  # nothing else opened up
    assert _post(client, SOUNDS[1]["file"], headers={"sec-fetch-site": "cross-site"}).status_code == 403
    monkeypatch.setenv("BIRDBRAIN_PUBLIC", "1")
    assert _post(client, SOUNDS[1]["file"]).status_code == 404  # the mirror takes no POST but the guest sign-in


def test_header_play_buttons_carry_the_card_name(client, conn, add_visit):
    """The header's Song/Call buttons say which species they belong to, so a clip that won't play can be reported."""
    add_visit(species="Bushtit")
    _save_card(conn)
    card = db.get_species_info(conn, "Bushtit")
    for sound in card[0]["sounds"]:
        sound.update(km=100, recordist="A")
    db.save_species_info(conn, "Bushtit", card[0], card[1])
    html = client.get("/species-info/Bushtit?fold=1").text
    buttons = re.findall(r'<button[^>]*class="sp-play"[^>]*>', html)
    assert len(buttons) == 2 and all('data-card="Bushtit"' in b for b in buttons)


HOOKS = [F("Blizzard", "A flock of juncos is called a blizzard."), F("75%", "Seeds make up 75% of the diet."),
         F("Arctic", "Some breed as far north as the Arctic."), F("1741", "Named by Linnaeus in 1741."),
         F("", "Old tile with no lead lives in flocks.")]


def _hero(client, conn, add_visit, facts):
    add_visit(species="Bushtit")
    _save_card(conn, facts=facts, sounds=[{**s, "km": 100, "recordist": "A"} for s in SOUNDS])
    return client.get("/species-info/Bushtit?fold=1").text


def test_species_header_shows_a_tile_per_fact_with_its_lead(client, conn, add_visit):
    html = _hero(client, conn, add_visit, HOOKS[:4])
    assert len(re.findall(r'class="fact-tile[ "]', html)) == 4 and 'data-n="4"' in html
    for lead in ("Blizzard", "75%", "Arctic", "1741"):
        assert f"<b>{lead}</b>" in html
    assert 'data-tip="AI summary based on Wikipedia"' in html
    assert "has-facts" in html
    # the sound buttons move under the name, inside the name block (before the tiles)
    assert html.index("sp-hero-text") < html.index("sp-plays") < html.index("fact-tiles")


def test_species_header_five_facts_and_a_leadless_tile(client, conn, add_visit):
    html = _hero(client, conn, add_visit, HOOKS)
    assert len(re.findall(r'class="fact-tile[ "]', html)) == 5 and html.count("<b>") == 4 and "no-lead" in html


def test_species_header_old_string_facts_still_render(client, conn, add_visit):
    html = _hero(client, conn, add_visit, ["Old plain fact one here.", "Old plain fact two here."])
    assert len(re.findall(r'class="fact-tile[ "]', html)) == 2 and "<b>" not in html and "Old plain fact two here." in html


def test_species_header_without_facts_is_as_before(client, conn, add_visit):
    for facts in ([], HOOKS[:1]):  # one fact alone is not a tile group
        html = _hero(client, conn, add_visit, facts)
        assert "fact-tile" not in html.replace(".fact-tile", "") and "has-facts" not in html
        assert html.index("sp-hero-text") < html.index("sp-hero-id") + 400 and "sp-plays" in html


def test_pop_up_card_has_no_tiles(client, conn, add_visit):
    add_visit(species="Bushtit")
    _save_card(conn, facts=HOOKS[:4], sounds=[{**s, "km": 100, "recordist": "A"} for s in SOUNDS])
    assert "fact-tile" not in client.get("/species-info/Bushtit").text


def _last_seen_visit(add_visit):
    from datetime import timedelta
    from helpers.clock import FROZEN_NOW
    add_visit(species="Bushtit", captured_at=(FROZEN_NOW - timedelta(days=2)).isoformat())


def test_hero_has_last_seen_line_under_the_songs(client, conn, add_visit):
    _last_seen_visit(add_visit)
    for facts in (HOOKS[:4], []):  # with tiles (inside the name block) and without
        _save_card(conn, facts=facts, sounds=[{**s, "km": 100, "recordist": "A"} for s in SOUNDS])
        html = client.get("/species-info/Bushtit?fold=1").text
        assert html.count('class="sp-last-seen') == 1
        assert html.index("sp-plays") < html.index("sp-last-seen")
        assert re.search(r'Last seen <span class="ago" data-bare="1" data-age="\d+">2 days ago</span>', html)


def test_hero_last_seen_line_without_sounds_sits_under_the_badge(client, conn, add_visit):
    _last_seen_visit(add_visit)
    _save_card(conn, facts=[], sounds=[])
    html = client.get("/species-info/Bushtit?fold=1").text
    assert "sp-plays" not in html and html.index("sp-status") < html.index("sp-last-seen") < html.index("sp-hero-row") + 4000


def test_info_tab_drops_last_seen_but_pop_up_keeps_it(client, conn, add_visit):
    _last_seen_visit(add_visit)
    _save_card(conn, facts=[], sounds=[])
    folded = client.get("/species-info/Bushtit?fold=1").text
    body = folded[folded.index('id="sp-info-body"'):]
    assert "Last seen" not in body and 'class="sp-facts' in body  # the confirm badge row stays
    popup = client.get("/species-info/Bushtit").text
    assert "sp-last-seen" not in popup
    assert re.search(r'Last seen <strong>[^<]+</strong> <span class="ago" data-age="\d+">\(2 days ago\)</span>', popup)
