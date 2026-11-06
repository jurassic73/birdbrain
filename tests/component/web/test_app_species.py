"""Component tests of the web app through TestClient on a temp database: species."""

import json
import re
from datetime import timedelta

from helpers.clock import FROZEN_NOW
from helpers.web import card_ids, _keys_dialog, more_link, _seed_swipe_species
from birdbrain.web import app as webapp


def test_confirmed_pill_after_three_sure_visits(client, add_visit, monkeypatch):
    from birdbrain import species_info
    monkeypatch.setattr(species_info, "get", lambda conn, name, fetch=True: {"name": name, "common": name})
    for _ in range(2):
        add_visit(species="Bushtit", confidence="high")
    medium = add_visit(species="Bushtit", confidence="medium")
    assert "Not confirmed yet" in client.get("/species").text
    assert "Not confirmed yet" in client.get(f"/visit/{medium}").text
    add_visit(species="Bushtit", confidence="high")
    assert ">Confirmed<" in client.get("/species-info/Bushtit").text
    assert ">Confirmed<" in client.get(f"/visit/{medium}").text


def test_species_pages(client, add_visit):
    add_visit(species="Bushtit")
    assert "Bushtit" in client.get("/species").text
    assert "Bushtit" in client.get("/species/Bushtit").text
    assert client.get("/species/Nope").status_code == 404


def _seed_filter_species(add_visit):
    """Junco visits: a day one today, a night one today, a two-bird one, an old one; plus a Bushtit that must never show."""
    today = FROZEN_NOW.replace(microsecond=0)
    add_visit(species="Dark-eyed Junco", captured_at=today.isoformat())
    add_visit(species="Dark-eyed Junco", captured_at=(today - timedelta(minutes=5)).isoformat(), is_night=1)
    add_visit(species="Dark-eyed Junco", count=2, captured_at=(today - timedelta(minutes=9)).isoformat())
    add_visit(species="Dark-eyed Junco", captured_at=(today - timedelta(days=20)).isoformat())
    add_visit(species="Bushtit", also=json.dumps([{"visitor": "bird", "species": "Dark-eyed Junco", "count": 1, "confidence": "high"}]))
    add_visit(species="Bushtit")


def _cards(html):
    import re
    return re.findall(r'<a class="card(?: hero-dup)?" href="/visit/(\d+)', html)


def test_species_visits_tab_has_the_home_filters_minus_species_and_kinds(client, add_visit):
    _seed_filter_species(add_visit)
    html = client.get("/species/Dark-eyed%20Junco").text
    panel = html[html.index('id="panel-visits"'):]
    assert 'class="filters-toggle"' in panel and 'id="more-filters"' in panel and "/static/species-filters.js" in html
    assert panel.index('class="chips"') < panel.index('id="more-filters"') < panel.index('id="visit-grid"')
    chips = panel[panel.index('class="chips"'):panel.index("</nav>")]
    for label in ("All", "Multiple", "Night", "Today", "This week"):
        assert f">{label}<" in chips
    assert ">Birds<" not in chips and ">Mammals<" not in chips
    form = panel[panel.index('id="more-filters"'):panel.index("</form>")]
    assert 'name="tod"' in form and 'name="date_from"' in form and 'name="date_to"' in form and 'name="species"' not in form
    assert "data-filtered" not in panel[:panel.index(">")] and "(5)" in html
    assert len(_cards(html)) == 5  # the Bushtit photo with a junco in it counts as a junco visit


def test_species_visits_filter_by_night_multiple_and_dates_and_open_on_the_visits_tab(client, add_visit):
    _seed_filter_species(add_visit)
    night = client.get("/species/Dark-eyed%20Junco?tod=night").text
    assert len(_cards(night)) == 1 and 'aria-current="page">Night<' in night and "(1 of 5)" in night
    assert "(1 of 5)" in night and 'id="panel-visits"' in night and "data-filtered" in night and ">Clear<" in night
    day = client.get("/species/Dark-eyed%20Junco?tod=day").text
    assert len(_cards(day)) == 4 and 'id="more-filters"' in day and 'aria-expanded="true"' in day
    multi = client.get("/species/Dark-eyed%20Junco?kind=multiple").text
    assert len(_cards(multi)) == 2 and 'aria-current="page">Multiple<' in multi  # the two-bird visit and the Bushtit with a junco too
    week = client.get("/species/Dark-eyed%20Junco?date_from=2026-01-01&date_to=2026-01-02").text
    assert "No visits match these filters." in week and "Show all" in week and 'id="visit-grid"' not in week
    # Birds / Mammals and a species in the URL don't apply on a species page.
    assert len(_cards(client.get("/species/Dark-eyed%20Junco?kind=mammal&species=Bushtit").text)) == 5


def test_species_filter_chips_keep_the_sort_and_end_in_the_visits_hash(client, add_visit):
    _seed_filter_species(add_visit)
    html = client.get("/species/Dark-eyed%20Junco?sort=name&tod=night").text
    assert 'href="/species/Dark-eyed%20Junco?tod=night&amp;sort=name#visits"' in html
    assert 'href="/species/Dark-eyed%20Junco?sort=name#visits"' in html  # All
    assert 'name="sort" value="name"' in html
    assert _swipe_links(html)["prev"] is None or "sort=name" in _swipe_links(html)["prev"]


def test_species_load_more_keeps_the_filters(client, add_visit, monkeypatch):
    monkeypatch.setattr(webapp, "SPECIES_PAGE", 2)
    _seed_filter_species(add_visit)
    first = client.get("/species/Dark-eyed%20Junco?tod=day").text
    assert len(_cards(first)) == 2
    import html as htmllib, re
    url = htmllib.unescape(re.search(r'hx-get="(/species/[^"]*/more[^"]*)"', first).group(1))
    assert "tod=day" in url and "before=" in url
    page2 = client.get(url).text
    got = _cards(first) + _cards(page2)
    assert len(got) == 4 and len(set(got)) == 4  # all four day visits, none of the night one, no repeats
    assert len(_cards(client.get("/species/Dark-eyed%20Junco/more?tod=night").text)) == 1


def _swipe_links(html):
    import re
    return {rel: (m.group(1) if (m := re.search(rf'<link rel="{rel}" href="([^"]*)"', html)) else None) for rel in ("prev", "next")}


def test_species_page_swipes_along_the_list_order(client, add_visit):
    _seed_swipe_species(add_visit)
    html = client.get("/species/Song%20Sparrow").text
    assert 'data-swipe="links"' in html
    assert _swipe_links(html) == {"prev": "/species/Bushtit", "next": "/species/Anna%27s%20Hummingbird"}
    # the ends have only one neighbour
    assert _swipe_links(client.get("/species/Bushtit").text) == {"prev": None, "next": "/species/Song%20Sparrow"}
    assert _swipe_links(client.get("/species/Anna's%20Hummingbird").text)["next"] is None


def test_species_page_swipe_follows_the_lists_sort(client, add_visit):
    _seed_swipe_species(add_visit)
    html = client.get("/species/Bushtit?sort=name").text
    assert _swipe_links(html) == {"prev": "/species/Anna%27s%20Hummingbird?sort=name", "next": "/species/Song%20Sparrow?sort=name"}
    # newest first seen: Anna's, Song, Bushtit
    assert _swipe_links(client.get("/species/Song%20Sparrow?sort=newest").text)["prev"] == "/species/Anna%27s%20Hummingbird?sort=newest"
    # an unknown sort is the default order, with clean links
    assert _swipe_links(client.get("/species/Song%20Sparrow?sort=bogus").text)["prev"] == "/species/Bushtit"


def test_species_list_tiles_carry_the_sort(client, add_visit):
    _seed_swipe_species(add_visit)
    assert 'href="/species/Bushtit"' in client.get("/species").text
    assert 'href="/species/Bushtit"' in client.get("/species?sort=visits").text
    assert 'href="/species/Bushtit?sort=name"' in client.get("/species?sort=name").text
    assert "?sort=bogus" not in client.get("/species?sort=bogus").text.replace('href="?sort=', "")


def test_species_page_has_the_full_size_photo_browser(client, add_visit):
    add_visit(species="Bushtit")
    html = client.get("/species/Bushtit").text
    assert 'class="lb-open"' in html and "/static/lightbox.js" in html
    assert 'data-full="/images/' in html


def test_species_list_names_open_the_info_card_and_card_has_no_jump_to_visits(client, add_visit, monkeypatch):
    add_visit(species="Bushtit")
    assert 'data-species="Bushtit"' in client.get("/species").text
    from birdbrain import species_info
    monkeypatch.setattr(species_info, "get", lambda conn, name, fetch=True: {"name": name, "common": name, "status": "Year-round resident"})
    html = client.get("/species-info/Bushtit").text
    assert "Jump to visits" not in html and "sp-jump" not in html
    assert "All visits" not in html and 'class="sp-facts"' in html
    assert 'id="visit-grid"' in client.get("/species/Bushtit").text
    page = client.get("/species/Bushtit").text
    for tab in re.findall(r'data-sp-tab="(\w+)"', page):  # a tab name in the URL hash must not match an element id, or the browser scrolls to it
        assert f'id="{tab}"' not in page
    assert re.findall(r'class="badge [^"]*"(?! data-tip)', html) == []


def test_visit_opened_from_species_page_keeps_the_species_scope(client, add_visit):
    old = add_visit(captured_at="2026-10-24T09:00:00", species="Bushtit")
    sparrow = add_visit(captured_at="2026-10-24T10:00:00", species="Song Sparrow")
    new = add_visit(captured_at="2026-10-24T11:00:00", species="Bushtit")
    assert f'href="/visit/{new}?species=Bushtit"' in client.get("/species/Bushtit").text
    html = client.get(f"/visit/{new}?species=Bushtit").text
    assert f'href="/visit/{old}?species=Bushtit" id="nav-next"' in html
    assert f'class="photo-arrow next" href="/visit/{old}?species=Bushtit"' in html
    assert "Browsing: Bushtit" in html
    plain = client.get(f"/visit/{new}").text
    assert "Browsing:" not in plain and "species=Bushtit" not in plain
    assert f'href="/visit/{sparrow}" id="nav-next"' in plain


def test_species_page_without_a_card_has_title_facts_and_charts(client, add_visit):
    add_visit(species="Bushtit")
    html = client.get("/species/Bushtit").text
    assert "<h1>Bushtit</h1>" in html and 'class="facts' in html and "First seen" not in html and "grid-count" not in html and 'class="tab-count"' in html
    assert 'id="daily"' in html and 'id="hourly"' in html


def test_species_page_while_the_card_loads_keeps_the_fallback(client, add_visit):
    add_visit(species="Bushtit")  # a species with a card (species_info.has_card), not cached yet
    html = client.get("/species/Bushtit").text
    # The card arrives over htmx and replaces these; if it comes back empty they stay.
    assert 'hx-get="/species-info/Bushtit?fold=1"' in html
    assert 'id="sp-fallback"' in html and "<h1>Bushtit</h1>" in html and 'id="daily"' in html


def test_species_page_pages_past_200_visits(client, add_visit, monkeypatch):
    monkeypatch.setattr(webapp, "SPECIES_PAGE", 4)
    ids = [add_visit(species="Bushtit", captured_at=f"2026-11-01T10:00:{i:02d}") for i in range(10)]
    add_visit(species="Song Sparrow")
    newest_first = ids[::-1]
    first = client.get("/species/Bushtit").text
    assert card_ids(first) == newest_first[:4] and "/species/Bushtit/more?before=" in first
    second = client.get(more_link(first)).text
    assert card_ids(second) == newest_first[4:8] and "?species=Bushtit" in second
    third = client.get(more_link(second)).text
    assert card_ids(third) == newest_first[8:] and "load-more" not in third


def test_species_title_has_a_play_button_only_when_the_card_has_sounds(client, add_visit, monkeypatch):
    add_visit(species="Bushtit")
    from birdbrain import species_info
    card = {"name": "Bushtit", "common": "Bushtit", "status": "Year-round resident", "links": [],
            "sounds": [{"label": "Call", "file": "https://xc.example/1.mp3", "recordist": "A", "license": "CC", "km": None, "id": "1"},
                       {"label": "Song", "file": "https://xc.example/2.mp3", "recordist": "B", "license": "CC", "km": 40, "id": "2"}]}
    monkeypatch.setattr(species_info, "get", lambda conn, name, fetch=True: dict(card))
    html = client.get("/species-info/Bushtit?fold=1").text
    assert 'class="sp-play"' in html and 'data-src="https://xc.example/2.mp3"' in html and "Song</span>" in html
    assert 'aria-pressed="false"' in html and "by B" in html
    assert html.count('class="sp-play"') == 2 and "Call</span>" in html
    assert html.index("Call</span>") < html.index("Song</span>")  # the list's order
    card["sounds"] = card["sounds"] + [{"label": "Song", "file": "https://xc.example/3.mp3", "recordist": "C", "license": "CC", "km": 9, "id": "3"}]
    html = client.get("/species-info/Bushtit?fold=1").text
    assert html.count('class="sp-play"') == 3 and "Song 1</span>" in html and "Song 2</span>" in html and "Call</span>" in html
    card["sounds"] = card["sounds"][:2]
    mini = client.get("/species-info/Bushtit?mini=1").text
    assert "sp-play" not in mini
    card["sounds"] = [card["sounds"][0]]
    assert 'data-src="https://xc.example/1.mp3"' in client.get("/species-info/Bushtit?fold=1").text
    card["sounds"] = []
    assert "sp-play" not in client.get("/species-info/Bushtit?fold=1").text


def test_species_page_has_desktop_arrows_counter_and_keyboard_hint(client, add_visit):
    _seed_swipe_species(add_visit)
    html = client.get("/species/Song%20Sparrow").text
    assert '<span class="pos" aria-label="Species 2 of 3">2/3</span>' in html
    assert 'id="sp-prev" href="/species/Bushtit"' in html
    assert 'id="sp-next" href="/species/Anna%27s%20Hummingbird"' in html
    assert 'Next species (→): Anna&#39;s Hummingbird' in html and 'Previous species (←): Bushtit' in html
    assert 'id="keys-btn"' in html and 'id="keys-dialog"' in html and "species-nav.js" in html and "shortcuts.js" in html
    # the keyboard button sits just left of the About icon, after the arrows
    assert html.index('class="sp-nav"') < html.index('id="keys-btn"') < html.index('href="/about"')
    # the list's sort is kept on the arrows
    assert 'id="sp-next" href="/species/Song%20Sparrow?sort=name"' in client.get("/species/Bushtit?sort=name").text


def test_species_arrows_dim_at_the_ends_and_stay_off_other_pages(client, add_visit):
    _seed_swipe_species(add_visit)
    first = client.get("/species/Bushtit").text
    assert 'id="sp-prev"' not in first and 'id="sp-next"' in first and ">1/3<" in first
    assert 'aria-label="No previous species"' in first
    last = client.get("/species/Anna's%20Hummingbird").text
    assert 'id="sp-next"' not in last and 'id="sp-prev"' in last and ">3/3<" in last
    for path in ("/", "/species", "/about"):
        page = client.get(path).text
        assert "sp-nav" not in page.replace(".sp-nav", "") and "species-nav.js" not in page and 'rel="up"' not in page


SITE_KEYS = [("?", "This list"), ("1", "Visits"), ("2", "Species"), ("3", "Live"), ("4", "Stats"),
             ("T", "Light or dark"), ("P", "Change colors")]


def test_keyboard_shortcuts_button_and_popup_on_every_page(client, add_visit):
    vid = add_visit(visitor="bird", species="Song Sparrow")
    for path in ("/", "/species", "/live", "/stats", "/about", f"/visit/{vid}", "/species/Song%20Sparrow"):
        html = client.get(path).text
        assert 'id="keys-btn"' in html and 'id="keys-dialog"' in html and "/static/shortcuts.js" in html, path
        assert html.index('id="keys-btn"') < html.index('href="/about"')  # just left of the About icon
        rows, heads = _keys_dialog(html)
        assert rows[-len(SITE_KEYS):] == SITE_KEYS, path
    # pages without keys of their own show the site keys alone, under no heading
    assert _keys_dialog(client.get("/species").text) == (SITE_KEYS, [])


def test_species_page_up_key_goes_to_the_list_in_the_same_sort(client, add_visit):
    _seed_swipe_species(add_visit)
    assert '<link rel="up" href="/species">' in client.get("/species/Song%20Sparrow").text
    assert '<link rel="up" href="/species?sort=name">' in client.get("/species/Song%20Sparrow?sort=name").text


def test_embed_pages_have_no_keyboard_shortcuts(client, add_visit):
    vid = add_visit(visitor="bird", species="Song Sparrow")
    for path in (f"/visit/{vid}?embed=1", "/?embed=1", "/species?embed=1"):
        html = client.get(path).text
        assert 'id="keys-btn"' not in html and 'id="keys-dialog"' not in html and "shortcuts.js" not in html, path
    assert "species-nav.js" not in client.get("/species/Song%20Sparrow?embed=1").text
