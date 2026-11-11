"""Component tests of the web app through TestClient on a temp database: visit clips."""

import re
from datetime import timedelta

import pytest

from birdbrain import clips, config, db

VIDEO = b"\x00\x00\x00\x18ftypisom" + b"\x00" * 4000


def cards(html: str) -> list[int]:
    """The visit ids of the grid's cards, in order (the home page's hero photo is not a card)."""
    return [int(i) for i in re.findall(r'<a class="card[^"]*" href="/visit/(\d+)', html)]


@pytest.fixture
def clip_visit(stored_visit, conn):
    """A real visit with photo files and a clip on disk. Returns (id, clip path under the clips folder)."""
    def add(**fields) -> tuple[int, str]:
        vid = stored_visit(**fields)
        rel = clips.rel_for(db.get_visit(conn, vid)["image_path"])
        (config.CLIPS_DIR / rel).parent.mkdir(parents=True, exist_ok=True)
        (config.CLIPS_DIR / rel).write_bytes(VIDEO)
        db.set_clip(conn, vid, rel)
        return vid, rel
    return add


def test_the_visit_page_offers_the_clip_and_the_file_plays(client, clip_visit):
    vid, rel = clip_visit()
    page = client.get(f"/visit/{vid}").text
    assert 'id="clip-play"' in page and f'data-src="/clips/{rel}"' in page and "/static/clip.js" in page
    # Fetched only on the first press, shown without sound, and it says how long it is kept.
    assert 'preload="none"' in page and " muted " in page and "Kept until" in page
    got = client.get(f"/clips/{rel}")
    assert got.status_code == 200 and got.content == VIDEO and got.headers["content-type"] == "video/mp4"
    # Seeking (and Safari at all) needs byte ranges.
    part = client.get(f"/clips/{rel}", headers={"Range": "bytes=0-9"})
    assert part.status_code == 206 and part.content == VIDEO[:10]


def test_a_visit_without_a_clip_shows_no_player(client, stored_visit):
    page = client.get(f"/visit/{stored_visit()}").text
    assert "clip-play" not in page and "clip.js" not in page


def test_a_starred_visits_clip_says_it_is_kept_for_good(client, conn, clip_visit):
    vid, _ = clip_visit()
    db.set_keep(conn, vid, True)
    page = client.get(f"/visit/{vid}").text
    assert "Kept for good: this visit is starred." in page and "Kept until" not in page


def test_visitors_on_the_home_network_can_play_clips_too(client, clip_visit, locked):
    vid, rel = clip_visit()
    assert 'id="clip-play"' in client.get(f"/visit/{vid}").text
    assert client.get(f"/clips/{rel}").status_code == 200


def test_the_public_mirror_has_no_clips_at_all(client, clip_visit, monkeypatch):
    # Belt and braces: the mirror's copy of the database names no clip (mirror.trim), and even if it did, the page
    # offers none, the chip is gone and the folder isn't one of the mirror's routes.
    vid, rel = clip_visit()
    monkeypatch.setenv("BIRDBRAIN_PUBLIC", "1")
    page = client.get(f"/visit/{vid}").text
    assert "clip-play" not in page and "/clips/" not in page
    assert client.get(f"/clips/{rel}").status_code == 404
    assert "kind=clip" not in client.get("/").text


def test_cards_mark_visits_with_a_clip_and_the_chip_lists_only_those(client, clip_visit, stored_visit):
    with_clip, _ = clip_visit()
    without = stored_visit(species="Bushtit")
    home = client.get("/").text
    assert home.count('class="badge clip"') == 1 and "?kind=clip" in home
    assert sorted(cards(home)) == [with_clip, without]
    assert cards(client.get("/?kind=clip").text) == [with_clip]
    species = client.get("/species/Dark-eyed%20Junco?kind=clip").text
    assert cards(species) == [with_clip] and "kind=clip#visits" in species


def test_deleting_a_visit_deletes_its_clip(client, clip_visit):
    vid, rel = clip_visit()
    assert client.post(f"/visit/{vid}/delete").status_code == 200
    assert not (config.CLIPS_DIR / rel).exists()


def test_settings_says_what_the_clips_take(client, clip_visit):
    clip_visit()
    page = client.get("/settings").text
    assert "<h3>Clips</h3>" in page and "(1 clip)" in page and "never go to the public mirror" in page


def test_about_mentions_clips_and_that_they_stay_home(client, conn):
    about = client.get("/about").text
    assert "<strong>Clips</strong>" in about and "without sound, on the Pi only" in about


def test_the_clips_news_pop_up_is_on_every_page_until_its_last_day(client, locked, now, monkeypatch):
    """Read-only visitors get it (the browser shows it once: static/news.js), and after the last day it is gone
    from the HTML altogether."""
    from birdbrain.web import news
    monkeypatch.setattr(news, "CLIPS_UNTIL", now.date())  # today is the last day
    for url in ("/", "/species", "/stats", "/about"):
        page = client.get(url).text
        assert 'id="news-popup" data-news="clips"' in page and f'data-until="{now.date().isoformat()}"' in page, url
        assert "/static/news.js" in page and 'href="/?kind=clip">See the clips</a>' in page
        assert f"New visits keep a {config.CLIP_SECONDS:g} second video clip. Open a visit" in page
        # why there is no sound, in the light text and before the retention rule (the apostrophe is escaped in HTML)
        assert page.index('<p class="news-rules">Video only. Sound is never recorded, to protect the neighbours') \
            < page.index('<p class="news-rules">Clips are kept for')
        assert f'<p class="news-rules">Clips are kept for {config.CLIP_DAYS} days;' in page
    monkeypatch.setattr(news, "CLIPS_UNTIL", now.date() - timedelta(days=1))  # yesterday was
    page = client.get("/").text
    assert "news-popup" not in page and "news.js" not in page


def test_no_clips_news_on_the_public_mirror_with_clips_off_or_in_an_embedded_page(client, now, monkeypatch):
    from birdbrain.web import news
    monkeypatch.setattr(news, "CLIPS_UNTIL", now.date() + timedelta(days=14))
    assert "news-popup" in client.get("/").text  # no password: the admin gets it too
    assert "news-popup" not in client.get("/?embed=1").text  # but not inside the False triggers pop-up's page
    monkeypatch.setattr(config, "CLIP_SECONDS", 0.0)
    assert "news-popup" not in client.get("/").text
    monkeypatch.setattr(config, "CLIP_SECONDS", 5.0)
    monkeypatch.setenv("BIRDBRAIN_PUBLIC", "1")
    assert "news-popup" not in client.get("/").text


def test_the_tour_tells_the_clips_news_and_the_pop_up_remembers_being_closed(client):
    tour = client.get("/static/tour.js").text
    assert "kind=clip" in tour and 'news: "clips"' in tour and '"bb_news_" + step.news' in tour  # the step marks it seen
    script = client.get("/static/news.js").text
    assert '"bb_news_" + dlg.dataset.news' in script and 'addEventListener("close", remember)' in script
    assert "dataset.until" in script  # a page left open past the last day stays quiet too
