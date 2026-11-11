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
    assert 'id="clip-play"' in page and f'data-src="/clips/{rel}"' in page and "/static/clip.js" in page and 'id="clip-autoplay"' in page
    # Fetched only on the first press, shown without sound, and it says how long it is kept.
    assert 'preload="none"' in page and " muted " in page and "Kept at least until" in page
    got = client.get(f"/clips/{rel}")
    assert got.status_code == 200 and got.content == VIDEO and got.headers["content-type"] == "video/mp4"
    # Seeking (and Safari at all) needs byte ranges.
    part = client.get(f"/clips/{rel}", headers={"Range": "bytes=0-9"})
    assert part.status_code == 206 and part.content == VIDEO[:10]


def test_a_visit_without_a_clip_shows_no_player(client, stored_visit):
    page = client.get(f"/visit/{stored_visit()}").text
    assert "clip-play" not in page and "clip.js" not in page and clips.NOT_CAPTURED not in page


def test_a_visit_whose_clip_was_dropped_says_the_video_was_not_captured(client, conn, stored_visit, monkeypatch):
    vid = stored_visit()
    db.set_clip(conn, vid, None, note="the camera's stream stalled for 4.0 s while it recorded")
    page = client.get(f"/visit/{vid}").text
    assert clips.NOT_CAPTURED in page and "stalled for 4.0 s" in page and "clip-play" not in page
    monkeypatch.setenv("BIRDBRAIN_PUBLIC", "1")  # the mirror says so too
    assert clips.NOT_CAPTURED in client.get(f"/visit/{vid}").text


def test_a_starred_visits_clip_says_it_is_kept_for_good(client, conn, clip_visit):
    vid, _ = clip_visit()
    db.set_keep(conn, vid, True)
    page = client.get(f"/visit/{vid}").text
    assert "Kept for good: this visit is starred." in page and "Kept until" not in page


def test_home_offers_auto_play_of_the_latest_visits_clip(client, clip_visit, stored_visit):
    vid, rel = clip_visit()
    page = client.get("/").text
    assert 'id="hero-autoplay"' in page and f'data-src="/clips/{rel}"' in page and "/static/hero-clip.js" in page
    stored_visit(species="Bushtit")  # a newer visit without a clip: no box
    assert 'id="hero-autoplay"' not in client.get("/").text


def test_visitors_on_the_home_network_can_play_clips_too(client, clip_visit, locked):
    vid, rel = clip_visit()
    assert 'id="clip-play"' in client.get(f"/visit/{vid}").text
    assert client.get(f"/clips/{rel}").status_code == 200


def test_the_public_mirror_plays_the_clips_it_was_sent(client, clip_visit, stored_visit, monkeypatch):
    # The mirror's copy names only the clips the person check passed (privacy.filter_clips); those play there like
    # on the Pi, with the visitors' wording (no stars), and the Clips chip lists them.
    vid, rel = clip_visit()
    plain = stored_visit(species="Bushtit")
    monkeypatch.setenv("BIRDBRAIN_PUBLIC", "1")
    page = client.get(f"/visit/{vid}").text
    assert 'id="clip-play"' in page and f'data-src="/clips/{rel}"' in page
    assert "star" not in page.split('class="vclip"')[1].split("</section>")[0].lower()
    assert "Kept at least until" not in page and 'id="clip-autoplay"' in page  # keep-note is home-site only
    assert client.get(f"/clips/{rel}").status_code == 200
    assert "clip-play" not in client.get(f"/visit/{plain}").text
    assert cards(client.get("/?kind=clip").text) == [vid]


def test_the_admin_sees_whether_a_clip_may_go_public(client, conn, clip_visit, monkeypatch):
    from birdbrain import privacy
    vid, rel = clip_visit()
    privacy.save_cache({})  # the filter has run, but not on this clip yet
    page = client.get(f"/visit/{vid}").text
    assert "clip not public yet" in page and "make-clip-public" not in page
    privacy.save_cache({privacy.clip_key(rel): 0.9})  # someone walked through
    page = client.get(f"/visit/{vid}").text
    assert "clip private" in page and 'id="make-clip-public"' in page
    assert client.post(f"/visit/{vid}/clip-public").status_code == 200
    assert "clip public (checked)" in client.get(f"/visit/{vid}").text
    assert privacy.parse_override(db.get_visit(conn, vid)["public_ok"]) == {privacy.clip_key(rel): 0.9}
    assert client.post(f"/visit/{vid}/clip-private").status_code == 200
    assert db.get_visit(conn, vid)["public_ok"] is None and "clip private" in client.get(f"/visit/{vid}").text
    privacy.save_cache({privacy.clip_key(rel): 0.01})  # nobody there: no pill at all
    assert "clip private" not in client.get(f"/visit/{vid}").text.replace("Clip private:", "")


def test_make_clip_public_needs_a_flagged_clip(client, clip_visit):
    from birdbrain import privacy
    vid, rel = clip_visit()
    assert client.post(f"/visit/{vid}/clip-public").status_code == 409  # not checked yet
    privacy.save_cache({privacy.clip_key(rel): 0.01})
    assert client.post(f"/visit/{vid}/clip-public").status_code == 409  # clean: nothing to clear
    assert client.post("/visit/9999/clip-public").status_code == 404


def test_make_public_and_its_undo_leave_a_cleared_clip_alone(client, conn, clip_visit):
    from birdbrain import privacy
    vid, rel = clip_visit()
    image = db.get_visit(conn, vid)["image_path"]
    privacy.save_cache({image: 0.8, privacy.clip_key(rel): 0.7})
    client.post(f"/visit/{vid}/clip-public")
    client.post(f"/visit/{vid}/public")
    assert privacy.parse_override(db.get_visit(conn, vid)["public_ok"]) == {image: 0.8, privacy.clip_key(rel): 0.7}
    client.post(f"/visit/{vid}/private")
    assert privacy.parse_override(db.get_visit(conn, vid)["public_ok"]) == {privacy.clip_key(rel): 0.7}


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
    assert "<h3>Clips</h3>" in page and "(1 clip)" in page and "person check has looked through it" in page


def test_about_mentions_clips_and_the_person_check(client, conn):
    about = client.get("/about").text
    assert "<strong>Clips</strong>" in about and "seconds of video without sound, for" in about
    assert "never gets a photo or a clip where the Pi's person check fires" in about


def test_the_clips_news_pop_up_is_on_every_page_until_its_last_day(client, locked, now, monkeypatch):
    """Read-only visitors get it (the browser shows it once: static/news.js), and after the last day it is gone
    from the HTML altogether."""
    from birdbrain.web import news
    monkeypatch.setattr(news, "CLIPS_UNTIL", now.date())  # today is the last day
    for url in ("/", "/species", "/stats", "/about"):
        page = client.get(url).text
        assert 'id="news-popup" data-news="clips"' in page and f'data-until="{now.date().isoformat()}"' in page, url
        assert "/static/news.js" in page and 'href="/?kind=clip">See the clips</a>' in page
        assert f"New visits have a {config.CLIP_SECONDS:g} second video clip. Open a visit" in page
        # why there is no sound, in the light text and before the retention rule (the apostrophe is escaped in HTML)
        assert page.index('<p class="news-rules">Video only. Sound is never recorded (for privacy).</p>') \
            < page.index('<p class="news-rules">The latest')
        assert (f'<p class="news-rules">The latest {config.CLIP_KEEP_PER_SPECIES} clips of each species are kept.</p>'
                in page)
    monkeypatch.setattr(news, "CLIPS_UNTIL", now.date() - timedelta(days=1))  # yesterday was
    page = client.get("/").text
    assert "news-popup" not in page and "news.js" not in page


def test_clips_news_on_the_mirror_too_but_not_with_clips_off_or_in_an_embedded_page(client, now, monkeypatch):
    from birdbrain.web import news
    monkeypatch.setattr(news, "CLIPS_UNTIL", now.date() + timedelta(days=14))
    assert "news-popup" in client.get("/").text  # no password: the admin gets it too
    assert "news-popup" not in client.get("/?embed=1").text  # but not inside the False triggers pop-up's page
    monkeypatch.setattr(config, "CLIP_SECONDS", 0.0)
    assert "news-popup" not in client.get("/").text
    monkeypatch.setattr(config, "CLIP_SECONDS", 5.0)
    monkeypatch.setenv("BIRDBRAIN_PUBLIC", "1")
    assert "news-popup" in client.get("/").text


def test_the_tour_tells_the_clips_news_and_the_pop_up_remembers_being_closed(client):
    tour = client.get("/static/tour.js").text
    assert "kind=clip" in tour and 'news: "clips"' in tour and '"bb_news_" + step.news' in tour  # the step marks it seen
    script = client.get("/static/news.js").text
    assert '"bb_news_" + dlg.dataset.news' in script and 'addEventListener("close", remember)' in script
    assert "dataset.until" in script  # a page left open past the last day stays quiet too
