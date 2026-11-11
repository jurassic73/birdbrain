"""Component tests of the web app through TestClient on a temp database: visit clips."""

import re

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
