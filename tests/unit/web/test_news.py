"""Unit tests of birdbrain/web/news: pure functions, no files, database or network."""
from datetime import date, timedelta

from birdbrain.web import news

ON = {"seconds": 5.0, "days": 21, "per_species": 10}


def test_the_clips_notice_shows_through_its_last_day_and_never_after():
    last = news.CLIPS_UNTIL
    assert last == date(2026, 11, 25)  # two weeks from the day it went out
    for day in (last - timedelta(days=14), last):
        notice = news.clips_notice(day, **ON)
        assert notice["id"] == "clips" and notice["until"] == "2026-11-25" and notice["href"] == "/?kind=clip"
    assert news.clips_notice(last + timedelta(days=1), **ON) is None
    assert news.clips_notice(last + timedelta(days=400), **ON) is None


def test_no_clips_notice_where_there_are_no_clips():
    today = news.CLIPS_UNTIL
    assert news.clips_notice(today, **{**ON, "seconds": 0}) is None  # clips turned off


def test_the_notice_says_the_clip_length_and_the_retention_rules_from_the_settings():
    notice = news.clips_notice(news.CLIPS_UNTIL, **ON)
    assert notice["text"] == "New visits keep a 5 second video clip. Open a visit and press Play clip to watch it."
    assert notice["silent"] == "Video only. Sound is never recorded, to protect the neighbours' privacy."
    assert notice["rules"] == "Clips are kept for 21 days; the newest 10 of each species are kept longer."
    assert "star" not in notice["rules"] + notice["text"] + notice["silent"]  # visitors don't know about starred visits
    assert "a 7.5 second video" in news.clips_notice(news.CLIPS_UNTIL, **{**ON, "seconds": 7.5})["text"]
    assert news.clips_rules(30, 0) == "Clips are kept for 30 days."
    assert news.clips_rules(0, 10) == "Clips are kept for good."
