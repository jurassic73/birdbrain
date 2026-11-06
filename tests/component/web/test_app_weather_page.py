"""The Weather & visits page: humidity and wind capture, the report math (db.weather_report) and the /weather route."""

import pytest
from fastapi.testclient import TestClient

from helpers.weather_page import NOW, week
from birdbrain import clock, config
from birdbrain.web import app as webapp, auth


@pytest.fixture
def client(conn, monkeypatch):
    monkeypatch.setattr(clock, "now", lambda: NOW)
    with TestClient(webapp.app) as c:
        yield c


def test_page_with_weather(client, week):
    html = client.get("/weather").text
    assert "Weather &amp; visits" in html and "2 days since Oct 31" in html
    assert "so far" in html and "wx-scatter" in html and "Dark-eyed Junco" in html
    assert "% cloud cover" in html and "mph" in html and "88%" in html


def test_page_shows_na_for_days_without_humidity(client, week):
    week.execute("UPDATE weather_days SET humidity_pct = NULL, wind_kmh = NULL, wind_max_kmh = NULL")
    week.commit()
    html = client.get("/weather").text
    assert "n/a" in html


def test_page_without_weather(client, add_visit, monkeypatch):
    add_visit(captured_at="2026-11-01T08:05:00")
    monkeypatch.setattr(config, "SITE_LAT", None)
    assert "Weather isn't set up" in client.get("/weather").text
    monkeypatch.setattr(config, "SITE_LAT", 1.5)
    monkeypatch.setattr(config, "SITE_LON", 2.5)
    monkeypatch.setattr(config, "WEATHER", True)
    assert "No weather saved yet" in client.get("/weather").text


def test_page_with_no_visits(client):
    assert "No photos yet" in client.get("/weather").text


def test_read_only_visitors_see_it(client, week, monkeypatch):
    monkeypatch.setenv("ADMIN_PASSWORD", "s3cret!")
    auth._fails.clear()
    page = client.get("/weather", follow_redirects=False)
    assert page.status_code == 200 and "Rain and visitors" in page.text


def test_stats_and_about_link_to_it(client, add_visit):
    add_visit()
    assert 'href="/weather"' in client.get("/stats").text
    assert 'href="/weather"' in client.get("/about").text


def test_research_is_folded_away_and_credit_stays_visible(client, week):
    html = client.get("/weather").text
    start = html.index('<details class="wx-more"')
    end = html.index("</details>", start)
    inside = html[start:end]
    assert "The numbers behind it" in inside
    assert 'id="wx-scatter"' in inside and "Hour by hour" in inside and "wx-rates" in inside
    assert "What the data says" in inside
    assert "open-meteo.com" in html[end:] and "CC BY 4.0" in html[end:]


def test_diary_is_slim_and_keeps_the_detail_in_tips(client, week):
    html = client.get("/weather").text
    for gone in ("wx-r-air", "wx-r-species", "wx-r-false", "False alarms"):
        assert gone not in html
    assert "false alarm" in html and "species." in html and "Humidity" in html
