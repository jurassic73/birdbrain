"""Unit tests of birdbrain/audience: what the public mirror counts, and how the Guests page sums it. Pure: no files,
database or network; the clock is passed in as Unix seconds."""
from datetime import datetime, timedelta

import pytest

from birdbrain import audience
from birdbrain.audience import Counter

NOON = datetime(2026, 11, 7, 12, 0).timestamp()
BROWSER = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 Version/18.0 Safari/605.1.15"
VIA_CF = {"cf-connecting-ip": "203.0.113.9", "user-agent": BROWSER}


def counts(c: Counter) -> dict[str, int]:
    out: dict[str, int] = {}
    for _, metric, n in c.take():
        out[metric] = out.get(metric, 0) + n
    return out


@pytest.mark.parametrize("header, code", [
    ("NZ", "NZ"), ("us", "US"), (" de ", "DE"), ("T1", "T1"), ("XX", "XX"),
    (None, "XX"), ("", "XX"), ("USA", "XX"), ("1A", "XX"), ("N", "XX"), ("<script>", "XX"),
])
def test_country_is_cloudflares_two_letters_or_xx(header, code):
    assert audience.country(header) == code


@pytest.mark.parametrize("ua", ["", None, "curl/8.4.0", "python-httpx/0.28", "Googlebot/2.1", "facebookexternalhit/1.1",
                                "Mozilla/5.0 (compatible; bingbot/2.0)", "Slackbot-LinkExpanding 1.0", "HeadlessChrome/120",
                                "WhatsApp/2.23"])
def test_bots_and_link_previewers_are_skipped(ua):
    assert audience.is_bot(ua)


def test_a_browser_is_not_a_bot():
    assert not audience.is_bot(BROWSER)
    assert not audience.is_bot("Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 Chrome/131.0 Mobile Safari/537.36")


@pytest.mark.parametrize("route, name", [
    ("/", "home"), ("/live", "live"), ("/visit/{visit_id}", "visit"), ("/species", "species_list"),
    ("/species/{name}", "species"), ("/stats", "stats"), ("/weather", "weather"), ("/about", "about"),
    ("/api/latest-visit", None), ("/static", None), ("/images", None), ("/visits/more", None), ("/scoreboard", None),
    ("/guest-login", None), ("/species-info/{name}", None), (None, None), ("/guests", None),
])
def test_page_names_come_from_the_route(route, name):
    assert audience.page_name(route) == name


def test_only_requests_through_cloudflare_count():
    assert audience.skipped({"user-agent": BROWSER}, {})
    assert not audience.skipped(VIA_CF, {})


def test_the_owners_cookie_and_switch():
    assert audience.skipped(VIA_CF, {audience.NOCOUNT_COOKIE: "1"})
    assert audience.skipped(VIA_CF, {}, "no")  # the request that sets the cookie doesn't carry it yet
    assert not audience.skipped(VIA_CF, {audience.NOCOUNT_COOKIE: "1"}, "yes")  # switched back on: counted again
    assert not audience.skipped(VIA_CF, {audience.NOCOUNT_COOKIE: "0"})
    assert audience.owner_switch("no") == "off" and audience.owner_switch("YES") == "on"
    assert audience.owner_switch("maybe") is None and audience.owner_switch(None) is None


def test_a_bot_is_skipped_even_through_cloudflare():
    assert audience.skipped({**VIA_CF, "user-agent": "Googlebot/2.1"}, {})


@pytest.mark.parametrize("method, status, ctype, headers, want", [
    ("GET", 200, "text/html; charset=utf-8", {}, True),
    ("GET", 200, "text/html", {"sec-fetch-dest": "document"}, True),
    ("HEAD", 200, "text/html", {}, False),
    ("GET", 303, "text/html", {}, False),
    ("GET", 404, "text/html", {}, False),
    ("GET", 200, "application/json", {}, False),
    ("GET", 200, "text/html", {"hx-request": "true"}, False),
    ("GET", 200, "text/html", {"sec-fetch-dest": "empty"}, False),  # a script's fetch (the chips' in-place swap)
    ("GET", 200, "text/html", {"sec-fetch-dest": "document", "sec-purpose": "prefetch;prerender"}, False),
    ("GET", 200, "text/html", {"purpose": "prefetch"}, False),
])
def test_what_is_a_page_view(method, status, ctype, headers, want):
    assert audience.is_page_view(method, status, ctype, headers) is want


def test_a_person_is_a_visitor_once_a_day_with_country_and_returning():
    c = Counter()
    c.view("home", "203.0.113.9", BROWSER, "NZ", True, NOON)
    c.view("live", "203.0.113.9", BROWSER, "NZ", True, NOON + 60)
    c.view("home", "203.0.113.9", BROWSER, "NZ", True, NOON + 120)
    c.view("home", "198.51.100.4", BROWSER, "US", False, NOON + 180)
    assert counts(c) == {"page:home": 3, "page:live": 1, "page_people:home": 2, "page_people:live": 1,
                         "visitors": 2, "country:NZ": 1, "country:US": 1, "returning": 1}


def test_the_same_address_with_another_browser_is_another_person():
    c = Counter()
    c.view("home", "203.0.113.9", BROWSER, "NZ", False, NOON)
    c.view("home", "203.0.113.9", BROWSER + " other", "NZ", False, NOON)
    assert counts(c)["visitors"] == 2


def test_a_new_utc_day_draws_a_new_salt_and_counts_people_again():
    c = Counter()
    c.view("home", "203.0.113.9", BROWSER, "NZ", False, NOON)
    salt, name = c.salt, c.person("203.0.113.9", BROWSER)
    c.view("home", "203.0.113.9", BROWSER, "NZ", True, NOON + 3600)  # same day: not again
    c.view("home", "203.0.113.9", BROWSER, "NZ", True, NOON + 86400)  # next day: counted again, as returning
    assert c.salt != salt and c.person("203.0.113.9", BROWSER) != name
    assert counts(c)["visitors"] == 2
    assert "203.0.113.9" not in repr(c.state())  # the address is never kept, only its salted hash


def test_counts_land_in_five_minute_slots():
    c = Counter()
    c.view("home", "203.0.113.9", BROWSER, "NZ", False, NOON + 299)
    c.view("home", "203.0.113.9", BROWSER, "NZ", False, NOON + 300)
    assert {(s, m) for s, m, _ in c.take() if m == "page:home"} == {(int(NOON), "page:home"), (int(NOON) + 300, "page:home")}


def test_only_allowlisted_controls_and_pages_count():
    c = Counter()
    assert c.click("theme", "203.0.113.9", BROWSER, NOON)
    assert c.click("theme", "203.0.113.9", BROWSER, NOON)
    assert not c.click("evil<script>", "203.0.113.9", BROWSER, NOON)
    assert not c.click("", "203.0.113.9", BROWSER, NOON)
    c.view("no_such_page", "203.0.113.9", BROWSER, "NZ", False, NOON)
    assert counts(c) == {"click:theme": 2, "click_people:theme": 1}


def test_a_restart_keeps_todays_names_but_not_yesterdays():
    before = Counter()
    before.view("home", "203.0.113.9", BROWSER, "NZ", False, NOON)
    saved = before.state()
    after = Counter()
    after.restore(saved)
    after.view("home", "203.0.113.9", BROWSER, "NZ", False, NOON + 60)
    assert "visitors" not in counts(after)
    stale = Counter()
    stale.restore({**saved, "day": "2020-01-01"})
    stale.view("home", "203.0.113.9", BROWSER, "NZ", False, NOON)
    assert counts(stale)["visitors"] == 1


def test_the_feed_token(monkeypatch):
    assert not audience.token_ok("anything")  # no token configured: nobody
    monkeypatch.setenv("AUDIENCE_TOKEN", "s3cret-token")
    assert audience.token_ok("s3cret-token")
    assert not audience.token_ok("s3cret-tokeN") and not audience.token_ok("") and not audience.token_ok(None)


def test_counting_is_the_mirrors_and_can_be_switched_off(monkeypatch):
    assert audience.counting_on(True) and not audience.counting_on(False)
    monkeypatch.setenv("AUDIENCE_COUNT", "0")
    assert not audience.counting_on(True)


@pytest.mark.parametrize("key, n, unit, first_label", [("today", 24, "hour", "12a"), ("7d", 7, "day", "Nov 1"),
                                                        ("30d", 30, "day", "Oct 9"), ("year", 52, "week", "Nov 9")])
def test_periods_and_their_buckets(key, n, unit, first_label):
    now = datetime(2026, 11, 7, 15, 30)
    p = audience.period_bounds(key, now)
    assert len(p["buckets"]) == n and p["unit"] == unit and p["buckets"][0][1] == first_label
    assert p["buckets"][0][0] == p["start"] and p["end"] == now
    held = [b for b, _ in p["buckets"] if b <= now < b + p["step"]]
    assert len(held) == 1 and (key == "today" or held[0] == p["buckets"][-1][0])  # today's chart runs on to 11p
    assert p["prev_end"] - p["prev_start"] == p["end"] - p["start"]  # the previous period is as long


def test_bucket_values_sum_slots_into_their_bucket():
    now = datetime(2026, 11, 7, 15, 30)
    p = audience.period_bounds("today", now)
    at = lambda h, m=0: int(now.replace(hour=h, minute=m).timestamp())  # noqa: E731
    slots = {at(0): 1, at(0, 55): 2, at(15): 4, at(15, 25): 1, int((p["start"] - timedelta(minutes=5)).timestamp()): 9}
    values = audience.bucket_values(p["buckets"], p["step"], slots)
    assert values[0] == 3 and values[15] == 5 and sum(values) == 8  # the slot before midnight is yesterday's


def test_change_needs_something_to_compare_with():
    assert audience.change(110, 100) == 10 and audience.change(96, 100) == -4 and audience.change(5, 0) is None


def test_flags_and_map_shades():
    assert audience.flag("NZ") == "\U0001F1F3\U0001F1FF" and audience.flag("XX") == "\U0001F310"
    assert audience.map_step(1, 1) == audience.MAP_STEPS
    assert audience.map_step(1, 1000) == 1 and audience.map_step(1000, 1000) == audience.MAP_STEPS
    assert [audience.map_step(n, 100) for n in (1, 3, 10, 40, 100)] == sorted(audience.map_step(n, 100) for n in (1, 3, 10, 40, 100))
