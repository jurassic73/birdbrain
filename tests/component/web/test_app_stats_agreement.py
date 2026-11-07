"""Component tests of the Stats page through TestClient on a temp database: the shadow agreement bar table
"Tight look bars (shadow)" in the BioCLIP 2 section, and the sentence shown when there are no looks yet."""
import re

from helpers.agreement import EXPECTED, seed
from helpers.classifier import raw

HEADING = "Tight look bars (shadow)"
AGREE_COLUMNS = ("Named when the bowl look agrees", "Wrong when it agrees")


def classifier_section(html: str) -> str:
    start = html.index('id="classifier"')
    return html[start:html.index("</section>", start)]


def table_rows(section: str) -> list[list[str]]:
    """The cell texts of each body row of the table under the heading."""
    table = section[section.index(HEADING):] if HEADING in section else ""
    if "<table" not in table:
        return []
    table = table[table.index("<table"):table.index("</table>")]
    body = table[table.index("<tbody"):] if "<tbody" in table else table
    rows = []
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", body, re.S):
        cells = [re.sub(r"<[^>]+>", "", c).strip() for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", tr, re.S)]
        if cells and not all(c in ("Named", "Wrong", *AGREE_COLUMNS, "Bar", "") for c in cells):
            rows.append(cells)
    return rows


def test_stats_page_shows_the_table_with_data(client, add_visit):
    seed(add_visit)
    html = client.get("/stats").text
    section = classifier_section(html)
    assert HEADING in section
    for column in ("Named", "Wrong", *AGREE_COLUMNS):
        assert column in section
    rows = table_rows(section)
    assert len(rows) == 3, rows
    for row, want in zip(rows, EXPECTED["bars"]):
        bar = f"{want['bar']:.2f}"
        assert bar in row[0] or f"{round(want['bar'] * 100)}" in row[0], row  # 0.90 or 90%
        assert row[-4:] == [str(want["named"]), str(want["wrong"]), str(want["agree_named"]), str(want["agree_wrong"])], row


def test_stats_page_explains_the_table_and_that_it_changes_nothing(client, add_visit):
    seed(add_visit)
    section = classifier_section(client.get("/stats").text)
    assert HEADING in section
    after = section[section.index(HEADING):]
    assert 'class="muted"' in after
    assert "settled" in after.lower()  # the settled-species rule is left out of these counts, and the page says so
    assert re.search(r"changes nothing|does not change|doesn't change|not used|nothing changes|only keeps score",
                     after, re.I), after


def test_stats_page_shows_a_sentence_instead_of_the_table_without_looks(client, add_visit):
    add_visit(species="Bushtit", raw_json=raw("Bushtit"))  # one scored visit, so the BioCLIP 2 section renders
    html = client.get("/stats").text
    section = classifier_section(html)
    assert 'id="classifier"' in html and "first guess right (1/1)" in html
    for column in AGREE_COLUMNS:
        assert column not in section
    muted = re.findall(r'<p class="muted">(.*?)</p>', section, re.S)
    assert any("tight look" in p.lower() for p in muted), muted
    if HEADING in section:  # the heading may stay above the sentence; the table may not
        assert table_rows(section) == []


def test_stats_page_renders_on_an_empty_database(client):
    r = client.get("/stats")
    assert r.status_code == 200
    assert "Named when the bowl look agrees" not in r.text
