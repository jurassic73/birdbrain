"""Component tests of the web app through TestClient on a temp database: Pi only (birdbrain/pi_only.py) on the Stats
page (one muted paragraph) and the visit page (one sentence under "How it was decided")."""
import json
import re

from birdbrain import config

CLAUDE = {"frames": 1, "answer": {"visitor": "bird", "species": "Bushtit", "confidence": "high"}}
PRESENCE = {"model": "bioclip-presence", "score": 0.934, "threshold": 0.02, "empty": False}


def raw(record, **calls):
    return json.dumps({"calls": {"pi_only": record, "presence": PRESENCE, **calls}})


def pills_explained(html: str) -> bool:
    return html.count('class="badge ') == len(re.findall(r'class="badge [^"]*" data-tip=', html))


def paragraph(html: str) -> str:
    seg = html[html.index('id="pi-only"'):]
    return seg[:seg.index("</p>")]


def test_stats_page_has_one_sentence_before_any_verdict_is_recorded(client, add_visit):
    add_visit(species="Bushtit")
    r = client.get("/stats")
    assert r.status_code == 200
    seg = paragraph(r.text)
    assert "Pi only (shadow)" in seg and "No photo to score yet" in seg and "It changes nothing." in seg
    assert "75%" in seg and "0.90" in seg and "PI_ONLY" in seg  # the tip names the two bars and the switch


def test_stats_page_scores_the_pis_verdicts(client, add_visit):
    named = {"verdict": "named", "score": 0.4, "species": "Bushtit", "visitor": "bird", "by": "tight"}
    add_visit(species="Bushtit", raw_json=raw(named, haiku=CLAUDE))
    add_visit(species="Bushtit", raw_json=raw({"verdict": "held", "score": 0.93, "why": "score"}, haiku=CLAUDE))
    add_visit(species="Bushtit", raw_json=raw({"verdict": "empty", "score": 0.2}, haiku=CLAUDE))
    add_visit(visitor="none", species=None, raw_json=raw({"verdict": "empty", "score": 0.1}, haiku=CLAUDE))
    html = client.get("/stats").text
    seg = re.sub(r"\s+", " ", paragraph(html))
    assert "4 photos Claude answered, 3 with an animal." in seg
    assert "would have named 1 (0 wrong, 0 empty), held 1 for you (0 empty) and missed 1 animal (1 of them Claude named)" in seg
    assert "1 empty photo would have cost nothing" in seg
    assert pills_explained(html)


def test_stats_page_drops_the_paragraph_when_pi_only_is_off(client, monkeypatch):
    monkeypatch.setattr(config, "PI_ONLY", "off")
    assert 'id="pi-only"' not in client.get("/stats").text


def test_visit_page_adds_the_shadow_sentence_under_claudes_answer(client, add_visit):
    plain = add_visit(species="Bushtit", raw_json=json.dumps({"calls": {"haiku": CLAUDE, "presence": PRESENCE}}))
    held = add_visit(species="Bushtit", raw_json=raw({"verdict": "held", "score": 0.934, "why": "score"}, haiku=CLAUDE))
    before = client.get(f"/visit/{plain}").text
    html = client.get(f"/visit/{held}").text
    assert 'id="pi-only-note"' not in before
    assert ("Pi only (shadow): without Claude the Pi would have held this photo for you (the Pi animal check scored "
            "it 0.93). Not used.") in html
    assert html.count('class="badge ') == before.count('class="badge ') and pills_explained(html)  # no new pill


def test_visit_page_of_a_frame_the_pi_held_alone_says_claude_is_switched_off(client, add_visit):
    record = {"verdict": "held", "score": 0.934, "why": "score"}
    vid = add_visit(visitor="other", species=None, confidence="low", model="bioclip-presence", needs_review=1,
                    notes="Held for you: the Pi animal check scored it 0.934.",
                    raw_json=raw(record, skipped_claude=True, settled_by="pi_only"))
    r = client.get(f"/visit/{vid}")
    assert r.status_code == 200
    html = r.text.replace("&#39;", "'")
    assert "Claude is switched off (PI_ONLY=on), so the Pi decided alone: nothing could name it, but the Pi animal check scored it 0.93" in html
    assert "Claude was asked" not in html[html.index("How it was decided"):]
    assert "Judged empty, so Claude wasn't asked" not in html  # the check did not call it empty
    assert pills_explained(html)


def test_visit_page_of_a_false_trigger_the_pi_filed_alone_renders(client, add_visit):
    vid = add_visit(visitor="none", species=None, confidence="medium", model="bioclip-presence", count=0,
                    raw_json=raw({"verdict": "empty", "score": 0.21}, skipped_claude=True, settled_by="pi_only"))
    r = client.get(f"/visit/{vid}")
    assert r.status_code == 200 and "filed as a false trigger" in r.text and pills_explained(r.text)
