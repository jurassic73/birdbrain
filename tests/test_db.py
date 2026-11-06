import json
import sqlite3
from datetime import date, datetime, timedelta

import pytest

from helpers.clock import FROZEN_NOW
from birdbrain import config, db

TODAY = FROZEN_NOW.date()


def at(days_ago: int = 0, hour: int = 9) -> str:
    return datetime.combine(TODAY - timedelta(days=days_ago), datetime.min.time()).replace(hour=hour).isoformat()


def labels(rows) -> list[str]:
    return [r["label"] for r in rows]


def test_init_db_adds_new_columns_to_an_old_database(tmp_path):
    path = tmp_path / "old.db"
    old = sqlite3.connect(path)
    old.execute(
        "CREATE TABLE visits (id INTEGER PRIMARY KEY, captured_at TEXT NOT NULL, image_path TEXT NOT NULL, "
        "thumb_path TEXT NOT NULL, visitor TEXT NOT NULL, species TEXT, count INTEGER, confidence TEXT, notes TEXT, "
        "model TEXT, escalated INTEGER NOT NULL DEFAULT 0, needs_review INTEGER NOT NULL DEFAULT 0, "
        "corrected_species TEXT, is_night INTEGER NOT NULL DEFAULT 0, raw_json TEXT)"
    )
    old.execute("INSERT INTO visits (captured_at, image_path, thumb_path, visitor) VALUES ('2026-01-01T09:00:00', 'a', 'b', 'bird')")
    old.commit()
    old.close()
    with db.session(path) as conn:
        db.init_db(conn)
        db.init_db(conn)  # idempotent
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(visits)")}
        assert {"corrected_visitor", "feedback", "public_ok"} <= cols
        assert db.count_visits(conn) == 1


def test_init_db_survives_a_column_the_other_process_added_first(tmp_path, monkeypatch):
    """After a deploy the watcher and the web app both start and migrate: one's look at the table is stale by the
    time it ALTERs."""
    path = tmp_path / "race.db"
    with db.session(path) as conn:
        db.init_db(conn)  # the other process got there first: every column is there now
        monkeypatch.setattr(db, "_columns", lambda c, table: set())  # ... but this one looked before that
        db.init_db(conn)
    monkeypatch.undo()
    import threading

    path = tmp_path / "together.db"
    with db.session(path):
        pass  # the file exists and is in WAL mode: it's the migration that races, not the first open
    start, errors = threading.Barrier(2), []

    def migrate():
        try:
            start.wait(timeout=10)
            with db.session(path) as c:
                db.init_db(c)
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=migrate) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    with db.session(path) as conn:
        assert set(db.ADDED_COLUMNS) <= {r["name"] for r in conn.execute("PRAGMA table_info(visits)")}


def test_set_raw_entry_keeps_what_landed_meanwhile(conn, add_visit):
    """A backfill reads the row, runs a slow model, then writes: only its own entry, so what the watcher and the
    owner stored in between stays."""
    vid = add_visit(raw_json=json.dumps({"trigger": "rim", "calls": {"circle": {"x": 0.5, "y": 0.5, "r": 0.1}}}))
    raw = json.loads(db.get_visit(conn, vid)["raw_json"])  # the backfill's read
    with db.session() as other:  # the watcher's and the web app's connections, while the model runs
        db.end_stay(other, vid, "2026-11-01T09:05:00", "left")
        db.set_raw_entry(other, vid, "$.calls.owner_checks", [{"source": "pi", "verdict": "sure"}])
    db.set_raw_entry(conn, vid, "$.calls.classifier", {"top": [], "backfilled": True})
    after = json.loads(db.get_visit(conn, vid)["raw_json"])
    assert after["stay_end"] == "left" and after["calls"]["owner_checks"][0]["verdict"] == "sure"
    assert after["calls"]["classifier"]["backfilled"] and after["calls"]["circle"] == raw["calls"]["circle"]
    assert after["trigger"] == "rim"
    for broken in ("not json", "[1, 2]", None):
        conn.execute("UPDATE visits SET raw_json = ? WHERE id = ?", (broken, vid))
        conn.commit()
        db.set_raw_entry(conn, vid, "$.calls.classifier", {"top": []})
        assert json.loads(db.get_visit(conn, vid)["raw_json"]) == {"calls": {"classifier": {"top": []}}}


def test_writing_takes_the_write_lock_before_reading(conn, add_visit):
    other = sqlite3.connect(config.DB_PATH, timeout=0.2)
    with db.writing(conn):
        with pytest.raises(sqlite3.OperationalError, match="locked"):
            other.execute("BEGIN IMMEDIATE")
    other.execute("BEGIN IMMEDIATE")  # free again once the block ends
    other.rollback()
    with pytest.raises(ValueError):
        with db.writing(conn):
            conn.execute("INSERT INTO settings (key, value) VALUES ('x', '1')")
            raise ValueError("rolled back")
    assert db.get_value(conn, "x") is None and not conn.in_transaction
    other.close()


def test_public_ok_is_stored_and_taken_back(conn, add_visit):
    vid, other = add_visit(), add_visit()
    db.set_public_ok(conn, vid, {"at": "2026-11-02T09:00:00", "files": {"img.jpg": 0.278}})
    assert json.loads(db.get_visit(conn, vid)["public_ok"])["files"] == {"img.jpg": 0.278}
    assert db.get_visit(conn, other)["public_ok"] is None
    db.set_public_ok(conn, vid, None)
    assert db.get_visit(conn, vid)["public_ok"] is None


def test_labels_prefer_corrections_then_species_then_capitalized_visitor(conn, add_visit):
    add_visit(species="Song Sparrow")
    add_visit(visitor="raccoon", species=None)
    add_visit(species="Dark-eyed Junco", corrected_species="Spotted Towhee")
    add_visit(species="Dark-eyed Junco", corrected_visitor="squirrel")
    assert sorted(labels(db.list_visits(conn))) == ["Raccoon", "Song Sparrow", "Spotted Towhee", "Squirrel"]


def test_false_triggers_are_separate_from_real_visits(conn, add_visit):
    real = add_visit()
    nothing = add_visit(visitor="none", species=None)
    rock = add_visit(species="Dark-eyed Junco", corrected_visitor="none")       # owner said: no animal
    legacy = add_visit(visitor="none", species=None, corrected_species="Bushtit")  # pre-corrected_visitor fix
    assert {r["id"] for r in db.list_visits(conn)} == {real, legacy}
    assert {r["id"] for r in db.list_visits(conn, false_triggers=True)} == {nothing, rock}
    assert db.count_visits(conn) == 2 and db.count_visits(conn, false_triggers=True) == 2


def test_neighbors_stay_within_real_or_false_trigger_set(conn, add_visit):
    old = add_visit(captured_at=at(3))
    ft_old = add_visit(captured_at=at(2), visitor="none", species=None)
    mid = add_visit(captured_at=at(2, hour=10))
    tie = add_visit(captured_at=at(2, hour=10))
    ft_new = add_visit(captured_at=at(1), visitor="none", species=None)
    new = add_visit(captured_at=at(0))
    assert db.get_neighbors(conn, mid) == (tie, old)
    assert db.get_neighbors(conn, tie) == (new, mid)
    assert db.get_neighbors(conn, new) == (None, tie)
    assert db.get_neighbors(conn, old) == (mid, None)
    assert db.get_neighbors(conn, ft_new, false_triggers=True) == (None, ft_old)
    assert db.is_false_trigger(conn, ft_old) and not db.is_false_trigger(conn, mid)


def test_filters(conn, add_visit):
    add_visit(species="Bushtit", captured_at=at(0, 9))
    add_visit(visitor="raccoon", species=None, is_night=1, captured_at=at(1, 2))
    add_visit(species="Bushtit", corrected_visitor="possum", corrected_species=None, captured_at=at(10, 22), is_night=1)
    assert labels(db.list_visits(conn, kind="bird")) == ["Bushtit"]
    assert sorted(labels(db.list_visits(conn, kind="mammal"))) == ["Possum", "Raccoon"]
    assert sorted(labels(db.list_visits(conn, tod="night"))) == ["Possum", "Raccoon"]
    assert labels(db.list_visits(conn, species="Raccoon")) == ["Raccoon"]
    assert labels(db.list_visits(conn, date_from=TODAY.isoformat(), date_to=TODAY.isoformat())) == ["Bushtit"]
    assert db.count_visits(conn, date_from=(TODAY - timedelta(days=2)).isoformat()) == 2


def test_multiple_filter_uses_effective_also(conn, add_visit):
    also = '[{"visitor": "bird", "species": "Song Sparrow", "count": 1}]'
    a = add_visit(species="Bushtit", also=also, captured_at=at(0, 9))
    add_visit(species="Bushtit", captured_at=at(0, 10))
    add_visit(species="Bushtit", also=also, corrected_also="[]", captured_at=at(0, 11))
    d = add_visit(species="Bushtit", corrected_also=also, captured_at=at(0, 12))
    pair = add_visit(species="Bushtit", count=2, captured_at=at(0, 13))  # two of one species count too
    ids = {r["id"] for r in db.list_visits(conn, kind="multiple")}
    assert ids == {a, d, pair}


def test_list_order_paging_and_review_queue_oldest_first(conn, add_visit):
    ids = [add_visit(captured_at=at(d), needs_review=1) for d in (2, 0, 1)]
    newest_first = [r["id"] for r in db.list_visits(conn)]
    assert newest_first == [ids[1], ids[2], ids[0]]
    assert [r["id"] for r in db.list_visits(conn, limit=1, offset=1)] == [ids[2]]
    assert [r["id"] for r in db.list_visits(conn, review=True)] == [ids[0], ids[2], ids[1]]


def test_species_summary_counts_dates_sorting_and_photo_choice(conn, add_visit):
    add_visit(species="Bushtit", captured_at=at(5), thumb_path="old_high.jpg", confidence="high")
    add_visit(species="Bushtit", captured_at=at(1), thumb_path="new_low.jpg", confidence="low")
    add_visit(species="Song Sparrow", captured_at=at(3))
    add_visit(visitor="none", species=None, captured_at=at(0))  # false triggers never become a species
    rows = {r["label"]: r for r in db.species_summary(conn)}
    assert set(rows) == {"Bushtit", "Song Sparrow"}
    bushtit = rows["Bushtit"]
    assert bushtit["visits"] == 2 and bushtit["first_seen"] == at(5) and bushtit["last_seen"] == at(1)
    assert bushtit["thumb_path"] == "old_high.jpg"  # high confidence beats newer
    assert labels(db.species_summary(conn, sort="newest")) == ["Song Sparrow", "Bushtit"]  # by first sighting
    assert labels(db.species_summary(conn, sort="name")) == ["Bushtit", "Song Sparrow"]
    assert labels(db.species_summary(conn, sort="visits", limit=1)) == ["Bushtit"]


def test_species_summary_prefers_owner_corrected_photos(conn, add_visit):
    add_visit(species="Bushtit", thumb_path="claude_high.jpg", confidence="high", captured_at=at(0))
    add_visit(species="Wren", corrected_species="Bushtit", thumb_path="owner.jpg", confidence="low", captured_at=at(3))
    assert db.species_row(conn, "Bushtit")["thumb_path"] == "owner.jpg"
    assert db.species_row(conn, "Nope") is None


def test_species_names_are_distinct_sorted_and_real_only(conn, add_visit):
    add_visit(species="Song Sparrow")
    add_visit(species="Bushtit")
    add_visit(species="Bushtit")
    add_visit(visitor="none", species=None)
    assert db.species_names(conn) == ["Bushtit", "Song Sparrow"]


def test_daily_and_hourly_counts_fill_gaps(conn, add_visit):
    add_visit(captured_at=at(0, 9))
    add_visit(captured_at=at(0, 9))
    add_visit(captured_at=at(2, 17))
    add_visit(visitor="none", species=None, captured_at=at(0, 3))
    daily = db.daily_counts(conn, days=5)
    assert len(daily) == 5 and daily[-1] == (TODAY.isoformat(), 2) and daily[-3] == ((TODAY - timedelta(days=2)).isoformat(), 1)
    assert sum(n for _, n in daily) == 3
    hourly = db.hourly_counts(conn)
    assert len(hourly) == 24 and hourly[9] == 2 and hourly[17] == 1 and hourly[3] == 0
    assert db.hourly_counts(conn, false_triggers=True)[3] == 1


def test_today_summary(conn, add_visit):
    assert db.today_summary(conn)["latest"] is None
    add_visit(species="Bushtit", captured_at=at(0, 8))
    add_visit(species="Song Sparrow", captured_at=at(0, 9), needs_review=1)
    add_visit(species="Bushtit", captured_at=at(3, 9))
    add_visit(visitor="none", species=None, captured_at=at(0, 10))
    s = db.today_summary(conn)
    assert (s["visits"], s["species"], s["false_triggers"], s["review"]) == (2, 2, 1, 1)
    assert s["latest"]["label"] == "Song Sparrow"
    assert [(c["label"], c["n"]) for c in s["critters"]] == [("Song Sparrow", 1), ("Bushtit", 1)]
    add_visit(species="Bushtit", captured_at=at(0, 11))
    assert [(c["label"], c["n"]) for c in db.today_summary(conn)["critters"]][0] == ("Bushtit", 2)
    assert s["last_motion"]["visitor"] == "none"


def test_scoreboard_counts_between_dates(conn, add_visit):
    add_visit(species="Bushtit", captured_at=at(0, 8))
    add_visit(species="Bushtit", captured_at=at(1, 8), also='[{"visitor": "bird", "species": "Song Sparrow", "count": 1}]')
    add_visit(species="Steller's Jay", captured_at=at(40, 8))
    add_visit(visitor="none", species=None, captured_at=at(1, 9))
    rows = lambda *a: [(r["label"], r["n"]) for r in db.scoreboard(conn, *a)]
    day = lambda n: at(n, 0)[:10]
    assert rows(day(1), day(1)) == [("Bushtit", 1), ("Song Sparrow", 1)]
    assert rows(day(6), day(0)) == [("Bushtit", 2), ("Song Sparrow", 1)]
    assert rows() == [("Bushtit", 2), ("Song Sparrow", 1), ("Steller's Jay", 1)]
    # The total counts photos: the two-animal photo is one visitor, the false trigger none.
    assert db.scoreboard_total(conn, day(1), day(1)) == 1
    assert db.scoreboard_total(conn, day(6), day(0)) == 2
    assert db.scoreboard_total(conn) == 3


def test_latest_visit_skips_false_triggers(conn, add_visit):
    bird = add_visit(captured_at=at(0, 8))
    add_visit(visitor="none", species=None, captured_at=at(0, 9))
    assert db.latest_visit(conn)["id"] == bird
    assert db.latest_visit(conn, real_only=False)["visitor"] == "none"


@pytest.mark.parametrize(
    "visitor, species, expected",
    [
        ("bird", "Dark-eyed Junco", (None, None)),          # unchanged: nothing stored
        ("bird", "Spotted Towhee", (None, "Spotted Towhee")),
        ("none", "Dark-eyed Junco", ("none", None)),        # no animal: species ignored
        ("other", "Cat", ("other", None)),
        ("squirrel", "", ("squirrel", None)),
        ("squirrel", "Douglas Squirrel", ("squirrel", "Douglas Squirrel")),
        (" bird ", " Spotted Towhee ", (None, "Spotted Towhee")),
        ("", "", (None, None)),
    ],
)
def test_plan_correction(conn, add_visit, visitor, species, expected):
    visit = db.get_visit(conn, add_visit(visitor="bird", species="Dark-eyed Junco"))
    assert db.plan_correction(visit, visitor, species) == expected


def test_plan_correction_rejects_unknown_visitor(conn, add_visit):
    visit = db.get_visit(conn, add_visit())
    with pytest.raises(ValueError):
        db.plan_correction(visit, "dragon", "")


def test_correct_visit_clears_review_and_undo(conn, add_visit):
    vid = add_visit(needs_review=1)
    db.correct_visit(conn, vid, "none", None, "  rock tips  ")
    v = db.get_visit(conn, vid)
    assert (v["corrected_visitor"], v["feedback"], v["needs_review"], v["label"]) == ("none", "rock tips", 0, "None")
    db.correct_visit(conn, vid, None, None, None)
    v = db.get_visit(conn, vid)
    assert (v["corrected_visitor"], v["feedback"], v["label"]) == (None, None, "Dark-eyed Junco")


def test_recent_lessons_wording_order_and_limit(conn, add_visit):
    add_visit()  # uncorrected visits are not lessons
    add_visit(species="Dark-eyed Junco", corrected_visitor="none", feedback="Rock tips\nafter a refill.", captured_at=at(0))
    add_visit(visitor="none", species=None, corrected_visitor="bird", corrected_species="Bushtit", is_night=1, captured_at=at(0))
    add_visit(species="Song Sparrow", feedback="Right, note the streaks.", captured_at=at(0))
    lessons = db.recent_lessons(conn)
    assert len(lessons) == 3
    newest, middle, oldest = lessons
    assert 'you said "Song Sparrow" and the owner confirmed it. Owner\'s note: Right, note the streaks.' in newest
    assert 'night (IR): you said "no animal"; the owner says it was "Bushtit".' in middle
    assert 'you said "Dark-eyed Junco"; the owner says it was "no animal". Owner\'s note: Rock tips after a refill.' in oldest
    assert len(db.recent_lessons(conn, limit=2)) == 2


def _skipped(by: str = "presence") -> str:
    """raw_json of a visit the Pi settled without asking Claude (identify.settled_*)."""
    calls = {"skipped_claude": True, "detector": {"best": None}}
    if by == "presence":
        calls["presence"] = {"score": 0.01, "threshold": 0.1, "empty": True}
    if by == "classifier":
        calls["settled_by"] = "classifier"
    return json.dumps({"calls": calls})


def test_a_lesson_on_a_visit_claude_never_saw_says_so(conn, add_visit):
    empty = {"visitor": "none", "species": None, "captured_at": at(0)}
    add_visit(**empty, raw_json=_skipped(), feedback="Wind ripples on the rim.")
    add_visit(**empty, raw_json=_skipped(), corrected_visitor="bird", corrected_species="Bushtit", is_night=1)
    add_visit(**empty, raw_json=_skipped("detector"), feedback="Leaf shadow.")
    add_visit(species="Bushtit", raw_json=_skipped("classifier"), corrected_species="Lesser Goldfinch", captured_at=at(0))
    add_visit(**empty, raw_json="{not json", feedback="Bad trail.")  # unreadable: treated as Claude's answer
    bad, classifier, detector, bushtit, ripples = db.recent_lessons(conn)
    day = at(0)[:10]
    assert ripples == (f"- {day}, day: the Pi animal check judged this frame empty, so you weren't asked; the owner "
                       "confirmed no animal was there. Owner's note: Wind ripples on the rim.")
    assert bushtit == (f"- {day}, night (IR): the Pi animal check judged this frame empty, so you weren't asked; "
                       'the owner says it was "Bushtit".')
    assert detector.endswith("day: the Pi's local detector found no animal, so you weren't asked; the owner confirmed "
                             "no animal was there. Owner's note: Leaf shadow.")
    assert classifier.endswith('the BioCLIP 2 species check named it "Bushtit", so you weren\'t asked; the owner says '
                               'it was "Lesser Goldfinch".')
    assert bad.endswith('day: you said "no animal" and the owner confirmed it. Owner\'s note: Bad trail.')
    rows = db.all_lessons(conn)
    assert [r["said_by"] for r in rows] == ["Claude", "the Pi", "the Pi", "the Pi", "the Pi"]


def test_agreeing_with_the_pi_and_saying_nothing_more_is_not_a_lesson(conn, add_visit):
    empty = {"visitor": "none", "species": None, "captured_at": at(0)}
    # Older rows could store the owner's "no animal" even when it matched: nothing for Claude to learn.
    add_visit(**empty, raw_json=_skipped(), corrected_visitor="none")
    add_visit(**empty, raw_json=_skipped(), corrected_visitor="none", corrected_also="[]")
    circled = add_visit(**empty, raw_json=_skipped(), corrected_visitor="none",
                        marks=json.dumps([{"x": 0.5, "y": 0.5, "r": 0.1}]))
    claude = add_visit(**empty, corrected_visitor="none")  # Claude's own answer confirmed: still a lesson
    assert [r["visit_id"] for r in db.all_lessons(conn)] == [claude, circled]
    assert "The owner circled spots" in db.recent_lessons(conn)[1]
    from birdbrain import lessons  # nor does the Lessons summary read it
    assert set(lessons.fingerprint(db.all_lessons(conn))) == {str(claude), str(circled)}


def test_the_prompt_preview_shows_a_skipped_visits_lesson(conn, add_visit):
    from fastapi.testclient import TestClient
    from birdbrain.web import app as webapp

    add_visit(visitor="none", species=None, raw_json=_skipped(), feedback="Wind ripples.", captured_at=at(0))
    with TestClient(webapp.app) as client:
        page = client.get("/notes").text
    assert "the Pi animal check judged this frame empty, so you weren&#39;t asked" in page
    assert "or the Pi&#39;s where a line says you weren&#39;t asked" in page  # the corrections intro
    assert "the Pi said &#34;no animal&#34;" in page or 'the Pi said "no animal"' in page


def test_settings_default_empty_save_trim_and_overwrite(conn):
    assert db.get_settings(conn) == {"location": "", "scene_notes": "", "target_notes": "", "ignore_notes": "", "lessons_summary": ""}
    db.save_settings(conn, location="  Vancouver, WA  ", ignore_notes="branch on the right")
    db.save_settings(conn, location="Portland, OR")
    assert db.get_settings(conn) == {"location": "Portland, OR", "scene_notes": "", "target_notes": "", "ignore_notes": "branch on the right",
                                     "lessons_summary": ""}
    with pytest.raises(ValueError):
        db.save_settings(conn, password="nope")


def test_detector_report_scores_gate_mode_against_final_answers(conn, add_visit):
    import json

    def raw(animal):
        return json.dumps({"calls": {"detector": {"animal": animal}}})

    add_visit(visitor="none", species=None, raw_json=raw(False))                       # would save a call
    add_visit(visitor="none", species=None, raw_json=raw(False))                       # would save a call
    missed = add_visit(species="Bushtit", raw_json=raw(False))                         # would miss a bird
    add_visit(species="Song Sparrow", raw_json=raw(True))                              # caught
    add_visit(species="Robin", corrected_visitor="none", raw_json=raw(False))          # owner: no animal, saved
    add_visit(visitor="raccoon", species=None, is_night=1, raw_json=raw(False))        # night: never gated
    add_visit(visitor="none", species=None, model="yolox-tiny", raw_json=raw(False))   # gated already, no Claude
    add_visit(species="Wren")                                                          # no detector data
    r = db.detector_report(conn)
    assert (r["checked"], r["would_skip"], r["would_save"], r["caught"], r["animals"]) == (5, 4, 3, 1, 2)
    assert r["would_miss"] == [{"id": missed, "label": "Bushtit"}]


def test_avg_haiku_call_cost_from_saved_token_counts(conn, add_visit):
    import json

    assert db.avg_haiku_call_usd(conn) == 0.0
    for i_tok, o_tok in ((1000, 100), (3000, 300)):
        add_visit(raw_json=json.dumps({"calls": {"haiku": {"input_tokens": i_tok, "output_tokens": o_tok}}}))
    # average 2000 in, 200 out: 2000 * 0.10/M + 200 * 0.50/M
    assert db.avg_haiku_call_usd(conn) == pytest.approx(0.0003)


def test_activity_rows_add_up_for_today(conn, add_visit):
    import json

    tokens = {"haiku": {"input_tokens": 2000, "output_tokens": 200}}

    def raw(animal):
        return json.dumps({"calls": {**tokens, "presence": {"empty": not animal}}})

    add_visit(visitor="none", species=None, raw_json=raw(False))                               # Claude said none; gate would keep
    add_visit(species="Bushtit", raw_json=raw(True))                                           # real visitor
    add_visit(visitor="none", species=None, is_night=1, raw_json=raw(True))                    # the check wasn't sure
    add_visit(visitor="none", species=None, model="bioclip-presence", raw_json='{"calls": {}}')  # settled on the Pi
    add_visit(visitor="none", species=None, raw_json=raw(False), captured_at=at(3))            # another day
    db.bump(conn, "probe_empty")
    db.bump(conn, "probe_empty")
    db.bump(conn, "probe_empty", day=(TODAY - timedelta(days=2)).isoformat())
    a = db.activity(conn)
    assert (a["visitors"], a["species"], a["false_alarms"], a["empty_on_pi"]) == (1, 1, 5, 3)
    assert a["empty_by_claude"] == 2 and a["motion"] == 6
    assert (a["sent_to_claude"], a["would_keep"]) == (3, 1)
    assert a["visitors"] + a["false_alarms"] == 4 + 2  # every motion event today: 4 stored visits + 2 probes
    assert a["claude_usd"] == pytest.approx(0.0009) and a["would_keep_usd"] == pytest.approx(0.0003)
    assert db.activity(conn, day=(TODAY - timedelta(days=2)).isoformat())["empty_on_pi"] == 1


def test_activity_funnel_adds_up_and_bioclip_visitors_are_not_empties(conn, add_visit):
    add_visit(species="Bushtit", model="bioclip-2")                                   # real, named on the Pi
    add_visit(species="Dark-eyed Junco", model="claude-haiku-5-5")                    # real, named by Claude
    add_visit(visitor="none", species=None, model="bioclip-presence")                 # empty, settled on the Pi
    add_visit(visitor="none", species=None, model="yolox-tiny")                       # empty, settled on the Pi
    add_visit(visitor="none", species=None, model="claude-haiku-5-5")                 # empty, Claude checked
    add_visit(visitor="none", species=None, model="bioclip-presence", captured_at=at(3))  # another day
    for _ in range(4):
        db.bump(conn, "probe_empty")
    a = db.activity(conn)
    assert (a["visitors"], a["false_alarms"], a["motion"]) == (2, 7, 9)
    assert a["motion"] == a["visitors"] + a["false_alarms"]
    assert (a["empty_on_pi"], a["empty_by_claude"]) == (2 + 4, 1)  # the bioclip-2 visitor is no empty
    assert a["empty_on_pi"] + a["empty_by_claude"] == a["false_alarms"]
    assert a["id_bioclip"] + a["id_claude"] == a["visitors"]


def test_activity_with_no_motion_is_all_zeros(conn):
    a = db.activity(conn)
    assert (a["motion"], a["visitors"], a["species"], a["false_alarms"], a["empty_on_pi"], a["empty_by_claude"]) == (0,) * 6


def test_first_visit_at_follows_the_total(conn, add_visit):
    assert db.first_visit_at(conn) is None
    add_visit(species="Bushtit", captured_at=at(5))
    add_visit(species="Bushtit", captured_at=at(2))
    add_visit(visitor="none", species=None, captured_at=at(9))  # a false trigger isn't in the total
    assert db.first_visit_at(conn) == at(5) and db.count_visits(conn) == 2


def test_activity_splits_visitors_by_who_named_them(conn, add_visit):
    add_visit(species="Bushtit", model="bioclip-2")
    add_visit(species="Dark-eyed Junco", model="claude-haiku-5-5")
    add_visit(species="Steller's Jay", model="claude-sonnet-5-5")
    add_visit(visitor="none", species=None, model="claude-haiku-5-5")  # a false trigger isn't a named visitor
    add_visit(species="Bushtit", model="bioclip-2", captured_at=at(3))  # another day
    a = db.activity(conn)
    assert (a["id_bioclip"], a["id_claude"], a["visitors"]) == (1, 2, 3)


def test_delete_visit_returns_the_row_once(conn, add_visit):
    vid = add_visit(image_path="a.jpg")
    assert db.delete_visit(conn, vid)["image_path"] == "a.jpg"
    assert db.delete_visit(conn, vid) is None
    assert db.get_visit(conn, vid) is None


def test_detector_mode_from_settings_overrides_env(conn, monkeypatch):
    monkeypatch.setattr(config, "DETECTOR", "shadow")
    assert db.detector_mode(conn) == "shadow"
    db.set_value(conn, "detector_mode", "gate")
    assert db.detector_mode(conn) == "gate"
    db.set_value(conn, "detector_mode", "nonsense")
    assert db.detector_mode(conn) == "shadow"


@pytest.mark.parametrize("animals, missed, mode, words", [
    (5, 0, "shadow", "5 of 20"),
    (25, 1, "shadow", "missed 1 of 25"),
    (25, 0, "gate", "all 25"),
])
def test_gate_advice(animals, missed, mode, words):
    report = {"animals": animals, "would_miss": [{"id": i, "label": "Bushtit"} for i in range(missed)]}
    got, why = db.gate_advice(report)
    assert got == mode and words in why


def test_totals_cover_every_day_and_add_up(conn, add_visit):
    import json

    haiku = {"input_tokens": 2000, "output_tokens": 200}
    sonnet = {"input_tokens": 3000, "output_tokens": 300}
    add_visit(species="Bushtit", raw_json=json.dumps({"calls": {"haiku": haiku}}), captured_at=at(10))
    add_visit(species="Bushtit", raw_json=json.dumps({"calls": {"haiku": haiku}}))
    add_visit(species="Steller's Jay", escalated=1, model="claude-sonnet-5-5", needs_review=1,
              raw_json=json.dumps({"calls": {"haiku": haiku, "sonnet": sonnet}}))
    add_visit(visitor="other", species=None)  # a real visitor, but not a species
    add_visit(visitor="none", species=None, raw_json=json.dumps({"calls": {"haiku": haiku}}))
    add_visit(visitor="none", species=None, model="yolox-tiny", raw_json='{"calls": {}}')
    vid = add_visit(visitor="none", species=None)
    db.correct_visit(conn, vid, None, None, feedback="Leaf on the rocks")
    db.bump(conn, "probe_empty", day=(TODAY - timedelta(days=5)).isoformat())
    db.bump(conn, "probe_empty")
    t = db.totals(conn)
    assert t["since"] == at(10)
    assert (t["visitors"], t["species"], t["false_alarms"], t["events"]) == (4, 2, 5, 9)
    assert (t["empty_on_pi"], t["sent_to_claude"], t["haiku_only"], t["escalated"]) == (3, 6, 5, 1)
    assert (t["waiting_review"], t["corrected"]) == (1, 1)
    assert t["claude_usd"] == pytest.approx(4 * 0.0003 + 0.009)


def test_totals_on_an_empty_database(conn):
    t = db.totals(conn)
    assert t["events"] == 0 and t["since"] is None and t["claude_usd"] == 0.0


def test_capture_rate_counts_quiet_days_and_gaps(conn, add_visit):
    assert db.capture_rate(conn) == {"photos": 0}
    day = (TODAY - timedelta(days=1)).isoformat()
    add_visit(captured_at=f"{day}T08:00:00")
    add_visit(captured_at=f"{day}T08:10:00")
    add_visit(captured_at=f"{day}T08:40:00")
    add_visit(visitor="none", species=None, captured_at=f"{day}T09:00:00")
    add_visit(captured_at=(TODAY - timedelta(days=30)).isoformat() + "T12:00:00")  # outside the week
    r = db.capture_rate(conn)
    assert r["recent_days"] == 7 and r["photos_per_day"] == pytest.approx(4 / 7)
    assert r["visitors_per_day"] == pytest.approx(3 / 7)
    assert r["all_photos_per_day"] == pytest.approx(5 / 31)
    assert r["busiest_hour"] == 8 and r["median_gap_min"] == 30


def test_switched_off_lessons_leave_the_prompt_and_the_next_newest_moves_up(conn, add_visit):
    ids = [add_visit(feedback=f"note {i}", captured_at=at(0)) for i in range(db.LESSON_LIMIT + 1)]
    rows = db.all_lessons(conn)
    assert [r["in_use"] for r in rows] == [True] * db.LESSON_LIMIT + [False]
    db.set_lesson_off(conn, ids[-1], True)
    rows = db.all_lessons(conn)
    assert rows[0]["off"] and not rows[0]["in_use"] and rows[-1]["in_use"]
    assert len(db.recent_lessons(conn)) == db.LESSON_LIMIT and "note 0" in db.recent_lessons(conn)[-1]
    assert db.get_visit(conn, ids[-1])["feedback"] == f"note {db.LESSON_LIMIT}"  # the label and note stay
    db.correct_visit(conn, ids[-1], None, None, "new note")  # saving again switches it back on
    assert "new note" in db.recent_lessons(conn)[0]


def test_a_long_owner_note_is_cut_for_the_prompt_but_kept_on_the_visit(conn, add_visit):
    note = "word " * 100  # 500 characters
    vid = add_visit(feedback=note, captured_at=at(0))
    line = db.recent_lessons(conn)[0]
    sent = line.split("Owner's note: ")[1]
    assert len(sent) == db.NOTE_LIMIT and sent.endswith("…") and sent.startswith("word word")
    assert db.get_visit(conn, vid)["feedback"] == note
    assert db.all_lessons(conn)[0]["feedback"] == note  # the edit form shows all of it
    add_visit(feedback="short", captured_at=at(0))
    assert db.recent_lessons(conn)[0].endswith("Owner's note: short")


def test_the_review_queue_includes_a_doubtful_empty_frame(conn, add_visit):
    doubt = add_visit(visitor="none", species=None, confidence="low", needs_review=1)
    add_visit(visitor="none", species=None, confidence="high")
    bird = add_visit(confidence="low", needs_review=1)
    assert [r["id"] for r in db.list_visits(conn, review=True)] == [doubt, bird]
    assert db.count_visits(conn, review=True) == 2
    assert db.count_visits(conn) == 1 and db.count_visits(conn, false_triggers=True) == 2  # still a false trigger
    db.mark_reviewed(conn, doubt)
    assert db.count_visits(conn, review=True) == 1


def test_cached_prompt_tokens_cost_their_own_price(conn, add_visit):
    import json

    haiku = {"input_tokens": 1000, "output_tokens": 100, "cache_read_input_tokens": 2000, "cache_creation_input_tokens": 0}
    sonnet = {"input_tokens": 4000, "output_tokens": 300, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 2000}
    add_visit(raw_json=json.dumps({"calls": {"haiku": haiku, "sonnet": sonnet}}), escalated=1, model="claude-sonnet-5-5")
    expect_haiku = (1000 * 0.10 + 100 * 0.50 + 2000 * 0.01) / 1e6
    expect_sonnet = (4000 * 2.00 + 300 * 10.00 + 2000 * 2.50) / 1e6
    assert db.call_usd(db.USD_PER_MTOK["haiku"], haiku) == pytest.approx(expect_haiku)
    assert db.totals(conn)["claude_usd"] == pytest.approx(expect_haiku + expect_sonnet)
    assert db.activity(conn)["claude_usd"] == pytest.approx(expect_haiku + expect_sonnet)
    assert db.avg_haiku_call_usd(conn) == pytest.approx(expect_haiku)
    for price in db.USD_PER_MTOK.values():  # reads are cheaper than fresh input, writes dearer
        assert price["cache_read"] < price["input"] < price["cache_write"]


def test_prompt_site_is_claudes_notes_only(conn):
    # an old Claude outline left in the settings table is no longer described to Claude: zones live on Settings
    db.set_value(conn, "scene", json.dumps({"objects": [{"name": "rock", "kind": "distraction", "note": ""}]}))
    db.save_settings(conn, ignore_notes="A drain tube.")
    site = db.prompt_site(conn)
    assert "objects" not in site and site["ignore_notes"] == "A drain tube."


def test_species_kinds_name_each_label_with_its_visitor_type(conn, add_visit):
    add_visit(species="Bushtit")
    add_visit(visitor="squirrel", species="Eastern Gray Squirrel")
    add_visit(visitor="squirrel", species=None)
    add_visit(visitor="none", species=None)
    assert db.species_kinds(conn) == [("Bushtit", "bird"), ("Eastern Gray Squirrel", "squirrel"), ("Squirrel", "squirrel")]


def test_neighbors_with_species_skip_other_species_and_count_also(conn, add_visit):
    also = '[{"visitor": "bird", "species": "Bushtit", "count": 1}]'
    old = add_visit(captured_at=at(3), species="Bushtit")
    add_visit(captured_at=at(2), species="Song Sparrow")
    mid = add_visit(captured_at=at(1, hour=10), species="Dark-eyed Junco", also=also)
    add_visit(captured_at=at(1), species="Song Sparrow")
    relabelled = add_visit(captured_at=at(0, hour=1), species="Bushtit", corrected_species="Song Sparrow")
    new = add_visit(captured_at=at(0), species="Bushtit")
    assert db.get_neighbors(conn, new, species="Bushtit") == (None, mid)
    assert db.get_neighbors(conn, mid, species="Bushtit") == (new, old)
    assert db.get_neighbors(conn, old, species="Bushtit") == (mid, None)
    # a visit not in the set still steps by its time
    assert db.get_neighbors(conn, relabelled, species="Bushtit") == (new, mid)
    assert db.get_neighbors(conn, mid) != db.get_neighbors(conn, mid, species="Bushtit")


def test_id_split_by_day_starts_at_the_first_bioclip_day(conn, add_visit):
    assert db.id_split_by_day(conn) == []
    add_visit(species="Dark-eyed Junco", model="claude-haiku-5-5", captured_at=at(5))  # before BioCLIP 2 named anyone
    assert db.id_split_by_day(conn) == []
    add_visit(species="Bushtit", model="bioclip-2", captured_at=at(3))
    add_visit(species="Bushtit", model="bioclip-2", captured_at=at(3, 10))
    add_visit(species="Steller's Jay", model="claude-sonnet-5-5", captured_at=at(3, 11))
    add_visit(visitor="none", species=None, model="claude-haiku-5-5", captured_at=at(2))  # false trigger: not counted
    add_visit(species="Dark-eyed Junco", model="claude-haiku-5-5", captured_at=at(0))
    out = db.id_split_by_day(conn)
    assert [d["day"] for d in out] == [(TODAY - timedelta(days=n)).isoformat() for n in (3, 2, 1, 0)]
    assert [(d["bioclip"], d["claude"]) for d in out] == [(2, 1), (0, 0), (0, 0), (0, 1)]


def test_id_split_by_day_is_capped_at_the_last_days(conn, add_visit):
    add_visit(species="Bushtit", model="bioclip-2", captured_at=at(60))
    add_visit(species="Bushtit", model="bioclip-2", captured_at=at(29))
    add_visit(species="Bushtit", model="claude-haiku-5-5", captured_at=at(30))
    out = db.id_split_by_day(conn, days=30)
    assert len(out) == 30 and out[0]["day"] == (TODAY - timedelta(days=29)).isoformat() and out[-1]["day"] == TODAY.isoformat()
    assert (out[0]["bioclip"], out[0]["claude"]) == (1, 0) and sum(d["claude"] for d in out) == 0
