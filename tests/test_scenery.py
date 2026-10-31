import json

from birdbrain import db, scenery

ROCK = [0.3, 0.40, 0.40, 0.05, 0.05]   # turns up in every empty frame
BIRD = [0.8, 0.60, 0.30, 0.08, 0.08]


def test_boxes_seen_in_empty_frames_are_scenery():
    background = [[ROCK], [ROCK], [ROCK]]
    assert scenery.novel_score([ROCK], background) == 0.0
    assert scenery.novel_score([ROCK, BIRD], background) == 0.8


def test_a_box_in_only_one_empty_frame_is_still_new():
    assert scenery.novel_score([BIRD], [[BIRD], [ROCK]]) == 0.8


def add_empty(conn, add_visit, objects, night=0, trigger="rim", visitor="none"):
    raw = {"trigger": trigger, "calls": {"detector": {"animal": False, "objects": objects}}}
    return add_visit(visitor=visitor, species=None, is_night=night, raw_json=json.dumps(raw))


def test_check_learns_from_empty_frames_of_the_same_zone_and_time(conn, add_visit):
    det = {"animal": False, "objects": [ROCK]}
    assert scenery.check(conn, det, ground=False, night=False) is None  # no history yet
    for _ in range(scenery.MIN_BACKGROUND):
        add_empty(conn, add_visit, [ROCK])
        add_empty(conn, add_visit, [BIRD], night=1)  # night frames don't teach the day
    assert scenery.check(conn, det, ground=False, night=False)["empty"] is True
    assert scenery.check(conn, {"animal": False, "objects": [BIRD]}, ground=False, night=False)["empty"] is False
    assert scenery.check(conn, det, ground=True, night=False) is None  # the ground keeps its own memory


def test_real_visits_never_become_scenery(conn, add_visit):
    for _ in range(scenery.MIN_BACKGROUND):
        add_empty(conn, add_visit, [ROCK])
        add_empty(conn, add_visit, [BIRD], visitor="bird")
    assert scenery.check(conn, {"animal": False, "objects": [BIRD]}, ground=False, night=False)["empty"] is False


def test_confirmed_species_needs_settled_visits(conn, add_visit):
    for _ in range(db.CONFIRMED_MIN):
        add_visit(species="Dark-eyed Junco", confidence="high")
        add_visit(species="Song Sparrow", confidence="medium")
    assert db.confirmed_species(conn) == {"Dark-eyed Junco"}
