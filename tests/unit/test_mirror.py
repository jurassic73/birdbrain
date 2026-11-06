"""Unit tests of birdbrain/mirror: pure functions, no files, database or network."""

import json

from helpers.mirror import CHECKED_RAW
from birdbrain import mirror


def test_public_raw_drops_what_nothing_public_reads():
    assert mirror.public_raw(None) is None and mirror.public_raw("not json") is None and mirror.public_raw("[1]") is None
    assert mirror.public_raw('{"motion": 1}') is None
    assert json.loads(mirror.public_raw('{"calls": {"x": 1}, "answer": {}, "stay_end": "t"}')) == {"calls": {"x": 1}}


def test_public_raw_can_drop_the_owner_checks_only():
    text = json.dumps(CHECKED_RAW)
    assert "owner_checks" in json.loads(mirror.public_raw(text))["calls"]
    off = json.loads(mirror.public_raw(text, owner_checks=False))
    assert set(off["calls"]) == {"haiku", "circle"} and off["trigger"] == "rim"
    assert json.loads(mirror.public_raw('{"calls": {"owner_checks": []}}', owner_checks=False)) == {"calls": {}}
    assert mirror.public_raw('{"calls": 3}', owner_checks=False) == '{"calls": 3}'  # not a dict: left as it is


def test_the_mirror_empties_the_whole_embeddings_table():
    from birdbrain import mirror
    assert "embeddings" in mirror.EMPTIED_TABLES
