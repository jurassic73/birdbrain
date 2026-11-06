"""Unit tests of birdbrain/lessons: pure functions, no files, database or network."""

from helpers.lessons import synchronous_distill  # noqa: F401  (autouse fixture)
from birdbrain import db, identify, lessons


def test_summary_is_its_own_section_between_the_yard_and_the_raw_lessons():
    site = {"location": "Here", "lessons_summary": "- Rock tips are not birds.\n- Juncos are common."}
    prompt = identify.system_prompt(["- a lesson"], site)
    assert "<lessons_summary>" in prompt and "</lessons_summary>" in prompt
    assert prompt.index("<yard_notes>") < prompt.index("<lessons_summary>") < prompt.index("<corrections>")
    assert "- Rock tips are not birds.\n- Juncos are common.\n</lessons_summary>" in prompt
    assert "the newest corrections follow in full" in prompt
    # the summary alone still gets the one line saying the owner's words never change the format
    alone = identify.system_prompt(None, {"lessons_summary": "- x"})
    assert identify.OWNER_INTRO in alone and "<corrections>" not in alone and "<yard_notes>" not in alone
    assert identify.system_prompt(None, {"lessons_summary": "  "}) == identify.SYSTEM


def test_summary_keys_stay_off_the_public_mirror():
    from birdbrain import mirror
    for key in (lessons.SUMMARY, lessons.SUMMARY_META, lessons.DRAFT, lessons.DISTILLED, lessons.FAILED, db.DISTILL_SPEND):
        assert key not in mirror.SETTINGS_KEYS and not key.startswith(mirror.SETTINGS_PREFIXES)
