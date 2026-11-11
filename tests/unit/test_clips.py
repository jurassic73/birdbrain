"""Unit tests of birdbrain/clips.py: which clips pruning takes, and where a clip is filed. No files, no database."""

from birdbrain import clips

CUTOFF = "2026-11-01T00:00:00"
OLD, OLDER, NEW = "2026-10-20T09:00:00", "2026-10-10T09:00:00", "2026-11-05T09:00:00"


def test_a_clip_is_filed_beside_its_photos_name():
    assert clips.rel_for("2026/11/10/101500_ab12cd.jpg") == "2026/11/10/101500_ab12cd.mp4"


def test_only_paths_inside_the_clips_folder_count():
    assert clips.inside("2026/11/10/a.mp4")
    assert not clips.inside("../a.mp4") and not clips.inside("/etc/passwd") and not clips.inside("")


def test_old_clips_go_and_new_ones_stay():
    rows = [(1, "Bushtit", OLD, False), (2, "Bushtit", NEW, False)]
    assert clips.prunable(rows, CUTOFF, per_species=0) == {1}


def test_a_starred_visit_keeps_its_clip_however_old():
    rows = [(1, "Bushtit", OLD, True), (2, "Bushtit", OLDER, False)]
    assert clips.prunable(rows, CUTOFF, per_species=0) == {2}


def test_a_starred_clip_takes_none_of_its_species_newest_places():
    # The favorite is the newest Bushtit clip, but it is kept anyway: the one place goes to the next clip.
    rows = [(1, "Bushtit", OLD, True), (2, "Bushtit", OLDER, False), (3, "Bushtit", "2026-10-01T09:00:00", False)]
    assert clips.prunable(rows, CUTOFF, per_species=1) == {3}


def test_each_species_keeps_its_newest_clips_even_past_the_cutoff():
    # Twelve old Bushtit clips and one old Varied Thrush: ten Bushtits stay (the newest), and the lone thrush.
    rows = [(i, "Bushtit", f"2026-10-{i:02d}T09:00:00", False) for i in range(1, 13)]
    rows.append((50, "Varied Thrush", OLDER, False))
    assert clips.prunable(rows, CUTOFF, per_species=10) == {1, 2}


def test_recent_clips_count_toward_a_species_ten():
    # Ten clips inside the window already make the ten: an older one beyond them goes.
    rows = [(i, "Bushtit", f"2026-11-{i:02d}T09:00:00", False) for i in range(1, 11)]
    rows.append((99, "Bushtit", OLD, False))
    assert clips.prunable(rows, CUTOFF, per_species=10) == {99}


def test_a_visit_with_two_species_stays_while_either_needs_it():
    # Visit 1 is the only Steller's Jay clip, though the Bushtits have newer ones to spare.
    rows = [(1, "Bushtit", OLDER, False), (1, "Steller's Jay", OLDER, False),
            (2, "Bushtit", OLD, False), (3, "Bushtit", NEW, False)]
    assert clips.prunable(rows, CUTOFF, per_species=1) == {2}


def test_a_clip_with_no_species_goes_by_age_alone():
    rows = [(1, None, OLD, False), (2, None, NEW, False)]
    assert clips.prunable(rows, CUTOFF, per_species=10) == {1}
