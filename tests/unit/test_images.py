"""Unit tests of birdbrain/images: pure functions, no files, database or network."""

from datetime import datetime

from birdbrain import images


def test_stamp_reads_like_a_label():
    assert images.stamp_text(datetime(2026, 11, 1, 9, 33, 50)) == "Sun Nov 1  \u00b7  9:33 AM"
    assert images.stamp_text(datetime(2026, 10, 27, 19, 5)) == "Tue Oct 27  \u00b7  7:05 PM"
    assert images.stamp_text(datetime(2026, 10, 27, 0, 5)) == "Tue Oct 27  \u00b7  12:05 AM"
