"""Unit tests of birdbrain/privacy: pure functions, no files, database or network."""

import numpy as np

from birdbrain import privacy


def test_views_cover_the_frame_with_overlapping_quarters():
    img = np.zeros((100, 200, 3), np.uint8)
    v = privacy.views(img)
    assert len(v) == 5 and v[0].shape == (100, 200, 3)
    assert all(q.shape[0] > 50 and q.shape[1] > 100 for q in v[1:])
