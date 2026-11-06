"""Component tests of the web app through TestClient on a temp database: view."""

import json
import re

import pytest

from helpers.presence_view import LIVE_CROP, NARROW_CROP, live, need
from birdbrain import zones


def dashed_box(html) -> tuple[float, float, float, float]:
    rects = re.findall(r'<rect class="vz-crop"[^>]*? x="([\d.]+)" y="([\d.]+)" width="([\d.]+)" height="([\d.]+)"', html)
    assert len(rects) == 1, "one dashed box on the photo"
    return tuple(float(v) for v in rects[0])


def test_the_dashed_box_of_a_rim_visit_is_the_view_box_not_the_crop_box(client, add_visit):
    z = live(crop=NARROW_CROP)
    vid = add_visit(raw_json=json.dumps({"trigger": "rim", "zones": z.to_dict()}))
    box = dashed_box(client.get(f"/visit/{vid}").text)
    x, y, w, h = need("presence_view_box")(z, "rim")
    assert box == pytest.approx((x * 160, y * 90, w * 160, h * 90), abs=0.02)
    assert box != pytest.approx((NARROW_CROP[0] * 160, NARROW_CROP[1] * 90, NARROW_CROP[2] * 160, NARROW_CROP[3] * 90),
                                abs=0.5)


def test_the_dashed_box_of_a_water_visit_is_the_bowl_view_too(client, add_visit):
    vid = add_visit(raw_json=json.dumps({"trigger": "water", "zones": live(crop=NARROW_CROP).to_dict()}))
    x, y, w, h = need("presence_view_box")(live(crop=NARROW_CROP), "water")
    assert dashed_box(client.get(f"/visit/{vid}").text) == pytest.approx((x * 160, y * 90, w * 160, h * 90), abs=0.02)


def test_the_dashed_box_of_a_ground_visit_is_the_ground_crop(client, add_visit):
    vid = add_visit(visitor="none", species=None, raw_json=json.dumps({"trigger": "ground", "zones": live().to_dict()}))
    assert dashed_box(client.get(f"/visit/{vid}").text) == pytest.approx((0.0, 45.0, 160.0, 45.0), abs=0.02)  # GROUND_CROP


def test_the_dashed_box_without_a_ring_is_the_crop_the_pi_pads(client, add_visit):
    z = zones.Zones(crop=LIVE_CROP)
    vid = add_visit(raw_json=json.dumps({"trigger": "rim", "zones": z.to_dict()}))
    assert dashed_box(client.get(f"/visit/{vid}").text) == pytest.approx(
        (LIVE_CROP[0] * 160, LIVE_CROP[1] * 90, LIVE_CROP[2] * 160, LIVE_CROP[3] * 90), abs=0.02)
