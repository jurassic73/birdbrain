import cv2
import numpy as np

from birdbrain import scene_fit

# A daylight frame (1280x720): mulch-colored noise, a light concrete bowl, dark water, a gray rock left of center.
BOWL = (0.5, 0.45, 0.2, 0.12)
WATER = (0.5, 0.45, 0.16, 0.095)
ROCK = (0.45, 0.46, 0.03, 0.02)


def frame(color: bool = True, water=(30, 32, 35), rim=(185, 190, 195)) -> bytes:
    W, H = 1280, 720
    rng = np.random.default_rng(3)
    img = rng.integers(60, 110, (H, W, 3), dtype=np.uint8)
    img[:, :, 1] = np.clip(img[:, :, 1].astype(int) + 25, 0, 255)  # greenish ground, so it isn't grayscale
    px = lambda e: ((int(e[0] * W), int(e[1] * H)), (int(e[2] * W), int(e[3] * H)))  # noqa: E731
    cv2.ellipse(img, *px(BOWL), 0, 0, 360, rim, -1)
    cv2.ellipse(img, *px(WATER), 0, 0, 360, water, -1)
    cv2.ellipse(img, *px(ROCK), 0, 0, 360, (140, 140, 140), -1)
    if not color:
        img = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    return cv2.imencode(".jpg", img)[1].tobytes()


def near(a, b, tol=0.012):
    return all(abs(x - y) <= tol for x, y in zip(a, b))


def test_round_objects_drawn_as_polygons_become_ellipses():
    o = scene_fit.as_ellipse({"name": "rock", "kind": "landing_spot", "note": "",
                              "points": [[0.4, 0.4], [0.5, 0.4], [0.5, 0.5], [0.4, 0.5]]})
    assert o["ellipse"] == [0.45, 0.45, 0.05, 0.05] and "points" not in o
    lawn = {"name": "lawn", "kind": "ground", "note": "", "points": [[0, 0.5], [1, 0.5], [1, 1]]}
    assert scene_fit.as_ellipse(lawn) is lawn


def test_a_rim_that_duplicates_the_bath_is_folded_in_and_a_water_object_becomes_its_inside():
    bath = {"name": "bowl", "kind": "birdbath", "note": "17.5 in.", "ellipse": list(BOWL)}
    rim = {"name": "rim", "kind": "perch", "note": "birds perch here",
           "points": [[0.3, 0.33], [0.7, 0.33], [0.7, 0.57], [0.3, 0.57]]}
    water = {"name": "water", "kind": "birdbath", "note": "", "ellipse": [0.5, 0.45, 0.17, 0.1]}
    rock = {"name": "rock", "kind": "landing_spot", "note": "", "ellipse": list(ROCK)}
    out = scene_fit.merge_bath([bath, rim, water, rock])
    assert [o["name"] for o in out] == ["bowl", "rock"]
    assert out[0]["inner"] == [0.5, 0.45, 0.17, 0.1] and out[0]["note"] == "17.5 in. birds perch here"


def test_rough_ellipses_snap_onto_the_bath_and_the_rock_by_day():
    objects = [{"name": "bowl", "kind": "birdbath", "note": "", "ellipse": [0.52, 0.43, 0.22, 0.13]},
               {"name": "rock", "kind": "distraction", "note": "", "ellipse": [0.5, 0.42, 0.04, 0.03]}]
    bath, rock = scene_fit.fit(objects, frame())
    assert near(bath["ellipse"], BOWL) and near(bath["inner"], WATER) and bath["fitted"] == ["water", "rim"]
    assert near(rock["ellipse"][:2], ROCK[:2]) and rock["fitted"] == ["photo"]


def test_at_night_the_bath_snaps_but_rocks_stay_where_claude_put_them():
    guess = [0.5, 0.42, 0.04, 0.03]
    bath, rock = scene_fit.fit([{"name": "bowl", "kind": "birdbath", "note": "", "ellipse": [0.52, 0.43, 0.22, 0.13]},
                                {"name": "rock", "kind": "landing_spot", "note": "", "ellipse": list(guess)}],
                               frame(color=False))
    assert near(bath["ellipse"], BOWL) and rock["ellipse"] == guess and "fitted" not in rock


def test_ground_is_cut_around_the_bath():
    bath = {"name": "bowl", "kind": "birdbath", "note": "", "ellipse": list(BOWL)}
    ground = {"name": "lawn", "kind": "ground", "note": "", "points": [[0, 0.4], [1, 0.4], [1, 1], [0, 1]]}
    _, cut = scene_fit.cut_ground([bath, ground])
    pts = [cut["points"], *cut.get("pieces", [])]
    # no corner of the new outline sits inside the bowl
    inside = [p for ps in pts for p in ps
              if ((p[0] - BOWL[0]) / BOWL[2]) ** 2 + ((p[1] - BOWL[1]) / BOWL[3]) ** 2 < 0.95]
    assert not inside and len(cut["points"]) > 4


def test_tidy_keeps_claudes_shapes_when_the_photo_is_unreadable():
    objects = [{"name": "bowl", "kind": "birdbath", "note": "", "ellipse": list(BOWL)}]
    assert scene_fit.tidy(objects, b"not a jpeg")[0]["ellipse"] == list(BOWL)


def test_a_bath_draws_as_a_ring_and_cut_ground_as_all_its_pieces():
    from birdbrain.web.app import scene_shape
    ring = scene_shape({"name": "bowl", "kind": "birdbath", "ellipse": list(BOWL), "inner": list(WATER)})
    assert ring["d"].count("M") == 2
    ground = scene_shape({"name": "lawn", "kind": "ground", "points": [[0, 0.5], [0.3, 0.5], [0.3, 1]],
                          "pieces": [[[0.7, 0.5], [1, 0.5], [1, 1]]]})
    assert ground["d"].count("M") == 2 and ground["d"].count("Z") == 2


def test_by_day_bright_water_in_a_darker_wet_rim_still_fits():
    # daylight on the real bath: the water mirrors the sky, the wet concrete is only a little lighter than the mulch
    jpeg = frame(water=(215, 205, 190), rim=(140, 145, 145))
    bath = scene_fit.fit([{"name": "bowl", "kind": "birdbath", "note": "", "ellipse": [0.47, 0.44, 0.24, 0.15]}], jpeg)[0]
    assert near(bath["ellipse"], BOWL, 0.015) and near(bath["inner"], WATER, 0.015)
