"""Component tests of birdbrain/camera.py: one real local dependency, a temp database or files on disk."""

from helpers.camera import cam
from birdbrain import camera


def test_picture_change_is_logged_and_undone(conn, cam):
    camera.set_picture(conn, {"brightness": 150, "contrast": 128})
    assert cam["posts"] == [("json-prudynt-save.cgi", {"image": {"brightness": 150, "contrast": 128}})]
    (row,) = camera.history(conn)
    assert row["before"] == {"brightness": 128} and row["after"] == {"brightness": 150}  # unchanged contrast left out
    assert camera.undo(conn, row["id"])
    assert cam["image"]["brightness"] == 128 and camera.history(conn) == []


def test_daynight_is_posted_under_its_key(conn, cam):
    camera.set_daynight(conn, {"night_threshold": 20})
    assert cam["posts"][-1] == ("json-config-daynight.cgi", {"daynight": {"night_threshold": 20}})


def test_nested_daynight_settings_are_flat_in_the_form_and_nested_on_the_camera(conn, cam):
    assert camera.daynight()["schedule.start_at"] == "06:00" and camera.daynight()["controls.ir850"] is True
    camera.set_daynight(conn, camera.clean_daynight({"controls.ir850": "0", "schedule.start_at": "07:30"}))
    assert cam["posts"][-1] == ("json-config-daynight.cgi",
                                {"daynight": {"controls": {"ir850": False}, "schedule": {"start_at": "07:30"}}})
    assert cam["daynight"]["schedule"]["stop_at"] == "22:00"  # the rest of the section is kept
    (row,) = camera.history(conn)
    assert camera.undo(conn, row["id"])
    assert cam["daynight"]["controls"]["ir850"] is True and cam["daynight"]["schedule"]["start_at"] == "06:00"
