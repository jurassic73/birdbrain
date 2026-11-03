"""Talks to the camera's own web API (Thingino) for the Settings > Camera section.

Thingino serves JSON CGI scripts under /x/. They accept a session cookie or an API key; the
key goes in ?token= because the camera's busybox httpd drops custom headers like X-API-Key.
The key is /etc/thingino-api.key on the camera (made in its web UI) and CAM_API_KEY in .env.

  picture   json-prudynt-config.cgi (read, the "image" section) and json-prudynt-save.cgi
            (applies live through prudyntctl and saves to /etc/prudynt.json, so it survives reboots)
  day/night json-config-daynight.cgi (the "daynight" key of /etc/thingino.json; reloads daynightd;
            nested sections like "schedule" are merged, so a partial one is safe)
  status    json-daynight-sensors.cgi (mode, light, camera clock), json-system-usage.cgi (memory)

Every change is logged in the settings table (camera_log) with the values it replaced, so it
can be undone: picture changes move the motion zones' background and the IR switch.
"""

import json
import logging
import re
import sqlite3
import time
from datetime import datetime
from typing import NamedTuple

import httpx

from . import config, db

log = logging.getLogger("birdbrain.camera")


class Field(NamedTuple):
    key: str          # prudynt image key, or a daynight key ("schedule.start_at" for nested ones)
    label: str
    tab: str
    tip: str
    lo: int = 0
    hi: int = 255
    default: int | None = None   # Thingino's default, for the Defaults button (picture fields)
    choices: dict | None = None  # value -> label: buttons instead of a slider
    kind: str = "range"          # range, choice, toggle or time


TABS = {"picture": "Picture", "light": "Light", "color": "Color", "daynight": "Day and night"}

# The picture (prudynt's image section). Flips are left out: they'd move every zone, and the zones
# and calibration depend on the framing. Ranges and names follow Thingino's own image page.
IMAGE = [
    Field("brightness", "Brightness", "picture", "Overall brightness. Default 128.", default=128),
    Field("contrast", "Contrast", "picture", "Difference between light and dark. Default 128.", default=128),
    Field("sharpness", "Sharpness", "picture", "Edge sharpening. Too high adds halos and noise. Default 128.", default=128),
    Field("sinter_strength", "Noise reduction", "picture", "Smooths grain, mostly at night. Too high smears feathers. Default 128.", default=128),
    Field("defog_strength", "Defog", "picture", "Adds contrast in haze and fog. Default 128.", default=128),
    Field("dpc_strength", "Dead pixels", "picture", "Hides stuck bright or dark pixels on the sensor. Default 128.", default=128),
    Field("ae_compensation", "Exposure", "light", "Auto-exposure target: higher lets in more light. Default 128.", default=128),
    Field("backlight_compensation", "Backlight", "light", "Brightens dark subjects against a bright background, like a bird against the sky in the water. Default 0.", 0, 10, default=0),
    Field("drc_strength", "Dynamic range", "light", "Lifts shadows and holds highlights in contrasty light (WDR). Default 128.", default=128),
    Field("highlight_depress", "Highlights", "light", "Pulls down bright spots, like sun glinting off the water. Default 0.", default=0),
    Field("max_again", "Max sensor gain", "light", "How far the sensor may amplify in low light. Lower is darker but less grainy at dusk and night. Default 160.", 0, 160, default=160),
    Field("max_dgain", "Max digital gain", "light", "Extra brightening after the sensor; adds more noise than sensor gain. Default 80.", 0, 160, default=80),
    Field("anti_flicker", "Anti-flicker", "light", "Matches exposure to mains lighting so porch lights don't band. North America is 60 Hz.",
          default=2, choices={0: "Off", 1: "50 Hz", 2: "60 Hz"}, kind="choice"),
    Field("saturation", "Saturation", "color", "Color strength by day (night is grayscale). Default 128.", default=128),
    Field("hue", "Hue", "color", "Shifts every color around the color wheel. Default 128.", default=128),
    Field("core_wb_mode", "White balance", "color", "How the camera judges white. Automatic suits a yard that goes from sun to shade.",
          default=0, kind="choice", choices={0: "Automatic", 1: "Manual", 2: "Daylight", 3: "Cloudy", 4: "Incandescent",
                                             5: "Fluorescent", 6: "Twilight", 7: "Shade", 8: "Warm fluorescent", 9: "Custom"}),
    Field("wb_rgain", "Red gain", "color", "Red strength. Used only with Manual or Custom white balance.", 0, 1024, default=0),
    Field("wb_bgain", "Blue gain", "color", "Blue strength. Used only with Manual or Custom white balance.", 0, 1024, default=0),
]
PICTURE_DEFAULTS = {f.key: f.default for f in IMAGE}

FORCE_MODES = {"": "Automatic", "day": "Always day", "night": "Always night"}
# The daynight config (Thingino calls it photosensing). Nested keys are flattened with a dot.
DAYNIGHT = [
    Field("force_mode", "Mode", "daynight", "Automatic follows the light; the others lock the camera in one mode.",
          choices=FORCE_MODES, kind="choice"),
    Field("night_threshold", "Switch to night below", "daynight", "Light level (percent) that turns infrared on.", 1, 99),
    Field("day_threshold", "Switch to day above", "daynight", "Light level (percent) that turns infrared off. Keep it above the night level so it doesn't flicker.", 1, 99),
    Field("transition_delay_s", "Wait before switching", "daynight", "Seconds the light must stay past a level before the camera switches.", 0, 120),
    Field("night_count_threshold", "Dark readings needed", "daynight", "Readings in a row below the night level before switching to night.", 1, 30),
    Field("day_count_threshold", "Bright readings needed", "daynight", "Readings in a row above the day level before switching to day.", 1, 30),
    Field("sample_interval_ms", "Reading every (ms)", "daynight", "How often the camera checks the light, in milliseconds.", 200, 10000),
    Field("controls.color", "Grayscale at night", "daynight", "Switch the picture to grayscale in night mode.", kind="toggle"),
    Field("controls.ircut", "IR-cut filter", "daynight", "Move the infrared filter aside at night. Leave on, or day colors turn pink.", kind="toggle"),
    Field("controls.ir850", "850 nm IR lights", "daynight", "The faint red-glowing infrared LEDs; most of the night light.", kind="toggle"),
    Field("controls.ir940", "940 nm IR lights", "daynight", "The invisible infrared LEDs; weaker, but nothing can see them glow.", kind="toggle"),
    Field("schedule.enabled", "Check the light only between", "daynight", "Outside these hours the camera stops checking the light and keeps its mode.", kind="toggle"),
    Field("schedule.start_at", "From", "daynight", "When light checks start.", kind="time"),
    Field("schedule.stop_at", "Until", "daynight", "When light checks stop.", kind="time"),
]
FIELDS = {f.key: f for f in IMAGE + DAYNIGHT}

LOG_KEY = "camera_log"
LOG_KEEP = 20
TIMEOUT = 10


class CameraError(Exception):
    pass


def configured() -> bool:
    return bool(config.CAM_API_KEY)


def _url(name: str) -> str:
    return f"http://{config.CAM_HOST}/x/{name}"


def _call(name: str, body: dict | None = None) -> dict:
    """GET (or POST JSON to) one CGI; its parsed answer. Never logs the URL: it holds the key."""
    if not configured():
        raise CameraError("CAM_API_KEY isn't set")
    params = {"token": config.CAM_API_KEY}
    try:
        with httpx.Client(timeout=TIMEOUT) as client:
            if body is None:
                r = client.get(_url(name), params=params)
            else:
                r = client.post(_url(name), params=params, content=json.dumps(body),
                                headers={"Content-Type": "application/json"})
        data = r.json()
    except (httpx.HTTPError, ValueError) as e:
        raise CameraError(f"camera didn't answer ({type(e).__name__})") from None
    if isinstance(data, dict) and data.get("error"):
        err = data["error"]
        raise CameraError(err.get("message", "error") if isinstance(err, dict) else str(err))
    return data


def _flat(d: dict) -> dict:
    """{"schedule": {"enabled": true}} -> {"schedule.enabled": true}, for the keys in DAYNIGHT."""
    out = {}
    for f in DAYNIGHT:
        head, _, tail = f.key.partition(".")
        v = d.get(head)
        if tail:
            v = v.get(tail) if isinstance(v, dict) else None
        if v is not None:
            out[f.key] = v
    return out


def _nest(values: dict) -> dict:
    out: dict = {}
    for k, v in values.items():
        head, _, tail = k.partition(".")
        if tail:
            out.setdefault(head, {})[tail] = v
        else:
            out[k] = v
    return out


def picture() -> dict[str, int]:
    image = _call("json-prudynt-config.cgi").get("image") or {}
    out = {k: int(image[k]) for k in PICTURE_DEFAULTS if k in image}
    out.setdefault("sinter_strength", 128)  # prudynt leaves it out until it's first saved
    return out


def daynight() -> dict:
    return _flat(_call("json-config-daynight.cgi"))


def status() -> dict:
    """What the camera says about itself right now; missing parts are left out."""
    out: dict = {}
    try:
        s = _call("json-daynight-sensors.cgi")
        out["mode"] = s.get("mode")
        out["light"] = s.get("daynight_brightness")
        try:
            if s.get("time_now"):
                out["clock_off"] = int(float(s["time_now"]) - time.time())
        except (TypeError, ValueError, OverflowError):
            pass  # a clock we can't read is unknown, not an error
    except CameraError:
        pass
    try:
        mem = _call("json-system-usage.cgi")["data"]["memory"]
        out["mem_free"] = round(100 * (mem["free"] + mem["cached"] + mem["buffers"]) / mem["total"])
    except (CameraError, KeyError, TypeError, ZeroDivisionError):
        pass
    return out


def _clean(fields: list[Field], form: dict, tab: str = "") -> dict:
    """The form's values for these fields (one tab's, if given), checked and clamped."""
    out: dict = {}
    for f in fields:
        if tab and f.tab != tab:
            continue
        raw = form.get(f.key)
        if raw is None:
            continue  # toggles always send a value (a hidden 0 before the checkbox)
        if f.kind == "toggle":
            out[f.key] = raw in ("1", "on", "true", True)
        elif f.kind == "time":
            if not re.fullmatch(r"([01]\d|2[0-3]):[0-5]\d", raw):
                raise ValueError(f"{f.label}: use a time like 06:00.")
            out[f.key] = raw
        elif f.kind == "choice":
            value = int(raw) if isinstance(next(iter(f.choices)), int) else raw
            if value not in f.choices:
                raise ValueError(f"Unknown {f.label.lower()}")
            out[f.key] = value
        elif raw != "":
            out[f.key] = max(f.lo, min(f.hi, int(float(raw))))
    return out


def clean_picture(form: dict, tab: str = "") -> dict[str, int]:
    """The picture values in a form, clamped to their ranges; unknown keys dropped."""
    return _clean(IMAGE, form, tab)


def defaults(tab: str = "") -> dict[str, int]:
    return {f.key: f.default for f in IMAGE if not tab or f.tab == tab}


def clean_daynight(form: dict) -> dict:
    out = _clean(DAYNIGHT, form)
    if "night_threshold" in out and "day_threshold" in out and out["day_threshold"] <= out["night_threshold"]:
        raise ValueError("The day level must be above the night level.")
    return out


def show(key: str, value) -> str:
    """A logged value the way the form shows it."""
    f = FIELDS.get(key)
    if value is None:
        return "?"
    if f and f.choices:
        return str(f.choices.get(value, value))
    if f and f.kind == "toggle":
        return "on" if value else "off"
    return str(value)


def history(conn: sqlite3.Connection) -> list[dict]:
    try:
        return json.loads(db.get_value(conn, LOG_KEY) or "[]")
    except ValueError:
        return []


def _record(conn: sqlite3.Connection, kind: str, before: dict, after: dict) -> None:
    changed = {k: v for k, v in after.items() if before.get(k) != v}
    if not changed:
        return
    rows = history(conn)
    rows.insert(0, {"id": int(time.time() * 1000), "at": datetime.now().isoformat(timespec="seconds"),
                    "kind": kind, "before": {k: before.get(k) for k in changed}, "after": changed})
    db.set_value(conn, LOG_KEY, json.dumps(rows[:LOG_KEEP]))


def set_picture(conn: sqlite3.Connection, values: dict[str, int], log_it: bool = True) -> dict[str, int]:
    before = picture()
    _call("json-prudynt-save.cgi", {"image": values})
    if log_it:
        _record(conn, "picture", before, values)
    return picture()


def set_daynight(conn: sqlite3.Connection, values: dict, log_it: bool = True) -> dict:
    """Values use the flat keys ("schedule.enabled"); they're posted nested."""
    before = daynight()
    _call("json-config-daynight.cgi", {"daynight": _nest(values)})
    if log_it:
        _record(conn, "daynight", before, values)
    return daynight()


def undo(conn: sqlite3.Connection, change_id: int) -> bool:
    """Put back what one logged change replaced; the entry is dropped from the log."""
    rows = history(conn)
    row = next((r for r in rows if r["id"] == change_id), None)
    if not row:
        return False
    before = {k: v for k, v in row["before"].items() if v is not None}
    if before:
        (set_picture if row["kind"] == "picture" else set_daynight)(conn, before, log_it=False)
    db.set_value(conn, LOG_KEY, json.dumps([r for r in rows if r["id"] != change_id]))
    return True


def reboot() -> None:
    """Restart the camera (~5 minutes to come back). It answers with a redirect page, not JSON."""
    if not configured():
        raise CameraError("CAM_API_KEY isn't set")
    try:
        r = httpx.get(_url("reboot.cgi"), params={"token": config.CAM_API_KEY}, timeout=TIMEOUT)
    except httpx.HTTPError:
        raise CameraError("camera didn't answer") from None
    if r.status_code == 401 or b"Authentication required" in r.content:
        raise CameraError("the camera refused the API key")
