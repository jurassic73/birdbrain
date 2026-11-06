"""Unit tests of birdbrain/scripts/readenv: pure functions, no files, database or network."""

import sys
from pathlib import Path

from helpers.scripts import FAKE, PASS

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from readenv import parse_env  # noqa: E402


def test_parse_env_handles_semicolons_quotes_export():
    env = parse_env(FAKE)
    assert env["GROUND_ZONE"] == "0.1,0.2;0.3,0.4;0.5,0.6"
    assert env["CAM_HOST"] == "10.0.0.9"
    assert env["RTSP_USER"] == "bob"
    assert env["RTSP_PASS"] == PASS
