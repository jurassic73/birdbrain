import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from readenv import parse_env  # noqa: E402

PASS = "p@ss!$;#x"
FAKE = f"""# comment

GROUND_ZONE=0.1,0.2;0.3,0.4;0.5,0.6
export CAM_HOST=10.0.0.9
RTSP_USER='bob'
RTSP_PASS="{PASS}"
RTSP_PORT=8554
"""


def test_parse_env_handles_semicolons_quotes_export():
    env = parse_env(FAKE)
    assert env["GROUND_ZONE"] == "0.1,0.2;0.3,0.4;0.5,0.6"
    assert env["CAM_HOST"] == "10.0.0.9"
    assert env["RTSP_USER"] == "bob"
    assert env["RTSP_PASS"] == PASS


def test_snap_sh_loads_env_and_redacts(tmp_path):
    envf = tmp_path / "env"
    envf.write_text(FAKE)
    bindir = tmp_path / "bin"
    bindir.mkdir()
    log = tmp_path / "url.txt"
    fake = bindir / "ffmpeg"
    fake.write_text(
        '#!/bin/sh\nwhile [ $# -gt 0 ]; do [ "$1" = -i ] && echo "$2" > '
        f'{log}; shift; done\necho "fail rtsp://u:pw@h/x" >&2; exit 1\n'
    )
    fake.chmod(0o755)
    env = dict(os.environ, SNAP_ENV=str(envf), PATH=f"{bindir}:{os.environ['PATH']}")
    r = subprocess.run(
        ["bash", str(ROOT / "scripts/snap.sh"), "sub", str(tmp_path / "o.jpg")],
        env=env, capture_output=True, text=True,
    )
    assert "command not found" not in r.stderr
    assert r.returncode == 1
    assert "pw@" not in r.stderr and "rtsp://***@" in r.stderr
    assert log.read_text().strip() == "rtsp://bob:p%40ss%21%24%3B%23x@10.0.0.9:8554/ch1"
