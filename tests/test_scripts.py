import os
import subprocess
import sys
from pathlib import Path

import pytest

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


def fake_tools(tmp_path, stage_marker: str = "ok"):
    """Stand-ins for everything push_public.sh calls, each logging its arguments."""
    bindir, log = tmp_path / "bin", tmp_path / "calls.txt"
    bindir.mkdir()
    logger = f'printf "%s\\n" "$(basename "$0") $*" >> {log}\n'
    tools = {
        "rsync": '#!/bin/sh\n[ "$1" = --version ] && { echo "rsync  version 3.2.7  protocol version 31"; exit 0; }\n'
                 + logger + "exit 0\n",
        "curl": "#!/bin/sh\nexit 22\n",  # no local site: the last live frame is used instead
        "timedatectl": '#!/bin/sh\necho "Europe/Paris"\n',
        # The venv python: `mirror prepare` stages a database and one photo link; `privacy check` finds a person.
        "fakepython": '#!/bin/sh\n' + logger +
                      'if [ "$2" = birdbrain.mirror ]; then mkdir -p "$5/images/2026/11/03" && touch "$5/birdbrain.db" '
                      '&& echo 2026/11/03/a.jpg > "$5/files.txt" && ln -s /nonexistent "$5/images/2026/11/03/a.jpg"; fi\n'
                      'if [ "$2" = birdbrain.privacy ]; then exit 1; fi\nexit 0\n',
    }
    for name, body in tools.items():
        (bindir / name).write_text(body)
        (bindir / name).chmod(0o755)
    return bindir, log


def push_env(tmp_path, bindir):
    data = tmp_path / "data"
    data.mkdir(exist_ok=True)
    (data / "live.jpg").write_bytes(b"jpeg")
    return dict(os.environ, PATH=f"{bindir}:{os.environ['PATH']}", PUBLIC_DEST="deploy@mirror:",
                PUBLIC_KEY="/nonexistent/key", BIRDBRAIN_DATA=str(data), PUBLIC_PYTHON=str(bindir / "fakepython"))


def test_push_public_sends_only_the_staged_files_with_timeouts(tmp_path):
    bindir, log = fake_tools(tmp_path)
    env = push_env(tmp_path, bindir)
    r = subprocess.run(["bash", str(ROOT / "scripts/push_public.sh")], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    calls = log.read_text().splitlines()
    prepare = next(c for c in calls if "birdbrain.mirror prepare" in c)
    assert prepare.endswith("Europe/Paris") and f"{tmp_path}/data/birdbrain.db" in prepare
    rsyncs = [c for c in calls if c.startswith("rsync ")]
    assert len(rsyncs) == 2  # the photos and the database; the live frame showed a person and stayed home
    assert " -L --delete " in rsyncs[0] and rsyncs[0].endswith("/images/ deploy@mirror:/images/")
    assert "--chmod=Dg+rwxs,Do-rwx,Fg+rw,Fo-rwx" in rsyncs[0] and "Fo+r" not in rsyncs[0]
    assert "--files-from" not in rsyncs[0] and f"{tmp_path}/data/images" not in rsyncs[0]  # the staged links, not the folder
    assert rsyncs[1].endswith("/birdbrain.db deploy@mirror:/birdbrain.db")
    assert not any("live.jpg" in c for c in rsyncs)
    # The staging folder is gone afterwards, and the lock file stays for the next run.
    guarded = subprocess.run(["bash", "-c", "command -v flock && command -v timeout"], capture_output=True).returncode == 0
    assert (tmp_path / "data" / "push_public.lock").exists() == guarded  # GNU flock (the Pi); a Mac runs unguarded


def test_push_public_runs_one_at_a_time(tmp_path):
    import fcntl

    if subprocess.run(["bash", "-c", "command -v flock && command -v timeout"], capture_output=True).returncode:
        pytest.skip("flock and timeout (GNU) are needed for the guard")
    bindir, log = fake_tools(tmp_path)
    env = push_env(tmp_path, bindir)
    lock = tmp_path / "data" / "push_public.lock"
    with open(lock, "a") as held:
        fcntl.flock(held, fcntl.LOCK_EX)  # another push is running
        r = subprocess.run(["bash", str(ROOT / "scripts/push_public.sh")], env=env, capture_output=True, text=True)
    assert r.returncode == 0 and "another push is still running" in r.stderr
    assert not log.exists()  # nothing ran


def test_push_public_ssh_gives_up_on_a_dead_connection():
    script = (ROOT / "scripts/push_public.sh").read_text()
    assert "ConnectTimeout=10" in script and "ServerAliveInterval=15" in script and "ServerAliveCountMax=3" in script
    assert "timeout -k 15 300" in script and "flock -n" in script
    assert subprocess.run(["bash", "-n", str(ROOT / "scripts/push_public.sh")]).returncode == 0
