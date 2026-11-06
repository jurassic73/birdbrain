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
# Every stand-in that push_public.sh calls starts with this line: its name and arguments go to the test's FAKE_LOG.
LOGGER = 'printf "%s\\n" "$(basename "$0") $*" >> "$FAKE_LOG"\n'


def test_parse_env_handles_semicolons_quotes_export():
    env = parse_env(FAKE)
    assert env["GROUND_ZONE"] == "0.1,0.2;0.3,0.4;0.5,0.6"
    assert env["CAM_HOST"] == "10.0.0.9"
    assert env["RTSP_USER"] == "bob"
    assert env["RTSP_PASS"] == PASS


# macOS checks every new executable the first time it runs (0.2 to 0.4 s each, and push_public.sh runs six), so the
# stand-ins below are written once per module. What each test must see on its own, the log of calls and the mirror's
# listing, comes to them through the environment: FAKE_LOG, FAKE_URL_LOG, FAKE_LISTING and FAKE_PYTHON_EXIT.
FAKE_TOOLS = {
    # snap.sh's ffmpeg: notes the stream URL it was given in FAKE_URL_LOG, then fails with the URL in its message.
    "ffmpeg": '#!/bin/sh\nwhile [ $# -gt 0 ]; do [ "$1" = -i ] && echo "$2" > "$FAKE_URL_LOG"; shift; done\n'
              'echo "fail rtsp://u:pw@h/x" >&2; exit 1\n',
    # A listing (--list-only) prints the file named by FAKE_LISTING, standing in for the mirror's folders.
    "rsync": '#!/bin/sh\n[ "$1" = --version ] && { echo "rsync  version 3.2.7  protocol version 31"; exit 0; }\n'
             + LOGGER + 'case " $* " in *" --list-only "*) [ -z "${FAKE_LISTING:-}" ] || cat "$FAKE_LISTING";; esac\n'
             + "exit 0\n",
    "logger": "#!/bin/sh\n" + LOGGER + "exit 0\n",  # the system log: never the Mac's or the Pi's real one
    "curl": "#!/bin/sh\nexit 22\n",  # no local site: the last live frame is used instead
    "timedatectl": '#!/bin/sh\necho "Europe/Paris"\n',
    # The venv python: `mirror prepare` stages a database and one photo link; `privacy check` finds a person.
    # FAKE_PYTHON_EXIT set: every call fails with that code (the mirror copy failed).
    "fakepython": '#!/bin/sh\n' + LOGGER + '[ -z "${FAKE_PYTHON_EXIT:-}" ] || exit "$FAKE_PYTHON_EXIT"\n'
                  'if [ "$2" = birdbrain.mirror ]; then mkdir -p "$5/images/2026/11/03" && touch "$5/birdbrain.db" '
                  '&& echo 2026/11/03/a.jpg > "$5/files.txt" && ln -s /nonexistent "$5/images/2026/11/03/a.jpg"; fi\n'
                  'if [ "$2" = birdbrain.privacy ]; then exit 1; fi\nexit 0\n',
}


@pytest.fixture(scope="module")
def bindir(tmp_path_factory) -> Path:
    """The folder of stand-in tools, made once for the module."""
    d = tmp_path_factory.mktemp("fake-tools")
    for name, body in FAKE_TOOLS.items():
        (d / name).write_text(body)
        (d / name).chmod(0o755)
    return d


def test_snap_sh_loads_env_and_redacts(tmp_path, bindir):
    envf = tmp_path / "env"
    envf.write_text(FAKE)
    log = tmp_path / "url.txt"
    env = dict(os.environ, SNAP_ENV=str(envf), PATH=f"{bindir}:{os.environ['PATH']}", FAKE_URL_LOG=str(log))
    r = subprocess.run(
        ["bash", str(ROOT / "scripts/snap.sh"), "sub", str(tmp_path / "o.jpg")],
        env=env, capture_output=True, text=True,
    )
    assert "command not found" not in r.stderr
    assert r.returncode == 1
    assert "pw@" not in r.stderr and "rtsp://***@" in r.stderr
    assert log.read_text().strip() == "rtsp://bob:p%40ss%21%24%3B%23x@10.0.0.9:8554/ch1"


GOOD_LISTING = """\
drwxrws---          4,096 2026/11/03 09:00:00 .
drwxrws---          4,096 2026/11/03 09:00:00 2026
drwxrws---          4,096 2026/11/03 09:00:00 2026/11
drwxrws---          4,096 2026/11/03 09:00:00 2026/11/03
"""


def push_env(tmp_path, bindir, listing: str = GOOD_LISTING, **more: str) -> dict:
    """push_public.sh's environment for one test: the shared stand-ins first on PATH, this test's own data folder,
    listing and call log (tmp_path/calls.txt, read by `calls`)."""
    data = tmp_path / "data"
    data.mkdir(exist_ok=True)
    (data / "live.jpg").write_bytes(b"jpeg")
    (tmp_path / "listing.txt").write_text(listing)
    return dict(os.environ, PATH=f"{bindir}:{os.environ['PATH']}", PUBLIC_DEST="deploy@mirror:",
                PUBLIC_KEY="/nonexistent/key", BIRDBRAIN_DATA=str(data), PUBLIC_PYTHON=str(bindir / "fakepython"),
                FAKE_LISTING=str(tmp_path / "listing.txt"), FAKE_LOG=str(tmp_path / "calls.txt"), **more)


def calls(tmp_path) -> list[str]:
    """What the stand-ins logged for this test, one call a line (an empty list when none ran)."""
    log = tmp_path / "calls.txt"
    return log.read_text().splitlines() if log.exists() else []


def test_push_public_sends_only_the_staged_files_with_timeouts(tmp_path, bindir):
    env = push_env(tmp_path, bindir)
    r = subprocess.run(["bash", str(ROOT / "scripts/push_public.sh")], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    logged = calls(tmp_path)
    prepare = next(c for c in logged if "birdbrain.mirror prepare" in c)
    assert prepare.endswith("Europe/Paris") and f"{tmp_path}/data/birdbrain.db" in prepare
    rsyncs = [c for c in logged if c.startswith("rsync ")]
    assert len(rsyncs) == 3  # the photos, the database and the self-check listing; the live frame showed a person and stayed home
    assert " -L --delete " in rsyncs[0] and rsyncs[0].endswith("/images/ deploy@mirror:/images/")
    assert "--files-from" not in rsyncs[0] and f"{tmp_path}/data/images" not in rsyncs[0]  # the staged links, not the folder
    assert rsyncs[1].endswith("/birdbrain.db deploy@mirror:/birdbrain.db")
    assert not any("live.jpg" in c for c in rsyncs)
    # The self-check is a folders-only listing of the mirror's photo tree; a sound tree says nothing.
    assert rsyncs[2].startswith("rsync -r --list-only --include=*/ --exclude=* ") and rsyncs[2].endswith(" deploy@mirror:/images/")
    assert "-L" not in rsyncs[2].split()  # -L would go to the mirror's rsync as an option, and rrsync refuses it
    assert "push_public:" not in r.stderr and not any(c.startswith("logger ") for c in logged)


def test_push_public_modes_let_the_service_group_read_and_nobody_else(tmp_path, bindir):
    """The mirror's service reads the pushed files through its group: absolute modes (folders setgid 2770,
    files 660, nothing for others), and the group itself is never sent, so each new file and folder takes the
    group of the setgid folder it lands in (the service's, set up once by hand) instead of the Pi's or the
    push user's own."""
    r = subprocess.run(["bash", str(ROOT / "scripts/push_public.sh")], env=push_env(tmp_path, bindir), capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    pushes = [c.split() for c in calls(tmp_path) if c.startswith("rsync ") and "--list-only" not in c]
    assert len(pushes) == 2
    for args in pushes:
        assert "--chmod=D2770,F660" in args
        assert "-rtpz" in args  # -p so the modes apply, and no -a: that would add -o and -g
        assert not any(a in ("-a", "-g", "-o", "--group", "--owner", "--archive") or a.startswith("--chown") for a in args)
        assert not any("+r" in a or "o+" in a for a in args if a.startswith("--chmod"))  # nothing relative, nothing for others


def test_push_public_warns_about_folders_the_service_cannot_read(tmp_path, bindir):
    """The one thing the pushes can't fix from the Pi: a folder that lost its setgid bit (or its group bits), under
    which every new folder takes the push user's group and the photos come up broken. The self-check names it on
    stderr and in the system log without failing the push."""
    listing = GOOD_LISTING.replace("drwxrws---          4,096 2026/11/03 09:00:00 .", "drwxrwx---          4,096 2026/11/03 09:00:00 .")
    listing = listing.replace("drwxrws---          4,096 2026/11/03 09:00:00 2026/11\n", "drwxr-s---          4,096 2026/11/03 09:00:00 2026/11\n")
    listing += "-rw-rw----        812,344 2026/11/03 09:00:00 2026/11/03/a.jpg\n"  # files aren't judged here
    r = subprocess.run(["bash", str(ROOT / "scripts/push_public.sh")], env=push_env(tmp_path, bindir, listing), capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert r.stderr.count("push_public:") == 1
    assert ". is drwxrwx---;2026/11 is drwxr-s---" in r.stderr and "2026/11/03" not in r.stderr and "a.jpg" not in r.stderr
    logged = [c for c in calls(tmp_path) if c.startswith("logger ")]
    assert len(logged) == 1 and "-t push_public" in logged[0] and ". is drwxrwx---" in logged[0]


def test_push_public_failure_reaches_the_system_log(tmp_path, bindir):
    env = push_env(tmp_path, bindir, FAKE_PYTHON_EXIT="2")  # the mirror copy failed
    r = subprocess.run(["bash", str(ROOT / "scripts/push_public.sh")], env=env, capture_output=True, text=True)
    assert r.returncode != 0
    assert "failed (exit 2)" in r.stderr and "birdbrain.mirror" in r.stderr
    assert not any(c.startswith("rsync ") for c in calls(tmp_path))  # fails closed: nothing pushed
    assert any(c.startswith("logger -t push_public") and "failed (exit 2)" in c for c in calls(tmp_path))
    # The staging folder is gone afterwards, and the lock file stays for the next run.
    guarded = subprocess.run(["bash", "-c", "command -v flock && command -v timeout"], capture_output=True).returncode == 0
    assert (tmp_path / "data" / "push_public.lock").exists() == guarded  # GNU flock (the Pi); a Mac runs unguarded


def test_push_public_runs_one_at_a_time(tmp_path, bindir):
    import fcntl

    if subprocess.run(["bash", "-c", "command -v flock && command -v timeout"], capture_output=True).returncode:
        pytest.skip("flock and timeout (GNU) are needed for the guard")
    env = push_env(tmp_path, bindir)
    lock = tmp_path / "data" / "push_public.lock"
    with open(lock, "a") as held:
        fcntl.flock(held, fcntl.LOCK_EX)  # another push is running
        r = subprocess.run(["bash", str(ROOT / "scripts/push_public.sh")], env=env, capture_output=True, text=True)
    assert r.returncode == 0 and "another push is still running" in r.stderr
    assert calls(tmp_path) == [] and not (tmp_path / "calls.txt").exists()  # nothing ran


def test_push_public_ssh_gives_up_on_a_dead_connection():
    script = (ROOT / "scripts/push_public.sh").read_text()
    assert "ConnectTimeout=10" in script and "ServerAliveInterval=15" in script and "ServerAliveCountMax=3" in script
    assert "timeout -k 15 300" in script and "flock -n" in script
    assert subprocess.run(["bash", "-n", str(ROOT / "scripts/push_public.sh")]).returncode == 0
