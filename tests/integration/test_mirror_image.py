"""Guard: the public mirror's container recipe stays small and keeps secrets out.

The image leaves out OpenCV and ONNX Runtime (the web process never loads them), so the web app must keep
importing without them, and the build context must never carry `.env`, private notes or data.
"""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# What the Dockerfile leaves out of the image; a test below checks the two lists agree.
LEFT_OUT = ("opencv-python-headless", "onnxruntime")


def _lines(name):
    return [ln.strip() for ln in (ROOT / name).read_text().splitlines() if ln.strip() and not ln.startswith("#")]


def test_web_app_imports_without_opencv_or_onnx(tmp_path):
    """Setting a module to None in sys.modules makes `import` raise ImportError, as if it were not installed."""
    code = (
        "import sys\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        "for m in ('cv2', 'onnxruntime'):\n"
        "    sys.modules[m] = None\n"
        "import birdbrain.web.app\n"
        "print('ok')\n"
    )
    env = {**os.environ, "BIRDBRAIN_PUBLIC": "1", "BIRDBRAIN_NO_DOTENV": "1", "BIRDBRAIN_DATA": str(tmp_path)}
    r = subprocess.run([sys.executable, "-I", "-c", code], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.strip() == "ok", (
        "the mirror image has no OpenCV or ONNX Runtime, so birdbrain.web.app must not import them at load time "
        "(import them inside the function that needs them):\n" + r.stderr[-1500:]
    )


def test_dockerfile_leaves_out_the_same_packages():
    text = (ROOT / "Dockerfile").read_text()
    for pkg in LEFT_OUT:
        assert f"--no-install-package {pkg}" in text, f"Dockerfile must leave out {pkg}"


def test_dockerfile_copies_only_the_package_and_runs_unprivileged():
    lines = _lines("Dockerfile")
    copies = [ln for ln in lines if ln.startswith("COPY ")]
    assert not any(ln.split()[1] in (".", "./") for ln in copies), "never COPY the whole repo into the image"
    assert any(ln.startswith("COPY birdbrain ") for ln in copies)
    assert any(ln.startswith("USER ") and ln.split()[1] != "root" for ln in lines), "run as a non-root user"
    assert "BIRDBRAIN_PUBLIC=1" in (ROOT / "Dockerfile").read_text(), "the image is the read-only mirror"


def test_dockerignore_keeps_secrets_and_data_out_of_the_build():
    ignored = set(_lines(".dockerignore"))
    for must in (".env", "CLAUDE.md", "CLAUDE.history.md", "data", "firmware-backup", ".git", "*.db"):
        assert must in ignored, f".dockerignore must list {must}"
