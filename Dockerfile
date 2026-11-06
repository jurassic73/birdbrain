# Birdbrain public read-only mirror (BIRDBRAIN_PUBLIC=1) as a container.
#
# This image is only the mirror: the web site reading a copy of the data that the home Pi pushes in. It has no
# camera, watcher, OpenCV or ONNX Runtime (the web process never loads them; tests/integration/test_mirror_image.py
# proves it). Build for the machine you run on (the mirror host is a Raspberry Pi, arm64):
#
#   docker build -t birdbrain-mirror .
#   docker run -d --name birdbrain-mirror --restart unless-stopped \
#     -p 127.0.0.1:8000:8000 --group-add <gid of the group that owns the pushed files> \
#     -v /path/to/pushed/data:/data -v birdbrain-state:/state \
#     birdbrain-mirror
#
# Put the Cloudflare tunnel or a reverse proxy in front; publish the port on 127.0.0.1 only (Docker bypasses
# ufw for bare -p 8000:8000). README.md > "Public mirror in Docker" has the details.

FROM python:3.11-slim-bookworm AS build
COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
# Locked dependencies only, minus the two heavy packages the mirror never imports.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project \
    --no-install-package opencv-python-headless --no-install-package onnxruntime

FROM python:3.11-slim-bookworm
# A fixed, unprivileged user (uid 10001). The pushed files are mode 660 for their group only, so run the container
# with --group-add (compose: group_add) naming the group that owns them; /state belongs to this user.
RUN useradd --system --uid 10001 --no-create-home --shell /usr/sbin/nologin birdbrain \
    && mkdir -p /data /state && chown birdbrain /state
WORKDIR /app
COPY --from=build /app/.venv /app/.venv
COPY birdbrain ./birdbrain
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    BIRDBRAIN_PUBLIC=1 \
    BIRDBRAIN_BASE=/birdbrain \
    BIRDBRAIN_NO_DOTENV=1 \
    BIRDBRAIN_DATA=/data \
    VIEWER_PASSWORD_FILE=/state/viewer.pw \
    VIEWER_INVITE_FILE=/state/viewer.invite
USER birdbrain
VOLUME ["/state"]
EXPOSE 8000
# Any answer below 500 means the site is up (a login redirect counts); /api/latest-visit is the lightest page.
HEALTHCHECK --interval=60s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request as u, urllib.error as e\ntry: u.urlopen('http://127.0.0.1:8000/api/latest-visit', timeout=4)\nexcept e.HTTPError as x: raise SystemExit(0 if x.code < 500 else 1)\nexcept Exception: raise SystemExit(1)"]
# Binds 0.0.0.0 inside the container so a published port works; publish it on 127.0.0.1 only. Behind a proxy on
# another address, name it with VIEWER_TRUSTED_PROXIES so the real client IP is read from CF-Connecting-IP.
CMD ["uvicorn", "birdbrain.web.app:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
