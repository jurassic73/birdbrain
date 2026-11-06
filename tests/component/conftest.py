"""Fixtures shared by the component tests: a TestClient on this test's database, and read-only mode."""
import pytest
from fastapi.testclient import TestClient

from birdbrain.web import app as webapp, auth


@pytest.fixture
def client(conn):
    with TestClient(webapp.app) as c:
        yield c


@pytest.fixture
def locked(monkeypatch):
    """Read-only mode: an admin password is set, so a client without the cookie is a visitor."""
    monkeypatch.setenv("ADMIN_PASSWORD", "s3cret!")
    auth._fails.clear()
