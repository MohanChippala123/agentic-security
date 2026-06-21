"""Auth + page-gating tests."""

from __future__ import annotations

import uuid

from fastapi.testclient import TestClient

from agentic_security.api.server import app

client = TestClient(app)


def test_landing_is_public():
    assert client.get("/").status_code == 200


def test_app_redirects_when_anonymous():
    r = client.get("/app", follow_redirects=False)
    assert r.status_code == 302
    assert r.headers["location"] == "/login"


def test_me_requires_auth():
    fresh = TestClient(app)
    assert fresh.get("/api/auth/me").status_code == 401


def test_demo_login_then_access_dashboard():
    c = TestClient(app)
    r = c.post("/api/auth/login",
               json={"email": "demo@agentic.security", "password": "demo1234"})
    assert r.status_code == 200
    assert "agsec_session" in r.cookies
    # cookie now on the client → dashboard + me work
    assert c.get("/app").status_code == 200
    assert c.get("/api/auth/me").json()["email"] == "demo@agentic.security"


def test_login_wrong_password_401():
    r = client.post("/api/auth/login",
                    json={"email": "demo@agentic.security", "password": "nope"})
    assert r.status_code == 401


def test_signup_and_login_roundtrip():
    email = f"user-{uuid.uuid4().hex[:8]}@example.com"
    c = TestClient(app)
    r = c.post("/api/auth/signup",
               json={"name": "Test User", "email": email, "password": "longenough"})
    assert r.status_code == 200 and r.json()["name"] == "Test User"
    # duplicate rejected
    assert c.post("/api/auth/signup",
                  json={"name": "x", "email": email, "password": "longenough"}).status_code == 400


def test_signup_short_password_rejected():
    r = client.post("/api/auth/signup",
                    json={"name": "x", "email": "x@y.com", "password": "short"})
    assert r.status_code == 400


def test_logout_clears_session():
    c = TestClient(app)
    c.post("/api/auth/login",
           json={"email": "demo@agentic.security", "password": "demo1234"})
    c.post("/api/auth/logout")
    assert c.get("/app", follow_redirects=False).status_code == 302
