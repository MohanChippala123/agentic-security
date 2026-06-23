"""FastAPI surface for Agentic Security."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import auth, db

app = FastAPI(
    title="AgentShield",
    description="The Security LLM platform for AI agents. Built by Mohan.",
    version="1.0.0",
)

auth.seed_demo_account()
db.purge_fake_users()

_WEB_DIR = Path(__file__).resolve().parents[2] / "web"
_MAX_BODY_BYTES = 64 * 1024


@app.middleware("http")
async def limit_body_size(request: Request, call_next):
    cl = request.headers.get("content-length")
    if cl is not None and cl.isdigit() and int(cl) > _MAX_BODY_BYTES:
        return JSONResponse(status_code=413, content={"detail": f"Request body exceeds {_MAX_BODY_BYTES} bytes."})
    return await call_next(request)


class Credentials(BaseModel):
    email: str
    password: str
    name: str = ""


def _set_session(resp: JSONResponse, email: str) -> None:
    resp.set_cookie(auth.COOKIE, auth.issue_token(email), httponly=True, samesite="lax", max_age=7 * 24 * 3600, path="/")


def _current_user(request: Request) -> dict | None:
    return auth.read_token(request.cookies.get(auth.COOKIE))


def _require_user(request: Request) -> dict:
    user = _current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    return user


# ── auth endpoints ────────────────────────────────────────────────────────────

@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "version": app.version}


@app.post("/api/auth/signup")
def signup(creds: Credentials) -> JSONResponse:
    try:
        user = auth.create_user(creds.email, creds.password, creds.name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    resp = JSONResponse({"ok": True, "name": user["name"]})
    _set_session(resp, user["email"])
    return resp


@app.post("/api/auth/login")
def login(creds: Credentials) -> JSONResponse:
    try:
        user = auth.verify_user(creds.email, creds.password)
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc))
    resp = JSONResponse({"ok": True, "name": user["name"]})
    _set_session(resp, user["email"])
    return resp


@app.post("/api/auth/logout")
def logout() -> JSONResponse:
    resp = JSONResponse({"ok": True})
    resp.delete_cookie(auth.COOKIE, path="/")
    return resp


@app.get("/api/auth/me")
def me(request: Request) -> dict:
    user = _current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user


# ── shield / AI firewall endpoints ───────────────────────────────────────────

from ..shield.detector import full_scan as shield_detect, ThreatType
from ..shield.proxy import shield_request, get_events as shield_events, get_stats as shield_stats


class ShieldChatRequest(BaseModel):
    messages: list[dict] = Field(..., description="OpenAI-format messages array")
    model: str = Field("gpt-4o-mini", description="Model to use")
    api_key: str | None = Field(None, description="OpenAI API key")
    system_prompt: str = Field("", description="System prompt to protect from leaking")
    enable_llm_detection: bool = Field(True, description="Use LLM-based threat detection")
    enable_pii_filter: bool = Field(True, description="Redact PII from output")
    enable_sanitization: bool = Field(True, description="Sanitize dangerous tokens from input")


class ShieldScanRequest(BaseModel):
    text: str = Field(..., description="Text to scan for threats")
    use_llm: bool = Field(False, description="Also run LLM-based detection")
    api_key: str | None = Field(None, description="API key for LLM detection")


@app.post("/api/shield/chat")
def shield_chat(req: ShieldChatRequest) -> dict:
    """Protected chat proxy — send messages through the security shield."""
    return shield_request(
        messages=req.messages,
        model=req.model,
        api_key=req.api_key,
        system_prompt=req.system_prompt,
        enable_llm_detection=req.enable_llm_detection,
        enable_pii_filter=req.enable_pii_filter,
        enable_sanitization=req.enable_sanitization,
    )


@app.post("/api/shield/scan")
def shield_scan(req: ShieldScanRequest) -> dict:
    """Scan text for prompt injection threats without sending to a model."""
    import os
    client = None
    model = "gpt-4o-mini"
    if req.use_llm:
        key = req.api_key or os.environ.get("OPENAI_API_KEY", "")
        if key:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=key)
            except ImportError:
                pass
    verdict = shield_detect(req.text, client=client, model=model)
    return {
        "blocked": verdict.blocked,
        "threat_type": verdict.threat_type.value,
        "confidence": verdict.confidence,
        "explanation": verdict.explanation,
        "layer": verdict.layer,
        "latency_ms": round(verdict.latency_ms, 2),
    }


@app.get("/api/shield/events")
def shield_event_log() -> dict:
    """Recent shield events (blocks, sanitizations, passes)."""
    return {"events": shield_events(50)}


@app.get("/api/shield/stats")
def shield_statistics() -> dict:
    """Shield stats: total requests, blocked, sanitized, passed."""
    return shield_stats()


# ── Guardrails: extra AI-security primitives ──
from ..shield.guardrails import (
    risk_score as gr_risk,
    moderate as gr_moderate,
    scan_secrets as gr_secrets,
    redact as gr_redact,
    check_policy as gr_policy,
    Policy,
)


class TextRequest(BaseModel):
    text: str = Field(..., description="Text to analyze")


class PolicyRequest(BaseModel):
    text: str = Field(..., description="Text to check")
    max_length: int = Field(8000)
    denied_keywords: list[str] = Field(default_factory=list)
    block_pii: bool = Field(True)
    block_secrets: bool = Field(True)
    block_on_risk: int = Field(85)


@app.post("/api/shield/risk")
def shield_risk(req: TextRequest) -> dict:
    """Multi-category risk score (0-100) with a recommendation."""
    return gr_risk(req.text)


@app.post("/api/shield/moderate")
def shield_moderate(req: TextRequest) -> dict:
    """Content moderation across toxicity categories."""
    return gr_moderate(req.text)


@app.post("/api/shield/secrets")
def shield_secrets(req: TextRequest) -> dict:
    """Scan text for leaked secrets and PII (masked previews)."""
    return gr_secrets(req.text)


@app.post("/api/shield/redact")
def shield_redact(req: TextRequest) -> dict:
    """Return a redacted copy of the text."""
    return gr_redact(req.text)


@app.post("/api/shield/policy")
def shield_policy(req: PolicyRequest) -> dict:
    """Enforce a configurable allow/deny policy."""
    pol = Policy(
        max_length=req.max_length,
        denied_keywords=req.denied_keywords,
        block_pii=req.block_pii,
        block_secrets=req.block_secrets,
        block_on_risk=req.block_on_risk,
    )
    return gr_policy(req.text, pol)


# ── API Key Guard / LLM Gateway ───────────────────────────────────────────────
from fastapi import Header
from ..shield import gateway as gw


class CreateKeyRequest(BaseModel):
    name: str = Field("unnamed", description="Label for this virtual key")
    budget_usd: float = Field(5.0, description="Spend cap in USD")
    rate_limit_per_min: int = Field(30, description="Max requests per minute")


class RevokeKeyRequest(BaseModel):
    key: str = Field(..., description="Virtual key (or masked prefix) to revoke")


class GatewayChatRequest(BaseModel):
    messages: list[dict] = Field(..., description="OpenAI-format messages")
    model: str = Field(gw.DEFAULT_MODEL, description="Model to use")
    max_tokens: int = Field(512, description="Max output tokens")


class UpstreamKeyRequest(BaseModel):
    api_key: str = Field(..., description="Your real provider key (stored server-side, never returned)")


@app.get("/api/gateway/status")
def gateway_status(request: Request) -> dict:
    """Is a real upstream key configured? Which models are available?"""
    user = _require_user(request)
    return gw.upstream_status(user["email"])


@app.post("/api/gateway/upstream")
def gateway_set_upstream(req: UpstreamKeyRequest, request: Request) -> dict:
    """Connect your real provider key. Held server-side; never exposed to clients."""
    user = _require_user(request)
    return gw.set_upstream_key(user["email"], req.api_key)


@app.delete("/api/gateway/upstream")
def gateway_clear_upstream(request: Request) -> dict:
    """Disconnect the real provider key (revert to local demo upstream)."""
    user = _require_user(request)
    return gw.clear_upstream_key(user["email"])


@app.get("/api/gateway/stats")
def gateway_stats(request: Request) -> dict:
    """Aggregate spend / blocked / saved across all virtual keys."""
    user = _require_user(request)
    return gw.stats(user["email"])


@app.get("/api/gateway/events")
def gateway_events(request: Request, limit: int = 50) -> dict:
    """Recent gateway events — every request flowing through the defense pipeline."""
    user = _require_user(request)
    return {"events": gw.recent_events(user["email"], limit)}


class JudgeRequest(BaseModel):
    text: str = Field(..., description="Text to evaluate with the Security LLM directly")


@app.post("/api/gateway/judge")
def gateway_judge(req: JudgeRequest, request: Request) -> dict:
    """Ask the AgentShield Security LLM to judge a request without going through the gateway."""
    _require_user(request)
    from ..llm.engine import judge_message
    return judge_message(req.text)


# ── AgentShield: Security LLM analyst endpoints ───────────────────────────────
from ..agentshield import (
    analyze_threat,
    verify_tool_call,
    scan_external_content,
    scan_memory_write,
    run_redteam,
    record_action,
    get_agent_profile,
    anomaly_report,
)
from ..agentshield.behavior import list_profiles


class AnalyzeRequest(BaseModel):
    text: str = Field(..., description="Input to run full Security LLM analysis on")
    source: str = Field("user", description="user | external_content | tool_input | memory_write")


@app.post("/api/agentshield/analyze")
def agentshield_analyze(req: AnalyzeRequest) -> dict:
    """Full Security LLM threat report - risk score, attack chain, severity, decision, reasoning."""
    return analyze_threat(req.text, source=req.source)


class ToolVerifyRequest(BaseModel):
    tool: str = Field(..., description="Tool name (e.g. delete_database, send_email)")
    arguments: dict = Field(default_factory=dict, description="Tool arguments")
    user_intent: str = Field("", description="The user request that supposedly triggered this tool call")
    require_human_for_destructive: bool = Field(True)


@app.post("/api/agentshield/verify-tool")
def agentshield_verify_tool(req: ToolVerifyRequest) -> dict:
    """Verify a tool call before it executes. Returns Security LLM verdict."""
    return verify_tool_call(
        req.tool, req.arguments,
        user_intent=req.user_intent,
        require_human_for_destructive=req.require_human_for_destructive,
    )


class ContentScanRequest(BaseModel):
    content: str = Field(..., description="Retrieved content to scan")
    source_url: str = Field("")
    content_type: str = Field("text/html")


@app.post("/api/agentshield/scan-content")
def agentshield_scan_content(req: ContentScanRequest) -> dict:
    """Scan retrieved/RAG content for indirect prompt injection. Returns sanitized version."""
    return scan_external_content(req.content, source_url=req.source_url, content_type=req.content_type)


class MemoryScanRequest(BaseModel):
    content: str = Field(..., description="Content the agent wants to write to memory")
    agent_id: str = Field("default")
    memory_key: str = Field("")


@app.post("/api/agentshield/scan-memory")
def agentshield_scan_memory(req: MemoryScanRequest) -> dict:
    """Scan a memory-write for instruction-poisoning / persistence attacks."""
    return scan_memory_write(req.content, agent_id=req.agent_id, memory_key=req.memory_key)


@app.post("/api/agentshield/redteam")
def agentshield_redteam(limit: int | None = None) -> dict:
    """Run the autonomous red-team suite against the Security LLM. Returns a security score."""
    return run_redteam(limit=limit)


@app.get("/api/agentshield/behavior/{agent_id}")
def agentshield_behavior(agent_id: str) -> dict:
    """Get behavior profile + anomaly report for a specific agent."""
    profile = get_agent_profile(agent_id)
    anomalies = anomaly_report(agent_id)
    return {"profile": profile, "anomalies": anomalies}


@app.get("/api/agentshield/agents")
def agentshield_agents() -> dict:
    """List all observed agents and their profiles."""
    return {"agents": list_profiles()}


class ConsoleRequest(BaseModel):
    question: str = Field(..., description="A question about your API-key activity")


@app.post("/api/agentshield/console")
def agentshield_console(req: ConsoleRequest, request: Request) -> dict:
    """Security Console - answers questions about live API-key activity (real data, not LLM)."""
    user = _require_user(request)
    from ..agentshield.console import answer
    return answer(req.question, user=user["email"])


@app.get("/api/admin/users")
def admin_users(request: Request) -> dict:
    """List all registered users (sign-in history). Requires an active session."""
    if not _current_user(request):
        raise HTTPException(status_code=401, detail="Not authenticated")
    users = db.user_all()
    safe = [
        {
            "email": u["email"],
            "name": u["name"],
            "created_at": u["created_at"],
            "last_login_at": u.get("last_login_at"),
            "login_count": u.get("login_count") or 0,
        }
        for u in users
    ]
    return {"users": safe, "total": len(safe)}


@app.get("/api/agentshield/dashboard")
def agentshield_dashboard(request: Request) -> dict:
    """Top-level dashboard data: aggregate stats for the authenticated user."""
    user = _require_user(request)
    return {
        "gateway": gw.stats(user["email"]),
        "gateway_events_recent": gw.recent_events(user["email"], 20),
        "agents": list_profiles(),
        "shield": shield_stats(),
        "security_llm": {
            "name": "AgentShield Security LLM",
            "version": "1.0.0",
            "built_by": "Mohan",
        },
    }


@app.get("/api/gateway/keys")
def gateway_list_keys(request: Request) -> dict:
    """List virtual keys (real key values are masked)."""
    user = _require_user(request)
    return {"keys": gw.list_keys(user["email"])}


@app.post("/api/gateway/keys")
def gateway_create_key(req: CreateKeyRequest, request: Request) -> dict:
    """Create a virtual key. The full key is returned ONCE, here."""
    user = _require_user(request)
    return gw.create_key(user["email"], req.name, req.budget_usd, req.rate_limit_per_min)


@app.post("/api/gateway/keys/revoke")
def gateway_revoke_key(req: RevokeKeyRequest, request: Request) -> dict:
    """Instantly disable a virtual key."""
    user = _require_user(request)
    return {"revoked": gw.revoke_key(user["email"], req.key)}


@app.post("/api/gateway/chat")
def gateway_chat_endpoint(req: GatewayChatRequest, request: Request, authorization: str = Header(None)) -> dict:
    """Protected chat. Clients authenticate with a virtual key:
       Authorization: Bearer agk-...   (the real provider key never leaves the server)."""
    user = _require_user(request)
    api_key = ""
    if authorization and authorization.lower().startswith("bearer "):
        api_key = authorization[7:].strip()
    return gw.gateway_chat(user["email"], api_key, req.messages, model=req.model, max_tokens=req.max_tokens)


# ── Agentic LLM (your own secured model) ──────────────────────────────────────

from ..llm.engine import (
    chat as llm_chat,
    get_model_info,
    get_audit_log,
    MODEL_NAME,
)


class LLMChatRequest(BaseModel):
    message: str = Field(..., description="User message")
    session_id: str | None = Field(None, description="Conversation session id")
    api_key: str | None = Field(None, description="OpenAI API key (or uses env)")
    temperature: float = Field(0.4, description="Sampling temperature")
    max_tokens: int = Field(1024, description="Max output tokens")


@app.post("/api/llm/chat")
def agentic_llm_chat(req: LLMChatRequest) -> dict:
    """Chat with Agentic LLM — your secured model with built-in defenses."""
    return llm_chat(
        message=req.message,
        session_id=req.session_id,
        api_key=req.api_key,
        temperature=req.temperature,
        max_tokens=req.max_tokens,
    )


@app.get("/api/llm/info")
def agentic_llm_info() -> dict:
    """Model card and live stats for Agentic LLM."""
    return get_model_info()


@app.get("/api/llm/audit")
def agentic_llm_audit() -> dict:
    """Audit log of all Agentic LLM activity."""
    return {"log": get_audit_log(50)}


# ── HTML pages ────────────────────────────────────────────────────────────────

def _page(name: str) -> str:
    return (_WEB_DIR / name).read_text(encoding="utf-8")


if _WEB_DIR.exists():
    @app.get("/", response_class=HTMLResponse)
    def landing() -> str:
        return _page("landing.html")

    @app.get("/login", response_class=HTMLResponse)
    def login_page() -> str:
        return _page("login.html")

    @app.get("/app", response_class=HTMLResponse)
    def dashboard(request: Request):
        if not _current_user(request):
            return RedirectResponse("/login", status_code=302)
        return HTMLResponse(_page("index.html"))

    app.mount("/static", StaticFiles(directory=str(_WEB_DIR)), name="static")
