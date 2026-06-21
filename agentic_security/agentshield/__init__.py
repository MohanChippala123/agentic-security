"""AgentShield - the Security LLM platform for AI agents."""

from .analyzer import analyze_threat, AnalystVerdict
from .tool_guard import verify_tool_call
from .content_guard import scan_external_content, scan_memory_write
from .redteam import run_redteam, RED_TEAM_PROMPTS
from .behavior import record_action, get_agent_profile, anomaly_report

__all__ = [
    "analyze_threat",
    "AnalystVerdict",
    "verify_tool_call",
    "scan_external_content",
    "scan_memory_write",
    "run_redteam",
    "RED_TEAM_PROMPTS",
    "record_action",
    "get_agent_profile",
    "anomaly_report",
]
