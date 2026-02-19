"""
Report Generator — Generates periodic performance reports.

Produces summary reports of agent performance, CSAT trends,
and knowledge base health.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import structlog

logger = structlog.get_logger()


def generate_performance_report(
    interaction_logs: list[dict],
    feedback_data: list[dict],
    period_label: str = "current",
) -> dict:
    """
    Generate a summary performance report.

    Returns:
        Dict with KPIs: total_interactions, avg_csat, resolution_rate,
        escalation_rate, avg_response_quality, generated_at timestamp.
    """
    total = len(interaction_logs)
    if total == 0:
        return {
            "period": period_label,
            "total_interactions": 0,
            "avg_csat": None,
            "resolution_rate": None,
            "escalation_rate": None,
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }

    # CSAT
    csat_scores = [
        fb.get("csat_score", 0.0) for fb in feedback_data if fb.get("csat_score") is not None
    ]
    avg_csat = sum(csat_scores) / len(csat_scores) if csat_scores else None

    # Resolution rate
    resolved = sum(
        1 for log in interaction_logs
        if log.get("intent", "unknown") not in ("unknown", "unclear", "unresolved")
    )
    resolution_rate = round(resolved / total, 4) if total > 0 else None

    # Escalation rate
    escalated = sum(1 for log in interaction_logs if log.get("escalation_flag", False))
    escalation_rate = round(escalated / total, 4) if total > 0 else None

    return {
        "period": period_label,
        "total_interactions": total,
        "avg_csat": round(avg_csat, 2) if avg_csat else None,
        "resolution_rate": resolution_rate,
        "escalation_rate": escalation_rate,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
