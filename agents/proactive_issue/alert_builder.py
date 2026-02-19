"""
Alert Builder — Structures detected anomalies into alerts with severity scoring.
"""

from __future__ import annotations

from datetime import datetime, timezone

import structlog

from api.schemas import ProactiveAlert, SeverityLevel

logger = structlog.get_logger()

# Map anomaly score ranges to severity levels
_SEVERITY_MAP: list[tuple[float, float, SeverityLevel]] = [
    (-float("inf"), -2.0, SeverityLevel.CRITICAL),
    (-2.0, -1.0, SeverityLevel.HIGH),
    (-1.0, -0.5, SeverityLevel.MEDIUM),
    (-0.5, float("inf"), SeverityLevel.LOW),
]

# Map data patterns to alert types and recommended actions
_ALERT_PATTERNS: dict[str, dict[str, str]] = {
    "error_count": {
        "alert_type": "high_error_rate",
        "action": "Investigate recent errors in customer's account. Consider proactive outreach.",
    },
    "login_failures": {
        "alert_type": "suspicious_login_activity",
        "action": "Review login attempts for potential unauthorized access. Consider account security measures.",
    },
    "latency_ms": {
        "alert_type": "performance_degradation",
        "action": "Customer may be experiencing slow service. Check backend performance metrics.",
    },
    "api_calls": {
        "alert_type": "unusual_usage_pattern",
        "action": "Usage pattern deviates from normal. Review for potential issues or plan upgrade needs.",
    },
}


def _determine_severity(anomaly_score: float) -> SeverityLevel:
    """Map anomaly score to severity level."""
    for low, high, severity in _SEVERITY_MAP:
        if low <= anomaly_score < high:
            return severity
    return SeverityLevel.LOW


def build_alerts(anomalies: list[dict], account_id: str) -> list[ProactiveAlert]:
    """
    Convert raw anomaly detections into structured ProactiveAlert objects.
    """
    alerts: list[ProactiveAlert] = []

    for anomaly in anomalies:
        score = anomaly.get("anomaly_score", 0.0)
        data = anomaly.get("data", {})
        severity = _determine_severity(score)

        # Determine alert type from the data fields
        alert_type = "general_anomaly"
        recommended_action = f"Anomaly detected in account {account_id}. Review usage patterns."

        for field, pattern in _ALERT_PATTERNS.items():
            if field in data:
                alert_type = pattern["alert_type"]
                recommended_action = pattern["action"]
                break

        alerts.append(
            ProactiveAlert(
                alert_type=alert_type,
                severity=severity,
                recommended_action=recommended_action,
                timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            )
        )

    return alerts
