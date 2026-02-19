"""
Aggregator — merges all agent outputs into the unified OrchestratorResponse.

Handles:
- Combining OCS, KFO, EIA, PIR, FAN outputs
- Escalation gate logic
- Null-safe merging when an agent fails (degraded output)
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

import structlog
import yaml
from pathlib import Path

from api.schemas import (
    AgentLogs,
    ChannelType,
    EIAOutput,
    FANOutput,
    FeedbackAnalysis,
    KFOOutput,
    OCSOutput,
    OrchestratorResponse,
    PIROutput,
    SeverityLevel,
    SupportedLanguage,
)

logger = structlog.get_logger()

# Load escalation thresholds
_thresholds_path = Path(__file__).resolve().parent.parent / "config" / "escalation_thresholds.yaml"
try:
    with open(_thresholds_path, "r", encoding="utf-8") as f:
        ESCALATION_CONFIG = yaml.safe_load(f).get("escalation", {})
except FileNotFoundError:
    ESCALATION_CONFIG = {}


def _check_escalation(
    ocs_output: Optional[OCSOutput],
    eia_output: Optional[EIAOutput],
    pir_output: Optional[PIROutput],
    context: dict,
) -> tuple[bool, Optional[str]]:
    """
    Evaluate escalation policy (§8) and return (flag, reason).
    """
    reasons: list[str] = []

    # OCS: explicit escalation request
    if ocs_output and ocs_output.escalation_flag:
        reasons.append("Customer explicitly requested human agent.")
    if ocs_output and ocs_output.intent == "escalation_request":
        reasons.append("Intent classified as escalation_request.")

    # EIA: sentiment threshold
    sentiment_threshold = ESCALATION_CONFIG.get("sentiment", {}).get("threshold", -0.65)
    if eia_output and eia_output.sentiment_score < sentiment_threshold:
        reasons.append(
            f"Sentiment score ({eia_output.sentiment_score:.2f}) below threshold ({sentiment_threshold})."
        )

    # EIA: consecutive negative emotions
    trigger_emotions = set(
        ESCALATION_CONFIG.get("emotion", {}).get("trigger_emotions", ["anger", "distress"])
    )
    consecutive_required = ESCALATION_CONFIG.get("emotion", {}).get("consecutive_turns", 2)
    emotion_trend = context.get("emotion_trend", [])
    if eia_output:
        emotion_trend_check = emotion_trend + [eia_output.dominant_emotion]
        consecutive_count = 0
        for emotion in reversed(emotion_trend_check):
            if emotion in trigger_emotions:
                consecutive_count += 1
            else:
                break
        if consecutive_count >= consecutive_required:
            reasons.append(
                f"Dominant emotion ({eia_output.dominant_emotion}) persisted for "
                f"{consecutive_count} consecutive turns."
            )

    # EIA escalation flag
    if eia_output and eia_output.escalation_flag:
        reasons.append("Emotional Intelligence Agent triggered escalation.")

    # PIR: critical severity
    if pir_output:
        for alert in pir_output.proactive_alerts:
            if alert.severity == SeverityLevel.CRITICAL:
                reasons.append(f"Critical proactive alert: {alert.alert_type}.")
                break

    # Unresolved turns
    max_unresolved = ESCALATION_CONFIG.get("unresolved", {}).get("max_unresolved_turns", 3)
    unresolved = context.get("unresolved_turns", 0)
    if unresolved >= max_unresolved:
        reasons.append(
            f"No resolution after {unresolved} consecutive turns (threshold: {max_unresolved})."
        )

    if reasons:
        return True, " | ".join(reasons)
    return False, None


def aggregate_outputs(
    interaction_id: str,
    customer_id: str,
    channel: ChannelType,
    language: SupportedLanguage,
    customer_message: str,
    context: dict,
    ocs_output: Optional[OCSOutput],
    kfo_output: Optional[KFOOutput],
    eia_output: Optional[EIAOutput],
    pir_output: Optional[PIROutput],
    fan_output: Optional[FANOutput] = None,
) -> OrchestratorResponse:
    """
    Merge all agent outputs into the unified response schema.
    Handles None outputs gracefully (degraded mode).
    """

    # Escalation gate
    escalation_flag, escalation_reason = _check_escalation(
        ocs_output, eia_output, pir_output, context
    )

    # Build response text — start from OCS, apply EIA tone recommendation
    response_text = ""
    if ocs_output:
        response_text = ocs_output.response_text

    # Tone adjustment hint (logged, not mutating text for now)
    tone_rec = eia_output.tone_recommendation if eia_output else "neutral"

    response = OrchestratorResponse(
        interaction_id=interaction_id,
        timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        customer_id=customer_id,
        channel=channel,
        language=language,
        response_text=response_text,
        intent=ocs_output.intent if ocs_output else "unknown",
        sentiment_score=eia_output.sentiment_score if eia_output else 0.0,
        dominant_emotion=eia_output.dominant_emotion if eia_output else "neutral",
        escalation_flag=escalation_flag,
        escalation_reason=escalation_reason,
        suggested_faq_articles=kfo_output.suggested_faq_articles if kfo_output else [],
        proactive_alerts=pir_output.proactive_alerts if pir_output else [],
        feedback_analysis=fan_output.feedback_analysis if fan_output else FeedbackAnalysis(),
        agent_logs=AgentLogs(
            ocs=ocs_output.model_dump() if ocs_output else {"status": "skipped"},
            kfo=kfo_output.model_dump() if kfo_output else {"status": "skipped"},
            eia=eia_output.model_dump() if eia_output else {"status": "skipped"},
            pir=pir_output.model_dump() if pir_output else {"status": "skipped"},
            fan=fan_output.model_dump() if fan_output else {"status": "pending_async"},
        ),
    )

    logger.info(
        "response_aggregated",
        interaction_id=interaction_id,
        escalation_flag=escalation_flag,
        tone_recommendation=tone_rec,
    )

    return response
