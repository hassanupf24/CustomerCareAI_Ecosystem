"""
Escalation Policy — configurable thresholds for escalation triggers.

Reads thresholds from YAML config and provides a check function
to decide if the current emotional state warrants escalation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
import structlog

logger = structlog.get_logger()

_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "escalation_thresholds.yaml"


def _load_config() -> dict:
    try:
        with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f).get("escalation", {})
    except FileNotFoundError:
        logger.warning("escalation_config_not_found", path=str(_CONFIG_PATH))
        return {}


def check_escalation(
    sentiment_score: float,
    dominant_emotion: str,
    emotion_history: list[str],
) -> tuple[bool, Optional[str]]:
    """
    Check if the current emotional state triggers escalation.

    Returns:
        (should_escalate, reason)
    """
    config = _load_config()
    reasons: list[str] = []

    # Sentiment threshold
    threshold = config.get("sentiment", {}).get("threshold", -0.65)
    if sentiment_score < threshold:
        reasons.append(
            f"Sentiment score ({sentiment_score:.2f}) below threshold ({threshold})"
        )

    # Consecutive negative emotions
    trigger_emotions = set(
        config.get("emotion", {}).get("trigger_emotions", ["anger", "distress"])
    )
    consecutive_required = config.get("emotion", {}).get("consecutive_turns", 2)

    # Count consecutive trigger emotions at the end of history
    check_list = emotion_history + [dominant_emotion]
    consecutive = 0
    for em in reversed(check_list):
        if em in trigger_emotions:
            consecutive += 1
        else:
            break

    if consecutive >= consecutive_required:
        reasons.append(
            f"Trigger emotion '{dominant_emotion}' detected for {consecutive} consecutive turns"
        )

    if reasons:
        return True, " | ".join(reasons)
    return False, None
