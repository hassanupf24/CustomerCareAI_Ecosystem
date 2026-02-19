"""
Trend Analyzer — Aggregates CSAT and issue trends over time.
"""

from __future__ import annotations

from typing import Optional

import structlog

logger = structlog.get_logger()


def analyze_sentiment_trend(
    feedback_history: list[dict],
) -> Optional[str]:
    """
    Analyze trends in feedback sentiment over time.

    Returns one of: 'improving', 'declining', 'stable', or None.
    """
    if not feedback_history or len(feedback_history) < 2:
        return None

    scores = []
    for fb in feedback_history:
        score = fb.get("csat_score") or fb.get("sentiment_score")
        if score is not None:
            scores.append(float(score))

    if len(scores) < 2:
        return None

    # Simple linear trend: compare first half avg to second half avg
    mid = len(scores) // 2
    first_half_avg = sum(scores[:mid]) / mid
    second_half_avg = sum(scores[mid:]) / (len(scores) - mid)

    diff = second_half_avg - first_half_avg

    if diff > 0.1:
        return "improving"
    elif diff < -0.1:
        return "declining"
    return "stable"


def extract_top_issues(
    interaction_logs: list[dict],
    max_issues: int = 5,
) -> list[str]:
    """
    Extract the most common issues/intents from interaction logs.
    """
    intent_counts: dict[str, int] = {}
    for log in interaction_logs:
        intent = log.get("intent", "unknown")
        if intent and intent != "unknown":
            intent_counts[intent] = intent_counts.get(intent, 0) + 1

    sorted_intents = sorted(intent_counts.items(), key=lambda x: x[1], reverse=True)
    return [intent for intent, _ in sorted_intents[:max_issues]]


def detect_knowledge_gaps(
    interaction_logs: list[dict],
) -> list[str]:
    """
    Identify knowledge gaps — intents or queries that frequently have
    low-confidence FAQ matches or 'unknown' classification.
    """
    gaps: list[str] = []

    unresolved_count = 0
    low_confidence_queries: list[str] = []

    for log in interaction_logs:
        intent = log.get("intent", "unknown")
        if intent in ("unknown", "unclear", "unresolved"):
            unresolved_count += 1
            query = log.get("customer_message", "")
            if query:
                low_confidence_queries.append(query[:100])

    if unresolved_count > 0:
        gaps.append(f"{unresolved_count} interactions with unresolved intent")

    if low_confidence_queries:
        unique_queries = list(set(low_confidence_queries))[:3]
        for q in unique_queries:
            gaps.append(f"Low-confidence query: '{q}'")

    return gaps
