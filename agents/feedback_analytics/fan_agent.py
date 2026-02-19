"""
Post-Interaction Feedback & Analytics Agent (FAN)

Responsibilities:
- Collect post-interaction CSAT/feedback
- Perform trend analysis on sentiment and issues
- Identify knowledge gaps
- Feed insights back into KFO and OCS
"""

from __future__ import annotations

from pydantic import BaseModel
import structlog

from agents.base_agent import BaseAgent
from agents.feedback_analytics.trend_analyzer import (
    analyze_sentiment_trend,
    detect_knowledge_gaps,
    extract_top_issues,
)
from api.schemas import FANInput, FANOutput, FeedbackAnalysis

logger = structlog.get_logger()


class FANAgent(BaseAgent):
    """Post-Interaction Feedback & Analytics Agent."""

    def __init__(self) -> None:
        super().__init__(agent_name="FAN")

    async def process(self, input_data: BaseModel) -> FANOutput:
        """
        1. Process customer feedback
        2. Analyze trends
        3. Detect knowledge gaps
        4. Return analysis + knowledge update recommendations
        """
        data: FANInput = input_data  # type: ignore[assignment]

        # Extract feedback info
        feedback = data.customer_feedback or {}
        interaction_log = data.interaction_log or {}

        # Compute CSAT score (if provided)
        csat_score = feedback.get("csat_score") or feedback.get("rating")
        if csat_score is not None:
            csat_score = float(csat_score)
            # Normalize to 0–5 if needed
            if csat_score > 5:
                csat_score = min(csat_score / 20.0, 5.0)

        # Build historical data from interaction log
        history = interaction_log.get("history", [])
        feedback_history = interaction_log.get("feedback_history", [])

        # Analyze sentiment trend
        sentiment_trend = analyze_sentiment_trend(feedback_history)

        # Extract top issues
        past_interactions = interaction_log.get("past_interactions", [])
        top_issues = extract_top_issues(past_interactions)

        # Detect knowledge gaps
        knowledge_gaps = detect_knowledge_gaps(past_interactions)

        self.logger.info(
            "fan_processed",
            interaction_id=data.interaction_id,
            csat_score=csat_score,
            sentiment_trend=sentiment_trend,
            top_issues_count=len(top_issues),
            knowledge_gaps_count=len(knowledge_gaps),
        )

        return FANOutput(
            feedback_analysis=FeedbackAnalysis(
                csat_score=csat_score,
                sentiment_trend=sentiment_trend,
                top_issues=top_issues,
                knowledge_gap_flags=knowledge_gaps,
            ),
            knowledge_base_update=[],
        )
