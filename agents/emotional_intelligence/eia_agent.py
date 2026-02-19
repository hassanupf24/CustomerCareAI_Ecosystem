"""
Emotional Intelligence Agent (EIA)

Responsibilities:
- Sentiment analysis (score: -1.0 to 1.0)
- Emotion classification (dominant emotion label)
- Tone recommendation for OCS response modulation
- Escalation triggering when distress thresholds are exceeded
"""

from __future__ import annotations

from pydantic import BaseModel
import structlog

from agents.base_agent import BaseAgent
from agents.emotional_intelligence.emotion_classifier import classify_emotion
from agents.emotional_intelligence.escalation_policy import check_escalation
from api.schemas import EIAInput, EIAOutput

logger = structlog.get_logger()

# Tone recommendations based on sentiment ranges
_TONE_MAP: list[tuple[float, float, str]] = [
    (-1.0, -0.65, "highly empathetic and apologetic"),
    (-0.65, -0.3, "empathetic and understanding"),
    (-0.3, 0.0, "warm and supportive"),
    (0.0, 0.3, "neutral and professional"),
    (0.3, 0.65, "friendly and positive"),
    (0.65, 1.0, "enthusiastic and celebratory"),
]


def _get_tone_recommendation(sentiment_score: float) -> str:
    """Map a sentiment score to a tone recommendation."""
    for low, high, tone in _TONE_MAP:
        if low <= sentiment_score < high:
            return tone
    return "neutral and professional"


def _compute_sentiment_from_emotions(emotions: dict[str, float]) -> float:
    """
    Convert emotion distribution to a single sentiment score in [-1, 1].
    Positive emotions add, negative emotions subtract.
    """
    positive_emotions = {"joy", "surprise", "love"}
    negative_emotions = {"anger", "disgust", "fear", "sadness", "distress"}

    pos_score = sum(emotions.get(e, 0.0) for e in positive_emotions)
    neg_score = sum(emotions.get(e, 0.0) for e in negative_emotions)
    neutral_score = emotions.get("neutral", 0.0)

    # Weighted combination
    sentiment = pos_score - neg_score
    # Clamp to [-1, 1]
    return max(-1.0, min(1.0, sentiment))


class EIAAgent(BaseAgent):
    """Emotional Intelligence Agent."""

    def __init__(self) -> None:
        super().__init__(agent_name="EIA")

    async def process(self, input_data: BaseModel) -> EIAOutput:
        """
        1. Classify emotions
        2. Compute sentiment score
        3. Determine tone recommendation
        4. Check escalation policy
        """
        data: EIAInput = input_data  # type: ignore[assignment]

        # Step 1 — Emotion classification
        emotions = await classify_emotion(data.conversation_text)

        # Dominant emotion
        dominant_emotion = max(emotions, key=emotions.get) if emotions else "neutral"

        # Step 2 — Sentiment score
        sentiment_score = _compute_sentiment_from_emotions(emotions)

        # Step 3 — Tone recommendation
        tone_recommendation = _get_tone_recommendation(sentiment_score)

        # Step 4 — Escalation check
        should_escalate, reason = check_escalation(
            sentiment_score=sentiment_score,
            dominant_emotion=dominant_emotion,
            emotion_history=data.conversation_history,
        )

        self.logger.info(
            "eia_processed",
            interaction_id=data.interaction_id,
            sentiment_score=round(sentiment_score, 3),
            dominant_emotion=dominant_emotion,
            escalation_flag=should_escalate,
            tone_recommendation=tone_recommendation,
        )

        return EIAOutput(
            sentiment_score=round(sentiment_score, 4),
            dominant_emotion=dominant_emotion,
            escalation_flag=should_escalate,
            tone_recommendation=tone_recommendation,
        )
