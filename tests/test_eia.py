"""
Tests for the Emotional Intelligence Agent (EIA).
"""

import pytest

from agents.emotional_intelligence.emotion_classifier import _fallback_classify
from agents.emotional_intelligence.escalation_policy import check_escalation
from agents.emotional_intelligence.eia_agent import (
    EIAAgent,
    _compute_sentiment_from_emotions,
    _get_tone_recommendation,
)
from api.schemas import EIAInput, EIAOutput


# ── Sentiment Computation ─────────────────────────────────────────────────────

class TestSentimentComputation:
    def test_positive_emotions(self):
        emotions = {"joy": 0.8, "neutral": 0.1, "sadness": 0.1}
        score = _compute_sentiment_from_emotions(emotions)
        assert score > 0

    def test_negative_emotions(self):
        emotions = {"anger": 0.6, "disgust": 0.3, "neutral": 0.1}
        score = _compute_sentiment_from_emotions(emotions)
        assert score < 0

    def test_neutral_emotions(self):
        emotions = {"neutral": 1.0}
        score = _compute_sentiment_from_emotions(emotions)
        assert -0.2 <= score <= 0.2

    def test_clamped_range(self):
        emotions = {"joy": 0.9, "surprise": 0.8, "love": 0.7}
        score = _compute_sentiment_from_emotions(emotions)
        assert -1.0 <= score <= 1.0


# ── Tone Recommendation ──────────────────────────────────────────────────────

class TestToneRecommendation:
    def test_very_negative(self):
        tone = _get_tone_recommendation(-0.8)
        assert "empathetic" in tone.lower() or "apologetic" in tone.lower()

    def test_neutral(self):
        tone = _get_tone_recommendation(0.1)
        assert "neutral" in tone.lower() or "professional" in tone.lower()

    def test_positive(self):
        tone = _get_tone_recommendation(0.5)
        assert "friendly" in tone.lower() or "positive" in tone.lower()


# ── Escalation Policy ────────────────────────────────────────────────────────

class TestEscalationPolicy:
    def test_low_sentiment_triggers_escalation(self):
        flag, reason = check_escalation(
            sentiment_score=-0.8,
            dominant_emotion="anger",
            emotion_history=[],
        )
        assert flag is True
        assert reason is not None

    def test_consecutive_anger_triggers_escalation(self):
        flag, reason = check_escalation(
            sentiment_score=-0.3,
            dominant_emotion="anger",
            emotion_history=["anger"],
        )
        assert flag is True

    def test_normal_state_no_escalation(self):
        flag, reason = check_escalation(
            sentiment_score=0.2,
            dominant_emotion="neutral",
            emotion_history=["neutral", "joy"],
        )
        assert flag is False
        assert reason is None


# ── Fallback Classifier ──────────────────────────────────────────────────────

class TestFallbackClassifier:
    def test_positive_text(self):
        result = _fallback_classify("I am very happy and delighted!")
        assert "joy" in result or "neutral" in result

    def test_negative_text(self):
        result = _fallback_classify("This is terrible and I am very angry.")
        assert "anger" in result or "sadness" in result

    def test_neutral_text(self):
        result = _fallback_classify("The weather is mild today.")
        assert "neutral" in result


# ── EIA Agent ─────────────────────────────────────────────────────────────────

class TestEIAAgent:
    @pytest.mark.asyncio
    async def test_process_returns_output(self):
        agent = EIAAgent()
        input_data = EIAInput(
            interaction_id="test-eia-001",
            conversation_text="I am very frustrated with your service!",
            conversation_history=[],
        )
        result = await agent.process(input_data)
        assert isinstance(result, EIAOutput)
        assert -1.0 <= result.sentiment_score <= 1.0
        assert result.dominant_emotion != ""
        assert result.tone_recommendation != ""

    @pytest.mark.asyncio
    async def test_safe_process_no_crash(self):
        agent = EIAAgent()
        input_data = EIAInput(
            interaction_id="test-eia-002",
            conversation_text="",
            conversation_history=[],
        )
        result = await agent.safe_process(input_data)
        assert result is not None
