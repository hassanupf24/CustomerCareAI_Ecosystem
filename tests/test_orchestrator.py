"""
Tests for the Orchestrator (integration-level).
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from api.schemas import (
    ChannelType,
    CustomerRequest,
    EIAOutput,
    KFOOutput,
    OCSOutput,
    OrchestratorResponse,
    PIROutput,
    SupportedLanguage,
)
from orchestrator.aggregator import aggregate_outputs, _check_escalation


# ── Aggregator ────────────────────────────────────────────────────────────────

class TestAggregator:
    def test_aggregate_all_outputs(self):
        ocs = OCSOutput(
            response_text="Hello! How can I help?",
            intent="greeting",
            suggested_faq_ids=[],
            escalation_flag=False,
            language=SupportedLanguage.EN,
        )
        kfo = KFOOutput(suggested_faq_articles=[], updated_knowledge=False)
        eia = EIAOutput(
            sentiment_score=0.3,
            dominant_emotion="joy",
            escalation_flag=False,
            tone_recommendation="friendly and positive",
        )
        pir = PIROutput(proactive_alerts=[])

        response = aggregate_outputs(
            interaction_id="test-agg-001",
            customer_id="CUST-001",
            channel=ChannelType.CHAT,
            language=SupportedLanguage.EN,
            customer_message="Hello",
            context={"emotion_trend": [], "unresolved_turns": 0},
            ocs_output=ocs,
            kfo_output=kfo,
            eia_output=eia,
            pir_output=pir,
        )

        assert isinstance(response, OrchestratorResponse)
        assert response.interaction_id == "test-agg-001"
        assert response.response_text == "Hello! How can I help?"
        assert response.escalation_flag is False
        assert response.intent == "greeting"

    def test_aggregate_with_none_agents(self):
        """Pipeline should handle agent failures gracefully."""
        response = aggregate_outputs(
            interaction_id="test-agg-002",
            customer_id="CUST-002",
            channel=ChannelType.EMAIL,
            language=SupportedLanguage.EN,
            customer_message="Help",
            context={"emotion_trend": [], "unresolved_turns": 0},
            ocs_output=None,
            kfo_output=None,
            eia_output=None,
            pir_output=None,
        )

        assert isinstance(response, OrchestratorResponse)
        assert response.response_text == ""
        assert response.intent == "unknown"


# ── Escalation Gate ───────────────────────────────────────────────────────────

class TestEscalationGate:
    def test_sentiment_escalation(self):
        eia = EIAOutput(
            sentiment_score=-0.8,
            dominant_emotion="anger",
            escalation_flag=True,
            tone_recommendation="highly empathetic",
        )
        flag, reason = _check_escalation(
            ocs_output=None,
            eia_output=eia,
            pir_output=None,
            context={"emotion_trend": [], "unresolved_turns": 0},
        )
        assert flag is True
        assert "Sentiment score" in reason

    def test_no_escalation_positive(self):
        eia = EIAOutput(
            sentiment_score=0.5,
            dominant_emotion="joy",
            escalation_flag=False,
            tone_recommendation="friendly",
        )
        ocs = OCSOutput(
            response_text="Hi!",
            intent="greeting",
            escalation_flag=False,
        )
        flag, reason = _check_escalation(
            ocs_output=ocs,
            eia_output=eia,
            pir_output=None,
            context={"emotion_trend": [], "unresolved_turns": 0},
        )
        assert flag is False

    def test_explicit_escalation_request(self):
        ocs = OCSOutput(
            response_text="Connecting you...",
            intent="escalation_request",
            escalation_flag=True,
        )
        flag, reason = _check_escalation(
            ocs_output=ocs,
            eia_output=None,
            pir_output=None,
            context={"emotion_trend": [], "unresolved_turns": 0},
        )
        assert flag is True

    def test_unresolved_turns_escalation(self):
        flag, reason = _check_escalation(
            ocs_output=None,
            eia_output=None,
            pir_output=None,
            context={"emotion_trend": [], "unresolved_turns": 5},
        )
        assert flag is True
        assert "resolution" in reason.lower() or "turns" in reason.lower()
