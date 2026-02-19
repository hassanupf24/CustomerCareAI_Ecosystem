"""
Tests for the Omni-Channel Support Agent (OCS).
"""

import pytest

from agents.omni_channel_support.language_detector import detect_language
from agents.omni_channel_support.intent_classifier import _rule_based_intent, classify_intent
from agents.omni_channel_support.ocs_agent import OCSAgent
from api.schemas import OCSInput, OCSOutput, SupportedLanguage


# ── Language Detector ─────────────────────────────────────────────────────────

class TestLanguageDetector:
    def test_english_text(self):
        assert detect_language("Hello, I need help with my account") == "en"

    def test_arabic_text(self):
        assert detect_language("مرحبا أحتاج مساعدة في حسابي") == "ar"

    def test_empty_text(self):
        assert detect_language("") == "en"

    def test_whitespace_only(self):
        assert detect_language("   ") == "en"


# ── Intent Classifier (Rule-Based) ───────────────────────────────────────────

class TestRuleBasedIntent:
    def test_billing_intent(self):
        intent, score = _rule_based_intent("I have a question about my bill")
        assert intent == "billing_inquiry"
        assert score > 0

    def test_escalation_intent(self):
        intent, score = _rule_based_intent("I want to speak to a human agent")
        assert intent == "escalation_request"

    def test_greeting_intent(self):
        intent, score = _rule_based_intent("Hello there!")
        assert intent == "greeting"

    def test_unknown_intent(self):
        intent, score = _rule_based_intent("xyzzy garble")
        assert intent == "general_inquiry"

    def test_technical_support(self):
        intent, score = _rule_based_intent("My app is not working properly")
        assert intent == "technical_support"


# ── OCS Agent ─────────────────────────────────────────────────────────────────

class TestOCSAgent:
    @pytest.mark.asyncio
    async def test_process_english_greeting(self):
        agent = OCSAgent()
        input_data = OCSInput(
            interaction_id="test-001",
            customer_message="Hello, I need help",
            conversation_context={},
        )
        result = await agent.process(input_data)
        assert isinstance(result, OCSOutput)
        assert result.response_text != ""
        assert result.language == SupportedLanguage.EN

    @pytest.mark.asyncio
    async def test_process_escalation_request(self):
        agent = OCSAgent()
        input_data = OCSInput(
            interaction_id="test-002",
            customer_message="I want to speak to a human agent please",
            conversation_context={},
        )
        result = await agent.process(input_data)
        assert isinstance(result, OCSOutput)
        assert result.escalation_flag is True
        assert result.intent == "escalation_request"

    @pytest.mark.asyncio
    async def test_safe_process_returns_result(self):
        agent = OCSAgent()
        input_data = OCSInput(
            interaction_id="test-003",
            customer_message="What are your payment methods?",
            conversation_context={},
        )
        result = await agent.safe_process(input_data)
        assert result is not None
        assert isinstance(result, OCSOutput)
