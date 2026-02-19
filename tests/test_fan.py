"""
Tests for the Post-Interaction Feedback & Analytics Agent (FAN).
"""

import pytest

from agents.feedback_analytics.trend_analyzer import (
    analyze_sentiment_trend,
    extract_top_issues,
    detect_knowledge_gaps,
)
from agents.feedback_analytics.report_generator import generate_performance_report
from agents.feedback_analytics.fan_agent import FANAgent
from api.schemas import FANInput, FANOutput


# ── Trend Analyzer ────────────────────────────────────────────────────────────

class TestTrendAnalyzer:
    def test_improving_trend(self):
        history = [
            {"csat_score": 2.0},
            {"csat_score": 2.5},
            {"csat_score": 3.0},
            {"csat_score": 4.0},
            {"csat_score": 4.5},
            {"csat_score": 5.0},
        ]
        result = analyze_sentiment_trend(history)
        assert result == "improving"

    def test_declining_trend(self):
        history = [
            {"csat_score": 5.0},
            {"csat_score": 4.5},
            {"csat_score": 3.5},
            {"csat_score": 2.0},
            {"csat_score": 1.5},
            {"csat_score": 1.0},
        ]
        result = analyze_sentiment_trend(history)
        assert result == "declining"

    def test_stable_trend(self):
        history = [
            {"csat_score": 3.5},
            {"csat_score": 3.4},
            {"csat_score": 3.5},
            {"csat_score": 3.6},
        ]
        result = analyze_sentiment_trend(history)
        assert result == "stable"

    def test_empty_history(self):
        assert analyze_sentiment_trend([]) is None

    def test_single_entry(self):
        assert analyze_sentiment_trend([{"csat_score": 4.0}]) is None


class TestTopIssues:
    def test_extract_top_issues(self):
        logs = [
            {"intent": "billing_inquiry"},
            {"intent": "billing_inquiry"},
            {"intent": "technical_support"},
            {"intent": "billing_inquiry"},
            {"intent": "refund_request"},
        ]
        issues = extract_top_issues(logs, max_issues=2)
        assert issues[0] == "billing_inquiry"
        assert len(issues) <= 2

    def test_empty_logs(self):
        assert extract_top_issues([]) == []


class TestKnowledgeGaps:
    def test_detects_unresolved(self):
        logs = [
            {"intent": "unknown", "customer_message": "What is your xyz feature?"},
            {"intent": "billing_inquiry"},
            {"intent": "unknown", "customer_message": "Tell me about abc"},
        ]
        gaps = detect_knowledge_gaps(logs)
        assert len(gaps) > 0
        assert any("unresolved" in g.lower() for g in gaps)

    def test_no_gaps(self):
        logs = [
            {"intent": "billing_inquiry"},
            {"intent": "technical_support"},
        ]
        gaps = detect_knowledge_gaps(logs)
        assert gaps == []


# ── Report Generator ──────────────────────────────────────────────────────────

class TestReportGenerator:
    def test_generate_report(self):
        logs = [
            {"intent": "billing_inquiry", "escalation_flag": False},
            {"intent": "unknown", "escalation_flag": True},
            {"intent": "technical_support", "escalation_flag": False},
        ]
        feedback = [
            {"csat_score": 4.0},
            {"csat_score": 3.5},
        ]
        report = generate_performance_report(logs, feedback)
        assert report["total_interactions"] == 3
        assert report["avg_csat"] is not None
        assert report["resolution_rate"] is not None
        assert report["escalation_rate"] is not None

    def test_empty_report(self):
        report = generate_performance_report([], [])
        assert report["total_interactions"] == 0


# ── FAN Agent ─────────────────────────────────────────────────────────────────

class TestFANAgent:
    @pytest.mark.asyncio
    async def test_process_with_feedback(self):
        agent = FANAgent()
        input_data = FANInput(
            interaction_id="test-fan-001",
            customer_feedback={"csat_score": 4.0, "comment": "Good service"},
            interaction_log={},
        )
        result = await agent.process(input_data)
        assert isinstance(result, FANOutput)
        assert result.feedback_analysis.csat_score == 4.0

    @pytest.mark.asyncio
    async def test_process_without_feedback(self):
        agent = FANAgent()
        input_data = FANInput(
            interaction_id="test-fan-002",
            customer_feedback=None,
            interaction_log={},
        )
        result = await agent.process(input_data)
        assert isinstance(result, FANOutput)
        assert result.feedback_analysis.csat_score is None
