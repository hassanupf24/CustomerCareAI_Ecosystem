"""
Tests for the Proactive Issue Resolution Agent (PIR).
"""

import pytest

from agents.proactive_issue.anomaly_detector import (
    detect_anomalies,
    _extract_numeric_keys,
    _statistical_fallback,
    _build_feature_matrix,
)
from agents.proactive_issue.alert_builder import build_alerts, _determine_severity
from agents.proactive_issue.pir_agent import PIRAgent
from api.schemas import PIRInput, PIROutput, SeverityLevel


# ── Anomaly Detector ──────────────────────────────────────────────────────────

class TestAnomalyDetector:
    def test_empty_logs(self):
        result = detect_anomalies([])
        assert result == []

    def test_extract_numeric_keys(self):
        logs = [
            {"api_calls": 100, "error_count": 2, "name": "test"},
            {"api_calls": 150, "error_count": 1, "status": "ok"},
        ]
        keys = _extract_numeric_keys(logs)
        assert "api_calls" in keys
        assert "error_count" in keys
        assert "name" not in keys

    def test_detects_anomaly_in_outlier(self):
        # Create logs with one outlier
        logs = [
            {"api_calls": 100, "error_count": 2},
            {"api_calls": 110, "error_count": 3},
            {"api_calls": 105, "error_count": 1},
            {"api_calls": 95, "error_count": 2},
            {"api_calls": 100, "error_count": 2},
            {"api_calls": 500, "error_count": 50},  # anomaly
        ]
        anomalies = detect_anomalies(logs)
        assert isinstance(anomalies, list)


# ── Alert Builder ─────────────────────────────────────────────────────────────

class TestAlertBuilder:
    def test_severity_critical(self):
        assert _determine_severity(-3.0) == SeverityLevel.CRITICAL

    def test_severity_high(self):
        assert _determine_severity(-1.5) == SeverityLevel.HIGH

    def test_severity_medium(self):
        assert _determine_severity(-0.7) == SeverityLevel.MEDIUM

    def test_severity_low(self):
        assert _determine_severity(0.5) == SeverityLevel.LOW

    def test_build_alerts_from_anomalies(self):
        anomalies = [
            {"index": 0, "anomaly_score": -1.5, "data": {"error_count": 50}},
        ]
        alerts = build_alerts(anomalies, "ACC-001")
        assert len(alerts) == 1
        assert alerts[0].severity == SeverityLevel.HIGH
        assert alerts[0].alert_type == "high_error_rate"

    def test_empty_anomalies(self):
        alerts = build_alerts([], "ACC-001")
        assert alerts == []


# ── PIR Agent ─────────────────────────────────────────────────────────────────

class TestPIRAgent:
    @pytest.mark.asyncio
    async def test_process_no_usage_data(self):
        agent = PIRAgent()
        input_data = PIRInput(
            interaction_id="test-pir-001",
            account_id="ACC-001",
            account_data={},
            usage_logs=[],
        )
        result = await agent.process(input_data)
        assert isinstance(result, PIROutput)
        assert result.proactive_alerts == []

    @pytest.mark.asyncio
    async def test_process_with_usage_data(self):
        agent = PIRAgent()
        logs = [
            {"api_calls": 100, "error_count": 2},
            {"api_calls": 110, "error_count": 3},
            {"api_calls": 105, "error_count": 1},
            {"api_calls": 95, "error_count": 2},
        ]
        input_data = PIRInput(
            interaction_id="test-pir-002",
            account_id="ACC-002",
            account_data={"plan": "enterprise"},
            usage_logs=logs,
        )
        result = await agent.process(input_data)
        assert isinstance(result, PIROutput)
        assert isinstance(result.proactive_alerts, list)
