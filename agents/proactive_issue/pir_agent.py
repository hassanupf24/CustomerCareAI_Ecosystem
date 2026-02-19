"""
Proactive Issue Resolution Agent (PIR)

Responsibilities:
- Monitor customer account data and usage telemetry
- Detect anomalies using Isolation Forest / statistical methods
- Generate proactive outreach alerts with severity scoring
"""

from __future__ import annotations

from pydantic import BaseModel
import structlog

from agents.base_agent import BaseAgent
from agents.proactive_issue.anomaly_detector import detect_anomalies
from agents.proactive_issue.alert_builder import build_alerts
from api.schemas import PIRInput, PIROutput

logger = structlog.get_logger()


class PIRAgent(BaseAgent):
    """Proactive Issue Resolution Agent."""

    def __init__(self) -> None:
        super().__init__(agent_name="PIR")

    async def process(self, input_data: BaseModel) -> PIROutput:
        """
        1. Analyze usage logs for anomalies
        2. Build structured alerts from detected anomalies
        """
        data: PIRInput = input_data  # type: ignore[assignment]

        if not data.usage_logs:
            self.logger.info(
                "pir_no_usage_data",
                interaction_id=data.interaction_id,
                account_id=data.account_id,
            )
            return PIROutput(proactive_alerts=[])

        # Step 1 — Anomaly detection
        anomalies = detect_anomalies(data.usage_logs)

        # Step 2 — Build alerts
        alerts = build_alerts(anomalies, data.account_id)

        self.logger.info(
            "pir_processed",
            interaction_id=data.interaction_id,
            account_id=data.account_id,
            anomalies_found=len(anomalies),
            alerts_generated=len(alerts),
        )

        return PIROutput(proactive_alerts=alerts)
