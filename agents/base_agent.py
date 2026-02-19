"""
Base Agent — Abstract class that all five agents inherit from.

Provides:
- Structured logging via structlog
- Uniform error-handling wrapper (safe_process)
- Consistent interface via abstract `process` method
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import structlog
from pydantic import BaseModel

logger = structlog.get_logger()


class BaseAgent(ABC):
    """Every agent must subclass BaseAgent and implement `process`."""

    def __init__(self, agent_name: str) -> None:
        self.agent_name = agent_name
        self.logger = logger.bind(agent=agent_name)

    @abstractmethod
    async def process(self, input_data: BaseModel) -> BaseModel:
        """Core logic — each agent overrides this."""
        raise NotImplementedError

    async def safe_process(self, input_data: BaseModel) -> BaseModel | None:
        """
        Wraps `process()` with structured error handling.
        Returns None (instead of raising) so the orchestrator can degrade
        gracefully when one agent fails.
        """
        try:
            interaction_id = getattr(input_data, "interaction_id", "N/A")
            self.logger.info(
                "agent_started",
                interaction_id=interaction_id,
            )
            result = await self.process(input_data)
            self.logger.info(
                "agent_completed",
                interaction_id=interaction_id,
            )
            return result
        except Exception as e:
            self.logger.error(
                "agent_failed",
                error=str(e),
                exc_info=True,
            )
            return None
