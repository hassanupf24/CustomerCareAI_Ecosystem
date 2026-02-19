"""
Structured JSON logger built on structlog.
Every event automatically includes interaction_id, agent_name, and ISO-8601 timestamp.
"""

from __future__ import annotations

import structlog


def configure_logging(log_level: str = "INFO") -> None:
    """Call once at application startup to wire structured logging."""
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            structlog.get_level_from_name(log_level),
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(agent_name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a bound logger, optionally pre-bound to an agent name."""
    log = structlog.get_logger()
    if agent_name:
        log = log.bind(agent=agent_name)
    return log
