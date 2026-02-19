"""
API Endpoints — All REST API route definitions (FastAPI routers).

Delegates to the orchestrator for the main interaction pipeline
and provides supporting endpoints for escalation, health, and feedback.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
import structlog

from api.schemas import (
    CustomerRequest,
    EscalationPayload,
    OrchestratorResponse,
    FANInput,
)
from agents.feedback_analytics.fan_agent import FANAgent

logger = structlog.get_logger()

router = APIRouter(prefix="/api/v1", tags=["CustomerCareAI"])

# Placeholder escalation queue (in production: Redis / message broker)
_escalation_queue: list[EscalationPayload] = []


@router.post("/interact", response_model=OrchestratorResponse)
async def interact(
    request: CustomerRequest,
    background_tasks: BackgroundTasks,
) -> OrchestratorResponse:
    """
    Main interaction endpoint — passes the request through the full
    agent orchestration pipeline and returns a unified response.
    """
    # Import here to avoid circular imports during startup
    from orchestrator.main import run_pipeline

    response = await run_pipeline(request, background_tasks)
    return response


@router.post("/escalate")
async def escalate(payload: EscalationPayload) -> dict:
    """
    Receives escalation payloads and queues them for human agents.
    """
    _escalation_queue.append(payload)
    logger.info(
        "escalation_queued",
        interaction_id=payload.interaction_id,
        reason=payload.escalation_reason,
    )
    return {
        "status": "queued",
        "interaction_id": payload.interaction_id,
        "position_in_queue": len(_escalation_queue),
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


@router.get("/escalation-queue")
async def get_escalation_queue() -> dict:
    """Return the current escalation queue status."""
    return {
        "queue_length": len(_escalation_queue),
        "items": [item.model_dump() for item in _escalation_queue[-10:]],
    }


@router.post("/feedback")
async def submit_feedback(
    interaction_id: str,
    feedback: dict,
) -> dict:
    """
    Post-interaction feedback submission endpoint.
    Triggers FAN agent asynchronously.
    """
    fan_input = FANInput(
        interaction_id=interaction_id,
        customer_feedback=feedback,
        interaction_log={},
    )
    fan_agent = FANAgent()
    result = await fan_agent.safe_process(fan_input)

    return {
        "status": "received",
        "interaction_id": interaction_id,
        "analysis": result.model_dump() if result else None,
    }


@router.get("/health")
async def health_check() -> dict:
    """Simple health-check endpoint."""
    return {
        "status": "healthy",
        "service": "CustomerCareAI_Ecosystem",
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
