"""
Orchestrator Main — FastAPI application.

Single entry point for all requests. Coordinates the sequential agent pipeline:
  [OCS] → [KFO] → [EIA] → [PIR] → (response) → [FAN async]
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import BackgroundTasks, FastAPI
import structlog

from agents.omni_channel_support.ocs_agent import OCSAgent
from agents.knowledge_base.kfo_agent import KFOAgent
from agents.emotional_intelligence.eia_agent import EIAAgent
from agents.proactive_issue.pir_agent import PIRAgent
from agents.feedback_analytics.fan_agent import FANAgent
from api.endpoints import router
from api.middleware import RateLimitMiddleware, RequestIDMiddleware
from api.schemas import (
    ChannelType,
    CustomerRequest,
    EIAInput,
    EscalationPayload,
    FANInput,
    KFOInput,
    OCSInput,
    OrchestratorResponse,
    PIRInput,
    SupportedLanguage,
)
from config.settings import get_settings
from db.models import init_db
from orchestrator.aggregator import aggregate_outputs
from orchestrator.context_manager import ContextManager
from orchestrator.logger import configure_logging

logger = structlog.get_logger()
settings = get_settings()


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    configure_logging(settings.log_level)
    logger.info("starting_up", app=settings.app_name, version=settings.app_version)
    await init_db()
    yield
    logger.info("shutting_down")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="CustomerCareAI Orchestrator",
    description="Multi-agent orchestration system for enterprise customer care.",
    version=settings.app_version,
    lifespan=lifespan,
)

# Middleware
app.add_middleware(RequestIDMiddleware)
app.add_middleware(RateLimitMiddleware, max_requests=100, window_seconds=60)

# Routers
app.include_router(router)

# Shared instances
context_mgr = ContextManager()


# ── Pipeline ──────────────────────────────────────────────────────────────────

async def run_pipeline(
    request: CustomerRequest,
    background_tasks: BackgroundTasks,
) -> OrchestratorResponse:
    """
    Execute the full agent orchestration pipeline sequentially.

    Steps 1–4 are sequential; Step 7 (FAN) is async.
    Any agent failure is caught and logged — the pipeline continues with
    degraded output.
    """
    interaction_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    logger.info(
        "pipeline_started",
        interaction_id=interaction_id,
        customer_id=request.customer_id,
        channel=request.channel.value,
    )

    # Load or create conversation context
    context = await context_mgr.get_or_create(
        conversation_id=request.conversation_id,
        customer_id=request.customer_id,
        channel=request.channel.value,
    )
    conversation_id = context["conversation_id"]

    # ── Step 1: OCS — Intent + Channel Normalization + Draft Response ─────
    ocs_input = OCSInput(
        interaction_id=interaction_id,
        customer_message=request.customer_message,
        conversation_context=context,
        channel=request.channel,
    )
    ocs_agent = OCSAgent()
    ocs_output = await ocs_agent.safe_process(ocs_input)

    # Determine language from OCS output
    detected_language = SupportedLanguage.EN
    if ocs_output and hasattr(ocs_output, "language"):
        detected_language = ocs_output.language

    # ── Step 2: KFO — Semantic FAQ Retrieval ──────────────────────────────
    query_text = request.customer_message
    if ocs_output:
        query_text = f"{ocs_output.intent} {request.customer_message}"

    kfo_input = KFOInput(
        interaction_id=interaction_id,
        query_text=query_text,
        top_k=5,
        language=detected_language,
    )
    kfo_agent = KFOAgent()
    kfo_output = await kfo_agent.safe_process(kfo_input)

    # ── Step 3: EIA — Sentiment + Emotion + Tone Adjustment ──────────────
    eia_input = EIAInput(
        interaction_id=interaction_id,
        conversation_text=request.customer_message,
        conversation_history=request.conversation_history,
    )
    eia_agent = EIAAgent()
    eia_output = await eia_agent.safe_process(eia_input)

    # ── Step 4: PIR — Account Anomaly Scan ────────────────────────────────
    pir_output = None
    if request.account_id:
        pir_input = PIRInput(
            interaction_id=interaction_id,
            account_id=request.account_id,
            account_data=request.account_data or {},
            usage_logs=request.usage_logs or [],
        )
        pir_agent = PIRAgent()
        pir_output = await pir_agent.safe_process(pir_input)

    # ── Step 5: Escalation Gate ───────────────────────────────────────────
    # (handled inside aggregate_outputs via _check_escalation)

    # ── Step 6: Unified Response Assembly ─────────────────────────────────
    response = aggregate_outputs(
        interaction_id=interaction_id,
        customer_id=request.customer_id,
        channel=request.channel,
        language=detected_language,
        customer_message=request.customer_message,
        context=context,
        ocs_output=ocs_output,
        kfo_output=kfo_output,
        eia_output=eia_output,
        pir_output=pir_output,
    )

    # Update conversation context
    await context_mgr.update(
        conversation_id=conversation_id,
        response_data={
            "customer_message": request.customer_message,
            "response_text": response.response_text,
            "intent": response.intent,
            "dominant_emotion": response.dominant_emotion,
            "escalation_flag": response.escalation_flag,
            "language": response.language.value,
        },
    )

    # ── If escalated, route to escalation queue ───────────────────────────
    if response.escalation_flag:
        escalation_payload = EscalationPayload(
            interaction_id=interaction_id,
            customer_id=request.customer_id,
            channel=request.channel,
            escalation_reason=response.escalation_reason or "Threshold exceeded",
            conversation_context=context,
            summary=response.response_text,
            timestamp=timestamp,
        )
        logger.info(
            "escalation_triggered",
            interaction_id=interaction_id,
            reason=response.escalation_reason,
        )
        # In production: POST to /api/v1/escalate or push to queue

    # ── Step 7: FAN — Async Feedback + Knowledge Update ───────────────────
    fan_input = FANInput(
        interaction_id=interaction_id,
        customer_feedback=request.customer_feedback,
        interaction_log={
            "customer_message": request.customer_message,
            "response_text": response.response_text,
            "intent": response.intent,
            "sentiment_score": response.sentiment_score,
            "history": context.get("history", []),
            "feedback_history": [],
            "past_interactions": [],
        },
    )
    fan_agent = FANAgent()
    background_tasks.add_task(fan_agent.safe_process, fan_input)

    logger.info(
        "pipeline_completed",
        interaction_id=interaction_id,
        escalation=response.escalation_flag,
    )

    return response


# ── Uvicorn entry point ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "orchestrator.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        workers=settings.workers,
    )
