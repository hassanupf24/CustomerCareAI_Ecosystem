"""
Pydantic v2 request/response models for the entire CustomerCareAI API.
All data flowing through the system is validated by these schemas.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ── Enums ──────────────────────────────────────────────────────────────────────

class ChannelType(str, Enum):
    CHAT = "chat"
    EMAIL = "email"
    SOCIAL = "social"


class SeverityLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SupportedLanguage(str, Enum):
    EN = "en"
    AR = "ar"


# ── Customer Request ──────────────────────────────────────────────────────────

class CustomerRequest(BaseModel):
    """Incoming customer interaction payload."""
    conversation_id: Optional[str] = None
    customer_id: str
    account_id: Optional[str] = None
    customer_message: str
    channel: ChannelType = ChannelType.CHAT
    conversation_history: list[str] = Field(default_factory=list)
    customer_feedback: Optional[dict] = None  # present for post-interaction
    account_data: Optional[dict] = None
    usage_logs: Optional[list[dict]] = None


# ── Agent-Specific Inputs ─────────────────────────────────────────────────────

class OCSInput(BaseModel):
    interaction_id: str
    customer_message: str
    conversation_context: dict = Field(default_factory=dict)
    channel: ChannelType = ChannelType.CHAT
    language: SupportedLanguage = SupportedLanguage.EN


class KFOInput(BaseModel):
    interaction_id: str
    query_text: str
    top_k: int = 5
    language: SupportedLanguage = SupportedLanguage.EN


class EIAInput(BaseModel):
    interaction_id: str
    conversation_text: str
    conversation_history: list[str] = Field(default_factory=list)


class PIRInput(BaseModel):
    interaction_id: str
    account_id: str
    account_data: dict = Field(default_factory=dict)
    usage_logs: list[dict] = Field(default_factory=list)


class FANInput(BaseModel):
    interaction_id: str
    customer_feedback: Optional[dict] = None
    interaction_log: dict = Field(default_factory=dict)


# ── Agent-Specific Outputs ────────────────────────────────────────────────────

class OCSOutput(BaseModel):
    response_text: str = ""
    intent: str = "unknown"
    suggested_faq_ids: list[str] = Field(default_factory=list)
    escalation_flag: bool = False
    language: SupportedLanguage = SupportedLanguage.EN


class FAQArticle(BaseModel):
    article_id: str
    title: str
    content_snippet: str
    confidence_score: float


class KFOOutput(BaseModel):
    suggested_faq_articles: list[FAQArticle] = Field(default_factory=list)
    updated_knowledge: bool = False


class EIAOutput(BaseModel):
    sentiment_score: float = 0.0
    dominant_emotion: str = "neutral"
    escalation_flag: bool = False
    tone_recommendation: str = "neutral"


class ProactiveAlert(BaseModel):
    alert_type: str
    severity: SeverityLevel
    recommended_action: str
    timestamp: str  # ISO-8601


class PIROutput(BaseModel):
    proactive_alerts: list[ProactiveAlert] = Field(default_factory=list)


class FeedbackAnalysis(BaseModel):
    csat_score: Optional[float] = None
    sentiment_trend: Optional[str] = None
    top_issues: list[str] = Field(default_factory=list)
    knowledge_gap_flags: list[str] = Field(default_factory=list)


class KnowledgeBaseUpdate(BaseModel):
    article_id: str
    update_type: str  # "new" | "revision"
    content: str


class FANOutput(BaseModel):
    feedback_analysis: FeedbackAnalysis = Field(default_factory=FeedbackAnalysis)
    knowledge_base_update: list[KnowledgeBaseUpdate] = Field(default_factory=list)


# ── Agent Logs ────────────────────────────────────────────────────────────────

class AgentLogs(BaseModel):
    ocs: dict = Field(default_factory=dict)
    kfo: dict = Field(default_factory=dict)
    eia: dict = Field(default_factory=dict)
    pir: dict = Field(default_factory=dict)
    fan: dict = Field(default_factory=dict)


# ── Orchestrator Unified Response ─────────────────────────────────────────────

class OrchestratorResponse(BaseModel):
    """Final output returned to the caller — conforms to §4 schema."""
    interaction_id: str
    timestamp: str  # ISO-8601
    customer_id: str
    channel: ChannelType
    language: SupportedLanguage = SupportedLanguage.EN
    response_text: str = ""
    intent: str = "unknown"
    sentiment_score: float = 0.0
    dominant_emotion: str = "neutral"
    escalation_flag: bool = False
    escalation_reason: Optional[str] = None
    suggested_faq_articles: list[FAQArticle] = Field(default_factory=list)
    proactive_alerts: list[ProactiveAlert] = Field(default_factory=list)
    feedback_analysis: FeedbackAnalysis = Field(default_factory=FeedbackAnalysis)
    agent_logs: AgentLogs = Field(default_factory=AgentLogs)


# ── Escalation Payload ────────────────────────────────────────────────────────

class EscalationPayload(BaseModel):
    interaction_id: str
    customer_id: str
    channel: ChannelType
    escalation_reason: str
    conversation_context: dict = Field(default_factory=dict)
    summary: str = ""
    timestamp: str  # ISO-8601
