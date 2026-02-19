"""
SQLAlchemy ORM models for conversations, interaction logs, and feedback.

Uses SQLAlchemy 2.0 style with Mapped annotations for full type safety.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    String,
    Text,
    JSON,
    create_engine,
)
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from config.settings import get_settings


# ── Base ──────────────────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _uuid() -> str:
    return str(uuid.uuid4())


# ── Conversation ──────────────────────────────────────────────────────────────

class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    customer_id: Mapped[str] = mapped_column(String(128), index=True)
    channel: Mapped[str] = mapped_column(String(16), default="chat")
    language: Mapped[str] = mapped_column(String(8), default="en")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)
    context: Mapped[dict | None] = mapped_column(JSON, default=dict)
    is_escalated: Mapped[bool] = mapped_column(Boolean, default=False)

    interactions: Mapped[list["InteractionLog"]] = relationship(back_populates="conversation")


# ── Interaction Log ───────────────────────────────────────────────────────────

class InteractionLog(Base):
    __tablename__ = "interaction_logs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    conversation_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("conversations.id"), index=True
    )
    customer_message: Mapped[str] = mapped_column(Text, default="")
    response_text: Mapped[str] = mapped_column(Text, default="")
    intent: Mapped[str] = mapped_column(String(64), default="unknown")
    sentiment_score: Mapped[float] = mapped_column(Float, default=0.0)
    dominant_emotion: Mapped[str] = mapped_column(String(32), default="neutral")
    escalation_flag: Mapped[bool] = mapped_column(Boolean, default=False)
    escalation_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    agent_logs: Mapped[dict | None] = mapped_column(JSON, default=dict)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    conversation: Mapped["Conversation"] = relationship(back_populates="interactions")
    feedback: Mapped["Feedback | None"] = relationship(back_populates="interaction", uselist=False)


# ── Feedback ──────────────────────────────────────────────────────────────────

class Feedback(Base):
    __tablename__ = "feedback"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    interaction_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("interaction_logs.id"), index=True
    )
    csat_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    sentiment_trend: Mapped[str | None] = mapped_column(String(32), nullable=True)
    top_issues: Mapped[list | None] = mapped_column(JSON, nullable=True)
    knowledge_gap_flags: Mapped[list | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    interaction: Mapped["InteractionLog"] = relationship(back_populates="feedback")


# ── Knowledge Base Article ────────────────────────────────────────────────────

class KBArticle(Base):
    __tablename__ = "kb_articles"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    title: Mapped[str] = mapped_column(String(256))
    content: Mapped[str] = mapped_column(Text)
    language: Mapped[str] = mapped_column(String(8), default="en")
    category: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)


# ── Engine helpers ────────────────────────────────────────────────────────────

_settings = get_settings()

engine = create_async_engine(_settings.database_url, echo=_settings.debug)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def init_db() -> None:
    """Create all tables (called on startup)."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
