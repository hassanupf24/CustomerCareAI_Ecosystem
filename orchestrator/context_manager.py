"""
Context Manager — reads / writes conversation context from the central DB.

Provides conversation-level state (history, previous intents, emotion trends)
so each agent has full context on every call.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

import structlog
from sqlalchemy import select

from db.models import Conversation, async_session


logger = structlog.get_logger()


class ContextManager:
    """Manages conversation context persistence."""

    async def get_or_create(
        self,
        conversation_id: Optional[str],
        customer_id: str = "",
        channel: str = "chat",
    ) -> dict:
        """
        Retrieve existing conversation context, or create a new one.
        Returns a plain dict containing the current context snapshot.
        """
        async with async_session() as session:
            if conversation_id:
                result = await session.execute(
                    select(Conversation).where(Conversation.id == conversation_id)
                )
                conv = result.scalar_one_or_none()
                if conv:
                    logger.info(
                        "context_loaded",
                        conversation_id=conversation_id,
                    )
                    return {
                        "conversation_id": conv.id,
                        "customer_id": conv.customer_id,
                        "channel": conv.channel,
                        "language": conv.language,
                        "history": conv.context.get("history", []) if conv.context else [],
                        "previous_intents": conv.context.get("previous_intents", []) if conv.context else [],
                        "emotion_trend": conv.context.get("emotion_trend", []) if conv.context else [],
                        "is_escalated": conv.is_escalated,
                        "turn_count": conv.context.get("turn_count", 0) if conv.context else 0,
                        "unresolved_turns": conv.context.get("unresolved_turns", 0) if conv.context else 0,
                    }

            # Create a new conversation
            new_id = conversation_id or str(uuid.uuid4())
            conv = Conversation(
                id=new_id,
                customer_id=customer_id,
                channel=channel,
                context={
                    "history": [],
                    "previous_intents": [],
                    "emotion_trend": [],
                    "turn_count": 0,
                    "unresolved_turns": 0,
                },
            )
            session.add(conv)
            await session.commit()
            logger.info("context_created", conversation_id=new_id)
            return {
                "conversation_id": new_id,
                "customer_id": customer_id,
                "channel": channel,
                "language": "en",
                "history": [],
                "previous_intents": [],
                "emotion_trend": [],
                "is_escalated": False,
                "turn_count": 0,
                "unresolved_turns": 0,
            }

    async def update(
        self,
        conversation_id: str,
        response_data: dict,
    ) -> None:
        """
        Update conversation context with the latest interaction data.
        """
        async with async_session() as session:
            result = await session.execute(
                select(Conversation).where(Conversation.id == conversation_id)
            )
            conv = result.scalar_one_or_none()
            if not conv:
                logger.warning("context_update_miss", conversation_id=conversation_id)
                return

            ctx = conv.context or {}
            history = ctx.get("history", [])
            history.append({
                "role": "customer",
                "text": response_data.get("customer_message", ""),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            history.append({
                "role": "assistant",
                "text": response_data.get("response_text", ""),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

            previous_intents = ctx.get("previous_intents", [])
            previous_intents.append(response_data.get("intent", "unknown"))

            emotion_trend = ctx.get("emotion_trend", [])
            emotion_trend.append(response_data.get("dominant_emotion", "neutral"))

            turn_count = ctx.get("turn_count", 0) + 1

            # Track unresolved turns
            unresolved_turns = ctx.get("unresolved_turns", 0)
            intent = response_data.get("intent", "unknown")
            if intent in ("unknown", "unclear", "unresolved"):
                unresolved_turns += 1
            else:
                unresolved_turns = 0

            conv.context = {
                "history": history[-20:],  # keep last 20 messages
                "previous_intents": previous_intents[-10:],
                "emotion_trend": emotion_trend[-10:],
                "turn_count": turn_count,
                "unresolved_turns": unresolved_turns,
            }
            conv.language = response_data.get("language", conv.language)
            conv.is_escalated = response_data.get("escalation_flag", False)
            conv.updated_at = datetime.now(timezone.utc)

            await session.commit()
            logger.info("context_updated", conversation_id=conversation_id, turn=turn_count)
