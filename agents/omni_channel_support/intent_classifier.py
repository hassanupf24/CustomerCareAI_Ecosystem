"""
Intent Classifier — Transformer-based intent recognition.

Uses a zero-shot classification pipeline from HuggingFace transformers
to classify customer intents into predefined categories.
"""

from __future__ import annotations

from typing import Optional

import structlog

logger = structlog.get_logger()

# Predefined intent labels for customer care
INTENT_LABELS: list[str] = [
    "billing_inquiry",
    "technical_support",
    "account_management",
    "product_information",
    "complaint",
    "feedback",
    "escalation_request",
    "greeting",
    "farewell",
    "order_status",
    "cancellation",
    "refund_request",
    "general_inquiry",
]

# Lazy-loaded pipeline
_classifier = None


def _get_classifier():
    """Lazy-load the zero-shot classifier to avoid heavy startup cost."""
    global _classifier
    if _classifier is None:
        try:
            from transformers import pipeline
            _classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=-1,  # CPU
            )
            logger.info("intent_classifier_loaded", model="facebook/bart-large-mnli")
        except Exception as e:
            logger.error("intent_classifier_load_failed", error=str(e))
            _classifier = None
    return _classifier


async def classify_intent(
    text: str,
    candidate_labels: Optional[list[str]] = None,
) -> tuple[str, float]:
    """
    Classify the customer's intent.

    Returns:
        (intent_label, confidence_score)
    """
    if not text or not text.strip():
        return "unknown", 0.0

    labels = candidate_labels or INTENT_LABELS

    classifier = _get_classifier()
    if classifier is None:
        # Fallback: rule-based matching
        return _rule_based_intent(text)

    try:
        result = classifier(text, labels, multi_label=False)
        top_label = result["labels"][0]
        top_score = result["scores"][0]

        if top_score < 0.25:
            return "unknown", top_score

        return top_label, top_score
    except Exception as e:
        logger.error("intent_classification_failed", error=str(e))
        return _rule_based_intent(text)


def _rule_based_intent(text: str) -> tuple[str, float]:
    """Simple keyword-based fallback when the transformer is unavailable."""
    text_lower = text.lower()

    rules: dict[str, list[str]] = {
        "billing_inquiry": ["bill", "invoice", "charge", "payment", "price"],
        "technical_support": ["error", "bug", "crash", "not working", "broken", "fix"],
        "account_management": ["account", "password", "login", "profile", "settings"],
        "complaint": ["complaint", "unhappy", "disappointed", "terrible", "worst"],
        "escalation_request": ["human", "agent", "manager", "supervisor", "speak to"],
        "order_status": ["order", "shipping", "delivery", "track"],
        "cancellation": ["cancel", "terminate", "end subscription"],
        "refund_request": ["refund", "money back", "return"],
        "greeting": ["hello", "hi", "hey", "good morning", "good afternoon"],
        "farewell": ["bye", "goodbye", "thank you", "thanks"],
    }

    for intent, keywords in rules.items():
        for kw in keywords:
            if kw in text_lower:
                return intent, 0.6

    return "general_inquiry", 0.3
