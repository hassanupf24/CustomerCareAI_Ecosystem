"""
Emotion Classifier — HuggingFace emotion model wrapper.

Uses j-hartmann/emotion-english-distilroberta-base for multi-class
emotion detection with a TextBlob/NLTK-based fallback.
"""

from __future__ import annotations

from typing import Optional

import structlog

logger = structlog.get_logger()

_classifier = None


def _get_classifier():
    """Lazy-load the emotion classifier pipeline."""
    global _classifier
    if _classifier is None:
        try:
            from transformers import pipeline
            _classifier = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                top_k=None,  # return all labels with scores
                device=-1,
            )
            logger.info("emotion_classifier_loaded")
        except Exception as e:
            logger.error("emotion_classifier_load_failed", error=str(e))
    return _classifier


async def classify_emotion(text: str) -> dict[str, float]:
    """
    Classify emotions in text.

    Returns:
        Dict mapping emotion labels to scores, e.g. {"joy": 0.8, "anger": 0.05, ...}
    """
    if not text or not text.strip():
        return {"neutral": 1.0}

    classifier = _get_classifier()
    if classifier is not None:
        try:
            results = classifier(text[:512])  # truncate for model limit
            if results and isinstance(results[0], list):
                return {r["label"]: round(r["score"], 4) for r in results[0]}
            elif results:
                return {r["label"]: round(r["score"], 4) for r in results}
        except Exception as e:
            logger.error("emotion_classification_failed", error=str(e))

    # Fallback: TextBlob-based sentiment → simple emotion mapping
    return _fallback_classify(text)


def _fallback_classify(text: str) -> dict[str, float]:
    """TextBlob polarity → approximate emotion mapping."""
    try:
        from textblob import TextBlob
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity  # -1 to 1

        if polarity > 0.5:
            return {"joy": 0.7, "neutral": 0.2, "surprise": 0.1}
        elif polarity > 0.1:
            return {"neutral": 0.5, "joy": 0.3, "surprise": 0.2}
        elif polarity > -0.1:
            return {"neutral": 0.8, "sadness": 0.1, "joy": 0.1}
        elif polarity > -0.5:
            return {"sadness": 0.4, "anger": 0.3, "disgust": 0.2, "neutral": 0.1}
        else:
            return {"anger": 0.5, "disgust": 0.3, "sadness": 0.2}
    except Exception:
        return {"neutral": 1.0}
