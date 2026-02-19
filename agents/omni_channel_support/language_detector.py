"""
Language Detector — langdetect wrapper.
Returns 'en' or 'ar' (defaults to 'en' for unsupported languages).
"""

from __future__ import annotations

from langdetect import detect, LangDetectException
import structlog

logger = structlog.get_logger()


def detect_language(text: str) -> str:
    """
    Detect language of input text.
    Returns 'en' or 'ar'. Falls back to 'en' for unsupported languages.
    """
    if not text or not text.strip():
        return "en"

    try:
        lang = detect(text)
        if lang == "ar":
            return "ar"
        return "en"
    except LangDetectException as e:
        logger.warning("language_detection_failed", error=str(e))
        return "en"
