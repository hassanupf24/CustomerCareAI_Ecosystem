"""
Centralized configuration via Pydantic BaseSettings + python-dotenv.
All environment variables are loaded from .env (or system env) and validated.
"""

from __future__ import annotations

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide settings — never hard-code values; use .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Application ---
    app_name: str = "CustomerCareAI_Ecosystem"
    app_version: str = "1.0.0"
    debug: bool = True
    log_level: str = "INFO"

    # --- API Server ---
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1

    # --- Database ---
    database_url: str = "sqlite+aiosqlite:///./customercareai.db"

    # --- FAISS Index ---
    faiss_index_dir: str = "./data/faiss_indices"

    # --- Model Paths ---
    intent_model_name: str = "distilbert-base-uncased"
    emotion_model_name: str = "j-hartmann/emotion-english-distilroberta-base"
    embedding_model_name: str = "all-MiniLM-L6-v2"

    # --- Escalation Thresholds ---
    sentiment_escalation_threshold: float = -0.65
    consecutive_emotion_turns: int = 2
    max_unresolved_turns: int = 3

    # --- Translation ---
    translation_enabled: bool = True
    supported_languages: str = "en,ar"

    # --- Inter-Agent Service URLs ---
    ocs_service_url: str = "http://localhost:8001"
    kfo_service_url: str = "http://localhost:8002"
    eia_service_url: str = "http://localhost:8003"
    pir_service_url: str = "http://localhost:8004"
    fan_service_url: str = "http://localhost:8005"

    @property
    def supported_languages_list(self) -> list[str]:
        return [lang.strip() for lang in self.supported_languages.split(",")]

    @property
    def faiss_index_path(self) -> Path:
        return Path(self.faiss_index_dir)


def get_settings() -> Settings:
    """Factory — allows easy overriding in tests."""
    return Settings()
