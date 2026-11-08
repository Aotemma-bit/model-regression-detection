"""Runtime configuration loaded from the environment."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, field_validator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PLACEHOLDER_API_KEYS = frozenset({"", "your_openai_api_key_here", "changeme"})


class Settings(BaseModel):
    """Service settings. The API key is excluded from repr and serialization."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    openai_api_key: str = Field(default="", repr=False, exclude=True)
    openai_model: str = "gpt-4o-mini"
    openai_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    openai_timeout_seconds: float = Field(default=30.0, gt=0.0, le=300.0)
    openai_max_retries: int = Field(default=2, ge=0, le=5)
    app_env: str = "development"
    database_path: Path = Field(default=PROJECT_ROOT / "data" / "evaluations.sqlite3")
    dataset_path: Path = Field(default=PROJECT_ROOT / "data" / "golden_dataset.jsonl")
    log_level: str = "INFO"
    regression_drop_threshold: float = Field(default=0.0, ge=0.0)

    @field_validator("database_path", "dataset_path", mode="before")
    @classmethod
    def resolve_path(cls, value: str | Path) -> Path:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path

    @field_validator("log_level")
    @classmethod
    def normalize_log_level(cls, value: str) -> str:
        normalized = value.upper()
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if normalized not in allowed:
            raise ValueError(f"log_level must be one of {sorted(allowed)}")
        return normalized

    @field_validator("openai_model", "app_env")
    @classmethod
    def require_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be blank")
        return stripped

    @property
    def api_key_configured(self) -> bool:
        return self.openai_api_key.strip() not in PLACEHOLDER_API_KEYS


def load_settings() -> Settings:
    """Load settings from the process environment and a local .env file.

    Existing process variables win over the file. The API key is read into
    memory for the OpenAI client and is never written to logs by this loader.
    """

    load_dotenv(PROJECT_ROOT / ".env", override=False)
    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        openai_temperature=float(os.getenv("OPENAI_TEMPERATURE", "0")),
        openai_timeout_seconds=float(os.getenv("OPENAI_TIMEOUT_SECONDS", "30")),
        openai_max_retries=int(os.getenv("OPENAI_MAX_RETRIES", "2")),
        app_env=os.getenv("APP_ENV", "development"),
        database_path=os.getenv("DATABASE_PATH", str(PROJECT_ROOT / "data" / "evaluations.sqlite3")),
        dataset_path=os.getenv("DATASET_PATH", str(PROJECT_ROOT / "data" / "golden_dataset.jsonl")),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        regression_drop_threshold=float(os.getenv("REGRESSION_DROP_THRESHOLD", "0")),
    )
