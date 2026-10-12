from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.document_types import DOCUMENTS_ROOT


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="OCRSD_", extra="ignore")
    storage: Path = Path("var/web")
    documents: Path = DOCUMENTS_ROOT
    max_count: int = Field(default=10, ge=1, le=100)
    max_pending: int = Field(default=20, ge=1, le=1000)
    jobs_per_hour: int = Field(default=10, ge=1)
    retention_hours: int = Field(default=24, ge=1)
    job_timeout: int = Field(default=900, ge=1)
    max_job_bytes: int = Field(default=256 * 1024 * 1024, ge=1024)
    max_upload_bytes: int = Field(default=12 * 1024 * 1024, ge=1024)
    max_storage_bytes: int = Field(default=4 * 1024**3, ge=1024)
    min_free_bytes: int = Field(default=512 * 1024**2, ge=0)
    memory_mb: int = Field(default=4096, ge=256)
    neural_enabled: bool = False
    accepting_jobs: bool = True
    ai_provider: Literal["none", "openai"] = "none"
    ai_base_url: str = "http://127.0.0.1:11434/v1"
    ai_model: str = "qwen3:1.7b"
    ai_timeout: int = Field(default=60, ge=1, le=600)
    ai_allow_remote: bool = False
    ai_api_key: SecretStr | None = None
    mapping_min_score: float = Field(default=0.5, ge=0, le=1)
    mapping_min_margin: float = Field(default=0.2, ge=0, le=1)
