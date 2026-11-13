import json
import os
from typing import Annotated

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

_DEFAULT_DEV_CORS_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://localhost:8080",
    "http://localhost:7860",
]


def _parse_origins(value: object) -> list[str]:
    """Parse a comma-separated string, JSON list string, or list of CORS origins."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(origin).strip() for origin in value if str(origin).strip()]
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return []
        if raw.startswith("["):
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    return [str(origin).strip() for origin in parsed if str(origin).strip()]
            except json.JSONDecodeError:
                pass
        return [origin.strip() for origin in raw.split(",") if origin.strip()]
    return []


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Database
    database_url: str = Field(
        default="postgresql+psycopg://localhost/hybrid_rag",
        description="SQLAlchemy connection string (pooled endpoint for app)",
    )
    database_url_direct: str | None = Field(
        default_factory=lambda: os.getenv(
            "DATABASE_URL_DIRECT", os.getenv("DATABASE_URL_UNPOOLED")
        ),
        description="Direct (non-pooler) endpoint for Alembic migrations",
    )
    db_pool_pre_ping: bool = True
    db_pool_recycle: int = 180
    db_pool_size: int = 2
    db_max_overflow: int = 0

    @field_validator("database_url", "database_url_direct", mode="before")
    @classmethod
    def normalize_postgres_url(cls, v: str | None) -> str | None:
        if not v:
            return v
        # Normalize driver prefix
        if v.startswith("postgres://"):
            v = "postgresql+psycopg://" + v[len("postgres://") :]
        elif v.startswith("postgresql://"):
            v = "postgresql+psycopg://" + v[len("postgresql://") :]

        # Robustly encode password if it contains '@' or special characters
        if "://" in v:
            import urllib.parse

            scheme, remainder = v.split("://", 1)
            if "@" in remainder:
                userinfo, host_path = remainder.rsplit("@", 1)
                if ":" in userinfo:
                    user, pw = userinfo.split(":", 1)
                    pw = urllib.parse.quote_plus(urllib.parse.unquote_plus(pw))
                    return f"{scheme}://{user}:{pw}@{host_path}"
        return v

    # Storage (S3-compatible)
    aws_endpoint_url_s3: str | None = None
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None
    aws_region: str = "ap-southeast-1"
    s3_bucket: str = "uploads"

    # Gemini
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-3.5-flash-lite"
    gemini_verifier_model: str = "gemini-3.5-flash"

    # Additional Providers (Groq & Local Ollama)
    groq_api_key: str | None = None
    ollama_host: str = "http://localhost:11434"

    # Embedding & reranking
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "BAAI/bge-reranker-base"

    # Cache paths
    hf_home: str = Field(
        default_factory=lambda: os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
        alias="HF_HOME",
    )
    fastembed_cache_path: str = Field(
        default_factory=lambda: os.getenv(
            "FASTEMBED_CACHE_PATH", os.path.expanduser("~/.cache/fastembed")
        ),
        alias="FASTEMBED_CACHE_PATH",
    )

    # Application
    app_env: str = "development"
    log_level: str = "INFO"
    uvicorn_workers: int = 1
    uvicorn_host: str = "0.0.0.0"
    uvicorn_port: int = 7860

    # Security
    frontend_url: str | None = None
    cors_origins: Annotated[list[str], NoDecode] = Field(default_factory=list)
    request_timeout_seconds: int = 120
    max_pdf_pages: int = 500

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: object) -> list[str]:
        return _parse_origins(v)

    @model_validator(mode="after")
    def resolve_cors_origins(self) -> "Settings":
        if not self.cors_origins:
            if self.frontend_url:
                self.cors_origins = _parse_origins(self.frontend_url)
            elif not self.is_production:
                self.cors_origins = list(_DEFAULT_DEV_CORS_ORIGINS)
        return self

    # Storage quotas (200 MB raw document cap provides safe headroom under Neon 500 MB free-tier limit)
    max_upload_size_mb: int = 50
    max_documents_count: int = 50
    max_storage_mb: int = 200

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    def apply_cache_env(self) -> None:
        """Set cache environment variables to protect C: drive."""
        os.environ.setdefault("HF_HOME", self.hf_home)
        os.environ.setdefault("FASTEMBED_CACHE_PATH", self.fastembed_cache_path)


settings = Settings()
