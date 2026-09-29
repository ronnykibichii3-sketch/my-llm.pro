"""Environment-backed application configuration."""

from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded exclusively from environment variables."""

    app_name: str = "MyLLM.pro API"
    environment: str = "development"
    debug: bool = False
    database_url: str = "sqlite+aiosqlite:///:memory:"
    redis_url: str = "redis://localhost:6379/0"
    secret_key: SecretStr = SecretStr("test-secret-key-for-development-only")
    access_token_expire_minutes: int = 30
    log_level: str = "INFO"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: SecretStr | None = None
    memory_collection_name: str = "myllm_memories"
    knowledge_collection_name: str = "myllm_knowledge"
    knowledge_chunk_max_chars: int = 1200
    knowledge_chunk_overlap: int = 200
    knowledge_max_document_chars: int = 200_000
    knowledge_max_upload_bytes: int = 2_000_000
    context_max_chars: int = 6000
    profile_max_chars: int = 1500
    knowledge_max_chars: int = 4500
    knowledge_top_k: int = 5

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide validated settings instance."""

    return Settings()
