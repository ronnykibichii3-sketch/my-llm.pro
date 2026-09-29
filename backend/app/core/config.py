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

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide validated settings instance."""

    return Settings()
