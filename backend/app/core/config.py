"""Environment-backed application configuration."""

from functools import lru_cache

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Provider names live here because the settings validator below, the provider
# factory and every log line must speak the same language (RULES.md Rule 7).
OLLAMA_PROVIDER = "ollama"
OPENAI_PROVIDER = "openai"
ANTHROPIC_PROVIDER = "anthropic"
SUPPORTED_LLM_PROVIDERS: tuple[str, ...] = (OLLAMA_PROVIDER, OPENAI_PROVIDER, ANTHROPIC_PROVIDER)

# Default endpoint and model per provider. A provider reads its model from these
# settings and never holds a model literal, so pointing the app at a different
# model is an environment change rather than a code change; the values below are
# only the convenience defaults a fresh checkout starts with.
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "llama3.2:3b"
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_ANTHROPIC_BASE_URL = "https://api.anthropic.com"
DEFAULT_ANTHROPIC_MODEL = "claude-3-haiku-20240307"


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
    # --- Inference provider ---
    llm_provider: str = OLLAMA_PROVIDER
    llm_request_timeout_seconds: float = 120.0
    llm_max_output_tokens: int = 512
    llm_temperature: float = 0.7
    llm_fallback_enabled: bool = True
    ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL
    ollama_model: str = DEFAULT_OLLAMA_MODEL
    openai_api_key: SecretStr | None = None
    openai_base_url: str = DEFAULT_OPENAI_BASE_URL
    openai_model: str = DEFAULT_OPENAI_MODEL
    anthropic_api_key: SecretStr | None = None
    anthropic_base_url: str = DEFAULT_ANTHROPIC_BASE_URL
    anthropic_model: str = DEFAULT_ANTHROPIC_MODEL

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @field_validator("llm_provider")
    @classmethod
    def _validate_llm_provider(cls, value: str) -> str:
        """Normalise the provider name and reject an unknown one while booting.

        Failing here rather than on the first chat turn means a typo in
        ``LLM_PROVIDER`` surfaces at startup instead of as a mystery 502.
        """
        provider = value.strip().lower()
        if provider not in SUPPORTED_LLM_PROVIDERS:
            raise ValueError(
                f"LLM_PROVIDER must be one of {', '.join(SUPPORTED_LLM_PROVIDERS)}, got {value!r}"
            )
        return provider


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide validated settings instance."""

    return Settings()
