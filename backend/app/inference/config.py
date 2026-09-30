"""Inference provider configuration and prompt-composition constants.

Every endpoint path, role, prompt phrase and tunable used while asking a model
for one completion lives here, so the provider implementations, the environment
settings and the tests cannot drift apart (RULES.md Rule 7).

``InferenceConfig`` is the provider-agnostic view of "who answers this turn": the
selected provider plus the model, base URL and credential that belong to *that*
provider. Resolving the pair here keeps the factory and the providers free of any
knowledge of the settings layout, and keeps model names out of the code: a
provider only ever sees ``config.model``.
"""

from dataclasses import dataclass

from pydantic import SecretStr

from app.core.config import (
    ANTHROPIC_PROVIDER,
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_MODEL,
    OLLAMA_PROVIDER,
    OPENAI_PROVIDER,
    SUPPORTED_LLM_PROVIDERS,
    Settings,
)

# Sampling and transport defaults; the environment values that override them are
# read in ``from_settings`` below.
DEFAULT_LLM_REQUEST_TIMEOUT_SECONDS = 120.0
DEFAULT_LLM_MAX_OUTPUT_TOKENS = 512
DEFAULT_LLM_TEMPERATURE = 0.7
MIN_TEMPERATURE = 0.0
MAX_TEMPERATURE = 2.0

# Ollama's chat endpoint. The model, not the path, is what changes between
# deployments, and it comes from OLLAMA_MODEL.
OLLAMA_CHAT_PATH = "/api/chat"

# Message roles understood by every provider wired so far.
SYSTEM_ROLE = "system"
USER_ROLE = "user"

# The behaviour instruction every turn is built on. Kept short and provider-neutral
# so switching providers does not change the assistant's voice; the turn's context
# is appended to it by ``compose_system_message``, never concatenated by a caller.
DEFAULT_SYSTEM_PROMPT = (
    "You are MyLLM.pro, a personal assistant running on the user's own machine. "
    "Answer directly and plainly, in the language the user writes in. "
    "Use the context when it is relevant; if it is not enough to answer, say what "
    "is missing instead of inventing a fact."
)

# How the context block is introduced inside the system message. The block itself
# already carries the context builder's preamble, so this is only a label.
CONTEXT_SECTION_HEADER = "CONTEXT FOR THIS TURN"
NO_CONTEXT_STATEMENT = (
    "No saved profile or knowledge applies to this turn; answer from the message alone."
)

# What the user is told when the provider could not be reached. Stored as a normal
# assistant message, so it must read as an apology and not as an error trace.
FALLBACK_RESPONSE = (
    "I could not reach the model just now, so I have not answered that. "
    "Your message is saved - check that the inference provider is running and ask me again."
)
FALLBACK_PROVIDER_NAME = "fallback"

# Error messages may echo a response body; keep them bounded so a huge error page
# cannot be written to the log or returned to a client verbatim.
MAX_PROVIDER_ERROR_CHARS = 300

# Messages raised by the interface-only providers and by unusable configuration.
IMPLEMENTATION_PENDING_MESSAGE = (
    "{provider} is an allowed LLM_PROVIDER value, but its client is not implemented yet; "
    "set LLM_PROVIDER=ollama to get answers in this phase"
)
UNSUPPORTED_PROVIDER_MESSAGE = "LLM_PROVIDER must be one of {providers}, got {provider!r}"
MISSING_MODEL_MESSAGE = "{provider} needs a model name; check its *_MODEL setting"
MISSING_BASE_URL_MESSAGE = "{provider} needs a base URL; check its *_BASE_URL setting"
MISSING_API_KEY_MESSAGE = "{provider} needs an API key; check its *_API_KEY setting"


@dataclass(frozen=True)
class InferenceConfig:
    """Everything needed to talk to the selected provider for one process.

    ``api_key`` holds a secret and is therefore never logged and never included in
    an exception message.
    """

    provider: str = OLLAMA_PROVIDER
    model: str = DEFAULT_OLLAMA_MODEL
    base_url: str = DEFAULT_OLLAMA_BASE_URL
    api_key: str | None = None
    request_timeout_seconds: float = DEFAULT_LLM_REQUEST_TIMEOUT_SECONDS
    max_output_tokens: int = DEFAULT_LLM_MAX_OUTPUT_TOKENS
    temperature: float = DEFAULT_LLM_TEMPERATURE
    fallback_enabled: bool = True
    system_prompt: str = DEFAULT_SYSTEM_PROMPT

    def __post_init__(self) -> None:
        if self.provider not in SUPPORTED_LLM_PROVIDERS:
            raise ValueError(
                UNSUPPORTED_PROVIDER_MESSAGE.format(providers=SUPPORTED_LLM_PROVIDERS, provider=self.provider)
            )
        if not self.model.strip():
            raise ValueError(MISSING_MODEL_MESSAGE.format(provider=self.provider))
        if not self.base_url.strip():
            raise ValueError(MISSING_BASE_URL_MESSAGE.format(provider=self.provider))
        if self.request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be greater than 0")
        if self.max_output_tokens < 1:
            raise ValueError("max_output_tokens must be at least 1")
        if not MIN_TEMPERATURE <= self.temperature <= MAX_TEMPERATURE:
            raise ValueError(f"temperature must be between {MIN_TEMPERATURE} and {MAX_TEMPERATURE}")

    @classmethod
    def from_settings(cls, settings: Settings) -> "InferenceConfig":
        """Select the model, endpoint and credential belonging to ``LLM_PROVIDER``."""
        model, base_url, api_key = _provider_settings(settings)
        return cls(
            provider=settings.llm_provider,
            model=model,
            base_url=base_url,
            api_key=api_key,
            request_timeout_seconds=settings.llm_request_timeout_seconds,
            max_output_tokens=settings.llm_max_output_tokens,
            temperature=settings.llm_temperature,
            fallback_enabled=settings.llm_fallback_enabled,
        )


def _provider_settings(settings: Settings) -> tuple[str, str, str | None]:
    """Resolve ``(model, base_url, api_key)`` for the selected provider.

    Ollama is served locally and is authenticated by being on the machine, so it
    deliberately has no credential and reports ``None``.
    """
    if settings.llm_provider == OLLAMA_PROVIDER:
        return settings.ollama_model, settings.ollama_base_url, None
    if settings.llm_provider == OPENAI_PROVIDER:
        return settings.openai_model, settings.openai_base_url, _secret_value(settings.openai_api_key)
    if settings.llm_provider == ANTHROPIC_PROVIDER:
        return settings.anthropic_model, settings.anthropic_base_url, _secret_value(settings.anthropic_api_key)
    raise ValueError(
        UNSUPPORTED_PROVIDER_MESSAGE.format(providers=SUPPORTED_LLM_PROVIDERS, provider=settings.llm_provider)
    )


def _secret_value(secret: SecretStr | None) -> str | None:
    """Unwrap a secret setting, treating an empty environment value as absent."""
    if secret is None:
        return None
    value = secret.get_secret_value()
    return value if value.strip() else None

