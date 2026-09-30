"""Anthropic provider - interface only in this phase.

Mirrors ``OpenAIProvider``: it validates the ``ANTHROPIC_*`` settings now so the
implementation later only adds the HTTP call. Its ``generate`` raises
``ProviderConfigurationError`` because unimplemented is not the same as broken.

The Messages API takes the composed system message as its own ``system``
parameter, so ``compose_system_message`` output is used unchanged and ``context``
needs no separate slot here either.
"""

from app.core.config import ANTHROPIC_PROVIDER
from app.inference.config import (
    DEFAULT_LLM_MAX_OUTPUT_TOKENS,
    DEFAULT_LLM_REQUEST_TIMEOUT_SECONDS,
    DEFAULT_LLM_TEMPERATURE,
    IMPLEMENTATION_PENDING_MESSAGE,
    MISSING_API_KEY_MESSAGE,
    MISSING_BASE_URL_MESSAGE,
    MISSING_MODEL_MESSAGE,
)
from app.inference.exceptions import ProviderConfigurationError
from app.inference.schemas.inference import LLMResponse


class AnthropicProvider:
    """Configured-but-unimplemented Messages backend."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None,
        request_timeout_seconds: float = DEFAULT_LLM_REQUEST_TIMEOUT_SECONDS,
        max_output_tokens: int = DEFAULT_LLM_MAX_OUTPUT_TOKENS,
        temperature: float = DEFAULT_LLM_TEMPERATURE,
    ) -> None:
        if not base_url.strip():
            raise ProviderConfigurationError(
                ANTHROPIC_PROVIDER, MISSING_BASE_URL_MESSAGE.format(provider=ANTHROPIC_PROVIDER)
            )
        if not model.strip():
            raise ProviderConfigurationError(ANTHROPIC_PROVIDER, MISSING_MODEL_MESSAGE.format(provider=ANTHROPIC_PROVIDER))
        if api_key is None or not api_key.strip():
            raise ProviderConfigurationError(ANTHROPIC_PROVIDER, MISSING_API_KEY_MESSAGE.format(provider=ANTHROPIC_PROVIDER))
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key
        self._request_timeout_seconds = request_timeout_seconds
        self._max_output_tokens = max_output_tokens
        self._temperature = temperature

    @property
    def name(self) -> str:
        return ANTHROPIC_PROVIDER

    @property
    def model(self) -> str:
        return self._model

    async def generate(self, system_prompt: str, user_prompt: str, context: str = "") -> LLMResponse:
        """Refuse the call: no client is implemented for this provider yet."""
        raise ProviderConfigurationError(
            ANTHROPIC_PROVIDER, IMPLEMENTATION_PENDING_MESSAGE.format(provider=ANTHROPIC_PROVIDER)
        )
