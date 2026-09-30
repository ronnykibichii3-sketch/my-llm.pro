"""OpenAI provider - interface only in this phase.

The class satisfies ``LLMProvider`` and validates the ``OPENAI_*`` settings the day
its client lands, so configuration is already settled and tested. ``generate``
raises ``ProviderConfigurationError`` instead of pretending to answer: selecting
``LLM_PROVIDER=openai`` today produces one clear 502 rather than a silent stub.

When it is implemented, the composed system message becomes the ``system`` chat
message, ``context`` is already inside it (``compose_system_message``), and the
reply maps onto ``LLMResponse`` exactly as the Ollama provider does.
"""

from app.core.config import OPENAI_PROVIDER
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


class OpenAIProvider:
    """Configured-but-unimplemented Chat Completions backend."""

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
            raise ProviderConfigurationError(OPENAI_PROVIDER, MISSING_BASE_URL_MESSAGE.format(provider=OPENAI_PROVIDER))
        if not model.strip():
            raise ProviderConfigurationError(OPENAI_PROVIDER, MISSING_MODEL_MESSAGE.format(provider=OPENAI_PROVIDER))
        if api_key is None or not api_key.strip():
            raise ProviderConfigurationError(OPENAI_PROVIDER, MISSING_API_KEY_MESSAGE.format(provider=OPENAI_PROVIDER))
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key
        self._request_timeout_seconds = request_timeout_seconds
        self._max_output_tokens = max_output_tokens
        self._temperature = temperature

    @property
    def name(self) -> str:
        return OPENAI_PROVIDER

    @property
    def model(self) -> str:
        return self._model

    async def generate(self, system_prompt: str, user_prompt: str, context: str = "") -> LLMResponse:
        """Refuse the call: no client is implemented for this provider yet.

        The credential is deliberately absent from the message - a secret never
        reaches a log or a response body.
        """
        raise ProviderConfigurationError(
            OPENAI_PROVIDER, IMPLEMENTATION_PENDING_MESSAGE.format(provider=OPENAI_PROVIDER)
        )
