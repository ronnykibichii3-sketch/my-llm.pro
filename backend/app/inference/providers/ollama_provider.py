"""Ollama provider: the one implemented backend in this phase.

Ollama exposes a chat endpoint at ``{OLLAMA_BASE_URL}/api/chat``. Everything that
varies between machines - host, port, model, sampling, timeout - arrives through
``InferenceConfig``, so this file contains no model name and no host (see
app/inference/README.md for what each setting does).

A fresh ``httpx.AsyncClient`` is built per call rather than kept on the instance.
That costs a connection setup on a loopback request, and buys a provider that is
stateless, safe to share across event loops and free of any client that has to be
closed on shutdown.
"""

from time import perf_counter
from typing import Any

import httpx

# Reused rather than reimplemented (RULES.md Rule 4): the context builder already
# publishes the character-based token estimate, so the estimates logged by the two
# layers are measured with the same ruler and can be compared turn by turn.
from app.context.services.context_builder_service import estimate_tokens
from app.core.config import OLLAMA_PROVIDER
from app.inference.config import (
    DEFAULT_LLM_MAX_OUTPUT_TOKENS,
    DEFAULT_LLM_REQUEST_TIMEOUT_SECONDS,
    DEFAULT_LLM_TEMPERATURE,
    OLLAMA_CHAT_PATH,
    SYSTEM_ROLE,
    USER_ROLE,
)
from app.inference.exceptions import ProviderResponseError, ProviderTransportError
from app.inference.providers.base import elapsed_ms, truncate_detail
from app.inference.schemas.inference import LLMResponse
from app.inference.services.prompt_builder import compose_system_message

# Request and response keys, spelled once.
MODEL_FIELD = "model"
MESSAGES_FIELD = "messages"
ROLE_FIELD = "role"
CONTENT_FIELD = "content"
STREAM_FIELD = "stream"
OPTIONS_FIELD = "options"
TEMPERATURE_OPTION = "temperature"
NUM_PREDICT_OPTION = "num_predict"
MESSAGE_FIELD = "message"
PROMPT_EVAL_COUNT_FIELD = "prompt_eval_count"
EVAL_COUNT_FIELD = "eval_count"

HTTP_ERROR_STATUS = 400


class OllamaProvider:
    """Chat completions served by an Ollama daemon, local or remote."""

    def __init__(
        self,
        base_url: str,
        model: str,
        request_timeout_seconds: float = DEFAULT_LLM_REQUEST_TIMEOUT_SECONDS,
        max_output_tokens: int = DEFAULT_LLM_MAX_OUTPUT_TOKENS,
        temperature: float = DEFAULT_LLM_TEMPERATURE,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._request_timeout_seconds = request_timeout_seconds
        self._max_output_tokens = max_output_tokens
        self._temperature = temperature

    @property
    def name(self) -> str:
        return OLLAMA_PROVIDER

    @property
    def model(self) -> str:
        return self._model

    async def generate(self, system_prompt: str, user_prompt: str, context: str = "") -> LLMResponse:
        """Ask the configured model for one non-streaming completion."""
        system_message = compose_system_message(system_prompt, context)
        payload = {
            MODEL_FIELD: self._model,
            MESSAGES_FIELD: [
                {ROLE_FIELD: SYSTEM_ROLE, CONTENT_FIELD: system_message},
                {ROLE_FIELD: USER_ROLE, CONTENT_FIELD: user_prompt},
            ],
            STREAM_FIELD: False,
            OPTIONS_FIELD: {
                TEMPERATURE_OPTION: self._temperature,
                NUM_PREDICT_OPTION: self._max_output_tokens,
            },
        }
        started_at = perf_counter()
        body = await self._post_chat(payload)
        latency_ms = elapsed_ms(started_at)
        content = _extract_content(body)
        return LLMResponse(
            content=content,
            model=_reported_model(body) or self._model,
            provider=OLLAMA_PROVIDER,
            input_tokens_estimate=_token_count(body, PROMPT_EVAL_COUNT_FIELD, system_message + user_prompt),
            output_tokens_estimate=_token_count(body, EVAL_COUNT_FIELD, content),
            latency_ms=latency_ms,
        )

    async def _post_chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Send one chat request and return its decoded body, translating every failure.

        Transport problems become ``ProviderTransportError`` and anything the server
        answered wrongly becomes ``ProviderResponseError``. Neither message can carry
        a credential: Ollama has none, and the base URL is not secret.
        """
        try:
            async with httpx.AsyncClient(
                base_url=self._base_url, timeout=httpx.Timeout(self._request_timeout_seconds)
            ) as client:
                response = await client.post(OLLAMA_CHAT_PATH, json=payload)
        except httpx.TimeoutException as error:
            raise ProviderTransportError(
                f"ollama did not answer {OLLAMA_CHAT_PATH} within {self._request_timeout_seconds} seconds"
            ) from error
        except httpx.HTTPError as error:
            raise ProviderTransportError(
                f"ollama request to {self._base_url}{OLLAMA_CHAT_PATH} failed: {truncate_detail(str(error))}"
            ) from error

        if response.status_code >= HTTP_ERROR_STATUS:
            raise ProviderResponseError(
                f"ollama returned HTTP {response.status_code} for {OLLAMA_CHAT_PATH}: {truncate_detail(response.text)}"
            )
        try:
            body: Any = response.json()
        except ValueError as error:
            raise ProviderResponseError(f"ollama replied with invalid JSON: {truncate_detail(response.text)}") from error
        if not isinstance(body, dict):
            raise ProviderResponseError(f"ollama replied with a {type(body).__name__} instead of a JSON object")
        return body


def _extract_content(body: dict[str, Any]) -> str:
    """Pull the assistant turn out of an Ollama response, refusing an empty one.

    A model that answers with nothing - a clipped context, a refused prompt - is a
    failed completion rather than a blank reply, so the caller can fall back.
    """
    message = body.get(MESSAGE_FIELD)
    if not isinstance(message, dict):
        raise ProviderResponseError(f"ollama response has no {MESSAGE_FIELD} object: {truncate_detail(str(body))}")
    raw = message.get(CONTENT_FIELD)
    if not isinstance(raw, str):
        raise ProviderResponseError(f"ollama {MESSAGE_FIELD} has no text {CONTENT_FIELD}")
    content = raw.strip()
    if not content:
        raise ProviderResponseError("ollama returned an empty completion")
    return content


def _reported_model(body: dict[str, Any]) -> str:
    """The model the daemon says it served, or an empty string when it does not say."""
    reported = body.get(MODEL_FIELD)
    return reported.strip() if isinstance(reported, str) else ""


def _token_count(body: dict[str, Any], field: str, text: str) -> int:
    """Provider-reported token count for ``field``, or the character estimate of ``text``.

    Ollama reports ``prompt_eval_count`` and ``eval_count`` for most models but
    leaves them out (or reports zero) for others, so both paths have to work.
    """
    reported = body.get(field)
    if isinstance(reported, int) and not isinstance(reported, bool) and reported > 0:
        return reported
    return estimate_tokens(text)

