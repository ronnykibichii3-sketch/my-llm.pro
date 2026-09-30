"""The contract every model backend implements, plus the shared measurements.

Keeping the contract a ``Protocol`` means a provider knows nothing about HTTP,
about the database, or about who is calling it: it receives the three parts of a
prompt and returns an ``LLMResponse``. Tests satisfy the same contract with a few
lines of Python, which is how the rest of the layer is exercised without a model.
"""

from time import perf_counter
from typing import Protocol, runtime_checkable

from app.inference.config import MAX_PROVIDER_ERROR_CHARS
from app.inference.schemas.inference import LLMResponse

DETAIL_SEPARATOR = " "


@runtime_checkable
class LLMProvider(Protocol):
    """A model backend that can answer one prompt.

    ``generate`` receives the parts a ``PromptBuilder`` composed and must not
    reassemble, add instructions to, or trim them: placement of the pieces into a
    provider-specific payload is the provider's only prompt-related job, and it
    does that through ``compose_system_message`` so the wording stays put.

    Implementations raise ``ProviderFailure`` (or a subclass) for anything that
    went wrong at call time so the service can fall back, and never return an
    empty completion.
    """

    @property
    def name(self) -> str:
        """Provider identifier as it appears in ``LLM_PROVIDER`` and in logs."""
        ...

    @property
    def model(self) -> str:
        """The model this provider was configured to call."""
        ...

    async def generate(self, system_prompt: str, user_prompt: str, context: str = "") -> LLMResponse:
        """Answer ``user_prompt`` given ``system_prompt`` and this turn's ``context``."""
        ...


def elapsed_ms(started_at: float) -> int:
    """Whole milliseconds since a ``perf_counter()`` reading."""
    return round((perf_counter() - started_at) * 1000)


def truncate_detail(text: str, limit: int = MAX_PROVIDER_ERROR_CHARS) -> str:
    """Collapse whitespace in ``text`` and clip it, for use inside error messages.

    Provider error bodies are untrusted and sometimes enormous; flattening them to
    one short line keeps them readable in a log line without letting one through
    verbatim.
    """
    flattened = DETAIL_SEPARATOR.join(text.split())
    if len(flattened) <= limit:
        return flattened
    return flattened[:limit] + "..."
