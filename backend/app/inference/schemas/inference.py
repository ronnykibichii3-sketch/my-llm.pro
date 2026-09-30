"""Types describing one model call and the reply it produced.

Both are internal hand-off types, not API responses: chat persists only the text
of ``LLMResponse.content`` today, while the counts alongside it exist so a turn
can be logged, compared and later metered without re-measuring anything.
"""

from pydantic import BaseModel, Field


class Prompt(BaseModel):
    """The three parts of one turn, composed by ``PromptBuilder`` and nothing else.

    ``system_prompt`` is the behaviour instruction, ``context`` is the block the
    context builder assembled for this turn (possibly empty) and ``user_prompt``
    is what the user typed. The two log-only fields ride along so the service can
    report the shape of a turn without reopening the ``ContextObject``.
    """

    system_prompt: str
    user_prompt: str
    context: str = ""
    context_size: int = 0
    context_token_estimate: int = 0
    citation_count: int = 0


class LLMResponse(BaseModel):
    """A completion plus the cost of producing it.

    ``input_tokens_estimate`` and ``output_tokens_estimate`` are the provider's own
    counts when it reports them (Ollama returns ``prompt_eval_count`` and
    ``eval_count``) and a character-based estimate of the same text otherwise, so
    the two are comparable between providers. ``latency_ms`` is the round trip of
    the completion call alone, excluding context assembly.
    """

    content: str
    model: str
    provider: str
    input_tokens_estimate: int = Field(ge=0)
    output_tokens_estimate: int = Field(ge=0)
    latency_ms: int = Field(ge=0)
