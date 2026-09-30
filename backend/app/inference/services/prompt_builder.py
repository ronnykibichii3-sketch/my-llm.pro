"""Compose the prompt for one chat turn - the only module that writes prompt text.

The order is fixed (see app/inference/README.md)::

    system prompt (behaviour instruction)
        +
    ContextObject.combined_context (profile + retrieved knowledge, may be empty)
        +
    user prompt

``ChatService`` never builds a string for the model, and neither does a provider:
the builder produces the parts, and ``compose_system_message`` - exported from
here - is the single place those parts are joined. That is what lets a provider
take ``context`` as its own argument without any module other than this one
deciding how the context is introduced to the model.
"""

from app.context.config import LINE_SEPARATOR, SECTION_SEPARATOR
from app.context.schemas.context import ContextObject
from app.inference.config import (
    CONTEXT_SECTION_HEADER,
    NO_CONTEXT_STATEMENT,
    InferenceConfig,
)
from app.inference.schemas.inference import Prompt


class PromptBuilder:
    """Turn a ``ContextObject`` plus the user's message into a ``Prompt``."""

    def __init__(self, config: InferenceConfig | None = None) -> None:
        self._config = config if config is not None else InferenceConfig()

    @property
    def config(self) -> InferenceConfig:
        """The configuration whose ``system_prompt`` this builder hands out."""
        return self._config

    def build_prompt(self, context: ContextObject, user_prompt: str) -> Prompt:
        """Split a turn into its three parts; an empty context is normal, not an error.

        ``user_prompt`` is stripped so a provider never sees stray whitespace, and a
        blank message is rejected here rather than burning a model call.
        """
        cleaned = user_prompt.strip()
        if not cleaned:
            raise ValueError("user_prompt must not be empty")
        return Prompt(
            system_prompt=self._config.system_prompt,
            user_prompt=cleaned,
            context=context.combined_context,
            context_size=len(context.combined_context),
            context_token_estimate=context.token_estimate,
            citation_count=len(context.citations),
        )


def compose_system_message(system_prompt: str, context: str) -> str:
    """The system message a provider sends: the instruction, then this turn's context.

    Called by providers instead of hand-built concatenations, so the wording stays
    in this module. With no context the model is told so explicitly - leaving the
    slot blank would leave a model expecting a context block wondering where it is.
    """
    if not context.strip():
        return system_prompt + SECTION_SEPARATOR + NO_CONTEXT_STATEMENT
    return system_prompt + SECTION_SEPARATOR + CONTEXT_SECTION_HEADER + LINE_SEPARATOR + context
