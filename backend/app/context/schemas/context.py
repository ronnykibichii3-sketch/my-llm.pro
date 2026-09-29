"""Models describing the context assembled for one chat turn.

These are internal types, not API responses: the assembled text is what reaches
the model, while the citations and the token estimate ride along for logging and
for a later step that stores or shows them (see app/context/README.md).
"""

from pydantic import BaseModel, Field


class ContextCitation(BaseModel):
    """One retrieved chunk that survived trimming and reached the prompt.

    ``document_id`` and ``chunk_id`` are the ids a future citation UI needs; the
    title and position are carried along so a citation stays readable after the
    chunk itself has scrolled out of the context window.
    """

    document_id: int
    chunk_id: int
    document_title: str
    position: int
    score: float


class ContextObject(BaseModel):
    """Everything the chat service is allowed to know about a user's context.

    ``combined_context`` is the single string meant for the system prompt; the
    per-source fields exist so callers and tests can assert where a sentence came
    from without parsing the combined block.
    """

    profile_context: str = ""
    knowledge_context: str = ""
    combined_context: str = ""
    citations: list[ContextCitation] = Field(default_factory=list)
    token_estimate: int = 0

    @property
    def is_empty(self) -> bool:
        """True when nothing was found, in which case the turn runs without context."""
        return not self.combined_context
