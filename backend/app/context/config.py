"""Prompt-context assembly configuration and limits.

Every budget, label and separator used while building the block of text handed
to the model lives here, so the builder, the environment settings and the tests
cannot drift apart (RULES.md Rule 7: no magic strings or numbers).

The budgets are nested on purpose: ``profile_max_chars`` and
``knowledge_max_chars`` bound their own section, while ``max_chars`` bounds the
combined block that actually reaches the model. The builder spends whatever the
profile section leaves over on knowledge, so a user with no profile still gets
the full knowledge budget and a user with a long profile loses retrieval rather
than seeing a truncated profile.
"""

from dataclasses import dataclass

from app.core.config import Settings

DEFAULT_CONTEXT_MAX_CHARS = 6000
DEFAULT_PROFILE_MAX_CHARS = 1500
DEFAULT_KNOWLEDGE_MAX_CHARS = 4500
DEFAULT_KNOWLEDGE_TOP_K = 5

# A rough character-to-token ratio is enough for budget bookkeeping and logs.
# A real tokenizer would add a heavyweight dependency for a number nobody acts
# on, and the estimate only ever needs to be comparable between turns.
CHARS_PER_TOKEN_ESTIMATE = 4

# The preamble tells the model what the block is and how to use it. It is part
# of the budget, never counted on top of it.
CONTEXT_PREAMBLE = (
    "The context below was assembled automatically for this user and may be empty or out of date. "
    "Use it when it is relevant; never quote it verbatim."
)
PROFILE_SECTION_HEADER = "USER PROFILE"
KNOWLEDGE_SECTION_HEADER = "RELEVANT KNOWLEDGE"

# Profile attributes rendered into the profile section, in the order they are
# listed. ``summary`` carries the durable free-text facts about the user; the
# profile model has no separate "facts" column.
PROFILE_SUMMARY_LABEL = "Summary"
PROFILE_PREFERENCES_LABEL = "Preferences"
PROFILE_INTERESTS_LABEL = "Interests"
PROFILE_GOALS_LABEL = "Goals"
PROFILE_TIMEZONE_LABEL = "Timezone"
PROFILE_LANGUAGE_LABEL = "Preferred language"

# Each retrieved excerpt is headed by its rank, source document and position so
# the model can attribute a claim, while the machine-readable citation ids stay
# in the ContextObject and out of the prompt.
EXCERPT_HEADER_TEMPLATE = "[{index}] {title} (chunk {position})"

SECTION_SEPARATOR = "\n\n"
LINE_SEPARATOR = "\n"
LABEL_SEPARATOR = ": "
LIST_SEPARATOR = ", "
PREFERENCE_SEPARATOR = "; "
PREFERENCE_VALUE_SEPARATOR = "="
EXCERPT_SEPARATOR = "\n\n"
TRUNCATION_SUFFIX = " [...]"

MIN_TRUNCATION_BUDGET = 1


@dataclass(frozen=True)
class ContextConfig:
    """Tunable budgets for one context build, validated at construction time."""

    max_chars: int = DEFAULT_CONTEXT_MAX_CHARS
    profile_max_chars: int = DEFAULT_PROFILE_MAX_CHARS
    knowledge_max_chars: int = DEFAULT_KNOWLEDGE_MAX_CHARS
    knowledge_top_k: int = DEFAULT_KNOWLEDGE_TOP_K

    def __post_init__(self) -> None:
        if self.max_chars < MIN_TRUNCATION_BUDGET:
            raise ValueError("max_chars must be at least 1")
        if self.profile_max_chars < MIN_TRUNCATION_BUDGET:
            raise ValueError("profile_max_chars must be at least 1")
        if self.knowledge_max_chars < MIN_TRUNCATION_BUDGET:
            raise ValueError("knowledge_max_chars must be at least 1")
        if self.knowledge_top_k < MIN_TRUNCATION_BUDGET:
            raise ValueError("knowledge_top_k must be at least 1")

    @classmethod
    def from_settings(cls, settings: Settings) -> "ContextConfig":
        return cls(
            max_chars=settings.context_max_chars,
            profile_max_chars=settings.profile_max_chars,
            knowledge_max_chars=settings.knowledge_max_chars,
            knowledge_top_k=settings.knowledge_top_k,
        )
