"""User profile module configuration and limits.

Every bound enforced by the profile API lives here rather than inline, so the
storage column widths, the request validation and the merge rules cannot drift
apart (RULES.md Rule 7: no magic strings or numbers).
"""

MAX_DISPLAY_NAME_LENGTH = 120
MAX_SUMMARY_LENGTH = 4000
MAX_TIMEZONE_LENGTH = 64
MAX_LANGUAGE_LENGTH = 32
MAX_TAG_LENGTH = 200
MAX_ITEMS_PER_LIST = 100
MAX_PREFERENCE_KEYS = 100
MAX_PREFERENCE_KEY_LENGTH = 80
MAX_PREFERENCE_VALUE_LENGTH = 500
