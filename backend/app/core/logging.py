"""Centralized structured application logging."""

import logging
import sys

from app.core.config import get_settings


def configure_logging() -> None:
    """Configure the root logger once for application processes."""
    settings = get_settings()
    logging.basicConfig(level=settings.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s %(message)s", stream=sys.stdout, force=True)


logger = logging.getLogger("myllm")
