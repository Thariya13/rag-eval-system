from __future__ import annotations

import logging
import os
import sys

logger = logging.getLogger(__name__)


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        stream=sys.stdout,
        force=True,
    )
    for noisy in ("httpx", "httpcore", "urllib3", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def langfuse_configured() -> bool:
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))


def get_callbacks(settings) -> list:
    if not (settings.langfuse_enabled and langfuse_configured()):
        return []
    try:
        from langfuse.langchain import CallbackHandler
    except ImportError:
        logger.warning("Langfuse keys set but the langfuse package is missing; tracing disabled")
        return []
    return [CallbackHandler()]


def flush() -> None:
    if not langfuse_configured():
        return
    try:
        from langfuse import get_client

        get_client().flush()
    except Exception:
        logger.exception("Failed to flush Langfuse")