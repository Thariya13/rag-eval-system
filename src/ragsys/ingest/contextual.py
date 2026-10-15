from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from langchain_core.documents import Document

from ragsys.config import Settings
from ragsys.ingest.loaders import RawDocument
from ragsys.models import get_chat_model, message_text

logger = logging.getLogger(__name__)
WINDOW_CHARS = 8000

PROMPT = """<document title="{title}">
{window}
</document>

Here is a chunk taken from the document above:
<chunk>
{chunk}
</chunk>

Write one or two sentences that situate this chunk within the document to improve search retrieval.
Mention the article or annex it belongs to, its topic, and who it applies to where relevant.
Reply with the context only."""


def _cache_path(cache_dir: Path, key: str) -> Path:
    return cache_dir / key[:2] / f"{key}.txt"


def contextualize(chunks: list[Document], doc: RawDocument, settings: Settings) -> list[Document]:
    cache_dir = settings.cache_dir / "contextual"
    llm = get_chat_model(settings.contextualizer_model, settings.llm_temperature)

    contexts: list[str | None] = []
    pending: list[tuple[int, str, str]] = []
    for i, chunk in enumerate(chunks):
        md = chunk.metadata
        window = doc.text[max(0, md["start"] - WINDOW_CHARS) : md["end"] + WINDOW_CHARS]
        prompt = PROMPT.format(title=doc.title, window=window, chunk=md["text"])
        key = hashlib.sha256(f"{settings.contextualizer_model}\n{prompt}".encode()).hexdigest()
        path = _cache_path(cache_dir, key)
        if path.exists():
            contexts.append(path.read_text(encoding="utf-8"))
        else:
            contexts.append(None)
            pending.append((i, key, prompt))

    logger.info("Contextualising %d chunks (%d cached)", len(pending), len(chunks) - len(pending))
    if pending:
        results = llm.batch(
            [p for _, _, p in pending],
            config={"max_concurrency": settings.llm_max_concurrency},
            return_exceptions=True,
        )
        for (i, key, _), result in zip(pending, results):
            if isinstance(result, Exception):
                logger.warning("Context generation failed for chunk %d: %s", i, result)
                continue
            text = message_text(result).strip()
            path = _cache_path(cache_dir, key)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
            contexts[i] = text

    enriched = []
    for chunk, context in zip(chunks, contexts):
        if not context:
            enriched.append(chunk)
            continue
        md = {**chunk.metadata, "context": context}
        enriched.append(Document(page_content=f"{context}\n\n{chunk.page_content}", metadata=md))
    return enriched