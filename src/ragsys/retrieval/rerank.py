from __future__ import annotations

import logging
import threading
from typing import Protocol

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class Reranker(Protocol):
    def rerank(self, query: str, docs: list[Document], top_k: int) -> list[Document]: ...


class NoopReranker:
    def rerank(self, query: str, docs: list[Document], top_k: int) -> list[Document]:
        return docs[:top_k]


class CrossEncoderReranker:
    def __init__(self, model_name: str, max_length: int = 512):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ImportError("Cross-encoder reranking needs: pip install 'ragsys[rerank]'") from exc
        logger.info("Loading cross-encoder %s", model_name)
        self._model = CrossEncoder(model_name, max_length=max_length)
        self._lock = threading.Lock()

    def rerank(self, query: str, docs: list[Document], top_k: int) -> list[Document]:
        if not docs:
            return []
        with self._lock:
            scores = self._model.predict([(query, d.page_content) for d in docs], batch_size=16, show_progress_bar=False)
        ranked = sorted(zip(docs, scores), key=lambda pair: float(pair[1]), reverse=True)[:top_k]
        return [Document(page_content=d.page_content, metadata={**d.metadata, "rerank_score": float(s)}) for d, s in ranked]


class CohereReranker:
    def __init__(self, model_name: str):
        try:
            from langchain_cohere import CohereRerank
        except ImportError as exc:
            raise ImportError("Cohere reranking needs: pip install 'ragsys[cohere]'") from exc
        self._cls, self._model_name = CohereRerank, model_name

    def rerank(self, query: str, docs: list[Document], top_k: int) -> list[Document]:
        if not docs:
            return []
        compressor = self._cls(model=self._model_name, top_n=top_k)
        out = []
        for d in compressor.compress_documents(docs, query):
            md = {**d.metadata, "rerank_score": float(d.metadata.get("relevance_score", 0.0))}
            out.append(Document(page_content=d.page_content, metadata=md))
        return out


_CACHE: dict[tuple[str, str], Reranker] = {}
_CACHE_LOCK = threading.Lock()


def get_reranker(kind: str, settings) -> Reranker:
    model = {"cross_encoder": settings.cross_encoder_model, "cohere": settings.cohere_rerank_model}.get(kind, "")
    with _CACHE_LOCK:
        if (kind, model) not in _CACHE:
            if kind == "none":
                _CACHE[(kind, model)] = NoopReranker()
            elif kind == "cross_encoder":
                _CACHE[(kind, model)] = CrossEncoderReranker(model)
            elif kind == "cohere":
                _CACHE[(kind, model)] = CohereReranker(model)
            else:
                raise ValueError(f"Unknown reranker '{kind}'")
        return _CACHE[(kind, model)]