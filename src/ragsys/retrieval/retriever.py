from __future__ import annotations

from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from ragsys.retrieval.filters import build_filter


class TwoStageRetriever(BaseRetriever):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    store: Any
    reranker: Any
    fetch_k: int = 30
    top_k: int = 6
    filters: dict[str, Any] | None = None

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        results = self.store.similarity_search_with_score(query, k=self.fetch_k, filter=build_filter(self.filters))
        candidates = [
            Document(page_content=doc.page_content, metadata={**doc.metadata, "retrieval_score": float(score)})
            for doc, score in results
        ]
        return self.reranker.rerank(query, candidates, self.top_k)