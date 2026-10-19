from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig, RunnableLambda

from ragsys.config import PipelineConfig, Settings
from ragsys.generation.citations import resolve_citations, strip_invalid_markers
from ragsys.generation.prompts import RAG_PROMPT, CitedAnswer, format_context
from ragsys.models import get_chat_model
from ragsys.retrieval.rerank import get_reranker
from ragsys.retrieval.retriever import TwoStageRetriever
from ragsys.retrieval.store import get_vector_store

logger = logging.getLogger(__name__)
NO_CONTEXT_ANSWER = "I could not find relevant information in the indexed documents."

try:
    from langchain_core.callbacks import get_usage_metadata_callback
except ImportError:
    get_usage_metadata_callback = None


@dataclass
class RAGResult:
    question: str
    answer: str
    answerable: bool
    citations: list[dict]
    retrieved: list[Document]
    latency_ms: float
    usage: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def contexts(self) -> list[str]:
        return [d.page_content for d in self.retrieved]

    @property
    def total_tokens(self) -> int:
        return sum(int(u.get("total_tokens", 0)) for u in self.usage.values())


class RAGPipeline:
    def __init__(self, settings: Settings, cfg: PipelineConfig):
        self.settings, self.cfg = settings, cfg
        self.store = get_vector_store(settings, cfg)
        self.reranker = get_reranker(cfg.reranker, settings)
        self.llm = get_chat_model(settings.llm_model, settings.llm_temperature).with_structured_output(CitedAnswer)
        self._runnable = RunnableLambda(self._run).with_config(run_name="rag_pipeline")

    def retriever(self, filters: dict | None = None) -> TwoStageRetriever:
        return TwoStageRetriever(
            store=self.store, reranker=self.reranker, fetch_k=self.cfg.fetch_k, top_k=self.cfg.top_k, filters=filters
        )

    def retrieve(self, question: str, filters: dict | None = None, callbacks: list | None = None) -> list[Document]:
        return self.retriever(filters).invoke(question, config={"callbacks": callbacks or []})

    def _run(self, inputs: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
        docs = self.retriever(inputs.get("filters")).invoke(inputs["question"], config=config)
        if not docs:
            return {"docs": [], "answer": None}
        messages = RAG_PROMPT.format_messages(question=inputs["question"], context=format_context(docs))
        return {"docs": docs, "answer": self.llm.invoke(messages, config=config)}

    def answer(
        self,
        question: str,
        filters: dict | None = None,
        *,
        callbacks: list | None = None,
        metadata: dict | None = None,
    ) -> RAGResult:
        start = time.perf_counter()
        config: RunnableConfig = {
            "callbacks": callbacks or [],
            "metadata": {"pipeline": self.cfg.name, "langfuse_tags": [self.cfg.name], **(metadata or {})},
        }
        usage: dict = {}
        if get_usage_metadata_callback is not None:
            with get_usage_metadata_callback() as usage_cb:
                out = self._runnable.invoke({"question": question, "filters": filters}, config=config)
            usage = dict(usage_cb.usage_metadata)
        else:
            out = self._runnable.invoke({"question": question, "filters": filters}, config=config)
        latency_ms = (time.perf_counter() - start) * 1000
        return self._to_result(question, out["docs"], out["answer"], latency_ms, usage)

    @staticmethod
    def _to_result(question, docs, parsed: CitedAnswer | None, latency_ms, usage) -> RAGResult:
        if not docs:
            return RAGResult(question, NO_CONTEXT_ANSWER, False, [], [], latency_ms, usage, ["no_documents_retrieved"])
        if parsed is None:
            return RAGResult(
                question, NO_CONTEXT_ANSWER, False, [], docs, latency_ms, usage, ["structured_output_failed"]
            )
        warnings: list[str] = []
        ids = resolve_citations(parsed.answer, parsed.cited_sources, len(docs))
        answer = strip_invalid_markers(parsed.answer, len(docs))
        if parsed.answerable and not ids:
            warnings.append("answer_without_citations")
        citations = []
        for i in ids:
            md = docs[i - 1].metadata
            citations.append(
                {
                    "id": i,
                    "ref": md.get("ref", ""),
                    "refs": md.get("refs", []),
                    "title": md.get("title", ""),
                    "page": md.get("page"),
                    "source": md.get("source", ""),
                    "snippet": (md.get("text") or docs[i - 1].page_content)[:400],
                    "score": md.get("rerank_score", md.get("retrieval_score")),
                }
            )
        return RAGResult(question, answer, parsed.answerable, citations, docs, latency_ms, usage, warnings)