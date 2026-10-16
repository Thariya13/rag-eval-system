from __future__ import annotations

from functools import lru_cache

from ragsys.config import PipelineConfig, Settings, collection_name
from ragsys.models import get_embeddings, get_sparse_embeddings


@lru_cache(maxsize=4)
def _client(url: str, api_key: str | None):
    from qdrant_client import QdrantClient

    return QdrantClient(url=url, api_key=api_key, timeout=30)


def get_qdrant_client(settings: Settings):
    return _client(settings.qdrant_url, settings.qdrant_api_key)


def get_vector_store(settings: Settings, cfg: PipelineConfig):
    from langchain_qdrant import QdrantVectorStore, RetrievalMode

    mode = {"dense": RetrievalMode.DENSE, "sparse": RetrievalMode.SPARSE, "hybrid": RetrievalMode.HYBRID}
    return QdrantVectorStore(
        client=get_qdrant_client(settings),
        collection_name=collection_name(settings, cfg),
        embedding=get_embeddings(settings.embedding_model),
        sparse_embedding=get_sparse_embeddings(settings.sparse_model),
        retrieval_mode=mode[cfg.retrieval_mode],
    )