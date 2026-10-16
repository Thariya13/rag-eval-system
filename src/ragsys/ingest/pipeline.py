from __future__ import annotations

import logging
import uuid

from langchain_core.documents import Document

from ragsys.config import PipelineConfig, Settings, collection_name
from ragsys.ingest.chunking import chunk_recursive, chunk_semantic, chunk_structure
from ragsys.ingest.contextual import contextualize
from ragsys.ingest.loaders import load_corpus
from ragsys.ingest.structure import parse_sections
from ragsys.models import get_embeddings, get_sparse_embeddings
from ragsys.retrieval.store import get_qdrant_client

logger = logging.getLogger(__name__)

PAYLOAD_INDEXES = {
    "metadata.refs": "keyword",
    "metadata.chapter": "keyword",
    "metadata.doc_id": "keyword",
    "metadata.section_type": "keyword",
    "metadata.article_number": "integer",
}


def build_documents(settings: Settings, cfg: PipelineConfig) -> list[Document]:
    raw_docs = load_corpus(settings.raw_data_dir)
    if not raw_docs:
        raise FileNotFoundError(f"No supported documents in {settings.raw_data_dir}")
    out: list[Document] = []
    for doc in raw_docs:
        sections = parse_sections(doc)
        logger.info("%s: %d sections detected", doc.source, len(sections))
        if cfg.chunking == "recursive":
            chunks = chunk_recursive(doc, sections, settings.chunk_size, settings.chunk_overlap)
        elif cfg.chunking == "structure":
            chunks = chunk_structure(doc, sections, settings.chunk_size, settings.chunk_overlap)
        else:
            chunks = chunk_semantic(
                doc,
                sections,
                get_embeddings(settings.embedding_model),
                settings.chunk_size,
                settings.semantic_breakpoint_percentile,
            )
        if cfg.contextual:
            chunks = contextualize(chunks, doc, settings)
        out.extend(chunks)
    if not out:
        raise ValueError("Chunking produced no chunks - check the documents in data/raw")
    lengths = sorted(len(d.page_content) for d in out)
    logger.info("Built %d chunks (median %d chars, max %d)", len(out), lengths[len(lengths) // 2], lengths[-1])
    return out


def chunk_id(doc: Document) -> str:
    md = doc.metadata
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{md['doc_id']}:{md['chunking']}:{md['start']}:{md['end']}"))


def index_documents(settings: Settings, cfg: PipelineConfig, docs: list[Document]) -> str:
    from langchain_qdrant import QdrantVectorStore, RetrievalMode
    from qdrant_client import models

    name = collection_name(settings, cfg)
    QdrantVectorStore.from_documents(
        docs,
        embedding=get_embeddings(settings.embedding_model),
        sparse_embedding=get_sparse_embeddings(settings.sparse_model),
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        collection_name=name,
        retrieval_mode=RetrievalMode.HYBRID,
        ids=[chunk_id(d) for d in docs],
        force_recreate=True,
        batch_size=64,
    )
    client = get_qdrant_client(settings)
    schema = {"keyword": models.PayloadSchemaType.KEYWORD, "integer": models.PayloadSchemaType.INTEGER}
    for field, kind in PAYLOAD_INDEXES.items():
        client.create_payload_index(name, field_name=field, field_schema=schema[kind])
    logger.info("Indexed %d chunks into collection '%s'", len(docs), name)
    return name


def ensure_index(settings: Settings, cfg: PipelineConfig, force: bool = False) -> str:
    name = collection_name(settings, cfg)
    if force or not get_qdrant_client(settings).collection_exists(name):
        index_documents(settings, cfg, build_documents(settings, cfg))
    return name