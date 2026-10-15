from __future__ import annotations

import re

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ragsys.ingest.loaders import RawDocument
from ragsys.ingest.structure import Section, sections_overlapping

LEGAL_SEPARATORS = [
    r"\n(?=\d{1,2}\.\s)",
    r"\n(?=\([a-z0-9]{1,4}\)\s)",
    r"\n\n",
    r"\n",
    r"(?<=[.;])\s+",
    r"\s+",
]
SENTENCE_BOUNDARY = re.compile(r"(?<=[.;:!?])\s+(?=[A-Z(\d\"'])|\n\s*\n")
MIN_SEMANTIC_CHARS = 250


def _metadata(doc: RawDocument, sections: list[Section], start: int, end: int, strategy: str, index: int) -> dict:
    overlapping = sections_overlapping(sections, start, end)
    primary = overlapping[0] if overlapping else None
    return {
        "doc_id": doc.doc_id,
        "source": doc.source,
        "doc_title": doc.title,
        "chunking": strategy,
        "chunk_index": index,
        "start": start,
        "end": end,
        "page": doc.page_at(start),
        "ref": primary.ref if primary else "Other",
        "refs": [s.ref for s in overlapping],
        "title": primary.title if primary else "",
        "section_type": primary.section_type if primary else "other",
        "chapter": primary.chapter if primary else None,
        "article_number": primary.article_number if primary else None,
    }


def _with_header(text: str, md: dict) -> str:
    if md["section_type"] in {"article", "annex"} and not text.startswith(md["ref"]):
        return f"{md['ref']} - {md['title']}\n{text}"
    return text


def chunk_recursive(doc: RawDocument, sections: list[Section], chunk_size: int, overlap: int) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap, add_start_index=True)
    out = []
    for i, piece in enumerate(splitter.create_documents([doc.text])):
        start = max(int(piece.metadata.get("start_index", 0)), 0)
        md = _metadata(doc, sections, start, start + len(piece.page_content), "recursive", i)
        md["text"] = piece.page_content
        out.append(Document(page_content=piece.page_content, metadata=md))
    return out


def chunk_structure(doc: RawDocument, sections: list[Section], chunk_size: int, overlap: int) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=LEGAL_SEPARATORS,
        is_separator_regex=True,
        add_start_index=True,
    )
    out: list[Document] = []
    for section in sections:
        section_text = doc.text[section.start : section.end]
        stripped = section_text.strip()
        if not stripped:
            continue
        if len(stripped) <= int(chunk_size * 1.25):
            pieces = [(section_text.find(stripped), stripped)]
        else:
            pieces = [
                (max(int(p.metadata.get("start_index", 0)), 0), p.page_content)
                for p in splitter.create_documents([section_text])
            ]
        for local_start, piece in pieces:
            start = section.start + local_start
            md = _metadata(doc, [section], start, start + len(piece), "structure", len(out))
            md["text"] = piece
            out.append(Document(page_content=_with_header(piece, md), metadata=md))
    return out


def sentence_spans(text: str) -> list[tuple[int, int]]:
    spans, pos = [], 0
    for match in SENTENCE_BOUNDARY.finditer(text):
        if match.start() > pos:
            spans.append((pos, match.start()))
        pos = match.end()
    if pos < len(text):
        spans.append((pos, len(text)))
    return [(s, e) for s, e in spans if text[s:e].strip()]


def semantic_groups(distances: np.ndarray, spans: list[tuple[int, int]], threshold: float, chunk_size: int):
    groups = []
    cur_start, cur_end = spans[0]
    for i in range(1, len(spans)):
        s, e = spans[i]
        is_break = distances[i - 1] > threshold and (cur_end - cur_start) >= MIN_SEMANTIC_CHARS
        too_big = (e - cur_start) > chunk_size
        if is_break or too_big:
            groups.append((cur_start, cur_end))
            cur_start = s
        cur_end = e
    groups.append((cur_start, cur_end))
    return groups


def chunk_semantic(
    doc: RawDocument,
    sections: list[Section],
    embeddings: Embeddings,
    chunk_size: int,
    percentile: float,
    batch_size: int = 256,
) -> list[Document]:
    text = doc.text
    spans = sentence_spans(text)
    if not spans:
        return []
    windows = [text[spans[max(i - 1, 0)][0] : spans[min(i + 1, len(spans) - 1)][1]] for i in range(len(spans))]
    vectors: list[list[float]] = []
    for b in range(0, len(windows), batch_size):
        vectors.extend(embeddings.embed_documents(windows[b : b + batch_size]))
    matrix = np.asarray(vectors, dtype=np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12
    distances = 1.0 - np.sum(matrix[:-1] * matrix[1:], axis=1)
    threshold = float(np.percentile(distances, percentile)) if len(distances) else 1.0

    out = []
    for i, (start, end) in enumerate(semantic_groups(distances, spans, threshold, chunk_size)):
        piece = text[start:end]
        md = _metadata(doc, sections, start, end, "semantic", i)
        md["text"] = piece
        out.append(Document(page_content=_with_header(piece, md), metadata=md))
    return out