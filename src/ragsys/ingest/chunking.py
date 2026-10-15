from __future__ import annotations

from langchain_core.documents import Document
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