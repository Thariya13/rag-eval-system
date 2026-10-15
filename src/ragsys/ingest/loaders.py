from __future__ import annotations

import bisect
import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".pdf", ".md", ".txt"}

BOILERPLATE_PATTERNS = [
    re.compile(r"^\s*EN\s*$"),
    re.compile(r"^\s*OJ L,?\s.*$"),
    re.compile(r"^\s*ELI:\s*http\S+.*$"),
    re.compile(r"^\s*\d{1,4}/\d{1,4}\s*$"),
]

PAGE_SEPARATOR = "\n\n"


@dataclass
class RawDocument:
    doc_id: str
    source: str
    title: str
    text: str
    page_starts: list[int] = field(default_factory=lambda: [0])

    def page_at(self, offset: int) -> int:
        return max(1, bisect.bisect_right(self.page_starts, offset))


def clean_page(text: str) -> str:
    text = re.sub(r"(\w)-\n(?=[a-z])", r"\1", text)
    lines = [line.rstrip() for line in text.splitlines()]
    lines = [line for line in lines if not any(p.match(line) for p in BOILERPLATE_PATTERNS)]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _doc_id(path: Path, text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", path.stem.lower()).strip("-")[:40]
    return f"{slug}-{hashlib.sha256(text.encode()).hexdigest()[:8]}"


def _join_pages(pages: list[str]) -> tuple[str, list[int]]:
    starts, offset = [], 0
    for page in pages:
        starts.append(offset)
        offset += len(page) + len(PAGE_SEPARATOR)
    return PAGE_SEPARATOR.join(pages), starts


def load_pdf(path: Path) -> RawDocument:
    import pymupdf

    with pymupdf.open(path) as pdf:
        pages = [clean_page(page.get_text("text")) for page in pdf]
        title = ((pdf.metadata or {}).get("title") or "").strip() or path.stem
    text, starts = _join_pages(pages)
    return RawDocument(doc_id=_doc_id(path, text), source=path.name, title=title, text=text, page_starts=starts)


def load_text(path: Path) -> RawDocument:
    text = clean_page(path.read_text(encoding="utf-8"))
    return RawDocument(doc_id=_doc_id(path, text), source=path.name, title=path.stem, text=text)


def load_corpus(directory: Path) -> list[RawDocument]:
    if not directory.exists():
        raise FileNotFoundError(f"Data directory {directory} does not exist")
    docs = []
    for path in sorted(directory.rglob("*")):
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        doc = load_pdf(path) if path.suffix.lower() == ".pdf" else load_text(path)
        if not doc.text.strip():
            logger.warning("Skipping %s: no extractable text (scanned PDF? add OCR)", path)
            continue
        logger.info("Loaded %s (%d chars, %d pages)", path.name, len(doc.text), len(doc.page_starts))
        docs.append(doc)
    return docs