from __future__ import annotations

import re
from dataclasses import dataclass

from ragsys.ingest.loaders import RawDocument

HEADING_RE = re.compile(r"^(CHAPTER|SECTION|Article|ANNEX)\s+([IVXLC]+|\d{1,3}[a-z]?)$")
MAX_TITLE_LEN = 160
_ROMAN = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}


@dataclass
class Section:
    ref: str
    section_type: str
    title: str
    start: int
    end: int
    chapter: str | None = None
    article_number: int | None = None


def roman_to_int(numeral: str) -> int:
    total, prev = 0, 0
    for ch in reversed(numeral.upper()):
        value = _ROMAN[ch]
        total = total - value if value < prev else total + value
        prev = max(prev, value)
    return total


def normalize_ref(ref: str) -> str:
    parts = ref.split()
    if len(parts) == 2 and parts[0].lower() == "article":
        return f"Article {parts[1]}"
    if len(parts) == 2 and parts[0].lower() == "annex":
        return f"Annex {parts[1].upper()}"
    return " ".join(parts)


def _lines_with_offsets(text: str) -> list[tuple[int, str]]:
    out, pos = [], 0
    for raw in text.splitlines(keepends=True):
        stripped = raw.strip()
        out.append((pos + (len(raw) - len(raw.lstrip())), stripped))
        pos += len(raw)
    return out


def _title_after(lines: list[tuple[int, str]], i: int) -> str | None:
    for j in range(i + 1, min(i + 3, len(lines))):
        candidate = lines[j][1]
        if not candidate:
            continue
        if len(candidate) <= MAX_TITLE_LEN and (candidate[0].isupper() or candidate[0].isdigit()):
            return candidate
        return None
    return None


def _find_markers(text: str) -> list[tuple[int, str, str, str]]:
    lines = _lines_with_offsets(text)
    markers: list[tuple[int, str, str, str]] = []
    last_article, last_annex = 0, 0
    for i, (start, line) in enumerate(lines):
        match = HEADING_RE.match(line)
        if not match:
            continue
        kind, ident = match.groups()
        title = _title_after(lines, i)
        if title is None:
            continue
        if kind == "Article":
            if not ident[0].isdigit():
                continue
            number = int(re.match(r"\d+", ident).group())
            if number <= last_article:
                continue
            last_article = number
        elif kind == "ANNEX":
            if not re.fullmatch(r"[IVXLC]+", ident):
                continue
            number = roman_to_int(ident)
            if number <= last_annex:
                continue
            last_annex = number
        markers.append((start, kind, ident, title))
    return markers


def parse_sections(doc: RawDocument) -> list[Section]:
    text = doc.text
    markers = _find_markers(text)
    if not any(kind in ("Article", "ANNEX") for _, kind, _, _ in markers):
        return [Section(ref=doc.title, section_type="body", title=doc.title, start=0, end=len(text))]

    sections: list[Section] = []
    if markers[0][0] > 0 and text[: markers[0][0]].strip():
        sections.append(Section("Recitals", "recital", "Preamble and recitals", 0, markers[0][0]))

    chapter: str | None = None
    for idx, (start, kind, ident, title) in enumerate(markers):
        end = markers[idx + 1][0] if idx + 1 < len(markers) else len(text)
        if kind == "CHAPTER":
            chapter = f"Chapter {ident}"
        elif kind == "Article":
            number = int(re.match(r"\d+", ident).group())
            sections.append(Section(f"Article {ident}", "article", title, start, end, chapter, number))
        elif kind == "ANNEX":
            chapter = None
            sections.append(Section(f"Annex {ident}", "annex", title, start, end))
    return sections


def sections_overlapping(sections: list[Section], start: int, end: int) -> list[Section]:
    return [s for s in sections if s.start < end and start < s.end]