import pytest

pytest.importorskip("langchain_text_splitters")

from ragsys.ingest.chunking import chunk_recursive, chunk_structure
from ragsys.ingest.loaders import RawDocument
from ragsys.ingest.structure import parse_sections
from tests.test_structure import TEXT


def test_structure_chunks_never_cross_sections_and_offsets_map_back():
    doc = RawDocument("d", "d.pdf", "Test Act", TEXT)
    sections = parse_sections(doc)
    for chunk in chunk_structure(doc, sections, chunk_size=120, overlap=0):
        md = chunk.metadata
        assert len(md["refs"]) == 1
        assert doc.text[md["start"] : md["end"]] == md["text"]


def test_recursive_chunks_carry_all_overlapping_refs():
    doc = RawDocument("d", "d.pdf", "Test Act", TEXT)
    chunks = chunk_recursive(doc, parse_sections(doc), chunk_size=400, overlap=0)
    assert any(len(c.metadata["refs"]) > 1 for c in chunks)