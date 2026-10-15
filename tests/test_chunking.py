import numpy as np
import pytest

pytest.importorskip("langchain_text_splitters")

from ragsys.ingest.chunking import chunk_recursive, chunk_structure, semantic_groups, sentence_spans
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


def test_semantic_grouping_breaks_on_distance_and_size(monkeypatch):
    import ragsys.ingest.chunking as chunking

    text = "Alpha one. Alpha two. Beta one. Beta two."
    spans = sentence_spans(text)
    assert len(spans) == 4
    distances = np.array([0.1, 0.9, 0.1])
    assert semantic_groups(distances, spans, threshold=0.5, chunk_size=10_000) == [(0, len(text))]
    monkeypatch.setattr(chunking, "MIN_SEMANTIC_CHARS", 0)
    groups = chunking.semantic_groups(distances, spans, threshold=0.5, chunk_size=10_000)
    assert [text[s:e] for s, e in groups] == ["Alpha one. Alpha two.", "Beta one. Beta two."]
    assert len(chunking.semantic_groups(np.zeros(3), spans, threshold=0.5, chunk_size=15)) == 4