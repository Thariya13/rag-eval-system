from ragsys.ingest.loaders import RawDocument, clean_page
from ragsys.ingest.structure import normalize_ref, parse_sections, roman_to_int

TEXT = """Whereas:
(1) The purpose of this Regulation is to improve the functioning of the internal market.
HAVE ADOPTED THIS REGULATION:
CHAPTER I
GENERAL PROVISIONS
Article 1
Subject matter
1. The purpose of this Regulation is to improve the internal market.
Article 2
Scope
This Regulation applies to providers as referred to in
Article 1
of this Regulation, and to deployers.
CHAPTER II
PROHIBITED AI PRACTICES
Article 5
Prohibited AI practices
1. The following AI practices shall be prohibited:
(a) the placing on the market of manipulative systems;
ANNEX I
List of Union harmonisation legislation
Section A. Machinery.
ANNEX III
High-risk AI systems referred to in Article 6(2)
1. Biometrics.
"""


def _doc(text=TEXT):
    return RawDocument(doc_id="d", source="d.pdf", title="Test Act", text=text)


def test_sections_detected_in_order():
    refs = [s.ref for s in parse_sections(_doc())]
    assert refs == ["Recitals", "Article 1", "Article 2", "Article 5", "Annex I", "Annex III"]


def test_inline_cross_reference_is_not_a_heading():
    article_2 = next(s for s in parse_sections(_doc()) if s.ref == "Article 2")
    assert "of this Regulation, and to deployers" in TEXT[article_2.start : article_2.end]


def test_chapter_assignment_and_article_numbers():
    sections = {s.ref: s for s in parse_sections(_doc())}
    assert sections["Article 1"].chapter == "Chapter I"
    assert sections["Article 5"].chapter == "Chapter II"
    assert sections["Article 5"].article_number == 5
    assert sections["Annex III"].chapter is None


def test_unstructured_document_falls_back_to_single_section():
    sections = parse_sections(_doc("Just a manual.\nPress the button to start."))
    assert len(sections) == 1 and sections[0].section_type == "body"


def test_helpers():
    assert roman_to_int("XIII") == 13
    assert normalize_ref("annex iii") == "Annex III"
    assert normalize_ref("article  5") == "Article 5"
    assert "ELI:" not in clean_page("Text line\nELI: http://data.europa.eu/eli/reg/2024/1689/oj\n12/144\nmore")
    doc = RawDocument("d", "s", "t", "aaaa\n\nbbbb", page_starts=[0, 6])
    assert doc.page_at(0) == 1 and doc.page_at(7) == 2