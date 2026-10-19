from ragsys.generation.citations import resolve_citations, strip_invalid_markers


def test_resolve_citations_orders_dedupes_and_bounds():
    answer = "Fines reach EUR 35 million [2]. Also 7% of turnover [2, 4]. Unknown [9]."
    assert resolve_citations(answer, declared=[1, 9], n_sources=6) == [2, 4, 1]


def test_strip_invalid_markers():
    assert strip_invalid_markers("A [1]. B [7]. C [2, 8].", n_sources=6) == "A [1]. B. C [2]."