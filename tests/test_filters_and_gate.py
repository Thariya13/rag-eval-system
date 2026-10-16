import pytest


def test_filter_builder_rejects_unknown_fields():
    pytest.importorskip("qdrant_client")
    from ragsys.retrieval.filters import build_filter

    f = build_filter({"ref": "article 5", "chapter": None})
    assert f.must[0].key == "metadata.refs" and f.must[0].match.value == "Article 5"
    with pytest.raises(ValueError):
        build_filter({"password": "x"})