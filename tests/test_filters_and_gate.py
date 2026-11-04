import json

import pytest

from ragsys.evaluation.gate import check_thresholds


def test_gate(tmp_path):
    summary = tmp_path / "s.json"
    summary.write_text(json.dumps({"metrics": {"recall@5": 0.7, "false_abstention_rate": 0.3}}))
    thresholds = tmp_path / "t.yaml"
    thresholds.write_text("min:\n  recall@5: 0.8\nmax:\n  false_abstention_rate: 0.2\n")
    failures = check_thresholds(summary, thresholds)
    assert len(failures) == 2


def test_filter_builder_rejects_unknown_fields():
    pytest.importorskip("qdrant_client")
    from ragsys.retrieval.filters import build_filter

    f = build_filter({"ref": "article 5", "chapter": None})
    assert f.must[0].key == "metadata.refs" and f.must[0].match.value == "Article 5"
    with pytest.raises(ValueError):
        build_filter({"password": "x"})