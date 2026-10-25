from types import SimpleNamespace

from ragsys.evaluation.metrics import retrieval_metrics


def _doc(*refs):
    return SimpleNamespace(metadata={"ref": refs[0], "refs": list(refs)})


def test_retrieval_metrics_multi_hop():
    docs = [_doc("Article 9"), _doc("Article 6"), _doc("Annex III"), _doc("Article 10")]
    m = retrieval_metrics(docs, ["Article 6", "annex iii"], ks=[1, 3])
    assert m["hit@1"] == 0.0 and m["recall@1"] == 0.0
    assert m["hit@3"] == 1.0 and m["recall@3"] == 1.0
    assert m["mrr"] == 0.5


def test_retrieval_metrics_partial_and_miss():
    docs = [_doc("Article 6", "Article 7")]
    assert retrieval_metrics(docs, ["Article 7", "Annex III"], ks=[5])["recall@5"] == 0.5
    assert retrieval_metrics([_doc("Article 1")], ["Article 2"], ks=[5])["mrr"] == 0.0