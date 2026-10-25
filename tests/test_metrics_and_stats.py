from types import SimpleNamespace

from ragsys.evaluation.metrics import retrieval_metrics
from ragsys.evaluation.stats import paired_bootstrap


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


def test_bootstrap_detects_real_improvement_and_ignores_noise():
    base = {str(i): 0.0 for i in range(100)}
    better = {str(i): 1.0 if i < 40 else 0.0 for i in range(100)}
    assert paired_bootstrap(base, better)["significant"]
    noisy = {str(i): 1.0 if i in (3, 50) else 0.0 for i in range(100)}
    base2 = {str(i): 1.0 if i in (7, 60) else 0.0 for i in range(100)}
    assert not paired_bootstrap(base2, noisy)["significant"]