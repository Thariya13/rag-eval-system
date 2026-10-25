from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import numpy as np

from ragsys.ingest.structure import normalize_ref


def _doc_refs(doc) -> set[str]:
    md = getattr(doc, "metadata", None) or {}
    refs = md.get("refs") or ([md["ref"]] if md.get("ref") else [])
    return {normalize_ref(r) for r in refs}


def retrieval_metrics(docs: Sequence, gold_refs: Iterable[str], ks: Iterable[int]) -> dict[str, float]:
    gold = {normalize_ref(r) for r in gold_refs}
    if not gold:
        return {}
    per_doc = [_doc_refs(d) for d in docs]
    out: dict[str, float] = {}
    for k in ks:
        found = set().union(*per_doc[:k]) & gold
        out[f"hit@{k}"] = float(bool(found))
        out[f"recall@{k}"] = len(found) / len(gold)
    out["mrr"] = next((1.0 / rank for rank, refs in enumerate(per_doc, 1) if refs & gold), 0.0)
    return out


def mean_of(rows: list[dict], key: str) -> float | None:
    values = []
    for row in rows:
        value = row.get(key)
        if value is None or isinstance(value, (list, dict, str)):
            continue
        value = float(value)
        if not math.isnan(value):
            values.append(value)
    return sum(values) / len(values) if values else None


def percentile(values: list[float], q: float) -> float | None:
    return float(np.percentile(values, q)) if values else None