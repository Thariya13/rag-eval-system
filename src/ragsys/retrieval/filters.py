from __future__ import annotations

from typing import Any

FILTERABLE_FIELDS = {
    "ref": "refs",
    "chapter": "chapter",
    "doc_id": "doc_id",
    "section_type": "section_type",
    "article_number": "article_number",
}


def build_filter(filters: dict[str, Any] | None):
    if not filters:
        return None
    from qdrant_client import models

    from ragsys.ingest.structure import normalize_ref

    conditions = []
    for key, value in filters.items():
        if value is None:
            continue
        if key not in FILTERABLE_FIELDS:
            raise ValueError(f"Unsupported filter '{key}'. Allowed: {sorted(FILTERABLE_FIELDS)}")
        if key == "ref":
            value = [normalize_ref(v) for v in value] if isinstance(value, list) else normalize_ref(value)
        field = f"metadata.{FILTERABLE_FIELDS[key]}"
        match = models.MatchAny(any=value) if isinstance(value, list) else models.MatchValue(value=value)
        conditions.append(models.FieldCondition(key=field, match=match))
    return models.Filter(must=conditions) if conditions else None