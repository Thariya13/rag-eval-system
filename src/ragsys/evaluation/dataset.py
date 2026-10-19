from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from ragsys.ingest.structure import normalize_ref

QuestionType = Literal["factual", "numeric", "definition", "multi_hop", "comparative", "unanswerable"]


class EvalExample(BaseModel):
    id: str
    question: str
    reference_answer: str | None = None
    gold_refs: list[str] = Field(default_factory=list)
    question_type: QuestionType = "factual"
    filters: dict | None = None
    verified: bool = False
    source: Literal["human", "synthetic"] = "synthetic"
    evidence: str | None = None
    notes: str | None = None

    @field_validator("gold_refs")
    @classmethod
    def _normalize(cls, refs: list[str]) -> list[str]:
        return [normalize_ref(r) for r in refs]

    @property
    def answerable(self) -> bool:
        return self.question_type != "unanswerable"


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def stratified_sample(examples: list[EvalExample], n: int, seed: int = 13) -> list[EvalExample]:
    rng = random.Random(seed)
    buckets: dict[str, list[EvalExample]] = defaultdict(list)
    for ex in examples:
        buckets[ex.question_type].append(ex)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    out: list[EvalExample] = []
    while len(out) < n and any(buckets.values()):
        for key in sorted(buckets):
            if buckets[key] and len(out) < n:
                out.append(buckets[key].pop())
    return out


def load_examples(
    path: Path, include_unverified: bool = False, limit: int | None = None, smoke: int | None = None
) -> list[EvalExample]:
    examples = [EvalExample(**row) for row in read_jsonl(path)]
    if not include_unverified:
        examples = [ex for ex in examples if ex.verified]
    if not examples:
        raise ValueError(f"No usable examples in {path} (use --include-unverified for unreviewed data)")
    if smoke:
        examples = stratified_sample(examples, smoke)
    if limit:
        examples = examples[:limit]
    return examples