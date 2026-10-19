from __future__ import annotations

import hashlib
import logging
import random
import re
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from ragsys.config import Settings
from ragsys.evaluation.dataset import EvalExample, read_jsonl, write_jsonl
from ragsys.ingest.loaders import load_corpus
from ragsys.ingest.structure import parse_sections
from ragsys.models import get_chat_model

logger = logging.getLogger(__name__)
MAX_EXCERPT = 6000


class GeneratedQA(BaseModel):
    question: str
    answer: str
    question_type: str = Field(description="factual | numeric | definition | multi_hop | comparative")


class GeneratedBatch(BaseModel):
    items: list[GeneratedQA]


class GeneratedQuestions(BaseModel):
    questions: list[str]


STYLE = (
    "Write like a real user (compliance officer, product manager, lawyer, engineer). Do NOT copy distinctive "
    "phrases from the text - paraphrase. Questions must be fully answerable from the excerpt(s). "
    "Reference answers must be concise and exact."
)

SINGLE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You create evaluation data for a document question-answering system. " + STYLE),
    ("human", "Document: {doc_title}\nSection: {ref} - {title}\n\n<excerpt>\n{text}\n</excerpt>\n\n"
              "Write {n} diverse question/answer pairs. Use types 'factual', 'numeric' (dates, amounts, "
              "thresholds, deadlines) or 'definition' as the excerpt allows."),
])

MULTI_HOP_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You create evaluation data for a document question-answering system. " + STYLE),
    ("human", "Document: {doc_title}\n\n<excerpt ref=\"{ref_a}\">\n{text_a}\n</excerpt>\n\n"
              "<excerpt ref=\"{ref_b}\">\n{text_b}\n</excerpt>\n\n"
              "Write 1 question whose correct answer REQUIRES information from BOTH excerpts. "
              "Use question_type 'multi_hop'."),
])

UNANSWERABLE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You create negative test cases for a document question-answering system."),
    ("human", "The system only knows the document '{doc_title}'. Its sections include:\n{toc}\n\n"
              "Write {n} realistic questions a user might ask this assistant that the document does NOT answer "
              "(other jurisdictions, unrelated laws, specific companies, tax, events after publication, "
              "operational details it does not specify). Avoid anything plausibly covered by the sections."),
])


def _id(*parts: str) -> str:
    return "syn-" + hashlib.sha256("|".join(parts).encode()).hexdigest()[:10]


def generate_candidates(
    settings: Settings,
    out_path: Path,
    n_sections: int = 60,
    per_section: int = 2,
    n_multi_hop: int = 20,
    n_unanswerable: int = 15,
    seed: int = 42,
) -> int:
    rng = random.Random(seed)
    llm = get_chat_model(settings.judge_model, settings.judge_temperature)
    batch_cfg = {"max_concurrency": settings.llm_max_concurrency}
    candidates: list[EvalExample] = []

    for doc in load_corpus(settings.raw_data_dir):
        sections = [s for s in parse_sections(doc) if s.section_type != "recital" and s.end - s.start >= 400]
        by_ref = {s.ref: s for s in sections}
        text_of = lambda s: doc.text[s.start : s.end][:MAX_EXCERPT]

        sampled = rng.sample(sections, min(n_sections, len(sections)))
        inputs = [
            {"doc_title": doc.title, "ref": s.ref, "title": s.title, "text": text_of(s), "n": per_section}
            for s in sampled
        ]
        outputs = (SINGLE_PROMPT | llm.with_structured_output(GeneratedBatch)).batch(
            inputs, config=batch_cfg, return_exceptions=True
        )
        for section, result in zip(sampled, outputs):
            if isinstance(result, Exception) or result is None:
                logger.warning("Generation failed for %s: %s", section.ref, result)
                continue
            for qa in result.items:
                qtype = qa.question_type if qa.question_type in {"factual", "numeric", "definition"} else "factual"
                candidates.append(EvalExample(
                    id=_id(section.ref, qa.question), question=qa.question, reference_answer=qa.answer,
                    gold_refs=[section.ref], question_type=qtype, evidence=text_of(section)[:1500],
                ))

        pairs = []
        for s in sections:
            for number in set(re.findall(r"Article (\d+)", text_of(s))):
                other = by_ref.get(f"Article {number}")
                if other and other.ref != s.ref:
                    pairs.append((s, other))
        pairs = rng.sample(pairs, min(n_multi_hop, len(pairs)))
        inputs = [
            {"doc_title": doc.title, "ref_a": a.ref, "text_a": text_of(a)[:3500], "ref_b": b.ref, "text_b": text_of(b)[:3500]}
            for a, b in pairs
        ]
        outputs = (MULTI_HOP_PROMPT | llm.with_structured_output(GeneratedBatch)).batch(
            inputs, config=batch_cfg, return_exceptions=True
        )
        for (a, b), result in zip(pairs, outputs):
            if isinstance(result, Exception) or result is None or not result.items:
                continue
            qa = result.items[0]
            candidates.append(EvalExample(
                id=_id(a.ref, b.ref, qa.question), question=qa.question, reference_answer=qa.answer,
                gold_refs=[a.ref, b.ref], question_type="multi_hop",
                evidence=f"{a.ref}: {text_of(a)[:700]}\n---\n{b.ref}: {text_of(b)[:700]}",
            ))

        toc = "\n".join(f"- {s.ref}: {s.title}" for s in sections[:120])
        result = (UNANSWERABLE_PROMPT | llm.with_structured_output(GeneratedQuestions)).invoke(
            {"doc_title": doc.title, "toc": toc, "n": n_unanswerable}
        )
        for q in result.questions if result else []:
            candidates.append(EvalExample(
                id=_id("unanswerable", q), question=q, question_type="unanswerable",
                reference_answer="The documents do not answer this question.",
            ))

    existing = {row["id"] for row in read_jsonl(out_path)}
    new_rows = [c.model_dump() for c in candidates if c.id not in existing]
    write_jsonl(out_path, read_jsonl(out_path) + new_rows)
    return len(new_rows)


def review_candidates(candidates_path: Path, out_path: Path) -> None:
    accepted = {row["id"]: row for row in read_jsonl(out_path)}
    rejected_path = out_path.with_suffix(".rejected.txt")
    rejected = set(rejected_path.read_text().split()) if rejected_path.exists() else set()
    todo = [EvalExample(**r) for r in read_jsonl(candidates_path) if r["id"] not in accepted and r["id"] not in rejected]
    print(f"{len(todo)} candidates to review. Keys: [a]ccept  [e]dit  [r]eject  [q]uit\n")

    for i, ex in enumerate(todo, 1):
        print("=" * 90)
        print(f"[{i}/{len(todo)}] type={ex.question_type} gold={ex.gold_refs}")
        print(f"Q: {ex.question}\nA: {ex.reference_answer}")
        if ex.evidence:
            print(f"\n--- evidence ---\n{ex.evidence[:1200]}\n")
        choice = input("> ").strip().lower()
        if choice == "q":
            break
        if choice == "r":
            rejected.add(ex.id)
            rejected_path.write_text("\n".join(sorted(rejected)))
            continue
        if choice == "e":
            ex.question = input("question [enter=keep]: ").strip() or ex.question
            ex.reference_answer = input("answer [enter=keep]: ").strip() or ex.reference_answer
            refs = input(f"gold refs, comma separated [enter=keep {ex.gold_refs}]: ").strip()
            if refs:
                ex.gold_refs = [r.strip() for r in refs.split(",")]
        if choice in {"a", "e"}:
            ex.verified = True
            accepted[ex.id] = EvalExample(**ex.model_dump()).model_dump()
            write_jsonl(out_path, list(accepted.values()))
    print(f"\n{len(accepted)} verified examples in {out_path}")