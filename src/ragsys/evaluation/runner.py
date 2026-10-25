from __future__ import annotations

import logging
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

from ragsys import observability
from ragsys.config import PipelineConfig, Settings
from ragsys.evaluation.dataset import EvalExample, write_jsonl
from ragsys.evaluation.metrics import mean_of, percentile, retrieval_metrics
from ragsys.evaluation.ragas_eval import score_with_ragas

logger = logging.getLogger(__name__)
GENERATION_METRICS = ["faithfulness", "answer_relevancy", "context_recall", "factual_correctness", "citation_hit"]


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def _evaluate_one(pipeline, ex: EvalExample, *, with_generation: bool, ks: list[int], settings: Settings) -> dict:
    row: dict = {
        "id": ex.id,
        "question": ex.question,
        "question_type": ex.question_type,
        "gold_refs": ex.gold_refs,
        "reference_answer": ex.reference_answer,
        "expected_answerable": ex.answerable,
    }
    start = time.perf_counter()
    try:
        if with_generation:
            result = pipeline.answer(
                ex.question, ex.filters,
                callbacks=observability.get_callbacks(settings),
                metadata={"eval_example_id": ex.id},
            )
            docs = result.retrieved
            row.update(
                answer=result.answer,
                answerable_pred=result.answerable,
                contexts=result.contexts,
                cited_refs=[c["ref"] for c in result.citations],
                warnings=result.warnings,
                total_tokens=result.total_tokens,
            )
            if ex.answerable and ex.gold_refs and result.answerable:
                cited = {r for c in result.citations for r in c["refs"]}
                row["citation_hit"] = float(bool(cited & set(ex.gold_refs)))
        else:
            docs = pipeline.retrieve(ex.question, ex.filters)
        row["retrieved_refs"] = [d.metadata.get("ref") for d in docs]
        if ex.answerable and ex.gold_refs:
            row.update(retrieval_metrics(docs, ex.gold_refs, ks))
    except Exception as exc:
        logger.exception("Example %s failed", ex.id)
        row["error"] = repr(exc)
    row["latency_ms"] = (time.perf_counter() - start) * 1000
    return row


def aggregate(rows: list[dict], ks: list[int]) -> dict:
    ok = [r for r in rows if not r.get("error")]
    keys = [f"hit@{k}" for k in ks] + [f"recall@{k}" for k in ks] + ["mrr"] + GENERATION_METRICS
    metrics = {k: v for k in keys if (v := mean_of(ok, k)) is not None}

    judged = [r for r in ok if "answerable_pred" in r]
    unanswerable = [r for r in judged if not r["expected_answerable"]]
    answerable = [r for r in judged if r["expected_answerable"]]
    if unanswerable:
        metrics["correct_abstention_rate"] = sum(not r["answerable_pred"] for r in unanswerable) / len(unanswerable)
    if answerable:
        metrics["false_abstention_rate"] = sum(not r["answerable_pred"] for r in answerable) / len(answerable)
    latencies = [r["latency_ms"] for r in ok]
    metrics["latency_ms_p50"] = percentile(latencies, 50)
    metrics["latency_ms_p95"] = percentile(latencies, 95)
    if judged:
        metrics["tokens_per_query"] = mean_of(judged, "total_tokens")

    by_type: dict[str, dict] = {}
    for qtype in sorted({r["question_type"] for r in ok}):
        subset = [r for r in ok if r["question_type"] == qtype]
        by_type[qtype] = {"n": len(subset)} | {
            k: v for k in ("recall@5", "mrr", "faithfulness", "factual_correctness") if (v := mean_of(subset, k)) is not None
        }
    return {"n_examples": len(rows), "n_errors": len(rows) - len(ok), "metrics": metrics, "by_question_type": by_type}


def run_experiment(
    settings: Settings,
    cfg: PipelineConfig,
    examples: list[EvalExample],
    *,
    with_generation: bool,
    out_root: Path,
    dataset_info: dict | None = None,
) -> tuple[dict, list[dict]]:
    from ragsys.generation.pipeline import RAGPipeline
    from ragsys.ingest.pipeline import ensure_index

    ensure_index(settings, cfg)
    pipeline = RAGPipeline(settings, cfg)
    ks = sorted({1, 3, 5, cfg.top_k})

    with ThreadPoolExecutor(max_workers=settings.eval_concurrency) as pool:
        work = lambda ex: _evaluate_one(pipeline, ex, with_generation=with_generation, ks=ks, settings=settings)  # noqa: E731
        rows = list(tqdm(pool.map(work, examples), total=len(examples), desc=cfg.name))

    if with_generation:
        scores = score_with_ragas(rows, settings)
        for row in rows:
            row.update(scores.get(row["id"], {}))
    observability.flush()

    summary = {
        "experiment": cfg.name,
        "config": cfg.model_dump(),
        "with_generation": with_generation,
        "models": {
            "llm": settings.llm_model, "judge": settings.judge_model, "embeddings": settings.embedding_model,
            "sparse": settings.sparse_model,
            "reranker": {"cross_encoder": settings.cross_encoder_model, "cohere": settings.cohere_rerank_model}.get(cfg.reranker),
        },
        "dataset": dataset_info or {},
        "git_commit": _git_commit(),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    } | aggregate(rows, ks)

    out_dir = out_root / cfg.name
    write_jsonl(out_dir / "per_question.jsonl", rows)
    from ragsys.evaluation.reporting import render_experiment_report, write_json

    write_json(out_dir / "summary.json", summary)
    (out_dir / "report.md").write_text(render_experiment_report(summary), encoding="utf-8")
    logger.info("%s: %s", cfg.name, {k: round(v, 3) for k, v in summary["metrics"].items() if v is not None})
    return summary, rows