from __future__ import annotations

import json
from pathlib import Path

from ragsys.evaluation.stats import paired_bootstrap

PCT_METRICS = {
    "hit@5": "Hit@5", "recall@5": "Recall@5", "mrr": "MRR", "faithfulness": "Faithfulness",
    "answer_relevancy": "Answer rel.", "context_recall": "Ctx recall", "factual_correctness": "Factual corr.",
    "citation_hit": "Citation hit", "correct_abstention_rate": "Correct abstain", "false_abstention_rate": "False abstain",
}
SIGNIFICANCE_METRICS = ["recall@5", "mrr", "faithfulness", "factual_correctness"]


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def _fmt(value, pct: bool = True) -> str:
    if value is None:
        return "-"
    return f"{value * 100:.1f}" if pct else f"{value:,.0f}"


def render_experiment_report(summary: dict) -> str:
    m = summary["metrics"]
    lines = [
        f"## RAG eval: `{summary['experiment']}`",
        "",
        f"{summary['n_examples']} examples, {summary['n_errors']} errors, commit `{summary.get('git_commit')}`, "
        f"LLM `{summary['models']['llm']}`, judge `{summary['models']['judge']}`",
        "",
        "| Metric | Value |",
        "|---|---|",
    ]
    for key, label in PCT_METRICS.items():
        if key in m:
            lines.append(f"| {label} | {_fmt(m[key])} |")
    for key in ("latency_ms_p50", "latency_ms_p95", "tokens_per_query"):
        if m.get(key) is not None:
            lines.append(f"| {key} | {_fmt(m[key], pct=False)} |")
    lines += ["", "### By question type", "", "| Type | n | Recall@5 | Faithfulness |", "|---|---|---|---|"]
    for qtype, values in summary.get("by_question_type", {}).items():
        lines.append(f"| {qtype} | {values['n']} | {_fmt(values.get('recall@5'))} | {_fmt(values.get('faithfulness'))} |")
    return "\n".join(lines) + "\n"


def _per_question(rows: list[dict], metric: str) -> dict[str, float]:
    return {r["id"]: float(r[metric]) for r in rows if r.get(metric) is not None and not r.get("error")}


def render_ablation(summaries: list[dict], rows_by_name: dict[str, list[dict]]) -> str:
    baseline = summaries[0]
    metric_keys = [k for k in PCT_METRICS if any(k in s["metrics"] for s in summaries)]
    header = ["Experiment", "Chunking", "Ctx", "Retrieval", "Reranker"] + [PCT_METRICS[k] for k in metric_keys] + ["p50 ms"]
    lines = [
        "# Ablation study",
        "",
        f"Dataset: `{baseline['dataset'].get('path')}` (sha `{baseline['dataset'].get('sha256')}`, "
        f"n={baseline['n_examples']}). LLM `{baseline['models']['llm']}`, judge `{baseline['models']['judge']}`, "
        f"embeddings `{baseline['models']['embeddings']}`. Scores in %.",
        "",
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
    ]
    for s in summaries:
        c = s["config"]
        cells = [s["experiment"], c["chunking"], "yes" if c["contextual"] else "no", c["retrieval_mode"], c["reranker"]]
        cells += [_fmt(s["metrics"].get(k)) for k in metric_keys]
        cells.append(_fmt(s["metrics"].get("latency_ms_p50"), pct=False))
        lines.append("| " + " | ".join(cells) + " |")

    lines += [
        "",
        f"## Paired bootstrap vs baseline (`{baseline['experiment']}`)",
        "",
        "Delta in percentage points with 95% CI over per-question scores. **Bold** = CI excludes zero.",
        "",
        "| Experiment | " + " | ".join(PCT_METRICS[k] for k in SIGNIFICANCE_METRICS) + " |",
        "|" + "---|" * (len(SIGNIFICANCE_METRICS) + 1),
    ]
    base_rows = rows_by_name[baseline["experiment"]]
    for s in summaries[1:]:
        cells = [s["experiment"]]
        for metric in SIGNIFICANCE_METRICS:
            res = paired_bootstrap(_per_question(base_rows, metric), _per_question(rows_by_name[s["experiment"]], metric))
            if res is None:
                cells.append("-")
                continue
            text = f"{res['delta'] * 100:+.1f} [{res['ci_low'] * 100:+.1f}, {res['ci_high'] * 100:+.1f}]"
            cells.append(f"**{text}**" if res["significant"] else text)
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"