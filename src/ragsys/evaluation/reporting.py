from __future__ import annotations

import json
from pathlib import Path

PCT_METRICS = {
    "hit@5": "Hit@5", "recall@5": "Recall@5", "mrr": "MRR", "faithfulness": "Faithfulness",
    "answer_relevancy": "Answer rel.", "context_recall": "Ctx recall", "factual_correctness": "Factual corr.",
    "citation_hit": "Citation hit", "correct_abstention_rate": "Correct abstain", "false_abstention_rate": "False abstain",
}


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