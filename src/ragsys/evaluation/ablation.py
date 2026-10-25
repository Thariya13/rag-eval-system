from __future__ import annotations

from pathlib import Path

import yaml

from ragsys.config import PipelineConfig, Settings
from ragsys.evaluation.dataset import EvalExample
from ragsys.evaluation.reporting import render_ablation, write_json
from ragsys.evaluation.runner import run_experiment


def load_experiments(path: Path) -> list[PipelineConfig]:
    data = yaml.safe_load(path.read_text())
    defaults = data.get("defaults", {})
    return [PipelineConfig(**(defaults | experiment)) for experiment in data["experiments"]]


def run_ablation(
    settings: Settings,
    experiments: list[PipelineConfig],
    examples: list[EvalExample],
    *,
    with_generation: bool,
    out_root: Path,
    dataset_info: dict,
) -> Path:
    summaries, rows_by_name = [], {}
    for cfg in experiments:
        summary, rows = run_experiment(
            settings, cfg, examples, with_generation=with_generation, out_root=out_root, dataset_info=dataset_info
        )
        summaries.append(summary)
        rows_by_name[cfg.name] = rows
    report = out_root / "ablation.md"
    report.write_text(render_ablation(summaries, rows_by_name), encoding="utf-8")
    write_json(out_root / "ablation_summary.json", {"experiments": summaries})
    return report