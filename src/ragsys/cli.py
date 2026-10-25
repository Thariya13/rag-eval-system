from __future__ import annotations

from pathlib import Path

import typer

from ragsys import observability
from ragsys.config import PipelineConfig, get_settings, load_pipeline_config

app = typer.Typer(no_args_is_help=True, help="Production RAG with an evaluation harness")
eval_app = typer.Typer(no_args_is_help=True, help="Test-set generation, evaluation, ablations, CI gate")
app.add_typer(eval_app, name="eval")

DEFAULT_DATASET = Path("data/eval/testset.jsonl")
DEFAULT_EXPERIMENTS = Path("configs/ablations.yaml")

def _init():
    settings = get_settings()
    observability.configure_logging(settings.log_level)
    return settings


def _resolve_config(settings, experiment: str | None, experiments_file: Path) -> PipelineConfig:
    if experiment:
        from ragsys.evaluation.ablation import load_experiments

        matches = [c for c in load_experiments(experiments_file) if c.name == experiment]
        if not matches:
            raise typer.BadParameter(f"No experiment named '{experiment}' in {experiments_file}")
        return matches[0]
    return load_pipeline_config(settings.pipeline_config_path)


@app.command()
def ingest(
    chunking: str = typer.Option("structure", help="recursive | structure | semantic"),
    contextual: bool = typer.Option(False, "--contextual/--no-contextual"),
    all_variants: bool = typer.Option(False, "--all", help="Build every index needed by the ablation file"),
):
    settings = _init()
    from ragsys.ingest.pipeline import ensure_index

    if all_variants:
        from ragsys.evaluation.ablation import load_experiments

        seen = set()
        for cfg in load_experiments(DEFAULT_EXPERIMENTS):
            if cfg.index_key not in seen:
                seen.add(cfg.index_key)
                typer.echo(f"Indexed: {ensure_index(settings, cfg, force=True)}")
        return
    cfg = PipelineConfig(chunking=chunking, contextual=contextual)
    typer.echo(f"Indexed: {ensure_index(settings, cfg, force=True)}")


@app.command()
def ask(
    question: str,
    ref: str | None = typer.Option(None, help='Filter, e.g. "Article 5"'),
    chapter: str | None = typer.Option(None, help='Filter, e.g. "Chapter III"'),
    experiment: str | None = typer.Option(None, help="Use a named config from ablations.yaml"),
):
    settings = _init()
    from ragsys.generation.pipeline import RAGPipeline

    cfg = _resolve_config(settings, experiment, DEFAULT_EXPERIMENTS)
    filters = {k: v for k, v in {"ref": ref, "chapter": chapter}.items() if v} or None
    result = RAGPipeline(settings, cfg).answer(question, filters, callbacks=observability.get_callbacks(settings))
    observability.flush()
    typer.echo(f"\n{result.answer}\n")
    for c in result.citations:
        score = f"{c['score']:.3f}" if c["score"] is not None else "n/a"
        typer.echo(f"  [{c['id']}] {c['ref']} - {c['title']} (p. {c['page']}, score {score})")
    typer.echo(f"\n{result.latency_ms:.0f} ms, {result.total_tokens} tokens, warnings={result.warnings}")


@app.command()
def serve(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn

    uvicorn.run("ragsys.api.app:app", host=host, port=port)


@eval_app.command("generate")
def eval_generate(
    out: Path = Path("data/eval/candidates.jsonl"),
    sections: int = 60,
    per_section: int = 2,
    multi_hop: int = 20,
    unanswerable: int = 15,
):
    settings = _init()
    from ragsys.evaluation.generate import generate_candidates

    n = generate_candidates(settings, out, sections, per_section, multi_hop, unanswerable)
    typer.echo(f"Wrote {n} new candidates to {out}. Next: ragsys eval review")


@eval_app.command("review")
def eval_review(candidates: Path = Path("data/eval/candidates.jsonl"), out: Path = DEFAULT_DATASET):
    from ragsys.evaluation.generate import review_candidates

    review_candidates(candidates, out)


def _examples(dataset: Path, include_unverified: bool, limit: int | None, smoke: int | None):
    from ragsys.evaluation.dataset import file_sha256, load_examples

    examples = load_examples(dataset, include_unverified, limit, smoke)
    info = {"path": str(dataset), "sha256": file_sha256(dataset), "n_used": len(examples), "smoke": smoke}
    return examples, info


@eval_app.command("run")
def eval_run(
    dataset: Path = DEFAULT_DATASET,
    experiment: str | None = typer.Option(None, help="Named config from ablations.yaml (default: pipeline.yaml)"),
    experiments_file: Path = DEFAULT_EXPERIMENTS,
    with_generation: bool = typer.Option(False, "--with-generation", help="Also generate answers + LLM-judge metrics"),
    include_unverified: bool = False,
    limit: int | None = None,
    smoke: int | None = typer.Option(None, help="Stratified subset size (for PR checks)"),
    out: Path = Path("results"),
):
    settings = _init()
    from ragsys.evaluation.runner import run_experiment

    cfg = _resolve_config(settings, experiment, experiments_file)
    examples, info = _examples(dataset, include_unverified, limit, smoke)
    summary, _ = run_experiment(settings, cfg, examples, with_generation=with_generation, out_root=out, dataset_info=info)
    typer.echo((out / cfg.name / "report.md").read_text())


@eval_app.command("ablate")
def eval_ablate(
    dataset: Path = DEFAULT_DATASET,
    experiments_file: Path = DEFAULT_EXPERIMENTS,
    with_generation: bool = typer.Option(False, "--with-generation"),
    include_unverified: bool = False,
    limit: int | None = None,
    out: Path = Path("results"),
):
    settings = _init()
    from ragsys.evaluation.ablation import load_experiments, run_ablation

    examples, info = _examples(dataset, include_unverified, limit, None)
    report = run_ablation(
        settings, load_experiments(experiments_file), examples,
        with_generation=with_generation, out_root=out, dataset_info=info,
    )
    typer.echo(report.read_text())


if __name__ == "__main__":
    app()