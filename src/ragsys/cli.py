from __future__ import annotations

from pathlib import Path

import typer

from ragsys import observability
from ragsys.config import PipelineConfig, get_settings, load_pipeline_config

app = typer.Typer(no_args_is_help=True, help="Production RAG with an evaluation harness")
eval_app = typer.Typer(no_args_is_help=True, help="Test-set generation, evaluation, ablations, CI gate")
app.add_typer(eval_app, name="eval")

DEFAULT_DATASET = Path("data/eval/testset.jsonl")

def _init():
    settings = get_settings()
    observability.configure_logging(settings.log_level)
    return settings


@app.command()
def ingest(
    chunking: str = typer.Option("structure", help="recursive | structure | semantic"),
    contextual: bool = typer.Option(False, "--contextual/--no-contextual"),
):
    settings = _init()
    from ragsys.ingest.pipeline import ensure_index

    cfg = PipelineConfig(chunking=chunking, contextual=contextual)
    typer.echo(f"Indexed: {ensure_index(settings, cfg, force=True)}")


@app.command()
def ask(
    question: str,
    ref: str | None = typer.Option(None, help='Filter, e.g. "Article 5"'),
    chapter: str | None = typer.Option(None, help='Filter, e.g. "Chapter III"'),
):
    settings = _init()
    from ragsys.generation.pipeline import RAGPipeline

    cfg = load_pipeline_config(settings.pipeline_config_path)
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
    with_generation: bool = typer.Option(False, "--with-generation", help="Also generate answers + LLM-judge metrics"),
    include_unverified: bool = False,
    limit: int | None = None,
    smoke: int | None = typer.Option(None, help="Stratified subset size (for PR checks)"),
    out: Path = Path("results"),
):
    settings = _init()
    from ragsys.evaluation.runner import run_experiment

    cfg = load_pipeline_config(settings.pipeline_config_path)
    examples, info = _examples(dataset, include_unverified, limit, smoke)
    summary, _ = run_experiment(settings, cfg, examples, with_generation=with_generation, out_root=out, dataset_info=info)
    typer.echo((out / cfg.name / "report.md").read_text())


if __name__ == "__main__":
    app()