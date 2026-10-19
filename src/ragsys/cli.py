from __future__ import annotations

import typer

from ragsys import observability
from ragsys.config import PipelineConfig, get_settings, load_pipeline_config

app = typer.Typer(no_args_is_help=True, help="Production RAG with an evaluation harness")


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


if __name__ == "__main__":
    app()