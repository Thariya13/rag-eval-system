from __future__ import annotations

import typer

from ragsys import observability
from ragsys.config import PipelineConfig, get_settings

app = typer.Typer(no_args_is_help=True, help="Production RAG with an evaluation harness")


@app.callback()
def main() -> None:
    """Pass"""


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


if __name__ == "__main__":
    app()