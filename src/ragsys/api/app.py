from __future__ import annotations

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from ragsys import observability
from ragsys.config import collection_name, get_settings, load_pipeline_config
from ragsys.generation.pipeline import RAGPipeline
from ragsys.retrieval.store import get_qdrant_client

logger = logging.getLogger("ragsys.api")


class SearchFilters(BaseModel):
    ref: str | None = Field(None, examples=["Article 5"])
    chapter: str | None = Field(None, examples=["Chapter III"])
    doc_id: str | None = None
    section_type: Literal["article", "annex", "recital", "body"] | None = None


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    filters: SearchFilters | None = None


class CitationOut(BaseModel):
    id: int
    ref: str
    title: str
    page: int | None
    source: str
    snippet: str
    score: float | None


class AskResponse(BaseModel):
    answer: str
    answerable: bool
    citations: list[CitationOut]
    warnings: list[str]
    latency_ms: float
    total_tokens: int
    pipeline: str
    request_id: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    observability.configure_logging(settings.log_level)
    cfg = load_pipeline_config(settings.pipeline_config_path)
    app.state.settings, app.state.cfg = settings, cfg
    app.state.pipeline = RAGPipeline(settings, cfg)
    logger.info("Pipeline '%s' ready: %s", cfg.name, cfg.model_dump())
    yield
    observability.flush()


app = FastAPI(title="Regulatory RAG API", version="0.1.0", lifespan=lifespan)


@app.middleware("http")
async def request_context(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
    request.state.request_id = request_id
    start = time.perf_counter()
    response = await call_next(request)
    response.headers["x-request-id"] = request_id
    logger.info(
        "%s %s -> %s in %.0f ms [%s]",
        request.method, request.url.path, response.status_code, (time.perf_counter() - start) * 1000, request_id,
    )
    return response


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.get("/readyz")
async def readyz(request: Request):
    settings, cfg = request.app.state.settings, request.app.state.cfg
    name = collection_name(settings, cfg)
    try:
        exists = await asyncio.to_thread(get_qdrant_client(settings).collection_exists, name)
    except Exception as exc:
        raise HTTPException(503, f"Qdrant unreachable: {exc}") from exc
    if not exists:
        raise HTTPException(503, f"Collection '{name}' missing - run ingestion")
    return {"status": "ready", "collection": name}


@app.post("/v1/ask", response_model=AskResponse)
async def ask(body: AskRequest, request: Request):
    settings = request.app.state.settings
    pipeline: RAGPipeline = request.app.state.pipeline
    filters = body.filters.model_dump(exclude_none=True) if body.filters else None
    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(
                pipeline.answer,
                body.question,
                filters,
                callbacks=observability.get_callbacks(settings),
                metadata={"request_id": request.state.request_id},
            ),
            timeout=settings.request_timeout_s,
        )
    except asyncio.TimeoutError as exc:
        raise HTTPException(504, "Answer generation timed out") from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        logger.exception("Pipeline failure [%s]", request.state.request_id)
        raise HTTPException(502, "Upstream model or vector store error") from exc

    return AskResponse(
        answer=result.answer,
        answerable=result.answerable,
        citations=[CitationOut(**{k: v for k, v in c.items() if k != "refs"}) for c in result.citations],
        warnings=result.warnings,
        latency_ms=round(result.latency_ms, 1),
        total_tokens=result.total_tokens,
        pipeline=pipeline.cfg.name,
        request_id=request.state.request_id,
    )