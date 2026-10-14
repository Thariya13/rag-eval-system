from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ChunkingStrategy = Literal["recursive", "structure", "semantic"]
RetrievalModeName = Literal["dense", "sparse", "hybrid"]
RerankerKind = Literal["none", "cross_encoder", "cohere"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Models
    llm_model: str = "openai:gpt-5-mini"
    llm_temperature: float | None = None
    contextualizer_model: str = "openai:gpt-5-mini"
    judge_model: str = "openai:gpt-4.1"
    judge_temperature: float | None = 0.0
    embedding_model: str = "openai:text-embedding-3-small"
    sparse_model: str = "Qdrant/bm25"
    cross_encoder_model: str = "BAAI/bge-reranker-v2-m3"
    cohere_rerank_model: str = "rerank-v3.5"

    # Vector store
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    collection_prefix: str = "docs"

    # Paths
    raw_data_dir: Path = Path("data/raw")
    cache_dir: Path = Path(".cache")
    pipeline_config_path: Path = Path("configs/pipeline.yaml")

    # Chunking
    chunk_size: int = 1200
    chunk_overlap: int = 150
    semantic_breakpoint_percentile: float = 90.0

    # Runtime
    llm_max_concurrency: int = 8
    eval_concurrency: int = 4
    request_timeout_s: float = 60.0
    langfuse_enabled: bool = True
    log_level: str = "INFO"

    @field_validator("llm_temperature", "judge_temperature", "qdrant_api_key", mode="before")
    @classmethod
    def _empty_to_none(cls, v):
        return None if v == "" else v


class PipelineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = "production"
    chunking: ChunkingStrategy = "structure"
    contextual: bool = False
    retrieval_mode: RetrievalModeName = "hybrid"
    reranker: RerankerKind = "cross_encoder"
    fetch_k: int = Field(30, ge=1, le=200, description="Candidates fetched before reranking")
    top_k: int = Field(6, ge=1, le=50, description="Chunks handed to the LLM")

    @model_validator(mode="after")
    def _check_k(self) -> "PipelineConfig":
        if self.fetch_k < self.top_k:
            raise ValueError("fetch_k must be >= top_k")
        return self

    @property
    def index_key(self) -> str:
        return f"{self.chunking}{'_ctx' if self.contextual else ''}"


def collection_name(settings: Settings, cfg: PipelineConfig) -> str:
    return f"{settings.collection_prefix}_{cfg.index_key}"


def load_pipeline_config(path: Path) -> PipelineConfig:
    if not path.exists():
        return PipelineConfig()
    return PipelineConfig(**(yaml.safe_load(path.read_text()) or {}))


@lru_cache
def get_settings() -> Settings:
    load_dotenv(override=False)
    return Settings()