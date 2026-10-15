from __future__ import annotations

from functools import lru_cache

from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage


@lru_cache(maxsize=8)
def get_chat_model(model: str, temperature: float | None = None) -> BaseChatModel:
    kwargs: dict = {"max_retries": 3}
    if temperature is not None:
        kwargs["temperature"] = temperature
    return init_chat_model(model, **kwargs)


@lru_cache(maxsize=4)
def get_embeddings(model: str) -> Embeddings:
    return init_embeddings(model)


@lru_cache(maxsize=2)
def get_sparse_embeddings(model: str):
    from langchain_qdrant import FastEmbedSparse

    return FastEmbedSparse(model_name=model)


def message_text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts)