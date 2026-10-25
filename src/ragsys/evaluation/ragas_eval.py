from __future__ import annotations

import logging
import math
import re

from ragsys.config import Settings
from ragsys.models import get_chat_model, get_embeddings

logger = logging.getLogger(__name__)
_INPUT_COLUMNS = {"user_input", "response", "retrieved_contexts", "reference", "reference_contexts"}


def score_with_ragas(rows: list[dict], settings: Settings) -> dict[str, dict[str, float]]:
    try:
        from ragas import EvaluationDataset, SingleTurnSample, evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import FactualCorrectness, Faithfulness, LLMContextRecall, ResponseRelevancy
        from ragas.run_config import RunConfig
    except ImportError:
        logger.warning("ragas not installed (pip install 'ragsys[eval]'); skipping LLM-judge metrics")
        return {}

    eligible = [
        r for r in rows
        if not r.get("error") and r.get("expected_answerable") and r.get("answer") and r.get("reference_answer")
    ]
    if not eligible:
        return {}
    samples = [
        SingleTurnSample(
            user_input=r["question"],
            response=r["answer"],
            retrieved_contexts=r.get("contexts") or [""],
            reference=r["reference_answer"],
        )
        for r in eligible
    ]
    result = evaluate(
        dataset=EvaluationDataset(samples=samples),
        metrics=[Faithfulness(), ResponseRelevancy(), LLMContextRecall(), FactualCorrectness()],
        llm=LangchainLLMWrapper(get_chat_model(settings.judge_model, settings.judge_temperature)),
        embeddings=LangchainEmbeddingsWrapper(get_embeddings(settings.embedding_model)),
        run_config=RunConfig(max_workers=max(2, settings.eval_concurrency * 2), timeout=180),
        raise_exceptions=False,
        show_progress=True,
    )
    frame = result.to_pandas()
    metric_columns = [c for c in frame.columns if c not in _INPUT_COLUMNS]
    scores: dict[str, dict[str, float]] = {}
    for row, (_, record) in zip(eligible, frame.iterrows()):
        values = {}
        for column in metric_columns:
            try:
                value = float(record[column])
            except (TypeError, ValueError):
                continue
            if not math.isnan(value):
                values[re.sub(r"\(.*\)$", "", column)] = value
        scores[row["id"]] = values
    return scores