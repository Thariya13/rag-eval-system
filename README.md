# Regulatory RAG with a Rigorous Evaluation Harness

# Results

Dataset: `data/eval/testset.jsonl` (sha `747b6077e326`, n=150). LLM `openai:gpt-5-mini`, judge `openai:gpt-4.1`, embeddings `openai:text-embedding-3-small`. Scores in %.

| Experiment | Chunking | Ctx | Retrieval | Reranker | Hit@5 | Recall@5 | MRR | Faithfulness | Answer rel. | Ctx recall | Factual corr. | Citation hit | Correct abstain | False abstain | p50 ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A_dense_recursive | recursive | no | dense | none | 89.6 | 86.3 | 74.5 | 95.7 | 72.3 | 86.3 | 24.4 | 92.5 | 73.3 | 11.1 | 10,165 |
| F_hybrid_structure_rerank | structure | no | hybrid | cross_encoder | 98.5 | 94.8 | 91.6 | 97.6 | 75.7 | 94.2 | 25.1 | 100.0 | 80.0 | 4.4 | 21,500 |
| G_full_contextual | structure | yes | hybrid | cross_encoder | 100.0 | 97.4 | 94.3 | 97.7 | 78.7 | 99.3 | 28.0 | 100.0 | 73.3 | 1.5 | 28,850 |

## Paired bootstrap vs baseline (`A_dense_recursive`)

Delta in percentage points with 95% CI over per-question scores. **Bold** = CI excludes zero.

| Experiment | Recall@5 | MRR | Faithfulness | Factual corr. |
|---|---|---|---|---|
| F_hybrid_structure_rerank | **+8.5 [+4.1, +13.7]** | **+17.1 [+11.4, +23.5]** | +1.9 [-0.5, +4.4] | +0.7 [-4.2, +5.6] |
| G_full_contextual | **+11.1 [+6.7, +16.3]** | **+19.8 [+13.5, +26.3]** | +1.9 [-0.8, +4.5] | +3.6 [-1.2, +8.4] |

Production configuration (`configs/pipeline.yaml`): G_full_contextual