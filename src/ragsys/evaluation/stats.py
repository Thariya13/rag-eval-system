from __future__ import annotations

import numpy as np


def paired_bootstrap(
    baseline: dict[str, float], candidate: dict[str, float], n_resamples: int = 5000, seed: int = 0
) -> dict | None:
    ids = sorted(set(baseline) & set(candidate))
    if len(ids) < 5:
        return None
    diffs = np.array([candidate[i] - baseline[i] for i in ids], dtype=float)
    rng = np.random.default_rng(seed)
    means = diffs[rng.integers(0, len(diffs), size=(n_resamples, len(diffs)))].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return {
        "delta": float(diffs.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "n": len(ids),
        "significant": bool(low > 0 or high < 0),
    }