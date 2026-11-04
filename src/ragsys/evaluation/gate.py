from __future__ import annotations

import json
from pathlib import Path

import yaml


def check_thresholds(summary_path: Path, thresholds_path: Path) -> list[str]:
    metrics = json.loads(summary_path.read_text())["metrics"]
    thresholds = yaml.safe_load(thresholds_path.read_text())
    failures = []
    for key, minimum in (thresholds.get("min") or {}).items():
        value = metrics.get(key)
        if value is None:
            failures.append(f"{key}: missing from summary")
        elif value < minimum:
            failures.append(f"{key}: {value:.3f} < min {minimum}")
    for key, maximum in (thresholds.get("max") or {}).items():
        value = metrics.get(key)
        if value is not None and value > maximum:
            failures.append(f"{key}: {value:.3f} > max {maximum}")
    return failures