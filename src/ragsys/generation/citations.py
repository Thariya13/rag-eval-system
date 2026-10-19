from __future__ import annotations

import re

_MARKER = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def _numbers(group: str) -> list[int]:
    return [int(x) for x in re.split(r"\s*,\s*", group.strip()) if x]


def resolve_citations(answer: str, declared: list[int], n_sources: int) -> list[int]:
    ordered: list[int] = []
    for match in _MARKER.finditer(answer):
        ordered.extend(_numbers(match.group(1)))
    ordered.extend(declared)
    seen: set[int] = set()
    out = []
    for i in ordered:
        if 1 <= i <= n_sources and i not in seen:
            seen.add(i)
            out.append(i)
    return out


def strip_invalid_markers(answer: str, n_sources: int) -> str:
    def repl(match: re.Match) -> str:
        valid = [str(i) for i in _numbers(match.group(1)) if 1 <= i <= n_sources]
        return f"[{', '.join(valid)}]" if valid else ""

    return re.sub(r"\s+\.", ".", _MARKER.sub(repl, answer))