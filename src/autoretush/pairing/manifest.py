from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path

from autoretush.pairing.matcher import PairCandidate


def append_candidates(path: Path, candidates: Iterable[PairCandidate]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for candidate in candidates:
            handle.write(json.dumps(candidate.to_dict(), ensure_ascii=False) + "\n")
            count += 1
        handle.flush()
    return count


def iter_records(path: Path) -> Iterator[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_number}: {path}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"Manifest line {line_number} is not an object: {path}")
            yield item
