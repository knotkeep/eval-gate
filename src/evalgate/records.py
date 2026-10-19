"""Result records: one JSON object per line, one line per case per trial.

    {"case_id": "claim_status-003", "slice": "happy", "passed": true,
     "metrics": {"latency_s": 3.2, "cost_usd": 0.0}, "error": null}

`passed` may be a boolean or a score in [0, 1] (`score` is accepted as an alias). A record with a
non-empty `error` is an infrastructure failure (timeout, server error): it is counted and reported
but left out of every pass rate, so a flaky model server cannot pass as an agent regression.
Several trials of a case are several records with the same `case_id`.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class RecordError(ValueError):
    """Malformed results file."""


@dataclass
class Case:
    case_id: str
    slice: str
    scores: list[float] = field(default_factory=list)
    errors: int = 0
    metrics: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))

    @property
    def score(self) -> float | None:
        """Mean over scored trials; None when every trial errored."""
        return sum(self.scores) / len(self.scores) if self.scores else None


@dataclass
class ResultSet:
    cases: dict[str, Case]
    sources: list[str]
    records: int
    errors: int

    @property
    def error_rate(self) -> float:
        return self.errors / self.records if self.records else 0.0


def _score(rec: dict[str, Any], where: str) -> float:
    raw = rec.get("passed", rec.get("score"))
    if isinstance(raw, bool):
        return float(raw)
    if isinstance(raw, int | float) and 0.0 <= raw <= 1.0:
        return float(raw)
    raise RecordError(f"{where}: 'passed' must be a boolean or a number in [0, 1], got {raw!r}")


def load(paths: list[str | Path]) -> ResultSet:
    """Merge one or more JSONL files; repeated case IDs become extra trials."""
    cases: dict[str, Case] = {}
    records = errors = 0
    for path in paths:
        p = Path(path)
        if not p.is_file():
            raise RecordError(f"{p}: no such file")
        for n, line in enumerate(p.read_text().splitlines(), 1):
            if not line.strip():
                continue
            where = f"{p}:{n}"
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                raise RecordError(f"{where}: invalid JSON ({e.msg})") from None
            cid = rec.get("case_id")
            if not isinstance(cid, str) or not cid:
                raise RecordError(f"{where}: missing case_id")
            sl = str(rec.get("slice") or "all")
            case = cases.setdefault(cid, Case(cid, sl))
            if case.slice != sl:
                raise RecordError(f"{where}: case {cid} is in slice {case.slice!r} and {sl!r}")
            records += 1
            if rec.get("error"):
                case.errors += 1
                errors += 1
                continue
            case.scores.append(_score(rec, where))
            for k, v in (rec.get("metrics") or {}).items():
                if isinstance(v, int | float) and not isinstance(v, bool):
                    case.metrics[k].append(float(v))
    return ResultSet(cases, [str(p) for p in paths], records, errors)
