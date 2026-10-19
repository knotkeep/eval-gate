"""Gate configuration and verdict.

Verdicts, in order of precedence, with their exit codes:

    error         4  the inputs cannot support a decision (too few cases, too many infra errors)
    floor         2  the candidate is below an absolute pass-rate floor
    regression    1  the candidate is significantly worse than the baseline, or a metric limit
                     (for example mean latency) is exceeded significantly
    inconclusive  3  no significant drop, but the interval does not rule out a drop larger than
                     `margin`; exits 0 with a warning unless `inconclusive: fail`
    pass          0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from evalgate.records import Case, ResultSet
from evalgate.stats import bootstrap_ci, drop_p_value_trials, welch_increase_p

EXIT = {"pass": 0, "regression": 1, "floor": 2, "inconclusive": 3, "error": 4}


@dataclass
class Config:
    alpha: float = 0.05
    margin: float = 0.05
    min_cases: int = 20
    max_error_rate: float = 0.05
    inconclusive: str = "warn"
    floors: dict[str, float] = field(default_factory=dict)
    metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    bootstrap_samples: int = 10_000
    seed: int = 0


def load_config(path: str | Path | None) -> Config:
    """Read gate.yaml. `floors` takes `overall` and slice names; `metrics` takes limits."""
    if path is None:
        return Config()
    raw: dict[str, Any] = yaml.safe_load(Path(path).read_text()) or {}
    floors_raw = raw.pop("floors", {}) or {}
    floors = {k: float(v) for k, v in (floors_raw.pop("slices", {}) or {}).items()}
    if "overall" in floors_raw:
        floors["overall"] = float(floors_raw.pop("overall"))
    if floors_raw:
        raise ValueError(f"unknown keys under floors: {sorted(floors_raw)}")
    boot = raw.pop("bootstrap", {}) or {}
    metrics = {k: dict(v or {}) for k, v in (raw.pop("metrics", {}) or {}).items()}
    cfg = Config(floors=floors, metrics=metrics)
    for key in ("alpha", "margin", "min_cases", "max_error_rate", "inconclusive"):
        if key in raw:
            setattr(cfg, key, type(getattr(cfg, key))(raw.pop(key)))
    cfg.bootstrap_samples = int(boot.get("samples", cfg.bootstrap_samples))
    cfg.seed = int(boot.get("seed", cfg.seed))
    if raw:
        raise ValueError(f"unknown config keys: {sorted(raw)}")
    if cfg.inconclusive not in ("warn", "fail"):
        raise ValueError("inconclusive must be 'warn' or 'fail'")
    return cfg


@dataclass
class SliceRow:
    name: str
    cases: int
    paired: int
    baseline: float | None
    candidate: float
    delta: float | None = None
    ci: tuple[float, float] | None = None
    p: float | None = None
    broke: int = 0
    fixed: int = 0
    floor: float | None = None
    floor_ok: bool | None = None


@dataclass
class MetricRow:
    name: str
    baseline: float
    candidate: float
    change: float
    p: float
    limit: float | None
    exceeded: bool


@dataclass
class Verdict:
    status: str
    exit_code: int
    summary: str
    reasons: list[str]
    warnings: list[str]
    overall: SliceRow | None
    slices: list[SliceRow]
    metrics: list[MetricRow]
    broke: list[str]
    fixed: list[str]
    candidate_only: list[str]
    baseline_only: list[str]
    inputs: dict[str, Any]
    config: dict[str, Any]


def _mean(xs: list[float]) -> float:
    return float(sum(xs) / len(xs))


def _broke(b: Case, c: Case) -> bool:
    """Passed in every baseline trial, fails in the candidate (majority of trials)."""
    return min(b.scores) >= 0.5 and c.score < 0.5


def _fixed(b: Case, c: Case) -> bool:
    """Failed in every baseline trial, passes in the candidate (majority of trials)."""
    return max(b.scores) < 0.5 and c.score >= 0.5


def _slice_row(
    name: str,
    cand: list[Case],
    pairs: list[tuple[Case, Case]],
    cfg: Config,
    has_baseline: bool,
) -> SliceRow:
    row = SliceRow(
        name=name,
        cases=len(cand),
        paired=len(pairs),
        baseline=_mean([b.score for b, _ in pairs]) if pairs else None,
        candidate=_mean([c.score for c in cand]),
        floor=cfg.floors.get(name),
    )
    if row.floor is not None:
        row.floor_ok = row.candidate >= row.floor - 1e-12
    if has_baseline and pairs:
        d = np.array([c.score - b.score for b, c in pairs])
        row.delta = float(d.mean())
        row.ci = bootstrap_ci(d, cfg.bootstrap_samples, cfg.seed)
        row.p = drop_p_value_trials(
            [b.scores for b, _ in pairs], [c.scores for _, c in pairs], seed=cfg.seed
        )
        row.broke = sum(_broke(b, c) for b, c in pairs)
        row.fixed = sum(_fixed(b, c) for b, c in pairs)
    return row


def _pts(x: float) -> str:
    return f"{x * 100:+.1f} pts"


def _p(p: float) -> str:
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def evaluate(candidate: ResultSet, baseline: ResultSet | None, cfg: Config) -> Verdict:
    reasons: dict[str, list[str]] = {k: [] for k in EXIT}
    warnings: list[str] = []
    inputs = {
        "candidate": {
            "files": candidate.sources,
            "records": candidate.records,
            "errors": candidate.errors,
        },
        "baseline": None
        if baseline is None
        else {"files": baseline.sources, "records": baseline.records, "errors": baseline.errors},
    }

    cand = {k: c for k, c in candidate.cases.items() if c.score is not None}
    base = (
        {} if baseline is None else {k: c for k, c in baseline.cases.items() if c.score is not None}
    )
    if candidate.error_rate > cfg.max_error_rate:
        reasons["error"].append(
            f"{candidate.errors} of {candidate.records} candidate records are infrastructure "
            f"errors ({candidate.error_rate:.0%} > {cfg.max_error_rate:.0%})"
        )
    if baseline is not None and baseline.error_rate > cfg.max_error_rate:
        warnings.append(
            f"baseline has {baseline.errors} errored records; those cases are left unpaired"
        )
    if len(cand) < cfg.min_cases:
        reasons["error"].append(f"only {len(cand)} scored candidate cases (< {cfg.min_cases})")

    pairs = {k: (base[k], cand[k]) for k in cand.keys() & base.keys()}
    if baseline is not None and len(pairs) < cfg.min_cases:
        reasons["error"].append(f"only {len(pairs)} cases in both runs (< {cfg.min_cases})")
    if baseline is None:
        warnings.append("no baseline: absolute floors only")

    has_baseline = baseline is not None and bool(pairs)
    overall = slices = None
    if cand:
        overall = _slice_row(
            "overall", list(cand.values()), list(pairs.values()), cfg, has_baseline
        )
        slices = []
        for name in sorted({c.slice for c in cand.values()}):
            sc = [c for c in cand.values() if c.slice == name]
            sp = [pairs[c.case_id] for c in sc if c.case_id in pairs]
            slices.append(_slice_row(name, sc, sp, cfg, has_baseline))
        for name in sorted(set(cfg.floors) - {"overall"} - {s.name for s in slices}):
            reasons["floor"].append(f"slice {name!r} has a floor but no candidate cases")
        for row in [overall, *slices]:
            if row.floor_ok is False:
                reasons["floor"].append(
                    f"{row.name}: {row.candidate:.1%} is below the floor of {row.floor:.1%}"
                )

    metrics: list[MetricRow] = []
    if has_baseline:
        names = sorted(
            {m for _, c in pairs.values() for m in c.metrics}
            & {m for b, _ in pairs.values() for m in b.metrics}
        )
        for m in names:
            bv = [v for b, _ in pairs.values() for v in b.metrics.get(m, [])]
            cv = [v for _, c in pairs.values() for v in c.metrics.get(m, [])]
            b_mean, c_mean = _mean(bv), _mean(cv)
            change = (c_mean - b_mean) / b_mean if b_mean else float("inf") if c_mean else 0.0
            p = welch_increase_p(bv, cv)
            limit = cfg.metrics.get(m, {}).get("max_increase")
            exceeded = limit is not None and change > limit and p < cfg.alpha
            metrics.append(MetricRow(m, b_mean, c_mean, change, p, limit, exceeded))
            if exceeded:
                reasons["regression"].append(
                    f"{m}: mean rose {change:+.0%} (limit {limit:+.0%}, one-sided {_p(p)})"
                )

    summary = ""
    if overall is not None and has_baseline and overall.delta is not None:
        lo, hi = overall.ci
        span = f"95% CI {_pts(lo)} to {_pts(hi)}"
        if overall.p < cfg.alpha and overall.delta < 0:
            reasons["regression"].append(
                f"pass rate fell {-overall.delta * 100:.1f} pts ({span}, one-sided {_p(overall.p)})"
            )
            summary = (
                f"Pass rate fell from {overall.baseline:.1%} to {overall.candidate:.1%} "
                f"({_pts(overall.delta)}, {span}, {_p(overall.p)}, "
                f"{overall.paired} paired cases)."
            )
        elif lo < -cfg.margin:
            msg = (
                f"no significant drop ({_p(overall.p)}), but the interval ({_pts(lo)} to "
                f"{_pts(hi)}) does not rule out a drop larger than {cfg.margin * 100:.1f} pts"
            )
            (reasons["inconclusive"] if cfg.inconclusive == "fail" else warnings).append(msg)
            summary = f"Pass rate {overall.baseline:.1%} to {overall.candidate:.1%}: {msg}."
        else:
            summary = (
                f"Pass rate {overall.baseline:.1%} to {overall.candidate:.1%} "
                f"({_pts(overall.delta)}, {span}, {overall.paired} paired cases); a drop larger "
                f"than {cfg.margin * 100:.1f} pts is ruled out."
            )
    elif overall is not None:
        summary = f"Pass rate {overall.candidate:.1%} on {overall.cases} cases (no baseline)."

    status = next(
        (s for s in ("error", "floor", "regression", "inconclusive") if reasons[s]), "pass"
    )
    broke = sorted(k for k, (b, c) in pairs.items() if _broke(b, c))
    fixed = sorted(k for k, (b, c) in pairs.items() if _fixed(b, c))
    return Verdict(
        status=status,
        exit_code=EXIT[status],
        summary=summary,
        reasons=[r for s in ("error", "floor", "regression", "inconclusive") for r in reasons[s]],
        warnings=warnings,
        overall=overall,
        slices=slices or [],
        metrics=metrics,
        broke=broke,
        fixed=fixed,
        candidate_only=sorted(cand.keys() - base.keys()) if baseline is not None else [],
        baseline_only=sorted(base.keys() - cand.keys()),
        inputs=inputs,
        config={
            "alpha": cfg.alpha,
            "margin": cfg.margin,
            "min_cases": cfg.min_cases,
            "max_error_rate": cfg.max_error_rate,
            "inconclusive": cfg.inconclusive,
            "floors": cfg.floors,
            "metrics": cfg.metrics,
        },
    )
