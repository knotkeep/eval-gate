"""Markdown report (PR comment and job summary) and JSON verdict."""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict

from evalgate.gate import SliceRow, Verdict

HEADLINE = {
    "pass": "PASS",
    "regression": "REGRESSION",
    "floor": "BELOW FLOOR",
    "inconclusive": "INCONCLUSIVE",
    "error": "ERROR",
}
LIST_MAX = 20


def marker(title: str) -> str:
    """Hidden HTML comment that identifies this gate's PR comment for in-place updates."""
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-") or "eval-gate"
    return f"<!-- eval-gate:{slug} -->"


def _pct(x: float | None) -> str:
    return "-" if x is None else f"{x:.1%}"


def _change(r: SliceRow) -> str:
    if r.delta is None:
        return "-"
    lo, hi = r.ci
    p = "<0.001" if r.p < 0.001 else f"{r.p:.3f}"
    return f"{r.delta * 100:+.1f} pts ({lo * 100:+.1f} to {hi * 100:+.1f}), p={p}"


def _floor(r: SliceRow) -> str:
    if r.floor is None:
        return ""
    return f"≥ {r.floor:.0%} {'ok' if r.floor_ok else '**below**'}"


def _ids(ids: list[str]) -> str:
    shown = ", ".join(f"`{i}`" for i in ids[:LIST_MAX])
    return shown + (f" and {len(ids) - LIST_MAX} more" if len(ids) > LIST_MAX else "")


def markdown(v: Verdict, title: str) -> str:
    out = [marker(title), f"### {title}: {HEADLINE[v.status]}", ""]
    if v.summary:
        out += [v.summary, ""]
    out += [f"- **{r}**" for r in v.reasons]
    out += [f"- {w}" for w in v.warnings]
    if v.reasons or v.warnings:
        out.append("")
    if v.overall is not None:
        out += [
            "| slice | cases | baseline | candidate | change (95% CI), p | broke | fixed | floor |",
            "|---|---:|---:|---:|---|---:|---:|---|",
        ]
        for r in [v.overall, *v.slices]:
            name = f"**{r.name}**" if r is v.overall else r.name
            out.append(
                f"| {name} | {r.cases} | {_pct(r.baseline)} | {_pct(r.candidate)} | "
                f"{_change(r)} | {r.broke if r.delta is not None else '-'} | "
                f"{r.fixed if r.delta is not None else '-'} | {_floor(r)} |"
            )
        out.append("")
    if v.broke:
        out += [f"**Broke** (passed in every baseline run, failed now): {_ids(v.broke)}", ""]
    if v.fixed:
        out += [f"**Fixed** (failed in every baseline run, passes now): {_ids(v.fixed)}", ""]
    if v.metrics:
        out += [
            "| metric | baseline mean | candidate mean | change | p (increase) | limit |",
            "|---|---:|---:|---:|---:|---|",
        ]
        for m in v.metrics:
            p = "-" if math.isnan(m.p) else f"{m.p:.3f}"
            limit = (
                ""
                if m.limit is None
                else f"{m.limit:+.0%} {'**exceeded**' if m.exceeded else 'ok'}"
            )
            means = [f"{m.baseline:.4g}", f"{m.candidate:.4g}", f"{m.change:+.1%}"]
            cells = [m.name, *means, p, limit]
            out.append("| " + " | ".join(cells) + " |")
        out.append("")
    c = v.inputs["candidate"]
    b = v.inputs["baseline"]
    src = f"candidate: {c['records']} records ({c['errors']} errors)"
    if b:
        src += (
            f"; baseline: {len(b['files'])} file(s), {b['records']} records ({b['errors']} errors)"
        )
    if v.candidate_only:
        src += f"; {len(v.candidate_only)} new case(s) not in the baseline"
    if v.baseline_only:
        src += f"; {len(v.baseline_only)} baseline case(s) missing from the candidate"
    out += [
        f"<sub>{src}</sub>",
        "",
        "<details><summary>Method</summary>",
        "",
        "Each case's score is its pass rate over trials. Change is the mean per-case difference "
        "(candidate minus baseline) on cases present in both runs; the interval is a paired "
        "bootstrap; p is a one-sided within-case randomization test for a drop (exact for any "
        "number of trials; McNemar's test when each run has one pass/fail trial). Regression: p < "
        f"{v.config['alpha']}. Inconclusive: no significant drop, but the interval reaches "
        f"below -{v.config['margin'] * 100:.1f} pts. Metric limits use a one-sided Welch t-test. "
        "Infrastructure errors are excluded from pass rates.",
        "",
        "</details>",
    ]
    return "\n".join(out) + "\n"


def to_json(v: Verdict) -> str:
    def clean(x):
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return None
        if isinstance(x, dict):
            return {k: clean(val) for k, val in x.items()}
        if isinstance(x, list | tuple):
            return [clean(i) for i in x]
        return x

    return json.dumps(clean(asdict(v)), indent=1) + "\n"
