"""Generate the example runs: three baseline nights, an unchanged candidate, a regressed one.

100 cases: 60 happy path, 20 edge, 10 refusal, 10 historical failures. 85% of happy and edge
cases pass 97% of the time and 15% are flaky (40-80%); refusal and historical cases always pass.
The regressed candidate breaks 12 happy/edge cases outright. Without a real change, about 3% of
seeds give an unchanged candidate that the gate flags at alpha 0.05; the seed below gives a
typical one.

    uv run python examples/make_examples.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SEED = 1
HERE = Path(__file__).parent
SLICES = ["happy"] * 60 + ["edge"] * 20 + ["refusal"] * 10 + ["historical"] * 10


def write(name: str, p: np.ndarray, rng: np.random.Generator) -> None:
    passed = rng.random(len(p)) < p
    latency = rng.normal(3.0, 0.6, len(p)).round(2)
    with (HERE / f"{name}.jsonl").open("w") as f:
        for i, sl in enumerate(SLICES):
            rec = {"case_id": f"{sl}-{i:03d}", "slice": sl, "passed": bool(passed[i])}
            rec["metrics"] = {"latency_s": float(latency[i])}
            f.write(json.dumps(rec) + "\n")


def main() -> None:
    rng = np.random.default_rng(SEED)
    p = np.where(rng.random(100) < 0.85, 0.97, rng.uniform(0.4, 0.8, 100))
    p[80:] = 1.0
    for night in (1, 2, 3):
        write(f"baseline-night{night}", p, rng)
    write("candidate-ok", p, rng)
    broken = p.copy()
    broken[rng.choice(80, 12, replace=False)] = 0.0
    write("candidate-regressed", broken, rng)


if __name__ == "__main__":
    main()
