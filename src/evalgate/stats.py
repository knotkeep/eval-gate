"""Paired tests on per-case scores, and Welch's t-test for continuous metrics.

The same cases run on the baseline and the candidate, so each case is its own control: the
analysis works on per-case differences d_i = candidate mean_i - baseline mean_i.

- Confidence interval: paired bootstrap of the mean difference (resample cases with replacement).
- p-value for a drop (`drop_p_value_trials`): within-case randomization test, one-sided. It is
  exact for any number of trials per run, which matters when a pooled baseline has more trials
  than the candidate. With one pass/fail trial on each side it is the exact McNemar test.
- `drop_p_value` is the sign-flip test on differences alone; it is exact only when every nonzero
  difference has the same size, and serves as the McNemar path.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

EXACT_MAX = 16  # enumerate all 2^k sign patterns up to this many nonzero differences
BOOT_CHUNK = 2000


def bootstrap_ci(
    d: np.ndarray, samples: int = 10_000, seed: int = 0, level: float = 0.95
) -> tuple[float, float]:
    """Percentile interval for the mean of d."""
    if len(d) == 0:
        return (float("nan"), float("nan"))
    if np.all(d == d[0]):
        return (float(d[0]), float(d[0]))
    rng = np.random.default_rng(seed)
    means = np.empty(samples)
    for start in range(0, samples, BOOT_CHUNK):
        k = min(BOOT_CHUNK, samples - start)
        idx = rng.integers(0, len(d), size=(k, len(d)))
        means[start : start + k] = d[idx].mean(axis=1)
    tail = (1 - level) / 2
    lo, hi = np.quantile(means, [tail, 1 - tail])
    return (float(lo), float(hi))


def drop_p_value(d: np.ndarray, samples: int = 20_000, seed: int = 0) -> float:
    """One-sided p-value for H1: mean(d) < 0."""
    nz = d[d != 0]
    k = len(nz)
    if k == 0:
        return 1.0
    mag = np.abs(nz)
    if np.allclose(mag, mag[0]):
        # Exact McNemar: count of positive differences ~ Binomial(k, 1/2) under the null.
        return float(stats.binom.cdf(int((nz > 0).sum()), k, 0.5))
    observed = nz.sum()
    if k <= EXACT_MAX:
        patterns = np.arange(2**k)[:, None] >> np.arange(k) & 1
        sums = (1 - 2 * patterns.astype(np.int8)) @ mag
        return float(np.mean(sums <= observed + 1e-12))
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(samples, k))
    sums = signs @ mag
    return float((1 + np.sum(sums <= observed + 1e-12)) / (samples + 1))


def drop_p_value_trials(
    base: list[list[float]], cand: list[list[float]], samples: int = 20_000, seed: int = 0
) -> float:
    """One-sided p-value for H1: candidate worse, from per-case trial values.

    Randomization test: under no change, a case's baseline and candidate trials come from the
    same distribution, so the case's trial values are pooled and re-split at random, keeping each
    run's trial count. The statistic is the sum over cases of (candidate mean - baseline mean).
    Exact for any trial counts; with one pass/fail trial on each side it is McNemar's test.
    """
    if all(len(b) == 1 and len(c) == 1 for b, c in zip(base, cand, strict=True)):
        return drop_p_value(np.array([c[0] - b[0] for b, c in zip(base, cand, strict=True)]))
    observed = sum(np.mean(c) - np.mean(b) for b, c in zip(base, cand, strict=True))
    rng = np.random.default_rng(seed)
    total = np.zeros(samples)
    for b, c in zip(base, cand, strict=True):
        vals = np.asarray(b + c, dtype=float)
        if np.all(vals == vals[0]):
            continue
        kb, kc = len(b), len(c)
        picks = np.argsort(rng.random((samples, kb + kc)), axis=1)[:, :kc]
        c_sum = vals[picks].sum(axis=1)
        total += c_sum / kc - (vals.sum() - c_sum) / kb
    return float((1 + np.sum(total <= observed + 1e-12)) / (samples + 1))


def welch_increase_p(baseline: list[float], candidate: list[float]) -> float:
    """One-sided Welch p-value for H1: candidate mean > baseline mean."""
    if len(baseline) < 2 or len(candidate) < 2:
        return float("nan")
    if np.var(baseline) == 0 and np.var(candidate) == 0:
        return 0.0 if np.mean(candidate) > np.mean(baseline) else 1.0
    res = stats.ttest_ind(candidate, baseline, equal_var=False, alternative="greater")
    return float(res.pvalue)
