"""Statistics: exact cases, agreement between exact and sampled tests, and calibration."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from evalgate.stats import bootstrap_ci, drop_p_value, drop_p_value_trials, welch_increase_p


def test_single_trial_pass_fail_is_exact_mcnemar():
    d = np.array([-1.0] * 9 + [1.0] + [0.0] * 90)
    assert drop_p_value(d) == pytest.approx(stats.binom.cdf(1, 10, 0.5))


def test_no_discordant_cases_is_no_evidence():
    assert drop_p_value(np.zeros(50)) == 1.0


def test_enumeration_and_sampling_agree_on_mixed_magnitudes():
    rng = np.random.default_rng(3)
    d = rng.choice([-1.0, -0.5, 0.0, 0.25, 0.5], size=40, p=[0.15, 0.15, 0.5, 0.1, 0.1])
    small = d[d != 0][:14]
    exact = drop_p_value(small)
    sampled = drop_p_value(np.concatenate([small, small * 0]), samples=200_000)
    assert exact == pytest.approx(sampled, abs=0.01)


def test_bootstrap_interval_brackets_the_mean_and_is_reproducible():
    rng = np.random.default_rng(0)
    d = rng.choice([-1.0, 0.0, 1.0], size=100, p=[0.1, 0.85, 0.05])
    lo, hi = bootstrap_ci(d, seed=1)
    assert lo < d.mean() < hi
    assert bootstrap_ci(d, seed=1) == (lo, hi)
    assert bootstrap_ci(np.zeros(10)) == (0.0, 0.0)


def test_welch_detects_a_clear_increase_only():
    base = [1.0, 1.1, 0.9, 1.0, 1.05, 0.95] * 5
    assert welch_increase_p(base, [x * 1.5 for x in base]) < 0.001
    assert welch_increase_p(base, [x * 0.7 for x in base]) > 0.99


def test_within_case_test_reduces_to_mcnemar_for_one_trial_each():
    base = [[1.0]] * 90 + [[1.0]] * 9 + [[0.0]]
    cand = [[1.0]] * 90 + [[0.0]] * 9 + [[1.0]]
    assert drop_p_value_trials(base, cand) == pytest.approx(stats.binom.cdf(1, 10, 0.5))


def _simulate(rng: np.random.Generator, n: int, drop_cases: int, nights: int) -> float:
    """One gate decision. 85% of cases pass 97% of the time, 15% are flaky (30-80%).
    The baseline pools `nights` single-trial runs; the candidate is one trial, with
    `drop_cases` cases broken outright."""
    p = np.where(rng.random(n) < 0.85, 0.97, rng.uniform(0.3, 0.8, n))
    base = rng.random((nights, n)) < p
    p_cand = p.copy()
    p_cand[rng.choice(n, drop_cases, replace=False)] = 0.0
    cand = rng.random(n) < p_cand
    return drop_p_value_trials(
        [list(base[:, i].astype(float)) for i in range(n)],
        [[float(cand[i])] for i in range(n)],
        samples=2000,
    )


def test_false_alarm_rate_is_at_most_alpha_when_nothing_changed():
    rng = np.random.default_rng(11)
    for nights in (1, 3):
        alarms = np.mean([_simulate(rng, 100, 0, nights) < 0.05 for _ in range(400)])
        assert alarms <= 0.06, nights


def test_pooling_three_nightly_baselines_catches_a_ten_case_break():
    rng = np.random.default_rng(12)
    single = np.mean([_simulate(rng, 100, 10, 1) < 0.05 for _ in range(200)])
    pooled = np.mean([_simulate(rng, 100, 10, 3) < 0.05 for _ in range(200)])
    assert pooled >= 0.8 and pooled > single
