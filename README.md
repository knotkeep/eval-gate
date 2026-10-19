# eval-gate

A CI gate for agent evaluations. It compares a candidate run (this pull request) with a baseline
(recent nightly runs) case by case, decides whether a drop is real or run-to-run noise, enforces
absolute floors, and reports the result on the pull request and in the job summary.

It does not run your agent or grade it. Your workflow produces a results file (one JSON line per
case per trial); the gate does the statistics and the reporting, so it works with any agent,
grader and runner.

## Use it

```yaml
# .github/workflows/eval.yml in the calling repo
on: pull_request
permissions:
  contents: read
  pull-requests: write
jobs:
  eval:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: ./run-golden-set.sh > results.jsonl          # your agent and grader
      - uses: knotkeep/eval-gate@v1
        with:
          candidate: results.jsonl
          baseline: baselines/night-1.jsonl baselines/night-2.jsonl baselines/night-3.jsonl
          config: eval/gate.yaml
          title: Golden set
```

The action posts one comment per `title` and updates it on later pushes. Outputs: `verdict`,
`exit-code`, `report` (path to the markdown).

Locally:

```bash
uv run evalgate compare --candidate results.jsonl --baseline night-*.jsonl \
  --config gate.yaml --markdown report.md --json verdict.json
uv run evalgate validate results.jsonl
```

## Results format

```json
{"case_id": "claim_status-003", "slice": "happy", "passed": true, "metrics": {"latency_s": 3.2, "cost_usd": 0.0}}
```

- `passed`: boolean, or a score in [0, 1].
- `slice`: any label (for example happy, edge, refusal, historical); floors can apply per slice.
- `metrics`: optional numbers compared with Welch's t-test; limits are optional.
- `error`: set it (any non-empty value) when the run failed for infrastructure reasons. Errored
  records are counted and reported but excluded from pass rates.
- Several records with the same `case_id` are trials. Several baseline files pool as trials, which
  is how a rolling baseline of the last few nightly runs works.

## Verdicts

| verdict | exit code | when |
|---|---|---|
| error | 4 | too few cases, or too many infrastructure errors |
| floor | 2 | a pass rate is below its floor (overall or a slice) |
| regression | 1 | pass rate significantly lower than the baseline, or a metric limit exceeded significantly |
| inconclusive | 3 | no significant drop, but the interval does not rule out a drop larger than `margin`; a warning (exit 0) unless `inconclusive: fail` |
| pass | 0 | none of the above |

When several apply, the first in this table wins; the report lists all of them.

## Config

```yaml
alpha: 0.05          # one-sided significance for a drop
margin: 0.10         # drop size a clean pass must rule out
min_cases: 50        # fewer scored (or paired) cases is an error
max_error_rate: 0.05 # share of candidate records with infrastructure errors
inconclusive: warn   # or fail
floors:
  overall: 0.85
  slices:
    refusal: 1.0
    historical: 1.0
metrics:
  latency_s:
    max_increase: 0.25   # relative increase in the mean
bootstrap:
  samples: 10000
  seed: 0
```

## Method

Each case's score is its pass rate over its trials. The comparison uses only cases present in both
runs, so each case is its own control.

- **Change** is the mean per-case difference, candidate minus baseline, with a paired bootstrap
  95% interval (cases resampled with replacement).
- **p** is a one-sided within-case randomization test for a drop: under "no change" a case's
  baseline and candidate trials are interchangeable, so they are pooled and re-split at random,
  keeping each run's trial count. It is exact for any number of trials. With one pass/fail trial
  on each side it is McNemar's test, computed exactly from the binomial distribution.
- **Broke** lists cases that passed in every baseline run and fail now; **fixed** the reverse.
  Flaky cases appear in neither list.
- **Metric limits** use a one-sided Welch t-test on the trial values.

Measured by simulation on 100 cases (85% stable at 97%, 15% flaky at 30-80%), one candidate trial:

| baseline | false alarms (no change) | 10 cases broken: caught | 5 cases broken: caught |
|---|---|---|---|
| one nightly run | 2.4% | 63% | 20% |
| three nightly runs pooled | 3.1% | 87% | 35% |

(1,000 simulated decisions without a change, 600 with; alpha 0.05.) Small breaks are beyond what
100 cases can detect statistically: a slice floor (for example refusals at 100%) is the right tool
for cases that must never break, and the broke list names every stable case that failed.
`tests/test_stats.py` checks the false-alarm rate and the effect of pooling.

## Using it from other private repos

The repo is private. Callers in the knotkeep organization need this repo's Settings, Actions,
General, Access set to "Accessible from repositories in the knotkeep organization".

## Development

```bash
uv sync --all-groups
uv run pytest
uv run python examples/make_examples.py   # regenerate the example runs
```
