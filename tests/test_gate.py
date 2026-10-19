"""Verdicts and exit codes on constructed result sets."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evalgate.cli import main
from evalgate.gate import Config, evaluate, load_config
from evalgate.records import RecordError, load
from evalgate.report import marker

SLICES = ["happy"] * 60 + ["edge"] * 20 + ["refusal"] * 10 + ["historical"] * 10


def _write(path: Path, passed: list[bool], latency: float = 2.0, errors: set[int] = frozenset()):
    with path.open("w") as f:
        for i, (ok, sl) in enumerate(zip(passed, SLICES, strict=True)):
            rec = {"case_id": f"case-{i:03d}", "slice": sl, "passed": ok}
            rec["metrics"] = {"latency_s": latency + (i % 7) * 0.1}
            if i in errors:
                rec = {"case_id": rec["case_id"], "slice": sl, "error": "timeout"}
            f.write(json.dumps(rec) + "\n")
    return path


def _base(fail: tuple[int, ...] = (5, 17, 33)) -> list[bool]:
    return [i not in fail for i in range(100)]


@pytest.fixture
def cfg() -> Config:
    return Config(
        floors={"overall": 0.85, "refusal": 1.0}, metrics={"latency_s": {"max_increase": 0.25}}
    )


def test_identical_runs_pass(tmp_path, cfg):
    b = load([_write(tmp_path / "b.jsonl", _base())])
    c = load([_write(tmp_path / "c.jsonl", _base())])
    v = evaluate(c, b, cfg)
    assert (v.status, v.exit_code) == ("pass", 0)
    assert v.overall.delta == 0 and v.broke == [] and "ruled out" in v.summary


def test_ten_broken_cases_are_a_regression(tmp_path, cfg):
    cand = _base()
    for i in range(40, 50):
        cand[i] = False
    v = evaluate(
        load([_write(tmp_path / "c.jsonl", cand)]),
        load([_write(tmp_path / "b.jsonl", _base())]),
        cfg,
    )
    assert (v.status, v.exit_code) == ("regression", 1)
    assert v.overall.p < 0.01 and len(v.broke) == 10


def test_floor_takes_precedence_over_regression(tmp_path, cfg):
    cand = _base()
    for i in [*range(40, 50), 85]:  # 85 is a refusal case
        cand[i] = False
    v = evaluate(
        load([_write(tmp_path / "c.jsonl", cand)]),
        load([_write(tmp_path / "b.jsonl", _base())]),
        cfg,
    )
    assert (v.status, v.exit_code) == ("floor", 2)
    assert any("refusal" in r for r in v.reasons) and any("fell" in r for r in v.reasons)


def test_infrastructure_errors_are_excluded_then_gate_errors_out(tmp_path, cfg):
    b = load([_write(tmp_path / "b.jsonl", _base())])
    few = load([_write(tmp_path / "c1.jsonl", _base(), errors={1, 2})])
    assert evaluate(few, b, cfg).status == "pass"
    assert few.cases["case-001"].score is None
    many = load([_write(tmp_path / "c2.jsonl", _base(), errors=set(range(10)))])
    v = evaluate(many, b, cfg)
    assert (v.status, v.exit_code) == ("error", 4)


def test_small_drop_on_few_cases_is_inconclusive(tmp_path):
    rows = [{"case_id": f"c{i}", "slice": "all", "passed": True} for i in range(20)]
    (tmp_path / "b.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    rows[0]["passed"] = rows[1]["passed"] = False
    (tmp_path / "c.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    b, c = load([tmp_path / "b.jsonl"]), load([tmp_path / "c.jsonl"])
    warn = evaluate(c, b, Config())
    assert (warn.status, warn.exit_code) == ("pass", 0) and warn.warnings
    fail = evaluate(c, b, Config(inconclusive="fail"))
    assert (fail.status, fail.exit_code) == ("inconclusive", 3)


def test_latency_limit_is_a_regression(tmp_path, cfg):
    b = load([_write(tmp_path / "b.jsonl", _base(), latency=2.0)])
    c = load([_write(tmp_path / "c.jsonl", _base(), latency=3.0)])
    v = evaluate(c, b, cfg)
    assert v.status == "regression" and v.metrics[0].exceeded


def test_several_baseline_files_pool_as_trials(tmp_path, cfg):
    b1 = _write(tmp_path / "b1.jsonl", _base(fail=(5,)))
    b2 = _write(tmp_path / "b2.jsonl", _base(fail=(5, 6)))
    pooled = load([b1, b2])
    assert pooled.cases["case-006"].scores == [1.0, 0.0]
    v = evaluate(load([_write(tmp_path / "c.jsonl", _base(fail=(5,)))]), pooled, cfg)
    assert v.status == "pass" and v.fixed == []


def test_no_baseline_checks_floors_only(tmp_path, cfg):
    v = evaluate(load([_write(tmp_path / "c.jsonl", _base())]), None, cfg)
    assert v.status == "pass" and v.overall.delta is None and "no baseline" in v.warnings[0]


def test_malformed_records_are_rejected(tmp_path):
    (tmp_path / "x.jsonl").write_text('{"case_id": "a", "passed": 2}\n')
    with pytest.raises(RecordError, match="passed"):
        load([tmp_path / "x.jsonl"])
    (tmp_path / "y.jsonl").write_text(
        '{"case_id": "a", "slice": "s1", "passed": true}\n'
        '{"case_id": "a", "slice": "s2", "passed": true}\n'
    )
    with pytest.raises(RecordError, match="slice"):
        load([tmp_path / "y.jsonl"])


def test_config_file_round_trip(tmp_path):
    (tmp_path / "gate.yaml").write_text(
        "alpha: 0.01\nmargin: 0.03\ninconclusive: fail\n"
        "floors:\n  overall: 0.9\n  slices:\n    refusal: 1.0\n"
        "metrics:\n  latency_s:\n    max_increase: 0.2\nbootstrap:\n  samples: 500\n"
    )
    cfg = load_config(tmp_path / "gate.yaml")
    assert cfg.alpha == 0.01 and cfg.floors == {"refusal": 1.0, "overall": 0.9}
    assert cfg.bootstrap_samples == 500 and cfg.metrics["latency_s"]["max_increase"] == 0.2
    (tmp_path / "bad.yaml").write_text("alpah: 0.05\n")
    with pytest.raises(ValueError, match="alpah"):
        load_config(tmp_path / "bad.yaml")


def test_cli_exit_codes_and_report(tmp_path, cfg):
    b = _write(tmp_path / "b.jsonl", _base())
    cand = _base()
    for i in range(40, 50):
        cand[i] = False
    c = _write(tmp_path / "c.jsonl", cand)
    md, js = tmp_path / "out.md", tmp_path / "out.json"
    code = main(
        [
            "compare",
            "--candidate",
            str(c),
            "--baseline",
            str(b),
            "--title",
            "Golden set",
            "--markdown",
            str(md),
            "--json",
            str(js),
        ]
    )
    assert code == 1
    text = md.read_text()
    assert text.startswith(marker("Golden set")) and "REGRESSION" in text and "`case-040`" in text
    assert json.loads(js.read_text())["status"] == "regression"
    assert main(["compare", "--candidate", str(tmp_path / "missing.jsonl")]) == 4


def test_score_field_gates_another_verdict_and_skips_records_without_it(tmp_path):
    rows = [
        {"case_id": f"c{i}", "slice": "s", "passed": True, "judge_passed": i != 3}
        for i in range(30)
    ]
    rows.append({"case_id": "seed", "slice": "s", "passed": True})
    (tmp_path / "r.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    rules = load([tmp_path / "r.jsonl"])
    judges = load([tmp_path / "r.jsonl"], field="judge_passed")
    assert len(rules.cases) == 31 and len(judges.cases) == 30
    assert judges.cases["c3"].scores == [0.0] and rules.cases["c3"].scores == [1.0]
