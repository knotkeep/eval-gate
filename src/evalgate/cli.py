"""evalgate compare | validate."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from evalgate.gate import EXIT, evaluate, load_config
from evalgate.records import RecordError, load
from evalgate.report import markdown, to_json


def _compare(args: argparse.Namespace) -> int:
    try:
        cfg = load_config(args.config)
        candidate = load(args.candidate, args.score_field)
        baseline = load(args.baseline, args.score_field) if args.baseline else None
    except (RecordError, ValueError, OSError) as e:
        print(f"evalgate: {e}", file=sys.stderr)
        return EXIT["error"]
    verdict = evaluate(candidate, baseline, cfg)
    md = markdown(verdict, args.title)
    print(md)
    if args.markdown:
        Path(args.markdown).write_text(md)
    if args.json:
        Path(args.json).write_text(to_json(verdict))
    out = os.environ.get("GITHUB_OUTPUT")
    if args.github_output and out:
        with open(out, "a") as f:
            f.write(f"verdict={verdict.status}\nexit-code={verdict.exit_code}\n")
    return verdict.exit_code


def _validate(args: argparse.Namespace) -> int:
    try:
        rs = load(args.files)
    except RecordError as e:
        print(f"evalgate: {e}", file=sys.stderr)
        return EXIT["error"]
    slices = sorted({c.slice for c in rs.cases.values()})
    print(f"{rs.records} records, {len(rs.cases)} cases, {rs.errors} errors, slices: {slices}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="evalgate", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compare", help="gate a candidate run against a baseline")
    c.add_argument("--candidate", nargs="+", required=True, help="candidate results (JSONL)")
    c.add_argument("--baseline", nargs="*", default=[], help="baseline results; pooled as trials")
    c.add_argument("--config", default=None, help="gate.yaml")
    c.add_argument("--title", default="Eval gate")
    c.add_argument("--score-field", default="passed", help="record field to gate on")
    c.add_argument("--markdown", default=None, help="write the report here")
    c.add_argument("--json", default=None, help="write the verdict as JSON here")
    c.add_argument("--github-output", action="store_true", help="append outputs to $GITHUB_OUTPUT")
    c.set_defaults(func=_compare)
    v = sub.add_parser("validate", help="check results files")
    v.add_argument("files", nargs="+")
    v.set_defaults(func=_validate)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
