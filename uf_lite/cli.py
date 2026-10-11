"""Command line interface: ``uf-lite boot|check|verify|truth-audit|scorecard|validate-profile``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .audit import AuditLogger
from .middleware import UFMiddleware, genesis_boot
from .profile import (
    STAKES_LEVELS,
    ProfileError,
    conformance_report,
    load_profile,
    read_profile_data,
)
from .scorecard import compute_scorecard, load_dataset, render_markdown
from .truth_audit import load_audit, score_audit
from .verification import verify_from_dict


def _consent(value: str) -> bool | float:
    lowered = value.lower()
    if lowered in ("yes", "true", "y"):
        return True
    if lowered in ("no", "false", "n"):
        return False
    return float(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="uf-lite", description="UF Lite ethical gates")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("boot", help="print the Genesis Boot statement")

    check = sub.add_parser("check", help="run text through the 8 gates")
    check.add_argument("text", nargs="?", help="text to check (default: read stdin)")
    check.add_argument("--profile", help="path to a gate profile YAML")
    check.add_argument("--stakes", choices=STAKES_LEVELS)
    check.add_argument("--consent", type=_consent, help="yes/no or a number in [0, 1]")
    check.add_argument("--source", action="append", default=[], dest="sources")
    check.add_argument("--reason", action="append", default=[], dest="reasons")
    check.add_argument("--audit-log", help="append audit records to this JSONL file")
    check.add_argument("--json", action="store_true", help="print the full gate report")

    card = sub.add_parser("scorecard", help="render a pilot scorecard from JSON data")
    card.add_argument("data", help="dataset JSON (see examples/pilot_data.json)")
    card.add_argument("-o", "--output", help="write Markdown here instead of stdout")
    card.add_argument("--json", action="store_true", help="print metrics as JSON")

    verify = sub.add_parser("verify", help="Verification Core report for a proposed action")
    verify.add_argument("data", help="action JSON (see examples/action_example.json)")
    verify.add_argument("--profile", help="path to a gate profile (YAML or JSON)")

    audit = sub.add_parser("truth-audit", help="score transcripts for truthfulness")
    audit.add_argument("data", help="audit JSON (see templates/truth_audit_schema.json)")
    audit.add_argument("--anchor", action="append", dest="anchors", help="anchor term to track")

    validate = sub.add_parser("validate-profile", help="conformance report for a gate profile")
    validate.add_argument("profile", help="profile YAML or JSON")
    return parser


def _load_json(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "boot":
        print(genesis_boot())
        return 0

    if args.command == "check":
        text = args.text if args.text is not None else sys.stdin.read()
        try:
            profile = load_profile(args.profile)
        except (ProfileError, OSError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        mw = UFMiddleware(profile, audit_logger=AuditLogger(args.audit_log))
        context = {"sources": args.sources, "reasons": args.reasons}
        if args.stakes:
            context["stakes"] = args.stakes
        if args.consent is not None:
            context["consent"] = args.consent
        report = mw.process(text, context)
        if args.json:
            print(json.dumps(report.to_dict(), indent=2))
        else:
            print(report.final_output)
            for r in report.failures:
                print(
                    f"[{r.gate_id} {r.name}] {r.action.value}: {'; '.join(r.reasons)}",
                    file=sys.stderr,
                )
        return 2 if report.blocked else 0

    if args.command == "scorecard":
        try:
            card = compute_scorecard(load_dataset(args.data))
        except (ValueError, OSError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        out = json.dumps(card.metrics(), indent=2) if args.json else render_markdown(card)
        if args.output:
            Path(args.output).write_text(out, encoding="utf-8")
        else:
            print(out, end="" if out.endswith("\n") else "\n")
        return 0

    if args.command == "verify":
        try:
            report = verify_from_dict(_load_json(args.data), load_profile(args.profile))
        except (ValueError, TypeError, KeyError, OSError, ProfileError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(json.dumps(report.to_dict(), indent=2))
        return {"pass": 0, "escalate": 3}.get(report.verdict, 2)

    if args.command == "truth-audit":
        try:
            result = score_audit(load_audit(args.data), args.anchors)
        except (ValueError, OSError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(json.dumps(result, indent=2))
        return 0

    if args.command == "validate-profile":
        try:
            report = conformance_report(read_profile_data(args.profile))
        except (ProfileError, OSError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(json.dumps(report, indent=2))
        return 0 if report["valid"] else 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
