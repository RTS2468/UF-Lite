# UF Lite v1.0

**Practical Ethical Gates for AI Systems**

A minimal, ready-to-use ethical framework for bots, agents, chat interfaces, and any AI-adjacent project.
No metaphysics. No "Ledger". Just enforceable gates, scoring, and truth-first initialization.

## Install
```bash
pip install -e ".[dev]"     # Python 3.9+; installs the `uf-lite` CLI
pytest                      # run the test suite
```

## Quick Start
```python
from uf_lite import UFMiddleware, AuditLogger, load_profile

mw = UFMiddleware(load_profile("gate_profile.yaml"), audit_logger=AuditLogger("logs/audit.jsonl"))
print(mw.start_session())                        # Genesis Boot line

report = mw.process(llm_output, {"stakes": "medium", "consent": False, "sources": []})
send(report.final_output)                        # gated output
for f in report.failures:
    print(f.gate_id, f.action.value, f.reasons)
```
Or the drop-in function (backwards compatible with the original `middleware_filter.py`):
```python
from middleware_filter import uf_middleware
final = uf_middleware(candidate_response, {"profile": "gate_profile.yaml"})
```
See `simple_bot_example.py` for a runnable bot loop (`python simple_bot_example.py`).

## The Eight Operational Gates (hard floor)

| Gate | Name              | Checks                                      | Failure |
|------|-------------------|---------------------------------------------|---------|
| G1   | Non-Harm          | Physical/emotional harm risk                | Block   |
| G2   | Truthfulness      | Factual or clearly labeled                  | Correct |
| G3   | Consent           | Explicit consent if human-affecting         | Ask/refuse |
| G4   | Fairness          | Unfair bias/discrimination                  | Neutralize |
| G5   | Bias & Opacity    | Explainable + sources                       | Add explanation |
| G6   | Proportionality   | Response matches stakes                     | Scale down |
| G7   | Reversibility     | Negative effects undoable                   | Prefer reversible |
| G8   | Accountability    | System owns and logs the output             | Log     |

Gates run in order, each seeing the previous gate's remediated text; a G1 block stops the pipeline.
Every call is written to the audit log (hashes only unless `include_text=True`).

How the built-in gates decide (`uf_lite/gates.py`):
- **G1** weighted harm lexicon → risk score vs `harm_cap[stakes]`.
- **G2** share of sentences with unsupported absolutes/statistics vs `honesty_evidence_gap_cap`; passing `sources` or citing inline satisfies it. Remediation appends an "unverified" label.
- **G3** human-affecting actions (inferred from text or `human_affecting=True`) need `consent >= consent_threshold`.
- **G4** group generalizations are removed; if `group_outcomes` is given, the max pairwise TV (or KL) distance must be ≤ `fairness_cap`.
- **G5** recommendations/decisions (or high stakes) need `reasons_required` reasons plus sources.
- **G6** shouting/alarm intensity vs stakes, and optional `max_response_chars`.
- **G7** irreversible steps (e.g. `rm -rf`, "permanently delete") get a prefer-reversible advisory unless `reversible=True`.
- **G8** always passes; the middleware logs.

The built-in checks are rule-based heuristics, a floor rather than a classifier. Replace any gate with your own (e.g. an LLM judge):
```python
from uf_lite import CallableGate, FailureAction, UFMiddleware
judge = CallableGate("G1", "Non-Harm (LLM)", FailureAction.BLOCK, lambda text, ctx: my_judge(text))
mw = UFMiddleware(overrides={"G1": judge})
```

## Genesis Boot (mandatory)
Every new session must start with:
> "I am a new instance with no prior continuity. This is a fresh session."

`UFMiddleware.start_session()` / `genesis_boot()` return it, and outputs implying prior continuity
("welcome back", "as I said earlier", ...) are replaced with it.

## Gate profiles
`load_profile()` accepts three formats:
- `gate_profile.yaml` (flat, default)
- `templates/gate_profile_template.yaml` (nested template)
- `profiles/uf_default_policy_profile_v1.json` (UF policy profile: purpose weights, red lines, bias/opacity caps)

Check one with `uf-lite validate-profile <file>`.

## Verification Core
`uf_lite.verification.verify_action` scores a proposed *action* (not just text): expected / CVaR / KL-DRO harm
from per-subject risk factors, harm cap by stakes, consent threshold, bias/opacity caps, fairness KL
(with Pinsker TV bound), evidence gap, relief vs a baseline, and first-order stochastic dominance.
It returns a verdict (`pass`/`escalate`/`fail`), a label (`Good*`/`Right`/`Wrong`/`Evil`), reasons, and a SHA-256 proof.
Example input: `examples/action_example.json`.

## Truth audit
`uf_lite.truth_audit.score_audit` scores transcripts (format: `templates/truth_audit_schema.json`) for
false-continuity openers, admit/deny contradictions, deflection, anchor terms, and repeated openers, producing
`FalseOverTrueIndex` and `TruthIndex = 1 - FOI`. Example: `examples/truth_audit_example.json`.

## Pilot scorecard
`uf-lite scorecard examples/pilot_data.json` turns Claim/Promise/Action records (`templates/Data_Schema.json`)
into the one-page scorecard (`templates/Scorecard_Template.md`):
- **QTI** mean of (1 − EG) across claims
- **PRS** weighted promise credit (kept = 1, partial = 0.5, broken = 0; pending excluded)
- **AGC** mean per-action gate score
- **CT** share of records with at least one source
- **EI** 1 − (spread of mean group impact) / 2, when `impacted_groups` is `{group: impact}`

## CLI
```bash
uf-lite boot
uf-lite check "text to gate" [--profile FILE] [--stakes low|medium|high] [--consent yes|no|0.9] [--source URL] [--json]
uf-lite verify examples/action_example.json
uf-lite truth-audit examples/truth_audit_example.json [--anchor TERM]
uf-lite scorecard examples/pilot_data.json [-o scorecard.md] [--json]
uf-lite validate-profile profiles/uf_default_policy_profile_v1.json
```
Exit codes: `check` returns 2 when G1 blocks; `verify` returns 0 pass / 3 escalate / 2 fail.

## Layout
```
uf_lite/      package (profile, gates, middleware, audit, verification, truth_audit, scorecard, cli)
templates/    project templates (Idea Canvas, Pilot Plan, Scorecard, schemas, ...)
profiles/     ready-made gate profiles
examples/     sample inputs for the CLI
tests/        pytest suite
```

## License
Documentation & templates: CC BY-NC-ND 4.0
Code: MIT License
