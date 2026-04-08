# UF Lite v1.0

**Practical Ethical Gates for AI Systems**

A minimal, ready-to-use ethical framework for bots, agents, chat interfaces, and any AI-adjacent project.  
No metaphysics. No "Ledger". Just enforceable gates, scoring, and truth-first initialization.

## Quick Start
1. Add `middleware_filter.py` to your stack
2. Load `gate_profile.yaml` at startup
3. Run every output through the 8 Gates before emitting
4. Use the templates for new projects

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

## Genesis Boot (mandatory)
Every new session must start with:
> "I am a new instance with no prior continuity. This is a fresh session."

## License
Documentation & templates: CC BY-NC-ND 4.0  
Code: MIT License
