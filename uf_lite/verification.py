"""Verification Core: quantitative verification of a proposed action.

Ported from the UF v1.1 reference implementation (verification_core_v1_1.py)
without the ledger. Caps come from the :class:`GateProfile`.

- Harm aggregation: ``expected`` (default), ``cvar`` (CVaR_alpha over loss
  scenarios), or ``kl_dro`` (expected + lambda * KL(claim || evidence)).
- Fairness: KL cap, with the Pinsker bound TV <= sqrt(KL / 2) reported.
- First-order stochastic dominance shortcut against baseline losses.
- Thresholded Right: relief index >= RI_MIN and harm <= harm cap for the stakes.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from .profile import STAKES_ALIASES, GateProfile

VERIFY_VERSION = "1.1.0-lite"


def _check_unit(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value!r}")


@dataclass
class EvidenceRef:
    uri: str
    hash: str | None = None


@dataclass
class SubjectRisk:
    """Risk to one affected subject; every factor is in [0, 1]."""

    p: float  # probability of adverse outcome
    s: float  # severity
    d: float  # duration
    r: float  # irreversibility
    k: float  # due-care quality (1 = best)
    consent: float  # informed consent
    w: float  # scope/need weight

    def __post_init__(self) -> None:
        for name in ("p", "s", "d", "r", "k", "consent", "w"):
            _check_unit(name, getattr(self, name))


@dataclass
class ActionProposal:
    action_id: str
    actor_id: str
    description: str
    timestamp: str
    stakes: str = "medium"
    evidence: list[EvidenceRef] = field(default_factory=list)
    subjects: list[SubjectRisk] = field(default_factory=list)
    loss_scenarios: list[float] | None = None
    baseline_loss_scenarios: list[float] | None = None

    def __post_init__(self) -> None:
        self.stakes = STAKES_ALIASES.get(self.stakes, self.stakes)
        self.evidence = [
            e if isinstance(e, EvidenceRef) else EvidenceRef(**e) for e in self.evidence
        ]
        self.subjects = [
            s if isinstance(s, SubjectRisk) else SubjectRisk(**s) for s in self.subjects
        ]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ActionProposal:
        return cls(**data)


@dataclass
class Signals:
    """Model-side measurements. R/D give alignment A = R / (R + D)."""

    R: float = 1.0
    D: float = 0.0
    E: float | None = None  # ethical compliance; default 1 - Harm * Bias * Opacity
    bias: float = 0.0
    opacity: float = 0.0
    fairness_kl: float | None = None  # KL(P_out || P_fair)
    claim_evid_kl: float | None = None  # KL(P_claim || P_evid)
    passion: float = 0.7


@dataclass
class VerifierConfig:
    EPS: float = 1e-12
    DELTA: float = 1e-18
    E_MIN: float = 0.999
    PASSION_CAP: float = 0.7
    RISK_MODE: str = "expected"  # "expected" | "cvar" | "kl_dro"
    CVAR_ALPHA: float = 0.95
    DRO_LAMBDA: float = 0.1
    RI_MIN: float = 0.0


@dataclass
class VerificationReport:
    verdict: str  # "pass" | "fail" | "escalate"
    action_label: str  # "Good*" | "Right" | "Wrong" | "Evil"
    severity: str  # "S0" | "S1" | "S2"
    metrics: dict[str, Any]
    reasons: list[str]
    proof: str  # SHA-256 of the canonical inputs and result
    version: str
    timestamp: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def expected_harm(subjects: list[SubjectRisk]) -> float:
    total = sum(s.w * s.p * s.s * s.d * s.r * (1 - s.k) * (1 - s.consent) for s in subjects)
    return max(0.0, min(1.0, 1.0 - math.exp(-total)))


def cvar(losses: list[float], alpha: float) -> float:
    """CVaR_alpha (mean of the worst 1 - alpha tail) for losses in [0, 1].

    The boundary sample contributes only its fractional share of the tail mass.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if not losses:
        return 0.0
    xs = sorted((max(0.0, min(1.0, x)) for x in losses), reverse=True)
    tail_mass = (1.0 - alpha) * len(xs)
    if tail_mass <= 1e-12:
        return xs[0]
    total, remaining = 0.0, tail_mass
    for x in xs:
        weight = min(1.0, remaining)
        total += weight * x
        remaining -= weight
        if remaining <= 1e-12:
            break
    return total / tail_mass


def fsd_dominates(loss_a: list[float], loss_b: list[float]) -> bool:
    """True if losses ``a`` first-order stochastically dominate ``b`` (a is better)."""
    if not loss_a or not loss_b:
        return False

    def cdf(samples: list[float], t: float) -> float:
        return sum(1 for v in samples if v <= t) / len(samples)

    return all(cdf(loss_a, t) >= cdf(loss_b, t) - 1e-12 for t in sorted(set(loss_a + loss_b)))


def robust_harm(action: ActionProposal, signals: Signals, cfg: VerifierConfig) -> float:
    if cfg.RISK_MODE == "cvar" and action.loss_scenarios:
        return cvar(action.loss_scenarios, cfg.CVAR_ALPHA)
    base = expected_harm(action.subjects)
    if cfg.RISK_MODE == "kl_dro":
        kl = signals.claim_evid_kl or 0.0
        return max(0.0, min(1.0, base + cfg.DRO_LAMBDA * kl))
    return base


def _proof(obj: dict[str, Any]) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def verify_action(
    action: ActionProposal,
    signals: Signals | None = None,
    profile: GateProfile | None = None,
    baseline_harm: float | None = None,
    cfg: VerifierConfig | None = None,
) -> VerificationReport:
    signals = signals or Signals()
    profile = profile or GateProfile()
    cfg = cfg or VerifierConfig()
    eps = cfg.EPS
    reasons: list[str] = []

    try:
        datetime.fromisoformat(action.timestamp.replace("Z", "+00:00"))
        timestamp_ok = True
    except (ValueError, AttributeError):
        timestamp_ok = False

    alignment = signals.R / (signals.R + signals.D + cfg.DELTA)
    misalignment = 1.0 - alignment
    harm = robust_harm(action, signals, cfg)
    e_score = signals.E if signals.E is not None else 1.0 - harm * signals.bias * signals.opacity
    if baseline_harm is None:
        relief = relief_index = 0.0
    else:
        relief = max(baseline_harm - harm, 0.0)
        relief_index = relief / max(baseline_harm, eps)

    harm_cap = profile.harm_cap_for(action.stakes)
    fairness_kl = signals.fairness_kl or 0.0
    evidence_kl = signals.claim_evid_kl or 0.0
    checks = {
        "passion": signals.passion <= cfg.PASSION_CAP + eps,
        "bias": signals.bias <= profile.bias_cap + eps,
        "opacity": signals.opacity <= profile.opacity_cap + eps,
        "ethical_compliance": e_score >= cfg.E_MIN - eps,
        "consent": all(s.consent >= profile.consent_threshold for s in action.subjects),
        "fairness": fairness_kl <= profile.fairness_cap + eps,
        "evidence": evidence_kl <= profile.honesty_evidence_gap_cap + eps,
        "harm_cap": harm <= harm_cap + eps,
        "timestamp": timestamp_ok,
    }

    dominance_ok = None
    if action.loss_scenarios and action.baseline_loss_scenarios:
        dominance_ok = fsd_dominates(action.loss_scenarios, action.baseline_loss_scenarios)

    severity, verdict, label = "S0", "pass", "Right"
    if misalignment >= 1.0 - eps:
        label, severity, verdict = "Evil", "S2", "fail"
        reasons.append("100% misalignment endpoint")
    elif alignment >= 1.0 - eps and harm <= eps and all(checks.values()):
        label = "Good*"
        if not action.evidence:
            severity, verdict = "S1", "escalate"
            reasons.append("no evidence references; escalate for human review")
    else:
        comparative_ok = relief_index >= cfg.RI_MIN - eps or dominance_ok is True
        if not (all(checks.values()) and comparative_ok):
            label = "Wrong"
            hard = ("ethical_compliance", "fairness", "consent", "harm_cap", "timestamp")
            hard_fail = not all(checks[k] for k in hard)
            severity = "S2" if hard_fail else "S1"
            verdict = "fail" if hard_fail else "escalate"

    messages = {
        "bias": "bias cap exceeded",
        "opacity": "opacity cap exceeded",
        "ethical_compliance": "ethical compliance E below minimum",
        "passion": "passion cap exceeded",
        "consent": f"one or more subjects below consent threshold {profile.consent_threshold}",
        "fairness": "fairness KL cap exceeded",
        "evidence": "evidence gap (KL claim||evidence) exceeded",
        "harm_cap": f"harm exceeds cap for stakes={action.stakes}",
        "timestamp": "invalid timestamp format",
    }
    reasons += [messages[k] for k, ok in checks.items() if not ok]

    metrics = {
        "A": round(alignment, 12),
        "M": round(misalignment, 12),
        "Harm": round(harm, 12),
        "E": round(e_score, 12),
        "Bias": signals.bias,
        "Opacity": signals.opacity,
        "Relief": relief,
        "ReliefIndex": relief_index,
        "fairness_kl": fairness_kl,
        "fairness_tv_bound": math.sqrt(0.5 * max(fairness_kl, 0.0)),
        "evidence_kl": evidence_kl,
        "harm_cap": harm_cap,
        "risk_mode": cfg.RISK_MODE,
        "dominance_ok": dominance_ok,
        "checks": checks,
    }
    canonical = {
        "action": asdict(action),
        "signals": asdict(signals),
        "profile_id": profile.profile_id,
        "profile": asdict(profile),
        "baseline_harm": baseline_harm,
        "cfg": asdict(cfg),
        "result": {"verdict": verdict, "label": label, "severity": severity, "reasons": reasons},
        "metrics": metrics,
    }
    return VerificationReport(
        verdict=verdict,
        action_label=label,
        severity=severity,
        metrics=metrics,
        reasons=reasons,
        proof=_proof(canonical),
        version=VERIFY_VERSION,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def verify_from_dict(
    data: dict[str, Any], profile: GateProfile | None = None
) -> VerificationReport:
    """Verify from JSON-shaped data: {action, signals?, baseline_harm?, config?}."""
    return verify_action(
        ActionProposal.from_dict(data["action"]),
        Signals(**data.get("signals", {})),
        profile,
        data.get("baseline_harm"),
        VerifierConfig(**data.get("config", {})),
    )
