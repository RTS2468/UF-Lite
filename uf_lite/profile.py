"""Gate profile loading and validation.

Supports three formats:
- flat YAML (``gate_profile.yaml``)
- nested YAML template (``templates/gate_profile_template.yaml``)
- UF policy profile JSON (``profiles/uf_default_policy_profile_v1.json``)
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

STAKES_LEVELS = ("low", "medium", "high")
STAKES_ALIASES = {"med": "medium"}
FAIRNESS_PRIORS = ("demographic_parity", "equal_opportunity", "individual_fairness", "custom")
PURPOSE_KEYS = ("self", "right", "help")

DEFAULT_CONTINUITY_PHRASES = [
    "i'm back",
    "as i said earlier",
    "welcome back",
    "as we discussed last time",
    "picking up where we left off",
    "i remember you",
    "we can continue",
    "continue our conversation",
    "pick up where",
]


class ProfileError(ValueError):
    """Raised when a gate profile is malformed."""


@dataclass
class GateProfile:
    profile_id: str = "default"
    stakes: str = "medium"
    harm_cap: dict[str, float] = field(
        default_factory=lambda: {"high": 0.01, "medium": 0.05, "low": 0.10}
    )
    consent_threshold: float = 0.80
    self_report_threshold: float = 0.80
    fairness_metric: str = "TV/KL bound"
    fairness_cap: float = 0.05
    fairness_prior: str = "equal_opportunity"
    honesty_evidence_gap_cap: float = 0.05
    honesty_sources: list[str] = field(default_factory=list)
    independent_review: bool = True
    reasons_required: int = 2
    bias_cap: float = 0.05
    opacity_cap: float = 0.01
    reversibility_required: bool = True
    reversibility_weight: float = 0.6
    purpose_weights: dict[str, float] = field(
        default_factory=lambda: {"self": 0.3, "right": 0.4, "help": 0.3}
    )
    red_lines: list[str] = field(default_factory=list)
    max_response_chars: int | None = None
    continuity_phrases: list[str] = field(default_factory=lambda: list(DEFAULT_CONTINUITY_PHRASES))
    notes: str = ""

    def __post_init__(self) -> None:
        self.validate()

    def harm_cap_for(self, stakes: str | None = None) -> float:
        stakes = stakes or self.stakes
        return self.harm_cap[STAKES_ALIASES.get(stakes, stakes)]

    def normalized_purpose_weights(self) -> dict[str, float]:
        total = sum(self.purpose_weights.values()) or 1.0
        return {k: v / total for k, v in self.purpose_weights.items()}

    def validate(self) -> None:
        if self.stakes not in STAKES_LEVELS:
            raise ProfileError(f"stakes must be one of {STAKES_LEVELS}, got {self.stakes!r}")
        missing = [level for level in STAKES_LEVELS if level not in self.harm_cap]
        if missing:
            raise ProfileError(f"harm_cap is missing levels: {missing}")
        caps = {f"harm_cap.{k}": v for k, v in self.harm_cap.items()}
        caps.update(
            consent_threshold=self.consent_threshold,
            self_report_threshold=self.self_report_threshold,
            fairness_cap=self.fairness_cap,
            honesty_evidence_gap_cap=self.honesty_evidence_gap_cap,
            bias_cap=self.bias_cap,
            opacity_cap=self.opacity_cap,
            reversibility_weight=self.reversibility_weight,
        )
        for name, value in caps.items():
            if not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0:
                raise ProfileError(f"{name} must be a number in [0, 1], got {value!r}")
        if not isinstance(self.reasons_required, int) or self.reasons_required < 0:
            raise ProfileError("reasons_required must be a non-negative integer")
        if self.max_response_chars is not None and self.max_response_chars <= 0:
            raise ProfileError("max_response_chars must be positive when set")
        if self.fairness_prior not in FAIRNESS_PRIORS:
            raise ProfileError(f"fairness_prior must be one of {FAIRNESS_PRIORS}")
        if set(self.purpose_weights) != set(PURPOSE_KEYS):
            raise ProfileError(f"purpose_weights must have exactly the keys {PURPOSE_KEYS}")
        if any(v < 0 for v in self.purpose_weights.values()):
            raise ProfileError("purpose_weights must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> GateProfile:
        return cls(**_normalize(dict(data or {})))


def _normalize(data: dict[str, Any]) -> dict[str, Any]:
    """Flatten the nested template format into GateProfile keyword arguments."""
    out: dict[str, Any] = {}
    nested = {
        "consent": {"threshold": "consent_threshold"},
        "fairness": {"metric": "fairness_metric", "cap": "fairness_cap"},
        "honesty": {"evidence_gap_cap": "honesty_evidence_gap_cap", "sources": "honesty_sources"},
        "bias_opacity": {
            "independent_review": "independent_review",
            "reasons_required": "reasons_required",
        },
        "reversibility": {"required": "reversibility_required"},
        "consent_thresholds": {
            "consent_tau": "consent_threshold",
            "sr_tau": "self_report_threshold",
        },
    }
    renamed = {
        "name": "profile_id",
        "harm_caps": "harm_cap",
        "evidence_gap_cap": "honesty_evidence_gap_cap",
    }
    for key, value in data.items():
        if key in nested and isinstance(value, Mapping):
            for sub_key, target in nested[key].items():
                if sub_key in value:
                    out[target] = value[sub_key]
        else:
            out[renamed.get(key, key)] = value

    if not out.get("profile_id"):
        out["profile_id"] = "default"
    for key in ("harm_cap",):
        if key in out:
            out[key] = {STAKES_ALIASES.get(str(k), str(k)): float(v) for k, v in out[key].items()}
    if "stakes" in out:
        out["stakes"] = STAKES_ALIASES.get(out["stakes"], out["stakes"])

    known = set(GateProfile.__dataclass_fields__)
    unknown = sorted(set(out) - known)
    if unknown:
        raise ProfileError(f"unknown profile keys: {unknown}")
    return out


def read_profile_data(path: str | Path) -> dict[str, Any]:
    """Read raw profile data from a YAML or JSON file."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)
    except (yaml.YAMLError, json.JSONDecodeError) as exc:
        raise ProfileError(f"invalid profile file {path}: {exc}") from exc
    if data is not None and not isinstance(data, Mapping):
        raise ProfileError(f"{path} must contain a mapping at the top level")
    return dict(data or {})


def load_profile(path: str | Path | None = None) -> GateProfile:
    """Load a gate profile from YAML or JSON. With no path, return the defaults."""
    if path is None:
        return GateProfile()
    return GateProfile.from_dict(read_profile_data(path))


def conformance_report(data: Mapping[str, Any]) -> dict[str, Any]:
    """Validate raw profile data and summarise it (port of adapter_universal.py)."""
    issues: list[str] = []
    warnings: list[str] = []
    profile = None
    try:
        profile = GateProfile.from_dict(data)
    except (ProfileError, TypeError, ValueError) as exc:
        issues.append(str(exc))
    if profile is not None and not profile.red_lines:
        warnings.append("red_lines is empty; list the outcomes that must never be allowed")
    return {
        "name": data.get("name") or data.get("profile_id") or "<unnamed>",
        "valid": not issues,
        "issues": issues,
        "warnings": warnings,
        "normalized_purpose_weights": profile.normalized_purpose_weights() if profile else None,
        "profile": profile.to_dict() if profile else None,
    }


def coerce_profile(value: GateProfile | Mapping[str, Any] | str | Path | None) -> GateProfile:
    """Accept a profile object, a dict, a YAML path, or None."""
    if isinstance(value, GateProfile):
        return value
    if isinstance(value, Mapping):
        return GateProfile.from_dict(value)
    return load_profile(value)
