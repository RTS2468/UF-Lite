"""Runs candidate outputs through Genesis Boot and the eight gates."""

from __future__ import annotations

import uuid
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, fields
from typing import Any

from .audit import AuditLogger
from .gates import FailureAction, Gate, GateContext, GateResult, default_gates
from .profile import GateProfile, coerce_profile

GENESIS_STATEMENT = "I am a new instance with no prior continuity. This is a fresh session."


def genesis_boot() -> str:
    """The mandatory first line of every new session."""
    return GENESIS_STATEMENT


@dataclass
class GateReport:
    original: str
    final_output: str
    results: list[GateResult] = field(default_factory=list)
    blocked: bool = False
    continuity_violation: bool = False
    session_id: str | None = None

    @property
    def passed(self) -> bool:
        return not self.continuity_violation and all(r.passed for r in self.results)

    @property
    def failures(self) -> list[GateResult]:
        return [r for r in self.results if not r.passed]

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "passed": self.passed,
            "blocked": self.blocked,
            "continuity_violation": self.continuity_violation,
            "final_output": self.final_output,
            "gates": [r.to_dict() for r in self.results],
        }


_CONTEXT_FIELDS = {f.name for f in fields(GateContext)} - {"profile", "metadata"}


class UFMiddleware:
    """Gate pipeline.

    ``overrides`` replaces built-in gates by id, e.g. ``{"G1": my_classifier_gate}``.
    """

    def __init__(
        self,
        profile: GateProfile | Mapping[str, Any] | str | None = None,
        gates: Iterable[Gate] | None = None,
        overrides: Mapping[str, Gate] | None = None,
        audit_logger: AuditLogger | None = None,
        session_id: str | None = None,
    ):
        self.profile = coerce_profile(profile)
        chosen = list(gates) if gates is not None else default_gates()
        if overrides:
            chosen = [overrides.get(g.gate_id, g) for g in chosen]
        self.gates = chosen
        self.audit_logger = audit_logger if audit_logger is not None else AuditLogger()
        self.session_id = session_id or uuid.uuid4().hex

    def start_session(self) -> str:
        """Begin a fresh session and return the Genesis Boot statement."""
        self.session_id = uuid.uuid4().hex
        return genesis_boot()

    def build_context(self, context: Mapping[str, Any] | GateContext | None) -> GateContext:
        if isinstance(context, GateContext):
            return context
        context = dict(context or {})
        profile = coerce_profile(context.pop("profile", None) or self.profile)
        known = {k: context.pop(k) for k in list(context) if k in _CONTEXT_FIELDS}
        return GateContext(profile=profile, metadata=context, **known)

    def _violates_continuity(self, text: str, profile: GateProfile) -> bool:
        lowered = text.lower()
        return any(phrase in lowered for phrase in profile.continuity_phrases)

    def process(
        self, response: str, context: Mapping[str, Any] | GateContext | None = None
    ) -> GateReport:
        ctx = self.build_context(context)
        report = GateReport(original=response, final_output=response, session_id=self.session_id)

        if self._violates_continuity(response, ctx.profile):
            report.continuity_violation = True
            report.final_output = GENESIS_STATEMENT
        else:
            text = response
            for gate in self.gates:
                result = gate.check(text, ctx)
                report.results.append(result)
                if not result.passed and result.output is not None:
                    text = result.output
                if not result.passed and result.action is FailureAction.BLOCK:
                    report.blocked = True
                    break
            report.final_output = text

        self.audit_logger.log(
            response,
            report.final_output,
            session_id=self.session_id,
            profile_id=ctx.profile.profile_id,
            passed=report.passed,
            blocked=report.blocked,
            continuity_violation=report.continuity_violation,
            gates=[r.to_dict() for r in report.results],
        )
        return report

    def __call__(self, response: str, context: Mapping[str, Any] | None = None) -> str:
        return self.process(response, context).final_output


_default: UFMiddleware | None = None


def uf_middleware(response: str, context: Mapping[str, Any] | None = None) -> str:
    """Drop-in filter: returns the gated output for ``response``.

    ``context`` may include ``profile`` (GateProfile, dict, or YAML path) and any
    :class:`GateContext` field (stakes, consent, sources, reasons, ...).
    """
    global _default
    if _default is None:
        _default = UFMiddleware()
    return _default(response, context)
