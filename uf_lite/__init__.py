"""UF Lite: practical ethical gates for AI systems."""

from .audit import AuditLogger
from .gates import (
    CallableGate,
    FailureAction,
    Gate,
    GateContext,
    GateResult,
    default_gates,
)
from .middleware import GENESIS_STATEMENT, GateReport, UFMiddleware, genesis_boot, uf_middleware
from .profile import GateProfile, ProfileError, conformance_report, load_profile
from .scorecard import compute_scorecard, load_dataset, render_markdown
from .truth_audit import load_audit, score_audit
from .verification import (
    ActionProposal,
    EvidenceRef,
    Signals,
    SubjectRisk,
    VerificationReport,
    VerifierConfig,
    verify_action,
)

__version__ = "1.0.0"

__all__ = [
    "GENESIS_STATEMENT",
    "ActionProposal",
    "AuditLogger",
    "CallableGate",
    "EvidenceRef",
    "FailureAction",
    "Gate",
    "GateContext",
    "GateProfile",
    "GateReport",
    "GateResult",
    "ProfileError",
    "Signals",
    "SubjectRisk",
    "UFMiddleware",
    "VerificationReport",
    "VerifierConfig",
    "compute_scorecard",
    "conformance_report",
    "default_gates",
    "genesis_boot",
    "load_audit",
    "load_dataset",
    "load_profile",
    "render_markdown",
    "score_audit",
    "uf_middleware",
    "verify_action",
]
