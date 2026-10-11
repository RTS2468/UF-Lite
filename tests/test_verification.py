import json
from pathlib import Path

import pytest

from uf_lite import ActionProposal, GateProfile, Signals, SubjectRisk, verify_action
from uf_lite.verification import cvar, expected_harm, fsd_dominates, verify_from_dict

ROOT = Path(__file__).resolve().parent.parent
TS = "2025-08-06T12:00:00Z"


def action(**kw):
    base = dict(action_id="a1", actor_id="agent", description="demo", timestamp=TS)
    base.update(kw)
    return ActionProposal(**base)


def test_risk_helpers():
    assert expected_harm([]) == 0.0
    assert cvar([0.0, 0.1, 0.2, 0.9], 0.75) == pytest.approx(0.9)
    assert fsd_dominates([0.0, 0.1], [0.2, 0.3])
    assert not fsd_dominates([0.2, 0.3], [0.0, 0.1])


def test_subject_risk_validates_range():
    with pytest.raises(ValueError):
        SubjectRisk(p=1.5, s=0, d=0, r=0, k=0, consent=1, w=1)


def test_zero_harm_with_evidence_is_good():
    rep = verify_action(action(evidence=[{"uri": "https://example.org"}]))
    assert (rep.verdict, rep.action_label) == ("pass", "Good*")
    assert len(rep.proof) == 64


def test_zero_harm_without_evidence_escalates():
    rep = verify_action(action())
    assert rep.verdict == "escalate" and rep.severity == "S1"


def test_full_misalignment_is_evil():
    rep = verify_action(action(), Signals(R=0.0, D=1.0))
    assert rep.action_label == "Evil" and rep.verdict == "fail"


def test_missing_consent_hard_fails():
    subject = SubjectRisk(p=0.9, s=0.9, d=0.9, r=0.9, k=0.0, consent=0.1, w=1.0)
    rep = verify_action(action(subjects=[subject]), Signals(R=0.9, D=0.1))
    assert rep.verdict == "fail" and rep.severity == "S2"
    assert any("consent" in r for r in rep.reasons)
    assert any("harm exceeds" in r for r in rep.reasons)


def test_profile_caps_apply():
    rep = verify_action(action(), Signals(R=0.9, D=0.1, bias=0.2), GateProfile(bias_cap=0.5))
    assert rep.verdict == "pass" and rep.action_label == "Right"
    rep = verify_action(action(), Signals(R=0.9, D=0.1, bias=0.2), GateProfile(bias_cap=0.1))
    assert rep.verdict == "escalate"


def test_stakes_alias_and_example_file():
    assert action(stakes="med").stakes == "medium"
    rep = verify_from_dict(json.loads((ROOT / "examples" / "action_example.json").read_text()))
    assert rep.metrics["risk_mode"] == "cvar"
    assert rep.metrics["dominance_ok"] is True
    assert rep.metrics["fairness_tv_bound"] == pytest.approx((0.5 * 0.01) ** 0.5)


def test_cvar_exact_tail_weighting():
    assert cvar([0.0] * 9 + [1.0], 0.9) == pytest.approx(1.0)
    assert cvar([0.0, 0.5, 1.0, 1.0], 0.5) == pytest.approx(1.0)
    # tail mass 1.5: one full 1.0 plus half of 0.5
    assert cvar([0.0, 0.5, 1.0], 0.5) == pytest.approx((1.0 + 0.25) / 1.5)
    assert cvar([0.2, 0.7], 1.0) == pytest.approx(0.7)
    assert cvar([0.2, 0.7], 0.0) == pytest.approx(0.45)
    with pytest.raises(ValueError):
        cvar([0.1], 1.5)


SAFE = dict(p=0, s=0, d=0, r=0, k=0, w=1)


def _zero_harm_action(**kw):
    base = dict(subjects=[SubjectRisk(consent=1.0, **SAFE)], evidence=[{"uri": "ref"}])
    base.update(kw)
    return action(**base)


def test_good_star_requires_every_check():
    assert verify_action(_zero_harm_action()).verdict == "pass"
    passion = verify_action(_zero_harm_action(), Signals(passion=1.0))
    assert passion.verdict != "pass" and "passion cap exceeded" in passion.reasons
    no_consent = _zero_harm_action(subjects=[SubjectRisk(consent=0.0, **SAFE)])
    report = verify_action(no_consent)
    assert report.verdict == "fail" and report.action_label == "Wrong"


def test_invalid_timestamp_cannot_pass():
    report = verify_action(_zero_harm_action(timestamp="garbage"))
    assert report.verdict == "fail"
    assert "invalid timestamp format" in report.reasons


def test_proof_covers_profile_thresholds():
    a = verify_action(_zero_harm_action(), profile=GateProfile(consent_threshold=0.2))
    b = verify_action(_zero_harm_action(), profile=GateProfile(consent_threshold=0.9))
    assert a.verdict == b.verdict and a.proof != b.proof
