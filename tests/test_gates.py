import pytest

from uf_lite import CallableGate, FailureAction, GateContext, GateProfile
from uf_lite.fairness import kl_divergence, max_pairwise_distance, total_variation
from uf_lite.gates import (
    BiasOpacityGate,
    ConsentGate,
    FairnessGate,
    NonHarmGate,
    ProportionalityGate,
    ReversibilityGate,
    TruthfulnessGate,
)


@pytest.fixture
def ctx():
    return GateContext(profile=GateProfile())


def test_non_harm_blocks(ctx):
    r = NonHarmGate().check("Here is how to make a pipe bomb at home.", ctx)
    assert not r.passed and r.action is FailureAction.BLOCK
    assert "can't help" in r.output


def test_non_harm_passes_benign(ctx):
    assert NonHarmGate().check("Drink water and get some sleep.", ctx).passed


def test_truthfulness_labels_unsupported_absolutes(ctx):
    r = TruthfulnessGate().check("This supplement is guaranteed to cure colds.", ctx)
    assert not r.passed
    assert TruthfulnessGate.LABEL in r.output


def test_truthfulness_accepts_sources_or_hedges(ctx):
    gate = TruthfulnessGate()
    assert gate.check("It might help some people.", ctx).passed
    assert gate.check("Unemployment is 4% (source: BLS).", ctx).passed
    ctx.sources = ["https://example.org"]
    assert gate.check("This is guaranteed.", ctx).passed


def test_consent_asks_when_missing(ctx):
    r = ConsentGate().check("I have sent the email to your team.", ctx)
    assert not r.passed and "explicit consent" in r.output


def test_consent_passes_with_consent(ctx):
    ctx.consent = True
    assert ConsentGate().check("I have sent the email.", ctx).passed
    ctx.consent = 0.5
    assert not ConsentGate().check("I have sent the email.", ctx).passed


def test_consent_ignores_non_actions(ctx):
    assert ConsentGate().check("Paris is the capital of France.", ctx).passed


def test_fairness_neutralizes_generalizations(ctx):
    r = FairnessGate().check("Try this recipe. Immigrants are lazy.", ctx)
    assert not r.passed
    assert "Immigrants" not in r.output and "Try this recipe." in r.output


def test_fairness_distribution_bound(ctx):
    ctx.group_outcomes = {
        "a": {"approved": 0.8, "denied": 0.2},
        "b": {"approved": 0.5, "denied": 0.5},
    }
    r = FairnessGate().check("Loan decisions are attached.", ctx)
    assert not r.passed and pytest.approx(r.score) == 0.3


def test_fairness_metrics():
    p, q = {"x": 0.5, "y": 0.5}, {"x": 0.5, "y": 0.5}
    assert total_variation(p, q) == 0
    assert kl_divergence(p, q) == pytest.approx(0)
    assert total_variation({"x": 1}, {"y": 1}) == 1
    assert max_pairwise_distance({"only": p}) == 0


def test_bias_opacity_requires_reasons_and_sources(ctx):
    gate = BiasOpacityGate()
    r = gate.check("I recommend option B.", ctx)
    assert not r.passed and "Explanation:" in r.output
    ctx.reasons = ["cheaper", "faster"]
    ctx.sources = ["vendor quote"]
    assert gate.check("I recommend option B.", ctx).passed


def test_proportionality_scales_down_shouting(ctx):
    ctx.stakes = "low"
    r = ProportionalityGate().check("URGENT!!! ACT NOW, this is an EMERGENCY!!!", ctx)
    assert not r.passed
    assert "!!!" not in r.output and "URGENT" not in r.output


def test_proportionality_max_chars():
    ctx = GateContext(profile=GateProfile(max_response_chars=20))
    r = ProportionalityGate().check("one two three four five six seven eight", ctx)
    assert not r.passed and len(r.output) <= 20


def test_reversibility(ctx):
    gate = ReversibilityGate()
    r = gate.check("Just run rm -rf / to fix it.", ctx)
    assert not r.passed and gate.ADVISORY in r.output
    ctx.reversible = True
    assert gate.check("Just run rm -rf / to fix it.", ctx).passed


def test_callable_gate_forms(ctx):
    as_bool = CallableGate("G1", "Custom", FailureAction.BLOCK, lambda t, c: "bad" not in t)
    assert as_bool.check("fine", ctx).passed
    assert not as_bool.check("bad", ctx).passed
    as_tuple = CallableGate(
        "G2", "Custom", FailureAction.CORRECT, lambda t, c: (False, 0.9, ["x"], "fixed")
    )
    r = as_tuple.check("anything", ctx)
    assert r.output == "fixed" and r.reasons == ["x"]


def test_fairness_removes_generalization_in_long_reply(ctx):
    text = "Immigrants are lazy. " + " ".join(["Okay."] * 30)
    r = FairnessGate().check(text, ctx)
    assert not r.passed and "lazy" not in r.output


def test_proportionality_calms_alarm_words(ctx):
    ctx.stakes = "low"
    gate = ProportionalityGate()
    r = gate.check("Emergency! Act now!", ctx)
    assert not r.passed
    assert "Emergency" not in r.output and "!" not in r.output
    assert gate.intensity(r.output) <= gate.INTENSITY_CAP["low"]


def test_reversibility_advisory_respects_max_chars():
    ctx = GateContext(profile=GateProfile(max_response_chars=80))
    r = ReversibilityGate().check("Delete all your files permanently and then reinstall.", ctx)
    assert not r.passed and len(r.output) <= 80
    assert ReversibilityGate.SHORT_ADVISORY in r.output
