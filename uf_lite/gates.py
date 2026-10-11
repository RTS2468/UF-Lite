"""The eight UF Lite operational gates.

Each gate inspects a candidate output and returns a :class:`GateResult`.
Built-in gates are rule-based heuristics (regex/lexicon). They are a hard
floor, not a substitute for a trained classifier: swap any gate for your own
via :class:`CallableGate` or by subclassing :class:`Gate`.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from .fairness import max_pairwise_distance
from .profile import GateProfile


class FailureAction(str, Enum):
    BLOCK = "block"
    CORRECT = "correct"
    ASK = "ask"
    NEUTRALIZE = "neutralize"
    ADD_EXPLANATION = "add_explanation"
    SCALE_DOWN = "scale_down"
    PREFER_REVERSIBLE = "prefer_reversible"
    LOG = "log"


@dataclass
class GateContext:
    """Per-call information the gates may use.

    consent: True/False, or a confidence in [0, 1]; None means unknown.
    human_affecting / reversible: None means "infer from the text".
    sources: citations backing the output (G2, G5).
    reasons: explanations for a recommendation (G5).
    group_outcomes: {group: {outcome: probability}} for the TV/KL fairness bound (G4).
    """

    profile: GateProfile | None = None
    """None means "use the middleware's profile" (or the defaults when used directly)."""
    user_input: str = ""
    stakes: str | None = None
    consent: bool | float | None = None
    human_affecting: bool | None = None
    reversible: bool | None = None
    sources: Sequence[str] = ()
    reasons: Sequence[str] = ()
    group_outcomes: Mapping[str, Mapping[str, float]] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    profile_explicit: bool = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.profile_explicit = self.profile is not None
        if self.profile is None:
            self.profile = GateProfile()

    @property
    def effective_stakes(self) -> str:
        return self.stakes or self.profile.stakes


@dataclass
class GateResult:
    gate_id: str
    name: str
    passed: bool
    action: FailureAction
    score: float = 0.0
    reasons: list[str] = field(default_factory=list)
    output: str | None = None
    """Revised output when the gate failed and remediated; None means unchanged."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate_id": self.gate_id,
            "name": self.name,
            "passed": self.passed,
            "action": self.action.value,
            "score": round(self.score, 4),
            "reasons": list(self.reasons),
            "modified": self.output is not None,
        }


class Gate:
    gate_id: str = ""
    name: str = ""
    action: FailureAction = FailureAction.LOG

    def check(self, text: str, ctx: GateContext) -> GateResult:
        raise NotImplementedError

    def _result(self, passed: bool, score: float = 0.0, reasons=None, output=None) -> GateResult:
        return GateResult(
            gate_id=self.gate_id,
            name=self.name,
            passed=passed,
            action=self.action,
            score=score,
            reasons=list(reasons or []),
            output=None if passed else output,
        )


class CallableGate(Gate):
    """Wrap a function as a gate, e.g. an LLM- or classifier-backed check.

    ``fn(text, ctx)`` may return a GateResult, a bool, or a tuple
    ``(passed, score, reasons[, output])``.
    """

    def __init__(self, gate_id: str, name: str, action: FailureAction, fn: Callable):
        self.gate_id, self.name, self.action, self.fn = gate_id, name, action, fn

    def check(self, text: str, ctx: GateContext) -> GateResult:
        value = self.fn(text, ctx)
        if isinstance(value, GateResult):
            return value
        if isinstance(value, bool):
            return self._result(value)
        passed, score, reasons, *rest = value
        return self._result(bool(passed), float(score), reasons, rest[0] if rest else None)


SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE_RE.split(text) if s.strip()]


def _compile(patterns: Sequence[tuple[str, float]]) -> list[tuple[re.Pattern, float]]:
    return [(re.compile(p, re.IGNORECASE), w) for p, w in patterns]


# --------------------------------------------------------------------------- G1
class NonHarmGate(Gate):
    gate_id, name, action = "G1", "Non-Harm", FailureAction.BLOCK

    PATTERNS = _compile(
        [
            (r"\b(kill|hurt|harm|cut)\s+(yourself|urself)\b", 1.0),
            (r"\byou\s+should\s+(die|disappear)\b", 1.0),
            (
                r"\b(how\s+to|steps?\s+to|instructions?\s+(for|to))\b.{0,40}"
                r"\b(make|build|assemble|synthesi[sz]e)\b.{0,40}"
                r"\b(bomb|explosive|nerve\s+agent|bioweapon|pipe\s+bomb)\b",
                1.0,
            ),
            (r"\b(make|build|assemble)\s+(a\s+)?(bomb|explosive|pipe\s+bomb)\b", 1.0),
            (
                r"\b(i\s+will|i'll|we\s+will|going\s+to)\s+"
                r"(kill|hurt|attack|destroy)\s+(you|him|her|them)\b",
                0.9,
            ),
            (r"\b(poison|stab|shoot)\s+(him|her|them|someone|your)\b", 0.9),
            (r"\b(stop|skip)\s+taking\s+(your|all)\s+(medication|meds|insulin)\b", 0.7),
            (r"\byou('re|\s+are)\s+(worthless|pathetic|a\s+failure|disgusting|an?\s+idiot)\b", 0.4),
            (r"\bno\s+one\s+(would\s+)?(cares?|love[sd]?)\s+(about\s+)?you\b", 0.6),
        ]
    )

    REFUSAL = (
        "I can't help with that because it could cause harm. "
        "If you or someone else is in danger, please contact local emergency services."
    )

    def check(self, text: str, ctx: GateContext) -> GateResult:
        hits = [(p.pattern, w) for p, w in self.PATTERNS if p.search(text)]
        risk = min(1.0, sum(w for _, w in hits))
        cap = ctx.profile.harm_cap_for(ctx.effective_stakes)
        passed = risk <= cap
        reasons = [f"harm risk {risk:.2f} exceeds cap {cap:.2f}"] if not passed else []
        return self._result(passed, risk, reasons, self.REFUSAL)


# --------------------------------------------------------------------------- G2
class TruthfulnessGate(Gate):
    gate_id, name, action = "G2", "Truthfulness", FailureAction.CORRECT

    ABSOLUTE = re.compile(
        r"\b(definitely|certainly|undeniabl[ey]|guaranteed?|100\s?%|always|never|"
        r"proven\s+fact|everyone\s+knows|scientifically\s+proven|without\s+(a\s+)?doubt|"
        r"no\s+doubt|indisputabl[ey])\b",
        re.IGNORECASE,
    )
    STATISTIC = re.compile(r"\b\d+(\.\d+)?\s?(%|percent)\b|\b\d+\s+(out\s+of|in)\s+\d+\b", re.I)
    CITATION = re.compile(r"https?://|\[\d+\]|\(source:|according\s+to\b", re.IGNORECASE)
    HEDGE = re.compile(
        r"\b(may|might|could|likely|possibly|i\s+think|i\s+believe|unverified|estimate)\b", re.I
    )
    LABEL = "Note: parts of this response are unverified claims, not established facts"

    def check(self, text: str, ctx: GateContext) -> GateResult:
        sentences = split_sentences(text)
        if not sentences:
            return self._result(True)
        if ctx.sources or ctx.metadata.get("labeled_fiction") or self.LABEL in text:
            return self._result(True)
        flagged = [
            s
            for s in sentences
            if (self.ABSOLUTE.search(s) or self.STATISTIC.search(s))
            and not self.CITATION.search(s)
            and not self.HEDGE.search(s)
        ]
        gap = len(flagged) / len(sentences)
        cap = ctx.profile.honesty_evidence_gap_cap
        passed = gap <= cap
        reasons = [f"unsupported claim: {s!r}" for s in flagged]
        output = f"{text.rstrip()}\n\n{self.LABEL}; verify before relying on them."
        return self._result(passed, gap, reasons, output)


# --------------------------------------------------------------------------- G3
class ConsentGate(Gate):
    gate_id, name, action = "G3", "Consent", FailureAction.ASK

    ACTION_RE = re.compile(
        r"\b(i|we)\s*('ve|\s+have|'ll|\s+will|\s+am\s+going\s+to)?\s*"
        r"(sent|send|e-?mail(ed)?|text(ed)?|messag(e|ed)|post(ed)?|publish(ed)?|share(d)?|"
        r"delete(d)?|remove(d)?|transfer(red)?|charg(e|ed)|book(ed)?|purchas(e|ed)|"
        r"cancel(l?ed)?|contact(ed)?|call(ed)?|submit(ted)?|forward(ed)?)\b",
        re.IGNORECASE,
    )

    def check(self, text: str, ctx: GateContext) -> GateResult:
        detected = bool(self.ACTION_RE.search(text))
        affecting = ctx.human_affecting if ctx.human_affecting is not None else detected
        if not affecting:
            return self._result(True, 1.0)
        if isinstance(ctx.consent, bool):
            level = 1.0 if ctx.consent else 0.0
        elif ctx.consent is None:
            level = 0.0
        else:
            level = float(ctx.consent)
        threshold = ctx.profile.consent_threshold
        passed = level >= threshold
        reasons = [f"human-affecting output with consent {level:.2f} < {threshold:.2f}"]
        output = (
            "This step affects people and needs explicit consent before I proceed. "
            "Please confirm you want me to go ahead (yes/no).\n\n"
            f"Proposed: {text.strip()}"
        )
        return self._result(passed, level, reasons, output)


# --------------------------------------------------------------------------- G4
GROUPS = (
    r"women|men|girls|boys|muslims|jews|christians|hindus|atheists|immigrants|refugees|"
    r"black\s+people|white\s+people|asians|latinos|hispanics|africans|arabs|mexicans|"
    r"gay\s+people|lesbians|trans\s+people|old\s+people|elderly\s+people|disabled\s+people|"
    r"poor\s+people|foreigners"
)


class FairnessGate(Gate):
    gate_id, name, action = "G4", "Fairness", FailureAction.NEUTRALIZE

    GENERALIZATION = re.compile(
        rf"\b(all|most|every|typical)?\s*({GROUPS})\s+(are|aren't|can't|cannot|should(n't)?|"
        rf"always|never|don't|do\s+not|tend\s+to\s+be)\b",
        re.IGNORECASE,
    )

    def check(self, text: str, ctx: GateContext) -> GateResult:
        sentences = split_sentences(text)
        flagged = [s for s in sentences if self.GENERALIZATION.search(s)]
        text_score = len(flagged) / len(sentences) if sentences else 0.0
        reasons = [f"group generalization: {s!r}" for s in flagged]

        metric = "kl" if "kl" in ctx.profile.fairness_metric.lower().split("/")[0] else "tv"
        dist_score = 0.0
        if ctx.group_outcomes:
            dist_score = max_pairwise_distance(ctx.group_outcomes, metric)
            if dist_score > ctx.profile.fairness_cap:
                reasons.append(
                    f"group outcome {metric.upper()} distance {dist_score:.3f} "
                    f"exceeds cap {ctx.profile.fairness_cap:.3f}"
                )

        score = max(text_score, dist_score)
        passed = not flagged and dist_score <= ctx.profile.fairness_cap
        kept = [s for s in sentences if s not in flagged]
        output = " ".join(kept).strip()
        note = "Note: generalizations about groups of people were removed."
        if dist_score > ctx.profile.fairness_cap:
            note += " Outcomes differ across groups beyond the fairness cap; review before acting."
        output = f"{output}\n\n{note}" if output else note
        return self._result(passed, score, reasons, output)


# --------------------------------------------------------------------------- G5
class BiasOpacityGate(Gate):
    gate_id, name, action = "G5", "Bias & Opacity", FailureAction.ADD_EXPLANATION

    DECISION = re.compile(
        r"\b(you\s+should|i\s+recommend|we\s+recommend|my\s+recommendation|the\s+best\s+option|"
        r"(is|are|has\s+been)\s+(approved|denied|rejected)|i\s+suggest|the\s+right\s+choice)\b",
        re.IGNORECASE,
    )
    REASON = re.compile(
        r"\b(because|since|due\s+to|therefore|as\s+a\s+result|given\s+that|based\s+on)\b", re.I
    )

    def check(self, text: str, ctx: GateContext) -> GateResult:
        is_decision = bool(self.DECISION.search(text)) or ctx.effective_stakes == "high"
        if not is_decision:
            return self._result(True)
        required = ctx.profile.reasons_required
        reason_count = len(self.REASON.findall(text)) + len(ctx.reasons)
        has_sources = bool(ctx.sources) or bool(TruthfulnessGate.CITATION.search(text))
        reasons = []
        if reason_count < required:
            reasons.append(f"{reason_count} reason(s) given, {required} required")
        if not has_sources:
            reasons.append("no sources given for a recommendation or decision")
        passed = not reasons
        score = min(1.0, reason_count / required) if required else 1.0

        parts = [text.rstrip(), "", "Explanation:"]
        if ctx.reasons:
            parts += [f"- {r}" for r in ctx.reasons]
        else:
            parts.append(
                f"- Fewer than {required} reasons were provided for this recommendation; "
                "treat it as provisional."
            )
        if ctx.sources:
            parts.append("Sources: " + "; ".join(ctx.sources))
        else:
            parts.append("Sources: none provided.")
        if ctx.profile.independent_review:
            parts.append("Independent review is recommended before acting on this.")
        return self._result(passed, score, reasons, "\n".join(parts))


# --------------------------------------------------------------------------- G6
class ProportionalityGate(Gate):
    gate_id, name, action = "G6", "Proportionality", FailureAction.SCALE_DOWN

    ALARM = re.compile(
        r"\b(emergency|catastroph\w*|disaster|urgent(ly)?|immediately|panic|terrifying|"
        r"life[- ]threatening|act\s+now|right\s+now)\b",
        re.IGNORECASE,
    )
    SHOUT = re.compile(r"\b[A-Z]{3,}\b")
    CALM = {
        "emergency": "important issue",
        "catastrophic": "serious",
        "catastrophically": "seriously",
        "catastrophe": "serious problem",
        "disaster": "serious problem",
        "urgent": "important",
        "urgently": "promptly",
        "immediately": "promptly",
        "panic": "concern",
        "terrifying": "concerning",
        "life-threatening": "serious",
        "life threatening": "serious",
        "act now": "consider acting soon",
        "right now": "soon",
    }
    INTENSITY_CAP = {"low": 0.15, "medium": 0.3, "high": 0.6}

    def intensity(self, text: str) -> float:
        words = re.findall(r"\b\w+\b", text)
        if not words:
            return 0.0
        alarms = len(self.ALARM.findall(text))
        shouts = len(self.SHOUT.findall(text))
        bangs = len(re.findall(r"!{2,}", text)) + 0.5 * text.count("!")
        return min(1.0, (2 * alarms + shouts + bangs) / len(words) * 5)

    @classmethod
    def calm(cls, text: str) -> str:
        """Replace alarm terms with measured wording and drop exclamation marks."""

        def repl(m: re.Match) -> str:
            word = m.group(0)
            key = re.sub(r"\s+", " ", word.lower())
            calm = cls.CALM.get(key, cls.CALM.get(key.rstrip("s"), "notable"))
            return calm[0].upper() + calm[1:] if word[0].isupper() else calm

        out = cls.ALARM.sub(repl, text)
        return re.sub(r"!+", ".", out)

    @staticmethod
    def truncate(text: str, max_chars: int | None) -> str:
        if max_chars and len(text) > max_chars:
            cut = text[: max_chars - 1].rsplit(" ", 1)[0]
            return cut.rstrip(" ,;:") + "…"
        return text

    @staticmethod
    def scale_down(text: str, max_chars: int | None, calm: bool = False) -> str:
        out = re.sub(r"!{2,}", "!", text)
        out = ProportionalityGate.SHOUT.sub(lambda m: m.group(0).capitalize(), out)
        if calm:
            out = ProportionalityGate.calm(out)
        return ProportionalityGate.truncate(out, max_chars)

    def check(self, text: str, ctx: GateContext) -> GateResult:
        stakes = ctx.effective_stakes
        score = self.intensity(text)
        cap = self.INTENSITY_CAP[stakes]
        max_chars = ctx.profile.max_response_chars
        reasons = []
        if score > cap:
            reasons.append(f"intensity {score:.2f} exceeds {cap:.2f} for {stakes} stakes")
        if max_chars and len(text) > max_chars:
            reasons.append(f"length {len(text)} exceeds max_response_chars {max_chars}")
        passed = not reasons
        return self._result(passed, score, reasons, self.scale_down(text, max_chars, score > cap))


# --------------------------------------------------------------------------- G7
class ReversibilityGate(Gate):
    gate_id, name, action = "G7", "Reversibility", FailureAction.PREFER_REVERSIBLE

    IRREVERSIBLE = re.compile(
        r"\b(permanent(ly)?|irreversibl[ey]|cannot\s+be\s+undone|can't\s+be\s+undone|"
        r"delete\s+(all|everything|the\s+(entire|whole))|wipe|factory\s+reset|"
        r"burn\s+(the\s+)?bridges?|quit\s+(your|my)\s+job|sell\s+everything|all[- ]in)\b"
        r"|rm\s+-rf|drop\s+(table|database)|git\s+push\s+(-f|--force)",
        re.IGNORECASE,
    )
    ADVISORY = (
        "Caution: this step may not be reversible. Prefer a reversible option first "
        "(back up, try a small pilot, or do a dry run) before committing."
    )
    SHORT_ADVISORY = "Caution: may be irreversible; prefer a reversible option first."

    def check(self, text: str, ctx: GateContext) -> GateResult:
        if not ctx.profile.reversibility_required or ctx.reversible is True:
            return self._result(True)
        matches = [m.group(0) for m in self.IRREVERSIBLE.finditer(text)]
        irreversible = bool(matches) or ctx.reversible is False
        if not irreversible or self.ADVISORY in text or self.SHORT_ADVISORY in text:
            return self._result(True)
        reasons = [f"irreversible step: {m!r}" for m in matches] or ["marked irreversible"]
        return self._result(False, 1.0, reasons, self.with_advisory(text, ctx))

    def with_advisory(self, text: str, ctx: GateContext) -> str:
        """Append the advisory, keeping the result within ``max_response_chars``."""
        max_chars = ctx.profile.max_response_chars
        full = f"{text.rstrip()}\n\n{self.ADVISORY}"
        if not max_chars or len(full) <= max_chars:
            return full
        advisory = self.ADVISORY if len(self.ADVISORY) * 2 <= max_chars else self.SHORT_ADVISORY
        room = max_chars - len(advisory) - 2
        if room < 10:
            return advisory[:max_chars]
        body = ProportionalityGate.truncate(text.rstrip(), room)
        return f"{body}\n\n{advisory}"


# --------------------------------------------------------------------------- G8
class AccountabilityGate(Gate):
    """Always passes; the middleware writes the audit record for every output."""

    gate_id, name, action = "G8", "Accountability", FailureAction.LOG

    def check(self, text: str, ctx: GateContext) -> GateResult:
        return self._result(True, 1.0, ["output logged"])


def default_gates() -> list[Gate]:
    return [
        NonHarmGate(),
        TruthfulnessGate(),
        ConsentGate(),
        FairnessGate(),
        BiasOpacityGate(),
        ProportionalityGate(),
        ReversibilityGate(),
        AccountabilityGate(),
    ]
