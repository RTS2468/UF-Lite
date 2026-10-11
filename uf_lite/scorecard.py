"""Pilot scorecard: Claim/Promise/Action records (templates/Data_Schema.json)
scored into QTI, PRS, AGC, CT and EI and rendered as a one-page Markdown file."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from statistics import mean
from typing import Any, Union


@dataclass
class Claim:
    id: str
    speaker: str = ""
    date: str = ""
    topic: str = ""
    proposition: str = ""
    support_evidence: list[str] = field(default_factory=list)
    EG: float = 1.0
    """Evidence gap in [0, 1]: 0 = fully supported, 1 = no support."""
    sources: list[str] = field(default_factory=list)


@dataclass
class Promise:
    id: str
    scope: str = ""
    due_date: str = ""
    criteria: str = ""
    status: str = "pending"
    """One of: kept, partial, broken, pending."""
    weight: float = 1.0
    sources: list[str] = field(default_factory=list)


@dataclass
class Action:
    id: str
    date: str = ""
    type: str = ""
    description: str = ""
    impacted_groups: Any = field(default_factory=list)
    """List of group names, or {group: impact in [-1, 1]}."""
    gate_scores: dict[str, Union[float, bool]] = field(default_factory=dict)
    """Compliance per gate in [0, 1] (or bool), keyed H/C/F/EG/B/O/RV or G1..G8."""
    sources: list[str] = field(default_factory=list)


PROMISE_CREDIT = {"kept": 1.0, "partial": 0.5, "broken": 0.0}


@dataclass
class Dataset:
    subject: str = ""
    topic: str = ""
    dates_covered: str = ""
    claims: list[Claim] = field(default_factory=list)
    promises: list[Promise] = field(default_factory=list)
    actions: list[Action] = field(default_factory=list)


def _build(cls, raw: Mapping[str, Any]):
    allowed = {f.name for f in fields(cls)}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"{cls.__name__} has unknown fields: {sorted(unknown)}")
    if "id" not in raw:
        raise ValueError(f"{cls.__name__} record is missing 'id'")
    return cls(**raw)


def dataset_from_dict(data: Mapping[str, Any]) -> Dataset:
    ds = Dataset(
        subject=data.get("subject", ""),
        topic=data.get("topic", ""),
        dates_covered=data.get("dates_covered", ""),
        claims=[_build(Claim, c) for c in data.get("claims", [])],
        promises=[_build(Promise, p) for p in data.get("promises", [])],
        actions=[_build(Action, a) for a in data.get("actions", [])],
    )
    for c in ds.claims:
        if not 0.0 <= float(c.EG) <= 1.0:
            raise ValueError(f"claim {c.id}: EG must be in [0, 1]")
    for p in ds.promises:
        if p.status not in (*PROMISE_CREDIT, "pending"):
            raise ValueError(f"promise {p.id}: unknown status {p.status!r}")
    return ds


def load_dataset(path: str | Path) -> Dataset:
    return dataset_from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _pct(value: float | None) -> float | None:
    return None if value is None else round(100.0 * value, 1)


def qti(claims: list[Claim]) -> float | None:
    """Claims vs evidence: mean of (1 - EG)."""
    return _pct(mean(1.0 - float(c.EG) for c in claims)) if claims else None


def prs(promises: list[Promise]) -> float | None:
    """Promises kept: weighted credit over resolved (non-pending) promises."""
    resolved = [p for p in promises if p.status in PROMISE_CREDIT]
    total = sum(p.weight for p in resolved)
    if not total:
        return None
    return _pct(sum(PROMISE_CREDIT[p.status] * p.weight for p in resolved) / total)


def agc(actions: list[Action]) -> float | None:
    """Action gate compliance: mean per-action gate score."""
    scored = [a for a in actions if a.gate_scores]
    if not scored:
        return None
    return _pct(mean(mean(float(v) for v in a.gate_scores.values()) for a in scored))


def ct(ds: Dataset) -> float | None:
    """Consistency/transparency: share of records with at least one source."""
    records = [*ds.claims, *ds.promises, *ds.actions]
    if not records:
        return None
    return _pct(sum(1 for r in records if r.sources) / len(records))


def ei(actions: list[Action]) -> float | None:
    """Equity impact: 1 - (spread of mean impact across groups) / 2.

    Needs ``impacted_groups`` as {group: impact in [-1, 1]}; 100 = even impact.
    """
    impacts: dict[str, list[float]] = {}
    for a in actions:
        if isinstance(a.impacted_groups, Mapping):
            for group, value in a.impacted_groups.items():
                impacts.setdefault(group, []).append(float(value))
    if len(impacts) < 2:
        return None
    means = [mean(v) for v in impacts.values()]
    return _pct(1.0 - (max(means) - min(means)) / 2.0)


@dataclass
class Scorecard:
    dataset: Dataset
    QTI: float | None
    PRS: float | None
    AGC: float | None
    CT: float | None
    EI: float | None

    def metrics(self) -> dict[str, float | None]:
        return {k: getattr(self, k) for k in ("QTI", "PRS", "AGC", "CT", "EI")}


def compute_scorecard(ds: Dataset) -> Scorecard:
    return Scorecard(ds, qti(ds.claims), prs(ds.promises), agc(ds.actions), ct(ds), ei(ds.actions))


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:g} / 100"


def render_markdown(card: Scorecard) -> str:
    ds = card.dataset
    lines = [
        "# Scorecard (One Page)",
        f"**Subject:** {ds.subject}  ",
        f"**Topic:** {ds.topic}  ",
        f"**Dates covered:** {ds.dates_covered}",
        "",
        "## Headline",
        f"- **QTI (claims vs evidence):** {_fmt(card.QTI)}",
        f"- **PRS (promises kept):** {_fmt(card.PRS)}",
        f"- **AGC (action gate compliance):** {_fmt(card.AGC)}",
        f"- **CT (consistency/transparency):** {_fmt(card.CT)}",
        f"- **EI (equity impact):** {_fmt(card.EI)}",
        "",
        "## Details",
        "### Top 3 claims (lowest evidence gap first)",
    ]
    for c in sorted(ds.claims, key=lambda c: c.EG)[:3]:
        cites = ", ".join(c.sources) or "no sources"
        lines.append(f'- {c.speaker} ({c.date}): "{c.proposition}" — EG {c.EG:g} [{cites}]')
    lines.append("### Top 3 actions and gate notes")
    for a in ds.actions[:3]:
        gates = ", ".join(f"{k}={float(v):g}" for k, v in a.gate_scores.items()) or "not scored"
        lines.append(f"- {a.date} {a.type}: {a.description} — {gates}")
    lines += ["", "Methods: see templates/Methods_Appendix.md and the README."]
    return "\n".join(lines) + "\n"
