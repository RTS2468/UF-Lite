import json
from dataclasses import fields
from pathlib import Path

import pytest

from uf_lite.scorecard import (
    Action,
    Claim,
    Promise,
    compute_scorecard,
    dataset_from_dict,
    load_dataset,
    render_markdown,
)

ROOT = Path(__file__).resolve().parent.parent


def test_dataclasses_match_schema():
    schema = json.loads((ROOT / "templates" / "Data_Schema.json").read_text())
    for cls in (Claim, Promise, Action):
        expected = schema["entities"][cls.__name__]["fields"]
        assert [f.name for f in fields(cls)] == expected


def test_example_scorecard():
    card = compute_scorecard(load_dataset(ROOT / "examples" / "pilot_data.json"))
    assert card.QTI == 66.7
    assert card.PRS == 83.3
    assert card.CT == 50.0
    assert card.EI == 85.0
    md = render_markdown(card)
    assert md.startswith("# Scorecard (One Page)")
    assert "Example City Council" in md


def test_empty_dataset_is_na():
    card = compute_scorecard(dataset_from_dict({}))
    assert all(v is None for v in card.metrics().values())
    assert "n/a" in render_markdown(card)


@pytest.mark.parametrize(
    "data",
    [
        {"claims": [{"id": "C1", "EG": 2}]},
        {"promises": [{"id": "P1", "status": "maybe"}]},
        {"actions": [{"description": "no id"}]},
        {"claims": [{"id": "C1", "bogus": 1}]},
    ],
)
def test_invalid_records(data):
    with pytest.raises(ValueError):
        dataset_from_dict(data)
