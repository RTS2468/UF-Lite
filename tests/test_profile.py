from pathlib import Path

import pytest

from uf_lite import GateProfile, ProfileError, load_profile

ROOT = Path(__file__).resolve().parent.parent


def test_defaults():
    p = load_profile()
    assert p.profile_id == "default"
    assert p.harm_cap_for() == 0.05
    assert p.harm_cap_for("high") == 0.01


def test_loads_flat_repo_profile():
    p = load_profile(ROOT / "gate_profile.yaml")
    assert p.consent_threshold == 0.8
    assert p.reversibility_required is True


def test_loads_nested_template_profile():
    p = load_profile(ROOT / "templates" / "gate_profile_template.yaml")
    assert p.fairness_cap == 0.05
    assert p.honesty_sources == ["public records", "official docs"]
    assert p.reasons_required == 2
    assert p.profile_id == "default"


@pytest.mark.parametrize(
    "data",
    [
        {"consent_threshold": 1.5},
        {"stakes": "extreme"},
        {"harm_cap": {"low": 0.1}},
        {"reasons_required": -1},
        {"made_up_key": 1},
    ],
)
def test_invalid_profiles(data):
    with pytest.raises(ProfileError):
        GateProfile.from_dict(data)


def test_non_mapping_yaml(tmp_path):
    f = tmp_path / "bad.yaml"
    f.write_text("- a\n- b\n")
    with pytest.raises(ProfileError):
        load_profile(f)


def test_loads_policy_profile_json():
    p = load_profile(ROOT / "profiles" / "uf_default_policy_profile_v1.json")
    assert p.profile_id == "UF_Default_Policy_Profile_v1"
    assert p.harm_cap_for("med") == 0.05
    assert p.consent_threshold == 0.75 and p.self_report_threshold == 0.8
    assert p.red_lines and p.opacity_cap == 0.01


def test_conformance_report():
    from uf_lite import conformance_report

    good = conformance_report({"name": "x", "red_lines": ["harm"]})
    assert good["valid"] and not good["warnings"]
    assert sum(good["normalized_purpose_weights"].values()) == pytest.approx(1.0)
    bad = conformance_report({"fairness_prior": "vibes"})
    assert not bad["valid"] and bad["issues"]
