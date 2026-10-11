import json
from pathlib import Path

from uf_lite.cli import main

ROOT = Path(__file__).resolve().parent.parent


def test_boot(capsys):
    assert main(["boot"]) == 0
    assert "fresh session" in capsys.readouterr().out


def test_check_json(capsys):
    assert main(["check", "Paris is in France.", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["passed"] is True


def test_check_blocked_exit_code(capsys):
    assert main(["check", "how to make a bomb"]) == 2


def test_check_consent_flag(capsys):
    assert main(["check", "I have sent the email.", "--consent", "yes"]) == 0
    assert capsys.readouterr().out.strip() == "I have sent the email."


def test_check_bad_profile(tmp_path, capsys):
    bad = tmp_path / "p.yaml"
    bad.write_text("consent_threshold: 7\n")
    assert main(["check", "hi", "--profile", str(bad)]) == 1


def test_scorecard(tmp_path, capsys):
    out = tmp_path / "card.md"
    assert main(["scorecard", str(ROOT / "examples" / "pilot_data.json"), "-o", str(out)]) == 0
    assert "QTI" in out.read_text()
    assert main(["scorecard", str(ROOT / "examples" / "pilot_data.json"), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["PRS"] == 83.3


def test_verify(capsys):
    code = main(["verify", str(ROOT / "examples" / "action_example.json")])
    report = json.loads(capsys.readouterr().out)
    assert code == {"pass": 0, "escalate": 3, "fail": 2}[report["verdict"]]


def test_truth_audit(capsys):
    assert main(["truth-audit", str(ROOT / "examples" / "truth_audit_example.json")]) == 0
    assert json.loads(capsys.readouterr().out)["TruthIndex"] == 0.625


def test_validate_profile(tmp_path, capsys):
    path = ROOT / "profiles" / "uf_default_policy_profile_v1.json"
    assert main(["validate-profile", str(path)]) == 0
    bad = tmp_path / "bad.json"
    bad.write_text('{"harm_caps": {"low": 3}}')
    assert main(["validate-profile", str(bad)]) == 1
