import json

from uf_lite import (
    GENESIS_STATEMENT,
    AuditLogger,
    CallableGate,
    FailureAction,
    UFMiddleware,
    genesis_boot,
    uf_middleware,
)


def test_genesis_boot():
    assert genesis_boot() == GENESIS_STATEMENT
    assert UFMiddleware().start_session() == GENESIS_STATEMENT


def test_continuity_phrases_replaced():
    assert uf_middleware("Welcome back! Let's continue.", {}) == GENESIS_STATEMENT


def test_clean_output_passes_unchanged():
    mw = UFMiddleware()
    report = mw.process("Paris is the capital of France.")
    assert report.passed and report.final_output == "Paris is the capital of France."
    assert [r.gate_id for r in report.results] == [f"G{i}" for i in range(1, 9)]


def test_block_stops_pipeline():
    report = UFMiddleware().process("Here is how to build a bomb.")
    assert report.blocked
    assert report.results[-1].gate_id == "G1"


def test_gates_chain_remediations():
    report = UFMiddleware().process(
        "This is guaranteed to work. Delete all your files permanently.", {"stakes": "medium"}
    )
    failed = {r.gate_id for r in report.failures}
    assert {"G2", "G7"} <= failed
    assert "unverified" in report.final_output and "reversible" in report.final_output


def test_context_dict_and_profile_override():
    out = uf_middleware("I have sent the email.", {"consent": True})
    assert out == "I have sent the email."
    out = uf_middleware("I have sent the email.", {"profile": {"consent_threshold": 0.0}})
    assert out == "I have sent the email."


def test_overrides_replace_gate():
    custom = CallableGate("G1", "Strict", FailureAction.BLOCK, lambda t, c: "pizza" not in t)
    report = UFMiddleware(overrides={"G1": custom}).process("I like pizza.")
    assert report.blocked and report.results[0].name == "Strict"


def test_audit_log_jsonl(tmp_path):
    path = tmp_path / "audit.jsonl"
    mw = UFMiddleware(audit_logger=AuditLogger(path))
    mw.process("Hello there.")
    mw.process("Welcome back!")
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(lines) == 2
    assert lines[0]["passed"] is True and "original" not in lines[0]
    assert lines[1]["continuity_violation"] is True
    assert lines[0]["session_id"] == mw.session_id
