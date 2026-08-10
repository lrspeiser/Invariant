from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from sigma_theory_compiler.llm_formula_proposal_adapter import (
    AdapterConfig,
    FormulaProposalAdapter,
    ProposalAdapterError,
    ProposalRequest,
    SpendLedger,
    build_readiness_artifact,
    validate_proposal_output,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs/llm_formula_proposal_adapter.json"
ARTIFACT_PATH = REPO_ROOT / "runs/engine/llm-formula-proposal-adapter-readiness.json"
SECRET = "mock-secret-must-never-persist"


def _config(*, enabled: bool) -> AdapterConfig:
    return AdapterConfig.from_mapping(
        {
            "allowed_data_classes": ["formal_artifact", "formula_dsl"],
            "api_key_env_var": "SIGMA_FORMULA_LLM_API_KEY",
            "maximum_attempts": 3,
            "maximum_call_usd": "5.000000",
            "maximum_total_usd": "500.000000",
            "paid_calls_enabled": enabled,
            "provider_id": "deterministic_mock",
        }
    )


def _request(request_id: str = "proposal:req:0001", prompt: str = "Combine curvature and coherence") -> ProposalRequest:
    return ProposalRequest(
        request_id=request_id,
        prompt=prompt,
        prompt_template_sha256="1" * 64,
        context_packets=(
            {"content_sha256": "2" * 64, "data_class": "formal_artifact"},
        ),
        dsl_version="sigma-gravity-dsl-3",
        deterministic_seed=7,
        maximum_call_usd="1.250000",
    )


def _output() -> dict[str, object]:
    return {
        "schema_version": "sigma-formula-proposals-1.0",
        "proposals": [
            {
                "proposal_id": "candidate:mock:0001",
                "expression": "R + alpha*coherence(phi)",
                "parameters": ["alpha"],
                "concept_tags": ["curvature", "coherence"],
            }
        ],
    }


def test_disabled_default_and_missing_secret_block_before_reservation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("SIGMA_FORMULA_LLM_API_KEY", raising=False)
    disabled = _config(enabled=False)
    ledger = SpendLedger(tmp_path / "disabled.sqlite", disabled)
    called = 0

    def provider(_request: object, _secret: str) -> dict[str, object]:
        nonlocal called
        called += 1
        return {"billed_usd": "1.0", "output": _output()}

    result = FormulaProposalAdapter(disabled, ledger, provider).propose(_request())
    assert result["reason"] == "paid_calls_disabled_by_default"
    assert ledger.status(_request().request_id) is None

    enabled = _config(enabled=True)
    enabled_ledger = SpendLedger(tmp_path / "missing.sqlite", enabled)
    result = FormulaProposalAdapter(enabled, enabled_ledger, provider).propose(_request())
    assert result["reason"] == "referenced_api_secret_absent"
    assert enabled_ledger.status(_request().request_id) is None
    assert called == 0


def test_success_is_settled_once_idempotent_and_quarantined(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SIGMA_FORMULA_LLM_API_KEY", SECRET)
    config = _config(enabled=True)
    ledger = SpendLedger(tmp_path / "ledger.sqlite", config)
    calls = []

    def provider(request: dict[str, object], secret: str) -> dict[str, object]:
        assert secret == SECRET
        calls.append(request["idempotency_key"])
        return {"billed_usd": "0.750000", "output": _output()}

    adapter = FormulaProposalAdapter(config, ledger, provider)
    first = adapter.propose(_request())
    second = adapter.propose(_request())
    assert first["decision"] == "quarantined"
    assert first["downstream_validation_required"] is True
    assert second["replayed"] is True
    assert calls == [_request().request_id]
    row = ledger.status(_request().request_id)
    assert row["status"] == "settled"
    assert row["attempts"] == 1
    assert ledger.telemetry()["settled_usd"] == "0.750000"


def test_retry_reuses_idempotency_key_and_never_double_charges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SIGMA_FORMULA_LLM_API_KEY", SECRET)
    config = _config(enabled=True)
    ledger = SpendLedger(tmp_path / "retry.sqlite", config)
    keys = []

    def provider(request: dict[str, object], _secret: str) -> dict[str, object]:
        keys.append(request["idempotency_key"])
        if len(keys) == 1:
            raise RuntimeError("synthetic transient")
        return {"billed_usd": "1.000000", "output": _output()}

    result = FormulaProposalAdapter(config, ledger, provider).propose(_request("proposal:req:retry"))
    assert result["settled_usd"] == "1.000000"
    assert keys == ["proposal:req:retry", "proposal:req:retry"]
    row = ledger.status("proposal:req:retry")
    assert row["attempts"] == 2
    assert row["settled_micro_usd"] == 1_000_000


def test_invalid_output_is_charged_once_but_rejected_from_quarantine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SIGMA_FORMULA_LLM_API_KEY", SECRET)
    config = _config(enabled=True)
    ledger = SpendLedger(tmp_path / "invalid.sqlite", config)

    def provider(_request: object, _secret: str) -> dict[str, object]:
        return {"billed_usd": "0.125000", "output": {"schema_version": "bad", "proposals": []}}

    adapter = FormulaProposalAdapter(config, ledger, provider)
    first = adapter.propose(_request("proposal:req:invalid"))
    second = adapter.propose(_request("proposal:req:invalid"))
    assert first["decision"] == "rejected_quarantine"
    assert second["decision"] == "rejected_quarantine"
    assert ledger.status("proposal:req:invalid")["status"] == "settled_invalid"
    assert ledger.telemetry()["settled_usd"] == "0.125000"


def test_atomic_cap_and_lineage_replay_controls(tmp_path: Path) -> None:
    config = _config(enabled=True)
    ledger = SpendLedger(tmp_path / "cap.sqlite", config)
    for index in range(100):
        request = ProposalRequest(
            **{
                **_request(f"proposal:cap:{index:04d}").__dict__,
                "maximum_call_usd": "5.000000",
            }
        )
        ledger.reserve(request)
    with pytest.raises(ProposalAdapterError, match=r"\$500 cap"):
        ledger.reserve(
            ProposalRequest(
                **{
                    **_request("proposal:cap:overflow").__dict__,
                    "maximum_call_usd": "5.000000",
                }
            )
        )
    with pytest.raises(ProposalAdapterError, match="different lineage"):
        ledger.reserve(_request("proposal:cap:0000", prompt="Different prompt"))


def test_secrets_prompt_and_output_are_absent_from_ledger_and_telemetry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SIGMA_FORMULA_LLM_API_KEY", SECRET)
    config = _config(enabled=True)
    ledger_path = tmp_path / "safe.sqlite"
    ledger = SpendLedger(ledger_path, config)
    prompt = "A uniquely identifying private prompt phrase"

    def provider(_request: object, _secret: str) -> dict[str, object]:
        return {"billed_usd": "0.010000", "output": _output()}

    FormulaProposalAdapter(config, ledger, provider).propose(_request("proposal:req:safe", prompt))
    with sqlite3.connect(ledger_path) as connection:
        dump = "\n".join(connection.iterdump())
    telemetry = json.dumps(ledger.telemetry(), sort_keys=True)
    for forbidden in (SECRET, prompt, "R + alpha*coherence(phi)"):
        assert forbidden not in dump
        assert forbidden not in telemetry


def test_forbidden_inputs_schema_and_settlement_overage_fail_closed(tmp_path: Path) -> None:
    config = _config(enabled=True)
    ledger = SpendLedger(tmp_path / "negative.sqlite", config)
    with pytest.raises(ProposalAdapterError, match="forbidden"):
        ledger.reserve(_request("proposal:req:forbid", "Fit a dark matter halo to redshift"))
    with pytest.raises(ProposalAdapterError, match="schema version"):
        validate_proposal_output({"schema_version": "wrong", "proposals": [_output()]})
    request = _request("proposal:req:overage")
    ledger.reserve(request)
    ledger.begin_attempt(request.request_id)
    with pytest.raises(ProposalAdapterError, match="exceeds reservation"):
        ledger.settle(request.request_id, billed_usd="2.000000", output_sha256="3" * 64)


def test_readiness_artifact_is_deterministic_disabled_and_zero_spend() -> None:
    first = build_readiness_artifact(REPO_ROOT, CONFIG_PATH)
    assert first == build_readiness_artifact(REPO_ROOT, CONFIG_PATH)
    assert first["maximum_total_usd"] == "500.000000"
    assert first["default_paid_calls_enabled"] is False
    assert first["paid_spend_usd"] == "0.000000"
    assert first["network_calls_made"] == 0
    assert first["credential_persistence"] is False
    assert first["output_status"] == "quarantine_until_downstream_validation"


def test_checked_in_artifact_matches() -> None:
    assert json.loads(ARTIFACT_PATH.read_text(encoding="utf-8")) == build_readiness_artifact(
        REPO_ROOT, CONFIG_PATH
    )

