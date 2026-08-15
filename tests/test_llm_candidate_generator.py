from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from sigma_theory_compiler.llm_candidate_generator import (
    RESPONSE_SCHEMA_VERSION,
    LLMBudgetState,
    LLMPolicy,
    LLMProposalManifest,
    LLMProposalRequest,
    generate_llm_candidates,
    llm_source_bindings,
    validate_llm_manifest,
)
from sigma_theory_compiler.sigma_core import (
    ArtifactKind,
    ArtifactRef,
    DomainPackRef,
    OutcomeStatus,
    SchemaViolation,
    SourceBinding,
    canonical_sha256,
)

ROOT = Path(__file__).resolve().parents[1]
DOMAIN = DomainPackRef("synthetic.llm", "1.0", "2" * 64)
POLICY = LLMPolicy(
    provider_id="mock.provider",
    credential_env_var="MOCK_PROVIDER_API_KEY",
    maximum_total_micro_usd=10_000,
    maximum_call_micro_usd=5_000,
    maximum_calls=2,
    maximum_prompt_tokens=100,
    maximum_completion_tokens=100,
    maximum_response_bytes=20_000,
    maximum_proposals=8,
)
REQUEST = LLMProposalRequest(
    request_id="proposal.fixture.0001",
    prompt="TRANSIENT_PROMPT_SENTINEL propose bounded formulas",
    prompt_token_count=7,
    completion_token_limit=20,
    deterministic_seed=42,
    context=(ArtifactRef("sig-context-fixture", "3" * 64),),
)
BUDGET = LLMBudgetState(calls=0, spent_micro_usd=100, prompt_tokens=2, completion_tokens=3)


def _sources() -> tuple[SourceBinding, ...]:
    return llm_source_bindings(ROOT)


def _response(
    *, billed: int = 1_234, prompt_tokens: int = 7, completion_tokens: int = 11
) -> dict[str, Any]:
    return {
        "schema_version": RESPONSE_SCHEMA_VERSION,
        "request_id": REQUEST.request_id,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "billed_micro_usd": billed,
        },
        "proposals": [
            {
                "proposal_id": "proposal.alpha",
                "kind": "formula",
                "statement": "A syntactic candidate formula alpha.",
                "representation": {"dsl": "x+x"},
                "assumptions": ["x is real."],
            },
            {
                "proposal_id": "proposal.alpha.duplicate",
                "kind": "formula",
                "statement": "A syntactic candidate formula alpha.",
                "representation": {"dsl": "x+x"},
                "assumptions": ["x is real."],
            },
            {
                "proposal_id": "proposal.beta",
                "kind": "conjecture",
                "statement": "A syntactic candidate conjecture beta.",
                "representation": {"dsl": "x*x"},
                "assumptions": [],
            },
        ],
    }


def _run(
    provider: Any = None,
    *,
    policy: LLMPolicy = POLICY,
    request: LLMProposalRequest = REQUEST,
    budget: LLMBudgetState = BUDGET,
) -> LLMProposalManifest:
    callback = provider if provider is not None else (lambda _request: _response())
    return generate_llm_candidates(policy, request, budget, DOMAIN, _sources(), callback)


def _reseal_receipt(value: dict[str, Any]) -> None:
    body = {key: child for key, child in value.items() if key != "receipt_sha256"}
    value["receipt_sha256"] = canonical_sha256(body)


def _reseal_lineage(value: dict[str, Any]) -> None:
    body = {key: child for key, child in value.items() if key != "lineage_sha256"}
    value["lineage_sha256"] = canonical_sha256(body)


def _reseal_manifest(value: dict[str, Any]) -> None:
    body = {key: child for key, child in value.items() if key != "manifest_sha256"}
    value["manifest_sha256"] = canonical_sha256(body)


def test_success_emits_quarantined_sigma_core_candidates_and_dedup_lineage() -> None:
    manifest = _run()

    assert manifest.status is OutcomeStatus.PASS
    assert len(manifest.candidates) == 2
    assert [item.kind for item in manifest.candidates] == [
        ArtifactKind.FORMULA,
        ArtifactKind.CONJECTURE,
    ]
    assert [item.duplicate for item in manifest.lineage] == [False, True, False]
    assert manifest.lineage[0].candidate == manifest.lineage[1].candidate
    assert manifest.receipt.proposal_count == 3
    assert manifest.receipt.unique_count == 2
    assert manifest.receipt.duplicate_count == 1
    assert all(
        item.claims == ("llm_generated_proposal", "requires_downstream_gates")
        for item in manifest.candidates
    )


def test_exact_integer_budget_token_call_and_byte_accounting() -> None:
    manifest = _run()
    receipt = manifest.receipt

    assert receipt.usage.to_dict() == {
        "prompt_tokens": 7,
        "completion_tokens": 11,
        "billed_micro_usd": 1_234,
    }
    assert receipt.budget_before == BUDGET
    assert receipt.budget_after == LLMBudgetState(1, 1_334, 9, 14)
    assert receipt.response_bytes == len(
        json.dumps(_response(), sort_keys=True, separators=(",", ":")).encode()
    )
    assert receipt.response_sha256 == canonical_sha256(_response())
    assert receipt.charge_applied is True


def test_callback_is_provider_neutral_and_receives_env_name_never_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "SUPER_SECRET_VALUE_MUST_NEVER_PERSIST"
    monkeypatch.setenv(POLICY.credential_env_var, secret)
    seen = {}

    def provider(payload: dict[str, Any]) -> dict[str, Any]:
        seen.update(payload)
        return _response()

    manifest = _run(provider)
    serialized = json.dumps(manifest.to_dict(), sort_keys=True)

    assert seen["credential_env_var"] == POLICY.credential_env_var
    assert seen["prompt"] == REQUEST.prompt
    assert secret not in json.dumps(seen)
    assert secret not in serialized
    assert REQUEST.prompt not in serialized
    assert "prompt" not in manifest.request_contract
    assert manifest.receipt.credential_env_var == POLICY.credential_env_var


def test_candidate_provenance_binds_request_response_policy_context_and_sources() -> None:
    manifest = _run()

    assert manifest.domain_pack == DOMAIN
    assert all(item.provenance.domain_pack == DOMAIN for item in manifest.candidates)
    assert all(item.provenance.inputs == REQUEST.context for item in manifest.candidates)
    assert all(item.provenance.sources == _sources() for item in manifest.candidates)
    assert manifest.receipt.policy_sha256 == canonical_sha256(POLICY.to_dict())
    assert manifest.receipt.request_sha256 == REQUEST.request_sha256
    assert validate_llm_manifest(manifest, request=REQUEST, project_root=ROOT) == manifest


def test_deterministic_mock_replay_is_byte_identical() -> None:
    first = _run()
    second = _run()

    assert first.to_dict() == second.to_dict()
    assert LLMProposalManifest.from_dict(first.to_dict()) == first
    assert first.manifest_sha256 == second.manifest_sha256


@pytest.mark.parametrize(
    ("policy", "budget", "request_case", "reason"),
    [
        (replace(POLICY, maximum_calls=1), LLMBudgetState(calls=1), REQUEST, "call_cap_reached"),
        (
            POLICY,
            LLMBudgetState(spent_micro_usd=10_000),
            REQUEST,
            "total_budget_exhausted",
        ),
        (
            POLICY,
            LLMBudgetState(spent_micro_usd=9_000),
            REQUEST,
            "total_budget_reservation_unavailable",
        ),
        (
            replace(POLICY, maximum_prompt_tokens=6),
            BUDGET,
            REQUEST,
            "prompt_token_cap_exceeded",
        ),
        (
            replace(POLICY, maximum_completion_tokens=19),
            BUDGET,
            REQUEST,
            "completion_token_request_cap_exceeded",
        ),
    ],
)
def test_preflight_caps_block_without_call_or_partial_outputs(
    policy: LLMPolicy,
    budget: LLMBudgetState,
    request_case: LLMProposalRequest,
    reason: str,
) -> None:
    called = False

    def provider(_payload: dict[str, Any]) -> dict[str, Any]:
        nonlocal called
        called = True
        return _response()

    manifest = _run(provider, policy=policy, budget=budget, request=request_case)

    assert manifest.status is OutcomeStatus.BLOCK
    assert manifest.receipt.reason_codes == (reason,)
    assert manifest.candidates == manifest.lineage == ()
    assert manifest.receipt.budget_after == budget
    assert called is False


@pytest.mark.parametrize(
    ("response", "budget", "reasons"),
    [
        (_response(prompt_tokens=8), BUDGET, ("prompt_token_accounting_mismatch",)),
        (_response(completion_tokens=21), BUDGET, ("completion_token_cap_exceeded",)),
        (_response(billed=5_001), BUDGET, ("call_budget_exceeded",)),
        (
            _response(billed=7_000, prompt_tokens=8, completion_tokens=21),
            LLMBudgetState(spent_micro_usd=4_000),
            (
                "call_budget_exceeded",
                "completion_token_cap_exceeded",
                "prompt_token_accounting_mismatch",
                "total_budget_exceeded",
            ),
        ),
    ],
)
def test_reported_usage_overages_reject_with_exact_charge_and_no_candidates(
    response: dict[str, Any], budget: LLMBudgetState, reasons: tuple[str, ...]
) -> None:
    manifest = _run(lambda _payload: response, budget=budget)

    assert manifest.status is OutcomeStatus.REJECT
    assert manifest.receipt.reason_codes == reasons
    assert manifest.receipt.usage is not None
    assert manifest.receipt.call_recorded is True
    assert manifest.receipt.charge_applied is True
    assert manifest.receipt.budget_after == budget.record_call(manifest.receipt.usage)
    assert manifest.candidates == manifest.lineage == ()


def test_response_byte_cap_is_exact_and_fail_closed() -> None:
    manifest = _run(policy=replace(POLICY, maximum_response_bytes=1))

    assert manifest.status is OutcomeStatus.REJECT
    assert manifest.receipt.reason_codes == ("response_byte_cap_exceeded",)
    assert manifest.receipt.response_bytes > 1
    assert manifest.receipt.response_sha256 == canonical_sha256(_response())
    assert manifest.receipt.usage is None
    assert manifest.receipt.budget_after == BUDGET.record_call()


@pytest.mark.parametrize(
    ("response", "charge_applied"),
    [
        (None, False),
        ({"unexpected": "shape"}, False),
        ({**_response(), "request_id": "proposal.wrong"}, True),
        (
            {
                **_response(),
                "usage": {
                    "prompt_tokens": True,
                    "completion_tokens": 1,
                    "billed_micro_usd": 1,
                },
            },
            False,
        ),
        (
            {**_response(), "proposals": [{**_response()["proposals"][0], "kind": "theorem"}]},
            True,
        ),
        (
            {
                **_response(),
                "proposals": [
                    {
                        **_response()["proposals"][0],
                        "representation": {"x": 1.5},
                    }
                ],
            },
            False,
        ),
    ],
)
def test_malformed_provider_payloads_reject_without_partial_outputs(
    response: Any, charge_applied: bool
) -> None:
    manifest = _run(lambda _payload: response)

    assert manifest.status is OutcomeStatus.REJECT
    assert manifest.receipt.reason_codes == ("malformed_provider_response",)
    assert manifest.candidates == manifest.lineage == ()
    assert manifest.receipt.call_recorded is True
    assert manifest.receipt.charge_applied is charge_applied
    expected_usage = manifest.receipt.usage if charge_applied else None
    assert manifest.receipt.budget_after == BUDGET.record_call(expected_usage)


def test_provider_exception_is_typed_error_without_exception_text_or_charge() -> None:
    def provider(_payload: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("SECRET_PROVIDER_EXCEPTION_BODY")

    manifest = _run(provider)
    serialized = json.dumps(manifest.to_dict())

    assert manifest.status is OutcomeStatus.ERROR
    assert manifest.receipt.reason_codes == ("provider_exception",)
    assert manifest.receipt.response_sha256 is None
    assert manifest.receipt.call_recorded is True
    assert manifest.receipt.budget_after == BUDGET.record_call()
    assert "SECRET_PROVIDER_EXCEPTION_BODY" not in serialized


@pytest.mark.parametrize("value", [True, -1, 1.5])
def test_money_token_and_call_accounting_rejects_non_integer_values(value: Any) -> None:
    with pytest.raises(SchemaViolation):
        LLMBudgetState(calls=value)
    with pytest.raises(SchemaViolation):
        replace(POLICY, maximum_call_micro_usd=value)


def test_source_bindings_are_exact_and_live_hash_tamper_fails() -> None:
    for binding in _sources():
        assert binding.file_sha256 == hashlib.sha256((ROOT / binding.path).read_bytes()).hexdigest()

    value = _run().to_dict()
    value["sources"][0]["file_sha256"] = "f" * 64
    _reseal_manifest(value)
    with pytest.raises(SchemaViolation, match="source bytes changed"):
        validate_llm_manifest(value, project_root=ROOT)


def test_unknown_keys_and_boundary_flag_tampers_fail_closed() -> None:
    value = _run().to_dict()
    value["unknown"] = True
    with pytest.raises(SchemaViolation, match="keys changed"):
        validate_llm_manifest(value)

    value = _run().to_dict()
    value["receipt"]["promotion_allowed"] = True
    _reseal_receipt(value["receipt"])
    _reseal_manifest(value)
    with pytest.raises(SchemaViolation, match="boundary changed"):
        validate_llm_manifest(value)


def test_resealed_receipt_and_lineage_tampers_fail_semantic_replay() -> None:
    receipt = _run().to_dict()
    receipt["receipt"]["response_sha256"] = "f" * 64
    _reseal_receipt(receipt["receipt"])
    _reseal_manifest(receipt)
    with pytest.raises(SchemaViolation, match="deterministic replay"):
        validate_llm_manifest(receipt)

    lineage = _run().to_dict()
    lineage["lineage"][0]["provider_proposal_id"] = "proposal.tampered"
    _reseal_lineage(lineage["lineage"][0])
    _reseal_manifest(lineage)
    with pytest.raises(SchemaViolation, match="content binding changed"):
        validate_llm_manifest(lineage)


def test_request_replay_and_detached_contract_tamper_fail() -> None:
    value = _run().to_dict()
    parsed = LLMProposalManifest.from_dict(copy.deepcopy(value))
    value["request_contract"]["prompt_sha256"] = "f" * 64
    assert parsed.to_dict() != value

    other = replace(REQUEST, prompt="different transient prompt")
    with pytest.raises(SchemaViolation, match="request replay changed"):
        validate_llm_manifest(parsed, request=other)


def test_no_network_provider_sdk_secret_access_or_body_persistence_code_paths() -> None:
    source = (ROOT / "src/sigma_theory_compiler/llm_candidate_generator.py").read_text(
        encoding="utf-8"
    )
    forbidden_imports = ("import requests", "import httpx", "import openai", "import anthropic")

    assert not any(item in source for item in forbidden_imports)
    assert "os.environ" not in source
    manifest_text = json.dumps(_run().to_dict(), sort_keys=True)
    assert "TRANSIENT_PROMPT_SENTINEL" not in manifest_text
    assert '"request_body_persisted": false' in manifest_text
    assert '"response_body_persisted": false' in manifest_text
    assert '"credential_value_accessed": false' in manifest_text
