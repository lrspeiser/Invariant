import hashlib
import json
from pathlib import Path

from sigma_theory_compiler.composite_covariant_lift_campaign import (
    compile_composite_aether_action,
)
from sigma_theory_compiler.formal_backend import load_field_contract
from sigma_theory_compiler.high_throughput import (
    build_basis,
    candidate_id,
    correction_expression,
    decode_ordinal,
)
from sigma_theory_compiler.promotion_orchestrator import ELIGIBILITY

ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "configs" / "generator_v2_billion.json"
CONTRACT = ROOT / "configs" / "covariant_field_contract.json"
ARTIFACT = ROOT / "runs" / "engine" / "composite-covariant-lift-campaign.json"
CONTRACT_SHA = "c31e42645fd6b7b9bd834cd77254451704946e6f6385517011343a74eb692c7c"
DICTIONARY_FILE_SHA = "54a50e6f20d8e8d59d7d34d4186a615637353175b44cca636c41fa80b873f7bf"
DICTIONARY_CONTENT_SHA = "0179ce22a456fe5845414563d3e97a0e9869f612caa8c61792da9b722020ff73"
SOURCE_SHA = "330c8f04e2da2e64dce39cf43f570e8a908367d3fd6e402ad90a8df55b913399"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _candidate(ordinal: int) -> dict:
    generator = _load(GENERATOR)
    decoded = decode_ordinal(
        generator["basis_count"], generator["max_action_terms"], ordinal
    )
    return {
        "candidate_id": candidate_id(generator["protocol_version"], decoded),
        "ordinal": ordinal,
        "term_ids": list(decoded["term_ids"]),
        "signs": list(decoded["signs"]),
        "correction_expression": correction_expression(
            decoded, build_basis(generator["basis_count"])
        ),
        "source_manifest_sha256": "a" * 64,
        "data_eligibility": dict(ELIGIBILITY),
    }


def _compile(ordinal: int) -> dict:
    return compile_composite_aether_action(
        _candidate(ordinal),
        load_field_contract(CONTRACT),
        field_contract_file_sha256=CONTRACT_SHA,
        static_dictionary_file_sha256=DICTIONARY_FILE_SHA,
        static_dictionary_content_sha256=DICTIONARY_CONTENT_SHA,
        source_sha256=SOURCE_SHA,
    )


def test_previously_unsupported_q_functions_receive_exact_static_shape_only() -> None:
    result = _compile(677)
    assert result["decision"] == "mapped"
    assert result["static_shape_certificate"]["universal_shape_residual"] == "0"
    assert result["action_ir"]["universal_matter_coupling_preserved"] is True
    provenance = result["covariant_action_provenance"]
    body = {key: value for key, value in provenance.items() if key != "provenance_binding_sha256"}
    assert provenance["provenance_binding_sha256"] == hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert result["formal_outcome"]["decision"] == "blocked"
    assert result["solar_bundle_outcome"]["bundle_generated"] is False
    assert result["galaxy_prediction_outcome"]["prediction_bundle_generated"] is False


def test_nonlinear_and_mixed_q_shape_maps_without_claiming_an_adm_adapter() -> None:
    result = _compile(3008915)
    assert result["decision"] == "mapped"
    expression = result["action_ir"]["terms"][1]["typed_expression_srepr"]
    assert "Q_a_u" in expression and "X_a_u" in expression
    assert result["formal_outcome"]["blocker"] == (
        "missing_candidate_specific_adm_dirac_principal_adapter_for_composite_qx_action"
    )
    assert result["data_eligibility"] == ELIGIBILITY


def test_z_remains_a_decisive_action_rejection() -> None:
    result = _compile(0)
    assert result == {
        "decision": "reject",
        "candidate_id": _candidate(0)["candidate_id"],
        "reason": "forbidden_baryonic_action_atom",
        "data_eligibility": ELIGIBILITY,
    }


def test_campaign_artifact_hash_and_fail_closed_counts() -> None:
    artifact = _load(ARTIFACT)
    body = {key: value for key, value in artifact.items() if key != "content_sha256"}
    assert artifact["content_sha256"] == hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert artifact["newly_exact_composite_lift_count"] == 68
    assert artifact["total_exact_lift_count"] == 70
    assert artifact["formal_outcome_counts"] == {"blocked": 68, "reject": 2}
    assert artifact["solar_bundle_outcome_counts"] == {"blocked": 70}
    assert artifact["galaxy_prediction_outcome_counts"] == {"blocked": 70}
    assert artifact["data_eligibility"] == ELIGIBILITY
