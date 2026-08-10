import copy
import hashlib
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.composite_covariant_lift_campaign import (
    compile_composite_aether_action,
)
from sigma_theory_compiler.composite_q_degenerate_formal_campaign import (
    evaluate_zero_local_acceleration_family,
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
ARTIFACT = ROOT / "runs" / "engine" / "composite-q-degenerate-formal-campaign.json"
CONTRACT_SHA = "c31e42645fd6b7b9bd834cd77254451704946e6f6385517011343a74eb692c7c"
DICTIONARY_FILE_SHA = "54a50e6f20d8e8d59d7d34d4186a615637353175b44cca636c41fa80b873f7bf"
DICTIONARY_CONTENT_SHA = "0179ce22a456fe5845414563d3e97a0e9869f612caa8c61792da9b722020ff73"
SOURCE_SHA = "330c8f04e2da2e64dce39cf43f570e8a908367d3fd6e402ad90a8df55b913399"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _lift(ordinal: int) -> dict:
    generator = _load(GENERATOR)
    decoded = decode_ordinal(
        generator["basis_count"], generator["max_action_terms"], ordinal
    )
    candidate = {
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
    return compile_composite_aether_action(
        candidate,
        load_field_contract(CONTRACT),
        field_contract_file_sha256=CONTRACT_SHA,
        static_dictionary_file_sha256=DICTIONARY_FILE_SHA,
        static_dictionary_content_sha256=DICTIONARY_CONTENT_SHA,
        source_sha256=SOURCE_SHA,
    )


def test_linear_q_quadratic_limit_is_rejected_by_exact_rank_jump() -> None:
    result = evaluate_zero_local_acceleration_family(_lift(677))
    assert result["decision"] == "reject"
    assert result["linearized_coefficients"] == {
        "F_at_origin": "0",
        "F_X_at_origin": "0",
        "F_Q_at_origin": "1/2",
    }
    assert result["weak_field_certificate"]["chain_rule_residual"] == "0"
    assert result["adm_kinetic_preflight"]["spatial_vector_rank_at_k_zero"] == 0
    assert result["adm_kinetic_preflight"]["spatial_vector_rank_at_k_nonzero"] == 3
    assert result["solar_bundle_outcome"]["bundle_generated"] is False


def test_purely_nonlinear_quadratic_limit_has_zero_vector_principal_symbol() -> None:
    result = evaluate_zero_local_acceleration_family(_lift(3008915))
    assert result["decision"] == "reject"
    assert result["reason"] == "composite_aether_sector_has_no_quadratic_vector_evolution"
    assert result["linearized_coefficients"]["F_X_at_origin"] == "0"
    assert result["linearized_coefficients"]["F_Q_at_origin"] == "0"
    assert result["principal_preflight"]["normalized_fourier_factor"] == "0"


def test_nonzero_local_acceleration_family_remains_blocked() -> None:
    result = evaluate_zero_local_acceleration_family(_lift(693))
    assert result["decision"] == "blocked"
    assert result["blocker"] == (
        "nonzero_local_acceleration_coefficient_outside_bounded_family"
    )


def test_action_and_provenance_tampering_fail_closed() -> None:
    lift = _lift(677)
    tampered = copy.deepcopy(lift)
    tampered["action_ir"]["ordinal"] += 1
    with pytest.raises(ValueError, match="action content hash"):
        evaluate_zero_local_acceleration_family(tampered)
    tampered = copy.deepcopy(lift)
    tampered["covariant_action_provenance"]["input_action_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="provenance binding"):
        evaluate_zero_local_acceleration_family(tampered)


def test_campaign_artifact_is_hash_bound_and_keeps_nonfamily_actions_blocked() -> None:
    artifact = _load(ARTIFACT)
    body = {key: value for key, value in artifact.items() if key != "content_sha256"}
    assert artifact["content_sha256"] == hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert artifact["input_composite_candidate_count"] == 68
    assert artifact["decision_counts"] == {"blocked": 47, "reject": 21}
    assert artifact["reason_counts"] == {
        "aligned_velocity_hessian_rank_jumps_between_k_zero_and_nonzero": 15,
        "composite_aether_sector_has_no_quadratic_vector_evolution": 6,
        "nonzero_local_acceleration_coefficient_outside_bounded_family": 47,
    }
    assert artifact["formal_pass_count"] == 0
    assert artifact["solar_bundle_generated_count"] == 0
    assert artifact["data_eligibility"] == ELIGIBILITY
