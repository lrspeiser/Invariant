import copy
import hashlib
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.composite_covariant_lift_campaign import (
    compile_composite_aether_action,
)
from sigma_theory_compiler.composite_positive_qx_tilt_campaign import (
    evaluate_positive_qx_tilt_family,
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
ARTIFACT = ROOT / "runs" / "engine" / "composite-positive-qx-tilt-campaign.json"
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


def test_positive_qx_action_has_exact_nonreal_generic_tilt_pair() -> None:
    result = evaluate_positive_qx_tilt_family(_lift(24000575))
    assert result["decision"] == "reject"
    assert result["linearized_coefficients"] == {
        "F_X_at_origin": "1/2",
        "F_Q_at_origin": "1",
    }
    assert result["adm_kinetic_preflight"]["status"] == (
        "pass_necessary_condition_only"
    )
    assert result["dirac_preflight"]["status"] == (
        "unresolved_after_uniform_tangent_hessian"
    )
    principal = result["principal_preflight"]
    assert principal["status"] == "reject"
    assert principal["nonreal_root_count"] == 2
    assert set(principal["root_substitution_residuals"].values()) == {"0"}
    assert result["solar_bundle_outcome"]["bundle_generated"] is False


def test_nonpositive_coefficient_is_not_absorbed_into_this_family() -> None:
    result = evaluate_positive_qx_tilt_family(_lift(693))
    assert result["decision"] == "blocked"
    assert result["blocker"] == "strictly_positive_qx_coefficients_not_certified"


def test_action_tampering_fails_closed() -> None:
    tampered = copy.deepcopy(_lift(24000575))
    tampered["action_ir"]["terms"][1]["typed_expression_srepr"] = "Integer(0)"
    with pytest.raises(ValueError, match="action content hash"):
        evaluate_positive_qx_tilt_family(tampered)


def test_campaign_artifact_closes_the_remaining_family_without_promotion() -> None:
    artifact = _load(ARTIFACT)
    body = {key: value for key, value in artifact.items() if key != "content_sha256"}
    assert artifact["content_sha256"] == hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert artifact["input_candidate_count"] == 20
    assert artifact["decision_counts"] == {"reject": 20}
    assert artifact["reason_counts"] == {
        "generic_tilt_lab_frequency_polynomial_has_nonreal_conjugate_pair": 20
    }
    assert artifact["formal_pass_count"] == 0
    assert artifact["remaining_formal_blocked_count"] == 0
    assert artifact["solar_bundle_generated_count"] == 0
    assert artifact["data_eligibility"] == ELIGIBILITY
