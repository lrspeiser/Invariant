from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.quartic_scalar_hessian_curl_invariance_gate import (
    CONFIG_PATH,
    EXPECTED_CLAIM_SEALS,
    EXPECTED_DATA_SEALS,
    FIRST_BLOCKER,
    OUTPUT_PATH,
    _content_sha,
    _coordinate_invariance_theorem,
    _independent_curl_manifest,
    _load_bound,
    _sha,
    _torsion_free_identity,
    _validate_config,
    _validate_result,
    build_gate,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / CONFIG_PATH
ARTIFACT = ROOT / OUTPUT_PATH
SCALAR = ROOT / "runs/physics-language/quartic-scalar-hessian-d2-integrability-gate/campaign.json"


def _reseal(value: dict) -> dict:
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    return {**body, "content_sha256": _sha(body)}


def test_exact_gate_matches_checked_artifact_and_replays() -> None:
    first = build_gate(CONFIG)
    second = build_gate(CONFIG)
    checked = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert first == second == checked
    _validate_result(first, root=ROOT)
    assert first["content_sha256"] == _content_sha(first)
    assert first["decision_counts"] == {"pass": 0, "reject": 0, "blocked": 12}
    assert first["first_blocker"] == FIRST_BLOCKER


def test_all_candidates_have_exact_nonzero_antisymmetric_curl_certificates() -> None:
    result = build_gate(CONFIG)
    assert len(result["candidate_records"]) == 12
    for record in result["candidate_records"]:
        manifest = record["curl_two_form_manifest"]
        assert manifest["ordered_nonzero_pair_count"] == 24
        assert manifest["independent_nonzero_pair_count"] == 12
        assert manifest["ordered_nonzero_component_count"] == 30
        assert manifest["independent_nonzero_component_count"] == 15
        assert manifest["ordered_antisymmetry_exact"] is True
        assert len(manifest["independent_components"]) == 12
        assert manifest["content_sha256"] == _sha(
            {key: item for key, item in manifest.items() if key != "content_sha256"}
        )
        assert record["registered_naive_chunk_curl_nonzero"] is True
        assert record["coordinate_only_repair_possible"] is False
        assert record["torsion_free_domain_connection_repair_possible"] is False
        assert record["corrected_source_or_output_bundle_or_torsion_repair_ruled_out"] is False
        assert record["candidate_decision"] == "blocked"
        assert record["candidate_rejection_authorized"] is False


def test_independent_manifest_reduces_the_registered_ordered_residuals_exactly() -> None:
    scalar = json.loads(SCALAR.read_text(encoding="utf-8"))
    for record in scalar["candidate_records"]:
        manifest = _independent_curl_manifest(record)
        assert (
            sum(component["component_count"] for component in manifest["independent_components"])
            == 15
        )
        assert all(
            component["left_atom"] < component["right_atom"]
            for component in manifest["independent_components"]
        )


def test_coordinate_and_torsion_free_identities_are_exact_and_narrow() -> None:
    coordinate = _coordinate_invariance_theorem()
    torsion_free = _torsion_free_identity()
    assert coordinate == {
        "pullback_law": "d(phi^*J_A)=phi^*(dJ_A)",
        "coordinate_Jacobian_dimension": 9,
        "exterior_square_dimension": 36,
        "determinant_identity": "det(Lambda^2 P)=det(P)^8",
        "determinant_exponent": 8,
        "determinant_proof": {
            "representation_identity": "Lambda^2(PQ)=Lambda^2(P)Lambda^2(Q)",
            "diagonal_basis_product": "product_(i<j)(p_i*p_j)=product_i(p_i)^8",
            "extension": "polynomial_identity_from_dense_diagonalizable_matrices",
        },
        "invertible_pullback_has_trivial_kernel": True,
        "conclusion": "dJ_A_nonzero_is_invariant_under_local_coordinate_diffeomorphisms",
        "proved": True,
    }
    assert torsion_free["symbolic_residual"] == "0"
    assert torsion_free["proved"] is True
    result = build_gate(CONFIG)
    assert result["claim_seals"] == EXPECTED_CLAIM_SEALS
    assert result["data_seals"] == EXPECTED_DATA_SEALS
    assert result["claim_seals"]["corrected_source_Jacobian_ruled_out"] is False
    assert result["claim_seals"]["output_bundle_connection_repair_ruled_out"] is False
    assert result["claim_seals"]["torsionful_domain_connection_repair_ruled_out"] is False


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("remove_component", "result boundary"),
        ("claim_global_h7", "result boundary"),
        ("reject_candidate", "result boundary"),
        ("forge_predecessor", "predecessor binding"),
        ("forge_local_source", "local source binding"),
    ],
)
def test_resealed_semantic_tampering_fails_closed(mutation: str, message: str) -> None:
    value = json.loads(json.dumps(build_gate(CONFIG)))
    if mutation == "remove_component":
        value["candidate_records"][0]["curl_two_form_manifest"]["independent_components"].pop()
    elif mutation == "claim_global_h7":
        value["claim_seals"]["global_H7_energy_closed"] = True
    elif mutation == "reject_candidate":
        value["candidate_records"][0]["candidate_decision"] = "reject"
    elif mutation == "forge_predecessor":
        value["source_bindings"]["scalar_hessian_d2"]["content_sha256"] = "0" * 64
    else:
        value["source_bindings"]["source"]["file_sha256"] = "0" * 64
    with pytest.raises(ValueError, match=message):
        _validate_result(_reseal(value), root=ROOT)


def test_config_and_bound_predecessor_contracts_fail_closed() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["policies"]["global_H7"] = "pass"
    with pytest.raises(ValueError, match="config boundary"):
        _validate_config(config)

    with pytest.raises(ValueError, match="path escapes"):
        _load_bound(
            ROOT,
            {"path": "../outside.json", "file_sha256": "0" * 64, "content_sha256": "0" * 64},
        )

    binding = json.loads(CONFIG.read_text(encoding="utf-8"))["predecessors"]["scalar_hessian_d2"]
    forged = {**binding, "file_sha256": "0" * 64}
    with pytest.raises(ValueError, match="file hash"):
        _load_bound(ROOT, forged)


def test_artifact_binds_the_exact_four_file_lane() -> None:
    value = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    bindings = value["source_bindings"]
    expected_local = {
        "config": CONFIG_PATH,
        "source": ("src/sigma_theory_compiler/quartic_scalar_hessian_curl_invariance_gate.py"),
        "test": "tests/test_quartic_scalar_hessian_curl_invariance_gate.py",
    }
    for label, relative in expected_local.items():
        assert bindings[label] == {
            "path": relative,
            "file_sha256": hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(),
        }
    assert ARTIFACT.relative_to(ROOT).as_posix() == OUTPUT_PATH
