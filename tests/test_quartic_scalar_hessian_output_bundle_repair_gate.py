from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.quartic_scalar_hessian_output_bundle_repair_gate import (
    CLAIM_SEALS,
    CONFIG_PATH,
    EXPECTED_SEALS,
    FIRST_BLOCKER,
    OUTPUT_PATH,
    SOURCE_PATH,
    TEST_PATH,
    _load_bound,
    _validate_config,
    _validate_result,
    build_gate,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / CONFIG_PATH
ARTIFACT = ROOT / OUTPUT_PATH


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _reseal(value: dict[str, object]) -> dict[str, object]:
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    return {**body, "content_sha256": hashlib.sha256(_canonical(body)).hexdigest()}


@pytest.fixture(scope="module")
def gate() -> dict[str, object]:
    return build_gate(CONFIG)


def test_exact_gate_matches_checked_artifact_and_replays(gate: dict[str, object]) -> None:
    checked = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert gate == checked
    assert build_gate(CONFIG) == checked
    assert (
        checked["content_sha256"]
        == hashlib.sha256(
            _canonical({key: value for key, value in checked.items() if key != "content_sha256"})
        ).hexdigest()
    )


def test_rank_one_one_form_makes_arbitrary_domain_torsion_insufficient(
    gate: dict[str, object],
) -> None:
    assert gate["decision_counts"] == {"pass": 0, "reject": 0, "blocked": 12}
    assert gate["gate_counts"]["arbitrary_domain_torsion_pair_no_go_certificates"] == 144
    assert gate["gate_counts"]["torsionful_domain_connection_repairs_admitted"] == 0
    for row in gate["candidate_records"]:
        assert row["registered_one_form"]["rank"] == 1
        assert row["registered_one_form"]["image_output_rows"] == [10]
        certificates = row["torsion_no_go"]["independent_pair_certificates"]
        assert len(certificates) == 12
        assert all(item["one_form_column_rank"] == 1 for item in certificates)
        assert all(item["curl_augmented_rank"] == 2 for item in certificates)
        assert not any(item["torsion_repair_possible"] for item in certificates)


def test_shared_output_connection_system_has_a_sparsest_exact_repair(
    gate: dict[str, object],
) -> None:
    counts = gate["gate_counts"]
    assert counts["output_connection_equations_per_candidate"] == 396
    assert counts["output_connection_unknowns_per_candidate"] == 99
    assert counts["output_connection_coefficient_rank"] == 88
    assert counts["output_connection_augmented_rank"] == 88
    assert counts["output_connection_affine_dimension"] == 11
    assert counts["sparse_output_connection_coefficients_per_candidate"] == 6
    for row in gate["candidate_records"]:
        repair = row["output_bundle_connection_repair"]
        assert repair["coefficient_rank"] == repair["augmented_rank"] == 88
        assert repair["affine_solution_dimension"] == 11
        assert repair["sparse_nonzero_coefficient_count"] == 6
        assert repair["sparse_support_minimal"] is True
        assert repair["corrected_curl_nonzero_components"] == 0
        assert {item["output_row"] for item in repair["sparse_nonzero_coefficients"]} == set(
            range(4, 10)
        )


def test_complete_corrected_891_entry_submanifest_is_symmetric_and_candidate_bound(
    gate: dict[str, object],
) -> None:
    assert gate["gate_counts"]["corrected_scalar_hessian_D2_entries_per_candidate"] == 891
    assert gate["gate_counts"]["corrected_scalar_hessian_D2_entries_total"] == 10_692
    for row in gate["candidate_records"]:
        manifest = row["corrected_D2_submanifest"]
        assert manifest["shape"] == [11, 9, 9]
        assert manifest["entry_count"] == len(manifest["entries"]) == 891
        assert manifest["ordered_symmetry_residual_count"] == 0
        assert manifest["status"] == (
            "complete_corrected_scalar_hessian_high_field10_D2_submanifest"
        )
        assert (
            manifest["content_sha256"]
            == hashlib.sha256(
                _canonical(
                    {key: value for key, value in manifest.items() if key != "content_sha256"}
                )
            ).hexdigest()
        )


def test_scope_and_claims_keep_every_broad_promotion_closed(gate: dict[str, object]) -> None:
    assert gate["first_blocker"] == FIRST_BLOCKER
    assert gate["claim_seals"] == CLAIM_SEALS
    assert gate["data_seals"] == EXPECTED_SEALS
    assert {key for key, enabled in gate["claim_seals"].items() if enabled} == {
        "arbitrary_domain_torsion_repair_ruled_out_for_registered_one_form",
        "sparse_output_bundle_connection_repair_constructed",
        "corrected_scalar_hessian_high_field10_D2_submanifest_registered",
    }
    assert gate["gate_counts"]["complete_ordered_D2_manifests_registered"] == 0
    assert gate["gate_counts"]["full_high_atom_good_unknown_identities_proved"] == 0
    assert gate["gate_counts"]["global_H7_closures"] == 0
    assert gate["gate_counts"]["nonlinear_PDE_closures"] == 0
    assert gate["gate_counts"]["lifespans_proved"] == 0
    assert all(row["candidate_decision"] == "blocked" for row in gate["candidate_records"])
    assert not any(row["candidate_rejection_authorized"] for row in gate["candidate_records"])
    assert not any(gate["data_seals"].values())


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("drop_dense_entry", "result boundary"),
        ("admit_torsion", "result boundary"),
        ("claim_full_d2", "result boundary"),
        ("forge_predecessor", "predecessor binding"),
        ("forge_local_source", "local source binding"),
    ],
)
def test_resealed_semantic_tampering_fails_closed(
    gate: dict[str, object], mutation: str, message: str
) -> None:
    value = json.loads(json.dumps(gate))
    if mutation == "drop_dense_entry":
        value["candidate_records"][0]["corrected_D2_submanifest"]["entries"].pop()
    elif mutation == "admit_torsion":
        value["candidate_records"][0]["torsion_no_go"]["independent_pair_certificates"][0][
            "torsion_repair_possible"
        ] = True
    elif mutation == "claim_full_d2":
        value["claim_seals"]["full_ordered_D2_tensor_registered"] = True
    elif mutation == "forge_predecessor":
        value["source_bindings"]["curl_invariance"]["content_sha256"] = "0" * 64
    else:
        value["source_bindings"]["source"]["file_sha256"] = "0" * 64
    with pytest.raises(ValueError, match=message):
        _validate_result(_reseal(value), root=ROOT)


def test_config_predecessor_and_four_file_bindings_fail_closed(gate: dict[str, object]) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["policies"]["global_H7"] = "pass"
    with pytest.raises(ValueError, match="config boundary"):
        _validate_config(config)
    with pytest.raises(ValueError, match="path escapes"):
        _load_bound(
            ROOT,
            {"path": "../outside.json", "file_sha256": "0" * 64, "content_sha256": "0" * 64},
        )
    bindings = gate["source_bindings"]
    assert bindings["source"] == {
        "path": SOURCE_PATH,
        "file_sha256": hashlib.sha256((ROOT / SOURCE_PATH).read_bytes()).hexdigest(),
    }
    assert bindings["config"] == {
        "path": CONFIG_PATH,
        "file_sha256": hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
    }
    assert bindings["test"] == {
        "path": TEST_PATH,
        "file_sha256": hashlib.sha256((ROOT / TEST_PATH).read_bytes()).hexdigest(),
    }
