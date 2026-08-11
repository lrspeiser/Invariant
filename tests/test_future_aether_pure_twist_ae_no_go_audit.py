from __future__ import annotations

import copy
import hashlib
import json
from fractions import Fraction
from pathlib import Path

import pytest

from sigma_theory_compiler.future_aether_pure_twist_ae_no_go_audit import (
    build_future_aether_pure_twist_ae_no_go_audit,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/future_aether_pure_twist_ae_no_go_audit.json"
ARTIFACT_PATH = ROOT / "runs/engine/future-aether-pure-twist-ae-no-go-audit.json"
SOURCE_PATH = ROOT / "runs/engine/future-aether-constraint-boundary-embedding-audit.json"


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def rebuilt() -> dict:
    return build_future_aether_pure_twist_ae_no_go_audit(_config(), ROOT)


def test_exact_blocked_partition_and_portable_artifact(rebuilt: dict) -> None:
    assert rebuilt == json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    body = {key: value for key, value in rebuilt.items() if key != "content_sha256"}
    assert rebuilt["content_sha256"] == _sha(body)
    assert rebuilt["candidate_count"] == 14
    assert rebuilt["decision_counts"] == {"blocked": 14}
    assert rebuilt["first_blocker_counts"] == {
        "candidate_bound_AE_coupled_constraint_solution_beyond_flat_static_global_pure_twist_class_with_negative_completed_boundary_energy": 14
    }
    assert rebuilt["formal_pass_count"] == 0
    assert rebuilt["candidate_rejection_authorized_count"] == 0
    assert rebuilt["constraint_satisfying_negative_total_energy_datum_count"] == 0


def test_source_candidate_action_and_record_bindings_are_preserved(rebuilt: dict) -> None:
    source = json.loads(SOURCE_PATH.read_text(encoding="utf-8"))
    expected = {item["candidate_id"]: item for item in source["candidate_records"]}
    assert len(expected) == 14
    for item in rebuilt["candidate_records"]:
        predecessor = expected[item["candidate_id"]]
        assert item["typed_action_ir_sha256"] == predecessor["typed_action_ir_sha256"]
        assert (
            item["action_density_equivalence_sha256"]
            == predecessor["action_density_equivalence_sha256"]
        )
        assert item["compilation_receipt_sha256"] == predecessor["compilation_receipt_sha256"]
        assert item["source_embedding_record_sha256"] == predecessor["content_sha256"]
        assert item["exact_specialization"] == predecessor["exact_specialization"]


def test_euclidean_killing_system_exhausts_global_pure_twist_class(rebuilt: dict) -> None:
    control = rebuilt["symbolic_obstruction_control"]
    system = control["differentiated_Killing_system"]
    assert system == {
        "second_jet": "T_ijk=partial_i partial_j A_k=T_jik",
        "equations": "T_kij+T_kji=0",
        "unknown_count": 18,
        "coefficient_rank": 18,
        "kernel_dimension": 0,
        "conclusion": "partial_i partial_j A_k=0",
    }
    assert control["global_solution"] == "A_i=a_i+B_ij*x^j with B_ij=-B_ji"
    assert control["AE_consequence"] == "A_i=o(1) forces a_i=0 and B_ij=0"
    assert control["positive_control_non_AE_affine_rotation"]["pure_twist"] is True
    assert control["positive_control_non_AE_affine_rotation"]["AE"] is False
    assert control["positive_control_trivial_AE_solution"] == {
        "field": "A_i=0",
        "pure_twist": True,
        "AE": True,
        "contains_negative_twist_witness": False,
    }


def test_compact_cutoff_preserves_center_but_requires_transition_shear(
    rebuilt: dict,
) -> None:
    assert rebuilt["flat_static_global_pure_twist_AE_completion_obstructed_count"] == 14
    assert rebuilt["compact_cutoff_non_pure_twist_transition_required_count"] == 14
    assert rebuilt["normalized_transition_symmetric_gradient_norm_squared_counts"] == {
        "34": 2,
        "6": 8,
        "10": 4,
    }
    control = rebuilt["symbolic_obstruction_control"]["negative_control_radial_cutoff"]
    assert control["axis_transition_norm_squared"] == ("2*R**2*eta_prime**2*(R**2 + 2*y)")
    for item in rebuilt["candidate_records"]:
        certificate = item["pure_twist_AE_no_go_certificate"]
        y = Fraction(certificate["tilt_squared_y"])
        assert certificate["center_witness_preserved_by_cutoff"] is True
        assert certificate["global_flat_static_pure_twist_AE_completion_exists"] is False
        assert (
            Fraction(certificate["normalized_transition_symmetric_gradient_norm_squared"])
            == 4 * y + 2
        )
        assert certificate["compact_cutoff_requires_non_pure_twist_transition"] is True


def test_completion_class_rejection_never_rejects_candidate(rebuilt: dict) -> None:
    for item in rebuilt["candidate_records"]:
        gates = item["gate_ledger"]
        assert gates["flat_static_global_pure_twist_AE_completion"]["status"] == (
            "reject_completion_class"
        )
        assert gates["compact_radial_localization"]["status"] == (
            "blocked_at_non_pure_twist_transition"
        )
        assert gates["coupled_constraint_solution_beyond_obstructed_class"]["status"] == "blocked"
        assert gates["completed_AE_boundary_energy"]["status"] == "blocked"
        assert item["decision"] == "blocked"
        assert item["formal_pass"] is False
        assert item["candidate_rejection_authorized"] is False
        assert item["constraint_satisfying_negative_total_energy_datum_proven"] is False


def test_hash_provenance_scope_and_data_seals(rebuilt: dict) -> None:
    assert rebuilt["bounded_pure_twist_AE_no_go_audit_completed"] is True
    assert rebuilt["full_candidate_specific_formal_completion_claimed"] is False
    assert rebuilt["automatic_downstream_enqueue_performed"] is False
    assert rebuilt["solar_bundle_count"] == 0
    assert rebuilt["observational_data_opened"] is False
    assert rebuilt["dark_matter_or_halo_inputs"] is False
    assert rebuilt["redshift_distance_inputs"] is False
    assert rebuilt["paid_llm_spend_usd"] == 0.0
    assert rebuilt["data_eligibility"] == _config()["data_eligibility"]
    provenance = rebuilt["provenance"]
    provenance_body = {key: value for key, value in provenance.items() if key != "binding_sha256"}
    assert provenance["binding_sha256"] == _sha(provenance_body)
    for item in rebuilt["candidate_records"]:
        body = {key: value for key, value in item.items() if key != "content_sha256"}
        assert item["content_sha256"] == _sha(body)
        record_provenance = item["provenance"]
        record_body = {
            key: value for key, value in record_provenance.items() if key != "binding_sha256"
        }
        assert record_provenance["binding_sha256"] == _sha(record_body)
        assert item["observational_data_opened"] is False
        assert item["solar_bundle_generated"] is False


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda config: config.update(
                data_eligibility={
                    **config["data_eligibility"],
                    "observational_data_opened": True,
                }
            ),
            "eligibility is open",
        ),
        (
            lambda config: config.update(observational_authorization=True),
            "opened observations",
        ),
        (
            lambda config: config.update(external_paid_llm_calls=True),
            "enabled paid LLM calls",
        ),
        (
            lambda config: config["budget"].update(maximum_symbolic_linear_system_unknowns=17),
            "budget is not exact",
        ),
        (
            lambda config: config["source_embedding_artifact"].update(content_sha256="0" * 64),
            "content hash mismatch",
        ),
        (
            lambda config: config["campaign_implementation"].update(file_sha256="0" * 64),
            "file hash mismatch",
        ),
    ],
)
def test_open_seals_budget_and_hash_tampering_fail_closed(mutation, message: str) -> None:
    config = copy.deepcopy(_config())
    mutation(config)
    with pytest.raises(ValueError, match=message):
        build_future_aether_pure_twist_ae_no_go_audit(config, ROOT)


def test_bound_paths_cannot_escape_repository() -> None:
    config = copy.deepcopy(_config())
    config["source_embedding_artifact"]["path"] = "../outside.json"
    with pytest.raises(ValueError, match="path escapes repository"):
        build_future_aether_pure_twist_ae_no_go_audit(config, ROOT)
