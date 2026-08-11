from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.future_aether_canonical_seed_constraint_dag_gate import (
    BLOCKER,
    build_future_aether_canonical_seed_constraint_dag_gate,
    exact_static_canonical_seed_control,
    validate_future_aether_canonical_seed_constraint_dag_gate,
)
from sigma_theory_compiler.future_aether_finite_tilt_york_symbol_gate import (
    YORK_SHELL_BLOCKER,
)
from sigma_theory_compiler.future_aether_nonlinear_lift_characteristic_gate import (
    CHARACTERISTIC_BLOCKER,
)
from sigma_theory_compiler.promotion_orchestrator import ELIGIBILITY

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/future_aether_canonical_seed_constraint_dag_gate.json"
ARTIFACT = ROOT / "runs/engine/future-aether-canonical-seed-constraint-dag-gate.json"


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


@pytest.fixture(scope="module")
def rebuilt() -> dict[str, object]:
    return build_future_aether_canonical_seed_constraint_dag_gate(_load(CONFIG), ROOT)


def test_exact_rebuild_and_candidate_partition(rebuilt: dict[str, object]) -> None:
    artifact = _load(ARTIFACT)
    assert rebuilt == artifact
    validate_future_aether_canonical_seed_constraint_dag_gate(artifact)
    assert artifact["candidate_count"] == 14
    assert artifact["decision_counts"] == {"blocked": 14}
    assert artifact["first_blocker_counts"] == {
        YORK_SHELL_BLOCKER: 2,
        BLOCKER: 1,
        CHARACTERISTIC_BLOCKER: 11,
    }
    assert artifact["full_canonical_background_point_registered_count"] == 1
    assert artifact["candidate_bound_flat_chart_D_residual_DAG_registered_count"] == 1
    assert artifact["spatially_distributed_canonical_H_core_registered_count"] == 0
    assert artifact["metric_covariantized_H_D_Frechet_DAG_registered_count"] == 0
    assert artifact["distributed_lower_order_coefficient_registry_complete_count"] == 0


def test_exact_candidate_bound_canonical_momenta(rebuilt: dict[str, object]) -> None:
    target = next(
        record
        for record in rebuilt["candidate_records"]
        if record["first_blocker"] == BLOCKER
    )
    assert target["candidate_id"] == "G3A-5e9f93eda83935f288c19571"
    assert target["parameters"] == {
        "c1": "1/32",
        "c2": "0",
        "c3": "0",
        "c4": "1/32",
    }
    certificate = target["canonical_seed_constraint_DAG_certificate"]
    seed = certificate["canonical_seed_point"]
    assert seed["canonical_aether_momentum"] == [
        "F*F_1/(32*sqrt(F**2 + 1))",
        "0",
        "0",
    ]
    assert seed["canonical_metric_momentum"] == [
        [
            "F_1*(F**2 - 1)/(64*sqrt(F**2 + 1))",
            "-F_2/(128*sqrt(F**2 + 1))",
            "-F_3/(128*sqrt(F**2 + 1))",
        ],
        ["-F_2/(128*sqrt(F**2 + 1))", "0", "0"],
        ["-F_3/(128*sqrt(F**2 + 1))", "0", "0"],
    ]
    assert seed["outside_compact_support"] == "F=partial_i F=0 implies pi^ij=p_A^i=0"
    assert len(seed["candidate_bound_flat_chart_D_residual"]) == 3
    assert all(seed["candidate_bound_flat_chart_D_residual"])
    assert certificate["full_canonical_background_point_registered"] is True
    assert certificate["candidate_bound_flat_chart_D_residual_DAG_registered"] is True


def test_missing_distributed_functional_is_fail_closed(
    rebuilt: dict[str, object],
) -> None:
    target = next(
        record
        for record in rebuilt["candidate_records"]
        if record["first_blocker"] == BLOCKER
    )
    certificate = target["canonical_seed_constraint_DAG_certificate"]
    dag = certificate["distributed_constraint_dependency_DAG"]
    assert dag["complete"] is False
    assert set(dag["missing_nodes"]) == {
        "spatially_distributed_canonical_H_core",
        "off_flat_metric_covariantization",
        "H_D_Frechet_edges",
        "B_and_C_tensor_registry",
    }
    assert certificate["spatially_distributed_canonical_H_core_registered"] is False
    assert certificate["metric_covariantized_H_D_Frechet_DAG_registered"] is False
    assert certificate["weighted_Fredholm_isomorphism_proven"] is False
    assert certificate["completed_boundary_sign_persistence_proven"] is False
    assert certificate["candidate_rejection_authorized"] is False


def test_unit_branch_negative_control_is_exact() -> None:
    control = exact_static_canonical_seed_control()
    negative = control["unit_branch_negative_control"]
    assert negative["rejected_mutated_derivation"] is True
    assert negative["corrupted_p_A"] == [
        "F*F_1*sqrt(F**2 + 1)/32",
        "0",
        "0",
    ]
    assert negative["exact_residual"] == [
        "F**3*F_1/(32*sqrt(F**2 + 1))",
        "0",
        "0",
    ]


def test_all_observation_and_downstream_seals_remain_closed(
    rebuilt: dict[str, object],
) -> None:
    assert rebuilt["formal_pass_count"] == 0
    assert rebuilt["candidate_rejection_authorized_count"] == 0
    assert rebuilt["constraint_satisfying_negative_total_energy_datum_count"] == 0
    assert rebuilt["full_candidate_specific_formal_completion_claimed"] is False
    assert rebuilt["automatic_downstream_enqueue_performed"] is False
    assert rebuilt["solar_bundle_count"] == 0
    assert rebuilt["data_eligibility"] == ELIGIBILITY
    for record in rebuilt["candidate_records"]:
        assert record["decision"] == "blocked"
        assert record["formal_pass"] is False
        assert record["candidate_rejection_authorized"] is False
        assert record["constraint_satisfying_negative_total_energy_datum_proven"] is False
        assert record["automatic_downstream_enqueue_performed"] is False
        assert record["solar_bundle_generated"] is False
        assert record["data_eligibility"] == ELIGIBILITY


def test_artifact_and_record_tampering_fail_closed() -> None:
    artifact = _load(ARTIFACT)
    tampered = copy.deepcopy(artifact)
    tampered["full_canonical_background_point_registered_count"] = 2
    with pytest.raises(ValueError, match="artifact is inconsistent"):
        validate_future_aether_canonical_seed_constraint_dag_gate(tampered)

    tampered = copy.deepcopy(artifact)
    tampered["candidate_records"][0]["candidate_rejection_authorized"] = True
    body = {key: item for key, item in tampered.items() if key != "content_sha256"}
    tampered["content_sha256"] = _sha(body)
    with pytest.raises(ValueError, match="record is inconsistent"):
        validate_future_aether_canonical_seed_constraint_dag_gate(tampered)


def test_source_and_ansatz_tampering_fail_before_execution() -> None:
    config = _load(CONFIG)
    tampered = copy.deepcopy(config)
    tampered["source_lower_order_artifact"]["file_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="lower-order artifact file hash mismatch"):
        build_future_aether_canonical_seed_constraint_dag_gate(tampered, ROOT)

    tampered = copy.deepcopy(config)
    tampered["canonical_seed_ansatz"]["K_ij"] = "arbitrary"
    with pytest.raises(ValueError, match="canonical seed ansatz is not exact"):
        build_future_aether_canonical_seed_constraint_dag_gate(tampered, ROOT)
