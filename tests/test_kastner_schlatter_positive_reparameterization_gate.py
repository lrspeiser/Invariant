from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.kastner_schlatter_positive_reparameterization_gate import (
    FIRST_BLOCKER,
    _validate_result,
    build_gate,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/kastner_schlatter_positive_reparameterization_gate.json"
ARTIFACT = ROOT / "runs/engine/kastner-schlatter-positive-reparameterization-gate.json"


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def rebuilt() -> dict[str, object]:
    return build_gate(CONFIG)


def test_exact_rebuild_partition_and_counts(rebuilt: dict[str, object]) -> None:
    artifact = _load(ARTIFACT)
    assert rebuilt == artifact
    _validate_result(artifact)
    assert artifact["decision_counts"] == {"pass": 0, "reject": 0, "blocked": 2}
    assert artifact["gate_counts"] == {
        "candidate_actions": 2,
        "exact_positive_field_diffeomorphism_pass": 2,
        "exact_reparameterized_action_pass": 2,
        "EL_equivalence_on_positive_sector_pass": 2,
        "regular_solution_strict_positivity_pass": 2,
        "original_unrestricted_phase_space_positivity_reject": 2,
        "paper_or_QED_positive_sector_selection_pass": 0,
        "action_derived_point_process_measure_pass": 0,
        "candidate_action_reject": 0,
        "paper_QED_ontology_observational_pass": 0,
    }
    assert artifact["first_blocker"] == FIRST_BLOCKER


def test_exact_exponential_rewrite_and_equations(rebuilt: dict[str, object]) -> None:
    for record in rebuilt["candidate_records"]:
        diffeomorphism = record["field_diffeomorphism"]
        action = record["reparameterized_action"]
        assert diffeomorphism["image"] == "q in (0,infinity)"
        assert diffeomorphism["Jacobian"] == "dq/dphi=q0*exp(phi)=q>0"
        assert diffeomorphism["global_on_declared_open_sector"] is True
        assert diffeomorphism["covers_q_equals_zero"] is False
        assert "exp(2*phi)" in action["bulk"]
        assert action["exact_EL_equivalence_on_q_positive"] is True
        assert action["original_equation_times_Jacobian"] == ("q*[B_q*Box(q)-A_q*(q-q0)]=0")
        assert action["metric_equation"] == "G_mn=(8*pi*G/c^4)*T_mn^+"
        assert action["off_shell_identity"] == "nabla^m(T_mn^+)=E_phi*grad_n(phi)"
        assert action["dimensions"]["phi"] == "1"


def test_positivity_is_restricted_and_not_action_rejection(rebuilt: dict[str, object]) -> None:
    for record in rebuilt["candidate_records"]:
        theorem = record["positivity_theorem"]
        ledger = record["gate_ledger"]
        assert theorem["scope"] == "restricted open field sector on the maximal regular solution"
        assert theorem["global_existence_or_geodesic_completeness_proved"] is False
        assert theorem["original_q_in_R_nonnegative_cone_invariant"] is False
        assert ledger["positive_intensity_on_regular_finite_phi_solutions"] == "pass"
        assert ledger["unrestricted_original_q_phase_space_positivity"] == "reject"
        assert ledger["paper_or_QED_selection_of_positive_sector"] == "blocked"
        assert record["candidate_action_rejection_authorized"] is False
        assert record["paper_or_QED_derived"] is False


def test_exact_positive_and_negative_controls(rebuilt: dict[str, object]) -> None:
    controls = rebuilt["deterministic_controls"]
    assert controls["stationary_positive_control"]["Euler_Lagrange_residual"] == "0"
    assert controls["linear_map_negative_control"]["rejected"] is True
    assert controls["missing_Jacobian_negative_control"]["rejected"] is True
    assert controls["crossing_import_negative_control"]["rejected"] is True
    assert controls["action_overclaim_negative_control"]["rejected"] is True


def test_attribution_and_data_seals_remain_closed(rebuilt: dict[str, object]) -> None:
    assert all(value is False for value in rebuilt["claim_seals"].values())
    assert all(value is False for value in rebuilt["data_seals"].values())


def test_lineage_config_and_result_tampering_fail_closed(
    tmp_path: Path, rebuilt: dict[str, object]
) -> None:
    config = _load(CONFIG)
    tampered = copy.deepcopy(config)
    tampered["predecessor"]["content_sha256"] = "0" * 64
    path = tmp_path / "configs" / CONFIG.name
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="file hash mismatch|content hash mismatch"):
        build_gate(path)

    opened = copy.deepcopy(config)
    opened["seals"]["observations_opened"] = True
    path.write_text(json.dumps(opened), encoding="utf-8")
    with pytest.raises(ValueError, match="seal opened"):
        build_gate(path)

    overclaim = copy.deepcopy(rebuilt)
    overclaim["candidate_records"][0]["candidate_action_rejection_authorized"] = True
    overclaim.pop("content_sha256")
    with pytest.raises(ValueError, match="overreached"):
        _validate_result(overclaim)

    attributed = copy.deepcopy(rebuilt)
    attributed["candidate_records"][0]["paper_or_QED_derived"] = True
    attributed.pop("content_sha256")
    with pytest.raises(ValueError, match="attribution changed"):
        _validate_result(attributed)


def test_source_bindings_are_exact_and_portable(rebuilt: dict[str, object]) -> None:
    bindings = rebuilt["source_bindings"]
    for label in ("config", "source", "test"):
        binding = bindings[label]
        assert _file_sha(ROOT / binding["path"]) == binding["file_sha256"]
        assert not Path(binding["path"]).is_absolute()
    assert all("C:/" not in str(value) and "C:\\" not in str(value) for value in bindings.values())
