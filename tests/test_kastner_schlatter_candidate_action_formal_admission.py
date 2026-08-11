from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.kastner_schlatter_candidate_action_formal_admission import (
    FIRST_BLOCKER,
    _validate_result,
    build_admission,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/kastner_schlatter_candidate_action_formal_admission.json"
ARTIFACT = ROOT / "runs/engine/kastner-schlatter-candidate-action-formal-admission.json"


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def rebuilt() -> dict[str, object]:
    return build_admission(CONFIG)


def test_exact_rebuild_counts_and_fail_closed_decisions(rebuilt: dict[str, object]) -> None:
    artifact = _load(ARTIFACT)
    assert rebuilt == artifact
    _validate_result(artifact)
    assert artifact["decision_counts"] == {"pass": 0, "reject": 0, "blocked": 2}
    assert artifact["formal_counts"] == {
        "candidate_actions": 2,
        "covariant_variation_pass": 2,
        "regular_ADM_Dirac_pass": 2,
        "three_local_DOF_pass": 2,
        "gauge_fixed_local_hyperbolicity_pass": 2,
        "ghost_gradient_tachyon_pass": 2,
        "scalar_Hamiltonian_positive_pass": 2,
        "global_positive_energy_pass": 0,
        "paper_or_QED_derived_actions": 0,
        "formal_admission_pass": 0,
    }
    assert artifact["first_blocker"] == FIRST_BLOCKER


def test_covariant_variation_replay_and_noether_identity(rebuilt: dict[str, object]) -> None:
    for record in rebuilt["candidate_records"]:
        replay = record["covariant_variation_replay"]
        assert replay["status"] == "pass_exact_local"
        assert replay["exact_residuals"] == {"metric": "0", "intensity": "0", "noether": "0"}
        assert "GHY" in replay["boundary_closure"]
        assert record["paper_or_QED_derived"] is False


def test_adm_dirac_constraint_and_dof_count(rebuilt: dict[str, object]) -> None:
    for record in rebuilt["candidate_records"]:
        adm = record["adm_dirac"]
        assert adm["velocity_hessian"] == {
            "rank": 7,
            "nullity": 4,
            "regularity": "N>0, positive h_ij, B_q>0",
            "null_directions": ["dot(N)", "dot(N^1)", "dot(N^2)", "dot(N^3)"],
        }
        assert adm["constraints"]["total_first_class"] == 8
        assert adm["constraints"]["second_class"] == 0
        assert adm["dof_count"]["physical_configuration_dof"] == 3
        assert adm["dof_count"]["tensor_dof"] == 2
        assert adm["dof_count"]["scalar_intensity_dof"] == 1


def test_principal_symbol_and_local_stability(rebuilt: dict[str, object]) -> None:
    for record in rebuilt["candidate_records"]:
        principal = record["principal_symbol"]
        stability = record["stability"]
        assert principal["status"] == "pass_local_after_declared_gauge_reduction"
        assert principal["characteristic_cone"] == "metric null cone"
        assert principal["ungauged_direct_symbol_invertibility"] is False
        assert principal["global_evolution_or_geodesic_completeness_proven"] is False
        assert stability["effective_mass_squared"] == "A_q/B_q>0"
        assert stability["registered_domain_satisfies_all_three"] is True
        assert stability["vacuum_energy_changes_principal_symbol"] is False


def test_positive_hamiltonian_scope_does_not_overreach(rebuilt: dict[str, object]) -> None:
    for record in rebuilt["candidate_records"]:
        hamiltonian = record["hamiltonian"]
        assert hamiltonian["scalar_density_nonnegative"] is True
        assert hamiltonian["reduced_quadratic_physical_hamiltonian_positive"] is True
        assert hamiltonian["global_boundary_charge_registered"] is False
        assert hamiltonian["global_nonlinear_positive_energy_proven"] is False
        assert record["background"]["Minkowski_stationary_background"] is False
        assert record["candidate_rejection_authorized"] is False


def test_exact_negative_controls(rebuilt: dict[str, object]) -> None:
    controls = rebuilt["negative_controls"]
    assert controls["B_q_negative"]["classification"] == "ghost_and_gradient_instability"
    assert controls["A_q_negative"]["classification"] == "tachyon_and_Hamiltonian_instability"
    assert "degenerate_Dirac_stratum" in controls["B_q_zero"]["classification"]
    assert controls["restricted_to_global_inference"]["classification"] == "scope_overreach"
    assert all(
        value["admitted"] is False for value in controls.values() if isinstance(value, dict)
    )
    assert controls["all_negative_controls_rejected"] is True


def test_lineage_config_and_result_tampering_fail_closed(tmp_path: Path, rebuilt: dict[str, object]) -> None:
    config = _load(CONFIG)
    tampered = copy.deepcopy(config)
    tampered["predecessors"]["completion_artifact"]["content_sha256"] = "0" * 64
    config_path = tmp_path / "configs" / CONFIG.name
    config_path.parent.mkdir(parents=True)
    config_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="file hash mismatch|content hash mismatch"):
        build_admission(config_path)

    opened = copy.deepcopy(config)
    opened["seals"]["observations_opened"] = True
    config_path.write_text(json.dumps(opened), encoding="utf-8")
    with pytest.raises(ValueError, match="seals changed"):
        build_admission(config_path)

    attributed = copy.deepcopy(rebuilt)
    attributed["candidate_records"][0]["paper_or_QED_derived"] = True
    attributed.pop("content_sha256")
    with pytest.raises(ValueError, match="falsely attributed"):
        _validate_result(attributed)

    rejected = copy.deepcopy(rebuilt)
    rejected["candidate_records"][0]["candidate_rejection_authorized"] = True
    rejected.pop("content_sha256")
    with pytest.raises(ValueError, match="overreached"):
        _validate_result(rejected)


def test_source_bindings_are_exact_and_portable(rebuilt: dict[str, object]) -> None:
    bindings = rebuilt["source_bindings"]
    for label in ("config", "source", "test"):
        binding = bindings[label]
        assert _file_sha(ROOT / binding["path"]) == binding["file_sha256"]
        assert not Path(binding["path"]).is_absolute()
    assert all("C:/" not in str(value) and "C:\\" not in str(value) for value in bindings.values())
