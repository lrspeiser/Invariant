from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.kastner_schlatter_positive_intensity_preservation_gate import (
    FIRST_BLOCKER,
    _validate_result,
    build_gate,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/kastner_schlatter_positive_intensity_preservation_gate.json"
ARTIFACT = ROOT / "runs/engine/kastner-schlatter-positive-intensity-preservation-gate.json"


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
        "exact_crossing_witnesses": 2,
        "fully_coupled_constraint_satisfying_witnesses": 2,
        "unrestricted_positive_intensity_preservation_pass": 0,
        "unrestricted_positive_intensity_preservation_reject": 2,
        "stationary_conditional_Poisson_interface_pass": 2,
        "restricted_invariant_nonnegative_cone_pass": 0,
        "positive_reparameterized_action_pass": 0,
        "action_derived_point_process_measure_pass": 0,
        "candidate_action_reject": 0,
        "paper_QED_ontology_observational_pass": 0,
    }
    assert artifact["first_blocker"] == FIRST_BLOCKER


def test_branch_hubble_bindings_are_exact(rebuilt: dict[str, object]) -> None:
    middle, printed = rebuilt["candidate_records"]
    assert middle["candidate_background_binding"]["stationary_Hubble_relation"] == (
        "H_vac^2=4*pi*G*h_planck*q0/(3*c^3)"
    )
    assert printed["candidate_background_binding"]["stationary_Hubble_relation"] == (
        "H_vac^2=2*pi*G*h_planck*q0/(3*c^3)"
    )


def test_fully_coupled_crossing_witness_satisfies_constraints(rebuilt: dict[str, object]) -> None:
    for record in rebuilt["candidate_records"]:
        witness = record["exact_crossing_witness"]
        assert witness["coupled_equations"]["scalar"] == "q''+3*H*q'+m^2*(q-q0)=0"
        assert witness["initial_data"]["Hamiltonian_constraint_residual"] == "0"
        assert witness["initial_data"]["momentum_constraint_residual"] == "0"
        assert witness["crossing_exists"] is True
        assert witness["crossing_time_bound"] == "0<Tau_cross<=q0/v"
        assert witness["full_metric_backreaction_included"] is True
        assert witness["Einstein_constraints_satisfied"] is True
        assert witness["finite_energy_on_compact_slice"] is True


def test_crossing_rejects_unrestricted_intensity_not_action(rebuilt: dict[str, object]) -> None:
    for record in rebuilt["candidate_records"]:
        ledger = record["gate_ledger"]
        assert ledger["unrestricted_nonnegative_intensity_phase_space_invariant"] == "reject"
        assert ledger["stationary_q_equals_q0_conditional_Poisson_interface"] == "pass"
        assert record["unrestricted_positive_intensity_preservation"] is False
        assert record["stationary_conditional_interface_preserved"] is True
        assert record["candidate_action_rejection_authorized"] is False
        assert record["paper_or_QED_derived"] is False


def test_deterministic_positive_and_negative_controls(rebuilt: dict[str, object]) -> None:
    controls = rebuilt["deterministic_controls"]
    assert controls["stationary_positive_control"]["solution"] == "q(Tau)=q0>0"
    assert controls["constraint_omission_negative_control"]["rejected"] is True
    assert controls["expanding_branch_comparison_negative_control"]["rejected"] is True
    assert controls["action_rejection_negative_control"]["rejected"] is True


def test_attribution_claims_and_data_remain_sealed(rebuilt: dict[str, object]) -> None:
    assert all(value is False for value in rebuilt["claim_seals"].values())
    assert all(value is False for value in rebuilt["data_seals"].values())


def test_lineage_config_and_result_tampering_fail_closed(
    tmp_path: Path, rebuilt: dict[str, object]
) -> None:
    config = _load(CONFIG)
    tampered = copy.deepcopy(config)
    tampered["predecessors"]["poisson_action_compatibility"]["content_sha256"] = "0" * 64
    path = tmp_path / "configs" / CONFIG.name
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="file hash mismatch|content hash mismatch"):
        build_gate(path)

    opened = copy.deepcopy(config)
    opened["seals"]["observations_opened"] = True
    path.write_text(json.dumps(opened), encoding="utf-8")
    with pytest.raises(ValueError, match="seals changed"):
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
