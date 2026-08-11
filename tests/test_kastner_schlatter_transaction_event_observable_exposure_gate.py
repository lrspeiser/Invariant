from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.kastner_schlatter_transaction_event_observable_exposure_gate import (
    FIRST_BLOCKER,
    SECOND_BLOCKER,
    _validate_result,
    build_gate,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/kastner_schlatter_transaction_event_observable_exposure_gate.json"
ARTIFACT = ROOT / "runs/engine/kastner-schlatter-transaction-event-observable-exposure-gate.json"


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
        "operational_obligations": 11,
        "operational_obligations_registered": 0,
        "operational_obligations_missing": 10,
        "operational_obligations_sealed": 1,
        "compiler_observation_operator_contracts": 1,
        "exact_nonidentifiability_witnesses": 4,
        "real_observation_bundles": 0,
        "latent_rate_identification_pass": 0,
        "candidate_action_reject": 0,
        "theory_ontology_observational_pass": 0,
    }
    assert artifact["first_blocker"] == FIRST_BLOCKER
    assert artifact["secondary_blockers"][0] == SECOND_BLOCKER
    assert artifact["synthetic_only"] is True


def test_minimal_observation_operator_is_exact_and_conditional(rebuilt: dict[str, object]) -> None:
    contract = rebuilt["minimal_operational_contract"]
    operator = contract["observation_operator"]
    assert "product_{x in N cap W}" in operator["conditional_Laplace_functional"]
    assert operator["observed_mean_measure"] == ("nu(dy)=b(dy)+Integral_W a(x)*R(dy|x)*mu(dx)")
    assert operator["Poisson_mapping_theorem"].startswith("if N~PRM(mu)")
    assert operator["converse_from_mean_only"] is False
    assert contract["latent_event_contract"]["status"] == "missing"
    assert contract["analysis_manifest"]["status"] == "missing"


def test_identifiability_theorem_states_exact_required_premises(rebuilt: dict[str, object]) -> None:
    theorem = rebuilt["identifiability_theorem"]
    assert theorem["forward_operator"] == "T_{a,R,b}(mu)=b+R_*(a*mu)"
    assert theorem["identified_from_first_moment"] == "only nu=T_{a,R,b}(mu)"
    assert len(theorem["necessary_conditions_for_latent_mean_identification"]) == 4
    assert theorem["current_contract_satisfies_conditions"] is False
    assert theorem["theory_or_ontology_consequence"] is False


def test_exact_forward_and_nonidentifiability_controls(rebuilt: dict[str, object]) -> None:
    controls = rebuilt["exact_controls"]
    forward = controls["forward_two_cell_positive_control"]
    assert forward["observed_nu"] == ["5/4", "3/2"]
    assert forward["pass"] is True
    scaling = controls["rate_efficiency_scaling_no_go"]
    assert scaling["model_A"]["latent_mu"] != scaling["model_B"]["latent_mu"]
    assert scaling["observed_signal_mean_both"] == "1"
    assert scaling["identifiable_without_acceptance_calibration"] is False
    background = controls["signal_background_no_go"]
    assert background["observed_mean_both"] == "1"
    assert background["identifiable_without_background_calibration"] is False
    response = controls["rank_deficient_response_no_go"]
    assert response["response_rank"] == 1
    assert response["observed_mean_both"] == ["1", "1"]
    assert response["latent_spatial_measure_identifiable"] is False
    laws = controls["same_mean_different_law_after_thinning"]
    assert laws["Poisson_observed_second_factorial_moment"] == "1"
    assert laws["Cox_Z_half_threehalves_observed_second_factorial_moment"] == "5/4"
    assert laws["mean_alone_selects_Poisson"] is False


def test_negative_controls_and_candidate_seals(rebuilt: dict[str, object]) -> None:
    controls = rebuilt["exact_controls"]
    assert controls["detector_click_equivalence_negative_control"]["rejected"] is True
    assert controls["zero_acceptance_negative_control"]["rejected"] is True
    assert [record["branch_id"] for record in rebuilt["candidate_records"]] == [
        "eq35_middle_h",
        "eq35_printed_planck",
    ]
    for record in rebuilt["candidate_records"]:
        assert record["operational_event_exposure_contract_registered_from_data"] is False
        assert record["latent_rate_identifiable"] is False
        assert record["observations_opened"] is False
        assert record["candidate_action_rejection_authorized"] is False
        assert record["candidate_decision"] == "blocked"


def test_lineage_config_and_result_tampering_fail_closed(
    tmp_path: Path, rebuilt: dict[str, object]
) -> None:
    config = _load(CONFIG)
    tampered = copy.deepcopy(config)
    tampered["predecessors"]["actualization_history_map"]["content_sha256"] = "0" * 64
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

    registered = copy.deepcopy(rebuilt)
    registered["candidate_records"][0][
        "operational_event_exposure_contract_registered_from_data"
    ] = True
    registered.pop("content_sha256")
    with pytest.raises(ValueError, match="data registration overclaim"):
        _validate_result(registered)

    identified = copy.deepcopy(rebuilt)
    identified["candidate_records"][0]["latent_rate_identifiable"] = True
    identified.pop("content_sha256")
    with pytest.raises(ValueError, match="latent-rate overclaim"):
        _validate_result(identified)

    rejected = copy.deepcopy(rebuilt)
    rejected["candidate_records"][0]["candidate_action_rejection_authorized"] = True
    rejected.pop("content_sha256")
    with pytest.raises(ValueError, match="overreached"):
        _validate_result(rejected)


def test_source_bindings_are_exact_portable_and_sealed(rebuilt: dict[str, object]) -> None:
    bindings = rebuilt["source_bindings"]
    for label in (
        "actualization_history_map",
        "observational_readiness",
        "config",
        "source",
        "test",
    ):
        binding = bindings[label]
        assert _file_sha(ROOT / binding["path"]) == binding["file_sha256"]
        assert not Path(binding["path"]).is_absolute()
    assert all("C:/" not in str(value) and "C:\\" not in str(value) for value in bindings.values())
    assert all(value is False for value in rebuilt["claim_seals"].values())
    assert all(value is False for value in rebuilt["data_seals"].values())
