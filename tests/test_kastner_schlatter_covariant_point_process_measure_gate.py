from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.kastner_schlatter_covariant_point_process_measure_gate import (
    FIRST_BLOCKER,
    _validate_result,
    build_gate,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/kastner_schlatter_covariant_point_process_measure_gate.json"
ARTIFACT = ROOT / "runs/engine/kastner-schlatter-covariant-point-process-measure-gate.json"


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
        "covariant_intensity_measure_pass": 2,
        "minimal_probability_measure_contracts_registered": 1,
        "exact_covariant_nonidentifiability_witnesses": 2,
        "action_only_Poisson_derivation_pass": 0,
        "action_only_Poisson_derivation_reject": 2,
        "external_Poisson_postulate_well_posed": 2,
        "paper_or_QED_Poisson_derivation_pass": 0,
        "candidate_action_reject": 0,
        "paper_QED_ontology_observational_pass": 0,
    }
    assert artifact["first_blocker"] == FIRST_BLOCKER


def test_minimal_poisson_measure_contract_is_explicit(rebuilt: dict[str, object]) -> None:
    contract = rebuilt["minimal_covariant_probability_measure_contract"]
    assert contract["deterministic_intensity_measure"]["formula"] == ("mu_q(B)=Integral_B q*dVol_g")
    assert contract["random_measure"]["requires_probability_space"] is True
    assert contract["poisson_Laplace_functional"]["uniquely_determines_probability_law"] is True
    assert contract["equivalent_count_contract"]["joint"] == (
        "counts on pairwise disjoint sets are independent"
    )
    assert contract["physical_event_bridge"]["provided_by_paper_bound_action_artifacts"] is False


def test_covariant_nonidentifiability_witness_is_exact(rebuilt: dict[str, object]) -> None:
    witness = rebuilt["exact_nonidentifiability_witness"]
    assert witness["model_P"]["mean_count"] == "E_P[N(B)]=mu"
    assert witness["model_C"]["mean_count"] == "E_C[N(B)]=E[Z]*mu=mu"
    assert witness["model_P"]["diffeomorphism_covariant"] is True
    assert witness["model_C"]["diffeomorphism_covariant"] is True
    separation = witness["exact_separation"]
    assert separation["same_first_moment"] is True
    assert separation["different_probability_laws"] is True
    assert separation["second_factorial_moment_P"] == "mu^2"
    assert separation["second_factorial_moment_C"] == "(1+epsilon^2)*mu^2"


def test_branch_decisions_distinguish_derivation_from_action_rejection(
    rebuilt: dict[str, object],
) -> None:
    for record in rebuilt["candidate_records"]:
        ledger = record["gate_ledger"]
        assert record["covariant_intensity_measure_construction"] == "pass"
        assert ledger["Poisson_probability_measure_from_action_alone"] == "reject"
        assert ledger["Poisson_probability_measure_as_explicit_external_postulate"] == "pass"
        assert ledger["Poisson_probability_measure_as_paper_or_QED_derivation"] == "blocked"
        assert record["external_poisson_postulate_is_action_derived"] is False
        assert record["candidate_action_rejection_authorized"] is False
        assert record["paper_or_QED_derived"] is False


def test_exact_positive_and_negative_controls(rebuilt: dict[str, object]) -> None:
    controls = rebuilt["deterministic_controls"]
    moments = controls["exact_moment_separation_control"]
    assert moments["Poisson_mean"] == moments["Cox_mean"] == "2"
    assert moments["Poisson_variance"] == "2"
    assert moments["Cox_variance"] == "3"
    assert moments["Poisson_second_factorial_moment"] == "4"
    assert moments["Cox_second_factorial_moment"] == "5"
    assert controls["zero_mixing_positive_control"]["Cox_reduces_to_Poisson"] is True
    assert controls["mean_only_negative_control"]["rejected"] is True
    assert controls["covariance_only_negative_control"]["rejected"] is True
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

    broken = copy.deepcopy(rebuilt)
    broken["exact_nonidentifiability_witness"]["exact_separation"]["different_probability_laws"] = (
        False
    )
    broken.pop("content_sha256")
    with pytest.raises(ValueError, match="no-go witness lost"):
        _validate_result(broken)


def test_source_bindings_are_exact_and_portable(rebuilt: dict[str, object]) -> None:
    bindings = rebuilt["source_bindings"]
    for label in ("config", "source", "test"):
        binding = bindings[label]
        assert _file_sha(ROOT / binding["path"]) == binding["file_sha256"]
        assert not Path(binding["path"]).is_absolute()
    assert all("C:/" not in str(value) and "C:\\" not in str(value) for value in bindings.values())
