from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.kastner_schlatter_poisson_selector_contract_gate import (
    FIRST_BLOCKER,
    _validate_result,
    build_gate,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/kastner_schlatter_poisson_selector_contract_gate.json"
ARTIFACT = ROOT / "runs/engine/kastner-schlatter-poisson-selector-contract-gate.json"


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
        "registered_scalar_Poisson_PMF_assertions": 1,
        "minimal_sufficient_selector_contracts": 3,
        "registered_selector_nodes": 0,
        "registered_action_derivation_edges_to_PMF": 0,
        "independent_increment_derivation_pass": 0,
        "Poisson_Laplace_functional_derivation_pass": 0,
        "QED_counting_measure_kernel_derivation_pass": 0,
        "registered_equations_imply_selector_reject": 2,
        "candidate_action_reject": 0,
        "paper_QED_ontology_observational_pass": 0,
    }
    assert artifact["first_blocker"] == FIRST_BLOCKER


def test_minimal_sufficient_selector_contract_is_exact(rebuilt: dict[str, object]) -> None:
    contract = rebuilt["minimal_Poisson_selector_contract"]
    assert contract["sufficient_selector_A_Laplace_functional"]["formula"] == (
        "L[f]=exp(-Integral_M (1-exp(-f))*q*dVol_g)"
    )
    assert contract["sufficient_selector_A_Laplace_functional"]["effect"] == (
        "uniquely determines the Poisson random measure"
    )
    assert "pairwise disjoint" in contract["sufficient_selector_B_joint_count_family"]["domain"]
    assert contract["sufficient_selector_C_Mecke_event_kernel"]["formula"] == (
        "E[Integral F(x,N)N(dx)]=Integral E[F(x,N+delta_x)]q(x)dVol_g(x)"
    )
    assert (
        len(contract["QED_bridge_required_before_selector_C_can_be_attributed"]["requirements"])
        == 5
    )


def test_registered_graph_has_assertion_but_no_selector_derivation(
    rebuilt: dict[str, object],
) -> None:
    audit = rebuilt["registered_dependency_audit"]
    assert audit["closed_world_counts"] == {"nodes": 54, "edges": 137}
    assert audit["registered_scalar_Poisson_PMF_nodes"] == 1
    assert audit["PMF_status"] == "paper_assertion_with_standard_implementation_formula"
    assert audit["PMF_printed_as_equation_in_paper"] is False
    assert audit["registered_selector_node_ids"] == []
    assert audit["PMF_action_derivation_edges"] == 0
    assert audit["PMF_not_derived_from_action_edge"] is True
    assert audit["registered_equations_imply_independent_increments"] is False
    assert audit["registered_equations_imply_Poisson_Laplace_functional"] is False
    assert audit["registered_equations_imply_QED_counting_measure_kernel"] is False


def test_scalar_marginal_does_not_imply_joint_independence(rebuilt: dict[str, object]) -> None:
    theorem = rebuilt["scalar_marginal_nonimplication_theorem"]
    assert theorem["single_cell_marginals"] == ("X and Y each obey exp(-lambda)*lambda^n/n!")
    assert theorem["dependent_joint_PGF"] == "G_dep(s,t)=exp(lambda*(s*t-1))"
    assert theorem["independent_joint_PGF"] == ("G_ind(s,t)=exp(lambda*(s-1)+lambda*(t-1))")
    assert theorem["covariance_dependent"] == "Cov(X,Y)=lambda>0"
    assert theorem["covariance_independent"] == "Cov(X,Y)=0"


def test_branch_records_preserve_attribution_and_action_boundary(
    rebuilt: dict[str, object],
) -> None:
    for record in rebuilt["candidate_records"]:
        outputs = record["registered_action_stochastic_outputs"]
        ledger = record["gate_ledger"]
        assert outputs["positive_intensity_measure"] is True
        assert outputs["probability_kernel"] is False
        assert outputs["Laplace_functional"] is False
        assert ledger["scalar_Poisson_PMF_as_registered_assertion"] == "pass"
        assert ledger["registered_equations_imply_a_Poisson_selector"] == "reject"
        assert ledger["candidate_action_rejection"] == "blocked"
        assert record["candidate_action_rejection_authorized"] is False
        assert record["paper_or_QED_derived"] is False


def test_exact_positive_and_negative_controls(rebuilt: dict[str, object]) -> None:
    controls = rebuilt["deterministic_controls"]
    assert controls["Laplace_selector_positive_control"]["Poisson_law_uniquely_selected"] is True
    assert controls["joint_family_positive_control"]["independent_increments_selected"] is True
    assert controls["single_PMF_negative_control"]["rejected"] is True
    assert controls["mean_only_negative_control"]["rejected"] is True
    assert controls["attribution_negative_control"]["rejected"] is True


def test_attribution_and_data_seals_remain_closed(rebuilt: dict[str, object]) -> None:
    assert all(value is False for value in rebuilt["claim_seals"].values())
    assert all(value is False for value in rebuilt["data_seals"].values())


def test_lineage_config_and_result_tampering_fail_closed(
    tmp_path: Path, rebuilt: dict[str, object]
) -> None:
    config = _load(CONFIG)
    tampered = copy.deepcopy(config)
    tampered["predecessors"]["equation_graph"]["content_sha256"] = "0" * 64
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

    invented = copy.deepcopy(rebuilt)
    invented["registered_dependency_audit"]["registered_selector_nodes"] = 1
    invented.pop("content_sha256")
    with pytest.raises(ValueError, match="dependency no-go changed"):
        _validate_result(invented)


def test_source_bindings_are_exact_and_portable(rebuilt: dict[str, object]) -> None:
    bindings = rebuilt["source_bindings"]
    for label in ("config", "source", "test"):
        binding = bindings[label]
        assert _file_sha(ROOT / binding["path"]) == binding["file_sha256"]
        assert not Path(binding["path"]).is_absolute()
    assert all("C:/" not in str(value) and "C:\\" not in str(value) for value in bindings.values())
