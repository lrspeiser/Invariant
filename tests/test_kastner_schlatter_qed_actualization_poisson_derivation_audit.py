from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.kastner_schlatter_qed_actualization_poisson_derivation_audit import (
    FIRST_BLOCKER,
    SECOND_BLOCKER,
    _validate_result,
    build_audit,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/kastner_schlatter_qed_actualization_poisson_derivation_audit.json"
ARTIFACT = ROOT / "runs/engine/kastner-schlatter-qed-actualization-poisson-derivation-audit.json"


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def rebuilt() -> dict[str, object]:
    return build_audit(CONFIG)


def test_exact_rebuild_partition_and_counts(rebuilt: dict[str, object]) -> None:
    artifact = _load(ARTIFACT)
    assert rebuilt == artifact
    _validate_result(artifact)
    assert artifact["decision_counts"] == {"pass": 0, "reject": 0, "blocked": 2}
    assert artifact["gate_counts"] == {
        "candidate_actions": 2,
        "source_evidence_clauses": 5,
        "microscopic_derivation_obligations": 12,
        "microscopic_obligations_closed": 0,
        "microscopic_obligations_partial": 2,
        "microscopic_obligations_absent": 10,
        "compiler_conditional_sufficient_theorems": 1,
        "exact_same_rate_non_Poisson_witnesses": 2,
        "paper_or_QED_channel_kernels_registered": 0,
        "paper_or_QED_Poisson_derivation_pass": 0,
        "candidate_action_reject": 0,
        "theory_ontology_observational_pass": 0,
    }
    assert artifact["first_blocker"] == FIRST_BLOCKER
    assert artifact["secondary_blockers"][0] == SECOND_BLOCKER
    assert artifact["synthetic_only"] is True


def test_primary_source_evidence_is_assertion_not_derivation(rebuilt: dict[str, object]) -> None:
    evidence = {item["evidence_id"]: item for item in rebuilt["primary_source_evidence"]}
    assert evidence["qed_poisson_sentence"]["status"] == (
        "assertion_without_registered_channel_kernel"
    )
    assert evidence["standard_poisson_pmf"]["status"] == (
        "implementation_not_paper_equation_and_not_derivation"
    )
    assert evidence["set_indexed_synthetic_discrimination"]["status"] == (
        "synthetic_only_no_QED_inference"
    )
    obligations = rebuilt["microscopic_derivation_obligations"]
    assert [item["status"] for item in obligations].count("absent") == 10
    assert [item["status"] for item in obligations].count("semantic_partial") == 1
    assert [item["status"] for item in obligations].count("rate_partial") == 1


def test_independent_rare_channel_theorem_is_exact_and_conditional(
    rebuilt: dict[str, object],
) -> None:
    theorem = rebuilt["independent_rare_channel_Poisson_limit"]
    assert theorem["finite_row_joint_PGF"] == ("G_m(z)=product_k[1+sum_i p_mki*(z_i-1)]")
    assert theorem["limit_joint_PGF"] == "exp(sum_i mu_i*(z_i-1))"
    assert theorem["conclusion"] == "cell counts converge jointly to independent Poisson(mu_i)"
    assert theorem["paper_or_registered_QED_closes_conditions"] is False
    assert len(theorem["conditions"]) == 4


def test_same_rate_non_poisson_and_common_shock_controls(rebuilt: dict[str, object]) -> None:
    controls = rebuilt["exact_controls"]
    independent = controls["independent_rare_channel_positive_control"]
    assert independent["limit_PGF"] == "exp(mu*(z-1))"
    assert independent["limit_mean"] == independent["limit_variance"] == "mu"
    cluster = controls["paired_cluster_same_rate_no_go"]
    assert cluster["limit_PGF"] == "exp((mu/2)*(z^2-1))"
    assert cluster["limit_mean"] == "mu"
    assert cluster["limit_variance"] == "2*mu"
    assert cluster["same_mean_rate_as_Poisson"] is True
    assert cluster["Poisson_conclusion_rejected"] is True
    shock = controls["two_cell_common_shock_no_go"]
    assert shock["marginals"] == ["Poisson(1)", "Poisson(1)"]
    assert shock["cross_covariance"] == "1/2"
    assert shock["independent_increment_conclusion"] is False


def test_candidate_and_attribution_boundaries_remain_closed(rebuilt: dict[str, object]) -> None:
    controls = rebuilt["exact_controls"]
    assert controls["external_reference_negative_control"]["rejected"] is True
    assert controls["attribution_negative_control"]["rejected"] is True
    assert [record["branch_id"] for record in rebuilt["candidate_records"]] == [
        "eq35_middle_h",
        "eq35_printed_planck",
    ]
    for record in rebuilt["candidate_records"]:
        assert record["compiler_conditional_rare_channel_theorem"] is True
        assert record["paper_or_QED_channel_kernel_registered"] is False
        assert record["paper_or_QED_Poisson_derivation_pass"] is False
        assert record["candidate_action_rejection_authorized"] is False
        assert record["candidate_decision"] == "blocked"


def test_lineage_config_and_result_tampering_fail_closed(
    tmp_path: Path, rebuilt: dict[str, object]
) -> None:
    config = _load(CONFIG)
    tampered = copy.deepcopy(config)
    tampered["predecessors"]["paper_intake"]["content_sha256"] = "0" * 64
    path = tmp_path / "configs" / CONFIG.name
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="file hash mismatch|content hash mismatch"):
        build_audit(path)

    opened = copy.deepcopy(config)
    opened["seals"]["QED_actualization_derivation_opened"] = True
    path.write_text(json.dumps(opened), encoding="utf-8")
    with pytest.raises(ValueError, match="seal opened"):
        build_audit(path)

    registered = copy.deepcopy(rebuilt)
    registered["candidate_records"][0]["paper_or_QED_channel_kernel_registered"] = True
    registered.pop("content_sha256")
    with pytest.raises(ValueError, match="channel-kernel overclaim"):
        _validate_result(registered)

    passed = copy.deepcopy(rebuilt)
    passed["candidate_records"][0]["paper_or_QED_Poisson_derivation_pass"] = True
    passed.pop("content_sha256")
    with pytest.raises(ValueError, match="derivation overclaim"):
        _validate_result(passed)

    rejected = copy.deepcopy(rebuilt)
    rejected["candidate_records"][0]["candidate_action_rejection_authorized"] = True
    rejected.pop("content_sha256")
    with pytest.raises(ValueError, match="overreached"):
        _validate_result(rejected)


def test_source_bindings_are_exact_portable_and_sealed(rebuilt: dict[str, object]) -> None:
    bindings = rebuilt["source_bindings"]
    for label in (
        "operational_event_exposure",
        "conditional_poisson_kernel",
        "set_indexed_synthetic_campaign",
        "paper_intake",
        "equation_graph",
        "config",
        "source",
        "test",
    ):
        binding = bindings[label]
        assert _file_sha(ROOT / binding["path"]) == binding["file_sha256"]
        assert not Path(binding["path"]).is_absolute()
    assert bindings["primary_pdf_sha256"] == rebuilt["audit_domain"]["paper_pdf_sha256"]
    assert all("C:/" not in str(value) and "C:\\" not in str(value) for value in bindings.values())
    assert all(value is False for value in rebuilt["claim_seals"].values())
    assert all(value is False for value in rebuilt["data_seals"].values())
