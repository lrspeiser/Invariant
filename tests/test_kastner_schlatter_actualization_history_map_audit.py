from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.kastner_schlatter_actualization_history_map_audit import (
    FIRST_BLOCKER,
    SECOND_BLOCKER,
    _validate_result,
    build_audit,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/kastner_schlatter_actualization_history_map_audit.json"
ARTIFACT = ROOT / "runs/engine/kastner-schlatter-actualization-history-map-audit.json"


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
        "paper_source_clauses_audited": 5,
        "typed_map_obligations": 12,
        "closed_by_paper_semantics": 4,
        "partially_specified": 2,
        "blocked_or_absent": 6,
        "paper_complete_history_to_counting_measure_maps": 0,
        "compiler_conditional_count_maps": 1,
        "paper_or_QED_Poisson_kernel_selections": 0,
        "candidate_action_reject": 0,
        "theory_ontology_observational_pass": 0,
    }
    assert artifact["first_blocker"] == FIRST_BLOCKER
    assert artifact["secondary_blockers"][0] == SECOND_BLOCKER


def test_paper_evidence_closes_semantics_not_map(rebuilt: dict[str, object]) -> None:
    evidence = rebuilt["paper_evidence_ledger"]
    assert len(evidence) == 5
    assert [item["status"] for item in evidence].count("paper_prose_semantics_only") == 3
    obligations = rebuilt["typed_map_obligations"]
    statuses = [item["status"] for item in obligations]
    assert statuses.count("closed_by_paper_semantics") == 4
    assert statuses.count("absent") == 4
    assert "partial_N_R_for_shrinking_regions_only" in statuses
    assert "absent_from_paper_equation_graph" in statuses


def test_compiler_map_theorem_is_conditional_and_not_a_law(rebuilt: dict[str, object]) -> None:
    contract = rebuilt["compiler_conditional_count_map"]
    assert contract["formula"] == "N_h(B)=sum_{tau in T_h} 1_B(a_h(tau))"
    assert contract["theorem"]["integer_valued"] is True
    assert contract["theorem"]["countably_additive"] is True
    assert contract["theorem"]["locally_finite_by_declared_domain"] is True
    assert contract["paper_supplies_H_lf_or_its_measurable_structure"] is False
    assert contract["selects_a_probability_law"] is False
    assert contract["selects_the_conditional_Poisson_kernel"] is False


def test_exact_nonidentifiability_and_negative_controls(rebuilt: dict[str, object]) -> None:
    controls = rebuilt["exact_nonidentifiability_and_negative_controls"]
    scalar = controls["same_scalar_rate_different_counting_measure"]
    assert sum(scalar["history_A_counts"]) == sum(scalar["history_B_counts"]) == 2
    assert scalar["history_A_counts"] != scalar["history_B_counts"]
    assert scalar["different_set_indexed_measures"] is True
    laws = controls["same_mean_measure_different_history_laws"]
    assert laws["Poisson_second_factorial_moment"] == "4"
    assert laws["Cox_second_factorial_moment"] == "5"
    assert laws["same_first_moment"] is True
    assert laws["different_laws"] is True
    assert controls["locally_infinite_history_negative_control"]["rejected_from_H_lf"] is True
    assert (
        controls["endpoint_double_count_negative_control"][
            "rejected_by_absorption_count_convention"
        ]
        is True
    )
    assert controls["assertion_as_derivation_negative_control"]["rejected"] is True


def test_both_branches_remain_blocked_without_rejection(rebuilt: dict[str, object]) -> None:
    records = rebuilt["candidate_records"]
    assert [record["branch_id"] for record in records] == [
        "eq35_middle_h",
        "eq35_printed_planck",
    ]
    for record in records:
        assert record["paper_semantics_audit_is_branch_independent"] is True
        assert record["paper_typed_history_map_complete"] is False
        assert record["paper_or_QED_kernel_selected"] is False
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
    opened["seals"]["observations_opened"] = True
    path.write_text(json.dumps(opened), encoding="utf-8")
    with pytest.raises(ValueError, match="seal opened"):
        build_audit(path)

    overclaimed = copy.deepcopy(rebuilt)
    overclaimed["candidate_records"][0]["paper_typed_history_map_complete"] = True
    overclaimed.pop("content_sha256")
    with pytest.raises(ValueError, match="paper-map overclaim"):
        _validate_result(overclaimed)

    selected = copy.deepcopy(rebuilt)
    selected["candidate_records"][0]["paper_or_QED_kernel_selected"] = True
    selected.pop("content_sha256")
    with pytest.raises(ValueError, match="kernel-selection overclaim"):
        _validate_result(selected)

    rejected = copy.deepcopy(rebuilt)
    rejected["candidate_records"][0]["candidate_action_rejection_authorized"] = True
    rejected.pop("content_sha256")
    with pytest.raises(ValueError, match="overreached"):
        _validate_result(rejected)


def test_source_bindings_are_exact_portable_and_sealed(rebuilt: dict[str, object]) -> None:
    bindings = rebuilt["source_bindings"]
    for label in (
        "conditional_poisson_kernel",
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
