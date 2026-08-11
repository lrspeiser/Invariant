from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.kastner_schlatter_countable_full_law_selector_admission import (
    EXPECTED_CLAIM_SEALS,
    EXPECTED_COUNTS,
    FIRST_BLOCKER,
    _validate_result,
    build_gate,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/kastner_schlatter_countable_full_law_selector_admission.json"
ARTIFACT = ROOT / "runs/engine/kastner-schlatter-countable-full-law-selector-admission.json"


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def rebuilt() -> dict[str, object]:
    return build_gate(CONFIG)


def test_exact_rebuild_counts_and_blocker(rebuilt: dict[str, object]) -> None:
    artifact = _load(ARTIFACT)
    assert rebuilt == artifact
    _validate_result(artifact)
    assert artifact["decision_counts"] == {"pass": 0, "reject": 0, "blocked": 2}
    assert artifact["gate_counts"] == EXPECTED_COUNTS
    assert artifact["first_blocker"] == FIRST_BLOCKER


def test_countable_laplace_core_is_full_law_determining(rebuilt: dict[str, object]) -> None:
    route = rebuilt["countable_selector_admission_theorem"]["laplace_core_route"]
    assert "rational r_i>=0" in route["certificate"]
    assert "monotone convergence" in route["extension"]
    assert "every finite disjoint-family joint Laplace transform" in route["selection"]
    assert "evaluation sigma algebra" in route["selection"]
    assert route["mathematically_sufficient"] is True


def test_countable_mecke_core_is_full_law_determining(rebuilt: dict[str, object]) -> None:
    route = rebuilt["countable_selector_admission_theorem"]["mecke_core_route"]
    assert "N+delta_x" in route["certificate"]
    assert "functional monotone-class" in route["extension"]
    assert "Laplace ODE" in route["selection"]
    assert route["mathematically_sufficient"] is True
    theorem = rebuilt["countable_selector_admission_theorem"]
    assert "countable, hashable compiler object" in theorem["countability_role"]
    assert "not a paper, QED, or action derivation" in theorem["scope_limit"]


def test_exact_laplace_and_mecke_controls(rebuilt: dict[str, object]) -> None:
    controls = rebuilt["exact_controls"]
    laplace = controls["two_cell_Laplace_core_positive_control"]
    assert laplace["joint_exponent"] == laplace["factorized_exponent"]
    assert laplace["pass"] is True
    ode = controls["Mecke_to_Laplace_ODE_positive_control"]
    assert ode["ODE"] == "L'(t)=-2*exp(-t)*L(t)"
    assert ode["unique_solution"] == "L(t)=exp(-2*(1-exp(-t)))"
    assert ode["pass"] is True


def test_closed_world_audit_identifies_exact_absent_source_premises(
    rebuilt: dict[str, object],
) -> None:
    audit = rebuilt["closed_world_source_audit"]
    assert audit["equation_graph_nodes"] == 54
    assert audit["equation_graph_edges"] == 137
    assert audit["registered_scalar_PMF_nodes"] == 1
    assert audit["registered_selector_nodes"] == 0
    assert audit["action_derivation_edges_to_PMF"] == 0
    assert audit["paper_typed_history_maps"] == 0
    assert audit["QED_channel_kernels"] == 0
    assert audit["Laplace_core_certificates"] == 0
    assert audit["Mecke_core_certificates"] == 0
    ledger = rebuilt["evidence_ledger"]
    assert sum(row["status"] == "closed_by_compiler" for row in ledger) == 6
    assert sum(row["status"] == "absent" for row in ledger) == 6


def test_scalar_pmf_and_compiler_construction_do_not_cross_attribution_boundary(
    rebuilt: dict[str, object],
) -> None:
    controls = rebuilt["exact_controls"]
    pmf = controls["scalar_PMF_negative_control"]
    assert pmf["registered_node"] == "EQ-KS-POISSON-PMF-IMPLEMENTATION"
    assert pmf["has_set_argument"] is False
    assert pmf["qualifies_as_Laplace_core_certificate"] is False
    assert pmf["qualifies_as_Mecke_core_certificate"] is False
    attribution = controls["compiler_attribution_negative_control"]
    assert attribution["canonical_PRM_construction_exists"] is True
    assert attribution["paper_or_QED_supplies_certificate"] is False
    assert attribution["candidate_action_selects_certificate"] is False


def test_both_candidates_replay_math_only_and_remain_blocked(
    rebuilt: dict[str, object],
) -> None:
    assert [(row["branch_id"], row["beta"]) for row in rebuilt["candidate_records"]] == [
        ("eq35_middle_h", "1/2"),
        ("eq35_printed_planck", "1/4"),
    ]
    for record in rebuilt["candidate_records"]:
        assert record["compiler_countable_Laplace_route_replay"] == "pass"
        assert record["compiler_countable_Mecke_route_replay"] == "pass"
        assert record["source_bound_countable_selector_certificate"] is False
        assert record["paper_or_QED_selector_derived"] is False
        assert record["candidate_action_selects_Poisson"] is False
        assert record["candidate_action_rejection_authorized"] is False
        assert record["candidate_decision"] == "blocked"


def test_lineage_and_overclaim_tampering_fail_closed(
    tmp_path: Path, rebuilt: dict[str, object]
) -> None:
    config = _load(CONFIG)
    tampered = copy.deepcopy(config)
    tampered["predecessors"]["finite_factorial_hierarchy_no_go"]["content_sha256"] = "0" * 64
    path = tmp_path / "configs" / CONFIG.name
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="file hash mismatch|content hash mismatch"):
        build_gate(path)

    opened = copy.deepcopy(config)
    opened["seals"]["QED_actualization_derivation_opened"] = True
    path.write_text(json.dumps(opened), encoding="utf-8")
    with pytest.raises(ValueError, match="seal opened"):
        build_gate(path)

    selected = copy.deepcopy(rebuilt)
    selected["candidate_records"][0]["candidate_action_selects_Poisson"] = True
    selected["content_sha256"] = None
    with pytest.raises(ValueError, match="candidate boundary changed"):
        _validate_result(selected)

    attributed = copy.deepcopy(rebuilt)
    attributed["claim_seals"]["compiler_replay_attributed_to_source"] = True
    attributed["content_sha256"] = None
    with pytest.raises(ValueError, match="seal changed"):
        _validate_result(attributed)

    rejected = copy.deepcopy(rebuilt)
    rejected["candidate_records"][0]["candidate_action_rejection_authorized"] = True
    rejected["content_sha256"] = None
    with pytest.raises(ValueError, match="candidate boundary changed"):
        _validate_result(rejected)


def test_source_bindings_are_exact_portable_and_sealed(rebuilt: dict[str, object]) -> None:
    bindings = rebuilt["source_bindings"]
    for label in (
        "finite_factorial_hierarchy_no_go",
        "poisson_selector_contract",
        "canonical_probability_space",
        "equation_graph",
        "qed_actualization_audit",
        "actualization_history_map",
        "config",
        "source",
        "test",
    ):
        binding = bindings[label]
        assert _file_sha(ROOT / binding["path"]) == binding["file_sha256"]
        assert not Path(binding["path"]).is_absolute()
    assert bindings["primary_pdf_sha256"] == (
        "c2f671293d07b21397e745da00a3ce1a2193c00da647a2ebf4147612b76c1780"
    )
    assert rebuilt["claim_seals"] == EXPECTED_CLAIM_SEALS
    assert all(value is False for value in rebuilt["claim_seals"].values())
    assert all(value is False for value in rebuilt["data_seals"].values())
    assert all("C:/" not in str(value) and "C:\\" not in str(value) for value in bindings.values())


def test_source_has_no_runtime_data_or_process_surface() -> None:
    source = (
        ROOT
        / "src/sigma_theory_compiler/kastner_schlatter_countable_full_law_selector_admission.py"
    ).read_text(encoding="utf-8")
    lowered = source.lower()
    for forbidden in (
        "sqlite",
        "subprocess",
        "socket",
        "requests",
        "urllib",
        "cupy",
        "torch",
        "os.kill",
        "popen",
    ):
        assert forbidden not in lowered
