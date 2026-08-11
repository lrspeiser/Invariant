from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.kastner_schlatter_source_selector_type_inhabitation_audit import (
    EXPECTED_CLAIM_SEALS,
    EXPECTED_COUNTS,
    FIRST_BLOCKER,
    _validate_result,
    build_gate,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/kastner_schlatter_source_selector_type_inhabitation_audit.json"
ARTIFACT = ROOT / "runs/engine/kastner-schlatter-source-selector-type-inhabitation-audit.json"


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


def test_source_evidence_classifications_are_preserved(rebuilt: dict[str, object]) -> None:
    evidence = rebuilt["source_evidence"]
    assert len(evidence) == 5
    assert {row["status"] for row in evidence} == {
        "semantic_only",
        "first_moment_or_density_only",
        "assertion_without_registered_channel_kernel",
        "implementation_not_paper_equation_and_not_derivation",
        "synthetic_only_no_QED_inference",
    }


def test_laplace_type_slots_have_no_complete_source_instance(
    rebuilt: dict[str, object],
) -> None:
    slots = rebuilt["Laplace_required_slots"]
    assert len(slots) == 10
    assert sum(row["status"] == "complete" for row in slots) == 0
    assert sum(row["status"] == "partial" for row in slots) == 3
    assert sum(row["status"] == "absent" for row in slots) == 7
    assert {row["slot"] for row in slots if row["status"] == "partial"} == {
        "measurable_region_or_continuity_ring",
        "set_indexed_counting_random_measure_N_of_A",
        "intensity_measure_mu",
    }


def test_mecke_type_slots_have_no_complete_source_instance(
    rebuilt: dict[str, object],
) -> None:
    slots = rebuilt["Mecke_required_slots"]
    assert len(slots) == 10
    assert sum(row["status"] == "complete" for row in slots) == 0
    assert sum(row["status"] == "partial" for row in slots) == 2
    assert sum(row["status"] == "absent" for row in slots) == 8
    assert {row["slot"] for row in slots if row["status"] == "partial"} == {
        "count_or_actualization_object",
        "intensity_measure_mu",
    }


def test_type_inhabitation_theorem_is_closed_world_and_narrow(
    rebuilt: dict[str, object],
) -> None:
    theorem = rebuilt["type_inhabitation_theorem"]
    assert theorem["theorem_name"] == (
        "registered_source_domain_countable_selector_noninhabitation"
    )
    assert "no nontrivial rational-simple-function core instance" in theorem["Laplace_conclusion"]
    assert "no nontrivial rational-cylinder/add-one core instance" in theorem["Mecke_conclusion"]
    assert "inhabits neither countable core" in theorem["scalar_PMF_boundary"]
    assert "not a claim that no future microscopic QED model" in theorem["scope_limit"]


def test_scalar_transform_partial_replay_is_not_promoted(rebuilt: dict[str, object]) -> None:
    control = rebuilt["exact_controls"]["compiler_scalar_transform_partial_replay"]
    assert control["derived_scalar_identity"] == ("sum_n exp(-t*n)*p(n|mu)=exp(-mu*(1-exp(-t)))")
    assert control["paper_printed_equation"] is False
    assert control["has_named_region_A"] is False
    assert control["has_counting_random_measure_N_of_A"] is False
    assert control["has_countable_core_quantifier"] is False
    assert control["qualifies_as_source_bound_core_certificate"] is False
    assert rebuilt["exact_controls"]["Poisson_prose_promotion_negative_control"]["rejected"] is True
    assert rebuilt["exact_controls"]["rate_to_compensator_negative_control"]["rejected"] is True


def test_both_candidates_have_zero_source_instances_and_remain_blocked(
    rebuilt: dict[str, object],
) -> None:
    assert [(row["branch_id"], row["beta"]) for row in rebuilt["candidate_records"]] == [
        ("eq35_middle_h", "1/2"),
        ("eq35_printed_planck", "1/4"),
    ]
    for record in rebuilt["candidate_records"]:
        assert record["source_Laplace_core_instances"] == 0
        assert record["source_Mecke_core_instances"] == 0
        assert record["compiler_scalar_transform_partial_replay"] is True
        assert record["paper_or_QED_selector_derived"] is False
        assert record["candidate_action_selects_Poisson"] is False
        assert record["candidate_action_rejection_authorized"] is False
        assert record["candidate_decision"] == "blocked"


def test_lineage_and_overclaim_tampering_fail_closed(
    tmp_path: Path, rebuilt: dict[str, object]
) -> None:
    config = _load(CONFIG)
    tampered = copy.deepcopy(config)
    tampered["predecessors"]["countable_full_law_admission"]["content_sha256"] = "0" * 64
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

    promoted = copy.deepcopy(rebuilt)
    promoted["claim_seals"]["compiler_scalar_replay_promoted_to_set_indexed_certificate"] = True
    promoted["content_sha256"] = None
    with pytest.raises(ValueError, match="seal changed"):
        _validate_result(promoted)

    rejected = copy.deepcopy(rebuilt)
    rejected["candidate_records"][0]["candidate_action_rejection_authorized"] = True
    rejected["content_sha256"] = None
    with pytest.raises(ValueError, match="candidate boundary changed"):
        _validate_result(rejected)


def test_source_bindings_are_exact_portable_and_sealed(rebuilt: dict[str, object]) -> None:
    bindings = rebuilt["source_bindings"]
    for label in (
        "countable_full_law_admission",
        "paper_intake",
        "poisson_selector_contract",
        "qed_actualization_audit",
        "actualization_history_map",
        "equation_graph",
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
        / "src/sigma_theory_compiler/kastner_schlatter_source_selector_type_inhabitation_audit.py"
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
