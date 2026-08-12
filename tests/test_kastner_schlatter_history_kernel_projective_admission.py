from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import pytest

from sigma_theory_compiler.kastner_schlatter_history_kernel_projective_admission import (
    EXPECTED_CLAIM_SEALS,
    EXPECTED_COUNTS,
    EXPECTED_DATA_SEALS,
    EXPECTED_DOMAIN,
    EXPECTED_SCOPE,
    EXPECTED_SECONDARY_BLOCKERS,
    FIRST_BLOCKER,
    _validate_result,
    build_gate,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/kastner_schlatter_history_kernel_projective_admission.json"
ARTIFACT = ROOT / "runs/engine/kastner-schlatter-history-kernel-projective-admission.json"


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _reseal(value: dict[str, object]) -> None:
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    value["content_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()


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


def test_projective_contract_sharpens_missing_history_kernel(rebuilt: dict[str, object]) -> None:
    obligations = rebuilt["admission_obligations"]
    assert len(obligations) == 6
    assert [row["source_status"] for row in obligations].count("partial_intensity_only") == 1
    assert [row["source_status"] for row in obligations].count("absent") == 5
    theorem = rebuilt["extension_theorem"]
    assert len(theorem["premises"]) == 4
    assert "unique candidate-conditioned probability kernel" in theorem["conclusion"]
    assert "neither derives the family from the paper/QED" in theorem["scope_limit"]


def test_same_total_poisson_and_cell_means_do_not_determine_history_law(
    rebuilt: dict[str, object],
) -> None:
    witness = rebuilt["exact_nonidentifiability_witness"]
    assert witness["shared_inputs"]["total_count"] == "K=N_A+N_B~Poisson(2)"
    assert witness["shared_inputs"]["cell_means"] == {"E[N_A]": "1", "E[N_B]": "1"}
    independent_void = math.exp(-1)
    coherent_void = (1 + math.exp(-2)) / 2
    assert coherent_void - independent_void > 0
    assert math.isclose(
        coherent_void - independent_void,
        (1 - math.exp(-1)) ** 2 / 2,
        rel_tol=0,
        abs_tol=1e-15,
    )
    assert witness["source_or_QED_attribution"] is False


def test_positive_and_negative_projective_controls_are_exact(
    rebuilt: dict[str, object],
) -> None:
    controls = rebuilt["exact_controls"]
    assert controls["compiler_independent_PRM_positive_control"]["projective_consistency"] == (
        "pass"
    )
    assert controls["compiler_independent_PRM_positive_control"][
        "source_or_QED_attribution"
    ] is False
    assert controls["inconsistent_coarsening_negative_control"]["actual_fine_pushforward"] == (
        "delta_1"
    )
    assert controls["inconsistent_coarsening_negative_control"]["rejected"] is True
    assert controls["scalar_PMF_promotion_negative_control"]["rejected"] is True


def test_both_candidate_actions_remain_blocked_and_unrejected(
    rebuilt: dict[str, object],
) -> None:
    assert [(row["branch_id"], row["beta"]) for row in rebuilt["candidate_records"]] == [
        ("eq35_middle_h", "1/2"),
        ("eq35_printed_planck", "1/4"),
    ]
    for record in rebuilt["candidate_records"]:
        assert record["same_input_distinct_history_law_witness"] is True
        assert record["paper_or_QED_history_kernel_registered"] is False
        assert record["candidate_action_selects_history_law"] is False
        assert record["candidate_action_rejection_authorized"] is False
        assert record["candidate_decision"] == "blocked"


def test_tamper_controls_fail_closed(tmp_path: Path, rebuilt: dict[str, object]) -> None:
    config = _load(CONFIG)
    tampered = copy.deepcopy(config)
    tampered["predecessors"]["actualization_probability_bridge"]["content_sha256"] = "0" * 64
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
    selected["candidate_records"][0]["candidate_action_selects_history_law"] = True
    _reseal(selected)
    with pytest.raises(ValueError, match="result boundary changed"):
        _validate_result(selected)

    attributed = copy.deepcopy(rebuilt)
    attributed["claim_seals"]["compiler_nonidentifiability_witness_attributed_to_source"] = True
    _reseal(attributed)
    with pytest.raises(ValueError, match="result boundary changed"):
        _validate_result(attributed)


@pytest.mark.parametrize(
    "mutation",
    [
        "source_predecessor_path",
        "source_predecessor_file_hash",
        "source_own_test_hash",
        "source_binding_removed",
        "source_bindings_empty",
        "scope_changed",
        "domain_changed",
        "secondary_removed",
        "secondary_reordered",
        "data_key_removed",
        "data_false_key_added",
        "data_seals_empty",
        "top_level_extra",
    ],
)
def test_resealed_envelope_and_provenance_tampering_fail_closed(
    rebuilt: dict[str, object], mutation: str
) -> None:
    value = copy.deepcopy(rebuilt)
    if mutation == "source_predecessor_path":
        value["source_bindings"]["actualization_probability_bridge"]["path"] = (
            "runs/engine/mutated.json"
        )
    elif mutation == "source_predecessor_file_hash":
        value["source_bindings"]["source_type_audit"]["file_sha256"] = "0" * 64
    elif mutation == "source_own_test_hash":
        value["source_bindings"]["test"]["file_sha256"] = "0" * 64
    elif mutation == "source_binding_removed":
        value["source_bindings"].pop("qed_actualization_audit")
    elif mutation == "source_bindings_empty":
        value["source_bindings"] = {}
    elif mutation == "scope_changed":
        value["scope"] = "physical history kernel proved"
    elif mutation == "domain_changed":
        value["admission_domain"]["target"] = "mutated target"
    elif mutation == "secondary_removed":
        value["secondary_blockers"].pop()
    elif mutation == "secondary_reordered":
        value["secondary_blockers"].reverse()
    elif mutation == "data_key_removed":
        value["data_seals"].pop("observations_opened")
    elif mutation == "data_false_key_added":
        value["data_seals"]["invented_false_key"] = False
    elif mutation == "data_seals_empty":
        value["data_seals"] = {}
    else:
        value["invented_field"] = False
    _reseal(value)
    with pytest.raises(ValueError, match="history-kernel projective"):
        _validate_result(value)


def test_source_bindings_and_seals_are_exact(rebuilt: dict[str, object]) -> None:
    bindings = rebuilt["source_bindings"]
    for label in (
        "actualization_probability_bridge",
        "source_type_audit",
        "qed_actualization_audit",
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
    assert rebuilt["data_seals"] == EXPECTED_DATA_SEALS
    assert rebuilt["admission_domain"] == EXPECTED_DOMAIN
    assert rebuilt["secondary_blockers"] == EXPECTED_SECONDARY_BLOCKERS
    assert rebuilt["scope"] == EXPECTED_SCOPE
    assert not any(rebuilt["claim_seals"].values())
    assert not any(rebuilt["data_seals"].values())


def test_source_has_no_runtime_or_data_surface() -> None:
    source = (
        ROOT
        / "src/sigma_theory_compiler/kastner_schlatter_history_kernel_projective_admission.py"
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
