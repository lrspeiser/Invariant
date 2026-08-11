from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.kastner_schlatter_finite_factorial_hierarchy_no_go import (
    EXPECTED_CLAIM_SEALS,
    EXPECTED_COUNTS,
    FIRST_BLOCKER,
    _validate_result,
    build_gate,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/kastner_schlatter_finite_factorial_hierarchy_no_go.json"
ARTIFACT = ROOT / "runs/engine/kastner-schlatter-finite-factorial-hierarchy-no-go.json"


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


def test_finite_difference_theorem_is_exact_and_narrow(rebuilt: dict[str, object]) -> None:
    theorem = rebuilt["finite_hierarchy_no_go"]
    assert theorem["theorem_name"] == ("arbitrary_finite_factorial_hierarchy_nonidentifiability")
    assert "0<=j<=k" in theorem["finite_difference_identity"]
    assert "(-1)^(k+1)*exp(-1)/2" in theorem["first_difference"]
    assert "varies the counterexample with k" in theorem["scope_limit"]
    assert "does not assert that one law matches every order" in theorem["scope_limit"]
    assert "without an analytic or exponential-integrability premise" in theorem["scope_limit"]


def test_orders_one_through_six_replay_exact_identities(rebuilt: dict[str, object]) -> None:
    controls = rebuilt["exact_controls"]["orders_1_through_6"]
    assert rebuilt["exact_controls"]["identity_count"] == 27
    assert [row["k"] for row in controls] == [1, 2, 3, 4, 5, 6]
    for row in controls:
        k = row["k"]
        assert len(row["scaled_modified_probabilities"]) == k + 2
        assert row["all_modified_probabilities_positive"] is True
        assert row["factorial_moment_perturbation_residuals_0_through_k"] == ["0"] * (k + 1)
        assert row["normalization_preserved"] is True
        assert row["orders_1_through_k_match_Poisson"] is True
        assert row["first_unmatched_factorial_order"] == k + 1
        assert row["first_unmatched_residual_coefficient_of_exp_minus_1"] == (
            "1/2" if k % 2 else "-1/2"
        )
        assert row["order_k_plus_1_differs"] is True


def test_point_process_lift_matches_every_prescribed_finite_order(
    rebuilt: dict[str, object],
) -> None:
    theorem = rebuilt["finite_hierarchy_no_go"]
    assert "factorial moment measures alpha_j equal mu^tensor_j" in theorem["point_process_lift"]
    assert "for every 1<=j<=k" in theorem["point_process_lift"]
    assert "differ on B^(k+1)" in theorem["point_process_lift"]
    assert "any preassigned finite factorial-moment hierarchy" in theorem["conclusion"]


def test_remaining_selector_contract_is_nonfinite_and_fail_closed(
    rebuilt: dict[str, object],
) -> None:
    contract = rebuilt["minimal_next_contract"]
    assert contract["finite_order_status"] == "ruled_out_for_every_fixed_k"
    assert contract["registered_remaining_selectors"] == 0
    assert contract["paper_or_QED_attribution"] is False
    assert contract["first_missing_premise"] == FIRST_BLOCKER
    assert len(contract["smallest_honest_remaining_selector_classes"]) == 5
    assert (
        "infinite factorial hierarchy plus an explicit moment-determinacy theorem"
        in contract["smallest_honest_remaining_selector_classes"]
    )


def test_both_candidates_advance_theorem_only_and_remain_blocked(
    rebuilt: dict[str, object],
) -> None:
    assert [(row["branch_id"], row["beta"]) for row in rebuilt["candidate_records"]] == [
        ("eq35_middle_h", "1/2"),
        ("eq35_printed_planck", "1/4"),
    ]
    for record in rebuilt["candidate_records"]:
        assert record["arbitrary_finite_factorial_hierarchy_no_go"] == "pass"
        assert record["candidate_bound_counterexample_for_every_fixed_k"] is True
        assert record["registered_nonfinite_selector"] is False
        assert record["paper_or_QED_selector_derived"] is False
        assert record["candidate_action_selects_Poisson"] is False
        assert record["candidate_action_rejection_authorized"] is False
        assert record["candidate_decision"] == "blocked"


def test_lineage_and_overclaim_tampering_fail_closed(
    tmp_path: Path, rebuilt: dict[str, object]
) -> None:
    config = _load(CONFIG)
    tampered = copy.deepcopy(config)
    tampered["predecessors"]["second_order_selector_no_go"]["content_sha256"] = "0" * 64
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

    overreach = copy.deepcopy(rebuilt)
    overreach["claim_seals"]["finite_factorial_hierarchy_claimed_sufficient"] = True
    overreach["content_sha256"] = None
    with pytest.raises(ValueError, match="seal changed"):
        _validate_result(overreach)

    rejected = copy.deepcopy(rebuilt)
    rejected["candidate_records"][0]["candidate_action_rejection_authorized"] = True
    rejected["content_sha256"] = None
    with pytest.raises(ValueError, match="candidate boundary changed"):
        _validate_result(rejected)


def test_source_bindings_are_exact_portable_and_sealed(rebuilt: dict[str, object]) -> None:
    bindings = rebuilt["source_bindings"]
    for label in (
        "second_order_selector_no_go",
        "canonical_probability_space",
        "candidate_action_completion",
        "positive_reparameterization",
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
    assert all(value is False for value in rebuilt["claim_seals"].values())
    assert all(value is False for value in rebuilt["data_seals"].values())
    assert all("C:/" not in str(value) and "C:\\" not in str(value) for value in bindings.values())


def test_source_has_no_runtime_data_or_process_surface() -> None:
    source = (
        ROOT / "src/sigma_theory_compiler/kastner_schlatter_finite_factorial_hierarchy_no_go.py"
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
