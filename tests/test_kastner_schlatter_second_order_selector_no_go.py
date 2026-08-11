from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.kastner_schlatter_second_order_selector_no_go import (
    EXPECTED_CLAIM_SEALS,
    EXPECTED_COUNTS,
    FIRST_BLOCKER,
    _validate_result,
    build_gate,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/kastner_schlatter_second_order_selector_no_go.json"
ARTIFACT = ROOT / "runs/engine/kastner-schlatter-second-order-selector-no-go.json"


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


def test_inside_count_matches_poisson_through_second_order(
    rebuilt: dict[str, object],
) -> None:
    moments = rebuilt["exact_controls"]["inside_count_moments"]
    assert moments["support"] == [0, 3]
    assert moments["probabilities"] == ["1/3", "2/3"]
    assert moments["mean"] == moments["Poisson_mean"] == "2"
    assert moments["variance"] == moments["Poisson_variance"] == "2"
    assert moments["second_factorial_moment"] == (moments["Poisson_second_factorial_moment"]) == "4"
    assert (
        moments["second_factorial_cumulant"]
        == (moments["Poisson_second_factorial_cumulant"])
        == "0"
    )


def test_third_order_and_void_probability_separate_the_laws(
    rebuilt: dict[str, object],
) -> None:
    moments = rebuilt["exact_controls"]["inside_count_moments"]
    assert moments["third_factorial_moment"] == "4"
    assert moments["Poisson_third_factorial_moment"] == "8"
    assert moments["witness_void_probability"] == "1/3"
    assert moments["Poisson_void_probability"] == "exp(-2)"


def test_point_process_extension_matches_full_first_and_second_measures(
    rebuilt: dict[str, object],
) -> None:
    theorem = rebuilt["second_order_no_go"]
    assert theorem["global_first_factorial_measure"] == "alpha_1=mu"
    assert theorem["global_second_factorial_measure"] == "alpha_2=mu tensor mu"
    assert theorem["global_pair_cumulant_measure"] == "kappa_2=0, exactly as for PRM(mu)"
    assert "outside PRM supplies the cross blocks" in theorem["proof"]
    assert "requires an admissible diffuse cell" in theorem["scope_limit"]
    assert "does not establish physical non-Poisson events" in theorem["scope_limit"]


def test_next_contract_requires_full_characterization(rebuilt: dict[str, object]) -> None:
    contract = rebuilt["minimal_next_contract"]
    assert contract["registered_sufficient_targets"] == 0
    assert contract["first_missing_premise"] == FIRST_BLOCKER
    assert "vanishing second factorial cumulant measure" in contract["ruled_out_as_sufficient"]
    assert "pair correlation g2=1" in contract["ruled_out_as_sufficient"]
    assert len(contract["honest_sufficient_targets"]) == 4
    assert "does not alone characterize" in contract["finite_third_order_witness_role"]


def test_both_candidates_advance_no_go_only_and_remain_blocked(
    rebuilt: dict[str, object],
) -> None:
    assert [(row["branch_id"], row["beta"]) for row in rebuilt["candidate_records"]] == [
        ("eq35_middle_h", "1/2"),
        ("eq35_printed_planck", "1/4"),
    ]
    for record in rebuilt["candidate_records"]:
        assert record["second_order_selector_no_go"] == "pass"
        assert record["exact_non_Poisson_same_first_second_measure_witness"] is True
        assert record["paper_or_QED_full_selector_derived"] is False
        assert record["candidate_action_selects_Poisson"] is False
        assert record["candidate_action_rejection_authorized"] is False
        assert record["candidate_decision"] == "blocked"


def test_lineage_and_overclaim_tampering_fail_closed(
    tmp_path: Path, rebuilt: dict[str, object]
) -> None:
    config = _load(CONFIG)
    tampered = copy.deepcopy(config)
    tampered["predecessors"]["deterministic_feature_selector_no_go"]["content_sha256"] = "0" * 64
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
    overreach["claim_seals"]["second_order_claimed_to_characterize_Poisson"] = True
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
        "deterministic_feature_selector_no_go",
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
        ROOT / "src/sigma_theory_compiler/kastner_schlatter_second_order_selector_no_go.py"
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
