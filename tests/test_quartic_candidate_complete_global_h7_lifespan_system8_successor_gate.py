from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.quartic_candidate_complete_global_h7_lifespan_system8_successor_gate import (
    EXPECTED_CLAIMS,
    EXPECTED_COUNTS,
    EXPECTED_DIRECTIONS,
    _content_sha,
    _load_documents,
    _normalized_text_sha,
    _validate_result,
    _validate_sources,
    build_gate,
    main,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT / "configs/backgrounds/"
    "quartic_candidate_complete_global_h7_lifespan_system8_successor_gate.json"
)
ARTIFACT = (
    ROOT / "runs/physics-language/"
    "quartic-candidate-complete-global-h7-lifespan-system8-successor-gate/campaign.json"
)


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _reseal(value: dict[str, object]) -> None:
    value["content_sha256"] = _content_sha(value)


@pytest.fixture(scope="module")
def rebuilt() -> dict[str, object]:
    return build_gate(ROOT, CONFIG)


def test_exact_rebuild_matches_checked_receipt(rebuilt: dict[str, object]) -> None:
    checked = _load(ARTIFACT)
    assert rebuilt == checked
    _validate_result(checked)
    assert checked["decision"] == "BLOCK_SYSTEM9"
    assert checked["counts"] == EXPECTED_COUNTS
    assert checked["claims"] == EXPECTED_CLAIMS


def test_all_15_transport_success_is_recorded_exactly(rebuilt: dict[str, object]) -> None:
    audit = rebuilt["system8_successor_evidence"]["transport_audit"]
    assert audit["required_evaluations"] == list(EXPECTED_DIRECTIONS)
    assert audit["proved_evaluations"] == sorted(EXPECTED_DIRECTIONS)
    assert audit["all_15_independent_G2_transports_proved"]
    assert rebuilt["counts"]["proved_independent_G2_transport_evaluations"] == 15


def test_exact_55_state_lift_obstruction_is_not_overclaimed(
    rebuilt: dict[str, object],
) -> None:
    lift = rebuilt["system8_successor_evidence"]["exact_55_state_lift_frontier"]
    assert lift["passed_evaluations"] == ["subset_0", "subset_1"]
    assert lift["first_failure_evaluation"] == "subset_2"
    assert lift["first_failure_Taylor_order"] == 3
    assert lift["symmetry_remainder_nonzero_entries_by_order"] == [0, 0, 0]
    assert lift["symmetrizer_remainder_nonzero_entries_by_order"] == [0, 0, 72]
    assert not lift["positive_tube_attempted"]
    assert not rebuilt["claims"]["all_15_exact_55_state_lifts_proved"]
    assert not rebuilt["claims"]["positive_symmetrizer_tube_proved"]


def test_no_candidate_is_upgraded_to_completion_grade(rebuilt: dict[str, object]) -> None:
    assert rebuilt["measured_upgrade"]["candidate_completion_grade_upgrade_count"] == 0
    assert rebuilt["measured_upgrade"]["candidate_completion_grade_obstructions_upgraded"] == []
    records = rebuilt["candidate_records"]
    assert len(records) == 12
    assert all(record["decision"] == "BLOCK_SYSTEM9" for record in records)
    assert all(not record["completion_grade_after"] for record in records)
    for record in records:
        scope = record["why_not_completion_grade"]
        assert not scope["candidate_bound"]
        assert not scope["positive_symmetrizer_tube_proved"]
        assert not scope["full_tensor_cancellation_excluded"]
        assert not scope["modified_energy_excluded"]
        assert not scope["Nash_Moser_or_derivative_loss_evolution_excluded"]
        assert not scope["analytic_or_Gevrey_closure_excluded"]


def test_exact_remaining_primitives_are_sealed(rebuilt: dict[str, object]) -> None:
    remaining = rebuilt["exact_remaining_contract"]
    assert remaining["first_System8_missing_primitive"] == (
        "exact_55_state_transverse_cross_lift_of_independent_G2_metric"
    )
    assert remaining["first_System8_failure_location"] == {
        "evaluation_id": "subset_2",
        "Taylor_order": 3,
        "symmetrizer_remainder_nonzero_entries": 72,
    }
    assert remaining["first_System9_completion_primitive"] == (
        "candidate_bound_full_tensor_source_good_unknown_B7_bound_or_"
        "all_closure_strategy_completion_grade_obstruction"
    )


def test_executed_source_negatives_all_reject(rebuilt: dict[str, object]) -> None:
    assert set(rebuilt["negative_controls"]) == {
        "omit_one_transport_evaluation",
        "erase_exact_72_entry_lift_obstruction",
        "promote_partial_lift_to_all_15",
        "upgrade_prior_candidate_without_theorem",
    }
    assert all(control["rejected"] for control in rebuilt["negative_controls"].values())


@pytest.mark.parametrize(
    "mutation",
    ["upgrade_candidate", "erase_72", "open_lifespan", "invent_key"],
)
def test_resealed_receipt_tampering_fails_closed(rebuilt: dict[str, object], mutation: str) -> None:
    changed = copy.deepcopy(rebuilt)
    if mutation == "upgrade_candidate":
        changed["candidate_records"][0]["completion_grade_after"] = True
    elif mutation == "erase_72":
        changed["system8_successor_evidence"]["exact_55_state_lift_frontier"][
            "symmetrizer_remainder_nonzero_entries_by_order"
        ] = [0, 0, 0]
    elif mutation == "open_lifespan":
        changed["claims"]["lifespan_proved"] = True
    else:
        changed["invented_completion"] = True
    _reseal(changed)
    with pytest.raises(ValueError, match="System9 successor"):
        _validate_result(changed)


@pytest.mark.parametrize("label", ["system9_candidate_gate", "system8_independent_g2"])
def test_predecessor_content_corruption_rejects(label: str) -> None:
    config = _load(CONFIG)
    documents = _load_documents(ROOT, config)
    documents[label]["invented_completion"] = True
    _reseal(documents[label])
    with pytest.raises(ValueError, match="System9 successor"):
        build_gate(ROOT, CONFIG, documents=documents)


def test_semantic_72_corruption_rejects_even_when_resealed_and_rebound() -> None:
    config = _load(CONFIG)
    documents = _load_documents(ROOT, config)
    broader = documents["system8_independent_g2"]
    broader["first_exact_55_state_lift_failure"]["K55_symmetrizer_remainder_entries"] = [0, 0, 0]
    _reseal(broader)
    config["predecessors"]["system8_independent_g2"]["content_sha256"] = broader["content_sha256"]
    with pytest.raises(ValueError, match="exact lift obstruction"):
        _validate_sources(config, documents)


def test_bindings_are_path_free_and_materializer_stable(
    rebuilt: dict[str, object], tmp_path: Path
) -> None:
    for binding in rebuilt["source_bindings"].values():
        assert not Path(binding["path"]).is_absolute()
        assert binding["content_sha256"] == binding["semantic_sha256"]
    for label, binding in rebuilt["local_bindings"].items():
        assert not Path(binding["path"]).is_absolute()
        path = ROOT / binding["path"]
        assert _normalized_text_sha(path) == binding["normalized_text_sha256"]
        text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
        materialized = tmp_path / f"{label}.txt"
        materialized.write_bytes(text.replace("\n", "\r\n").encode())
        assert _normalized_text_sha(materialized) == binding["normalized_text_sha256"]
        assert hashlib.sha256(text.encode()).hexdigest() == binding["normalized_text_sha256"]


def test_cli_replays_checked_contract(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--root", str(ROOT), "--config", str(CONFIG)]) == 0
    assert json.loads(capsys.readouterr().out) == _load(ARTIFACT)
