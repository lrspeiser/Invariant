from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.quartic_candidate_complete_global_h7_lifespan_gate import (
    EXPECTED_CANDIDATES,
    EXPECTED_CLAIMS,
    EXPECTED_COUNTS,
    EXPECTED_DIRECTIONS,
    _content_sha,
    _load_documents,
    _normalized_text_sha,
    _validate_config,
    _validate_result,
    build_gate,
    main,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/backgrounds/quartic_candidate_complete_global_h7_lifespan_gate.json"
ARTIFACT = (
    ROOT / "runs/physics-language/quartic-candidate-complete-global-h7-lifespan-gate/campaign.json"
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
    assert not any(checked["claims"].values())


def test_all_twelve_candidates_are_audited_without_redefining_completion(
    rebuilt: dict[str, object],
) -> None:
    records = rebuilt["candidate_records"]
    assert len(records) == EXPECTED_CANDIDATES
    assert len({record["candidate_id"] for record in records}) == EXPECTED_CANDIDATES
    assert all(record["decision"] == "BLOCK_SYSTEM9" for record in records)
    assert all(record["completion_grade"] is False for record in records)
    assert all(
        record["global_H7_proof_branch"]["energy_equivalence_certified"]
        and record["global_H7_proof_branch"]["known_nonremainder_terms_summed"]
        and record["global_H7_proof_branch"]["strongest_inequality_has_explicit_B7_remainder"]
        and not record["global_H7_proof_branch"]["closed_global_H7_inequality"]
        and not record["global_H7_proof_branch"]["positive_lifespan_proved"]
        for record in records
    )


def test_scoped_finite_sobolev_no_go_is_preserved_exactly(
    rebuilt: dict[str, object],
) -> None:
    records = rebuilt["candidate_records"]
    assert {record["obstruction_branch"]["representative_D2_value"] for record in records} == {
        "-2",
        "-1",
        "1",
        "2",
    }
    assert {record["obstruction_branch"]["absolute_growth_multiplier"] for record in records} == {
        "1",
        "2",
    }
    for record in records:
        obstruction = record["obstruction_branch"]
        assert obstruction["finite_unmodified_Sobolev_no_go_proved"]
        assert "every finite integer s>=4" in obstruction["theorem_domain"]
        assert not obstruction["full_tensor_cancellation_excluded"]
        assert not obstruction["modified_energy_excluded"]
        assert not obstruction["Nash_Moser_or_derivative_loss_evolution_excluded"]
        assert not obstruction["analytic_or_Gevrey_closure_excluded"]
        assert not obstruction["full_direction_completion_obstruction_proved"]


def test_full_direction_replay_boundary_is_measured_not_inferred(
    rebuilt: dict[str, object],
) -> None:
    replay = rebuilt["full_direction_replay_audit"]
    assert replay["required_evaluations"] == list(EXPECTED_DIRECTIONS)
    assert replay["P55_input_evaluations_replayed"] == list(EXPECTED_DIRECTIONS)
    assert replay["H_star_input_evaluations_replayed"] == list(EXPECTED_DIRECTIONS)
    assert replay["accepted_alternative_recurrence_evaluations"] == []
    assert replay["witness_local_only_evaluations"] == ["subset_2"]
    assert len(replay["missing_accepted_alternative_recurrence_evaluations"]) == 14
    assert "subset_2" not in replay["missing_accepted_alternative_recurrence_evaluations"]
    assert not replay["full_direction_completion_replay_closed"]
    assert all(
        record["shared_System8_scope"]["accepted_full_direction_replays"] == 0
        and record["shared_System8_scope"]["required_full_direction_replays"] == 15
        and not record["shared_System8_scope"]["accepted_global_coordinate_free_recurrence"]
        for record in rebuilt["candidate_records"]
    )


def test_executed_corruption_and_scope_promotion_negatives_all_reject(
    rebuilt: dict[str, object],
) -> None:
    assert set(rebuilt["negative_controls"]) == {
        "omit_one_candidate",
        "erase_nonzero_candidate_D2_slice",
        "promote_scoped_no_go_to_completion",
        "corrupt_full_direction_manifest",
        "promote_witness_local_recurrence_to_full_direction",
    }
    assert all(control["rejected"] for control in rebuilt["negative_controls"].values())


@pytest.mark.parametrize(
    "mutation",
    [
        "candidate_pass",
        "completion_grade_obstruction",
        "full_direction_pass",
        "open_claim",
        "erase_candidate",
        "extra_key",
    ],
)
def test_resealed_receipt_tampering_fails_closed(rebuilt: dict[str, object], mutation: str) -> None:
    changed = copy.deepcopy(rebuilt)
    if mutation == "candidate_pass":
        changed["candidate_records"][0]["decision"] = "PASS"
    elif mutation == "completion_grade_obstruction":
        changed["candidate_records"][0]["obstruction_branch"][
            "full_direction_completion_obstruction_proved"
        ] = True
    elif mutation == "full_direction_pass":
        changed["full_direction_replay_audit"]["accepted_alternative_recurrence_evaluations"] = (
            list(EXPECTED_DIRECTIONS)
        )
        changed["full_direction_replay_audit"]["full_direction_completion_replay_closed"] = True
    elif mutation == "open_claim":
        changed["claims"]["lifespan_proved"] = True
    elif mutation == "erase_candidate":
        changed["candidate_records"].pop()
    else:
        changed["invented_completion"] = True
    _reseal(changed)
    with pytest.raises(ValueError, match="System9"):
        _validate_result(changed)


@pytest.mark.parametrize(
    "source_label",
    [
        "global_h7",
        "finite_sobolev_no_go",
        "higher_p55",
        "higher_h_star",
        "higher_k55",
        "k55_sylvester_obstruction",
        "physical_metric_transport_no_go",
        "alternative_symmetrizer",
    ],
)
def test_each_predecessor_content_corruption_rejects(source_label: str) -> None:
    config = _load(CONFIG)
    documents = _load_documents(ROOT, config)
    documents[source_label]["invented_completion"] = True
    _reseal(documents[source_label])
    with pytest.raises(ValueError, match="System9"):
        build_gate(ROOT, CONFIG, documents=documents)


def test_config_completion_policy_tamper_rejects(tmp_path: Path) -> None:
    config = _load(CONFIG)
    config["completion_contract"]["scoped_no_go_is_completion_grade"] = True
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="completion definition"):
        build_gate(ROOT, changed)


def test_bindings_are_path_free_and_materializer_stable(rebuilt: dict[str, object]) -> None:
    for binding in rebuilt["source_bindings"].values():
        assert not Path(binding["path"]).is_absolute()
        assert binding["content_sha256"] == binding["semantic_sha256"]
    for binding in rebuilt["local_bindings"].values():
        assert not Path(binding["path"]).is_absolute()
        path = ROOT / binding["path"]
        normalized = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
        assert hashlib.sha256(normalized.encode()).hexdigest() == binding["normalized_text_sha256"]


def test_local_text_seals_survive_crlf_materialization(
    rebuilt: dict[str, object], tmp_path: Path
) -> None:
    for label in ("source", "test"):
        binding = rebuilt["local_bindings"][label]
        text = (ROOT / binding["path"]).read_text(encoding="utf-8")
        materialized = tmp_path / f"{label}.txt"
        materialized.write_bytes(text.replace("\r\n", "\n").replace("\n", "\r\n").encode())
        assert _normalized_text_sha(materialized) == binding["normalized_text_sha256"]


def test_cli_replays_checked_contract(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--root", str(ROOT), "--config", str(CONFIG)]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output == _load(ARTIFACT)


def test_source_has_no_runtime_data_gpu_or_process_surface() -> None:
    source = (
        ROOT / "src/sigma_theory_compiler/quartic_candidate_complete_global_h7_lifespan_gate.py"
    ).read_text(encoding="utf-8")
    lowered = source.lower()
    for forbidden in (
        "sqlite3",
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


def test_config_closed_world() -> None:
    config = _load(CONFIG)
    _validate_config(config)
    changed = copy.deepcopy(config)
    changed["required_polarization_evaluations"].pop()
    with pytest.raises(ValueError, match="config boundary"):
        _validate_config(changed)
