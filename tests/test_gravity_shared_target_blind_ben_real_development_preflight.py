from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler import (
    gravity_shared_target_blind_ben_real_development_preflight as preflight,
)


def config_sha256() -> str:
    return preflight.file_sha256(preflight.ROOT / preflight.CONFIG_PATH)


def config() -> dict:
    return preflight.load_config(preflight.ROOT / preflight.CONFIG_PATH, config_sha256())


def test_committed_candidate_registry_predates_incident_and_is_unchanged() -> None:
    frozen = config()
    registry = frozen["frozen_candidate_registry"]
    assert registry == {
        "freeze_commit": preflight.SYNTHETIC_COMMIT,
        "freeze_commit_time": preflight.SYNTHETIC_COMMIT_TIME,
        "architecture_id": "BEN-additive-cross-scale-v1",
        "formula_template": "A_nuisance*E_local_base+B_continuous_gate*N_additive_channel",
        "A_role": "bounded_source_calibration_nuisance_only",
        "M_temporal_phase_operator_included": False,
        "raw_candidate_count": 240,
        "equivalence_class_count": 60,
        "candidate_registry_content_sha256": preflight.REGISTRY_SHA256,
        "candidate_proposals_may_only_come_from_committed_registry": True,
    }
    assert frozen["chronology"]["candidate_registry_frozen_before_mixed_file_access"] is True


def test_incident_is_precise_and_confirmation_is_not_claimed_sealed() -> None:
    incident = config()["access_incident"]
    assert incident["mixed_file_path"] == "configs/sparc_rotation_curves_full_v1.json"
    assert incident["mixed_file_git_blob_sha1"] == preflight.MIXED_GIT_BLOB_SHA1
    assert incident["mixed_file_git_blob_bytes"] == 247_315
    assert incident["command_class"] == (
        "read_only_regex_search_rg_against_one_line_consolidated_json"
    )
    assert incident["process_read_entire_mixed_file"] is True
    assert incident["tool_output_was_truncated"] is True
    assert incident["exact_raw_rows_visible_to_agent_reconstructable"] is False
    assert incident["known_confirmation_objects_with_raw_rows_visible"] == [
        "D512-2",
        "DDO154",
        "ESO444-G084",
    ]
    assert incident["known_visible_confirmation_raw_row_lower_bound"] == 23
    assert incident["exact_confirmation_raw_rows_surfaced"] is None
    assert incident["local_sparc_confirmation_remains_sealed_for_this_descendant"] is False


def test_preflight_stopped_before_every_metric_score_and_other_domain_access() -> None:
    frozen = config()
    chronology = frozen["chronology"]
    boundary = frozen["data_boundary"]
    assert chronology["real_metric_definitions_after_access"] == 0
    assert chronology["real_thresholds_after_access"] == 0
    assert chronology["real_candidates_selected_after_access"] == 0
    assert chronology["real_candidate_scores_after_access"] == 0
    assert boundary["sparc_real_candidate_score_calls"] == 0
    assert boundary["xcop_target_rows_read"] == 0
    assert boundary["xcop_target_files_opened"] == 0
    assert boundary["group_rows_read"] == 0
    assert boundary["lensing_rows_read"] == 0
    assert boundary["inferred_total_mass_rows_read"] == 0
    assert boundary["network_calls"] == 0
    assert boundary["model_calls"] == 0
    assert boundary["paid_calls"] == 0


def test_recovery_options_are_frozen_but_unauthorized_and_unexecuted() -> None:
    recovery = config()["recovery_options"]
    assert [row["option_id"] for row in recovery] == [
        "RECLASSIFY_LOCAL_SPARC_DEVELOPMENT_ONLY",
        "OBTAIN_GENUINELY_EXTERNAL_CONFIRMATION",
    ]
    assert all(row["authorized_here"] is False for row in recovery)
    assert all(row["executed_here"] is False for row in recovery)


@pytest.mark.parametrize(
    ("section", "field", "replacement"),
    [
        ("access_incident", "local_sparc_confirmation_remains_sealed_for_this_descendant", True),
        ("access_incident", "exact_confirmation_raw_rows_surfaced", 23),
        ("chronology", "real_candidate_scores_after_access", 1),
        ("data_boundary", "xcop_target_rows_read", 1),
        ("claim_boundary", "scientific_claim_allowed", True),
        ("frozen_candidate_registry", "equivalence_class_count", 59),
    ],
)
def test_critical_contract_mutations_fail_closed(
    section: str, field: str, replacement: object
) -> None:
    mutated = copy.deepcopy(config())
    mutated[section][field] = replacement
    with pytest.raises(preflight.BENRealDevelopmentPreflightError, match=f"frozen {section}"):
        preflight.validate_contract(mutated)


def test_source_binding_mutation_fails_closed() -> None:
    mutated = copy.deepcopy(config())
    mutated["source_bindings"]["synthetic_receipt"]["file_sha256"] = "0" * 64
    with pytest.raises(preflight.BENRealDevelopmentPreflightError, match="source_bindings"):
        preflight.validate_contract(mutated)


def test_verifier_binding_mutation_fails_closed() -> None:
    mutated = copy.deepcopy(config())
    mutated["verifier_test"]["file_sha256"] = "0" * 64
    with pytest.raises(preflight.BENRealDevelopmentPreflightError, match="verifier_test"):
        preflight.validate_contract(mutated)


def test_access_ledger_has_no_scientific_result() -> None:
    ledger = preflight.expected_access_ledger(config())
    assert ledger["decision"] == preflight.DECISION
    assert ledger["consequence"] == {
        "real_evaluation_aborted_before_metric_or_score": True,
        "local_sparc_reserved_confirmation_cannot_be_claimed_sealed_for_this_descendant": True,
        "candidate_registry_remains_the_committed_240_raw_60_equivalence_classes": True,
        "no_candidate_failure_or_success_exists_to_retain": True,
    }
    claimed = ledger["content_sha256"]
    assert claimed == preflight.content_sha256(
        {key: value for key, value in ledger.items() if key != "content_sha256"}
    )


def test_atomic_writer_refuses_overwrite_and_preserves_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(preflight, "ROOT", tmp_path)
    target = tmp_path / "ledger.json"
    preflight.write_json_no_clobber(target, {"first": True})
    original = target.read_bytes()
    with pytest.raises(preflight.BENRealDevelopmentPreflightError, match="no-clobber"):
        preflight.write_json_no_clobber(target, {"first": False})
    assert target.read_bytes() == original
    assert json.loads(original) == {"first": True}


def test_path_escape_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(preflight.BENRealDevelopmentPreflightError, match="escaped"):
        preflight.confined(tmp_path / "outside.json")


def test_frozen_receipt_reconstructs_exactly() -> None:
    status = preflight.check(
        preflight.ROOT / preflight.CONFIG_PATH,
        config_sha256(),
        preflight.ROOT / preflight.RECEIPT_PATH,
    )
    assert status["valid"] is True
    assert status["decision"] == preflight.DECISION
    assert status["real_evaluation_executed"] is False
    assert status["real_candidates_scored"] == 0
    assert status["local_sparc_confirmation_sealed_for_descendant"] is False
    assert status["xcop_target_rows_read"] == 0
    assert status["scientific_claim_allowed"] is False
