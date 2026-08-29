from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler import (
    gravity_shared_target_blind_ben_real_development_preflight_v2 as preflight,
)


def config() -> dict:
    return preflight.load_config()


def test_every_local_sparc_row_is_development_only_but_score_subset_is_fixed() -> None:
    frozen = config()
    reclassified = frozen["sparc_reclassification"]
    assert (
        reclassified["all_rows_in_locally_accessible_mixed_packet_role"]
        == "development_only_for_this_descendant"
    )
    assert reclassified["locally_accessible_published_rows_metadata_count"] == 3391
    assert reclassified["local_confirmation_role_exists"] is False
    assert reclassified["local_confirmation_claim_allowed"] is False
    assert reclassified["historical_confirmation_names_may_be_used_as_confirmation"] is False
    assert reclassified["planned_score_subset_objects"] == 139
    assert reclassified["planned_score_subset_rows"] == 2720
    assert reclassified["rows_outside_planned_score_subset_scored"] == 0


def test_committed_registry_and_ef466_incident_are_exactly_bound() -> None:
    frozen = config()
    registry = frozen["candidate_registry"]
    incident = frozen["source_bindings"]["incident_receipt"]
    assert registry["raw_candidate_count"] == 240
    assert registry["canonical_equivalence_class_count"] == 60
    assert registry["content_sha256"] == (
        "45966eae73d7641ea982a7eea47aad883a9ff344baf121b91b901c32ef819f19"
    )
    assert registry["score_raw_equivalent_duplicates"] is False
    assert incident["commit"] == "ef466cb09be39481302b5247488036b6e25d3d3a"
    assert incident["git_blob_sha1"] == "c9eda276a818ae2030a847ad0a1c8ea6eb3fa694"
    assert incident["file_sha256"] == (
        "a3d42d0008deaee2fd1bfa4f8def84aa6425ff98425ffc464199b376034ae5fa"
    )


def test_item61_metric_comparators_object_units_and_no_single_veto_are_reused() -> None:
    metric = config()["item61_metric_reuse"]
    assert metric["primary_metric_id"] == ("equal_object_mean_squared_standardized_residual")
    assert metric["primary_metric_equation"] == (
        "mean_over_objects(mean_over_rows(((prediction-observed)/sigma)^2))"
    )
    assert metric["comparators"] == ["newtonian_baryons", "empirical_rar"]
    assert metric["row_level_pooling_for_selection"] is False
    assert metric["single_counterexample_terminal"] is False
    assert metric["counterexample_count_alone_terminal"] is False
    assert metric["finite_sample_may_prune_formula_family"] is False
    assert metric["numeric_improvement_threshold"] is None
    assert metric["threshold_tuning_from_rows_allowed"] is False
    assert "No new folds" in metric["object_fold_contract"]


def test_mapping_is_predictor_only_and_xcop_output_blocks_before_payload() -> None:
    mapping = config()["predictor_only_mapping"]
    assert mapping["sparc"]["response_fields_used_in_mapping"] == []
    assert mapping["xcop"]["response_fields_used_in_input_mapping"] == []
    assert mapping["xcop"]["raw_predictor_fields"] == ["RW_X", "NE"]
    assert mapping["xcop"]["candidate_output_projection"] is None
    assert mapping["xcop"]["output_mapping_ready"] is False
    assert "outer SZ pressure boundary" in mapping["xcop"]["blocker"]
    assert "P500/T500" in mapping["xcop"]["blocker"]
    assert mapping["xcop"]["failure_action"] == ("BLOCK_BEFORE_PAYLOAD_LOAD_AND_SCORE")
    assert mapping["response_used_for_candidate_generation"] is False


def test_only_eight_xcop_development_clusters_are_in_scope() -> None:
    populations = config()["planned_populations"]
    assert populations["xcop"]["objects"] == preflight.EXPECTED_XCOP_OBJECTS
    assert populations["xcop"]["role"] == "already_exposed_development_only"
    assert populations["forbidden"]["xcop_item59_holdout_objects"] == [
        "A2029",
        "A3158",
        "A644",
        "RXC1825",
    ]
    assert populations["forbidden"]["little_things"] is True
    assert populations["forbidden"]["groups"] is True
    assert populations["forbidden"]["lensing"] is True


def test_candidate_selection_and_ablations_are_frozen_without_row_tuning() -> None:
    selection = config()["selection_and_ablation"]
    assert selection["full_candidates_per_domain"] == 60
    assert selection["fixed_ablations_per_candidate"] == [
        "N_zero_ablation",
        "B_unity_gate_ablation",
        "A_off_nuisance_ablation",
    ]
    assert selection["comparators_per_domain"] == [
        "newtonian_baryons",
        "empirical_rar",
    ]
    assert selection["post_response_candidate_generation_allowed"] is False
    assert selection["post_response_formula_repair_allowed"] is False
    assert selection["single_object_veto_allowed"] is False
    assert "ascending class_id first" in selection["selection_rule"]


def test_compute_ceiling_is_exact_and_zero_cost() -> None:
    ceiling = config()["compute_ceiling"]
    assert ceiling["ablation_variants"] == 3 * 60
    assert ceiling["formula_domain_batches_per_backend"] == (60 + 180) * 2 + 4
    assert ceiling["sparc_formula_row_cells_per_backend"] == 242 * 2720
    assert ceiling["xcop_formula_row_cells_per_backend"] == 242 * 521
    assert ceiling["total_formula_row_cells_per_backend"] == 784_322
    assert ceiling["total_formula_row_cells_both_backends"] == 1_568_644
    assert ceiling["maximum_object_score_reductions"] == 242 * (139 + 8)
    assert ceiling["maximum_response_row_score_terms"] == 242 * (2720 + 184)
    assert ceiling["cpu_formula_domain_batches"] == 484
    assert ceiling["gpu_formula_domain_batches"] == 484
    assert ceiling["cpu_gpu_parity_comparisons"] == 484
    assert ceiling["threshold_tuning_calls"] == 0
    assert ceiling["model_calls"] == 0
    assert ceiling["paid_calls"] == 0
    assert ceiling["maximum_api_spend_usd"] == 0.0


def test_zero_access_chronology_is_exact() -> None:
    chronology = config()["zero_access_chronology"]
    assert chronology["v2_contract_frozen_before_payload_access"] is True
    assert chronology["authorization_artifacts_accepted"] == 0
    for key, value in chronology.items():
        if key != "v2_contract_frozen_before_payload_access":
            assert value == 0


def test_current_manifest_is_unauthorized_and_grants_nothing() -> None:
    frozen = config()
    authorization = preflight.read_json(preflight.ROOT / preflight.AUTHORIZATION_PATH)
    preflight.validate_authorization(authorization, frozen, require_authorized=False)
    assert authorization["authorized"] is False
    assert authorization["sparc_objects"] == 0
    assert authorization["sparc_rows"] == 0
    assert authorization["xcop_objects"] == []
    assert all(value == 0 or value == 0.0 for value in authorization["compute_ceiling"].values())
    with pytest.raises(
        preflight.BENRealDevelopmentPreflightV2Error,
        match="UNAUTHORIZED_BEFORE_PAYLOAD_LOAD",
    ):
        preflight.production_preflight(preflight.ROOT / preflight.AUTHORIZATION_PATH)


def test_authorization_cannot_override_mapping_blocker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frozen = config()
    receipt = preflight.read_json(preflight.ROOT / preflight.RECEIPT_PATH)
    authorized = {
        "schema_version": preflight.AUTHORIZATION_SCHEMA,
        "authorization_id": "future-explicit-approval",
        "authorized": True,
        "approved_by": "future-approver",
        "approved_at": "2099-01-01T00:00:00Z",
        "config_file_sha256": preflight.CONFIG_FILE_SHA256,
        "preflight_receipt_file_sha256": preflight.file_sha256(
            preflight.ROOT / preflight.RECEIPT_PATH
        ),
        "preflight_receipt_content_sha256": receipt["content_sha256"],
        "incident_commit": frozen["lineage"]["incident_commit"],
        "candidate_registry_content_sha256": frozen["candidate_registry"]["content_sha256"],
        "sparc_role": "development_only",
        "sparc_objects": 139,
        "sparc_rows": 2720,
        "xcop_role": "development_only",
        "xcop_objects": preflight.EXPECTED_XCOP_OBJECTS,
        "compute_ceiling": frozen["compute_ceiling"],
        "mapping_resolution_contract": {"successor_contract_required": True},
        "claim_acknowledgements": preflight.EXPECTED_ACKNOWLEDGEMENTS,
    }
    preflight.validate_authorization(authorized, frozen, require_authorized=True)
    original = preflight.read_json

    def fake_read(path: Path) -> dict:
        if path == Path("authorized-v2.json"):
            return authorized
        return original(path)

    monkeypatch.setattr(preflight, "read_json", fake_read)
    with pytest.raises(
        preflight.BENRealDevelopmentPreflightV2Error,
        match="BLOCKED_XCOP_RESPONSE_DERIVED_OUTPUT_MAPPING_BEFORE_PAYLOAD_LOAD",
    ):
        preflight.production_preflight(Path("authorized-v2.json"))


@pytest.mark.parametrize(
    "section",
    [
        "source_bindings",
        "sparc_reclassification",
        "item61_metric_reuse",
        "predictor_only_mapping",
        "selection_and_ablation",
        "compute_ceiling",
        "zero_access_chronology",
        "approval_schema",
        "claim_boundary",
    ],
)
def test_section_mutations_fail_closed(section: str) -> None:
    mutated = copy.deepcopy(config())
    mutated[section][next(iter(mutated[section]))] = "tampered"
    with pytest.raises(preflight.BENRealDevelopmentPreflightV2Error, match=f"frozen {section}"):
        preflight.validate_config(mutated)


def test_authorization_mutation_fails_closed() -> None:
    frozen = config()
    authorization = preflight.read_json(preflight.ROOT / preflight.AUTHORIZATION_PATH)
    authorization["sparc_rows"] = 1
    with pytest.raises(preflight.BENRealDevelopmentPreflightV2Error):
        preflight.validate_authorization(authorization, frozen, require_authorized=False)


def test_atomic_writer_refuses_overwrite_and_preserves_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(preflight, "ROOT", tmp_path)
    target = tmp_path / "receipt.json"
    preflight.write_json_no_clobber(target, {"first": True})
    original = target.read_bytes()
    with pytest.raises(preflight.BENRealDevelopmentPreflightV2Error, match="no-clobber"):
        preflight.write_json_no_clobber(target, {"first": False})
    assert target.read_bytes() == original
    assert json.loads(original) == {"first": True}


def test_v2_source_has_no_payload_or_scoring_loader() -> None:
    source = (
        preflight.ROOT
        / "src/sigma_theory_compiler/gravity_shared_target_blind_ben_real_development_preflight_v2.py"
    ).read_text(encoding="utf-8")
    for forbidden in (
        "fits.open",
        "import astropy",
        "import numpy",
        "import pandas",
        'sparc_rotation_curves_full_v1.json").read',
        "Vobs",
        "P_SZ",
        "T_X",
    ):
        assert forbidden not in source
    assert "payload_loader_present_in_v2" in source


def test_frozen_receipt_reconstructs_exactly() -> None:
    result = preflight.check()
    assert result["valid"] is True
    assert result["decision"] == preflight.DECISION
    assert result["authorized"] is False
    assert result["mapping_ready"] is False
    assert result["real_candidates_scored"] == 0
    assert result["payload_rows_read"] == 0
