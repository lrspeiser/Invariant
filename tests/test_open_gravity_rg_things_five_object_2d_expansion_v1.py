from __future__ import annotations

import copy
from pathlib import Path

import pytest

from sigma_theory_compiler import open_gravity_rg_things_five_object_2d_expansion_v1 as expansion


@pytest.fixture(scope="module")
def config() -> dict:
    return expansion.load_config(verify_package=False)


@pytest.fixture(scope="module")
def receipt(config: dict) -> dict:
    return expansion.build_receipt(config)


def test_full_set_and_new_blind_subset_are_frozen(config: dict) -> None:
    selection = config["selection_contract"]
    assert selection["full_source_ready_set"] == list(expansion._OBJECTS)
    assert selection["new_response_blind_objects_at_original_freeze"] == list(
        expansion._NEW_OBJECTS
    )
    assert selection["selection_used_new_response_pixels"] is False
    assert selection["one_failure_never_prunes_family"] is True
    repair = config["failed_run_and_numerical_repair"]
    assert repair["failed_run_response_pixels_were_decoded"] is True
    assert repair["failed_run_receipt_was_not_written"] is True
    assert repair["repair_pcg_relative_tolerance"] == 1e-10
    assert repair["repair_pcg_max_iterations"] == 400
    assert repair["physical_parameters_changed"] is False
    assert repair["decision_rule_changed"] is False


def test_every_builder_admission_dimension_is_bound(config: dict) -> None:
    admission = config["source_admission"]
    assert admission["real_public_source_and_response_data_required"] is True
    assert admission["primary_measurement_and_method_papers_required"] is True
    assert admission["target_free_operator_projection_and_rar_benchmarks_required"] is True
    assert admission["newtonian_and_rar_controls_required"] is True
    assert admission["missing_source_disposition"] == "SOURCE_BLOCKED"
    assert admission["paper_only_disposition"] == "THEORY_BENCHMARK_ONLY"
    assert admission["model_lifted_vertical_structure_disposition"] == "MODEL_LIFTED_2P5D"
    assert admission["general_3d_validation_allowed"] is False


def test_new_response_headers_and_bytes_are_exact_without_pixel_decode(config: dict) -> None:
    assert len(config["new_response_files"]) == 6
    assert sum(row["bytes"] for row in config["new_response_files"]) == 25_655_040
    for row in config["new_response_files"]:
        header = expansion._validate_response_header(row)
        assert int(header["NAXIS1"]) == 1024
        assert int(header["NAXIS2"]) == 1024
        assert header["BUNIT"] == "METR/SEC"


def test_incidental_wrong_archive_object_is_excluded(config: dict) -> None:
    incident = config["incidental_archive_access"]
    assert incident["object_id"] == "NGC5055"
    assert incident["scientific_pixels_decoded"] == 0
    assert incident["used_in_campaign"] is False
    assert "NGC5055" not in config["selection_contract"]["full_source_ready_set"]


def test_predecessor_bindings_and_rar_benchmarks_pass(config: dict) -> None:
    receipts = expansion._validate_bindings(config)
    assert set(receipts) == {
        "FIVE_OBJECT_REAL_SOURCE_BUILDER",
        "SEALED_MATCHED_PAIR_COMPARATOR_DIAGNOSTICS",
    }
    assert expansion.diagnostics.rar_target_free_benchmarks()["all_pass"] is True


def test_complete_five_object_real_score_and_exact_decision(receipt: dict) -> None:
    assert receipt["status"] == "PASS_COMPLETE_FIVE_OBJECT_REAL_THINGS_2D_EXPANSION"
    assert [row["object_id"] for row in receipt["objects"]] == list(expansion._OBJECTS)
    assert len(receipt["new_object_details"]) == 3
    assert {row["object_id"] for row in receipt["new_object_details"]} == set(
        expansion._NEW_OBJECTS
    )
    for row in receipt["objects"]:
        assert row["common_pixel_count"] > 0
        assert set(row["models"]) == set(expansion._MODELS)
        assert isinstance(row["rg_broader_signal_object_gate"], bool)
    gate = receipt["expansion_gate"]
    expected = gate["signal_object_count"] >= 3 and gate["new_blind_signal_object_count"] >= 2
    assert gate["broader_development_signal"] is expected
    assert gate["confirmation"] is False


def test_new_object_solver_and_response_evidence_are_retained(receipt: dict) -> None:
    for row in receipt["new_object_details"]:
        assert row["result_source"] == "NEW_RESPONSE_BLIND_AT_FREEZE"
        assert row["beam_equivalent_count"] > 0.0
        assert row["rotation_sign"] in (-1.0, 1.0)
        assert row["solver"]["fine_newton_residual"] < 1.0e-8
        assert row["solver"]["fine_rg_residual"] < 1.0e-8
        assert row["solver"]["fine_source_mass_relative_error"] < 2.0e-9
        assert row["maximum_common_tangential_ratio"] >= 0.0


def test_claim_ceiling_remains_restrictive(receipt: dict) -> None:
    claims = receipt["claim_boundary"]
    assert claims["five_object_real_response_scored"] is True
    assert claims["broader_development_signal"] is False
    assert claims["preregistered_confirmation"] is False
    assert claims["general_3d_validated"] is False
    assert claims["unique_theory_established"] is False
    assert claims["publication_candidate"] is False
    assert claims["publication_ready"] is False


def test_receipt_is_deterministically_self_hashed(receipt: dict) -> None:
    assert receipt["content_sha256"] == expansion.content_sha256({**receipt, "content_sha256": ""})


def test_config_mutations_fail_closed(config: dict) -> None:
    for path, value in (
        (("status",), "CONFIRMED"),
        (("selection_contract", "new_response_blind_objects_at_original_freeze"), ["NGC5055"]),
        (("selection_contract", "selection_used_new_response_pixels"), True),
        (("source_admission", "general_3d_validation_allowed"), True),
        (("incidental_archive_access", "used_in_campaign"), True),
        (("failed_run_and_numerical_repair", "physical_parameters_changed"), True),
        (("fixed_model_contract", "rar_g_dagger_m_s2"), 1.0e-10),
        (("expansion_decision_rule", "minimum_objects_for_broader_signal"), 1),
        (("claim_boundary", "publication_candidate"), True),
    ):
        mutated = copy.deepcopy(config)
        target = mutated
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        with pytest.raises(expansion.FiveObjectExpansionError):
            expansion.validate_config(mutated)


def test_receipt_mutation_fails(config: dict, receipt: dict) -> None:
    mutated = copy.deepcopy(receipt)
    mutated["claim_boundary"]["unique_theory_established"] = True
    mutated["content_sha256"] = expansion.content_sha256({**mutated, "content_sha256": ""})
    with pytest.raises(expansion.FiveObjectExpansionError):
        expansion.validate_receipt_payload(config, mutated, receipt)


def test_atomic_no_clobber(tmp_path: Path) -> None:
    output = tmp_path / "receipt.json"
    assert expansion._atomic_no_clobber(output, b"one\n") == "CREATED"
    assert expansion._atomic_no_clobber(output, b"one\n") == "EXISTING_IDENTICAL"
    with pytest.raises(expansion.FiveObjectExpansionError):
        expansion._atomic_no_clobber(output, b"two\n")
