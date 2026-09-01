from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from sigma_theory_compiler import (
    open_gravity_rg_things_matched_pair_2d_comparator_diagnostics_v1 as diagnostics,
)


@pytest.fixture(scope="module")
def config() -> dict:
    return diagnostics.load_config(verify_package=False)


@pytest.fixture(scope="module")
def receipt(config: dict) -> dict:
    return diagnostics.build_receipt(config)


def test_post_score_exploratory_disclosure_is_explicit(config: dict) -> None:
    audit = config["audit_disclosure"]
    assert audit["primary_newton_vs_rg_pixel_score_was_opened_before_this_packet"] is True
    assert audit["rar_formula_and_gdagger_are_published_fixed_values_not_fit_here"] is True
    assert audit["all_sensitivity_cells_are_exploratory_not_preregistered_confirmation"] is True
    assert audit["no_cell_may_replace_or erase_the_sealed_primary_result"] is True
    assert audit["one_failure_never_prunes_family"] is True


def test_source_paper_benchmark_and_control_admission(config: dict) -> None:
    admission = config["source_admission"]
    assert admission["real_public_sources_and_responses_bound_by_predecessor"] is True
    assert admission["rar_primary_paper_bound"] is True
    assert admission["target_free_operator_and_projection_benchmarks_bound_by_predecessor"] is True
    assert admission["newtonian_control_required"] is True
    assert admission["missing_source_disposition"] == "SOURCE_BLOCKED"
    assert admission["paper_only_disposition"] == "THEORY_BENCHMARK_ONLY"
    assert admission["model_lifted_vertical_structure_disposition"] == "MODEL_LIFTED_2P5D"
    assert admission["general_3d_validation_allowed"] is False


def test_sealed_primary_score_is_bound_exactly(config: dict) -> None:
    receipt = diagnostics._load_score_evidence(config)
    primary = next(row for row in receipt["objects"] if row["object_id"] == "NGC2976")
    assert primary["rg_fractional_rmse_improvement_over_newton"] == pytest.approx(
        0.1533607170491626
    )


def test_rar_target_free_limits_pass() -> None:
    benchmark = diagnostics.rar_target_free_benchmarks()
    assert benchmark["all_pass"] is True
    assert all(benchmark["checks"].values())


def test_rar_vector_preserves_direction_and_zero() -> None:
    gx = np.asarray([[3.0, 0.0]])
    gy = np.asarray([[4.0, 0.0]])
    out_x, out_y = diagnostics.rar_vector_field((gx, gy), a0_m_s2=1.2e-10, g_dagger_m_s2=1.2e-10)
    assert float(out_x[0, 0] * gy[0, 0] - out_y[0, 0] * gx[0, 0]) == pytest.approx(0.0, abs=1.0e-14)
    assert out_x[0, 1] == 0.0
    assert out_y[0, 1] == 0.0


def test_real_rar_comparator_and_all_diagnostic_cells_complete(receipt: dict) -> None:
    assert receipt["status"] == ("PASS_POST_PRIMARY_SCORE_RAR_AND_SOURCE_SENSITIVITY_DIAGNOSTICS")
    assert len(receipt["objects"]) == 2
    for row in receipt["objects"]:
        assert row["common_pixel_count"] > 0
        assert set(row["all_pixel_metrics"]) == set(diagnostics._MODELS)
        assert row["sensitivity_cell_count"] == 15
        assert len(row["sensitivity_cells"]) == 15
        assert row["rar_solver"]["maximum_parent_solver_residual"] < 1.0e-8
        assert row["rar_solver"]["converged_pixels"] > 0
        for metrics in row["all_pixel_metrics"].values():
            assert metrics is not None
            assert metrics["rmse_m_s"] > 0.0


def test_diagnostics_do_not_promote_theory_claims(receipt: dict) -> None:
    assert receipt["diagnostic_summary"]["preregistered_confirmation"] is False
    assert receipt["scientific_boundary"]["post_score_exploratory"] is True
    assert receipt["scientific_boundary"]["general_3d_validated"] is False
    assert receipt["claim_boundary"]["unique_theory_established"] is False
    assert receipt["claim_boundary"]["publication_candidate"] is False
    assert receipt["claim_boundary"]["publication_ready"] is False


def test_receipt_is_deterministically_self_hashed(receipt: dict) -> None:
    assert receipt["content_sha256"] == diagnostics.content_sha256(
        {**receipt, "content_sha256": ""}
    )


def test_config_mutations_fail_closed(config: dict) -> None:
    for path, value in (
        (("status",), "CONFIRMED"),
        (
            ("audit_disclosure", "primary_newton_vs_rg_pixel_score_was_opened_before_this_packet"),
            False,
        ),
        (("source_admission", "general_3d_validation_allowed"), True),
        (("rar_comparator", "g_dagger_m_s2"), 1.0e-10),
        (("rar_comparator", "parameter_fit_to_things"), True),
        (("common_score_contract", "response_tuning_calls"), 1),
        (("exploratory_sensitivity_grid", "subcell_systemic_offsets_refit"), True),
        (("claim_boundary", "publication_candidate"), True),
    ):
        mutated = copy.deepcopy(config)
        target = mutated
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        with pytest.raises(diagnostics.ComparatorDiagnosticError):
            diagnostics.validate_config(mutated)


def test_receipt_mutation_fails(config: dict, receipt: dict) -> None:
    mutated = copy.deepcopy(receipt)
    mutated["claim_boundary"]["refracted_gravity_beats_known_family"] = True
    mutated["content_sha256"] = diagnostics.content_sha256({**mutated, "content_sha256": ""})
    with pytest.raises(diagnostics.ComparatorDiagnosticError):
        diagnostics.validate_receipt_payload(config, mutated, receipt)


def test_atomic_no_clobber(tmp_path: Path) -> None:
    output = tmp_path / "receipt.json"
    assert diagnostics._atomic_no_clobber(output, b"one\n") == "CREATED"
    assert diagnostics._atomic_no_clobber(output, b"one\n") == "EXISTING_IDENTICAL"
    with pytest.raises(diagnostics.ComparatorDiagnosticError):
        diagnostics._atomic_no_clobber(output, b"two\n")
