from __future__ import annotations

import copy
import csv
import io
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from sigma_theory_compiler import gravity_cluster_manuscript_renderer as renderer

ROOT = Path(__file__).resolve().parents[1]


def test_frozen_inventory_defines_and_renders_every_primary_artifact() -> None:
    config = renderer.load_config(ROOT)
    artifacts = renderer.build_artifacts(ROOT)
    expected = {row["filename"] for row in config["primary_tables"] + config["primary_figures"]}
    assert set(artifacts) == expected
    assert len(artifacts) == 13
    assert tuple(row["artifact_id"] for row in config["primary_tables"]) == renderer.TABLE_IDS
    assert tuple(row["artifact_id"] for row in config["primary_figures"]) == renderer.FIGURE_IDS


def test_tables_are_parseable_nonempty_csv_with_expected_scientific_coverage() -> None:
    artifacts = renderer.build_artifacts(ROOT)
    parsed = {
        filename: list(csv.reader(io.StringIO(value.decode("utf-8"))))
        for filename, value in artifacts.items()
        if filename.endswith(".csv")
    }
    assert len(parsed) == 7
    assert all(len(rows) > 2 and len({len(row) for row in rows}) == 1 for rows in parsed.values())
    table_1 = parsed["table-1-candidate-and-claims.csv"]
    assert any("g(r)=g_bar(r)+1.5" in cell for row in table_1 for cell in row)
    table_2 = parsed["table-2-split-performance.csv"]
    assert {row[0] for row in table_2[1:]} == {
        "development_train",
        "development_holdout",
        "confirmation",
    }
    table_3 = parsed["table-3-comparators-and-ablations.csv"]
    assert {row[0] for row in table_3[1:]} == {"candidate", "comparator", "ablation"}
    table_4 = parsed["table-4-object-performance.csv"]
    assert len(table_4) == 21
    table_5 = parsed["table-5-robustness-and-controls.csv"]
    assert any(
        row[:3] == ["quotient_sbc", "v3_synthetic_sbc_passed", "true"] for row in table_5[1:]
    )
    assert any(
        row[:3]
        == [
            "pressure_covariance",
            "scoring_decision",
            "FAIL_FROZEN_PRESSURE_RANKING_ROBUSTNESS",
        ]
        for row in table_5[1:]
    )
    assert any(row[:3] == ["shared_ben_synthetic", "raw_candidates", "240"] for row in table_5[1:])
    assert any(
        row[:3] == ["shared_ben_real", "real_scoring_executed", "false"] for row in table_5[1:]
    )
    assert any(row[:3] == ["shared_ben_v4", "canonical_full_classes", "60"] for row in table_5[1:])
    assert any(row[:3] == ["shared_ben_v4", "production_executed", "false"] for row in table_5[1:])
    assert any(
        row[:3] == ["cluster_strata", "candidate_absolute_gate_passed", "false"]
        for row in table_5[1:]
    )
    assert any(
        row[:3] == ["cluster_strata", "candidate_object_win_gate_passed", "false"]
        for row in table_5[1:]
    )
    assert any(
        row[:3] == ["xcop_shape_bridge", "real_scoring_executed", "false"] for row in table_5[1:]
    )
    assert any(
        row[:3] == ["missing_variables", "continuous_measurement_ready_rows", "0"]
        for row in table_5[1:]
    )
    assert any(
        row[:3] == ["act_erass_overlap", "population_gate_evaluated", "false"]
        for row in table_5[1:]
    )
    assert any(row[:3] == ["act_erass_executor", "authorized", "false"] for row in table_5[1:])
    assert any(row[:3] == ["act_erass_executor", "catalog_rows_opened", "0"] for row in table_5[1:])
    assert any(row[:3] == ["group_source_v3", "ready_science_lanes", "0"] for row in table_5[1:])
    assert any(
        row[:3] == ["xclass_identity_executor", "authorized", "false"] for row in table_5[1:]
    )
    assert any(
        row[:3] == ["xclass_identity_executor", "identity_rows_opened", "0"] for row in table_5[1:]
    )
    assert any(
        row[:3] == ["matter_lensing_theory", "template_level_gates_passed", "1"]
        for row in table_5[1:]
    )
    assert any(
        row[:3] == ["matter_lensing_symbolic", "full_H2_passed", "false"] for row in table_5[1:]
    )
    assert any(
        row[:3]
        == [
            "matter_lensing_external_symbol",
            "designed_u_above_one_third_failure_preserved",
            "true",
        ]
        for row in table_5[1:]
    )
    assert any(
        row[:3]
        == [
            "matter_lensing_source_bound",
            "restricted_static_source_bound_established",
            "true",
        ]
        for row in table_5[1:]
    )
    assert any(
        row[:3]
        == [
            "matter_lensing_conformal_source",
            "universal_conformal_source_identity_established",
            "true",
        ]
        for row in table_5[1:]
    )
    assert any(
        row[:3] == ["matter_lensing_solar_gw", "necessary_conditions_established", "true"]
        for row in table_5[1:]
    )
    assert any(
        row[:3] == ["matter_lensing_flrw", "restricted_flat_flrw_equations_established", "true"]
        for row in table_5[1:]
    )
    assert any(
        row[:3]
        == [
            "matter_lensing_covariant",
            "scalar_stress_and_exchange_established",
            "true",
        ]
        for row in table_5[1:]
    )
    assert any(
        row[:3] == ["matter_lensing_adm_constraints", "CP11_3_complete", "true"]
        for row in table_5[1:]
    )
    assert any(
        row[:3]
        == [
            "matter_lensing_scalar_hamiltonian",
            "restricted_scalar_canonical_hamiltonian_derived",
            "true",
        ]
        for row in table_5[1:]
    )
    assert any(
        row[:3]
        == [
            "matter_lensing_deep_aqual_transition",
            "conditional_exact_transition_no_go_established",
            "true",
        ]
        for row in table_5[1:]
    )
    assert any(
        row[:3]
        == [
            "shared_formula_scalar_kinetic_reconstruction",
            "formula_to_minimal_kinetic_map_derived",
            "true",
        ]
        for row in table_5[1:]
    )
    assert any(
        row[:3]
        == [
            "shared_quadrature_covariant_action",
            "restricted_quadrature_action_embedding_established",
            "true",
        ]
        for row in table_5[1:]
    )
    assert any(
        row[:3]
        == [
            "shared_quadrature_lensing_backreaction",
            "restricted_exterior_lensing_backreaction_derived",
            "true",
        ]
        for row in table_5[1:]
    )
    assert any(
        row[:3]
        == [
            "shared_quadrature_universal_vector_metric",
            "restricted_same_metric_motion_lensing_architecture",
            "true",
        ]
        for row in table_5[1:]
    )
    assert any(
        row[:3]
        == [
            "shared_quadrature_aether_mode_conditions",
            "finite_positive_all_luminal_pure_aether_locus",
            "true",
        ]
        for row in table_5[1:]
    )
    assert any(
        row[:3]
        == [
            "shared_quadrature_reduced_principal_factorization",
            "static_branch_six_mode_reduced_factorization",
            "true",
        ]
        for row in table_5[1:]
    )
    assert any(
        row[:3]
        == [
            "matter_lensing_kinetic_gate",
            "conditional_timelike_mixing_no_go",
            "PASS_MACHINE_DERIVED_UNDER_FROZEN_HYPOTHESES",
        ]
        for row in table_5[1:]
    )
    table_6 = parsed["table-6-access-claims-limitations.csv"]
    assert any(
        row
        == [
            "sampler_calibration_boundary",
            "candidate_production_unlock",
            "false",
        ]
        for row in table_6[1:]
    )
    assert any(row == ["group_scale_source_boundary", "ready_lanes", "0"] for row in table_6[1:])
    assert any(
        row == ["cluster_strata_boundary", "CP5_13_complete", "false"] for row in table_6[1:]
    )
    assert any(
        row
        == [
            "shared_ben_boundary",
            "xcop_predictor_output_mapping_ready",
            "false",
        ]
        for row in table_6[1:]
    )
    assert any(
        row == ["shared_ben_v4_boundary", "registered_ablations", "180"] for row in table_6[1:]
    )
    assert any(row == ["shared_ben_v4_boundary", "scores_computed", "0"] for row in table_6[1:])
    assert any(
        row
        == [
            "shape_missing_variable_boundary",
            "continuous_measurement_ready_rows",
            "0",
        ]
        for row in table_6[1:]
    )
    assert any(
        row == ["group_act_acquisition_boundary", "act_catalog_rows_opened", "0"]
        for row in table_6[1:]
    )
    assert any(
        row == ["group_act_acquisition_boundary", "act_population_gate_evaluated", "false"]
        for row in table_6[1:]
    )
    assert any(
        row == ["group_act_acquisition_boundary", "act_executor_authorized", "false"]
        for row in table_6[1:]
    )
    assert any(
        row == ["group_act_acquisition_boundary", "act_executor_catalog_rows_opened", "0"]
        for row in table_6[1:]
    )
    assert any(row == ["group_scale_source_boundary", "v3_ready_lanes", "0"] for row in table_6[1:])
    assert any(
        row == ["group_act_acquisition_boundary", "xclass_executor_identity_rows", "0"]
        for row in table_6[1:]
    )
    assert any(
        row == ["matter_lensing_theory_boundary", "full_H2_passed", "false"] for row in table_6[1:]
    )
    assert any(
        row == ["matter_lensing_theory_boundary", "full_H3_passed", "false"] for row in table_6[1:]
    )
    assert any(
        row
        == [
            "matter_lensing_theory_boundary",
            "designed_u_above_one_third_failure_preserved",
            "true",
        ]
        for row in table_6[1:]
    )
    assert any(
        row
        == [
            "matter_lensing_theory_boundary",
            "physical_source_law_established",
            "false",
        ]
        for row in table_6[1:]
    )
    assert any(
        row == ["matter_lensing_theory_boundary", "solar_gate_passed", "false"]
        for row in table_6[1:]
    )
    assert any(
        row
        == [
            "matter_lensing_theory_boundary",
            "healthy_late_time_history_exists",
            "false",
        ]
        for row in table_6[1:]
    )
    assert any(
        row
        == [
            "matter_lensing_theory_boundary",
            "covariant_ADM_constraints_derived",
            "false",
        ]
        for row in table_6[1:]
    )
    assert any(
        row == ["matter_lensing_theory_boundary", "CP11_3_complete", "true"] for row in table_6[1:]
    )
    assert any(
        row
        == [
            "matter_lensing_theory_boundary",
            "full_metric_scalar_matter_system_strongly_hyperbolic",
            "false",
        ]
        for row in table_6[1:]
    )
    assert any(
        row
        == [
            "matter_lensing_theory_boundary",
            "physical_hamiltonian_positive",
            "false",
        ]
        for row in table_6[1:]
    )
    assert any(
        row
        == [
            "matter_lensing_theory_boundary",
            "scalar_hamiltonian_CP11_4_complete",
            "false",
        ]
        for row in table_6[1:]
    )
    assert any(
        row
        == [
            "matter_lensing_theory_boundary",
            "positive_principal_negative_energy_case_preserved",
            "true",
        ]
        for row in table_6[1:]
    )
    assert any(
        row
        == [
            "matter_lensing_theory_boundary",
            "deep_aqual_transition_CP11_4_complete",
            "false",
        ]
        for row in table_6[1:]
    )
    assert any(
        row
        == [
            "matter_lensing_theory_boundary",
            "positive_floor_regulator_removes_transition_degeneracy",
            "true",
        ]
        for row in table_6[1:]
    )
    assert any(
        row
        == [
            "matter_lensing_theory_boundary",
            "formula_source_only_classes",
            "3",
        ]
        for row in table_6[1:]
    )
    assert any(
        row
        == [
            "matter_lensing_theory_boundary",
            "formula_full_covariant_bridge_derived",
            "false",
        ]
        for row in table_6[1:]
    )
    assert any(
        row
        == [
            "matter_lensing_theory_boundary",
            "quadrature_quantitative_lensing_solution_derived",
            "false",
        ]
        for row in table_6[1:]
    )
    assert any(
        row
        == [
            "matter_lensing_theory_boundary",
            "quadrature_asymptotic_motion_lensing_match",
            "false",
        ]
        for row in table_6[1:]
    )
    assert any(
        row
        == [
            "matter_lensing_theory_boundary",
            "quadrature_vector_metric_full_causality",
            "false",
        ]
        for row in table_6[1:]
    )
    assert any(
        row
        == [
            "matter_lensing_theory_boundary",
            "unconditional_action_no_go_established",
            "false",
        ]
        for row in table_6[1:]
    )
    assert any(
        row
        == [
            "sampler_calibration_boundary",
            "newtonian_production_runs",
            "0",
        ]
        for row in table_6[1:]
    )
    table_7 = parsed["table-7-prior-art-boundary.csv"]
    assert sum(row[0] == "source" for row in table_7[1:]) == 10


def test_all_six_figures_are_standalone_parseable_svg() -> None:
    artifacts = renderer.build_artifacts(ROOT)
    figures = {name: value for name, value in artifacts.items() if name.endswith(".svg")}
    assert len(figures) == 6
    for value in figures.values():
        root = ET.fromstring(value)
        assert root.tag == "{http://www.w3.org/2000/svg}svg"
        assert root.attrib["viewBox"] == "0 0 960 640"
        assert b"http://" not in value.replace(b'xmlns="http://www.w3.org/2000/svg"', b"")
        assert b"https://" not in value
    observed = figures["figure-1-observed-vs-predicted.svg"]
    assert observed.count(b"<circle") >= 233


def test_receipt_keeps_artifact_reproduction_separate_from_scientific_readiness() -> None:
    receipt = renderer.build_receipt(ROOT)
    assert receipt["decision"] == (
        "PRIMARY_DEVELOPMENT_TABLES_AND_FIGURES_REPRODUCIBLE_NOT_PAPER_READY"
    )
    assert receipt["completed_goal_evidence"] == {
        "CP12.1": "one_command_recreates_all_7_frozen_primary_tables_and_6_frozen_primary_figures"
    }
    assert receipt["counts"] == {
        "primary_tables": 7,
        "primary_figures": 6,
        "artifacts": 13,
        "source_candidate_rows": 233,
        "independent_target_rows_opened": 0,
    }
    assert receipt["claims"]["development_artifacts_reproducible"] is True
    assert receipt["claims"]["external_reproduction"] is False
    assert receipt["claims"]["independent_replication"] is False
    assert receipt["claims"]["bounded_paper_ready"] is False


def test_renderer_is_byte_deterministic_and_stored_artifacts_validate() -> None:
    first = renderer.build_artifacts(ROOT)
    second = renderer.build_artifacts(ROOT)
    assert first == second
    stored = json.loads(
        (
            ROOT / "runs/gravity/publication-readiness/manuscript-artifact-manifest-v1.json"
        ).read_text(encoding="utf-8")
    )
    renderer.validate_receipt(stored, ROOT)
    assert stored == renderer.build_receipt(ROOT)


@pytest.mark.parametrize(
    "mutation,match",
    [
        (
            lambda value: value["rendering_contract"].__setitem__(
                "independent_target_access_allowed", True
            ),
            "rendering contract",
        ),
        (
            lambda value: value["primary_figures"].pop(),
            "artifact inventory",
        ),
        (
            lambda value: value["source_bindings"][0].__setitem__("file_sha256", "0" * 64),
            "source file changed",
        ),
    ],
)
def test_claim_inventory_and_source_mutations_fail_closed(mutation: object, match: str) -> None:
    config = copy.deepcopy(renderer.load_config(ROOT))
    mutation(config)  # type: ignore[operator]
    if match == "source file changed":
        with pytest.raises(renderer.GravityClusterManuscriptRendererError, match=match):
            renderer._load_sources(ROOT, config)
    else:
        with pytest.raises(renderer.GravityClusterManuscriptRendererError, match=match):
            renderer.validate_config(config)
