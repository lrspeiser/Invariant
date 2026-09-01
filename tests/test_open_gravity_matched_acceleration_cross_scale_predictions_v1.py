from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest

from sigma_theory_compiler import (
    open_gravity_matched_acceleration_cross_scale_predictions_v1 as compiler,
)


@pytest.fixture(scope="module")
def config() -> dict:
    value = json.loads(Path(compiler.CONFIG_PATH).read_text(encoding="utf-8"))
    compiler.validate_config(value)
    return value


@pytest.fixture(scope="module")
def candidates(config: dict) -> list[dict]:
    return compiler.candidate_registry(config)


@pytest.fixture(scope="module")
def predictions(config: dict) -> dict:
    return compiler.build_predictions(config)


def test_method_admission_gate_requires_real_sources_and_benchmarks(config: dict) -> None:
    gate = config["method_admission_gate"]
    assert gate["missing_real_source_disposition"] == "SOURCE_BLOCKED"
    assert gate["missing_paper_or_analytic_benchmark_disposition"] == "THEORY_ONLY_UNVALIDATED"
    assert gate["failed_benchmark_disposition"] == "BENCHMARK_FAILED_RETAINED_NOT_SCORED"
    assert gate["proxy_may_impersonate_published_solver"] is False
    assert gate["response_values_may_repair_or_tune_method"] is False
    assert gate["novel_exact_formula_paper_required"] is False
    for evidence in config["current_method_evidence"]:
        assert evidence["real_data"]
        assert evidence["primary_or_analytic_anchors"]
        assert evidence["required_passed_checks"]


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("method_admission_gate", "missing_real_source_disposition"), "SCORE_ANYWAY"),
        (("method_admission_gate", "proxy_may_impersonate_published_solver"), True),
        (("method_admission_gate", "response_values_may_repair_or_tune_method"), True),
        (("current_method_evidence", 0, "real_data"), []),
        (("current_method_evidence", 1, "primary_or_analytic_anchors"), []),
        (("scientific_boundary", "galaxy_response_rows_opened"), 1),
        (("claims", "unique_publishable_signal"), True),
    ],
)
def test_material_contract_mutations_fail_closed(
    config: dict, path: tuple[str | int, ...], replacement: object
) -> None:
    mutated = copy.deepcopy(config)
    cursor = mutated
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = replacement
    with pytest.raises(compiler.PredictionCompilerError):
        compiler.validate_config(mutated)


def test_all_candidates_have_source_and_benchmark_evidence(candidates: list[dict]) -> None:
    assert len(candidates) == 32
    assert len({row["candidate_id"] for row in candidates}) == 32
    for row in candidates:
        evidence = row["admission_evidence"]
        assert evidence["real_source_domains"]
        assert evidence["required_checks"]
        assert evidence["class"] in {
            "PRIMARY_PUBLISHED_FORMULA",
            "NOVEL_FALSIFIABLE_HYPOTHESIS",
        }
    geometry = [
        row
        for row in candidates
        if row.get("driver") == "GEOMETRY" or "GEOMETRY" in row.get("drivers", [])
    ]
    assert geometry
    assert all(
        row["admission_evidence"]["real_source_domains"] == ["GALAXY_2P5D"] for row in geometry
    )


def test_mond_standard_exact_equation_and_limits(config: dict) -> None:
    a0 = config["constants"]["a0_m_s2"]
    for g_newton in (1.0e-16, a0, 1.0e-3):
        g = compiler.mond_standard(g_newton, a0)
        mu = (g / a0) / math.sqrt(1.0 + (g / a0) ** 2)
        assert mu * g == pytest.approx(g_newton, rel=2.0e-15)
    assert compiler.mond_standard(1.0e-16, a0) == pytest.approx(math.sqrt(1.0e-16 * a0), rel=5.0e-4)
    assert compiler.mond_standard(1.0e-3, a0) / 1.0e-3 - 1.0 < 1.0e-12


def test_published_rar_and_emond_controls_are_finite_and_bounded(config: dict) -> None:
    a0 = config["constants"]["a0_m_s2"]
    assert compiler.rar_2016(a0, a0) > a0
    assert compiler.rar_2016(1.0e6 * a0, a0) == pytest.approx(1.0e6 * a0)
    shallow = compiler.emond_a0(config, 1.0e-8)
    deep = compiler.emond_a0(config, 1.0e-5)
    assert 0.0 < shallow < deep < config["constants"]["emond_cH0_m_s2"]


def test_exact_source_population_and_no_response_access(predictions: dict) -> None:
    assert predictions["source_row_count"] == 14_021
    assert predictions["galaxy_source_row_count"] == 13_500
    assert predictions["cluster_source_row_count"] == 521
    assert predictions["scientific_boundary"]["galaxy_response_rows_opened"] == 0
    assert predictions["scientific_boundary"]["cluster_response_files_opened"] == 0
    assert predictions["scientific_boundary"]["scores_computed"] == 0
    assert predictions["scientific_boundary"]["network_calls"] == 0


def test_geometry_is_not_invented_for_spherical_cluster_sources(predictions: dict) -> None:
    geometry_ids = {
        row["candidate_id"]
        for row in predictions["candidate_registry"]
        if row.get("driver") == "GEOMETRY" or "GEOMETRY" in row.get("drivers", [])
    }
    assert len(geometry_ids) == 8
    for row in predictions["prediction_rows"]:
        for candidate_id in geometry_ids:
            disposition = row["predictions"][candidate_id]["disposition"]
            if row["source"]["domain"] == "CLUSTER":
                assert disposition == "SOURCE_BLOCKED_DRIVER_UNAVAILABLE"
            else:
                assert disposition == "COMPILED"


def test_every_compiled_prediction_is_finite_positive(predictions: dict) -> None:
    compiled = 0
    for row in predictions["prediction_rows"]:
        for value in row["predictions"].values():
            if value["disposition"] == "COMPILED":
                assert math.isfinite(value["g_prediction_m_s2"])
                assert value["g_prediction_m_s2"] > 0.0
                assert math.isfinite(value["effective_a0_m_s2"])
                assert value["effective_a0_m_s2"] >= 0.0
                compiled += 1
    assert compiled == predictions["compiled_prediction_count"] == 444_504
    assert predictions["blocked_prediction_count"] == 4_168


def test_potential_depth_provides_real_cross_scale_leverage(predictions: dict) -> None:
    galaxy_reference = predictions["reference_values"]["potential_depth_c2"]
    cluster_depths = [
        row["source"]["potential_depth_c2"]
        for row in predictions["prediction_rows"]
        if row["source"]["domain"] == "CLUSTER"
    ]
    assert min(cluster_depths) / galaxy_reference > 7.0
    assert max(cluster_depths) / galaxy_reference > 50.0


def test_worst_case_solar_recovery_passes_without_tuning(config: dict, predictions: dict) -> None:
    assert (
        predictions["solar_worst_case_fractional_deviation"]
        < config["constants"]["solar_max_fractional_deviation"]
    )
    assert config["candidate_program"]["same_constants_across_every_object"] is True
    assert config["candidate_program"]["per_object_parameters"] == 0
    assert config["candidate_program"]["response_tuning"] is False


def test_required_full_solvers_are_blocked_not_impersonated(config: dict) -> None:
    required = config["required_but_not_impersonated_comparators"]
    assert [row["id"] for row in required] == [
        "AQUAL_FULL_3D",
        "QUMOND_FULL_3D",
        "REFRACTED_GRAVITY",
        "MOG_STVG",
        "NONLOCAL_GRAVITY",
        "GP01_ELLIPTIC_FULL_3D",
        "NFW_BURKERT_EINASTO",
    ]
    assert all(row["status"] != "COMPILED" for row in required)


def test_package_hash_pins_match_after_final_seal() -> None:
    if compiler._MODULE_SEMANTIC_SHA256 == "0" * 64 or compiler._TEST_RAW_SHA256 == "0" * 64:
        pytest.skip("package self-pins are installed only at the final mutation seal")
    assert (
        compiler.module_semantic_sha256(compiler._repo_path(compiler.MODULE_PATH))
        == compiler._MODULE_SEMANTIC_SHA256
    )
    assert (
        compiler.file_sha256(compiler._repo_path(compiler.TEST_PATH)) == compiler._TEST_RAW_SHA256
    )
