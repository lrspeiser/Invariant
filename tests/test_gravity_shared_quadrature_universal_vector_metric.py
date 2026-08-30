from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler import gravity_shared_quadrature_universal_vector_metric as vector

ROOT = Path(__file__).resolve().parents[1]


def test_config_and_both_predecessor_commits_are_exact() -> None:
    config = vector.load_config(ROOT)
    validated = vector.validate_predecessors(ROOT, config["predecessor_bindings"])
    assert [row["binding_id"] for row in validated] == [
        "shared_quadrature_covariant_action",
        "shared_quadrature_lensing_backreaction",
    ]
    assert [row["artifact_count"] for row in validated] == [4, 4]
    assert all(row["valid"] for row in validated)


@pytest.mark.parametrize("section", list(vector.EXPECTED_SECTION_HASHES))
def test_every_frozen_section_rejects_mutation(section: str) -> None:
    config = vector.load_config(ROOT)
    changed = copy.deepcopy(config)
    value = changed[section]
    if isinstance(value, dict):
        value["unexpected"] = False
    else:
        value.append({"unexpected": False})
    with pytest.raises(vector.QuadratureUniversalVectorMetricError, match=f"config {section}"):
        vector.validate_config(changed)


def test_symbolic_metric_force_lensing_and_cone_checks_pass() -> None:
    checks, expressions = vector.symbolic_checks()
    assert len(checks) == 21
    allowed_residuals = {
        "0",
        "Matrix([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])",
    }
    assert all(row["passed"] and row["residual"] in allowed_residuals for row in checks)
    assert checks[0]["check_id"] == "S01_PHYSICAL_METRIC_EQUIVALENT_FORMS"
    assert checks[-1]["check_id"] == "S21_WEAK_DUST_SOURCE_NORMALIZATION"
    assert "exp" in expressions["physical_metric_rest_frame"]
    assert "1 - 2*s" in expressions["K"] or "2*s - 1" in expressions["K"]


def test_numeric_fixed_aether_scalar_cones_pass() -> None:
    rows = vector.numeric_cases(vector.load_config(ROOT))
    assert len(rows) == 4
    assert all(row["passed"] for row in rows)
    assert all(
        row["scalar_parallel_speed_squared"]
        == pytest.approx(row["physical_photon_speed_squared"], abs=1e-12)
        for row in rows
    )
    assert all(
        0 < row["scalar_transverse_speed_squared"] < row["physical_photon_speed_squared"]
        for row in rows
    )


def test_receipt_keeps_positive_result_and_global_failures_separate() -> None:
    receipt = vector.build_receipt(ROOT)
    assert receipt["decision"] == vector.DECISION
    assert receipt["counts"] == {
        "predecessor_bindings": 2,
        "predecessor_artifacts": 8,
        "symbolic_checks": 21,
        "symbolic_checks_passed": 21,
        "numeric_cases": 4,
        "numeric_cases_passed": 4,
        "observational_files_opened": 0,
        "observational_rows_opened": 0,
        "network_calls": 0,
        "model_or_paid_calls": 0,
        "gpu_calls": 0,
    }
    adjudication = receipt["adjudication"]
    assert adjudication["one_universal_massive_matter_and_photon_metric_defined"] is True
    assert adjudication["separate_photon_adjustment_present"] is False
    assert adjudication["leading_scalar_motion_and_lensing_relation_matched"] is True
    assert adjudication["fixed_aether_scalar_block_causal_relative_to_physical_cone"] is True
    assert adjudication["full_metric_vector_scalar_matter_causality_established"] is False
    assert adjudication["gw_physical_gate_passed"] is False
    assert adjudication["low_gradient_transition_nondegenerate"] is False
    assert adjudication["finite_gradient_endpoint_regular"] is False
    assert adjudication["timelike_cosmological_branch_defined"] is False
    assert receipt["claim_boundary"]["full_covariant_health_established"] is False
    assert receipt["claim_boundary"]["scientific_observational_claim_allowed"] is False
    assert set(receipt["zero_access_and_compute"].values()) == {0}


def test_stored_receipt_rebuilds_and_second_write_is_identical() -> None:
    stored = json.loads((ROOT / vector.OUTPUT_PATH).read_text(encoding="utf-8"))
    vector.validate_receipt(stored, ROOT)
    assert stored == vector.build_receipt(ROOT)
    path, publication = vector.write_receipt(ROOT)
    assert path == ROOT / vector.OUTPUT_PATH
    assert publication == "EXISTING_IDENTICAL"


def test_rehashed_health_overclaim_fails() -> None:
    receipt = vector.build_receipt(ROOT)
    receipt["claim_boundary"]["full_covariant_health_established"] = True
    body = {key: value for key, value in receipt.items() if key != "content_sha256"}
    receipt["content_sha256"] = vector._sha(body)
    with pytest.raises(vector.QuadratureUniversalVectorMetricError, match="evidence changed"):
        vector.validate_receipt(receipt, ROOT)


def test_source_has_no_observational_or_network_loader() -> None:
    source = (
        ROOT / "src/sigma_theory_compiler/gravity_shared_quadrature_universal_vector_metric.py"
    ).read_text(encoding="utf-8")
    assert "urllib" not in source
    assert "requests" not in source
    assert "pandas" not in source
    assert "astropy" not in source
