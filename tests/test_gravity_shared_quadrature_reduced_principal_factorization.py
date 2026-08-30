from __future__ import annotations

import copy
import json
from fractions import Fraction
from pathlib import Path

import pytest

from sigma_theory_compiler import (
    gravity_shared_quadrature_reduced_principal_factorization as factorization,
)

ROOT = Path(__file__).resolve().parents[1]


def test_config_and_predecessor_commits_are_exact() -> None:
    config = factorization.load_config(ROOT)
    validated = factorization.validate_predecessors(ROOT, config["predecessor_bindings"])
    assert [row["binding_id"] for row in validated] == [
        "quadrature_universal_vector_metric",
        "quadrature_aether_mode_necessary_conditions",
    ]
    assert [row["artifact_count"] for row in validated] == [4, 4]
    assert all(row["valid"] for row in validated)


@pytest.mark.parametrize("section", list(factorization.EXPECTED_SECTION_HASHES))
def test_every_frozen_section_rejects_mutation(section: str) -> None:
    config = factorization.load_config(ROOT)
    changed = copy.deepcopy(config)
    value = changed[section]
    if isinstance(value, dict):
        value["unexpected"] = False
    else:
        value.append({"unexpected": False})
    with pytest.raises(factorization.QuadratureReducedPrincipalError, match=f"config {section}"):
        factorization.validate_config(changed)


def test_symbolic_derivative_order_factorization_passes() -> None:
    checks, expressions = factorization.symbolic_checks()
    assert len(checks) == 22
    allowed_residuals = {
        "0",
        "Matrix([[0, 0], [0, 0], [0, 0]])",
        "Matrix([[0, 0, 0], [0, 0, 0]])",
        "Matrix([[0], [0]])",
        "Matrix([[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]])",
    }
    assert all(row["passed"] and row["residual"] in allowed_residuals for row in checks)
    assert checks[0]["check_id"] == "S01_SCALAR_DERIVATIVE_HESSIAN"
    assert checks[-1]["check_id"] == "S22_PHYSICAL_CONE_RATIO_AT_VARPHI_ZERO"
    assert "A*v" in expressions["lower_order_scalar_aether_mixing"]
    assert expressions["lower_order_scalar_metric_mixing"] != "0"
    assert expressions["predecessor_vector_checks_passed"] == 21
    assert expressions["predecessor_aether_checks_passed"] == 25


def test_numeric_cases_keep_all_six_reduced_modes_inside_photon_cone() -> None:
    rows = factorization.numeric_cases(factorization.load_config(ROOT))
    assert len(rows) == 4
    assert all(row["passed"] for row in rows)
    assert all(row["aether_speed_squared"] == ["1"] * 5 for row in rows)
    assert all(row["scalar_speed_squared"]["parallel"] == "1" for row in rows)
    transverse = [Fraction(row["scalar_speed_squared"]["transverse"]) for row in rows]
    assert all(0 < value < 1 for value in transverse)
    assert all(row["lower_order_scalar_aether_mixing_nonzero"] for row in rows)
    assert all(row["maximum_physical_speed_squared"] == "1" for row in rows)


def test_receipt_preserves_local_result_and_global_blockers() -> None:
    receipt = factorization.build_receipt(ROOT)
    assert receipt["decision"] == factorization.DECISION
    assert receipt["counts"] == {
        "predecessor_bindings": 2,
        "predecessor_artifacts": 8,
        "reduced_physical_modes": 6,
        "symbolic_checks": 22,
        "symbolic_checks_passed": 22,
        "numeric_cases": 4,
        "numeric_cases_passed": 4,
        "observational_files_opened": 0,
        "observational_rows_opened": 0,
        "network_calls_by_builder": 0,
        "model_or_paid_calls": 0,
        "gpu_calls": 0,
    }
    adjudication = receipt["adjudication"]
    assert adjudication["static_W_zero_scalar_to_aether_metric_principal_mixing_present"] is False
    assert adjudication["restricted_reduced_physical_principal_factorization_established"] is True
    assert (
        adjudication["finite_locus_six_reduced_modes_causal_relative_to_physical_photons"] is True
    )
    assert adjudication["exact_preferred_frame_free_limit_regular"] is False
    assert adjudication["unreduced_gauge_constraint_system_strongly_hyperbolic"] is False
    assert adjudication["nonzero_W_or_varying_background_factorization_established"] is False
    assert receipt["claim_boundary"]["healthy_action_established"] is False
    assert receipt["claim_boundary"]["scientific_observational_claim_allowed"] is False
    assert set(receipt["zero_access_and_compute"].values()) == {0}


def test_stored_receipt_rebuilds_and_second_write_is_identical() -> None:
    stored = json.loads((ROOT / factorization.OUTPUT_PATH).read_text(encoding="utf-8"))
    factorization.validate_receipt(stored, ROOT)
    assert stored == factorization.build_receipt(ROOT)
    path, publication = factorization.write_receipt(ROOT)
    assert path == ROOT / factorization.OUTPUT_PATH
    assert publication == "EXISTING_IDENTICAL"


def test_rehashed_global_health_overclaim_fails() -> None:
    receipt = factorization.build_receipt(ROOT)
    receipt["claim_boundary"]["healthy_action_established"] = True
    body = {key: value for key, value in receipt.items() if key != "content_sha256"}
    receipt["content_sha256"] = factorization._sha(body)
    with pytest.raises(factorization.QuadratureReducedPrincipalError, match="evidence changed"):
        factorization.validate_receipt(receipt, ROOT)


def test_source_has_no_observational_or_network_loader() -> None:
    source = (
        ROOT
        / "src/sigma_theory_compiler/gravity_shared_quadrature_reduced_principal_factorization.py"
    ).read_text(encoding="utf-8")
    assert "urllib" not in source
    assert "requests" not in source
    assert "pandas" not in source
    assert "astropy" not in source
