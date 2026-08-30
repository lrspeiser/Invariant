from __future__ import annotations

import copy
import json
from fractions import Fraction
from pathlib import Path

import pytest

from sigma_theory_compiler import (
    gravity_shared_quadrature_aether_mode_necessary_conditions as aether_modes,
)

ROOT = Path(__file__).resolve().parents[1]


def test_config_and_both_predecessor_commits_are_exact() -> None:
    config = aether_modes.load_config(ROOT)
    validated = aether_modes.validate_predecessors(ROOT, config["predecessor_bindings"])
    assert [row["binding_id"] for row in validated] == [
        "quadrature_universal_vector_metric",
        "committed_einstein_aether_reduced_principal_control",
    ]
    assert [row["artifact_count"] for row in validated] == [4, 3]
    assert all(row["valid"] for row in validated)


@pytest.mark.parametrize("section", list(aether_modes.EXPECTED_SECTION_HASHES))
def test_every_frozen_section_rejects_mutation(section: str) -> None:
    config = aether_modes.load_config(ROOT)
    changed = copy.deepcopy(config)
    value = changed[section]
    if isinstance(value, dict):
        value["unexpected"] = False
    else:
        value.append({"unexpected": False})
    with pytest.raises(aether_modes.QuadratureAetherModeError, match=f"config {section}"):
        aether_modes.validate_config(changed)


def test_symbolic_specialization_and_inherited_controls_pass() -> None:
    checks, expressions, inherited = aether_modes.symbolic_checks()
    assert len(checks) == 25
    assert all(row["passed"] and row["residual"] == "0" for row in checks)
    assert checks[0]["check_id"] == "S01_COMMITTED_REDUCED_PRINCIPAL_CONTROL"
    assert checks[-1]["check_id"] == "S25_NO_UNIFORM_POSITIVE_KINETIC_MARGIN"
    assert inherited["principal_mode_count"] == 5
    assert inherited["principal_negative_controls_pass"] is True
    assert inherited["energy_speed_only_negative_controls_pass"] is True
    assert "c1 + c3" in expressions["spin_2_speed_squared"]
    assert "exp(-2*varphi)" == expressions["mode_to_photon_speed_ratio"]


def test_finite_luminal_epsilon_cases_have_positive_but_vanishing_residues() -> None:
    rows = aether_modes.epsilon_cases(aether_modes.load_config(ROOT))
    assert [row["epsilon"] for row in rows] == ["1/100", "1/1000", "1/1000000"]
    assert all(row["passed"] for row in rows)
    assert all(row["speed_squared"] == ["1", "1", "1"] for row in rows)
    assert all(row["alpha2"] == "0" for row in rows)
    assert all(row["alpha1"].startswith("-") for row in rows)
    assert [Fraction(row["spin_1_energy"]) for row in rows] == sorted(
        [Fraction(row["spin_1_energy"]) for row in rows], reverse=True
    )
    assert [Fraction(row["spin_0_energy"]) for row in rows] == sorted(
        [Fraction(row["spin_0_energy"]) for row in rows], reverse=True
    )


def test_receipt_separates_finite_locus_from_exact_singular_intersection() -> None:
    receipt = aether_modes.build_receipt(ROOT)
    assert receipt["decision"] == aether_modes.DECISION
    assert receipt["counts"] == {
        "predecessor_bindings": 2,
        "predecessor_artifacts": 7,
        "inherited_pure_aether_modes": 5,
        "symbolic_checks": 25,
        "symbolic_checks_passed": 25,
        "epsilon_cases": 3,
        "epsilon_cases_passed": 3,
        "observational_files_opened": 0,
        "observational_rows_opened": 0,
        "network_calls_by_builder": 0,
        "model_or_paid_calls": 0,
        "gpu_calls": 0,
    }
    adjudication = receipt["adjudication"]
    assert adjudication["exact_c13_alpha1_alpha2_zero_is_regular"] is False
    assert adjudication["finite_positive_all_g_luminal_locus_exists"] is True
    assert adjudication["finite_locus_alpha2_zero"] is True
    assert adjudication["finite_locus_alpha1_zero"] is False
    assert adjudication["uniform_positive_kinetic_margin_as_alpha1_tolerance_goes_zero"] is False
    assert (
        adjudication["full_quadrature_scalar_vector_metric_principal_system_established"] is False
    )
    assert receipt["claim_boundary"]["full_covariant_health_established"] is False
    assert receipt["claim_boundary"]["scientific_observational_claim_allowed"] is False
    assert set(receipt["zero_access_and_compute"].values()) == {0}


def test_stored_receipt_rebuilds_and_second_write_is_identical() -> None:
    stored = json.loads((ROOT / aether_modes.OUTPUT_PATH).read_text(encoding="utf-8"))
    aether_modes.validate_receipt(stored, ROOT)
    assert stored == aether_modes.build_receipt(ROOT)
    path, publication = aether_modes.write_receipt(ROOT)
    assert path == ROOT / aether_modes.OUTPUT_PATH
    assert publication == "EXISTING_IDENTICAL"


def test_rehashed_health_overclaim_fails() -> None:
    receipt = aether_modes.build_receipt(ROOT)
    receipt["claim_boundary"]["full_covariant_health_established"] = True
    body = {key: value for key, value in receipt.items() if key != "content_sha256"}
    receipt["content_sha256"] = aether_modes._sha(body)
    with pytest.raises(aether_modes.QuadratureAetherModeError, match="evidence changed"):
        aether_modes.validate_receipt(receipt, ROOT)


def test_source_has_no_observational_or_network_loader() -> None:
    source = (
        ROOT
        / "src/sigma_theory_compiler/gravity_shared_quadrature_aether_mode_necessary_conditions.py"
    ).read_text(encoding="utf-8")
    assert "urllib" not in source
    assert "requests" not in source
    assert "pandas" not in source
    assert "astropy" not in source
