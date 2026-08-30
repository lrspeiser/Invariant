from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import sympy as sp

from sigma_theory_compiler import gravity_shared_quadrature_scalar_cherenkov_cutoff_rate as rate

ROOT = Path(__file__).resolve().parents[1]


def test_symbolic_derivation_passes_exact_inventory() -> None:
    config = rate.load_config(ROOT)
    checks, expressions = rate.symbolic_checks()
    assert [row["check_id"] for row in checks] == config["machine_check_contract"][
        "required_symbolic_checks"
    ]
    assert len(checks) == 25
    assert all(row["passed"] for row in checks)
    assert expressions["stationary_power_magnitude"].startswith("E**2")


def test_numeric_cases_preserve_open_and_closed_gates() -> None:
    config = rate.load_config(ROOT)
    rows = rate.numeric_checks(config)
    assert len(rows) == 4
    assert all(row["passed"] for row in rows)
    assert [row["open_emission_gate"] for row in rows] == [True, False, False, True]
    assert rows[2]["dimensionless_rate_coefficient"] == "0"


def test_final_power_and_survival_formulas_are_exact() -> None:
    _, expressions = rate.symbolic_checks()
    symbols = {
        name: sp.Symbol(name, positive=True)
        for name in ("E", "v", "L", "Mpl", "s", "mu2", "OmegaIR", "OmegaUV")
    }
    power = sp.sympify(expressions["stationary_power_magnitude"], locals=symbols)
    fraction = sp.sympify(expressions["small_loss_fraction"], locals=symbols)
    energy, velocity, distance = symbols["E"], symbols["v"], symbols["L"]
    assert sp.simplify(fraction - distance * power / (energy * velocity)) == 0


def test_alpha_cancels_but_planck_suppression_remains() -> None:
    _, expressions = rate.symbolic_checks()
    power = expressions["stationary_power_magnitude"]
    assert "alpha" not in power
    assert "Mpl**2" in power


def test_source_is_not_trace_suppressed_ultrarelativistically() -> None:
    _, expressions = rate.symbolic_checks()
    assert expressions["point_particle_charge"] == "E*(v**2 + 1)"
    assert "E**2" in expressions["ultrarelativistic_power_magnitude"]


def test_low_gradient_and_endpoint_scalings_are_distinct() -> None:
    _, expressions = rate.symbolic_checks()
    assert expressions["low_s_scaled_rate_limit"] == "1/(v*sqrt(2 - mu2))"
    assert expressions["endpoint_scaled_rate_limit"] == "2/(v*sqrt(1 - mu2))"


def test_predecessor_commit_and_bytes_validate() -> None:
    config = rate.load_config(ROOT)
    rows = rate.validate_predecessors(config, ROOT)
    assert len(rows) == 1
    assert rows[0]["valid"] is True
    assert rows[0]["git_commit"] == "fd5e2deb23a71f2f962961b7442235e4650aae6c"


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("frozen_local_action_contract", "quadratic_density", "forged"),
        ("anisotropic_rate_contract", "stationary_power", "forged"),
        ("survival_contract", "cutoff_inequality", "forged"),
        ("validity_and_missing_inputs", "cutoff", "forged"),
        ("adjudication", "physical_UV_cutoff_established", True),
        ("claim_boundary", "observational_scalar_cherenkov_exclusion_established", True),
        ("zero_access_and_compute", "cosmic_ray_rows_opened", 1),
    ],
)
def test_nested_contract_mutations_fail_closed(section: str, key: str, value: object) -> None:
    config = rate.load_config(ROOT)
    forged = copy.deepcopy(config)
    forged[section][key] = value
    with pytest.raises(rate.QuadratureScalarCherenkovRateError):
        rate.validate_config(forged, ROOT)


def test_machine_inventory_mutation_fails_closed() -> None:
    config = rate.load_config(ROOT)
    forged = copy.deepcopy(config)
    forged["machine_check_contract"]["required_symbolic_checks"].pop()
    with pytest.raises(rate.QuadratureScalarCherenkovRateError):
        rate.validate_config(forged, ROOT)


def test_primary_source_scope_is_restrained() -> None:
    config = rate.load_config(ROOT)
    source = config["primary_source_context"][0]
    assert source["url"] == "https://arxiv.org/abs/2109.10812"
    assert "none of its cosmological model coefficients" in source["scope"]


def test_build_receipt_is_zero_access_and_restrained() -> None:
    receipt = rate.build_receipt(ROOT)
    assert receipt["adjudication"]["restricted_stationary_cutoff_dependent_radiation_rate_derived"]
    assert receipt["adjudication"]["physical_UV_cutoff_established"] is False
    assert receipt["adjudication"]["cosmic_ray_survival_test_passed"] is False
    assert (
        receipt["claim_boundary"]["observational_scalar_cherenkov_exclusion_established"] is False
    )
    assert receipt["counts"]["observational_rows_opened"] == 0
    assert all(value == 0 for value in receipt["zero_access_and_compute"].values())


def test_stored_receipt_matches_exact_rebuild() -> None:
    stored = rate.check_receipt(ROOT)
    rebuilt = rate.build_receipt(ROOT)
    assert stored == rebuilt
    assert stored["content_sha256"] == rate._content_sha(
        {key: value for key, value in stored.items() if key != "content_sha256"}
    )


def test_tampered_receipt_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    receipt = rate.build_receipt(ROOT)
    receipt["adjudication"]["observational_scalar_cherenkov_exclusion_established"] = True
    payload = dict(receipt)
    payload.pop("content_sha256")
    receipt["content_sha256"] = rate._content_sha(payload)
    output = tmp_path / "receipt.json"
    output.write_text(json.dumps(receipt, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    monkeypatch.setattr(rate, "OUTPUT_PATH", output.relative_to(tmp_path))
    monkeypatch.setattr(rate, "_repo_root", lambda: tmp_path)
    with pytest.raises(rate.QuadratureScalarCherenkovRateError):
        rate.check_receipt(tmp_path)


def test_atomic_no_clobber_preserves_different_bytes(tmp_path: Path) -> None:
    target = tmp_path / "receipt.json"
    target.write_bytes(b"existing")
    with pytest.raises(rate.QuadratureScalarCherenkovRateError):
        rate._atomic_no_clobber(target, b"new")
    assert target.read_bytes() == b"existing"


def test_atomic_no_clobber_accepts_identical_bytes(tmp_path: Path) -> None:
    target = tmp_path / "receipt.json"
    target.write_bytes(b"same")
    assert rate._atomic_no_clobber(target, b"same") == "EXISTING_IDENTICAL"
    assert target.read_bytes() == b"same"


def test_status_reports_conditional_rate_not_observational_pass() -> None:
    result = rate.status(ROOT)
    assert result["valid"] is True
    assert result["restricted_rate_derived"] is True
    assert result["physical_cutoff_established"] is False
    assert result["observational_exclusion"] is False
    assert result["observational_rows_opened"] == 0
