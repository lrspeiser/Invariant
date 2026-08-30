from __future__ import annotations

import copy
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import sympy as sp

from sigma_theory_compiler import (
    gravity_shared_quadrature_scalar_full_local_fluctuation as full_scalar,
)

ROOT = Path(__file__).resolve().parents[1]


def test_symbolic_derivation_passes_exact_inventory() -> None:
    config = full_scalar.load_config(ROOT)
    checks, expressions = full_scalar.symbolic_checks(config)
    assert [row["check_id"] for row in checks] == config["machine_check_contract"][
        "required_symbolic_checks"
    ]
    assert len(checks) == 30
    assert all(row["passed"] for row in checks)
    assert expressions["L3_new"] != "0"
    assert expressions["L4_amplitude"] != "0"
    assert expressions["L4_mixed"] != "0"


def test_full_expansion_restores_only_frozen_conformal_vertices() -> None:
    _, expressions = full_scalar.symbolic_checks(full_scalar.load_config(ROOT))
    assert expressions["L3"] != expressions["L3_derivative"]
    assert expressions["L4"] != expressions["L4_derivative"]
    assert "z" in expressions["L3_new"]
    assert "z**2" in expressions["L4_amplitude"]
    assert "r" in expressions["L4_mixed"]


def test_new_canonical_coefficients_match_closed_forms() -> None:
    _, expressions = full_scalar.symbolic_checks(full_scalar.load_config(ROOT))
    s = sp.Symbol("s", positive=True)
    values = {
        key: sp.sympify(expressions[key], locals={"s": s})
        for key in ("e3_conformal", "e4_conformal", "e4_mixed")
    }
    d = 1 - 2 * s
    assert sp.simplify(values["e3_conformal"] ** 2 - 2 * d / s) == 0
    assert sp.simplify(values["e4_conformal"] - 2 * d / s) == 0
    assert sp.simplify(values["e4_mixed"] - 1 / (s**2 * (1 - s))) == 0
    assert sp.simplify(values["e4_conformal"] - values["e3_conformal"] ** 2) == 0


def test_numeric_probes_are_finite_and_derivative_limited() -> None:
    rows = full_scalar.numeric_checks(full_scalar.load_config(ROOT))
    assert [row["s"] for row in rows] == ["1/100", "1/10", "1/6", "1/4", "49/100"]
    assert all(row["passed"] for row in rows)
    assert all(row["limiting_scale_at_probe"] == "derivative" for row in rows)
    assert all(
        row["lambda_full_coefficient_over_Lambda0"] == row["lambda_derivative_over_Lambda0"]
        for row in rows
    )


def test_endpoint_hierarchy_is_exact_not_a_physical_cutoff() -> None:
    config = full_scalar.load_config(ROOT)
    assert config["endpoint_and_hierarchy_contract"]["result"].startswith(
        "Restoring the full scalar-only"
    )
    assert config["adjudication"]["derivative_scale_remains_low_s_asymptotic_limiter"]
    assert config["adjudication"]["derivative_scale_remains_high_s_asymptotic_limiter"]
    assert config["adjudication"]["physical_UV_cutoff_established"] is False
    assert config["adjudication"]["strong_coupling_theorem_established"] is False


def test_predecessor_commits_and_bytes_validate() -> None:
    rows = full_scalar.validate_predecessors(full_scalar.load_config(ROOT), ROOT)
    assert [row["git_commit"] for row in rows] == [
        "38b1d4a66f79a6275098e4505b6fb55275079378",
        "5faee2f4eee462b2e25ceb2f2e4e861bbab2409f",
    ]
    assert sum(row["artifact_count"] for row in rows) == 8
    assert all(row["valid"] for row in rows)


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("full_scalar_expansion_contract", "exact_density", "forged"),
        ("canonical_interaction_contract", "new_cubic", "forged"),
        ("coefficient_scale_contract", "conformal_scale", "forged"),
        ("endpoint_and_hierarchy_contract", "result", "forged"),
        ("adjudication", "physical_UV_cutoff_established", True),
        ("claim_boundary", "strong_coupling_scale_established", True),
        ("zero_access_and_compute", "observational_rows_opened", 1),
    ],
)
def test_nested_contract_mutations_fail_closed(section: str, key: str, value: object) -> None:
    forged = copy.deepcopy(full_scalar.load_config(ROOT))
    forged[section][key] = value
    with pytest.raises(full_scalar.QuadratureScalarFullLocalError):
        full_scalar.validate_config(forged, ROOT)


def test_predecessor_source_and_machine_mutations_fail_closed() -> None:
    mutations = []
    predecessor = copy.deepcopy(full_scalar.load_config(ROOT))
    predecessor["predecessor_bindings"][0]["git_commit"] = "0" * 40
    mutations.append(predecessor)
    source = copy.deepcopy(full_scalar.load_config(ROOT))
    source["primary_source_context"][0]["scope"] = "forged"
    mutations.append(source)
    machine = copy.deepcopy(full_scalar.load_config(ROOT))
    machine["machine_check_contract"]["required_symbolic_checks"].pop()
    mutations.append(machine)
    for forged in mutations:
        with pytest.raises(full_scalar.QuadratureScalarFullLocalError):
            full_scalar.validate_config(forged, ROOT)


def test_build_receipt_is_zero_access_and_restrained() -> None:
    receipt = full_scalar.build_receipt(ROOT)
    assert receipt["adjudication"]["fixed_metric_aether_full_scalar_expansion_through_quartic"]
    assert receipt["adjudication"]["full_coupled_metric_aether_matter_fluctuation_action"] is False
    assert receipt["claim_boundary"]["physical_cutoff_established"] is False
    assert receipt["counts"]["symbolic_checks"] == 30
    assert receipt["counts"]["numeric_cases"] == 5
    assert all(value == 0 for value in receipt["zero_access_and_compute"].values())


def test_stored_receipt_matches_exact_rebuild() -> None:
    stored = full_scalar.check_receipt(ROOT)
    rebuilt = full_scalar.build_receipt(ROOT)
    assert stored == rebuilt
    payload = dict(stored)
    content = payload.pop("content_sha256")
    assert content == full_scalar._content_sha(payload)


def test_tampered_receipt_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    receipt = full_scalar.build_receipt(ROOT)
    receipt["claim_boundary"]["physical_cutoff_established"] = True
    payload = dict(receipt)
    payload.pop("content_sha256")
    receipt["content_sha256"] = full_scalar._content_sha(payload)
    output = tmp_path / "receipt.json"
    output.write_text(json.dumps(receipt, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    monkeypatch.setattr(full_scalar, "OUTPUT_PATH", output.relative_to(tmp_path))
    with pytest.raises(full_scalar.QuadratureScalarFullLocalError):
        full_scalar.check_receipt(tmp_path)


def test_atomic_no_clobber_preserves_different_bytes(tmp_path: Path) -> None:
    target = tmp_path / "receipt.json"
    target.write_bytes(b"existing")
    with pytest.raises(full_scalar.QuadratureScalarFullLocalError):
        full_scalar._atomic_no_clobber(target, b"new")
    assert target.read_bytes() == b"existing"


def test_atomic_no_clobber_accepts_identical_bytes(tmp_path: Path) -> None:
    target = tmp_path / "receipt.json"
    target.write_bytes(b"same")
    assert full_scalar._atomic_no_clobber(target, b"same") == "EXISTING_IDENTICAL"


def test_atomic_no_clobber_race_retains_one_complete_payload(tmp_path: Path) -> None:
    target = tmp_path / "receipt.json"

    def publish(payload: bytes) -> str:
        try:
            return full_scalar._atomic_no_clobber(target, payload)
        except full_scalar.QuadratureScalarFullLocalError:
            return "REFUSED"

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(publish, (b"first", b"second")))
    assert target.read_bytes() in {b"first", b"second"}
    assert outcomes.count("CREATED") == 1
    assert outcomes.count("REFUSED") == 1


def test_status_preserves_claim_ceiling() -> None:
    result = full_scalar.status(ROOT)
    assert result["valid"] is True
    assert result["full_fixed_background_scalar_quartic"] is True
    assert result["new_interactions_canonicalized"] is True
    assert result["derivative_scale_endpoint_limiter"] is True
    assert result["full_coupled_action"] is False
    assert result["physical_cutoff"] is False
    assert result["observational_rows_opened"] == 0
