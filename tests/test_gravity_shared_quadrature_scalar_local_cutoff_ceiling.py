from __future__ import annotations

import copy
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import sympy as sp

from sigma_theory_compiler import gravity_shared_quadrature_scalar_local_cutoff_ceiling as cutoff

ROOT = Path(__file__).resolve().parents[1]


def test_symbolic_derivation_passes_exact_inventory() -> None:
    config = cutoff.load_config(ROOT)
    checks, expressions = cutoff.symbolic_checks(config)
    assert [row["check_id"] for row in checks] == config["machine_check_contract"][
        "required_symbolic_checks"
    ]
    assert len(checks) == 33
    assert all(row["passed"] for row in checks)
    assert expressions["d3_transverse"] == "0"


def test_exact_action_expansion_contains_committed_k_term() -> None:
    _, expressions = cutoff.symbolic_checks(cutoff.load_config(ROOT))
    assert "Lambda0sq" not in expressions["L2"]
    assert "t" in expressions["L2"]
    assert "Lambda0sq" in expressions["L3"]
    assert "Lambda0sq**2" in expressions["L4"]
    assert "K" in cutoff.load_config(ROOT)["frozen_action_expansion"]["committed_scalar_density"]


def test_canonical_coefficients_match_closed_forms() -> None:
    _, expressions = cutoff.symbolic_checks(cutoff.load_config(ROOT))
    s = sp.Symbol("s", positive=True)
    locals_map = {"s": s}
    d3 = sp.sympify(expressions["d3_longitudinal"], locals=locals_map)
    d4l = sp.sympify(expressions["d4_longitudinal"], locals=locals_map)
    d4t = sp.sympify(expressions["d4_transverse"], locals=locals_map)
    expected_d3 = 1 / (3 * sp.sqrt(2) * s ** sp.Rational(3, 2) * (1 - s) * sp.sqrt(1 - 2 * s))
    assert sp.simplify(d3 - expected_d3) == 0
    assert sp.simplify(d4l - 5 / (8 * s**2 * (1 - s) * (1 - 2 * s))) == 0
    assert sp.simplify(d4t - (1 + s) / (8 * s**3 * (1 - 2 * s))) == 0


def test_numeric_cases_are_positive_finite_interior_probes() -> None:
    rows = cutoff.numeric_checks(cutoff.load_config(ROOT))
    assert [row["s"] for row in rows] == ["1/100", "1/10", "1/6", "1/4", "49/100"]
    assert all(row["passed"] for row in rows)
    assert all(row["d3_transverse"] == 0.0 for row in rows)
    assert all(row["lambda_coefficient_over_Lambda0"] > 0 for row in rows)


def test_endpoint_scales_collapse_but_not_at_fixed_interior_points() -> None:
    _, expressions = cutoff.symbolic_checks(cutoff.load_config(ROOT))
    s = sp.Symbol("s", positive=True)
    locals_map = {"s": s}
    scales = [
        sp.sympify(expressions[key], locals=locals_map)
        for key in (
            "lambda3_longitudinal_over_Lambda0",
            "lambda4_longitudinal_over_Lambda0",
            "lambda4_transverse_over_Lambda0",
        )
    ]
    assert min(float(value.subs(s, sp.Rational(1, 4))) for value in scales) > 0
    assert sp.limit(scales[2], s, 0, dir="+") == 0
    endpoint = sp.Symbol("endpoint", positive=True)
    assert sp.limit(scales[1].subs(s, sp.Rational(1, 2) - endpoint), endpoint, 0) == 0


def test_predecessor_commits_and_bytes_validate() -> None:
    config = cutoff.load_config(ROOT)
    rows = cutoff.validate_predecessors(config, ROOT)
    assert [row["git_commit"] for row in rows] == [
        "38b1d4a66f79a6275098e4505b6fb55275079378",
        "c37326ebb479bf1182419a2e752c1962db2d0056",
    ]
    assert all(row["valid"] for row in rows)
    assert sum(row["artifact_count"] for row in rows) == 8


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("frozen_action_expansion", "committed_scalar_density", "forged"),
        ("canonicalization_contract", "canonical_field", "forged"),
        ("local_coefficient_scale_contract", "transverse_quartic_coefficient", "forged"),
        ("endpoint_contract", "uniformity_result", "forged"),
        ("cherenkov_comparison_contract", "conditional_substitution", "forged"),
        ("adjudication", "physical_UV_cutoff_established", True),
        ("claim_boundary", "strong_coupling_scale_established", True),
        ("zero_access_and_compute", "cosmic_ray_rows_opened", 1),
    ],
)
def test_nested_contract_mutations_fail_closed(section: str, key: str, value: object) -> None:
    config = cutoff.load_config(ROOT)
    forged = copy.deepcopy(config)
    forged[section][key] = value
    with pytest.raises(cutoff.QuadratureScalarLocalCutoffError):
        cutoff.validate_config(forged, ROOT)


def test_predecessor_and_source_context_mutations_fail_closed() -> None:
    for section in ("predecessor_bindings", "primary_source_context"):
        config = cutoff.load_config(ROOT)
        forged = copy.deepcopy(config)
        forged[section][0]["binding_id" if section == "predecessor_bindings" else "scope"] = (
            "forged"
        )
        with pytest.raises(cutoff.QuadratureScalarLocalCutoffError):
            cutoff.validate_config(forged, ROOT)


def test_machine_inventory_mutation_fails_closed() -> None:
    config = cutoff.load_config(ROOT)
    forged = copy.deepcopy(config)
    forged["machine_check_contract"]["required_symbolic_checks"].pop()
    with pytest.raises(cutoff.QuadratureScalarLocalCutoffError):
        cutoff.validate_config(forged, ROOT)


def test_primary_source_scope_is_restrained() -> None:
    sources = cutoff.load_config(ROOT)["primary_source_context"]
    assert [row["url"] for row in sources] == [
        "https://arxiv.org/abs/hep-th/0602178",
        "https://arxiv.org/abs/2109.10812",
    ]
    assert "No positivity or UV-completion result" in sources[0]["scope"]
    assert "supplies no cutoff" in sources[1]["scope"]


def test_build_receipt_is_zero_access_and_restrained() -> None:
    receipt = cutoff.build_receipt(ROOT)
    assert receipt["adjudication"]["exact_derivative_leading_scalar_expansion_through_quartic"]
    assert receipt["adjudication"]["full_scalar_fluctuation_action_through_quartic"] is False
    assert receipt["adjudication"]["physical_UV_cutoff_established"] is False
    assert receipt["adjudication"]["cosmic_ray_survival_test_passed"] is False
    assert receipt["claim_boundary"]["strong_coupling_scale_established"] is False
    assert receipt["counts"]["symbolic_checks"] == 33
    assert receipt["counts"]["numeric_cases"] == 5
    assert all(value == 0 for value in receipt["zero_access_and_compute"].values())


def test_stored_receipt_matches_exact_rebuild() -> None:
    stored = cutoff.check_receipt(ROOT)
    rebuilt = cutoff.build_receipt(ROOT)
    assert stored == rebuilt
    assert stored["content_sha256"] == cutoff._content_sha(
        {key: value for key, value in stored.items() if key != "content_sha256"}
    )


def test_tampered_receipt_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    receipt = cutoff.build_receipt(ROOT)
    receipt["adjudication"]["physical_UV_cutoff_established"] = True
    payload = dict(receipt)
    payload.pop("content_sha256")
    receipt["content_sha256"] = cutoff._content_sha(payload)
    output = tmp_path / "receipt.json"
    output.write_text(json.dumps(receipt, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    monkeypatch.setattr(cutoff, "OUTPUT_PATH", output.relative_to(tmp_path))
    with pytest.raises(cutoff.QuadratureScalarLocalCutoffError):
        cutoff.check_receipt(tmp_path)


def test_atomic_no_clobber_preserves_different_bytes(tmp_path: Path) -> None:
    target = tmp_path / "receipt.json"
    target.write_bytes(b"existing")
    with pytest.raises(cutoff.QuadratureScalarLocalCutoffError):
        cutoff._atomic_no_clobber(target, b"new")
    assert target.read_bytes() == b"existing"


def test_atomic_no_clobber_accepts_identical_bytes(tmp_path: Path) -> None:
    target = tmp_path / "receipt.json"
    target.write_bytes(b"same")
    assert cutoff._atomic_no_clobber(target, b"same") == "EXISTING_IDENTICAL"
    assert target.read_bytes() == b"same"


def test_atomic_no_clobber_race_retains_one_complete_payload(tmp_path: Path) -> None:
    target = tmp_path / "receipt.json"

    def publish(payload: bytes) -> str:
        try:
            return cutoff._atomic_no_clobber(target, payload)
        except cutoff.QuadratureScalarLocalCutoffError:
            return "REFUSED"

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(publish, (b"first", b"second")))
    assert target.read_bytes() in {b"first", b"second"}
    assert outcomes.count("CREATED") == 1
    assert outcomes.count("REFUSED") == 1


def test_status_reports_coefficient_ceiling_not_physical_cutoff() -> None:
    result = cutoff.status(ROOT)
    assert result["valid"] is True
    assert result["derivative_leading_expansion_derived"] is True
    assert result["local_coefficient_scale_derived"] is True
    assert result["physical_cutoff_established"] is False
    assert result["uniform_positive_scale"] is False
    assert result["observational_rows_opened"] == 0
