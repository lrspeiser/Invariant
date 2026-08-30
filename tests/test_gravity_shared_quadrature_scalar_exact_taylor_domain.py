from __future__ import annotations

import copy
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import sympy as sp

from sigma_theory_compiler import (
    gravity_shared_quadrature_scalar_exact_taylor_domain as taylor_domain,
)

ROOT = Path(__file__).resolve().parents[1]


def test_symbolic_derivation_passes_exact_inventory() -> None:
    config = taylor_domain.load_config(ROOT)
    checks, expressions = taylor_domain.symbolic_checks(config)
    assert [row["check_id"] for row in checks] == config["machine_check_contract"][
        "required_symbolic_checks"
    ]
    assert len(checks) == 25
    assert all(row["passed"] for row in checks)
    assert expressions["rho_zero_gradient_singularity"] == "-s"
    assert expressions["rho_outer_pole_singularity"] == "1/2 - s"
    assert expressions["upsilon_sqrt_branch"] == "-s**2"
    assert expressions["upsilon_outer_pole"] == "1/4 - s**2"


def test_exact_real_domain_and_one_variable_singularities() -> None:
    _, expressions = taylor_domain.symbolic_checks(taylor_domain.load_config(ROOT))
    s, rho, upsilon = sp.symbols("s rho upsilon", positive=True)
    argument = sp.sympify(
        expressions["s_total_squared"], locals={"s": s, "rho": rho, "upsilon": upsilon}
    )
    assert sp.simplify(argument.subs({rho: -s, upsilon: 0})) == 0
    assert sp.simplify(argument.subs({rho: sp.Rational(1, 2) - s, upsilon: 0})) == sp.Rational(1, 4)
    assert sp.simplify(argument.subs({rho: 0, upsilon: -(s**2)})) == 0
    assert sp.simplify(argument.subs({rho: 0, upsilon: sp.Rational(1, 4) - s**2})) == sp.Rational(
        1, 4
    )


def test_canonical_radius_closed_forms_and_crossover() -> None:
    rows = taylor_domain.numeric_checks(taylor_domain.load_config(ROOT))
    assert [row["s"] for row in rows] == ["1/100", "1/10", "1/6", "1/4", "49/100"]
    assert all(row["passed"] for row in rows)
    assert all(row["upsilon_limiting_singularity"] == "complex_sqrt_branch" for row in rows[:-1])
    assert rows[-1]["upsilon_limiting_singularity"] == "outer_pK_pole"
    assert rows[0]["rho_limiting_singularity"] == "zero_gradient_kink"
    assert rows[-1]["rho_limiting_singularity"] == "outer_pK_pole"
    assert all(
        row["canonical_transverse_radius"] <= row["canonical_real_transverse_pole_radius"] + 1e-12
        for row in rows
    )


def test_endpoint_limits_and_quartic_consistency_are_exact() -> None:
    config = taylor_domain.load_config(ROOT)
    checks, _ = taylor_domain.symbolic_checks(config)
    by_id = {row["check_id"]: row for row in checks}
    for check_id in (
        "S16_LOW_S_LONGITUDINAL_RADIUS",
        "S17_LOW_S_TRANSVERSE_RADIUS",
        "S18_HIGH_S_LONGITUDINAL_RADIUS",
        "S19_HIGH_S_TRANSVERSE_RADIUS",
        "S20_LOW_S_QUARTIC_RADIUS_CONSISTENCY",
        "S21_HIGH_S_QUARTIC_RADIUS_CONSISTENCY",
        "S22_BOTH_ENDPOINTS_COLLAPSE",
    ):
        assert by_id[check_id]["passed"] is True
    assert config["endpoint_and_quartic_consistency_contract"]["result"].endswith(
        "not created solely by truncating the action at quartic order."
    )


def test_amplitude_direction_is_entire_but_scope_remains_restricted() -> None:
    config = taylor_domain.load_config(ROOT)
    assert "entire" in config["exact_jet_domain_contract"]["amplitude_direction"]
    assert config["adjudication"]["full_coupled_metric_aether_matter_analyticity_domain"] is False
    assert config["adjudication"]["physical_UV_cutoff_established"] is False
    assert config["adjudication"]["tree_level_unitarity_bound_established"] is False
    assert config["claim_boundary"]["healthy_action_established"] is False


def test_predecessor_commit_and_bytes_validate() -> None:
    rows = taylor_domain.validate_predecessors(taylor_domain.load_config(ROOT), ROOT)
    assert [row["git_commit"] for row in rows] == ["c2380194b35d317c8945f77255ba0370d918318d"]
    assert sum(row["artifact_count"] for row in rows) == 4
    assert all(row["valid"] for row in rows)


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("exact_jet_domain_contract", "exact_argument", "forged"),
        ("canonical_radius_contract", "longitudinal_radius", "forged"),
        ("endpoint_and_quartic_consistency_contract", "result", "forged"),
        ("adjudication", "physical_UV_cutoff_established", True),
        ("claim_boundary", "tree_unitarity_established", True),
        ("zero_access_and_compute", "observational_rows_opened", 1),
    ],
)
def test_nested_contract_mutations_fail_closed(section: str, key: str, value: object) -> None:
    forged = copy.deepcopy(taylor_domain.load_config(ROOT))
    forged[section][key] = value
    with pytest.raises(taylor_domain.QuadratureScalarTaylorDomainError):
        taylor_domain.validate_config(forged, ROOT)


def test_inventory_and_predecessor_mutations_fail_closed() -> None:
    inventory = copy.deepcopy(taylor_domain.load_config(ROOT))
    inventory["machine_check_contract"]["required_symbolic_checks"].pop()
    predecessor = copy.deepcopy(taylor_domain.load_config(ROOT))
    predecessor["predecessor_bindings"][0]["git_commit"] = "0" * 40
    for forged in (inventory, predecessor):
        with pytest.raises(taylor_domain.QuadratureScalarTaylorDomainError):
            taylor_domain.validate_config(forged, ROOT)


def test_build_receipt_is_zero_access_and_restrained() -> None:
    receipt = taylor_domain.build_receipt(ROOT)
    assert receipt["adjudication"]["exact_fixed_background_real_scalar_jet_domain_derived"]
    assert receipt["adjudication"]["canonical_jet_radii_derived"]
    assert receipt["adjudication"]["endpoint_collapse_is_quartic_truncation_artifact"] is False
    assert receipt["claim_boundary"]["physical_cutoff_established"] is False
    assert receipt["counts"]["symbolic_checks"] == 25
    assert receipt["counts"]["numeric_cases"] == 5
    assert all(value == 0 for value in receipt["zero_access_and_compute"].values())


def test_stored_receipt_matches_exact_rebuild() -> None:
    stored = taylor_domain.check_receipt(ROOT)
    rebuilt = taylor_domain.build_receipt(ROOT)
    assert stored == rebuilt
    payload = dict(stored)
    content = payload.pop("content_sha256")
    assert content == taylor_domain._content_sha(payload)


def test_tampered_receipt_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    receipt = taylor_domain.build_receipt(ROOT)
    receipt["claim_boundary"]["physical_cutoff_established"] = True
    payload = dict(receipt)
    payload.pop("content_sha256")
    receipt["content_sha256"] = taylor_domain._content_sha(payload)
    output = tmp_path / "receipt.json"
    output.write_text(json.dumps(receipt, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    monkeypatch.setattr(taylor_domain, "OUTPUT_PATH", output.relative_to(tmp_path))
    with pytest.raises(taylor_domain.QuadratureScalarTaylorDomainError):
        taylor_domain.check_receipt(tmp_path)


def test_atomic_no_clobber_preserves_different_bytes(tmp_path: Path) -> None:
    target = tmp_path / "receipt.json"
    target.write_bytes(b"existing")
    with pytest.raises(taylor_domain.QuadratureScalarTaylorDomainError):
        taylor_domain._atomic_no_clobber(target, b"new")
    assert target.read_bytes() == b"existing"


def test_atomic_no_clobber_accepts_identical_bytes(tmp_path: Path) -> None:
    target = tmp_path / "receipt.json"
    target.write_bytes(b"same")
    assert taylor_domain._atomic_no_clobber(target, b"same") == "EXISTING_IDENTICAL"


def test_atomic_no_clobber_race_retains_one_complete_payload(tmp_path: Path) -> None:
    target = tmp_path / "receipt.json"

    def publish(payload: bytes) -> str:
        try:
            return taylor_domain._atomic_no_clobber(target, payload)
        except taylor_domain.QuadratureScalarTaylorDomainError:
            return "REFUSED"

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(publish, (b"first", b"second")))
    assert target.read_bytes() in {b"first", b"second"}
    assert outcomes.count("CREATED") == 1
    assert outcomes.count("REFUSED") == 1


def test_status_preserves_claim_ceiling() -> None:
    result = taylor_domain.status(ROOT)
    assert result["valid"] is True
    assert result["exact_scalar_jet_domain"] is True
    assert result["canonical_radii"] is True
    assert result["quartic_truncation_artifact"] is False
    assert result["full_coupled_domain"] is False
    assert result["physical_cutoff"] is False
    assert result["observational_rows_opened"] == 0
