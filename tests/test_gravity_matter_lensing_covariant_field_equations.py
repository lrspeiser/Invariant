from __future__ import annotations

import copy
import subprocess
from pathlib import Path

import pytest

from sigma_theory_compiler import (
    gravity_matter_lensing_covariant_field_equations as covariant,
)

ROOT = Path(__file__).resolve().parents[1]


def test_frozen_config_loads_and_has_expected_identity() -> None:
    config = covariant.load_config(ROOT)
    assert config["schema_version"] == covariant.CONFIG_SCHEMA
    assert config["analysis_id"] == "gravity-matter-lensing-covariant-field-equations-v1"
    assert config["adjudication"]["overall_decision"] == covariant.DECISION
    assert (
        covariant._file_sha(ROOT / covariant.CONFIG_PATH) == covariant.EXPECTED_CONFIG_FILE_SHA256
    )
    assert covariant._sha(config) == covariant.EXPECTED_CONFIG_CONTENT_SHA256


def test_all_frozen_sections_are_independently_bound() -> None:
    config = covariant.load_config(ROOT)
    assert set(covariant.EXPECTED_SECTION_SHA256).issubset(config)
    for section, expected in covariant.EXPECTED_SECTION_SHA256.items():
        assert covariant._sha(config[section]) == expected


def test_predecessor_files_receipts_and_commits_are_exact() -> None:
    config = covariant.load_config(ROOT)
    covariant._validate_predecessors(ROOT, config)
    for binding in config["predecessor_bindings"]:
        completed = subprocess.run(
            ["git", "cat-file", "-t", binding["git_commit"]],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        assert completed.stdout.strip() == "commit"
        for path_key in ("config_path", "module_path", "test_path", "receipt_path"):
            path = binding[path_key]
            committed = subprocess.run(
                ["git", "show", f"{binding['git_commit']}:{path}"],
                cwd=ROOT,
                check=True,
                capture_output=True,
            ).stdout
            assert committed == (ROOT / path).read_bytes()


def test_symbolic_suite_has_all_nontrivial_checks() -> None:
    result = covariant.run_symbolic_suite()
    assert result["all_passed"] is True
    assert tuple(item["check_id"] for item in result["checks"]) == covariant.SYMBOLIC_CHECK_IDS
    assert len(result["checks"]) == 21
    assert all(item["residual"] == "0" and item["passed"] is True for item in result["checks"])


def test_metric_variation_finite_difference_suite() -> None:
    result = covariant.run_numeric_suite(covariant.load_config(ROOT))
    assert result["all_passed"] is True
    assert len(result["cases"]) == 3
    assert sum(len(item["components"]) for item in result["cases"]) == 9
    assert result["max_scaled_error"] <= result["gate"]
    assert all(item["lorentzian_determinant"] < 0 for item in result["cases"])


def test_field_equations_and_exchange_are_same_action() -> None:
    config = covariant.load_config(ROOT)
    field = config["covariant_field_equation_contract"]
    exchange = config["same_action_exchange_contract"]
    conventions = config["action_and_variation_conventions"]
    assert "M_Pl^2*G_munu=T_E,munu+T_s,munu" == field["einstein_equation"]
    assert "E_phi=nabla_mu(C*nabla^mu(phi))-V_phi=Q_phi" == field["phi_eom"]
    assert "E_chi=Y0*box(chi)-m_chi^2*Z*chi=Q_chi" == field["chi_eom"]
    assert "=-Q_phi*partial_nu(phi)-Q_chi*partial_nu(chi)" in exchange["matter_identity"]
    assert "No independent photon metric" in conventions["forbidden"]


def test_flrw_regression_is_exactly_retained() -> None:
    regression = covariant.load_config(ROOT)["flrw_regression_contract"]
    assert regression == {
        "homogeneous_energy_density": "rho_s=2*X*C-P+V+Y0*X_chi+Q*Z",
        "homogeneous_pressure": "p_s=L_s=P-V+Y0*X_chi-Q*Z",
        "homogeneous_enthalpy": "rho_s+p_s=2*X*C+2*Y0*X_chi",
        "regression_requirement": "All three expressions must equal the committed exact flat-FLRW package after specialization of the covariant stress tensor.",
    }


def test_claim_ceiling_remains_partial() -> None:
    config = covariant.load_config(ROOT)
    adjudication = config["adjudication"]
    claims = config["claim_boundary"]
    assert adjudication["scalar_metric_variation_derived"] is True
    assert adjudication["same_action_exchange_identity_derived"] is True
    assert adjudication["full_H2"] is False
    assert adjudication["full_H3"] is False
    assert adjudication["full_H4"] is False
    assert adjudication["metric_backreaction_solved"] is False
    assert claims["covariant_scalar_stress_and_exchange_established"] is True
    assert claims["formal_same_action_field_equation_contract_established"] is True
    assert claims["closed_healthy_theory_established"] is False
    assert claims["lensing_success_established"] is False
    assert claims["scientific_observational_claim_allowed"] is False


def test_zero_access_ledger_is_literal_zero() -> None:
    access = covariant.load_config(ROOT)["zero_access_and_compute"]
    assert set(access) == covariant.ZERO_KEYS
    assert all(type(value) is int and value == 0 for value in access.values())


@pytest.mark.parametrize(
    ("section", "mutator"),
    [
        (
            "covariant_field_equation_contract",
            lambda value: value.__setitem__("einstein_equation", "M_Pl^2*G_munu=0"),
        ),
        (
            "same_action_exchange_contract",
            lambda value: value.__setitem__("on_shell_total", "nabla_mu(T_E^mu_nu)=0"),
        ),
        (
            "flrw_regression_contract",
            lambda value: value.__setitem__("homogeneous_pressure", "p_s=0"),
        ),
        (
            "machine_check_contract",
            lambda value: value.__setitem__("max_scaled_error", 1.0),
        ),
        (
            "adjudication",
            lambda value: value.__setitem__("full_H2", True),
        ),
        (
            "claim_boundary",
            lambda value: value.__setitem__("lensing_success_established", True),
        ),
        (
            "zero_access_and_compute",
            lambda value: value.__setitem__("observational_rows_opened", 1),
        ),
    ],
)
def test_nested_semantic_mutations_fail_even_if_global_hash_is_rebound(
    monkeypatch: pytest.MonkeyPatch, section: str, mutator: object
) -> None:
    config = copy.deepcopy(covariant.load_config(ROOT))
    mutator(config[section])  # type: ignore[operator]
    monkeypatch.setattr(covariant, "EXPECTED_CONFIG_CONTENT_SHA256", covariant._sha(config))
    with pytest.raises(covariant.CovariantFieldEquationsError, match="frozen section changed"):
        covariant.validate_config(config)


def test_predecessor_mutation_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    config = copy.deepcopy(covariant.load_config(ROOT))
    config["predecessor_bindings"][0]["git_commit"] = "0" * 40
    monkeypatch.setattr(covariant, "EXPECTED_CONFIG_CONTENT_SHA256", covariant._sha(config))
    monkeypatch.setitem(
        covariant.EXPECTED_SECTION_SHA256,
        "predecessor_bindings",
        covariant._sha(config["predecessor_bindings"]),
    )
    with pytest.raises(covariant.CovariantFieldEquationsError, match="predecessor commits changed"):
        covariant.validate_config(config)


def test_receipt_rebuild_is_deterministic_and_binds_implementation() -> None:
    first = covariant.build_receipt(ROOT)
    second = covariant.build_receipt(ROOT)
    assert first == second
    assert first["content_sha256"] == covariant._sha(
        {key: value for key, value in first.items() if key != "content_sha256"}
    )
    binding = first["implementation_binding"]
    assert binding["source_file_sha256"] == covariant._file_sha(ROOT / covariant.SOURCE_PATH)
    assert binding["test_file_sha256"] == covariant._file_sha(ROOT / covariant.TEST_PATH)
    assert first["counts"]["symbolic_checks_passed"] == 21
    assert first["counts"]["numeric_cases_passed"] == 3


def test_forged_receipt_claim_is_rejected() -> None:
    config = covariant.load_config(ROOT)
    receipt = covariant.build_receipt(ROOT)
    receipt["claim_boundary"]["lensing_success_established"] = True
    body = {key: value for key, value in receipt.items() if key != "content_sha256"}
    receipt["content_sha256"] = covariant._sha(body)
    with pytest.raises(covariant.CovariantFieldEquationsError, match="receipt contract changed"):
        covariant.validate_receipt(receipt, config)


def test_atomic_publication_is_idempotent_and_no_clobber(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "receipt.json"
    payload = b'{"safe":true}\n'
    assert covariant._atomic_no_replace(output, payload) == "CREATED"
    assert covariant._atomic_no_replace(output, payload) == "EXISTING_IDENTICAL"
    with pytest.raises(covariant.CovariantFieldEquationsError, match="refusing to overwrite"):
        covariant._atomic_no_replace(output, b'{"safe":false}\n')
    assert output.read_bytes() == payload


def test_source_has_no_data_or_network_loader() -> None:
    source = (ROOT / covariant.SOURCE_PATH).read_text(encoding="utf-8")
    forbidden = ("requests", "urllib", "httpx", "pandas", "astropy.io", "fits.open", "cupy")
    assert all(token not in source for token in forbidden)


def test_stored_receipt_matches_exact_rebuild_after_write() -> None:
    stored = covariant.check_receipt(ROOT)
    rebuilt = covariant.build_receipt(ROOT)
    assert stored == rebuilt
    assert stored["decision"] == covariant.DECISION
    assert stored["counts"]["observational_rows_opened"] == 0
