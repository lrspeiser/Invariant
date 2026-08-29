from __future__ import annotations

import copy
import shutil
from pathlib import Path

import pytest

from sigma_theory_compiler import gravity_matter_lensing_symbolic_derivation as derivation


def test_config_binds_corrected_committed_preflight_and_action() -> None:
    config = derivation.load_config()
    predecessor = config["predecessor_binding"]
    assert predecessor["git_commit"] == "27d8cae5"
    assert predecessor["receipt_content_sha256"] == (
        "e8537efee368b08e0571565d058754718bf78f70cd5d032740e8560e63cb4768"
    )
    assert "Z(u)*(X_chi-m_chi^2*chi^2/2)" in config["action_and_invariants"]["scalar_lagrangian"]


def test_generic_scalar_variation_is_machine_zero() -> None:
    suite = derivation.run_symbolic_suite()
    by_id = {item["check_id"]: item for item in suite["checks"]}
    for check_id in (
        "S1_GENERIC_DERIVATIVE_XPHI",
        "S2_GENERIC_DERIVATIVE_XCHI",
        "S3_GENERIC_DERIVATIVE_CHI",
    ):
        assert by_id[check_id]["passed"] is True
        assert by_id[check_id]["residual"] == "0"
    expressions = suite["expressions"]["generic_scalar"]
    assert "2*X_chi - chi**2*m_chi**2" in expressions["dL_dX_phi"]
    assert "chi*m_chi**2" in expressions["dL_dchi"]


def test_constant_background_kinetic_gradient_and_mass_blocks_are_checked() -> None:
    suite = derivation.run_symbolic_suite()
    by_id = {item["check_id"]: item for item in suite["checks"]}
    for check_id in (
        "S4_TIME_KINETIC_PHI",
        "S5_TIME_KINETIC_CHI",
        "S6_TIME_KINETIC_CROSS_ZERO",
        "S7_GRADIENT_PHI",
        "S8_GRADIENT_CHI",
        "S9_GRADIENT_CROSS_ZERO",
        "S10_MASS_PHI",
        "S11_MASS_CHI",
        "S12_MASS_CROSS_ZERO",
    ):
        assert by_id[check_id]["passed"] is True
        assert by_id[check_id]["residual"] == "0"
    blocks = suite["expressions"]["constant_background"]
    assert blocks["time_kinetic_matrix"] == "diag(P_y+2*y*P_yy,Z_bar)"
    assert blocks["gradient_matrix"] == "diag(P_y,Z_bar)"
    assert blocks["mass_matrix"] == "diag(V_phi,phiphi,Z_bar*m_chi^2)"
    assert "v_chi*v_phi" in blocks["general_velocity_cross_derivative"]


def test_reduced_noether_identity_is_not_mislabelled_covariant() -> None:
    suite = derivation.run_symbolic_suite()
    check = next(
        item for item in suite["checks"] if item["check_id"] == "S13_REDUCED_NOETHER_IDENTITY"
    )
    assert check["passed"] is True
    assert check["residual"] == "0"
    assert "not the four-dimensional diffeomorphism identity" in check["detail"]
    config = derivation.load_config()
    assert config["adjudication_contract"]["H2_covariant_noether_identity"].startswith("BLOCKED_")


def test_bounded_euler_operators_include_potential_and_explicit_q_sign() -> None:
    suite = derivation.run_symbolic_suite()
    by_id = {item["check_id"]: item for item in suite["checks"]}
    for check_id in ("S19_BOUNDED_0P1_EULER_PHI", "S20_BOUNDED_0P1_EULER_CHI"):
        assert by_id[check_id]["passed"] is True
        assert by_id[check_id]["residual"] == "0"
        assert "covariant-oriented operator" in by_id[check_id]["detail"]
    equations = suite["expressions"]["bounded_0p1_euler"]
    assert equations["source_definition"] == "Q_i=-delta(L_m)/delta(field_i)"
    assert "V_phi,phi" in equations["phi_covariant_oriented_equation"]
    assert "+Z*m_chi^2*chi" in equations["chi_covariant_oriented_equation"]
    assert equations["general_covariant_equations_machine_verified"] == "false"
    config = derivation.load_config()
    assert (
        config["expected_equations"]["general_covariant_field_equations_machine_verified"] is False
    )
    assert config["adjudication_contract"]["H2_general_covariant_scalar_equations"] == (
        "UNVERIFIED_STORED_CONTRACT_ONLY"
    )


def test_weak_field_conformal_cancellation_and_disformal_scope() -> None:
    suite = derivation.run_symbolic_suite()
    by_id = {item["check_id"]: item for item in suite["checks"]}
    assert by_id["S14_LINEAR_CONFORMAL_CANCELLATION"]["passed"] is True
    assert by_id["S15_DISFORMAL_LINEAR_TERM_ZERO"]["passed"] is True
    weak = suite["expressions"]["weak_field"]
    assert weak == {
        "Phi": "Phi_E + a",
        "Psi": "Psi_E - a",
        "lensing_sum": "Phi_E + Psi_E",
    }
    assert derivation.load_config()["adjudication_contract"][
        "H9_joint_matter_lensing_completion"
    ].startswith("BLOCKED_")


def test_green_function_has_fixed_range_and_gated_amplitude_only_locally() -> None:
    suite = derivation.run_symbolic_suite()
    by_id = {item["check_id"]: item for item in suite["checks"]}
    for check_id in (
        "S16_GREEN_HELMHOLTZ_R_GT_ZERO",
        "S17_GREEN_FIXED_RANGE",
        "S18_GATE_AMPLITUDE_IDENTITY",
    ):
        assert by_id[check_id]["passed"] is True
    green = suite["expressions"]["green_function"]
    assert green["radial_kernel_outside_source"] == "exp(-m_chi*r)/r"
    assert green["inverse_range"] == "m_chi"
    assert green["source_amplitude"] == "(u + 1)**(-2)"
    assumption = derivation.load_config()["frozen_assumptions"]["green_function"]
    assert "locally constant" in assumption
    assert "contact delta" in assumption


def test_all_symbolic_and_independent_numeric_checks_pass() -> None:
    symbolic = derivation.run_symbolic_suite()
    numeric = derivation.run_numeric_suite(derivation.load_config())
    assert tuple(item["check_id"] for item in symbolic["checks"]) == derivation.SYMBOLIC_CHECK_IDS
    assert tuple(item["check_id"] for item in numeric["checks"]) == derivation.NUMERIC_CHECK_IDS
    assert symbolic["all_passed"] is True
    assert numeric["all_passed"] is True
    assert float(numeric["maximum_scaled_error"]) < 1e-9
    lens = next(
        item for item in numeric["checks"] if item["check_id"] == "N6_CONFORMAL_LENSING_SUM"
    )
    assert float(lens["inputs"]["a"]) != 0.0
    assert float(lens["derived"]["Phi"]) == pytest.approx(
        float(lens["inputs"]["Phi_E"]) + float(lens["inputs"]["a"])
    )
    assert float(lens["derived"]["Psi"]) == pytest.approx(
        float(lens["inputs"]["Psi_E"]) - float(lens["inputs"]["a"])
    )


def test_adjudication_keeps_full_h2_h3_h4_h9_blocked() -> None:
    config = derivation.load_config()
    adjudication = config["adjudication_contract"]
    assert adjudication["H2_scalar_lagrangian_coefficient_identities"].startswith(
        "PASS_MACHINE_SYMBOLIC_"
    )
    assert adjudication["H2_bounded_scalar_euler_lagrange"].startswith("PASS_MACHINE_SYMBOLIC_0P1_")
    assert adjudication["H2_general_covariant_scalar_equations"] == (
        "UNVERIFIED_STORED_CONTRACT_ONLY"
    )
    assert adjudication["H3_constant_background_scalar_kinetic_form"].startswith(
        "PASS_MACHINE_SYMBOLIC_"
    )
    assert adjudication["H4_constant_background_scalar_principal_block"].startswith(
        "PASS_MACHINE_SYMBOLIC_"
    )
    for key in (
        "H2_full_metric_variation",
        "H2_covariant_noether_identity",
        "H2_hamiltonian_constraints",
        "H3_no_ghost_health",
        "H4_global_hyperbolicity",
        "H9_joint_matter_lensing_completion",
    ):
        assert adjudication[key].startswith("BLOCKED_")
    assert adjudication["overall_decision"] == derivation.DECISION
    assert adjudication["H3_chi_bar_zero_background_status"].startswith(
        "ON_SHELL_ONLY_IF_Q_CHI_ZERO_"
    )
    background = config["frozen_assumptions"]["quadratic_background"]
    assert "Q_chi=0" in background
    assert "off-shell Hessians" in background
    assert set(config["claim_boundary"].values()) == {False}
    assert set(config["zero_access_and_compute"].values()) == {0}


@pytest.mark.parametrize(
    ("section", "mutation"),
    [
        (
            "predecessor_binding",
            lambda value: value.__setitem__("receipt_content_sha256", "0" * 64),
        ),
        ("action_and_invariants", lambda value: value.__setitem__("Z", "tampered")),
        ("frozen_assumptions", lambda value: value.__setitem__("green_function", "global")),
        ("expected_equations", lambda value: value.__setitem__("local_chi_range", "wrong")),
        (
            "machine_check_contract",
            lambda value: value.__setitem__("numeric_absolute_tolerance", 1.0),
        ),
        (
            "adjudication_contract",
            lambda value: value.__setitem__("H2_full_metric_variation", "PASS"),
        ),
        ("claim_boundary", lambda value: value.__setitem__("full_H2_passed", True)),
        (
            "zero_access_and_compute",
            lambda value: value.__setitem__("observational_files_opened", 1),
        ),
    ],
)
def test_nested_contract_mutations_fail_closed(section: str, mutation: object) -> None:
    config = copy.deepcopy(derivation.load_config())
    mutation(config[section])  # type: ignore[operator]
    with pytest.raises(
        derivation.GravityMatterLensingSymbolicDerivationError, match="content changed"
    ):
        derivation.validate_config(config)


def test_predecessor_tampering_fails_closed(tmp_path: Path) -> None:
    config = derivation.load_config()
    binding = config["predecessor_binding"]
    for path_key in ("config_path", "module_path", "test_path", "receipt_path"):
        source = Path(binding[path_key])
        destination = tmp_path / source
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    target = tmp_path / binding["receipt_path"]
    target.write_bytes(target.read_bytes() + b" ")
    with pytest.raises(
        derivation.GravityMatterLensingSymbolicDerivationError, match="receipt_path"
    ):
        derivation._validate_predecessor(tmp_path, config)


def test_atomic_no_replace_preserves_existing_bytes(tmp_path: Path) -> None:
    target = tmp_path / "receipt.json"
    target.write_bytes(b"winner")
    with pytest.raises(derivation.GravityMatterLensingSymbolicDerivationError, match="refusing"):
        derivation._atomic_no_replace(target, b"candidate")
    assert target.read_bytes() == b"winner"
    assert derivation._atomic_no_replace(target, b"winner") == "EXISTING_IDENTICAL"


def test_receipt_binds_implementation_and_exact_check_counts() -> None:
    receipt = derivation.build_receipt()
    assert receipt["implementation_binding"]["source_file_sha256"] == derivation._file_sha(
        derivation.SOURCE_PATH
    )
    assert receipt["implementation_binding"]["test_file_sha256"] == derivation._file_sha(
        derivation.TEST_PATH
    )
    assert receipt["counts"]["symbolic_checks"] == 20
    assert receipt["counts"]["symbolic_checks_passed"] == 20
    assert receipt["counts"]["independent_numeric_checks"] == 6
    assert receipt["counts"]["independent_numeric_checks_passed"] == 6


def test_stored_receipt_is_exact_deterministic_rebuild() -> None:
    stored = derivation.check_receipt()
    assert stored == derivation.build_receipt()
    assert stored["claim_boundary"]["full_H2_passed"] is False
    assert stored["counts"]["observational_files_opened"] == 0


def test_receipt_tampering_fails_closed() -> None:
    config = derivation.load_config()
    receipt = derivation.build_receipt()
    receipt["counts"]["symbolic_checks_passed"] = 17
    with pytest.raises(
        derivation.GravityMatterLensingSymbolicDerivationError, match="content hash changed"
    ):
        derivation.validate_receipt(receipt, config)


def test_module_has_no_network_or_observational_loader() -> None:
    source = derivation.SOURCE_PATH.read_text(encoding="utf-8")
    for forbidden in ("requests", "urllib", "pandas", "astropy", "fits.open", "subprocess"):
        assert forbidden not in source
