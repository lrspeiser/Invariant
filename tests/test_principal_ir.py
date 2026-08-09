from __future__ import annotations

import copy
from pathlib import Path

from sigma_theory_compiler.action_ir import compile_action_file
from sigma_theory_compiler.adm_ir import compile_adm_ir
from sigma_theory_compiler.cli import _parser
from sigma_theory_compiler.dirac_ir import compile_dirac_ir
from sigma_theory_compiler.legendre_ir import compile_legendre_ir
from sigma_theory_compiler.principal_ir import compile_physical_principal_ir
from sigma_theory_compiler.stability_ir import compile_stability_ir

ROOT = Path(__file__).resolve().parents[1]
GRAMMAR = ROOT / "configs" / "covariant_action_grammar.json"
CONTRACT = ROOT / "configs" / "covariant_field_contract.json"


def _controls() -> dict[str, bool]:
    return {
        name: True
        for name in (
            "cadabra_adm_spatial_curvature_variation",
            "nonlinear_adm_hamiltonian_constraint_algebra",
            "canonical_metric_diffeomorphism_algebra",
            "canonical_metric_dewitt_kinetic_covariance",
            "spatial_curvature_density_diffeomorphism_covariance",
            "canonical_scalar",
            "canonical_scalar_noether_identity",
            "canonical_scalar_gravity_cross_constraint_identities",
            "three_spatial_dimensional_smeared_brackets",
            "proca_adm_dirac",
            "proca_divergence_identity",
            "proca_stress_noether_identity",
            "proca_reduced_smeared_constraint_algebra",
            "einstein_aether_generic_3plus1_legendre",
            "einstein_aether_generic_lapse_shift_constraint_seeds",
            "einstein_aether_spatial_diffeomorphism_algebra",
            "einstein_aether_generic_dh_covariance",
            "einstein_aether_generic_hh_deformation_kinematics",
            "einstein_aether_arbitrary_background_4d_noether",
            "regular_holonomic_multiplier_dirac_theorem",
            "unit_timelike_vector_dirac_chain",
            "einstein_hilbert_linearized_adm",
            "principal_symbol_controls",
            "curved_background_principal_controls",
            "einstein_aether_linearized_physical_energy",
            "einstein_aether_restricted_nonlinear_total_energy",
            "einstein_aether_reduced_five_mode_principal_domain",
            "einstein_aether_global_tilt_legendre_strata",
            "einstein_aether_covariant_arbitrary_background_hyperbolicity",
            "quartic_horndeski_covariant_adm_degeneracy",
            "quartic_horndeski_unitary_flrw_dirac_chain",
            "quartic_horndeski_unitary_distributed_dirac_closure",
            "quartic_horndeski_timelike_flat_principal_symbol",
            "quartic_horndeski_arbitrary_curvature_scalar_principal",
            "quartic_horndeski_coupled_formulation_hyperbolicity",
            "quartic_horndeski_full_local_principal_extraction",
            "quartic_horndeski_global_timelike_gradient_no_go",
        )
    }


def _compile(name: str) -> tuple[dict, dict, dict, dict]:
    controls = _controls()
    action = compile_action_file(ROOT / "configs" / "actions" / name, GRAMMAR, CONTRACT)
    adm = compile_adm_ir(action, controls)
    legendre = compile_legendre_ir(action, adm)
    dirac = compile_dirac_ir(action, adm, legendre, controls)
    stability = compile_stability_ir(action, dirac, controls)
    return action, dirac, stability, compile_physical_principal_ir(action, dirac, stability)


def test_eh_scalar_and_proca_principal_ir_retains_only_physical_modes() -> None:
    expected = {
        "einstein_hilbert_control.json": (2, {"tensor_speed_squared": "1"}),
        "canonical_scalar_control.json": (
            3,
            {"scalar_speed_squared": "1", "tensor_speed_squared": "1"},
        ),
        "proca_control.json": (
            5,
            {"proca_speed_squared": "1", "tensor_speed_squared": "1"},
        ),
    }
    for name, (mode_count, speeds) in expected.items():
        action, dirac, stability, result = _compile(name)
        assert result == compile_physical_principal_ir(action, dirac, stability)
        assert result["status"] == "pass"
        reduction = result["gauge_reduction_certificate"]
        assert reduction["physical_dof_from_constraint_surface"] == mode_count
        assert reduction["retained_mode_count"] == mode_count
        assert reduction["constrained_or_gauge_variables_retained"] == []
        assert result["characteristic_speed_squared"] == speeds
        assert result["propagation_residual"] == f"Matrix({[[0] * mode_count] * mode_count})"


def test_aether_principal_ir_is_action_bound_and_keeps_all_five_speeds() -> None:
    action, dirac, stability, result = _compile("einstein_aether_control.json")
    assert result["status"] == "pass"
    assert result["input_action_sha256"] == action["content_sha256"]
    assert result["input_dirac_ir_sha256"] == dirac["content_sha256"]
    assert result["input_stability_ir_sha256"] == stability["content_sha256"]
    assert result["gauge_reduction_certificate"]["retained_physical_basis"] == [
        "spin_2_plus",
        "spin_2_cross",
        "spin_1_x",
        "spin_1_y",
        "spin_0",
    ]
    assert set(result["characteristic_speed_squared"]) == {
        "spin_0_speed_squared",
        "spin_1_speed_squared",
        "spin_2_speed_squared",
    }
    assert result["exact_diagonal_physical_eigenbasis"]
    assert result["cone_policy"].endswith("without imposing an observational subluminality cut")


def test_horndeski_local_patch_symbol_passes_but_coupled_background_stays_open() -> None:
    action, _dirac, stability, result = _compile("quartic_horndeski_control.json")
    assert result["status"] == "unresolved"
    assert result["input_action_sha256"] == action["content_sha256"]
    assert result["gauge_reduction_certificate"]["physical_dof_from_constraint_surface"] == 3
    assert result["declared_background_patch_certificate"]["status"] == "pass"
    assert result["pointwise_domain_certificate_status"] == "pass"
    assert result["source_principal_certificate_status"] == "unresolved"
    assert not stability["principal_symbol"]["generic_coupled_background_supported"]
    assert stability["principal_symbol"]["coupled_formulation"][
        "generalized_harmonic"
    ] == "reject"


def test_principal_ir_rejects_wrong_sign_domain_and_broken_hash_chain() -> None:
    action, dirac, stability, _ = _compile("canonical_scalar_control.json")
    wrong_sign = copy.deepcopy(stability)
    wrong_sign["condition_certificate"]["status"] = "reject"
    wrong_sign["principal_symbol"]["status"] = "reject"
    rejected = compile_physical_principal_ir(action, dirac, wrong_sign)
    assert rejected["status"] == "reject"

    wrong_hash = copy.deepcopy(stability)
    wrong_hash["input_dirac_ir_sha256"] = "0" * 64
    broken = compile_physical_principal_ir(action, dirac, wrong_hash)
    assert broken["status"] == "reject"
    assert broken["errors"] == ["stability IR belongs to a different Dirac hash"]


def test_principal_ir_stays_unresolved_without_a_completed_dirac_reduction() -> None:
    action, dirac, stability, _ = _compile("einstein_hilbert_control.json")
    unresolved_dirac = copy.deepcopy(dirac)
    unresolved_dirac["status"] = "unresolved"
    unresolved_dirac["content_sha256"] = "unresolved-dirac"
    unresolved_stability = copy.deepcopy(stability)
    unresolved_stability["input_dirac_ir_sha256"] = "unresolved-dirac"
    result = compile_physical_principal_ir(
        action, unresolved_dirac, unresolved_stability
    )
    assert result["status"] == "unresolved"
    assert result["gauge_reduction_certificate"]["status"] == "unresolved"


def test_formal_ir_cli_stages_have_explicit_artifact_contracts() -> None:
    for command in ("action-stability", "action-principal", "action-hamiltonian"):
        parsed = _parser().parse_args(
            [command, "--spec", "action.json", "--output", f"{command}.json"]
        )
        assert parsed.command == command
        assert parsed.grammar == Path("configs/covariant_action_grammar.json")
        assert parsed.contract == Path("configs/covariant_field_contract.json")
