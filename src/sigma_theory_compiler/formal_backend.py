from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import sympy as sp

from .adm import (
    linearized_einstein_hilbert_adm_control,
    nonlinear_adm_hamiltonian_constraint_control,
    spatial_curvature_density_diffeomorphism_control,
)
from .adm_aether import (
    einstein_aether_3plus1_decomposition_control,
    einstein_aether_adm_kinetic_control,
    einstein_aether_coupled_unit_normal_control,
    einstein_aether_covariant_strong_hyperbolicity_control,
    einstein_aether_generic_dh_covariance_control,
    einstein_aether_generic_hh_deformation_control,
    einstein_aether_global_tilt_legendre_control,
    einstein_aether_lapse_shift_constraint_seed_control,
    einstein_aether_linearized_energy_control,
    einstein_aether_nonlinear_positive_energy_theorem_control,
    einstein_aether_reduced_principal_domain_control,
    einstein_aether_spatial_diffeomorphism_control,
    maxwell_unit_aether_nonlinear_hamiltonian_control,
    unit_timelike_vector_dirac_chain_control,
)
from .backgrounds import curved_background_principal_controls
from .covariant_identities import (
    einstein_aether_flrw_variation_control,
    proca_curved_background_noether_controls,
)
from .cubic_bssn_domain import (
    certify_cubic_bssn_domain,
    generic_cubic_scalar_effective_metric_control,
    run_cubic_bssn_domain_campaign,
)
from .dhost import (
    dhost_reduced_dirac_control,
    generic_horndeski_l2_l4_unitary_adm_control,
    quartic_horndeski_covariant_adm_control,
    quartic_horndeski_unitary_flrw_dirac_control,
)
from .dirac import (
    analyze_quadratic_lagrangian,
    proca_fourier_dirac_control,
    reduce_poisson_matrix_on_constraint_surface,
    regular_holonomic_multiplier_dirac_control,
)
from .field_dirac import (
    canonical_metric_dewitt_kinetic_control,
    canonical_metric_diffeomorphism_control,
    canonical_scalar_spatial_density_certificate,
    proca_reduced_smeared_constraint_control,
    three_dimensional_smeared_bracket_control,
    virasoro_constraint_algebra_control,
)
from .flrw_background import certify_flrw_background
from .g4_curved_witness import (
    generic_g4_curved_rnc_witness_control,
    generic_g4_curved_symbolic_rnc_control,
)
from .horndeski import (
    generic_cubic_horndeski_bssn_hyperbolicity_control,
    generic_horndeski_l2_l4_flrw_scalar_reduction_control,
    generic_horndeski_l2_l4_tensor_stability_control,
    generic_horndeski_l2_l4_unitary_dirac_control,
    generic_kessence_nonlinear_adm_legendre_control,
    generic_kessence_timelike_principal_hamiltonian_control,
    quartic_horndeski_arbitrary_curvature_scalar_principal_control,
    quartic_horndeski_boundary_and_flrw_noether_control,
    quartic_horndeski_coupled_formulation_hyperbolicity_control,
    quartic_horndeski_flrw_domain_crossing_control,
    quartic_horndeski_global_timelike_gradient_no_go_control,
    quartic_horndeski_scalar_euler_reduction_control,
    quartic_horndeski_timelike_flat_hamiltonian_control,
    quartic_horndeski_timelike_flat_principal_control,
    quartic_horndeski_unitary_distributed_dirac_control,
)
from .horndeski_principal import (
    quartic_horndeski_baseline_riesz_symmetrizer_control,
    quartic_horndeski_full_local_principal_control,
)
from .principal_symbol import (
    run_anisotropic_principal_symbol_controls,
    run_extracted_principal_symbol_controls,
    run_principal_symbol_controls,
    run_uniform_multifield_block_controls,
    run_uniform_scalar_anisotropy_controls,
)
from .q_adm import projected_aether_q_3plus1_control
from .q_dirac import projected_aether_q_aligned_auxiliary_dirac_control
from .q_tilt import projected_aether_q_constant_tilt_root_audit
from .quartic_auxiliary_time_campaign import run_quartic_auxiliary_time_campaign
from .quartic_constraint_reconstruction_campaign import (
    run_quartic_constraint_reconstruction_campaign,
)
from .quartic_dirac_hamiltonian_campaign import (
    run_quartic_dirac_hamiltonian_campaign,
)
from .quartic_first_order_reduction_campaign import (
    run_quartic_first_order_reduction_campaign,
)
from .quartic_linear_x_campaign import run_quartic_linear_x_symbol_campaign
from .quartic_linearized_energy_campaign import (
    run_quartic_linearized_energy_campaign,
)
from .quartic_quasilinear_moser_campaign import (
    run_quartic_quasilinear_moser_campaign,
)
from .quartic_symmetrizer_domain import run_quartic_symmetrizer_domain_campaign
from .scalar_tensor_pack import (
    generic_g2_variation_noether_control,
    generic_g3_variation_noether_control,
    generic_g4_phi_variation_noether_control,
    generic_g4_scalar_variation_control,
)
from .x_completion_nogo import static_null_k14_multiplicative_no_go_control
from .x_nonlinear import nonlinear_aether_acceleration_convexity_control


@dataclass(frozen=True)
class FormalCheck:
    name: str
    status: str
    claim: str
    evidence: dict[str, Any]
    scope: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "claim": self.claim,
            "scope": self.scope,
            "evidence": self.evidence,
        }


def load_field_contract(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != "sigma-covariant-field-contract-1.0":
        raise ValueError("Unsupported or missing covariant field-contract version")
    return payload


def validate_field_contract(contract: dict[str, Any]) -> dict[str, Any]:
    fields = {item["id"]: item for item in contract["fields"]}
    generators = {item["id"]: item for item in contract["generator_invariants"]}
    diagnostics = {item["id"]: item for item in contract["diagnostic_invariants"]}
    errors: list[str] = []
    for required in ("g_mu_nu", "psi_m", "J_b_mu"):
        if required not in fields:
            errors.append(f"missing required field declaration: {required}")
    if contract["action_contract"].get("physical_metric") != "g_mu_nu":
        errors.append("the unique physical metric must be g_mu_nu")
    if "z_b" not in diagnostics:
        errors.append("z_b must be explicitly defined as a diagnostic invariant")
    elif diagnostics["z_b"].get("generator_status") != "forbidden":
        errors.append("z_b must not be admitted to the universal-minimal-coupling grammar")
    overlap = sorted(set(generators) & set(diagnostics))
    if overlap:
        errors.append(f"invariants cannot be both generator and diagnostic entries: {overlap}")
    dimensionful = sorted(
        name for name, item in generators.items() if item.get("dimensionless") is not True
    )
    if dimensionful:
        errors.append(f"generator invariants are not declared dimensionless: {dimensionful}")
    return {
        "valid": not errors,
        "errors": errors,
        "field_ids": sorted(fields),
        "generator_invariant_ids": sorted(generators),
        "diagnostic_invariant_ids": sorted(diagnostics),
        "z_b_definition": diagnostics.get("z_b", {}).get("definition"),
        "z_b_generator_status": diagnostics.get("z_b", {}).get("generator_status"),
    }


def validate_covariant_action_spec(
    spec: dict[str, Any], contract: dict[str, Any]
) -> dict[str, Any]:
    """Static, fail-closed validation before any expensive tensor calculation."""

    text = json.dumps(spec, sort_keys=True).casefold()
    generator_ids = {item["id"].casefold() for item in contract["generator_invariants"]}
    diagnostic_ids = {item["id"].casefold() for item in contract["diagnostic_invariants"]}
    used = {str(item).casefold() for item in spec.get("invariants", [])}
    errors: list[str] = []
    if spec.get("matter_metric") != "g_mu_nu":
        errors.append("matter_metric must be the unique physical metric g_mu_nu")
    prohibited_diagnostics = sorted(used & diagnostic_ids)
    if prohibited_diagnostics:
        errors.append(
            "diagnostic matter invariants are forbidden in this gravitational action grammar: "
            + ", ".join(prohibited_diagnostics)
        )
    unknown = sorted(used - generator_ids)
    if unknown:
        errors.append("undeclared generator invariants: " + ", ".join(unknown))
    baryon_tokens = ("j_b", "rho_b", "t_b", "z_b", "baryon")
    if any(token in text for token in baryon_tokens):
        errors.append(
            "direct baryon-specific dependence violates universal minimal matter coupling"
        )
    if spec.get("static_dictionary_status") != "derived":
        errors.append("the static-to-covariant dictionary has not been derived")
    return {
        "valid": not errors,
        "errors": errors,
        "used_invariants": sorted(used),
        "prohibited_diagnostic_invariants": prohibited_diagnostics,
    }


def _run_check(
    name: str,
    claim: str,
    scope: str,
    function: Callable[[], tuple[bool, dict[str, Any]]],
) -> FormalCheck:
    try:
        passed, evidence = function()
        return FormalCheck(name, "pass" if passed else "fail", claim, evidence, scope)
    except Exception as error:  # noqa: BLE001 - controls expose failures instead of hiding them.
        return FormalCheck(
            name,
            "fail",
            claim,
            {"exception": type(error).__name__, "message": str(error)},
            scope,
        )


def _projected_aether_q_first_variation_control() -> tuple[bool, dict[str, Any]]:
    # Lazy import avoids action_ir -> formal_backend -> q_variation_ir -> action_ir initialization.
    from .q_variation_ir import projected_aether_q_first_variation_control

    return projected_aether_q_first_variation_control()


def _linearized_einstein_bianchi() -> tuple[bool, dict[str, Any]]:
    """Verify k^mu G^(1)_mu,nu = 0 for a general symmetric h_mu,nu in flat space."""

    eta = sp.diag(-1, 1, 1, 1)
    k_up = sp.Matrix(sp.symbols("k0:4"))
    k_down = eta * k_up
    h_symbols = sp.symbols("h00 h01 h02 h03 h11 h12 h13 h22 h23 h33")
    h = sp.zeros(4)
    cursor = 0
    for mu in range(4):
        for nu in range(mu, 4):
            h[mu, nu] = h_symbols[cursor]
            h[nu, mu] = h_symbols[cursor]
            cursor += 1
    trace_h = sp.trace(eta * h)
    k_sq = (k_up.T * eta * k_up)[0]
    khk = (k_up.T * h * k_up)[0]
    einstein = sp.zeros(4)
    for mu in range(4):
        for nu in range(4):
            einstein[mu, nu] = sp.Rational(1, 2) * (
                k_down[mu] * sum(k_up[rho] * h[rho, nu] for rho in range(4))
                + k_down[nu] * sum(k_up[rho] * h[rho, mu] for rho in range(4))
                - k_sq * h[mu, nu]
                - k_down[mu] * k_down[nu] * trace_h
                - eta[mu, nu] * (khk - k_sq * trace_h)
            )
    residuals = [sp.factor(sum(k_up[mu] * einstein[mu, nu] for mu in range(4))) for nu in range(4)]
    return all(value == 0 for value in residuals), {
        "identity": "k^mu G^(1)_mu_nu = 0",
        "residuals": [str(value) for value in residuals],
        "independent_metric_perturbations": len(h_symbols),
    }


def _canonical_scalar_control() -> tuple[bool, dict[str, Any]]:
    velocity, gradient, mass, field = sp.symbols("v grad m phi", real=True)
    lagrangian = (
        sp.Rational(1, 2) * velocity**2
        - sp.Rational(1, 2) * gradient**2
        - sp.Rational(1, 2) * mass**2 * field**2
    )
    hessian = sp.hessian(lagrangian, (velocity,))
    omega2, wave_number2 = sp.symbols("omega2 k2", nonnegative=True)
    characteristic = -omega2 + wave_number2
    return hessian.det() == 1 and sp.solve(characteristic, omega2) == [wave_number2], {
        "action": "S_phi = integral sqrt(-g)[-1/2 (nabla phi)^2 - V(phi)]",
        "kinetic_hessian": str(hessian),
        "kinetic_rank": int(hessian.rank()),
        "principal_polynomial": str(characteristic),
        "characteristic_speed_squared": 1,
        "physical_scalar_dof": 1,
    }


def _canonical_scalar_gravity_cross_constraint_control() -> tuple[bool, dict[str, Any]]:
    evidence = canonical_scalar_spatial_density_certificate()
    return bool(evidence["passed"]), evidence


def _canonical_scalar_noether_identity() -> tuple[bool, dict[str, Any]]:
    """Verify div(T)_nu = (box(phi)-m^2 phi) partial_nu(phi) in normal coordinates."""

    eta = sp.diag(-1, 1, 1, 1)
    field, mass = sp.symbols("phi m", real=True)
    gradient = sp.Matrix(sp.symbols("p0:4", real=True))
    hessian_symbols = sp.symbols("h00 h01 h02 h03 h11 h12 h13 h22 h23 h33")
    hessian = sp.zeros(4)
    cursor = 0
    for mu in range(4):
        for nu in range(mu, 4):
            hessian[mu, nu] = hessian_symbols[cursor]
            hessian[nu, mu] = hessian_symbols[cursor]
            cursor += 1
    gradient_squared = (gradient.T * eta * gradient)[0]
    box_field = sp.trace(eta * hessian)
    residuals: list[sp.Expr] = []
    divergences: list[sp.Expr] = []
    for nu in range(4):
        divergence = 0
        for mu in range(4):
            for alpha in range(4):
                derivative_t = (
                    hessian[alpha, mu] * gradient[nu]
                    + gradient[mu] * hessian[alpha, nu]
                    - eta[mu, nu]
                    * (
                        sum(
                            eta[rho, sigma] * hessian[alpha, rho] * gradient[sigma]
                            for rho in range(4)
                            for sigma in range(4)
                        )
                        + mass**2 * field * gradient[alpha]
                    )
                )
                divergence += eta[mu, alpha] * derivative_t
        divergence = sp.factor(divergence)
        expected = sp.factor((box_field - mass**2 * field) * gradient[nu])
        divergences.append(divergence)
        residuals.append(sp.factor(divergence - expected))
    passed = all(item == 0 for item in residuals)
    return passed, {
        "identity": "nabla^mu T_mu_nu = E_phi nabla_nu(phi)",
        "scalar_euler_derivative": "E_phi = box(phi) - m^2 phi",
        "calculation_frame": "Riemann normal coordinates at an arbitrary point; the residual is tensorial",
        "gradient_squared": str(gradient_squared),
        "divergences": [str(item) for item in divergences],
        "residuals": [str(item) for item in residuals],
    }


def _proca_dirac_control() -> tuple[bool, dict[str, Any]]:
    evidence = proca_fourier_dirac_control()
    passed = (
        evidence["hessian_rank"] == 3
        and evidence["hessian_nullity"] == 1
        and len(evidence["primary_constraints"]) == 1
        and len(evidence["secondary_constraints"]) == 1
        and evidence["constraint_matrix_rank"] == 2
        and evidence["physical_dof"] == 3
        and evidence["closure"]
        and evidence["hamiltonian_positive_definite"]
    )
    evidence["action"] = "S_A = integral sqrt(-g)[-F_mu_nu F^mu_nu/4 - m^2 A_mu A^mu/2]"
    evidence["physical_vector_dof"] = evidence["physical_dof"]
    evidence["reduced_characteristic_polynomial"] = "(-omega^2 + |k|^2)^3"
    return passed, evidence


def _proca_divergence_identity() -> tuple[bool, dict[str, Any]]:
    second_derivative = sp.Matrix(
        4,
        4,
        lambda row, column: sp.Symbol(f"q{min(row, column)}{max(row, column)}", real=True),
    )
    antisymmetric = sp.zeros(4)
    for mu in range(4):
        for nu in range(mu + 1, 4):
            component = sp.Symbol(f"F{mu}{nu}", real=True)
            antisymmetric[mu, nu] = component
            antisymmetric[nu, mu] = -component
    maxwell_divergence = sp.factor(
        sum(second_derivative[mu, nu] * antisymmetric[mu, nu] for mu in range(4) for nu in range(4))
    )
    mass = sp.Symbol("m", nonzero=True, real=True)
    divergence_a = sp.Symbol("divA", real=True)
    eom_divergence = sp.factor(maxwell_divergence - mass**2 * divergence_a)
    constraint_solution = sp.solve(sp.Eq(eom_divergence, 0), divergence_a)
    passed = maxwell_divergence == 0 and constraint_solution == [0]
    return passed, {
        "identity": "partial_nu partial_mu F^{mu nu} = 0",
        "reason": "commuting derivative pair is symmetric while F is antisymmetric",
        "maxwell_divergence_residual": str(maxwell_divergence),
        "proca_eom_divergence": str(eom_divergence),
        "mass_domain": "m != 0",
        "derived_constraint": "partial_mu A^mu = 0",
        "constraint_solution": [str(item) for item in constraint_solution],
    }


def _proca_stress_noether_identity() -> tuple[bool, dict[str, Any]]:
    """Independently reduce the full off-shell Proca stress identity on Minkowski."""

    dimension = 4
    coordinates = sp.symbols("x0:4", real=True)
    metric = sp.diag(-1, 1, 1, 1)
    mass = sp.Symbol("m", real=True)
    vector_down = sp.Matrix([sp.Function(f"A{mu}")(*coordinates) for mu in range(dimension)])
    vector_up = metric * vector_down
    field_strength_down = sp.MutableDenseMatrix(
        dimension,
        dimension,
        lambda mu, nu: (
            sp.diff(vector_down[nu], coordinates[mu]) - sp.diff(vector_down[mu], coordinates[nu])
        ),
    )
    field_strength_up = metric * field_strength_down * metric
    field_strength_squared = sum(
        field_strength_down[mu, nu] * field_strength_up[mu, nu]
        for mu in range(dimension)
        for nu in range(dimension)
    )
    vector_squared = (vector_down.T * metric * vector_down)[0]
    stress_down = sp.MutableDenseMatrix(
        dimension,
        dimension,
        lambda mu, nu: (
            sum(
                field_strength_down[mu, rho]
                * sum(
                    metric[rho, sigma] * field_strength_down[nu, sigma]
                    for sigma in range(dimension)
                )
                for rho in range(dimension)
            )
            - sp.Rational(1, 4) * metric[mu, nu] * field_strength_squared
            + mass**2
            * (
                vector_down[mu] * vector_down[nu]
                - sp.Rational(1, 2) * metric[mu, nu] * vector_squared
            )
        ),
    )
    euler_up = [
        sp.factor(
            sum(sp.diff(field_strength_up[mu, rho], coordinates[mu]) for mu in range(dimension))
            - mass**2 * vector_up[rho]
        )
        for rho in range(dimension)
    ]
    euler_divergence = sp.factor(
        sum(sp.diff(euler_up[rho], coordinates[rho]) for rho in range(dimension))
    )
    residuals: list[sp.Expr] = []
    for nu in range(dimension):
        stress_divergence = sp.factor(
            sum(
                metric[mu, alpha] * sp.diff(stress_down[mu, nu], coordinates[alpha])
                for mu in range(dimension)
                for alpha in range(dimension)
            )
        )
        euler_force = sum(field_strength_down[nu, rho] * euler_up[rho] for rho in range(dimension))
        residuals.append(
            sp.simplify(stress_divergence - euler_force + vector_down[nu] * euler_divergence)
        )
    passed = all(item == 0 for item in residuals)
    return passed, {
        "identity": "partial^mu T_mu_nu - F_nu_rho E^rho + A_nu partial_rho E^rho = 0",
        "vector_euler_derivative": "E^rho = partial_mu F^{mu rho} - m^2 A^rho",
        "stress_tensor": "T_mu_nu = F_mu_rho F_nu^rho - eta_mu_nu F^2/4 + m^2(A_mu A_nu - eta_mu_nu A^2/2)",
        "calculation_scope": "exact off-shell 4D Minkowski component identity for arbitrary smooth A_mu",
        "residuals": [str(item) for item in residuals],
        "curved_background_extension": "unresolved; requires covariant derivative commutators and metric compatibility in the executable reducer",
    }


def _proca_curved_noether_identity() -> tuple[bool, dict[str, Any]]:
    evidence = proca_curved_background_noether_controls()
    return bool(evidence["passed"]), evidence


def _einstein_aether_flrw_variation_identity() -> tuple[bool, dict[str, Any]]:
    evidence = einstein_aether_flrw_variation_control()
    return bool(evidence["passed"]), evidence


def _einstein_aether_adm_kinetic() -> tuple[bool, dict[str, Any]]:
    evidence = einstein_aether_adm_kinetic_control()
    return bool(evidence["passed"]), evidence


def _einstein_aether_3plus1_decomposition() -> tuple[bool, dict[str, Any]]:
    evidence = einstein_aether_3plus1_decomposition_control()
    return bool(evidence["passed"]), evidence


def _einstein_aether_generic_dh_covariance() -> tuple[bool, dict[str, Any]]:
    evidence = einstein_aether_generic_dh_covariance_control()
    return bool(evidence["passed"]), evidence


def _einstein_aether_generic_hh_deformation() -> tuple[bool, dict[str, Any]]:
    evidence = einstein_aether_generic_hh_deformation_control()
    return bool(evidence["passed"]), evidence


def _einstein_aether_linearized_energy() -> tuple[bool, dict[str, Any]]:
    evidence = einstein_aether_linearized_energy_control()
    return bool(evidence["passed"]), evidence


def _einstein_aether_nonlinear_positive_energy_theorem() -> tuple[bool, dict[str, Any]]:
    evidence = einstein_aether_nonlinear_positive_energy_theorem_control()
    return bool(evidence["passed"]), evidence


def _einstein_aether_reduced_principal_domain() -> tuple[bool, dict[str, Any]]:
    evidence = einstein_aether_reduced_principal_domain_control()
    return bool(evidence["passed"]), evidence


def _einstein_aether_global_tilt_legendre() -> tuple[bool, dict[str, Any]]:
    evidence = einstein_aether_global_tilt_legendre_control()
    return bool(evidence["passed"]), evidence


def _einstein_aether_covariant_strong_hyperbolicity() -> tuple[bool, dict[str, Any]]:
    evidence = einstein_aether_covariant_strong_hyperbolicity_control()
    return bool(evidence["passed"]), evidence


def _einstein_aether_lapse_shift_constraint_seed() -> tuple[bool, dict[str, Any]]:
    evidence = einstein_aether_lapse_shift_constraint_seed_control()
    return bool(evidence["passed"]), evidence


def _einstein_aether_coupled_unit_normal() -> tuple[bool, dict[str, Any]]:
    evidence = einstein_aether_coupled_unit_normal_control()
    return bool(evidence["passed"]), evidence


def _einstein_aether_spatial_diffeomorphism() -> tuple[bool, dict[str, Any]]:
    evidence = einstein_aether_spatial_diffeomorphism_control()
    return bool(evidence["passed"]), evidence


def _maxwell_unit_aether_nonlinear_hamiltonian() -> tuple[bool, dict[str, Any]]:
    evidence = maxwell_unit_aether_nonlinear_hamiltonian_control()
    return bool(evidence["passed"]), evidence


def _unit_timelike_vector_dirac_chain() -> tuple[bool, dict[str, Any]]:
    evidence = unit_timelike_vector_dirac_chain_control()
    return bool(evidence["passed"]), evidence


def _regular_holonomic_multiplier_dirac() -> tuple[bool, dict[str, Any]]:
    evidence = regular_holonomic_multiplier_dirac_control()
    return bool(evidence["passed"]), evidence


def _constraint_surface_poisson_rank_control() -> tuple[bool, dict[str, Any]]:
    q1, q2, p1, p2 = sp.symbols("q1 q2 p1 p2")
    constraints = (p1, p2 + q1 * p1)
    off_surface, on_surface, independent = reduce_poisson_matrix_on_constraint_surface(
        constraints, (q1, q2), (p1, p2)
    )
    passed = off_surface.rank() == 2 and on_surface == sp.zeros(2) and independent == 2
    return passed, {
        "constraints": [str(item) for item in constraints],
        "off_surface_poisson_matrix": str(off_surface),
        "off_surface_rank": int(off_surface.rank()),
        "constraint_surface_poisson_matrix": str(on_surface),
        "constraint_surface_rank": int(on_surface.rank()),
        "independent_constraints": independent,
        "interpretation": (
            "The nonzero off-surface bracket is proportional to a constraint; quotient-ring "
            "reduction correctly classifies both independent constraints as first class."
        ),
    }


def _tertiary_dirac_chain_control() -> tuple[bool, dict[str, Any]]:
    q1, q2, q3 = sp.symbols("q1 q2 q3")
    v1, v2, v3 = sp.symbols("v1 v2 v3")
    lagrangian = (v1 - q2) ** 2 / 2 + q3 * q1
    result = analyze_quadratic_lagrangian(lagrangian, (q1, q2, q3), (v1, v2, v3))
    evidence = result.as_dict()
    evidence["lagrangian"] = str(lagrangian)
    passed = (
        len(result.primary_constraints) == 2
        and len(result.secondary_constraints) == 2
        and len(result.higher_generation_constraints) == 2
        and result.independent_constraints == 6
        and result.constraint_matrix_rank == 6
        and result.physical_dof == 0
        and result.closure
    )
    return passed, evidence


def _field_theory_constraint_algebra_control() -> tuple[bool, dict[str, Any]]:
    evidence = virasoro_constraint_algebra_control()
    return bool(evidence["passed"]), evidence


def _three_dimensional_field_bracket_control() -> tuple[bool, dict[str, Any]]:
    evidence = three_dimensional_smeared_bracket_control()
    return bool(evidence["passed"]), evidence


def _proca_reduced_field_bracket_control() -> tuple[bool, dict[str, Any]]:
    evidence = proca_reduced_smeared_constraint_control()
    return bool(evidence["passed"]), evidence


def _canonical_metric_diffeomorphism_control() -> tuple[bool, dict[str, Any]]:
    evidence = canonical_metric_diffeomorphism_control()
    return bool(evidence["passed"]), evidence


def _canonical_metric_dewitt_kinetic_control() -> tuple[bool, dict[str, Any]]:
    evidence = canonical_metric_dewitt_kinetic_control()
    return bool(evidence["passed"]), evidence


def _einstein_aether_2d_noether_artifact(root: Path) -> tuple[bool, dict[str, Any]]:
    """Validate the independently generated arbitrary-jet identity artifact.

    The script hash makes this a bound executable result rather than an untracked
    JSON assertion.  It remains a two-dimensional control and cannot stand in for
    the outstanding full four-dimensional tensor identity.
    """

    artifact_path = root / "runs" / "formal-controls-v1" / "aether-noether-2d.json"
    verifier_path = root / "scripts" / "verify_aether_noether_2d.py"
    if not artifact_path.exists():
        return False, {"reason": "missing arbitrary-jet artifact", "path": str(artifact_path)}
    if not verifier_path.exists():
        return False, {"reason": "missing verifier script", "path": str(verifier_path)}
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    verifier_sha256 = hashlib.sha256(verifier_path.read_bytes()).hexdigest()
    expected_terms = {"K1", "K2", "K3", "K4", "unit_constraint"}
    terms = artifact.get("terms", {})
    residuals_are_zero = all(
        item.get("passed") is True and item.get("residuals") == ["0", "0"]
        for item in terms.values()
    )
    passed = (
        artifact.get("schema_version") == "sigma-aether-noether-jet-1.0"
        and artifact.get("dimension") == 2
        and artifact.get("complete") is True
        and artifact.get("passed") is True
        and set(terms) == expected_terms
        and residuals_are_zero
        and artifact.get("verifier_sha256") == verifier_sha256
    )
    return passed, {
        "artifact": str(artifact_path),
        "artifact_complete": artifact.get("complete"),
        "artifact_passed": artifact.get("passed"),
        "dimension": artifact.get("dimension"),
        "field_jet_scope": artifact.get("field_jet_scope"),
        "identity": artifact.get("identity"),
        "term_residuals": {name: item.get("residuals") for name, item in terms.items()},
        "expected_terms_present": set(terms) == expected_terms,
        "verifier_sha256": verifier_sha256,
        "artifact_verifier_sha256": artifact.get("verifier_sha256"),
        "script_hash_matches": artifact.get("verifier_sha256") == verifier_sha256,
        "interpretation": (
            "Exact arbitrary inhomogeneous 2D diffeomorphism identity, term by term; "
            "the full arbitrary-background 4D identity remains unresolved."
        ),
    }


def _einstein_aether_4d_numeric_noether_artifact(root: Path) -> tuple[bool, dict[str, Any]]:
    """Validate the source-bound full-coordinate 4D arbitrary-jet numerical artifact."""

    artifact_path = root / "runs" / "formal-controls-v1" / "aether-noether-4d-numeric.json"
    verifier_path = root / "scripts" / "verify_aether_noether_4d_numeric.py"
    if not artifact_path.exists():
        return False, {"reason": "missing 4D arbitrary-jet artifact", "path": str(artifact_path)}
    if not verifier_path.exists():
        return False, {"reason": "missing 4D verifier script", "path": str(verifier_path)}
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    verifier_sha256 = hashlib.sha256(verifier_path.read_bytes()).hexdigest()
    expected_terms = {"K1", "K2", "K3", "K4", "unit_constraint"}
    terms = artifact.get("terms", {})
    samples = [sample for item in terms.values() for sample in item.get("samples", [])]
    maximum_residual = max(
        (float(sample.get("maximum_absolute_residual", float("inf"))) for sample in samples),
        default=float("inf"),
    )
    expected_seed_count = len(artifact.get("seeds", []))
    terms_pass = all(
        item.get("passed") is True
        and len(item.get("samples", [])) == expected_seed_count
        and all(sample.get("passed") is True for sample in item.get("samples", []))
        for item in terms.values()
    )
    negative = artifact.get("negative_control", {})
    passed = (
        artifact.get("schema_version") == "sigma-aether-noether-4d-numeric-1.0"
        and artifact.get("dimension") == 4
        and artifact.get("fields") == 15
        and artifact.get("complete") is True
        and artifact.get("passed") is True
        and set(terms) == expected_terms
        and expected_seed_count >= 3
        and terms_pass
        and negative.get("rejected") is True
        and artifact.get("verifier_sha256") == verifier_sha256
    )
    return passed, {
        "artifact": str(artifact_path),
        "artifact_complete": artifact.get("complete"),
        "artifact_passed": artifact.get("passed"),
        "dimension": artifact.get("dimension"),
        "fields": artifact.get("fields"),
        "field_jet_scope": artifact.get("field_jet_scope"),
        "identity": artifact.get("identity"),
        "proof_scope": artifact.get("proof_scope"),
        "seeds": artifact.get("seeds"),
        "expected_terms_present": set(terms) == expected_terms,
        "sample_count": len(samples),
        "maximum_absolute_residual": maximum_residual,
        "absolute_tolerance": artifact.get("absolute_tolerance"),
        "negative_control": negative,
        "verifier_sha256": verifier_sha256,
        "artifact_verifier_sha256": artifact.get("verifier_sha256"),
        "script_hash_matches": artifact.get("verifier_sha256") == verifier_sha256,
        "interpretation": (
            "Full-coordinate 4D arbitrary-jet differential falsification at three independent "
            "samples; floating-point agreement is not the outstanding exact symbolic proof."
        ),
    }


def _dhost_degeneracy_control() -> tuple[bool, dict[str, Any]]:
    evidence = dhost_reduced_dirac_control()
    return bool(evidence["passed"]), evidence


def _generic_horndeski_l2_l4_adm_control() -> tuple[bool, dict[str, Any]]:
    evidence = generic_horndeski_l2_l4_unitary_adm_control()
    return bool(evidence["passed"]), evidence


def _generic_horndeski_flrw_interval_background_control(
    root: Path,
) -> tuple[bool, dict[str, Any]]:
    ir_path = root / "runs" / "physics-language" / "horndeski-l2-l4-polynomial-ir.json"
    config_path = root / "configs" / "backgrounds" / "canonical_scalar_stiff_interval.json"
    artifact_path = (
        root
        / "runs"
        / "physics-language"
        / "canonical-scalar-stiff-background-certificate.json"
    )
    ir = json.loads(ir_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    result = certify_flrw_background(ir, config)

    off_constraint_config = json.loads(json.dumps(config))
    off_constraint_config["initial_state"]["x"] = "0.031"
    off_constraint = certify_flrw_background(ir, off_constraint_config)
    singular_ir = json.loads(json.dumps(ir))
    singular_ir["compiled_flrw_background_system"]["evolution_matrix"] = (
        "Matrix([[1, 1], [1, 1]])"
    )
    singular = certify_flrw_background(singular_ir, config)
    ghost_ir = json.loads(json.dumps(ir))
    ghost_ir["compiled_tensor_G_T"] = "-1"
    ghost = certify_flrw_background(ghost_ir, config)

    uniform = result.get("uniform_certificate", {})
    reference = result.get("analytic_reference", {})
    formulation = result.get("formulation_certificate", {})
    health = uniform.get("health_lower_bounds", {})
    passed = (
        result.get("status") == "pass_interval_certified"
        and artifact.get("content_sha256") == result.get("content_sha256")
        and result.get("source_ir_sha256") == ir.get("content_sha256")
        and reference.get("passed") is True
        and all(reference.get("contained", {}).values())
        and health
        and all(value > config["health_margin"] for value in health.values())
        and uniform.get("Theta_min_abs", 0) > config["health_margin"]
        and uniform.get("evolution_determinant_min_abs", 0)
        > config["determinant_floor"]
        and uniform.get("constraint_max_abs_enclosure", float("inf"))
        <= config["constraint_tolerance"]
        and len(result.get("step_certificates", [])) == result.get("time", {}).get("steps")
        and formulation.get("status")
        == "pass_generalized_harmonic_kessence_on_certified_trajectory"
        and formulation.get("uniform_kessence_health_lower_bounds")
        and all(
            value > config["health_margin"]
            for value in formulation.get(
                "uniform_kessence_health_lower_bounds", {}
            ).values()
        )
        and off_constraint.get("status") == "reject"
        and singular.get("status") == "reject"
        and ghost.get("status") == "reject"
    )
    return passed, {
        "ir": str(ir_path),
        "config": str(config_path),
        "artifact": str(artifact_path),
        "status": result.get("status"),
        "steps": result.get("time", {}).get("steps"),
        "endpoint_enclosure": result.get("endpoint_enclosure"),
        "uniform_certificate": uniform,
        "analytic_reference": reference,
        "formulation_certificate": formulation,
        "artifact_hash_matches_reexecution": (
            artifact.get("content_sha256") == result.get("content_sha256")
        ),
        "source_ir_hash_matches": result.get("source_ir_sha256") == ir.get("content_sha256"),
        "negative_controls": {
            "off_constraint_initial_state": {
                "status": off_constraint.get("status"),
                "errors": off_constraint.get("errors"),
            },
            "singular_evolution_matrix": {
                "status": singular.get("status"),
                "errors": singular.get("errors"),
            },
            "tensor_ghost": {
                "status": ghost.get("status"),
                "errors": ghost.get("errors"),
            },
        },
        "scope": (
            "outward-rounded interval Picard certificate for one exact canonical-scalar FLRW "
            "known answer; generated candidates require their own initial constraint surface, "
            "coefficient binding, time domain, and interval certificate"
        ),
    }


def _cubic_bssn_uniform_domain_campaign_control(
    root: Path,
) -> tuple[bool, dict[str, Any]]:
    ir_path = root / "runs" / "physics-language" / "horndeski-l2-l4-polynomial-ir.json"
    flrw_campaign_path = (
        root
        / "runs"
        / "physics-language"
        / "horndeski-l2-l4-interval-campaign"
        / "campaign.json"
    )
    config_path = (
        root
        / "configs"
        / "backgrounds"
        / "cubic_bssn_uniform_domain_campaign.json"
    )
    artifact_path = (
        root
        / "runs"
        / "physics-language"
        / "cubic-bssn-uniform-domain-campaign"
        / "campaign.json"
    )
    ir = json.loads(ir_path.read_text(encoding="utf-8"))
    flrw_campaign = json.loads(flrw_campaign_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    trajectory_certificates = {
        record["candidate_id"]: json.loads(
            (flrw_campaign_path.parent / record["certificate"]).read_text(
                encoding="utf-8"
            )
        )
        for record in flrw_campaign["candidates"]
        if record.get("status")
        == "pass_flrw_interval_cubic_weak_field_bounds_unresolved"
    }
    result = run_cubic_bssn_domain_campaign(
        ir, flrw_campaign, trajectory_certificates, config
    )
    manifest = result["manifest"]
    first_trajectory = next(iter(trajectory_certificates.values()))
    bad_sigma_config = json.loads(json.dumps(config["domain_template"]))
    bad_sigma_config["slicing_parameter_sigma"] = 0.5
    bad_sigma = certify_cubic_bssn_domain(ir, first_trajectory, bad_sigma_config)
    non_timelike_config = json.loads(json.dumps(config["domain_template"]))
    non_timelike_config["domain_extension"]["spatial_gradient_abs"] = 1.0
    non_timelike = certify_cubic_bssn_domain(
        ir, first_trajectory, non_timelike_config
    )
    ranking = manifest.get("ranking", [])
    passed = (
        manifest.get("status")
        == "pass_all_screened_cubic_candidates_have_uniform_local_jet_boxes"
        and manifest.get("counts", {}).get("uniform_domain_certified") == 5
        and manifest.get("counts", {}).get("rejected") == 0
        and artifact.get("content_sha256") == manifest.get("content_sha256")
        and len(result["certificates"]) == 5
        and len(ranking) == 5
        and all(
            record["certified_hessian_component_radius_lower"] > 0
            and record["spatial_block_eigenvalue_lower"] > 0
            and record["slicing_cone_polynomial_upper"] < 0
            for record in ranking
        )
        and bad_sigma.get("status") == "reject"
        and non_timelike.get("status") == "reject"
    )
    return passed, {
        "ir": str(ir_path),
        "flrw_campaign": str(flrw_campaign_path),
        "config": str(config_path),
        "artifact": str(artifact_path),
        "status": manifest.get("status"),
        "counts": manifest.get("counts"),
        "ranking": ranking,
        "artifact_hash_matches_reexecution": (
            artifact.get("content_sha256") == manifest.get("content_sha256")
        ),
        "negative_controls": {
            "sigma_equals_one_half": {
                "status": bad_sigma.get("status"),
                "errors": bad_sigma.get("errors"),
            },
            "gradient_box_crosses_X_zero": {
                "status": non_timelike.get("status"),
                "errors": non_timelike.get("errors"),
            },
        },
        "scope": manifest.get("scope"),
    }


def _quartic_linear_x_symbol_campaign_control(
    root: Path,
) -> tuple[bool, dict[str, Any]]:
    ir_path = root / "runs" / "physics-language" / "horndeski-l2-l4-polynomial-ir.json"
    config_path = (
        root
        / "configs"
        / "backgrounds"
        / "quartic_linear_x_symbol_campaign.json"
    )
    artifact_path = (
        root
        / "runs"
        / "physics-language"
        / "quartic-linear-x-symbol-campaign"
        / "campaign.json"
    )
    ir = json.loads(ir_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    result = run_quartic_linear_x_symbol_campaign(ir, config)
    corrupted = json.loads(json.dumps(config))
    corrupted["fixed_coefficients"]["a01"] = "1"
    negative = run_quartic_linear_x_symbol_campaign(ir, corrupted)
    counts = result.get("counts", {})
    passed = bool(
        result.get("status")
        == "pass_exact_symbol_binding_uniform_symmetrizer_unresolved"
        and counts.get("selected") == 12
        and counts.get("exactly_bound") == 12
        and counts.get("canonical_G2") == 4
        and counts.get("quadratic_kessence_G2") == 8
        and artifact.get("content_sha256") == result.get("content_sha256")
        and result.get("source_ir_sha256") == ir.get("content_sha256")
        and negative.get("status") == "reject"
        and "a01" in " ".join(negative.get("errors", []))
    )
    return passed, {
        "ir": str(ir_path),
        "config": str(config_path),
        "artifact": str(artifact_path),
        "status": result.get("status"),
        "counts": counts,
        "artifact_hash_matches_reexecution": (
            artifact.get("content_sha256") == result.get("content_sha256")
        ),
        "source_ir_hash_matches": (
            result.get("source_ir_sha256") == ir.get("content_sha256")
        ),
        "quadratic_kessence_effective_metric_residual": result.get(
            "proof_controls", {}
        ).get("quadratic_kessence_extension", {}).get(
            "arbitrary_covector_effective_metric_residual"
        ),
        "negative_controls": {
            "phi_dependent_G4": {
                "status": negative.get("status"),
                "errors": negative.get("errors"),
            }
        },
        "scope": result.get("scope"),
    }


def _quartic_symmetrizer_domain_campaign_control(
    root: Path,
) -> tuple[bool, dict[str, Any]]:
    ir_path = root / "runs" / "physics-language" / "horndeski-l2-l4-polynomial-ir.json"
    binding_path = (
        root
        / "runs"
        / "physics-language"
        / "quartic-linear-x-symbol-campaign"
        / "campaign.json"
    )
    config_path = (
        root
        / "configs"
        / "backgrounds"
        / "quartic_symmetrizer_uniform_domain_campaign.json"
    )
    artifact_path = (
        root
        / "runs"
        / "physics-language"
        / "quartic-symmetrizer-uniform-domain-campaign"
        / "campaign.json"
    )
    ir = json.loads(ir_path.read_text(encoding="utf-8"))
    bindings = json.loads(binding_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    result = run_quartic_symmetrizer_domain_campaign(ir, bindings, config)
    corrupted = json.loads(json.dumps(bindings))
    corrupted["source_ir_sha256"] = "corrupted"
    negative = run_quartic_symmetrizer_domain_campaign(ir, corrupted, config)
    counts = result.get("counts", {})
    certificates = result.get("certificates", [])
    passed = bool(
        result.get("status")
        == "pass_all_linear_X_quartic_candidates_strongly_hyperbolic_on_local_boxes"
        and counts.get("selected") == 12
        and counts.get("uniform_local_jet_strong_hyperbolicity_passed") == 12
        and counts.get("rejected") == 0
        and len(certificates) == 12
        and all(
            record.get("status")
            == "pass_uniform_local_jet_strong_hyperbolicity"
            and record.get("uniform_matrix_bounds", {}).get(
                "companion_margin_numeric", -1
            )
            > 0
            for record in certificates
        )
        and artifact.get("content_sha256") == result.get("content_sha256")
        and negative.get("status") == "reject"
    )
    return passed, {
        "ir": str(ir_path),
        "bindings": str(binding_path),
        "config": str(config_path),
        "artifact": str(artifact_path),
        "status": result.get("status"),
        "counts": counts,
        "artifact_hash_matches_reexecution": (
            artifact.get("content_sha256") == result.get("content_sha256")
        ),
        "common_normalized_local_jet_component_radius": config.get(
            "normalized_local_jet_component_abs"
        ),
        "companion_margin_numeric_range": [
            min(
                record["uniform_matrix_bounds"]["companion_margin_numeric"]
                for record in certificates
            ),
            max(
                record["uniform_matrix_bounds"]["companion_margin_numeric"]
                for record in certificates
            ),
        ]
        if certificates
        else [],
        "negative_controls": result.get("negative_controls"),
        "hash_mismatch_negative": {
            "status": negative.get("status"),
            "errors": negative.get("errors"),
        },
        "scope": result.get("scope"),
    }


def _quartic_dirac_hamiltonian_campaign_control(
    root: Path,
) -> tuple[bool, dict[str, Any]]:
    base = root / "runs" / "physics-language"
    ir_path = base / "horndeski-l2-l4-polynomial-ir.json"
    binding_path = base / "quartic-linear-x-symbol-campaign" / "campaign.json"
    symmetrizer_path = (
        base / "quartic-symmetrizer-uniform-domain-campaign" / "campaign.json"
    )
    config_path = (
        root
        / "configs"
        / "backgrounds"
        / "quartic_dirac_hamiltonian_campaign.json"
    )
    artifact_path = (
        base / "quartic-dirac-hamiltonian-campaign" / "campaign.json"
    )
    ir = json.loads(ir_path.read_text(encoding="utf-8"))
    bindings = json.loads(binding_path.read_text(encoding="utf-8"))
    symmetrizers = json.loads(symmetrizer_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    result = run_quartic_dirac_hamiltonian_campaign(
        ir, bindings, symmetrizers, config
    )
    corrupted = json.loads(json.dumps(symmetrizers))
    corrupted["binding_campaign_sha256"] = "corrupted"
    negative = run_quartic_dirac_hamiltonian_campaign(
        ir, bindings, corrupted, config
    )
    certificates = result.get("certificates", [])
    counts = result.get("counts", {})
    passed = bool(
        result.get("status")
        == "pass_all_12_local_on_shell_adm_dirac_and_quadratic_hamiltonian"
        and counts.get("selected") == 12
        and counts.get("local_on_shell_adm_dirac_hamiltonian_passed") == 12
        and counts.get("rejected") == 0
        and len(certificates) == 12
        and all(
            item.get("certified_local_jet_embedding", {}).get("all_inside")
            and item.get("dirac_chain", {}).get("pairing_is_strictly_positive")
            and item.get("dirac_chain", {})
            .get("constraint_count", {})
            .get("physical_configuration_dof")
            == 3
            and item.get("on_shell_quadratic_physical_hamiltonian", {}).get(
                "strictly_positive"
            )
            and item.get("forward_homogeneous_invariant_domain", {}).get("passed")
            for item in certificates
        )
        and artifact.get("content_sha256") == result.get("content_sha256")
        and negative.get("status") == "reject"
    )
    return passed, {
        "ir": str(ir_path),
        "bindings": str(binding_path),
        "symmetrizers": str(symmetrizer_path),
        "config": str(config_path),
        "artifact": str(artifact_path),
        "status": result.get("status"),
        "counts": counts,
        "artifact_hash_matches_reexecution": (
            artifact.get("content_sha256") == result.get("content_sha256")
        ),
        "timelike_gradient_amplitude": config.get("timelike_gradient_amplitude"),
        "all_on_shell_equation_residuals_zero": all(
            set(
                item.get("on_shell_local_flrw_witness", {})
                .get("equation_residuals", {})
                .values()
            )
            == {"0"}
            for item in certificates
        ),
        "all_forward_homogeneous_domains_invariant": all(
            item.get("forward_homogeneous_invariant_domain", {}).get("passed")
            for item in certificates
        ),
        "lapse_pairing_numeric_range": [
            min(
                item["dirac_chain"]["background_lapse_pairing_numeric"]
                for item in certificates
            ),
            max(
                item["dirac_chain"]["background_lapse_pairing_numeric"]
                for item in certificates
            ),
        ]
        if certificates
        else [],
        "negative_controls": result.get("negative_controls"),
        "hash_mismatch_negative": {
            "status": negative.get("status"),
            "errors": negative.get("errors"),
        },
        "scope": result.get("scope"),
    }


def _quartic_linearized_energy_campaign_control(
    root: Path,
) -> tuple[bool, dict[str, Any]]:
    base = root / "runs" / "physics-language"
    dirac_path = base / "quartic-dirac-hamiltonian-campaign" / "campaign.json"
    config_path = (
        root
        / "configs"
        / "backgrounds"
        / "quartic_linearized_energy_campaign.json"
    )
    artifact_path = base / "quartic-linearized-energy-campaign" / "campaign.json"
    dirac_campaign = json.loads(dirac_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    result = run_quartic_linearized_energy_campaign(dirac_campaign, config)
    corrupted = json.loads(json.dumps(dirac_campaign))
    corrupted["status"] = "reject"
    negative = run_quartic_linearized_energy_campaign(corrupted, config)
    certificates = result.get("certificates", [])
    counts = result.get("counts", {})
    passed = bool(
        result.get("status")
        == "pass_all_12_finite_horizon_linearized_inhomogeneous_energies"
        and counts.get("selected") == 12
        and counts.get("finite_horizon_linearized_energy_passed") == 12
        and counts.get("rejected") == 0
        and len(certificates) == 12
        and all(
            item.get("quadratic_energy", {}).get("all_spatial_wavenumbers")
            and item.get("quadratic_energy", {}).get(
                "energy_amplification_upper_numeric", 0
            )
            >= 1
            and item.get("physical_derivative_tube", {}).get(
                "initial_E_s_strict_upper_numeric", 0
            )
            > 0
            and "not a nonlinear PDE trapping theorem" in item.get("scope", "")
            for item in certificates
        )
        and artifact.get("content_sha256") == result.get("content_sha256")
        and negative.get("status") == "reject"
    )
    return passed, {
        "dirac_campaign": str(dirac_path),
        "config": str(config_path),
        "artifact": str(artifact_path),
        "status": result.get("status"),
        "counts": counts,
        "artifact_hash_matches_reexecution": (
            artifact.get("content_sha256") == result.get("content_sha256")
        ),
        "proper_time_horizon_numeric_range": [
            min(
                item["background_compact_subdomain"][
                    "proper_time_horizon_lower_numeric"
                ]
                for item in certificates
            ),
            max(
                item["background_compact_subdomain"][
                    "proper_time_horizon_lower_numeric"
                ]
                for item in certificates
            ),
        ]
        if certificates
        else [],
        "energy_amplification_numeric_range": [
            min(
                item["quadratic_energy"]["energy_amplification_upper_numeric"]
                for item in certificates
            ),
            max(
                item["quadratic_energy"]["energy_amplification_upper_numeric"]
                for item in certificates
            ),
        ]
        if certificates
        else [],
        "negative_controls": result.get("negative_controls"),
        "prerequisite_status_negative": {
            "status": negative.get("status"),
            "errors": negative.get("errors"),
        },
        "scope": result.get("scope"),
    }


def _quartic_constraint_reconstruction_campaign_control(
    root: Path,
) -> tuple[bool, dict[str, Any]]:
    base = root / "runs" / "physics-language"
    dirac_path = base / "quartic-dirac-hamiltonian-campaign" / "campaign.json"
    energy_path = base / "quartic-linearized-energy-campaign" / "campaign.json"
    config_path = (
        root
        / "configs"
        / "backgrounds"
        / "quartic_constraint_reconstruction_campaign.json"
    )
    artifact_path = (
        base / "quartic-constraint-reconstruction-campaign" / "campaign.json"
    )
    dirac_campaign = json.loads(dirac_path.read_text(encoding="utf-8"))
    energy_campaign = json.loads(energy_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    result = run_quartic_constraint_reconstruction_campaign(
        dirac_campaign, energy_campaign, config
    )
    corrupted = json.loads(json.dumps(energy_campaign))
    corrupted["dirac_campaign_sha256"] = "corrupted"
    negative = run_quartic_constraint_reconstruction_campaign(
        dirac_campaign, corrupted, config
    )
    certificates = result.get("certificates", [])
    counts = result.get("counts", {})
    passed = bool(
        result.get("status") == "pass_all_12_linear_constraint_reconstructions"
        and counts.get("selected") == 12
        and counts.get("linear_constraint_reconstruction_passed") == 12
        and counts.get("rejected") == 0
        and len(certificates) == 12
        and all(
            item.get("operator_norm_bounds", {}).get(
                "combined_reconstruction_upper_numeric", 0
            )
            > 0
            and item.get("chained_energy_tube", {}).get(
                "final_initial_E_s_strict_upper_numeric", 0
            )
            > 0
            and "does not control their time derivatives" in item.get("scope", "")
            for item in certificates
        )
        and artifact.get("content_sha256") == result.get("content_sha256")
        and negative.get("status") == "reject"
    )
    return passed, {
        "dirac_campaign": str(dirac_path),
        "energy_campaign": str(energy_path),
        "config": str(config_path),
        "artifact": str(artifact_path),
        "status": result.get("status"),
        "counts": counts,
        "artifact_hash_matches_reexecution": (
            artifact.get("content_sha256") == result.get("content_sha256")
        ),
        "reconstruction_operator_numeric_range": [
            min(
                item["operator_norm_bounds"][
                    "combined_reconstruction_upper_numeric"
                ]
                for item in certificates
            ),
            max(
                item["operator_norm_bounds"][
                    "combined_reconstruction_upper_numeric"
                ]
                for item in certificates
            ),
        ]
        if certificates
        else [],
        "chained_initial_energy_numeric_range": [
            min(
                item["chained_energy_tube"][
                    "final_initial_E_s_strict_upper_numeric"
                ]
                for item in certificates
            ),
            max(
                item["chained_energy_tube"][
                    "final_initial_E_s_strict_upper_numeric"
                ]
                for item in certificates
            ),
        ]
        if certificates
        else [],
        "negative_controls": result.get("negative_controls"),
        "hash_mismatch_negative": {
            "status": negative.get("status"),
            "errors": negative.get("errors"),
        },
        "scope": result.get("scope"),
    }


def _quartic_auxiliary_time_campaign_control(
    root: Path,
) -> tuple[bool, dict[str, Any]]:
    base = root / "runs" / "physics-language"
    dirac_path = base / "quartic-dirac-hamiltonian-campaign" / "campaign.json"
    energy_path = base / "quartic-linearized-energy-campaign" / "campaign.json"
    reconstruction_path = (
        base / "quartic-constraint-reconstruction-campaign" / "campaign.json"
    )
    config_path = (
        root / "configs" / "backgrounds" / "quartic_auxiliary_time_campaign.json"
    )
    artifact_path = base / "quartic-auxiliary-time-campaign" / "campaign.json"
    dirac = json.loads(dirac_path.read_text(encoding="utf-8"))
    energy = json.loads(energy_path.read_text(encoding="utf-8"))
    reconstruction = json.loads(reconstruction_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    result = run_quartic_auxiliary_time_campaign(
        dirac, energy, reconstruction, config
    )
    corrupted = json.loads(json.dumps(reconstruction))
    corrupted["energy_campaign_sha256"] = "corrupted"
    negative = run_quartic_auxiliary_time_campaign(
        dirac, energy, corrupted, config
    )
    certificates = result.get("certificates", [])
    counts = result.get("counts", {})
    passed = bool(
        result.get("status")
        == "pass_all_12_linear_auxiliary_time_reconstructions"
        and counts.get("selected") == 12
        and counts.get("linear_auxiliary_time_reconstruction_passed") == 12
        and counts.get("rejected") == 0
        and len(certificates) == 12
        and all(
            item.get("time_reconstruction_operator", {}).get(
                "combined_upper_numeric", 0
            )
            > 0
            and item.get("chained_energy_tube", {}).get(
                "final_initial_E_s_strict_upper_numeric", 0
            )
            > 0
            and "does not bound nonlinear constraint products" in item.get(
                "scope", ""
            )
            for item in certificates
        )
        and artifact.get("content_sha256") == result.get("content_sha256")
        and negative.get("status") == "reject"
    )
    return passed, {
        "dirac_campaign": str(dirac_path),
        "energy_campaign": str(energy_path),
        "reconstruction_campaign": str(reconstruction_path),
        "config": str(config_path),
        "artifact": str(artifact_path),
        "status": result.get("status"),
        "counts": counts,
        "artifact_hash_matches_reexecution": (
            artifact.get("content_sha256") == result.get("content_sha256")
        ),
        "time_reconstruction_operator_numeric_range": [
            min(
                item["time_reconstruction_operator"]["combined_upper_numeric"]
                for item in certificates
            ),
            max(
                item["time_reconstruction_operator"]["combined_upper_numeric"]
                for item in certificates
            ),
        ]
        if certificates
        else [],
        "final_initial_energy_numeric_range": [
            min(
                item["chained_energy_tube"][
                    "final_initial_E_s_strict_upper_numeric"
                ]
                for item in certificates
            ),
            max(
                item["chained_energy_tube"][
                    "final_initial_E_s_strict_upper_numeric"
                ]
                for item in certificates
            ),
        ]
        if certificates
        else [],
        "negative_controls": result.get("negative_controls"),
        "hash_mismatch_negative": {
            "status": negative.get("status"),
            "errors": negative.get("errors"),
        },
        "scope": result.get("scope"),
    }


def _quartic_quasilinear_moser_campaign_control(
    root: Path,
) -> tuple[bool, dict[str, Any]]:
    base = root / "runs" / "physics-language"
    symmetrizer_path = (
        base / "quartic-symmetrizer-uniform-domain-campaign" / "campaign.json"
    )
    auxiliary_path = base / "quartic-auxiliary-time-campaign" / "campaign.json"
    config_path = (
        root
        / "configs"
        / "backgrounds"
        / "quartic_quasilinear_moser_campaign.json"
    )
    artifact_path = base / "quartic-quasilinear-moser-campaign" / "campaign.json"
    symmetrizer = json.loads(symmetrizer_path.read_text(encoding="utf-8"))
    auxiliary = json.loads(auxiliary_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    result = run_quartic_quasilinear_moser_campaign(
        symmetrizer, auxiliary, config
    )
    corrupted = json.loads(json.dumps(auxiliary))
    corrupted["certificates"][0]["candidate_id"] = "corrupted-candidate"
    negative = run_quartic_quasilinear_moser_campaign(
        symmetrizer, corrupted, config
    )
    certificates = result.get("certificates", [])
    counts = result.get("counts", {})
    passed = bool(
        result.get("status")
        == "pass_all_12_quasilinear_coefficient_derivative_envelopes"
        and counts.get("selected") == 12
        and counts.get("quasilinear_coefficient_envelopes_passed") == 12
        and counts.get("rejected") == 0
        and len(certificates) == 12
        and all(
            item.get("raw_coefficient_degree") == {"A": 2, "B": 2, "C": 2}
            and all(
                item.get("raw_Frechet_derivative_2_norm_envelopes", {})
                .get(name, {})
                .get(order)
                == "0"
                for name in ("A", "B", "C")
                for order in ("3", "4")
            )
            and item.get(
                "companion_Frechet_derivative_2_norm_envelopes_numeric", {}
            ).get("4", 0)
            > 0
            and "does not reconstruct the nonlinear state-to-covariant-jet map"
            in item.get("scope", "")
            for item in certificates
        )
        and artifact.get("content_sha256") == result.get("content_sha256")
        and negative.get("status") == "reject"
    )
    return passed, {
        "symmetrizer_campaign": str(symmetrizer_path),
        "auxiliary_time_campaign": str(auxiliary_path),
        "config": str(config_path),
        "artifact": str(artifact_path),
        "status": result.get("status"),
        "counts": counts,
        "artifact_hash_matches_reexecution": (
            artifact.get("content_sha256") == result.get("content_sha256")
        ),
        "companion_C4_envelope_numeric_range": [
            min(
                item["companion_Frechet_derivative_2_norm_envelopes_numeric"]["4"]
                for item in certificates
            ),
            max(
                item["companion_Frechet_derivative_2_norm_envelopes_numeric"]["4"]
                for item in certificates
            ),
        ]
        if certificates
        else [],
        "negative_controls": result.get("negative_controls"),
        "candidate_set_corruption_negative": {
            "status": negative.get("status"),
            "errors": negative.get("errors"),
        },
        "scope": result.get("scope"),
    }


def _quartic_first_order_reduction_campaign_control(
    root: Path,
) -> tuple[bool, dict[str, Any]]:
    base = root / "runs" / "physics-language"
    symmetrizer_path = (
        base / "quartic-symmetrizer-uniform-domain-campaign" / "campaign.json"
    )
    moser_path = base / "quartic-quasilinear-moser-campaign" / "campaign.json"
    config_path = (
        root
        / "configs"
        / "backgrounds"
        / "quartic_first_order_reduction_campaign.json"
    )
    artifact_path = base / "quartic-first-order-reduction-campaign" / "campaign.json"
    symmetrizer = json.loads(symmetrizer_path.read_text(encoding="utf-8"))
    moser = json.loads(moser_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    result = run_quartic_first_order_reduction_campaign(
        symmetrizer, moser, config
    )
    corrupted = json.loads(json.dumps(moser))
    corrupted["certificates"][0]["candidate_id"] = "corrupted-candidate"
    negative = run_quartic_first_order_reduction_campaign(
        symmetrizer, corrupted, config
    )
    certificates = result.get("certificates", [])
    counts = result.get("counts", {})
    control = result.get("generic_reduction_control", {})
    passed = bool(
        result.get("status")
        == "pass_all_12_exact_55_variable_principal_first_order_reductions"
        and counts.get("selected") == 12
        and counts.get("exact_55_variable_reductions_passed") == 12
        and counts.get("rejected") == 0
        and len(certificates) == 12
        and control.get("passed") is True
        and control.get("spatial_block_extraction", {}).get(
            "B_reconstruction_residual_zero"
        )
        is True
        and control.get("spatial_block_extraction", {}).get(
            "C_reconstruction_residual_zero"
        )
        is True
        and control.get("full_pencil", {}).get(
            "nonzero_characteristic_lift_residual_zero"
        )
        is True
        and control.get("full_pencil", {}).get(
            "directional_companion_lift_residual_zero"
        )
        is True
        and all(
            item.get("state_dimensions", {}).get("physical_space_first_order")
            == 55
            and item.get("state_dimensions", {}).get("directional_companion")
            == 22
            and item.get("constraint_counts")
            == {"derivative_definition": 33, "independent_spatial_curl": 33}
            and "does not yet provide the nonlinear lower-order source"
            in item.get("scope", "")
            for item in certificates
        )
        and artifact.get("content_sha256") == result.get("content_sha256")
        and negative.get("status") == "reject"
    )
    return passed, {
        "symmetrizer_campaign": str(symmetrizer_path),
        "moser_campaign": str(moser_path),
        "config": str(config_path),
        "artifact": str(artifact_path),
        "status": result.get("status"),
        "counts": counts,
        "artifact_hash_matches_reexecution": (
            artifact.get("content_sha256") == result.get("content_sha256")
        ),
        "state_dimensions": {
            "second_order_fields": 11,
            "directional_companion": 22,
            "physical_space_first_order": 55,
            "zero_speed_auxiliary": 33,
        },
        "constraint_counts": {
            "derivative_definition": 33,
            "independent_spatial_curl": 33,
        },
        "spatial_block_content_sha256": control.get(
            "spatial_block_extraction", {}
        ).get("block_content_sha256"),
        "negative_controls": result.get("negative_controls"),
        "candidate_set_corruption_negative": {
            "status": negative.get("status"),
            "errors": negative.get("errors"),
        },
        "scope": result.get("scope"),
    }


def _quartic_horndeski_covariant_adm_control(
    root: Path,
) -> tuple[bool, dict[str, Any]]:
    # Local import avoids the action_ir -> formal_backend validation import cycle.
    from .action_ir import compile_action_file

    action_ir = compile_action_file(
        root / "configs" / "actions" / "quartic_horndeski_control.json",
        root / "configs" / "covariant_action_grammar.json",
        root / "configs" / "covariant_field_contract.json",
    )
    evidence = quartic_horndeski_covariant_adm_control()
    term_ids = {item["id"] for item in action_ir.get("canonical", {}).get("terms", [])}
    expected_terms = {"EH_R", "SCALAR_X", "HORNDESKI_L4_LINEAR_X"}
    action_bound = action_ir.get("valid") is True and term_ids == expected_terms
    evidence["input_action_sha256"] = action_ir.get("content_sha256")
    evidence["action_ir_valid"] = action_ir.get("valid", False)
    evidence["action_term_ids"] = sorted(term_ids)
    evidence["action_binding_passed"] = action_bound
    return bool(evidence["passed"] and action_bound), evidence


def _quartic_horndeski_unitary_flrw_dirac_control(
    root: Path,
) -> tuple[bool, dict[str, Any]]:
    # Bind the reduced constraint calculation to the same named covariant
    # action instead of allowing an unattached minisuperspace toy model to
    # count as theory evidence.
    from .action_ir import compile_action_file

    action_ir = compile_action_file(
        root / "configs" / "actions" / "quartic_horndeski_control.json",
        root / "configs" / "covariant_action_grammar.json",
        root / "configs" / "covariant_field_contract.json",
    )
    evidence = quartic_horndeski_unitary_flrw_dirac_control()
    term_ids = {item["id"] for item in action_ir.get("canonical", {}).get("terms", [])}
    expected_terms = {"EH_R", "SCALAR_X", "HORNDESKI_L4_LINEAR_X"}
    action_bound = action_ir.get("valid") is True and term_ids == expected_terms
    evidence["input_action_sha256"] = action_ir.get("content_sha256")
    evidence["action_ir_valid"] = action_ir.get("valid", False)
    evidence["action_term_ids"] = sorted(term_ids)
    evidence["expected_action_term_ids"] = sorted(expected_terms)
    evidence["action_binding_passed"] = action_bound
    return bool(evidence["passed"] and action_bound), evidence


def _quartic_horndeski_scalar_variation_control(
    root: Path, backend: dict[str, Any]
) -> tuple[bool, dict[str, Any]]:
    from .action_ir import compile_action_file

    action_ir = compile_action_file(
        root / "configs" / "actions" / "quartic_horndeski_control.json",
        root / "configs" / "covariant_action_grammar.json",
        root / "configs" / "covariant_field_contract.json",
    )
    cadabra_passed, cadabra = run_cadabra_script(
        root,
        backend,
        "formal/cadabra/quartic_horndeski_scalar_variation.cdb",
        [
            "SIGMA_QUARTIC_HORNDESKI_SCALAR_VARIATION_FINAL",
            "\\nabla^{a}(\\nabla_{a}(\\nabla^{b}(\\nabla_{b}(φ))))",
            "R",
        ],
    )
    reduction_passed, reduction = quartic_horndeski_scalar_euler_reduction_control()
    term_ids = {item["id"] for item in action_ir.get("canonical", {}).get("terms", [])}
    expected_terms = {"EH_R", "SCALAR_X", "HORNDESKI_L4_LINEAR_X"}
    action_bound = action_ir.get("valid") is True and term_ids == expected_terms
    script = root / "formal" / "cadabra" / "quartic_horndeski_scalar_variation.cdb"
    evidence = {
        "input_action_sha256": action_ir.get("content_sha256"),
        "action_ir_valid": action_ir.get("valid", False),
        "action_term_ids": sorted(term_ids),
        "expected_action_term_ids": sorted(expected_terms),
        "action_binding_passed": action_bound,
        "cadabra_variation": cadabra,
        "cadabra_script_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
        "second_order_reduction": reduction,
        "algorithm_chain": [
            "vary the action-IR-bound covariant scalar sector in Cadabra",
            "integrate first- and second-derivative delta(phi) terms by parts",
            "apply the scalar-Hessian curvature commutator",
            "apply the exact contracted Bianchi identity",
            "verify cancellation of fourth derivatives and curvature-gradient terms",
        ],
        "scope": reduction["scope"],
    }
    passed = bool(action_bound and cadabra_passed and reduction_passed)
    return passed, evidence


def _quartic_horndeski_metric_noether_control(
    root: Path, backend: dict[str, Any]
) -> tuple[bool, dict[str, Any]]:
    from .action_ir import compile_action_file

    action_ir = compile_action_file(
        root / "configs" / "actions" / "quartic_horndeski_control.json",
        root / "configs" / "covariant_action_grammar.json",
        root / "configs" / "covariant_field_contract.json",
    )
    script_specs = {
        "metric_variation": (
            "formal/cadabra/quartic_horndeski_metric_variation.cdb",
            [
                "SIGMA_QUARTIC_HORNDESKI_METRIC_VARIATION_RAW_PALATINI",
                "SIGMA_QUARTIC_HORNDESKI_METRIC_VARIATION_NO_H_DERIVATIVES",
                "SIGMA_QUARTIC_HORNDESKI_METRIC_VARIATION_FINAL",
            ],
        ),
        "euler_noether_identity": (
            "formal/cadabra/quartic_horndeski_noether_euler_identity.cdb",
            [
                "SIGMA_QUARTIC_HORNDESKI_NOETHER_EULER_COEFFICIENT_DERIVED",
                "SIGMA_QUARTIC_HORNDESKI_NOETHER_CORRUPTED_SIGN_REJECTED",
            ],
        ),
        "action_diffeomorphism_covariance": (
            "formal/cadabra/quartic_horndeski_diffeomorphism_covariance.cdb",
            [
                "SIGMA_QUARTIC_HORNDESKI_DIFFEO_COVARIANCE_ZERO",
                "SIGMA_QUARTIC_HORNDESKI_OMITTED_TENSOR_INDEX_TERMS_REJECTED",
            ],
        ),
    }
    scripts: dict[str, Any] = {}
    scripts_pass = True
    for name, (relative_path, fragments) in script_specs.items():
        passed, execution = run_cadabra_script(root, backend, relative_path, fragments)
        path = root / relative_path
        execution["script_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        scripts[name] = execution
        scripts_pass = scripts_pass and passed
    reduced_passed, reduced = quartic_horndeski_boundary_and_flrw_noether_control()
    term_ids = {item["id"] for item in action_ir.get("canonical", {}).get("terms", [])}
    expected_terms = {"EH_R", "SCALAR_X", "HORNDESKI_L4_LINEAR_X"}
    action_bound = action_ir.get("valid") is True and term_ids == expected_terms
    evidence = {
        "input_action_sha256": action_ir.get("content_sha256"),
        "action_ir_valid": action_ir.get("valid", False),
        "action_term_ids": sorted(term_ids),
        "expected_action_term_ids": sorted(expected_terms),
        "action_binding_passed": action_bound,
        "boundary_and_flrw_noether": reduced,
        "scripts": scripts,
        "metric_variation_method": (
            "Cadabra expands the full inverse-metric Palatini variation of the boundary-equivalent "
            "G_ab grad^a(phi) grad^b(phi) density; the exact twice-integrated Palatini adjoint "
            "is instantiated explicitly because Cadabra 2.4 does not reliably integrate two "
            "derivatives off a tensor-valued metric variation"
        ),
        "source_binding": (
            "metric and scalar Euler scripts, action-density covariance, and the Euler Noether "
            "coefficient are all bound to the same compiled Horndeski action hash"
        ),
        "scope": (
            "exact arbitrary-background covariant metric Euler construction and structural "
            "metric-scalar Noether identity, with exact nonlinear lapse-FLRW corroboration"
        ),
    }
    return bool(action_bound and scripts_pass and reduced_passed), evidence


def _quartic_horndeski_timelike_flat_principal_control(
    root: Path,
) -> tuple[bool, dict[str, Any]]:
    from .action_ir import compile_action_file

    action_ir = compile_action_file(
        root / "configs" / "actions" / "quartic_horndeski_control.json",
        root / "configs" / "covariant_action_grammar.json",
        root / "configs" / "covariant_field_contract.json",
    )
    control_passed, evidence = quartic_horndeski_timelike_flat_principal_control()
    term_ids = {item["id"] for item in action_ir.get("canonical", {}).get("terms", [])}
    expected_terms = {"EH_R", "SCALAR_X", "HORNDESKI_L4_LINEAR_X"}
    action_bound = action_ir.get("valid") is True and term_ids == expected_terms
    evidence["input_action_sha256"] = action_ir.get("content_sha256")
    evidence["action_ir_valid"] = action_ir.get("valid", False)
    evidence["action_term_ids"] = sorted(term_ids)
    evidence["expected_action_term_ids"] = sorted(expected_terms)
    evidence["action_binding_passed"] = action_bound
    return bool(control_passed and action_bound), evidence


def _quartic_horndeski_arbitrary_curvature_scalar_principal_control(
    root: Path,
) -> tuple[bool, dict[str, Any]]:
    from .action_ir import compile_action_file

    action_ir = compile_action_file(
        root / "configs" / "actions" / "quartic_horndeski_control.json",
        root / "configs" / "covariant_action_grammar.json",
        root / "configs" / "covariant_field_contract.json",
    )
    control_passed, evidence = (
        quartic_horndeski_arbitrary_curvature_scalar_principal_control()
    )
    term_ids = {item["id"] for item in action_ir.get("canonical", {}).get("terms", [])}
    expected_terms = {"EH_R", "SCALAR_X", "HORNDESKI_L4_LINEAR_X"}
    action_bound = action_ir.get("valid") is True and term_ids == expected_terms
    evidence["input_action_sha256"] = action_ir.get("content_sha256")
    evidence["action_ir_valid"] = action_ir.get("valid", False)
    evidence["action_term_ids"] = sorted(term_ids)
    evidence["expected_action_term_ids"] = sorted(expected_terms)
    evidence["action_binding_passed"] = action_bound
    return bool(control_passed and action_bound), evidence


def _quartic_horndeski_coupled_formulation_hyperbolicity_control(
    root: Path,
) -> tuple[bool, dict[str, Any]]:
    from .action_ir import compile_action_file

    action_ir = compile_action_file(
        root / "configs" / "actions" / "quartic_horndeski_control.json",
        root / "configs" / "covariant_action_grammar.json",
        root / "configs" / "covariant_field_contract.json",
    )
    control_passed, evidence = (
        quartic_horndeski_coupled_formulation_hyperbolicity_control()
    )
    term_ids = {item["id"] for item in action_ir.get("canonical", {}).get("terms", [])}
    expected_terms = {"EH_R", "SCALAR_X", "HORNDESKI_L4_LINEAR_X"}
    action_bound = action_ir.get("valid") is True and term_ids == expected_terms
    evidence["input_action_sha256"] = action_ir.get("content_sha256")
    evidence["action_ir_valid"] = action_ir.get("valid", False)
    evidence["action_term_ids"] = sorted(term_ids)
    evidence["expected_action_term_ids"] = sorted(expected_terms)
    evidence["action_binding_passed"] = action_bound
    return bool(control_passed and action_bound), evidence


def _quartic_horndeski_full_local_principal_control(
    root: Path,
) -> tuple[bool, dict[str, Any]]:
    from .action_ir import compile_action_file

    action_ir = compile_action_file(
        root / "configs" / "actions" / "quartic_horndeski_control.json",
        root / "configs" / "covariant_action_grammar.json",
        root / "configs" / "covariant_field_contract.json",
    )
    control_passed, evidence = quartic_horndeski_full_local_principal_control()
    term_ids = {item["id"] for item in action_ir.get("canonical", {}).get("terms", [])}
    expected_terms = {"EH_R", "SCALAR_X", "HORNDESKI_L4_LINEAR_X"}
    action_bound = action_ir.get("valid") is True and term_ids == expected_terms
    evidence["input_action_sha256"] = action_ir.get("content_sha256")
    evidence["action_ir_valid"] = action_ir.get("valid", False)
    evidence["action_term_ids"] = sorted(term_ids)
    evidence["expected_action_term_ids"] = sorted(expected_terms)
    evidence["action_binding_passed"] = action_bound
    return bool(control_passed and action_bound), evidence


def _quartic_horndeski_unitary_distributed_dirac_control(
    root: Path,
) -> tuple[bool, dict[str, Any]]:
    from .action_ir import compile_action_file

    action_ir = compile_action_file(
        root / "configs" / "actions" / "quartic_horndeski_control.json",
        root / "configs" / "covariant_action_grammar.json",
        root / "configs" / "covariant_field_contract.json",
    )
    control_passed, evidence = quartic_horndeski_unitary_distributed_dirac_control()
    term_ids = {item["id"] for item in action_ir.get("canonical", {}).get("terms", [])}
    expected_terms = {"EH_R", "SCALAR_X", "HORNDESKI_L4_LINEAR_X"}
    action_bound = action_ir.get("valid") is True and term_ids == expected_terms
    evidence["input_action_sha256"] = action_ir.get("content_sha256")
    evidence["action_ir_valid"] = action_ir.get("valid", False)
    evidence["action_term_ids"] = sorted(term_ids)
    evidence["expected_action_term_ids"] = sorted(expected_terms)
    evidence["action_binding_passed"] = action_bound
    return bool(control_passed and action_bound), evidence


def _quartic_horndeski_timelike_flat_hamiltonian_control(
    root: Path,
) -> tuple[bool, dict[str, Any]]:
    from .action_ir import compile_action_file

    action_ir = compile_action_file(
        root / "configs" / "actions" / "quartic_horndeski_control.json",
        root / "configs" / "covariant_action_grammar.json",
        root / "configs" / "covariant_field_contract.json",
    )
    control_passed, evidence = quartic_horndeski_timelike_flat_hamiltonian_control()
    term_ids = {item["id"] for item in action_ir.get("canonical", {}).get("terms", [])}
    expected_terms = {"EH_R", "SCALAR_X", "HORNDESKI_L4_LINEAR_X"}
    action_bound = action_ir.get("valid") is True and term_ids == expected_terms
    evidence["input_action_sha256"] = action_ir.get("content_sha256")
    evidence["action_ir_valid"] = action_ir.get("valid", False)
    evidence["action_term_ids"] = sorted(term_ids)
    evidence["expected_action_term_ids"] = sorted(expected_terms)
    evidence["action_binding_passed"] = action_bound
    return bool(control_passed and action_bound), evidence


def _quartic_horndeski_global_timelike_gradient_no_go_control(
    root: Path,
) -> tuple[bool, dict[str, Any]]:
    from .action_ir import compile_action_file

    action_ir = compile_action_file(
        root / "configs" / "actions" / "quartic_horndeski_control.json",
        root / "configs" / "covariant_action_grammar.json",
        root / "configs" / "covariant_field_contract.json",
    )
    control_passed, evidence = quartic_horndeski_global_timelike_gradient_no_go_control()
    term_ids = {item["id"] for item in action_ir.get("canonical", {}).get("terms", [])}
    expected_terms = {"EH_R", "SCALAR_X", "HORNDESKI_L4_LINEAR_X"}
    action_bound = action_ir.get("valid") is True and term_ids == expected_terms
    evidence["input_action_sha256"] = action_ir.get("content_sha256")
    evidence["action_ir_valid"] = action_ir.get("valid", False)
    evidence["action_term_ids"] = sorted(term_ids)
    evidence["expected_action_term_ids"] = sorted(expected_terms)
    evidence["action_binding_passed"] = action_bound
    return bool(control_passed and action_bound), evidence


def _quartic_horndeski_flrw_domain_crossing_control(
    root: Path,
) -> tuple[bool, dict[str, Any]]:
    from .action_ir import compile_action_file

    action_ir = compile_action_file(
        root / "configs" / "actions" / "quartic_horndeski_control.json",
        root / "configs" / "covariant_action_grammar.json",
        root / "configs" / "covariant_field_contract.json",
    )
    control_passed, evidence = quartic_horndeski_flrw_domain_crossing_control()
    term_ids = {item["id"] for item in action_ir.get("canonical", {}).get("terms", [])}
    expected_terms = {"EH_R", "SCALAR_X", "HORNDESKI_L4_LINEAR_X"}
    action_bound = action_ir.get("valid") is True and term_ids == expected_terms
    evidence["input_action_sha256"] = action_ir.get("content_sha256")
    evidence["action_ir_valid"] = action_ir.get("valid", False)
    evidence["action_term_ids"] = sorted(term_ids)
    evidence["expected_action_term_ids"] = sorted(expected_terms)
    evidence["action_binding_passed"] = action_bound
    return bool(control_passed and action_bound), evidence


def _einstein_aether_modes() -> tuple[bool, dict[str, Any]]:
    c1 = sp.Rational(1, 10)
    c2 = sp.Rational(1, 20)
    c3 = sp.Integer(0)
    c4 = sp.Rational(1, 20)
    c13, c14, c123 = c1 + c3, c1 + c4, c1 + c2 + c3
    spin2 = sp.simplify(1 / (1 - c13))
    spin1 = sp.simplify((2 * c1 - c1**2 + c3**2) / (2 * c14 * (1 - c13)))
    spin0 = sp.simplify(c123 * (2 - c14) / (c14 * (1 - c13) * (2 + c13 + 3 * c2)))
    speeds = {"spin_2": spin2, "spin_1": spin1, "spin_0": spin0}
    passed = all(value.is_positive for value in speeds.values()) and c13 != 1 and c14 != 0
    return passed, {
        "action": "Einstein-Hilbert plus c1..c4 unit-timelike-vector kinetic invariants and lambda(u^2+1)",
        "couplings": {"c1": str(c1), "c2": str(c2), "c3": str(c3), "c4": str(c4)},
        "mode_speed_squared": {name: str(value) for name, value in speeds.items()},
        "mode_count": {"tensor": 2, "vector": 2, "scalar": 1, "total": 5},
        "domain_checks": {"1-c13_nonzero": c13 != 1, "c14_nonzero": c14 != 0},
        "interpretation": "Known linearized Minkowski mode formulas are evaluated at one healthy algebraic control point; this is not a global viability claim for Einstein-Aether theory.",
    }


def _field_contract_control(contract: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    result = validate_field_contract(contract)
    invalid_spec = {
        "matter_metric": "g_mu_nu",
        "invariants": ["z_b"],
        "action": "F(z_b)",
        "static_dictionary_status": "derived",
    }
    rejected = validate_covariant_action_spec(invalid_spec, contract)
    result["baryon_specific_negative_control"] = rejected
    return result["valid"] and not rejected["valid"], result


def _principal_symbol_control() -> tuple[bool, dict[str, Any]]:
    evidence = run_principal_symbol_controls()
    return bool(evidence["passed"]), evidence


def _anisotropic_principal_symbol_control() -> tuple[bool, dict[str, Any]]:
    evidence = run_anisotropic_principal_symbol_controls()
    return bool(evidence["passed"]), evidence


def _extracted_principal_symbol_control() -> tuple[bool, dict[str, Any]]:
    evidence = run_extracted_principal_symbol_controls()
    return bool(evidence["passed"]), evidence


def _uniform_multifield_block_control() -> tuple[bool, dict[str, Any]]:
    evidence = run_uniform_multifield_block_controls()
    return bool(evidence["passed"]), evidence


def _uniform_scalar_anisotropy_control() -> tuple[bool, dict[str, Any]]:
    evidence = run_uniform_scalar_anisotropy_controls()
    return bool(evidence["passed"]), evidence


def _einstein_hilbert_adm_control() -> tuple[bool, dict[str, Any]]:
    evidence = linearized_einstein_hilbert_adm_control()
    return bool(evidence["passed"]), evidence


def _nonlinear_adm_hamiltonian_constraint_control() -> tuple[bool, dict[str, Any]]:
    evidence = nonlinear_adm_hamiltonian_constraint_control()
    return bool(evidence["passed"]), evidence


def _spatial_curvature_density_diffeomorphism_control() -> tuple[bool, dict[str, Any]]:
    evidence = spatial_curvature_density_diffeomorphism_control()
    return bool(evidence["passed"]), evidence


def _curved_background_principal_control() -> tuple[bool, dict[str, Any]]:
    evidence = curved_background_principal_controls()
    return bool(evidence["passed"]), evidence


def _candidate_cadabra_roots(project_root: Path) -> list[Path]:
    roots: list[Path] = []
    if os.environ.get("SIGMA_CADABRA_ROOT"):
        roots.append(Path(os.environ["SIGMA_CADABRA_ROOT"]))
    # Default produced by scripts/bootstrap_cadabra_wsl.ps1.
    roots.append(project_root.parent.parent / "work" / "cadabra2-root" / "root")
    return roots


def probe_cadabra(project_root: str | Path) -> dict[str, Any]:
    project_root = Path(project_root).resolve()
    native = shutil.which("cadabra2")
    if native:
        return {"available": True, "mode": "native", "executable": native}
    wsl = shutil.which("wsl")
    if not wsl:
        return {
            "available": False,
            "mode": "missing",
            "reason": "wsl and native cadabra2 are absent",
        }
    for root in _candidate_cadabra_roots(project_root):
        executable = root / "usr" / "bin" / "cadabra2"
        module = (
            root
            / "usr"
            / "lib"
            / "python3"
            / "dist-packages"
            / "cadabra2.cpython-312-x86_64-linux-gnu.so"
        )
        if executable.exists() and module.exists():
            return {
                "available": True,
                "mode": "wsl-local",
                "root": str(root),
                "executable": str(executable),
                "python_module": str(module),
                "version": "2.4.5.4",
            }
    completed = subprocess.run(
        [wsl, "-d", "Ubuntu-24.04", "--", "bash", "-lc", "command -v cadabra2"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        timeout=20,
    )
    if completed.returncode == 0 and completed.stdout.strip():
        return {
            "available": True,
            "mode": "wsl-system",
            "executable": completed.stdout.strip(),
        }
    return {
        "available": False,
        "mode": "missing",
        "reason": "Cadabra 2 is not installed; run scripts/bootstrap_cadabra_wsl.ps1",
    }


def _windows_to_wsl(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").casefold()
    if not drive:
        raise ValueError(f"A Windows drive path is required for WSL translation: {resolved}")
    tail = resolved.as_posix().split(":", 1)[1]
    return f"/mnt/{drive}{tail}"


def run_cadabra_script(
    project_root: Path,
    backend: dict[str, Any],
    script: str | Path,
    expected_fragments: list[str],
) -> tuple[bool, dict[str, Any]]:
    project_root = Path(project_root).resolve()
    script = Path(script)
    if not script.is_absolute():
        script = project_root / script
    if not script.exists():
        return False, {"reason": f"missing Cadabra control script: {script}"}
    mode = backend.get("mode")
    if mode == "native":
        command = [backend["executable"], str(script)]
    elif mode in {"wsl-local", "wsl-system"}:
        wsl = shutil.which("wsl")
        if not wsl:
            return False, {"reason": "wsl executable disappeared after backend probe"}
        script_wsl = _windows_to_wsl(script)
        if mode == "wsl-local":
            root_wsl = _windows_to_wsl(Path(backend["root"]))
            shell_command = (
                f"PYTHONPATH={shlex.quote(root_wsl + '/usr/lib/python3/dist-packages')} "
                f"LD_LIBRARY_PATH={shlex.quote(root_wsl + '/usr/lib/x86_64-linux-gnu')} "
                f"{shlex.quote(root_wsl + '/usr/bin/cadabra2')} {shlex.quote(script_wsl)}"
            )
        else:
            shell_command = f"cadabra2 {shlex.quote(script_wsl)}"
        command = [wsl, "-d", "Ubuntu-24.04", "--", "bash", "-lc", shell_command]
    else:
        return False, {"reason": "Cadabra backend is unavailable", "backend": backend}
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        timeout=60,
    )
    normalized = completed.stdout.replace(" ", "").replace("\r", "")
    passed = completed.returncode == 0 and all(
        fragment.replace(" ", "") in normalized for fragment in expected_fragments
    )
    return passed, {
        "script": str(script),
        "backend_mode": mode,
        "return_code": completed.returncode,
        "expected_fragments": expected_fragments,
        "stdout_tail": completed.stdout[-1000:],
        "stderr_tail": completed.stderr[-1000:],
    }


def run_cadabra_metric_control(
    project_root: str | Path, backend: dict[str, Any]
) -> tuple[bool, dict[str, Any]]:
    return run_cadabra_script(
        Path(project_root), backend, "formal/cadabra/metric_identity.cdb", ["g_{a}^{c}"]
    )


def run_cadabra_scalar_variation_control(
    project_root: str | Path, backend: dict[str, Any]
) -> tuple[bool, dict[str, Any]]:
    passed, evidence = run_cadabra_script(
        Path(project_root),
        backend,
        "formal/cadabra/canonical_scalar_variation.cdb",
        ["δ(φ)", "\\partial^{", "(m)**2"],
    )
    evidence["expected_euler_lagrange_equation"] = "box(phi) - m^2 phi = 0"
    evidence["algorithm_chain"] = [
        "vary",
        "integrate_by_parts",
        "canonicalise",
        "sort_product",
        "factor_out",
    ]
    return passed, evidence


def run_cadabra_adm_curvature_variation_control(
    project_root: str | Path, backend: dict[str, Any]
) -> tuple[bool, dict[str, Any]]:
    root = Path(project_root)
    relative_script = "formal/cadabra/adm_spatial_curvature_variation.cdb"
    passed, evidence = run_cadabra_script(
        root,
        backend,
        relative_script,
        [
            "SIGMA_ADM_CURVATURE_PALATINI_BOUNDARY_TRACKED",
            "SIGMA_ADM_CURVATURE_FIRST_IBP_EXPLICIT",
            "SIGMA_ADM_CURVATURE_NO_H_DERIVATIVES",
            "SIGMA_ADM_CURVATURE_FULLY_COVARIANT",
            "SIGMA_ADM_CURVATURE_LAPSE_DERIVATIVES_RETAINED",
            "SIGMA_ADM_CURVATURE_VARIATION_FINAL",
        ],
    )
    script_path = root / relative_script
    evidence["script_sha256"] = hashlib.sha256(script_path.read_bytes()).hexdigest()
    evidence["euler_coefficient"] = "sqrt(q)[N G^{ij}+q^{ij} nabla^2 N-nabla^i nabla^j N]"
    evidence["boundary_policy"] = (
        "first covariant divergence step explicit; second executed by Cadabra; compact support "
        "or matching spatial boundary term"
    )
    return passed, evidence


def run_cadabra_aether_noether_control(
    project_root: str | Path, backend: dict[str, Any]
) -> tuple[bool, dict[str, Any]]:
    """Bind Aether Euler variations to an exact arbitrary-background Noether proof."""

    root = Path(project_root).resolve()
    scripts = {
        "metric_euler": "formal/cadabra/einstein_aether_metric_variation.cdb",
        "vector_multiplier_euler": "formal/cadabra/einstein_aether_vector_variation.cdb",
        "euler_noether_coefficient": ("formal/cadabra/einstein_aether_noether_euler_identity.cdb"),
        "action_diffeomorphism_covariance": (
            "formal/cadabra/einstein_aether_diffeomorphism_covariance.cdb"
        ),
    }
    expected = {
        "metric_euler": [
            "SIGMA_AETHER_METRIC_VARIATION_NO_H_DERIVATIVES",
            "SIGMA_AETHER_METRIC_VARIATION_CONNECTION_INCLUDED",
            "SIGMA_AETHER_METRIC_VARIATION_FINAL",
        ],
        "vector_multiplier_euler": [
            "SIGMA_AETHER_VECTOR_VARIATION_FINAL",
            "SIGMA_AETHER_NORM_CONSTRAINT_FINAL",
        ],
        "euler_noether_coefficient": [
            "SIGMA_AETHER_4D_NOETHER_EULER_COEFFICIENT_DERIVED",
            "SIGMA_AETHER_4D_NOETHER_CORRUPTED_SIGN_REJECTED",
        ],
        "action_diffeomorphism_covariance": [
            "SIGMA_AETHER_K1_K2_K3_K4_UNIT_DIFFEO_COVARIANCE_ZERO",
            "SIGMA_AETHER_OMITTED_CONNECTION_TERMS_REJECTED",
        ],
    }
    results: dict[str, Any] = {}
    passed = True
    for role, relative in scripts.items():
        item_passed, item_evidence = run_cadabra_script(root, backend, relative, expected[role])
        script_path = root / relative
        item_evidence["sha256"] = hashlib.sha256(script_path.read_bytes()).hexdigest()
        item_evidence["passed"] = item_passed
        results[role] = item_evidence
        passed = passed and item_passed
    return passed, {
        "passed": passed,
        "dimension": 4,
        "dimension_dependence": (
            "the abstract-index derivation is dimension independent and therefore includes 4D"
        ),
        "independent_field_convention": "fixed covector u_a, inverse metric g^ab, multiplier lambda",
        "identity": (
            "2 nabla^a E^(g)_ab + E_u^a nabla_b u_a "
            "- nabla_a(E_u^a u_b) + E_lambda nabla_b lambda = 0"
        ),
        "termwise_scope": (
            "K1, K2, K3, K4, and the unit constraint carry independent symbolic coefficients, "
            "so the zero combined polynomial establishes each coefficient separately"
        ),
        "proof_chain": [
            "derive the complete fixed-covector metric Euler coefficient including delta Gamma",
            "derive the vector and multiplier Euler coefficients",
            "insert exact field Lie derivatives and integrate derivatives of xi by parts",
            "derive the displayed Euler Noether coefficient",
            "prove the complete action-density Lie variation is a covariant divergence",
        ],
        "negative_controls": [
            "corrupt the metric-Euler divergence sign",
            "omit both D_ab connection/index Lie-derivative terms",
        ],
        "scripts": results,
        "scope": (
            "exact off-shell arbitrary-background abstract-tensor identity for the standard "
            "Einstein-Aether K1..K4 plus unit action in the fixed-covector convention"
        ),
    }


def run_formal_control_suite(
    contract_path: str | Path,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    contract_path = Path(contract_path).resolve()
    root = Path(project_root).resolve() if project_root else contract_path.parent.parent
    contract = load_field_contract(contract_path)
    cadabra = probe_cadabra(root)
    checks = [
        _run_check(
            "covariant_field_contract",
            "The field content is explicit and baryon-specific z_b is rejected as a gravitational-action atom.",
            "Exact schema and policy validation.",
            lambda: _field_contract_control(contract),
        ),
        _run_check(
            "projected_aether_q_fixed_metric_first_variation",
            "The Q_a_u projector, acceleration, and acceleration-gradient first variations agree with direct tensor differentiation.",
            "Exact rational fixed-metric tensor contraction; connection-dependent metric variation and covariant boundary completion remain separate.",
            _projected_aether_q_first_variation_control,
        ),
        _run_check(
            "projected_aether_q_generic_3plus1_decomposition",
            "The generic-tilt Q_a_u projector contraction has the declared exact ADM block form and exposes its normal higher-jet channels.",
            "Exact rational kinematic 3+1 identity; Dirac degeneracy, constraint closure, and boundedness remain separate.",
            projected_aether_q_3plus1_control,
        ),
        _run_check(
            "projected_aether_q_aligned_auxiliary_dirac",
            "The lifted aligned quadratic Q sector closes its complete finite-mode Dirac chain and reduces to one positive dispersive mode per polarization.",
            "Exact one-polarization Fourier Dirac reduction on the aligned frozen-metric background; generic tilt and gravitational constraint mixing remain separate.",
            projected_aether_q_aligned_auxiliary_dirac_control,
        ),
        _run_check(
            "projected_aether_q_constant_tilt_root_audit",
            "The complete nonzero-tilt lab-frequency polynomial exposes the nonreal Q root pair missed by physical-branch mapping alone.",
            "Exact monotone real-branch bijection, quartic leading coefficient, and Sturm root count on a rational interior point; this is a frozen-coefficient negative control.",
            projected_aether_q_constant_tilt_root_audit,
        ),
        _run_check(
            "nonlinear_aether_acceleration_global_convexity",
            "The covariant-first F_p(X_a) family has a positive full velocity Hessian for X_a>=0 and 1/2<=p<1.",
            "Exact radial Hessian eigenvalues and high-field scaling; coupled ADM/Dirac closure and nonlinear total energy remain separate.",
            nonlinear_aether_acceleration_convexity_control,
        ),
        _run_check(
            "static_null_k14_multiplicative_completion_no_go",
            "A positive decaying multiplicative W(X_a)(K1+K4) completion cannot be both globally shear-Legendre-regular and high-X speed bounded.",
            "Exact coupled shear Schur coefficient plus a C2 concavity contradiction; applies only to the declared multiplicative completion class.",
            static_null_k14_multiplicative_no_go_control,
        ),
        _run_check(
            "einstein_hilbert_linearized_bianchi",
            "The linearized Einstein tensor obeys its off-shell differential identity.",
            "Exact Fourier-space identity around Minkowski; not nonlinear arbitrary-background variation.",
            _linearized_einstein_bianchi,
        ),
        _run_check(
            "einstein_hilbert_linearized_adm",
            "Linearized Einstein-Hilbert ADM has eight first-class constraints, two TT modes, a positive TT Hamiltonian, and the metric cone.",
            "Complete nonzero Fourier-mode control around Minkowski; nonlinear ADM remains separate.",
            _einstein_hilbert_adm_control,
        ),
        _run_check(
            "nonlinear_adm_hamiltonian_constraint_algebra",
            "The nonlinear pure-GR Hamiltonian constraints close into the metric-dependent spatial momentum constraint.",
            "Exact 3D covariant H-H bracket with general q_ij, pi^ij, lapse jets, DeWitt kinetic term, curvature lapse Hessian, and compact-support boundary reduction.",
            _nonlinear_adm_hamiltonian_constraint_control,
        ),
        _run_check(
            "spatial_curvature_density_diffeomorphism_covariance",
            "The spatial-curvature Hamiltonian potential transforms as a weight-one density under the canonical metric momentum constraint.",
            "Exact 3D D-H curvature-sector identity modulo a compact-support boundary; the lapse-smeared metric derivative is independently derived by Cadabra.",
            _spatial_curvature_density_diffeomorphism_control,
        ),
        _run_check(
            "canonical_scalar",
            "A canonical scalar has one positive kinetic mode and the metric characteristic cone.",
            "Exact quadratic control around Minkowski.",
            _canonical_scalar_control,
        ),
        _run_check(
            "canonical_scalar_noether_identity",
            "The canonical scalar stress tensor divergence equals its Euler derivative times the field gradient.",
            "Exact local tensor identity in Riemann normal coordinates at an arbitrary point.",
            _canonical_scalar_noether_identity,
        ),
        _run_check(
            "generic_g2_variation_noether_identity",
            "An arbitrary normalized G2(phi,X) scalar sector has the derived field Euler coefficient and metric stress tensor satisfying its off-shell Noether identity.",
            "Exact arbitrary local scalar-gradient/Hessian jet with independent G2 derivative coefficients and a corrupted metric-pressure-sign negative control; the G3 and scoped G4 controls are separate, as is constraint analysis.",
            generic_g2_variation_noether_control,
        ),
        _run_check(
            "generic_g3_variation_noether_identity",
            "An arbitrary cubic Horndeski -G3(phi,X) box(phi) sector has a covariantly derived second-order field equation and Hilbert stress tensor satisfying its off-shell Noether identity.",
            "Exact covariant contraction algebra on arbitrary scalar and Ricci jets, including the Hessian commutator, explicit third-derivative cancellation, and omitted-braiding-stress and omitted-Ricci negative controls; G4 and constraint analysis remain separate.",
            generic_g3_variation_noether_control,
        ),
        _run_check(
            "generic_g4_phi_variation_noether_identity",
            "Every smooth X-independent quartic-Horndeski function G4=F(phi) has the exact nonminimal F(phi)R metric/scalar variation and off-shell diffeomorphism identity.",
            "Exact arbitrary-background contracted-Bianchi and scalar-commutator proof with omitted metric-completion and wrong-scalar-sign negative controls; any G4_X dependence remains unresolved.",
            generic_g4_phi_variation_noether_control,
        ),
        _run_check(
            "generic_g4_fixed_metric_scalar_variation",
            "The arbitrary quartic-Horndeski G4(phi,X) scalar current is instantiated, its apparent higher derivatives cancel, and its full nonlinear-X metric-scalar Noether identity closes on a general flat third jet, with exact curved linear-X and phi-only reductions.",
            "Source-form equations B.4/B.8/B.12, automatic 4D component differentiation over 4 gradient, 10 Hessian, and 20 third-jet components, an omitted-G4_XX tensor-term negative, plus the covariant Einstein-tensor and F(phi)R controls; generic curved nonlinear-X metric/Noether closure remains unresolved.",
            generic_g4_scalar_variation_control,
        ),
        _run_check(
            "generic_g4_curved_rnc_exact_witnesses",
            "The complete source-form nonlinear G4(phi,X) metric and scalar Euler tensors satisfy their combined Noether identity on independent exact, fully curved four-dimensional local jets.",
            "Three deterministic exact-rational normal-frame witnesses generated from independent second/third metric Taylor coefficients and scalar Taylor jets; algebraic/contracted Bianchi, metric symmetry, four Noether components, nonzero curvature gradients, and an omitted-G4_XX term negative are checked. This is not yet a symbolic all-jet theorem.",
            generic_g4_curved_rnc_witness_control,
        ),
        _run_check(
            "generic_g4_curved_symbolic_all_jet_noether",
            "The complete source-form curved nonlinear G4(phi,X) metric and scalar Euler tensors satisfy their combined Noether identity as an exact symbolic polynomial on the full four-dimensional local jet.",
            "Exact Riemann-normal-coordinate expansion with 345 independent symbols covering arbitrary second/third metric Taylor data, scalar gradient/Hessian/third data, and the complete local G4 function 3-jet; Bianchi and commutator identities are derived from Taylor data, all residuals expand to zero, and an omitted-G4_XX term is nonzero. Independent Cadabra metric variation is enforced by a separate control.",
            generic_g4_curved_symbolic_rnc_control,
        ),
        _run_check(
            "canonical_scalar_gravity_cross_constraint_identities",
            "The canonical scalar Hamiltonian closes D-H as a spatial weight-one density, and its gravity-matter cross H-H terms cancel under lapse antisymmetrization.",
            "Exact local orthonormal-frame GL(3) density identity plus the ultralocal cross-HH antisymmetry residual; combined with the separate exact scalar and pure-GR H-H controls to certify the minimally coupled algebra.",
            _canonical_scalar_gravity_cross_constraint_control,
        ),
        _run_check(
            "proca_adm_dirac",
            "Massive Proca has a rank-three velocity Hessian, two second-class constraints, and three physical modes.",
            "Exact flat-background quadratic Hamiltonian control.",
            _proca_dirac_control,
        ),
        _run_check(
            "proca_divergence_identity",
            "The divergence of the Proca equation yields the Lorenz constraint for nonzero mass.",
            "Exact local principal-derivative identity and algebraic consequence of the vector equation.",
            _proca_divergence_identity,
        ),
        _run_check(
            "proca_stress_noether_identity",
            "The Proca stress divergence obeys the complete off-shell vector Noether identity.",
            "Exact 4D Minkowski component identity for arbitrary smooth A_mu; curved-background executable reduction remains separate.",
            _proca_stress_noether_identity,
        ),
        _run_check(
            "proca_curved_background_noether_identity",
            "The complete off-shell Proca stress Noether identity survives nonzero connection and volume-element terms.",
            "Exact FLRW homogeneous-vector and static-spherical radial-vector controls; not every metric/profile.",
            _proca_curved_noether_identity,
        ),
        _run_check(
            "einstein_aether_modes",
            "Known Einstein-Aether mode formulas give five real positive-speed modes at the declared control point.",
            "Known linearized Minkowski formulas; not a derivation of the full nonlinear constraint algebra.",
            _einstein_aether_modes,
        ),
        _run_check(
            "einstein_aether_flrw_variation_noether",
            "The complete K1..K4 plus unit-constraint action obeys its reduced nonlinear metric-vector-multiplier Noether identity.",
            "Exact lapse-FLRW homogeneous reduction with independent N, a, U, and lambda; not the full inhomogeneous identity.",
            _einstein_aether_flrw_variation_identity,
        ),
        _run_check(
            "einstein_aether_adm_kinetic_hessian",
            "The standard-sign Einstein–Æther ADM kinetic block has the expected generic rank and c14 vector coefficient.",
            "Exact pointwise aligned and rationally tilted unit-aether controls; secondary constraints and Poisson closure remain separate.",
            _einstein_aether_adm_kinetic,
        ),
        _run_check(
            "einstein_aether_generic_3plus1_legendre",
            "Every K1..K4 Einstein-Aether invariant has an exact spatially inhomogeneous 3+1 decomposition and a verified positive-unit-branch local Legendre map.",
            "Exact block decomposition, symbolic aligned determinant, and one inhomogeneous tilted rational nine-velocity Legendre patch; distributed Hamiltonian constraints and H-D/H-H brackets remain separate.",
            _einstein_aether_3plus1_decomposition,
        ),
        _run_check(
            "einstein_aether_generic_lapse_shift_constraint_seeds",
            "The generic K1..K4 positive-unit-branch Hamiltonian is bulk-linear in lapse and shift and generates the expected four primary and four secondary constraint seeds.",
            "Exact lapse-acceleration Legendre cancellation, boundary reduction, and spatial cotangent lift; H-H closure, global rank, nonlinear degree count, and boundedness remain unresolved.",
            _einstein_aether_lapse_shift_constraint_seed,
        ),
        _run_check(
            "einstein_aether_generic_dh_covariance",
            "The generic independent-c1..c4 Einstein-Aether Legendre Hamiltonian is a weight-one spatial scalar density and closes with the momentum constraint.",
            "Exact arbitrary-GL(3) tensor-contraction and canonical-density proof of D-H covariance; the normal-deformation H-H bracket, global rank, nonlinear degree count, and boundedness remain unresolved.",
            _einstein_aether_generic_dh_covariance,
        ),
        _run_check(
            "einstein_aether_generic_hh_deformation_kinematics",
            "The generic independent-c1..c4 Einstein-Aether Hamiltonian closes H-H into the spatial momentum constraint on every regular positive-unit-branch Legendre patch.",
            "Exact normal-embedding deformation algebra, Jacobi reduction, inverse-metric structure function, and Aether-specific -chi D_i N Hamilton-flow check; requires the separately executable arbitrary-background Noether control and assumes compact support or a completed boundary generator. Singular coupling strata and Hamiltonian boundedness remain unresolved.",
            _einstein_aether_generic_hh_deformation,
        ),
        _run_check(
            "einstein_aether_linearized_physical_energy",
            "All five reduced Einstein-Aether modes have positive cycle-averaged energy and positive squared speed at the declared rational Minkowski control point.",
            "Exact on-shell spin-2/spin-1/spin-0 wave-energy coefficients after linearized gauge and constraint reduction, with two positive-speed negative-energy controls and the restricted hypersurface-orthogonal nonlinear positive-energy domain; generic nonlinear Hamiltonian boundedness remains unresolved.",
            _einstein_aether_linearized_energy,
        ),
        _run_check(
            "einstein_aether_restricted_nonlinear_total_energy",
            "The exact asymptotic Einstein-Aether energy is nonnegative in the hypersurface-orthogonal maximal-slice sector throughout the declared coupling domain.",
            "Executable conformal-curvature and boundary-charge reduction to the Schoen-Yau positive-mass theorem with nonnegative matter energy, 0<=c14<=2, and c13<=1. Twisting Aether, nonmaximal data, and out-of-domain couplings remain unresolved rather than rejected; generic nonlinear reduced-Hamiltonian stability is not claimed.",
            _einstein_aether_nonlinear_positive_energy_theorem,
        ),
        _run_check(
            "einstein_aether_reduced_five_mode_principal_domain",
            "The complete reduced five-mode Einstein-Aether principal symbol is ghost-free, gradient-stable, real-characteristic, and strongly hyperbolic exactly on the certified open coupling domain.",
            "Necessary-and-sufficient aligned-Minkowski linearized certificate with exact spin-2/spin-1/spin-0 kinetic and gradient matrices, five negative witnesses, and six singular or strong-coupling strata; arbitrary nonlinear backgrounds, global tilted strata, and observational cone cuts remain outside this proof.",
            _einstein_aether_reduced_principal_domain,
        ),
        _run_check(
            "einstein_aether_global_tilt_legendre_strata",
            "The unit-reduced nine-velocity Einstein-Aether Legendre determinant factors globally by spin sector, and every healthy superluminal sector becomes singular exactly when the foliation normal reaches its characteristic cone.",
            "Exact pointwise theorem for every unit-timelike tilt magnitude and orientation by spatial rotational covariance, with a rank-loss witness and a globally subluminal noncharacteristic certificate; arbitrary inhomogeneous-background principal symbols, boundary charges, and nonlinear Hamiltonian boundedness remain outside this proof.",
            _einstein_aether_global_tilt_legendre,
        ),
        _run_check(
            "einstein_aether_covariant_arbitrary_background_hyperbolicity",
            "A covariant Aether-aligned first-order tetrad formulation is strongly hyperbolic on arbitrary smooth vacuum backgrounds throughout its declared sufficient coupling domain.",
            "Executable five-mode covariant effective-cone and exact-boost controls plus the Sarbach-Barausse-Preciado-Lopez frozen-principal theorem: all speeds positive and finite with nonluminal spin-1 and spin-0 sectors. Luminal formulation boundaries remain unresolved rather than rejected; nonlinear Hamiltonian boundedness and generated-action automation are separate.",
            _einstein_aether_covariant_strong_hyperbolicity,
        ),
        _run_check(
            "einstein_aether_coupled_unit_normal",
            "The unit-norm constraint remains second-class after the complete pointwise metric-vector kinetic mixing is retained.",
            "Exact inverse-kinetic normality in aligned, axis-tilted, and oblique rational unit-timelike patches; spatial Hamiltonian brackets, global coupling-domain regularity, and reduced stability remain separate.",
            _einstein_aether_coupled_unit_normal,
        ),
        _run_check(
            "einstein_aether_spatial_diffeomorphism_algebra",
            "The canonical spatial metric, Aether covector, normal Aether scalar, and their momentum densities form the exact cotangent lift of three-dimensional diffeomorphisms.",
            "Exact Einstein-Aether D-D momentum-constraint sector; unit/Hamiltonian constraints, higher consistency, and reduced Hamiltonian remain separate.",
            _einstein_aether_spatial_diffeomorphism,
        ),
        _run_check(
            "unit_timelike_vector_dirac_chain",
            "A regular unit-timelike vector kinetic block generates the complete multiplier, norm, tangency, and multiplier-fixing Dirac chain with three physical vector modes.",
            "Exact finite-point four-generation unit-vector control; spatial derivatives and coupling to the full Einstein-Aether metric Hamiltonian remain separate.",
            _unit_timelike_vector_dirac_chain,
        ),
        _run_check(
            "regular_holonomic_multiplier_dirac_theorem",
            "A regular holonomic constraint retains a four-second-class multiplier chain under arbitrary coordinate-dependent kinetic mixing and potential terms.",
            "Exact local dimension-independent Dirac theorem on the patch C_,A G^AB C_,B != 0; it does not establish lapse/shift Hamiltonian constraints or H-D/H-H closure for a specific field theory.",
            _regular_holonomic_multiplier_dirac,
        ),
        _run_check(
            "maxwell_unit_aether_nonlinear_hamiltonian",
            "The Maxwell-form unit-Aether subclass has a complete nonlinear hypersurface-deformation algebra and five physical modes, but its reduced Hamiltonian is unbounded below.",
            "Exact nonlinear c3=-c1, c2=c4=0 subclass after solving the positive unit branch; this is a stability-reject control and not the generic K1..K4 Einstein-Aether Hamiltonian.",
            _maxwell_unit_aether_nonlinear_hamiltonian,
        ),
        _run_check(
            "dirac_constraint_surface_poisson_rank",
            "Poisson rank is evaluated after quotient-ring reduction on the full constraint surface.",
            "Exact finite-dimensional polynomial negative control for an off-surface structure-function false positive.",
            _constraint_surface_poisson_rank_control,
        ),
        _run_check(
            "dirac_tertiary_constraint_chain",
            "The reusable Dirac engine iterates beyond secondary constraints and closes a tertiary chain.",
            "Exact finite-dimensional velocity-quadratic control with two primary, two secondary, and two higher-generation constraints.",
            _tertiary_dirac_chain_control,
        ),
        _run_check(
            "field_theory_smeared_constraint_algebra",
            "Smeared Hamiltonian and spatial-diffeomorphism constraints close with the expected derivative-of-smearing structure.",
            "Exact 1+1 local-functional Virasoro/hypersurface-deformation control modulo spatial boundary terms; not the 3+1 Einstein-Aether algebra.",
            _field_theory_constraint_algebra_control,
        ),
        _run_check(
            "three_spatial_dimensional_smeared_brackets",
            "Three-dimensional smeared Hamiltonian-Hamiltonian and spatial-diffeomorphism brackets reproduce their expected local-functional algebra.",
            "Exact three-spatial-dimensional scalar canonical control, with DD equality modulo spatial boundaries; the canonical metric sector and mixed gravity bracket remain separate.",
            _three_dimensional_field_bracket_control,
        ),
        _run_check(
            "proca_reduced_smeared_constraint_algebra",
            "Massive Proca's reduced three-dimensional Hamiltonian closes H-H into its covector momentum constraint after the normal-component second-class pair is solved.",
            "Exact nonlinear local-functional bracket modulo compact-support spatial boundaries, with all A_i, p^i, lapse-N, and lapse-M Euler residuals zero and a nonzero primary-secondary bracket for m_A>0.",
            _proca_reduced_field_bracket_control,
        ),
        _run_check(
            "canonical_metric_diffeomorphism_algebra",
            "The six-component spatial metric and its symmetric momentum density form an exact cotangent lift of three-dimensional diffeomorphisms, and the momentum constraints close.",
            "Exact 3D canonical-metric D-D algebra via componentwise Lie-generator identities; the curvature-dependent GR H-H bracket remains separate.",
            _canonical_metric_diffeomorphism_control,
        ),
        _run_check(
            "canonical_metric_dewitt_kinetic_covariance",
            "The complete six-component DeWitt kinetic Hamiltonian is a spatial scalar density of weight one under the canonical metric momentum constraint.",
            "Exact arbitrary-first-jet 3D D-H kinetic-sector identity; the spatial-curvature potential and H-H bracket remain separate.",
            _canonical_metric_dewitt_kinetic_control,
        ),
        _run_check(
            "einstein_aether_inhomogeneous_2d_noether",
            "Every standard Einstein-Aether kinetic invariant and its unit constraint obeys the off-shell diffeomorphism identity for arbitrary inhomogeneous two-dimensional jets.",
            "Exact arbitrary-jet 2D control through third derivatives; not the outstanding arbitrary-background 4D tensor identity.",
            lambda: _einstein_aether_2d_noether_artifact(root),
        ),
        _run_check(
            "einstein_aether_inhomogeneous_4d_numeric_noether",
            "Every standard Einstein-Aether kinetic invariant and its unit constraint passes the off-shell diffeomorphism identity on unrestricted four-dimensional coordinate jets.",
            "Source-bound floating-point 4D falsification on three general Lorentzian jets; not an exact symbolic proof.",
            lambda: _einstein_aether_4d_numeric_noether_artifact(root),
        ),
        _run_check(
            "dhost_degenerate_kinetic_block",
            "A known degenerate scalar-tensor ADM kinetic block generates a primary and secondary second-class pair, removes the extra scalar mode, and has a positive reduced control Hamiltonian.",
            "Exact finite-point quadratic-DHOST ADM scalar Dirac chain with a nondegenerate extra-mode control; tensor/momentum constraints, spatial derivatives, arbitrary coefficient functions, and the full covariant DHOST classification remain separate.",
            _dhost_degeneracy_control,
        ),
        _run_check(
            "generic_horndeski_l2_l4_unitary_adm_primary_degeneracy",
            "Every smooth Horndeski L2-L4 function family cancels the scalar normal-Hessian velocity and has exactly one ADM primary null direction on regular unitary-gauge patches.",
            "Exact seven-velocity Hessian theorem with arbitrary local G4 and G4_X values, the complete six-component DeWitt metric block, the G4-2 X G4_X singular stratum, and a wrong-completion rank-seven negative control. Secondary preservation, distributed Poisson closure, degree count, and Hamiltonian boundedness remain unresolved for arbitrary functions.",
            _generic_horndeski_l2_l4_adm_control,
        ),
        _run_check(
            "generic_horndeski_l2_l4_unitary_distributed_dirac_closure",
            "Every smooth Horndeski L2-L4 family has the complete unitary-gauge secondary chain, spatial-diffeomorphism covariance, second-class lapse pair, and three-mode count on regular lapse-Hessian patches.",
            "Exact conditional arbitrary-function theorem: p_N preservation generates C_N, D-D and D-C close, and invertible Delta_N fixes the lapse multiplier. The G2=X/G4=constant family proves the regular set is nonempty; Delta_N=0 is rejected. Global operator invertibility, boundary zero modes, singular strata, and Hamiltonian boundedness remain unresolved.",
            generic_horndeski_l2_l4_unitary_dirac_control,
        ),
        _run_check(
            "generic_horndeski_l2_l4_flrw_tensor_stability",
            "Every smooth Horndeski L2-L4 family has two healthy tensor modes, a hyperbolic tensor cone, and a positive reduced tensor Hamiltonian wherever G4>0 and G4-2 X G4_X>0 on a homogeneous timelike-gradient background.",
            "Exact source-bound arbitrary-function tensor quadratic action, two-polarization principal polynomial, Legendre transform, and five negative controls. Scalar perturbations, arbitrary inhomogeneous-background strong hyperbolicity, and nonlinear global energy boundedness remain unresolved.",
            generic_horndeski_l2_l4_tensor_stability_control,
        ),
        _run_check(
            "generic_horndeski_l2_l4_flrw_scalar_reduction",
            "Every smooth Horndeski L2-L4 family has one constraint-reduced FLRW scalar mode with a hyperbolic principal cone and bounded quadratic Hamiltonian wherever Theta is nonzero and G_S and F_S are positive.",
            "Exact source-bound arbitrary-function lapse/shift elimination, integration by parts, scalar principal polynomial, Legendre transform, and five pathology controls. Each candidate must supply an on-shell background and prove the coefficient signs over its declared domain; arbitrary inhomogeneous strong hyperbolicity and nonlinear global energy remain unresolved.",
            generic_horndeski_l2_l4_flrw_scalar_reduction_control,
        ),
        _run_check(
            "generic_horndeski_l2_l4_flrw_interval_background",
            "The compiled Horndeski FLRW evolution matrix can generate an outward-rounded interval trajectory certificate that preserves the energy constraint and uniformly excludes all tensor/scalar health boundaries.",
            "Reexecuted 40-step canonical massless-scalar stiff-FLRW known answer with analytic endpoint containment and off-constraint, singular-matrix, and tensor-ghost negatives. This certifies the declared homogeneous trajectory only, not arbitrary generated candidates or inhomogeneous/global stability.",
            lambda: _generic_horndeski_flrw_interval_background_control(root),
        ),
        _run_check(
            "generic_kessence_timelike_principal_hamiltonian",
            "The generalized-harmonic-eligible Einstein-plus-k-essence Horndeski subclass has an exact scalar effective metric, hyperbolic cone, and positive reduced Hamiltonian wherever G2_X and G2_X+2 X G2_XX are positive.",
            "Exact arbitrary-G2 aligned timelike-gradient principal polynomial, effective-metric determinant, Legendre transform, and four pathology controls, source-bound to the Papallo generic weak-field generalized-harmonic subclass theorem. Common-time and uniform arbitrary-background bounds remain candidate-specific.",
            generic_kessence_timelike_principal_hamiltonian_control,
        ),
        _run_check(
            "generic_kessence_nonlinear_adm_legendre",
            "Every smooth k-essence G2 has the exact nonlinear pointwise ADM scalar Legendre map p=G2_X v_n, with a locally convex Hamiltonian branch wherever G2_X+v_n^2 G2_XX is positive.",
            "Exact arbitrary-G2 jet theorem for the nonlinear momentum, Legendre Jacobian, Hamiltonian density, and inverse Hessian, with canonical, wrong-sign, nonconvex, and singular controls. Candidate energy nonnegativity remains an explicit inequality; gravitational boundary charges and a global positive-energy theorem remain unresolved.",
            generic_kessence_nonlinear_adm_legendre_control,
        ),
        _run_check(
            "generic_cubic_horndeski_bssn_hyperbolicity",
            "Cubic Horndeski theories have a dedicated strongly hyperbolic BSSN formulation in the weak-field regime for m>1/4, suitable sigma>1/2, and a uniformly separated scalar/slicing cone.",
            "Source-bound principal-speed and weak-field contract with a healthy m=sigma=1 witness, the full G2/G3 derivative ledger, and momentum-boundary, harmonic-slicing crossing, and scalar-cone crossing negatives. The source supplies no universal numerical threshold for 'much less than one', so each candidate still requires uniform weak-field and cone bounds on its declared domain.",
            generic_cubic_horndeski_bssn_hyperbolicity_control,
        ),
        _run_check(
            "generic_cubic_horndeski_scalar_effective_metric",
            "The cubic-Horndeski scalar effective metric includes the exact G3_X-squared correction generated by substituting the trace-reversed metric equation into the scalar equation.",
            "Exact arbitrary-gradient contraction of the source P_gphi tensor, trace reversal, closed-form effective-metric correction, symmetry identities, and an omitted-trace-reversal witness with residual 27/4. Candidate domain signs remain a separate interval proof.",
            generic_cubic_scalar_effective_metric_control,
        ),
        _run_check(
            "cubic_horndeski_bssn_uniform_local_jet_domains",
            "Every one of the five FLRW-screened cubic-Horndeski candidates has a nonzero arbitrary-local-jet box with a common time covector, positive scalar spatial block, real distinct scalar characteristics, and a slicing cone uniformly separated over the complete direction sphere.",
            "Reexecuted nested interval/Gershgorin campaign anchored to each trajectory hull, binding the source normalization and complete P'_phi_phi metric. Sigma=1/2 and a gradient box crossing X=0 reject. The certificate is pointwise over the declared box; nonlinear PDE invariance of that box and global positive energy remain unresolved.",
            lambda: _cubic_bssn_uniform_domain_campaign_control(root),
        ),
        _run_check(
            "quartic_horndeski_covariant_adm_degeneracy",
            "The named covariant quartic-Horndeski G4(X) action cancels its scalar normal-Hessian velocity and produces the required ADM primary null direction.",
            "Exact unitary-gauge local kinetic reduction for G4=M2/2+alpha X, including the Gauss-Codazzi boundary term and a wrong-completion nondegenerate control; distributed constraints, Euler variation, and physical stability remain separate.",
            lambda: _quartic_horndeski_covariant_adm_control(root),
        ),
        _run_check(
            "quartic_horndeski_unitary_flrw_dirac_chain",
            "The named quartic-Horndeski action produces the expected primary/secondary lapse pair when its scalar is used as the clock.",
            "Exact action-hash-bound curved-FLRW unitary-gauge Dirac chain and quotient-surface rank on a declared regular patch; it is not the inhomogeneous distributed constraint algebra.",
            lambda: _quartic_horndeski_unitary_flrw_dirac_control(root),
        ),
        _run_check(
            "quartic_horndeski_unitary_distributed_dirac_closure",
            "The named quartic-Horndeski action has the exact unitary-gauge spatial-diffeomorphism algebra and a three-mode Dirac count on regular lapse-Hessian operator patches.",
            "Action-hash-bound 3D metric+lapse cotangent lift, secondary-density covariance, regular second-class lapse-pair theorem, and nonempty curved-FLRW witness; global operator invertibility and boundary zero modes remain separate.",
            lambda: _quartic_horndeski_unitary_distributed_dirac_control(root),
        ),
        _run_check(
            "quartic_horndeski_scalar_covariant_variation",
            "The action-hash-bound linear-X quartic-Horndeski scalar variation reduces to a second-order Einstein-tensor Hessian equation.",
            "Cadabra fixed-metric scalar variation plus exact Hessian-commutator and contracted-Bianchi reduction; metric variation and the combined Noether identity remain separate.",
            lambda: _quartic_horndeski_scalar_variation_control(root, cadabra),
        ),
        _run_check(
            "quartic_horndeski_metric_variation_and_noether",
            "The named quartic-Horndeski metric Euler tensor and combined metric-scalar Noether identity are derived on the same action hash.",
            "Full Palatini raw variation with an explicit twice-integrated adjoint, exact John boundary equivalence, arbitrary-background Euler Noether coefficient and action-density covariance, plus nonlinear lapse-FLRW corroboration.",
            lambda: _quartic_horndeski_metric_noether_control(root, cadabra),
        ),
        _run_check(
            "quartic_horndeski_timelike_flat_principal_symbol",
            "The named canonical-plus-quartic Horndeski control has the exact two-tensor plus one-scalar principal block on a flat constant timelike scalar-gradient background.",
            "Action-hash-bound reduced quadratic principal symbol with explicit tensor ghost, tensor-gradient, and omitted-canonical-scalar negative controls; arbitrary-background strong hyperbolicity remains separate.",
            lambda: _quartic_horndeski_timelike_flat_principal_control(root),
        ),
        _run_check(
            "quartic_horndeski_arbitrary_curvature_scalar_principal",
            "The named canonical-plus-quartic Horndeski control has the exact fixed-metric scalar effective cone on an arbitrary local curvature jet.",
            "Action-hash-bound covariant scalar principal tensor P^{mu nu}=g^{mu nu}-2 alpha G^{mu nu}, including diagonal and time-space-flux Lorentzian witnesses, cone-collapse/gradient/kinetic negatives, and metric-cone comparison; the full coupled metric-scalar symbol remains unresolved.",
            lambda: _quartic_horndeski_arbitrary_curvature_scalar_principal_control(
                root
            ),
        ),
        _run_check(
            "quartic_horndeski_coupled_formulation_hyperbolicity",
            "The coupled quartic-Horndeski formulation is classified correctly: generalized harmonic gauge fails generically, while modified harmonic gauge is strongly hyperbolic only under a weak-coupling and auxiliary-cone-separation contract.",
            "Action-hash-bound implementation of the Papallo generalized-harmonic obstruction and Kovacs--Reall modified-harmonic weak-coupling theorem, with an exact nonempty auxiliary-cone witness; the action-specific 11-by-11 correction norm and uniform background bound remain unresolved.",
            lambda: _quartic_horndeski_coupled_formulation_hyperbolicity_control(
                root
            ),
        ),
        _run_check(
            "quartic_horndeski_full_local_principal_extraction",
            "The complete local 11-by-11 quartic-Horndeski metric-scalar principal matrix and its exact 22-by-22 generalized first-order pencil are extracted in modified harmonic gauge and reproduce the independent ADM, scalar-cone, and Einstein-scalar mode controls.",
            "Action-hash-bound local orthonormal-frame extraction with independent scalar-gradient, Hessian, Einstein-tensor, and covector jets. The time block has an exact conditional Frobenius radius, but the declared domain lacks the needed uniform jet bounds; a positive symmetrizer and direction/background induced-norm bound also remain unresolved.",
            lambda: _quartic_horndeski_full_local_principal_control(root),
        ),
        _run_check(
            "quartic_horndeski_baseline_riesz_symmetrizer",
            "The modified-harmonic Einstein-scalar baseline has an exact positive 22-by-22 six-group Riesz symmetrizer, and the physical Horndeski groups retain a positive H_star form under an explicit nonzero matrix-norm perturbation contract.",
            "Exact projectors for speeds +/-1, +/-1/2, and +/-1/3; source-form H_star^+/- on the two three-dimensional physical groups; exact positive LDL pivots and K M=M^T K; resolvent/Neumann projector drift bound with three negative controls. Candidate local-jet companion/H_star bounds and the hat-group restricted-block invertibility proof remain unresolved.",
            quartic_horndeski_baseline_riesz_symmetrizer_control,
        ),
        _run_check(
            "quartic_linear_x_candidate_symbol_bindings",
            "All 12 G3-free, linear-X G4 mutations in the fixed-coefficient campaign are bound to the exact local 11-by-11 quartic principal symbol, including the eight quadratic-kessence extensions.",
            "Hash-replayed source-IR campaign with a symbolic arbitrary-covector G2_XX scalar-block extension and a phi-dependent-G4 fail-closed control. Exact extraction does not yet supply the uniform local-jet symmetrizer and induced-norm bound required for a completed strong-hyperbolicity theorem.",
            lambda: _quartic_linear_x_symbol_campaign_control(root),
        ),
        _run_check(
            "quartic_linear_x_uniform_symmetrizer_domains",
            "All 12 fixed-coefficient G3-free, linear-X quartic candidates possess a nonzero arbitrary-local-jet box on which the complete modified-harmonic 22-by-22 system is strongly hyperbolic for every spatial direction.",
            "Hash-replayed matrix-norm campaign around the exact Minkowski/constant-scalar solution. It combines the six Riesz-group projector bound, positive physical H_star forms, a Neumann-invertible time block, exact diffeomorphism kernel, rank-seven hat quotient, and fixed separated auxiliary cones. A larger box rejects. Nonlinear evolution-invariance of the box and global energy remain unresolved.",
            lambda: _quartic_symmetrizer_domain_campaign_control(root),
        ),
        _run_check(
            "quartic_linear_x_local_on_shell_adm_dirac_hamiltonians",
            "All 12 fixed-coefficient linear-X quartic candidates possess an on-shell expanding FLRW branch inside the certified strong-hyperbolicity box where the ADM Hessian, regular Dirac chain, three-mode count, and reduced quadratic Hamiltonian pass for every finite future time.",
            "Hash-replayed action-to-symbol-to-symmetrizer-to-Dirac campaign. Each candidate has exact zero FLRW equation residuals, an invertible acceleration system, rank-six seven-velocity Hessian with p_V_star=0, strictly nonzero lapse pairing, six first-class plus two second-class constraints, and positive G_T/F_T/G_S/F_S. Exact interval sign bounds prove A_star^2 decreases and all local jets remain inside the box on 0<A_star^2<=1e-20; zero is approached only asymptotically. Zero clock gradient, an out-of-box witness, a tensor ghost, and a k-essence ghost reject. Inhomogeneous PDE trapping and nonlinear global energy remain unresolved.",
            lambda: _quartic_dirac_hamiltonian_campaign_control(root),
        ),
        _run_check(
            "quartic_linear_x_finite_horizon_inhomogeneous_physical_energies",
            "All 12 fixed-coefficient linear-X quartic candidates possess a coercive all-wavenumber linearized physical-mode Sobolev energy with an explicit finite-horizon amplification bound on a compact segment of the exact expanding FLRW branch.",
            "Hash-replayed KYY tensor/scalar quadratic Hamiltonians with exact rational coefficient bounds, Hamiltonian-mode cancellation, a Gronwall amplification factor, an explicit three-torus Sobolev C1 majorant, and positive initial-energy radii. This is deliberately limited to the three reduced linear physical modes; lapse/shift/constraint reconstruction, full physical-space nonlinear trapping, nonlinear boundary energy, and observations remain fail-closed.",
            lambda: _quartic_linearized_energy_campaign_control(root),
        ),
        _run_check(
            "quartic_linear_x_constraint_and_gauge_reconstruction",
            "All 12 fixed-coefficient linear-X quartic candidates possess exact bounded linear lapse and physical longitudinal-shift reconstruction operators chained to positive Sobolev-energy radii on the compact FLRW segment.",
            "Source-bound KYY lapse/shift constraint elimination with an exact beta-k closed form, explicit Theta and infrared singular controls, harmless treatment of the zero-mode shift-potential kernel, candidate-specific operator bounds, and tightened initial-energy radii. Spatial C1 auxiliaries pass; their time derivatives, nonlinear constraint products, full physical-space jet trapping, and boundary energy remain unresolved.",
            lambda: _quartic_constraint_reconstruction_campaign_control(root),
        ),
        _run_check(
            "quartic_linear_x_auxiliary_time_reconstruction",
            "All 12 fixed-coefficient linear-X quartic candidates possess bounded linear lapse and physical longitudinal-shift reconstruction through one time derivative, chained to explicit positive Sobolev-energy radii.",
            "Exact time differentiation of the KYY auxiliary solutions followed by elimination of ddot(zeta) with the reduced scalar equation; candidate-specific R/S/G_S drift, damping, sound-cone, C2 Sobolev, and infrared bounds. An omitted-acceleration equation and insufficient Sobolev order reject. Nonlinear products, modified-harmonic gauge sectors, quasilinear commutators, full PDE trapping, and boundary energy remain unresolved.",
            lambda: _quartic_auxiliary_time_campaign_control(root),
        ),
        _run_check(
            "quartic_linear_x_quasilinear_coefficient_moser_envelopes",
            "All 12 fixed-coefficient linear-X quartic candidates possess explicit uniform C4 derivative envelopes for the action-derived 22-variable quasilinear companion coefficient on their certified strong-hyperbolicity boxes.",
            "The exact A/B/C blocks are quadratic in 24 covariant jet components, so their third and fourth raw derivatives vanish. Candidate symmetrizer bounds give a verified rational ceiling on the inverse time block, and the differentiated A F=X identity propagates exact companion bounds through Sobolev order four. False degree-one and H3 declarations reject. This is coefficient-composition readiness only: the nonlinear state-to-jet map, source and symmetrizer derivatives, gauge reconstruction, commuted energy closure, and PDE bootstrap remain unresolved.",
            lambda: _quartic_quasilinear_moser_campaign_control(root),
        ),
        _run_check(
            "quartic_linear_x_physical_space_first_order_reduction",
            "All 12 fixed-coefficient linear-X quartic candidates are bound to an exact 55-variable three-dimensional first-order principal reduction whose nonzero characteristic modes reproduce the proven 22-by-22 directional companion pencil.",
            "The reduction introduces 11 fields, 11 time derivatives, and 33 spatial derivatives. Exact extraction of B_i and symmetric C_ij reconstructs the second-order symbol; the characteristic lift has zero residual; and 33 derivative-definition plus 33 independent curl constraints propagate. Omitting one spatial-derivative evolution equation rejects. This corrects the state dimension but does not yet supply nonlinear lower-order sources, connection terms, gauge drivers, a 55-variable symmetrizer, state-to-jet Sobolev bounds, commuted energy closure, or PDE bootstrap.",
            lambda: _quartic_first_order_reduction_campaign_control(root),
        ),
        _run_check(
            "quartic_horndeski_timelike_flat_physical_hamiltonian",
            "The named quartic-Horndeski action has a positive three-mode reduced quadratic Hamiltonian on its healthy flat constant-timelike-gradient patch.",
            "Action/Dirac/principal-count-matched exact Legendre transform with positive tensor/scalar momentum and coordinate Hessians plus ghost and gradient negative controls; nonlinear global energy remains unresolved.",
            lambda: _quartic_horndeski_timelike_flat_hamiltonian_control(root),
        ),
        _run_check(
            "quartic_horndeski_global_timelike_gradient_no_go",
            "A nonzero linear-X quartic-Horndeski coupling cannot keep both tensor kinetic and tensor gradient coefficients positive for arbitrarily large timelike scalar gradients.",
            "Exact all-amplitude sign split: alpha>0 reaches kinetic rank loss/ghost, alpha<0 reaches cone collapse/gradient instability; an explicit bounded EFT/background domain is required.",
            lambda: _quartic_horndeski_global_timelike_gradient_no_go_control(root),
        ),
        _run_check(
            "quartic_horndeski_flrw_background_domain_crossing",
            "The named quartic-Horndeski action does not preserve its bounded healthy tensor domain under unrestricted nonlinear FLRW evolution.",
            "Exact action-hash-bound contracting closed-FLRW boundary-crossing witness with a regular homogeneous acceleration system; restricted solution classes, EFT stopping boundaries, and nonlinear G4 completions remain separate.",
            lambda: _quartic_horndeski_flrw_domain_crossing_control(root),
        ),
        _run_check(
            "principal_symbol_controls",
            "The reduced principal-symbol analyzer accepts scalar/Proca metric cones and rejects ghost, gradient, and superluminal controls.",
            "Exact reduced isotropic quadratic systems on a frozen local background.",
            _principal_symbol_control,
        ),
        _run_check(
            "anisotropic_principal_symbol_directions",
            "Direction-dependent kinetic/gradient symbols are checked on exact rational directions, including cross-gradient terms.",
            "Finite declared-direction anisotropic falsification; not uniform strong hyperbolicity over the complete direction sphere.",
            _anisotropic_principal_symbol_control,
        ),
        _run_check(
            "reduced_lagrangian_principal_extraction",
            "Kinetic and anisotropic spatial-gradient principal blocks are differentiated directly from a reduced quadratic Lagrangian.",
            "Automatic reduced K/G^{ij}/B^i extraction with mixed omega-k matrix-polynomial characteristics; gauge reduction and arbitrary action/background export remain separate.",
            _extracted_principal_symbol_control,
        ),
        _run_check(
            "uniform_scalar_anisotropy_sphere",
            "Scalar-sector anisotropic speed and stability bounds hold uniformly over the complete spatial direction sphere.",
            "Exact Rayleigh-quotient eigenvalue proof for one reduced scalar mode without time-space mixed terms; multi-field uniformity remains separate.",
            _uniform_scalar_anisotropy_control,
        ),
        _run_check(
            "uniform_multifield_block_certificate",
            "A sufficient spatial-field block certificate proves multi-field stability and cone bounds uniformly over every spatial direction.",
            "Exact sufficient certificate for symmetric reduced systems without time-space mixed terms; an inconclusive block test is unresolved because block positivity is stronger than rank-one positivity.",
            _uniform_multifield_block_control,
        ),
        _run_check(
            "curved_background_principal_controls",
            "Canonical scalar, reduced Proca, and GR TT sectors remain metric-cone hyperbolic on FLRW and static spherical backgrounds.",
            "Exact principal two-derivative sectors after reduction; not arbitrary nonminimal candidate extraction.",
            _curved_background_principal_control,
        ),
    ]
    if cadabra["available"]:
        checks.append(
            _run_check(
                "einstein_aether_arbitrary_background_4d_noether",
                "The complete standard Einstein-Aether action obeys its off-shell metric-vector-multiplier Noether identity on an arbitrary four-dimensional background.",
                "Exact abstract-tensor proof in the fixed-covector convention, source-bound to the metric/vector/multiplier Euler variations; arbitrary independent c1..c4 coefficients establish K1..K4 termwise, with corrupted-sign and omitted-connection negative controls.",
                lambda: run_cadabra_aether_noether_control(root, cadabra),
            )
        )
        checks.append(
            _run_check(
                "cadabra_metric_contraction",
                "Cadabra 2 executes a tensor script and eliminates an inverse-metric contraction.",
                "Exact tensor-algebra backend smoke control; not action variation.",
                lambda: run_cadabra_metric_control(root, cadabra),
            )
        )
        checks.append(
            _run_check(
                "cadabra_proca_variation",
                "Cadabra substitutes the field strength and varies the Proca action with respect to A_mu.",
                "Exact vector-field variation in flat derivative notation; metric variation is not included.",
                lambda: run_cadabra_script(
                    root,
                    cadabra,
                    "formal/cadabra/proca_variation.cdb",
                    ["SIGMA_PROCA_VARIATION_FINAL", "δ{A", "\\partial", "m^{2}"],
                ),
            )
        )
        checks.append(
            _run_check(
                "cadabra_einstein_aether_vector_variation",
                "Cadabra varies the complete K1..K4 Einstein-Aether vector sector and its unit-norm multiplier.",
                "Exact vector and multiplier variations; metric variation and nonlinear Hamiltonian analysis remain separate.",
                lambda: run_cadabra_script(
                    root,
                    cadabra,
                    "formal/cadabra/einstein_aether_vector_variation.cdb",
                    [
                        "SIGMA_AETHER_VECTOR_VARIATION_FINAL",
                        "SIGMA_AETHER_NORM_CONSTRAINT_FINAL",
                        "δ{u",
                        "δ{λ}",
                        "u_{a}u^{a}+1",
                    ],
                ),
            )
        )
        checks.append(
            _run_check(
                "cadabra_einstein_aether_metric_variation",
                "Cadabra varies the complete K1..K4 and unit-constraint action with respect to the inverse metric, including delta Gamma.",
                "Exact nonlinear abstract metric variation holding u_a fixed; total divergences are discarded after integration by parts.",
                lambda: run_cadabra_script(
                    root,
                    cadabra,
                    "formal/cadabra/einstein_aether_metric_variation.cdb",
                    [
                        "SIGMA_AETHER_METRIC_VARIATION_NO_H_DERIVATIVES",
                        "SIGMA_AETHER_METRIC_VARIATION_CONNECTION_INCLUDED",
                        "SIGMA_AETHER_METRIC_VARIATION_FINAL",
                        "c1",
                        "c2",
                        "c3",
                        "c4",
                        "u_{a}u_{b}",
                    ],
                ),
            )
        )
        checks.append(
            _run_check(
                "cadabra_generic_g4_metric_raw_variation",
                "Cadabra independently varies the complete generic quartic-Horndeski G4(phi,X) action with respect to the inverse metric, including G4/G4_X chain rules and the scalar-Hessian connection variation.",
                "Exact determinant, inverse-metric, G4, G4_X, Hessian-connection, and twice-integrated Palatini contributions with every derivative removed from h^ab. An exact rank-one polarization certificate cancels the complete symmetric third scalar jet, leaving a second-order metric Euler coefficient; termwise normalization against the published B.4 tensor remains separate.",
                lambda: run_cadabra_script(
                    root,
                    cadabra,
                    "formal/cadabra/generic_g4_metric_variation.cdb",
                    [
                        "SIGMA_GENERIC_G4_PALATINI_ADJOINT_INCLUDED",
                        "SIGMA_GENERIC_G4_METRIC_VARIATION_NO_H_DERIVATIVES",
                        "SIGMA_GENERIC_G4_METRIC_VARIATION_CONNECTION_INCLUDED",
                        "SIGMA_GENERIC_G4_METRIC_VARIATION_POLARIZATION_CERTIFIED",
                        "SIGMA_GENERIC_G4_METRIC_VARIATION_OMITTED_PALATINI_REJECTED",
                        "SIGMA_GENERIC_G4_METRIC_VARIATION_THIRD_DERIVATIVES_CANCELLED",
                        "SIGMA_GENERIC_G4_METRIC_VARIATION_FINAL",
                    ],
                ),
            )
        )
        checks.append(
            _run_check(
                "cadabra_einstein_hilbert_metric_variation",
                "Cadabra applies the determinant and Palatini rules to Einstein-Hilbert and isolates the Einstein tensor bulk coefficient.",
                "Exact nonlinear metric variation with the total divergence explicitly tracked; assumes compact support or the matching boundary completion.",
                lambda: run_cadabra_script(
                    root,
                    cadabra,
                    "formal/cadabra/einstein_hilbert_metric_variation.cdb",
                    [
                        "SIGMA_EH_BOUNDARY_TRACKED",
                        "SIGMA_EH_METRIC_VARIATION_FINAL",
                        "R^{ab}h_{ab}sqrtg",
                        "R g^{ab}h_{ab}sqrtg",
                    ],
                ),
            )
        )
        checks.append(
            _run_check(
                "cadabra_adm_spatial_curvature_variation",
                "Cadabra retains the lapse derivatives in the spatial Einstein-Hilbert variation and isolates the curvature Hamiltonian Euler coefficient.",
                "Exact fully covariant variation of integral N sqrt(q) R^(3), with the first divergence step explicit and the second integration by parts executed by Cadabra.",
                lambda: run_cadabra_adm_curvature_variation_control(root, cadabra),
            )
        )
        checks.append(
            _run_check(
                "cadabra_nonlinear_contracted_bianchi",
                "Cadabra contracts the exact differential Riemann Bianchi identity and obtains zero.",
                "Exact nonlinear abstract-tensor identity for a Levi-Civita connection.",
                lambda: run_cadabra_script(
                    root,
                    cadabra,
                    "formal/cadabra/contracted_bianchi.cdb",
                    ["SIGMA_NONLINEAR_CONTRACTED_BIANCHI_ZERO"],
                ),
            )
        )
        checks.append(
            _run_check(
                "cadabra_canonical_scalar_metric_variation",
                "Cadabra derives the canonical scalar metric Euler coefficient from explicit inverse-metric and determinant dependence.",
                "Exact nonlinear matter metric variation; p_a denotes nabla_a phi with its index down.",
                lambda: run_cadabra_script(
                    root,
                    cadabra,
                    "formal/cadabra/canonical_scalar_metric_variation.cdb",
                    [
                        "SIGMA_SCALAR_METRIC_VARIATION_FINAL",
                        "h^{ab}p_{a}p_{b}sqrtg",
                        "g^{cd}h_{cd}mphi^2phi^{2}sqrtg",
                    ],
                ),
            )
        )
        checks.append(
            _run_check(
                "cadabra_proca_metric_variation",
                "Cadabra derives the Proca metric Euler coefficient from explicit F^2, A^2, and determinant dependence.",
                "Exact nonlinear matter metric variation; F_ab is connection independent by antisymmetry.",
                lambda: run_cadabra_script(
                    root,
                    cadabra,
                    "formal/cadabra/proca_metric_variation.cdb",
                    [
                        "SIGMA_PROCA_METRIC_VARIATION_FINAL",
                        "F_{ac}F_{bd}",
                        "A_{a}A_{b}h^{ab}mA^{2}sqrtg",
                    ],
                ),
            )
        )
        checks.append(
            _run_check(
                "cadabra_canonical_scalar_variation",
                "Cadabra varies the canonical scalar action, integrates by parts, and factors out the field variation.",
                "Exact covariant scalar control in flat derivative notation; metric variation is not included.",
                lambda: run_cadabra_scalar_variation_control(root, cadabra),
            )
        )
    passed = sum(item.status == "pass" for item in checks)
    return {
        "schema_version": "sigma-formal-controls-1.0",
        "created_utc": datetime.now(UTC).isoformat(),
        "field_contract": str(contract_path),
        "backends": {
            "sympy": {
                "available": True,
                "version": sp.__version__,
                "role": "known-answer algebra and reduced Hamiltonian controls",
            },
            "cadabra2": {**cadabra, "role": "covariant tensor variation adapter target"},
        },
        "counts": {"total": len(checks), "passed": passed, "failed": len(checks) - passed},
        "checks": [item.as_dict() for item in checks],
        "candidate_readiness": {
            "arbitrary_covariant_variation": (
                "family adapters implemented for the declared EH, canonical-scalar, Proca, "
                "complete K1..K4 Einstein-Aether, arbitrary G2/G3, arbitrary G4 fixed-metric "
                "scalar, and X-independent G4 metric controls; generic nonlinear-G4_X now has "
                "an independent Cadabra raw metric variation and an exact source-form all-jet "
                "identity, but their third-derivative commutator reduction/equivalence remains "
                "unresolved; arbitrary future generated actions remain unsupported"
            ),
            "nonlinear_bianchi_noether_verification": (
                "exact family-scoped identities implemented for EH, scalar, Proca, "
                "Einstein-Aether, arbitrary G2/G3, and X-independent G4; nonlinear-G4_X and "
                "arbitrary future generated-action identity construction remain unsupported"
            ),
            "full_adm_dirac_constraint_closure": (
                "implemented for pure GR and regular positive-unit-branch generic "
                "Einstein-Aether; exact Proca Fourier and finite-point DHOST chains are "
                "known-answer controls, not full coupled arbitrary-action closure"
            ),
            "background_principal_symbols": (
                "implemented for declared reduced standard sectors and a sufficient arbitrary-"
                "smooth-background Einstein-Aether formulation; arbitrary generated-action "
                "gauge reduction and symbol extraction remain unsupported"
            ),
            "observational_gates_unsealed": False,
        },
        "interpretation": "Passing controls validates only the stated algebraic scopes. It is not evidence that a generated gravity candidate is healthy or empirically correct.",
    }


def write_formal_report(report: dict[str, Any], output_directory: str | Path) -> tuple[Path, Path]:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "formal-controls.json"
    markdown_path = output / "formal-controls.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Sigma formal-backend controls",
        "",
        f"- Passed: {report['counts']['passed']} / {report['counts']['total']}",
        f"- Cadabra 2 available: {report['backends']['cadabra2']['available']}",
        "",
        "| Control | Status | Verified scope |",
        "|---|---:|---|",
    ]
    for item in report["checks"]:
        lines.append(f"| `{item['name']}` | {item['status']} | {item['scope']} |")
    lines.extend(
        [
            "",
            "## Candidate readiness",
            "",
            (
                "The known-answer families now cover covariant variation and identities, pure-GR "
                "and regular-patch generic Einstein-Aether constraint closure, exact Proca and "
                "reduced DHOST Dirac chains, and declared principal-symbol controls. Arbitrary "
                "generated actions without a family adapter remain fail-closed. Observational "
                "gates stay sealed."
            ),
            "",
        ]
    )
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, markdown_path
