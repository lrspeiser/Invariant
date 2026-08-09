from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import sympy as sp

from .legendre_ir import build_local_kinetic_model
from .principal_ir import build_physical_principal_blocks

SCHEMA_VERSION = "sigma-physical-hamiltonian-ir-1.0"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _reject(action_ir: dict[str, Any], message: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "reject",
        "promotion_allowed": False,
        "input_action_sha256": action_ir.get("content_sha256"),
        "errors": [message],
    }


def _unresolved(action_ir: dict[str, Any], message: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "unresolved",
        "promotion_allowed": False,
        "input_action_sha256": action_ir.get("content_sha256"),
        "errors": [message],
    }


def _mass_symbol(action_ir: dict[str, Any], name: str) -> sp.Symbol:
    if name not in action_ir["canonical"]["universal_constants"]:
        raise ValueError(f"{name} must be declared for the physical Hamiltonian adapter")
    return sp.Symbol(name, real=True)


def _hamiltonian_blocks(
    action_ir: dict[str, Any], family: str, wave_number: sp.Symbol
) -> tuple[list[str], sp.Matrix, sp.Matrix, dict[str, Any]]:
    basis, principal_kinetic, gradient, _ = build_physical_principal_blocks(
        action_ir, family
    )
    coefficients = build_local_kinetic_model(action_ir)["coefficients"]
    coordinate = sp.Matrix(sp.simplify(wave_number**2 * gradient))
    momentum = sp.Matrix(sp.simplify(principal_kinetic.inv()))
    detail: dict[str, Any] = {
        "construction": "H2=(p^T K^-1 p+q^T[k^2 G+M^2]q)/2",
        "mass_or_constraint_reduction": "none",
    }
    if family == "canonical_scalar_gravity":
        mass = _mass_symbol(action_ir, "m_phi")
        scalar_mass = coefficients.get("SCALAR_MASS", sp.Integer(0))
        coordinate[2, 2] += sp.factor(-2 * scalar_mass * mass**2)
        detail["mass_or_constraint_reduction"] = (
            "scalar potential Hessian -2*C_SCALAR_MASS*m_phi^2 in the physical scalar entry"
        )
    elif family == "proca_gravity":
        mass = _mass_symbol(action_ir, "m_A")
        gauge = sp.factor(-4 * coefficients.get("PROCA_F2", 0))
        mass_amplitude = sp.factor(-2 * coefficients.get("PROCA_MASS", 0))
        coordinate = sp.diag(
            *(list(coordinate.diagonal()[:2]) + [
                gauge * wave_number**2 + mass_amplitude * mass**2,
                gauge * wave_number**2 + mass_amplitude * mass**2,
                mass_amplitude * mass**2,
            ])
        )
        momentum = sp.diag(
            *(list(momentum.diagonal()[:2]) + [
                sp.factor(1 / gauge),
                sp.factor(1 / gauge),
                sp.factor(1 / gauge + wave_number**2 / (mass_amplitude * mass**2)),
            ])
        )
        detail.update(
            {
                "construction": (
                    "exact Fourier reduction after solving p_A_perp=0 and "
                    "div(p)+b*m_A^2*A_perp=0"
                ),
                "mass_or_constraint_reduction": (
                    "H_A=p_i^2/(2a)+a F_ij^2/4+b m_A^2 A_i^2/2+"
                    "(div p)^2/(2 b m_A^2), a=-4*C_F2, b=-2*C_mass"
                ),
                "gauge_kinetic_amplitude": str(gauge),
                "mass_energy_amplitude": str(mass_amplitude),
            }
        )
    return basis, sp.simplify(coordinate), sp.simplify(momentum), detail


def compile_physical_hamiltonian_ir(
    action_ir: dict[str, Any],
    dirac_ir: dict[str, Any],
    stability_ir: dict[str, Any],
    principal_ir: dict[str, Any],
) -> dict[str, Any]:
    """Derive the constraint/gauge-reduced quadratic Hamiltonian on one artifact chain."""

    if not action_ir.get("valid"):
        return _reject(action_ir, "covariant action IR is invalid")
    action_hash = action_ir.get("content_sha256")
    if dirac_ir.get("input_action_sha256") != action_hash:
        return _reject(action_ir, "Dirac IR belongs to a different action hash")
    if stability_ir.get("input_action_sha256") != action_hash:
        return _reject(action_ir, "stability IR belongs to a different action hash")
    if stability_ir.get("input_dirac_ir_sha256") != dirac_ir.get("content_sha256"):
        return _reject(action_ir, "stability IR belongs to a different Dirac hash")
    if principal_ir.get("input_action_sha256") != action_hash:
        return _reject(action_ir, "principal IR belongs to a different action hash")
    if principal_ir.get("input_dirac_ir_sha256") != dirac_ir.get("content_sha256"):
        return _reject(action_ir, "principal IR belongs to a different Dirac hash")
    if principal_ir.get("input_stability_ir_sha256") != stability_ir.get("content_sha256"):
        return _reject(action_ir, "principal IR belongs to a different stability hash")

    family = stability_ir.get("family", "unsupported")
    if family == "unsupported":
        return _unresolved(action_ir, "no physical Hamiltonian adapter for this action family")
    wave_number = sp.Symbol("k", positive=True, finite=True)
    try:
        basis, coordinate_block, momentum_block, construction = _hamiltonian_blocks(
            action_ir, family, wave_number
        )
    except (ValueError, ZeroDivisionError) as error:
        return _reject(action_ir, str(error))

    mode_count = len(basis)
    closure = dirac_ir.get("distributed_constraint_closure", {})
    physical_dof = closure.get("constraint_surface_rank", {}).get("physical_dof")
    reduction_pass = (
        dirac_ir.get("status") == "pass"
        and closure.get("status") == "pass"
        and physical_dof == mode_count
        and principal_ir.get("gauge_reduction_certificate", {}).get("status") == "pass"
        and principal_ir.get("gauge_reduction_certificate", {}).get("retained_mode_count")
        == mode_count
        and principal_ir.get("gauge_reduction_certificate", {}).get(
            "retained_physical_basis"
        )
        == basis
    )
    exact_symmetric = (
        coordinate_block == coordinate_block.T
        and momentum_block == momentum_block.T
        and coordinate_block.shape == (mode_count, mode_count)
        and momentum_block.shape == (mode_count, mode_count)
    )
    exact_regular = exact_symmetric and momentum_block.det() != 0

    coordinates = sp.Matrix(sp.symbols(f"Q0:{mode_count}", real=True))
    momenta = sp.Matrix(sp.symbols(f"P0:{mode_count}", real=True))
    velocities = sp.Matrix(sp.symbols(f"V0:{mode_count}", real=True))
    reduced_hamiltonian = sp.factor(
        ((momenta.T * momentum_block * momenta)[0]
        + (coordinates.T * coordinate_block * coordinates)[0])
        / 2
    )
    if exact_regular:
        lagrangian_kinetic = sp.simplify(momentum_block.inv())
        reduced_lagrangian = sp.factor(
            ((velocities.T * lagrangian_kinetic * velocities)[0]
            - (coordinates.T * coordinate_block * coordinates)[0])
            / 2
        )
        canonical_momenta = lagrangian_kinetic * velocities
        momentum_substitutions = dict(zip(momenta, canonical_momenta, strict=True))
        legendre_residual = sp.factor(
            (canonical_momenta.T * velocities)[0]
            - reduced_lagrangian
            - reduced_hamiltonian.subs(momentum_substitutions)
        )
    else:
        lagrangian_kinetic = sp.zeros(mode_count)
        reduced_lagrangian = sp.Integer(0)
        legendre_residual = sp.nan

    condition_certificate = stability_ir.get("condition_certificate", {})
    source_hamiltonian = stability_ir.get("physical_hamiltonian", {})
    domain_pass = condition_certificate.get("pointwise_status") == "pass"
    source_status = source_hamiltonian.get("status", "unresolved")
    quadratic_pass = (
        reduction_pass
        and principal_ir.get("status") == "pass"
        and domain_pass
        and source_status != "reject"
        and exact_regular
        and legendre_residual == 0
    )
    patchwise_quadratic_pass = (
        family == "quartic_horndeski"
        and reduction_pass
        and exact_regular
        and legendre_residual == 0
        and not source_hamiltonian.get("missing_or_failed_controls", [])
        and "quartic_horndeski_timelike_flat_physical_hamiltonian"
        in source_hamiltonian.get("required_controls", [])
    )
    if source_status == "reject" or not exact_symmetric:
        status = "reject"
    elif source_status == "pass" and quadratic_pass:
        status = "pass"
    else:
        status = "unresolved"

    conditions = {
        item["name"]: item["status"]
        for item in condition_certificate.get("conditions", [])
    }
    body = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "promotion_allowed": False,
        "input_action_sha256": action_hash,
        "input_dirac_ir_sha256": dirac_ir.get("content_sha256"),
        "input_stability_ir_sha256": stability_ir.get("content_sha256"),
        "input_principal_ir_sha256": principal_ir.get("content_sha256"),
        "family": family,
        "background": "one nonzero Fourier mode in a frozen local physical frame",
        "wave_number_domain": "k>0",
        "physical_basis": basis,
        "physical_mode_count": mode_count,
        "constraint_surface_physical_dof": physical_dof,
        "gauge_reduction_status": "pass" if reduction_pass else "unresolved",
        "coordinate_hessian": str(coordinate_block),
        "momentum_hessian": str(momentum_block),
        "phase_space_hessian": str(sp.diag(coordinate_block, momentum_block)),
        "coordinate_hessian_determinant": str(sp.factor(coordinate_block.det())),
        "momentum_hessian_determinant": str(sp.factor(momentum_block.det())),
        "reduced_hamiltonian": str(reduced_hamiltonian),
        "inverse_legendre_kinetic_matrix": str(lagrangian_kinetic),
        "reconstructed_reduced_lagrangian": str(reduced_lagrangian),
        "legendre_transform_residual": str(legendre_residual),
        "construction": construction,
        "positivity_certificate": {
            "status": "pass" if quadratic_pass else (
                "reject" if source_status == "reject" else "unresolved"
            ),
            "parameter_condition_statuses": conditions,
            "coordinate_and_momentum_blocks_diagonal": bool(
                coordinate_block.is_diagonal() and momentum_block.is_diagonal()
            ),
            "exact_legendre_transform": legendre_residual == 0,
            "source_reduced_energy_status": source_status,
            "source_required_controls": source_hamiltonian.get("required_controls", []),
            "declared_background_patch_status": (
                "pass" if patchwise_quadratic_pass else "unresolved"
            ),
            "declared_background_patch_scope": (
                "flat constant-timelike-gradient patch satisfying positive tensor kinetic, "
                "positive tensor gradient, positive canonical scalar kinetic, and k>0"
                if family == "quartic_horndeski"
                else "not separately applicable"
            ),
        },
        "generic_nonlinear_total_energy": {
            "status": (
                "unresolved"
                if family in {"einstein_aether", "quartic_horndeski"}
                else "not_claimed"
            ),
            "scope": (
                "generic twisting-Aether/nonmaximal nonlinear total-energy positivity is not established"
                if family == "einstein_aether"
                else (
                    "nonlinear curved-background Horndeski energy, boundary charges, and a global positive-energy theorem are not established"
                    if family == "quartic_horndeski"
                    else "this artifact is a local physical perturbation/free-matter stability certificate, not a new global positive-mass theorem"
                )
            ),
        },
        "proof_scope": (
            "action/Dirac/stability/principal-hash-bound physical quadratic Hamiltonian after "
            "constraint and gauge reduction; exact Proca second-class elimination is included, "
            "while generic nonlinear gravitational total-energy positivity is stated separately"
        ),
    }
    content = _canonical_json(body)
    return {**body, "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest()}


def write_physical_hamiltonian_ir(
    hamiltonian_ir: dict[str, Any], output: str | Path
) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(hamiltonian_ir, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path
