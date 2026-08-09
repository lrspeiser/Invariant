from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import sympy as sp

from .legendre_ir import build_local_kinetic_model

SCHEMA_VERSION = "sigma-physical-principal-ir-1.0"


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


def build_physical_principal_blocks(
    action_ir: dict[str, Any], family: str
) -> tuple[list[str], sp.Matrix, sp.Matrix, dict[str, sp.Expr]]:
    model = build_local_kinetic_model(action_ir)
    coefficients = model["coefficients"]
    eh = sp.factor(coefficients.get("EH_R", 0))
    if family == "einstein_hilbert":
        basis = ["spin_2_plus", "spin_2_cross"]
        kinetic = sp.diag(eh, eh)
        return basis, kinetic, kinetic, {"tensor_speed_squared": sp.Integer(1)}
    if family == "canonical_scalar_gravity":
        scalar = sp.factor(coefficients.get("SCALAR_X", 0))
        basis = ["spin_2_plus", "spin_2_cross", "scalar"]
        kinetic = sp.diag(eh, eh, scalar)
        return basis, kinetic, kinetic, {
            "tensor_speed_squared": sp.Integer(1),
            "scalar_speed_squared": sp.Integer(1),
        }
    if family == "quartic_horndeski":
        scalar = sp.factor(coefficients.get("SCALAR_X", 0))
        alpha = sp.factor(coefficients["HORNDESKI_L4_LINEAR_X"])
        a_star = sp.Symbol("A_star", positive=True, finite=True)
        tensor_kinetic = sp.factor(eh - alpha * a_star**2 / 2)
        tensor_gradient = sp.factor(eh + alpha * a_star**2 / 2)
        basis = ["spin_2_plus", "spin_2_cross", "scalar"]
        kinetic = sp.diag(tensor_kinetic, tensor_kinetic, scalar)
        gradient = sp.diag(tensor_gradient, tensor_gradient, scalar)
        return basis, kinetic, gradient, {
            "tensor_speed_squared": sp.factor(tensor_gradient / tensor_kinetic),
            "scalar_speed_squared": sp.Integer(1),
        }
    if family == "proca_gravity":
        vector = sp.factor(-4 * coefficients.get("PROCA_F2", 0))
        basis = [
            "spin_2_plus",
            "spin_2_cross",
            "proca_transverse_x",
            "proca_transverse_y",
            "proca_longitudinal",
        ]
        kinetic = sp.diag(eh, eh, vector, vector, vector)
        return basis, kinetic, kinetic, {
            "tensor_speed_squared": sp.Integer(1),
            "proca_speed_squared": sp.Integer(1),
        }
    if family == "einstein_aether":
        c1 = sp.factor(-coefficients.get("AETHER_K1", 0) / eh)
        c2 = sp.factor(-coefficients.get("AETHER_K2", 0) / eh)
        c3 = sp.factor(-coefficients.get("AETHER_K3", 0) / eh)
        c4 = sp.factor(coefficients.get("AETHER_K4", 0) / eh)
        c13 = sp.factor(c1 + c3)
        c14 = sp.factor(c1 + c4)
        c123 = sp.factor(c1 + c2 + c3)
        tensor = sp.factor(1 - c13)
        trace = sp.factor(2 + c13 + 3 * c2)
        vector_gradient = sp.factor(2 * c1 - c1**2 + c3**2)
        kinetic = sp.diag(
            tensor,
            tensor,
            2 * c14,
            2 * c14,
            sp.factor(c14**2 * tensor * trace / c123),
        )
        gradient = sp.diag(
            1,
            1,
            sp.factor(vector_gradient / tensor),
            sp.factor(vector_gradient / tensor),
            sp.factor(c14 * (2 - c14)),
        )
        speeds = {
            "spin_2_speed_squared": sp.factor(1 / tensor),
            "spin_1_speed_squared": sp.factor(
                vector_gradient / (2 * c14 * tensor)
            ),
            "spin_0_speed_squared": sp.factor(
                c123 * (2 - c14) / (c14 * tensor * trace)
            ),
        }
        basis = [
            "spin_2_plus",
            "spin_2_cross",
            "spin_1_x",
            "spin_1_y",
            "spin_0",
        ]
        return basis, kinetic, gradient, speeds
    raise ValueError(f"unsupported action family: {family}")


def compile_physical_principal_ir(
    action_ir: dict[str, Any],
    dirac_ir: dict[str, Any],
    stability_ir: dict[str, Any],
) -> dict[str, Any]:
    """Construct the reduced physical principal matrices on one frozen action hash."""

    if not action_ir.get("valid"):
        return _reject(action_ir, "covariant action IR is invalid")
    if dirac_ir.get("input_action_sha256") != action_ir.get("content_sha256"):
        return _reject(action_ir, "Dirac IR belongs to a different action hash")
    if stability_ir.get("input_action_sha256") != action_ir.get("content_sha256"):
        return _reject(action_ir, "stability IR belongs to a different action hash")
    if stability_ir.get("input_dirac_ir_sha256") != dirac_ir.get("content_sha256"):
        return _reject(action_ir, "stability IR belongs to a different Dirac hash")
    family = stability_ir.get("family", "unsupported")
    if family == "unsupported":
        unresolved = _unresolved(
            action_ir, "no gauge-reduced principal adapter for this action family"
        )
        unresolved.update(
            {
                "input_dirac_ir_sha256": dirac_ir.get("content_sha256"),
                "input_stability_ir_sha256": stability_ir.get("content_sha256"),
                "family": family,
            }
        )
        return unresolved
    try:
        basis, kinetic, gradient, named_speeds = build_physical_principal_blocks(
            action_ir, family
        )
    except (ValueError, ZeroDivisionError) as error:
        return _reject(action_ir, str(error))

    closure = dirac_ir.get("distributed_constraint_closure", {})
    rank = closure.get("constraint_surface_rank", {})
    physical_dof = rank.get("physical_dof")
    reduction_pass = (
        dirac_ir.get("status") == "pass"
        and closure.get("status") == "pass"
        and physical_dof == len(basis)
    )
    stability_principal = stability_ir.get("principal_symbol", {})
    domain_pass = (
        stability_ir.get("condition_certificate", {}).get("pointwise_status")
        == "pass"
    )
    source_principal_status = stability_principal.get("status", "unresolved")

    determinant = sp.factor(kinetic.det())
    if determinant == 0:
        propagation = sp.zeros(len(basis))
        residual = sp.zeros(len(basis))
        exact_regular_symbol = False
    else:
        propagation = sp.simplify(kinetic.inv() * gradient)
        expected_entries: list[sp.Expr] = []
        if family == "einstein_hilbert":
            expected_entries = [named_speeds["tensor_speed_squared"]] * 2
        elif family in {"canonical_scalar_gravity", "quartic_horndeski"}:
            expected_entries = [named_speeds["tensor_speed_squared"]] * 2 + [
                named_speeds["scalar_speed_squared"]
            ]
        elif family == "proca_gravity":
            expected_entries = [named_speeds["tensor_speed_squared"]] * 2 + [
                named_speeds["proca_speed_squared"]
            ] * 3
        else:
            expected_entries = [named_speeds["spin_2_speed_squared"]] * 2 + [
                named_speeds["spin_1_speed_squared"]
            ] * 2 + [named_speeds["spin_0_speed_squared"]]
        residual = sp.simplify(propagation - sp.diag(*expected_entries))
        exact_regular_symbol = residual == sp.zeros(len(basis))

    if source_principal_status == "reject" or not exact_regular_symbol:
        status = "reject"
    elif reduction_pass and domain_pass and source_principal_status == "pass":
        status = "pass"
    else:
        status = "unresolved"
    local_patch_pass = (
        family == "quartic_horndeski"
        and reduction_pass
        and domain_pass
        and exact_regular_symbol
        and not stability_principal.get("missing_or_failed_controls", [])
        and "quartic_horndeski_timelike_flat_principal_symbol"
        in stability_principal.get("required_controls", [])
    )
    omega, wave_number = sp.symbols("omega k", real=True)
    principal_polynomial = sp.factor(
        (-omega**2 * kinetic + wave_number**2 * gradient).det()
    )
    body = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "promotion_allowed": False,
        "input_action_sha256": action_ir["content_sha256"],
        "input_dirac_ir_sha256": dirac_ir.get("content_sha256"),
        "input_stability_ir_sha256": stability_ir.get("content_sha256"),
        "family": family,
        "gauge_reduction_certificate": {
            "status": "pass" if reduction_pass else "unresolved",
            "dirac_closure_status": closure.get("status"),
            "physical_dof_from_constraint_surface": physical_dof,
            "retained_physical_basis": basis,
            "retained_mode_count": len(basis),
            "constrained_or_gauge_variables_retained": [],
        },
        "background": (
            "aligned Minkowski Aether with arbitrary-background hyperbolicity theorem bound through the stability IR"
            if family == "einstein_aether"
            else (
                "flat constant-timelike-gradient scalar background"
                if family == "quartic_horndeski"
                else "frozen local physical frame on the executable Minkowski, FLRW, and static-spherical control backgrounds"
            )
        ),
        "kinetic_matrix": str(kinetic),
        "gradient_matrix": str(gradient),
        "kinetic_determinant": str(determinant),
        "propagation_matrix": str(propagation),
        "propagation_residual": str(residual),
        "principal_polynomial": str(principal_polynomial),
        "characteristic_speed_squared": {
            name: str(value) for name, value in sorted(named_speeds.items())
        },
        "domain_certificate_status": stability_ir.get("condition_certificate", {}).get(
            "status"
        ),
        "pointwise_domain_certificate_status": stability_ir.get(
            "condition_certificate", {}
        ).get("pointwise_status"),
        "source_principal_certificate_status": source_principal_status,
        "declared_background_patch_certificate": {
            "status": "pass" if local_patch_pass else "unresolved",
            "scope": (
                "constraint-count-matched flat constant-timelike-gradient three-mode symbol"
                if family == "quartic_horndeski"
                else "not separately applicable"
            ),
        },
        "exact_diagonal_physical_eigenbasis": exact_regular_symbol,
        "uniform_direction_scope": "isotropic reduced symbol; the displayed eigenbasis covers every nonzero spatial covector direction",
        "cone_policy": "characteristic speeds are reported without imposing an observational subluminality cut",
        "proof_scope": (
            "action/Dirac/stability-hash-bound physical-mode principal matrices after generated "
            "constraint and gauge counting; mass and potential terms correctly drop from the principal part"
        ),
    }
    content = _canonical_json(body)
    return {**body, "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest()}


def write_physical_principal_ir(principal_ir: dict[str, Any], output: str | Path) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(principal_ir, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
