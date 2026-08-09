from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import sympy as sp

from .dhost import quartic_horndeski_unitary_flrw_dirac_control
from .dirac import partial_velocity_solution
from .field_dirac import canonical_scalar_spatial_density_certificate
from .legendre_ir import build_local_kinetic_model

SCHEMA_VERSION = "sigma-dirac-ir-1.0"


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


def _canonical_local_model(action_ir: dict[str, Any]) -> dict[str, Any]:
    model = build_local_kinetic_model(action_ir)
    channel_velocities = model["velocities"]
    metric_coordinates = sp.symbols("h11 h22 h33 h12 h13 h23", real=True)
    metric_velocities = sp.symbols("dot_h11 dot_h22 dot_h33 dot_h12 dot_h13 dot_h23", real=True)
    coordinates: list[sp.Symbol] = list(metric_coordinates)
    canonical_velocities: list[sp.Symbol] = list(metric_velocities)
    substitutions: dict[sp.Symbol, sp.Expr] = {
        channel: velocity / 2
        for channel, velocity in zip(channel_velocities[:6], metric_velocities, strict=True)
    }
    channel_map = [
        f"{channel}={velocity}/2 at N=1,N^i=0"
        for channel, velocity in zip(channel_velocities[:6], metric_velocities, strict=True)
    ]

    for channel in channel_velocities[6:]:
        if str(channel) == "Pi_phi":
            coordinate = sp.Symbol("phi", real=True)
            velocity = sp.Symbol("dot_phi", real=True)
        elif str(channel).startswith("E_A"):
            index = str(channel).removeprefix("E_A")
            coordinate = sp.Symbol(f"A{index}", real=True)
            velocity = sp.Symbol(f"dot_A{index}", real=True)
        elif str(channel).startswith("V_u"):
            index = str(channel).removeprefix("V_u")
            coordinate = sp.Symbol(f"u{index}", real=True)
            velocity = sp.Symbol(f"dot_u{index}", real=True)
        elif str(channel) == "V_star":
            coordinate = sp.Symbol("A_star", positive=True, finite=True)
            velocity = sp.Symbol("dot_A_star", real=True)
        else:
            raise ValueError(f"unsupported kinetic channel: {channel}")
        coordinates.append(coordinate)
        canonical_velocities.append(velocity)
        substitutions[channel] = velocity
        channel_map.append(f"{channel}={velocity} at N=1,N^i=0")

    lagrangian = sp.expand(model["kinetic_lagrangian"].subs(substitutions))
    velocity_tuple = tuple(canonical_velocities)
    coordinate_tuple = tuple(coordinates)
    hessian = sp.hessian(lagrangian, velocity_tuple)
    channel_hessian = sp.hessian(model["kinetic_lagrangian"], channel_velocities)
    jacobian = sp.diag(*([sp.Rational(1, 2)] * 6 + [sp.Integer(1)] * (len(velocity_tuple) - 6)))
    hessian_residual = (hessian - jacobian.T * channel_hessian * jacobian).applyfunc(sp.factor)
    momenta = tuple(sp.Symbol(f"P_{coordinate}", real=True) for coordinate in coordinate_tuple)
    canonical_momenta = tuple(sp.diff(lagrangian, item) for item in velocity_tuple)
    affine = sp.Matrix(canonical_momenta) - hessian * sp.Matrix(velocity_tuple)
    nullspace = hessian.nullspace()
    primary_constraints = tuple(
        sp.factor((null.T * (sp.Matrix(momenta) - affine))[0]) for null in nullspace
    )
    velocity_solution, unresolved_velocities = partial_velocity_solution(
        hessian, affine, velocity_tuple, momenta
    )
    canonical_hamiltonian = sp.expand(
        sum(momentum * velocity for momentum, velocity in zip(momenta, velocity_tuple, strict=True))
        - lagrangian
    )
    canonical_hamiltonian = sp.factor(
        canonical_hamiltonian.subs(velocity_solution).subs(
            {item: 0 for item in unresolved_velocities}
        )
    )
    legendre_residuals = tuple(
        sp.factor(
            sp.diff(canonical_hamiltonian, momentum)
            - velocity_solution.get(velocity, sp.Integer(0)).subs(
                {item: 0 for item in unresolved_velocities}
            )
        )
        for momentum, velocity in zip(momenta, velocity_tuple, strict=True)
        if velocity in velocity_solution
    )
    return {
        "coordinates": coordinate_tuple,
        "velocities": velocity_tuple,
        "momenta": momenta,
        "channel_map": channel_map,
        "lagrangian": lagrangian,
        "hessian": hessian,
        "hessian_residual": hessian_residual,
        "canonical_momenta": canonical_momenta,
        "primary_constraints": primary_constraints,
        "velocity_solution": velocity_solution,
        "unresolved_velocities": unresolved_velocities,
        "canonical_hamiltonian": canonical_hamiltonian,
        "legendre_residuals": legendre_residuals,
        "unsupported_kinetic_terms": list(model["unsupported_kinetic_terms"]),
    }


def _closure_certificate(
    action_ir: dict[str, Any],
    local: dict[str, Any],
    control_status: dict[str, bool] | None,
) -> dict[str, Any]:
    controls = control_status or {}
    term_ids = {item["id"] for item in action_ir["canonical"]["terms"]}
    fields = set(action_ir["canonical"]["fields"])
    regular = int(local["hessian"].rank()) == len(local["velocities"])
    gravity_controls = [
        "canonical_metric_diffeomorphism_algebra",
        "canonical_metric_dewitt_kinetic_covariance",
        "spatial_curvature_density_diffeomorphism_covariance",
        "cadabra_adm_spatial_curvature_variation",
        "nonlinear_adm_hamiltonian_constraint_algebra",
    ]
    family = "unsupported"
    required_controls: list[str] = []
    proof_route: list[str] = []
    reduced_pairs: int | None = None
    physical_dof: int | None = None
    reduced_first_class: int | None = None
    reduced_second_class: int | None = None
    extended_first_class: int | None = None
    extended_second_class: int | None = None
    extended_pairs: int | None = None
    closure_scope = "no generated distributed closure adapter"

    if term_ids == {"EH_R"}:
        family = "einstein_hilbert"
        required_controls = gravity_controls
        proof_route = [
            "exact canonical metric D-D cotangent-lift algebra",
            "exact kinetic and curvature D-H covariance",
            "exact nonlinear GR H-H bracket with inverse-metric structure function",
        ]
        reduced_pairs, physical_dof = 6, 2
        reduced_first_class, reduced_second_class = 4, 0
        extended_first_class, extended_second_class = 8, 0
        closure_scope = "complete nonlinear pure-GR distributed constraint algebra"
    elif "phi" in fields and term_ids <= {"EH_R", "SCALAR_X", "SCALAR_MASS"}:
        family = "canonical_scalar_gravity"
        scalar_certificate = canonical_scalar_spatial_density_certificate()
        required_controls = gravity_controls + [
            "canonical_scalar",
            "canonical_scalar_noether_identity",
            "canonical_scalar_gravity_cross_constraint_identities",
            "three_spatial_dimensional_smeared_brackets",
        ]
        proof_route = [
            "pure GR D-D/D-H/H-H closure",
            "exact three-dimensional scalar H-H and D-D local-functional brackets",
            "exact scalar Hamiltonian weight-one D-H identity",
            "ultralocal gravity-scalar H-H cross terms cancel under lapse antisymmetrization",
        ]
        reduced_pairs, physical_dof = 7, 3
        reduced_first_class, reduced_second_class = 4, 0
        extended_first_class, extended_second_class = 8, 0
        closure_scope = "complete minimally coupled canonical-scalar plus GR algebra"
        if not scalar_certificate["passed"]:
            regular = False
    elif (
        "u_mu" in fields
        and "UNIT_VECTOR_CONSTRAINT" in term_ids
        and term_ids
        <= {
            "EH_R",
            "AETHER_K1",
            "AETHER_K2",
            "AETHER_K3",
            "AETHER_K4",
            "UNIT_VECTOR_CONSTRAINT",
        }
    ):
        family = "einstein_aether"
        required_controls = [
            "einstein_aether_generic_3plus1_legendre",
            "einstein_aether_generic_lapse_shift_constraint_seeds",
            "einstein_aether_spatial_diffeomorphism_algebra",
            "einstein_aether_generic_dh_covariance",
            "einstein_aether_generic_hh_deformation_kinematics",
            "einstein_aether_arbitrary_background_4d_noether",
            "regular_holonomic_multiplier_dirac_theorem",
        ]
        proof_route = [
            "positive unit branch solves the regular four-constraint holonomic multiplier chain",
            "exact Aether-metric spatial cotangent lift closes D-D",
            "generic independent-coefficient Hamiltonian density closes D-H",
            "normal-deformation/Jacobi theorem closes H-H on regular Legendre patches",
        ]
        reduced_pairs, physical_dof = 9, 5
        reduced_first_class, reduced_second_class = 4, 0
        extended_first_class, extended_second_class = 8, 0
        closure_scope = (
            "complete generic K1..K4 Aether distributed algebra on regular positive-unit-branch "
            "Legendre patches; coefficient specializations inherit the proof only while regular"
        )
    elif "A_mu" in fields:
        family = "proca_gravity"
        required_controls = gravity_controls + [
            "proca_adm_dirac",
            "proca_divergence_identity",
            "proca_stress_noether_identity",
            "proca_reduced_smeared_constraint_algebra",
            "einstein_aether_spatial_diffeomorphism_algebra",
        ]
        proof_route = [
            "the normal Proca component and its momentum form an exact regular second-class pair",
            "the positive reduced 3D Proca Hamiltonian closes H-H into its covector momentum constraint modulo a spatial boundary",
            "the universal covector cotangent lift closes D-D and establishes D-H covariance",
            "gravity-Proca metric cross terms are ultralocal in both lapses and cancel under antisymmetrization",
        ]
        reduced_pairs, physical_dof = 9, 5
        reduced_first_class, reduced_second_class = 4, 0
        extended_first_class, extended_second_class = 8, 2
        closure_scope = (
            "complete massive minimally coupled Proca+GR algebra on m_A>0, positive-metric "
            "patches after exact elimination of the normal-component second-class pair"
        )
    elif term_ids == {"EH_R", "SCALAR_X", "HORNDESKI_L4_LINEAR_X"}:
        family = "quartic_horndeski"
        required_controls = [
            "quartic_horndeski_covariant_adm_degeneracy",
            "quartic_horndeski_unitary_flrw_dirac_chain",
            "quartic_horndeski_unitary_distributed_dirac_closure",
        ]
        proof_route = [
            "exact unitary-gauge ADM cancellation exposes the auxiliary scalar null velocity",
            "exact curved-FLRW quotient-surface Dirac chain pairs p_N with its secondary constraint",
            "the exact 3D metric+lapse cotangent lift closes D-D and D-C_N",
            "invertible distributed lapse Hessian pairs p_N with C_N and fixes the lapse multiplier",
        ]
        closure_scope = (
            "complete unitary-gauge distributed constraint algebra and three-mode count on regular "
            "invertible lapse-Hessian operator patches; global invertibility and boundary zero "
            "modes remain separate"
        )
        reduced_pairs, physical_dof = 6, 3
        reduced_first_class, reduced_second_class = 3, 0
        extended_first_class, extended_second_class = 6, 2
        extended_pairs = 10

    failed_controls = [name for name in required_controls if not controls.get(name, False)]
    adapter_complete = family in {
        "einstein_hilbert",
        "canonical_scalar_gravity",
        "einstein_aether",
        "proca_gravity",
        "quartic_horndeski",
    }
    horndeski_degenerate_patch = (
        family == "quartic_horndeski"
        and int(local["hessian"].rank()) == len(local["velocities"]) - 1
        and [str(item) for item in local["primary_constraints"]] == ["P_A_star"]
    )
    legendre_admissible = regular or horndeski_degenerate_patch
    closure_pass = legendre_admissible and adapter_complete and not failed_controls
    primary = ["p_N=0", "p_(N^i)=0 (three components)"]
    secondary = ["H_perp=0", "H_i=0 (three components)"]
    if family == "proca_gravity":
        primary.append("p_(A_perp)=0")
        secondary.append("D_i p^i+m_A^2 sqrt(h) A_perp=0")
    if family == "quartic_horndeski":
        primary = ["p_N=0", "p_(N^i)=0 (three components)"]
        secondary = [
            "C_N=delta(H)/delta(N)=0 (second class with p_N on an invertible lapse-Hessian patch)",
            "H_i=0 (three spatial-diffeomorphism constraints)",
        ]
    if extended_pairs is None and reduced_pairs is not None:
        extended_pairs = reduced_pairs + (5 if family == "proca_gravity" else 4)
    if family == "quartic_horndeski":
        poisson_algebra = {
            "D_D": "{D[M],D[L]}=D[[M,L]]",
            "D_C": "{D[M],C_N[f]}=-C_N[Lie_M f]",
            "lapse_pair": "{p_N(x),C_N(y)}=Delta_N(x,y), invertible on the declared regular patch",
            "H_H": "not applicable as a first-class bracket after the unitary scalar-clock gauge",
            "status": "pass" if closure_pass else "unresolved",
        }
    else:
        poisson_algebra = {
            "D_D": "{D[M],D[L]}=D[[M,L]]",
            "D_H": "{D[M],H[N]}=H[Lie_M N]",
            "H_H": "{H[N],H[M]}=D[h^ij(N D_j M-M D_j N)]",
            "status": "pass" if closure_pass else "unresolved",
        }
    return {
        "family": family,
        "status": "pass" if closure_pass else "unresolved",
        "regular_local_legendre_patch": legendre_admissible,
        "required_controls": required_controls,
        "missing_or_failed_controls": failed_controls,
        "proof_route": proof_route,
        "constraint_generations": {"primary": primary, "secondary": secondary},
        "poisson_algebra": poisson_algebra,
        "constraint_surface_rank": {
            "reduced_canonical_pairs": reduced_pairs,
            "first_class_constraints": reduced_first_class if closure_pass else None,
            "second_class_constraints_after_reduction": (
                reduced_second_class if closure_pass else None
            ),
            "physical_dof": physical_dof if closure_pass else None,
            "extended_pairs": extended_pairs,
            "extended_first_class_constraints": (
                extended_first_class if closure_pass else None
            ),
            "extended_second_class_constraints": (extended_second_class if closure_pass else None),
        },
        "scope": closure_scope,
    }


def compile_dirac_ir(
    action_ir: dict[str, Any],
    adm_ir: dict[str, Any],
    legendre_ir: dict[str, Any],
    control_status: dict[str, bool] | None = None,
) -> dict[str, Any]:
    """Compile an exact local canonical transform plus fail-closed distributed closure IR."""

    if not action_ir.get("valid"):
        return _reject(action_ir, "covariant action IR is invalid")
    if adm_ir.get("input_action_sha256") != action_ir.get("content_sha256"):
        return _reject(action_ir, "ADM IR belongs to a different action hash")
    if legendre_ir.get("input_action_sha256") != action_ir.get("content_sha256"):
        return _reject(action_ir, "Legendre IR belongs to a different action hash")
    if legendre_ir.get("input_adm_ir_sha256") != adm_ir.get("content_sha256"):
        return _reject(action_ir, "Legendre IR belongs to a different ADM hash")

    local = _canonical_local_model(action_ir)
    rank = int(local["hessian"].rank())
    nullity = len(local["velocities"]) - rank
    hessian_consistent = local["hessian_residual"] == sp.zeros(len(local["velocities"]))
    legendre_consistent = all(item == 0 for item in local["legendre_residuals"])
    closure = _closure_certificate(action_ir, local, control_status)
    unitary_horndeski_dirac = (
        quartic_horndeski_unitary_flrw_dirac_control()
        if closure["family"] == "quartic_horndeski"
        else None
    )
    local_status = (
        "unresolved"
        if local["unsupported_kinetic_terms"]
        else ("pass" if hessian_consistent and legendre_consistent else "reject")
    )
    status = "pass" if local_status == "pass" and closure["status"] == "pass" else "unresolved"
    body = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "promotion_allowed": False,
        "input_action_sha256": action_ir["content_sha256"],
        "input_adm_ir_sha256": adm_ir.get("content_sha256"),
        "input_legendre_ir_sha256": legendre_ir.get("content_sha256"),
        "source_role": action_ir["canonical"]["source_role"],
        "local_canonical_transform": {
            "status": local_status,
            "background": "aligned/frozen orthonormal frame with N=1,N^i=0",
            "coordinates": [str(item) for item in local["coordinates"]],
            "velocities": [str(item) for item in local["velocities"]],
            "momenta": [str(item) for item in local["momenta"]],
            "channel_map": local["channel_map"],
            "kinetic_lagrangian": str(sp.factor(local["lagrangian"])),
            "canonical_momenta": [str(sp.factor(item)) for item in local["canonical_momenta"]],
            "hessian": str(local["hessian"]),
            "hessian_rank": rank,
            "hessian_nullity": nullity,
            "hessian_chain_residual": str(local["hessian_residual"]),
            "primary_constraints": [str(item) for item in local["primary_constraints"]],
            "solved_velocities": {
                str(key): str(value) for key, value in local["velocity_solution"].items()
            },
            "unresolved_velocities": [str(item) for item in local["unresolved_velocities"]],
            "kinetic_canonical_hamiltonian": str(local["canonical_hamiltonian"]),
            "legendre_residuals": [str(item) for item in local["legendre_residuals"]],
            "unsupported_kinetic_terms": local["unsupported_kinetic_terms"],
            "scope": (
                "exact local canonical transform with the metric relation dot(h)_ij=2K_ij; "
                "spatial potential terms and boundary generators are certified separately"
            ),
        },
        "distributed_constraint_closure": closure,
        "unitary_gauge_local_dirac_control": unitary_horndeski_dirac,
        "hamiltonian_stability": {
            "status": "unresolved",
            "reason": (
                "the local kinetic Hamiltonian is not the gauge-reduced physical Hamiltonian; "
                "spatial potentials, all constraints, gauge fixing, and boundary charges must be included"
            ),
        },
        "proof_scope": (
            "hash-bound local canonical transformation plus action-family distributed closure "
            "instantiation only where exact smeared-functional/tensor controls cover the same "
            "coefficient specialization; unsupported or singular branches remain unresolved"
        ),
    }
    content = _canonical_json(body)
    return {**body, "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest()}


def write_dirac_ir(dirac_ir: dict[str, Any], output: str | Path) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dirac_ir, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
