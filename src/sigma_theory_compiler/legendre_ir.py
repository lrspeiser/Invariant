from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import sympy as sp

from .action_ir import _coefficient_symbols

SCHEMA_VERSION = "sigma-legendre-ir-1.0"

_SUPPORTED_LOCAL_KINETIC_TERMS = {
    "EH_R",
    "SCALAR_X",
    "SCALAR_MASS",
    "HORNDESKI_L4_LINEAR_X",
    "PROCA_F2",
    "PROCA_MASS",
    "AETHER_K1",
    "AETHER_K2",
    "AETHER_K3",
    "AETHER_K4",
    "AETHER_X_SQRT1P",
    "UNIT_VECTOR_CONSTRAINT",
}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _parse_coefficients(action_ir: dict[str, Any]) -> dict[str, sp.Expr]:
    names = sorted(
        {
            name
            for term in action_ir["canonical"]["terms"]
            for name in _coefficient_symbols(term["coefficient"])
        }
    )
    symbols = {name: sp.Symbol(name, real=True) for name in names}
    return {
        term["id"]: sp.sympify(
            str(term["coefficient"]).replace("^", "**"),
            locals=symbols,
            evaluate=True,
        )
        for term in action_ir["canonical"]["terms"]
    }


def _factor_records(expression: sp.Expr) -> list[dict[str, Any]]:
    if expression == 0:
        return []
    _, factors = sp.factor_list(expression)
    return [
        {
            "factor": str(sp.factor(factor)),
            "multiplicity": int(multiplicity),
            "regularity_condition": f"{sp.factor(factor)} != 0",
        }
        for factor, multiplicity in factors
        if not factor.is_number
    ]


def build_local_kinetic_model(action_ir: dict[str, Any]) -> dict[str, Any]:
    """Return the exact symbolic aligned/frozen kinetic model behind Legendre IR.

    This is an in-memory representation: SymPy expressions are intentionally retained so the
    canonical/Dirac stage can derive momenta and Hamiltonians from the same source rather than
    parsing serialized display strings.
    """

    if not action_ir.get("valid"):
        raise ValueError("covariant action IR is invalid")

    coefficients = _parse_coefficients(action_ir)
    unsupported_kinetic_terms = sorted(set(coefficients) - _SUPPORTED_LOCAL_KINETIC_TERMS)
    k11, k22, k33, k12, k13, k23 = sp.symbols("K11 K22 K33 K12 K13 K23", real=True)
    metric_velocities = (k11, k22, k33, k12, k13, k23)
    extrinsic = sp.Matrix([[k11, k12, k13], [k12, k22, k23], [k13, k23, k33]])
    trace_k = sp.trace(extrinsic)
    k_squared = sp.trace(extrinsic.T * extrinsic)
    metric_kinetic_coefficient = coefficients.get("EH_R", sp.Integer(0))
    is_quartic_horndeski = "HORNDESKI_L4_LINEAR_X" in coefficients
    a_star: sp.Symbol | None = None
    if is_quartic_horndeski:
        # For G4=M_Pl^2/2+alpha*X_c, the exact unitary-gauge Horndeski
        # cancellation leaves (G4-2 X_c G4_X)(K_ij K^ij-K^2), with
        # X_c=A_star^2/2.  V_star=L_n A_star is retained explicitly below as
        # the cancelled/null auxiliary velocity rather than being confused
        # with the ordinary first-derivative scalar channel Pi_phi.
        a_star = sp.Symbol("A_star", positive=True, finite=True)
        metric_kinetic_coefficient -= (
            coefficients["HORNDESKI_L4_LINEAR_X"] * a_star**2 / 2
        )
    kinetic = metric_kinetic_coefficient * (k_squared - trace_k**2)
    velocities: list[sp.Symbol] = list(metric_velocities)
    sectors: dict[str, list[int]] = {"metric_K_ij": list(range(6))}

    fields = set(action_ir["canonical"]["fields"])
    if "phi" in fields:
        if is_quartic_horndeski:
            v_star = sp.Symbol("V_star", real=True)
            sectors["quartic_horndeski_auxiliary_scalar"] = [len(velocities)]
            velocities.append(v_star)
            # The exact named Horndeski relation makes the V_star coefficient
            # identically zero.  Keeping the channel makes the primary null
            # direction visible to the Legendre/Dirac stages.
        else:
            pi_phi = sp.Symbol("Pi_phi", real=True)
            sectors["scalar_phi"] = [len(velocities)]
            velocities.append(pi_phi)
            kinetic += coefficients.get("SCALAR_X", sp.Integer(0)) * pi_phi**2 / 2

    if "A_mu" in fields:
        electric = tuple(sp.symbols("E_A0:3", real=True))
        sectors["proca_spatial_vector"] = list(range(len(velocities), len(velocities) + 3))
        velocities.extend(electric)
        kinetic += coefficients.get("PROCA_F2", sp.Integer(0)) * (
            -2 * sum(item**2 for item in electric)
        )

    if "u_mu" in fields:
        aether_velocity = tuple(sp.symbols("V_u0:3", real=True))
        sectors["unit_aether_spatial_vector"] = list(range(len(velocities), len(velocities) + 3))
        velocities.extend(aether_velocity)
        vector_square = sum(item**2 for item in aether_velocity)
        kinetic += coefficients.get("AETHER_K1", sp.Integer(0)) * (-vector_square + k_squared)
        kinetic += coefficients.get("AETHER_K2", sp.Integer(0)) * trace_k**2
        kinetic += coefficients.get("AETHER_K3", sp.Integer(0)) * k_squared
        kinetic += coefficients.get("AETHER_K4", sp.Integer(0)) * vector_square
        if "AETHER_X_SQRT1P" in coefficients:
            coefficient_symbols = {
                str(symbol): symbol
                for expression in coefficients.values()
                for symbol in expression.free_symbols
            }
            acceleration_scale = coefficient_symbols.get(
                "a_sigma", sp.Symbol("a_sigma", nonzero=True, real=True)
            )
            kinetic += (
                coefficients["AETHER_X_SQRT1P"]
                * vector_square
                / (2 * acceleration_scale**2)
            )

    return {
        "coefficients": coefficients,
        "fields": fields,
        "velocities": tuple(velocities),
        "sectors": sectors,
        "kinetic_lagrangian": sp.expand(kinetic),
        "unitary_gauge_auxiliary": (
            {"A_star": a_star, "X_c": a_star**2 / 2} if a_star is not None else None
        ),
        "unsupported_kinetic_terms": unsupported_kinetic_terms,
    }


def compile_legendre_ir(action_ir: dict[str, Any], adm_ir: dict[str, Any]) -> dict[str, Any]:
    """Build the exact local kinetic Hessian for one compiled bounded-grammar action."""

    if not action_ir.get("valid"):
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "reject",
            "promotion_allowed": False,
            "input_action_sha256": action_ir.get("content_sha256"),
            "errors": ["covariant action IR is invalid"],
        }
    if adm_ir.get("input_action_sha256") != action_ir.get("content_sha256"):
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "reject",
            "promotion_allowed": False,
            "input_action_sha256": action_ir.get("content_sha256"),
            "errors": ["ADM IR belongs to a different action hash"],
        }

    model = build_local_kinetic_model(action_ir)
    coefficients = model["coefficients"]
    fields = model["fields"]
    sectors = model["sectors"]
    kinetic = model["kinetic_lagrangian"]
    term_ids = set(coefficients)
    unsupported_kinetic_terms = list(model["unsupported_kinetic_terms"])

    velocity_tuple = model["velocities"]
    hessian = sp.hessian(sp.expand(kinetic), velocity_tuple)
    determinant = sp.factor(hessian.det())
    rank = int(hessian.rank())
    nullspace = hessian.nullspace()
    momenta = sp.Matrix([sp.Symbol(f"p_{velocity}", real=True) for velocity in velocity_tuple])
    kinetic_primary_constraints = [str(sp.factor((null.T * momenta)[0])) for null in nullspace]
    sector_records: dict[str, Any] = {}
    for name, indices in sectors.items():
        block = hessian.extract(indices, indices)
        sector_records[name] = {
            "indices": indices,
            "velocities": [str(velocity_tuple[index]) for index in indices],
            "hessian": str(block),
            "determinant": str(sp.factor(block.det())),
            "generic_rank": int(block.rank()),
            "dimension": len(indices),
        }

    regular = rank == len(velocity_tuple)
    expected_horndeski_degeneracy = (
        "HORNDESKI_L4_LINEAR_X" in term_ids
        and len(velocity_tuple) == 7
        and rank == 6
        and len(nullspace) == 1
        and list(nullspace[0]) == [0, 0, 0, 0, 0, 0, 1]
        and kinetic_primary_constraints == ["p_V_star"]
    )
    verified_legendre_structure = regular or expected_horndeski_degeneracy
    if expected_horndeski_degeneracy:
        metric_indices = sectors["metric_K_ij"]
        rank_minor = hessian.extract(metric_indices, metric_indices).det()
    else:
        rank_minor = determinant
    body = {
        "schema_version": SCHEMA_VERSION,
        "status": (
            "pass"
            if adm_ir.get("status") == "pass"
            and verified_legendre_structure
            and not unsupported_kinetic_terms
            else "unresolved"
        ),
        "promotion_allowed": False,
        "input_action_sha256": action_ir["content_sha256"],
        "input_adm_ir_sha256": adm_ir.get("content_sha256"),
        "source_role": action_ir["canonical"]["source_role"],
        "coefficient_map": {name: str(value) for name, value in sorted(coefficients.items())},
        "term_ids": sorted(term_ids),
        "unsupported_kinetic_terms": unsupported_kinetic_terms,
        "background": "local aligned/frozen spatial orthonormal frame",
        "unit_aether_branch": "chi=1,A_i=0" if "u_mu" in fields else None,
        "quartic_horndeski_branch": (
            "unitary gauge, X_c=A_star^2/2, V_star=L_n A_star"
            if expected_horndeski_degeneracy
            else None
        ),
        "velocity_order": [str(item) for item in velocity_tuple],
        "velocity_count": len(velocity_tuple),
        "local_kinetic_lagrangian": str(sp.factor(kinetic)),
        "hessian": str(hessian),
        "hessian_determinant": str(determinant),
        "generic_hessian_rank": rank,
        "generic_hessian_nullity": len(velocity_tuple) - rank,
        "legendre_status": (
            "regular_generic"
            if regular
            else (
                "degenerate_primary_verified"
                if expected_horndeski_degeneracy
                else "singular_requires_dirac"
            )
        ),
        "rank_regularity_minor": str(sp.factor(rank_minor)),
        "regularity_factors": _factor_records(rank_minor),
        "kinetic_primary_constraints": kinetic_primary_constraints,
        "nondynamical_primary_seeds": list(adm_ir.get("primary_constraint_seeds", [])),
        "secondary_seeds": list(adm_ir.get("secondary_constraint_seeds", [])),
        "sector_blocks": sector_records,
        "boundary_contract": list(adm_ir.get("boundary_contract", [])),
        "proof_scope": (
            "exact pointwise kinetic Hessian and generic Legendre-rank strata for the frozen "
            "action coefficients only for terms listed as locally supported; unsupported "
            "higher-jet terms are reported and force unresolved status. Lapse/shift and algebraic-field "
            "primaries are carried from ADM IR, while distributed consistency, Poisson closure, "
            "gauge reduction, and reduced Hamiltonian boundedness remain separate"
        ),
    }
    content = _canonical_json(body)
    return {**body, "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest()}


def write_legendre_ir(legendre_ir: dict[str, Any], output: str | Path) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(legendre_ir, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
