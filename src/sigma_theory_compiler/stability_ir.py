from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import sympy as sp
from sympy.core.relational import (
    Equality,
    GreaterThan,
    LessThan,
    Relational,
    StrictGreaterThan,
    StrictLessThan,
    Unequality,
)
from sympy.logic.boolalg import BooleanFalse, BooleanTrue

from .legendre_ir import build_local_kinetic_model

SCHEMA_VERSION = "sigma-stability-ir-1.0"


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


def _property(relation: Relational) -> tuple[str, sp.Expr]:
    if isinstance(relation, StrictGreaterThan):
        return "positive", sp.factor(relation.lhs - relation.rhs)
    if isinstance(relation, StrictLessThan):
        return "positive", sp.factor(relation.rhs - relation.lhs)
    if isinstance(relation, GreaterThan):
        return "nonnegative", sp.factor(relation.lhs - relation.rhs)
    if isinstance(relation, LessThan):
        return "nonnegative", sp.factor(relation.rhs - relation.lhs)
    if isinstance(relation, Unequality):
        return "nonzero", sp.factor(relation.lhs - relation.rhs)
    if isinstance(relation, Equality):
        return "zero", sp.factor(relation.lhs - relation.rhs)
    raise TypeError(f"unsupported relation type: {type(relation).__name__}")


def _declared_domain(action_ir: dict[str, Any]) -> dict[str, Any]:
    canonical = action_ir["canonical"]
    names = canonical["universal_constants"]
    background = canonical.get("background_domain")
    background_variables = {
        item["id"]: item for item in (background or {}).get("variables", [])
    }
    symbols = {
        name: sp.Symbol(name, real=True)
        for name in [*names, *background_variables]
    }
    raw = canonical.get("parameter_domain", {})
    predicates: list[sp.logic.boolalg.Boolean] = []
    for name in raw.get("positive", []):
        predicates.append(sp.Q.positive(symbols[name]))
    for name in raw.get("nonnegative", []):
        predicates.append(sp.Q.nonnegative(symbols[name]))
    for name in raw.get("negative", []):
        predicates.append(sp.Q.negative(symbols[name]))
    for name in raw.get("nonpositive", []):
        predicates.append(sp.Q.nonpositive(symbols[name]))
    for name in raw.get("nonzero", []):
        predicates.append(sp.Q.nonzero(symbols[name]))
    relations = [
        sp.sympify(text, locals=symbols, evaluate=True)
        for text in raw.get("inequalities", [])
    ]
    background_relations = [
        sp.sympify(text, locals=symbols, evaluate=True)
        for text in (background or {}).get("inequalities", [])
    ]
    for name, record in background_variables.items():
        if record.get("nonnegative"):
            predicates.append(sp.Q.nonnegative(symbols[name]))
    return {
        "symbols": symbols,
        "predicates": predicates,
        "relations": [*relations, *background_relations],
        "canonical": raw,
        "background_canonical": background,
        "background_preservation": (background or {}).get("preservation"),
    }


def _same_expression(left: sp.Expr, right: sp.Expr) -> bool:
    return sp.cancel(left - right) == 0


def _prove_relation(relation: Relational | BooleanTrue | BooleanFalse, domain: dict[str, Any]) -> dict[str, Any]:
    if isinstance(relation, BooleanTrue):
        return {
            "status": "pass",
            "condition": "True",
            "normalized_property": "exact_true",
            "proof": "required coefficient condition simplifies exactly to true",
        }
    if isinstance(relation, BooleanFalse):
        return {
            "status": "reject",
            "condition": "False",
            "normalized_property": "exact_false",
            "proof": "required coefficient condition simplifies exactly to false",
        }
    kind, expression = _property(relation)
    expression = sp.factor(expression)
    direct: bool | None
    if kind == "positive":
        direct = expression.is_positive
        query = sp.Q.positive(expression)
        contradiction_queries = (sp.Q.negative(expression), sp.Q.zero(expression))
    elif kind == "nonnegative":
        direct = expression.is_nonnegative
        query = sp.Q.nonnegative(expression)
        contradiction_queries = (sp.Q.negative(expression),)
    elif kind == "nonzero":
        direct = expression.is_nonzero
        query = sp.Q.nonzero(expression)
        contradiction_queries = (sp.Q.zero(expression),)
    else:
        direct = expression.is_zero
        query = sp.Q.zero(expression)
        contradiction_queries = (sp.Q.nonzero(expression),)
    if direct is True:
        return {
            "status": "pass",
            "condition": str(relation),
            "normalized_property": f"{kind}({expression})",
            "proof": "exact expression assumptions",
        }
    if direct is False:
        return {
            "status": "reject",
            "condition": str(relation),
            "normalized_property": f"{kind}({expression})",
            "proof": "exact expression contradicts the required sign",
        }

    entailment = {
        "positive": {"positive"},
        "nonnegative": {"positive", "nonnegative"},
        "nonzero": {"positive", "nonzero"},
        "zero": {"zero"},
    }
    contradiction = {
        "positive": {"zero"},
        "nonnegative": set(),
        "nonzero": {"zero"},
        "zero": {"positive", "nonzero"},
    }
    for declared in domain["relations"]:
        declared_kind, declared_expression = _property(declared)
        if _same_expression(expression, declared_expression):
            if declared_kind in entailment[kind]:
                return {
                    "status": "pass",
                    "condition": str(relation),
                    "normalized_property": f"{kind}({expression})",
                    "proof": f"declared inequality: {declared}",
                }
            if declared_kind in contradiction[kind]:
                return {
                    "status": "reject",
                    "condition": str(relation),
                    "normalized_property": f"{kind}({expression})",
                    "proof": f"contradicted by declared inequality: {declared}",
                }
        if _same_expression(expression, -declared_expression):
            if kind in {"positive", "nonnegative"} and declared_kind == "positive":
                return {
                    "status": "reject",
                    "condition": str(relation),
                    "normalized_property": f"{kind}({expression})",
                    "proof": f"opposite sign declared by: {declared}",
                }
            if kind == "nonzero" and declared_kind == "positive":
                return {
                    "status": "pass",
                    "condition": str(relation),
                    "normalized_property": f"{kind}({expression})",
                    "proof": f"opposite expression is strictly signed by: {declared}",
                }

    assumptions = sp.And(*domain["predicates"]) if domain["predicates"] else True
    asked = sp.ask(query, assumptions)
    if asked is True:
        return {
            "status": "pass",
            "condition": str(relation),
            "normalized_property": f"{kind}({expression})",
            "proof": "declared atomic sign assumptions",
        }
    if asked is False or any(sp.ask(item, assumptions) is True for item in contradiction_queries):
        return {
            "status": "reject",
            "condition": str(relation),
            "normalized_property": f"{kind}({expression})",
            "proof": "declared atomic sign assumptions contradict the requirement",
        }
    return {
        "status": "unresolved",
        "condition": str(relation),
        "normalized_property": f"{kind}({expression})",
        "proof": "not implied or contradicted by the frozen parameter domain",
    }


def _condition_summary(
    conditions: list[tuple[str, Relational | BooleanTrue | BooleanFalse]],
    domain: dict[str, Any],
) -> dict[str, Any]:
    records = []
    for name, relation in conditions:
        record = _prove_relation(relation, domain)
        records.append({"name": name, **record})
    if any(item["status"] == "reject" for item in records):
        status = "reject"
    elif all(item["status"] == "pass" for item in records):
        status = "pass"
    else:
        status = "unresolved"
    return {"status": status, "conditions": records}


def _family_conditions(
    action_ir: dict[str, Any], coefficients: dict[str, sp.Expr]
) -> tuple[str, list[tuple[str, Relational]], dict[str, sp.Expr]]:
    fields = set(action_ir["canonical"]["fields"])
    terms = {item["id"] for item in action_ir["canonical"]["terms"]}
    constants = {
        name: sp.Symbol(name, real=True)
        for name in action_ir["canonical"]["universal_constants"]
    }
    eh = coefficients.get("EH_R", sp.Integer(0))
    conditions: list[tuple[str, Relational | BooleanTrue | BooleanFalse]] = [
        ("positive_tensor_kinetic_and_newton_coupling", sp.Gt(eh, 0))
    ]
    derived: dict[str, sp.Expr] = {"EH_coefficient": eh}
    if terms == {"EH_R"}:
        return "einstein_hilbert", conditions, derived
    if "phi" in fields and terms <= {"EH_R", "SCALAR_X", "SCALAR_MASS"}:
        scalar_kinetic = coefficients.get("SCALAR_X", sp.Integer(0))
        scalar_mass = coefficients.get("SCALAR_MASS", sp.Integer(0))
        conditions.extend(
            [
                ("positive_scalar_kinetic_and_gradient", sp.Gt(scalar_kinetic, 0)),
                ("nonnegative_scalar_mass_energy", sp.Ge(-scalar_mass, 0)),
            ]
        )
        if "Lambda_phi" in constants:
            conditions.append(
                ("nonzero_scalar_normalization", sp.Ne(constants["Lambda_phi"], 0))
            )
        derived.update(
            {
                "scalar_kinetic_coefficient": scalar_kinetic,
                "scalar_mass_lagrangian_coefficient": scalar_mass,
                "scalar_speed_squared": sp.Integer(1),
            }
        )
        return "canonical_scalar_gravity", conditions, derived
    if terms == {"EH_R", "SCALAR_X", "HORNDESKI_L4_LINEAR_X"}:
        scalar_kinetic = coefficients.get("SCALAR_X", sp.Integer(0))
        alpha = coefficients["HORNDESKI_L4_LINEAR_X"]
        # Match the canonical action-domain symbol exactly. Nonnegativity is a
        # declared, hash-bound background predicate rather than an implicit
        # SymPy assumption that could differ from the compiled action symbol.
        a_star_squared = sp.Symbol("A_star_squared", real=True)
        tensor_kinetic = sp.factor(eh - alpha * a_star_squared / 2)
        tensor_gradient = sp.factor(eh + alpha * a_star_squared / 2)
        tensor_speed = sp.factor(tensor_gradient / tensor_kinetic)
        conditions.extend(
            [
                (
                    "positive_tensor_kinetic_gradient_product_on_declared_background",
                    sp.Gt(
                        (2 * tensor_kinetic) * (2 * tensor_gradient),
                        0,
                    ),
                ),
                (
                    "positive_tensor_kinetic_gradient_sum_on_declared_background",
                    sp.Gt(tensor_kinetic + tensor_gradient, 0),
                ),
                ("positive_canonical_scalar_kinetic", sp.Gt(scalar_kinetic, 0)),
            ]
        )
        if "Lambda_phi" in constants:
            conditions.append(
                ("nonzero_scalar_normalization", sp.Ne(constants["Lambda_phi"], 0))
            )
        derived.update(
            {
                "background_A_star_squared": a_star_squared,
                "tensor_kinetic_coefficient": tensor_kinetic,
                "tensor_gradient_coefficient": tensor_gradient,
                "tensor_speed_squared": tensor_speed,
                "scalar_kinetic_coefficient": scalar_kinetic,
                "scalar_speed_squared": sp.Integer(1),
            }
        )
        return "quartic_horndeski", conditions, derived
    if "A_mu" in fields:
        field_strength = coefficients.get("PROCA_F2", sp.Integer(0))
        mass_coefficient = coefficients.get("PROCA_MASS", sp.Integer(0))
        conditions.extend(
            [
                ("positive_proca_electric_and_magnetic_energy", sp.Gt(-field_strength, 0)),
                ("positive_proca_mass_energy", sp.Gt(-mass_coefficient, 0)),
            ]
        )
        if "m_A" in constants:
            conditions.append(("massive_second_class_domain", sp.Ne(constants["m_A"], 0)))
        derived.update(
            {
                "proca_F2_coefficient": field_strength,
                "proca_mass_lagrangian_coefficient": mass_coefficient,
                "proca_speed_squared": sp.Integer(1),
            }
        )
        return "proca_gravity", conditions, derived
    if "u_mu" in fields and terms <= {
        "EH_R",
        "AETHER_K1",
        "AETHER_K2",
        "AETHER_K3",
        "AETHER_K4",
        "UNIT_VECTOR_CONSTRAINT",
    }:
        c1 = sp.factor(-coefficients.get("AETHER_K1", 0) / eh)
        c2 = sp.factor(-coefficients.get("AETHER_K2", 0) / eh)
        c3 = sp.factor(-coefficients.get("AETHER_K3", 0) / eh)
        c4 = sp.factor(coefficients.get("AETHER_K4", 0) / eh)
        c13 = sp.factor(c1 + c3)
        c14 = sp.factor(c1 + c4)
        c123 = sp.factor(c1 + c2 + c3)
        spin1_numerator = sp.factor(2 * c1 - c1**2 + c3**2)
        scalar_denominator_factor = sp.factor(2 + c13 + 3 * c2)
        spin1_speed = sp.factor(spin1_numerator / (2 * c14 * (1 - c13)))
        spin0_speed = sp.factor(
            c123 * (2 - c14) / (c14 * (1 - c13) * scalar_denominator_factor)
        )
        conditions.extend(
            [
                ("positive_spin2_kinetic", sp.Gt(1 - c13, 0)),
                ("positive_aether_vector_amplitude", sp.Gt(c14, 0)),
                ("subcritical_aether_vector_amplitude", sp.Gt(2 - c14, 0)),
                ("positive_spin1_energy", sp.Gt(spin1_numerator, 0)),
                (
                    "positive_spin0_kinetic_gradient_product",
                    sp.Gt(c123 * scalar_denominator_factor, 0),
                ),
                ("nonluminal_spin1_sufficient_formulation", sp.Ne(spin1_speed, 1)),
                ("nonluminal_spin0_sufficient_formulation", sp.Ne(spin0_speed, 1)),
            ]
        )
        derived.update(
            {
                "c1_effective": c1,
                "c2_effective": c2,
                "c3_effective": c3,
                "c4_effective": c4,
                "c13_effective": c13,
                "c14_effective": c14,
                "c123_effective": c123,
                "spin2_speed_squared": sp.factor(1 / (1 - c13)),
                "spin1_speed_squared": spin1_speed,
                "spin0_speed_squared": spin0_speed,
            }
        )
        return "einstein_aether", conditions, derived
    return "unsupported", conditions, derived


def compile_stability_ir(
    action_ir: dict[str, Any],
    dirac_ir: dict[str, Any],
    control_status: dict[str, bool] | None = None,
) -> dict[str, Any]:
    """Bind coefficient-domain proofs to reduced Hamiltonian and principal-symbol controls."""

    if not action_ir.get("valid"):
        return _reject(action_ir, "covariant action IR is invalid")
    if dirac_ir.get("input_action_sha256") != action_ir.get("content_sha256"):
        return _reject(action_ir, "Dirac IR belongs to a different action hash")
    controls = control_status or {}
    model = build_local_kinetic_model(action_ir)
    family, conditions, derived = _family_conditions(action_ir, model["coefficients"])
    domain = _declared_domain(action_ir)
    condition_summary = _condition_summary(conditions, domain)
    pointwise_condition_status = condition_summary["status"]
    preservation = domain.get("background_preservation")
    if preservation is None:
        preservation_certificate = {
            "status": "not_applicable",
            "declared_status": None,
            "required_controls": [],
            "missing_or_failed_controls": [],
        }
    else:
        preservation_controls = list(preservation.get("required_controls", []))
        missing_preservation = [
            name for name in preservation_controls if not controls.get(name, False)
        ]
        declared_preservation = preservation.get("status", "unresolved")
        if declared_preservation == "rejected" and not missing_preservation:
            preservation_status = "reject"
        elif declared_preservation == "proved" and not missing_preservation:
            preservation_status = "pass"
        else:
            preservation_status = "unresolved"
        preservation_certificate = {
            "status": preservation_status,
            "declared_status": declared_preservation,
            "statement": preservation.get("statement"),
            "required_controls": preservation_controls,
            "missing_or_failed_controls": missing_preservation,
        }
        if pointwise_condition_status == "pass" and preservation_status == "unresolved":
            condition_summary["status"] = "unresolved"
        elif preservation_status == "reject":
            condition_summary["status"] = "reject"
    condition_summary["pointwise_status"] = pointwise_condition_status
    condition_summary["background_domain_preservation"] = preservation_certificate
    dirac_pass = dirac_ir.get("status") == "pass"

    common_principal = ["principal_symbol_controls", "curved_background_principal_controls"]
    if family == "einstein_hilbert":
        hamiltonian_controls = ["einstein_hilbert_linearized_adm"]
        principal_controls = common_principal
        generic_hamiltonian_supported = True
    elif family == "canonical_scalar_gravity":
        hamiltonian_controls = ["einstein_hilbert_linearized_adm", "canonical_scalar"]
        principal_controls = common_principal
        generic_hamiltonian_supported = True
    elif family == "proca_gravity":
        hamiltonian_controls = [
            "einstein_hilbert_linearized_adm",
            "proca_adm_dirac",
            "proca_reduced_smeared_constraint_algebra",
        ]
        principal_controls = common_principal
        generic_hamiltonian_supported = True
    elif family == "einstein_aether":
        hamiltonian_controls = [
            "einstein_aether_linearized_physical_energy",
            "einstein_aether_restricted_nonlinear_total_energy",
        ]
        principal_controls = [
            "einstein_aether_reduced_five_mode_principal_domain",
            "einstein_aether_global_tilt_legendre_strata",
            "einstein_aether_covariant_arbitrary_background_hyperbolicity",
        ]
        generic_hamiltonian_supported = False
    elif family == "quartic_horndeski":
        hamiltonian_controls = [
            "quartic_horndeski_timelike_flat_physical_hamiltonian"
        ]
        principal_controls = [
            "quartic_horndeski_timelike_flat_principal_symbol",
            "quartic_horndeski_arbitrary_curvature_scalar_principal",
            "quartic_horndeski_coupled_formulation_hyperbolicity",
            "quartic_horndeski_full_local_principal_extraction",
            "quartic_horndeski_global_timelike_gradient_no_go",
        ]
        generic_hamiltonian_supported = False
    else:
        hamiltonian_controls = []
        principal_controls = []
        generic_hamiltonian_supported = False

    generic_principal_supported = bool(principal_controls) and family != (
        "quartic_horndeski"
    )
    missing_hamiltonian = [name for name in hamiltonian_controls if not controls.get(name, False)]
    missing_principal = [name for name in principal_controls if not controls.get(name, False)]
    base_pass = dirac_pass and pointwise_condition_status == "pass"
    if pointwise_condition_status == "reject":
        hamiltonian_status = "reject"
        principal_status = "reject"
    else:
        hamiltonian_status = (
            "pass"
            if base_pass and generic_hamiltonian_supported and not missing_hamiltonian
            else "unresolved"
        )
        principal_status = (
            "pass"
            if (
                base_pass
                and generic_principal_supported
                and principal_controls
                and not missing_principal
            )
            else "unresolved"
        )
    if (
        condition_summary["status"] == "reject"
        or hamiltonian_status == "reject"
        or principal_status == "reject"
    ):
        status = "reject"
    elif (
        condition_summary["status"] == "pass"
        and hamiltonian_status == "pass"
        and principal_status == "pass"
    ):
        status = "pass"
    else:
        status = "unresolved"
    body = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "promotion_allowed": False,
        "input_action_sha256": action_ir["content_sha256"],
        "input_dirac_ir_sha256": dirac_ir.get("content_sha256"),
        "source_role": action_ir["canonical"]["source_role"],
        "family": family,
        "parameter_domain": domain["canonical"],
        "background_domain": domain.get("background_canonical"),
        "condition_certificate": condition_summary,
        "derived_effective_parameters": {
            name: str(value) for name, value in sorted(derived.items())
        },
        "physical_hamiltonian": {
            "status": hamiltonian_status,
            "required_controls": hamiltonian_controls,
            "missing_or_failed_controls": missing_hamiltonian,
            "scope": (
                "positive gauge/constraint-reduced quadratic Hamiltonian on declared Minkowski "
                "control domains for EH/scalar/Proca"
                if generic_hamiltonian_supported
                else "generic nonlinear Einstein-Aether physical Hamiltonian remains unresolved; only aligned linear and restricted hypersurface-orthogonal maximal-slice total energy are certified"
            ),
        },
        "principal_symbol": {
            "status": principal_status,
            "generic_coupled_background_supported": generic_principal_supported,
            "required_controls": principal_controls,
            "missing_or_failed_controls": missing_principal,
            "characteristic_speed_squared": {
                name: str(value)
                for name, value in sorted(derived.items())
                if "speed_squared" in name
            },
            "scope": (
                "coefficient-domain-bound reduced ghost, gradient, characteristic, and strong-hyperbolicity controls; Einstein-Aether arbitrary-background pass uses the declared nonluminal sufficient-formulation conditions"
            ),
            "coupled_formulation": (
                {
                    "generalized_harmonic": "reject",
                    "modified_harmonic_weak_coupling": "unresolved",
                    "missing": [
                        "uniform background-jet bounds satisfying the exact Frobenius time-block condition",
                        "a positive symmetrizer for the extracted 22-by-22 generalized first-order pencil",
                        "uniform symmetrizer-induced correction-norm bound over the background and direction sphere",
                        "uniform separation of physical characteristics from both auxiliary cones",
                    ],
                }
                if family == "quartic_horndeski"
                else {"status": "not_applicable"}
            ),
        },
        "proof_scope": (
            "machine-checked implication from the frozen parameter domain to the actual effective "
            "coefficient inequalities, then binding to executable reduced Hamiltonian and principal controls"
        ),
    }
    content = _canonical_json(body)
    return {**body, "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest()}


def write_stability_ir(stability_ir: dict[str, Any], output: str | Path) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(stability_ir, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path
