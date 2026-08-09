from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import sympy as sp

from .dhost import dhost_reduced_dirac_control
from .scalar_tensor_pack import parse_dimensionless_expression

SCHEMA_VERSION = "sigma-reduced-dhost-pack-1.0"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _mutation_axes(
    raw_axes: Any, coefficients: set[str], errors: list[str]
) -> tuple[int, list[dict[str, Any]]]:
    if not isinstance(raw_axes, list):
        errors.append("mutation_axes must be a list")
        return 0, []
    cardinality = 1
    axes: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_axes):
        if not isinstance(raw, dict):
            errors.append(f"mutation_axes[{index}] must be an object")
            continue
        coefficient = str(raw.get("coefficient", ""))
        if coefficient not in coefficients:
            errors.append(f"mutation axis uses undeclared coefficient {coefficient!r}")
        if coefficient in seen:
            errors.append(f"duplicate mutation axis for {coefficient}")
        seen.add(coefficient)
        values = raw.get("values", [])
        if not isinstance(values, list) or not values:
            errors.append(f"mutation axis {coefficient} requires values")
            continue
        parsed: list[str] = []
        for value in values:
            expression = sp.sympify(str(value), evaluate=True)
            if expression.free_symbols or expression.is_real is not True:
                errors.append(f"mutation value for {coefficient} must be an exact real constant")
            else:
                parsed.append(str(expression))
        cardinality *= len(parsed)
        axes.append({"coefficient": coefficient, "values": parsed})
    return cardinality, axes


def compile_reduced_dhost_pack(spec: dict[str, Any]) -> dict[str, Any]:
    """Derive a rank-one two-velocity DHOST kinetic family before enumeration.

    The reduced Hessian is [[a,b],[b,c]]. On the regular a!=0 branch, c=b^2/a is generated rather
    than proposed independently. This proves one local primary null direction; full covariant DHOST
    classification and distributed constraint closure remain separate.
    """

    errors: list[str] = []
    if spec.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}")
    raw_coefficients = spec.get("coefficients", [])
    if not isinstance(raw_coefficients, list) or any(
        not isinstance(item, str) or not item.isidentifier() for item in raw_coefficients
    ):
        errors.append("coefficients must be a list of valid identifiers")
        raw_coefficients = []
    coefficients = set(raw_coefficients)
    if len(coefficients) != len(raw_coefficients):
        errors.append("coefficients must be unique")

    kinetic = spec.get("kinetic", {})
    if not isinstance(kinetic, dict):
        errors.append("kinetic must be an object")
        kinetic = {}
    if kinetic.get("solve_for") != "c":
        errors.append("the first pack version requires solve_for='c' on the regular a!=0 branch")
    expressions: dict[str, sp.Expr] = {}
    for name in ("a", "b"):
        raw = kinetic.get(name)
        if not isinstance(raw, str):
            errors.append(f"kinetic.{name} must be a string")
            continue
        try:
            expressions[name] = sp.factor(
                parse_dimensionless_expression(raw, coefficients)
            )
        except (TypeError, ValueError, sp.SympifyError) as error:
            errors.append(f"kinetic.{name}: {error}")

    a = expressions.get("a", sp.Symbol("invalid_a"))
    b = expressions.get("b", sp.Symbol("invalid_b"))
    if a == 0:
        errors.append("the solve-for-c branch requires a!=0")
    c = sp.factor(sp.cancel(b**2 / a))
    determinant = sp.factor(sp.cancel(a * c - b**2))
    hessian = sp.Matrix([[a, b], [b, c]])
    null_vector = sp.Matrix([-b, a])
    null_residual = [sp.factor(sp.cancel(item)) for item in hessian * null_vector]

    override = kinetic.get("c_override")
    override_residual: sp.Expr | None = None
    if override is not None:
        if not isinstance(override, str):
            errors.append("kinetic.c_override must be a string")
        else:
            try:
                proposed = parse_dimensionless_expression(override, coefficients)
                override_residual = sp.factor(sp.cancel(proposed - c))
                if override_residual != 0:
                    errors.append("kinetic.c_override violates the generated degeneracy relation")
            except (TypeError, ValueError, sp.SympifyError) as error:
                errors.append(f"kinetic.c_override: {error}")

    cardinality, axes = _mutation_axes(spec.get("mutation_axes", []), coefficients, errors)
    dirac = dhost_reduced_dirac_control()
    passed = (
        not errors
        and determinant == 0
        and all(item == 0 for item in null_residual)
        and dirac["passed"]
    )
    body = {
        "schema_version": "sigma-reduced-dhost-pack-ir-1.0",
        "source_schema_version": SCHEMA_VERSION,
        "status": "compiled_full_covariant_adapters_unresolved" if passed else "reject",
        "errors": errors,
        "kinetic_hessian": [[str(a), str(b)], [str(b), str(c)]],
        "generated_coefficients": {"a": str(a), "b": str(b), "c": str(c)},
        "degeneracy_relation": "c=b^2/a on a!=0",
        "determinant_residual": str(determinant),
        "regular_branch_guard": f"{a} != 0",
        "hessian_null_vector": [str(item) for item in null_vector],
        "null_vector_residual": [str(item) for item in null_residual],
        "primary_constraint": f"-({b}) p_V + ({a}) p_K = 0",
        "c_override_residual": None if override_residual is None else str(override_residual),
        "mutation_space": {
            "axes": axes,
            "declared_cardinality": cardinality,
            "log10_cardinality": (
                None if cardinality <= 0 else 0.0 if cardinality == 1 else math.log10(cardinality)
            ),
            "enumerated": False,
        },
        "known_dirac_mechanism_control": {
            "passed": dirac["passed"],
            "primary_constraint": dirac["primary_constraint"],
            "secondary_constraint": dirac["secondary_constraint"],
            "constraint_matrix_rank": dirac["constraint_matrix_rank"],
            "physical_scalar_dof": dirac["physical_scalar_dof"],
            "reduced_hamiltonian": dirac["reduced_hamiltonian"],
        },
        "capability_status": {
            "reduced_adm_kinetic_degeneracy": "pass" if passed else "reject",
            "primary_null_direction": "pass" if passed else "reject",
            "generic_secondary_constraint": "unresolved",
            "full_covariant_dhost_classification": "unresolved",
            "distributed_gravitational_constraint_algebra": "unresolved",
            "generic_reduced_hamiltonian": "unresolved",
            "generic_principal_symbol": "unresolved",
            "observations": "sealed",
        },
        "scope": (
            "exact rank-one two-velocity reduced ADM kinetic family on the regular a!=0 branch; "
            "the embedded known control proves the primary-secondary mechanism for one potential, "
            "not for every generated covariant theory"
        ),
        "primary_source": "https://arxiv.org/abs/1512.06820",
    }
    canonical = _canonical_json(body)
    return {**body, "content_sha256": hashlib.sha256(canonical.encode()).hexdigest()}
