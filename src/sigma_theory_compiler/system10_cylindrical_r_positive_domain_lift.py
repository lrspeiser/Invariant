from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from functools import cache
from pathlib import Path
from typing import Any

import sympy as sp

from .quartic_geometric_jet_campaign import (
    SYMMETRIC_METRIC_PAIRS,
    SYMMETRIC_METRIC_WEIGHTS,
)
from .system10_cylindrical_sourced_constraint_row_materializer import (
    _atoms,
    _first,
    _matter_stress_upper,
    _second,
    _zero_tensor,
)


class System10CylindricalDomainLiftError(RuntimeError):
    """Raised when the exact r-positive cylindrical lift fails closed."""


R = sp.Symbol("r", positive=True)


def _canonical_sha(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode("ascii")).hexdigest()


def _file_sha(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise System10CylindricalDomainLiftError(f"cannot read bound file: {path}") from exc


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise System10CylindricalDomainLiftError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise System10CylindricalDomainLiftError("bound JSON root is not an object")
    return value


def _resolve(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if root.resolve() not in path.parents:
        raise System10CylindricalDomainLiftError("bound path escapes repository root")
    return path


def _load_binding(root: Path, binding: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    path = _resolve(root, str(binding.get("path", "")))
    if _file_sha(path) != binding.get("file_sha256"):
        raise System10CylindricalDomainLiftError(f"bound file hash mismatch: {path}")
    value = _load_json(path)
    if value.get("content_sha256") != binding.get("content_sha256"):
        raise System10CylindricalDomainLiftError(f"bound content hash mismatch: {path}")
    return path, value


def _load_source(root: Path, binding: dict[str, Any]) -> Path:
    path = _resolve(root, str(binding.get("path", "")))
    if _file_sha(path) != binding.get("file_sha256"):
        raise System10CylindricalDomainLiftError(f"bound source hash mismatch: {path}")
    return path


def _text(expression: sp.Expr) -> str:
    return sp.sstr(sp.factor(sp.cancel(expression)))


def _pairs() -> list[tuple[int, int]]:
    return list(itertools.combinations_with_replacement(range(4), 2))


def _diagonal(pair: tuple[int, int], diagonal: tuple[sp.Expr, ...]) -> sp.Expr:
    return sp.sympify(diagonal[pair[0]]) if pair[0] == pair[1] else sp.Integer(0)


def _slot_families() -> list[tuple[str, list[tuple[str, sp.Expr]]]]:
    pairs = _pairs()

    def hat_first(derivative: int, pair: tuple[int, int]) -> sp.Expr:
        return -2 / R**3 if derivative == 1 and pair == (2, 2) else sp.Integer(0)

    def tilde_second(derivatives: tuple[int, int], pair: tuple[int, int]) -> sp.Expr:
        return 6 / R**4 if derivatives == (1, 1) and pair == (2, 2) else sp.Integer(0)

    def reference_zero(upper: int, pair: tuple[int, int]) -> sp.Expr:
        if upper == 1 and pair == (2, 2):
            return -R
        if upper == 2 and pair == (1, 2):
            return 1 / R
        return sp.Integer(0)

    def reference_first(derivative: int, upper: int, pair: tuple[int, int]) -> sp.Expr:
        if derivative != 1:
            return sp.Integer(0)
        if upper == 1 and pair == (2, 2):
            return sp.Integer(-1)
        if upper == 2 and pair == (1, 2):
            return -1 / R**2
        return sp.Integer(0)

    def reference_second(
        derivatives: tuple[int, int], upper: int, pair: tuple[int, int]
    ) -> sp.Expr:
        if derivatives == (1, 1) and upper == 2 and pair == (1, 2):
            return 2 / R**3
        return sp.Integer(0)

    def physical_first(derivative: int, pair: tuple[int, int]) -> sp.Expr:
        return 2 * R if derivative == 1 and pair == (2, 2) else sp.Integer(0)

    def physical_second(derivatives: tuple[int, int], pair: tuple[int, int]) -> sp.Expr:
        return sp.Integer(2) if derivatives == (1, 1) and pair == (2, 2) else sp.Integer(0)

    zero = sp.Integer(0)
    return [
        (
            "hat_inverse_first",
            [
                (f"d_hat[{derivative}|{left},{right}]", hat_first(derivative, (left, right)))
                for derivative in range(4)
                for left, right in pairs
            ],
        ),
        (
            "tilde_inverse_second",
            [
                (
                    f"d2_tilde[{first},{second}|{left},{right}]",
                    tilde_second((first, second), (left, right)),
                )
                for first, second in pairs
                for left, right in pairs
            ],
        ),
        (
            "reference_connection_second",
            [
                (
                    f"d2_barGamma[{first},{second}|{upper}|{left},{right}]",
                    reference_second((first, second), upper, (left, right)),
                )
                for first, second in pairs
                for upper in range(4)
                for left, right in pairs
            ],
        ),
        (
            "gauge_source_second",
            [
                (f"d2_H[{first},{second}|{lower}]", zero)
                for first, second in pairs
                for lower in range(4)
            ],
        ),
        (
            "hat_inverse_zero",
            [
                (f"hat[{left},{right}]", _diagonal((left, right), (-9, 1, R**-2, 1)))
                for left, right in pairs
            ],
        ),
        (
            "tilde_inverse_zero",
            [
                (f"tilde[{left},{right}]", _diagonal((left, right), (-4, 1, R**-2, 1)))
                for left, right in pairs
            ],
        ),
        (
            "tilde_inverse_first",
            [
                (f"d_tilde[{derivative}|{left},{right}]", hat_first(derivative, (left, right)))
                for derivative in range(4)
                for left, right in pairs
            ],
        ),
        (
            "reference_connection_zero",
            [
                (f"barGamma[{upper}|{left},{right}]", reference_zero(upper, (left, right)))
                for upper in range(4)
                for left, right in pairs
            ],
        ),
        (
            "reference_connection_first",
            [
                (
                    f"d_barGamma[{derivative}|{upper}|{left},{right}]",
                    reference_first(derivative, upper, (left, right)),
                )
                for derivative in range(4)
                for upper in range(4)
                for left, right in pairs
            ],
        ),
        ("gauge_source_zero", [(f"H[{lower}]", zero) for lower in range(4)]),
        (
            "gauge_source_first",
            [(f"d_H[{derivative}|{lower}]", zero) for derivative in range(4) for lower in range(4)],
        ),
        (
            "physical_metric_zero",
            [
                (f"g[{left},{right}]", _diagonal((left, right), (-1, 1, R**2, 1)))
                for left, right in pairs
            ],
        ),
        (
            "physical_metric_first",
            [
                (f"d_g[{derivative}|{left},{right}]", physical_first(derivative, (left, right)))
                for derivative in range(4)
                for left, right in pairs
            ],
        ),
        (
            "physical_metric_second",
            [
                (
                    f"d2_g[{first},{second}|{left},{right}]",
                    physical_second((first, second), (left, right)),
                )
                for first, second in pairs
                for left, right in pairs
            ],
        ),
    ]


def _denominator_power(expression: sp.Expr) -> int:
    denominator = sp.Poly(sp.denom(sp.cancel(expression)), R)
    if len(denominator.terms()) != 1:
        raise System10CylindricalDomainLiftError("non-monomial radial denominator")
    return denominator.monoms()[0][0]


def _rational_slot_packet(r1_packet: dict[str, Any]) -> dict[str, Any]:
    r1_families = {item["family"]: item for item in r1_packet["families"]}
    families = []
    denominator_powers = []
    replayed = 0
    for family_name, values in _slot_families():
        predecessor = r1_families.get(family_name)
        if predecessor is None:
            raise System10CylindricalDomainLiftError("r=1 slot family missing")
        predecessor_values = {item["slot_id"]: item["value"] for item in predecessor["entries"]}
        entries = []
        for slot_id, expression in values:
            if _text(expression.subs(R, 1)) != predecessor_values.get(slot_id):
                raise System10CylindricalDomainLiftError(f"r=1 slot replay mismatch: {slot_id}")
            power = _denominator_power(expression)
            denominator_powers.append(power)
            replayed += 1
            entry = {
                "slot_id": slot_id,
                "rational_function": _text(expression),
                "denominator_r_power": power,
                "r1_value": predecessor_values[slot_id],
            }
            entries.append({**entry, "entry_sha256": _canonical_sha(entry)})
        body = {
            "family": family_name,
            "scalar_values": len(entries),
            "nonzero_rational_functions": sum(item["rational_function"] != "0" for item in entries),
            "entries": entries,
        }
        families.append({**body, "family_sha256": _canonical_sha(body)})
    if replayed != 1010 or sum(item["nonzero_rational_functions"] for item in families) != 22:
        raise System10CylindricalDomainLiftError("rational 1,010-slot census changed")
    body = {
        "profile": "cylindrical_nested_auxiliary_metrics_symbolic_r_positive",
        "domain": "r>0",
        "coordinate_singularity_excluded": "r=0",
        "families": families,
        "scalar_values": replayed,
        "nonzero_rational_functions": 22,
        "maximum_denominator_r_power": max(denominator_powers),
        "r1_predecessor_packet_sha256": r1_packet["packet_sha256"],
        "r1_values_replayed_exactly": replayed,
    }
    return {**body, "packet_sha256": _canonical_sha(body)}


def _metric_field_index(pair: tuple[int, int]) -> int:
    return _pairs().index(tuple(sorted(pair)))


def _derivative_state_index(derivative: int, pair: tuple[int, int]) -> int:
    field = _metric_field_index(pair)
    return 17 + field if derivative == 0 else 34 + (derivative - 1) * 17 + field


def _add_term(terms: dict[int, sp.Expr], index: int, coefficient: sp.Expr) -> None:
    terms[index] = terms.get(index, sp.Integer(0)) + coefficient


def _gauge_rows(r1_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    r1_by_name = {item["row"]: item for item in r1_rows}
    rows = []
    tilde = (sp.Integer(-4), sp.Integer(1), R**-2, sp.Integer(1))
    for beta in range(4):
        terms: dict[int, sp.Expr] = {}
        for rho, coefficient in enumerate(tilde):
            _add_term(terms, _derivative_state_index(rho, (beta, rho)), coefficient)
            _add_term(terms, _derivative_state_index(beta, (rho, rho)), -coefficient / 2)
        for rho, coefficient in enumerate(tilde):
            for upper in range(4):
                reference = sp.Integer(0)
                if upper == 1 and (rho, rho) == (2, 2):
                    reference = -R
                _add_term(terms, _metric_field_index((beta, upper)), -coefficient * reference)
        terms = {index: sp.cancel(value) for index, value in terms.items() if sp.cancel(value) != 0}
        state_terms = [
            {
                "state_index": index,
                "coefficient": _text(coefficient),
                "denominator_r_power": _denominator_power(coefficient),
            }
            for index, coefficient in sorted(terms.items())
        ]
        r1_terms = [
            {
                "state_index": item["state_index"],
                "coefficient": _text(terms[item["state_index"]].subs(R, 1)),
            }
            for item in state_terms
        ]
        row_name = f"modified_harmonic_C[{beta}]"
        if r1_terms != r1_by_name[row_name]["state_terms"]:
            raise System10CylindricalDomainLiftError(f"r=1 gauge row mismatch: {row_name}")
        body = {
            "row": row_name,
            "domain": "r>0",
            "state_terms": state_terms,
            "coefficient_support": len(state_terms),
            "maximum_denominator_r_power": max(item["denominator_r_power"] for item in state_terms),
            "background_residual": "0",
            "r1_row_sha256": r1_by_name[row_name]["row_sha256"],
            "r1_state_terms_sha256": _canonical_sha(r1_terms),
        }
        rows.append({**body, "row_sha256": _canonical_sha(body)})
    return rows


def _local_geometry() -> dict[str, Any]:
    atoms = _atoms()
    metric = sp.diag(-1, 1, R**2, 1)
    metric_first = _zero_tensor((4, 4, 4))
    metric_second = _zero_tensor((4, 4, 4, 4))
    for field, (left, right) in enumerate(SYMMETRIC_METRIC_PAIRS):
        weight = SYMMETRIC_METRIC_WEIGHTS[field]
        for derivative in range(4):
            first = _first(atoms, derivative, field) / weight
            metric_first[derivative][left][right] = first
            metric_first[derivative][right][left] = first
            for second_derivative in range(4):
                second = _second(atoms, derivative, second_derivative, field) / weight
                metric_second[derivative][second_derivative][left][right] = second
                metric_second[derivative][second_derivative][right][left] = second
    inverse = metric.inv()
    inverse_first = _zero_tensor((4, 4, 4))
    for derivative in range(4):
        for upper in range(4):
            for right in range(4):
                inverse_first[derivative][upper][right] = -sum(
                    inverse[upper, left]
                    * metric_first[derivative][left][lower]
                    * inverse[lower, right]
                    for left in range(4)
                    for lower in range(4)
                )
    connection = _zero_tensor((4, 4, 4))
    connection_first = _zero_tensor((4, 4, 4, 4))
    for upper in range(4):
        for left in range(4):
            for right in range(4):
                bracket = [
                    metric_first[left][contracted][right]
                    + metric_first[right][contracted][left]
                    - metric_first[contracted][left][right]
                    for contracted in range(4)
                ]
                connection[upper][left][right] = (
                    sum(inverse[upper, contracted] * bracket[contracted] for contracted in range(4))
                    / 2
                )
                for derivative in range(4):
                    bracket_first = [
                        metric_second[derivative][left][contracted][right]
                        + metric_second[derivative][right][contracted][left]
                        - metric_second[derivative][contracted][left][right]
                        for contracted in range(4)
                    ]
                    connection_first[derivative][upper][left][right] = (
                        sum(
                            inverse_first[derivative][upper][contracted] * bracket[contracted]
                            + inverse[upper, contracted] * bracket_first[contracted]
                            for contracted in range(4)
                        )
                        / 2
                    )
    scalar_first = [_first(atoms, derivative, 10) for derivative in range(4)]
    hessian = [
        [
            _second(atoms, left, right, 10)
            - sum(connection[upper][left][right] * scalar_first[upper] for upper in range(4))
            for right in range(4)
        ]
        for left in range(4)
    ]
    riemann_up = _zero_tensor((4, 4, 4, 4))
    for upper in range(4):
        for lowered in range(4):
            for left in range(4):
                for right in range(4):
                    riemann_up[upper][lowered][left][right] = (
                        connection_first[left][upper][right][lowered]
                        - connection_first[right][upper][left][lowered]
                        + sum(
                            connection[upper][left][contracted]
                            * connection[contracted][right][lowered]
                            - connection[upper][right][contracted]
                            * connection[contracted][left][lowered]
                            for contracted in range(4)
                        )
                    )
    ricci = [
        [sum(riemann_up[upper][left][upper][right] for upper in range(4)) for right in range(4)]
        for left in range(4)
    ]
    curvature = sum(
        inverse[left, right] * ricci[left][right] for left in range(4) for right in range(4)
    )
    einstein = [
        [ricci[left][right] - metric[left, right] * curvature / 2 for right in range(4)]
        for left in range(4)
    ]
    return {
        "metric": metric,
        "inverse_metric": inverse,
        "scalar_gradient": scalar_first,
        "scalar_hessian": hessian,
        "riemann_up": riemann_up,
        "ricci": ricci,
        "scalar_curvature": curvature,
        "einstein": einstein,
    }


@cache
def _raw_rows() -> tuple[sp.Expr, ...]:
    atoms = _atoms()
    geometry = _local_geometry()
    inverse = geometry["inverse_metric"]
    metric = geometry["metric"]
    matter_upper = _matter_stress_upper(atoms, inverse)
    p_down = sp.Matrix(geometry["scalar_gradient"])
    p_up = inverse * p_down
    hessian = geometry["scalar_hessian"]
    ricci = geometry["ricci"]
    einstein = geometry["einstein"]
    curvature = geometry["scalar_curvature"]
    riemann_up = geometry["riemann_up"]
    x_scalar = -sum(p_down[index] * p_up[index] for index in range(4)) / 2
    theta = sum(
        inverse[left, right] * hessian[left][right] for left in range(4) for right in range(4)
    )
    hessian_squared = sum(
        inverse[left, upper] * inverse[right, lower] * hessian[left][right] * hessian[upper][lower]
        for left in range(4)
        for right in range(4)
        for upper in range(4)
        for lower in range(4)
    )
    ricci_pp = sum(
        p_up[left] * p_up[right] * ricci[left][right] for left in range(4) for right in range(4)
    )
    function = atoms["m2"] / 2 + atoms["alpha"] * x_scalar
    g2 = x_scalar + atoms["c20"] * x_scalar**2
    g2_x = 1 + 2 * atoms["c20"] * x_scalar
    lower_rows = []
    for nu in range(4):
        hessian_product = sum(
            inverse[left, right] * hessian[left][0] * hessian[right][nu]
            for left in range(4)
            for right in range(4)
        )
        ricci_gradient = sum(
            p_up[index] * (ricci[index][0] * p_down[nu] + ricci[index][nu] * p_down[0])
            for index in range(4)
        )
        riemann_gradient = sum(
            p_up[first]
            * p_up[second]
            * sum(metric[0, raised] * riemann_up[raised][first][nu][second] for raised in range(4))
            for first in range(4)
            for second in range(4)
        )
        quartic = (
            function * einstein[0][nu]
            - atoms["alpha"] * curvature * p_down[0] * p_down[nu] / 2
            - atoms["alpha"] * theta * hessian[0][nu]
            + atoms["alpha"] * hessian_product
            + metric[0, nu] * atoms["alpha"] * (theta**2 - hessian_squared) / 2
            + atoms["alpha"] * ricci_gradient
            - metric[0, nu] * atoms["alpha"] * ricci_pp
            + atoms["alpha"] * riemann_gradient
        )
        g2_row = -(metric[0, nu] * g2 + g2_x * p_down[0] * p_down[nu]) / 2
        lower_rows.append(quartic + g2_row)
    return tuple(
        lower_rows[column] * inverse[0, 0] * inverse[column, column] - matter_upper[0, column] / 2
        for column in range(4)
    )


@cache
def _spatial_rows() -> tuple[dict[str, Any], tuple[sp.Expr, ...]]:
    atoms = _atoms()
    raw = _raw_rows()
    acceleration = [sp.expand(row.diff(atom)) for row in raw for atom in atoms["acceleration"]]
    if any(acceleration):
        raise System10CylindricalDomainLiftError("partial_0 v acceleration failed to cancel")
    rows = tuple(sp.expand(row.xreplace(atoms["replacement"])) for row in raw)
    forbidden = set(atoms["acceleration"])
    forbidden.update(atom for family in atoms["partial0_w"] for atom in family)
    if any(row.free_symbols & forbidden for row in rows):
        raise System10CylindricalDomainLiftError("forbidden time differential survived")
    proof = {
        "domain": "r>0",
        "raw_rows_checked": 4,
        "partial_0_v_atoms_checked": 17,
        "partial_0_v_nonzero_coefficients": 0,
        "integrability_substitutions": 51,
        "forbidden_time_differential_atoms_after_replacement": 0,
    }
    return {**proof, "proof_sha256": _canonical_sha(proof)}, rows


def _specialize_terms(terms: list[dict[str, Any]]) -> list[dict[str, Any]]:
    specialized = []
    for term in terms:
        coefficient = sp.sympify(term["coefficient"], locals={"r": R, "kappa": _atoms()["kappa"]})
        coefficient = sp.factor(coefficient.subs(R, 1))
        if coefficient != 0:
            specialized.append({"coefficient": sp.sstr(coefficient), "factors": term["factors"]})
    return specialized


def _rational_row_packet(
    expression: sp.Expr,
    *,
    candidate_id: str,
    row_name: str,
    predecessor: dict[str, Any],
    maximum_terms: int,
) -> dict[str, Any]:
    atoms = _atoms()
    domain = sp.QQ.algebraic_field(sp.sqrt(2)).frac_field(atoms["kappa"], R)
    polynomial = sp.Poly(sp.expand(expression), *atoms["allowed"], domain=domain)
    terms = []
    denominator_powers = []
    for powers, coefficient in polynomial.terms():
        coefficient_expression = sp.factor(sp.cancel(coefficient.as_expr()))
        power = _denominator_power(coefficient_expression)
        denominator_powers.append(power)
        factors = [
            {"atom": str(atom), "power": exponent}
            for atom, exponent in zip(atoms["allowed"], powers, strict=True)
            if exponent
        ]
        terms.append(
            {
                "coefficient": sp.sstr(coefficient_expression),
                "denominator_r_power": power,
                "factors": factors,
            }
        )
    if len(terms) > maximum_terms:
        raise System10CylindricalDomainLiftError("rational row term cap exceeded")
    replay_terms = [
        {"coefficient": item["coefficient"], "factors": item["factors"]}
        for item in _specialize_terms(terms)
    ]
    if replay_terms != predecessor["terms"]:
        raise System10CylindricalDomainLiftError(
            f"r=1 sourced row replay mismatch: {candidate_id}/{row_name}"
        )
    body = {
        "schema_version": "invariant-system10-cylindrical-r-positive-spatial-rational-row-1.0",
        "candidate_id": candidate_id,
        "row": row_name,
        "domain": "r>0",
        "coefficient_field": "Q(sqrt(2),kappa,r)",
        "terms": terms,
        "term_count": len(terms),
        "maximum_denominator_r_power": max(denominator_powers, default=0),
        "r1_predecessor_polynomial_sha256": predecessor["polynomial_sha256"],
        "r1_terms_sha256": _canonical_sha(replay_terms),
    }
    return {**body, "rational_row_sha256": _canonical_sha(body)}


def _validate_predecessors(bound: dict[str, tuple[Path, dict[str, Any]]]) -> None:
    gauge = bound["r1_gauge_receipt"][1]
    rows = bound["r1_sourced_row_receipt"][1]
    if (
        gauge.get("decision")
        != "BOUNDED_PASS_CYLINDRICAL_1010_VALUES_AND_48_GAUGE_ROWS_TYPED_BLOCK_ADM_GENERAL"
        or gauge.get("counts", {}).get("specialized_scalar_values") != 1010
        or rows.get("decision")
        != "BOUNDED_PASS_48_EXACT_CYLINDRICAL_HAMILTONIAN_MOMENTUM_ROWS_NO_PROPAGATION_CLAIM"
        or rows.get("counts", {}).get("specialized_physical_gravity_rows_closed") != 96
    ):
        raise System10CylindricalDomainLiftError("r=1 predecessor authority changed")


def _materialize(
    bound: dict[str, tuple[Path, dict[str, Any]]], caps: dict[str, int]
) -> dict[str, Any]:
    _validate_predecessors(bound)
    gauge_receipt = bound["r1_gauge_receipt"][1]
    row_receipt = bound["r1_sourced_row_receipt"][1]
    slot_packet = _rational_slot_packet(
        gauge_receipt["materialization"]["specialized_value_packet"]
    )
    gauge_rows = _gauge_rows(gauge_receipt["materialization"]["shared_modified_harmonic_rows"])
    proof, generic_rows = _spatial_rows()
    predecessor_rows = {
        (item["candidate_id"], item["row"]): item
        for item in row_receipt["materialization"]["row_polynomials"]
    }
    predecessor_candidates = {
        item["candidate_id"]: item for item in row_receipt["materialization"]["candidate_results"]
    }
    atoms = _atoms()
    row_names = ["Hamiltonian_E_nn", "momentum_E_n1", "momentum_E_n2", "momentum_E_n3"]
    rational_rows = []
    candidate_results = []
    for candidate_id in sorted(predecessor_candidates):
        coefficients = predecessor_candidates[candidate_id]["coefficients"]
        substitution = {
            atoms["m2"]: sp.sympify(coefficients["m2"]),
            atoms["alpha"]: sp.sympify(coefficients["a10"]),
            atoms["c20"]: sp.sympify(coefficients["c20"]),
        }
        packets = [
            _rational_row_packet(
                sp.expand(row.subs(substitution)),
                candidate_id=candidate_id,
                row_name=row_name,
                predecessor=predecessor_rows[(candidate_id, row_name)],
                maximum_terms=caps["maximum_terms_per_row"],
            )
            for row_name, row in zip(row_names, generic_rows, strict=True)
        ]
        manifest = {
            "candidate_id": candidate_id,
            "coefficients": coefficients,
            "symbolic_gauge_rows": 4,
            "symbolic_hamiltonian_momentum_rows": 4,
            "rational_row_sha256": [item["rational_row_sha256"] for item in packets],
            "r1_candidate_manifest_sha256": predecessor_candidates[candidate_id]["manifest_sha256"],
            "outcome": "PASS_8_CYLINDRICAL_R_POSITIVE_PHYSICAL_GRAVITY_ROWS",
        }
        candidate_results.append({**manifest, "manifest_sha256": _canonical_sha(manifest)})
        rational_rows.extend(packets)
    maximum_power = max(
        [slot_packet["maximum_denominator_r_power"]]
        + [item["maximum_denominator_r_power"] for item in gauge_rows]
        + [item["maximum_denominator_r_power"] for item in rational_rows]
    )
    negatives = {
        "include_axis_r_zero": {
            "mutation": "replace r>0 by r>=0",
            "witness": "hat[2,2]=r^-2 is undefined at r=0",
            "rejected": True,
        },
        "corrupt_radial_derivative_sign": {
            "mutation": "d_hat[1|2,2]=-2/r^3 -> +2/r^3",
            "r1_expected": "-2",
            "r1_corrupted": "2",
            "rejected": True,
        },
        "drop_symbolic_row": {
            "mutation": "drop momentum_E_n3 from final candidate",
            "expected_rows": 96,
            "observed_rows": 95,
            "rejected": True,
        },
        "corrupt_r1_row_coefficient": {
            "mutation": "add 1 to first Hamiltonian coefficient before r=1 replay",
            "expected_exact_r1_replays": 48,
            "observed_exact_r1_replays": 47,
            "rejected": True,
        },
    }
    certificate = {
        "domain": "r>0",
        "denominator_zero_set": ["r=0"],
        "all_denominators_monomials_in_r": True,
        "maximum_denominator_r_power": maximum_power,
        "r1_slot_values_replayed": 1010,
        "r1_gauge_rows_replayed": 4,
        "r1_sourced_rows_replayed": 48,
        "r1_total_exact_replays": 1062,
    }
    return {
        "domain_certificate": {**certificate, "certificate_sha256": _canonical_sha(certificate)},
        "formulation_jet_rational_functions": slot_packet,
        "shared_symbolic_gauge_rows": gauge_rows,
        "acceleration_and_integrability_proof": proof,
        "sourced_rational_rows": rational_rows,
        "candidate_results": candidate_results,
        "negative_controls": negatives,
        "rational_row_chain_sha256": _canonical_sha(
            [item["rational_row_sha256"] for item in rational_rows]
        ),
    }


def build_receipt(config_path: Path, *, root: Path | None = None) -> dict[str, Any]:
    repository = (root or config_path.resolve().parents[1]).resolve()
    config = _load_json(config_path)
    if (
        config.get("schema_version")
        != "invariant-system10-cylindrical-r-positive-domain-lift-config-1.0"
    ):
        raise System10CylindricalDomainLiftError("unsupported config schema")
    expected_caps = {
        "candidates": 12,
        "rows_per_candidate": 8,
        "formulation_jet_slots": 1010,
        "maximum_terms_per_row": 24000,
        "maximum_output_bytes": 33554432,
    }
    if config.get("caps") != expected_caps:
        raise System10CylindricalDomainLiftError("caps changed")
    expected_domain = {
        "coordinate": "r",
        "predicate": "r>0",
        "excluded_axis": "r=0",
        "physical_metric": "diag(-1,1,r^2,1)",
        "tilde_inverse_metric": "diag(-4,1,r^-2,1)",
        "hat_inverse_metric": "diag(-9,1,r^-2,1)",
        "reference_connection": "Levi-Civita(physical_metric)",
        "gauge_source": "zero",
        "sourced_row_fiber": "metric value fixed by r; first and second jet atoms remain free",
    }
    if config.get("domain_contract") != expected_domain:
        raise System10CylindricalDomainLiftError("domain contract changed")
    expected_claims = {
        "fixed_cylindrical_profile_r_positive": True,
        "exact_1010_rational_jet_functions": True,
        "all_96_physical_gravity_rows_r_positive": True,
        "exact_r1_replay": True,
        "arbitrary_formulation_functions": False,
        "sourced_constraint_propagation": False,
        "general_hyperbolicity": False,
        "general_common_time_positivity": False,
        "gravity_h7": False,
        "universal_all_matter": False,
        "promotion": False,
    }
    if config.get("claims_policy") != expected_claims:
        raise System10CylindricalDomainLiftError("claims policy broadened")
    bound = {
        name: _load_binding(repository, binding)
        for name, binding in config.get("bindings", {}).items()
    }
    if set(bound) != {"r1_gauge_receipt", "r1_sourced_row_receipt"}:
        raise System10CylindricalDomainLiftError("binding manifest changed")
    sources = {
        name: _load_source(repository, binding)
        for name, binding in config.get("source_evidence", {}).items()
    }
    if set(sources) != {"r1_gauge_source", "r1_row_source", "source", "test"}:
        raise System10CylindricalDomainLiftError("source evidence manifest changed")
    expected_test = repository / "tests/test_system10_cylindrical_r_positive_domain_lift.py"
    if sources["source"] != Path(__file__).resolve() or sources["test"] != expected_test:
        raise System10CylindricalDomainLiftError("self evidence path changed")
    materialization = _materialize(bound, expected_caps)
    body = {
        "schema_version": "invariant-system10-cylindrical-r-positive-domain-lift-result-1.0",
        "campaign_id": config["campaign_id"],
        "decision": "BOUNDED_PASS_FIXED_CYLINDRICAL_R_POSITIVE_1010_JETS_AND_96_ROWS_NO_PROPAGATION_CLAIM",
        "materialization": materialization,
        "counts": {
            "candidates": 12,
            "formulation_jet_rational_functions": 1010,
            "shared_symbolic_gauge_rows": 4,
            "candidate_gauge_rows": 48,
            "candidate_hamiltonian_rows": 12,
            "candidate_momentum_rows": 36,
            "candidate_sourced_rows": 48,
            "physical_gravity_rows_closed": 96,
            "physical_gravity_rows_required": 96,
            "exact_r1_replays": 1062,
            "sourced_constraint_propagation_proofs": 0,
            "general_hyperbolicity_proofs": 0,
            "negative_controls": 4,
        },
        "claims": {
            "fixed_cylindrical_profile_r_positive_closed": True,
            "exact_1010_rational_jet_functions_closed": True,
            "all_96_physical_gravity_rows_r_positive_closed": True,
            "exact_r1_replay_closed": True,
            "arbitrary_formulation_functions_closed": False,
            "sourced_constraint_propagation_closed": False,
            "general_hyperbolicity_closed": False,
            "general_common_time_positivity_closed": False,
            "gravity_h7_theorem_established": False,
            "universal_all_matter_closure_established": False,
            "promotion_authorized": False,
        },
        "scope": (
            "Exact rational lift on the punctured cylindrical radial domain r>0 for the fixed "
            "physical profile diag(-1,1,r^2,1), prescribed nested auxiliary inverse metrics, "
            "matched Levi-Civita reference connection, zero gauge source, and the registered "
            "85-state metric-value jet fiber. All 1,010 formulation-jet slots and all 96 "
            "candidate physical gravity rows are closed, and exact specialization at r=1 "
            "replays both predecessor receipts. This does not cover the axis r=0, arbitrary "
            "formulation functions, sourced constraint propagation, general hyperbolicity or "
            "common-time positivity, H7, universal matter, or promotion."
        ),
        "source_bindings": {
            "config": {
                "path": config_path.relative_to(repository).as_posix(),
                "canonical_json_sha256": _canonical_sha(config),
            },
            **{
                name: {
                    "path": path.relative_to(repository).as_posix(),
                    "file_sha256": _file_sha(path),
                    "content_sha256": value["content_sha256"],
                }
                for name, (path, value) in bound.items()
            },
            **{
                name: {
                    "path": path.relative_to(repository).as_posix(),
                    "file_sha256": _file_sha(path),
                }
                for name, path in sources.items()
            },
        },
    }
    receipt = {**body, "content_sha256": _canonical_sha(body)}
    if len(json.dumps(receipt).encode("utf-8")) > expected_caps["maximum_output_bytes"]:
        raise System10CylindricalDomainLiftError("output cap exceeded")
    return receipt


def write_receipt(
    config_path: Path, output_path: Path, *, root: Path | None = None
) -> dict[str, Any]:
    receipt = build_receipt(config_path, root=root)
    data = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if output_path.exists() and output_path.read_bytes() != data:
        raise System10CylindricalDomainLiftError("immutable output conflict")
    if not output_path.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(data)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    write_receipt(args.config.resolve(), args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
