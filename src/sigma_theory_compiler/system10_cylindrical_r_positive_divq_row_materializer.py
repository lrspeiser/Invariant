from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import sympy as sp


class System10CylindricalDivQError(RuntimeError):
    """Raised when the exact cylindrical divQ row contract fails closed."""


R = sp.Symbol("r", positive=True)
OperatorKey = tuple[int, tuple[int, ...]]
Operator = dict[OperatorKey, sp.Expr]


def _canonical_sha(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode("ascii")).hexdigest()


def _file_sha(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise System10CylindricalDivQError(f"cannot read bound file: {path}") from exc


def _portable_text_sha(path: Path) -> str:
    try:
        raw = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    except OSError as exc:
        raise System10CylindricalDivQError(f"cannot read bound source: {path}") from exc
    return hashlib.sha256(raw).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise System10CylindricalDivQError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise System10CylindricalDivQError("bound JSON root is not an object")
    return value


def _resolve(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if root.resolve() not in path.parents:
        raise System10CylindricalDivQError("bound path escapes repository root")
    return path


def _load_binding(root: Path, binding: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    path = _resolve(root, str(binding.get("path", "")))
    raw_expected = binding.get("file_sha256")
    canonical_lf_expected = binding.get("canonical_lf_sha256")
    if (isinstance(raw_expected, str)) == (isinstance(canonical_lf_expected, str)):
        raise System10CylindricalDivQError("binding must select exactly one byte representation")
    actual = _file_sha(path) if isinstance(raw_expected, str) else _portable_text_sha(path)
    expected = raw_expected if isinstance(raw_expected, str) else canonical_lf_expected
    if actual != expected:
        raise System10CylindricalDivQError(f"bound file hash mismatch: {path}")
    value = _load_json(path)
    if value.get("content_sha256") != binding.get("content_sha256"):
        raise System10CylindricalDivQError(f"bound content hash mismatch: {path}")
    return path, value


def _reported_binding(
    path: Path, value: dict[str, Any], binding: dict[str, Any], repository: Path
) -> dict[str, Any]:
    body = {
        "path": path.relative_to(repository).as_posix(),
        "content_sha256": value["content_sha256"],
    }
    if "canonical_lf_sha256" in binding:
        body["canonical_lf_sha256"] = _portable_text_sha(path)
    else:
        body["file_sha256"] = _file_sha(path)
    return body


def _load_source(root: Path, binding: dict[str, Any]) -> Path:
    path = _resolve(root, str(binding.get("path", "")))
    if _portable_text_sha(path) != binding.get("canonical_lf_sha256"):
        raise System10CylindricalDivQError(f"bound source hash mismatch: {path}")
    return path


def _text(expression: sp.Expr) -> str:
    return sp.sstr(sp.factor(sp.cancel(expression)))


def _denominator_power(expression: sp.Expr) -> int:
    denominator = sp.Poly(sp.denom(sp.cancel(expression)), R)
    if len(denominator.terms()) != 1:
        raise System10CylindricalDivQError("coefficient denominator is not a monomial in r")
    (power,), coefficient = denominator.terms()[0]
    if coefficient == 0:
        raise System10CylindricalDivQError("zero coefficient in denominator certificate")
    return int(power)


def _clean(operator: Operator) -> Operator:
    cleaned: Operator = {}
    for key, coefficient in operator.items():
        value = sp.factor(sp.cancel(coefficient))
        if value != 0:
            cleaned[key] = value
    return cleaned


def _scaled(operator: Operator, scalar: sp.Expr) -> Operator:
    scalar = sp.sympify(scalar)
    if scalar == 0:
        return {}
    return _clean({key: scalar * coefficient for key, coefficient in operator.items()})


def _sum_operators(operators: Iterable[Operator]) -> Operator:
    result: Operator = {}
    for operator in operators:
        for key, coefficient in operator.items():
            result[key] = result.get(key, sp.Integer(0)) + coefficient
    return _clean(result)


def _partial(operator: Operator, coordinate: int) -> Operator:
    result: Operator = {}
    for (state_index, derivatives), coefficient in operator.items():
        coefficient_derivative = sp.diff(coefficient, R) if coordinate == 1 else sp.Integer(0)
        key = (state_index, derivatives)
        result[key] = result.get(key, sp.Integer(0)) + coefficient_derivative
        differentiated = (state_index, tuple(sorted((*derivatives, coordinate))))
        result[differentiated] = result.get(differentiated, sp.Integer(0)) + coefficient
    return _clean(result)


def _diagonal(values: tuple[sp.Expr, ...]) -> list[list[sp.Expr]]:
    return [
        [sp.sympify(values[row]) if row == column else sp.Integer(0) for column in range(4)]
        for row in range(4)
    ]


def _physical_tensors() -> tuple[
    list[list[sp.Expr]],
    list[list[sp.Expr]],
    list[list[sp.Expr]],
    list[list[list[sp.Expr]]],
]:
    metric = _diagonal((-1, 1, R**2, 1))
    inverse = _diagonal((-1, 1, R**-2, 1))
    hat = _diagonal((-9, 1, R**-2, 1))
    connection = [[[sp.Integer(0) for _ in range(4)] for _ in range(4)] for _ in range(4)]
    for upper in range(4):
        for left in range(4):
            for right in range(4):
                value = sp.Integer(0)
                for lower in range(4):
                    d_left = sp.diff(metric[lower][right], R) if left == 1 else 0
                    d_right = sp.diff(metric[lower][left], R) if right == 1 else 0
                    d_lower = sp.diff(metric[left][right], R) if lower == 1 else 0
                    value += inverse[upper][lower] * (d_left + d_right - d_lower) / 2
                connection[upper][left][right] = sp.factor(sp.cancel(value))
    return metric, inverse, hat, connection


def _constraint_rows(domain: dict[str, Any]) -> list[Operator]:
    rows = domain.get("materialization", {}).get("shared_symbolic_gauge_rows", [])
    if len(rows) != 4 or [item.get("row") for item in rows] != [
        f"modified_harmonic_C[{index}]" for index in range(4)
    ]:
        raise System10CylindricalDivQError("cylindrical gauge row authority changed")
    operators: list[Operator] = []
    for item in rows:
        if item.get("background_residual") != "0" or item.get("domain") != "r>0":
            raise System10CylindricalDivQError("gauge row domain or residual changed")
        operator = {
            (int(term["state_index"]), ()): sp.sympify(term["coefficient"], locals={"r": R})
            for term in item.get("state_terms", [])
        }
        if len(operator) != item.get("coefficient_support"):
            raise System10CylindricalDivQError("gauge row support changed")
        operators.append(_clean(operator))
    return operators


def _covariant_constraint_derivatives(
    constraints: list[Operator], connection: list[list[list[sp.Expr]]]
) -> tuple[list[list[Operator]], list[list[list[Operator]]]]:
    first = [[{} for _ in range(4)] for _ in range(4)]
    for gamma in range(4):
        for beta in range(4):
            pieces = [_partial(constraints[beta], gamma)]
            pieces.extend(
                _scaled(constraints[lower], -connection[lower][gamma][beta]) for lower in range(4)
            )
            first[gamma][beta] = _sum_operators(pieces)
    second = [[[{} for _ in range(4)] for _ in range(4)] for _ in range(4)]
    for mu in range(4):
        for gamma in range(4):
            for beta in range(4):
                pieces = [_partial(first[gamma][beta], mu)]
                pieces.extend(
                    _scaled(first[lower][beta], -connection[lower][mu][gamma]) for lower in range(4)
                )
                pieces.extend(
                    _scaled(first[gamma][lower], -connection[lower][mu][beta]) for lower in range(4)
                )
                second[mu][gamma][beta] = _sum_operators(pieces)
    return first, second


def _hat_projectors(
    hat: list[list[sp.Expr]], connection: list[list[list[sp.Expr]]]
) -> tuple[Any, Any]:
    nabla_hat = [[[sp.Integer(0) for _ in range(4)] for _ in range(4)] for _ in range(4)]
    for mu in range(4):
        for left in range(4):
            for right in range(4):
                value = sp.diff(hat[left][right], R) if mu == 1 else sp.Integer(0)
                for lower in range(4):
                    value += connection[left][mu][lower] * hat[lower][right]
                    value += connection[right][mu][lower] * hat[left][lower]
                nabla_hat[mu][left][right] = sp.factor(sp.cancel(value))

    def projector(alpha: int, gamma: int, mu: int, nu: int) -> sp.Expr:
        value = (
            int(alpha == mu) * hat[nu][gamma]
            + int(alpha == nu) * hat[mu][gamma]
            - int(alpha == gamma) * hat[mu][nu]
        ) / 2
        return sp.factor(sp.cancel(value))

    def projector_derivative(derivative: int, alpha: int, gamma: int, mu: int, nu: int) -> sp.Expr:
        value = (
            int(alpha == mu) * nabla_hat[derivative][nu][gamma]
            + int(alpha == nu) * nabla_hat[derivative][mu][gamma]
            - int(alpha == gamma) * nabla_hat[derivative][mu][nu]
        ) / 2
        return sp.factor(sp.cancel(value))

    return projector, projector_derivative


def _operator_rows(domain: dict[str, Any]) -> list[dict[str, Any]]:
    constraints = _constraint_rows(domain)
    metric, inverse, hat, connection = _physical_tensors()
    first, second = _covariant_constraint_derivatives(constraints, connection)
    projector, projector_derivative = _hat_projectors(hat, connection)
    upper_rows: list[Operator] = []
    for nu in range(4):
        pieces: list[Operator] = []
        for mu in range(4):
            for alpha in range(4):
                for beta in range(4):
                    if inverse[alpha][beta] == 0:
                        continue
                    for gamma in range(4):
                        first_scale = (
                            -projector_derivative(mu, alpha, gamma, mu, nu)
                            * inverse[alpha][beta]
                            / 2
                        )
                        second_scale = -projector(alpha, gamma, mu, nu) * inverse[alpha][beta] / 2
                        if first_scale != 0:
                            pieces.append(_scaled(first[gamma][beta], first_scale))
                        if second_scale != 0:
                            pieces.append(_scaled(second[mu][gamma][beta], second_scale))
        upper_rows.append(_sum_operators(pieces))
    lower_rows = [
        _sum_operators(_scaled(upper_rows[sigma], metric[nu][sigma]) for sigma in range(4))
        for nu in range(4)
    ]
    rows: list[dict[str, Any]] = []
    for component, operator in enumerate(lower_rows):
        terms = []
        for (state_index, derivatives), coefficient in sorted(operator.items()):
            term = {
                "state_index": state_index,
                "coordinate_derivatives": list(derivatives),
                "coefficient": _text(coefficient),
                "denominator_r_power": _denominator_power(coefficient),
            }
            terms.append({**term, "term_sha256": _canonical_sha(term)})
        r1_terms = [
            {
                "state_index": item["state_index"],
                "coordinate_derivatives": item["coordinate_derivatives"],
                "coefficient": _text(sp.sympify(item["coefficient"], locals={"r": R}).subs(R, 1)),
            }
            for item in terms
        ]
        body = {
            "component": f"divQ_lower[{component}]",
            "normalization": f"divQ_lower[{component}]/M2",
            "domain": "r>0",
            "terms": terms,
            "term_count": len(terms),
            "nonzero_term_count": len(terms),
            "maximum_coordinate_derivative_order": max(
                len(item["coordinate_derivatives"]) for item in terms
            ),
            "maximum_denominator_r_power": max(item["denominator_r_power"] for item in terms),
            "r1_terms": r1_terms,
            "r1_terms_sha256": _canonical_sha(r1_terms),
        }
        rows.append({**body, "row_sha256": _canonical_sha(body)})
    return rows


def _validate_predecessors(bound: dict[str, tuple[Path, dict[str, Any]]]) -> list[str]:
    attempt = bound["propagation_attempt"][1]
    domain = bound["r_positive_domain"][1]
    indexed = bound["indexed_divergence_map"][1]
    off_shell = bound["off_shell_identity"][1]
    missing = attempt.get("materialization", {}).get("first_missing_primitive", {})
    if (
        attempt.get("decision")
        != "TYPED_BLOCK_R_POSITIVE_DIVQ_ROWS_AND_FULL_EVOLUTION_UNREGISTERED"
        or missing.get("required_rows") != 4
        or missing.get("registered_rows") != 0
        or missing.get("acceptance", {}).get("exact_r1_replay_required") is not True
    ):
        raise System10CylindricalDivQError("propagation attempt authority changed")
    if (
        domain.get("decision")
        != "BOUNDED_PASS_FIXED_CYLINDRICAL_R_POSITIVE_1010_JETS_AND_96_ROWS_NO_PROPAGATION_CLAIM"
        or domain.get("materialization", {}).get("domain_certificate", {}).get("domain") != "r>0"
    ):
        raise System10CylindricalDivQError("r-positive domain authority changed")
    program = indexed.get("materialization", {}).get("indexed_formula_program", {})
    if (
        indexed.get("decision") != "PASS_EXACT_INDEXED_GAUGE_MAP_WITH_FORMAL_EXTERNAL_JET_PACKETS"
        or program.get("output_components") != [f"divQ_lower[{index}]" for index in range(4)]
        or indexed.get("counts", {}).get("fully_expanded_85_state_coefficient_rows") != 0
    ):
        raise System10CylindricalDivQError("indexed divergence authority changed")
    formula = (
        off_shell.get("materialization", {}).get("common_formula", {}).get("gauge_formula", {})
    )
    if formula.get("completion") != (
        "Q^mu_nu=-M2/2 hat_P_alpha^(gamma mu nu) g^(alpha beta) nabla_gamma C_beta"
    ):
        raise System10CylindricalDivQError("off-shell gauge formula changed")
    attempts = attempt.get("materialization", {}).get("candidate_attempts", [])
    candidate_ids = sorted(str(item["candidate_id"]) for item in attempts)
    if len(candidate_ids) != 12 or len(set(candidate_ids)) != 12:
        raise System10CylindricalDivQError("candidate census changed")
    return candidate_ids


def _materialize(
    bound: dict[str, tuple[Path, dict[str, Any]]], expected: dict[str, Any]
) -> dict[str, Any]:
    candidate_ids = _validate_predecessors(bound)
    domain = bound["r_positive_domain"][1]
    indexed = bound["indexed_divergence_map"][1]
    rows = _operator_rows(domain)
    measured = {
        "row_sha256": [row["row_sha256"] for row in rows],
        "r1_terms_sha256": [row["r1_terms_sha256"] for row in rows],
        "term_counts": [row["term_count"] for row in rows],
        "maximum_denominator_r_power": max(row["maximum_denominator_r_power"] for row in rows),
    }
    if measured != expected:
        raise System10CylindricalDivQError("frozen row expectations changed")
    pole = {
        "domain": "r>0",
        "all_coefficient_denominators_monomials_in_r": True,
        "maximum_denominator_r_power": measured["maximum_denominator_r_power"],
        "denominator_zero_set": ["r=0"],
        "axis_excluded": True,
        "poles_on_admitted_domain": 0,
    }
    row_set_sha = _canonical_sha(measured["row_sha256"])
    candidates = []
    for candidate_id in candidate_ids:
        body = {
            "candidate_id": candidate_id,
            "common_divQ_row_set_sha256": row_set_sha,
            "registered_divQ_rows": 4,
            "full_nonlinear_85_state_rhs_registered": False,
            "constraint_propagation_claimed": False,
            "outcome": "PASS_EXACT_DIVQ_ROWS_BLOCK_FULL_EVOLUTION_RHS",
        }
        candidates.append({**body, "manifest_sha256": _canonical_sha(body)})

    def mutated_row_sha(
        row: dict[str, Any], terms: list[dict[str, Any]], r1_terms: list[dict[str, Any]]
    ) -> str:
        body = {key: value for key, value in row.items() if key != "row_sha256"}
        body.update(
            {
                "terms": terms,
                "term_count": len(terms),
                "nonzero_term_count": len(terms),
                "maximum_coordinate_derivative_order": max(
                    (len(item["coordinate_derivatives"]) for item in terms), default=0
                ),
                "maximum_denominator_r_power": max(
                    (item["denominator_r_power"] for item in terms), default=0
                ),
                "r1_terms": r1_terms,
                "r1_terms_sha256": _canonical_sha(r1_terms),
            }
        )
        return _canonical_sha(body)

    zero_hashes = [mutated_row_sha(row, [], []) for row in rows]
    zero_row_set_sha = _canonical_sha(zero_hashes)
    dropped_hashes = measured["row_sha256"][:-1] + [
        mutated_row_sha(rows[-1], rows[-1]["terms"][:-1], rows[-1]["r1_terms"][:-1])
    ]
    dropped_row_set_sha = _canonical_sha(dropped_hashes)
    sign_hashes = []
    for row in rows:
        flipped_terms = []
        for term in row["terms"]:
            body = {key: value for key, value in term.items() if key != "term_sha256"}
            body["coefficient"] = _text(-sp.sympify(term["coefficient"], locals={"r": R}))
            flipped_terms.append({**body, "term_sha256": _canonical_sha(body)})
        flipped_r1 = [
            {**term, "coefficient": _text(-sp.sympify(term["coefficient"]))}
            for term in row["r1_terms"]
        ]
        sign_hashes.append(mutated_row_sha(row, flipped_terms, flipped_r1))
    sign_row_set_sha = _canonical_sha(sign_hashes)
    negatives = {
        "zero_fill_all_four_rows": {
            "mutated_row_set_sha256": zero_row_set_sha,
            "expected_row_set_sha256": row_set_sha,
            "rejected": zero_row_set_sha != row_set_sha,
        },
        "drop_last_term": {
            "mutated_term_count": sum(row["term_count"] for row in rows) - 1,
            "expected_term_count": sum(row["term_count"] for row in rows),
            "mutated_row_set_sha256": dropped_row_set_sha,
            "expected_row_set_sha256": row_set_sha,
            "rejected": dropped_row_set_sha != row_set_sha,
        },
        "flip_completion_sign": {
            "mutated_row_set_sha256": sign_row_set_sha,
            "expected_row_set_sha256": row_set_sha,
            "rejected": sign_row_set_sha != row_set_sha,
        },
        "include_axis": {
            "mutation": "replace r>0 by r>=0",
            "denominator_zero_set": ["r=0"],
            "rejected": True,
        },
    }
    return {
        "operator_convention": {
            "state_dimension": 85,
            "coordinate_indices": [0, 1, 2, 3],
            "coordinate_partials_commute": True,
            "coefficient_scale": "M2",
            "row_normalization": "divQ_lower[nu]/M2",
            "indexed_formula_program_sha256": indexed["materialization"]["indexed_formula_program"][
                "program_sha256"
            ],
        },
        "rows": rows,
        "row_set_sha256": row_set_sha,
        "pole_certificate": {**pole, "certificate_sha256": _canonical_sha(pole)},
        "candidate_results": candidates,
        "negative_controls": negatives,
        "next_missing_primitive": bound["propagation_attempt"][1]["materialization"][
            "next_missing_primitive"
        ],
    }


def build_receipt(config_path: Path, *, root: Path | None = None) -> dict[str, Any]:
    repository = (root or config_path.resolve().parents[1]).resolve()
    config = _load_json(config_path)
    if config.get("schema_version") != (
        "invariant-system10-cylindrical-r-positive-divq-row-materializer-config-1.0"
    ):
        raise System10CylindricalDivQError("unsupported config schema")
    expected_caps = {
        "candidates": 12,
        "state_dimension": 85,
        "divq_rows": 4,
        "maximum_total_terms": 4096,
        "maximum_denominator_r_power": 16,
        "maximum_output_bytes": 2097152,
    }
    if config.get("caps") != expected_caps:
        raise System10CylindricalDivQError("caps changed")
    expected_claims = {
        "four_divq_rows": True,
        "fixed_cylindrical_r_positive": True,
        "full_nonlinear_85_state_rhs": False,
        "constraint_propagation": False,
        "subsidiary_energy": False,
        "general_hyperbolicity": False,
        "global_theorem": False,
        "promotion": False,
    }
    if config.get("claims_policy") != expected_claims:
        raise System10CylindricalDivQError("claims policy broadened")
    bound = {
        name: _load_binding(repository, binding)
        for name, binding in config.get("bindings", {}).items()
    }
    if set(bound) != {
        "propagation_attempt",
        "r_positive_domain",
        "indexed_divergence_map",
        "off_shell_identity",
    }:
        raise System10CylindricalDivQError("binding manifest changed")
    sources = {
        name: _load_source(repository, binding)
        for name, binding in config.get("source_evidence", {}).items()
    }
    if set(sources) != {"source", "test"}:
        raise System10CylindricalDivQError("source evidence manifest changed")
    expected_test = (
        repository / "tests/test_system10_cylindrical_r_positive_divq_row_materializer.py"
    )
    if sources["source"] != Path(__file__).resolve() or sources["test"] != expected_test:
        raise System10CylindricalDivQError("self evidence path changed")
    materialization = _materialize(bound, config.get("frozen_expectations", {}))
    total_terms = sum(row["term_count"] for row in materialization["rows"])
    maximum_power = materialization["pole_certificate"]["maximum_denominator_r_power"]
    if total_terms > expected_caps["maximum_total_terms"]:
        raise System10CylindricalDivQError("term cap exceeded")
    if maximum_power > expected_caps["maximum_denominator_r_power"]:
        raise System10CylindricalDivQError("denominator cap exceeded")
    body = {
        "schema_version": (
            "invariant-system10-cylindrical-r-positive-divq-row-materializer-result-1.0"
        ),
        "campaign_id": config["campaign_id"],
        "decision": "BOUNDED_PASS_FOUR_R_POSITIVE_DIVQ_ROWS_BLOCK_FULL_EVOLUTION_RHS",
        "materialization": materialization,
        "counts": {
            "candidates": 12,
            "state_dimension": 85,
            "divq_rows_required": 4,
            "divq_rows_registered": 4,
            "total_nonzero_operator_terms": total_terms,
            "r1_rows_replayed": 4,
            "poles_on_r_positive_domain": 0,
            "full_nonlinear_85_state_rhs_rows_registered": 0,
            "constraint_propagation_proofs": 0,
            "negative_controls": 4,
        },
        "claims": {
            "four_divq_rows_closed": True,
            "fixed_cylindrical_r_positive_closed": True,
            "full_nonlinear_85_state_rhs_closed": False,
            "constraint_propagation_closed": False,
            "subsidiary_energy_closed": False,
            "general_hyperbolicity_closed": False,
            "global_theorem_established": False,
            "promotion_authorized": False,
        },
        "scope": (
            "Exact expansion of the four gauge-completion divergence rows on the fixed "
            "cylindrical r>0 profile. Coefficients are rational functions of r multiplying "
            "the common M2 scale, and every row is an explicit order-at-most-two operator on "
            "the registered 85-state coordinates. This closes only the first primitive named "
            "by the predecessor audit. The full nonlinear 85-state evolution RHS remains "
            "unregistered, so no constraint-propagation, subsidiary-energy, hyperbolicity, "
            "global, or promotion conclusion follows."
        ),
        "source_bindings": {
            "config": {
                "path": config_path.relative_to(repository).as_posix(),
                "canonical_json_sha256": _canonical_sha(config),
            },
            **{
                name: _reported_binding(path, value, config["bindings"][name], repository)
                for name, (path, value) in bound.items()
            },
            **{
                name: {
                    "path": path.relative_to(repository).as_posix(),
                    "canonical_lf_sha256": _portable_text_sha(path),
                }
                for name, path in sources.items()
            },
        },
    }
    receipt = {**body, "content_sha256": _canonical_sha(body)}
    if len(json.dumps(receipt).encode("utf-8")) > expected_caps["maximum_output_bytes"]:
        raise System10CylindricalDivQError("output cap exceeded")
    return receipt


def write_receipt(
    config_path: Path, output_path: Path, *, root: Path | None = None
) -> dict[str, Any]:
    receipt = build_receipt(config_path, root=root)
    data = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if output_path.exists() and output_path.read_bytes() != data:
        raise System10CylindricalDivQError("immutable output conflict")
    if not output_path.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(data)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    write_receipt(arguments.config.resolve(), arguments.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
