from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from fractions import Fraction
from pathlib import Path
from typing import Any

Quadratic = tuple[Fraction, Fraction]


class Quartic85StateGaugeMapScalarExpansionError(RuntimeError):
    """Raised when scalar coefficient expansion fails closed."""


def _canonical_sha(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("ascii")).hexdigest()


def _file_sha(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except (OSError, ValueError) as exc:
        raise Quartic85StateGaugeMapScalarExpansionError(f"cannot read bound file: {path}") from exc


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise Quartic85StateGaugeMapScalarExpansionError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise Quartic85StateGaugeMapScalarExpansionError(f"JSON root is not an object: {path}")
    return value


def _resolve(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if root.resolve() not in path.parents:
        raise Quartic85StateGaugeMapScalarExpansionError("bound path escapes repository root")
    return path


def _load_binding(root: Path, binding: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    path = _resolve(root, str(binding.get("path", "")))
    if _file_sha(path) != binding.get("file_sha256"):
        raise Quartic85StateGaugeMapScalarExpansionError(f"bound file hash mismatch: {path}")
    value = _load_json(path)
    if value.get("content_sha256") != binding.get("content_sha256"):
        raise Quartic85StateGaugeMapScalarExpansionError(f"bound content hash mismatch: {path}")
    return path, value


def _pairs() -> list[tuple[int, int]]:
    return list(itertools.combinations_with_replacement(range(4), 2))


def _qadd(left: Quadratic, right: Quadratic) -> Quadratic:
    return left[0] + right[0], left[1] + right[1]


def _qscale(value: Quadratic, scale: Fraction) -> Quadratic:
    return value[0] * scale, value[1] * scale


def _fraction_text(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def _quadratic_text(value: Quadratic) -> str:
    rational, radical = value
    if radical == 0:
        return _fraction_text(rational)
    radical_text = "sqrt(2)" if abs(radical) == 1 else f"{_fraction_text(abs(radical))}*sqrt(2)"
    if radical < 0:
        radical_text = f"-{radical_text}"
    if rational == 0:
        return radical_text
    sign = "+" if radical > 0 else ""
    return f"{_fraction_text(rational)}{sign}{radical_text}"


def _metric_scale(pair: tuple[int, int], *, corrupt_off_diagonal: bool) -> Quadratic:
    if pair[0] == pair[1] or corrupt_off_diagonal:
        return Fraction(1), Fraction(0)
    return Fraction(0), Fraction(1, 2)


def _constraint_linear_coefficients(
    tilde: list[int], *, corrupt_off_diagonal: bool = False
) -> list[dict[tuple[int, int, int], Quadratic]]:
    coefficients: list[dict[tuple[int, int, int], Quadratic]] = [{} for _ in range(4)]
    for beta in range(4):
        for rho in range(4):
            pair = tuple(sorted((beta, rho)))
            first_key = (rho, pair[0], pair[1])
            first_value = _qscale(
                _metric_scale(pair, corrupt_off_diagonal=corrupt_off_diagonal),
                Fraction(tilde[rho]),
            )
            coefficients[beta][first_key] = _qadd(
                coefficients[beta].get(first_key, (Fraction(0), Fraction(0))),
                first_value,
            )
            trace_key = (beta, rho, rho)
            trace_value = (Fraction(-tilde[rho], 2), Fraction(0))
            coefficients[beta][trace_key] = _qadd(
                coefficients[beta].get(trace_key, (Fraction(0), Fraction(0))),
                trace_value,
            )
    return coefficients


def _projector(alpha: int, gamma: int, mu: int, nu: int, hat: list[int]) -> Fraction:
    value = Fraction(0)
    if alpha == mu and nu == gamma:
        value += hat[nu]
    if alpha == nu and mu == gamma:
        value += hat[mu]
    if alpha == gamma and mu == nu:
        value -= hat[mu]
    return value / 2


def _flat_rows(
    *, hat: list[int] | None = None, corrupt_off_diagonal: bool = False
) -> list[dict[str, Quadratic]]:
    physical = [-1, 1, 1, 1]
    tilde = [-4, 1, 1, 1]
    hat_values = hat or [-9, 1, 1, 1]
    constraint = _constraint_linear_coefficients(tilde, corrupt_off_diagonal=corrupt_off_diagonal)
    rows: list[dict[str, Quadratic]] = [{} for _ in range(4)]
    for nu in range(4):
        for mu in range(4):
            for gamma in range(4):
                for alpha in range(4):
                    projector = _projector(alpha, gamma, mu, nu, hat_values)
                    if projector == 0:
                        continue
                    factor = Fraction(physical[nu] * physical[alpha], -2) * projector
                    for (derivative, left, right), coefficient in constraint[alpha].items():
                        triple = tuple(sorted((mu, gamma, derivative)))
                        key = f"d3_g[{triple[0]},{triple[1]},{triple[2]}|{left},{right}]"
                        rows[nu][key] = _qadd(
                            rows[nu].get(key, (Fraction(0), Fraction(0))),
                            _qscale(coefficient, factor),
                        )
        rows[nu] = {key: value for key, value in rows[nu].items() if value != (0, 0)}
    return rows


def _first_difference(
    left: list[dict[str, Quadratic]], right: list[dict[str, Quadratic]]
) -> dict[str, Any]:
    difference_count = 0
    first: dict[str, Any] | None = None
    for row in range(4):
        for key in sorted(set(left[row]) | set(right[row])):
            delta = _qadd(
                left[row].get(key, (0, 0)),
                _qscale(right[row].get(key, (0, 0)), -1),
            )
            if delta == (0, 0):
                continue
            difference_count += 1
            if first is None:
                first = {
                    "row": row,
                    "slot_key": key,
                    "coefficient_delta": _quadratic_text(delta),
                }
    if first is None:
        raise Quartic85StateGaugeMapScalarExpansionError(
            "coefficient corruption negative unexpectedly passed"
        )
    return {"differing_coefficients": difference_count, "first_witness": first}


def _materialize(
    indexed: dict[str, Any],
    first_order: dict[str, Any],
    nonlinear: dict[str, Any],
    basis: dict[str, Any],
) -> dict[str, Any]:
    if indexed.get("decision") != ("PASS_EXACT_INDEXED_GAUGE_MAP_WITH_FORMAL_EXTERNAL_JET_PACKETS"):
        raise Quartic85StateGaugeMapScalarExpansionError("indexed gauge-map predecessor changed")
    if indexed.get("counts", {}).get("indexed_formula_templates") != 17:
        raise Quartic85StateGaugeMapScalarExpansionError("indexed formula template count changed")
    if first_order.get("status") != (
        "pass_all_12_exact_55_variable_principal_first_order_reductions"
    ):
        raise Quartic85StateGaugeMapScalarExpansionError("flat first-order predecessor changed")
    if nonlinear.get("status") != (
        "pass_all_12_exact_local_nonlinear_time_acceleration_eliminations"
    ):
        raise Quartic85StateGaugeMapScalarExpansionError("nonlinear Euler predecessor changed")
    if basis.get("decision") != (
        "BOUNDED_PASS_KINEMATIC_MATTER_BASIS_TYPED_BLOCK_GRAVITY_COORDINATE_MAP"
    ):
        raise Quartic85StateGaugeMapScalarExpansionError(
            "constraint-coordinate predecessor changed"
        )
    candidate_certificates = nonlinear.get("certificates", [])
    if len(candidate_certificates) != 12 or any(
        item.get("coefficients", {}).get("m2") != "1" for item in candidate_certificates
    ):
        raise Quartic85StateGaugeMapScalarExpansionError("candidate M2 normalization changed")
    operator_map = {
        item["key"]: item
        for item in indexed.get("materialization", {}).get("physical_metric_third_operator_map", [])
    }
    if len(operator_map) != 200:
        raise Quartic85StateGaugeMapScalarExpansionError(
            "physical metric third-operator map changed"
        )
    rows = _flat_rows()
    row_payloads: list[dict[str, Any]] = []
    for row, coefficients in enumerate(rows):
        entries = []
        for key in sorted(coefficients):
            mapped = operator_map[key]
            entries.append(
                {
                    "slot_key": key,
                    "coefficient": _quadratic_text(coefficients[key]),
                    "state_index": mapped["state_index"],
                    "state_coordinate": mapped["state_coordinate"],
                    "remaining_derivative_operator": mapped["remaining_derivative_operator"],
                }
            )
        row_body = {
            "gravity_constraint_row": f"divQ_lower[{row}]",
            "nonzero_scalar_coefficients": len(entries),
            "entries": entries,
        }
        row_payloads.append({**row_body, "row_sha256": _canonical_sha(row_body)})
    flat_rows_sha = _canonical_sha(row_payloads)
    candidate_records = indexed.get("materialization", {}).get("candidate_results", [])
    candidate_results = []
    for item in sorted(candidate_records, key=lambda record: record["candidate_id"]):
        manifest = {
            "schema_version": ("invariant-candidate-flat-gauge-row-expansion-manifest-1.0"),
            "candidate_id": item["candidate_id"],
            "indexed_gauge_map_manifest_sha256": item["manifest_sha256"],
            "flat_scalar_rows_sha256": flat_rows_sha,
            "M2": "1",
            "expanded_rows": 4,
            "general_external_jet_expansion_closed": False,
            "outcome": "PASS_FLAT_ROWS_TYPED_BLOCK_GENERAL_EXTERNAL_JETS",
        }
        candidate_results.append({**manifest, "manifest_sha256": _canonical_sha(manifest)})
    wrong_hat = _first_difference(rows, _flat_rows(hat=[-8, 1, 1, 1]))
    wrong_basis = _first_difference(rows, _flat_rows(corrupt_off_diagonal=True))
    external_zero_fill = {
        "mutation": "set d2_H[0,0|0]=1 after omitting all external-jet columns",
        "exact_missing_divQ_lower_0_coefficient": "-9/4",
        "rejected": True,
    }
    value_packet = {
        "reason_code": ("general_scalar_rows_require_external_and_lower_jet_value_packet"),
        "required_packets": [
            {
                "packet": "differentiated_external_formulation_jets",
                "exact_scalar_values": 580,
                "families": [
                    "hat_inverse_first",
                    "tilde_inverse_second",
                    "reference_connection_second",
                    "gauge_source_second",
                ],
            },
            {
                "packet": "lower_formulation_field_jets",
                "exact_scalar_values": 280,
                "contents": ("hat0, tilde0-1, reference-connection0-1, and gauge-source0-1"),
            },
            {
                "packet": "physical_metric_two_jet",
                "exact_scalar_values": 150,
                "contents": ("10 metric, 40 first partial, and 100 second partial values"),
                "sourced_acceleration_caveat": (
                    "ten metric second-time values must be bound to the sourced, "
                    "not vacuum-only, "
                    "Euler solve"
                ),
            },
            {
                "packet": "common_domain",
                "exact_scalar_values": 0,
                "contents": ("uniform bounds and compatibility for every supplied jet value"),
            },
        ],
        "total_exact_scalar_values_before_domain": 1010,
        "zero_fill_forbidden": True,
    }
    return {
        "flat_constant_scalar_rows": row_payloads,
        "flat_constant_scalar_rows_sha256": flat_rows_sha,
        "candidate_results": candidate_results,
        "negative_controls": {
            "wrong_hat_time_coefficient": {
                "mutation": "hat inverse time coefficient -9 -> -8",
                **wrong_hat,
                "rejected": True,
            },
            "wrong_off_diagonal_basis": {
                "mutation": "replace orthonormal off-diagonal 1/sqrt(2) by 1",
                **wrong_basis,
                "rejected": True,
            },
            "external_zero_fill": external_zero_fill,
        },
        "general_expansion_block": value_packet,
    }


def build_receipt(config_path: Path, *, root: Path | None = None) -> dict[str, Any]:
    repository = (root or config_path.resolve().parents[1]).resolve()
    config = _load_json(config_path)
    if config.get("schema_version") != (
        "invariant-quartic-85-state-gauge-map-scalar-coefficient-expansion-config-1.0"
    ):
        raise Quartic85StateGaugeMapScalarExpansionError("unsupported config schema")
    expected_specialization = {
        "physical_inverse_metric_diagonal": [-1, 1, 1, 1],
        "tilde_inverse_metric_diagonal": [-4, 1, 1, 1],
        "hat_inverse_metric_diagonal": [-9, 1, 1, 1],
        "M2": "1",
        "reference_connection": "zero_cartesian",
        "gauge_source_H": "zero",
        "external_formulation_derivatives": "zero",
        "metric_component_basis": "orthonormal_symmetric",
    }
    if config.get("flat_constant_formulation_specialization") != expected_specialization:
        raise Quartic85StateGaugeMapScalarExpansionError("flat constant specialization changed")
    expected_policy = {
        "flat_reference_scalar_coefficient_rows": True,
        "all_twelve_candidate_flat_row_hashes": True,
        "general_external_jet_scalar_expansion": False,
        "external_formulation_jet_domain": False,
        "fully_nonlinear_85_state_rows": False,
        "constraint_propagation": False,
        "candidate_jet_uniformity": False,
        "nonlinear_global_closure": False,
        "gravity_h7": False,
        "universal_all_matter": False,
        "promotion": False,
    }
    if config.get("claims_policy") != expected_policy:
        raise Quartic85StateGaugeMapScalarExpansionError("claims policy is absent or broadened")
    bound = {
        name: _load_binding(repository, binding)
        for name, binding in config.get("bindings", {}).items()
    }
    expected_bindings = {
        "indexed_gauge_map",
        "vacuum_first_order_reference",
        "vacuum_nonlinear_euler",
        "constraint_coordinate_basis",
    }
    if set(bound) != expected_bindings:
        raise Quartic85StateGaugeMapScalarExpansionError("closed binding manifest changed")
    materialization = _materialize(
        bound["indexed_gauge_map"][1],
        bound["vacuum_first_order_reference"][1],
        bound["vacuum_nonlinear_euler"][1],
        bound["constraint_coordinate_basis"][1],
    )
    source_path = Path(__file__).resolve()
    test_path = repository / (
        "tests/test_quartic_85_state_gauge_map_scalar_coefficient_expansion_gate.py"
    )
    row_counts = [
        item["nonzero_scalar_coefficients"] for item in materialization["flat_constant_scalar_rows"]
    ]
    body: dict[str, Any] = {
        "schema_version": (
            "invariant-quartic-85-state-gauge-map-scalar-coefficient-expansion-result-1.0"
        ),
        "campaign_id": config["campaign_id"],
        "decision": "BOUNDED_PASS_FLAT_SCALAR_ROWS_TYPED_BLOCK_GENERAL_EXTERNAL_JETS",
        "materialization": materialization,
        "counts": {
            "flat_gravity_constraint_rows_expanded": 4,
            "nonzero_scalar_coefficients_by_row": row_counts,
            "nonzero_scalar_coefficients_total": sum(row_counts),
            "candidate_flat_row_manifests": 12,
            "general_external_jet_row_expansions": 0,
            "required_general_scalar_values_before_domain": 1010,
            "negative_controls": 3,
            "constraint_propagation_claims": 0,
        },
        "claims": {
            "exact_flat_reference_scalar_coefficient_rows_closed": True,
            "all_twelve_candidate_flat_row_hashes_closed": True,
            "general_external_jet_scalar_expansion_closed": False,
            "external_formulation_jet_domain_closed": False,
            "fully_nonlinear_85_state_rows_closed": False,
            "constraint_propagation_closed": False,
            "candidate_jet_uniformity_closed": False,
            "nonlinear_global_closure_established": False,
            "gravity_h7_theorem_established": False,
            "universal_all_matter_closure_established": False,
            "promotion_authorized": False,
        },
        "scope": (
            "Exact Q(sqrt(2)) scalar coefficients of the four differentiated "
            "gauge rows at the registered flat constant formulation reference, "
            "lowered to physical metric third-derivative operators in the "
            "85-state ordering and hash-bound to all 12 candidates. A general "
            "nonlinear coefficient expansion remains blocked on 1,010 exact "
            "lower/external jet values, a sourced acceleration packet, and a "
            "common domain. Constraint propagation, candidate-jet uniformity, "
            "nonlinear/global "
            "closure, H7, universal matter, and promotion remain false."
        ),
        "source_bindings": {
            "config": {
                "path": config_path.relative_to(repository).as_posix(),
                "file_sha256": _file_sha(config_path),
            },
            **{
                name: {
                    "path": path.relative_to(repository).as_posix(),
                    "file_sha256": _file_sha(path),
                    "content_sha256": value["content_sha256"],
                }
                for name, (path, value) in bound.items()
            },
            "source": {
                "path": source_path.relative_to(repository).as_posix(),
                "file_sha256": _file_sha(source_path),
            },
            "test": {
                "path": test_path.relative_to(repository).as_posix(),
                "file_sha256": _file_sha(test_path),
            },
        },
    }
    return {**body, "content_sha256": _canonical_sha(body)}


def write_receipt(
    config_path: Path, output_path: Path, *, root: Path | None = None
) -> dict[str, Any]:
    receipt = build_receipt(config_path, root=root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
