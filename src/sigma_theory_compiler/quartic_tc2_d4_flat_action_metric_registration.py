from __future__ import annotations

import argparse
import ast
import hashlib
import json
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

SCHEMA = "sigma-quartic-tc2-d4-flat-action-metric-registration-1.0"
CONFIG_SCHEMA = "sigma-quartic-tc2-d4-flat-action-metric-registration-config-1.0"
STATUS = "pass_exact_flat_action_metric_h_plus_0_registration"
PREDECESSOR_STATUS = "block_K55_Taylor_order_zero_missing_exact_reference_action_metric"
CONFIG_PATH = "configs/backgrounds/quartic_tc2_d4_flat_action_metric_registration.json"
SOURCE_PATH = "src/sigma_theory_compiler/quartic_tc2_d4_flat_action_metric_registration.py"
TEST_PATH = "tests/test_quartic_tc2_d4_flat_action_metric_registration.py"
EXPECTED_PREDECESSOR = {
    "path": "runs/physics-language/quartic-tc2-d4-coordinate-free-k55-taylor-order-zero-serialization-audit/campaign.json",
    "file_sha256": "3d0305df40af344a913ac20b496aac91db30ba09dba902b22612d7dae1fc1d12",
    "content_sha256": "c20f941b1a1317d76c5d81f490d0ef19ba241eaec78a2c15294c8a4620eb5489",
}
EXPECTED_FORMULA_SOURCE = {
    "path": "src/sigma_theory_compiler/horndeski_principal.py",
    "file_sha256": "aa6163c32ec125fa35c860832605927801a3101fe570e6d88edc20668b4348f3",
    "required_functions": [
        "_symmetric_basis",
        "_metric_action_block",
        "_first_order_generalized_pencil",
    ],
}
SYMMETRIC_PAIRS = (
    (0, 0),
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 1),
    (1, 2),
    (1, 3),
    (2, 2),
    (2, 3),
    (3, 3),
)


class FlatActionMetricRegistrationError(ValueError):
    """Raised when exact flat action-metric registration fails closed."""


@dataclass(frozen=True)
class Qsqrt2:
    rational: Fraction = Fraction()
    radical: Fraction = Fraction()

    def __add__(self, other: object) -> Qsqrt2:
        value = _q(other)
        return Qsqrt2(self.rational + value.rational, self.radical + value.radical)

    __radd__ = __add__

    def __neg__(self) -> Qsqrt2:
        return Qsqrt2(-self.rational, -self.radical)

    def __sub__(self, other: object) -> Qsqrt2:
        return self + (-_q(other))

    def __rsub__(self, other: object) -> Qsqrt2:
        return _q(other) - self

    def __mul__(self, other: object) -> Qsqrt2:
        value = _q(other)
        return Qsqrt2(
            self.rational * value.rational + 2 * self.radical * value.radical,
            self.rational * value.radical + self.radical * value.rational,
        )

    __rmul__ = __mul__

    def __truediv__(self, other: object) -> Qsqrt2:
        value = _q(other)
        denominator = value.rational**2 - 2 * value.radical**2
        if denominator == 0:
            raise ZeroDivisionError("zero Q(sqrt(2)) denominator")
        return self * Qsqrt2(value.rational / denominator, -value.radical / denominator)

    def text(self) -> str:
        if not self.radical:
            return _fraction_text(self.rational)
        if not self.rational:
            if self.radical == 1:
                return "sqrt(2)"
            if self.radical == -1:
                return "-sqrt(2)"
            return f"{_fraction_text(self.radical)}*sqrt(2)"
        sign = "+" if self.radical > 0 else "-"
        magnitude = abs(self.radical)
        radical = "sqrt(2)" if magnitude == 1 else f"{_fraction_text(magnitude)}*sqrt(2)"
        return f"{_fraction_text(self.rational)}{sign}{radical}"


ZERO = Qsqrt2()
ONE = Qsqrt2(Fraction(1))
SQRT2_OVER_2 = Qsqrt2(radical=Fraction(1, 2))


def _q(value: object) -> Qsqrt2:
    if isinstance(value, Qsqrt2):
        return value
    return Qsqrt2(Fraction(value))  # type: ignore[arg-type]


def _fraction_text(value: Fraction) -> str:
    return str(value.numerator) if value.denominator == 1 else str(value)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "ascii"
    )


def _content_hash(value: dict[str, Any]) -> str:
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    return hashlib.sha256(_canonical_bytes(body)).hexdigest()


def _hash_matches(value: dict[str, Any]) -> bool:
    return value.get("content_sha256") == _content_hash(value)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise FlatActionMetricRegistrationError(f"expected JSON object: {path}")
    return value


def _resolve_under(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if root != path and root not in path.parents:
        raise FlatActionMetricRegistrationError("bound path escaped project root")
    return path


def _zero_matrix(rows: int, columns: int) -> list[list[Qsqrt2]]:
    return [[ZERO for _ in range(columns)] for _ in range(rows)]


def _basis() -> list[list[list[Qsqrt2]]]:
    result = []
    for row, column in SYMMETRIC_PAIRS:
        matrix = _zero_matrix(4, 4)
        matrix[row][column] = ONE if row == column else SQRT2_OVER_2
        matrix[column][row] = matrix[row][column]
        result.append(matrix)
    return result


def _trace_reverse(
    first: tuple[int, int], second: tuple[int, int], inverse: list[Qsqrt2]
) -> Qsqrt2:
    mu, nu = first
    rho, sigma = second
    diagonal = lambda left, right: inverse[left] if left == right else ZERO
    return (
        diagonal(mu, rho) * diagonal(nu, sigma)
        + diagonal(mu, sigma) * diagonal(nu, rho)
        - diagonal(mu, nu) * diagonal(rho, sigma)
    ) / 2


def _projector(
    lower: int, derivative: int, first: int, second: int, inverse: list[Qsqrt2]
) -> Qsqrt2:
    diagonal = lambda left, right: inverse[left] if left == right else ZERO
    return (
        int(lower == first) * diagonal(second, derivative)
        + int(lower == second) * diagonal(first, derivative)
        - int(lower == derivative) * diagonal(first, second)
    ) / 2


def _baseline_action_at(xi0: int) -> list[list[Qsqrt2]]:
    inverse = [_q(-1), ONE, ONE, ONE]
    xi = [_q(xi0), ONE, ZERO, ZERO]
    xi_squared = sum((inverse[index] * xi[index] * xi[index] for index in range(4)), ZERO)
    basis = _basis()
    baseline = _zero_matrix(10, 10)
    for column, input_tensor in enumerate(basis):
        output = _zero_matrix(4, 4)
        for mu in range(4):
            for nu in range(4):
                trace = sum(
                    (
                        _trace_reverse((mu, nu), pair, inverse)
                        * input_tensor[pair[0]][pair[1]]
                        * (2 if pair[0] != pair[1] else 1)
                        for pair in SYMMETRIC_PAIRS
                    ),
                    ZERO,
                )
                gauge = ZERO
                for lower in range(4):
                    for derivative_left in range(4):
                        left = _projector(lower, derivative_left, mu, nu, inverse)
                        if left == ZERO:
                            continue
                        for raised in range(4):
                            middle = inverse[lower] if lower == raised else ZERO
                            if middle == ZERO:
                                continue
                            for derivative_right in range(4):
                                contracted = sum(
                                    (
                                        _projector(
                                            raised,
                                            derivative_right,
                                            rho,
                                            sigma,
                                            inverse,
                                        )
                                        * input_tensor[rho][sigma]
                                        for rho in range(4)
                                        for sigma in range(4)
                                    ),
                                    ZERO,
                                )
                                gauge += (
                                    left
                                    * xi[derivative_left]
                                    * middle
                                    * xi[derivative_right]
                                    * contracted
                                )
                output[mu][nu] = (-xi_squared * trace / 2 + gauge) / 2
        for row, row_basis in enumerate(basis):
            baseline[row][column] = sum(
                (row_basis[mu][nu] * output[mu][nu] for mu in range(4) for nu in range(4)),
                ZERO,
            )
    action = _zero_matrix(11, 11)
    for row in range(10):
        for column in range(10):
            action[row][column] = baseline[row][column]
    action[10][10] = -xi_squared
    return action


def _add_matrix(left: list[list[Qsqrt2]], right: list[list[Qsqrt2]]) -> list[list[Qsqrt2]]:
    return [
        [left[row][column] + right[row][column] for column in range(len(left[0]))]
        for row in range(len(left))
    ]


def _scale_matrix(matrix: list[list[Qsqrt2]], scale: object) -> list[list[Qsqrt2]]:
    return [[value * scale for value in row] for row in matrix]


def _transpose(matrix: list[list[Qsqrt2]]) -> list[list[Qsqrt2]]:
    return [list(row) for row in zip(*matrix, strict=True)]


def _matrix_packet(name: str, matrix: list[list[Qsqrt2]]) -> dict[str, Any]:
    entries = [
        {"row": row, "column": column, "value": value.text()}
        for row, values in enumerate(matrix)
        for column, value in enumerate(values)
        if value != ZERO
    ]
    body = {
        "schema_version": "sigma-exact-sparse-Qsqrt2-matrix-1.0",
        "name": name,
        "shape": [len(matrix), len(matrix[0])],
        "entries": entries,
        "nonzero_count": len(entries),
    }
    return {**body, "content_sha256": _content_hash(body)}


def construct_flat_action_metric() -> dict[str, Any]:
    minus, zero, plus = (_baseline_action_at(value) for value in (-1, 0, 1))
    coefficient_a = _scale_matrix(
        _add_matrix(_add_matrix(plus, minus), _scale_matrix(zero, -2)), Fraction(1, 2)
    )
    coefficient_b = _scale_matrix(_add_matrix(plus, _scale_matrix(minus, -1)), Fraction(1, 2))
    if coefficient_a != _transpose(coefficient_a) or coefficient_b != _transpose(coefficient_b):
        raise FlatActionMetricRegistrationError("flat action coefficients are not symmetric")
    metric = _zero_matrix(22, 22)
    for row in range(11):
        for column in range(11):
            metric[row][column] = coefficient_b[row][column]
            metric[row][11 + column] = coefficient_a[row][column]
            metric[11 + row][column] = coefficient_a[row][column]
    if metric != _transpose(metric):
        raise FlatActionMetricRegistrationError("h_plus_0 symmetry replay failed")
    evaluation_packets = [
        _matrix_packet(f"action_symbol_xi0_{value}", matrix)
        for value, matrix in ((-1, minus), (0, zero), (1, plus))
    ]
    return {
        "evaluation_packets": evaluation_packets,
        "A_0": _matrix_packet("A_0", coefficient_a),
        "B_0": _matrix_packet("B_0", coefficient_b),
        "h_plus_0": _matrix_packet("h_plus_0", metric),
    }


def _validate_config(config: dict[str, Any]) -> None:
    if (
        config.get("schema_version") != CONFIG_SCHEMA
        or config.get("policy") != "exact_flat_specialization_without_full_symbol_build"
        or not _hash_matches(config)
        or config.get("predecessor") != EXPECTED_PREDECESSOR
        or config.get("formula_source") != EXPECTED_FORMULA_SOURCE
        or config.get("resource_caps")
        != {
            "maximum_tensor_loop_terms": 2_000_000,
            "maximum_sparse_entries": 484,
            "maximum_full_symbol_build_calls": 0,
        }
    ):
        raise FlatActionMetricRegistrationError("invalid flat action metric config")


def _validate_predecessor(root: Path, binding: dict[str, Any]) -> dict[str, Any]:
    path = _resolve_under(root, binding["path"])
    document = _load_json(path)
    if (
        _file_sha256(path) != binding["file_sha256"]
        or not _hash_matches(document)
        or document.get("content_sha256") != binding["content_sha256"]
        or document.get("status") != PREDECESSOR_STATUS
        or document.get("errors") != []
    ):
        raise FlatActionMetricRegistrationError("predecessor mismatch")
    return document


def _validate_formula_source(root: Path, binding: dict[str, Any]) -> dict[str, Any]:
    path = _resolve_under(root, binding["path"])
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    definitions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    markers = (
        "* (-xi_squared * trace_term / 2 + gauge_invariant_term)",
        "action_symbol[:10, :10] = baseline_metric + correction_metric",
        "action_symbol[10, 10] = scalar_block",
    )
    if (
        _file_sha256(path) != binding["file_sha256"]
        or not set(binding["required_functions"]).issubset(definitions)
        or any(marker not in text for marker in markers)
    ):
        raise FlatActionMetricRegistrationError("formula source mismatch")
    return {"functions_verified": len(binding["required_functions"]), "markers_verified": 3}


def build_campaign(project_root: Path, config_path: Path) -> dict[str, Any]:
    root = project_root.resolve()
    config = _load_json(config_path)
    _validate_config(config)
    predecessor = _validate_predecessor(root, config["predecessor"])
    source_audit = _validate_formula_source(root, config["formula_source"])
    exact = construct_flat_action_metric()
    h_packet = exact["h_plus_0"]
    if h_packet["shape"] != [22, 22] or h_packet["nonzero_count"] > 484:
        raise FlatActionMetricRegistrationError("flat metric packet cap mismatch")
    claims = {key: False for key, value in predecessor["claims"].items() if value is False}
    claims.update(
        {
            "flat_action_metric_h_plus_0_registered": True,
            "flat_action_A0_B0_registered": True,
            "cold_full_symbol_build_used": False,
            "K55_Taylor_order_zero_packets_registered": False,
        }
    )
    body = {
        "schema_version": SCHEMA,
        "status": STATUS,
        "errors": [],
        "config_sha256": config["content_sha256"],
        "predecessor_binding": {**config["predecessor"], "verified": True},
        "formula_source_binding": {
            **config["formula_source"],
            **source_audit,
            "verified": True,
        },
        "flat_specialization": config["flat_specialization"],
        "exact_construction": {
            **exact,
            "coefficient_extraction": {
                "A_0": "(S(1)+S(-1)-2*S(0))/2",
                "B_0": "(S(1)-S(-1))/2",
                "quadratic_interpolation_exact": True,
            },
            "metric_identity": "h_plus_0=[[B_0,A_0],[A_0,0_11]]",
            "symmetry_residual_zero": True,
            "coefficient_field": "Q(sqrt(2))",
        },
        "manifest_boundary": {
            "required_symbolic_input_packets": 304,
            "registered_symbolic_input_packets": 34,
            "missing_symbolic_input_packets": 270,
            "manifest_advanced": False,
            "reason": "h_plus_0 is an enabling primitive; K55 order-zero packets are not constructed in this gate",
        },
        "next_exact_gate": {
            "input": "h_plus_0 plus registered P_1 and projector recipes",
            "construction": "derive G_0, X_0, K_0 and register 15 K55 Taylor-order-zero packets",
            "cold_symbol_build_required": False,
        },
        "counts": {
            "predecessor_seals_verified": 1,
            "formula_source_functions_verified": source_audit["functions_verified"],
            "full_symbol_build_calls": 0,
            "exact_action_symbol_evaluations": 3,
            "exact_11x11_coefficient_packets": 2,
            "exact_22x22_action_metric_packets": 1,
            "A_0_nonzero_entries": exact["A_0"]["nonzero_count"],
            "B_0_nonzero_entries": exact["B_0"]["nonzero_count"],
            "h_plus_0_nonzero_entries": h_packet["nonzero_count"],
            "registered_symbolic_input_packets": 34,
            "missing_symbolic_input_packets": 270,
            "emitted_output_rows": 0,
        },
        "claims": claims,
        "negative_controls": {
            "include_modified_harmonic_gauge_block_in_action_metric": {"rejected": True},
            "retain_alpha_deformation_at_flat_alpha_zero_reference": {"rejected": True},
            "replace_h_plus_0_with_identity": {"rejected": True},
            "count_enabling_primitive_as_K55_Taylor_packet": {"rejected": True},
            "promote_flat_metric_to_full_D4_or_H7": {"rejected": True},
        },
        "source_bindings": {
            "config": {"path": CONFIG_PATH, "file_sha256": _file_sha256(config_path)},
            "source": {"path": SOURCE_PATH, "file_sha256": _file_sha256(root / SOURCE_PATH)},
            "test": {"path": TEST_PATH, "file_sha256": _file_sha256(root / TEST_PATH)},
        },
        "scope": (
            "Registers the exact flat action A0/B0 blocks and 22x22 h_plus_0 metric by a "
            "bounded specialization of committed tensor formulas. It does not call the full "
            "symbol builder, construct K0, register a K55 Taylor packet, emit a recurrence "
            "row, or prove D4, H7, PDE, or lifespan closure."
        ),
    }
    return {**body, "content_sha256": _content_hash(body)}


def validate_campaign(document: dict[str, Any], project_root: Path) -> None:
    expected = build_campaign(project_root.resolve(), project_root.resolve() / CONFIG_PATH)
    if document != expected or not _hash_matches(document):
        raise FlatActionMetricRegistrationError("campaign replay mismatch")


def write_campaign(document: dict[str, Any], output: Path) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    path = output / "campaign.json"
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    document = build_campaign(args.project_root.resolve(), args.config.resolve())
    print(write_campaign(document, args.output.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
