"""AST-executed quantum/entity/wave gravity diagnostics and source gates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import itertools
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

CONFIG_PATH = Path("configs/open_gravity_quantum_entity_wave_atlas_v3.json")
MODULE_PATH = Path("src/sigma_theory_compiler/open_gravity_quantum_entity_wave_atlas_v3.py")
TEST_PATH = Path("tests/test_open_gravity_quantum_entity_wave_atlas_v3.py")
OUTPUT_PATH = Path("runs/gravity/theory/open-gravity-quantum-entity-wave-atlas-v3/receipt.json")
ARTIFACT_DIR = OUTPUT_PATH.parent / "artifacts"
CONFIG_SCHEMA = "invariant-open-gravity-quantum-entity-wave-programs-config-3.0"
RECEIPT_SCHEMA = "invariant-open-gravity-quantum-entity-wave-programs-receipt-3.0"
DECISION = (
    "PASS_INTERNAL_AST_DIMENSION_MAPPING_AND_SOURCE_GATE_REPAIR_"
    "REDUCED_DIAGNOSTICS_ONLY_NO_EMPIRICAL_EXECUTION"
)

CARD_IDS = (
    "Q00_GR_COHERENT_SPIN2_CONTROL",
    "Q01_MASSIVE_SPIN2",
    "Q02_SCALAR_VECTOR_TENSOR_MIXTURE",
    "Q03_DISCRETE_GRAVITY_IMPULSES",
    "Q04_CLASSICAL_STOCHASTIC_METRIC",
    "Q05_SEMICLASSICAL_EXPECTATION_SOURCE",
    "Q06_GRAVITATIONAL_COLLAPSE_DECOHERENCE",
    "Q07_ENTANGLEMENT_MEDIATED_GRAVITY",
    "Q08_EMERGENT_SUPERFLUID_MEDIUM",
    "Q09_DISPERSIVE_GRAVITY_WAVE_PACKET",
    "Q10_POLARIZATION_BIREFRINGENT_GRAVITY",
    "Q11_FINITE_OCCUPATION_COHERENCE",
    "Q12_QUANTIZED_TIMEWELL_MEMORY_MODE",
    "Q13_QUANTIZED_CAPTURE_JUMP_MEMORY",
    "Q14_POSTQUANTUM_CLASSICAL_QUANTUM_GRAVITY",
    "Q15_KTM_CLASSICAL_CHANNEL_BOUNDARY",
)


class QuantumAtlasV3Error(RuntimeError):
    """Raised when a program, dimension, mapping, source gate, or seal fails."""


@dataclass(frozen=True)
class TypedValue:
    value: Any
    type: str
    shape: tuple[int, ...]
    unit: tuple[Fraction, Fraction, Fraction]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise QuantumAtlasV3Error(message)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _self_hash(value: Mapping[str, Any]) -> str:
    body = dict(value)
    body["content_sha256"] = ""
    return _sha256_bytes(_canonical(body))


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise QuantumAtlasV3Error(f"invalid JSON: {path}") from exc
    _require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def _unit(raw: Sequence[int | float | Fraction]) -> tuple[Fraction, Fraction, Fraction]:
    _require(len(raw) == 3, "unit is not an L,M,T triple")
    return tuple(Fraction(str(value)) for value in raw)  # type: ignore[return-value]


def _unit_add(
    left: tuple[Fraction, Fraction, Fraction],
    right: tuple[Fraction, Fraction, Fraction],
) -> tuple[Fraction, Fraction, Fraction]:
    return tuple(a + b for a, b in zip(left, right, strict=True))  # type: ignore[return-value]


def _unit_sub(
    left: tuple[Fraction, Fraction, Fraction],
    right: tuple[Fraction, Fraction, Fraction],
) -> tuple[Fraction, Fraction, Fraction]:
    return tuple(a - b for a, b in zip(left, right, strict=True))  # type: ignore[return-value]


def _unit_scale(
    unit: tuple[Fraction, Fraction, Fraction], exponent: Fraction
) -> tuple[Fraction, Fraction, Fraction]:
    return tuple(value * exponent for value in unit)  # type: ignore[return-value]


DIMENSIONLESS = _unit([0, 0, 0])


def _shape(value: Any) -> tuple[int, ...]:
    if isinstance(value, (bool, int, float)):
        return ()
    if isinstance(value, list):
        if not value:
            return (0,)
        child_shapes = {_shape(item) for item in value}
        _require(len(child_shapes) == 1, "ragged array")
        child_shape = next(iter(child_shapes))
        return (len(value), *child_shape)
    raise QuantumAtlasV3Error(f"unsupported runtime value: {type(value).__name__}")


def _type_from_shape(shape: tuple[int, ...]) -> str:
    if not shape:
        return "scalar"
    if len(shape) == 1:
        return "vector"
    if len(shape) == 2:
        return "matrix"
    raise QuantumAtlasV3Error(f"unsupported tensor rank: {len(shape)}")


def _all_numbers(value: Any) -> list[float]:
    if isinstance(value, bool):
        return [float(value)]
    if isinstance(value, (int, float)):
        _require(math.isfinite(float(value)), "nonfinite numeric value")
        return [float(value)]
    if isinstance(value, list):
        return [number for item in value for number in _all_numbers(item)]
    raise QuantumAtlasV3Error("nonnumeric value")


def _elementwise_unary(value: Any, operation: Any) -> Any:
    if isinstance(value, list):
        return [_elementwise_unary(item, operation) for item in value]
    return operation(value)


def _elementwise_binary(left: Any, right: Any, operation: Any) -> Any:
    if isinstance(left, list) and isinstance(right, list):
        _require(_shape(left) == _shape(right), "elementwise shape mismatch")
        return [_elementwise_binary(a, b, operation) for a, b in zip(left, right, strict=True)]
    if isinstance(left, list):
        return [_elementwise_binary(item, right, operation) for item in left]
    if isinstance(right, list):
        return [_elementwise_binary(left, item, operation) for item in right]
    return operation(left, right)


def _matrix_multiply(left: Any, right: Any) -> Any:
    left_shape, right_shape = _shape(left), _shape(right)
    if len(left_shape) == 2 and len(right_shape) == 1:
        _require(left_shape[1] == right_shape[0], "matrix-vector shape mismatch")
        return [sum(row[index] * right[index] for index in range(right_shape[0])) for row in left]
    if len(left_shape) == 2 and len(right_shape) == 2:
        _require(left_shape[1] == right_shape[0], "matrix-matrix shape mismatch")
        return [
            [
                sum(left[i][k] * right[k][j] for k in range(left_shape[1]))
                for j in range(right_shape[1])
            ]
            for i in range(left_shape[0])
        ]
    raise QuantumAtlasV3Error("matmul requires matrix-vector or matrix-matrix")


def _determinant(matrix: Sequence[Sequence[float]]) -> float:
    size = len(matrix)
    _require(
        size > 0 and all(len(row) == size for row in matrix), "determinant requires square matrix"
    )
    if size == 1:
        return float(matrix[0][0])
    if size == 2:
        return float(matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0])
    return sum(
        ((-1) ** column)
        * matrix[0][column]
        * _determinant([list(row[:column]) + list(row[column + 1 :]) for row in matrix[1:]])
        for column in range(size)
    )


def _inverse(matrix: Sequence[Sequence[float]]) -> list[list[float]]:
    size = len(matrix)
    _require(size > 0 and all(len(row) == size for row in matrix), "inverse requires square matrix")
    work = [
        list(map(float, row)) + [1.0 if i == j else 0.0 for j in range(size)]
        for i, row in enumerate(matrix)
    ]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(work[row][column]))
        _require(abs(work[pivot][column]) > 1e-14, "singular matrix")
        work[column], work[pivot] = work[pivot], work[column]
        scale = work[column][column]
        work[column] = [value / scale for value in work[column]]
        for row in range(size):
            if row == column:
                continue
            factor = work[row][column]
            work[row] = [
                value - factor * pivot_value
                for value, pivot_value in zip(work[row], work[column], strict=True)
            ]
    return [row[size:] for row in work]


def _rank(matrix: Sequence[Sequence[float]], tolerance: float) -> int:
    work = [list(map(float, row)) for row in matrix]
    if not work:
        return 0
    rows, columns = len(work), len(work[0])
    rank = 0
    for column in range(columns):
        if rank >= rows:
            break
        pivot = max(range(rank, rows), key=lambda row: abs(work[row][column]))
        if abs(work[pivot][column]) <= tolerance:
            continue
        work[rank], work[pivot] = work[pivot], work[rank]
        scale = work[rank][column]
        work[rank] = [value / scale for value in work[rank]]
        for row in range(rows):
            if row == rank:
                continue
            factor = work[row][column]
            work[row] = [
                value - factor * pivot_value
                for value, pivot_value in zip(work[row], work[rank], strict=True)
            ]
        rank += 1
    return rank


def _is_symmetric(matrix: Any, tolerance: float) -> bool:
    shape = _shape(matrix)
    if len(shape) != 2 or shape[0] != shape[1]:
        return False
    return all(
        abs(matrix[i][j] - matrix[j][i]) <= tolerance
        for i in range(shape[0])
        for j in range(shape[1])
    )


def _is_psd(matrix: Any, tolerance: float, *, strict: bool = False) -> bool:
    if not _is_symmetric(matrix, tolerance):
        return False
    size = len(matrix)
    for order in range(1, size + 1):
        for subset in itertools.combinations(range(size), order):
            minor = [[matrix[i][j] for j in subset] for i in subset]
            determinant = _determinant(minor)
            if strict and determinant <= tolerance:
                return False
            if not strict and determinant < -tolerance:
                return False
    return True


def _domain_ok(value: Any, domain: Mapping[str, Any]) -> bool:
    kind = domain["kind"]
    numbers = _all_numbers(value)
    tolerance = 1e-12
    if kind == "real":
        return True
    if kind == "positive":
        return len(numbers) == 1 and numbers[0] > 0
    if kind == "nonnegative":
        return len(numbers) == 1 and numbers[0] >= 0
    if kind == "all_positive":
        return all(number > 0 for number in numbers)
    if kind == "all_nonnegative":
        return all(number >= 0 for number in numbers)
    if kind == "simplex":
        return all(number >= 0 for number in numbers) and abs(sum(numbers) - 1.0) <= tolerance
    if kind == "nonnegative_integers":
        return all(number >= 0 and number.is_integer() for number in numbers)
    if kind == "positive_integers":
        return all(number > 0 and number.is_integer() for number in numbers)
    if kind == "symmetric_positive_semidefinite":
        return _is_psd(value, tolerance)
    if kind == "symmetric_positive_definite":
        return _is_psd(value, tolerance, strict=True)
    raise QuantumAtlasV3Error(f"unknown domain: {kind}")


def _typed_from_declaration(declaration: Mapping[str, Any], value: Any | None = None) -> TypedValue:
    observed = declaration["fixture"] if value is None else value
    observed_shape = _shape(observed)
    expected_shape = tuple(declaration["shape"])
    _require(observed_shape == expected_shape, f"shape mismatch for {declaration['name']}")
    expected_type = declaration["type"]
    _require(
        _type_from_shape(observed_shape) == expected_type,
        f"type mismatch for {declaration['name']}",
    )
    _require(
        _domain_ok(observed, declaration["domain"]), f"domain violation for {declaration['name']}"
    )
    return TypedValue(observed, expected_type, expected_shape, _unit(declaration["unit"]))


def _collect_vars(node: Any) -> set[str]:
    if isinstance(node, list):
        return set().union(*(_collect_vars(item) for item in node)) if node else set()
    if not isinstance(node, dict):
        return set()
    found = {node["name"]} if node.get("op") == "var" else set()
    for value in node.values():
        found.update(_collect_vars(value))
    return found


def _same_unit(values: Sequence[TypedValue], label: str) -> tuple[Fraction, Fraction, Fraction]:
    _require(bool(values), f"{label} has no arguments")
    unit = values[0].unit
    _require(all(value.unit == unit for value in values), f"unit mismatch in {label}")
    return unit


def evaluate_ast(node: Mapping[str, Any], environment: Mapping[str, TypedValue]) -> TypedValue:
    op = node.get("op")
    if op == "var":
        name = node["name"]
        _require(name in environment, f"undeclared or unavailable variable: {name}")
        return environment[name]
    if op == "const":
        return TypedValue(node["value"], "scalar", (), _unit(node["unit"]))
    if op in {"vector", "matrix"}:
        raw_items = node["items"] if op == "vector" else node["rows"]
        evaluated = (
            [[evaluate_ast(item, environment) for item in row] for row in raw_items]
            if op == "matrix"
            else [evaluate_ast(item, environment) for item in raw_items]
        )
        flattened = [item for row in evaluated for item in row] if op == "matrix" else evaluated
        unit = _same_unit(flattened, op)
        _require(all(item.type == "scalar" for item in flattened), f"{op} literals require scalars")
        value = (
            [[item.value for item in row] for row in evaluated]
            if op == "matrix"
            else [item.value for item in evaluated]
        )
        shape = _shape(value)
        return TypedValue(value, _type_from_shape(shape), shape, unit)
    if op in {"add", "mul", "max"}:
        values = [evaluate_ast(argument, environment) for argument in node["args"]]
        if op in {"add", "max"}:
            unit = _same_unit(values, op)
        else:
            unit = DIMENSIONLESS
            for item in values:
                unit = _unit_add(unit, item.unit)
        result = values[0].value
        operation = (
            (lambda a, b: a + b) if op == "add" else ((lambda a, b: a * b) if op == "mul" else max)
        )
        for item in values[1:]:
            result = _elementwise_binary(result, item.value, operation)
        shape = _shape(result)
        return TypedValue(result, _type_from_shape(shape), shape, unit)
    if op in {"sub", "div"}:
        left_node, right_node = (
            (node["left"], node["right"]) if op == "sub" else (node["num"], node["den"])
        )
        left, right = evaluate_ast(left_node, environment), evaluate_ast(right_node, environment)
        if op == "sub":
            _require(left.unit == right.unit, "unit mismatch in sub")
            result = _elementwise_binary(left.value, right.value, lambda a, b: a - b)
            unit = left.unit
        else:
            result = _elementwise_binary(left.value, right.value, lambda a, b: a / b)
            unit = _unit_sub(left.unit, right.unit)
        shape = _shape(result)
        return TypedValue(result, _type_from_shape(shape), shape, unit)
    if op == "pow":
        base = evaluate_ast(node["base"], environment)
        exponent = Fraction(str(node["exponent"]))
        result = _elementwise_unary(base.value, lambda value: value ** float(exponent))
        return TypedValue(result, base.type, base.shape, _unit_scale(base.unit, exponent))
    if op == "sqrt":
        argument = evaluate_ast(node["arg"], environment)
        _require(
            all(number >= 0 for number in _all_numbers(argument.value)), "sqrt of negative value"
        )
        result = _elementwise_unary(argument.value, math.sqrt)
        return TypedValue(
            result, argument.type, argument.shape, _unit_scale(argument.unit, Fraction(1, 2))
        )
    if op in {"exp", "sin"}:
        argument = evaluate_ast(node["arg"], environment)
        _require(argument.unit == DIMENSIONLESS, f"{op} requires dimensionless input")
        function = math.exp if op == "exp" else math.sin
        result = _elementwise_unary(argument.value, function)
        return TypedValue(result, argument.type, argument.shape, DIMENSIONLESS)
    if op in {"abs", "neg"}:
        argument = evaluate_ast(node["arg"], environment)
        function = abs if op == "abs" else (lambda value: -value)
        return TypedValue(
            _elementwise_unary(argument.value, function),
            argument.type,
            argument.shape,
            argument.unit,
        )
    if op == "matmul":
        left, right = (
            evaluate_ast(node["left"], environment),
            evaluate_ast(node["right"], environment),
        )
        result = _matrix_multiply(left.value, right.value)
        shape = _shape(result)
        return TypedValue(result, _type_from_shape(shape), shape, _unit_add(left.unit, right.unit))
    if op == "transpose":
        argument = evaluate_ast(node["arg"], environment)
        _require(len(argument.shape) == 2, "transpose requires matrix")
        value = [list(column) for column in zip(*argument.value, strict=True)]
        return TypedValue(value, "matrix", (argument.shape[1], argument.shape[0]), argument.unit)
    if op == "inverse":
        argument = evaluate_ast(node["arg"], environment)
        _require(len(argument.shape) == 2, "inverse requires matrix")
        value = _inverse(argument.value)
        return TypedValue(value, "matrix", argument.shape, _unit_scale(argument.unit, Fraction(-1)))
    if op == "dot":
        left, right = (
            evaluate_ast(node["left"], environment),
            evaluate_ast(node["right"], environment),
        )
        _require(
            left.type == right.type == "vector" and left.shape == right.shape, "dot shape mismatch"
        )
        value = sum(a * b for a, b in zip(left.value, right.value, strict=True))
        return TypedValue(value, "scalar", (), _unit_add(left.unit, right.unit))
    if op in {"index", "slice"}:
        argument = evaluate_ast(node["value"], environment)
        _require(argument.type == "vector", f"{op} requires vector")
        if op == "index":
            return TypedValue(argument.value[node["index"]], "scalar", (), argument.unit)
        value = argument.value[node["start"] : node["stop"]]
        return TypedValue(value, "vector", (len(value),), argument.unit)
    if op == "rank":
        matrix, tolerance = (
            evaluate_ast(node["matrix"], environment),
            evaluate_ast(node["tolerance"], environment),
        )
        _require(matrix.type == "matrix" and tolerance.type == "scalar", "rank arguments invalid")
        _require(tolerance.unit == DIMENSIONLESS, "rank tolerance must be dimensionless")
        return TypedValue(_rank(matrix.value, tolerance.value), "scalar", (), DIMENSIONLESS)
    if op == "atan2":
        y_value, x_value = (
            evaluate_ast(node["y"], environment),
            evaluate_ast(node["x"], environment),
        )
        _require(
            y_value.type == x_value.type == "scalar" and y_value.unit == x_value.unit,
            "atan2 arguments invalid",
        )
        return TypedValue(math.atan2(y_value.value, x_value.value), "scalar", (), DIMENSIONLESS)
    if op in {"min", "max_abs"}:
        argument = evaluate_ast(node["arg"], environment)
        numbers = _all_numbers(argument.value)
        value = min(numbers) if op == "min" else max(abs(number) for number in numbers)
        return TypedValue(value, "scalar", (), argument.unit)
    if op == "poisson_pmf":
        mean, indices = (
            evaluate_ast(node["mean"], environment),
            evaluate_ast(node["n"], environment),
        )
        _require(mean.type == "scalar" and indices.type == "vector", "Poisson arguments invalid")
        _require(
            mean.unit == indices.unit == DIMENSIONLESS, "Poisson arguments must be dimensionless"
        )
        _require(mean.value >= 0, "negative Poisson mean")
        values = [
            math.exp(-mean.value) * mean.value ** int(index) / math.factorial(int(index))
            for index in indices.value
        ]
        return TypedValue(values, "vector", indices.shape, DIMENSIONLESS)
    if op in {"is_symmetric", "is_psd"}:
        argument = evaluate_ast(node["arg"], environment)
        _require(argument.type == "matrix", f"{op} requires matrix")
        value = (
            _is_symmetric(argument.value, node["tolerance"])
            if op == "is_symmetric"
            else _is_psd(argument.value, node["tolerance"])
        )
        return TypedValue(value, "scalar", (), DIMENSIONLESS)
    if op in {"gte", "lte", "lt"}:
        left, right = (
            evaluate_ast(node["left"], environment),
            evaluate_ast(node["right"], environment),
        )
        _require(
            left.type == right.type == "scalar" and left.unit == right.unit,
            f"{op} arguments invalid",
        )
        operation = {
            "gte": lambda a, b: a >= b,
            "lte": lambda a, b: a <= b,
            "lt": lambda a, b: a < b,
        }[op]
        return TypedValue(operation(left.value, right.value), "scalar", (), DIMENSIONLESS)
    if op in {"all_gte", "all_lte"}:
        left = evaluate_ast(node["left"], environment)
        raw_right = node["right"]
        right = (
            evaluate_ast(raw_right, environment)
            if isinstance(raw_right, dict)
            else TypedValue(raw_right, "scalar", (), DIMENSIONLESS)
        )
        _require(left.unit == right.unit, f"{op} unit mismatch")
        operation = (lambda a, b: a >= b) if op == "all_gte" else (lambda a, b: a <= b)
        result = _elementwise_binary(left.value, right.value, operation)
        return TypedValue(
            all(bool(value) for value in _all_numbers(result)), "scalar", (), DIMENSIONLESS
        )
    if op in {"all_nonnegative", "all_positive"}:
        argument = evaluate_ast(node["arg"], environment)
        values = _all_numbers(argument.value)
        result = (
            all(value >= 0 for value in values)
            if op == "all_nonnegative"
            else all(value > 0 for value in values)
        )
        return TypedValue(result, "scalar", (), DIMENSIONLESS)
    if op == "contains":
        argument = evaluate_ast(node["value"], environment)
        _require(argument.type == "vector", "contains requires vector")
        return TypedValue(node["item"] in argument.value, "scalar", (), DIMENSIONLESS)
    if op == "approx_eq":
        left, right = (
            evaluate_ast(node["left"], environment),
            evaluate_ast(node["right"], environment),
        )
        _require(
            left.type == right.type == "scalar" and left.unit == right.unit,
            "approx_eq arguments invalid",
        )
        return TypedValue(
            abs(left.value - right.value) <= node["tolerance"], "scalar", (), DIMENSIONLESS
        )
    if op == "in_interval":
        argument = evaluate_ast(node["value"], environment)
        _require(
            argument.type == "scalar" and argument.unit == DIMENSIONLESS,
            "interval argument invalid",
        )
        return TypedValue(node["min"] <= argument.value <= node["max"], "scalar", (), DIMENSIONLESS)
    if op == "all_in_interval":
        argument = evaluate_ast(node["value"], environment)
        _require(argument.unit == DIMENSIONLESS, "interval vector must be dimensionless")
        return TypedValue(
            all(node["min"] <= value <= node["max"] for value in _all_numbers(argument.value)),
            "scalar",
            (),
            DIMENSIONLESS,
        )
    raise QuantumAtlasV3Error(f"unknown AST operation: {op}")


def _printable_unit(unit: tuple[Fraction, Fraction, Fraction]) -> list[int | str]:
    return [
        value.numerator if value.denominator == 1 else f"{value.numerator}/{value.denominator}"
        for value in unit
    ]


def validate_and_execute_card(
    card: Mapping[str, Any], overrides: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    _require(card.get("implementation_level") and card.get("claim"), "implementation claim missing")
    conditions = card.get("conditions")
    _require(
        isinstance(conditions, dict)
        and set(conditions) == {"initial", "boundary"}
        and conditions["initial"].get("status")
        and conditions["boundary"].get("status"),
        f"condition status missing: {card.get('id')}",
    )
    declarations = card.get("variables")
    _require(
        isinstance(declarations, list) and declarations, f"variables missing: {card.get('id')}"
    )
    names = [row["name"] for row in declarations]
    _require(len(names) == len(set(names)), f"duplicate variables: {card.get('id')}")
    environment: dict[str, TypedValue] = {}
    for declaration in declarations:
        _require(
            set(declaration) == {"name", "role", "type", "shape", "unit", "domain", "fixture"},
            f"variable typing changed: {card.get('id')}",
        )
        _require(declaration["role"] in {"parameter", "input"}, "unknown variable role")
        replacement = overrides.get(declaration["name"]) if overrides else None
        environment[declaration["name"]] = _typed_from_declaration(declaration, replacement)
    program = card.get("program")
    _require(
        isinstance(program, dict) and set(program) == {"outputs", "assertions"},
        "program shape changed",
    )
    output_names = [row["name"] for row in program["outputs"]]
    _require(len(output_names) == len(set(output_names)), "duplicate output")
    declared_variables = set(names)
    used_input_variables: set[str] = set()
    output_environment = dict(environment)
    outputs: list[dict[str, Any]] = []
    for output in program["outputs"]:
        _require(
            set(output) == {"name", "type", "shape", "unit", "ast"},
            f"output typing changed: {card.get('id')}",
        )
        ast_vars = _collect_vars(output["ast"])
        _require(
            ast_vars <= set(output_environment),
            f"forward or undeclared variable in {output['name']}",
        )
        used_input_variables.update(ast_vars & declared_variables)
        result = evaluate_ast(output["ast"], output_environment)
        _require(result.type == output["type"], f"output type mismatch: {output['name']}")
        _require(result.shape == tuple(output["shape"]), f"output shape mismatch: {output['name']}")
        _require(result.unit == _unit(output["unit"]), f"output unit mismatch: {output['name']}")
        output_environment[output["name"]] = result
        outputs.append(
            {
                "name": output["name"],
                "value": result.value,
                "type": result.type,
                "shape": list(result.shape),
                "unit": _printable_unit(result.unit),
                "ast_sha256": _sha256_bytes(_canonical(output["ast"])),
            }
        )
    assertion_rows = []
    for assertion in program["assertions"]:
        _require(set(assertion) == {"id", "ast"}, "assertion shape changed")
        ast_vars = _collect_vars(assertion["ast"])
        _require(
            ast_vars <= set(output_environment), f"undeclared assertion variable: {assertion['id']}"
        )
        used_input_variables.update(ast_vars & declared_variables)
        result = evaluate_ast(assertion["ast"], output_environment)
        _require(
            result.type == "scalar" and isinstance(result.value, bool), "assertion is not boolean"
        )
        _require(result.value, f"assertion failed: {card['id']}:{assertion['id']}")
        assertion_rows.append(
            {
                "id": assertion["id"],
                "passed": True,
                "ast_sha256": _sha256_bytes(_canonical(assertion["ast"])),
            }
        )
    _require(
        used_input_variables == declared_variables,
        f"declared variables not exactly used: {card['id']} missing={sorted(declared_variables - used_input_variables)}",
    )
    return {
        "card_id": card["id"],
        "implementation_level": card["implementation_level"],
        "program_sha256": _sha256_bytes(_canonical(program)),
        "outputs": outputs,
        "assertions": assertion_rows,
        "dimension_check": "PASS",
        "domain_shape_check": "PASS",
    }


def _output_value(execution: Mapping[str, Any], name: str) -> Any:
    matches = [row["value"] for row in execution["outputs"] if row["name"] == name]
    _require(len(matches) == 1, f"output not unique: {name}")
    return matches[0]


def _max_difference(left: Any, right: Any) -> float:
    _require(_shape(left) == _shape(right), "comparison shape mismatch")
    if isinstance(left, list):
        return max((_max_difference(a, b) for a, b in zip(left, right, strict=True)), default=0.0)
    return abs(float(left) - float(right))


def execute_parameter_mapping(
    config: Mapping[str, Any], mapping: Mapping[str, Any]
) -> dict[str, Any]:
    cards = {card["id"]: card for card in config["cards"]}
    _require(
        mapping["source_card"] in cards and mapping["target_card"] in cards, "mapping card missing"
    )
    source, target = cards[mapping["source_card"]], cards[mapping["target_card"]]
    source_declarations = {row["name"]: row for row in source["variables"]}
    target_declarations = {row["name"]: row for row in target["variables"]}
    _require(
        set(mapping["source_classification"]) == set(source_declarations),
        "source classification coverage mismatch",
    )
    _require(
        set(mapping["target_assignments"]) == set(target_declarations),
        "target assignment coverage mismatch",
    )
    source_environment = {
        name: _typed_from_declaration(declaration)
        for name, declaration in source_declarations.items()
    }
    overrides: dict[str, Any] = {}
    assignment_rows = []
    for target_name, ast in mapping["target_assignments"].items():
        used = _collect_vars(ast)
        _require(
            used <= set(source_environment), f"mapping uses unknown source variable: {target_name}"
        )
        result = evaluate_ast(ast, source_environment)
        declaration = target_declarations[target_name]
        _require(result.type == declaration["type"], f"mapping type mismatch: {target_name}")
        _require(
            result.shape == tuple(declaration["shape"]), f"mapping shape mismatch: {target_name}"
        )
        _require(result.unit == _unit(declaration["unit"]), f"mapping unit mismatch: {target_name}")
        _require(
            _domain_ok(result.value, declaration["domain"]),
            f"mapping domain mismatch: {target_name}",
        )
        overrides[target_name] = result.value
        assignment_rows.append(
            {
                "target": target_name,
                "source_variables": sorted(used),
                "type": result.type,
                "shape": list(result.shape),
                "unit": _printable_unit(result.unit),
                "value": result.value,
            }
        )
    source_execution = validate_and_execute_card(source)
    target_execution = validate_and_execute_card(target, overrides)
    output = mapping["compare_output"]
    residual = _max_difference(
        _output_value(source_execution, output), _output_value(target_execution, output)
    )
    _require(residual <= 1e-12, f"mapping output mismatch: {mapping['id']}")
    return {
        "mapping_id": mapping["id"],
        "status": mapping["status"],
        "source_card": source["id"],
        "target_card": target["id"],
        "compare_output": output,
        "max_abs_residual": residual,
        "source_key_coverage": "EXACT",
        "target_key_coverage": "EXACT",
        "type_shape_unit_domain_checks": "PASS",
        "assignments": assignment_rows,
        "scope": mapping["scope"],
    }


def validate_config(config: Mapping[str, Any], base: Path | None = None) -> None:
    _require(config.get("schema_version") == CONFIG_SCHEMA, "config schema changed")
    _require(
        config.get("analysis_id") == "open-gravity-quantum-entity-wave-atlas-v3",
        "analysis ID changed",
    )
    _require(config.get("status") == "FROZEN_INTERNAL_AST_AND_SOURCE_GATE_REPAIR", "status changed")
    _require(
        config.get("package")
        == {
            "module_path": MODULE_PATH.as_posix(),
            "test_path": TEST_PATH.as_posix(),
            "output_path": OUTPUT_PATH.as_posix(),
            "artifact_directory": ARTIFACT_DIR.as_posix(),
        },
        "package paths changed",
    )
    predecessors = config.get("predecessors")
    _require(
        isinstance(predecessors, list) and len(predecessors) == 2, "predecessor inventory changed"
    )
    if base is not None:
        for predecessor in predecessors:
            path = base / predecessor["path"]
            _require(path.is_file(), f"predecessor missing: {path}")
            _require(
                _sha256_file(path) == predecessor["raw_sha256"],
                f"predecessor bytes changed: {path}",
            )
            _require(
                _read_json(path).get("content_sha256") == predecessor["content_sha256"],
                f"predecessor content changed: {path}",
            )
    contract = config.get("program_contract")
    _require(
        isinstance(contract, dict)
        and contract.get("unit_basis") == ["L", "M", "T"]
        and contract.get("fixtures_execute_stored_AST_only") is True
        and contract.get("card_specific_evaluators") is False
        and contract.get("all_variables_declared") is True
        and contract.get("all_declared_variables_used") is True,
        "program contract changed",
    )
    scope = config.get("gaussian_scope")
    _require(
        isinstance(scope, dict)
        and len(scope.get("required", [])) == 6
        and "commuting" in scope["required"][0]
        and "POVM" in scope["required"][1]
        and "back-action" in scope["required"][4]
        and scope.get("novelty_claimed") is False,
        "Gaussian scope widened",
    )
    cards = config.get("cards")
    _require(
        isinstance(cards, list) and tuple(card.get("id") for card in cards) == CARD_IDS,
        "card inventory changed",
    )
    required_card_keys = {
        "id",
        "implementation_level",
        "claim",
        "variables",
        "conditions",
        "program",
        "falsifier_manifest",
        "data_readiness",
    }
    for card in cards:
        _require(set(card) == required_card_keys, f"card fields changed: {card.get('id')}")
        validate_and_execute_card(card)
    card_by_id = {card["id"]: card for card in cards}
    _require(
        card_by_id["Q14_POSTQUANTUM_CLASSICAL_QUANTUM_GRAVITY"]["implementation_level"]
        == "REDUCED_DIAGNOSTIC_NOT_CQ_GENERATOR",
        "Q14 overclaimed",
    )
    _require(
        card_by_id["Q15_KTM_CLASSICAL_CHANNEL_BOUNDARY"]["implementation_level"]
        == "REDUCED_DIAGNOSTIC_NOT_FEEDBACK_MASTER_EQUATION",
        "Q15 overclaimed",
    )
    _require(
        "no bath spectral density" in card_by_id["Q12_QUANTIZED_TIMEWELL_MEMORY_MODE"]["claim"],
        "Q12 boundary widened",
    )
    mappings = config.get("parameter_mappings")
    _require(isinstance(mappings, list) and len(mappings) == 2, "mapping inventory changed")
    for mapping in mappings:
        _require(
            set(mapping)
            == {
                "id",
                "source_card",
                "target_card",
                "status",
                "compare_output",
                "source_classification",
                "target_assignments",
                "scope",
            },
            f"mapping fields changed: {mapping.get('id')}",
        )
        execute_parameter_mapping(config, mapping)
    boundaries = config.get("equivalence_boundaries")
    _require(
        isinstance(boundaries, list) and len(boundaries) == 3, "equivalence boundaries changed"
    )
    _require(
        boundaries[0]["status"] == "SCOPED_EXISTENCE_NOT_INSTANTIATED"
        and "Q14 is not included" in boundaries[0]["reason"],
        "EF03 scope changed",
    )
    _require(
        boundaries[1]["status"] == "NO_PARAMETER_EQUIVALENCE_CLAIMED",
        "memory equivalence overclaimed",
    )
    manifests = config.get("source_manifests")
    _require(isinstance(manifests, list) and len(manifests) == 11, "manifest inventory changed")
    manifest_by_id = {row["id"]: row for row in manifests}
    _require(len(manifest_by_id) == len(manifests), "duplicate manifest")
    _require(
        set(manifest_by_id) == {card["falsifier_manifest"] for card in cards},
        "manifest-card coverage changed",
    )
    for manifest in manifests:
        _require(
            manifest.get("status") == "SOURCE_BLOCKED",
            f"source incorrectly executable: {manifest['id']}",
        )
        _require(manifest.get("missing"), f"missing source blockers: {manifest['id']}")
    for card in cards:
        _require(card["data_readiness"] <= 2, f"readiness overstated: {card['id']}")
    _require(
        config.get("provenance", {}).get("repository_state") == "UNCOMMITTED_WORKTREE_FILES"
        and config["provenance"].get("commit_sha") is None,
        "uncommitted provenance hidden",
    )
    _require(set(config.get("access_contract", {}).values()) == {0}, "access contract changed")
    boundary = config.get("claim_boundary")
    _require(
        isinstance(boundary, dict)
        and boundary.get("AST_programs_validated") is True
        and boundary.get("target_free_fixtures_only") is True
        and boundary.get("reduced_diagnostics_not_full_theories") is True
        and boundary.get("source_execution_ready") is False
        and boundary.get("real_observational_rows_scored") is False
        and boundary.get("any_branch_empirically_supported") is False
        and boundary.get("historical_novelty_established") is False
        and boundary.get("publication_ready") is False
        and boundary.get("strict_reaudit_submission_ready") is True,
        "claim boundary widened",
    )


def load_config(root: Path | None = None) -> dict[str, Any]:
    base = (root or _repo_root()).resolve()
    config = _read_json(base / CONFIG_PATH)
    validate_config(config, base)
    return config


def counterexamples() -> list[dict[str, str]]:
    return [
        {
            "id": "CEX_AST_TEXT",
            "failure": "Treat free-text formulas or a card-specific evaluator as the executable law.",
            "repair": "The stored AST is evaluated by one generic typed interpreter and every output binds its AST hash.",
        },
        {
            "id": "CEX_DIMENSION",
            "failure": "Allow addition, transcendental functions, or mappings with incompatible units.",
            "repair": "Propagate L,M,T exponents and reject mismatches before fixture output.",
        },
        {
            "id": "CEX_GWOSC_METADATA",
            "failure": "Call an event landing page an executable response manifest.",
            "repair": "M01/M02 are SOURCE_BLOCKED until direct product URLs, hashes, waveform/statistical implementations, and validation contracts are frozen.",
        },
        {
            "id": "CEX_ARBITRARY_GAUSSIAN_MAP",
            "failure": "Map Q11 or Q14 arbitrarily into Q04.",
            "repair": "Q11 lacks the full measurement map and Q14 is only a reduced PSD diagnostic; EF03 is scoped existence, not an instantiated parameter map.",
        },
        {
            "id": "CEX_MEMORY_CQ_EQUIVALENCE",
            "failure": "Equate Q12, Q13, or Q14 because each contains noise or memory language.",
            "repair": "No parameter equivalence is claimed; their state spaces and implemented programs differ.",
        },
        {
            "id": "CEX_Q14_FULL_THEORY",
            "failure": "Present a 2x2 PSD determinant as the Oppenheim CQ generator.",
            "repair": "Q14 is explicitly REDUCED_DIAGNOSTIC and lists the unimplemented drift, Lindblad, backreaction, diffusion, and physical map.",
        },
        {
            "id": "CEX_Q15_FULL_CHANNEL",
            "failure": "Present an AM-GM curve as the KTM record/feedback master equation.",
            "repair": "Q15 is explicitly REDUCED_DIAGNOSTIC and source-blocked pending the full channel and physical conversion.",
        },
        {
            "id": "CEX_UNCOMMITTED_PROVENANCE",
            "failure": "Imply git-commit provenance for uncommitted files.",
            "repair": "Receipt claims only byte/content hashes and records commit_sha=null.",
        },
    ]


def _csv_bytes(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _gaussian_proof(config: Mapping[str, Any]) -> str:
    scope = config["gaussian_scope"]
    assumptions = "\n".join(f"- {item}" for item in scope["required"])
    exclusions = "\n".join(f"- {item}" for item in scope["excluded"])
    return f"""# Fixed finite-dimensional Gaussian pushforward

Let `h` be an arbitrary finite-dimensional Gaussian random vector with mean `m` and
covariance `C`.  For a fixed linear readout `R` and an independent Gaussian
measurement-noise vector `n` with mean `b` and covariance `N`, define the
jointly commuting output record `y = R h + n`.

For every real test vector `t`, independence gives the characteristic function

`E exp(i t^T y) = exp(i t^T(Rm+b) - t^T(R C R^T+N)t/2)`.

The characteristic function uniquely fixes the pushforward distribution:
`y` is Gaussian with mean `Rm+b` and covariance `R C R^T+N`.  Thus two models
sharing `m`, `C`, `R`, `b`, and `N` are equivalent only for this specified
measurement class.

## Required assumptions

{assumptions}

## Excluded cases

{exclusions}

This is a standard finite-dimensional probability result and an audit gate,
not a novelty claim or evidence that gravity is classical or quantum.  Q04's
stored AST evaluates the mean and covariance formulas directly; the proof does
not extend Q04 beyond its declared reduced fixed-measurement diagnostic.
"""


def _report(
    config: Mapping[str, Any],
    executions: Sequence[Mapping[str, Any]],
    mappings: Sequence[Mapping[str, Any]],
) -> str:
    return f"""# Quantum/entity-wave gravity atlas v3 strict repair

## Exact result

All {len(executions)} reduced card fixtures are evaluated from their stored machine ASTs by one generic interpreter. Variable declarations, shapes, domains, L/M/T dimensions, initial/boundary-condition status, and assertions are checked before sealing. No card-specific evaluator exists.

The two exact reduced free-pole mappings have residuals `{[row["max_abs_residual"] for row in mappings]}`. Each mapping covers every source key by an explicit mapped/dropped classification and assigns every target key with checked type, shape, unit, and domain.

## Honest limits

Q14 is only a normalized matrix-positivity diagnostic; it is not an Oppenheim CQ generator. Q15 is only a dimensionless AM-GM noise-cost diagnostic; it is not a KTM measurement-record, feedback, or master-equation implementation. Q12 contains only a damped mean transfer and ringdown; no quantum bath spectrum is implemented.

EF03 is not instantiated for Q11 because Q11 lacks a complete measurement map, and Q14 is excluded. No Q12/Q13/Q14 equivalence is claimed.

## Source gate

All {len(config["source_manifests"])} empirical manifests are `SOURCE_BLOCKED`. In particular, GWOSC event metadata do not provide the resolved product URLs, hashes, waveform implementation, statistical program, calibration treatment, and validation suite needed to call M01/M02 executable. No observational payload or response row was opened.

## Provenance and publication boundary

These are uncommitted worktree files. Integrity is hash-bound only; no VCS commit provenance is claimed. This package passes its internal AST/dimension/mapping/source-gate checks and is ready to submit for another strict audit. It does not establish a full theory implementation, empirical support, novelty, quantum gravity, or publication readiness.
"""


def artifact_payloads(config: Mapping[str, Any]) -> dict[str, bytes]:
    executions = [validate_and_execute_card(card) for card in config["cards"]]
    mappings = [
        execute_parameter_mapping(config, mapping) for mapping in config["parameter_mappings"]
    ]
    program_object = {
        "schema_version": "invariant-open-gravity-AST-program-cards-3.0",
        "program_contract": config["program_contract"],
        "cards": config["cards"],
        "reduced_diagnostics_only": True,
    }
    execution_object = {
        "schema_version": "invariant-open-gravity-AST-fixtures-3.0",
        "executions": executions,
        "observational_rows": 0,
    }
    mapping_object = {
        "schema_version": "invariant-open-gravity-parameter-mappings-3.0",
        "exact_mappings": mappings,
        "boundaries": config["equivalence_boundaries"],
        "arbitrary_maps_allowed": False,
    }
    source_object = {
        "schema_version": "invariant-open-gravity-source-gates-3.0",
        "manifests": config["source_manifests"],
        "all_empirical_sources_blocked": True,
    }
    counterexample_object = {
        "schema_version": "invariant-open-gravity-AST-counterexamples-3.0",
        "counterexamples": counterexamples(),
    }
    output_rows = [
        [
            execution["card_id"],
            output["name"],
            output["type"],
            "x".join(map(str, output["shape"])),
            json.dumps(output["unit"], separators=(",", ":")),
            output["ast_sha256"],
            json.dumps(output["value"], sort_keys=True, separators=(",", ":")),
        ]
        for execution in executions
        for output in execution["outputs"]
    ]
    mapping_rows = [
        [
            row["mapping_id"],
            row["source_card"],
            row["target_card"],
            row["compare_output"],
            row["max_abs_residual"],
            row["source_key_coverage"],
            row["target_key_coverage"],
            row["type_shape_unit_domain_checks"],
            row["scope"],
        ]
        for row in mappings
    ]
    readiness_rows = [
        [
            card["id"],
            card["implementation_level"],
            card["data_readiness"],
            card["falsifier_manifest"],
            "SOURCE_BLOCKED",
            card["claim"],
        ]
        for card in config["cards"]
    ]
    return {
        "AST-program-cards.json": _canonical(program_object) + b"\n",
        "AST-target-free-executions.json": _canonical(execution_object) + b"\n",
        "AST-output-index.csv": _csv_bytes(
            ["card_id", "output", "type", "shape", "unit_LMT", "ast_sha256", "value"], output_rows
        ),
        "parameter-mappings.json": _canonical(mapping_object) + b"\n",
        "parameter-mappings.csv": _csv_bytes(
            [
                "mapping_id",
                "source",
                "target",
                "output",
                "max_abs_residual",
                "source_coverage",
                "target_coverage",
                "typed_checks",
                "scope",
            ],
            mapping_rows,
        ),
        "gaussian-fixed-measurement-proof.md": _gaussian_proof(config).encode("utf-8"),
        "source-gates.json": _canonical(source_object) + b"\n",
        "readiness-and-implementation.csv": _csv_bytes(
            [
                "card_id",
                "implementation_level",
                "data_readiness",
                "manifest",
                "manifest_status",
                "claim",
            ],
            readiness_rows,
        ),
        "counterexamples.json": _canonical(counterexample_object) + b"\n",
        "report.md": _report(config, executions, mappings).encode("utf-8"),
    }


def _package_hashes(base: Path) -> dict[str, str]:
    return {
        "config_raw_sha256": _sha256_file(base / CONFIG_PATH),
        "module_raw_sha256": _sha256_file(base / MODULE_PATH),
        "test_raw_sha256": _sha256_file(base / TEST_PATH),
    }


def build_receipt(config: Mapping[str, Any], base: Path) -> tuple[dict[str, Any], dict[str, bytes]]:
    payloads = artifact_payloads(config)
    executions = [validate_and_execute_card(card) for card in config["cards"]]
    mappings = [
        execute_parameter_mapping(config, mapping) for mapping in config["parameter_mappings"]
    ]
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "analysis_id": config["analysis_id"],
        "decision": DECISION,
        "content_sha256": "",
        "predecessors": config["predecessors"],
        "provenance": config["provenance"],
        "package_hashes": _package_hashes(base),
        "config_content_sha256": _sha256_bytes(_canonical(config)),
        "artifact_sha256": {
            name: _sha256_bytes(payload) for name, payload in sorted(payloads.items())
        },
        "counts": {
            "reduced_AST_cards": len(executions),
            "AST_outputs": sum(len(row["outputs"]) for row in executions),
            "AST_assertions": sum(len(row["assertions"]) for row in executions),
            "exact_parameter_mappings": len(mappings),
            "scoped_nonmappings": len(config["equivalence_boundaries"]),
            "source_blocked_manifests": len(config["source_manifests"]),
            "counterexamples_retained": len(counterexamples()),
            "observational_payloads": 0,
            "observational_rows": 0,
        },
        "internal_checks": {
            "stored_AST_only": True,
            "card_specific_evaluators_absent": True,
            "dimension_shape_domain_checks": True,
            "initial_boundary_statuses": True,
            "PSD_symmetry_gates": True,
            "mutation_sensitive_programs": ["Q01", "Q14", "Q15"],
            "mapping_key_type_shape_unit_domain_coverage": True,
            "all_empirical_sources_blocked": True,
        },
        "claim_scope": {
            "submission_status": "READY_FOR_STRICT_REAUDIT",
            "full_theory_implementations": False,
            "empirical_execution_ready": False,
            "empirical_support": False,
            "novelty_established": False,
            "publication_ready": False,
        },
        "access_ledger": config["access_contract"],
        "claim_boundary": config["claim_boundary"],
    }
    receipt["content_sha256"] = _self_hash(receipt)
    return receipt, payloads


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def build(root: Path | None = None) -> str:
    base = (root or _repo_root()).resolve()
    config = load_config(base)
    receipt, payloads = build_receipt(config, base)
    targets = {base / ARTIFACT_DIR / name: payload for name, payload in payloads.items()}
    targets[base / OUTPUT_PATH] = _canonical(receipt) + b"\n"
    existing = [path for path in targets if path.exists()]
    if existing:
        _require(len(existing) == len(targets), "partial output package exists")
        for path, payload in targets.items():
            _require(path.read_bytes() == payload, f"existing output differs: {path}")
        return "EXISTING_IDENTICAL"
    for path, payload in targets.items():
        _atomic_write(path, payload)
    return "CREATED"


def check(root: Path | None = None) -> str:
    base = (root or _repo_root()).resolve()
    config = load_config(base)
    expected, payloads = build_receipt(config, base)
    observed = _read_json(base / OUTPUT_PATH)
    _require(observed.get("content_sha256") == _self_hash(observed), "receipt self-hash invalid")
    _require(observed == expected, "receipt differs from deterministic rebuild")
    for name, payload in payloads.items():
        path = base / ARTIFACT_DIR / name
        _require(path.is_file(), f"missing artifact: {name}")
        _require(path.read_bytes() == payload, f"artifact differs: {name}")
    return "VALID"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("build")
    subparsers.add_parser("check")
    subparsers.add_parser("status")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "build":
        print(build())
        return 0
    if args.command == "check":
        print(check())
        return 0
    config = load_config()
    print(
        json.dumps(
            {
                "analysis_id": config["analysis_id"],
                "status": config["status"],
                "cards": len(config["cards"]),
                "sources_ready": 0,
                "observational_rows": 0,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
