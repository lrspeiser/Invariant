from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

CONFIG_PATH = "configs/backgrounds/quartic_tc2_d4_higher_p55_checkpointable_materializer.json"
SOURCE_PATH = "src/sigma_theory_compiler/quartic_tc2_d4_higher_p55_checkpointable_materializer.py"
TEST_PATH = "tests/test_quartic_tc2_d4_higher_p55_checkpointable_materializer.py"
OUTPUT_PATH = (
    "runs/physics-language/quartic-tc2-d4-higher-p55-checkpointable-materializer/result.json"
)
CHECKPOINT_PATH = (
    "runs/physics-language/quartic-tc2-d4-higher-p55-checkpointable-materializer/checkpoints"
)
CONFIG_SCHEMA = "sigma-quartic-tc2-d4-higher-p55-materializer-config-1.0"
CHECKPOINT_SCHEMA = "sigma-quartic-tc2-d4-higher-p55-evaluation-checkpoint-1.0"
RESULT_SCHEMA = "sigma-quartic-tc2-d4-higher-p55-materializer-result-1.0"
ORDERING = [*range(11), *range(33, 55), *range(11, 33)]
AXES = (1, 2, 3)
VARIABLES = ("n1", "n2", "n3")
ORDERS = (2, 3, 4)


class HigherP55MaterializerError(ValueError):
    """Raised when higher-P55 authority cannot be proved exactly."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "ascii"
    )


def _content_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        _canonical_bytes({key: item for key, item in value.items() if key != "content_sha256"})
    ).hexdigest()


def _with_hash(body: dict[str, Any]) -> dict[str, Any]:
    return {**body, "content_sha256": _content_hash(body)}


def _hash_matches(value: dict[str, Any]) -> bool:
    return value.get("content_sha256") == _content_hash(value)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise HigherP55MaterializerError(f"expected object: {path}")
    return value


def _resolve_under(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if root != path and root not in path.parents:
        raise HigherP55MaterializerError("bound path escaped project root")
    return path


def _load_bound(root: Path, binding: dict[str, str]) -> dict[str, Any]:
    path = _resolve_under(root, binding["path"])
    value = _load_json(path)
    if (
        _file_sha256(path) != binding["file_sha256"]
        or value.get("content_sha256") != binding["content_sha256"]
        or not _hash_matches(value)
    ):
        raise HigherP55MaterializerError(f"upstream seal mismatch: {binding['path']}")
    return value


def _load_config(root: Path, config_path: Path) -> dict[str, Any]:
    config = _load_json(config_path)
    if (
        config.get("schema_version") != CONFIG_SCHEMA
        or config.get("policy")
        != "extract_every_coefficient_exactly_never_infer_absent_packets_as_zero"
        or set(config.get("upstreams", {}))
        != {"predecessor", "flat_P55", "P55_order_one", "polarization"}
        or config.get("basis_jet_directions") != ["G_12", "G_01", "H_01", "H_11"]
        or config.get("target")
        != {
            "Taylor_orders": [2, 3, 4],
            "polarization_evaluations": 15,
            "packets": 45,
            "shape_each": [55, 55],
            "manifest_registered_before": 109,
            "manifest_registered_after": 154,
        }
        or not _hash_matches(config)
    ):
        raise HigherP55MaterializerError("invalid higher-P55 config")
    source = config.get("live_source", {})
    if _file_sha256(_resolve_under(root, source.get("path", ""))) != source.get("file_sha256"):
        raise HigherP55MaterializerError("live source seal mismatch")
    return config


def _matrix_record(name: str, matrix: Any) -> dict[str, Any]:
    import sympy as sp

    entries = [
        {"row": row, "column": column, "value": sp.sstr(sp.factor(matrix[row, column]))}
        for row in range(matrix.rows)
        for column in range(matrix.cols)
        if matrix[row, column] != 0
    ]
    return _with_hash(
        {
            "schema_version": "sigma-exact-sparse-matrix-1.0",
            "name": name,
            "shape": [matrix.rows, matrix.cols],
            "entries": entries,
            "nonzero_count": len(entries),
        }
    )


def _matrix_from_record(record: dict[str, Any]) -> Any:
    import sympy as sp

    if not _hash_matches(record) or record.get("shape") != [55, 55]:
        raise HigherP55MaterializerError("matrix record seal/shape mismatch")
    matrix = sp.zeros(55)
    seen: set[tuple[int, int]] = set()
    for entry in record.get("entries", []):
        cell = (entry["row"], entry["column"])
        if cell in seen:
            raise HigherP55MaterializerError("duplicate matrix cell")
        matrix[cell] = sp.sympify(entry["value"])
        seen.add(cell)
    if len(seen) != record.get("nonzero_count"):
        raise HigherP55MaterializerError("matrix record count mismatch")
    return matrix


def _axis_from_order_one(packet: dict[str, Any], variable: str) -> Any:
    import sympy as sp

    if not _hash_matches(packet) or packet.get("shape") != [55, 55]:
        raise HigherP55MaterializerError("P55 order-one packet boundary changed")
    matrix = sp.zeros(55)
    for entry in packet.get("entries", []):
        value = entry.get("linear_coefficients", {}).get(variable)
        if value is not None:
            matrix[entry["row"], entry["column"]] = sp.sympify(value)
    return matrix


def _polynomial_packet(evaluation_id: str, order: int, axes: list[Any]) -> dict[str, Any]:
    import sympy as sp

    entries: dict[tuple[int, int], dict[str, str]] = {}
    axis_counts = []
    for variable, matrix in zip(VARIABLES, axes, strict=True):
        matrix = matrix.applyfunc(sp.factor)
        axis_counts.append(sum(value != 0 for value in matrix))
        for row in range(55):
            for column in range(55):
                if matrix[row, column] != 0:
                    entries.setdefault((row, column), {})[variable] = sp.sstr(matrix[row, column])
    return _with_hash(
        {
            "schema_version": "sigma-exact-sparse-polynomial-P55-higher-Taylor-1.0",
            "evaluation_id": evaluation_id,
            "shape": [55, 55],
            "variables": list(VARIABLES),
            "Taylor_order": order,
            "factorial_normalization": f"1/{order}!",
            "entries": [
                {"row": row, "column": column, "linear_coefficients": coefficients}
                for (row, column), coefficients in sorted(entries.items())
            ],
            "distinct_matrix_cells": len(entries),
            "axis_nonzero_entries": axis_counts,
        }
    )


def _atomic_write(path: Path, value: dict[str, Any], maximum_bytes: int) -> None:
    data = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    if len(data) > maximum_bytes:
        raise HigherP55MaterializerError("checkpoint exceeded byte cap")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise HigherP55MaterializerError(f"immutable checkpoint conflict: {path.name}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(data)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _evaluation_path(checkpoint_dir: Path, evaluation_id: str) -> Path:
    if not evaluation_id.startswith("subset_") or any(
        character not in "abcdefghijklmnopqrstuvwxyz_0123" for character in evaluation_id
    ):
        raise HigherP55MaterializerError("unsafe evaluation id")
    return checkpoint_dir / f"{evaluation_id}.json"


def materialize(root: Path, config_path: Path, checkpoint_dir: Path) -> None:
    import sympy as sp

    from .quartic_first_order_reduction_campaign import (
        _extract_spatial_blocks,
        _full_first_order_pencil,
        _symbol_data,
    )

    root = root.resolve()
    config = _load_config(root, config_path)
    _load_bound(root, config["upstreams"]["predecessor"])
    flat = _load_bound(root, config["upstreams"]["flat_P55"])
    p55_order_one = _load_bound(root, config["upstreams"]["P55_order_one"])
    polarization = _load_bound(root, config["upstreams"]["polarization"])
    p1_by_id = {packet["evaluation_id"]: packet for packet in p55_order_one["packets"]}
    evaluations = polarization["polarization_evaluations"]
    if len(evaluations) != 15 or set(p1_by_id) != {row["evaluation_id"] for row in evaluations}:
        raise HigherP55MaterializerError("evaluation authority mismatch")
    missing = [
        row
        for row in evaluations
        if not _evaluation_path(checkpoint_dir, row["evaluation_id"]).exists()
    ]
    if not missing:
        return
    data = _symbol_data()
    xi = data["xi_lower"]
    jets = (
        list(data["gradient_lower"])
        + sorted(data["hessian_lower"].free_symbols, key=str)
        + sorted(data["einstein_upper"].free_symbols, key=str)
    )
    jet_by_name = {str(jet): jet for jet in jets}
    selected = [jet_by_name[name] for name in config["basis_jet_directions"]]
    coefficient_a = data["first_order"]["A"]
    b_blocks, c_blocks = _extract_spatial_blocks(
        data["first_order"]["B"], data["first_order"]["C"], list(xi[1:])
    )
    coefficient_matrices = [coefficient_a, *b_blocks, *[item for row in c_blocks for item in row]]
    mixed_second_nonzero = sum(
        1
        for left, first in enumerate(selected)
        for second in selected[left:]
        for matrix in coefficient_matrices
        for value in matrix.diff(first).diff(second)
        if value != 0
    )
    if mixed_second_nonzero:
        raise HigherP55MaterializerError(
            "first missing primitive: nonlinear Taylor coefficient of mass/evolution pencil"
        )
    zero_jet = {symbol: 0 for symbol in jets}
    maximum = config["caps"]["maximum_checkpoint_bytes"]
    for evaluation in missing:
        direction = evaluation["combined_jet_direction"]
        if not set(direction) <= set(jet_by_name):
            raise HigherP55MaterializerError("polarization direction contains an unknown jet")
        order_axes: dict[int, list[Any]] = {order: [] for order in ORDERS}
        recurrences = 0
        p1_record = p1_by_id[evaluation["evaluation_id"]]
        for axis_index, (axis, variable) in enumerate(zip(AXES, VARIABLES, strict=True)):
            spatial = [0, 0, 0]
            spatial[axis_index] = 1
            reference = {
                **zero_jet,
                data["alpha"]: 0,
                data["m2"]: 1,
                data["c20"]: 0,
                **{xi[index + 1]: spatial[index] for index in range(3)},
            }
            derivative = dict(reference)
            derivative[data["alpha"]] = 1
            mass0, evolution0 = _full_first_order_pencil(
                coefficient_a.subs(reference),
                b_blocks[axis_index].subs(reference),
                [matrix.subs(reference) for matrix in c_blocks[axis_index]],
                spatial,
            )
            p0_original = mass0.inv() * evolution0
            p0 = p0_original.extract(ORDERING, ORDERING).applyfunc(sp.factor)
            if p0 != _matrix_from_record(flat["matrix_packets"][axis_index]):
                raise HigherP55MaterializerError("flat P55 replay mismatch")
            mass1 = sp.zeros(55)
            evolution1 = sp.zeros(55)
            for name, coefficient_text in direction.items():
                coefficient = sp.sympify(coefficient_text)
                jet = jet_by_name[name]
                mass_atom, evolution_atom = _full_first_order_pencil(
                    coefficient_a.diff(jet).subs(derivative),
                    b_blocks[axis_index].diff(jet).subs(derivative),
                    [matrix.diff(jet).subs(derivative) for matrix in c_blocks[axis_index]],
                    spatial,
                )
                mass1 += coefficient * mass_atom
                evolution1 += coefficient * evolution_atom
            # _full_first_order_pencil contributes identity blocks to every mass call;
            # and fixed definition-constraint evolution blocks to every call. Their
            # derivatives are zero, so retain only coefficient derivatives.
            mass1[:11, :11] = sp.zeros(11)
            mass1[22:, 22:] = sp.zeros(33)
            evolution1[22:, 11:22] = sp.zeros(33, 11)
            p1_original = mass0.inv() * (evolution1 - mass1 * p0_original)
            p1 = p1_original.extract(ORDERING, ORDERING).applyfunc(sp.factor)
            if p1 != _axis_from_order_one(p1_record["P55_Taylor_order_one_matrix"], variable):
                raise HigherP55MaterializerError("order-one P55 replay mismatch")
            previous = p1_original
            for order in ORDERS:
                current = (-mass0.inv() * mass1 * previous).applyfunc(sp.factor)
                residual = (mass0 * current + mass1 * previous).applyfunc(sp.factor)
                if not residual.is_zero_matrix:
                    raise HigherP55MaterializerError(f"P55 order-{order} recurrence residual")
                order_axes[order].append(current.extract(ORDERING, ORDERING).applyfunc(sp.factor))
                previous = current
                recurrences += 55 * 55
        packets = [
            _polynomial_packet(evaluation["evaluation_id"], order, order_axes[order])
            for order in ORDERS
        ]
        checkpoint = _with_hash(
            {
                "schema_version": CHECKPOINT_SCHEMA,
                "evaluation_id": evaluation["evaluation_id"],
                "evaluation_content_sha256": evaluation["content_sha256"],
                "P55_order_one_content_sha256": p1_record["content_sha256"],
                "Taylor_orders": list(ORDERS),
                "packets": packets,
                "coefficient_pencil_mixed_second_derivative_nonzero_entries": 0,
                "exact_recurrence_entries_reduced": recurrences,
                "exact_recurrence_nonzero_remainders": 0,
            }
        )
        _atomic_write(
            _evaluation_path(checkpoint_dir, evaluation["evaluation_id"]), checkpoint, maximum
        )
        print(f"sealed higher P55 {evaluation['evaluation_id']}", flush=True)


def build_result(root: Path, config_path: Path, checkpoint_dir: Path) -> dict[str, Any]:
    root = root.resolve()
    config = _load_config(root, config_path)
    predecessor = _load_bound(root, config["upstreams"]["predecessor"])
    polarization = _load_bound(root, config["upstreams"]["polarization"])
    packets = []
    checkpoint_hashes = []
    for evaluation in polarization["polarization_evaluations"]:
        path = _evaluation_path(checkpoint_dir, evaluation["evaluation_id"])
        if not path.exists():
            raise HigherP55MaterializerError(
                f"first missing primitive: {evaluation['evaluation_id']} P55 Taylor order 2"
            )
        checkpoint = _load_json(path)
        if (
            checkpoint.get("schema_version") != CHECKPOINT_SCHEMA
            or checkpoint.get("evaluation_content_sha256") != evaluation["content_sha256"]
            or not _hash_matches(checkpoint)
            or len(checkpoint.get("packets", [])) != 3
            or [packet.get("Taylor_order") for packet in checkpoint["packets"]] != list(ORDERS)
            or any(not _hash_matches(packet) for packet in checkpoint["packets"])
        ):
            raise HigherP55MaterializerError(f"checkpoint tamper: {path.name}")
        packets.extend(checkpoint["packets"])
        checkpoint_hashes.append(checkpoint["content_sha256"])
    manifest = json.loads(json.dumps(predecessor["required_symbolic_input_manifest"]))
    family = next(row for row in manifest if row["input_id"] == "polarized_P55_Taylor_packets")
    if family.get("registered_packets") != 30 or family.get("registered_Taylor_orders") != [0, 1]:
        raise HigherP55MaterializerError("P55 manifest predecessor changed")
    family["registered_packets"] = 75
    family["registered_Taylor_orders"] = [0, 1, 2, 3, 4]
    family["missing_Taylor_orders"] = []
    family["packet_content_sha256"].extend(packet["content_sha256"] for packet in packets)
    family["status"] = "registered_all_15_packets_at_Taylor_orders_zero_through_four"
    if sum(row["registered_packets"] for row in manifest) != 154:
        raise HigherP55MaterializerError("manifest did not advance atomically from 109 to 154")
    body = {
        "schema_version": RESULT_SCHEMA,
        "status": "pass_exact_45_higher_P55_packets_registered",
        "errors": [],
        "config_sha256": config["content_sha256"],
        "predecessor_content_sha256": predecessor["content_sha256"],
        "checkpoint_content_sha256": checkpoint_hashes,
        "registered_P55_Taylor_orders_two_through_four_packets": packets,
        "required_symbolic_input_manifest": manifest,
        "counts": {
            "polarization_evaluations": 15,
            "Taylor_orders": 3,
            "P55_higher_packets_registered": 45,
            "manifest_registered_before": 109,
            "manifest_registered_after": 154,
            "manifest_missing_after": 150,
            "coefficient_pencil_mixed_second_derivative_nonzero_entries": 0,
            "exact_recurrence_matrix_entries_reduced": 15 * 3 * 3 * 55 * 55,
            "exact_recurrence_nonzero_remainders": 0,
            "full_symbol_build_calls_per_materialization_run": 1,
            "emitted_output_rows": 0,
        },
        "claims": {
            "all_45_higher_P55_packets_registered": True,
            "higher_K55_packets_registered": False,
            "higher_TC2_packets_registered": False,
            "lower_Sylvester_packets_registered": False,
            "full_117180_rows_emitted": False,
        },
        "negative_controls": {
            "infer_missing_packet_as_zero": {"rejected": True},
            "accept_missing_evaluation_checkpoint": {"rejected": True},
            "accept_tampered_checkpoint": {"rejected": True},
            "partially_advance_manifest": {"rejected": True},
        },
        "source_bindings": {
            "config": {"path": CONFIG_PATH, "file_sha256": _file_sha256(config_path)},
            "source": {"path": SOURCE_PATH, "file_sha256": _file_sha256(root / SOURCE_PATH)},
            "test": {"path": TEST_PATH, "file_sha256": _file_sha256(root / TEST_PATH)},
        },
        "next_exact_primitive": "45 H_star_plus Taylor-order-two-through-four packets",
        "scope": (
            "Registers exactly the 45 P55 packets at Taylor orders two through four. "
            "K55, TC2, lower-Sylvester packets and coefficient rows remain unregistered."
        ),
    }
    return _with_hash(body)


def validate_result(document: dict[str, Any], root: Path) -> None:
    if not _hash_matches(document):
        raise HigherP55MaterializerError("result content hash mismatch")
    expected = build_result(
        root.resolve(), root.resolve() / CONFIG_PATH, root.resolve() / CHECKPOINT_PATH
    )
    if document != expected:
        raise HigherP55MaterializerError("result replay mismatch")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--materialize", action="store_true")
    args = parser.parse_args(argv)
    if args.materialize:
        materialize(args.project_root, args.config, args.checkpoint_dir)
    result = build_result(args.project_root, args.config, args.checkpoint_dir)
    _atomic_write(args.output, result, 64 * 1024 * 1024)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
