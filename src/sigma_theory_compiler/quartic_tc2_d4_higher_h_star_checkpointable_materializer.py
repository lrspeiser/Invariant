from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

CONFIG_PATH = "configs/backgrounds/quartic_tc2_d4_higher_h_star_checkpointable_materializer.json"
SOURCE_PATH = (
    "src/sigma_theory_compiler/quartic_tc2_d4_higher_h_star_checkpointable_materializer.py"
)
TEST_PATH = "tests/test_quartic_tc2_d4_higher_h_star_checkpointable_materializer.py"
OUTPUT_PATH = (
    "runs/physics-language/quartic-tc2-d4-higher-h-star-checkpointable-materializer/result.json"
)
CHECKPOINT_PATH = (
    "runs/physics-language/quartic-tc2-d4-higher-h-star-checkpointable-materializer/checkpoints"
)
CONFIG_SCHEMA = "sigma-quartic-tc2-d4-higher-h-star-materializer-config-1.0"
CHECKPOINT_SCHEMA = "sigma-quartic-tc2-d4-higher-h-star-evaluation-checkpoint-1.0"
RESULT_SCHEMA = "sigma-quartic-tc2-d4-higher-h-star-materializer-result-1.0"
ORDERS = (2, 3, 4)
AXES = (1, 2, 3)
VARIABLES = ("n1", "n2", "n3")


class HigherHStarMaterializerError(ValueError):
    """Raised when higher physical H-star authority fails closed."""


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
        raise HigherHStarMaterializerError(f"expected object: {path}")
    return value


def _resolve_under(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if root != path and root not in path.parents:
        raise HigherHStarMaterializerError("bound path escaped root")
    return path


def _load_bound(root: Path, binding: dict[str, str]) -> dict[str, Any]:
    path = _resolve_under(root, binding["path"])
    value = _load_json(path)
    if (
        _file_sha256(path) != binding["file_sha256"]
        or value.get("content_sha256") != binding["content_sha256"]
        or not _hash_matches(value)
    ):
        raise HigherHStarMaterializerError(f"upstream mismatch: {binding['path']}")
    return value


def _load_config(root: Path, config_path: Path) -> dict[str, Any]:
    config = _load_json(config_path)
    if (
        config.get("schema_version") != CONFIG_SCHEMA
        or config.get("policy")
        != "differentiate_authoritative_action_symbol_never_infer_zero_packets"
        or set(config.get("upstreams", {}))
        != {"higher_P55", "H_star_order_one", "polarization", "flat_action_metric"}
        or config.get("target")
        != {
            "Taylor_orders": [2, 3, 4],
            "polarization_evaluations": 15,
            "packets": 45,
            "shape_each": [22, 22],
        }
        or config.get("basis_jet_directions") != ["G_12", "G_01", "H_01", "H_11"]
        or not _hash_matches(config)
    ):
        raise HigherHStarMaterializerError("invalid higher H-star config")
    live_sources = config.get("live_sources", {})
    if set(live_sources) != {"symbol_builder", "generalized_pencil"} or any(
        _file_sha256(_resolve_under(root, binding.get("path", ""))) != binding.get("file_sha256")
        for binding in live_sources.values()
    ):
        raise HigherHStarMaterializerError("live action-symbol source seal mismatch")
    return config


def _atomic_write(path: Path, value: dict[str, Any], maximum: int) -> None:
    data = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    if len(data) > maximum:
        raise HigherHStarMaterializerError("checkpoint exceeded byte cap")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise HigherHStarMaterializerError(f"immutable checkpoint conflict: {path.name}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(data)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _checkpoint_path(directory: Path, evaluation_id: str) -> Path:
    if not evaluation_id.startswith("subset_") or any(
        character not in "abcdefghijklmnopqrstuvwxyz_0123" for character in evaluation_id
    ):
        raise HigherHStarMaterializerError("unsafe evaluation id")
    return directory / f"{evaluation_id}.json"


def _matrix_packet(
    evaluation_id: str, order: int, constant: Any, axes: list[Any]
) -> dict[str, Any]:
    import sympy as sp

    entries = []
    for row in range(22):
        for column in range(22):
            item: dict[str, Any] = {"row": row, "column": column}
            if constant[row, column] != 0:
                item["constant"] = sp.sstr(sp.factor(constant[row, column]))
            linear = {
                variable: sp.sstr(sp.factor(matrix[row, column]))
                for variable, matrix in zip(VARIABLES, axes, strict=True)
                if matrix[row, column] != 0
            }
            if linear:
                item["linear_coefficients"] = linear
            if len(item) > 2:
                entries.append(item)
    return _with_hash(
        {
            "schema_version": "sigma-exact-sparse-polynomial-H-star-plus-higher-Taylor-1.0",
            "evaluation_id": evaluation_id,
            "shape": [22, 22],
            "variables": list(VARIABLES),
            "Taylor_order": order,
            "factorial_normalization": f"1/{order}!",
            "entries": entries,
            "distinct_matrix_cells": len(entries),
            "constant_nonzero_entries": sum(value != 0 for value in constant),
            "axis_nonzero_entries": [sum(value != 0 for value in matrix) for matrix in axes],
        }
    )


def _packet_to_parts(packet: dict[str, Any]) -> tuple[Any, list[Any]]:
    import sympy as sp

    if not _hash_matches(packet) or packet.get("shape") != [22, 22]:
        raise HigherHStarMaterializerError("H-star packet boundary changed")
    constant = sp.zeros(22)
    axes = [sp.zeros(22) for _ in AXES]
    for entry in packet.get("entries", []):
        cell = (entry["row"], entry["column"])
        if "constant" in entry:
            constant[cell] = sp.sympify(entry["constant"])
        for index, variable in enumerate(VARIABLES):
            if variable in entry.get("linear_coefficients", {}):
                axes[index][cell] = sp.sympify(entry["linear_coefficients"][variable])
    return constant, axes


def materialize(root: Path, config_path: Path, checkpoint_dir: Path) -> None:
    import sympy as sp

    from .horndeski_principal import _first_order_generalized_pencil
    from .quartic_first_order_reduction_campaign import _symbol_data

    root = root.resolve()
    config = _load_config(root, config_path)
    higher_p = _load_bound(root, config["upstreams"]["higher_P55"])
    h1_result = _load_bound(root, config["upstreams"]["H_star_order_one"])
    polarization = _load_bound(root, config["upstreams"]["polarization"])
    flat = _load_bound(root, config["upstreams"]["flat_action_metric"])
    evaluations = polarization["polarization_evaluations"]
    h1_by_id = {packet["evaluation_id"]: packet for packet in h1_result["packets"]}
    p_ids = {
        packet["evaluation_id"]
        for packet in higher_p["registered_P55_Taylor_orders_two_through_four_packets"]
    }
    if (
        len(evaluations) != 15
        or set(h1_by_id) != p_ids
        or p_ids != {row["evaluation_id"] for row in evaluations}
    ):
        raise HigherHStarMaterializerError("evaluation authority mismatch")
    missing = [
        row
        for row in evaluations
        if not _checkpoint_path(checkpoint_dir, row["evaluation_id"]).exists()
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
    action = _first_order_generalized_pencil(data["action_symbol"], xi[0])
    zero_direction = {symbol: 0 for symbol in xi[1:]}
    b_axes = [
        action["B"].diff(symbol).subs(zero_direction).applyfunc(sp.factor) for symbol in xi[1:]
    ]
    reference = {
        **{symbol: 0 for symbol in jets},
        **zero_direction,
        data["alpha"]: 0,
        data["m2"]: 1,
        data["c20"]: 0,
    }
    if action["A"].subs(reference).applyfunc(sp.factor) != _parts_from_flat(flat)[0]:
        raise HigherHStarMaterializerError("flat action A replay mismatch")
    t = sp.Symbol("higher_H_star_Taylor_parameter")
    maximum = config["caps"]["maximum_checkpoint_bytes"]
    for evaluation in missing:
        direction = evaluation["combined_jet_direction"]
        substitution = {symbol: 0 for symbol in jets}
        for name, coefficient in direction.items():
            substitution[jet_by_name[name]] = sp.sympify(coefficient) * t
        substitution.update({**zero_direction, data["alpha"]: 1, data["m2"]: 1, data["c20"]: 0})
        a_curve = action["A"].subs(substitution)
        b_curves = [matrix.subs(substitution) for matrix in b_axes]
        # Replay order one before authorizing any higher packet.
        a1 = a_curve.diff(t).subs(t, 0).applyfunc(sp.factor)
        b1 = [matrix.diff(t).subs(t, 0).applyfunc(sp.factor) for matrix in b_curves]
        h1_constant = sp.zeros(22)
        h1_constant[:11, 11:] = a1
        h1_constant[11:, :11] = a1
        h1_axes = []
        for matrix in b1:
            h_axis = sp.zeros(22)
            h_axis[:11, :11] = matrix
            h1_axes.append(h_axis)
        expected_constant, expected_axes = _packet_to_parts(
            h1_by_id[evaluation["evaluation_id"]]["H_star_plus_order_one_matrix"]
        )
        if h1_constant != expected_constant or h1_axes != expected_axes:
            raise HigherHStarMaterializerError("H-star order-one replay mismatch")
        packets = []
        derivative_entries = 0
        for order in ORDERS:
            divisor = math.factorial(order)
            a_order = (a_curve.diff(t, order).subs(t, 0) / divisor).applyfunc(sp.factor)
            b_order = [
                (matrix.diff(t, order).subs(t, 0) / divisor).applyfunc(sp.factor)
                for matrix in b_curves
            ]
            constant = sp.zeros(22)
            constant[:11, 11:] = a_order
            constant[11:, :11] = a_order
            axes = []
            for matrix in b_order:
                h_axis = sp.zeros(22)
                h_axis[:11, :11] = matrix
                axes.append(h_axis)
            if constant != constant.T or any(matrix != matrix.T for matrix in axes):
                raise HigherHStarMaterializerError(f"H-star order-{order} symmetry failed")
            derivative_entries += sum(value != 0 for value in a_order) + sum(
                value != 0 for matrix in b_order for value in matrix
            )
            packets.append(_matrix_packet(evaluation["evaluation_id"], order, constant, axes))
        document = _with_hash(
            {
                "schema_version": CHECKPOINT_SCHEMA,
                "evaluation_id": evaluation["evaluation_id"],
                "evaluation_content_sha256": evaluation["content_sha256"],
                "H_star_order_one_content_sha256": h1_by_id[evaluation["evaluation_id"]][
                    "content_sha256"
                ],
                "Taylor_orders": list(ORDERS),
                "packets": packets,
                "authoritative_action_derivative_nonzero_entries": derivative_entries,
                "symmetry_remainder_entries": 0,
                "zero_packets_authorized_by_exact_derivative": all(
                    packet["distinct_matrix_cells"] == 0 for packet in packets
                ),
            }
        )
        _atomic_write(
            _checkpoint_path(checkpoint_dir, evaluation["evaluation_id"]), document, maximum
        )
        print(f"sealed higher H-star {evaluation['evaluation_id']}", flush=True)


def _parts_from_flat(flat: dict[str, Any]) -> tuple[Any, Any]:
    from .quartic_tc2_d4_h_star_order_one_checkpointable_materializer import _matrix_from_record

    exact = flat["exact_construction"]
    return _matrix_from_record(exact["A_0"], [11, 11]), _matrix_from_record(exact["B_0"], [11, 11])


def build_result(root: Path, config_path: Path, checkpoint_dir: Path) -> dict[str, Any]:
    root = root.resolve()
    config = _load_config(root, config_path)
    higher_p = _load_bound(root, config["upstreams"]["higher_P55"])
    polarization = _load_bound(root, config["upstreams"]["polarization"])
    packets = []
    checkpoint_hashes = []
    derivative_entries = 0
    for evaluation in polarization["polarization_evaluations"]:
        path = _checkpoint_path(checkpoint_dir, evaluation["evaluation_id"])
        if not path.exists():
            raise HigherHStarMaterializerError(
                f"first missing primitive: {evaluation['evaluation_id']} H_star_plus Taylor order 2"
            )
        checkpoint = _load_json(path)
        if (
            checkpoint.get("schema_version") != CHECKPOINT_SCHEMA
            or checkpoint.get("evaluation_content_sha256") != evaluation["content_sha256"]
            or not _hash_matches(checkpoint)
            or [packet.get("Taylor_order") for packet in checkpoint.get("packets", [])]
            != list(ORDERS)
            or any(not _hash_matches(packet) for packet in checkpoint["packets"])
        ):
            raise HigherHStarMaterializerError(f"checkpoint tamper: {path.name}")
        packets.extend(checkpoint["packets"])
        checkpoint_hashes.append(checkpoint["content_sha256"])
        derivative_entries += checkpoint["authoritative_action_derivative_nonzero_entries"]
    if len(packets) != 45:
        raise HigherHStarMaterializerError("incomplete higher H-star family")
    body = {
        "schema_version": RESULT_SCHEMA,
        "status": "pass_exact_45_physical_H_star_plus_higher_Taylor_packets_materialized",
        "errors": [],
        "config_sha256": config["content_sha256"],
        "higher_P55_content_sha256": higher_p["content_sha256"],
        "checkpoint_content_sha256": checkpoint_hashes,
        "packets": packets,
        "counts": {
            "polarization_evaluations": 15,
            "Taylor_orders": 3,
            "H_star_plus_higher_packets": 45,
            "authoritative_action_derivative_nonzero_entries": derivative_entries,
            "zero_packets_exactly_derived": sum(
                packet["distinct_matrix_cells"] == 0 for packet in packets
            ),
            "symmetry_remainder_entries": 0,
            "full_symbol_build_calls_per_materialization_run": 1,
        },
        "claims": {
            "all_45_physical_H_star_plus_higher_packets_materialized": True,
            "higher_K55_registered": False,
            "higher_TC2_registered": False,
            "lower_Sylvester_registered": False,
            "rows_emitted": False,
        },
        "negative_controls": {
            "infer_missing_derivative_as_zero": {"rejected": True},
            "accept_missing_evaluation_checkpoint": {"rejected": True},
            "accept_tampered_checkpoint": {"rejected": True},
            "accept_nonsymmetric_H_star_packet": {"rejected": True},
        },
        "source_bindings": {
            "config": {"path": CONFIG_PATH, "file_sha256": _file_sha256(config_path)},
            "source": {"path": SOURCE_PATH, "file_sha256": _file_sha256(root / SOURCE_PATH)},
            "test": {"path": TEST_PATH, "file_sha256": _file_sha256(root / TEST_PATH)},
        },
        "scope": "Materializes all physical H_star_plus Taylor packets at orders two through four directly from authoritative action-symbol derivatives. It does not register K55, TC2, lower-Sylvester packets, or rows.",
    }
    return _with_hash(body)


def validate_result(document: dict[str, Any], root: Path) -> None:
    if not _hash_matches(document) or document != build_result(
        root, root / CONFIG_PATH, root / CHECKPOINT_PATH
    ):
        raise HigherHStarMaterializerError("higher H-star result replay mismatch")


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
    result = build_result(
        args.project_root.resolve(), args.config.resolve(), args.checkpoint_dir.resolve()
    )
    _atomic_write(args.output.resolve(), result, 64 * 1024 * 1024)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
