from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

PLAN_SCHEMA = "sigma-quartic-tc2-d4-h-star-order-one-materializer-plan-1.0"
CONFIG_SCHEMA = "sigma-quartic-tc2-d4-h-star-order-one-checkpointable-materializer-config-1.0"
C0_SCHEMA = "sigma-quartic-tc2-d4-h-star-order-one-materializer-C0-seals-1.0"
C1_SCHEMA = "sigma-quartic-tc2-d4-h-star-order-one-materializer-C1-basis-jet-1.0"
C2_SCHEMA = "sigma-quartic-tc2-d4-h-star-order-one-materializer-C2-evaluation-1.0"
FINAL_SCHEMA = "sigma-quartic-tc2-d4-h-star-order-one-materializer-result-1.0"
CONFIG_PATH = "configs/backgrounds/quartic_tc2_d4_h_star_order_one_checkpointable_materializer.json"
SOURCE_PATH = (
    "src/sigma_theory_compiler/quartic_tc2_d4_h_star_order_one_checkpointable_materializer.py"
)
TEST_PATH = "tests/test_quartic_tc2_d4_h_star_order_one_checkpointable_materializer.py"
PLAN_PATH = (
    "runs/physics-language/quartic-tc2-d4-h-star-order-one-checkpointable-materializer/plan.json"
)
DEFAULT_CHECKPOINT_DIR = (
    "runs/physics-language/quartic-tc2-d4-h-star-order-one-checkpointable-materializer/checkpoints"
)
BASIS_ATOMS = ["G_12", "G_01", "H_01", "H_11"]
AXES = (1, 2, 3)
VARIABLES = ("n1", "n2", "n3")


class HStarOrderOneMaterializerError(ValueError):
    """Raised when an H-star materializer checkpoint fails closed."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "ascii"
    )


def _content_hash(value: dict[str, Any]) -> str:
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    return hashlib.sha256(_canonical_bytes(body)).hexdigest()


def _with_hash(body: dict[str, Any]) -> dict[str, Any]:
    return {**body, "content_sha256": _content_hash(body)}


def _hash_matches(value: dict[str, Any]) -> bool:
    return value.get("content_sha256") == _content_hash(value)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve_under(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if root != path and root not in path.parents:
        raise HStarOrderOneMaterializerError("bound path escaped project root")
    return path


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise HStarOrderOneMaterializerError(
            f"cannot load JSON checkpoint {path}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise HStarOrderOneMaterializerError(f"expected JSON object: {path}")
    return value


def _json_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _atomic_write_immutable(path: Path, value: dict[str, Any], maximum: int) -> None:
    data = _json_bytes(value)
    if len(data) > maximum:
        raise HStarOrderOneMaterializerError(f"checkpoint exceeds byte cap: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise HStarOrderOneMaterializerError(f"immutable checkpoint conflict: {path.name}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(data)
    try:
        if path.exists():
            if path.read_bytes() != data:
                raise HStarOrderOneMaterializerError(f"immutable checkpoint race: {path.name}")
            temporary.unlink(missing_ok=True)
        else:
            temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_hashed(value: dict[str, Any], schema: str, label: str) -> None:
    if value.get("schema_version") != schema or not _hash_matches(value):
        raise HStarOrderOneMaterializerError(f"{label} checkpoint seal mismatch")


def _load_bound(root: Path, binding: dict[str, str]) -> dict[str, Any]:
    path = _resolve_under(root, binding["path"])
    value = _load_json(path)
    if (
        _file_sha256(path) != binding["file_sha256"]
        or value.get("content_sha256") != binding["content_sha256"]
        or not _hash_matches(value)
    ):
        raise HStarOrderOneMaterializerError(f"upstream seal mismatch: {binding['path']}")
    return value


def _validate_live_source(root: Path, binding: dict[str, Any]) -> None:
    path = _resolve_under(root, binding["path"])
    tree = ast.parse(path.read_text(encoding="utf-8"))
    functions = {
        node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    if (
        _file_sha256(path) != binding["file_sha256"]
        or not set(binding["required_functions"]) <= functions
    ):
        raise HStarOrderOneMaterializerError("live formula source mismatch")


def _load_config(root: Path, config_path: Path) -> dict[str, Any]:
    config = _load_json(config_path)
    target = config.get("target", {})
    caps = config.get("caps", {})
    if (
        config.get("schema_version") != CONFIG_SCHEMA
        or config.get("policy")
        != "exact_basis_jet_H_star_plus_order_one_materialization_fail_closed"
        or not _hash_matches(config)
        or set(config.get("upstreams", {})) != {"readiness", "P55_order_one", "flat_action_metric"}
        or set(config.get("live_sources", {})) != {"symbol_builder", "action_pencil"}
        or config.get("basis_jet_directions") != BASIS_ATOMS
        or target
        != {
            "basis_jet_packets": 4,
            "basis_A_star_order_one_matrices": 4,
            "basis_B_star_order_one_axis_matrices": 12,
            "polarization_evaluations": 15,
            "state_matrix_dimension": 22,
            "Taylor_order": 1,
            "registered_manifest_before": 79,
            "registered_manifest_after_materializer": 79,
        }
        or set(caps) != {"C1", "C2", "poll_seconds", "maximum_checkpoint_bytes"}
        or any(
            not isinstance(caps[phase].get(key), int) or caps[phase][key] <= 0
            for phase in ("C1", "C2")
            for key in ("wall_seconds", "rss_bytes")
        )
        or not isinstance(caps.get("poll_seconds"), int)
        or caps["poll_seconds"] <= 0
        or not isinstance(caps.get("maximum_checkpoint_bytes"), int)
        or caps["maximum_checkpoint_bytes"] <= 0
        or any(config.get("seals", {}).values())
    ):
        raise HStarOrderOneMaterializerError("invalid H-star materializer config")
    for binding in config["live_sources"].values():
        _validate_live_source(root, binding)
    return config


def _evaluation_packets(root: Path, config: dict[str, Any]) -> list[dict[str, Any]]:
    result = _load_bound(root, config["upstreams"]["P55_order_one"])
    packets = result.get("packets", [])
    if (
        result.get("status") != "pass_exact_15_P55_Taylor_order_one_packets_materialized"
        or len(packets) != 15
        or any(not packet.get("basis_bindings") for packet in packets)
    ):
        raise HStarOrderOneMaterializerError("P55 evaluation boundary changed")
    return packets


def build_c0(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    receipts = {name: _load_bound(root, binding) for name, binding in config["upstreams"].items()}
    readiness = receipts["readiness"]
    flat = receipts["flat_action_metric"]
    evaluations = _evaluation_packets(root, config)
    if (
        readiness.get("decision") != "READY_CHECKPOINTABLE_MATERIALIZER"
        or readiness.get("counts", {}).get("registered_polarized_H_star_plus_order_one_packets")
        != 0
        or flat.get("exact_construction", {}).get("h_plus_0", {}).get("shape") != [22, 22]
        or len(evaluations) != 15
    ):
        raise HStarOrderOneMaterializerError("C0 predecessor boundary changed")
    return _with_hash(
        {
            "schema_version": C0_SCHEMA,
            "config_sha256": config["content_sha256"],
            "upstreams": {
                name: {
                    "path": binding["path"],
                    "content_sha256": binding["content_sha256"],
                }
                for name, binding in config["upstreams"].items()
            },
            "live_sources": config["live_sources"],
            "basis_jet_directions": BASIS_ATOMS,
            "polarization_evaluations": 15,
            "readiness_BLOCK_unchanged": True,
        }
    )


def _safe_slug(value: str) -> str:
    allowed = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
    if not value or any(character not in allowed for character in value):
        raise HStarOrderOneMaterializerError("unsafe checkpoint unit id")
    return value.lower()


def _checkpoint_paths(checkpoint_dir: Path, evaluations: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "C0": checkpoint_dir / "c0-seals.json",
        "C1": {atom: checkpoint_dir / f"c1-basis-{_safe_slug(atom)}.json" for atom in BASIS_ATOMS},
        "C2": {
            row["evaluation_id"]: checkpoint_dir / "c2-evaluations" / f"{row['evaluation_id']}.json"
            for row in evaluations
        },
    }


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
            "schema_version": "sigma-exact-sparse-action-matrix-1.0",
            "name": name,
            "shape": [matrix.rows, matrix.cols],
            "entries": entries,
            "nonzero_count": len(entries),
        }
    )


def _validate_matrix_record(record: dict[str, Any], shape: list[int]) -> None:
    if (
        not _hash_matches(record)
        or record.get("schema_version")
        not in {
            "sigma-exact-sparse-action-matrix-1.0",
            "sigma-exact-sparse-Qsqrt2-matrix-1.0",
        }
        or record.get("shape") != shape
        or record.get("nonzero_count") != len(record.get("entries", []))
    ):
        raise HStarOrderOneMaterializerError("matrix record seal/shape mismatch")
    seen = set()
    for entry in record["entries"]:
        coordinate = (entry.get("row"), entry.get("column"))
        if (
            coordinate in seen
            or not all(isinstance(index, int) for index in coordinate)
            or not (0 <= coordinate[0] < shape[0] and 0 <= coordinate[1] < shape[1])
            or not isinstance(entry.get("value"), str)
            or entry["value"] in {"0", "0.0"}
        ):
            raise HStarOrderOneMaterializerError("invalid sparse matrix entry")
        seen.add(coordinate)


def _matrix_from_record(record: dict[str, Any], shape: list[int]) -> Any:
    import sympy as sp

    _validate_matrix_record(record, shape)
    matrix = sp.zeros(*shape)
    for entry in record["entries"]:
        matrix[entry["row"], entry["column"]] = sp.sympify(entry["value"])
    return matrix


def _validate_c1(document: dict[str, Any], atom: str) -> None:
    _validate_hashed(document, C1_SCHEMA, f"C1 {atom}")
    if (
        document.get("basis_jet_direction") != atom
        or len(document.get("B_star_order_one_axis_matrices", [])) != 3
        or [row.get("spatial_axis") for row in document["B_star_order_one_axis_matrices"]]
        != list(AXES)
        or document.get("symmetry_residual_nonzero_entries") != 0
    ):
        raise HStarOrderOneMaterializerError("C1 packet boundary changed")
    _validate_matrix_record(document["A_star_order_one_matrix"], [11, 11])
    for row in document["B_star_order_one_axis_matrices"]:
        _validate_matrix_record(row["matrix"], [11, 11])


def _validate_c2(document: dict[str, Any], evaluation_id: str) -> None:
    _validate_hashed(document, C2_SCHEMA, f"C2 {evaluation_id}")
    matrix = document.get("H_star_plus_order_one_matrix", {})
    if (
        document.get("evaluation_id") != evaluation_id
        or not document.get("basis_bindings")
        or matrix.get("schema_version")
        != "sigma-exact-sparse-polynomial-H-star-plus-Taylor-order-one-1.0"
        or not _hash_matches(matrix)
        or matrix.get("shape") != [22, 22]
        or matrix.get("variables") != list(VARIABLES)
        or matrix.get("Taylor_order") != 1
        or matrix.get("factorial_normalization") != "1/1!=1"
        or matrix.get("distinct_matrix_cells") != len(matrix.get("entries", []))
        or document.get("symmetry_residual_nonzero_entries") != 0
        or document.get("H_star_minus_negation_residual_nonzero_entries") != 0
    ):
        raise HStarOrderOneMaterializerError("C2 packet boundary changed")


def _state(
    root: Path, config: dict[str, Any], checkpoint_dir: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    evaluations = _evaluation_packets(root, config)
    paths = _checkpoint_paths(checkpoint_dir, evaluations)
    if paths["C0"].exists():
        c0 = _load_json(paths["C0"])
        _validate_hashed(c0, C0_SCHEMA, "C0")
        if c0.get("config_sha256") != config["content_sha256"]:
            raise HStarOrderOneMaterializerError("C0 config binding changed")
    c1 = [atom for atom, path in paths["C1"].items() if path.exists()]
    c2 = [name for name, path in paths["C2"].items() if path.exists()]
    for atom in c1:
        _validate_c1(_load_json(paths["C1"][atom]), atom)
    for name in c2:
        _validate_c2(_load_json(paths["C2"][name]), name)
    return {"C0": paths["C0"].exists(), "C1_atoms": c1, "C2_evaluations": c2}, paths


def next_units(root: Path, config: dict[str, Any], checkpoint_dir: Path) -> list[str]:
    state, paths = _state(root, config, checkpoint_dir)
    if not state["C0"]:
        return ["C0"]
    missing_c1 = [atom for atom in paths["C1"] if atom not in state["C1_atoms"]]
    if missing_c1:
        return [f"C1_basis_{atom}" for atom in missing_c1]
    missing_c2 = [name for name in paths["C2"] if name not in state["C2_evaluations"]]
    return [f"C2_evaluation_{name}" for name in missing_c2]


def build_plan(root: Path, config_path: Path, checkpoint_dir: Path) -> dict[str, Any]:
    config = _load_config(root, config_path)
    build_c0(root, config)
    state, _ = _state(root, config, checkpoint_dir)
    body = {
        "schema_version": PLAN_SCHEMA,
        "status": ("ready_checkpointable_exact_H_star_plus_Taylor_order_one_materialization"),
        "config_sha256": config["content_sha256"],
        "checkpoint_directory": "caller_owned_scratch",
        "phases": [
            {"phase": "C0", "units": 1, "output": "c0-seals.json"},
            {
                "phase": "C1",
                "units": 4,
                "unit_key": "basis_jet_direction",
                "A_star_matrices_per_unit": 1,
                "B_star_axis_matrices_per_unit": 3,
                "cold_dependency": "one _symbol_data call per worker invocation",
                "durability": "each completed basis atom is sealed immediately",
            },
            {
                "phase": "C2",
                "units": 15,
                "unit_key": "polarization_evaluation",
                "cold_dependency": None,
            },
            {"phase": "C3", "units": 1, "output": "portable result.json"},
        ],
        "caps": config["caps"],
        "observed_state": state,
        "progress": {
            "basis_jet_packets_complete": len(state["C1_atoms"]),
            "basis_jet_packets_required": 4,
            "basis_action_matrices_complete": 4 * len(state["C1_atoms"]),
            "basis_action_matrices_required": 16,
            "evaluation_packets_complete": len(state["C2_evaluations"]),
            "evaluation_packets_required": 15,
        },
        "next_units": next_units(root, config, checkpoint_dir),
        "claims": {
            "H_star_plus_Taylor_order_one_packets_materialized": False,
            "K55_Taylor_order_one_registered": False,
            "manifest_advanced_beyond_79": False,
            "full_direction_sphere_D4_compatibility_proved": False,
            "global_H7_closed": False,
            "nonlinear_PDE_closure_proved": False,
            "lifespan_proved": False,
        },
        "source_bindings": {
            "config": {"path": CONFIG_PATH, "file_sha256": _file_sha256(config_path)},
            "source": {
                "path": SOURCE_PATH,
                "file_sha256": _file_sha256(root / SOURCE_PATH),
            },
            "test": {"path": TEST_PATH, "file_sha256": _file_sha256(root / TEST_PATH)},
        },
    }
    return _with_hash(body)


def worker_c1(root: Path, config: dict[str, Any], checkpoint_dir: Path) -> None:
    import sympy as sp

    from .horndeski_principal import _first_order_generalized_pencil
    from .quartic_first_order_reduction_campaign import _symbol_data

    evaluations = _evaluation_packets(root, config)
    paths = _checkpoint_paths(checkpoint_dir, evaluations)
    c0 = _load_json(paths["C0"])
    _validate_hashed(c0, C0_SCHEMA, "C0")
    missing = [atom for atom, path in paths["C1"].items() if not path.exists()]
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
    if not set(BASIS_ATOMS) <= set(jet_by_name):
        raise HStarOrderOneMaterializerError("basis jets absent from live symbol data")
    action = _first_order_generalized_pencil(data["action_symbol"], xi[0])
    zero_direction = {symbol: 0 for symbol in xi[1:]}
    b_axes = [
        action["B"].diff(symbol).subs(zero_direction).applyfunc(sp.factor) for symbol in xi[1:]
    ]
    zero_jets = {symbol: 0 for symbol in jets}
    reference = {
        **zero_jets,
        **zero_direction,
        data["alpha"]: 0,
        data["m2"]: 1,
        data["c20"]: 0,
    }
    derivative = dict(reference)
    derivative[data["alpha"]] = 1
    flat = _load_bound(root, config["upstreams"]["flat_action_metric"])
    exact = flat["exact_construction"]
    if action["A"].subs(reference).applyfunc(sp.factor) != _matrix_from_record(
        exact["A_0"], [11, 11]
    ) or b_axes[0].subs(reference).applyfunc(sp.factor) != _matrix_from_record(
        exact["B_0"], [11, 11]
    ):
        raise HStarOrderOneMaterializerError("flat action A0/B0 replay mismatch")
    maximum = config["caps"]["maximum_checkpoint_bytes"]
    for atom in missing:
        jet = jet_by_name[atom]
        a1 = action["A"].diff(jet).subs(derivative).applyfunc(sp.factor)
        b1_axes = [matrix.diff(jet).subs(derivative).applyfunc(sp.factor) for matrix in b_axes]
        if a1 != a1.T or any(matrix != matrix.T for matrix in b1_axes):
            raise HStarOrderOneMaterializerError("action derivative symmetry failed")
        document = _with_hash(
            {
                "schema_version": C1_SCHEMA,
                "C0_sha256": c0["content_sha256"],
                "basis_jet_direction": atom,
                "normalization": "alpha=1; Taylor coefficient 1/1!",
                "flat_specialization": "all jets=0; m2=1; c20=0",
                "A_star_order_one_matrix": _matrix_record(f"D_{atom}_A_star", a1),
                "B_star_order_one_axis_matrices": [
                    {
                        "spatial_axis": axis,
                        "matrix": _matrix_record(f"D_{atom}_B_star_axis_{axis}", matrix),
                    }
                    for axis, matrix in zip(AXES, b1_axes, strict=True)
                ],
                "symmetry_residual_nonzero_entries": 0,
                "counts": {
                    "A_star_matrices": 1,
                    "B_star_axis_matrices": 3,
                    "sparse_entries": sum(value != 0 for value in a1)
                    + sum(value != 0 for matrix in b1_axes for value in matrix),
                },
            }
        )
        _atomic_write_immutable(paths["C1"][atom], document, maximum)
        print(f"sealed C1 basis atom {atom}", flush=True)


def worker_c2(root: Path, config: dict[str, Any], checkpoint_dir: Path, evaluation_id: str) -> None:
    import sympy as sp

    evaluations = _evaluation_packets(root, config)
    paths = _checkpoint_paths(checkpoint_dir, evaluations)
    if evaluation_id not in paths["C2"]:
        raise HStarOrderOneMaterializerError("unknown polarization evaluation")
    evaluation = next(row for row in evaluations if row["evaluation_id"] == evaluation_id)
    a1 = sp.zeros(11)
    b1 = {axis: sp.zeros(11) for axis in AXES}
    basis_bindings = []
    for binding in evaluation["basis_bindings"]:
        atom = binding["basis_jet_direction"]
        c1 = _load_json(paths["C1"][atom])
        _validate_c1(c1, atom)
        coefficient = sp.sympify(binding["coefficient"])
        a1 += coefficient * _matrix_from_record(c1["A_star_order_one_matrix"], [11, 11])
        for row in c1["B_star_order_one_axis_matrices"]:
            b1[row["spatial_axis"]] += coefficient * _matrix_from_record(row["matrix"], [11, 11])
        basis_bindings.append(
            {
                "basis_jet_direction": atom,
                "coefficient": binding["coefficient"],
                "C1_content_sha256": c1["content_sha256"],
            }
        )
    h_constant = sp.zeros(22)
    h_constant[:11, 11:] = a1
    h_constant[11:, :11] = a1
    h_axes = []
    for axis in AXES:
        matrix = sp.zeros(22)
        matrix[:11, :11] = b1[axis]
        h_axes.append(matrix)
    if h_constant != h_constant.T or any(matrix != matrix.T for matrix in h_axes):
        raise HStarOrderOneMaterializerError("H-star order-one symmetry failed")
    entries = []
    for row in range(22):
        for column in range(22):
            item: dict[str, Any] = {"row": row, "column": column}
            if h_constant[row, column] != 0:
                item["constant"] = sp.sstr(sp.factor(h_constant[row, column]))
            linear = {
                variable: sp.sstr(sp.factor(matrix[row, column]))
                for variable, matrix in zip(VARIABLES, h_axes, strict=True)
                if matrix[row, column] != 0
            }
            if linear:
                item["linear_coefficients"] = linear
            if len(item) > 2:
                entries.append(item)
    matrix_packet = _with_hash(
        {
            "schema_version": ("sigma-exact-sparse-polynomial-H-star-plus-Taylor-order-one-1.0"),
            "shape": [22, 22],
            "variables": list(VARIABLES),
            "Taylor_order": 1,
            "factorial_normalization": "1/1!=1",
            "entries": entries,
            "distinct_matrix_cells": len(entries),
            "constant_nonzero_entries": sum(value != 0 for value in h_constant),
            "axis_nonzero_entries": [sum(value != 0 for value in matrix) for matrix in h_axes],
        }
    )
    document = _with_hash(
        {
            "schema_version": C2_SCHEMA,
            "evaluation_id": evaluation_id,
            "evaluation_content_sha256": evaluation["evaluation_content_sha256"],
            "P55_order_one_content_sha256": evaluation["content_sha256"],
            "basis_bindings": basis_bindings,
            "H_star_plus_order_one_matrix": matrix_packet,
            "exact_linearity_from_basis_packets": True,
            "symmetry_residual_nonzero_entries": 0,
            "H_star_minus_identity": "H_star_minus_order_one=-H_star_plus_order_one",
            "H_star_minus_negation_residual_nonzero_entries": 0,
        }
    )
    _atomic_write_immutable(
        paths["C2"][evaluation_id],
        document,
        config["caps"]["maximum_checkpoint_bytes"],
    )


def build_final_result(root: Path, config_path: Path, checkpoint_dir: Path) -> dict[str, Any]:
    config = _load_config(root, config_path)
    if next_units(root, config, checkpoint_dir):
        raise HStarOrderOneMaterializerError("cannot finalize incomplete checkpoints")
    evaluations = _evaluation_packets(root, config)
    paths = _checkpoint_paths(checkpoint_dir, evaluations)
    packets = []
    for evaluation in evaluations:
        name = evaluation["evaluation_id"]
        document = _load_json(paths["C2"][name])
        _validate_c2(document, name)
        packets.append(document)
    body = {
        "schema_version": FINAL_SCHEMA,
        "status": "pass_exact_15_H_star_plus_Taylor_order_one_packets_materialized",
        "config_sha256": config["content_sha256"],
        "packet_content_sha256": [row["content_sha256"] for row in packets],
        "packets": packets,
        "counts": {
            "basis_jet_packets": 4,
            "basis_A_star_order_one_matrices": 4,
            "basis_B_star_order_one_axis_matrices": 12,
            "H_star_plus_Taylor_order_one_packets": 15,
            "manifest_registered_before": 79,
            "manifest_registered_after": 79,
            "manifest_missing_after": 225,
            "full_symbol_build_calls_per_complete_C1_worker": 1,
        },
        "claims": {
            "all_15_H_star_plus_Taylor_order_one_packets_materialized": True,
            "K55_Taylor_order_one_registered": False,
            "manifest_advanced_beyond_79": False,
            "full_direction_sphere_D4_compatibility_proved": False,
            "global_H7_closed": False,
            "nonlinear_PDE_closure_proved": False,
            "lifespan_proved": False,
        },
        "source_bindings": {
            "config": {"path": CONFIG_PATH, "file_sha256": _file_sha256(config_path)},
            "source": {
                "path": SOURCE_PATH,
                "file_sha256": _file_sha256(root / SOURCE_PATH),
            },
            "test": {"path": TEST_PATH, "file_sha256": _file_sha256(root / TEST_PATH)},
        },
        "scope": (
            "Exact materialization of 15 coordinate-free H_star_plus Taylor-order-one "
            "packets only. A separate K55 consumer must construct and validate K55 "
            "order "
            "one; no manifest, D4, H7, PDE, or lifespan conclusion follows."
        ),
    }
    return _with_hash(body)


def _rss_bytes(process_id: int) -> int | None:
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            class Counters(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            psapi = ctypes.WinDLL("psapi", use_last_error=True)
            kernel.OpenProcess.argtypes = [
                wintypes.DWORD,
                wintypes.BOOL,
                wintypes.DWORD,
            ]
            kernel.OpenProcess.restype = wintypes.HANDLE
            kernel.CloseHandle.argtypes = [wintypes.HANDLE]
            psapi.GetProcessMemoryInfo.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(Counters),
                wintypes.DWORD,
            ]
            handle = kernel.OpenProcess(0x0400 | 0x0010, False, process_id)
            if not handle:
                return None
            try:
                counters = Counters()
                counters.cb = ctypes.sizeof(counters)
                if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
                    return None
                return int(counters.WorkingSetSize)
            finally:
                kernel.CloseHandle(handle)
        except (AttributeError, OSError):
            return None
    status = Path(f"/proc/{process_id}/status")
    if status.exists():
        for line in status.read_text(encoding="ascii").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    return None


def _run_supervised(command: list[str], wall_seconds: int, rss_bytes: int, poll: int) -> None:
    process = subprocess.Popen(command)
    started = time.monotonic()
    peak_rss = 0
    try:
        while process.poll() is None:
            elapsed = time.monotonic() - started
            observed = _rss_bytes(process.pid)
            if observed is not None:
                peak_rss = max(peak_rss, observed)
            if elapsed > wall_seconds or peak_rss > rss_bytes:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                reason = "wall" if elapsed > wall_seconds else "RSS"
                raise HStarOrderOneMaterializerError(
                    f"worker exceeded {reason} cap; elapsed={elapsed:.1f}s peak_rss={peak_rss}"
                )
            time.sleep(poll)
        if process.returncode != 0:
            raise HStarOrderOneMaterializerError(
                f"worker failed with exit code {process.returncode}"
            )
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def resume(root: Path, config_path: Path, checkpoint_dir: Path) -> dict[str, Any]:
    config = _load_config(root, config_path)
    evaluations = _evaluation_packets(root, config)
    paths = _checkpoint_paths(checkpoint_dir, evaluations)
    _atomic_write_immutable(
        paths["C0"], build_c0(root, config), config["caps"]["maximum_checkpoint_bytes"]
    )
    units = next_units(root, config, checkpoint_dir)
    if not units:
        return build_plan(root, config_path, checkpoint_dir)
    unit = units[0]
    phase = "C1" if unit.startswith("C1_") else "C2"
    command = [
        sys.executable,
        "-m",
        ("sigma_theory_compiler.quartic_tc2_d4_h_star_order_one_checkpointable_materializer"),
        "--worker-phase",
        phase,
        "--project-root",
        str(root),
        "--config",
        str(config_path),
        "--checkpoint-dir",
        str(checkpoint_dir),
    ]
    if phase == "C2":
        command.extend(["--evaluation-id", unit.removeprefix("C2_evaluation_")])
    _run_supervised(
        command,
        config["caps"][phase]["wall_seconds"],
        config["caps"][phase]["rss_bytes"],
        config["caps"]["poll_seconds"],
    )
    return build_plan(root, config_path, checkpoint_dir)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--finalize", action="store_true")
    mode.add_argument("--worker-phase", choices=("C1", "C2"))
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--evaluation-id")
    args = parser.parse_args(argv)
    root = args.project_root.resolve()
    config_path = (args.config or root / CONFIG_PATH).resolve()
    checkpoint_dir = (args.checkpoint_dir or root / DEFAULT_CHECKPOINT_DIR).resolve()
    if checkpoint_dir == root or checkpoint_dir.parent == checkpoint_dir:
        raise HStarOrderOneMaterializerError("checkpoint directory target is too broad")
    config = _load_config(root, config_path)
    if args.worker_phase == "C1":
        worker_c1(root, config, checkpoint_dir)
        return 0
    if args.worker_phase == "C2":
        if not args.evaluation_id:
            raise HStarOrderOneMaterializerError("C2 requires --evaluation-id")
        worker_c2(root, config, checkpoint_dir, args.evaluation_id)
        return 0
    if args.finalize:
        if args.output is None:
            raise HStarOrderOneMaterializerError("--finalize requires --output")
        result = build_final_result(root, config_path, checkpoint_dir)
        _atomic_write_immutable(
            args.output.resolve(), result, config["caps"]["maximum_checkpoint_bytes"]
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    document = (
        build_plan(root, config_path, checkpoint_dir)
        if args.plan
        else resume(root, config_path, checkpoint_dir)
    )
    if args.output is not None:
        _atomic_write_immutable(
            args.output.resolve(), document, config["caps"]["maximum_checkpoint_bytes"]
        )
    print(json.dumps(document, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
