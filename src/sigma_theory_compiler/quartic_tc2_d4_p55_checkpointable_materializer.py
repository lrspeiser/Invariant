from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

SCHEMA = "sigma-quartic-tc2-d4-p55-checkpointable-materializer-plan-1.0"
CONFIG_SCHEMA = "sigma-quartic-tc2-d4-p55-checkpointable-materializer-config-1.0"
C0_SCHEMA = "sigma-quartic-tc2-d4-p55-materializer-C0-seals-1.0"
C1_SCHEMA = "sigma-quartic-tc2-d4-p55-materializer-C1-flat-blocks-1.0"
C2_SCHEMA = "sigma-quartic-tc2-d4-p55-materializer-C2-axis-packet-1.0"
C3_SCHEMA = "sigma-quartic-tc2-d4-p55-materializer-C3-linearity-row-1.0"
C4_SCHEMA = "sigma-quartic-tc2-d4-p55-materializer-C4-minimal-polynomial-row-1.0"
FINAL_SCHEMA = "sigma-quartic-tc2-d4-p55-checkpointable-materializer-result-1.0"
CONFIG_PATH = "configs/backgrounds/quartic_tc2_d4_p55_checkpointable_materializer.json"
DEFAULT_CHECKPOINT_DIR = (
    "runs/physics-language/quartic-tc2-d4-p55-checkpointable-materializer/checkpoints"
)
STATE_DIMENSION = 55
SECOND_ORDER_DIMENSION = 11
AXES = (1, 2, 3)
EXPECTED_NONZEROS = 48


class P55MaterializerError(ValueError):
    """Raised when an exact materializer seal or checkpoint fails closed."""


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


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise P55MaterializerError(f"cannot load JSON checkpoint {path}: {error}") from error
    if not isinstance(value, dict):
        raise P55MaterializerError(f"expected JSON object: {path}")
    return value


def _resolve_under(root: Path, relative: str | Path) -> Path:
    path = (root / relative).resolve()
    if root != path and root not in path.parents:
        raise P55MaterializerError("path escaped project root")
    return path


def _validate_hashed(value: dict[str, Any], schema: str, *, label: str) -> None:
    if value.get("schema_version") != schema or not _hash_matches(value):
        raise P55MaterializerError(f"{label} schema/content seal mismatch")


def _atomic_write_immutable(path: Path, document: dict[str, Any], maximum_bytes: int) -> None:
    encoded = json.dumps(document, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    if len(encoded) > maximum_bytes:
        raise P55MaterializerError(f"checkpoint exceeds byte cap: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != encoded:
            raise P55MaterializerError(f"immutable checkpoint conflict: {path.name}")
        return
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != encoded:
                raise P55MaterializerError(f"immutable checkpoint race: {path.name}")
    finally:
        temporary.unlink(missing_ok=True)


def _validate_config(config: dict[str, Any]) -> None:
    target = config.get("target", {})
    caps = config.get("caps", {})
    ordered = target.get("ordered_indices", [])
    cap_names = {"C1", "C2_axis", "C3", "C4"}
    if (
        config.get("schema_version") != CONFIG_SCHEMA
        or config.get("policy") != "exact_live_construction_fail_closed_no_count_inference"
        or not _hash_matches(config)
        or target.get("state_dimension") != STATE_DIMENSION
        or target.get("second_order_dimension") != SECOND_ORDER_DIMENSION
        or target.get("spatial_axes") != len(AXES)
        or target.get("expected_nonzero_entries_each_validation_only") != EXPECTED_NONZEROS
        or len(ordered) != STATE_DIMENSION
        or sorted(ordered) != list(range(STATE_DIMENSION))
        or not cap_names.issubset(caps)
        or any(
            not isinstance(caps[name].get(field), int) or caps[name][field] <= 0
            for name in cap_names
            for field in ("wall_seconds", "rss_bytes")
        )
        or not isinstance(caps.get("poll_seconds"), int)
        or caps["poll_seconds"] <= 0
        or not isinstance(caps.get("maximum_checkpoint_bytes"), int)
        or caps["maximum_checkpoint_bytes"] <= 0
    ):
        raise P55MaterializerError("invalid materializer config")


def _load_config(root: Path, config_path: Path) -> dict[str, Any]:
    config = _load_json(config_path)
    _validate_config(config)
    return config


def _validate_source_contract(path: Path, required: list[str]) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    if not set(required).issubset(functions):
        raise P55MaterializerError("live source function contract mismatch")


def build_c0(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    source_binding = config["live_source"]
    source_path = _resolve_under(root, source_binding["path"])
    if _file_sha256(source_path) != source_binding["file_sha256"]:
        raise P55MaterializerError("live source file seal mismatch")
    _validate_source_contract(source_path, source_binding["required_functions"])

    receipts: dict[str, dict[str, Any]] = {}
    for label in ("authoritative_block_receipt", "predecessor"):
        binding = config[label]
        path = _resolve_under(root, binding["path"])
        document = _load_json(path)
        if (
            not _hash_matches(document)
            or document.get("content_sha256") != binding["content_sha256"]
        ):
            raise P55MaterializerError(f"{label} content seal mismatch")
        receipts[label] = {
            "path": binding["path"],
            "content_sha256": binding["content_sha256"],
            "status": document.get("status"),
        }
    required_status = config["authoritative_block_receipt"]["required_status"]
    if receipts["authoritative_block_receipt"]["status"] != required_status:
        raise P55MaterializerError("authoritative BLOCK status mismatch")
    return _with_hash(
        {
            "schema_version": C0_SCHEMA,
            "config_sha256": config["content_sha256"],
            "source": {
                "path": source_binding["path"],
                "file_sha256": source_binding["file_sha256"],
                "required_functions": source_binding["required_functions"],
            },
            "receipts": receipts,
            "authoritative_BLOCK_unchanged": True,
        }
    )


def _checkpoint_paths(checkpoint_dir: Path) -> dict[str, Any]:
    return {
        "C0": checkpoint_dir / "c0-seals.json",
        "C1": checkpoint_dir / "c1-flat-blocks.json",
        "C2": {axis: checkpoint_dir / f"c2-axis-{axis}.json" for axis in AXES},
        "C3": {
            row: checkpoint_dir / "c3-linearity" / f"row-{row:03d}.json"
            for row in range(STATE_DIMENSION)
        },
        "C4": {
            row: checkpoint_dir / "c4-minimal-polynomial" / f"row-{row:03d}.json"
            for row in range(STATE_DIMENSION)
        },
    }


def _checkpoint_state(checkpoint_dir: Path) -> dict[str, Any]:
    paths = _checkpoint_paths(checkpoint_dir)
    return {
        "C0": paths["C0"].exists(),
        "C1": paths["C1"].exists(),
        "C2_axes": [axis for axis, path in paths["C2"].items() if path.exists()],
        "C3_rows": [row for row, path in paths["C3"].items() if path.exists()],
        "C4_rows": [row for row, path in paths["C4"].items() if path.exists()],
    }


def next_units(checkpoint_dir: Path) -> list[str]:
    state = _checkpoint_state(checkpoint_dir)
    if not state["C0"]:
        return ["C0"]
    if not state["C1"]:
        return ["C1"]
    missing_axes = [axis for axis in AXES if axis not in state["C2_axes"]]
    if missing_axes:
        return [f"C2_axis_{axis}" for axis in missing_axes]
    missing_c3 = [row for row in range(STATE_DIMENSION) if row not in state["C3_rows"]]
    if missing_c3:
        return [f"C3_row_{row:03d}" for row in missing_c3]
    missing_c4 = [row for row in range(STATE_DIMENSION) if row not in state["C4_rows"]]
    if missing_c4:
        return [f"C4_row_{row:03d}" for row in missing_c4]
    return []


def build_plan(root: Path, config_path: Path, checkpoint_dir: Path) -> dict[str, Any]:
    config = _load_config(root, config_path)
    # Planning is read-only, but it still performs C0 so stale sources/receipts
    # cannot produce a misleading ready plan.
    build_c0(root, config)
    state = _checkpoint_state(checkpoint_dir)
    body = {
        "schema_version": SCHEMA,
        "status": "ready_checkpointable_exact_materialization",
        "config_sha256": config["content_sha256"],
        # The caller owns this resumable scratch directory.  Persisting its
        # host path would make an otherwise deterministic plan machine-bound.
        "checkpoint_directory": "caller_owned_scratch",
        "authoritative_BLOCK_receipt_mutated": False,
        "phases": [
            {"phase": "C0", "units": 1, "output": "c0-seals.json"},
            {
                "phase": "C1",
                "units": 1,
                "output": "c1-flat-blocks.json",
                "cold_dependency": "_symbol_data returns before first durable checkpoint",
            },
            {"phase": "C2", "units": 3, "unit_key": "spatial_axis"},
            {"phase": "C3", "units": 55, "unit_key": "matrix_row", "entries_per_unit": 55},
            {"phase": "C4", "units": 55, "unit_key": "matrix_row", "entries_per_unit": 55},
        ],
        "caps": config["caps"],
        "observed_state": state,
        "next_units": next_units(checkpoint_dir),
        "claims": {
            "P55_spatial_pencil_registered": False,
            "P55_minimal_polynomial_certified": False,
            "full_direction_sphere_D4_compatibility_proved": False,
            "global_H7_closed": False,
            "nonlinear_PDE_closure_proved": False,
            "lifespan_proved": False,
        },
    }
    return _with_hash(body)


def _matrix_record(name: str, matrix: Any) -> dict[str, Any]:
    import sympy as sp

    entries = [
        {"row": row, "column": column, "value": sp.sstr(matrix[row, column])}
        for row in range(matrix.rows)
        for column in range(matrix.cols)
        if matrix[row, column] != 0
    ]
    return _with_hash(
        {
            "name": name,
            "shape": [matrix.rows, matrix.cols],
            "nonzero_count": len(entries),
            "entries": entries,
        }
    )


def _matrix_from_record(record: dict[str, Any], shape: tuple[int, int]) -> Any:
    import sympy as sp

    if not _hash_matches(record) or record.get("shape") != list(shape):
        raise P55MaterializerError(f"matrix record seal/shape mismatch: {record.get('name')}")
    matrix = sp.zeros(*shape)
    seen: set[tuple[int, int]] = set()
    for entry in record.get("entries", []):
        row, column = entry.get("row"), entry.get("column")
        if (
            not isinstance(row, int)
            or not isinstance(column, int)
            or not (0 <= row < shape[0] and 0 <= column < shape[1])
            or (row, column) in seen
        ):
            raise P55MaterializerError("invalid or duplicate sparse coordinate")
        value = sp.sympify(entry.get("value"), evaluate=True)
        if value == 0:
            raise P55MaterializerError("zero stored as sparse entry")
        matrix[row, column] = value
        seen.add((row, column))
    if len(seen) != record.get("nonzero_count"):
        raise P55MaterializerError("sparse entry count mismatch")
    return matrix


def worker_c1(root: Path, config: dict[str, Any], checkpoint_dir: Path) -> None:
    import sympy as sp

    from .quartic_first_order_reduction_campaign import _extract_spatial_blocks, _symbol_data

    paths = _checkpoint_paths(checkpoint_dir)
    c0 = _load_json(paths["C0"])
    _validate_hashed(c0, C0_SCHEMA, label="C0")
    data = _symbol_data()
    substitutions = {data["alpha"]: sp.Integer(0), data["m2"]: sp.Integer(1)}
    if "c20" in data:
        substitutions[data["c20"]] = sp.Integer(0)
    substitutions.update({symbol: 0 for symbol in data["gradient_lower"]})
    substitutions.update({symbol: 0 for symbol in data["hessian_lower"].free_symbols})
    substitutions.update({symbol: 0 for symbol in data["einstein_upper"].free_symbols})
    xi = list(data["xi_lower"])
    coefficient_a = data["first_order"]["A"].subs(substitutions).applyfunc(sp.factor)
    b_blocks, c_blocks = _extract_spatial_blocks(
        data["first_order"]["B"], data["first_order"]["C"], xi[1:]
    )
    b_blocks = [block.subs(substitutions).applyfunc(sp.factor) for block in b_blocks]
    c_blocks = [
        [block.subs(substitutions).applyfunc(sp.factor) for block in row] for row in c_blocks
    ]
    records = [_matrix_record("A", coefficient_a)]
    records.extend(_matrix_record(f"B_{axis}", b_blocks[axis - 1]) for axis in AXES)
    records.extend(
        _matrix_record(f"C_{left}_{right}", c_blocks[left - 1][right - 1])
        for left in AXES
        for right in AXES
    )
    document = _with_hash(
        {
            "schema_version": C1_SCHEMA,
            "config_sha256": config["content_sha256"],
            "C0_sha256": c0["content_sha256"],
            "construction": "live _symbol_data then exact flat substitution and _extract_spatial_blocks",
            "flat_substitutions": config["flat_substitutions"],
            "matrix_records": records,
            "counts": {
                "A_matrices": 1,
                "B_axis_matrices": 3,
                "C_ordered_axis_pair_matrices": 9,
                "matrix_records": len(records),
                "observed_sparse_entries": sum(item["nonzero_count"] for item in records),
            },
        }
    )
    _atomic_write_immutable(paths["C1"], document, config["caps"]["maximum_checkpoint_bytes"])


def _load_c1(checkpoint_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    document = _load_json(_checkpoint_paths(checkpoint_dir)["C1"])
    _validate_hashed(document, C1_SCHEMA, label="C1")
    records = {record.get("name"): record for record in document.get("matrix_records", [])}
    expected = {"A", *(f"B_{axis}" for axis in AXES)} | {
        f"C_{left}_{right}" for left in AXES for right in AXES
    }
    if set(records) != expected:
        raise P55MaterializerError("C1 matrix record set mismatch")
    return document, records


def worker_c2(root: Path, config: dict[str, Any], checkpoint_dir: Path, axis: int) -> None:
    import sympy as sp

    from .quartic_first_order_reduction_campaign import _full_first_order_pencil

    if axis not in AXES:
        raise P55MaterializerError("invalid C2 axis")
    c1, records = _load_c1(checkpoint_dir)
    coefficient_a = _matrix_from_record(records["A"], (11, 11))
    b_axis = _matrix_from_record(records[f"B_{axis}"], (11, 11))
    c_flux = [_matrix_from_record(records[f"C_{axis}_{right}"], (11, 11)) for right in AXES]
    direction = [sp.Integer(index == axis) for index in AXES]
    mass, evolution = _full_first_order_pencil(coefficient_a, b_axis, c_flux, direction)
    physical = mass.inv() * evolution
    ordering = config["target"]["ordered_indices"]
    physical = physical.extract(ordering, ordering).applyfunc(sp.factor)
    packet = _matrix_record(f"P_{axis}", physical)
    if packet["nonzero_count"] != EXPECTED_NONZEROS:
        raise P55MaterializerError(
            f"axis {axis} observed {packet['nonzero_count']} nonzeros; expected-count validation failed"
        )
    document = _with_hash(
        {
            "schema_version": C2_SCHEMA,
            "config_sha256": config["content_sha256"],
            "C1_sha256": c1["content_sha256"],
            "spatial_axis": axis,
            "expected_nonzero_count_used_as_input": False,
            "expected_nonzero_count_validation_only": EXPECTED_NONZEROS,
            "matrix_packet": packet,
        }
    )
    _atomic_write_immutable(
        _checkpoint_paths(checkpoint_dir)["C2"][axis],
        document,
        config["caps"]["maximum_checkpoint_bytes"],
    )


def _load_axes(checkpoint_dir: Path) -> tuple[list[Any], list[dict[str, Any]]]:
    matrices, documents = [], []
    for axis in AXES:
        document = _load_json(_checkpoint_paths(checkpoint_dir)["C2"][axis])
        _validate_hashed(document, C2_SCHEMA, label=f"C2 axis {axis}")
        if document.get("spatial_axis") != axis:
            raise P55MaterializerError("C2 axis identity mismatch")
        matrices.append(_matrix_from_record(document["matrix_packet"], (55, 55)))
        documents.append(document)
    return matrices, documents


def worker_c3(root: Path, config: dict[str, Any], checkpoint_dir: Path) -> None:
    import sympy as sp

    from .quartic_first_order_reduction_campaign import _full_first_order_pencil

    paths = _checkpoint_paths(checkpoint_dir)
    c1, records = _load_c1(checkpoint_dir)
    axes, axis_documents = _load_axes(checkpoint_dir)
    coefficient_a = _matrix_from_record(records["A"], (11, 11))
    b_blocks = [_matrix_from_record(records[f"B_{axis}"], (11, 11)) for axis in AXES]
    c_blocks = [
        [_matrix_from_record(records[f"C_{left}_{right}"], (11, 11)) for right in AXES]
        for left in AXES
    ]
    n = list(sp.symbols("n1:4"))
    b_direction = sum((n[index] * b_blocks[index] for index in range(3)), sp.zeros(11))
    c_flux = [
        sum((n[left] * c_blocks[left][right] for left in range(3)), sp.zeros(11))
        for right in range(3)
    ]
    mass, evolution = _full_first_order_pencil(coefficient_a, b_direction, c_flux, n)
    ordering = config["target"]["ordered_indices"]
    direct = (mass.inv() * evolution).extract(ordering, ordering)
    registered = sum((n[index] * axes[index] for index in range(3)), sp.zeros(55))
    residual = direct - registered
    axis_hashes = [document["content_sha256"] for document in axis_documents]
    for row in range(STATE_DIMENSION):
        target = paths["C3"][row]
        if target.exists():
            existing = _load_json(target)
            _validate_hashed(existing, C3_SCHEMA, label=f"C3 row {row}")
            continue
        entries = [sp.factor(residual[row, column]) for column in range(STATE_DIMENSION)]
        if any(value != 0 for value in entries):
            raise P55MaterializerError(f"C3 linearity residual nonzero in row {row}")
        document = _with_hash(
            {
                "schema_version": C3_SCHEMA,
                "config_sha256": config["content_sha256"],
                "C1_sha256": c1["content_sha256"],
                "C2_sha256": axis_hashes,
                "row": row,
                "entries_checked": STATE_DIMENSION,
                "nonzero_residuals": 0,
                "formula": "P(n)=n1*P1+n2*P2+n3*P3",
            }
        )
        _atomic_write_immutable(target, document, config["caps"]["maximum_checkpoint_bytes"])


def _sphere_groebner(n1: Any, n2: Any, n3: Any) -> Any:
    import sympy as sp

    return sp.groebner(
        [n1**2 + n2**2 + n3**2 - 1],
        n1,
        n2,
        n3,
        extension=sp.sqrt(2),
    )


def worker_c4(root: Path, config: dict[str, Any], checkpoint_dir: Path) -> None:
    import sympy as sp

    paths = _checkpoint_paths(checkpoint_dir)
    for row, path in paths["C3"].items():
        document = _load_json(path)
        _validate_hashed(document, C3_SCHEMA, label=f"C3 row {row}")
        if document.get("nonzero_residuals") != 0:
            raise P55MaterializerError("C3 is not a zero linearity certificate")
    axes, axis_documents = _load_axes(checkpoint_dir)
    n1, n2, n3 = sp.symbols("n1 n2 n3")
    pencil = n1 * axes[0] + n2 * axes[1] + n3 * axes[2]
    identity = sp.eye(55)
    square = pencil * pencil
    residual = pencil * (square - identity) * (square - identity / 4) * (square - identity / 9)
    # The live matrices contain rational multiples of sqrt(2).  An implicit
    # integer-domain Groebner basis rejects those exact coefficients, so bind
    # the reducer to the smallest registered algebraic coefficient field.
    reducer = _sphere_groebner(n1, n2, n3)
    axis_hashes = [document["content_sha256"] for document in axis_documents]
    for row in range(STATE_DIMENSION):
        target = paths["C4"][row]
        if target.exists():
            existing = _load_json(target)
            _validate_hashed(existing, C4_SCHEMA, label=f"C4 row {row}")
            continue
        raw_nonzero = 0
        nonzero_remainders = 0
        for column in range(STATE_DIMENSION):
            entry = sp.expand(residual[row, column])
            raw_nonzero += int(entry != 0)
            remainder = reducer.reduce(entry)[1]
            nonzero_remainders += int(remainder != 0)
        if nonzero_remainders:
            raise P55MaterializerError(f"C4 sphere remainder nonzero in row {row}")
        document = _with_hash(
            {
                "schema_version": C4_SCHEMA,
                "config_sha256": config["content_sha256"],
                "C2_sha256": axis_hashes,
                "row": row,
                "entries_reduced": STATE_DIMENSION,
                "raw_nonzero_entries": raw_nonzero,
                "nonzero_remainders": 0,
                "sphere_relation": config["target"]["sphere_relation"],
                "minimal_polynomial": config["target"]["minimal_polynomial"],
            }
        )
        _atomic_write_immutable(target, document, config["caps"]["maximum_checkpoint_bytes"])


def build_final_result(root: Path, config_path: Path, checkpoint_dir: Path) -> dict[str, Any]:
    """Collapse complete scratch checkpoints into one portable exact result."""
    root = root.resolve()
    config = _load_config(root, config_path)
    if next_units(checkpoint_dir):
        raise P55MaterializerError("cannot finalize incomplete P55 checkpoints")
    paths = _checkpoint_paths(checkpoint_dir)
    c0 = _load_json(paths["C0"])
    c1 = _load_json(paths["C1"])
    _validate_hashed(c0, C0_SCHEMA, label="C0")
    _validate_hashed(c1, C1_SCHEMA, label="C1")
    _, axis_documents = _load_axes(checkpoint_dir)
    matrix_packets = [document["matrix_packet"] for document in axis_documents]
    if any(packet.get("nonzero_count") != EXPECTED_NONZEROS for packet in matrix_packets):
        raise P55MaterializerError("final P55 sparse count mismatch")

    c3_documents = []
    c4_documents = []
    for row in range(STATE_DIMENSION):
        c3 = _load_json(paths["C3"][row])
        c4 = _load_json(paths["C4"][row])
        _validate_hashed(c3, C3_SCHEMA, label=f"C3 row {row}")
        _validate_hashed(c4, C4_SCHEMA, label=f"C4 row {row}")
        if c3.get("nonzero_residuals") != 0 or c4.get("nonzero_remainders") != 0:
            raise P55MaterializerError("final P55 row certificate is nonzero")
        c3_documents.append(c3)
        c4_documents.append(c4)

    c3_hashes = [document["content_sha256"] for document in c3_documents]
    c4_hashes = [document["content_sha256"] for document in c4_documents]
    source_path = Path(__file__).resolve()
    test_path = root / "tests/test_quartic_tc2_d4_p55_checkpointable_materializer.py"
    body = {
        "schema_version": FINAL_SCHEMA,
        "status": "pass_exact_flat_reference_P55_spatial_pencil_registration",
        "config_sha256": config["content_sha256"],
        "matrix_packets": matrix_packets,
        "certificate_roots": {
            "C0_sha256": c0["content_sha256"],
            "C1_sha256": c1["content_sha256"],
            "C2_sha256": [document["content_sha256"] for document in axis_documents],
            "C3_row_root_sha256": hashlib.sha256(_canonical_bytes(c3_hashes)).hexdigest(),
            "C4_row_root_sha256": hashlib.sha256(_canonical_bytes(c4_hashes)).hexdigest(),
        },
        "counts": {
            "matrix_packets": 3,
            "sparse_entries": sum(packet["nonzero_count"] for packet in matrix_packets),
            "linearity_entries_certified": sum(
                document["entries_checked"] for document in c3_documents
            ),
            "linearity_nonzero_residuals": 0,
            "minimal_polynomial_entries_reduced": sum(
                document["entries_reduced"] for document in c4_documents
            ),
            "minimal_polynomial_nonzero_remainders": 0,
            "raw_nonzero_polynomial_entries": sum(
                document["raw_nonzero_entries"] for document in c4_documents
            ),
        },
        "claims": {
            "flat_reference_P55_spatial_pencils_registered": True,
            "flat_reference_P55_linearity_certified": True,
            "flat_reference_P55_sphere_minimal_polynomial_certified": True,
            "full_direction_sphere_D4_compatibility_proved": False,
            "global_H7_closed": False,
            "nonlinear_PDE_closure_proved": False,
            "lifespan_proved": False,
            "promotion_authorized": False,
        },
        "source_bindings": {
            "config": {
                "path": config_path.resolve().relative_to(root).as_posix(),
                "file_sha256": _file_sha256(config_path),
            },
            "source": {
                "path": source_path.relative_to(root).as_posix(),
                "file_sha256": _file_sha256(source_path),
            },
            "test": {
                "path": test_path.relative_to(root).as_posix(),
                "file_sha256": _file_sha256(test_path),
            },
        },
        "scope": (
            "exact three-axis flat-reference 55x55 P55 registration, linearity, and sphere "
            "minimal-polynomial certificate only; full D4 compatibility, nonlinear H7, PDE "
            "closure, lifespan, and promotion remain false"
        ),
    }
    return _with_hash(body)


def validate_final_result(result: dict[str, Any], root: Path, config_path: Path) -> None:
    import sympy as sp

    expected_keys = {
        "certificate_roots",
        "claims",
        "config_sha256",
        "content_sha256",
        "counts",
        "matrix_packets",
        "schema_version",
        "scope",
        "source_bindings",
        "status",
    }
    if set(result) != expected_keys or not _hash_matches(result):
        raise P55MaterializerError("final P55 result schema/content seal mismatch")
    config = _load_config(root, config_path)
    if (
        result.get("schema_version") != FINAL_SCHEMA
        or result.get("config_sha256") != config["content_sha256"]
    ):
        raise P55MaterializerError("final P55 config binding mismatch")
    for label, path in {
        "config": config_path,
        "source": Path(__file__).resolve(),
        "test": root / "tests/test_quartic_tc2_d4_p55_checkpointable_materializer.py",
    }.items():
        binding = result["source_bindings"].get(label, {})
        if binding.get("path") != path.resolve().relative_to(root.resolve()).as_posix():
            raise P55MaterializerError(f"final P55 {label} path binding mismatch")
        if binding.get("file_sha256") != _file_sha256(path):
            raise P55MaterializerError(f"final P55 {label} file binding mismatch")

    packets = result.get("matrix_packets")
    if not isinstance(packets, list) or len(packets) != len(AXES):
        raise P55MaterializerError("final P55 packet count mismatch")
    matrices = [_matrix_from_record(packet, (55, 55)) for packet in packets]
    if any(packet.get("nonzero_count") != EXPECTED_NONZEROS for packet in packets):
        raise P55MaterializerError("final P55 sparse count mismatch")
    n1, n2, n3 = sp.symbols("n1 n2 n3")
    pencil = n1 * matrices[0] + n2 * matrices[1] + n3 * matrices[2]
    identity = sp.eye(55)
    square = pencil * pencil
    residual = pencil * (square - identity) * (square - identity / 4) * (square - identity / 9)
    reducer = _sphere_groebner(n1, n2, n3)
    raw_nonzero = 0
    nonzero_remainders = 0
    for entry in residual:
        expanded = sp.expand(entry)
        raw_nonzero += int(expanded != 0)
        nonzero_remainders += int(reducer.reduce(expanded)[1] != 0)
    expected_counts = {
        "matrix_packets": 3,
        "sparse_entries": 144,
        "linearity_entries_certified": 3025,
        "linearity_nonzero_residuals": 0,
        "minimal_polynomial_entries_reduced": 3025,
        "minimal_polynomial_nonzero_remainders": 0,
        "raw_nonzero_polynomial_entries": raw_nonzero,
    }
    if nonzero_remainders or result.get("counts") != expected_counts:
        raise P55MaterializerError("final P55 exact polynomial replay mismatch")
    expected_claims = {
        "flat_reference_P55_spatial_pencils_registered": True,
        "flat_reference_P55_linearity_certified": True,
        "flat_reference_P55_sphere_minimal_polynomial_certified": True,
        "full_direction_sphere_D4_compatibility_proved": False,
        "global_H7_closed": False,
        "nonlinear_PDE_closure_proved": False,
        "lifespan_proved": False,
        "promotion_authorized": False,
    }
    if result.get("claims") != expected_claims:
        raise P55MaterializerError("final P55 claim boundary mismatch")


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
            kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
            kernel.OpenProcess.restype = wintypes.HANDLE
            kernel.CloseHandle.argtypes = [wintypes.HANDLE]
            kernel.CloseHandle.restype = wintypes.BOOL
            psapi.GetProcessMemoryInfo.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(Counters),
                wintypes.DWORD,
            ]
            psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
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


def _run_supervised(command: list[str], *, wall_seconds: int, rss_bytes: int, poll: int) -> None:
    process = subprocess.Popen(command)
    started = time.monotonic()
    peak_rss = 0
    next_progress = 60.0
    try:
        while process.poll() is None:
            elapsed = time.monotonic() - started
            observed_rss = _rss_bytes(process.pid)
            if observed_rss is not None:
                peak_rss = max(peak_rss, observed_rss)
            if elapsed > wall_seconds or peak_rss > rss_bytes:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                reason = "wall" if elapsed > wall_seconds else "RSS"
                raise P55MaterializerError(
                    f"worker exceeded {reason} cap; elapsed={elapsed:.1f}s peak_rss={peak_rss}"
                )
            if elapsed >= next_progress:
                print(
                    f"materializer worker progress: elapsed={elapsed:.0f}s peak_rss={peak_rss}",
                    flush=True,
                )
                next_progress += 60.0
            time.sleep(poll)
        if process.returncode != 0:
            raise P55MaterializerError(f"worker failed with exit code {process.returncode}")
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def resume(root: Path, config_path: Path, checkpoint_dir: Path) -> dict[str, Any]:
    config = _load_config(root, config_path)
    maximum = config["caps"]["maximum_checkpoint_bytes"]
    paths = _checkpoint_paths(checkpoint_dir)
    c0 = build_c0(root, config)
    _atomic_write_immutable(paths["C0"], c0, maximum)
    units = next_units(checkpoint_dir)
    if not units:
        return build_plan(root, config_path, checkpoint_dir)
    unit = units[0]
    if unit == "C1":
        phase, extra = "C1", []
    elif unit.startswith("C2_axis_"):
        phase, extra = "C2", ["--axis", unit.rsplit("_", 1)[1]]
    elif unit.startswith("C3_row_"):
        phase, extra = "C3", []
    else:
        phase, extra = "C4", []
    cap_name = "C2_axis" if phase == "C2" else phase
    command = [
        sys.executable,
        "-m",
        "sigma_theory_compiler.quartic_tc2_d4_p55_checkpointable_materializer",
        "--worker-phase",
        phase,
        "--project-root",
        str(root),
        "--config",
        str(config_path),
        "--checkpoint-dir",
        str(checkpoint_dir),
        *extra,
    ]
    _run_supervised(
        command,
        wall_seconds=config["caps"][cap_name]["wall_seconds"],
        rss_bytes=config["caps"][cap_name]["rss_bytes"],
        poll=config["caps"]["poll_seconds"],
    )
    return build_plan(root, config_path, checkpoint_dir)


def _run_worker(
    phase: str, root: Path, config_path: Path, checkpoint_dir: Path, axis: int | None
) -> None:
    config = _load_config(root, config_path)
    if phase == "C1":
        worker_c1(root, config, checkpoint_dir)
    elif phase == "C2" and axis is not None:
        worker_c2(root, config, checkpoint_dir, axis)
    elif phase == "C3":
        worker_c3(root, config, checkpoint_dir)
    elif phase == "C4":
        worker_c4(root, config, checkpoint_dir)
    else:
        raise P55MaterializerError("invalid worker phase")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--finalize", action="store_true")
    mode.add_argument("--worker-phase", choices=("C1", "C2", "C3", "C4"))
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--axis", type=int)
    args = parser.parse_args(argv)
    root = args.project_root.resolve()
    config_path = (args.config or root / CONFIG_PATH).resolve()
    checkpoint_dir = (args.checkpoint_dir or root / DEFAULT_CHECKPOINT_DIR).resolve()
    if checkpoint_dir == root or checkpoint_dir.parent == checkpoint_dir:
        raise P55MaterializerError("checkpoint directory target is too broad")
    if args.worker_phase:
        _run_worker(args.worker_phase, root, config_path, checkpoint_dir, args.axis)
        return 0
    if args.finalize:
        if args.output is None:
            raise P55MaterializerError("--finalize requires --output")
        result = build_final_result(root, config_path, checkpoint_dir)
        _atomic_write_immutable(
            args.output.resolve(),
            result,
            _load_config(root, config_path)["caps"]["maximum_checkpoint_bytes"],
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    document = (
        build_plan(root, config_path, checkpoint_dir)
        if args.plan
        else resume(root, config_path, checkpoint_dir)
    )
    print(json.dumps(document, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
