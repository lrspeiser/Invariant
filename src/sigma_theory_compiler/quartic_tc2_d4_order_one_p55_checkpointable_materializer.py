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

PLAN_SCHEMA = "sigma-quartic-tc2-d4-order-one-p55-materializer-plan-1.0"
CONFIG_SCHEMA = "sigma-quartic-tc2-d4-order-one-p55-checkpointable-materializer-config-1.0"
C0_SCHEMA = "sigma-quartic-tc2-d4-order-one-p55-materializer-C0-seals-1.0"
C1_SCHEMA = "sigma-quartic-tc2-d4-order-one-p55-materializer-C1-basis-jet-1.0"
C2_SCHEMA = "sigma-quartic-tc2-d4-order-one-p55-materializer-C2-evaluation-1.0"
FINAL_SCHEMA = "sigma-quartic-tc2-d4-order-one-p55-materializer-result-1.0"
CONFIG_PATH = "configs/backgrounds/quartic_tc2_d4_order_one_p55_checkpointable_materializer.json"
SOURCE_PATH = (
    "src/sigma_theory_compiler/quartic_tc2_d4_order_one_p55_checkpointable_materializer.py"
)
TEST_PATH = "tests/test_quartic_tc2_d4_order_one_p55_checkpointable_materializer.py"
DEFAULT_CHECKPOINT_DIR = (
    "runs/physics-language/quartic-tc2-d4-order-one-p55-checkpointable-materializer/checkpoints"
)
AXES = (1, 2, 3)
VARIABLES = ("n1", "n2", "n3")
ORDERING = [*range(11), *range(33, 55), *range(11, 33)]


class OrderOneP55MaterializerError(ValueError):
    """Raised when an order-one P55 checkpoint fails closed."""


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
        raise OrderOneP55MaterializerError("bound path escaped project root")
    return path


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise OrderOneP55MaterializerError(
            f"cannot load JSON checkpoint {path}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise OrderOneP55MaterializerError(f"expected JSON object: {path}")
    return value


def _json_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _atomic_write_immutable(path: Path, value: dict[str, Any], maximum: int) -> None:
    data = _json_bytes(value)
    if len(data) > maximum:
        raise OrderOneP55MaterializerError(f"checkpoint exceeds byte cap: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise OrderOneP55MaterializerError(f"immutable checkpoint conflict: {path.name}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(data)
    try:
        if path.exists():
            if path.read_bytes() != data:
                raise OrderOneP55MaterializerError(f"immutable checkpoint race: {path.name}")
            temporary.unlink(missing_ok=True)
        else:
            temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_hashed(value: dict[str, Any], schema: str, label: str) -> None:
    if value.get("schema_version") != schema or not _hash_matches(value):
        raise OrderOneP55MaterializerError(f"{label} checkpoint seal mismatch")


def _load_config(root: Path, config_path: Path) -> dict[str, Any]:
    config = _load_json(config_path)
    caps = config.get("caps", {})
    target = config.get("target", {})
    if (
        config.get("schema_version") != CONFIG_SCHEMA
        or config.get("policy") != "exact_basis_jet_P55_order_one_materialization_fail_closed"
        or not _hash_matches(config)
        or set(config.get("upstreams", {}))
        != {"order_one_frontier", "flat_P55_checkpoint", "polarization_registration"}
        or config.get("basis_jet_directions") != ["G_12", "G_01", "H_01", "H_11"]
        or target.get("state_dimension") != 55
        or target.get("spatial_axes") != 3
        or target.get("basis_jet_packets") != 4
        or target.get("basis_axis_matrices") != 12
        or target.get("polarization_evaluations") != 15
        or target.get("Taylor_order") != 1
        or target.get("registered_manifest_before") != 64
        or target.get("registered_manifest_after_complete") != 79
        or target.get("required_manifest_packets") != 304
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
        raise OrderOneP55MaterializerError("invalid materializer config")
    source = config.get("live_source", {})
    source_path = _resolve_under(root, source.get("path", ""))
    if _file_sha256(source_path) != source.get("file_sha256"):
        raise OrderOneP55MaterializerError("live source file seal mismatch")
    functions = {
        node.name
        for node in ast.parse(source_path.read_text(encoding="utf-8")).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    if not set(source.get("required_functions", [])) <= functions:
        raise OrderOneP55MaterializerError("live source function contract mismatch")
    return config


def _load_bound(root: Path, binding: dict[str, str]) -> dict[str, Any]:
    path = _resolve_under(root, binding["path"])
    value = _load_json(path)
    if (
        _file_sha256(path) != binding["file_sha256"]
        or value.get("content_sha256") != binding["content_sha256"]
        or not _hash_matches(value)
    ):
        raise OrderOneP55MaterializerError(f"upstream seal mismatch: {binding['path']}")
    return value


def build_c0(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    receipts = {name: _load_bound(root, binding) for name, binding in config["upstreams"].items()}
    frontier = receipts["order_one_frontier"]
    flat = receipts["flat_P55_checkpoint"]
    polarization = receipts["polarization_registration"]
    if (
        frontier.get("counts", {}).get("registered_symbolic_input_packets") != 64
        or frontier.get("counts", {}).get("serialized_P55_Taylor_order_one_packets") != 0
        or len(flat.get("matrix_packets", [])) != 3
        or len(polarization.get("polarization_evaluations", [])) != 15
    ):
        raise OrderOneP55MaterializerError("C0 predecessor boundary changed")
    return _with_hash(
        {
            "schema_version": C0_SCHEMA,
            "config_sha256": config["content_sha256"],
            "source": config["live_source"],
            "upstreams": {
                name: {
                    "path": binding["path"],
                    "content_sha256": binding["content_sha256"],
                }
                for name, binding in config["upstreams"].items()
            },
            "basis_jet_directions": config["basis_jet_directions"],
            "spatial_axes": list(AXES),
            "frontier_BLOCK_unchanged": True,
        }
    )


def _safe_slug(value: str) -> str:
    allowed = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
    if not value or any(character not in allowed for character in value):
        raise OrderOneP55MaterializerError("unsafe checkpoint unit id")
    return value.lower()


def _checkpoint_paths(
    checkpoint_dir: Path, config: dict[str, Any], polarization: dict[str, Any]
) -> dict[str, Any]:
    atoms = config["basis_jet_directions"]
    evaluations = [row["evaluation_id"] for row in polarization["polarization_evaluations"]]
    return {
        "C0": checkpoint_dir / "c0-seals.json",
        "C1": {atom: checkpoint_dir / f"c1-basis-{_safe_slug(atom)}.json" for atom in atoms},
        "C2": {
            evaluation: checkpoint_dir / "c2-evaluations" / f"{evaluation}.json"
            for evaluation in evaluations
        },
    }


def _state(
    root: Path, config: dict[str, Any], checkpoint_dir: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    polarization = _load_bound(root, config["upstreams"]["polarization_registration"])
    paths = _checkpoint_paths(checkpoint_dir, config, polarization)
    if paths["C0"].exists():
        _validate_hashed(_load_json(paths["C0"]), C0_SCHEMA, "C0")
    c1 = [atom for atom, path in paths["C1"].items() if path.exists()]
    c2 = [name for name, path in paths["C2"].items() if path.exists()]
    for atom in c1:
        _validate_hashed(_load_json(paths["C1"][atom]), C1_SCHEMA, f"C1 {atom}")
    for name in c2:
        _validate_hashed(_load_json(paths["C2"][name]), C2_SCHEMA, f"C2 {name}")
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
        "status": "ready_checkpointable_exact_P55_Taylor_order_one_materialization",
        "config_sha256": config["content_sha256"],
        "checkpoint_directory": "caller_owned_scratch",
        "phases": [
            {"phase": "C0", "units": 1, "output": "c0-seals.json"},
            {
                "phase": "C1",
                "units": 4,
                "unit_key": "basis_jet_direction",
                "axis_matrices_per_unit": 3,
                "cold_dependency": "one _symbol_data call per worker invocation",
                "durability": "each completed basis atom is sealed immediately",
            },
            {
                "phase": "C2",
                "units": 15,
                "unit_key": "polarization_evaluation",
                "cold_dependency": None,
            },
        ],
        "caps": config["caps"],
        "observed_state": state,
        "progress": {
            "basis_jet_packets_complete": len(state["C1_atoms"]),
            "basis_jet_packets_required": 4,
            "basis_axis_matrices_complete": 3 * len(state["C1_atoms"]),
            "basis_axis_matrices_required": 12,
            "evaluation_packets_complete": len(state["C2_evaluations"]),
            "evaluation_packets_required": 15,
        },
        "next_units": next_units(root, config, checkpoint_dir),
        "claims": {
            "P55_Taylor_order_one_packets_registered": False,
            "manifest_advanced_to_79": False,
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
        {"row": row, "column": column, "value": sp.sstr(sp.factor(matrix[row, column]))}
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


def _matrix_from_record(record: dict[str, Any]) -> Any:
    import sympy as sp

    if not _hash_matches(record) or record.get("shape") != [55, 55]:
        raise OrderOneP55MaterializerError("matrix record seal/shape mismatch")
    matrix = sp.zeros(55)
    seen = set()
    for entry in record.get("entries", []):
        coordinate = (entry["row"], entry["column"])
        if coordinate in seen or not all(0 <= value < 55 for value in coordinate):
            raise OrderOneP55MaterializerError("invalid sparse matrix coordinate")
        matrix[coordinate] = sp.sympify(entry["value"])
        seen.add(coordinate)
    if len(seen) != record.get("nonzero_count"):
        raise OrderOneP55MaterializerError("sparse matrix count mismatch")
    return matrix


def worker_c1(root: Path, config: dict[str, Any], checkpoint_dir: Path) -> None:
    import sympy as sp

    from .quartic_first_order_reduction_campaign import (
        _extract_spatial_blocks,
        _full_first_order_pencil,
        _symbol_data,
    )

    polarization = _load_bound(root, config["upstreams"]["polarization_registration"])
    paths = _checkpoint_paths(checkpoint_dir, config, polarization)
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
    if not set(config["basis_jet_directions"]) <= set(jet_by_name):
        raise OrderOneP55MaterializerError("basis jets absent from live symbol data")
    coefficient_a = data["first_order"]["A"]
    b_blocks, c_blocks = _extract_spatial_blocks(
        data["first_order"]["B"], data["first_order"]["C"], list(xi[1:])
    )
    flat = _load_bound(root, config["upstreams"]["flat_P55_checkpoint"])
    flat_axes = [_matrix_from_record(record) for record in flat["matrix_packets"]]
    maximum = config["caps"]["maximum_checkpoint_bytes"]
    for atom in missing:
        jet = jet_by_name[atom]
        axes = []
        for axis_index, axis in enumerate(AXES):
            spatial = [0, 0, 0]
            spatial[axis_index] = 1
            zero_jet = {symbol: 0 for symbol in jets}
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
            physical0_original = mass0.inv() * evolution0
            physical0 = physical0_original.extract(ORDERING, ORDERING).applyfunc(sp.factor)
            if physical0 != flat_axes[axis_index]:
                raise OrderOneP55MaterializerError("flat P55 axis replay mismatch")
            mass1, evolution1 = _full_first_order_pencil(
                coefficient_a.diff(jet).subs(derivative),
                b_blocks[axis_index].diff(jet).subs(derivative),
                [matrix.diff(jet).subs(derivative) for matrix in c_blocks[axis_index]],
                spatial,
            )
            physical1_original = mass0.inv() * (evolution1 - mass1 * physical0_original)
            residual = (
                mass0 * physical1_original + mass1 * physical0_original - evolution1
            ).applyfunc(sp.factor)
            if not residual.is_zero_matrix:
                raise OrderOneP55MaterializerError("P55 order-one recurrence residual nonzero")
            physical1 = physical1_original.extract(ORDERING, ORDERING).applyfunc(sp.factor)
            axes.append(
                {
                    "spatial_axis": axis,
                    "P55_Taylor_order_one": _matrix_record(f"D_{atom}_P_axis_{axis}", physical1),
                    "recurrence_residual_nonzero_entries": 0,
                    "flat_P55_axis_content_sha256": flat["matrix_packets"][axis_index][
                        "content_sha256"
                    ],
                }
            )
        document = _with_hash(
            {
                "schema_version": C1_SCHEMA,
                "C0_sha256": c0["content_sha256"],
                "basis_jet_direction": atom,
                "normalization": "alpha=1; Taylor coefficient 1/1!",
                "axis_packets": axes,
                "counts": {
                    "spatial_axes": 3,
                    "recurrence_residual_nonzero_entries": 0,
                    "sparse_entries": sum(
                        row["P55_Taylor_order_one"]["nonzero_count"] for row in axes
                    ),
                },
            }
        )
        _atomic_write_immutable(paths["C1"][atom], document, maximum)
        print(f"sealed C1 basis atom {atom}", flush=True)


def worker_c2(root: Path, config: dict[str, Any], checkpoint_dir: Path, evaluation_id: str) -> None:
    import sympy as sp

    polarization = _load_bound(root, config["upstreams"]["polarization_registration"])
    paths = _checkpoint_paths(checkpoint_dir, config, polarization)
    if evaluation_id not in paths["C2"]:
        raise OrderOneP55MaterializerError("unknown polarization evaluation")
    evaluation = next(
        row
        for row in polarization["polarization_evaluations"]
        if row["evaluation_id"] == evaluation_id
    )
    matrices = {axis: sp.zeros(55) for axis in AXES}
    basis_bindings = []
    for atom, coefficient_text in evaluation["combined_jet_direction"].items():
        c1 = _load_json(paths["C1"][atom])
        _validate_hashed(c1, C1_SCHEMA, f"C1 {atom}")
        coefficient = sp.sympify(coefficient_text)
        for row in c1["axis_packets"]:
            matrices[row["spatial_axis"]] += coefficient * _matrix_from_record(
                row["P55_Taylor_order_one"]
            )
        basis_bindings.append(
            {
                "basis_jet_direction": atom,
                "coefficient": coefficient_text,
                "C1_content_sha256": c1["content_sha256"],
            }
        )
    entries_by_cell: dict[tuple[int, int], dict[str, str]] = {}
    axis_nonzero = []
    for axis, variable in zip(AXES, VARIABLES, strict=True):
        matrix = matrices[axis].applyfunc(sp.factor)
        axis_nonzero.append(sum(value != 0 for value in matrix))
        for row in range(55):
            for column in range(55):
                if matrix[row, column] != 0:
                    entries_by_cell.setdefault((row, column), {})[variable] = sp.sstr(
                        matrix[row, column]
                    )
    matrix_body = {
        "schema_version": "sigma-exact-sparse-polynomial-P55-Taylor-order-one-1.0",
        "shape": [55, 55],
        "variables": list(VARIABLES),
        "Taylor_order": 1,
        "factorial_normalization": "1/1!=1",
        "entries": [
            {"row": row, "column": column, "linear_coefficients": coefficients}
            for (row, column), coefficients in sorted(entries_by_cell.items())
        ],
        "distinct_matrix_cells": len(entries_by_cell),
        "axis_nonzero_entries": axis_nonzero,
    }
    matrix_packet = _with_hash(matrix_body)
    document = _with_hash(
        {
            "schema_version": C2_SCHEMA,
            "evaluation_id": evaluation_id,
            "evaluation_content_sha256": evaluation["content_sha256"],
            "basis_bindings": basis_bindings,
            "P55_Taylor_order_one_matrix": matrix_packet,
            "exact_linearity_from_basis_packets": True,
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
        raise OrderOneP55MaterializerError("cannot finalize incomplete checkpoints")
    polarization = _load_bound(root, config["upstreams"]["polarization_registration"])
    paths = _checkpoint_paths(checkpoint_dir, config, polarization)
    packets = []
    for evaluation in polarization["polarization_evaluations"]:
        document = _load_json(paths["C2"][evaluation["evaluation_id"]])
        _validate_hashed(document, C2_SCHEMA, "C2")
        packets.append(document)
    body = {
        "schema_version": FINAL_SCHEMA,
        "status": "pass_exact_15_P55_Taylor_order_one_packets_materialized",
        "config_sha256": config["content_sha256"],
        "packet_content_sha256": [row["content_sha256"] for row in packets],
        "packets": packets,
        "counts": {
            "basis_jet_packets": 4,
            "basis_axis_matrices": 12,
            "P55_Taylor_order_one_packets": 15,
            "manifest_registered_before": 64,
            "manifest_registered_after": 79,
            "manifest_missing_after": 225,
            "full_symbol_build_calls_per_complete_C1_worker": 1,
        },
        "claims": {
            "all_15_P55_Taylor_order_one_packets_materialized": True,
            "manifest_consumer_receipt_emitted": False,
            "K55_Taylor_order_one_registered": False,
            "TC2_Taylor_order_one_registered": False,
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
            "Exact materialization of the 15 coordinate-free P55 Taylor-order-one "
            "packets only. A separate consumer receipt must validate and advance the "
            "304-packet "
            "manifest; no K55, TC2, D4, H7, PDE, or lifespan conclusion follows."
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
                raise OrderOneP55MaterializerError(
                    f"worker exceeded {reason} cap; elapsed={elapsed:.1f}s peak_rss={peak_rss}"
                )
            time.sleep(poll)
        if process.returncode != 0:
            raise OrderOneP55MaterializerError(f"worker failed with exit code {process.returncode}")
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def resume(root: Path, config_path: Path, checkpoint_dir: Path) -> dict[str, Any]:
    config = _load_config(root, config_path)
    polarization = _load_bound(root, config["upstreams"]["polarization_registration"])
    paths = _checkpoint_paths(checkpoint_dir, config, polarization)
    _atomic_write_immutable(
        paths["C0"],
        build_c0(root, config),
        config["caps"]["maximum_checkpoint_bytes"],
    )
    units = next_units(root, config, checkpoint_dir)
    if not units:
        return build_plan(root, config_path, checkpoint_dir)
    unit = units[0]
    phase = "C1" if unit.startswith("C1_") else "C2"
    command = [
        sys.executable,
        "-m",
        ("sigma_theory_compiler.quartic_tc2_d4_order_one_p55_checkpointable_materializer"),
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
        raise OrderOneP55MaterializerError("checkpoint directory target is too broad")
    config = _load_config(root, config_path)
    if args.worker_phase == "C1":
        worker_c1(root, config, checkpoint_dir)
        return 0
    if args.worker_phase == "C2":
        if not args.evaluation_id:
            raise OrderOneP55MaterializerError("C2 requires --evaluation-id")
        worker_c2(root, config, checkpoint_dir, args.evaluation_id)
        return 0
    if args.finalize:
        if args.output is None:
            raise OrderOneP55MaterializerError("--finalize requires --output")
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
    print(json.dumps(document, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
