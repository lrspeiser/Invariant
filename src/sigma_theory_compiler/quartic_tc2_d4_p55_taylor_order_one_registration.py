from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

SCHEMA = "sigma-quartic-tc2-d4-p55-taylor-order-one-registration-1.0"
CONFIG_SCHEMA = "sigma-quartic-tc2-d4-p55-taylor-order-one-registration-config-1.0"
STATUS = "block_coordinate_free_D4_recurrence_emitter_missing_225_symbolic_packets"
CONFIG_PATH = "configs/backgrounds/quartic_tc2_d4_p55_taylor_order_one_registration.json"
SOURCE_PATH = "src/sigma_theory_compiler/quartic_tc2_d4_p55_taylor_order_one_registration.py"
TEST_PATH = "tests/test_quartic_tc2_d4_p55_taylor_order_one_registration.py"
OUTPUT_PATH = "runs/physics-language/quartic-tc2-d4-p55-taylor-order-one-registration/campaign.json"
REQUIRED_PACKETS = 304
PREDECESSOR_REGISTERED = 64
NEW_PACKETS = 15
REGISTERED_PACKETS = 79
MISSING_PACKETS = 225
REQUIRED_ROWS = 117_180
EXPECTED_EVALUATIONS = [
    "subset_0",
    "subset_1",
    "subset_2",
    "subset_3",
    "subset_01",
    "subset_02",
    "subset_03",
    "subset_12",
    "subset_13",
    "subset_23",
    "subset_012",
    "subset_013",
    "subset_023",
    "subset_123",
    "subset_0123",
]


class P55TaylorOrderOneRegistrationError(ValueError):
    """Raised when exact P55 Taylor-order-one registration fails closed."""


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


def _resolve_under(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if root != path and root not in path.parents:
        raise P55TaylorOrderOneRegistrationError("bound path escaped project root")
    return path


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise P55TaylorOrderOneRegistrationError(f"expected JSON object: {path}")
    return value


def _load_bound(root: Path, binding: dict[str, str]) -> dict[str, Any]:
    path = _resolve_under(root, binding["path"])
    value = _load_json(path)
    if (
        _file_sha256(path) != binding["file_sha256"]
        or value.get("content_sha256") != binding["content_sha256"]
        or not _hash_matches(value)
    ):
        raise P55TaylorOrderOneRegistrationError(f"upstream mismatch: {binding['path']}")
    return value


def _validate_config(config: dict[str, Any]) -> None:
    if (
        set(config)
        != {
            "schema_version",
            "policy",
            "upstreams",
            "target",
            "seals",
            "content_sha256",
        }
        or config.get("schema_version") != CONFIG_SCHEMA
        or config.get("policy") != "consume_only_exact_sealed_P55_order_one_packets_fail_closed"
        or not _hash_matches(config)
        or set(config.get("upstreams", {})) != {"manifest_predecessor", "order_one_materializer"}
        or config.get("target")
        != {
            "required_symbolic_input_packets": REQUIRED_PACKETS,
            "predecessor_registered_packets": PREDECESSOR_REGISTERED,
            "new_P55_Taylor_order_one_packets": NEW_PACKETS,
            "registered_packets": REGISTERED_PACKETS,
            "missing_packets": MISSING_PACKETS,
            "required_output_rows": REQUIRED_ROWS,
        }
        or any(config.get("seals", {}).values())
    ):
        raise P55TaylorOrderOneRegistrationError("invalid P55 order-one config")


def _validate_source_bindings(root: Path, result: dict[str, Any]) -> None:
    bindings = result.get("source_bindings", {})
    if set(bindings) != {"config", "source", "test"}:
        raise P55TaylorOrderOneRegistrationError("materializer source binding set changed")
    for binding in bindings.values():
        path = _resolve_under(root, binding["path"])
        if _file_sha256(path) != binding["file_sha256"]:
            raise P55TaylorOrderOneRegistrationError("materializer source provenance changed")


def _validate_matrix(matrix: dict[str, Any]) -> dict[str, int]:
    if (
        not _hash_matches(matrix)
        or matrix.get("schema_version") != "sigma-exact-sparse-polynomial-P55-Taylor-order-one-1.0"
        or matrix.get("shape") != [55, 55]
        or matrix.get("variables") != ["n1", "n2", "n3"]
        or matrix.get("Taylor_order") != 1
        or matrix.get("factorial_normalization") != "1/1!=1"
        or len(matrix.get("axis_nonzero_entries", [])) != 3
    ):
        raise P55TaylorOrderOneRegistrationError("P55 order-one matrix boundary changed")
    coordinates = set()
    coefficients = 0
    variables = set()
    for entry in matrix.get("entries", []):
        coordinate = (entry.get("row"), entry.get("column"))
        linear = entry.get("linear_coefficients", {})
        if (
            coordinate in coordinates
            or not all(isinstance(index, int) and 0 <= index < 55 for index in coordinate)
            or not linear
            or not set(linear) <= {"n1", "n2", "n3"}
            or any(not isinstance(value, str) or value in {"0", "0.0"} for value in linear.values())
        ):
            raise P55TaylorOrderOneRegistrationError("invalid sparse P55 order-one entry")
        coordinates.add(coordinate)
        coefficients += len(linear)
        variables.update(linear)
    if len(coordinates) != matrix.get("distinct_matrix_cells"):
        raise P55TaylorOrderOneRegistrationError("P55 distinct-cell count changed")
    return {
        "distinct_cells": len(coordinates),
        "linear_coefficients": coefficients,
        "active_variables": len(variables),
    }


def _validate_materializer(root: Path, result: dict[str, Any]) -> dict[str, Any]:
    counts = result.get("counts", {})
    packets = result.get("packets", [])
    if (
        result.get("status") != "pass_exact_15_P55_Taylor_order_one_packets_materialized"
        or counts
        != {
            "P55_Taylor_order_one_packets": 15,
            "basis_axis_matrices": 12,
            "basis_jet_packets": 4,
            "full_symbol_build_calls_per_complete_C1_worker": 1,
            "manifest_missing_after": 225,
            "manifest_registered_after": 79,
            "manifest_registered_before": 64,
        }
        or result.get("packet_content_sha256")
        != [packet.get("content_sha256") for packet in packets]
        or len(packets) != NEW_PACKETS
        or [packet.get("evaluation_id") for packet in packets] != EXPECTED_EVALUATIONS
    ):
        raise P55TaylorOrderOneRegistrationError("materializer result boundary changed")
    _validate_source_bindings(root, result)
    matrix_counts = []
    for packet in packets:
        if (
            not _hash_matches(packet)
            or packet.get("schema_version")
            != "sigma-quartic-tc2-d4-order-one-p55-materializer-C2-evaluation-1.0"
            or packet.get("exact_linearity_from_basis_packets") is not True
            or not isinstance(packet.get("evaluation_content_sha256"), str)
            or not packet.get("basis_bindings")
            or any(
                set(binding) != {"basis_jet_direction", "coefficient", "C1_content_sha256"}
                for binding in packet["basis_bindings"]
            )
        ):
            raise P55TaylorOrderOneRegistrationError("invalid materializer C2 packet")
        matrix_counts.append(_validate_matrix(packet["P55_Taylor_order_one_matrix"]))
    return {
        "packets": packets,
        "matrix_counts": matrix_counts,
        "packet_content_sha256": result["packet_content_sha256"],
    }


def build_campaign(project_root: Path, config_path: Path) -> dict[str, Any]:
    root = project_root.resolve()
    config = _load_json(config_path)
    _validate_config(config)
    predecessor = _load_bound(root, config["upstreams"]["manifest_predecessor"])
    result = _load_bound(root, config["upstreams"]["order_one_materializer"])
    validated = _validate_materializer(root, result)
    if (
        predecessor.get("status")
        != "block_coordinate_free_D4_recurrence_emitter_missing_240_symbolic_packets"
        or predecessor.get("counts", {}).get("registered_symbolic_input_packets")
        != PREDECESSOR_REGISTERED
    ):
        raise P55TaylorOrderOneRegistrationError("manifest predecessor boundary changed")
    manifest = deepcopy(predecessor["required_symbolic_input_manifest"])
    records = {row["input_id"]: row for row in manifest}
    family = records.get("polarized_P55_Taylor_packets")
    if (
        len(records) != 8
        or family is None
        or family.get("required_packets") != 75
        or family.get("registered_packets") != 15
        or family.get("registered_Taylor_orders") != [0]
        or family.get("missing_Taylor_orders") != [1, 2, 3, 4]
        or len(family.get("packet_content_sha256", [])) != 15
    ):
        raise P55TaylorOrderOneRegistrationError("P55 manifest family boundary changed")
    family.update(
        {
            "registered_packets": 30,
            "registered_Taylor_orders": [0, 1],
            "missing_Taylor_orders": [2, 3, 4],
            "status": "partially_registered_all_15_packets_at_Taylor_orders_zero_and_one",
            "packet_content_sha256": [
                *family["packet_content_sha256"],
                *validated["packet_content_sha256"],
            ],
            "Taylor_order_one_materializer_content_sha256": result["content_sha256"],
        }
    )
    if (
        sum(row["registered_packets"] for row in manifest) != REGISTERED_PACKETS
        or sum(row["required_packets"] for row in manifest) != REQUIRED_PACKETS
    ):
        raise P55TaylorOrderOneRegistrationError("updated manifest total mismatch")
    missing = [
        {
            "input_id": row["input_id"],
            "required_packets": row["required_packets"],
            "registered_packets": row["registered_packets"],
            "missing_packets": row["required_packets"] - row["registered_packets"],
        }
        for row in manifest
        if row["registered_packets"] < row["required_packets"]
    ]
    if sum(row["missing_packets"] for row in missing) != MISSING_PACKETS:
        raise P55TaylorOrderOneRegistrationError("updated missing count changed")
    claims = {key: value for key, value in predecessor["claims"].items() if value is False}
    claims.update(
        {
            "all_15_P55_Taylor_order_one_packets_registered": True,
            "P55_Taylor_orders_zero_and_one_registered": True,
            "P55_Taylor_orders_two_through_four_registered": False,
            "manifest_recomputed_from_exact_packets": True,
            "cold_full_symbol_build_used_in_consumer": False,
        }
    )
    totals = validated["matrix_counts"]
    body = {
        "schema_version": SCHEMA,
        "status": STATUS,
        "errors": [],
        "config_sha256": config["content_sha256"],
        "upstream_bindings": {
            name: {**binding, "verified": True} for name, binding in config["upstreams"].items()
        },
        "registered_P55_Taylor_order_one_packets": validated["packets"],
        "required_symbolic_input_manifest": manifest,
        "remaining_missing_inputs": missing,
        "bounded_emitter_checkpoint": {
            "complete": False,
            "first_missing_input": "polarized_P55_Taylor_packets/orders_2_through_4",
            "required_output_rows": REQUIRED_ROWS,
            "emitted_output_rows": 0,
            "emitted_rhs_rows": 0,
            "emitted_sparse_entries": 0,
        },
        "phase_two": {
            "decision": "BLOCK",
            "admitted": False,
            "attempted": False,
            "blocker": "225 required symbolic input packets remain unregistered",
        },
        "counts": {
            "upstream_seals_verified": 2,
            "required_symbolic_input_packets": REQUIRED_PACKETS,
            "predecessor_registered_symbolic_input_packets": PREDECESSOR_REGISTERED,
            "new_P55_Taylor_order_one_packets_registered": NEW_PACKETS,
            "registered_symbolic_input_packets": REGISTERED_PACKETS,
            "missing_symbolic_input_packets": MISSING_PACKETS,
            "P55_Taylor_order_one_distinct_matrix_cells": sum(
                row["distinct_cells"] for row in totals
            ),
            "P55_Taylor_order_one_linear_coefficients": sum(
                row["linear_coefficients"] for row in totals
            ),
            "P55_Taylor_order_one_packets_with_all_three_axes": sum(
                row["active_variables"] == 3 for row in totals
            ),
            "materializer_basis_jet_packets": 4,
            "materializer_basis_axis_matrices": 12,
            "consumer_full_symbol_build_calls": 0,
            "required_output_rows": REQUIRED_ROWS,
            "emitted_output_rows": 0,
            "phase_two_solve_attempts": 0,
        },
        "claims": claims,
        "negative_controls": {
            "accept_materializer_with_missing_packet": {"rejected": True},
            "accept_resealed_sparse_matrix_tamper": {"rejected": True},
            "count_order_one_packet_as_orders_two_through_four": {"rejected": True},
            "infer_K55_or_TC2_order_one_from_P55_order_one": {"rejected": True},
            "emit_rows_with_225_missing_packets": {"rejected": True},
            "promote_partial_P55_family_to_D4_or_H7": {"rejected": True},
        },
        "source_bindings": {
            "config": {"path": CONFIG_PATH, "file_sha256": _file_sha256(config_path)},
            "source": {
                "path": SOURCE_PATH,
                "file_sha256": _file_sha256(root / SOURCE_PATH),
            },
            "test": {"path": TEST_PATH, "file_sha256": _file_sha256(root / TEST_PATH)},
        },
        "data_seals": deepcopy(config["seals"]),
        "scope": (
            "Consumes and registers exactly the 15 sealed coordinate-free P55 "
            "Taylor-order-one packets, advancing the checked recurrence manifest from "
            "64 to 79 of 304. P55 orders two through four and all missing K55, TC2, and "
            "lower-Sylvester packets remain absent. No recurrence row, D4 theorem, H7 "
            "closure, PDE theorem, lifespan, physical no-go, or candidate rejection follows."
        ),
    }
    return {**body, "content_sha256": _content_hash(body)}


def validate_campaign(document: dict[str, Any], project_root: Path) -> None:
    expected = build_campaign(project_root.resolve(), project_root.resolve() / CONFIG_PATH)
    if document != expected or not _hash_matches(document):
        raise P55TaylorOrderOneRegistrationError("campaign replay mismatch")


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
