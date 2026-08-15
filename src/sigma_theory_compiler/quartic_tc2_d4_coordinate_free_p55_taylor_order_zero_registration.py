from __future__ import annotations

import argparse
import ast
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

SCHEMA = "sigma-quartic-tc2-d4-coordinate-free-p55-taylor-order-zero-registration-1.0"
CONFIG_SCHEMA = "sigma-quartic-tc2-d4-coordinate-free-p55-taylor-order-zero-registration-config-1.0"
STATUS = "block_coordinate_free_D4_recurrence_emitter_missing_270_symbolic_packets"
PREDECESSOR_STATUS = "block_coordinate_free_D4_recurrence_emitter_missing_285_symbolic_packets"
P55_STATUS = "pass_exact_flat_reference_P55_spatial_pencil_registration"
SELECTOR_STATUS = "pass_exact_fourth_jet_minimal_selector_manifest_no_evaluations_tube_fail_closed"
CONFIG_PATH = (
    "configs/backgrounds/quartic_tc2_d4_coordinate_free_p55_taylor_order_zero_registration.json"
)
SOURCE_PATH = (
    "src/sigma_theory_compiler/quartic_tc2_d4_coordinate_free_p55_taylor_order_zero_registration.py"
)
TEST_PATH = "tests/test_quartic_tc2_d4_coordinate_free_p55_taylor_order_zero_registration.py"
REQUIRED_PACKETS = 304
PREDECESSOR_REGISTERED = 19
NEW_PACKETS = 15
REGISTERED_PACKETS = 34
MISSING_PACKETS = 270
REQUIRED_ROWS = 117_180
ACTIVE_INDICES = (0, 2, 3, 9)
EXPECTED_UPSTREAMS = {
    "predecessor": {
        "path": "runs/physics-language/quartic-tc2-d4-coordinate-free-sphere-normal-form-reducer-registration/campaign.json",
        "file_sha256": "03b399ed6817dbea87ab561ab70ef6c7e0a2fc7065fcf9e11108af667f116d7c",
        "content_sha256": "2feafb6dfb1eb3a65035b337e780892ee36550f411be014492e9e3b89604c418",
    },
    "P55_checkpoint": {
        "path": "runs/physics-language/quartic-tc2-d4-p55-checkpointable-materializer/result.json",
        "file_sha256": "9e79245d45c248d5abf1b2fd11b12bb7ca0c857286d31f8bf46b7ec7c43490b2",
        "content_sha256": "25ced42ac83bd592332f3d3a8bc97eeaac9c55b32ba4bb2be3e3da1fa6b503b7",
    },
    "polarization_selector": {
        "path": "runs/physics-language/quartic-tc2-fourth-jet-range-obligation-campaign/campaign.json",
        "file_sha256": "1bcdad6095cdb6ca483748da2ed08196ad8f5d73e4782586cd0c505991c03b05",
        "content_sha256": "b87e76db34c828fe2d9ec1623c34af4ead33cd9f627140bc4dd776999ee9462e",
    },
}
EXPECTED_POLARIZATION_SOURCE = {
    "path": "src/sigma_theory_compiler/quartic_tc2_d4_parity_cubic_generic_direction_campaign.py",
    "file_sha256": "02e10e12db83fd6120ae13f441f025598401aa5037ef7a53fbd43361c9cbfd4b",
    "active_indices": [0, 2, 3, 9],
    "required_functions": ["_generic_directional_packet", "_polarized_payload"],
}


class P55TaylorOrderZeroRegistrationError(ValueError):
    """Raised when exact P55 Taylor-order-zero registration fails closed."""


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
        raise P55TaylorOrderZeroRegistrationError(f"expected JSON object: {path}")
    return value


def _resolve_under(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if root != path and root not in path.parents:
        raise P55TaylorOrderZeroRegistrationError("bound path escaped project root")
    return path


def _validate_config(config: dict[str, Any]) -> None:
    if (
        config.get("schema_version") != CONFIG_SCHEMA
        or config.get("policy") != "register_only_exactly_derivable_P55_Taylor_orders_fail_closed"
        or not _hash_matches(config)
        or config.get("target")
        != {
            "required_symbolic_input_packets": REQUIRED_PACKETS,
            "predecessor_registered_packets": PREDECESSOR_REGISTERED,
            "P55_Taylor_packets_required": 75,
            "P55_order_zero_packets_expected": NEW_PACKETS,
            "expected_registered_packets": REGISTERED_PACKETS,
            "expected_missing_packets": MISSING_PACKETS,
            "required_output_rows": REQUIRED_ROWS,
        }
        or config.get("polarization_source", {}).get("active_indices") != list(ACTIVE_INDICES)
        or config.get("upstreams") != EXPECTED_UPSTREAMS
        or config.get("polarization_source") != EXPECTED_POLARIZATION_SOURCE
        or config.get("resource_caps")
        != {
            "maximum_axis_sparse_entries": 144,
            "maximum_polarization_evaluations": 15,
            "maximum_registered_Taylor_order": 0,
            "maximum_output_rows_emitted": 0,
        }
    ):
        raise P55TaylorOrderZeroRegistrationError("invalid order-zero config")


def _load_bound(root: Path, binding: dict[str, Any], status: str) -> dict[str, Any]:
    path = _resolve_under(root, binding["path"])
    document = _load_json(path)
    if (
        _file_sha256(path) != binding.get("file_sha256")
        or not _hash_matches(document)
        or document.get("content_sha256") != binding.get("content_sha256")
        or document.get("status") != status
        or document.get("errors", []) != []
    ):
        raise P55TaylorOrderZeroRegistrationError(f"upstream mismatch: {binding['path']}")
    return document


def _validate_polarization_source(root: Path, binding: dict[str, Any]) -> None:
    path = _resolve_under(root, binding["path"])
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    definitions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    required = binding.get("required_functions")
    semantic_markers = (
        "physical_original[0] = mass[0].inv() * evolution[0]",
        "extra = {alpha: 0, c20: 0}",
        'if physical[0] != reference["physical0"]:',
        "for subset_size in range(1, JET_ORDER + 1):",
    )
    if (
        _file_sha256(path) != binding.get("file_sha256")
        or required != ["_generic_directional_packet", "_polarized_payload"]
        or not set(required).issubset(definitions)
        or any(marker not in text for marker in semantic_markers)
    ):
        raise P55TaylorOrderZeroRegistrationError("polarization source contract mismatch")


def _coordinate_free_pencil(axis_packets: list[dict[str, Any]]) -> dict[str, Any]:
    if [packet.get("name") for packet in axis_packets] != ["P_1", "P_2", "P_3"]:
        raise P55TaylorOrderZeroRegistrationError("P55 axis packet order mismatch")
    grouped: dict[tuple[int, int], dict[str, str]] = {}
    axis_hashes: list[str] = []
    for axis, packet in enumerate(axis_packets, start=1):
        if (
            packet.get("shape") != [55, 55]
            or packet.get("nonzero_count") != 48
            or len(packet.get("entries", [])) != 48
            or not _hash_matches(packet)
        ):
            raise P55TaylorOrderZeroRegistrationError("invalid P55 axis packet")
        axis_hashes.append(packet["content_sha256"])
        for entry in packet["entries"]:
            key = (entry["row"], entry["column"])
            grouped.setdefault(key, {})[f"n{axis}"] = entry["value"]
    entries = [
        {
            "row": row,
            "column": column,
            "linear_coefficients": grouped[(row, column)],
        }
        for row, column in sorted(grouped)
    ]
    body = {
        "schema_version": "sigma-exact-sparse-linear-P55-coordinate-free-matrix-1.0",
        "shape": [55, 55],
        "variables": ["n1", "n2", "n3"],
        "identity": "P(n)=n1*P_1+n2*P_2+n3*P_3",
        "axis_packet_content_sha256": axis_hashes,
        "axis_sparse_entries_consumed": sum(len(packet["entries"]) for packet in axis_packets),
        "distinct_matrix_cells": len(entries),
        "entries": entries,
    }
    return {**body, "content_sha256": _content_hash(body)}


def _polarization_evaluations(selector: dict[str, Any]) -> list[dict[str, Any]]:
    basis = selector.get("selector", {}).get("basis_directions", [])
    if len(basis) != 15 or [row.get("basis_index") for row in basis] != list(range(15)):
        raise P55TaylorOrderZeroRegistrationError("polarization basis mismatch")
    selected = [basis[index] for index in ACTIVE_INDICES]
    if any(not row.get("direction") for row in selected):
        raise P55TaylorOrderZeroRegistrationError("empty selected polarization direction")
    evaluations: list[dict[str, Any]] = []
    for size in range(1, 5):
        sign = -1 if (4 - size) % 2 else 1
        for subset in itertools.combinations(range(4), size):
            components: dict[str, str] = {}
            for local_index in subset:
                direction = selected[local_index]["direction"]
                if set(components).intersection(direction):
                    raise P55TaylorOrderZeroRegistrationError(
                        "selected direction supports unexpectedly overlap"
                    )
                components.update(direction)
            body = {
                "evaluation_id": f"subset_{''.join(str(index) for index in subset)}",
                "selected_basis_indices": [ACTIVE_INDICES[index] for index in subset],
                "selected_active_positions": [
                    selected[index]["active_position"] for index in subset
                ],
                "combined_jet_direction": components,
                "polarization_weight": sign,
            }
            evaluations.append({**body, "content_sha256": _content_hash(body)})
    if len(evaluations) != NEW_PACKETS:
        raise P55TaylorOrderZeroRegistrationError("polarization evaluation count mismatch")
    return evaluations


def _order_zero_packets(
    evaluations: list[dict[str, Any]], pencil: dict[str, Any]
) -> list[dict[str, Any]]:
    packets = []
    for evaluation in evaluations:
        body = {
            "schema_version": "sigma-coordinate-free-P55-Taylor-polarization-packet-1.0",
            "evaluation_id": evaluation["evaluation_id"],
            "evaluation_content_sha256": evaluation["content_sha256"],
            "Taylor_order": 0,
            "factorial_normalization": "1/0!=1",
            "shape": [55, 55],
            "coordinate_free_matrix_content_sha256": pencil["content_sha256"],
            "identity": "P55_evaluation_order_0(n)=P(n)",
            "jet_direction_independent_at_flat_reference": True,
        }
        packets.append({**body, "content_sha256": _content_hash(body)})
    return packets


def _missing_serialization_contract(evaluations: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "sigma-coordinate-free-P55-Taylor-orders-one-through-four-input-contract-1.0",
        "status": "MISSING_REQUIRED_SERIALIZATION",
        "evaluation_content_sha256": [row["content_sha256"] for row in evaluations],
        "required_Taylor_orders": [1, 2, 3, 4],
        "required_output_packets": 60,
        "minimal_direct_contract": {
            "matrix_packets": 60,
            "shape_each": [55, 55],
            "required_fields": [
                "evaluation_content_sha256",
                "Taylor_order",
                "factorial_normalization_1_over_k_factorial",
                "exact_sparse_polynomial_entries_in_n1_n2_n3",
                "ordered_state_indices",
                "matrix_content_sha256",
            ],
        },
        "sufficient_recurrence_input_alternative": {
            "per_evaluation": (
                "exact M_0 inverse and exact Taylor coefficients M_k(n), E_k(n) for k=1..4"
            ),
            "recurrence": "P_k=M_0^{-1}*(E_k-sum_{i=1}^k M_i*P_{k-i})",
            "required_residual": "M_0*P_k+sum_{i=1}^k M_i*P_{k-i}-E_k=0 entrywise",
        },
        "why_current_checkpoint_is_insufficient": (
            "the checkpoint stores only the flat-substituted three axis matrices P_j; "
            "flat substitution erased every state-jet derivative needed for orders 1..4"
        ),
        "forbidden_inferences": [
            "infer any order 1..4 packet as zero",
            "differentiate the spatial direction n instead of the state jets",
            "treat 15 order-zero copies as evidence for a higher Taylor order",
            "recover M_k or E_k from P_1,P_2,P_3",
        ],
    }


def build_campaign(project_root: Path, config_path: Path) -> dict[str, Any]:
    root = project_root.resolve()
    config = _load_json(config_path)
    _validate_config(config)
    predecessor = _load_bound(root, config["upstreams"]["predecessor"], PREDECESSOR_STATUS)
    p55 = _load_bound(root, config["upstreams"]["P55_checkpoint"], P55_STATUS)
    selector = _load_bound(root, config["upstreams"]["polarization_selector"], SELECTOR_STATUS)
    _validate_polarization_source(root, config["polarization_source"])
    pencil = _coordinate_free_pencil(p55.get("matrix_packets", []))
    evaluations = _polarization_evaluations(selector)
    packets = _order_zero_packets(evaluations, pencil)
    manifest = json.loads(json.dumps(predecessor["required_symbolic_input_manifest"]))
    records = {row["input_id"]: row for row in manifest}
    family = records.get("polarized_P55_Taylor_packets")
    if (
        len(records) != 8
        or family is None
        or family.get("required_packets") != 75
        or family.get("registered_packets") != 0
    ):
        raise P55TaylorOrderZeroRegistrationError("predecessor manifest boundary mismatch")
    family.update(
        {
            "registered_packets": NEW_PACKETS,
            "status": "partially_registered_all_15_Taylor_order_zero_packets",
            "registered_Taylor_orders": [0],
            "missing_Taylor_orders": [1, 2, 3, 4],
            "packet_content_sha256": [packet["content_sha256"] for packet in packets],
            "coordinate_free_matrix_content_sha256": pencil["content_sha256"],
        }
    )
    if (
        sum(row["registered_packets"] for row in manifest) != REGISTERED_PACKETS
        or sum(row["required_packets"] for row in manifest) != REQUIRED_PACKETS
    ):
        raise P55TaylorOrderZeroRegistrationError("updated manifest total mismatch")
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
        raise P55TaylorOrderZeroRegistrationError("updated missing total mismatch")
    claims = {key: False for key, value in predecessor["claims"].items() if value is False}
    claims.update(
        {
            "all_15_P55_Taylor_order_zero_packets_registered": True,
            "coordinate_free_P55_order_zero_pencil_constructed_from_axes": True,
            "P55_Taylor_orders_one_through_four_registered": False,
            "manifest_recomputed_from_exact_packets": True,
        }
    )
    body = {
        "schema_version": SCHEMA,
        "status": STATUS,
        "errors": [],
        "config_sha256": config["content_sha256"],
        "upstream_bindings": {
            name: {**binding, "verified": True} for name, binding in config["upstreams"].items()
        },
        "polarization_source_binding": {
            **config["polarization_source"],
            "verified": True,
        },
        "coordinate_free_P55_order_zero_matrix": pencil,
        "polarization_evaluations": evaluations,
        "registered_P55_Taylor_order_zero_packets": packets,
        "missing_P55_Taylor_serialization_contract": _missing_serialization_contract(evaluations),
        "required_symbolic_input_manifest": manifest,
        "remaining_missing_inputs": missing,
        "bounded_emitter_checkpoint": {
            "complete": False,
            "first_missing_input": "polarized_P55_Taylor_packets/orders_1_through_4",
            "required_output_rows": REQUIRED_ROWS,
            "emitted_output_rows": 0,
            "emitted_rhs_rows": 0,
            "emitted_sparse_entries": 0,
        },
        "phase_two": {
            "decision": "BLOCK",
            "admitted": False,
            "attempted": False,
            "blocker": "270 required symbolic input packets remain unregistered",
        },
        "counts": {
            "upstream_seals_verified": 3,
            "required_symbolic_input_packets": REQUIRED_PACKETS,
            "predecessor_registered_symbolic_input_packets": PREDECESSOR_REGISTERED,
            "new_P55_Taylor_order_zero_packets_registered": NEW_PACKETS,
            "registered_symbolic_input_packets": REGISTERED_PACKETS,
            "missing_symbolic_input_packets": MISSING_PACKETS,
            "P55_axis_sparse_entries_consumed": 144,
            "polarization_evaluations_registered": 15,
            "P55_Taylor_orders_registered": 1,
            "P55_Taylor_orders_missing": 4,
            "required_output_rows": REQUIRED_ROWS,
            "emitted_output_rows": 0,
            "phase_two_solve_attempts": 0,
        },
        "claims": claims,
        "negative_controls": {
            "infer_orders_one_through_four_as_zero": {"rejected": True},
            "differentiate_spatial_direction_instead_of_state_jets": {"rejected": True},
            "recover_erased_state_derivatives_from_flat_axis_packets": {"rejected": True},
            "count_order_zero_packet_as_higher_order_packet": {"rejected": True},
            "emit_rows_with_270_missing_packets": {"rejected": True},
            "promote_partial_P55_family_to_full_D4_or_H7": {"rejected": True},
        },
        "source_bindings": {
            "config": {"path": CONFIG_PATH, "file_sha256": _file_sha256(config_path)},
            "source": {"path": SOURCE_PATH, "file_sha256": _file_sha256(root / SOURCE_PATH)},
            "test": {"path": TEST_PATH, "file_sha256": _file_sha256(root / TEST_PATH)},
        },
        "scope": (
            "Registers only the 15 Taylor-order-zero P55 polarization packets, whose "
            "coordinate-free matrix is the sealed flat-reference pencil P(n). Orders one "
            "through four require state-jet derivative serialization erased by the flat "
            "checkpoint. No recurrence row, D4 theorem, H7 closure, PDE theorem, lifespan, "
            "or candidate rejection follows."
        ),
    }
    return {**body, "content_sha256": _content_hash(body)}


def validate_campaign(document: dict[str, Any], project_root: Path) -> None:
    expected = build_campaign(project_root.resolve(), project_root.resolve() / CONFIG_PATH)
    if document != expected or not _hash_matches(document):
        raise P55TaylorOrderZeroRegistrationError("campaign replay mismatch")


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
