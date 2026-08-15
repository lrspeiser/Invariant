from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

SCHEMA = "sigma-quartic-tc2-d4-order-one-serialization-frontier-gate-1.0"
CONFIG_SCHEMA = "sigma-quartic-tc2-d4-order-one-serialization-frontier-config-1.0"
STATUS = "block_coordinate_free_P55_Taylor_order_one_serialization_absent"
CONFIG_PATH = "configs/backgrounds/quartic_tc2_d4_order_one_serialization_frontier_gate.json"
SOURCE_PATH = "src/sigma_theory_compiler/quartic_tc2_d4_order_one_serialization_frontier_gate.py"
TEST_PATH = "tests/test_quartic_tc2_d4_order_one_serialization_frontier_gate.py"
OUTPUT_PATH = (
    "runs/physics-language/quartic-tc2-d4-order-one-serialization-frontier-gate/campaign.json"
)
REQUIRED_PACKETS = 304
REGISTERED_PACKETS = 64
MISSING_PACKETS = 240
TARGET_PACKETS = 15
REQUIRED_ROWS = 117_180


class OrderOneSerializationFrontierError(ValueError):
    """Raised when the exact order-one serialization audit fails closed."""


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
        raise OrderOneSerializationFrontierError("bound path escaped project root")
    return path


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise OrderOneSerializationFrontierError(f"expected JSON object: {path}")
    return value


def _load_bound(root: Path, binding: dict[str, str]) -> dict[str, Any]:
    path = _resolve_under(root, binding["path"])
    value = _load_json(path)
    if (
        _file_sha256(path) != binding["file_sha256"]
        or value.get("content_sha256") != binding["content_sha256"]
        or not _hash_matches(value)
    ):
        raise OrderOneSerializationFrontierError(f"order-one upstream mismatch: {binding['path']}")
    return value


def _validate_raw_binding(root: Path, binding: dict[str, str]) -> None:
    path = _resolve_under(root, binding["path"])
    if not path.is_file() or _file_sha256(path) != binding["file_sha256"]:
        raise OrderOneSerializationFrontierError("recurrence-source binding changed")


def _validate_config(config: dict[str, Any]) -> None:
    target = config.get("target", {})
    if (
        set(config)
        != {
            "schema_version",
            "policy",
            "upstreams",
            "recurrence_source",
            "target",
            "seals",
            "content_sha256",
        }
        or config.get("schema_version") != CONFIG_SCHEMA
        or config.get("policy")
        != (
            "audit_exact_order_one_inputs_and_block_without_coordinate_free_state_jet_serialization"
        )
        or not _hash_matches(config)
        or set(config.get("upstreams", {}))
        != {
            "manifest_predecessor",
            "P55_order_zero_registration",
            "full_reference_Sylvester",
        }
        or target
        != {
            "required_symbolic_input_packets": REQUIRED_PACKETS,
            "predecessor_registered_packets": REGISTERED_PACKETS,
            "predecessor_missing_packets": MISSING_PACKETS,
            "target_family": "polarized_P55_Taylor_packets",
            "target_Taylor_order": 1,
            "target_evaluations": TARGET_PACKETS,
            "admissible_serialized_target_packets": 0,
            "new_registered_packets": 0,
            "registered_packets": REGISTERED_PACKETS,
            "missing_packets": MISSING_PACKETS,
            "required_output_rows": REQUIRED_ROWS,
        }
        or any(config.get("seals", {}).values())
    ):
        raise OrderOneSerializationFrontierError("invalid order-one frontier config")


def _manifest_records(document: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["input_id"]: row for row in document["required_symbolic_input_manifest"]}


def _audit_predecessor(predecessor: dict[str, Any]) -> None:
    records = _manifest_records(predecessor)
    if (
        predecessor.get("status")
        != "block_coordinate_free_D4_recurrence_emitter_missing_240_symbolic_packets"
        or predecessor.get("counts", {}).get("registered_symbolic_input_packets")
        != REGISTERED_PACKETS
        or predecessor.get("counts", {}).get("missing_symbolic_input_packets") != MISSING_PACKETS
        or len(records) != 8
        or records["polarized_P55_Taylor_packets"].get("registered_Taylor_orders") != [0]
        or records["polarized_P55_Taylor_packets"].get("missing_Taylor_orders") != [1, 2, 3, 4]
        or records["polarized_K55_Taylor_packets"].get("registered_Taylor_orders") != [0]
        or records["polarized_TC2_Taylor_packets"].get("registered_Taylor_orders") != [0]
        or records["lower_Sylvester_correction_recurrence"].get("registered_packets") != 0
    ):
        raise OrderOneSerializationFrontierError("predecessor manifest boundary changed")


def _audit_P55_serialization(
    p55: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    packets = p55.get("registered_P55_Taylor_order_zero_packets", [])
    contract = p55.get("missing_P55_Taylor_serialization_contract", {})
    evaluations = p55.get("polarization_evaluations", [])
    if (
        len(packets) != TARGET_PACKETS
        or len(evaluations) != TARGET_PACKETS
        or any(packet.get("Taylor_order") != 0 for packet in packets)
        or contract.get("status") != "MISSING_REQUIRED_SERIALIZATION"
        or contract.get("required_Taylor_orders") != [1, 2, 3, 4]
        or contract.get("required_output_packets") != 60
        or contract.get("minimal_direct_contract", {}).get("matrix_packets") != 60
        or contract.get("sufficient_recurrence_input_alternative", {}).get("recurrence")
        != "P_k=M_0^{-1}*(E_k-sum_{i=1}^k M_i*P_{k-i})"
    ):
        raise OrderOneSerializationFrontierError("P55 serialization boundary changed")
    return evaluations, contract


def _audit_reference_only(reference: dict[str, Any]) -> dict[str, Any]:
    packet = reference.get("common_full_reference_Sylvester_packet", {})
    controls = reference.get("generic_full_Sylvester_control", {})
    negative = controls.get("negative_controls", {}).get(
        "claim_reference_solution_is_variable_coefficient_solution", {}
    )
    column10 = next(
        (row for row in packet.get("TC2_columns", []) if row.get("low_field_column") == 10),
        None,
    )
    if (
        packet.get("reference") != "flat zero jet, M2=1, direction e1"
        or len(packet.get("deltaK_column10_entries", [])) != 24
        or column10 is None
        or column10.get("Sylvester_residual_zero") is not True
        or column10.get("deltaK_Hermitian") is not True
        or reference.get("counts", {}).get("variable_coefficient_solvability_proofs") != 0
        or negative.get("rejected") is not True
        or negative.get("missing")
        != "D_Y of the eigenspace-diagonal solvability blocks and of deltaK"
    ):
        raise OrderOneSerializationFrontierError("reference-only Sylvester boundary changed")
    return {
        "serialized_reference_packets": 1,
        "serialized_reference_direction": "e1",
        "serialized_deltaK_entries": 24,
        "admissible_coordinate_free_deltaK_packets": 0,
        "variable_coefficient_solvability_proofs": 0,
        "reference_promotion_rejected": True,
        "missing_derivative_schema": negative["missing"],
        "reference_packet_content_sha256": packet["content_sha256"],
    }


def _required_order_one_packets(
    evaluations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    packets = []
    for evaluation in evaluations:
        body = {
            "packet_id": f"{evaluation['evaluation_id']}:P55:Taylor_order_1",
            "evaluation_id": evaluation["evaluation_id"],
            "evaluation_content_sha256": evaluation["content_sha256"],
            "Taylor_order": 1,
            "factorial_normalization": "1/1!=1",
            "required_shape": [55, 55],
            "status": "MISSING_REQUIRED_SERIALIZATION",
        }
        packets.append({**body, "content_sha256": _content_hash(body)})
    if len(packets) != TARGET_PACKETS:
        raise OrderOneSerializationFrontierError("order-one packet census changed")
    return packets


def build_campaign(project_root: Path, config_path: Path) -> dict[str, Any]:
    root = project_root.resolve()
    config = _load_json(config_path)
    _validate_config(config)
    upstreams = {name: _load_bound(root, binding) for name, binding in config["upstreams"].items()}
    _validate_raw_binding(root, config["recurrence_source"])
    predecessor = upstreams["manifest_predecessor"]
    _audit_predecessor(predecessor)
    evaluations, p55_contract = _audit_P55_serialization(upstreams["P55_order_zero_registration"])
    reference_audit = _audit_reference_only(upstreams["full_reference_Sylvester"])
    required_packets = _required_order_one_packets(evaluations)
    manifest = deepcopy(predecessor["required_symbolic_input_manifest"])
    if (
        sum(row["registered_packets"] for row in manifest) != REGISTERED_PACKETS
        or sum(row["required_packets"] for row in manifest) != REQUIRED_PACKETS
    ):
        raise OrderOneSerializationFrontierError("frontier manifest arithmetic changed")
    claims = {key: value for key, value in predecessor["claims"].items() if value is False}
    claims.update(
        {
            "order_one_serialization_frontier_exactly_measured": True,
            "P55_Taylor_order_one_registered": False,
            "K55_Taylor_order_one_registered": False,
            "TC2_Taylor_order_one_registered": False,
            "coordinate_free_deltaK_order_zero_registered": False,
            "cold_full_symbol_build_used": False,
        }
    )
    body = {
        "schema_version": SCHEMA,
        "status": STATUS,
        "decision": "BLOCK_SERIALIZATION",
        "errors": [],
        "config_sha256": config["content_sha256"],
        "upstream_bindings": {
            name: {**binding, "verified": True} for name, binding in config["upstreams"].items()
        },
        "recurrence_source_binding": {**config["recurrence_source"], "verified": True},
        "first_blocker": (
            "serialize_all_15_exact_coordinate_free_P55_Taylor_order_one_packets_or_"
            "the_equivalent_M0_inverse_M1_E1_recurrence_inputs"
        ),
        "required_P55_Taylor_order_one_packets": required_packets,
        "minimal_serialization_contract": {
            "schema_version": ("sigma-coordinate-free-P55-Taylor-order-one-input-contract-1.0"),
            "required_packets": TARGET_PACKETS,
            "registered_packets": 0,
            "required_shape_each": [55, 55],
            "direct_packet_required_fields": [
                "evaluation_content_sha256",
                "Taylor_order",
                "factorial_normalization_1_over_1_factorial",
                "exact_sparse_polynomial_entries_in_n1_n2_n3",
                "ordered_state_indices",
                "matrix_content_sha256",
                "entrywise_recurrence_residual_zero",
            ],
            "equivalent_recurrence_inputs": {
                "per_evaluation": ["exact_M0_inverse", "exact_M1(n)", "exact_E1(n)"],
                "identity": "P1=M0^{-1}*(E1-M1*P0)",
                "required_residual": "M0*P1+M1*P0-E1=0 entrywise",
            },
            "inherited_four_order_contract_content": p55_contract,
        },
        "dependency_frontier": [
            {
                "family": "polarized_P55_Taylor_packets/order_1",
                "required_packets": 15,
                "admissible_serialized_packets": 0,
                "blocked_dependents": [
                    "polarized_K55_Taylor_packets/order_1",
                    "polarized_TC2_Taylor_packets/order_1",
                ],
            },
            {
                "family": "lower_Sylvester_correction_recurrence/order_0",
                "required_packets": 15,
                "admissible_serialized_packets": 0,
                "inadmissible_reference_packets": 1,
                "reason": "the only serialized deltaK is direction-e1 reference data",
            },
        ],
        "reference_only_deltaK_audit": reference_audit,
        "required_symbolic_input_manifest": manifest,
        "remaining_missing_inputs": deepcopy(predecessor["remaining_missing_inputs"]),
        "bounded_emitter_checkpoint": {
            "complete": False,
            "required_output_rows": REQUIRED_ROWS,
            "emitted_output_rows": 0,
            "emitted_rhs_rows": 0,
            "emitted_sparse_entries": 0,
        },
        "phase_two": {
            "decision": "BLOCK",
            "admitted": False,
            "attempted": False,
            "blocker": "240 required symbolic input packets remain unregistered",
        },
        "counts": {
            "upstream_seals_verified": 3,
            "required_symbolic_input_packets": REQUIRED_PACKETS,
            "predecessor_registered_symbolic_input_packets": REGISTERED_PACKETS,
            "new_symbolic_input_packets_registered": 0,
            "registered_symbolic_input_packets": REGISTERED_PACKETS,
            "missing_symbolic_input_packets": MISSING_PACKETS,
            "target_P55_Taylor_order_one_packets": TARGET_PACKETS,
            "serialized_P55_Taylor_order_one_packets": 0,
            "serialized_M0_inverse_packets": 0,
            "serialized_M1_packets": 0,
            "serialized_E1_packets": 0,
            "reference_deltaK_packets_seen": 1,
            "admissible_coordinate_free_deltaK_order_zero_packets": 0,
            "variable_coefficient_Sylvester_proofs": 0,
            "full_symbol_build_calls": 0,
            "required_output_rows": REQUIRED_ROWS,
            "emitted_output_rows": 0,
            "phase_two_solve_attempts": 0,
        },
        "claims": claims,
        "negative_controls": {
            "infer_P55_order_one_as_zero": {"rejected": True},
            "differentiate_spatial_n_instead_of_state_jets": {"rejected": True},
            "promote_direction_e1_deltaK_to_coordinate_free_deltaK": {"rejected": True},
            "count_required_schema_as_serialized_packet": {"rejected": True},
            "emit_rows_with_240_missing_packets": {"rejected": True},
            "promote_serialization_audit_to_D4_or_H7": {"rejected": True},
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
            "Exact readiness obstruction for the 15 coordinate-free P55 "
            "Taylor-order-one packets and their K55/TC2 dependents. No coefficients "
            "are inferred from the flat order-zero pencil or the single e1 "
            "Sylvester reference. The manifest stays "
            "at 64/304 and no recurrence row, D4 theorem, TC2 closure, H7 closure, PDE "
            "theorem, lifespan, physical no-go, or candidate rejection follows."
        ),
    }
    return {**body, "content_sha256": _content_hash(body)}


def validate_campaign(document: dict[str, Any], project_root: Path) -> None:
    expected = build_campaign(project_root.resolve(), project_root.resolve() / CONFIG_PATH)
    if document != expected or not _hash_matches(document):
        raise OrderOneSerializationFrontierError("campaign replay mismatch")


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
