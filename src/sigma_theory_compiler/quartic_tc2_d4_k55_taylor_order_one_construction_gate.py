from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

SCHEMA = "sigma-quartic-tc2-d4-k55-taylor-order-one-construction-gate-1.0"
CONFIG_SCHEMA = f"{SCHEMA.removesuffix('-1.0')}-config-1.0"
STATUS = "block_K55_Taylor_order_one_missing_physical_H_star_order_one_packets"
CONFIG_PATH = "configs/backgrounds/quartic_tc2_d4_k55_taylor_order_one_construction_gate.json"
SOURCE_PATH = "src/sigma_theory_compiler/quartic_tc2_d4_k55_taylor_order_one_construction_gate.py"
TEST_PATH = "tests/test_quartic_tc2_d4_k55_taylor_order_one_construction_gate.py"
OUTPUT_PATH = (
    "runs/physics-language/quartic-tc2-d4-k55-taylor-order-one-construction-gate/campaign.json"
)
REQUIRED_PACKETS = 304
REGISTERED_PACKETS = 79
MISSING_PACKETS = 225
TARGET_NEW_PACKETS = 15
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


class K55TaylorOrderOneConstructionError(ValueError):
    """Raised when the K55 order-one construction boundary changes."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "ascii"
    )


def _content_hash(value: dict[str, Any]) -> str:
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    return hashlib.sha256(_canonical_bytes(body)).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hash_matches(value: dict[str, Any]) -> bool:
    return value.get("content_sha256") == _content_hash(value)


def _resolve_under(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if root != path and root not in path.parents:
        raise K55TaylorOrderOneConstructionError("bound path escaped project root")
    return path


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise K55TaylorOrderOneConstructionError(f"expected JSON object: {path}")
    return value


def _load_bound(root: Path, binding: dict[str, str]) -> dict[str, Any]:
    path = _resolve_under(root, binding["path"])
    value = _load_json(path)
    if (
        _file_sha256(path) != binding["file_sha256"]
        or value.get("content_sha256") != binding["content_sha256"]
        or not _hash_matches(value)
    ):
        raise K55TaylorOrderOneConstructionError(f"upstream mismatch: {binding['path']}")
    return value


def _validate_config(config: dict[str, Any]) -> None:
    if (
        config.get("schema_version") != CONFIG_SCHEMA
        or config.get("policy") != "derive_K55_order_one_only_from_exact_registered_inputs_or_block"
        or not _hash_matches(config)
        or set(config.get("upstreams", {})) != {"manifest_predecessor", "K55_order_zero"}
        or config.get("target")
        != {
            "required_symbolic_input_packets": REQUIRED_PACKETS,
            "predecessor_registered_packets": REGISTERED_PACKETS,
            "target_new_K55_Taylor_order_one_packets": TARGET_NEW_PACKETS,
            "registered_packets_if_blocked": REGISTERED_PACKETS,
            "missing_packets_if_blocked": MISSING_PACKETS,
            "required_output_rows": REQUIRED_ROWS,
        }
        or config.get("resource_caps", {}).get("maximum_full_symbol_build_calls") != 0
        or config.get("resource_caps", {}).get("maximum_output_rows_emitted") != 0
    ):
        raise K55TaylorOrderOneConstructionError("invalid K55 order-one gate config")


def _validate_p55_packets(predecessor: dict[str, Any]) -> list[dict[str, Any]]:
    packets = predecessor.get("registered_P55_Taylor_order_one_packets", [])
    if (
        predecessor.get("status")
        != "block_coordinate_free_D4_recurrence_emitter_missing_225_symbolic_packets"
        or predecessor.get("counts", {}).get("registered_symbolic_input_packets")
        != REGISTERED_PACKETS
        or len(packets) != TARGET_NEW_PACKETS
        or [packet.get("evaluation_id") for packet in packets] != EXPECTED_EVALUATIONS
    ):
        raise K55TaylorOrderOneConstructionError("P55 predecessor boundary changed")
    for packet in packets:
        matrix = packet.get("P55_Taylor_order_one_matrix", {})
        entries = matrix.get("entries", [])
        if (
            not _hash_matches(packet)
            or not _hash_matches(matrix)
            or matrix.get("shape") != [55, 55]
            or matrix.get("Taylor_order") != 1
            or any(entry.get("row", -1) < 33 for entry in entries)
            or any(
                not set(entry.get("linear_coefficients", {})) <= {"n1", "n2", "n3"}
                for entry in entries
            )
        ):
            raise K55TaylorOrderOneConstructionError("invalid exact P55 order-one packet")
    return packets


def _validate_k0(receipt: dict[str, Any], predecessor: dict[str, Any]) -> dict[str, Any]:
    exact = receipt.get("exact_K0_construction", {})
    k0 = exact.get("K0", {})
    manifest = {
        row.get("input_id"): row for row in predecessor.get("required_symbolic_input_manifest", [])
    }
    k55_family = manifest.get("polarized_K55_Taylor_packets", {})
    if (
        receipt.get("claims", {}).get("exact_flat_K0_constructed") is not True
        or not _hash_matches(k0)
        or k0.get("shape") != [55, 55]
        or exact.get("K0_symmetry_residual_nonzero_entries") != 0
        or exact.get("K0_P1_symmetrizer_residual_nonzero_entries") != 0
        or k55_family.get("K0_content_sha256") != k0.get("content_sha256")
        or k55_family.get("registered_Taylor_orders") != [0]
        or k55_family.get("registered_packets") != 15
    ):
        raise K55TaylorOrderOneConstructionError("K0 construction boundary changed")
    return k0


def _block_census(packets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    census = []
    for packet in packets:
        entries = packet["P55_Taylor_order_one_matrix"]["entries"]
        lift = [entry for entry in entries if entry["column"] < 33]
        companion = [entry for entry in entries if entry["column"] >= 33]
        census.append(
            {
                "evaluation_id": packet["evaluation_id"],
                "evaluation_content_sha256": packet["evaluation_content_sha256"],
                "P55_order_one_content_sha256": packet["content_sha256"],
                "determined_L1_sparse_entries": len(lift),
                "determined_M22_order_one_sparse_entries": len(companion),
                "missing_H_star_plus_order_one_packets": 1,
                "K55_order_one_admissible": False,
            }
        )
    return census


def build_campaign(project_root: Path, config_path: Path) -> dict[str, Any]:
    root = project_root.resolve()
    config = _load_json(config_path)
    _validate_config(config)
    predecessor = _load_bound(root, config["upstreams"]["manifest_predecessor"])
    k0_receipt = _load_bound(root, config["upstreams"]["K55_order_zero"])
    packets = _validate_p55_packets(predecessor)
    k0 = _validate_k0(k0_receipt, predecessor)
    census = _block_census(packets)
    lift_entries = sum(row["determined_L1_sparse_entries"] for row in census)
    companion_entries = sum(row["determined_M22_order_one_sparse_entries"] for row in census)
    if lift_entries != 248 or companion_entries != 192:
        raise K55TaylorOrderOneConstructionError("P55 block census changed")
    manifest = deepcopy(predecessor["required_symbolic_input_manifest"])
    family = {row["input_id"]: row for row in manifest}["polarized_K55_Taylor_packets"]
    if (
        family.get("registered_packets") != 15
        or family.get("registered_Taylor_orders") != [0]
        or family.get("missing_Taylor_orders") != [1, 2, 3, 4]
    ):
        raise K55TaylorOrderOneConstructionError("K55 manifest boundary changed")
    missing_contract = {
        "schema_version": ("sigma-coordinate-free-H-star-plus-Taylor-order-one-contract-1.0"),
        "required_packets": 15,
        "registered_packets": 0,
        "shape_each": [22, 22],
        "Taylor_order": 1,
        "factorial_normalization": "1/1!=1",
        "variables": ["n1", "n2", "n3"],
        "required_fields": [
            "evaluation_id",
            "evaluation_content_sha256",
            "exact_sparse_polynomial_entries_in_n1_n2_n3",
            "H_star_plus_order_one_matrix_content_sha256",
            "H_star_minus_order_one_equals_negative_H_star_plus_order_one",
            "physical_action_A_star_B_star_derivative_provenance",
        ],
        "reason_P55_is_insufficient": (
            "P55=M55_inverse_E55 fixes L1 and M22_1 but does not recover the "
            "independent derivative of the physical action inner product H_star."
        ),
    }
    claims = {key: value for key, value in predecessor["claims"].items() if value is False}
    claims.update(
        {
            "P55_order_one_block_decomposition_exactly_audited": True,
            "K55_Taylor_order_one_registered": False,
            "physical_H_star_order_one_registered": False,
            "manifest_advanced_beyond_79": False,
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
        "determined_order_one_block_census": census,
        "exact_K55_derivative_boundary": {
            "K0_content_sha256": k0["content_sha256"],
            "determined_from_P55_order_one": [
                "L1(n)",
                "M22_1(n)",
                "Riesz_projector_derivatives_from_M22_1_given_fixed_spectrum",
                "M22_inverse_order_one_from_M22_1",
            ],
            "first_undetermined_term": "Pi_physical_0^T*H_star_plus_1*Pi_physical_0",
            "K22_derivative_identity": (
                "K22_1=sum_lambda(Pi_lambda_1^T H_lambda_0 Pi_lambda_0+"
                "Pi_lambda_0^T H_lambda_1 Pi_lambda_0+"
                "Pi_lambda_0^T H_lambda_0 Pi_lambda_1)"
            ),
            "cross_block_derivative_identity": (
                "F1=L1^T K22_0 M22_0^-1+L0^T K22_1 M22_0^-1-L0^T K22_0 M22_0^-1 M22_1 M22_0^-1"
            ),
            "missing_input_contract": missing_contract,
        },
        "exact_nonuniqueness_witness": {
            "P0": [["1", "0"], ["0", "-1"]],
            "P1": [["0", "0"], ["0", "0"]],
            "K0": [["1", "0"], ["0", "1"]],
            "K1_candidate_A": [["0", "0"], ["0", "0"]],
            "K1_candidate_B": [["1", "0"], ["0", "1"]],
            "candidate_A_first_order_symmetrizer_residual_nonzero_entries": 0,
            "candidate_B_first_order_symmetrizer_residual_nonzero_entries": 0,
            "candidates_distinct": True,
            "conclusion": (
                "P0, P1, and K0 do not select a unique K1; the physical H_star_1 "
                "construction data or an equivalent canonical symmetrizer derivative "
                "is required."
            ),
        },
        "required_symbolic_input_manifest": manifest,
        "bounded_emitter_checkpoint": {
            "complete": False,
            "first_missing_input": ("15 coordinate-free H_star_plus Taylor-order-one packets"),
            "required_output_rows": REQUIRED_ROWS,
            "emitted_output_rows": 0,
            "emitted_rhs_rows": 0,
            "emitted_sparse_entries": 0,
        },
        "phase_two": {
            "decision": "BLOCK",
            "admitted": False,
            "attempted": False,
            "blocker": ("225 required symbolic packets remain; K55 order one lacks H_star_plus_1"),
        },
        "counts": {
            "upstream_seals_verified": 2,
            "P55_Taylor_order_one_packets_validated": 15,
            "determined_L1_sparse_entries": lift_entries,
            "determined_M22_order_one_sparse_entries": companion_entries,
            "required_H_star_plus_order_one_packets": 15,
            "registered_H_star_plus_order_one_packets": 0,
            "new_K55_Taylor_order_one_packets_registered": 0,
            "required_symbolic_input_packets": REQUIRED_PACKETS,
            "predecessor_registered_symbolic_input_packets": REGISTERED_PACKETS,
            "registered_symbolic_input_packets": REGISTERED_PACKETS,
            "missing_symbolic_input_packets": MISSING_PACKETS,
            "full_symbol_build_calls": 0,
            "required_output_rows": REQUIRED_ROWS,
            "emitted_output_rows": 0,
            "phase_two_solve_attempts": 0,
        },
        "claims": claims,
        "negative_controls": {
            "infer_K55_order_one_from_symmetrizer_identity_alone": {"rejected": True},
            "set_H_star_plus_order_one_to_zero_without_source": {"rejected": True},
            "choose_one_of_two_valid_K1_witnesses_as_canonical": {"rejected": True},
            "count_required_H_star_schema_as_registered_packets": {"rejected": True},
            "emit_rows_with_225_missing_packets": {"rejected": True},
            "promote_block_audit_to_D4_H7_or_PDE": {"rejected": True},
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
            "Validates and block-decomposes all 15 exact P55 Taylor-order-one packets. "
            "It measures the exact K55 construction frontier and seals the missing "
            "15-packet physical H_star_plus derivative contract. No K55 packet, "
            "recurrence row, D4 or H7 theorem, nonlinear PDE closure, lifespan, or "
            "candidate rejection follows."
        ),
    }
    return {**body, "content_sha256": _content_hash(body)}


def validate_campaign(document: dict[str, Any], project_root: Path) -> None:
    expected = build_campaign(project_root.resolve(), project_root.resolve() / CONFIG_PATH)
    if document != expected or not _hash_matches(document):
        raise K55TaylorOrderOneConstructionError("campaign replay mismatch")


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
