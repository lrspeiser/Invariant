from __future__ import annotations

import argparse
import hashlib
import json
from functools import cache
from pathlib import Path
from typing import Any

from .quartic_tc2_d4_coordinate_free_symbolic_recurrence_emitter_readiness import (
    validate_campaign as validate_readiness_campaign,
)
from .quartic_tc2_d4_p55_checkpointable_materializer import validate_final_result

SCHEMA = "sigma-quartic-tc2-d4-coordinate-free-symbolic-recurrence-emitter-p55-registration-1.0"
CONFIG_SCHEMA = (
    "sigma-quartic-tc2-d4-coordinate-free-symbolic-recurrence-emitter-p55-registration-config-1.0"
)
STATUS = "block_coordinate_free_D4_recurrence_emitter_missing_298_symbolic_packets"
CONFIG_PATH = (
    "configs/backgrounds/"
    "quartic_tc2_d4_coordinate_free_symbolic_recurrence_emitter_p55_registration.json"
)
SOURCE_PATH = (
    "src/sigma_theory_compiler/"
    "quartic_tc2_d4_coordinate_free_symbolic_recurrence_emitter_p55_registration.py"
)
TEST_PATH = (
    "tests/test_quartic_tc2_d4_coordinate_free_symbolic_recurrence_emitter_p55_registration.py"
)
READINESS_SHA256 = "893b5b5daacc749a593d4eddd709e0e61f63f9f7954f46f02c0fd6970f48badb"
P55_RESULT_SHA256 = "25ced42ac83bd592332f3d3a8bc97eeaac9c55b32ba4bb2be3e3da1fa6b503b7"
P55_RESULT_FILE_SHA256 = "9e79245d45c248d5abf1b2fd11b12bb7ca0c857286d31f8bf46b7ec7c43490b2"
P55_CONFIG_SHA256 = "4085aa176c41cec5f912bc1db0369683e39987c82a7d5440daf0892cb9b9837a"
REQUIRED_PACKETS = 304
REGISTERED_PACKETS = 6
MISSING_PACKETS = 298
REQUIRED_ROWS = 117_180

FALSE_CLAIMS = {
    "B7_closed",
    "CK1_closed",
    "CK3_closed",
    "TC2_closed",
    "complete_D2F_tensor_registered",
    "complete_coordinate_free_coefficient_map_emitted",
    "complete_coordinate_free_rhs_emitted",
    "full_direction_sphere_D4_compatibility_proved",
    "full_high_atom_identity_proved",
    "full_tube_Sylvester_identity_proved",
    "global_H7_closed",
    "lifespan_proved",
    "matrix_projectors_evaluated",
    "nonlinear_PDE_closure_proved",
    "phase_two_exact_solve_admitted",
    "theory_candidate_rejected",
    "variable_coefficient_constraint_calculus_proved",
}


class P55EmitterRegistrationError(ValueError):
    """Raised when the isolated P55 emitter registration fails closed."""


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
        raise P55EmitterRegistrationError(f"expected JSON object: {path}")
    return value


def _resolve_under(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if root != path and root not in path.parents:
        raise P55EmitterRegistrationError("bound path escaped project root")
    return path


def _validate_config(config: dict[str, Any]) -> None:
    target = config.get("target", {})
    caps = config.get("resource_caps", {})
    readiness = config.get("upstreams", {}).get("recurrence_emitter_readiness", {})
    p55 = config.get("upstreams", {}).get("checkpointable_P55_result", {})
    if (
        config.get("schema_version") != CONFIG_SCHEMA
        or config.get("policy") != "register_only_native_validated_exact_packets_fail_closed"
        or not _hash_matches(config)
        or target
        != {
            "required_symbolic_input_packets": REQUIRED_PACKETS,
            "previously_registered_symbolic_input_packets": 3,
            "new_P55_packets": 3,
            "expected_registered_symbolic_input_packets": REGISTERED_PACKETS,
            "expected_missing_symbolic_input_packets": MISSING_PACKETS,
            "required_output_rows": REQUIRED_ROWS,
        }
        or caps
        != {
            "maximum_manifest_records": 8,
            "maximum_registered_P55_sparse_entries": 144,
            "maximum_output_rows_emitted": 0,
        }
        or readiness.get("content_sha256") != READINESS_SHA256
        or p55.get("file_sha256") != P55_RESULT_FILE_SHA256
        or p55.get("content_sha256") != P55_RESULT_SHA256
        or p55.get("config_content_sha256") != P55_CONFIG_SHA256
    ):
        raise P55EmitterRegistrationError("invalid P55 emitter registration config")


def _validate_bound_p55_files(
    root: Path, p55_result: dict[str, Any], binding: dict[str, Any]
) -> dict[str, Any]:
    expected_files = {
        binding["path"]: binding["file_sha256"],
        **{
            record["path"]: record["file_sha256"]
            for record in p55_result.get("source_bindings", {}).values()
        },
    }
    verified: list[dict[str, Any]] = []
    for relative, expected_sha256 in sorted(expected_files.items()):
        current_path = _resolve_under(root, relative)
        if _file_sha256(current_path) != expected_sha256:
            raise P55EmitterRegistrationError(f"bound P55 file mismatch: {relative}")
        verified.append(
            {
                "path": relative,
                "file_sha256": expected_sha256,
                "current_file_matches_immutable_binding": True,
            }
        )
    return {
        "files_verified": len(verified),
        "files": verified,
        "history_independent": True,
    }


def _registered_manifest(
    readiness: dict[str, Any], p55_result: dict[str, Any]
) -> list[dict[str, Any]]:
    manifest = json.loads(json.dumps(readiness["required_symbolic_input_manifest"]))
    if len(manifest) != 8:
        raise P55EmitterRegistrationError("readiness manifest record count mismatch")
    records = {record.get("input_id"): record for record in manifest}
    physical = records.get("physical_spatial_pencil_coefficients")
    if (
        physical is None
        or physical.get("required_packets") != 3
        or physical.get("registered_packets") != 0
        or physical.get("status") != "missing"
    ):
        raise P55EmitterRegistrationError("prior physical P55 manifest boundary mismatch")
    physical.pop("evidence_boundary", None)
    physical.update(
        {
            "registered_packets": 3,
            "status": "registered_exact_flat_reference_P55_packets",
            "registered_shape_each": [55, 55],
            "registered_nonzero_entries": 144,
            "matrix_packet_content_sha256": [
                packet["content_sha256"] for packet in p55_result["matrix_packets"]
            ],
            "registration_receipt_content_sha256": p55_result["content_sha256"],
            "exact_linearity_entries_certified": 3025,
            "exact_sphere_minimal_polynomial_entries_reduced": 3025,
            "exact_sphere_minimal_polynomial_nonzero_remainders": 0,
        }
    )
    if sum(record["required_packets"] for record in manifest) != REQUIRED_PACKETS:
        raise P55EmitterRegistrationError("required manifest packet total mismatch")
    if sum(record["registered_packets"] for record in manifest) != REGISTERED_PACKETS:
        raise P55EmitterRegistrationError("registered manifest packet total mismatch")
    return manifest


@cache
def build_campaign(project_root: Path, config_path: Path) -> dict[str, Any]:
    root = project_root.resolve()
    config = _load_json(config_path)
    _validate_config(config)
    readiness_binding = config["upstreams"]["recurrence_emitter_readiness"]
    p55_binding = config["upstreams"]["checkpointable_P55_result"]
    readiness_path = _resolve_under(root, readiness_binding["path"])
    p55_path = _resolve_under(root, p55_binding["path"])
    p55_config_path = _resolve_under(root, p55_binding["config_path"])
    readiness = _load_json(readiness_path)
    p55_result = _load_json(p55_path)
    if (
        readiness.get("content_sha256") != READINESS_SHA256
        or not _hash_matches(readiness)
        or readiness.get("status")
        != "block_coordinate_free_D4_recurrence_emitter_missing_symbolic_P_and_Taylor_packets"
    ):
        raise P55EmitterRegistrationError("recurrence readiness seal/status mismatch")
    validate_readiness_campaign(readiness, root)
    if _file_sha256(p55_path) != P55_RESULT_FILE_SHA256:
        raise P55EmitterRegistrationError("P55 result file seal mismatch")
    try:
        validate_final_result(p55_result, root, p55_config_path)
    except ValueError as error:
        raise P55EmitterRegistrationError(f"native P55 validation failed: {error}") from error
    bound_input_receipt = _validate_bound_p55_files(root, p55_result, p55_binding)
    manifest = _registered_manifest(readiness, p55_result)
    missing = [
        {
            "input_id": record["input_id"],
            "required_packets": record["required_packets"],
            "registered_packets": record["registered_packets"],
            "missing_packets": record["required_packets"] - record["registered_packets"],
            "status": record["status"],
        }
        for record in manifest
        if record["registered_packets"] < record["required_packets"]
    ]
    if sum(record["missing_packets"] for record in missing) != MISSING_PACKETS:
        raise P55EmitterRegistrationError("missing manifest packet total mismatch")
    packet_summaries = [
        {
            "name": packet["name"],
            "shape": packet["shape"],
            "nonzero_count": packet["nonzero_count"],
            "content_sha256": packet["content_sha256"],
        }
        for packet in p55_result["matrix_packets"]
    ]
    claims = {claim: False for claim in sorted(FALSE_CLAIMS)}
    claims.update(
        {
            "checkpointable_P55_result_native_validated": True,
            "checkpointable_P55_immutable_file_bindings_verified": True,
            "exact_three_axis_flat_reference_P55_packets_registered": True,
            "manifest_recomputed_from_registered_packets": True,
            "all_seven_scalar_Lagrange_projector_recipes_remain_registered": True,
            "exact_gradient_lift_pencil_remains_registered": True,
        }
    )
    body = {
        "schema_version": SCHEMA,
        "status": STATUS,
        "errors": [],
        "config_sha256": config["content_sha256"],
        "upstream_bindings": {
            "recurrence_emitter_readiness": {
                "path": readiness_binding["path"],
                "content_sha256": readiness["content_sha256"],
                "verified": True,
            },
            "checkpointable_P55_result": {
                "path": p55_binding["path"],
                "file_sha256": _file_sha256(p55_path),
                "content_sha256": p55_result["content_sha256"],
                "native_exact_validation": True,
            },
        },
        "bound_input_receipt": bound_input_receipt,
        "registered_P55_packets": packet_summaries,
        "required_symbolic_input_manifest": manifest,
        "remaining_missing_inputs": missing,
        "bounded_emitter_checkpoint": {
            "complete": False,
            "first_missing_input": "polarized_P55_Taylor_packets",
            "row_emission_cursor": {
                "next_cokernel_coordinate": 0,
                "next_odd_sphere_mode": 0,
                "next_flat_row_offset": 0,
            },
            "emitted_output_rows": 0,
            "emitted_rhs_rows": 0,
            "emitted_sparse_entries": 0,
            "required_output_rows": REQUIRED_ROWS,
        },
        "phase_two": {
            "decision": "BLOCK",
            "admitted": False,
            "attempted": False,
            "blocker": "298 required symbolic input packets remain unregistered",
        },
        "counts": {
            "upstream_content_seals_verified": 2,
            "immutable_files_verified": bound_input_receipt["files_verified"],
            "required_symbolic_input_packets": REQUIRED_PACKETS,
            "registered_symbolic_input_packets": REGISTERED_PACKETS,
            "missing_symbolic_input_packets": MISSING_PACKETS,
            "new_P55_matrix_packets_registered": 3,
            "new_P55_sparse_entries_registered": 144,
            "gradient_lift_matrix_packets_registered": 3,
            "gradient_lift_nonzero_entries_registered": 33,
            "Lagrange_projector_recipes_registered": 7,
            "matrix_projectors_evaluated": 0,
            "required_output_rows": REQUIRED_ROWS,
            "emitted_output_rows": 0,
            "emitted_rhs_rows": 0,
            "phase_two_solve_attempts": 0,
        },
        "claims": claims,
        "negative_controls": {
            "trust_P55_result_counts_without_native_replay": {"rejected": True},
            "consume_hash_mismatched_P55_file": {"rejected": True},
            "count_expected_nonzeros_as_coefficient_data": {"rejected": True},
            "promote_P55_registration_to_matrix_projector_evaluation": {"rejected": True},
            "emit_recurrence_rows_before_remaining_packets_exist": {"rejected": True},
            "promote_flat_reference_P55_to_full_D4_or_H7": {"rejected": True},
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
            "Registers only the three hash-bound, native-validated flat-reference P55 axis "
            "packets in the coordinate-free recurrence input manifest. The remaining 298 "
            "Taylor, lower-recurrence, candidate-normalization, and sphere-reducer packets "
            "remain absent; no recurrence row, D4 theorem, H7 closure, PDE theorem, lifespan, "
            "or rejection follows."
        ),
    }
    return {**body, "content_sha256": _content_hash(body)}


def validate_campaign(document: dict[str, Any], project_root: Path) -> None:
    expected = build_campaign(project_root.resolve(), project_root.resolve() / CONFIG_PATH)
    if document != expected or not _hash_matches(document):
        raise P55EmitterRegistrationError("campaign replay mismatch")


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
