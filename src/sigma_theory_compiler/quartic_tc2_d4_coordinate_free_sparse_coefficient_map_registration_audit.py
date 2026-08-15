from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

SCHEMA = "sigma-quartic-tc2-d4-coordinate-free-sparse-coefficient-map-registration-audit-1.0"
CONFIG_SCHEMA = (
    "sigma-quartic-tc2-d4-coordinate-free-sparse-coefficient-map-registration-config-1.0"
)
CHECKPOINT_SCHEMA = "sigma-quartic-tc2-d4-coordinate-free-sparse-coefficient-map-checkpoint-1.0"
STATUS = "block_coordinate_free_D4_sparse_coefficient_map_not_symbolically_emitted"
CONFIG_PATH = (
    "configs/backgrounds/"
    "quartic_tc2_d4_coordinate_free_sparse_coefficient_map_registration_audit.json"
)
SOURCE_PATH = (
    "src/sigma_theory_compiler/"
    "quartic_tc2_d4_coordinate_free_sparse_coefficient_map_registration_audit.py"
)
TEST_PATH = "tests/test_quartic_tc2_d4_coordinate_free_sparse_coefficient_map_registration_audit.py"
ARTIFACT_PATH = (
    "runs/physics-language/"
    "quartic-tc2-d4-coordinate-free-sparse-coefficient-map-registration-audit/"
    "campaign.json"
)

UNKNOWN_COLUMNS = 33_880
COKERNEL_COORDINATES = 558
ODD_SPHERE_MODES = 210
REQUIRED_ROWS = 117_180
EXPECTED_CAPS = {
    "maximum_registered_rows": REQUIRED_ROWS,
    "maximum_registered_sparse_entries": 2_000_000,
    "maximum_checkpoint_bytes": 16_777_216,
    "maximum_point_evidence_packets": 16,
}
EXPECTED_UPSTREAMS = {
    "full_sphere_readiness_gate": {
        "content_sha256": ("07ba08c8057823b03733d39bf8c2d1c04ce3d506d4dd4103c18d195943a1724b"),
        "status": ("block_full_sphere_degree_seven_exact_solve_coefficient_map_not_registered"),
    },
    "revised_thirteen_frame_predecessor": {
        "content_sha256": ("55a68d34961739728a6ae111ea1c76f83f51524614712d7e960f4f37a1139267"),
        "status": "pass_exact_second_height_two_point_and_bounded_classification",
    },
    "canonical_D4_obstruction_certificate": {
        "content_sha256": ("bef3246a17942c74e8f3cdb09bc14a36c6bdc44d030a9a70ce833c30ec04bc65"),
        "status": "pass_exact_canonical_d4_obstruction_cokernel_classification",
    },
    "rational_chart_counterexample_gate": {
        "content_sha256": ("48b8ecfe63336071721baeb90a41f379ac1b4235629b380abdf0124e7008152c"),
        "status": (
            "pass_exact_rational_chart_counterexample_disproves_current_full_sphere_"
            "D4_compatibility"
        ),
    },
}

FALSE_CLAIMS = {
    "B7_closed",
    "CK1_closed",
    "CK3_closed",
    "TC2_closed",
    "complete_D2F_tensor_registered",
    "complete_coordinate_free_coefficient_map_registered",
    "complete_exact_rhs_registered",
    "covariant_action_origin_proved",
    "full_direction_sphere_D4_compatibility_proved",
    "full_high_atom_identity_proved",
    "full_tube_Sylvester_identity_proved",
    "global_H7_closed",
    "lifespan_proved",
    "local_differential_operator_origin_proved",
    "nonlinear_PDE_closure_proved",
    "phase_two_exact_solve_admitted",
    "point_samples_inferred_as_polynomial_coefficients",
    "theory_candidate_rejected",
    "variable_coefficient_constraint_calculus_proved",
}

NEGATIVE_CONTROLS = {
    "count_local_compatibility_as_global_row_registration": {"rejected": True},
    "count_pointwise_obstruction_as_sphere_mode_coefficient": {"rejected": True},
    "infer_210_modes_from_13_local_certificates": {"rejected": True},
    "infer_missing_rows_as_zero": {"rejected": True},
    "promote_partial_checkpoint_to_phase_two": {"rejected": True},
    "register_float_or_sampled_coefficients": {"rejected": True},
    "skip_exact_rhs_registration": {"rejected": True},
    "weaken_complete_map_admission_count": {"rejected": True},
}


class CoordinateFreeCoefficientMapAuditError(ValueError):
    """Raised when the coefficient-map audit or one of its seals is inconsistent."""


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
        raise CoordinateFreeCoefficientMapAuditError(f"expected JSON object: {path}")
    return value


def _resolve_under(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if root != path and root not in path.parents:
        raise CoordinateFreeCoefficientMapAuditError("bound path escaped project root")
    return path


def _validate_config(config: dict[str, Any]) -> None:
    if (
        set(config)
        != {
            "schema_version",
            "global_claim_policy",
            "registration_policy",
            "target_map",
            "resource_caps",
            "phase_two_admission",
            "upstream_bindings",
            "content_sha256",
        }
        or config.get("schema_version") != CONFIG_SCHEMA
        or config.get("global_claim_policy") != "fail_closed"
        or config.get("registration_policy")
        != "exact_symbolic_coefficients_only_no_point_sample_inference"
        or not _hash_matches(config)
        or config.get("target_map")
        != {
            "unknown_columns": UNKNOWN_COLUMNS,
            "equal_eigenspace_cokernel_coordinates": COKERNEL_COORDINATES,
            "odd_sphere_modes": ODD_SPHERE_MODES,
            "required_coefficient_rows": REQUIRED_ROWS,
            "required_rhs_rows": REQUIRED_ROWS,
            "sphere_polynomial_degree_ceiling": 19,
            "recurrence_orders": [1, 2, 3, 4],
        }
        or config.get("resource_caps") != EXPECTED_CAPS
        or config.get("phase_two_admission")
        != {
            "require_complete_coefficient_rows": REQUIRED_ROWS,
            "require_complete_rhs_rows": REQUIRED_ROWS,
            "require_missing_rows": 0,
            "require_exact_row_replay": True,
        }
        or set(config.get("upstream_bindings", {})) != set(EXPECTED_UPSTREAMS)
    ):
        raise CoordinateFreeCoefficientMapAuditError("invalid coefficient-map audit config")


def _validate_upstream(name: str, value: dict[str, Any]) -> dict[str, Any]:
    expected = EXPECTED_UPSTREAMS[name]
    if (
        not _hash_matches(value)
        or value.get("content_sha256") != expected["content_sha256"]
        or value.get("status") != expected["status"]
        or value.get("errors") != []
    ):
        raise CoordinateFreeCoefficientMapAuditError(f"upstream seal mismatch: {name}")
    if name == "full_sphere_readiness_gate":
        counts = value.get("counts", {})
        topology = value.get("phase_one", {}).get("symbolic_sparse_topology", {})
        phase_two = value.get("phase_two", {})
        if (
            counts.get("unknown_columns") != UNKNOWN_COLUMNS
            or counts.get("symbolic_row_descriptors") != REQUIRED_ROWS
            or topology.get("coefficient_entries_materialized") != 0
            or topology.get("coefficient_map_registered") is not False
            or phase_two.get("decision") != "BLOCK"
            or phase_two.get("attempted") is not False
        ):
            raise CoordinateFreeCoefficientMapAuditError("readiness boundary mismatch")
    elif name == "revised_thirteen_frame_predecessor":
        counts = value.get("counts", {})
        repair = value.get("exact_gate", {}).get("bounded_classification", {}).get("repair", {})
        if (
            counts.get("total_local_direction_certificates") != 13
            or counts.get("inferred_global_passes") != 0
            or repair.get("local_certificate_constructed") is not True
            or len(repair.get("candidate_records", [])) != 12
        ):
            raise CoordinateFreeCoefficientMapAuditError("local-certificate boundary mismatch")
    elif name == "canonical_D4_obstruction_certificate":
        certificate = value.get("exact_symbolic_certificate", {})
        zero = certificate.get("equal_eigenspace_compressions", {}).get("zero_eigenspace", {})
        if (
            certificate.get("directional_evaluations") != 15
            or zero.get("generic_rank") != 2
            or zero.get("nonzero_entries") != 4
            or zero.get("factorization") != "(34816/15)*alpha^5*W"
        ):
            raise CoordinateFreeCoefficientMapAuditError("canonical obstruction mismatch")
    else:
        counts = value.get("counts", {})
        reduction = value.get("exact_gate", {}).get("symbolic_chart_reduction", {})
        atlas = value.get("exact_gate", {}).get("atlas", {})
        if (
            counts.get("rational_SO3_charts") != 2
            or counts.get("rational_counterexample_points") != 1
            or counts.get("global_chart_numerator_polynomials_materialized") != 0
            or reduction.get("global_two_variable_numerator_polynomials_materialized") != 0
            or atlas.get("union_covers_real_S2") is not True
        ):
            raise CoordinateFreeCoefficientMapAuditError("rational-chart boundary mismatch")
    return {
        "path_status": "verified",
        "schema_version": value.get("schema_version"),
        "status": value["status"],
        "content_sha256": value["content_sha256"],
    }


def _point_evidence_packets(upstreams: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    predecessor = upstreams["revised_thirteen_frame_predecessor"]
    obstruction = upstreams["canonical_D4_obstruction_certificate"]
    rational = upstreams["rational_chart_counterexample_gate"]
    predecessor_gate = predecessor["exact_gate"]
    obstruction_certificate = obstruction["exact_symbolic_certificate"]
    zero = obstruction_certificate["equal_eigenspace_compressions"]["zero_eigenspace"]
    rational_gate = rational["exact_gate"]
    return [
        {
            "packet_id": "revised_thirteen_frame_local_certificate_summary",
            "evidence_kind": "finite_local_direction_certificates",
            "local_direction_certificates": 13,
            "last_selector": predecessor_gate["selector"],
            "candidate_records": 12,
            "registered_coordinate_free_rows": 0,
            "reason_not_a_row_registration": (
                "finite local compatibility values do not identify any odd sphere-mode "
                "coefficient without a registered determining map"
            ),
        },
        {
            "packet_id": "canonical_e1_D4_obstruction",
            "evidence_kind": "single_direction_exact_compression",
            "directional_polarization_evaluations": 15,
            "zero_speed_compression_rank": 2,
            "zero_speed_compression_nonzero_entries": 4,
            "factorization": "(34816/15)*alpha^5*W",
            "witness_matrix_W_sparse": zero["witness_matrix_W_sparse"],
            "registered_coordinate_free_rows": 0,
            "reason_not_a_row_registration": (
                "one point evaluation is a linear functional of 210 sphere modes, not one "
                "mode coefficient"
            ),
        },
        {
            "packet_id": "regular_rational_chart_counterexample",
            "evidence_kind": "single_regular_chart_point",
            "selector": rational_gate["counterexample_selector"],
            "eigenspace_compressions_checked": 84,
            "candidate_obstructions": 12,
            "global_numerator_polynomials_materialized": 0,
            "registered_coordinate_free_rows": 0,
            "reason_not_a_row_registration": (
                "the upstream gate terminated at a counterexample and explicitly materialized "
                "zero global numerator polynomials"
            ),
        },
    ]


def _checkpoint(point_packets: list[dict[str, Any]]) -> dict[str, Any]:
    checkpoint = {
        "schema_version": CHECKPOINT_SCHEMA,
        "target": {
            "unknown_columns": UNKNOWN_COLUMNS,
            "cokernel_coordinates": COKERNEL_COORDINATES,
            "odd_sphere_modes": ODD_SPHERE_MODES,
            "required_coefficient_rows": REQUIRED_ROWS,
            "required_rhs_rows": REQUIRED_ROWS,
        },
        "registration_cursor": {
            "next_cokernel_coordinate": 0,
            "next_odd_sphere_mode": 0,
            "next_flat_row_offset": 0,
        },
        "registered_coefficient_rows": [],
        "registered_rhs_rows": [],
        "registered_sparse_entries": [],
        "point_evidence_packet_sha256": hashlib.sha256(_canonical_bytes(point_packets)).hexdigest(),
        "counts": {
            "registered_coefficient_rows": 0,
            "registered_rhs_rows": 0,
            "registered_sparse_entries": 0,
            "missing_coefficient_rows": REQUIRED_ROWS,
            "missing_rhs_rows": REQUIRED_ROWS,
            "point_evidence_packets": len(point_packets),
        },
        "complete": False,
        "first_missing_row": {
            "flat_offset": 0,
            "equal_eigenspace_cokernel_coordinate": 0,
            "odd_sphere_mode": 0,
        },
    }
    checkpoint["content_sha256"] = _content_hash(checkpoint)
    return checkpoint


def build_campaign(project_root: Path, config_path: Path) -> dict[str, Any]:
    root = project_root.resolve()
    config = _load_json(config_path.resolve())
    _validate_config(config)
    upstream_receipts: dict[str, Any] = {}
    upstream_documents: dict[str, dict[str, Any]] = {}
    for name, binding in config["upstream_bindings"].items():
        if set(binding) != {"path", "content_sha256"}:
            raise CoordinateFreeCoefficientMapAuditError(f"invalid upstream binding: {name}")
        if binding["content_sha256"] != EXPECTED_UPSTREAMS[name]["content_sha256"]:
            raise CoordinateFreeCoefficientMapAuditError(f"config upstream hash mismatch: {name}")
        document = _load_json(_resolve_under(root, binding["path"]))
        upstream_documents[name] = document
        upstream_receipts[name] = _validate_upstream(name, document) | {"path": binding["path"]}
    packets = _point_evidence_packets(upstream_documents)
    if len(packets) > config["resource_caps"]["maximum_point_evidence_packets"]:
        raise CoordinateFreeCoefficientMapAuditError("point evidence cap exceeded")
    checkpoint = _checkpoint(packets)
    checkpoint_bytes = len(_canonical_bytes(checkpoint))
    if checkpoint_bytes > config["resource_caps"]["maximum_checkpoint_bytes"]:
        raise CoordinateFreeCoefficientMapAuditError("checkpoint byte cap exceeded")
    counts = checkpoint["counts"]
    phase_two_admitted = bool(
        counts["registered_coefficient_rows"] == REQUIRED_ROWS
        and counts["registered_rhs_rows"] == REQUIRED_ROWS
        and counts["missing_coefficient_rows"] == 0
        and counts["missing_rhs_rows"] == 0
        and checkpoint["complete"] is True
    )
    if phase_two_admitted:
        raise CoordinateFreeCoefficientMapAuditError(
            "unexpected complete map requires a new solve-bearing schema"
        )
    source_path = _resolve_under(root, SOURCE_PATH)
    test_path = _resolve_under(root, TEST_PATH)
    artifact = {
        "schema_version": SCHEMA,
        "status": STATUS,
        "config_sha256": config["content_sha256"],
        "implementation_bindings": {
            "config": {"path": CONFIG_PATH, "file_sha256": _file_sha256(config_path)},
            "source": {"path": SOURCE_PATH, "file_sha256": _file_sha256(source_path)},
            "test": {"path": TEST_PATH, "file_sha256": _file_sha256(test_path)},
        },
        "upstream_receipts": upstream_receipts,
        "point_evidence_packets": packets,
        "sparse_registration_checkpoint": checkpoint,
        "resource_admission": {
            "caps": config["resource_caps"],
            "checkpoint_bytes": checkpoint_bytes,
            "checkpoint_within_cap": True,
            "registered_rows_within_cap": True,
            "registered_sparse_entries_within_cap": True,
        },
        "phase_two": {
            "decision": "BLOCK",
            "attempted": False,
            "admitted": False,
            "PASS": False,
            "OBSTRUCTED_CLASS": False,
            "BLOCK": True,
            "admission_requirements": config["phase_two_admission"],
        },
        "counts": {
            "upstream_seals_verified": 4,
            "point_evidence_packets": len(packets),
            "finite_local_direction_certificates_observed": 13,
            "single_direction_obstruction_packets": 2,
            "required_coefficient_rows": REQUIRED_ROWS,
            "registered_coefficient_rows": 0,
            "missing_coefficient_rows": REQUIRED_ROWS,
            "required_rhs_rows": REQUIRED_ROWS,
            "registered_rhs_rows": 0,
            "missing_rhs_rows": REQUIRED_ROWS,
            "registered_sparse_entries": 0,
            "global_numerator_polynomials_materialized_upstream": 0,
            "phase_two_solve_attempts": 0,
            "negative_controls": len(NEGATIVE_CONTROLS),
        },
        "claims": {
            "all_four_upstream_content_seals_verified": True,
            "deterministic_sparse_checkpoint_materialized": True,
            "exact_registered_and_missing_counts_reported": True,
            "point_evidence_separated_from_coefficient_registration": True,
        }
        | {claim: False for claim in sorted(FALSE_CLAIMS)},
        "negative_controls": NEGATIVE_CONTROLS,
        "scope": (
            "Exact audit of what the sealed pointwise D4 evidence registers in the required "
            "558-by-210 coordinate-free sphere-mode coefficient map. Three point-evidence "
            "packets are preserved, but none is inferred to be an individual polynomial-mode "
            "coefficient. Therefore zero of 117,180 coefficient rows, zero of 117,180 RHS "
            "rows, and zero sparse entries are registered; phase two remains blocked. No "
            "D4 full-sphere, D2F, high-atom, TC2, H7, tube, PDE, lifespan, local, covariant, "
            "candidate-pass, or candidate-rejection claim follows."
        ),
        "first_blocker": (
            "register_a_coordinate_free_symbolic_D4_recurrence_emitter_that_expands_all_"
            "seven_equal_eigenspace_compressions_into_the_210_odd_sphere_modes"
        ),
        "errors": [],
    }
    artifact["content_sha256"] = _content_hash(artifact)
    return artifact


def validate_campaign(document: dict[str, Any], project_root: Path | None = None) -> None:
    root = (project_root or Path(__file__).resolve().parents[2]).resolve()
    expected = build_campaign(root, root / CONFIG_PATH)
    checkpoint = document.get("sparse_registration_checkpoint", {})
    if (
        document != expected
        or not _hash_matches(document)
        or not isinstance(checkpoint, dict)
        or not _hash_matches(checkpoint)
    ):
        raise CoordinateFreeCoefficientMapAuditError("campaign replay mismatch")


def write_campaign(document: dict[str, Any], output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(document, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8", newline="\n")
    temporary.replace(output)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    document = build_campaign(args.project_root, args.config)
    write_campaign(document, args.output)
    validate_campaign(document, args.project_root)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
