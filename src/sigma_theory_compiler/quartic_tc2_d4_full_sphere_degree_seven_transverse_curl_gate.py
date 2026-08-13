from __future__ import annotations

import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path
from typing import Any

SCHEMA = "sigma-quartic-tc2-d4-full-sphere-degree-seven-transverse-curl-gate-1.0"
CONFIG_SCHEMA = "sigma-quartic-tc2-d4-full-sphere-degree-seven-transverse-curl-config-1.0"
STATUS_BLOCKED = "block_full_sphere_degree_seven_exact_solve_coefficient_map_not_registered"
SOURCE_PATH = (
    "src/sigma_theory_compiler/quartic_tc2_d4_full_sphere_degree_seven_transverse_curl_gate.py"
)
TEST_PATH = "tests/test_quartic_tc2_d4_full_sphere_degree_seven_transverse_curl_gate.py"
CONFIG_PATH = (
    "configs/backgrounds/quartic_tc2_d4_full_sphere_degree_seven_transverse_curl_gate.json"
)
ARTIFACT_PATH = (
    "runs/physics-language/"
    "quartic-tc2-d4-full-sphere-degree-seven-transverse-curl-gate/campaign.json"
)

STATE_DIMENSION = 55
GRADIENT_LIFT_COLUMNS = 33
TRANSVERSE_CURL_CHANNELS = 22
ENVELOPE_DEGREE = 6
SYMBOL_DEGREE = 7
SPHERE_POLYNOMIAL_DEGREE = 19
EQUAL_EIGENSPACE_COKERNEL_DIMENSION = 558
ODD_SPHERE_MODE_DIMENSION = 210
UNKNOWN_COLUMNS = 33_880
SYMBOLIC_ROW_DESCRIPTORS = 117_180
DENSE_RATIONAL_ENTRY_CEILING = 3_970_058_400
SPECTRUM = ["0", "1", "-1", "1/2", "-1/2", "1/3", "-1/3"]
RECURRENCE_ORDERS = [1, 2, 3, 4]
EXPECTED_RESOURCE_CAPS = {
    "maximum_unknown_columns": 40_000,
    "maximum_symbolic_row_descriptors": 120_000,
    "maximum_descriptor_bytes": 33_554_432,
    "maximum_dense_rational_entries": 100_000_000,
    "maximum_exact_sparse_nonzeros": 2_000_000,
    "maximum_exact_solver_rows": 4_096,
    "maximum_exact_solver_columns": 4_096,
}

EXPECTED_UPSTREAMS = {
    "revised_thirteen_frame_predecessor": {
        "content_sha256": ("55a68d34961739728a6ae111ea1c76f83f51524614712d7e960f4f37a1139267"),
        "status": "pass_exact_second_height_two_point_and_bounded_classification",
    },
    "D4_obstruction_cokernel_certificate": {
        "content_sha256": ("bef3246a17942c74e8f3cdb09bc14a36c6bdc44d030a9a70ce833c30ec04bc65"),
        "status": "pass_exact_canonical_d4_obstruction_cokernel_classification",
    },
    "rational_direction_chart_gate": {
        "content_sha256": ("48b8ecfe63336071721baeb90a41f379ac1b4235629b380abdf0124e7008152c"),
        "status": (
            "pass_exact_rational_chart_counterexample_disproves_current_full_sphere_"
            "D4_compatibility"
        ),
    },
    "full_Sylvester_reference": {
        "content_sha256": ("61af809a74e50cf379375512fca61559ff3fb3df63faef51a3640d727fea8523"),
        "status": (
            "pass_all_12_full_reference_TC2_Sylvester_solutions_variable_extension_"
            "global_H7_fail_closed"
        ),
    },
}

FALSE_CLAIMS = {
    "B7_closed",
    "CK1_closed",
    "CK3_closed",
    "TC2_closed",
    "boundary_energy_admission_proved",
    "complete_D2F_tensor_registered",
    "corrected_candidate_family_registered",
    "covariant_action_origin_proved",
    "full_direction_sphere_D4_compatibility_proved",
    "full_high_atom_identity_proved",
    "full_tube_Sylvester_identity_proved",
    "global_H7_closed",
    "lifespan_proved",
    "local_differential_operator_origin_proved",
    "nonlinear_PDE_closure_proved",
    "phase_two_exact_solve_admitted",
    "theory_candidate_rejected",
    "variable_coefficient_constraint_calculus_proved",
}

NEGATIVE_CONTROLS = {
    "accept_dense_solve_above_entry_cap": {"rejected": True},
    "accept_float_or_modular_rank_without_exact_witness": {"rejected": True},
    "fit_twelve_unrelated_candidate_symbols": {"rejected": True},
    "infer_full_sphere_from_thirteen_points": {"rejected": True},
    "infer_H7_tube_or_lifespan_from_D4": {"rejected": True},
    "omit_any_equal_eigenspace_block": {"rejected": True},
    "omit_lower_recurrence_orders": {"rejected": True},
    "solve_without_bound_sparse_coefficient_map": {"rejected": True},
    "use_even_total_symbol_parity": {"rejected": True},
    "use_noncurl_covector_with_gradient_residual": {"rejected": True},
}


class FullSphereDegreeSevenGateError(ValueError):
    """Raised when the full-sphere gate or one of its seals is inconsistent."""


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
        raise FullSphereDegreeSevenGateError(f"expected JSON object: {path}")
    return value


def _resolve_under(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if root != path and root not in path.parents:
        raise FullSphereDegreeSevenGateError("bound path escaped project root")
    return path


def _homogeneous_monomials(degree: int) -> list[tuple[int, int, int]]:
    return [(a, b, degree - a - b) for a in range(degree + 1) for b in range(degree - a + 1)]


def _odd_sphere_modes(degree_ceiling: int) -> list[tuple[int, int]]:
    return [
        (degree, mode)
        for degree in range(1, degree_ceiling + 1, 2)
        for mode in range(2 * degree + 1)
    ]


def _descriptor_seal(descriptors: list[tuple[int, ...]]) -> tuple[str, int, int]:
    digest = hashlib.sha256()
    seen: set[tuple[int, ...]] = set()
    byte_count = 0
    for descriptor in descriptors:
        if descriptor in seen:
            continue
        seen.add(descriptor)
        payload = _canonical_bytes(list(descriptor)) + b"\n"
        digest.update(payload)
        byte_count += len(payload)
    return digest.hexdigest(), len(seen), byte_count


def _construct_symbolic_sparse_topology() -> dict[str, Any]:
    monomials = _homogeneous_monomials(ENVELOPE_DEGREE)
    modes = _odd_sphere_modes(SPHERE_POLYNOMIAL_DEGREE)
    columns = [
        (output, curl, a, b, c)
        for output in range(STATE_DIMENSION)
        for curl in range(TRANSVERSE_CURL_CHANNELS)
        for a, b, c in monomials
    ]
    rows = [
        (cokernel, degree, mode)
        for cokernel in range(EQUAL_EIGENSPACE_COKERNEL_DIMENSION)
        for degree, mode in modes
    ]
    column_sha, deduplicated_columns, column_bytes = _descriptor_seal(columns)
    row_sha, deduplicated_rows, row_bytes = _descriptor_seal(rows)
    if (
        len(monomials) != 28
        or len(modes) != ODD_SPHERE_MODE_DIMENSION
        or len(columns) != UNKNOWN_COLUMNS
        or len(rows) != SYMBOLIC_ROW_DESCRIPTORS
        or deduplicated_columns != len(columns)
        or deduplicated_rows != len(rows)
    ):
        raise FullSphereDegreeSevenGateError("symbolic sparse topology count mismatch")
    return {
        "column_index": {
            "descriptor": "(output_state,curl_channel,n1_power,n2_power,n3_power)",
            "homogeneous_monomials": len(monomials),
            "raw_descriptors": len(columns),
            "deduplicated_descriptors": deduplicated_columns,
            "duplicates_removed": len(columns) - deduplicated_columns,
            "descriptor_sha256": column_sha,
            "first_descriptor": list(columns[0]),
            "last_descriptor": list(columns[-1]),
        },
        "row_index": {
            "descriptor": "(equal_eigenspace_cokernel_coordinate,odd_sphere_degree,mode)",
            "odd_sphere_modes": len(modes),
            "raw_descriptors": len(rows),
            "deduplicated_descriptors": deduplicated_rows,
            "duplicates_removed": len(rows) - deduplicated_rows,
            "descriptor_sha256": row_sha,
            "first_descriptor": list(rows[0]),
            "last_descriptor": list(rows[-1]),
        },
        "descriptor_bytes": column_bytes + row_bytes,
        "dense_rational_entry_ceiling": len(columns) * len(rows),
        "coefficient_entries_materialized": 0,
        "coefficient_map_registered": False,
        "rhs_registered": False,
    }


def _validate_config(config: dict[str, Any]) -> None:
    expected_keys = {
        "schema_version",
        "global_claim_policy",
        "phase_architecture",
        "declared_class",
        "resource_caps",
        "solve_inputs",
        "upstream_bindings",
        "content_sha256",
    }
    declared = config.get("declared_class", {})
    caps = config.get("resource_caps", {})
    if (
        set(config) != expected_keys
        or config.get("schema_version") != CONFIG_SCHEMA
        or config.get("global_claim_policy") != "fail_closed"
        or config.get("phase_architecture")
        != [
            "exact_preregistration_and_sparse_topology_readiness",
            "admitted_exact_rational_solve",
        ]
        or not _hash_matches(config)
        or declared.get("state_dimension") != STATE_DIMENSION
        or declared.get("gradient_lift_columns") != GRADIENT_LIFT_COLUMNS
        or declared.get("transverse_curl_channels") != TRANSVERSE_CURL_CHANNELS
        or declared.get("homogeneous_even_envelope_degree") != ENVELOPE_DEGREE
        or declared.get("odd_symbol_degree_ceiling") != SYMBOL_DEGREE
        or declared.get("sphere_polynomial_degree_ceiling") != SPHERE_POLYNOMIAL_DEGREE
        or declared.get("spectrum") != SPECTRUM
        or declared.get("equal_eigenspace_skew_cokernel_dimension")
        != EQUAL_EIGENSPACE_COKERNEL_DIMENSION
        or declared.get("recurrence_orders") != RECURRENCE_ORDERS
        or declared.get("directional_polarization_evaluations") != 15
        or declared.get("candidate_count") != 12
        or declared.get("prior_local_direction_certificates") != 13
        or caps != EXPECTED_RESOURCE_CAPS
        or set(config.get("solve_inputs", {}))
        != {"coordinate_free_D4_sparse_coefficient_map", "exact_rhs_vector"}
        or config.get("solve_inputs")
        != {
            "coordinate_free_D4_sparse_coefficient_map": None,
            "exact_rhs_vector": None,
        }
        or set(config.get("upstream_bindings", {})) != set(EXPECTED_UPSTREAMS)
    ):
        raise FullSphereDegreeSevenGateError("invalid full-sphere gate config")


def _validate_upstream(name: str, value: dict[str, Any]) -> dict[str, Any]:
    expected = EXPECTED_UPSTREAMS[name]
    if (
        not _hash_matches(value)
        or value.get("content_sha256") != expected["content_sha256"]
        or value.get("status") != expected["status"]
        or value.get("errors") != []
    ):
        raise FullSphereDegreeSevenGateError(f"upstream seal mismatch: {name}")
    if name == "revised_thirteen_frame_predecessor":
        counts = value.get("counts", {})
        if (
            counts.get("total_local_direction_certificates") != 13
            or counts.get("inferred_global_passes") != 0
            or counts.get("candidate_conditions_checked") != 12
        ):
            raise FullSphereDegreeSevenGateError("thirteen-frame scope mismatch")
    elif name == "D4_obstruction_cokernel_certificate":
        counts = value.get("counts", {})
        certificate = value.get("exact_symbolic_certificate", {})
        zero = certificate.get("equal_eigenspace_compressions", {}).get("zero_eigenspace", {})
        if (
            counts.get("candidate_obstructions_certified") != 12
            or counts.get("compression_generic_rank") != 2
            or zero.get("generic_rank") != 2
            or zero.get("factorization") != "(34816/15)*alpha^5*W"
        ):
            raise FullSphereDegreeSevenGateError("D4 cokernel certificate mismatch")
    elif name == "rational_direction_chart_gate":
        counts = value.get("counts", {})
        atlas = value.get("exact_gate", {}).get("atlas", {})
        if (
            counts.get("rational_SO3_charts") != 2
            or counts.get("inferred_global_passes") != 0
            or atlas.get("union_covers_real_S2") is not True
            or atlas.get("real_singular_strata") != 0
        ):
            raise FullSphereDegreeSevenGateError("rational chart scope mismatch")
    else:
        counts = value.get("counts", {})
        packet = value.get("common_full_reference_Sylvester_packet", {})
        if (
            counts.get("full_reference_Sylvester_solutions") != 12
            or counts.get("variable_coefficient_solvability_proofs") != 0
            or packet.get("spectrum") != SPECTRUM
            or packet.get("minimum_distinct_spectral_gap") != "1/6"
        ):
            raise FullSphereDegreeSevenGateError("full Sylvester reference mismatch")
    return {
        "content_sha256": value["content_sha256"],
        "schema_version": value.get("schema_version"),
        "status": value["status"],
    }


def _fraction(value: int | str | Fraction) -> Fraction:
    if isinstance(value, Fraction):
        return value
    return Fraction(value)


def _fraction_text(value: Fraction) -> str:
    return str(value.numerator) if value.denominator == 1 else str(value)


def solve_exact_sparse_system(
    rows: list[dict[int, int | str | Fraction]],
    rhs: list[int | str | Fraction],
    unknown_count: int,
    *,
    maximum_rows: int,
    maximum_columns: int,
    maximum_nonzeros: int,
) -> dict[str, Any]:
    """Exactly solve a bounded rational system or return a verified obstruction."""

    if unknown_count < 0 or len(rows) != len(rhs):
        raise FullSphereDegreeSevenGateError("invalid exact sparse system shape")
    nonzeros = sum(len(row) for row in rows)
    if len(rows) > maximum_rows or unknown_count > maximum_columns or nonzeros > maximum_nonzeros:
        return {
            "decision": "BLOCK",
            "reason": "exact_solver_resource_cap_not_admitted",
            "rows": len(rows),
            "columns": unknown_count,
            "nonzeros": nonzeros,
            "attempted": False,
        }
    matrix: list[list[Fraction]] = []
    original_rows: list[list[Fraction]] = []
    original_rhs = [_fraction(value) for value in rhs]
    for row_index, sparse in enumerate(rows):
        if any(index < 0 or index >= unknown_count for index in sparse):
            raise FullSphereDegreeSevenGateError("sparse column outside declared shape")
        dense = [Fraction(0) for _ in range(unknown_count)]
        for index, value in sparse.items():
            dense[index] = _fraction(value)
        original_rows.append(dense[:])
        tracker = [Fraction(int(index == row_index)) for index in range(len(rows))]
        matrix.append(dense + [original_rhs[row_index]] + tracker)
    pivot_rows: dict[int, int] = {}
    next_row = 0
    for column in range(unknown_count):
        pivot = next(
            (index for index in range(next_row, len(matrix)) if matrix[index][column]),
            None,
        )
        if pivot is None:
            continue
        matrix[next_row], matrix[pivot] = matrix[pivot], matrix[next_row]
        divisor = matrix[next_row][column]
        matrix[next_row] = [value / divisor for value in matrix[next_row]]
        for row_index in range(len(matrix)):
            if row_index == next_row or not matrix[row_index][column]:
                continue
            factor = matrix[row_index][column]
            matrix[row_index] = [
                left - factor * right
                for left, right in zip(matrix[row_index], matrix[next_row], strict=True)
            ]
        pivot_rows[column] = next_row
        next_row += 1
        if next_row == len(matrix):
            break
    for reduced in matrix:
        if all(value == 0 for value in reduced[:unknown_count]) and reduced[unknown_count] != 0:
            witness = reduced[unknown_count + 1 :]
            witness_left = [
                sum(witness[row] * original_rows[row][column] for row in range(len(rows)))
                for column in range(unknown_count)
            ]
            witness_rhs = sum(witness[row] * original_rhs[row] for row in range(len(rows)))
            if any(witness_left) or witness_rhs == 0:
                raise FullSphereDegreeSevenGateError("invalid rational obstruction witness")
            return {
                "decision": "OBSTRUCTED_CLASS",
                "attempted": True,
                "rows": len(rows),
                "columns": unknown_count,
                "nonzeros": nonzeros,
                "rank": len(pivot_rows),
                "left_nullspace_witness": [_fraction_text(value) for value in witness],
                "witness_times_matrix": ["0"] * unknown_count,
                "witness_times_rhs": _fraction_text(witness_rhs),
                "exact_witness_verified": True,
            }
    solution = [Fraction(0) for _ in range(unknown_count)]
    for column, row_index in pivot_rows.items():
        solution[column] = matrix[row_index][unknown_count]
    residuals = [
        sum(original_rows[row][column] * solution[column] for column in range(unknown_count))
        - original_rhs[row]
        for row in range(len(rows))
    ]
    if any(residuals):
        raise FullSphereDegreeSevenGateError("exact solution did not replay")
    return {
        "decision": "PASS",
        "attempted": True,
        "rows": len(rows),
        "columns": unknown_count,
        "nonzeros": nonzeros,
        "rank": len(pivot_rows),
        "solution": [_fraction_text(value) for value in solution],
        "exact_residuals": ["0"] * len(rows),
        "exact_solution_verified": True,
    }


def _phase_two(config: dict[str, Any], topology: dict[str, Any]) -> dict[str, Any]:
    solve_inputs = config["solve_inputs"]
    coefficient_map = solve_inputs["coordinate_free_D4_sparse_coefficient_map"]
    rhs_vector = solve_inputs["exact_rhs_vector"]
    caps = config["resource_caps"]
    dense_admitted = (
        topology["dense_rational_entry_ceiling"] <= caps["maximum_dense_rational_entries"]
    )
    if coefficient_map is None or rhs_vector is None:
        return {
            "decision": "BLOCK",
            "attempted": False,
            "solve_admitted": False,
            "dense_solve_admitted": dense_admitted,
            "sparse_solve_required": not dense_admitted,
            "first_blocker": (
                "register_the_coordinate_free_D4_sparse_coefficient_map_and_exact_rhs_"
                "for_all_558_equal_eigenspace_cokernel_coordinates_and_210_odd_sphere_modes"
            ),
            "PASS": False,
            "OBSTRUCTED_CLASS": False,
            "BLOCK": True,
        }
    raise FullSphereDegreeSevenGateError(
        "bound production coefficient-map ingestion is not registered in schema 1.0"
    )


def build_campaign(project_root: Path, config_path: Path) -> dict[str, Any]:
    root = project_root.resolve()
    config = _load_json(config_path.resolve())
    _validate_config(config)
    upstream_receipts: dict[str, Any] = {}
    for name, binding in config["upstream_bindings"].items():
        if set(binding) != {"path", "content_sha256"}:
            raise FullSphereDegreeSevenGateError(f"invalid upstream binding: {name}")
        if binding["content_sha256"] != EXPECTED_UPSTREAMS[name]["content_sha256"]:
            raise FullSphereDegreeSevenGateError(f"config upstream hash mismatch: {name}")
        upstream = _load_json(_resolve_under(root, binding["path"]))
        upstream_receipts[name] = _validate_upstream(name, upstream) | {"path": binding["path"]}
    topology = _construct_symbolic_sparse_topology()
    caps = config["resource_caps"]
    descriptor_admitted = (
        topology["column_index"]["deduplicated_descriptors"] <= caps["maximum_unknown_columns"]
        and topology["row_index"]["deduplicated_descriptors"]
        <= caps["maximum_symbolic_row_descriptors"]
        and topology["descriptor_bytes"] <= caps["maximum_descriptor_bytes"]
    )
    if not descriptor_admitted:
        raise FullSphereDegreeSevenGateError(
            "symbolic sparse topology exceeded preregistered readiness caps"
        )
    phase_one = {
        "decision": "PASS_EXACT_PREREGISTRATION_AND_SPARSE_TOPOLOGY_READINESS",
        "upstream_seals_verified": len(upstream_receipts),
        "descriptor_topology_admitted": True,
        "declared_class": config["declared_class"],
        "resource_caps": caps,
        "symbolic_sparse_topology": topology,
        "exact_dimension_identities": {
            "degree_six_monomials": "binomial(8,2)=28",
            "unknown_columns": "55*22*28=33880",
            "odd_sphere_modes_through_19": "sum_(l=1,3,...,19)(2*l+1)=210",
            "symbolic_rows": "558*210=117180",
            "dense_entry_ceiling": "33880*117180=3970058400",
        },
    }
    phase_two = _phase_two(config, topology)
    source_path = _resolve_under(root, SOURCE_PATH)
    test_path = _resolve_under(root, TEST_PATH)
    claims = {
        "all_four_upstream_content_seals_verified": True,
        "degree_seven_transverse_curl_class_preregistered": True,
        "exact_sparse_column_and_row_topology_constructed": True,
        "exact_sparse_topology_deduplicated": True,
        "phase_one_readiness_passed": True,
    } | {claim: False for claim in sorted(FALSE_CLAIMS)}
    artifact = {
        "schema_version": SCHEMA,
        "status": STATUS_BLOCKED,
        "config_sha256": config["content_sha256"],
        "implementation_bindings": {
            "config": {"path": CONFIG_PATH, "file_sha256": _file_sha256(config_path)},
            "source": {"path": SOURCE_PATH, "file_sha256": _file_sha256(source_path)},
            "test": {"path": TEST_PATH, "file_sha256": _file_sha256(test_path)},
        },
        "upstream_receipts": upstream_receipts,
        "phase_one": phase_one,
        "phase_two": phase_two,
        "counts": {
            "upstream_seals_verified": 4,
            "state_dimension": STATE_DIMENSION,
            "gradient_lift_columns": GRADIENT_LIFT_COLUMNS,
            "transverse_curl_channels": TRANSVERSE_CURL_CHANNELS,
            "degree_six_homogeneous_monomials": 28,
            "unknown_columns": UNKNOWN_COLUMNS,
            "equal_eigenspace_cokernel_coordinates": (EQUAL_EIGENSPACE_COKERNEL_DIMENSION),
            "odd_sphere_modes": ODD_SPHERE_MODE_DIMENSION,
            "symbolic_row_descriptors": SYMBOLIC_ROW_DESCRIPTORS,
            "descriptor_duplicates_removed": 0,
            "coefficient_entries_materialized": 0,
            "candidate_replays_attempted": 0,
            "full_sphere_passes": 0,
            "obstructed_classes": 0,
            "blocked_classes": 1,
            "negative_controls": len(NEGATIVE_CONTROLS),
        },
        "claims": claims,
        "negative_controls": NEGATIVE_CONTROLS,
        "scope": (
            "Exact upstream-seal validation and exact construction/deduplication of the "
            "33,880-column by 117,180-row symbolic sparse index topology for the declared "
            "degree-seven transverse-curl full-sphere D4 class. The coordinate-free sparse "
            "coefficient map and exact RHS are not registered, so phase two is not admitted. "
            "No D4 full-sphere, D2F, high-atom, TC2, H7, tube, PDE, lifespan, local, covariant, "
            "candidate-pass, or candidate-rejection claim follows."
        ),
        "first_blocker": phase_two["first_blocker"],
        "errors": [],
    }
    artifact["content_sha256"] = _content_hash(artifact)
    return artifact


def validate_campaign(document: dict[str, Any], project_root: Path | None = None) -> None:
    root = (project_root or Path(__file__).resolve().parents[2]).resolve()
    expected = build_campaign(root, root / CONFIG_PATH)
    if document != expected or not _hash_matches(document):
        raise FullSphereDegreeSevenGateError("campaign replay mismatch")


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
