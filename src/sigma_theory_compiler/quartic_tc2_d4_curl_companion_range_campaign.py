from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import sympy as sp

from .horndeski_principal import _symmetric_basis
from .quartic_first_order_reduction_campaign import (
    _extract_spatial_blocks,
    _full_first_order_pencil,
    _symbol_data,
)
from .quartic_tc2_d4_minimal_tc2_escape_campaign import _correction_basis
from .quartic_tc2_diagonal_third_jet_campaign import _content_hash, _matrix_payload
from .quartic_tc2_mixed_third_jet_continuation_service import (
    _atomic_write,
    _file_sha256,
    _hash_matches,
    _json_bytes,
    _load_file,
    _with_hash,
)
from .quartic_tc2_variable_sylvester_campaign import (
    STATE_DIMENSION,
    _reference_and_first_jet_packet,
)

SCHEMA = "sigma-quartic-tc2-d4-curl-companion-range-campaign-1.0"
CONFIG_SCHEMA = "sigma-quartic-tc2-d4-curl-companion-range-config-1.0"
OBLIGATION_OFFSET = 244
ACTIVE_INDICES = (0, 2, 3, 9)
EXPECTED_CANDIDATES = 12
EXPECTED_V_SHA256 = "a8a6cb0588ebae512db867990f937a3a9e5a9a38bf90be807fc62a8eb928f9c0"
EXPECTED_COMPANION_BLOCK_SHA256 = (
    "9ef0bdb7ea7009ebba9b25ccb1225e1b955351d62a4b16c7989d339508a3b195"
)
EXPECTED_COMPANION_COMPRESSION_SHA256 = (
    "def5dc985fa3356a9a21b2b06d4ebe0f0365058403e3e762eab161d7fb2822be"
)
EXPECTED_ROTATED_W_SHA256 = (
    "3d5c76103f2f85c97383cfc7e7430309e51e5496bdf0c26b2198fe55f2c201da"
)


class QuarticTC2D4CurlCompanionRangeError(ValueError):
    """Raised when the curl-companion range certificate is inconsistent."""


def _load_bound(root: Path, binding: Mapping[str, Any]) -> dict[str, Any]:
    path = (root / str(binding["path"])).resolve()
    if root.resolve() not in path.parents:
        raise QuarticTC2D4CurlCompanionRangeError(
            "bound input escaped project root"
        )
    value, data = _load_file(path)
    if (
        _file_sha256(data) != binding.get("file_sha256")
        or value.get("content_sha256") != binding.get("content_sha256")
        or not _hash_matches(value)
    ):
        raise QuarticTC2D4CurlCompanionRangeError(
            f"bound input mismatch: {binding.get('path')}"
        )
    return value


def _check_raw_binding(root: Path, binding: Mapping[str, Any]) -> None:
    path = (root / str(binding["path"])).resolve()
    if root.resolve() not in path.parents or not path.is_file():
        raise QuarticTC2D4CurlCompanionRangeError(
            "raw binding escaped project root or is absent"
        )
    if _file_sha256(path.read_bytes()) != binding.get("file_sha256"):
        raise QuarticTC2D4CurlCompanionRangeError(
            f"raw binding mismatch: {binding.get('path')}"
        )


def _sparse_matrix(matrix: sp.MatrixBase) -> list[dict[str, Any]]:
    return [
        {"row": row, "column": column, "value": str(sp.factor(matrix[row, column]))}
        for row in range(matrix.rows)
        for column in range(matrix.cols)
        if matrix[row, column] != 0
    ]


def _axis_swap_1_2() -> sp.Matrix:
    spacetime_swap = sp.zeros(4)
    for row, column in enumerate((0, 2, 1, 3)):
        spacetime_swap[row, column] = 1
    basis = _symmetric_basis()
    field_rotation = sp.zeros(11)
    for source, source_basis in enumerate(basis):
        rotated = spacetime_swap * source_basis * spacetime_swap.T
        for target, target_basis in enumerate(basis):
            field_rotation[target, source] = sp.trace(target_basis.T * rotated)
    field_rotation[10, 10] = 1
    spatial_swap = sp.Matrix([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    original = sp.zeros(STATE_DIMENSION)
    original[0:11, 0:11] = field_rotation
    original[11:22, 11:22] = field_rotation
    for target in range(3):
        for source in range(3):
            original[
                22 + 11 * target : 33 + 11 * target,
                22 + 11 * source : 33 + 11 * source,
            ] = spatial_swap[target, source] * field_rotation
    ordering = [*range(11), *range(33, 55), *range(11, 33)]
    result = original.extract(ordering, ordering)
    if result * result.T != sp.eye(STATE_DIMENSION) or result.det() not in (-1, 1):
        raise QuarticTC2D4CurlCompanionRangeError(
            "axis-swap state representation is not orthogonal"
        )
    return result


def _direct_axis_2_physical() -> sp.Matrix:
    data = _symbol_data()
    xi = data["xi_lower"]
    substitutions: dict[sp.Symbol, sp.Expr] = {
        data["alpha"]: 0,
        data["m2"]: 1,
        data["c20"]: 0,
        xi[1]: 0,
        xi[2]: 1,
        xi[3]: 0,
    }
    substitutions.update({symbol: 0 for symbol in data["gradient_lower"]})
    substitutions.update({symbol: 0 for symbol in data["hessian_lower"]})
    substitutions.update({symbol: 0 for symbol in data["einstein_upper"]})
    b_blocks, c_blocks = _extract_spatial_blocks(
        data["first_order"]["B"], data["first_order"]["C"], list(xi[1:])
    )
    mass, evolution = _full_first_order_pencil(
        data["first_order"]["A"].subs(substitutions),
        b_blocks[1].subs(substitutions),
        [c_blocks[1][index].subs(substitutions) for index in range(3)],
        [0, 1, 0],
    )
    ordering = [*range(11), *range(33, 55), *range(11, 33)]
    return (mass.inv() * evolution).extract(ordering, ordering).applyfunc(sp.factor)


def _reduced_exterior_range(
    projector2: sp.Matrix, target: sp.Matrix
) -> dict[str, Any]:
    _, pivot_columns = projector2.T.rref()
    covector_basis = projector2.T[:, list(pivot_columns)]
    _, pivot_rows = covector_basis.T.rref()
    selected_rows = list(pivot_rows)
    coordinate_map = (
        covector_basis[selected_rows, :].inv()
        * sp.eye(STATE_DIMENSION)[selected_rows, :]
    )
    if coordinate_map * covector_basis != sp.eye(33):
        raise QuarticTC2D4CurlCompanionRangeError(
            "zero-speed covector coordinate map mismatch"
        )
    target_coordinates = (coordinate_map * target * coordinate_map.T).applyfunc(
        sp.factor
    )
    if covector_basis * target_coordinates * covector_basis.T != target:
        raise QuarticTC2D4CurlCompanionRangeError(
            "companion target coordinate reconstruction mismatch"
        )

    # With the direction-1 block fixed exactly, all C12 coefficients are fixed and
    # every C13 coefficient is zero. Only arbitrary output multiples of the eleven
    # C23^[b] constraints remain. Their direction-2 selectors are w3[b], indices 22:33.
    selector_coordinates = sp.Matrix.hstack(
        *[
            coordinate_map * projector2.T * sp.eye(STATE_DIMENSION)[:, 22 + field]
            for field in range(11)
        ]
    )
    upper = [(left, right) for left in range(33) for right in range(left + 1, 33)]
    columns: list[list[sp.Expr]] = []
    identity33 = sp.eye(33)
    for field in range(11):
        selector = selector_coordinates[:, field]
        for output_index in range(33):
            output = identity33[:, output_index]
            wedge = output * selector.T - selector * output.T
            columns.append([wedge[left, right] for left, right in upper])
    range_matrix = sp.Matrix(
        len(upper), len(columns), lambda row, column: columns[column][row]
    )
    target_vector = sp.Matrix(
        [target_coordinates[left, right] for left, right in upper]
    )
    map_rank = range_matrix.rank()
    augmented_rank = range_matrix.row_join(target_vector).rank()
    if (
        covector_basis.rank() != 33
        or selector_coordinates.rank() != 11
        or range_matrix.shape != (528, 363)
        or map_rank != 297
        or augmented_rank != 298
    ):
        raise QuarticTC2D4CurlCompanionRangeError(
            "pure-C23 companion range classification mismatch"
        )
    return {
        "declared_completion_class": {
            "name": "pure_curl_constraint_additions_preserving_the_exact_direction_1_V_slice",
            "fixed_C12_coefficients": True,
            "forced_C13_coefficients_zero": True,
            "remaining_free_constraints": [f"C_23^[{field}]" for field in range(11)],
            "raw_output_dimension": 55,
            "source_constraint_count": 11,
            "raw_parameter_dimension": 605,
            "zero_speed_output_dimension": 33,
            "effective_parameter_count": 363,
        },
        "exact_range_map": {
            "codomain_skew_dimension": 528,
            "matrix_shape": [528, 363],
            "right_selector_rank": selector_coordinates.rank(),
            "rank": map_rank,
            "target_augmented_rank": augmented_rank,
            "target_in_image": False,
            "selector_coordinates_sha256": _content_hash(
                _matrix_payload(selector_coordinates)
            ),
            "range_matrix_sha256": _content_hash(_matrix_payload(range_matrix)),
            "target_coordinates_sha256": _content_hash(
                _matrix_payload(target_coordinates)
            ),
        },
        "result": {
            "pure_curl_self_compatible_completion_exists": False,
            "additional_C23_constraints_can_cancel_companion_witness": False,
            "proof": (
                "The exact projected C23 range has rank 297; adjoining the rank-two "
                "companion target raises the rank to 298."
            ),
        },
    }


def _exact_companion_audit(curl_admission: Mapping[str, Any]) -> dict[str, Any]:
    reference = _reference_and_first_jet_packet()
    physical1 = reference["physical0"]
    energy1 = reference["energy0"]
    projectors1 = reference["projectors"]
    basis = _correction_basis()
    block1 = basis["block"]
    wedge1 = basis["wedge"]
    output = block1[:, 21]
    block2 = (-output * sp.eye(STATE_DIMENSION)[:, 54].T).applyfunc(sp.factor)
    swap = _axis_swap_1_2()
    physical2 = (swap * physical1 * swap.T).applyfunc(sp.factor)
    direct_physical2 = _direct_axis_2_physical()
    energy2 = (swap * energy1 * swap.T).applyfunc(sp.factor)
    projectors2 = {
        eigenvalue: (swap * projector * swap.T).applyfunc(sp.factor)
        for eigenvalue, projector in projectors1.items()
    }
    if physical2 != direct_physical2:
        raise QuarticTC2D4CurlCompanionRangeError(
            "rotated and direct axis-2 reference symbols disagree"
        )
    skew2 = (energy2 * block2 - block2.T * energy2).applyfunc(sp.factor)
    compressions = {
        eigenvalue: (projector.T * skew2 * projector).applyfunc(sp.factor)
        for eigenvalue, projector in projectors2.items()
    }
    target = compressions[sp.S.Zero]
    nonzero_records = []
    for eigenvalue, compression in compressions.items():
        nonzero_records.append(
            {
                "eigenvalue": str(eigenvalue),
                "rank": compression.rank(),
                "nonzero_entries": sum(value != 0 for value in compression),
                "zero": compression.is_zero_matrix,
                "sha256": _content_hash(_matrix_payload(compression)),
            }
        )
    rotated_wedge = (swap * wedge1 * swap.T).applyfunc(sp.factor)
    witness_span = sp.Matrix.hstack(
        sp.Matrix([target[row, column] for row in range(55) for column in range(55)]),
        sp.Matrix(
            [rotated_wedge[row, column] for row in range(55) for column in range(55)]
        ),
    )
    if (
        _content_hash(_matrix_payload(block1)) != EXPECTED_V_SHA256
        or _content_hash(_matrix_payload(block2)) != EXPECTED_COMPANION_BLOCK_SHA256
        or target.rank() != 2
        or sum(value != 0 for value in target) != 10
        or _content_hash(_matrix_payload(target))
        != EXPECTED_COMPANION_COMPRESSION_SHA256
        or any(
            not compression.is_zero_matrix
            for eigenvalue, compression in compressions.items()
            if eigenvalue != 0
        )
        or rotated_wedge.rank() != 2
        or _content_hash(_matrix_payload(rotated_wedge)) != EXPECTED_ROTATED_W_SHA256
        or witness_span.rank() != 2
    ):
        raise QuarticTC2D4CurlCompanionRangeError(
            "axis-2 companion equal-eigenspace audit mismatch"
        )

    range_audit = _reduced_exterior_range(projectors2[sp.S.Zero], target)
    candidate_rows = []
    source_rows = curl_admission.get("exact_admission", {}).get(
        "reference_D4_binding", {}
    ).get("candidate_specializations", [])
    for source in source_rows:
        eta = sp.sympify(source["eta_unique_tuning"])
        scaled = (eta * target).applyfunc(sp.factor)
        candidate_rows.append(
            {
                "candidate_id": source["candidate_id"],
                "a10": source["a10"],
                "eta": str(eta),
                "companion_correction_compression_rank": scaled.rank(),
                "companion_correction_compression_nonzero_entries": sum(
                    value != 0 for value in scaled
                ),
                "companion_correction_compression_sha256": _content_hash(
                    _matrix_payload(scaled)
                ),
                "companion_correction_alone_Sylvester_compatible": False,
                "required_base_D4_compression": f"-({eta})*C_companion",
                "full_base_D4_RHS_evaluated": False,
            }
        )
    if (
        len(candidate_rows) != EXPECTED_CANDIDATES
        or len(
            {
                row["companion_correction_compression_sha256"]
                for row in candidate_rows
            }
        )
        != 4
    ):
        raise QuarticTC2D4CurlCompanionRangeError(
            "candidate companion witness classification mismatch"
        )
    return {
        "axis_2_reference": {
            "axis_swap_state_sha256": _content_hash(_matrix_payload(swap)),
            "axis_swap_orthogonal": True,
            "rotated_P55_sha256": _content_hash(_matrix_payload(physical2)),
            "direct_axis_2_P55_matches_rotation": True,
            "rotated_K55_sha256": _content_hash(_matrix_payload(energy2)),
            "zero_projector_sha256": _content_hash(
                _matrix_payload(projectors2[sp.S.Zero])
            ),
            "spectrum": ["0", "1", "-1", "1/2", "-1/2", "1/3", "-1/3"],
        },
        "companion_block": {
            "definition": "B2=-K55(0)^(-1)*(e16+e28)*e54^T",
            "sha256": EXPECTED_COMPANION_BLOCK_SHA256,
            "rank": block2.rank(),
            "nonzero_entries": sum(value != 0 for value in block2),
            "energy_skew_sha256": _content_hash(_matrix_payload(skew2)),
            "energy_skew_rank": skew2.rank(),
            "energy_skew_nonzero_entries": sum(value != 0 for value in skew2),
        },
        "equal_eigenspace_audit": {
            "eigenspaces_checked": len(compressions),
            "records": nonzero_records,
            "nonzero_compression_count": sum(
                not compression.is_zero_matrix for compression in compressions.values()
            ),
            "sole_nonzero_eigenvalue": "0",
            "companion_compression_rank": target.rank(),
            "companion_compression_nonzero_entries": sum(
                value != 0 for value in target
            ),
            "companion_compression_sha256": EXPECTED_COMPANION_COMPRESSION_SHA256,
            "companion_compression_sparse": _sparse_matrix(target),
            "companion_block_alone_Sylvester_compatible": False,
        },
        "rotation_control": {
            "rotated_direction_1_W_sha256": EXPECTED_ROTATED_W_SHA256,
            "rotated_direction_1_W_rank": rotated_wedge.rank(),
            "companion_and_rotated_W_span_dimension": witness_span.rank(),
            "companion_is_rotated_W_multiple": False,
            "simple_rotation_of_direction_1_certificate_cancels_companion": False,
        },
        "pure_curl_completion_range": range_audit,
        "candidate_companion_witnesses": candidate_rows,
        "necessary_full_D4_condition": {
            "formula": (
                "R02^T*RHS_base_D4(e2;same_active_tensor_inputs)*R02="
                "-eta*C_companion"
            ),
            "candidate_conditions": EXPECTED_CANDIDATES,
            "base_D4_RHS_computed": False,
            "condition_verified": False,
            "condition_refuted": False,
        },
    }


def build_campaign(project_root: Path, config_path: Path) -> dict[str, Any]:
    root = project_root.resolve()
    config, _ = _load_file(config_path.resolve())
    if (
        config.get("schema_version") != CONFIG_SCHEMA
        or not _hash_matches(config)
        or config.get("global_claim_policy") != "fail_closed"
        or config.get("obligation_offset") != OBLIGATION_OFFSET
        or tuple(config.get("active_indices", ())) != ACTIVE_INDICES
        or config.get("expected_candidate_count") != EXPECTED_CANDIDATES
    ):
        raise QuarticTC2D4CurlCompanionRangeError(
            "curl-companion range config mismatch"
        )
    for key in ("campaign_source", "campaign_test", "first_order_reduction_source"):
        _check_raw_binding(root, config[key])
    curl_admission = _load_bound(root, config["curl_admission"])
    topology = _load_bound(root, config["topology_classification"])
    if (
        curl_admission.get("status")
        != "pass_exact_gauge_fixed_curl_constraint_admission_for_minimal_V"
        or curl_admission.get("exact_admission", {})
        .get("gauge_fixed_operator", {})
        .get("direction_2_companion_sha256")
        != EXPECTED_COMPANION_BLOCK_SHA256
        or topology.get("exact_classification", {})
        .get("explicit_TC2_selector_classification", {})
        .get("canonical_capable_indices")
        != [21, 44, 48, 51, 53]
    ):
        raise QuarticTC2D4CurlCompanionRangeError(
            "bound predecessor semantic contract mismatch"
        )
    exact = _exact_companion_audit(curl_admission)
    body = {
        "schema_version": SCHEMA,
        "status": "pass_exact_axis2_companion_obstruction_and_pure_curl_range_no_go",
        "config_sha256": config["content_sha256"],
        "source_bindings": {
            key: dict(config[key])
            for key in (
                "curl_admission",
                "topology_classification",
                "first_order_reduction_source",
                "campaign_source",
                "campaign_test",
            )
        },
        "selector_binding": {
            "obligation_offset": OBLIGATION_OFFSET,
            "active_indices": list(ACTIVE_INDICES),
            "reference_direction": "e2",
            "companion_input_selector": 54,
            "companion_block_sha256": EXPECTED_COMPANION_BLOCK_SHA256,
        },
        "exact_companion_audit": exact,
        "counts": {
            "reference_eigenspaces_checked": 7,
            "nonzero_equal_eigenspace_compressions": 1,
            "companion_compression_rank": 2,
            "companion_compression_nonzero_entries": 10,
            "pure_C23_raw_parameters": 605,
            "pure_C23_effective_parameters": 363,
            "pure_C23_range_rank": 297,
            "target_augmented_rank": 298,
            "candidate_companion_witnesses": 12,
            "distinct_scaled_witnesses": 4,
            "negative_controls": 6,
            "inferred_global_passes": 0,
        },
        "negative_controls": {
            "declare_companion_block_sylvester_compatible": {
                "zero_eigenspace_compression_rank": 2,
                "rejected": True,
            },
            "cancel_with_additional_C23_curl_constraints": {
                "range_rank": 297,
                "augmented_rank": 298,
                "rejected": True,
            },
            "reuse_rotated_direction_1_W_line": {
                "witness_span_dimension": 2,
                "rejected": True,
            },
            "omit_required_direction_2_companion": {
                "physical_gradient_subspace_equivalence_lost": True,
                "rejected": True,
            },
            "infer_full_axis2_D4_obstruction_without_base_RHS": {
                "base_RHS_evaluated": False,
                "rejected": True,
            },
            "promote_companion_audit_to_global_closure": {
                "remaining_D4_selector_closed": False,
                "tube_theorem_proved": False,
                "rejected": True,
            },
        },
        "claims": {
            "axis2_companion_all_eigenspaces_audited": True,
            "companion_block_alone_Sylvester_compatible": False,
            "pure_curl_self_compatible_completion_ruled_out": True,
            "full_axis2_base_D4_RHS_evaluated": False,
            "full_axis2_D4_compatibility_proved": False,
            "full_axis2_D4_obstruction_proved": False,
            "spatially_covariant_tensor_completion_proved": False,
            "all_spatial_direction_compatibility_proved": False,
            "corrected_candidate_family_registered": False,
            "remaining_D4_selector_closed": False,
            "full_tube_Sylvester_identity": False,
            "CK1_closed": False,
            "CK3_closed": False,
            "TC2_closed": False,
            "B7_closed": False,
            "global_H7_closed": False,
            "lifespan_proved": False,
        },
        "next_gate": (
            "Compute the complete polarized base D4 Sylvester RHS at the e2 reference for the "
            "same active tensor-component inputs and test its zero-speed compression against "
            "the exact required value -eta*C_companion. The current result rules out repairing "
            "the companion with further pure curl constraints but does not infer the missing "
            "base-RHS value."
        ),
        "scope": (
            "Exact all-seven-eigenspace audit of the required axis-2 companion block and an "
            "exact range no-go for every additional pure C23 curl-constraint term that preserves "
            "the admitted direction-1 V slice. The companion has a sole rank-two zero-speed "
            "compression outside that 297-dimensional range. The full axis-2 base D4 RHS is "
            "not computed, so neither full axis-2 obstruction nor compatibility, spatial "
            "covariance, remaining D4 closure, tube closure, CK1, CK3, TC2, B7, global-H7, or "
            "lifespan is claimed."
        ),
        "errors": [],
    }
    return _with_hash(body)


def validate_campaign(document: Mapping[str, Any]) -> None:
    if document.get("schema_version") != SCHEMA or not _hash_matches(dict(document)):
        raise QuarticTC2D4CurlCompanionRangeError(
            "curl-companion range content identity mismatch"
        )
    exact = document.get("exact_companion_audit", {})
    eigenspaces = exact.get("equal_eigenspace_audit", {})
    range_audit = exact.get("pure_curl_completion_range", {})
    range_map = range_audit.get("exact_range_map", {})
    condition = exact.get("necessary_full_D4_condition", {})
    expected_claims = {
        "axis2_companion_all_eigenspaces_audited": True,
        "companion_block_alone_Sylvester_compatible": False,
        "pure_curl_self_compatible_completion_ruled_out": True,
        "full_axis2_base_D4_RHS_evaluated": False,
        "full_axis2_D4_compatibility_proved": False,
        "full_axis2_D4_obstruction_proved": False,
        "spatially_covariant_tensor_completion_proved": False,
        "all_spatial_direction_compatibility_proved": False,
        "corrected_candidate_family_registered": False,
        "remaining_D4_selector_closed": False,
        "full_tube_Sylvester_identity": False,
        "CK1_closed": False,
        "CK3_closed": False,
        "TC2_closed": False,
        "B7_closed": False,
        "global_H7_closed": False,
        "lifespan_proved": False,
    }
    expected_counts = {
        "reference_eigenspaces_checked": 7,
        "nonzero_equal_eigenspace_compressions": 1,
        "companion_compression_rank": 2,
        "companion_compression_nonzero_entries": 10,
        "pure_C23_raw_parameters": 605,
        "pure_C23_effective_parameters": 363,
        "pure_C23_range_rank": 297,
        "target_augmented_rank": 298,
        "candidate_companion_witnesses": 12,
        "distinct_scaled_witnesses": 4,
        "negative_controls": 6,
        "inferred_global_passes": 0,
    }
    if (
        document.get("status")
        != "pass_exact_axis2_companion_obstruction_and_pure_curl_range_no_go"
        or document.get("claims") != expected_claims
        or document.get("counts") != expected_counts
        or exact.get("axis_2_reference", {}).get("direct_axis_2_P55_matches_rotation")
        is not True
        or exact.get("companion_block", {}).get("sha256")
        != EXPECTED_COMPANION_BLOCK_SHA256
        or eigenspaces.get("eigenspaces_checked") != 7
        or eigenspaces.get("nonzero_compression_count") != 1
        or eigenspaces.get("sole_nonzero_eigenvalue") != "0"
        or eigenspaces.get("companion_compression_rank") != 2
        or eigenspaces.get("companion_compression_nonzero_entries") != 10
        or eigenspaces.get("companion_compression_sha256")
        != EXPECTED_COMPANION_COMPRESSION_SHA256
        or eigenspaces.get("companion_block_alone_Sylvester_compatible") is not False
        or range_map.get("rank") != 297
        or range_map.get("target_augmented_rank") != 298
        or range_map.get("target_in_image") is not False
        or range_audit.get("result", {}).get(
            "pure_curl_self_compatible_completion_exists"
        )
        is not False
        or len(exact.get("candidate_companion_witnesses", [])) != EXPECTED_CANDIDATES
        or condition.get("base_D4_RHS_computed") is not False
        or condition.get("condition_verified") is not False
        or condition.get("condition_refuted") is not False
        or len(document.get("negative_controls", {})) != 6
        or any(
            control.get("rejected") is not True
            for control in document.get("negative_controls", {}).values()
        )
    ):
        raise QuarticTC2D4CurlCompanionRangeError(
            "curl-companion range exact/fail-closed contract mismatch"
        )


def run_campaign(
    project_root: Path, config_path: Path, output_path: Path
) -> dict[str, Any]:
    artifact = build_campaign(project_root, config_path)
    validate_campaign(artifact)
    _atomic_write(output_path.resolve(), _json_bytes(artifact))
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit the D4 curl companion and pure-curl completion range."
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    artifact = run_campaign(args.project_root, args.config, args.output)
    print(
        json.dumps(
            {
                "status": artifact["status"],
                "content_sha256": artifact["content_sha256"],
                "companion_rank": artifact["counts"]["companion_compression_rank"],
                "range_rank": artifact["counts"]["pure_C23_range_rank"],
                "augmented_rank": artifact["counts"]["target_augmented_rank"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
