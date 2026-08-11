from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import sympy as sp

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

SCHEMA = "sigma-quartic-tc2-d4-topology-changing-origin-classification-1.0"
CONFIG_SCHEMA = (
    "sigma-quartic-tc2-d4-topology-changing-origin-classification-config-1.0"
)
OBLIGATION_OFFSET = 244
ACTIVE_INDICES = (0, 2, 3, 9)
EXPECTED_V_SHA256 = "a8a6cb0588ebae512db867990f937a3a9e5a9a38bf90be807fc62a8eb928f9c0"
EXPECTED_W_SHA256 = "e44c769b1eaf44c6e0ffc411007d98f9de24c6e8a20bac112d9a0a062e913500"
EXPECTED_CANONICAL_CAPABLE = (21, 44, 48, 51, 53)
EXPECTED_CANONICAL_KERNEL = (
    33,
    34,
    35,
    36,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    45,
    46,
    47,
    52,
    54,
)


class QuarticTC2D4TopologyChangingOriginClassificationError(ValueError):
    """Raised when the topology-changing origin classification is inconsistent."""


def _load_bound(root: Path, binding: Mapping[str, Any]) -> dict[str, Any]:
    path = (root / str(binding["path"])).resolve()
    if root.resolve() not in path.parents:
        raise QuarticTC2D4TopologyChangingOriginClassificationError(
            "bound input escaped project root"
        )
    value, data = _load_file(path)
    if (
        _file_sha256(data) != binding.get("file_sha256")
        or value.get("content_sha256") != binding.get("content_sha256")
        or not _hash_matches(value)
    ):
        raise QuarticTC2D4TopologyChangingOriginClassificationError(
            f"bound input mismatch: {binding.get('path')}"
        )
    return value


def _check_raw_binding(root: Path, binding: Mapping[str, Any]) -> None:
    path = (root / str(binding["path"])).resolve()
    if root.resolve() not in path.parents or not path.is_file():
        raise QuarticTC2D4TopologyChangingOriginClassificationError(
            "raw binding escaped project root or is absent"
        )
    if _file_sha256(path.read_bytes()) != binding.get("file_sha256"):
        raise QuarticTC2D4TopologyChangingOriginClassificationError(
            f"raw binding mismatch: {binding.get('path')}"
        )


def _state_label(index: int) -> str:
    if 0 <= index < 11:
        return f"q[{index}]"
    if 11 <= index < 22:
        return f"w2[{index - 11}]"
    if 22 <= index < 33:
        return f"w3[{index - 22}]"
    if 33 <= index < 44:
        return f"v[{index - 33}]"
    if 44 <= index < 55:
        return f"w1[{index - 44}]"
    raise QuarticTC2D4TopologyChangingOriginClassificationError(
        "state index outside canonical ordering"
    )


def _sparse_vector(vector: sp.MatrixBase) -> list[dict[str, Any]]:
    if vector.cols != 1:
        raise QuarticTC2D4TopologyChangingOriginClassificationError(
            "expected a column vector"
        )
    return [
        {"index": index, "value": str(sp.factor(vector[index]))}
        for index in range(vector.rows)
        if vector[index] != 0
    ]


def _direct_action_no_go(
    physical0: sp.Matrix,
    energy0: sp.Matrix,
    zero_projector: sp.Matrix,
    target: sp.Matrix,
) -> dict[str, Any]:
    velocity_embedding = sp.zeros(STATE_DIMENSION, 11)
    velocity_embedding[33:44, :] = sp.eye(11)
    pairing = (zero_projector.T * energy0 * velocity_embedding).applyfunc(sp.factor)
    right_zero = (physical0 * zero_projector).applyfunc(sp.factor)
    left_zero = (zero_projector.T * physical0.T).applyfunc(sp.factor)
    if (
        not pairing.is_zero_matrix
        or not right_zero.is_zero_matrix
        or not left_zero.is_zero_matrix
        or target.is_zero_matrix
    ):
        raise QuarticTC2D4TopologyChangingOriginClassificationError(
            "direct action-lift annihilation identity mismatch"
        )
    return {
        "declared_class": {
            "name": "direct_D4_second_order_action_principal_deformations",
            "second_order_field_dimension": 11,
            "canonical_first_order_state_dimension": 55,
            "orders_0_through_3_fixed": True,
            "allowed_second_order_changes": ["deltaA", "deltaB^i", "deltaC^ij"],
            "allowed_coefficients": "arbitrary exact D4 coefficient tensors",
            "canonical_lift_formula": "deltaP55=E_v*Z for arbitrary Z in Q^(11x55)",
            "broad_deltaP_domain_dimension": 605,
            "symmetric_deltaK_domain_dimension": 1540,
            "joint_domain_dimension": 2145,
            "covers_mass_variation": True,
            "covers_mixed_Hessian_Cij_variation": True,
            "covers_arbitrary_direct_symmetrizer_variation": True,
            "excludes_lower_jet_coupled_changes": True,
            "excludes_changes_to_definition_or_curl_equations": True,
            "excludes_explicit_nonprincipal_TC2_constraint_rows": True,
        },
        "canonical_lift_support": {
            "output_rows": list(range(33, 44)),
            "output_labels": [_state_label(index) for index in range(33, 44)],
            "velocity_embedding_sha256": _content_hash(
                _matrix_payload(velocity_embedding)
            ),
            "R0T_K0_Ev_zero": True,
            "R0T_K0_Ev_nonzero_entries": 0,
            "R0T_K0_Ev_sha256": _content_hash(_matrix_payload(pairing)),
        },
        "deltaP_cokernel_map": {
            "formula": (
                "R0^T(K0*E_v*Z-Z^T*E_v^T*K0)R0=0 because R0^T*K0*E_v=0"
            ),
            "domain_dimension": 605,
            "rank": 0,
            "target_W_in_image": False,
            "target_augmented_rank": 1,
        },
        "deltaK_cokernel_map": {
            "formula": (
                "R0^T(deltaK*P0-P0^T*deltaK)R0=0 because P0*R0=0"
            ),
            "symmetric_domain_dimension": 1540,
            "rank": 0,
            "P0_R0_zero": True,
            "P0_R0_sha256": _content_hash(_matrix_payload(right_zero)),
            "R0T_P0T_zero": True,
            "R0T_P0T_sha256": _content_hash(_matrix_payload(left_zero)),
            "target_W_in_image": False,
        },
        "joint_result": {
            "joint_domain_dimension": 2145,
            "joint_map_rank": 0,
            "target_W_rank": target.rank(),
            "target_W_nonzero_entries": sum(value != 0 for value in target),
            "target_W_sha256": EXPECTED_W_SHA256,
            "target_W_in_image": False,
            "direct_action_principal_origin_ruled_out": True,
        },
    }


def _selector_classification(
    energy0: sp.Matrix, zero_projector: sp.Matrix, target: sp.Matrix
) -> dict[str, Any]:
    identity = sp.eye(STATE_DIMENSION)
    left = identity[:, 16] + identity[:, 28]
    right = identity[:, 21]
    target_plane = sp.Matrix.hstack(left, right)
    if (
        target != left * right.T - right * left.T
        or (zero_projector.T * left) != left
        or (zero_projector.T * right) != right
        or target_plane.rank() != 2
    ):
        raise QuarticTC2D4TopologyChangingOriginClassificationError(
            "target plane replay mismatch"
        )

    records: list[dict[str, Any]] = []
    capable_blocks: list[dict[str, Any]] = []
    for index in range(STATE_DIMENSION):
        selector = identity[:, index]
        projection = (zero_projector.T * selector).applyfunc(sp.factor)
        projection_zero = projection.is_zero_matrix
        in_target_plane = (
            not projection_zero and target_plane.row_join(projection).rank() == 2
        )
        map_rank = 0 if projection_zero else 32
        augmented_rank = map_rank if in_target_plane else map_rank + 1
        coefficients: list[str] | None = None
        block_sha256: str | None = None
        if in_target_plane:
            coefficient_left = sp.factor(projection[16])
            coefficient_right = sp.factor(projection[21])
            if projection != coefficient_left * left + coefficient_right * right:
                raise QuarticTC2D4TopologyChangingOriginClassificationError(
                    "target-plane selector coordinates mismatch"
                )
            coefficients = [str(coefficient_left), str(coefficient_right)]
            if coefficient_right != 0:
                desired_output = left / coefficient_right
            elif coefficient_left != 0:
                desired_output = -right / coefficient_left
            else:
                raise QuarticTC2D4TopologyChangingOriginClassificationError(
                    "capable selector has zero target-plane coordinates"
                )
            output = energy0.inv() * desired_output
            block = (output * selector.T).applyfunc(sp.factor)
            compression = (
                zero_projector.T
                * (energy0 * block - block.T * energy0)
                * zero_projector
            ).applyfunc(sp.factor)
            if block.rank() != 1 or compression != target:
                raise QuarticTC2D4TopologyChangingOriginClassificationError(
                    "canonical selector constructive sufficiency failed"
                )
            block_sha256 = _content_hash(_matrix_payload(block))
            capable_blocks.append(
                {
                    "selector_index": index,
                    "selector_label": _state_label(index),
                    "projection_target_plane_coordinates": coefficients,
                    "rank_one_block_sha256": block_sha256,
                    "rank_one_block_rank": 1,
                    "projected_energy_skew_equals_W": True,
                    "full_equal_eigenspace_compatibility_checked": False,
                    "covariant_origin_proved": False,
                }
            )
        records.append(
            {
                "selector_index": index,
                "selector_label": _state_label(index),
                "projected_selector_sparse": _sparse_vector(projection),
                "projected_selector_sha256": _content_hash(
                    _matrix_payload(projection)
                ),
                "projection_zero": projection_zero,
                "induced_map_rank": map_rank,
                "target_W_in_image": in_target_plane,
                "target_augmented_rank": augmented_rank,
                **(
                    {"target_plane_coordinates": coefficients}
                    if coefficients is not None
                    else {}
                ),
            }
        )

    kernel_indices = tuple(
        record["selector_index"] for record in records if record["projection_zero"]
    )
    capable_indices = tuple(
        record["selector_index"] for record in records if record["target_W_in_image"]
    )
    if (
        zero_projector.T.rank() != 33
        or len(zero_projector.T.nullspace()) != 22
        or kernel_indices != EXPECTED_CANONICAL_KERNEL
        or capable_indices != EXPECTED_CANONICAL_CAPABLE
        or len(records) != STATE_DIMENSION
        or len(capable_blocks) != len(EXPECTED_CANONICAL_CAPABLE)
    ):
        raise QuarticTC2D4TopologyChangingOriginClassificationError(
            "canonical selector classification mismatch"
        )

    return {
        "necessary_and_sufficient_condition": {
            "selector": "h in Q^55",
            "condition": (
                "R0^T*h is a nonzero member of span{e16+e28,e21}"
            ),
            "proof": (
                "The image is {a wedge (R0^T h): a in im(R0^T K0)}. Since K0 is "
                "invertible, im(R0^T K0)=im(R0^T), dimension 33. A nonzero selector "
                "projection gives rank 32. The rank-two W=L wedge Q belongs to this image "
                "iff the selector projection lies in span{L,Q}."
            ),
            "R0T_rank": 33,
            "R0T_kernel_dimension": 22,
            "target_plane_dimension": 2,
            "capable_selector_preimage_dimension": 24,
            "capable_nonzero_projection_condition": "(a,b)!=(0,0)",
            "general_capable_selector": (
                "h=a*(e16+e28)+b*e21+k, k in ker(R0^T), (a,b)!=(0,0)"
            ),
        },
        "canonical_selector_records": records,
        "canonical_counts": {
            "selectors_checked": 55,
            "zero_projection_selectors": len(kernel_indices),
            "nonzero_projection_incapable_selectors": (
                STATE_DIMENSION - len(kernel_indices) - len(capable_indices)
            ),
            "cokernel_capable_selectors": len(capable_indices),
        },
        "canonical_kernel_indices": list(kernel_indices),
        "canonical_capable_indices": list(capable_indices),
        "canonical_capable_labels": [
            _state_label(index) for index in capable_indices
        ],
        "constructive_rank_one_blocks": capable_blocks,
        "registered_selector_control": {
            "selector_index": 54,
            "selector_label": _state_label(54),
            "projection_zero": True,
            "target_W_in_image": False,
            "rejected": True,
        },
        "minimal_escape_control": {
            "selector_index": 21,
            "selector_label": _state_label(21),
            "projection_target_plane_coordinates": ["0", "1"],
            "V_sha256": EXPECTED_V_SHA256,
            "target_W_in_image": True,
            "accepted": True,
        },
    }


def _exact_classification() -> dict[str, Any]:
    reference = _reference_and_first_jet_packet()
    physical0 = reference["physical0"]
    energy0 = reference["energy0"]
    zero_projector = reference["projectors"][sp.S.Zero]
    escape = _correction_basis()
    target = escape["wedge"]
    if (
        _content_hash(_matrix_payload(escape["block"])) != EXPECTED_V_SHA256
        or _content_hash(_matrix_payload(target)) != EXPECTED_W_SHA256
        or (zero_projector.T * target * zero_projector).applyfunc(sp.factor)
        != target
    ):
        raise QuarticTC2D4TopologyChangingOriginClassificationError(
            "minimal escape predecessor matrix replay mismatch"
        )
    return {
        "direct_action_origin_no_go": _direct_action_no_go(
            physical0, energy0, zero_projector, target
        ),
        "explicit_TC2_selector_classification": _selector_classification(
            energy0, zero_projector, target
        ),
        "exact_first_blocker": {
            "statement": (
                "A successful origin must leave the direct second-order action-principal "
                "class: it must either modify a definition/curl constraint row, introduce "
                "an explicit nonprincipal TC2 constraint operator with a capable selector, "
                "or use coupled lower-jet changes. No covariant derivation or constraint "
                "propagation proof for any such mechanism is currently bound."
            ),
            "smallest_canonical_selector_to_test": {
                "index": 21,
                "label": "w2[10]",
                "required_spatial_Hessian_companion": (
                    "A direction-1 input w2 term from a covariant second-order Hessian "
                    "coefficient C12 must be accompanied by the direction-2 input w1 term "
                    "because C12=C21; the direct canonical lift of that pair is already "
                    "covered by, and obstructed in, the output-v no-go."
                ),
            },
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
        or config.get("expected_canonical_selector_count") != STATE_DIMENSION
    ):
        raise QuarticTC2D4TopologyChangingOriginClassificationError(
            "topology-changing origin config mismatch"
        )
    for key in ("campaign_source", "campaign_test", "first_order_reduction_source"):
        _check_raw_binding(root, config[key])
    minimal = _load_bound(root, config["minimal_escape"])
    registered_no_go = _load_bound(root, config["registered_operator_no_go"])
    first_order = _load_bound(root, config["first_order_reduction"])
    if (
        minimal.get("exact_escape", {}).get("correction_ansatz", {}).get("V_sha256")
        != EXPECTED_V_SHA256
        or registered_no_go.get("claims", {}).get(
            "registered_support_preserving_gauge_deformation_ruled_out"
        )
        is not True
        or registered_no_go.get("exact_no_go", {})
        .get("induced_cokernel_map", {})
        .get("rank")
        != 0
        or first_order.get("status")
        != "pass_all_12_exact_55_variable_principal_first_order_reductions"
        or first_order.get("counts", {}).get("exact_55_variable_reductions_passed")
        != 12
    ):
        raise QuarticTC2D4TopologyChangingOriginClassificationError(
            "bound predecessor semantic contract mismatch"
        )
    exact = _exact_classification()
    body = {
        "schema_version": SCHEMA,
        "status": (
            "pass_exact_direct_action_origin_no_go_and_complete_selector_classification"
        ),
        "config_sha256": config["content_sha256"],
        "source_bindings": {
            key: dict(config[key])
            for key in (
                "minimal_escape",
                "registered_operator_no_go",
                "first_order_reduction",
                "first_order_reduction_source",
                "campaign_source",
                "campaign_test",
            )
        },
        "selector_binding": {
            "obligation_offset": OBLIGATION_OFFSET,
            "active_indices": list(ACTIVE_INDICES),
            "V_sha256": EXPECTED_V_SHA256,
            "W_sha256": EXPECTED_W_SHA256,
        },
        "exact_classification": exact,
        "counts": {
            "direct_deltaP_domain_dimension": 605,
            "direct_symmetric_deltaK_domain_dimension": 1540,
            "direct_joint_domain_dimension": 2145,
            "direct_joint_cokernel_map_rank": 0,
            "canonical_selectors_checked": 55,
            "canonical_cokernel_capable_selectors": 5,
            "canonical_kernel_selectors": 16,
            "canonical_nonzero_incapable_selectors": 34,
            "constructive_rank_one_cokernel_blocks": 5,
            "negative_controls": 6,
            "inferred_global_passes": 0,
        },
        "negative_controls": {
            "direct_C12_cross_Hessian_pair": {
                "covariant_pairing_C12_equals_C21_required": True,
                "canonical_output_rows": list(range(33, 44)),
                "joint_direct_map_rank": 0,
                "target_in_image": False,
                "rejected": True,
            },
            "direct_symmetrizer_deltaK_only": {
                "equal_zero_eigenspace_map_rank": 0,
                "target_in_image": False,
                "rejected": True,
            },
            "registered_e54_selector": {
                "R0T_selector_zero": True,
                "target_in_image": False,
                "rejected": True,
            },
            "infer_full_compatibility_from_zero_block": {
                "five_rank_one_zero_block_constructions": True,
                "other_equal_eigenspaces_checked": False,
                "rejected": True,
            },
            "infer_covariant_origin_from_capable_selector": {
                "selector_condition_only_necessary_for_origin": True,
                "constraint_propagation_proved": False,
                "rejected": True,
            },
            "promote_local_classification_to_global_closure": {
                "remaining_D4_selector_closed": False,
                "tube_theorem_proved": False,
                "rejected": True,
            },
        },
        "claims": {
            "direct_second_order_action_principal_origin_ruled_out": True,
            "all_55_canonical_TC2_input_selectors_classified": True,
            "canonical_cokernel_capable_selector_count": 5,
            "explicit_constraint_row_covariant_origin_constructed": False,
            "lower_jet_coupled_action_origin_ruled_out": False,
            "arbitrary_covariant_operator_origin_ruled_out": False,
            "constraint_propagation_for_topology_change_proved": False,
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
        "next_gate": exact["exact_first_blocker"]["statement"],
        "scope": (
            "Exact reference-point theorem for direct D4 deformations of the canonical "
            "11-field second-order action/principal operator, plus a complete classification "
            "of canonical and general input covectors capable of reaching the obligation-244 "
            "zero-speed witness through an explicit TC2 block. Coupled lower-jet action "
            "changes, new constraint-row operators, other equal eigenspaces, the remaining "
            "D4 selector, tube closure, CK1, CK3, TC2, B7, global-H7, and lifespan remain open."
        ),
        "errors": [],
    }
    return _with_hash(body)


def validate_campaign(document: Mapping[str, Any]) -> None:
    if document.get("schema_version") != SCHEMA or not _hash_matches(dict(document)):
        raise QuarticTC2D4TopologyChangingOriginClassificationError(
            "topology-changing origin content identity mismatch"
        )
    exact = document.get("exact_classification", {})
    direct = exact.get("direct_action_origin_no_go", {})
    selector = exact.get("explicit_TC2_selector_classification", {})
    expected_claims = {
        "direct_second_order_action_principal_origin_ruled_out": True,
        "all_55_canonical_TC2_input_selectors_classified": True,
        "canonical_cokernel_capable_selector_count": 5,
        "explicit_constraint_row_covariant_origin_constructed": False,
        "lower_jet_coupled_action_origin_ruled_out": False,
        "arbitrary_covariant_operator_origin_ruled_out": False,
        "constraint_propagation_for_topology_change_proved": False,
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
        "direct_deltaP_domain_dimension": 605,
        "direct_symmetric_deltaK_domain_dimension": 1540,
        "direct_joint_domain_dimension": 2145,
        "direct_joint_cokernel_map_rank": 0,
        "canonical_selectors_checked": 55,
        "canonical_cokernel_capable_selectors": 5,
        "canonical_kernel_selectors": 16,
        "canonical_nonzero_incapable_selectors": 34,
        "constructive_rank_one_cokernel_blocks": 5,
        "negative_controls": 6,
        "inferred_global_passes": 0,
    }
    direct_joint = direct.get("joint_result", {})
    selector_counts = selector.get("canonical_counts", {})
    if (
        document.get("status")
        != "pass_exact_direct_action_origin_no_go_and_complete_selector_classification"
        or document.get("claims") != expected_claims
        or document.get("counts") != expected_counts
        or direct.get("canonical_lift_support", {}).get("R0T_K0_Ev_zero") is not True
        or direct.get("deltaP_cokernel_map", {}).get("rank") != 0
        or direct.get("deltaK_cokernel_map", {}).get("rank") != 0
        or direct_joint.get("joint_domain_dimension") != 2145
        or direct_joint.get("joint_map_rank") != 0
        or direct_joint.get("target_W_in_image") is not False
        or selector_counts
        != {
            "selectors_checked": 55,
            "zero_projection_selectors": 16,
            "nonzero_projection_incapable_selectors": 34,
            "cokernel_capable_selectors": 5,
        }
        or selector.get("canonical_kernel_indices") != list(EXPECTED_CANONICAL_KERNEL)
        or selector.get("canonical_capable_indices") != list(EXPECTED_CANONICAL_CAPABLE)
        or len(selector.get("canonical_selector_records", [])) != 55
        or len(selector.get("constructive_rank_one_blocks", [])) != 5
        or any(
            row.get("projected_energy_skew_equals_W") is not True
            or row.get("full_equal_eigenspace_compatibility_checked") is not False
            or row.get("covariant_origin_proved") is not False
            for row in selector.get("constructive_rank_one_blocks", [])
        )
        or selector.get("registered_selector_control", {}).get("rejected") is not True
        or selector.get("minimal_escape_control", {}).get("accepted") is not True
        or len(document.get("negative_controls", {})) != 6
        or any(
            control.get("rejected") is not True
            for control in document.get("negative_controls", {}).values()
        )
    ):
        raise QuarticTC2D4TopologyChangingOriginClassificationError(
            "topology-changing origin exact/fail-closed contract mismatch"
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
        description="Classify topology-changing origins for the D4 TC2 escape."
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
                "direct_map_rank": artifact["counts"][
                    "direct_joint_cokernel_map_rank"
                ],
                "capable_selectors": artifact["counts"][
                    "canonical_cokernel_capable_selectors"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
