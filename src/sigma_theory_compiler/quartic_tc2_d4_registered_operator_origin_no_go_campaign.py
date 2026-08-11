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

SCHEMA = "sigma-quartic-tc2-d4-registered-operator-origin-no-go-campaign-1.0"
CONFIG_SCHEMA = "sigma-quartic-tc2-d4-registered-operator-origin-no-go-config-1.0"
ACTIVE_INDICES = (0, 2, 3, 9)
OBLIGATION_OFFSET = 244
EXPECTED_ESCAPE_FILE_SHA256 = (
    "abb4ba1e6e0abe89c5fd8ad2166774377531b59556c1eca4732e89b8dada76e5"
)
EXPECTED_ESCAPE_CONTENT_SHA256 = (
    "8b15318e04bfe89f6dd321b04634878b57639fcd48503cade30739db61e033d4"
)
EXPECTED_W_SHA256 = "e44c769b1eaf44c6e0ffc411007d98f9de24c6e8a20bac112d9a0a062e913500"
EXPECTED_V_SHA256 = "a8a6cb0588ebae512db867990f937a3a9e5a9a38bf90be807fc62a8eb928f9c0"


class QuarticTC2D4RegisteredOperatorOriginNoGoError(ValueError):
    """Raised when the registered-operator no-go certificate is inconsistent."""


def _load_bound(root: Path, binding: Mapping[str, Any]) -> dict[str, Any]:
    path = (root / str(binding["path"])).resolve()
    if root.resolve() not in path.parents:
        raise QuarticTC2D4RegisteredOperatorOriginNoGoError(
            "bound input escaped project root"
        )
    value, data = _load_file(path)
    if (
        _file_sha256(data) != binding.get("file_sha256")
        or value.get("content_sha256") != binding.get("content_sha256")
        or not _hash_matches(value)
    ):
        raise QuarticTC2D4RegisteredOperatorOriginNoGoError(
            f"bound input mismatch: {binding.get('path')}"
        )
    return value


def _check_raw_binding(root: Path, binding: Mapping[str, Any]) -> None:
    path = (root / str(binding["path"])).resolve()
    if root.resolve() not in path.parents or not path.is_file():
        raise QuarticTC2D4RegisteredOperatorOriginNoGoError(
            "raw binding escaped project root or is absent"
        )
    if _file_sha256(path.read_bytes()) != binding.get("file_sha256"):
        raise QuarticTC2D4RegisteredOperatorOriginNoGoError(
            f"raw binding mismatch: {binding.get('path')}"
        )


def _nonzero_columns(matrix: sp.MatrixBase) -> list[int]:
    return [
        column
        for column in range(matrix.cols)
        if any(matrix[row, column] != 0 for row in range(matrix.rows))
    ]


def _registered_blocks(physical0: sp.Matrix) -> dict[str, sp.Matrix]:
    q = sp.zeros(11)
    q[0, 10] = 2
    q[4, 10] = -8
    q[10, 7] = 2
    q[10, 9] = 2
    high = sp.zeros(STATE_DIMENSION, 1)
    high[54] = 1
    p_eq = sp.zeros(STATE_DIMENSION, 11)
    p_eq[44:55, :] = q
    embedded_q = sp.zeros(STATE_DIMENSION, 11)
    embedded_q[33:44, :] = q
    blocks = {
        f"reference_column_{column}": p_eq[:, column] * high.T
        for column in (7, 9, 10)
    }
    blocks["variable_column_10"] = physical0 * embedded_q[:, 10] * high.T
    return blocks


def _exact_certificate() -> dict[str, Any]:
    reference = _reference_and_first_jet_packet()
    physical0 = reference["physical0"]
    energy0 = reference["energy0"]
    zero_projector = reference["projectors"][sp.S.Zero]
    escape = _correction_basis()
    block_v, target = escape["block"], escape["wedge"]
    high = sp.zeros(STATE_DIMENSION, 1)
    high[54] = 1
    stationary = sp.zeros(STATE_DIMENSION, 1)
    stationary[21] = 1
    projected_high = (high.T * zero_projector).applyfunc(sp.factor)
    projected_stationary = (stationary.T * zero_projector).applyfunc(sp.factor)
    target_compression = (
        zero_projector.T * target * zero_projector
    ).applyfunc(sp.factor)

    # This is the broadest support-preserving class: the output vector u is
    # arbitrary, while the registered first-order TC2 input covector e_54^T is
    # retained.  The factorization below proves the entire 55-dimensional map is
    # zero without sampling its coefficients.
    broad_compressions: list[sp.Matrix] = []
    for row in range(STATE_DIMENSION):
        output = sp.zeros(STATE_DIMENSION, 1)
        output[row] = 1
        candidate = output * high.T
        broad_compressions.append(
            (
                zero_projector.T
                * (energy0 * candidate - candidate.T * energy0)
                * zero_projector
            ).applyfunc(sp.factor)
        )
    broad_map_zero = all(matrix.is_zero_matrix for matrix in broad_compressions)
    target_coordinates = [
        target_compression[row, column]
        for row in range(STATE_DIMENSION)
        for column in range(STATE_DIMENSION)
        if target_compression[row, column] != 0
    ]
    broad_rank = 0 if broad_map_zero else -1
    augmented_rank = 1 if broad_map_zero and target_coordinates else -1

    registered_rows: list[dict[str, Any]] = []
    for name, block in _registered_blocks(physical0).items():
        compression = (
            zero_projector.T
            * (energy0 * block - block.T * energy0)
            * zero_projector
        ).applyfunc(sp.factor)
        registered_rows.append(
            {
                "name": name,
                "block_sha256": _content_hash(_matrix_payload(block)),
                "block_rank": block.rank(),
                "right_support_columns": _nonzero_columns(block),
                "zero_eigenspace_compression_zero": compression.is_zero_matrix,
                "zero_eigenspace_compression_sha256": _content_hash(
                    _matrix_payload(compression)
                ),
            }
        )

    induced_v = (energy0 * block_v - block_v.T * energy0).applyfunc(sp.factor)
    if (
        zero_projector.rank() != 33
        or not projected_high.is_zero_matrix
        or projected_stationary.is_zero_matrix
        or target_compression != target
        or target.rank() != 2
        or sum(value != 0 for value in target) != 4
        or not broad_map_zero
        or broad_rank != 0
        or augmented_rank != 1
        or any(
            row["right_support_columns"] != [54]
            or row["zero_eigenspace_compression_zero"] is not True
            for row in registered_rows
        )
        or _nonzero_columns(block_v) != [21]
        or induced_v != target
        or _content_hash(_matrix_payload(block_v)) != EXPECTED_V_SHA256
        or _content_hash(_matrix_payload(target)) != EXPECTED_W_SHA256
    ):
        raise QuarticTC2D4RegisteredOperatorOriginNoGoError(
            "exact registered-operator cokernel audit mismatch"
        )

    return {
        "declared_operator_class": {
            "name": "registered_support_preserving_quartic_TC2_lifts",
            "state_dimension": STATE_DIMENSION,
            "ordered_state": "z=(q,w2,w3), y=(v,w1)",
            "general_block": "B_u(Y)=u(Y)*e_54^T with arbitrary u(Y) in Q(Y)^55",
            "domain_dimension_at_one_jet_monomial": 55,
            "fixed_input_state": {"index": 54, "label": "w1[10]"},
            "contains_registered_reference_columns": [7, 9, 10],
            "contains_registered_variable_column": 10,
            "contains_arbitrary_coefficient_and_output_row_changes": True,
            "preserves_first_order_state_ordering": True,
            "preserves_fixed_high_derivative_selector": True,
            "excludes_new_state_variables": True,
            "excludes_new_input_columns": True,
            "scope_limit": (
                "This class contains the currently registered linear-X quartic-Horndeski "
                "TC2 lift and every gauge-fixed coefficient/output-row deformation that "
                "retains its first-order constraint topology. It does not contain a new "
                "covariant invariant or a deformation that changes the input selector."
            ),
        },
        "constraint_support_audit": {
            "zero_projector_rank": zero_projector.rank(),
            "zero_projector_sha256": _content_hash(_matrix_payload(zero_projector)),
            "fixed_high_covector_times_zero_projector_zero": True,
            "fixed_high_covector_projection_nonzero_entries": 0,
            "fixed_high_covector_projection_sha256": _content_hash(
                _matrix_payload(projected_high)
            ),
            "escape_input_state": {"index": 21, "label": "w2[10]"},
            "escape_input_covector_times_zero_projector_zero": False,
            "escape_input_covector_projection_nonzero_entries": sum(
                value != 0 for value in projected_stationary
            ),
            "escape_input_covector_projection_sha256": _content_hash(
                _matrix_payload(projected_stationary)
            ),
            "escape_V_right_support_columns": [21],
            "registered_right_support_columns": [54],
            "support_intersection_empty": True,
            "interpretation": (
                "V consumes the stationary w2[10] constraint-sector state, whereas every "
                "registered TC2 block consumes the high w1[10] state. Adding V therefore "
                "changes the registered derivative-definition/constraint topology."
            ),
        },
        "induced_cokernel_map": {
            "formula": (
                "R0^T(K0*u*e54^T-e54*u^T*K0)R0="
                "(R0^T*K0*u)(e54^T*R0)-(R0^T*e54)(u^T*K0*R0)=0"
            ),
            "domain_dimension": 55,
            "rank": broad_rank,
            "image_dimension": broad_rank,
            "target_W_rank": target.rank(),
            "target_W_nonzero_entries": sum(value != 0 for value in target),
            "target_W_sha256": EXPECTED_W_SHA256,
            "target_compression_equals_W": True,
            "target_in_image": False,
            "augmented_rank": augmented_rank,
            "codomain_test_coordinates": len(target_coordinates),
        },
        "registered_block_checks": registered_rows,
        "sharp_result": {
            "registered_linear_X_quartic_Horndeski_TC2_realizes_V": False,
            "support_preserving_gauge_fixed_deformation_realizes_V": False,
            "support_preserving_class_can_cancel_obligation_244_W": False,
            "reason": "induced map rank 0 while adjoining W raises rank to 1",
            "sharpness": (
                "The no-go permits arbitrary output u and arbitrary quartic coefficient "
                "dependence; its only structural restriction is the registered e54 input "
                "selector. The algebraic V succeeds precisely by replacing that selector "
                "with e21, so changing constraint topology remains an open escape route."
            ),
        },
        "positive_control": {
            "name": "algebraic_e21_selector",
            "V_sha256": EXPECTED_V_SHA256,
            "V_rank": block_v.rank(),
            "V_right_support_columns": [21],
            "energy_skew_equals_W": induced_v == target,
            "projected_energy_skew_equals_W": (
                zero_projector.T * induced_v * zero_projector
            ).applyfunc(sp.factor)
            == target,
            "accepted": True,
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
        or config.get("expected_registered_block_count") != 4
    ):
        raise QuarticTC2D4RegisteredOperatorOriginNoGoError(
            "registered-operator no-go config mismatch"
        )
    for key in (
        "campaign_source",
        "campaign_test",
        "registered_action_spec",
        "registered_reference_source",
        "registered_variable_source",
        "first_order_reduction_source",
    ):
        _check_raw_binding(root, config[key])
    minimal_escape = _load_bound(root, config["minimal_escape"])
    reference_campaign = _load_bound(root, config["reference_campaign"])
    if (
        config["minimal_escape"]["file_sha256"] != EXPECTED_ESCAPE_FILE_SHA256
        or config["minimal_escape"]["content_sha256"]
        != EXPECTED_ESCAPE_CONTENT_SHA256
        or minimal_escape.get("exact_escape", {})
        .get("correction_ansatz", {})
        .get("V_sha256")
        != EXPECTED_V_SHA256
        or minimal_escape.get("claims", {}).get(
            "correction_covariant_or_action_derived"
        )
        is not False
        or reference_campaign.get("counts", {}).get("selected") != 12
    ):
        raise QuarticTC2D4RegisteredOperatorOriginNoGoError(
            "predecessor semantic binding mismatch"
        )
    action_spec = json.loads(
        (root / config["registered_action_spec"]["path"]).read_text(encoding="utf-8")
    )
    if (
        action_spec.get("terms")
        != ["EH_R", "SCALAR_X", "HORNDESKI_L4_LINEAR_X"]
        or action_spec.get("coefficients", {}).get("HORNDESKI_L4_LINEAR_X")
        != "alpha"
    ):
        raise QuarticTC2D4RegisteredOperatorOriginNoGoError(
            "registered action class changed"
        )
    exact = _exact_certificate()
    body = {
        "schema_version": SCHEMA,
        "status": "pass_exact_no_go_for_registered_support_preserving_TC2_operator_class",
        "config_sha256": config["content_sha256"],
        "source_bindings": {
            key: dict(config[key])
            for key in (
                "minimal_escape",
                "reference_campaign",
                "registered_action_spec",
                "registered_reference_source",
                "registered_variable_source",
                "first_order_reduction_source",
                "campaign_source",
                "campaign_test",
            )
        },
        "selector_binding": {
            "obligation_offset": OBLIGATION_OFFSET,
            "active_indices": list(ACTIVE_INDICES),
            "active_monomial": "Y_0*Y_2*Y_3*Y_9",
            "V_sha256": EXPECTED_V_SHA256,
            "W_sha256": EXPECTED_W_SHA256,
        },
        "exact_no_go": exact,
        "counts": {
            "registered_action_terms_checked": 1,
            "registered_TC2_blocks_checked": 4,
            "broad_support_preserving_domain_dimension": 55,
            "broad_induced_cokernel_map_rank": 0,
            "target_augmented_rank": 1,
            "positive_controls": 1,
            "negative_controls": 5,
            "inferred_global_passes": 0,
        },
        "negative_controls": {
            "infer_from_matching_block_rank": {
                "V_rank": 1,
                "registered_block_rank_can_equal_one": True,
                "support_and_projected_image_still_mismatch": True,
                "rejected": True,
            },
            "allow_arbitrary_output_but_keep_e54": {
                "domain_dimension": 55,
                "induced_map_rank": 0,
                "target_in_image": False,
                "rejected": True,
            },
            "silently_replace_e54_by_e21": {
                "changes_constraint_topology": True,
                "covariant_derivation_present": False,
                "rejected": True,
            },
            "infer_all_covariant_quartic_no_go": {
                "new_invariants_or_input_selectors_classified": False,
                "rejected": True,
            },
            "promote_obligation_244_to_global_closure": {
                "remaining_D4_selector_closed": False,
                "tube_theorem_proved": False,
                "rejected": True,
            },
        },
        "claims": {
            "registered_linear_X_quartic_Horndeski_TC2_origin_ruled_out": True,
            "registered_support_preserving_gauge_deformation_ruled_out": True,
            "arbitrary_covariant_quartic_operator_ruled_out": False,
            "arbitrary_gauge_fixed_operator_deformation_ruled_out": False,
            "covariant_realization_constructed": False,
            "constraint_topology_changing_realization_constructed": False,
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
            "Derive and constraint-check a covariant invariant or gauge-fixed reduction that "
            "genuinely changes the TC2 input selector from e54 to a component with nonzero "
            "zero-speed projection (the minimal escape uses e21), then recompute the principal "
            "constraint subsystem and the affected D4 obligations."
        ),
        "scope": (
            "Exact reference-point incompatibility for the currently registered linear-X "
            "quartic-Horndeski TC2 lift and the broader 55-dimensional class of deformations "
            "B_u=u*e54^T that preserve its first-order input selector. No statement is made "
            "about new covariant invariants, topology-changing gauge reductions, the remaining "
            "D4 selector, a tube theorem, CK1, CK3, TC2, B7, global-H7, or lifespan."
        ),
        "errors": [],
    }
    return _with_hash(body)


def validate_campaign(document: Mapping[str, Any]) -> None:
    if document.get("schema_version") != SCHEMA or not _hash_matches(dict(document)):
        raise QuarticTC2D4RegisteredOperatorOriginNoGoError(
            "registered-operator no-go content identity mismatch"
        )
    exact = document.get("exact_no_go", {})
    operator = exact.get("declared_operator_class", {})
    support = exact.get("constraint_support_audit", {})
    induced = exact.get("induced_cokernel_map", {})
    sharp = exact.get("sharp_result", {})
    positive = exact.get("positive_control", {})
    expected_claims = {
        "registered_linear_X_quartic_Horndeski_TC2_origin_ruled_out": True,
        "registered_support_preserving_gauge_deformation_ruled_out": True,
        "arbitrary_covariant_quartic_operator_ruled_out": False,
        "arbitrary_gauge_fixed_operator_deformation_ruled_out": False,
        "covariant_realization_constructed": False,
        "constraint_topology_changing_realization_constructed": False,
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
    if (
        document.get("status")
        != "pass_exact_no_go_for_registered_support_preserving_TC2_operator_class"
        or document.get("claims") != expected_claims
        or document.get("counts")
        != {
            "registered_action_terms_checked": 1,
            "registered_TC2_blocks_checked": 4,
            "broad_support_preserving_domain_dimension": 55,
            "broad_induced_cokernel_map_rank": 0,
            "target_augmented_rank": 1,
            "positive_controls": 1,
            "negative_controls": 5,
            "inferred_global_passes": 0,
        }
        or operator.get("domain_dimension_at_one_jet_monomial") != 55
        or operator.get("fixed_input_state") != {"index": 54, "label": "w1[10]"}
        or support.get("zero_projector_rank") != 33
        or support.get("fixed_high_covector_times_zero_projector_zero") is not True
        or support.get("escape_input_covector_times_zero_projector_zero") is not False
        or support.get("escape_V_right_support_columns") != [21]
        or support.get("registered_right_support_columns") != [54]
        or induced.get("rank") != 0
        or induced.get("target_in_image") is not False
        or induced.get("augmented_rank") != 1
        or induced.get("target_W_sha256") != EXPECTED_W_SHA256
        or len(exact.get("registered_block_checks", [])) != 4
        or any(
            row.get("right_support_columns") != [54]
            or row.get("zero_eigenspace_compression_zero") is not True
            for row in exact.get("registered_block_checks", [])
        )
        or sharp.get("registered_linear_X_quartic_Horndeski_TC2_realizes_V")
        is not False
        or sharp.get("support_preserving_gauge_fixed_deformation_realizes_V")
        is not False
        or positive.get("V_sha256") != EXPECTED_V_SHA256
        or positive.get("energy_skew_equals_W") is not True
        or positive.get("accepted") is not True
        or len(document.get("negative_controls", {})) != 5
        or any(
            control.get("rejected") is not True
            for control in document.get("negative_controls", {}).values()
        )
    ):
        raise QuarticTC2D4RegisteredOperatorOriginNoGoError(
            "registered-operator exact/fail-closed contract mismatch"
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
        description="Prove the bounded registered-operator origin no-go for D4 escape V."
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
                "map_rank": artifact["counts"][
                    "broad_induced_cokernel_map_rank"
                ],
                "augmented_rank": artifact["counts"]["target_augmented_rank"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
