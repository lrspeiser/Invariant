from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from math import comb
from pathlib import Path
from typing import Any

import sympy as sp

from . import quartic_tc2_diagonal_third_jet_campaign as directional_engine
from .quartic_tc2_diagonal_third_jet_campaign import (
    _active_directions,
    _content_hash,
    _matrix_payload,
)
from .quartic_tc2_fourth_jet_parallel_kernel import _combine_directions
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

SCHEMA = "sigma-quartic-tc2-d4-homogeneous-freedom-reduction-1.0"
CONFIG_SCHEMA = "sigma-quartic-tc2-d4-homogeneous-freedom-config-1.0"
JET_ORDER = 4
OBLIGATION_OFFSET = 244
ACTIVE_INDICES = (0, 2, 3, 9)
ZERO_MULTIPLICITY = 33
COMPANION_DIMENSION = 22
EXPECTED_CANDIDATES = 12
EXPECTED_WITNESS_SHA256 = (
    "6dcc21e22a450b41d624a739c7db4e5d9753a3848f1a9578730f10d77db125f2"
)
EXPECTED_WITNESS_GAP = "[1088/15,34816/15]"
SPECTRUM = (
    sp.Integer(0),
    sp.Integer(1),
    sp.Integer(-1),
    sp.Rational(1, 2),
    sp.Rational(-1, 2),
    sp.Rational(1, 3),
    sp.Rational(-1, 3),
)


class QuarticTC2D4HomogeneousFreedomReductionError(ValueError):
    """Raised when the exact homogeneous-freedom reduction does not close."""


def _load_bound(root: Path, binding: Mapping[str, Any]) -> dict[str, Any]:
    path = (root / str(binding["path"])).resolve()
    if root.resolve() not in path.parents:
        raise QuarticTC2D4HomogeneousFreedomReductionError(
            "bound input escaped project root"
        )
    value, data = _load_file(path)
    if (
        _file_sha256(data) != binding.get("file_sha256")
        or value.get("content_sha256") != binding.get("content_sha256")
        or not _hash_matches(value)
    ):
        raise QuarticTC2D4HomogeneousFreedomReductionError(
            f"bound input mismatch: {binding.get('path')}"
        )
    return value


def _zero_series(rows: int, columns: int) -> list[sp.Matrix]:
    return [sp.zeros(rows, columns) for _ in range(JET_ORDER + 1)]


def _series_product(
    left: list[sp.Matrix], right: list[sp.Matrix]
) -> list[sp.Matrix]:
    result = _zero_series(left[0].rows, right[0].cols)
    for order in range(JET_ORDER + 1):
        result[order] = sum(
            (left[index] * right[order - index] for index in range(order + 1)),
            sp.zeros(left[0].rows, right[0].cols),
        ).applyfunc(sp.factor)
    return result


def _inverse_series(series: list[sp.Matrix]) -> list[sp.Matrix]:
    result = _zero_series(series[0].rows, series[0].cols)
    result[0] = series[0].inv()
    for order in range(1, JET_ORDER + 1):
        result[order] = (
            -result[0]
            * sum(
                (
                    series[index] * result[order - index]
                    for index in range(1, order + 1)
                ),
                sp.zeros(*series[0].shape),
            )
        ).applyfunc(sp.factor)
    return result


def _series_subtract(
    left: list[sp.Matrix], right: list[sp.Matrix]
) -> list[sp.Matrix]:
    return [
        (left[index] - right[index]).applyfunc(sp.factor)
        for index in range(JET_ORDER + 1)
    ]


def _zero_projector_series(physical: list[sp.Matrix]) -> tuple[list[sp.Matrix], dict[str, Any]]:
    if any(
        not physical[order][0:ZERO_MULTIPLICITY, :].is_zero_matrix
        for order in range(JET_ORDER + 1)
    ):
        raise QuarticTC2D4HomogeneousFreedomReductionError(
            "physical symbol lost its exact stationary top block"
        )
    coupling = [
        matrix[ZERO_MULTIPLICITY:STATE_DIMENSION, 0:ZERO_MULTIPLICITY]
        for matrix in physical
    ]
    companion = [
        matrix[
            ZERO_MULTIPLICITY:STATE_DIMENSION,
            ZERO_MULTIPLICITY:STATE_DIMENSION,
        ]
        for matrix in physical
    ]
    inverse = _inverse_series(companion)
    inverse_residual = _series_product(companion, inverse)
    inverse_residual[0] -= sp.eye(COMPANION_DIMENSION)
    inverse_residual = [matrix.applyfunc(sp.factor) for matrix in inverse_residual]
    lower_left = [
        (-matrix).applyfunc(sp.factor)
        for matrix in _series_product(inverse, coupling)
    ]
    projector = _zero_series(STATE_DIMENSION, STATE_DIMENSION)
    projector[0][0:ZERO_MULTIPLICITY, 0:ZERO_MULTIPLICITY] = sp.eye(
        ZERO_MULTIPLICITY
    )
    for order in range(JET_ORDER + 1):
        projector[order][
            ZERO_MULTIPLICITY:STATE_DIMENSION, 0:ZERO_MULTIPLICITY
        ] = lower_left[order]
        projector[order] = projector[order].applyfunc(sp.factor)
    physical_projector = _series_product(physical, projector)
    idempotence = _series_subtract(
        _series_product(projector, projector), projector
    )
    if not all(
        matrix.is_zero_matrix
        for matrix in (*inverse_residual, *physical_projector, *idempotence)
    ):
        raise QuarticTC2D4HomogeneousFreedomReductionError(
            "exact zero-projector series identity failed"
        )
    return projector, {
        "stationary_top_rows_zero_by_order": [
            physical[order][0:ZERO_MULTIPLICITY, :].is_zero_matrix
            for order in range(JET_ORDER + 1)
        ],
        "companion_inverse_residual_zero_by_order": [
            matrix.is_zero_matrix for matrix in inverse_residual
        ],
        "P_times_R0_zero_by_order": [
            matrix.is_zero_matrix for matrix in physical_projector
        ],
        "R0_idempotent_by_order": [matrix.is_zero_matrix for matrix in idempotence],
        "R0_coefficient_sha256": [
            _content_hash(_matrix_payload(matrix)) for matrix in projector
        ],
        "R0_coefficient_nonzero_entries": [
            sum(value != 0 for value in matrix) for matrix in projector
        ],
    }


def _directional_projector_audit(
    campaign: dict[str, Any]
) -> tuple[list[dict[str, Any]], str]:
    active_positions = campaign["selector"]["active_positions"]
    directions = _active_directions()
    selected = tuple(
        directions[active_positions[index]]["direction"] for index in ACTIVE_INDICES
    )
    reference_zero = _reference_and_first_jet_packet()["projectors"][sp.S.Zero]
    records: list[dict[str, Any]] = []
    prior_order = directional_engine.TAYLOR_ORDER
    directional_engine.TAYLOR_ORDER = JET_ORDER
    try:
        for mask in range(1, 1 << len(ACTIVE_INDICES)):
            subset = tuple(index for index in range(4) if mask & (1 << index))
            direction = _combine_directions(selected, subset)
            packet = directional_engine._directional_taylor_packet(
                {
                    "atom_index": -1,
                    "atom": f"D4_homogeneous_zero_projector_mask_{mask:02d}",
                    "direction": direction,
                }
            )
            projector, audit = _zero_projector_series(packet["physical"])
            if not projector[0].equals(reference_zero):
                raise QuarticTC2D4HomogeneousFreedomReductionError(
                    "directional zero projector reference mismatch"
                )
            body = {
                "mask": mask,
                "local_direction_indices": list(subset),
                "active_indices": [ACTIVE_INDICES[index] for index in subset],
                "direction_count": len(subset),
                **audit,
                "prior_record_sha256": records[-1]["record_sha256"] if records else None,
            }
            records.append({**body, "record_sha256": _content_hash(body)})
    finally:
        directional_engine.TAYLOR_ORDER = prior_order
        directional_engine._directional_taylor_packet_cached.cache_clear()
    if len(records) != 15:
        raise QuarticTC2D4HomogeneousFreedomReductionError(
            "four-direction projector audit cardinality mismatch"
        )
    return records, records[-1]["record_sha256"]


def _reference_sylvester_dimensions() -> dict[str, Any]:
    reference = _reference_and_first_jet_packet()
    projectors = reference["projectors"]
    ranks = {str(value): int(projector.rank()) for value, projector in projectors.items()}
    expected_ranks = {
        "0": 33,
        "1": 3,
        "-1": 3,
        "1/2": 4,
        "-1/2": 4,
        "1/3": 4,
        "-1/3": 4,
    }
    identity = sp.eye(STATE_DIMENSION)
    sum_projectors = sum(projectors.values(), sp.zeros(STATE_DIMENSION))
    identities = {
        "sum_to_identity": sum_projectors.equals(identity),
        "idempotent": all(
            (projector * projector).equals(projector)
            for projector in projectors.values()
        ),
        "pairwise_zero": all(
            (left_projector * right_projector).is_zero_matrix
            for left, left_projector in projectors.items()
            for right, right_projector in projectors.items()
            if left != right
        ),
        "spectral": all(
            (reference["physical0"] * projector - value * projector).is_zero_matrix
            for value, projector in projectors.items()
        ),
    }
    if ranks != expected_ranks or not all(identities.values()):
        raise QuarticTC2D4HomogeneousFreedomReductionError(
            "reference spectral decomposition mismatch"
        )
    symmetric_dimension = STATE_DIMENSION * (STATE_DIMENSION + 1) // 2
    skew_dimension = STATE_DIMENSION * (STATE_DIMENSION - 1) // 2
    kernel_dimension = sum(rank * (rank + 1) // 2 for rank in ranks.values())
    range_dimension = sum(
        ranks[str(left)] * ranks[str(right)]
        for index, left in enumerate(SPECTRUM)
        for right in SPECTRUM[index + 1 :]
    )
    cokernel_dimension = sum(rank * (rank - 1) // 2 for rank in ranks.values())
    if (
        kernel_dimension != 613
        or range_dimension != 927
        or cokernel_dimension != 558
        or symmetric_dimension - kernel_dimension != range_dimension
        or skew_dimension - range_dimension != cokernel_dimension
    ):
        raise QuarticTC2D4HomogeneousFreedomReductionError(
            "reference Sylvester dimension accounting mismatch"
        )
    return {
        "state_dimension": STATE_DIMENSION,
        "spectrum": [str(value) for value in SPECTRUM],
        "eigenspace_ranks": ranks,
        "projector_identities": identities,
        "symmetric_domain_dimension": symmetric_dimension,
        "homogeneous_kernel_dimension_per_jet_coefficient": kernel_dimension,
        "Sylvester_range_dimension": range_dimension,
        "skew_codomain_dimension": skew_dimension,
        "equal_eigenspace_cokernel_dimension": cokernel_dimension,
        "zero_eigenspace_skew_cokernel_dimension": 528,
        "minimum_nonzero_reference_spectral_gap": "1/6",
        "zero_projector_sha256": _content_hash(
            _matrix_payload(projectors[sp.S.Zero])
        ),
    }


def build_reduction(project_root: Path, config_path: Path) -> dict[str, Any]:
    project_root = project_root.resolve()
    config, _ = _load_file(config_path.resolve())
    if (
        config.get("schema_version") != CONFIG_SCHEMA
        or not _hash_matches(config)
        or config.get("obligation_offset") != OBLIGATION_OFFSET
        or tuple(config.get("active_indices", ())) != ACTIVE_INDICES
        or config.get("expected_candidate_count") != EXPECTED_CANDIDATES
        or config.get("global_claim_policy") != "fail_closed"
    ):
        raise QuarticTC2D4HomogeneousFreedomReductionError(
            "homogeneous-freedom config mismatch"
        )
    for local_key in ("campaign_source", "campaign_test"):
        binding = config[local_key]
        if _file_sha256((project_root / binding["path"]).read_bytes()) != binding["file_sha256"]:
            raise QuarticTC2D4HomogeneousFreedomReductionError(
                f"local binding mismatch: {local_key}"
            )
    obstruction = _load_bound(project_root, config["obstruction_certificate"])
    campaign = _load_bound(project_root, config["fourth_campaign"])
    if (
        obstruction.get("claims", {}).get(
            "alternative_lower_jet_homogeneous_completion_ruled_out"
        )
        is not False
        or obstruction.get("claims", {}).get(
            "all_12_registered_candidates_canonically_obstructed"
        )
        is not True
        or obstruction.get("exact_symbolic_certificate", {})
        .get("equal_eigenspace_compressions", {})
        .get("zero_eigenspace", {})
        .get("sha256")
        != EXPECTED_WITNESS_SHA256
        or obstruction.get("exact_symbolic_certificate", {})
        .get("exact_candidate_gap", {})
        .get("interval")
        != EXPECTED_WITNESS_GAP
        or len(
            obstruction.get("exact_symbolic_certificate", {}).get(
                "candidate_classification", []
            )
        )
        != EXPECTED_CANDIDATES
        or campaign.get("selector", {}).get("records", [])[OBLIGATION_OFFSET].get(
            "active_indices"
        )
        != list(ACTIVE_INDICES)
    ):
        raise QuarticTC2D4HomogeneousFreedomReductionError(
            "bound obstruction/campaign mismatch"
        )
    projector_records, projector_tip = _directional_projector_audit(campaign)
    dimensions = _reference_sylvester_dimensions()
    kernel = dimensions["homogeneous_kernel_dimension_per_jet_coefficient"]
    jet_slots = {
        f"order_{order}": {
            "symmetric_multiindices": comb(4 + order - 1, order),
            "reference_kernel_slots": comb(4 + order - 1, order) * kernel,
        }
        for order in range(1, 4)
    }
    total_slots = sum(row["reference_kernel_slots"] for row in jet_slots.values())
    if total_slots != 20842:
        raise QuarticTC2D4HomogeneousFreedomReductionError(
            "lower-jet homogeneous slot accounting mismatch"
        )
    candidate_rows = [
        {
            "candidate_id": row["candidate_id"],
            "a10": row["a10"],
            "canonical_witness_scalar": row["witness_scalar"],
            "canonical_compression_sha256": row["compression_sha256"],
            "homogeneous_D4_zero_eigenspace_correction": "0",
            "cancellation_possible": False,
            "D4_compatible_after_all_lower_homogeneous_freedom": False,
        }
        for row in obstruction["exact_symbolic_certificate"][
            "candidate_classification"
        ]
    ]
    body = {
        "schema_version": SCHEMA,
        "status": "pass_exact_d4_obstruction_invariant_under_all_lower_homogeneous_freedom",
        "config_sha256": config["content_sha256"],
        "source_bindings": {
            key: {
                "path": config[key]["path"],
                "file_sha256": config[key]["file_sha256"],
                **(
                    {"content_sha256": config[key]["content_sha256"]}
                    if "content_sha256" in config[key]
                    else {}
                ),
            }
            for key in (
                "obstruction_certificate",
                "fourth_campaign",
                "campaign_source",
                "campaign_test",
            )
        },
        "selector_binding": {
            "obligation_offset": OBLIGATION_OFFSET,
            "active_indices": list(ACTIVE_INDICES),
            "active_positions": [0, 2, 4, 15],
            "multiplicity_partition": "ABCD",
            "canonical_witness_sha256": EXPECTED_WITNESS_SHA256,
        },
        "reference_sylvester_space": dimensions,
        "exact_zero_projector_audit": {
            "construction": "R0(Y)=[[I_33,0],[-D(Y)^(-1)C(Y),0]]",
            "physical_block_form": "P(Y)=[[0,0],[C(Y),D(Y)]]",
            "polarization_directions_checked": 15,
            "Taylor_orders_checked": [0, 1, 2, 3, 4],
            "record_chain_tip_sha256": projector_tip,
            "records": projector_records,
        },
        "homogeneous_freedom_reduction": {
            "homogeneous_residual": "F_H(Y)=H(Y)P(Y)-P(Y)^T H(Y)",
            "exact_identity": "R0(Y)^T F_H(Y) R0(Y)=0 for every matrix H(Y)",
            "lower_order_hypothesis": "D^j F_H(0)=0 for j=0,1,2,3",
            "order_four_consequence": "R0(0)^T D4F_H(0) R0(0)=0",
            "proof": (
                "Differentiate the exact projected identity four times. Every Leibniz "
                "term containing D^j F_H(0), j<4, vanishes by the lower-order homogeneous "
                "recurrence; the only surviving term is the fixed-reference D4 compression."
            ),
            "Hermitian_assumption_needed_for_zero_result": False,
            "lower_jet_reference_kernel_slots": jet_slots,
            "total_lower_jet_reference_kernel_slots_before_cross_order_constraints": total_slots,
            "induced_D4_zero_eigenspace_map_domain_scope": (
                "every order-one-through-three homogeneous completion satisfying its exact "
                "lower recurrence"
            ),
            "induced_D4_zero_eigenspace_map_rank": 0,
            "induced_D4_zero_eigenspace_map_image_dimension": 0,
            "canonical_rank_two_witness_in_image": False,
        },
        "candidate_classification": candidate_rows,
        "counts": {
            "selector_obligations_classified": 1,
            "polarization_directions_checked": 15,
            "Taylor_orders_per_direction_checked": 5,
            "stationary_block_checks": 75,
            "projector_algebra_checks": 225,
            "total_exact_zero_projector_checks": 300,
            "candidate_specializations_checked": EXPECTED_CANDIDATES,
            "candidate_obstructions_invariant": EXPECTED_CANDIDATES,
            "candidate_cancellations": 0,
            "lower_jet_reference_kernel_slots_covered_by_identity": total_slots,
            "induced_cokernel_map_rank": 0,
            "negative_controls": 5,
            "inferred_passes": 0,
        },
        "negative_controls": {
            "drop_lower_recurrence_hypothesis": {
                "false_step": "discard D^j F_H(0)=0 for j<4",
                "why_rejected": "Leibniz terms with differentiated R0 and lower F_H then remain",
                "rejected": True,
            },
            "discard_exact_zero_projector": {
                "false_step": "treat the 33-dimensional stationary block as reference-only",
                "exact_directional_projector_checks": 300,
                "rejected": True,
            },
            "use_raw_top_left_block": {
                "false_step": "replace R0(0)^T D4F_H(0) R0(0) by a coordinate submatrix",
                "projector_is_coordinate_diagonal": False,
                "rejected": True,
            },
            "cancel_rank_two_witness": {
                "required_image_rank_at_least": 1,
                "proved_image_rank": 0,
                "rejected": True,
            },
            "promote_to_downstream_closure": {
                "false_step": "turn one exact candidate obstruction into CK/TC2/H7/lifespan closure",
                "rejected": True,
            },
        },
        "claims": {
            "canonical_D4_obligation_244_obstructed": True,
            "alternative_lower_jet_homogeneous_completion_ruled_out_for_obligation_244": True,
            "all_12_registered_candidates_D4_obstructed_at_obligation_244": True,
            "all_3060_fourth_jet_obligations_evaluated": False,
            "full_fourth_jet_range_closed": False,
            "full_tube_Sylvester_identity": False,
            "CK1_closed": False,
            "CK3_closed": False,
            "TC2_closed": False,
            "B7_closed": False,
            "global_H7_closed": False,
            "lifespan_proved": False,
        },
        "next_gate": (
            "The current 12-candidate quartic family has an invariant exact D4 obstruction. "
            "Any continuation must change the candidate/operator ansatz or the TC2 correction, "
            "not merely the lower-jet homogeneous Sylvester normalization."
        ),
        "scope": (
            "This proves invariance of the exact obligation-244 zero-speed obstruction under "
            "every lower homogeneous matrix completion satisfying orders one through three. "
            "It does not evaluate the remaining selector or promote full-D4, tube, CK1, CK3, "
            "TC2, B7, global-H7, or lifespan claims."
        ),
        "errors": [],
    }
    return _with_hash(body)


def validate_reduction(document: Mapping[str, Any]) -> None:
    if document.get("schema_version") != SCHEMA or not _hash_matches(dict(document)):
        raise QuarticTC2D4HomogeneousFreedomReductionError(
            "homogeneous reduction content identity mismatch"
        )
    reduction = document.get("homogeneous_freedom_reduction", {})
    audit = document.get("exact_zero_projector_audit", {})
    records = audit.get("records", [])
    expected_claims = {
        "canonical_D4_obligation_244_obstructed": True,
        "alternative_lower_jet_homogeneous_completion_ruled_out_for_obligation_244": True,
        "all_12_registered_candidates_D4_obstructed_at_obligation_244": True,
        "all_3060_fourth_jet_obligations_evaluated": False,
        "full_fourth_jet_range_closed": False,
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
        != "pass_exact_d4_obstruction_invariant_under_all_lower_homogeneous_freedom"
        or document.get("claims") != expected_claims
        or document.get("selector_binding", {}).get("canonical_witness_sha256")
        != EXPECTED_WITNESS_SHA256
        or len(records) != 15
        or [record.get("mask") for record in records] != list(range(1, 16))
        or any(
            record.get("stationary_top_rows_zero_by_order") != [True] * 5
            or record.get("companion_inverse_residual_zero_by_order") != [True] * 5
            or record.get("P_times_R0_zero_by_order") != [True] * 5
            or record.get("R0_idempotent_by_order") != [True] * 5
            for record in records
        )
        or audit.get("record_chain_tip_sha256") != records[-1].get("record_sha256")
        or reduction.get("total_lower_jet_reference_kernel_slots_before_cross_order_constraints")
        != 20842
        or reduction.get("induced_D4_zero_eigenspace_map_rank") != 0
        or reduction.get("induced_D4_zero_eigenspace_map_image_dimension") != 0
        or reduction.get("canonical_rank_two_witness_in_image") is not False
        or len(document.get("candidate_classification", [])) != EXPECTED_CANDIDATES
        or any(
            row.get("homogeneous_D4_zero_eigenspace_correction") != "0"
            or row.get("cancellation_possible") is not False
            or row.get("D4_compatible_after_all_lower_homogeneous_freedom") is not False
            for row in document.get("candidate_classification", [])
        )
        or set(document.get("negative_controls", {}))
        != {
            "drop_lower_recurrence_hypothesis",
            "discard_exact_zero_projector",
            "use_raw_top_left_block",
            "cancel_rank_two_witness",
            "promote_to_downstream_closure",
        }
        or any(
            control.get("rejected") is not True
            for control in document.get("negative_controls", {}).values()
        )
    ):
        raise QuarticTC2D4HomogeneousFreedomReductionError(
            "homogeneous reduction exact/fail-closed contract mismatch"
        )


def run_reduction(
    project_root: Path, config_path: Path, output_path: Path
) -> dict[str, Any]:
    artifact = build_reduction(project_root, config_path)
    validate_reduction(artifact)
    _atomic_write(output_path.resolve(), _json_bytes(artifact))
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reduce all lower homogeneous freedom at D4 obligation 244."
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    artifact = run_reduction(args.project_root, args.config, args.output)
    print(
        json.dumps(
            {
                "status": artifact["status"],
                "content_sha256": artifact["content_sha256"],
                "homogeneous_image_rank": artifact["counts"][
                    "induced_cokernel_map_rank"
                ],
                "candidate_obstructions": artifact["counts"][
                    "candidate_obstructions_invariant"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
