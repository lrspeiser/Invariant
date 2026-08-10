from __future__ import annotations

import hashlib
import json
from typing import Any

from .quartic_arithmetic_expansion_shared import build_output_row_arithmetic_packet
from .quartic_row0_arithmetic_expansion_campaign import (
    _candidate_records,
    _content_hash,
    _content_hash_matches,
    generic_arithmetic_materialization_control,
)


class QuarticArithmeticCampaignRunnerError(ValueError):
    """Raised when a next-row arithmetic campaign contract is invalid."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _certify_candidate(
    previous: dict[str, Any],
    metric: dict[str, Any],
    packet: dict[str, Any],
    output_row: int,
    certificate_schema: str,
) -> dict[str, Any]:
    candidate_id = str(previous.get("candidate_id"))
    if (
        metric.get("candidate_id") != candidate_id
        or metric.get("coefficients") != previous.get("coefficients")
    ):
        raise QuarticArithmeticCampaignRunnerError("candidate identity mismatch")
    prior_hashes = {
        key: value
        for key, value in previous["provenance"].items()
        if key.startswith("row") and key.endswith("_arithmetic_dag_sha256")
    }
    row_coverage = json.loads(json.dumps(previous["row_coverage"]))
    row_coverage[str(output_row)] = {
        "lower_entries_arithmetic_normalized": 54,
        "mixed_entries_orders_2_to_4_normalized": 6,
        "source": f"row{output_row} arithmetic campaign",
    }
    return {
        "schema_version": certificate_schema,
        "status": (
            f"pass_row{output_row}_lower_arithmetic_materialization_"
            "other_rows_fail_closed"
        ),
        "candidate_id": candidate_id,
        "coefficients": previous["coefficients"],
        "provenance": {
            **prior_hashes,
            "metric_tensor_dag_sha256": metric["provenance"][
                "common_tensor_dag_sha256"
            ],
            "metric_root_packet_sha256": metric["provenance"][
                "common_root_packet_sha256"
            ],
            f"row{output_row}_arithmetic_dag_sha256": packet["arithmetic_dag"][
                "content_sha256"
            ],
            "component_input_contract_sha256": _content_hash(
                {
                    "metric_roots": metric["provenance"][
                        "common_root_packet_sha256"
                    ],
                    "coefficients": previous["coefficients"],
                }
            ),
        },
        "row_coverage": row_coverage,
        "current_lower_entries_normalized": 54,
        "cumulative_lower_entries_normalized": 54 * (output_row + 1),
        "current_selected_mixed_entries_normalized": 6,
        "cumulative_selected_mixed_entries_normalized": 6 * (output_row + 1),
        "full_11x153_source_Jacobian_entrywise_materialized": False,
        "full_component_Frechet_tensors_complete": False,
        "paralinearization_remainder_bound_proved": False,
        "full_H7_commutator_closed": False,
        "global_dyadic_summation_applied": False,
        "remaining_gate": (
            f"materialize output rows {output_row + 1} through 10 and all remaining "
            "mixed atom multi-indices before applying the component remainder"
        ),
    }


def run_next_row_arithmetic_campaign(
    previous_campaign: dict[str, Any],
    metric_rows_campaign: dict[str, Any],
    config: dict[str, Any],
    *,
    schema_version: str,
    certificate_schema: str,
    output_row: int,
    previous_status: str,
    success_status: str,
    packet_key: str,
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        if config.get("schema_version") != schema_version:
            raise QuarticArithmeticCampaignRunnerError(
                "unsupported campaign schema_version"
            )
        if (
            previous_campaign.get("status") != previous_status
            or metric_rows_campaign.get("status")
            != "pass_all_12_all_Euler_rows_tensor_lowered_mixed_incomplete_fail_closed"
        ):
            raise QuarticArithmeticCampaignRunnerError(
                "campaign prerequisite status mismatch"
            )
        if not _content_hash_matches(previous_campaign) or not _content_hash_matches(
            metric_rows_campaign
        ):
            raise QuarticArithmeticCampaignRunnerError("campaign content hash mismatch")
        if previous_campaign.get("upstream_sha256", {}).get(
            "metric_rows_tensor_dag"
        ) != metric_rows_campaign.get("content_sha256"):
            raise QuarticArithmeticCampaignRunnerError("previous-row provenance mismatch")
        if (
            int(config["output_row"]) != output_row
            or int(config["lower_column_count"]) != 54
            or list(config["mixed_atom_pair"]) != ["p0[10]", "p1[10]"]
            or int(config["max_mixed_derivative_order"]) != 4
        ):
            raise QuarticArithmeticCampaignRunnerError(
                "unsupported arithmetic checkpoint"
            )
        if bool(config.get("declare_component_remainder_proved", False)):
            raise QuarticArithmeticCampaignRunnerError(
                f"component remainder cannot be declared from {output_row + 1} output rows"
            )
        control_passed, control = generic_arithmetic_materialization_control()
        if not control_passed:
            raise QuarticArithmeticCampaignRunnerError(
                "generic arithmetic materialization failed"
            )
        component_provenance = metric_rows_campaign[
            "common_explicit_tensor_dag_packet"
        ]["root_packet"]["content_sha256"]
        packet = build_output_row_arithmetic_packet(output_row, component_provenance)
        allowed = set(packet["arithmetic_dag"]["allowed_operations"])
        actual = {node["op"] for node in packet["arithmetic_dag"]["nodes"]}
        if actual - allowed:
            raise QuarticArithmeticCampaignRunnerError(
                "non-arithmetic operation leaked into output DAG"
            )
        maps = (
            _candidate_records(previous_campaign),
            _candidate_records(metric_rows_campaign),
        )
        candidate_ids = set(maps[0])
        if len(candidate_ids) != int(config.get("expected_candidate_count", 12)) or set(
            maps[1]
        ) != candidate_ids:
            raise QuarticArithmeticCampaignRunnerError("candidate-set mismatch")
        certificates = [
            _certify_candidate(
                maps[0][candidate_id],
                maps[1][candidate_id],
                packet,
                output_row,
                certificate_schema,
            )
            for candidate_id in sorted(candidate_ids)
        ]
        body = {
            "schema_version": schema_version,
            "status": success_status,
            "errors": [],
            "upstream_sha256": {
                f"row{output_row - 1}_arithmetic": previous_campaign.get(
                    "content_sha256"
                ),
                "metric_rows_tensor_dag": metric_rows_campaign.get("content_sha256"),
            },
            "config_sha256": hashlib.sha256(_canonical_json(config).encode()).hexdigest(),
            "generic_arithmetic_materialization_control": control,
            packet_key: packet,
            "counts": {
                "selected": len(certificates),
                "current_output_rows_materialized_per_candidate": 1,
                "cumulative_output_rows_materialized_per_candidate": output_row + 1,
                "current_lower_entries_normalized_per_candidate": 54,
                "cumulative_lower_entries_normalized_per_candidate": 54
                * (output_row + 1),
                "current_selected_mixed_entries_per_candidate": 6,
                "cumulative_selected_mixed_entries_per_candidate": 6
                * (output_row + 1),
                "full_component_remainders_proved": 0,
                "H7_closures": 0,
                "rejected": 0,
            },
            "certificates": certificates,
            "claim": (
                f"Output rows zero through {output_row} have arithmetic-only lower "
                f"Jacobian and selected mixed roots; output rows {output_row + 1} "
                "through ten remain fail-closed."
            ),
            "scope": (
                f"The row-{output_row} DAG uses the shared audited Faddeev-LeVerrier "
                "and normalized bivariate recurrence implementation."
            ),
        }
    except (KeyError, TypeError, ValueError, QuarticArithmeticCampaignRunnerError) as error:
        errors.append(str(error))
        body = {
            "schema_version": schema_version,
            "status": "reject",
            "errors": errors,
            "certificates": [],
            "counts": {
                "selected": 0,
                "current_output_rows_materialized_per_candidate": 0,
                "cumulative_output_rows_materialized_per_candidate": 0,
                "current_lower_entries_normalized_per_candidate": 0,
                "cumulative_lower_entries_normalized_per_candidate": 0,
                "current_selected_mixed_entries_per_candidate": 0,
                "cumulative_selected_mixed_entries_per_candidate": 0,
                "full_component_remainders_proved": 0,
                "H7_closures": 0,
                "rejected": 0,
            },
        }
    return {**body, "content_sha256": _content_hash(body)}
