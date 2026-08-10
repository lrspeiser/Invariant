from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .quartic_arithmetic_expansion_shared import build_output_row_arithmetic_packet
from .quartic_row0_arithmetic_expansion_campaign import (
    _candidate_records,
    _content_hash,
    _content_hash_matches,
    generic_arithmetic_materialization_control,
)

SCHEMA_VERSION = "sigma-quartic-row2-arithmetic-expansion-campaign-1.0"
OUTPUT_ROW = 2


class QuarticRow2ArithmeticExpansionError(ValueError):
    """Raised when row-two arithmetic materialization is invalid."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _certify_candidate(
    row1: dict[str, Any], metric: dict[str, Any], packet: dict[str, Any]
) -> dict[str, Any]:
    candidate_id = str(row1.get("candidate_id"))
    if (
        metric.get("candidate_id") != candidate_id
        or metric.get("coefficients") != row1.get("coefficients")
    ):
        raise QuarticRow2ArithmeticExpansionError("candidate identity mismatch")
    return {
        "schema_version": "sigma-quartic-row2-arithmetic-expansion-certificate-1.0",
        "status": "pass_row2_lower_arithmetic_materialization_other_rows_fail_closed",
        "candidate_id": candidate_id,
        "coefficients": row1["coefficients"],
        "provenance": {
            "row0_arithmetic_dag_sha256": row1["provenance"][
                "row0_arithmetic_dag_sha256"
            ],
            "row1_arithmetic_dag_sha256": row1["provenance"][
                "row1_arithmetic_dag_sha256"
            ],
            "metric_tensor_dag_sha256": metric["provenance"][
                "common_tensor_dag_sha256"
            ],
            "metric_root_packet_sha256": metric["provenance"][
                "common_root_packet_sha256"
            ],
            "row2_arithmetic_dag_sha256": packet["arithmetic_dag"][
                "content_sha256"
            ],
            "component_input_contract_sha256": _content_hash(
                {
                    "metric_roots": metric["provenance"][
                        "common_root_packet_sha256"
                    ],
                    "coefficients": row1["coefficients"],
                }
            ),
        },
        "row_coverage": {
            **{
                str(row): {
                    "lower_entries_arithmetic_normalized": 54,
                    "mixed_entries_orders_2_to_4_normalized": 6,
                    "source": f"row{row} arithmetic campaign",
                }
                for row in range(3)
            },
            **{
                str(row): {
                    "lower_entries_arithmetic_normalized": 0,
                    "mixed_entries_orders_2_to_4_normalized": 0,
                    "source": "unmaterialized",
                }
                for row in range(3, 11)
            },
        },
        "current_lower_entries_normalized": 54,
        "cumulative_lower_entries_normalized": 162,
        "current_selected_mixed_entries_normalized": 6,
        "cumulative_selected_mixed_entries_normalized": 18,
        "full_11x153_source_Jacobian_entrywise_materialized": False,
        "full_component_Frechet_tensors_complete": False,
        "paralinearization_remainder_bound_proved": False,
        "full_H7_commutator_closed": False,
        "global_dyadic_summation_applied": False,
        "remaining_gate": (
            "materialize output rows 3 through 10 and all remaining mixed atom "
            "multi-indices before applying the component remainder"
        ),
    }


def run_quartic_row2_arithmetic_expansion_campaign(
    row1_campaign: dict[str, Any],
    metric_rows_campaign: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        if config.get("schema_version") != SCHEMA_VERSION:
            raise QuarticRow2ArithmeticExpansionError(
                "unsupported campaign schema_version"
            )
        if (
            row1_campaign.get("status")
            != "pass_all_12_rows0_1_arithmetic_materialized_other_rows_fail_closed"
            or metric_rows_campaign.get("status")
            != "pass_all_12_all_Euler_rows_tensor_lowered_mixed_incomplete_fail_closed"
        ):
            raise QuarticRow2ArithmeticExpansionError(
                "campaign prerequisite status mismatch"
            )
        if not _content_hash_matches(row1_campaign) or not _content_hash_matches(
            metric_rows_campaign
        ):
            raise QuarticRow2ArithmeticExpansionError("campaign content hash mismatch")
        if row1_campaign.get("upstream_sha256", {}).get(
            "metric_rows_tensor_dag"
        ) != metric_rows_campaign.get("content_sha256"):
            raise QuarticRow2ArithmeticExpansionError("row1 provenance mismatch")
        if (
            int(config["output_row"]) != OUTPUT_ROW
            or int(config["lower_column_count"]) != 54
            or list(config["mixed_atom_pair"]) != ["p0[10]", "p1[10]"]
            or int(config["max_mixed_derivative_order"]) != 4
        ):
            raise QuarticRow2ArithmeticExpansionError(
                "unsupported arithmetic checkpoint"
            )
        if bool(config.get("declare_component_remainder_proved", False)):
            raise QuarticRow2ArithmeticExpansionError(
                "component remainder cannot be declared from three output rows"
            )
        control_passed, control = generic_arithmetic_materialization_control()
        if not control_passed:
            raise QuarticRow2ArithmeticExpansionError(
                "generic arithmetic materialization failed"
            )
        component_provenance = metric_rows_campaign[
            "common_explicit_tensor_dag_packet"
        ]["root_packet"]["content_sha256"]
        packet = build_output_row_arithmetic_packet(
            OUTPUT_ROW, component_provenance
        )
        allowed = set(packet["arithmetic_dag"]["allowed_operations"])
        actual = {node["op"] for node in packet["arithmetic_dag"]["nodes"]}
        if actual - allowed:
            raise QuarticRow2ArithmeticExpansionError(
                "non-arithmetic operation leaked into output DAG"
            )
        maps = (
            _candidate_records(row1_campaign),
            _candidate_records(metric_rows_campaign),
        )
        candidate_ids = set(maps[0])
        if len(candidate_ids) != int(config.get("expected_candidate_count", 12)) or set(
            maps[1]
        ) != candidate_ids:
            raise QuarticRow2ArithmeticExpansionError("candidate-set mismatch")
        certificates = [
            _certify_candidate(maps[0][candidate_id], maps[1][candidate_id], packet)
            for candidate_id in sorted(candidate_ids)
        ]
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": "pass_all_12_rows0_2_arithmetic_materialized_other_rows_fail_closed",
            "errors": [],
            "upstream_sha256": {
                "row1_arithmetic": row1_campaign.get("content_sha256"),
                "metric_rows_tensor_dag": metric_rows_campaign.get("content_sha256"),
            },
            "config_sha256": hashlib.sha256(_canonical_json(config).encode()).hexdigest(),
            "generic_arithmetic_materialization_control": control,
            "common_row2_arithmetic_packet": packet,
            "counts": {
                "selected": len(certificates),
                "current_output_rows_materialized_per_candidate": 1,
                "cumulative_output_rows_materialized_per_candidate": 3,
                "current_lower_entries_normalized_per_candidate": 54,
                "cumulative_lower_entries_normalized_per_candidate": 162,
                "current_selected_mixed_entries_per_candidate": 6,
                "cumulative_selected_mixed_entries_per_candidate": 18,
                "full_component_remainders_proved": 0,
                "H7_closures": 0,
                "rejected": 0,
            },
            "certificates": certificates,
            "claim": (
                "Output rows zero through two have arithmetic-only lower Jacobian and "
                "selected mixed roots; output rows three through ten remain fail-closed."
            ),
            "scope": (
                "The row-two DAG uses the shared audited Faddeev-LeVerrier and "
                "normalized bivariate recurrence implementation."
            ),
        }
    except (
        KeyError,
        TypeError,
        ValueError,
        QuarticRow2ArithmeticExpansionError,
    ) as error:
        errors.append(str(error))
        body = {
            "schema_version": SCHEMA_VERSION,
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


def write_quartic_row2_arithmetic_expansion_campaign(
    result: dict[str, Any], output: Path
) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    path = output / "campaign.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
