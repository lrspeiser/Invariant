from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .quartic_arithmetic_multirow_shared import build_output_rows_arithmetic_packet
from .quartic_row0_arithmetic_expansion_campaign import (
    _candidate_records,
    _content_hash,
    _content_hash_matches,
    generic_arithmetic_materialization_control,
)

SCHEMA_VERSION = "sigma-quartic-rows5-10-arithmetic-expansion-campaign-1.0"
OUTPUT_ROWS = (5, 6, 7, 8, 9, 10)


class QuarticRows5To10ArithmeticExpansionError(ValueError):
    """Raised when rows five through ten cannot be materialized exactly."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _certify_candidate(
    row4: dict[str, Any], metric: dict[str, Any], packet: dict[str, Any]
) -> dict[str, Any]:
    candidate_id = str(row4.get("candidate_id"))
    if (
        metric.get("candidate_id") != candidate_id
        or metric.get("coefficients") != row4.get("coefficients")
    ):
        raise QuarticRows5To10ArithmeticExpansionError("candidate identity mismatch")
    prior_hashes = {
        key: value
        for key, value in row4["provenance"].items()
        if key.startswith("row") and key.endswith("_arithmetic_dag_sha256")
    }
    required_prior_hashes = {
        f"row{row}_arithmetic_dag_sha256" for row in range(5)
    }
    if set(prior_hashes) != required_prior_hashes or any(
        not isinstance(value, str) or len(value) != 64
        for value in prior_hashes.values()
    ):
        raise QuarticRows5To10ArithmeticExpansionError(
            "rows0-4 arithmetic provenance is incomplete"
        )
    for row in range(5):
        coverage = row4["row_coverage"].get(str(row), {})
        if (
            coverage.get("lower_entries_arithmetic_normalized") != 54
            or coverage.get("mixed_entries_orders_2_to_4_normalized") != 6
        ):
            raise QuarticRows5To10ArithmeticExpansionError(
                "rows0-4 arithmetic coverage is incomplete"
            )
    row_coverage = json.loads(json.dumps(row4["row_coverage"]))
    for row in OUTPUT_ROWS:
        row_coverage[str(row)] = {
            "lower_entries_arithmetic_normalized": 54,
            "mixed_entries_orders_2_to_4_normalized": 6,
            "source": "shared rows5-10 arithmetic campaign",
        }
    return {
        "schema_version": "sigma-quartic-rows5-10-arithmetic-expansion-certificate-1.0",
        "status": "pass_all_output_rows_lower_arithmetic_selected_mixed_only",
        "candidate_id": candidate_id,
        "coefficients": row4["coefficients"],
        "provenance": {
            **prior_hashes,
            "metric_tensor_dag_sha256": metric["provenance"][
                "common_tensor_dag_sha256"
            ],
            "metric_root_packet_sha256": metric["provenance"][
                "common_root_packet_sha256"
            ],
            "rows5_10_arithmetic_dag_sha256": packet["arithmetic_dag"][
                "content_sha256"
            ],
            "component_input_contract_sha256": _content_hash(
                {
                    "metric_roots": metric["provenance"][
                        "common_root_packet_sha256"
                    ],
                    "coefficients": row4["coefficients"],
                }
            ),
        },
        "row_coverage": row_coverage,
        "current_output_rows_materialized": 6,
        "current_lower_entries_normalized": 324,
        "cumulative_lower_entries_normalized": 594,
        "current_selected_mixed_entries_normalized": 36,
        "cumulative_selected_mixed_entries_normalized": 66,
        "full_lower_11x54_Jacobian_entrywise_materialized": True,
        "full_11x153_source_Jacobian_entrywise_materialized": False,
        "full_component_Frechet_tensors_complete": False,
        "paralinearization_remainder_bound_proved": False,
        "full_H7_commutator_closed": False,
        "global_dyadic_summation_applied": False,
        "remaining_gate": (
            "materialize every required atom multi-index in D^2F,D^3F,D^4F and "
            "materialize the 1089 principal entries before applying the remainder"
        ),
    }


def run_quartic_rows5_10_arithmetic_expansion_campaign(
    row4_campaign: dict[str, Any],
    metric_rows_campaign: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        if config.get("schema_version") != SCHEMA_VERSION:
            raise QuarticRows5To10ArithmeticExpansionError(
                "unsupported campaign schema_version"
            )
        if (
            row4_campaign.get("status")
            != "pass_all_12_rows0_4_arithmetic_materialized_other_rows_fail_closed"
            or metric_rows_campaign.get("status")
            != "pass_all_12_all_Euler_rows_tensor_lowered_mixed_incomplete_fail_closed"
        ):
            raise QuarticRows5To10ArithmeticExpansionError(
                "campaign prerequisite status mismatch"
            )
        if not _content_hash_matches(row4_campaign) or not _content_hash_matches(
            metric_rows_campaign
        ):
            raise QuarticRows5To10ArithmeticExpansionError(
                "campaign content hash mismatch"
            )
        if row4_campaign.get("upstream_sha256", {}).get(
            "metric_rows_tensor_dag"
        ) != metric_rows_campaign.get("content_sha256"):
            raise QuarticRows5To10ArithmeticExpansionError(
                "row4 provenance mismatch"
            )
        if (
            tuple(config["output_rows"]) != OUTPUT_ROWS
            or int(config["lower_column_count"]) != 54
            or list(config["mixed_atom_pair"]) != ["p0[10]", "p1[10]"]
            or int(config["max_mixed_derivative_order"]) != 4
        ):
            raise QuarticRows5To10ArithmeticExpansionError(
                "unsupported multirow arithmetic checkpoint"
            )
        if bool(config.get("declare_component_remainder_proved", False)):
            raise QuarticRows5To10ArithmeticExpansionError(
                "component remainder cannot be declared from one selected atom pair"
            )
        control_passed, control = generic_arithmetic_materialization_control()
        if not control_passed:
            raise QuarticRows5To10ArithmeticExpansionError(
                "generic arithmetic materialization failed"
            )
        component_provenance = metric_rows_campaign[
            "common_explicit_tensor_dag_packet"
        ]["root_packet"]["content_sha256"]
        packet = build_output_rows_arithmetic_packet(
            OUTPUT_ROWS, component_provenance
        )
        allowed = set(packet["arithmetic_dag"]["allowed_operations"])
        actual = {node["op"] for node in packet["arithmetic_dag"]["nodes"]}
        if actual - allowed:
            raise QuarticRows5To10ArithmeticExpansionError(
                "non-arithmetic operation leaked into output DAG"
            )
        maps = (
            _candidate_records(row4_campaign),
            _candidate_records(metric_rows_campaign),
        )
        candidate_ids = set(maps[0])
        if len(candidate_ids) != int(config.get("expected_candidate_count", 12)) or set(
            maps[1]
        ) != candidate_ids:
            raise QuarticRows5To10ArithmeticExpansionError("candidate-set mismatch")
        certificates = [
            _certify_candidate(maps[0][candidate_id], maps[1][candidate_id], packet)
            for candidate_id in sorted(candidate_ids)
        ]
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": "pass_all_12_all_lower_rows_arithmetic_mixed_tensor_fail_closed",
            "errors": [],
            "upstream_sha256": {
                "row4_arithmetic": row4_campaign.get("content_sha256"),
                "metric_rows_tensor_dag": metric_rows_campaign.get("content_sha256"),
            },
            "config_sha256": hashlib.sha256(_canonical_json(config).encode()).hexdigest(),
            "generic_arithmetic_materialization_control": control,
            "common_rows5_10_arithmetic_packet": packet,
            "counts": {
                "selected": len(certificates),
                "current_output_rows_materialized_per_candidate": 6,
                "cumulative_output_rows_materialized_per_candidate": 11,
                "current_lower_entries_normalized_per_candidate": 324,
                "cumulative_lower_entries_normalized_per_candidate": 594,
                "current_selected_mixed_entries_per_candidate": 36,
                "cumulative_selected_mixed_entries_per_candidate": 66,
                "full_lower_Jacobians_materialized": len(certificates),
                "full_component_remainders_proved": 0,
                "H7_closures": 0,
                "rejected": 0,
            },
            "certificates": certificates,
            "claim": (
                "All eleven output rows have arithmetic-only lower Jacobian entries; "
                "only one selected atom pair has mixed derivatives through order four."
            ),
            "scope": (
                "Rows five through ten share one Faddeev-LeVerrier inverse and one "
                "normalized bivariate recurrence DAG without weakening rowwise roots."
            ),
        }
    except (
        KeyError,
        TypeError,
        ValueError,
        QuarticRows5To10ArithmeticExpansionError,
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
                "full_lower_Jacobians_materialized": 0,
                "full_component_remainders_proved": 0,
                "H7_closures": 0,
                "rejected": 0,
            },
        }
    return {**body, "content_sha256": _content_hash(body)}


def write_quartic_rows5_10_arithmetic_expansion_campaign(
    result: dict[str, Any], output: Path
) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    path = output / "campaign.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
