from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .quartic_principal_arithmetic_shared import (
    build_principal_source_arithmetic_packet,
)
from .quartic_row0_arithmetic_expansion_campaign import (
    _candidate_records,
    _content_hash,
    _content_hash_matches,
)

SCHEMA_VERSION = "sigma-quartic-full-source-jacobian-arithmetic-campaign-1.0"
ROW_STATUSES = (
    "pass_all_12_row0_arithmetic_materialized_other_rows_fail_closed",
    "pass_all_12_rows0_1_arithmetic_materialized_other_rows_fail_closed",
    "pass_all_12_rows0_2_arithmetic_materialized_other_rows_fail_closed",
    "pass_all_12_rows0_3_arithmetic_materialized_other_rows_fail_closed",
    "pass_all_12_rows0_4_arithmetic_materialized_other_rows_fail_closed",
    "pass_all_12_all_lower_rows_arithmetic_mixed_tensor_fail_closed",
)


class QuarticFullSourceJacobianArithmeticError(ValueError):
    """Raised when the full entrywise source-Jacobian bridge is not exact."""


def _packet(campaign: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    keys = [
        key
        for key in campaign
        if key.startswith("common_") and key.endswith("_arithmetic_packet")
    ]
    if len(keys) != 1:
        raise QuarticFullSourceJacobianArithmeticError(
            "row arithmetic packet is missing or ambiguous"
        )
    return keys[0], campaign[keys[0]]


def _lower_entries(row_campaigns: tuple[dict[str, Any], ...]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for campaign in row_campaigns:
        packet_key, packet = _packet(campaign)
        lower_keys = [key for key in packet if key.startswith("lower_Jacobian")]
        if len(lower_keys) != 1:
            raise QuarticFullSourceJacobianArithmeticError(
                "lower Jacobian root packet is missing or ambiguous"
            )
        dag = packet["arithmetic_dag"]
        if dag.get("node_count") != len(dag.get("nodes", [])):
            raise QuarticFullSourceJacobianArithmeticError(
                "lower arithmetic DAG node count mismatch"
            )
        for item in packet[lower_keys[0]]:
            root = int(item["arithmetic_root"])
            if not 0 <= root < int(dag["node_count"]):
                raise QuarticFullSourceJacobianArithmeticError(
                    "lower arithmetic root is outside its DAG"
                )
            entries.append(
                {
                    "source_row": int(item["output_row"]),
                    "coordinate_column": int(item["column"]),
                    "coordinate_atom": item["atom"],
                    "family": "lower",
                    "arithmetic_dag_sha256": dag["content_sha256"],
                    "arithmetic_root": root,
                    "source_campaign_sha256": campaign["content_sha256"],
                    "source_packet": packet_key,
                }
            )
    return entries


def _validate_row_chain(row_campaigns: tuple[dict[str, Any], ...]) -> None:
    if tuple(item.get("status") for item in row_campaigns) != ROW_STATUSES:
        raise QuarticFullSourceJacobianArithmeticError(
            "row campaign prerequisite status mismatch"
        )
    for index in range(1, 5):
        if row_campaigns[index].get("upstream_sha256", {}).get(
            f"row{index - 1}_arithmetic"
        ) != row_campaigns[index - 1].get("content_sha256"):
            raise QuarticFullSourceJacobianArithmeticError(
                "row arithmetic provenance chain mismatch"
            )
    if row_campaigns[5].get("upstream_sha256", {}).get(
        "row4_arithmetic"
    ) != row_campaigns[4].get("content_sha256"):
        raise QuarticFullSourceJacobianArithmeticError(
            "rows5-10 arithmetic provenance mismatch"
        )


def _certify_candidate(
    principal: dict[str, Any],
    rows: tuple[dict[str, Any], ...],
    principal_packet: dict[str, Any],
    manifest_sha256: str,
) -> dict[str, Any]:
    candidate_id = str(principal.get("candidate_id"))
    if any(
        row.get("candidate_id") != candidate_id
        or row.get("coefficients") != principal.get("coefficients")
        for row in rows
    ):
        raise QuarticFullSourceJacobianArithmeticError(
            "candidate identity mismatch"
        )
    identity = principal.get("principal_composed_identity", {})
    if not identity.get("proved") or identity.get("entry_residuals_proved_zero") != 3025:
        raise QuarticFullSourceJacobianArithmeticError(
            "physical pencil/J identity is not proved"
        )
    basis = principal["basis_and_injection_provenance"]
    physical = principal_packet["physical_provenance"]
    if (
        basis.get("coordinate_atom_basis_sha256")
        != physical["coordinate_atom_basis_sha256"]
        or basis.get("principal_jet_injection_sha256")
        != physical["principal_jet_injection_sha256"]
    ):
        raise QuarticFullSourceJacobianArithmeticError(
            "candidate basis/J provenance mismatch"
        )
    chunk_packet = principal["source_jacobian_chunk_packet"]
    if (
        chunk_packet.get("unspecialized_physical_block_sha256")
        != physical["unspecialized_physical_block_sha256"]
        or chunk_packet.get("completed_entries") != 1089
    ):
        raise QuarticFullSourceJacobianArithmeticError(
            "principal chunk provenance mismatch"
        )
    return {
        "schema_version": "sigma-quartic-full-source-jacobian-arithmetic-certificate-1.0",
        "status": "pass_full_11x153_entrywise_arithmetic_mixed_tensors_fail_closed",
        "candidate_id": candidate_id,
        "coefficients": principal["coefficients"],
        "provenance": {
            "principal_chunk_packet_sha256": chunk_packet["content_sha256"],
            "principal_arithmetic_dag_sha256": principal_packet["arithmetic_dag"][
                "content_sha256"
            ],
            "full_entry_manifest_sha256": manifest_sha256,
            "state_basis_sha256": basis["state_basis_sha256"],
            "coordinate_atom_basis_sha256": basis[
                "coordinate_atom_basis_sha256"
            ],
            "principal_jet_injection_sha256": basis[
                "principal_jet_injection_sha256"
            ],
            "row_artifact_sha256": [row["campaign_hash"] for row in rows],
            "row_arithmetic_dag_sha256": [row["dag_hash"] for row in rows],
        },
        "source_Jacobian_shape": [11, 153],
        "lower_entries_entrywise_arithmetic": 594,
        "principal_entries_entrywise_arithmetic": 1089,
        "total_entries_entrywise_arithmetic": 1683,
        "full_11x153_source_Jacobian_entrywise_materialized": True,
        "physical_pencil_J_identity_entry_residuals_zero": 3025,
        "physical_pencil_J_identity_proved": True,
        "full_component_Frechet_tensors_orders_2_to_4_complete": False,
        "paralinearization_remainder_bound_proved": False,
        "H7_derivative_loss_resolved": False,
        "full_H7_commutator_closed": False,
        "global_dyadic_summation_applied": False,
        "remaining_gate": (
            "materialize every required component of D2F,D3F,D4F and prove the "
            "paralinearization remainder before any H7 closure"
        ),
    }


def run_quartic_full_source_jacobian_arithmetic_campaign(
    principal_campaign: dict[str, Any],
    row0_campaign: dict[str, Any],
    row1_campaign: dict[str, Any],
    row2_campaign: dict[str, Any],
    row3_campaign: dict[str, Any],
    row4_campaign: dict[str, Any],
    rows5_10_campaign: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        rows = (
            row0_campaign,
            row1_campaign,
            row2_campaign,
            row3_campaign,
            row4_campaign,
            rows5_10_campaign,
        )
        if config.get("schema_version") != SCHEMA_VERSION:
            raise QuarticFullSourceJacobianArithmeticError(
                "unsupported campaign schema_version"
            )
        if principal_campaign.get("status") != (
            "pass_all_12_complete_unspecialized_principal_source_jacobians_"
            "remainder_fail_closed"
        ):
            raise QuarticFullSourceJacobianArithmeticError(
                "principal campaign prerequisite status mismatch"
            )
        if not all(_content_hash_matches(item) for item in (principal_campaign, *rows)):
            raise QuarticFullSourceJacobianArithmeticError(
                "campaign content hash mismatch"
            )
        _validate_row_chain(rows)
        if row0_campaign.get("upstream_sha256", {}).get(
            "principal_source"
        ) != principal_campaign.get("content_sha256"):
            raise QuarticFullSourceJacobianArithmeticError(
                "row0/principal provenance mismatch"
            )
        if (
            int(config["dynamic_source_rows"]) != 11
            or int(config["coordinate_atom_columns"]) != 153
            or int(config["lower_entries"]) != 594
            or int(config["principal_entries"]) != 1089
        ):
            raise QuarticFullSourceJacobianArithmeticError(
                "unsupported full source-Jacobian contract"
            )
        if bool(config.get("declare_component_remainder_proved", False)):
            raise QuarticFullSourceJacobianArithmeticError(
                "component remainder cannot be declared from first derivatives"
            )
        principal_records = _candidate_records(principal_campaign)
        row_records = tuple(_candidate_records(item) for item in rows)
        candidate_ids = set(principal_records)
        if len(candidate_ids) != int(config.get("expected_candidate_count", 12)) or any(
            set(records) != candidate_ids for records in row_records
        ):
            raise QuarticFullSourceJacobianArithmeticError("candidate-set mismatch")
        reference = principal_records[min(candidate_ids)]
        basis = reference["basis_and_injection_provenance"]
        physical_hash = principal_campaign[
            "generic_unspecialized_source_jacobian_control"
        ]["unspecialized_block_extraction"]["block_content_sha256"]
        principal_packet = build_principal_source_arithmetic_packet(
            physical_hash,
            basis["coordinate_atom_basis_sha256"],
            basis["principal_jet_injection_sha256"],
        )
        allowed = set(principal_packet["arithmetic_dag"]["allowed_operations"])
        actual = {
            node["op"] for node in principal_packet["arithmetic_dag"]["nodes"]
        }
        if actual - allowed:
            raise QuarticFullSourceJacobianArithmeticError(
                "non-arithmetic operation leaked into principal DAG"
            )
        lower = _lower_entries(rows)
        principal_entries = [
            {
                **item,
                "family": "principal",
                "arithmetic_dag_sha256": principal_packet["arithmetic_dag"][
                    "content_sha256"
                ],
            }
            for item in principal_packet["entries"]
        ]
        manifest_entries = lower + principal_entries
        positions = {
            (item["source_row"], item["coordinate_column"])
            for item in manifest_entries
        }
        expected_positions = {
            (row, column) for row in range(11) for column in range(153)
        }
        if len(lower) != 594 or len(principal_entries) != 1089 or positions != expected_positions:
            raise QuarticFullSourceJacobianArithmeticError(
                "full entrywise source-Jacobian position coverage failed"
            )
        manifest_body = {
            "schema_version": "sigma-full-11x153-arithmetic-entry-manifest-1.0",
            "shape": [11, 153],
            "entries": manifest_entries,
            "lower_entry_count": len(lower),
            "principal_entry_count": len(principal_entries),
            "total_entry_count": len(manifest_entries),
        }
        manifest = {**manifest_body, "content_sha256": _content_hash(manifest_body)}
        row_refs = []
        for row_campaign in rows:
            packet_key, packet = _packet(row_campaign)
            row_refs.append(
                {
                    "campaign_hash": row_campaign["content_sha256"],
                    "packet_key": packet_key,
                    "dag_hash": packet["arithmetic_dag"]["content_sha256"],
                }
            )
        certificates = [
            _certify_candidate(
                principal_records[candidate_id],
                tuple(records[candidate_id] | refs for records, refs in zip(row_records, row_refs, strict=True)),
                principal_packet,
                manifest["content_sha256"],
            )
            for candidate_id in sorted(candidate_ids)
        ]
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": "pass_all_12_full_11x153_entrywise_arithmetic_mixed_tensors_fail_closed",
            "errors": [],
            "upstream_sha256": {
                "principal_source": principal_campaign["content_sha256"],
                "row_arithmetic_artifacts": [item["content_sha256"] for item in rows],
            },
            "config_sha256": _content_hash(config),
            "physical_pencil_J_identity": {
                "identity": "D_Y E55 J_153x55(xi)=iP55(Y,xi)",
                "entry_residuals_proved_zero": 3025,
                "proved": True,
                "source_campaign_sha256": principal_campaign["content_sha256"],
            },
            "common_principal_arithmetic_packet": principal_packet,
            "common_full_entry_manifest": manifest,
            "counts": {
                "selected": len(certificates),
                "lower_entries_per_candidate": 594,
                "principal_entries_per_candidate": 1089,
                "full_source_entries_per_candidate": 1683,
                "full_source_jacobians_materialized": len(certificates),
                "component_remainders_proved": 0,
                "H7_closures": 0,
                "rejected": 0,
            },
            "certificates": certificates,
            "claim": (
                "All 11x153 solved-source Jacobian entries have exact arithmetic roots "
                "bound to the physical pencil, canonical coordinate basis, and sparse J."
            ),
            "scope": (
                "This closes first-derivative component materialization only; complete "
                "D2-D4 tensors, the Bony remainder, and H7 summation remain fail-closed."
            ),
        }
    except (
        KeyError,
        TypeError,
        ValueError,
        QuarticFullSourceJacobianArithmeticError,
    ) as error:
        errors.append(str(error))
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": "reject",
            "errors": errors,
            "certificates": [],
            "counts": {
                "selected": 0,
                "lower_entries_per_candidate": 0,
                "principal_entries_per_candidate": 0,
                "full_source_entries_per_candidate": 0,
                "full_source_jacobians_materialized": 0,
                "component_remainders_proved": 0,
                "H7_closures": 0,
                "rejected": 0,
            },
        }
    return {**body, "content_sha256": _content_hash(body)}


def write_quartic_full_source_jacobian_arithmetic_campaign(
    result: dict[str, Any], output: Path
) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    path = output / "campaign.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
