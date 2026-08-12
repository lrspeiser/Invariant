"""Reconcile the finite-Sobolev witness with the exact two-channel slice."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from fractions import Fraction
from pathlib import Path
from typing import Any

CONFIG_SCHEMA = "sigma-quartic-full-tensor-good-unknown-reconciliation-config-1.0"
RESULT_SCHEMA = "sigma-quartic-full-tensor-good-unknown-reconciliation-gate-1.0"
CAMPAIGN_ID = "quartic-full-tensor-good-unknown-reconciliation-001"
FIRST_BLOCKER = (
    "complete_candidate_bound_ordered_D2F_component_manifest_and_full_high_atom_"
    "good_unknown_identity_not_registered"
)
EXPECTED_PREDECESSORS = {
    "finite_sobolev_no_go": {
        "path": (
            "runs/physics-language/quartic-finite-sobolev-hierarchy-no-go-"
            "campaign/campaign.json"
        ),
        "file_sha256": "404a869cd676fb57389535c74ff4fea73ec1ae3da43cad1ff74951f817ae1308",
        "content_sha256": "a74fb10a523cef935695d99fff595de7aa51a8ead322171373827c8b6db9288a",
    },
    "full_source_jacobian": {
        "path": (
            "runs/physics-language/quartic-full-source-jacobian-arithmetic-"
            "campaign/campaign.json"
        ),
        "file_sha256": "e893ebcaef464b958516279c557382fb76ecdb0fd542b3e3fed6a347076fcdae",
        "content_sha256": "1707b7258fd434f68b06c7af6bc447b4136624b9916992df8b412e048ab6538a",
    },
    "h7_topology": {
        "path": (
            "runs/physics-language/quartic-h7-paracomposition-topology-"
            "campaign/campaign.json"
        ),
        "file_sha256": "14cdfc8012745d5e8d8206f1572ff1364bccb802aab55a561fc72def6e46813a",
        "content_sha256": "c694af6a9d412cf587f3673fce802e0da0834561e38a2823daebc4a308b43c34",
    },
    "representative_d2": {
        "path": (
            "runs/physics-language/quartic-high-atom-d2-good-unknown-"
            "campaign/campaign.json"
        ),
        "file_sha256": "5848e62c811baf4a005e821d73c3dcc6d29a285fa2be57cfbe6842b56dfd3513",
        "content_sha256": "5b6a5c43d9e22c2780f3987e3271b8c863c802129b3837777da246a5d635b466",
    },
    "two_channel_slice": {
        "path": (
            "runs/physics-language/quartic-two-channel-good-unknown-slice-"
            "campaign/campaign.json"
        ),
        "file_sha256": "b83dc512fe3c3679c9013b23538513685d6fae76a493b01fb97330507f43ba06",
        "content_sha256": "fa3fef69a2f2a5443ad3b2a4f9c47aa8fcf617e2525382e698cd58a9198f60d4",
    },
}
EXPECTED_STATUSES = {
    "full_source_jacobian": (
        "pass_all_12_full_11x153_entrywise_arithmetic_mixed_tensors_fail_closed"
    ),
    "h7_topology": (
        "pass_all_12_H7_atom_topologies_and_recombined_tame_ledgers_high_low_"
        "paraproduct_fail_closed"
    ),
    "representative_d2": (
        "pass_all_12_exact_representative_D2_obstructions_named_good_unknown_"
        "cancellation_refuted_global_H7_fail_closed"
    ),
    "two_channel_slice": (
        "pass_all_12_exact_two_channel_s01_slice_identities_induced_commutators_"
        "global_H7_fail_closed"
    ),
}
EXPECTED_COUNTS = {
    "selected": 12,
    "candidate_pass": 0,
    "candidate_reject": 0,
    "candidate_blocked": 12,
    "full_D1_entries_per_candidate": 1683,
    "closed_world_ordered_D2_target_per_candidate": 257499,
    "complete_ordered_D2_manifests_registered": 0,
    "representative_D2_entries_replayed": 48,
    "unmodified_slice_witnesses_reconciled": 12,
    "two_channel_four_entry_slice_cancellations_proved": 12,
    "full_high_atom_families_closed": 0,
    "all_induced_term_bounds_closed": 0,
    "global_H7_closures": 0,
    "lifespans_proved": 0,
}
EXPECTED_CLAIM_SEALS = {
    "full_ordered_D2_tensor_registered": False,
    "all_99_second_atom_families_cancelled": False,
    "all_low_direction_slices_cancelled": False,
    "all_induced_good_unknown_terms_bounded": False,
    "B7_replaced_by_closed_expression": False,
    "global_H7_energy_closed": False,
    "global_dyadic_summation_applied": False,
    "nonlinear_lifespan_proved": False,
    "candidate_theory_rejected": False,
    "observational_claim_made": False,
}
EXPECTED_DATA_SEALS = {
    "observations_opened": False,
    "solar_system_inputs_opened": False,
    "cosmology_inputs_opened": False,
    "paid_llm_calls": False,
    "live_SQLite_opened": False,
    "GPU_execution_used": False,
}
EXPECTED_RESULT_KEYS = {
    "schema_version",
    "campaign_id",
    "decision",
    "decision_counts",
    "gate_counts",
    "first_blocker",
    "coverage_theorem",
    "exact_controls",
    "candidate_records",
    "secondary_blockers",
    "claim_seals",
    "data_seals",
    "scope",
    "source_bindings",
    "content_sha256",
}


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _content_sha(value: Mapping[str, Any]) -> str:
    return _sha({key: item for key, item in value.items() if key != "content_sha256"})


def _inside(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError("full-tensor reconciliation path escapes repository") from error
    return path


def _bound_artifact(root: Path, binding: Mapping[str, Any]) -> dict[str, Any]:
    if set(binding) != {"path", "file_sha256", "content_sha256"}:
        raise ValueError("full-tensor predecessor binding shape changed")
    path = _inside(root, str(binding["path"]))
    if not path.is_file() or _file_sha(path) != binding["file_sha256"]:
        raise ValueError("full-tensor predecessor file hash mismatch")
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or value.get("content_sha256") != binding["content_sha256"]
        or _content_sha(value) != binding["content_sha256"]
    ):
        raise ValueError("full-tensor predecessor content hash mismatch")
    return value


def _validate_config(config: Mapping[str, Any]) -> None:
    if (
        set(config)
        != {
            "schema_version",
            "campaign_id",
            "output_path",
            "predecessors",
            "tensor_contract",
            "policies",
            "seals",
        }
        or config.get("schema_version") != CONFIG_SCHEMA
        or config.get("campaign_id") != CAMPAIGN_ID
        or config.get("output_path")
        != (
            "runs/physics-language/quartic-full-tensor-good-unknown-"
            "reconciliation-gate/campaign.json"
        )
        or config.get("predecessors") != EXPECTED_PREDECESSORS
        or config.get("tensor_contract")
        != {
            "source_outputs": 11,
            "coordinate_atoms": 153,
            "second_atoms": 99,
            "representative_high_atom": "s01[10]",
            "representative_low_direction": "H_01",
        }
        or config.get("policies")
        != {
            "representative_slice_cancellation": "exact_replay",
            "full_D2_promotion": "fail_closed",
            "induced_terms": "fail_closed",
            "global_H7": "fail_closed",
            "lifespan": "fail_closed",
            "candidate_rejection": "forbidden",
        }
        or config.get("seals") != EXPECTED_DATA_SEALS
    ):
        raise ValueError("full-tensor reconciliation config boundary changed")


def _records(campaign: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = campaign.get("certificates", campaign.get("candidate_records", []))
    result = {
        str(row["candidate_id"]): row
        for row in rows
        if isinstance(row, Mapping) and isinstance(row.get("candidate_id"), str)
    }
    if len(rows) != 12 or len(result) != 12:
        raise ValueError("full-tensor candidate set changed")
    return result


def _fraction_text(value: Fraction) -> str:
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def _slice_entries(a10: str) -> list[dict[str, Any]]:
    alpha = Fraction(a10)
    values = (
        (0, 10, -2 * alpha),
        (4, 10, 8 * alpha),
        (10, 7, -2 * alpha),
        (10, 9, -2 * alpha),
    )
    return [
        {"row": row, "column": column, "value": _fraction_text(value)}
        for row, column, value in values
    ]


def _negate_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "row": row["row"],
            "column": row["column"],
            "value": _fraction_text(-Fraction(str(row["value"]))),
        }
        for row in entries
    ]


def _validate_predecessors(predecessors: Mapping[str, Mapping[str, Any]]) -> None:
    if set(predecessors) != set(EXPECTED_PREDECESSORS):
        raise ValueError("full-tensor predecessor set changed")
    finite = predecessors["finite_sobolev_no_go"]
    if (
        finite.get("decision")
        != "finite_unmodified_Sobolev_hierarchy_refuted_candidates_blocked"
        or finite.get("decision_counts") != {"blocked": 12, "pass": 0, "reject": 0}
        or finite.get("gate_counts", {}).get("full_tensor_cancellations_proved") != 0
    ):
        raise ValueError("finite-Sobolev predecessor boundary changed")
    for label, status in EXPECTED_STATUSES.items():
        if predecessors[label].get("status") != status:
            raise ValueError("full-tensor predecessor status changed")
    full = predecessors["full_source_jacobian"]
    topology = predecessors["h7_topology"]
    d2 = predecessors["representative_d2"]
    two = predecessors["two_channel_slice"]
    if (
        full.get("counts", {}).get("full_source_entries_per_candidate") != 1683
        or full.get("counts", {}).get("component_remainders_proved") != 0
        or any(
            row.get("source_Jacobian_shape") != [11, 153]
            or row.get("full_component_Frechet_tensors_orders_2_to_4_complete")
            is not False
            for row in full.get("certificates", [])
        )
        or topology.get("counts", {}).get("coefficient_high_state_low_branches_closed")
        != 0
        or any(
            row.get("coordinate_atom_topology", {})
            .get("groups", {})
            .get("acceleration_free_second_partial_atoms", {})
            .get("count")
            != 99
            for row in topology.get("certificates", [])
        )
        or d2.get("counts", {}).get("nonzero_obstructions") != 12
        or two.get("counts", {}).get("four_entry_cancellations_proved") != 12
        or two.get("counts", {}).get("full_high_atom_families_closed") != 0
        or two.get("counts", {}).get("all_induced_term_bounds_closed") != 0
        or topology.get("upstream_sha256", {}).get("dyadic_localization")
        != "43dcfcd4ab08a548175e6971f39bcd7c2dd3a15a8817689aa95f10dd2412bcb6"
        or finite.get("source_bindings", {})
        .get("dyadic_localization", {})
        .get("content_sha256")
        != "ce7afcaf428144cc7149dfdd67be5139d09a2e33d3d2bb8a19b867799313f3b5"
    ):
        raise ValueError("full-tensor predecessor semantic boundary changed")


def _candidate_records(predecessors: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    maps = {label: _records(value) for label, value in predecessors.items()}
    candidate_ids = set(maps["finite_sobolev_no_go"])
    if any(set(rows) != candidate_ids for rows in maps.values()):
        raise ValueError("full-tensor candidate identity mismatch")
    result = []
    for candidate_id in sorted(candidate_ids):
        finite = maps["finite_sobolev_no_go"][candidate_id]
        full = maps["full_source_jacobian"][candidate_id]
        topology = maps["h7_topology"][candidate_id]
        d2 = maps["representative_d2"][candidate_id]
        two = maps["two_channel_slice"][candidate_id]
        coefficients = finite.get("coefficients")
        if any(
            row.get("coefficients") != coefficients for row in (full, topology, d2, two)
        ):
            raise ValueError("full-tensor candidate coefficient mismatch")
        if not isinstance(coefficients, Mapping):
            raise TypeError("full-tensor candidate coefficients absent")
        source_entries = _slice_entries(str(coefficients["a10"]))
        correction_entries = _negate_entries(source_entries)
        identity = two.get("principal_slice_identity", {})
        if (
            d2.get("representative_slice", {}).get("component_D2_value")
            != source_entries[0]["value"]
            or identity.get("source_obstruction_entries") != source_entries
            or identity.get("correction_entries") != correction_entries
            or identity.get("after_J_s01_residual_entries") != []
            or identity.get("after_J_s01_residual_zero") is not True
            or identity.get("all_four_nonzero_entries_cancelled") is not True
        ):
            raise ValueError("full-tensor four-entry cancellation replay failed")
        if (
            finite.get("direct_finite_Sobolev_hierarchy_closure") is not False
            or finite.get("full_tensor_cancellation_proved") is not False
            or two.get("connection_to_B7_global_H7", {}).get(
                "representative_full_s01_H01_slice_removed_from_B7"
            )
            is not True
            or two.get("connection_to_B7_global_H7", {}).get("B7_fully_replaced")
            is not False
        ):
            raise ValueError("full-tensor scope reconciliation failed")
        result.append(
            {
                "candidate_id": candidate_id,
                "a10": str(coefficients["a10"]),
                "unmodified_slice": {
                    "finite_Sobolev_growth_multiplier": finite[
                        "absolute_growth_multiplier"
                    ],
                    "representative_D2_value": source_entries[0]["value"],
                    "direct_Hs_estimate_closed": False,
                },
                "two_channel_modified_slice": {
                    "source_entries": [
                        [row["row"], row["column"], row["value"]]
                        for row in source_entries
                    ],
                    "source_matrix_rank": 2,
                    "correction_entries": [
                        [row["row"], row["column"], row["value"]]
                        for row in correction_entries
                    ],
                    "residual_entries": [],
                    "all_four_entries_cancelled": True,
                    "reference_s01_H01_slice_removed": True,
                },
                "coverage_boundary": {
                    "full_D1_entries_registered": 1683,
                    "closed_world_ordered_D2_target": 257499,
                    "complete_ordered_D2_manifest_registered": False,
                    "all_99_second_atom_families_cancelled": False,
                    "all_induced_terms_bounded": False,
                },
                "candidate_decision": "blocked",
                "candidate_rejection_authorized": False,
                "first_blocker": FIRST_BLOCKER,
            }
        )
    return result


def _coverage_theorem() -> dict[str, Any]:
    return {
        "name": "D1_manifest_does_not_determine_full_D2_good_unknown_identity",
        "registered_D1_shape": [11, 153],
        "registered_D1_entries_per_candidate": 1683,
        "closed_world_ordered_D2_shape": [11, 153, 153],
        "closed_world_ordered_D2_entries_per_candidate": 257499,
        "complete_ordered_D2_entries_registered": 0,
        "representative_exact_slice": {
            "high_atom": "s01[10]",
            "low_direction": "H_01",
            "nonzero_source_entries": 4,
            "source_matrix_rank": 2,
            "two_channel_correction_rank": 2,
            "residual_entries": 0,
            "candidate_cancellations": 12,
        },
        "topology_lineage_audit": {
            "bound_topology_snapshot_content_sha256": (
                "c694af6a9d412cf587f3673fce802e0da0834561e38a2823daebc4a308b43c34"
            ),
            "topology_snapshot_dyadic_content_sha256": (
                "43dcfcd4ab08a548175e6971f39bcd7c2dd3a15a8817689aa95f10dd2412bcb6"
            ),
            "current_finite_gate_dyadic_content_sha256": (
                "ce7afcaf428144cc7149dfdd67be5139d09a2e33d3d2bb8a19b867799313f3b5"
            ),
            "current_dyadic_binding_matches": False,
            "policy": (
                "use the exact bound topology snapshot only; do not claim a regenerated "
                "current-dyadic topology lineage"
            ),
        },
        "logical_reconciliation": (
            "the N-growing finite-Sobolev witness is exact for the unmodified variable, "
            "while the registered two-channel modified variable cancels its complete "
            "four-entry principal s01/H01 slice"
        ),
        "non_promotion": (
            "a complete D1 manifest and one cancelled D2 slice do not determine the "
            "remaining ordered D2 tensor, the other high-atom/low-direction identities, "
            "or the induced TC1/TC2/TC3/TC5 bounds"
        ),
    }


def _controls() -> dict[str, Any]:
    return {
        "reverse_two_channel_sign": {
            "surviving_nonzero_entries_per_candidate": 4,
            "residual_multiplier_of_source": 2,
            "rejected": True,
        },
        "promote_full_D1_to_full_D2": {
            "D1_entries": 1683,
            "ordered_D2_target": 257499,
            "missing_complete_D2_manifest": True,
            "rejected": True,
        },
        "promote_one_slice_to_99_second_atom_families": {
            "representative_slices_cancelled": 1,
            "full_high_atom_families_closed": 0,
            "rejected": True,
        },
        "erase_unmodified_finite_Sobolev_no_go": {
            "unmodified_growth_exponent": 1,
            "modified_variable_required_for_slice_cancellation": True,
            "rejected": True,
        },
        "promote_principal_slice_to_global_H7": {
            "induced_terms_without_complete_bounds": ["TC1", "TC2", "TC3", "TC5"],
            "rejected": True,
        },
    }


def _expected_body(
    config: Mapping[str, Any],
    root: Path,
    config_path: Path,
    predecessors: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    source_path = (
        root
        / "src/sigma_theory_compiler/quartic_full_tensor_good_unknown_reconciliation_gate.py"
    ).resolve()
    test_path = root / "tests/test_quartic_full_tensor_good_unknown_reconciliation_gate.py"
    return {
        "schema_version": RESULT_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "decision": "representative_slice_cancelled_full_D2_identity_blocked",
        "decision_counts": {"pass": 0, "reject": 0, "blocked": 12},
        "gate_counts": EXPECTED_COUNTS,
        "first_blocker": FIRST_BLOCKER,
        "coverage_theorem": _coverage_theorem(),
        "exact_controls": _controls(),
        "candidate_records": _candidate_records(predecessors),
        "secondary_blockers": [
            "H7_topology_snapshot_not_resealed_against_current_dyadic_artifact",
            "other_high_atom_and_low_direction_good_unknown_identities_not_registered",
            "induced_TC1_TC2_TC3_TC5_H7_bounds_not_all_closed",
            "B7_global_dyadic_summation_and_lifespan_not_closed",
        ],
        "claim_seals": EXPECTED_CLAIM_SEALS,
        "data_seals": EXPECTED_DATA_SEALS,
        "scope": (
            "exact all-candidate cancellation of the complete four-entry s01/H01 "
            "principal slice and an exact D1-versus-D2 coverage obstruction; no full "
            "high-atom identity, global H7 closure, lifespan, or theory rejection"
        ),
        "source_bindings": {
            **EXPECTED_PREDECESSORS,
            "config": {
                "path": config_path.relative_to(root).as_posix(),
                "file_sha256": _file_sha(config_path),
            },
            "source": {
                "path": source_path.relative_to(root).as_posix(),
                "file_sha256": _file_sha(source_path),
            },
            "test": {
                "path": test_path.relative_to(root).as_posix(),
                "file_sha256": _file_sha(test_path),
            },
        },
    }


def _validate_source_bindings(result: Mapping[str, Any], root: Path) -> None:
    bindings = result.get("source_bindings", {})
    if not isinstance(bindings, Mapping) or set(bindings) != {
        *EXPECTED_PREDECESSORS,
        "config",
        "source",
        "test",
    }:
        raise ValueError("full-tensor source binding set changed")
    for label, expected in EXPECTED_PREDECESSORS.items():
        if bindings.get(label) != expected:
            raise ValueError("full-tensor predecessor binding changed")
        _bound_artifact(root, expected)
    paths = {
        "config": (
            "configs/backgrounds/"
            "quartic_full_tensor_good_unknown_reconciliation_gate.json"
        ),
        "source": (
            "src/sigma_theory_compiler/"
            "quartic_full_tensor_good_unknown_reconciliation_gate.py"
        ),
        "test": "tests/test_quartic_full_tensor_good_unknown_reconciliation_gate.py",
    }
    for label, relative in paths.items():
        binding = bindings.get(label, {})
        if (
            not isinstance(binding, Mapping)
            or set(binding) != {"path", "file_sha256"}
            or binding.get("path") != relative
            or _file_sha(_inside(root, relative)) != binding.get("file_sha256")
        ):
            raise ValueError("full-tensor local source binding changed")


def _validate_result(result: Mapping[str, Any], *, root: Path | None = None) -> None:
    validation_root = (root or Path(__file__).resolve().parents[2]).resolve()
    if result.get("content_sha256") != _content_sha(result):
        raise ValueError("full-tensor content hash changed")
    _validate_source_bindings(result, validation_root)
    config_path = _inside(
        validation_root,
        "configs/backgrounds/quartic_full_tensor_good_unknown_reconciliation_gate.json",
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    predecessors = {
        label: _bound_artifact(validation_root, binding)
        for label, binding in EXPECTED_PREDECESSORS.items()
    }
    _validate_predecessors(predecessors)
    expected = _expected_body(config, validation_root, config_path, predecessors)
    if set(result) != EXPECTED_RESULT_KEYS or {
        key: value for key, value in result.items() if key != "content_sha256"
    } != expected:
        raise ValueError("full-tensor result boundary changed")


def build_gate(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    root = config_path.parents[2]
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    predecessors = {
        label: _bound_artifact(root, binding)
        for label, binding in config["predecessors"].items()
    }
    _validate_predecessors(predecessors)
    body = _expected_body(config, root, config_path, predecessors)
    result = {**body, "content_sha256": _sha(body)}
    _validate_result(result, root=root)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = build_gate(args.config)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output = args.output or args.config.resolve().parents[2] / config["output_path"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
