"""Test whether the registered principal chunks admit a D2F extension."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from functools import cache
from pathlib import Path
from typing import Any

import sympy as sp

from .quartic_unspecialized_source_jacobian_campaign import (
    _unspecialized_principal_blocks,
)

CONFIG_SCHEMA = "sigma-quartic-scalar-hessian-d2-integrability-config-1.0"
RESULT_SCHEMA = "sigma-quartic-scalar-hessian-d2-integrability-gate-1.0"
CAMPAIGN_ID = "quartic-scalar-hessian-d2-integrability-001"
FIRST_BLOCKER = (
    "typed_coordinate_to_block_Frechet_map_or_covariant_connection_terms_"
    "restoring_Schwarz_integrability_not_registered"
)
EXPECTED_DATA_SEALS = {
    "observations_opened": False,
    "solar_system_inputs_opened": False,
    "cosmology_inputs_opened": False,
    "paid_llm_calls": False,
    "live_SQLite_opened": False,
    "GPU_execution_used": False,
}
EXPECTED_PREDECESSORS = {
    "full_tensor_reconciliation": {
        "path": "runs/physics-language/quartic-full-tensor-good-unknown-reconciliation-gate/campaign.json",
        "file_sha256": "cf7957c2efad52a1fa91761fc6259e17a58011cc6093365f9e86e8e7eea0dfd6",
        "content_sha256": "9994df86948a4419dd999b66610e9fea847dece6d5300f68152e942ffb2b87c8",
    },
    "full_source_jacobian": {
        "path": "runs/physics-language/quartic-full-source-jacobian-arithmetic-campaign/campaign.json",
        "file_sha256": "e893ebcaef464b958516279c557382fb76ecdb0fd542b3e3fed6a347076fcdae",
        "content_sha256": "1707b7258fd434f68b06c7af6bc447b4136624b9916992df8b412e048ab6538a",
    },
    "unspecialized_principal_chunks": {
        "path": "runs/physics-language/quartic-unspecialized-source-jacobian-campaign/campaign.json",
        "file_sha256": "8ecae346f75ba5bbeb266e486b96f48a0c76387513ff92d20d1bc68d8ecef22b",
        "content_sha256": "b60dbbb191f43d84d3d9c9e44e4adf70e4e7d729143905561b695cfabcaa7c72",
    },
    "representative_d2": {
        "path": "runs/physics-language/quartic-high-atom-d2-good-unknown-campaign/campaign.json",
        "file_sha256": "5848e62c811baf4a005e821d73c3dcc6d29a285fa2be57cfbe6842b56dfd3513",
        "content_sha256": "5b6a5c43d9e22c2780f3987e3271b8c863c802129b3837777da246a5d635b466",
    },
    "two_channel_slice": {
        "path": "runs/physics-language/quartic-two-channel-good-unknown-slice-campaign/campaign.json",
        "file_sha256": "b83dc512fe3c3679c9013b23538513685d6fae76a493b01fb97330507f43ba06",
        "content_sha256": "fa3fef69a2f2a5443ad3b2a4f9c47aa8fcf617e2525382e698cd58a9198f60d4",
    },
}
FAMILY_SPECS = (
    ("s01", 0, 1, "B_i", 0, 0, 1),
    ("s02", 0, 2, "B_i", 1, 0, 1),
    ("s03", 0, 3, "B_i", 2, 0, 1),
    ("s11", 1, 1, "C_ij", 0, 0, 1),
    ("s12", 1, 2, "C_ij", 0, 1, 2),
    ("s13", 1, 3, "C_ij", 0, 2, 2),
    ("s22", 2, 2, "C_ij", 1, 1, 1),
    ("s23", 2, 3, "C_ij", 1, 2, 2),
    ("s33", 3, 3, "C_ij", 2, 2, 1),
)


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
        raise ValueError("scalar-Hessian D2 path escapes repository") from error
    return path


def _load_bound(root: Path, binding: Mapping[str, Any]) -> dict[str, Any]:
    if set(binding) != {"path", "file_sha256", "content_sha256"}:
        raise ValueError("scalar-Hessian predecessor binding shape changed")
    path = _inside(root, str(binding["path"]))
    if not path.is_file() or _file_sha(path) != binding["file_sha256"]:
        raise ValueError("scalar-Hessian predecessor file hash mismatch")
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        value.get("content_sha256") != binding["content_sha256"]
        or _content_sha(value) != binding["content_sha256"]
    ):
        raise ValueError("scalar-Hessian predecessor content hash mismatch")
    return value


def _validate_config(config: Mapping[str, Any]) -> None:
    if (
        set(config)
        != {
            "schema_version",
            "campaign_id",
            "output_path",
            "predecessors",
            "slice_contract",
            "policies",
            "seals",
        }
        or config.get("schema_version") != CONFIG_SCHEMA
        or config.get("campaign_id") != CAMPAIGN_ID
        or config.get("output_path")
        != "runs/physics-language/quartic-scalar-hessian-d2-integrability-gate/campaign.json"
        or config.get("predecessors") != EXPECTED_PREDECESSORS
        or config.get("slice_contract")
        != {
            "source_outputs": 11,
            "coordinate_atoms": 153,
            "scalar_hessian_directions": 9,
            "acceleration_free_second_atoms": 99,
            "reference": "flat_zero_jet_M2_1",
            "schwarz_subslice": "scalar_hessian_field10_by_scalar_hessian_field10",
        }
        or config.get("policies")
        != {
            "naive_chunk_differentiation_admission": "require_schwarz_integrability",
            "full_D2_promotion": "fail_closed",
            "good_unknown_promotion": "fail_closed",
            "global_H7": "fail_closed",
            "lifespan": "fail_closed",
            "candidate_rejection": "forbidden",
        }
        or config.get("seals") != EXPECTED_DATA_SEALS
    ):
        raise ValueError("scalar-Hessian D2 config boundary changed")


def _records(value: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = value.get("certificates", value.get("candidate_records", []))
    result = {str(row["candidate_id"]): row for row in rows if isinstance(row, Mapping)}
    if len(rows) != 12 or len(result) != 12:
        raise ValueError("scalar-Hessian candidate set changed")
    return result


def _validate_predecessors(values: Mapping[str, Mapping[str, Any]]) -> None:
    if (
        values["full_tensor_reconciliation"].get("decision")
        != "representative_slice_cancelled_full_D2_identity_blocked"
        or values["full_source_jacobian"].get("status")
        != "pass_all_12_full_11x153_entrywise_arithmetic_mixed_tensors_fail_closed"
        or values["unspecialized_principal_chunks"].get("status")
        != "pass_all_12_complete_unspecialized_principal_source_jacobians_remainder_fail_closed"
        or values["representative_d2"].get("status")
        != "pass_all_12_exact_representative_D2_obstructions_named_good_unknown_cancellation_refuted_global_H7_fail_closed"
        or values["two_channel_slice"].get("status")
        != "pass_all_12_exact_two_channel_s01_slice_identities_induced_commutators_global_H7_fail_closed"
    ):
        raise ValueError("scalar-Hessian predecessor status changed")
    ids = [set(_records(value)) for value in values.values()]
    if any(candidate_ids != ids[0] for candidate_ids in ids[1:]):
        raise ValueError("scalar-Hessian predecessor candidates disagree")


def _entries(matrix: sp.MatrixBase) -> list[dict[str, Any]]:
    return [
        {"output_row": row, "high_field": column, "value": str(sp.factor(matrix[row, column]))}
        for row in range(matrix.rows)
        for column in range(matrix.cols)
        if matrix[row, column] != 0
    ]


@cache
def _generic_packet() -> dict[str, Any]:
    blocks = _unspecialized_principal_blocks()
    data = blocks["data"]
    zero = {
        symbol: 0
        for symbol in list(data["gradient_lower"])
        + list(data["hessian_lower"])
        + list(data["einstein_upper"])
    }
    zero[data["m2"]] = 1
    zero[data["c20"]] = data["c20"]
    inverse = blocks["A"].subs(zero).inv()
    chunks: dict[str, sp.MatrixBase] = {}
    directions: dict[str, sp.Symbol] = {}
    for name, left, right, kind, first, second, multiplicity in FAMILY_SPECS:
        base = blocks[kind][first] if kind == "B_i" else blocks[kind][first][second]
        chunks[name] = multiplicity * base
        directions[name] = data["hessian_lower"][left, right]
    matrices: dict[tuple[str, str], sp.MatrixBase] = {}
    packets = []
    for low_name, *_ in FAMILY_SPECS:
        direction = directions[low_name]
        derivative_a = sp.diff(blocks["A"], direction).subs(zero)
        for high_name, *_ in FAMILY_SPECS:
            chunk = chunks[high_name]
            matrix = (
                inverse * derivative_a * inverse * chunk.subs(zero)
                - inverse * sp.diff(chunk, direction).subs(zero)
            ).applyfunc(sp.factor)
            matrices[(low_name, high_name)] = matrix
            nonzero = _entries(matrix)
            packets.append(
                {
                    "low_direction": f"{low_name}[10]",
                    "high_family": high_name,
                    "entry_shape": [11, 11],
                    "entry_count": 121,
                    "nonzero_entries": nonzero,
                    "nonzero_count": len(nonzero),
                    "packet_sha256": _sha(nonzero),
                }
            )
    residuals = []
    failed_pairs = []
    for left_name, *_ in FAMILY_SPECS:
        for right_name, *_ in FAMILY_SPECS:
            residual = (
                matrices[(left_name, right_name)][:, 10] - matrices[(right_name, left_name)][:, 10]
            ).applyfunc(sp.factor)
            entries = [
                {"output_row": row, "value": str(residual[row])}
                for row in range(11)
                if residual[row] != 0
            ]
            if entries:
                failed_pairs.append([left_name, right_name])
                residuals.append(
                    {
                        "left_atom": f"{left_name}[10]",
                        "right_atom": f"{right_name}[10]",
                        "nonzero_residuals": entries,
                        "residual_sha256": _sha(entries),
                    }
                )
    return {
        "alpha": data["alpha"],
        "c20": data["c20"],
        "reference_A_determinant": str(sp.factor(blocks["A"].subs(zero).det())),
        "generic_blocks": packets,
        "generic_schwarz_residuals": residuals,
        "failed_ordered_family_pairs": failed_pairs,
        "unspecialized_block_sha256": blocks["content_sha256"],
    }


def _specialize(packet: Mapping[str, Any], coefficients: Mapping[str, Any]) -> dict[str, Any]:
    substitutions = {
        packet["alpha"]: sp.sympify(coefficients["a10"]),
        packet["c20"]: sp.sympify(coefficients["c20"]),
    }
    symbols = {"alpha": packet["alpha"], "c20": packet["c20"]}
    blocks = []
    for block in packet["generic_blocks"]:
        entries = [
            {
                **entry,
                "value": str(
                    sp.factor(sp.sympify(entry["value"], locals=symbols).subs(substitutions))
                ),
            }
            for entry in block["nonzero_entries"]
        ]
        blocks.append(
            {
                **{
                    key: value
                    for key, value in block.items()
                    if key not in {"nonzero_entries", "packet_sha256"}
                },
                "nonzero_entries": entries,
                "packet_sha256": _sha(entries),
            }
        )
    residuals = []
    for residual in packet["generic_schwarz_residuals"]:
        entries = [
            {
                **entry,
                "value": str(
                    sp.factor(sp.sympify(entry["value"], locals=symbols).subs(substitutions))
                ),
            }
            for entry in residual["nonzero_residuals"]
        ]
        residuals.append(
            {
                **{
                    key: value
                    for key, value in residual.items()
                    if key not in {"nonzero_residuals", "residual_sha256"}
                },
                "nonzero_residuals": entries,
                "residual_sha256": _sha(entries),
            }
        )
    manifest = {
        "blocks": blocks,
        "closed_world_entries": 9801,
        "nonzero_entries": sum(item["nonzero_count"] for item in blocks),
    }
    schwarz = {
        "ordered_atom_pairs_checked": 81,
        "vector_entries_checked": 891,
        "failed_ordered_family_pairs": packet["failed_ordered_family_pairs"],
        "failed_ordered_family_pair_count": len(packet["failed_ordered_family_pairs"]),
        "nonzero_residual_entries": sum(len(item["nonzero_residuals"]) for item in residuals),
        "residuals": residuals,
    }
    return {
        "manifest": {**manifest, "content_sha256": _sha(manifest)},
        "schwarz": {**schwarz, "content_sha256": _sha(schwarz)},
    }


def _candidate_records(predecessors: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    maps = {label: _records(value) for label, value in predecessors.items()}
    packet = _generic_packet()
    result = []
    for candidate_id in sorted(maps["full_tensor_reconciliation"]):
        coefficients = maps["representative_d2"][candidate_id]["coefficients"]
        if any(
            rows[candidate_id].get("coefficients") != coefficients
            for label, rows in maps.items()
            if label != "full_tensor_reconciliation"
        ):
            raise ValueError("scalar-Hessian candidate coefficients disagree")
        specialized = _specialize(packet, coefficients)
        representative = next(
            item
            for item in specialized["manifest"]["blocks"]
            if item["low_direction"] == "s01[10]" and item["high_family"] == "s01"
        )
        prior_entries = maps["two_channel_slice"][candidate_id]["principal_slice_identity"][
            "source_obstruction_entries"
        ]
        if representative["nonzero_entries"] != [
            {"output_row": item["row"], "high_field": item["column"], "value": item["value"]}
            for item in prior_entries
        ]:
            raise ValueError("scalar-Hessian representative slice replay changed")
        if specialized["schwarz"]["failed_ordered_family_pair_count"] == 0:
            raise ValueError("scalar-Hessian expected integrability obstruction vanished")
        result.append(
            {
                "candidate_id": candidate_id,
                "coefficients": coefficients,
                "registered_chunk_extension": specialized["manifest"],
                "schwarz_integrability": specialized["schwarz"],
                "representative_four_entry_slice_replayed": True,
                "naive_chunk_extension_admitted_as_D2F": False,
                "candidate_decision": "blocked",
                "candidate_rejection_authorized": False,
                "first_blocker": FIRST_BLOCKER,
            }
        )
    return result


def _expected_body(
    config_path: Path, root: Path, predecessors: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    records = _candidate_records(predecessors)
    packet = _generic_packet()
    return {
        "schema_version": RESULT_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "decision": "naive_scalar_hessian_chunk_extension_fails_Schwarz_integrability_candidates_blocked",
        "decision_counts": {"pass": 0, "reject": 0, "blocked": 12},
        "gate_counts": {
            "selected": 12,
            "ordered_D2_entries_materialized_per_candidate": 9801,
            "ordered_D2_entries_materialized_total": 117612,
            "nonzero_chunk_derivatives_per_candidate": 186,
            "nonzero_chunk_derivatives_total": 2232,
            "scalar_scalar_Schwarz_entries_checked_per_candidate": 891,
            "failed_ordered_family_pairs_per_candidate": 24,
            "nonzero_Schwarz_residuals_per_candidate": 30,
            "integrable_candidate_manifests": 0,
            "representative_four_entry_slices_replayed": 12,
            "full_ordered_D2_manifests_admitted": 0,
            "global_H7_closures": 0,
            "lifespans_proved": 0,
        },
        "first_blocker": FIRST_BLOCKER,
        "theorem": {
            "name": "registered_principal_chunk_naive_D2_extension_integrability_obstruction",
            "reference": "flat zero jet with M2=1; candidate a10,c20 retained exactly",
            "candidate_bound_submanifest_shape": [11, 9, 99],
            "candidate_bound_submanifest_entries": 9801,
            "full_ordered_D2_target_entries": 257499,
            "coverage_fraction": "11/289",
            "generic_nonzero_entries": 186,
            "Schwarz_test": "D_(s_ab[10])D_(s_cd[10])F equals its transposed atom order for every output row",
            "failed_ordered_family_pairs": packet["failed_ordered_family_pairs"],
            "failed_ordered_family_pair_count": 24,
            "nonzero_residual_entries": 30,
            "conclusion": (
                "Direct differentiation of the registered -A^-1 B_i and -m A^-1 C_ij chunk formulas cannot yet be admitted as a D2F tensor: its scalar-Hessian field-10 subslice violates Schwarz symmetry. This is an obstruction to the naive extension, not a proof that no covariant source or corrected coordinate map exists."
            ),
        },
        "exact_controls": {
            "representative_s01_s01_replay": {
                "entries_per_candidate": 4,
                "matches_predecessor": True,
                "passed": True,
            },
            "ignore_Schwarz_residual": {"would_admit_nonintegrable_D2F": True, "rejected": True},
            "promote_9801_entries_to_full_257499": {"missing_entries": 247698, "rejected": True},
            "infer_no_covariant_completion": {
                "connection_or_coordinate_terms_unregistered": True,
                "rejected": True,
            },
            "infer_candidate_rejection": {"theory_level_evidence": False, "rejected": True},
        },
        "candidate_records": records,
        "secondary_blockers": [
            "remaining_247698_ordered_D2_entries_not_materialized",
            "complete_high_atom_good_unknown_identity_not_registered",
            "induced_TC1_TC2_TC3_TC5_bounds_not_closed",
            "B7_global_H7_dyadic_summation_and_lifespan_not_closed",
        ],
        "claim_seals": {
            "naive_chunk_extension_obstructed": True,
            "full_ordered_D2_tensor_registered": False,
            "corrected_covariant_D2_tensor_ruled_out": False,
            "full_high_atom_good_unknown_identity_proved": False,
            "global_H7_energy_closed": False,
            "nonlinear_lifespan_proved": False,
            "candidate_theory_rejected": False,
            "observational_claim_made": False,
        },
        "data_seals": EXPECTED_DATA_SEALS,
        "scope": (
            "exact candidate-bound 11x9x99 flat-reference derivative submanifest and Schwarz-integrability obstruction for the naive registered A/B/C chunk extension; no full D2F, corrected covariant source no-go, good-unknown closure, H7, lifespan, or theory rejection"
        ),
        "source_bindings": {
            **EXPECTED_PREDECESSORS,
            "config": {
                "path": config_path.relative_to(root).as_posix(),
                "file_sha256": _file_sha(config_path),
            },
            "source": {
                "path": "src/sigma_theory_compiler/quartic_scalar_hessian_d2_integrability_gate.py",
                "file_sha256": _file_sha(
                    root
                    / "src/sigma_theory_compiler/quartic_scalar_hessian_d2_integrability_gate.py"
                ),
            },
            "test": {
                "path": "tests/test_quartic_scalar_hessian_d2_integrability_gate.py",
                "file_sha256": _file_sha(
                    root / "tests/test_quartic_scalar_hessian_d2_integrability_gate.py"
                ),
            },
        },
    }


def _validate_result(result: Mapping[str, Any], *, root: Path | None = None) -> None:
    validation_root = (root or Path(__file__).resolve().parents[2]).resolve()
    if result.get("content_sha256") != _content_sha(result):
        raise ValueError("scalar-Hessian D2 content hash changed")
    config_path = _inside(
        validation_root, "configs/backgrounds/quartic_scalar_hessian_d2_integrability_gate.json"
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    bindings = result.get("source_bindings", {})
    if not isinstance(bindings, Mapping) or set(bindings) != {
        *EXPECTED_PREDECESSORS,
        "config",
        "source",
        "test",
    }:
        raise ValueError("scalar-Hessian D2 source binding set changed")
    predecessors = {
        label: _load_bound(validation_root, EXPECTED_PREDECESSORS[label])
        for label in EXPECTED_PREDECESSORS
    }
    _validate_predecessors(predecessors)
    for label in ("config", "source", "test"):
        binding = bindings[label]
        if (
            set(binding) != {"path", "file_sha256"}
            or _file_sha(_inside(validation_root, binding["path"])) != binding["file_sha256"]
        ):
            raise ValueError("scalar-Hessian D2 local binding changed")
    expected = _expected_body(config_path, validation_root, predecessors)
    if {key: value for key, value in result.items() if key != "content_sha256"} != expected:
        raise ValueError("scalar-Hessian D2 result boundary changed")


def build_gate(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    root = config_path.parents[2]
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    predecessors = {
        label: _load_bound(root, binding) for label, binding in config["predecessors"].items()
    }
    _validate_predecessors(predecessors)
    body = _expected_body(config_path, root, predecessors)
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
