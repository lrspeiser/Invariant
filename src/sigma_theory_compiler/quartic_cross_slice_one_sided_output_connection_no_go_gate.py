"""Audit arbitrary output rows on only the Pother connection directions."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import sympy as sp

from .quartic_reverse_principal_typed_map_curl_gate import (
    _validate_result as validate_typed_map_curl,
)
from .quartic_scalar_hessian_output_bundle_repair_gate import (
    _validate_result as validate_output_bundle_repair,
)

CONFIG_SCHEMA = "sigma-quartic-cross-slice-one-sided-output-connection-no-go-config-1.0"
RESULT_SCHEMA = "sigma-quartic-cross-slice-one-sided-output-connection-no-go-gate-1.0"
CAMPAIGN_ID = "quartic-cross-slice-one-sided-output-connection-no-go-001"
CONFIG_PATH = (
    "configs/backgrounds/quartic_cross_slice_one_sided_output_connection_no_go_gate.json"
)
OUTPUT_PATH = (
    "runs/physics-language/quartic-cross-slice-one-sided-output-connection-no-go-gate/"
    "campaign.json"
)
SOURCE_PATH = (
    "src/sigma_theory_compiler/quartic_cross_slice_one_sided_output_connection_no_go_gate.py"
)
TEST_PATH = "tests/test_quartic_cross_slice_one_sided_output_connection_no_go_gate.py"
FIRST_BLOCKER = (
    "connection_variations_on_P10_directions_or_corrected_source_extension_needed_for_"
    "one_sided_system_augmented_rank_991_over_coefficient_rank_990"
)
DIAGONAL_P10 = ("s11[10]", "s22[10]", "s33[10]")
ZERO_P10 = ("s01[10]", "s02[10]", "s03[10]", "s12[10]", "s13[10]", "s23[10]")

EXPECTED_PREDECESSORS = {
    "typed_map_curl": {
        "path": "runs/physics-language/quartic-reverse-principal-typed-map-curl-gate/campaign.json",
        "file_sha256": "4e432566b16e44b7d5ca05a2ce6e60b5ebd849e2fe8c88fa6523297f1fc111b4",
        "content_sha256": "79d06514c1dd8fd7933bdc36b19622fc3cce8ddcaf14712f0b908fbe6c9f2664",
    },
    "output_bundle_repair": {
        "path": "runs/physics-language/quartic-scalar-hessian-output-bundle-repair-gate/campaign.json",
        "file_sha256": "e1ae98ebcb3c2739f7c84938d61ce9e7d2d209d4025f54a7d1d499a8495acfdb",
        "content_sha256": "688dcb478b86d44330f8a3623183e91c237bd91f31bd4e91bf5869098175973f",
    },
}
EXPECTED_CONNECTION_CONTRACT = {
    "fixed_connection_directions": "nine_P10_directions_from_predecessor",
    "new_connection_directions": "ninety_Pother_directions",
    "output_rows": 11,
    "new_output_input_rows": 11,
    "system_equations": 8910,
    "system_unknowns": 10890,
}
EXPECTED_POLICIES = {
    "cross_slice_admission": "require_complete_connection_system_consistency",
    "full_D2_promotion": "fail_closed",
    "full_high_atom_identity": "fail_closed",
    "global_H7": "fail_closed",
    "nonlinear_PDE": "fail_closed",
    "lifespan": "fail_closed",
    "candidate_rejection": "forbidden",
}
EXPECTED_SEALS = {
    "observations_opened": False,
    "solar_system_inputs_opened": False,
    "cosmology_inputs_opened": False,
    "paid_llm_calls": False,
    "live_SQLite_opened": False,
    "GPU_execution_used": False,
}
CLAIM_SEALS = {
    "one_sided_arbitrary_output_row_connection_system_classified": True,
    "one_sided_connection_system_inconsistent": True,
    "maximal_declared_compatible_subdomain_repair_constructed": True,
    "two_sided_general_output_connection_classified": False,
    "corrected_source_extension_registered": False,
    "corrected_cross_slice_curl_zero": False,
    "cross_slice_D2F_entries_admitted": False,
    "complete_ordered_D2F_tensor_registered": False,
    "full_high_atom_good_unknown_identity_proved": False,
    "global_H7_energy_closed": False,
    "nonlinear_PDE_closed": False,
    "nonlinear_lifespan_proved": False,
    "candidate_theory_rejected": False,
    "observational_claim_made": False,
}


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _content_sha(value: Mapping[str, Any]) -> str:
    return _sha({key: item for key, item in value.items() if key != "content_sha256"})


def _inside(root: Path, relative: str) -> Path:
    target = (root / relative).resolve()
    if target != root and root not in target.parents:
        raise ValueError("one-sided output connection path escapes project root")
    return target


def _validate_config(value: Mapping[str, Any]) -> None:
    expected = {
        "schema_version": CONFIG_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "output_path": OUTPUT_PATH,
        "predecessors": EXPECTED_PREDECESSORS,
        "connection_contract": EXPECTED_CONNECTION_CONTRACT,
        "policies": EXPECTED_POLICIES,
        "seals": EXPECTED_SEALS,
    }
    if value != expected:
        raise ValueError("one-sided output connection config boundary changed")


def _load_bound(root: Path, binding: Mapping[str, Any]) -> dict[str, Any]:
    if set(binding) != {"path", "file_sha256", "content_sha256"}:
        raise ValueError("one-sided output connection predecessor binding changed")
    path = _inside(root, str(binding["path"]))
    if not path.is_file() or _file_sha(path) != binding["file_sha256"]:
        raise ValueError("one-sided output connection predecessor file hash changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("content_sha256") != binding["content_sha256"]:
        raise ValueError("one-sided output connection predecessor content binding changed")
    if value.get("content_sha256") != _content_sha(value):
        raise ValueError("one-sided output connection predecessor content hash changed")
    return value


def _records(value: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    raw = value.get("candidate_records", [])
    if not isinstance(raw, list):
        raise TypeError("one-sided output connection candidate records missing")
    result = {str(row.get("candidate_id")): row for row in raw if isinstance(row, Mapping)}
    if len(raw) != 12 or len(result) != 12:
        raise ValueError("one-sided output connection candidate set changed")
    return result


def _validate_predecessors(values: Mapping[str, Mapping[str, Any]], root: Path) -> None:
    if set(values) != set(EXPECTED_PREDECESSORS):
        raise ValueError("one-sided output connection predecessor set changed")
    validate_typed_map_curl(values["typed_map_curl"], root=root)
    validate_output_bundle_repair(values["output_bundle_repair"], root=root)
    typed = _records(values["typed_map_curl"])
    repair = _records(values["output_bundle_repair"])
    if set(typed) != set(repair):
        raise ValueError("one-sided output connection predecessor candidates disagree")
    for candidate_id in typed:
        if typed[candidate_id]["coefficients"] != repair[candidate_id]["coefficients"]:
            raise ValueError("one-sided output connection predecessor coefficients disagree")
        one_form = repair[candidate_id]["registered_one_form"]
        if one_form != {
            "shape": [11, 9],
            "rank": 1,
            "nonzero_entries": [
                {"output_row": 10, "domain_direction": atom, "value": "1"}
                for atom in DIAGONAL_P10
            ],
            "image_output_rows": [10],
        }:
            raise ValueError("one-sided output connection registered one-form changed")


def _classify_candidate(row: Mapping[str, Any]) -> dict[str, Any]:
    groups: dict[tuple[str, int], dict[str, sp.Expr]] = defaultdict(dict)
    curl = row["corrected_ordered_curl_manifest"]
    if curl.get("entry_count") != 8910 or curl.get("nonzero_entry_count") != 63:
        raise ValueError("one-sided output connection curl coverage changed")
    for entry in curl["nonzero_entries"]:
        groups[(str(entry["left_atom"]), int(entry["output_row"]))][
            str(entry["right_atom"])
        ] = sp.sympify(entry["value"])
    off_diagonal = []
    inconsistent_diagonal = []
    compatible = []
    for (left_atom, output_row), values in sorted(groups.items()):
        zero_direction_values = {atom: values[atom] for atom in ZERO_P10 if atom in values}
        diagonal_values = [values.get(atom, sp.S.Zero) for atom in DIAGONAL_P10]
        certificate = {
            "left_atom": left_atom,
            "output_row": output_row,
            "curl_values": {atom: str(value) for atom, value in sorted(values.items())},
        }
        if zero_direction_values:
            off_diagonal.append(
                {
                    **certificate,
                    "obstruction": "J_Bj_zero_for_every_B_on_zero_one_form_P10_direction",
                }
            )
        elif len(set(diagonal_values)) != 1:
            inconsistent_diagonal.append(
                {
                    **certificate,
                    "obstruction": "one_Omega_A10_i_cannot_match_three_diagonal_equations",
                }
            )
        else:
            omega = sp.factor(-diagonal_values[0])
            compatible.append(
                {
                    **certificate,
                    "connection_output_input_row": 10,
                    "connection_value": str(omega),
                    "repaired_pair_entries": 3,
                }
            )
    if len(groups) != 36 or len(off_diagonal) != 18 or len(inconsistent_diagonal) != 15:
        raise ValueError("one-sided output connection obstruction partition changed")
    if len(compatible) != 3 or sum(item["repaired_pair_entries"] for item in compatible) != 9:
        raise ValueError("one-sided output connection compatible partition changed")
    if {item["left_atom"] for item in compatible} != {"s12[5]", "s13[6]", "s23[8]"}:
        raise ValueError("one-sided output connection compatible atoms changed")
    return {
        "coefficient_system": {
            "equations": 8910,
            "unknowns": 10890,
            "coefficient_rank": 990,
            "augmented_rank": 991,
            "consistent": False,
            "active_output_row": 10,
            "active_output_row_coefficient_rank": 90,
            "active_output_row_augmented_rank": 91,
        },
        "obstruction_partition": {
            "nonzero_curl_entries": 63,
            "active_left_output_groups": 36,
            "zero_one_form_direction_obstruction_groups": off_diagonal,
            "zero_one_form_direction_obstruction_group_count": 18,
            "inconsistent_diagonal_groups": inconsistent_diagonal,
            "inconsistent_diagonal_group_count": 15,
            "compatible_groups": compatible,
            "compatible_group_count": 3,
            "compatible_pair_entries_repaired": 9,
        },
    }


def _candidate_records(values: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    typed = _records(values["typed_map_curl"])
    return [
        {
            "candidate_id": candidate_id,
            "coefficients": row["coefficients"],
            "predecessor_curl_dense_content_sha256": row["corrected_ordered_curl_manifest"][
                "dense_content_sha256"
            ],
            **_classify_candidate(row),
            "cross_slice_admitted_entries": 0,
            "candidate_decision": "blocked",
            "candidate_rejection_authorized": False,
            "first_blocker": FIRST_BLOCKER,
        }
        for candidate_id, row in sorted(typed.items())
    ]


def _expected_body(
    root: Path, config_path: Path, values: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    records = _candidate_records(values)
    return {
        "schema_version": RESULT_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "decision": (
            "arbitrary_output_rows_on_Pother_directions_leave_exact_rank_obstruction_"
            "cross_slice_not_admitted_candidates_blocked"
        ),
        "decision_counts": {"pass": 0, "reject": 0, "blocked": 12},
        "gate_counts": {
            "selected": 12,
            "connection_system_equations_per_candidate": 8910,
            "connection_system_unknowns_per_candidate": 10890,
            "connection_system_coefficient_rank": 990,
            "connection_system_augmented_rank": 991,
            "consistent_connection_systems": 0,
            "zero_one_form_direction_obstruction_groups_per_candidate": 18,
            "inconsistent_diagonal_groups_per_candidate": 15,
            "compatible_groups_per_candidate": 3,
            "compatible_pair_entries_repaired_per_candidate": 9,
            "corrected_curl_nonzero_entries_per_candidate": 63,
            "cross_slice_entries_admitted": 0,
            "principal_high_atom_entries_missing_per_candidate": 106920,
            "complete_ordered_D2F_tensors_registered": 0,
            "full_high_atom_good_unknown_identities_proved": 0,
            "global_H7_closures": 0,
            "nonlinear_PDE_closures": 0,
            "lifespans_proved": 0,
        },
        "no_go_theorem": {
            "name": "one_sided_Pother_output_connection_rank_no_go",
            "declared_class": (
                "Omega_A^B_i arbitrary for all 90 Pother directions and B=0..10, while the "
                "registered connection on all nine P10 directions is held fixed"
            ),
            "equation": "C_Aij+Omega_A^B_i*J_Bj=0",
            "rank_reason": (
                "The registered P10 one-form is zero on six directions and equals output-row-10 "
                "on three diagonal directions, so only one of eleven unknown input rows is active "
                "per Pother direction."
            ),
            "conclusion": (
                "The 8910-by-10890 system has coefficient rank 990 and augmented rank 991. "
                "Eighteen active groups hit zero one-form directions and fifteen more demand "
                "inconsistent values across the three diagonal directions. Only three groups, "
                "covering nine curl entries, admit a connection coefficient in this class."
            ),
        },
        "candidate_records": records,
        "exact_controls": {
            "promote_one_sided_no_go_to_two_sided_connection": {
                "rejected": True,
                "P10_direction_variations_not_classified": True,
            },
            "promote_partial_nine_entry_repair_to_cross_slice": {
                "rejected": True,
                "full_system_consistent": False,
            },
            "reject_candidates_from_declared_class_no_go": {
                "rejected": True,
                "corrected_source_and_two_sided_connections_open": True,
            },
        },
        "first_blocker": FIRST_BLOCKER,
        "secondary_blockers": [
            "two_sided_connection_system_with_P10_direction_variations_not_classified",
            "corrected_source_extension_not_registered",
            "nonlinear_arbitrary_background_typed_map_not_registered",
            "remaining_other_principal_by_other_principal_values_not_registered",
            "complete_high_atom_identity_TC1_TC2_TC3_TC5_B7_H7_PDE_lifespan_not_closed",
        ],
        "claim_seals": CLAIM_SEALS,
        "data_seals": EXPECTED_SEALS,
        "source_bindings": {
            "source": {"path": SOURCE_PATH, "file_sha256": _file_sha(_inside(root, SOURCE_PATH))},
            "config": {"path": CONFIG_PATH, "file_sha256": _file_sha(config_path)},
            "test": {"path": TEST_PATH, "file_sha256": _file_sha(_inside(root, TEST_PATH))},
            **EXPECTED_PREDECESSORS,
        },
        "scope": (
            "candidate-bound exact consistency and rank audit of arbitrary output connection rows "
            "on the 90 Pother directions with all P10-direction connection values fixed; no "
            "two-sided connection classification, corrected source, cross-slice admission, full "
            "D2F, high-atom identity, H7, PDE, lifespan, candidate rejection, or observation"
        ),
    }


def _validate_source_bindings(value: Mapping[str, Any], root: Path) -> None:
    bindings = value.get("source_bindings")
    if not isinstance(bindings, Mapping):
        raise TypeError("one-sided output connection source bindings missing")
    if set(bindings) != {"source", "config", "test", *EXPECTED_PREDECESSORS}:
        raise ValueError("one-sided output connection source binding keys changed")
    for label, relative in {"source": SOURCE_PATH, "config": CONFIG_PATH, "test": TEST_PATH}.items():
        binding = bindings[label]
        if not isinstance(binding, Mapping) or binding.get("path") != relative:
            raise ValueError("one-sided output connection local binding changed")
        if binding.get("file_sha256") != _file_sha(_inside(root, relative)):
            raise ValueError("one-sided output connection local binding hash changed")
    for label, expected in EXPECTED_PREDECESSORS.items():
        if bindings[label] != expected:
            raise ValueError("one-sided output connection predecessor binding changed")


def _validate_result(value: Mapping[str, Any], *, root: Path | None = None) -> None:
    validation_root = (root or Path(__file__).resolve().parents[2]).resolve()
    if value.get("content_sha256") != _content_sha(value):
        raise ValueError("one-sided output connection content hash changed")
    _validate_source_bindings(value, validation_root)
    config_path = _inside(validation_root, CONFIG_PATH)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    predecessors = {
        label: _load_bound(validation_root, binding)
        for label, binding in EXPECTED_PREDECESSORS.items()
    }
    _validate_predecessors(predecessors, validation_root)
    expected = _expected_body(validation_root, config_path, predecessors)
    if {key: item for key, item in value.items() if key != "content_sha256"} != expected:
        raise ValueError("one-sided output connection result boundary changed")


def build_gate(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    root = config_path.parents[2]
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    predecessors = {
        label: _load_bound(root, binding) for label, binding in config["predecessors"].items()
    }
    _validate_predecessors(predecessors, root)
    body = _expected_body(root, config_path, predecessors)
    result = {**body, "content_sha256": _sha(body)}
    _validate_result(result, root=root)
    return result


def write_gate(config_path: Path) -> Path:
    result = build_gate(config_path)
    root = config_path.resolve().parents[2]
    output = _inside(root, OUTPUT_PATH)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(CONFIG_PATH))
    args = parser.parse_args()
    print(write_gate(args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
