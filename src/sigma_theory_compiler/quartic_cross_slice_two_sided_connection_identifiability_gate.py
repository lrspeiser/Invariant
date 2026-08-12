"""Classify the missing premise for two-sided cross-slice connection repair."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from functools import cache
from pathlib import Path
from typing import Any

import sympy as sp

from .quartic_cross_slice_one_sided_output_connection_no_go_gate import (
    _validate_result as validate_one_sided_no_go,
)

CONFIG_SCHEMA = "sigma-quartic-cross-slice-two-sided-connection-identifiability-config-1.0"
RESULT_SCHEMA = "sigma-quartic-cross-slice-two-sided-connection-identifiability-gate-1.0"
CAMPAIGN_ID = "quartic-cross-slice-two-sided-connection-identifiability-001"
CONFIG_PATH = (
    "configs/backgrounds/quartic_cross_slice_two_sided_connection_identifiability_gate.json"
)
OUTPUT_PATH = (
    "runs/physics-language/quartic-cross-slice-two-sided-connection-identifiability-gate/"
    "campaign.json"
)
SOURCE_PATH = (
    "src/sigma_theory_compiler/"
    "quartic_cross_slice_two_sided_connection_identifiability_gate.py"
)
TEST_PATH = "tests/test_quartic_cross_slice_two_sided_connection_identifiability_gate.py"
FIRST_BLOCKER = (
    "candidate_bound_Pother_one_form_or_corrected_source_jet_required_to_select_between_"
    "inconsistent_zero_completion_and_consistent_rank_six_completion"
)
FAMILIES = ("s01", "s02", "s03", "s11", "s12", "s13", "s22", "s23", "s33")
POTHER = tuple(f"{family}[{index}]" for family in FAMILIES for index in range(10))
P10 = tuple(f"{family}[10]" for family in FAMILIES)

EXPECTED_PREDECESSORS = {
    "one_sided_output_connection_no_go": {
        "path": (
            "runs/physics-language/quartic-cross-slice-one-sided-output-connection-no-go-gate/"
            "campaign.json"
        ),
        "file_sha256": "2a6823b9dda087b1a336b85b559305850dd2a9b9c987aa274bcd9ec0d08a55ba",
        "content_sha256": "250bec41dafd002930516c250ca5bbda3eb35c633a53d3e397d8040f1ead6554",
    }
}
EXPECTED_EXTENSION_CONTRACT = {
    "equation": "C_Aij+Omega_A^B_i*J_Bj-Omega_A^B_j*K_Bi=0",
    "registered_P10_one_form_rank": 1,
    "missing_Pother_one_form_shape": [11, 90],
    "Pother_connection_unknowns": 10890,
    "P10_connection_variation_unknowns": 1089,
    "total_connection_unknowns": 11979,
    "equations": 8910,
    "constructive_subclass": "Omega_A^B_i_zero_on_all_Pother_directions",
}
EXPECTED_POLICIES = {
    "two_sided_admission": "require_candidate_bound_Pother_one_form",
    "synthetic_completion_admission": "forbidden",
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
    "two_sided_connection_premise_identifiability_classified": True,
    "zero_Pother_one_form_completion_inconsistent": True,
    "rank_six_synthetic_completion_constructed": True,
    "rank_six_minimal_in_zero_Pother_connection_subclass": True,
    "candidate_bound_Pother_one_form_registered": False,
    "corrected_source_extension_registered": False,
    "physical_two_sided_connection_registered": False,
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
        raise ValueError("two-sided connection identifiability path escapes project root")
    return target


def _validate_config(value: Mapping[str, Any]) -> None:
    expected = {
        "schema_version": CONFIG_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "output_path": OUTPUT_PATH,
        "predecessors": EXPECTED_PREDECESSORS,
        "extension_contract": EXPECTED_EXTENSION_CONTRACT,
        "policies": EXPECTED_POLICIES,
        "seals": EXPECTED_SEALS,
    }
    if value != expected:
        raise ValueError("two-sided connection identifiability config boundary changed")


def _load_bound(root: Path, binding: Mapping[str, Any]) -> dict[str, Any]:
    if set(binding) != {"path", "file_sha256", "content_sha256"}:
        raise ValueError("two-sided connection identifiability predecessor binding changed")
    path = _inside(root, str(binding["path"]))
    if not path.is_file() or _file_sha(path) != binding["file_sha256"]:
        raise ValueError("two-sided connection identifiability predecessor file hash changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("content_sha256") != binding["content_sha256"]:
        raise ValueError("two-sided connection identifiability content binding changed")
    if value.get("content_sha256") != _content_sha(value):
        raise ValueError("two-sided connection identifiability predecessor content changed")
    return value


def _records(value: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    raw = value.get("candidate_records")
    if not isinstance(raw, list):
        raise TypeError("two-sided connection identifiability candidate records missing")
    result = {str(row.get("candidate_id")): row for row in raw if isinstance(row, Mapping)}
    if len(raw) != 12 or len(result) != 12:
        raise ValueError("two-sided connection identifiability candidate set changed")
    return result


def _validate_predecessor(value: Mapping[str, Any], root: Path) -> None:
    validate_one_sided_no_go(value, root=root)
    if value.get("decision_counts") != {"pass": 0, "reject": 0, "blocked": 12}:
        raise ValueError("two-sided connection identifiability predecessor decisions changed")
    if value.get("gate_counts", {}).get("connection_system_coefficient_rank") != 990:
        raise ValueError("two-sided connection identifiability predecessor coefficient rank changed")
    if value.get("gate_counts", {}).get("connection_system_augmented_rank") != 991:
        raise ValueError("two-sided connection identifiability predecessor augmented rank changed")


def _curl_matrix(row: Mapping[str, Any]) -> sp.ImmutableDenseMatrix:
    matrix = sp.zeros(99, 90)
    seen: set[tuple[int, str, str]] = set()
    partition = row["obstruction_partition"]
    lists = (
        partition["zero_one_form_direction_obstruction_groups"],
        partition["inconsistent_diagonal_groups"],
        partition["compatible_groups"],
    )
    for certificates in lists:
        for certificate in certificates:
            output_row = int(certificate["output_row"])
            left_atom = str(certificate["left_atom"])
            if output_row not in range(11) or left_atom not in POTHER:
                raise ValueError("two-sided connection identifiability curl coordinate changed")
            for right_atom, raw in certificate["curl_values"].items():
                key = (output_row, left_atom, str(right_atom))
                if key in seen or right_atom not in P10:
                    raise ValueError("two-sided connection identifiability curl support changed")
                seen.add(key)
                value = sp.sympify(raw)
                if value == 0:
                    raise ValueError("two-sided connection identifiability sparse curl gained zero")
                matrix[output_row * 9 + P10.index(right_atom), POTHER.index(left_atom)] = value
    if len(seen) != 63 or sum(item != 0 for item in matrix) != 63:
        raise ValueError("two-sided connection identifiability curl count changed")
    return sp.ImmutableDenseMatrix(matrix)


def _sparse_matrix(
    matrix: sp.MatrixBase, row_coordinates: list[dict[str, Any]], column_name: str
) -> list[dict[str, Any]]:
    result = []
    for row_index in range(matrix.rows):
        for column_index in range(matrix.cols):
            value = matrix[row_index, column_index]
            if value != 0:
                result.append(
                    {
                        **row_coordinates[row_index],
                        column_name: column_index,
                        "value": str(sp.factor(value)),
                    }
                )
    return result


@cache
def _completion_from_partition(partition_packet: str) -> dict[str, Any]:
    curl = _curl_matrix({"obstruction_partition": json.loads(partition_packet)})
    left_factor, right_factor = curl.rank_decomposition()
    if left_factor.shape != (99, 6) or right_factor.shape != (6, 90):
        raise ValueError("two-sided connection identifiability rank decomposition changed")
    if left_factor * right_factor != curl:
        raise ValueError("two-sided connection identifiability factor replay failed")

    pother_one_form = sp.zeros(11, 90)
    pother_one_form[:6, :] = right_factor
    p10_connection = sp.zeros(11 * 11, 9)
    for output_row in range(11):
        for p10_index in range(9):
            curl_row = output_row * 9 + p10_index
            for input_row in range(6):
                p10_connection[output_row * 11 + input_row, p10_index] = left_factor[
                    curl_row, input_row
                ]

    residual = sp.zeros(99, 90)
    for output_row in range(11):
        for p10_index in range(9):
            for pother_index in range(90):
                repaired = sum(
                    p10_connection[output_row * 11 + input_row, p10_index]
                    * pother_one_form[input_row, pother_index]
                    for input_row in range(11)
                )
                residual[output_row * 9 + p10_index, pother_index] = sp.factor(
                    curl[output_row * 9 + p10_index, pother_index] - repaired
                )
    if any(residual):
        raise ValueError("two-sided connection identifiability completion residual changed")

    one_form_sparse = []
    for input_row in range(11):
        for pother_index, atom in enumerate(POTHER):
            value = pother_one_form[input_row, pother_index]
            if value != 0:
                one_form_sparse.append(
                    {"input_row": input_row, "Pother_atom": atom, "value": str(sp.factor(value))}
                )
    connection_rows = [
        {"output_row": output_row, "P10_atom": P10[p10_index]}
        for output_row in range(11)
        for p10_index in range(9)
    ]
    connection_sparse = _sparse_matrix(
        left_factor, connection_rows, "input_row"
    )
    if len(one_form_sparse) != 54 or len(connection_sparse) != 9:
        raise ValueError("two-sided connection identifiability sparse completion changed")
    return {
        "curl_flattening": {
            "shape": [99, 90],
            "rank": 6,
            "nonzero_entry_count": 63,
            "dense_content_sha256": _sha([str(item) for item in curl]),
        },
        "zero_completion_witness": {
            "Pother_one_form_rank": 0,
            "coefficient_rank": 990,
            "augmented_rank": 991,
            "consistent": False,
        },
        "rank_six_completion_witness": {
            "synthetic_not_source_registered": True,
            "Pother_one_form": {
                "shape": [11, 90],
                "rank": 6,
                "nonzero_entry_count": 54,
                "nonzero_entries": one_form_sparse,
                "dense_content_sha256": _sha([str(item) for item in pother_one_form]),
            },
            "P10_connection_variation": {
                "shape": [11, 11, 9],
                "nonzero_entry_count": 9,
                "nonzero_entries": connection_sparse,
                "dense_content_sha256": _sha([str(item) for item in p10_connection]),
            },
            "Pother_connection_variation_nonzero_entries": 0,
            "coefficient_rank": 1518,
            "augmented_rank": 1518,
            "consistent": True,
            "equations_checked": 8910,
            "nonzero_residuals": 0,
            "residual_dense_content_sha256": _sha(["0"] * 8910),
        },
    }


def _completion(row: Mapping[str, Any]) -> dict[str, Any]:
    return _completion_from_partition(
        json.dumps(row["obstruction_partition"], sort_keys=True, separators=(",", ":"))
    )


def _candidate_records(predecessor: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "candidate_id": candidate_id,
            "coefficients": row["coefficients"],
            "predecessor_curl_dense_content_sha256": row[
                "predecessor_curl_dense_content_sha256"
            ],
            **_completion(row),
            "cross_slice_admitted_entries": 0,
            "candidate_decision": "blocked",
            "candidate_rejection_authorized": False,
            "first_blocker": FIRST_BLOCKER,
        }
        for candidate_id, row in sorted(_records(predecessor).items())
    ]


def _expected_body(root: Path, config_path: Path, predecessor: Mapping[str, Any]) -> dict[str, Any]:
    records = _candidate_records(predecessor)
    return {
        "schema_version": RESULT_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "decision": (
            "two_sided_connection_consistency_depends_on_unregistered_Pother_one_form_"
            "synthetic_rank_six_completion_not_admissible_candidates_blocked"
        ),
        "decision_counts": {"pass": 0, "reject": 0, "blocked": 12},
        "gate_counts": {
            "selected": 12,
            "two_sided_equations_per_candidate": 8910,
            "two_sided_connection_unknowns_per_candidate": 11979,
            "missing_Pother_one_form_entries_per_candidate": 990,
            "curl_flattening_rank_per_candidate": 6,
            "zero_completion_coefficient_rank": 990,
            "zero_completion_augmented_rank": 991,
            "rank_six_completion_coefficient_rank": 1518,
            "rank_six_completion_augmented_rank": 1518,
            "rank_six_completion_one_form_nonzero_entries_per_candidate": 54,
            "rank_six_completion_connection_nonzero_entries_per_candidate": 9,
            "rank_six_completion_residual_nonzero_entries": 0,
            "physically_registered_completions": 0,
            "cross_slice_entries_admitted": 0,
            "principal_high_atom_entries_missing_per_candidate": 106920,
            "complete_ordered_D2F_tensors_registered": 0,
            "full_high_atom_good_unknown_identities_proved": 0,
            "global_H7_closures": 0,
            "nonlinear_PDE_closures": 0,
            "lifespans_proved": 0,
        },
        "identifiability_theorem": {
            "name": "Pother_one_form_two_sided_connection_non_identifiability",
            "declared_class": (
                "cross-slice equation C_Aij+Omega_A^B_i J_Bj-Omega_A^B_j K_Bi=0, "
                "with registered rank-one J on P10 and unregistered K on Pother"
            ),
            "zero_completion": (
                "K=0 reduces exactly to the predecessor one-sided system, whose coefficient "
                "rank is 990 and augmented rank is 991"
            ),
            "constructive_completion": (
                "The 99-by-90 curl flattening has exact rank six. Its rank decomposition C=L*K "
                "gives a synthetic rank-six Pother one-form K and P10 connection variation L, "
                "with Omega_i=0, that satisfies all 8910 equations."
            ),
            "minimality_boundary": (
                "Rank six is necessary and sufficient only in the declared Omega_i=0 "
                "constructive subclass because rank(C)<=rank(K); no general physical "
                "connection or corrected source is inferred."
            ),
            "conclusion": (
                "Registered evidence does not select K and therefore does not determine whether "
                "P10-direction variations close the curl. A candidate-bound source-derived "
                "Pother one-form or corrected source jet is required before admission."
            ),
        },
        "candidate_records": records,
        "exact_controls": {
            "promote_synthetic_completion_to_source_derived": {"rejected": True},
            "promote_subclass_minimality_to_general_no_go": {"rejected": True},
            "admit_cross_slice_from_existential_completion": {
                "rejected": True,
                "physically_registered_completions": 0,
            },
            "reject_candidates_from_non_identifiability": {"rejected": True},
        },
        "first_blocker": FIRST_BLOCKER,
        "secondary_blockers": [
            "corrected_source_extension_not_registered",
            "general_covariant_typed_map_not_registered",
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
            "candidate-bound identifiability audit for adding P10-direction output-connection "
            "variations to the registered cross slice; explicit zero and synthetic rank-six "
            "Pother one-form completions only, with no source registration, cross-slice "
            "admission, full D2F, high-atom identity, H7, PDE, lifespan, rejection, or observation"
        ),
    }


def _validate_source_bindings(value: Mapping[str, Any], root: Path) -> None:
    bindings = value.get("source_bindings")
    if not isinstance(bindings, Mapping):
        raise TypeError("two-sided connection identifiability source bindings missing")
    if set(bindings) != {"source", "config", "test", *EXPECTED_PREDECESSORS}:
        raise ValueError("two-sided connection identifiability source binding keys changed")
    for label, relative in {"source": SOURCE_PATH, "config": CONFIG_PATH, "test": TEST_PATH}.items():
        binding = bindings[label]
        if not isinstance(binding, Mapping) or binding != {
            "path": relative,
            "file_sha256": _file_sha(_inside(root, relative)),
        }:
            raise ValueError("two-sided connection identifiability local binding changed")
    for label, expected in EXPECTED_PREDECESSORS.items():
        if bindings[label] != expected:
            raise ValueError("two-sided connection identifiability predecessor binding changed")


def _validate_result(value: Mapping[str, Any], *, root: Path | None = None) -> None:
    validation_root = (root or Path(__file__).resolve().parents[2]).resolve()
    if value.get("content_sha256") != _content_sha(value):
        raise ValueError("two-sided connection identifiability content hash changed")
    _validate_source_bindings(value, validation_root)
    config_path = _inside(validation_root, CONFIG_PATH)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    predecessor = _load_bound(
        validation_root, EXPECTED_PREDECESSORS["one_sided_output_connection_no_go"]
    )
    _validate_predecessor(predecessor, validation_root)
    expected = _expected_body(validation_root, config_path, predecessor)
    if {key: item for key, item in value.items() if key != "content_sha256"} != expected:
        raise ValueError("two-sided connection identifiability result boundary changed")


def build_gate(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    root = config_path.parents[2]
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    predecessor = _load_bound(root, config["predecessors"]["one_sided_output_connection_no_go"])
    _validate_predecessor(predecessor, root)
    body = _expected_body(root, config_path, predecessor)
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
