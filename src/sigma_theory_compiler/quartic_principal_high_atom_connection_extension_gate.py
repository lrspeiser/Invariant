"""Audit the registered B=10 output connection on a missing principal slice."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from functools import cache
from pathlib import Path
from typing import Any

import sympy as sp

from .quartic_full_d2f_high_atom_coverage_gate import (
    _validate_result as validate_full_d2f_coverage,
)
from .quartic_scalar_hessian_d2_integrability_gate import (
    FAMILY_SPECS,
    _generic_packet,
    _specialize,
)
from .quartic_scalar_hessian_d2_integrability_gate import (
    _validate_result as validate_scalar_hessian_d2,
)
from .quartic_scalar_hessian_output_bundle_repair_gate import (
    _validate_result as validate_output_bundle_repair,
)
from .quartic_unspecialized_source_jacobian_campaign import _unspecialized_principal_blocks

CONFIG_SCHEMA = "sigma-quartic-principal-high-atom-connection-extension-config-1.0"
RESULT_SCHEMA = "sigma-quartic-principal-high-atom-connection-extension-gate-1.0"
CAMPAIGN_ID = "quartic-principal-high-atom-connection-extension-001"
CONFIG_PATH = "configs/backgrounds/quartic_principal_high_atom_connection_extension_gate.json"
OUTPUT_PATH = (
    "runs/physics-language/quartic-principal-high-atom-connection-extension-gate/campaign.json"
)
SOURCE_PATH = "src/sigma_theory_compiler/quartic_principal_high_atom_connection_extension_gate.py"
TEST_PATH = "tests/test_quartic_principal_high_atom_connection_extension_gate.py"
FIRST_BLOCKER = (
    "reverse_Pother_by_P10_candidate_bound_source_derivatives_and_zero_corrected_"
    "curl_for_810_ordered_pairs_not_registered"
)
LEFT_ATOMS = tuple(f"{spec[0]}[10]" for spec in FAMILY_SPECS)

EXPECTED_PREDECESSORS = {
    "full_d2f_high_atom_coverage": {
        "path": "runs/physics-language/quartic-full-d2f-high-atom-coverage-gate/campaign.json",
        "file_sha256": "b9ce34960b766a6fe74a36a13190b0f050a1447884d599fad1eebfe189b32590",
        "content_sha256": "e7e4e4171aed90d07d68791183c58a696e77b9bed745f1018da2c5ee9438c38a",
    },
    "scalar_hessian_d2": {
        "path": "runs/physics-language/quartic-scalar-hessian-d2-integrability-gate/campaign.json",
        "file_sha256": "654a0442e0d6ec0166eed1d15da163260658d6cd44192da9cbb4e4b2b88f105a",
        "content_sha256": "66f680eec0ab93169163f7f1e2055aac7713f45fcc2abf2f12907183e701b45c",
    },
    "scalar_hessian_output_bundle_repair": {
        "path": "runs/physics-language/quartic-scalar-hessian-output-bundle-repair-gate/campaign.json",
        "file_sha256": "e1ae98ebcb3c2739f7c84938d61ce9e7d2d209d4025f54a7d1d499a8495acfdb",
        "content_sha256": "688dcb478b86d44330f8a3623183e91c237bd91f31bd4e91bf5869098175973f",
    },
}
EXPECTED_EXTENSION_CONTRACT = {
    "output_dimension": 11,
    "left_atoms": "nine_scalar_hessian_families_at_field_10",
    "right_atoms": "ninety_other_principal_scalar_hessian_atoms",
    "right_field_range": [0, 9],
    "connection_input_output_row": 10,
    "registered_connection": (
        "predecessor_sparse_Omega_A^10_on_left_atoms_zero_on_other_principal_atoms"
    ),
}
EXPECTED_POLICIES = {
    "one_sided_value_admission": (
        "require_registered_reverse_values_and_zero_corrected_curl"
    ),
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
    "other_principal_atom_subset_exactly_registered": True,
    "scalar_source_row_10_zero_on_other_principal_subset": True,
    "registered_B10_connection_correction_zero_on_P10_by_Pother": True,
    "one_sided_P10_by_Pother_values_materialized": True,
    "one_sided_P10_by_Pother_values_admitted_as_covariant_D2F": False,
    "reverse_Pother_by_P10_values_registered": False,
    "corrected_cross_slice_curl_zero": False,
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
        raise ValueError("principal connection extension path escapes project root")
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
        raise ValueError("principal connection extension config boundary changed")


def _load_bound(root: Path, binding: Mapping[str, Any]) -> dict[str, Any]:
    if set(binding) != {"path", "file_sha256", "content_sha256"}:
        raise ValueError("principal connection predecessor binding changed")
    path = _inside(root, str(binding["path"]))
    if not path.is_file() or _file_sha(path) != binding["file_sha256"]:
        raise ValueError("principal connection predecessor file hash changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("content_sha256") != binding["content_sha256"]:
        raise ValueError("principal connection predecessor content binding changed")
    if value.get("content_sha256") != _content_sha(value):
        raise ValueError("principal connection predecessor content hash changed")
    return value


def _records(value: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = value.get("candidate_records", [])
    if not isinstance(rows, list):
        raise TypeError("principal connection candidate records missing")
    result = {str(row.get("candidate_id")): row for row in rows if isinstance(row, Mapping)}
    if len(rows) != 12 or len(result) != 12:
        raise ValueError("principal connection candidate coverage changed")
    return result


def _right_atoms(coverage: Mapping[str, Any]) -> list[dict[str, Any]]:
    registry = coverage.get("atom_registry")
    if not isinstance(registry, list):
        raise TypeError("principal connection atom registry missing")
    result = [row for row in registry if row.get("atom_class") == "principal_other"]
    expected = [
        {
            "coordinate_column": 54 + family_index * 11 + field,
            "coordinate_atom": f"{family}[{field}]",
            "atom_class": "principal_other",
        }
        for family_index, (family, *_) in enumerate(FAMILY_SPECS)
        for field in range(10)
    ]
    if result != expected:
        raise ValueError("principal connection right-atom subset changed")
    return result


@cache
def _generic_principal_source() -> dict[str, Any]:
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
    chunks = {}
    for family, _, _, kind, first, second, multiplicity in FAMILY_SPECS:
        base = blocks[kind][first] if kind == "B_i" else blocks[kind][first][second]
        chunks[family] = (-inverse * (multiplicity * base).subs(zero)).applyfunc(sp.factor)
    return {
        "alpha": data["alpha"],
        "c20": data["c20"],
        "chunks": chunks,
        "unspecialized_block_sha256": blocks["content_sha256"],
    }


def _validate_predecessors(values: Mapping[str, Mapping[str, Any]], root: Path) -> None:
    if set(values) != set(EXPECTED_PREDECESSORS):
        raise ValueError("principal connection predecessor set changed")
    validate_full_d2f_coverage(values["full_d2f_high_atom_coverage"], root=root)
    validate_scalar_hessian_d2(values["scalar_hessian_d2"], root=root)
    validate_output_bundle_repair(values["scalar_hessian_output_bundle_repair"], root=root)
    maps = {label: _records(value) for label, value in values.items()}
    ids = set(maps["full_d2f_high_atom_coverage"])
    if any(set(rows) != ids for rows in maps.values()):
        raise ValueError("principal connection predecessor candidates disagree")
    for candidate_id in ids:
        coefficients = maps["full_d2f_high_atom_coverage"][candidate_id]["coefficients"]
        if any(rows[candidate_id].get("coefficients") != coefficients for rows in maps.values()):
            raise ValueError("principal connection predecessor coefficients disagree")
    _right_atoms(values["full_d2f_high_atom_coverage"])


def _dense_root_and_sparse(
    scalar_record: Mapping[str, Any],
    repair_record: Mapping[str, Any],
    right_atoms: list[dict[str, Any]],
) -> dict[str, Any]:
    coefficients = scalar_record["coefficients"]
    source = _generic_principal_source()
    substitutions = {
        source["alpha"]: sp.sympify(coefficients["a10"]),
        source["c20"]: sp.sympify(coefficients["c20"]),
    }
    scalar_row = []
    for atom in right_atoms:
        family, field_text = str(atom["coordinate_atom"]).rstrip("]").split("[")
        value = sp.factor(source["chunks"][family][10, int(field_text)].subs(substitutions))
        scalar_row.append({"right_atom": atom["coordinate_atom"], "value": str(value)})
    if any(item["value"] != "0" for item in scalar_row):
        raise ValueError("principal connection scalar source support changed")

    omega = {
        (int(entry["output_row"]), str(entry["domain_direction"])): sp.sympify(entry["value"])
        for entry in repair_record["output_bundle_connection_repair"][
            "sparse_nonzero_coefficients"
        ]
    }
    if len(omega) != 6:
        raise ValueError("principal connection sparse predecessor changed")
    blocks = {
        (str(block["low_direction"]), str(block["high_family"])): block
        for block in _specialize(_generic_packet(), coefficients)["manifest"]["blocks"]
    }
    dense = []
    corrections = []
    sparse = []
    for left_atom in LEFT_ATOMS:
        for right in right_atoms:
            family, field_text = str(right["coordinate_atom"]).rstrip("]").split("[")
            field = int(field_text)
            block = blocks[(left_atom, family)]
            naive = {
                (int(entry["output_row"]), int(entry["high_field"])): sp.sympify(
                    entry["value"]
                )
                for entry in block["nonzero_entries"]
            }
            source_value = sp.sympify(scalar_row[(FAMILY_NAMES.index(family) * 10) + field]["value"])
            for output_row in range(11):
                correction = sp.factor(omega.get((output_row, left_atom), 0) * source_value)
                value = sp.factor(naive.get((output_row, field), 0) + correction)
                coordinate = {
                    "output_row": output_row,
                    "left_atom": left_atom,
                    "right_atom": right["coordinate_atom"],
                }
                corrections.append({**coordinate, "value": str(correction)})
                dense.append({**coordinate, "value": str(value)})
                if value != 0:
                    sparse.append({**coordinate, "value": str(value)})
    if len(dense) != 8910 or len(sparse) != 93:
        raise ValueError("principal connection one-sided manifest changed")
    if any(item["value"] != "0" for item in corrections):
        raise ValueError("principal connection restricted correction became nonzero")
    return {
        "source_scalar_row_manifest": {
            "shape": [90],
            "entry_count": 90,
            "nonzero_entry_count": 0,
            "dense_content_sha256": _sha(scalar_row),
        },
        "connection_correction_manifest": {
            "shape": [11, 9, 90],
            "entry_count": 8910,
            "nonzero_entry_count": 0,
            "dense_content_sha256": _sha(corrections),
        },
        "one_sided_value_manifest": {
            "shape": [11, 9, 90],
            "entry_count": 8910,
            "nonzero_entry_count": 93,
            "nonzero_entries": sparse,
            "dense_content_sha256": _sha(dense),
            "status": "materialized_not_admitted_reverse_and_curl_unregistered",
        },
    }


FAMILY_NAMES = tuple(spec[0] for spec in FAMILY_SPECS)


def _candidate_records(values: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    coverage = _records(values["full_d2f_high_atom_coverage"])
    scalar = _records(values["scalar_hessian_d2"])
    repair = _records(values["scalar_hessian_output_bundle_repair"])
    right_atoms = _right_atoms(values["full_d2f_high_atom_coverage"])
    result = []
    for candidate_id in sorted(coverage):
        manifests = _dense_root_and_sparse(scalar[candidate_id], repair[candidate_id], right_atoms)
        result.append(
            {
                "candidate_id": candidate_id,
                "coefficients": coverage[candidate_id]["coefficients"],
                "predecessor_ordered_pair_classification_root_sha256": coverage[candidate_id][
                    "ordered_pair_classification_root_sha256"
                ],
                "predecessor_corrected_D2_submanifest_content_sha256": coverage[candidate_id][
                    "corrected_D2_submanifest_content_sha256"
                ],
                **manifests,
                "restricted_connection_extension_decision": (
                    "exact_no_effect_on_P10_by_Pother_because_J_10_right_equals_zero"
                ),
                "cross_slice_admitted_entries": 0,
                "candidate_decision": "blocked",
                "candidate_rejection_authorized": False,
                "first_blocker": FIRST_BLOCKER,
            }
        )
    return result


def _expected_body(
    root: Path, config_path: Path, values: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    records = _candidate_records(values)
    right_atoms = _right_atoms(values["full_d2f_high_atom_coverage"])
    return {
        "schema_version": RESULT_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "decision": (
            "restricted_B10_connection_extension_exactly_ineffective_one_sided_values_"
            "materialized_cross_slice_not_admitted_candidates_blocked"
        ),
        "decision_counts": {"pass": 0, "reject": 0, "blocked": 12},
        "gate_counts": {
            "selected": 12,
            "left_P10_atoms": 9,
            "right_other_principal_atoms": 90,
            "ordered_pair_cells_audited_per_candidate": 810,
            "one_sided_values_materialized_per_candidate": 8910,
            "one_sided_nonzero_values_per_candidate": 93,
            "source_scalar_row_entries_checked_per_candidate": 90,
            "source_scalar_row_nonzero_entries": 0,
            "restricted_connection_correction_entries_checked_per_candidate": 8910,
            "restricted_connection_nonzero_corrections": 0,
            "cross_slice_entries_admitted": 0,
            "principal_high_atom_entries_missing_per_candidate": 106920,
            "complete_ordered_D2F_tensors_registered": 0,
            "full_high_atom_good_unknown_identities_proved": 0,
            "global_H7_closures": 0,
            "nonlinear_PDE_closures": 0,
            "lifespans_proved": 0,
        },
        "subset_registry": {
            "left_atoms": list(LEFT_ATOMS),
            "right_atoms": right_atoms,
            "ordered_pair_cell_count": 810,
            "output_entry_count": 8910,
            "content_sha256": _sha({"left_atoms": LEFT_ATOMS, "right_atoms": right_atoms}),
        },
        "theorem": {
            "name": "restricted_B10_output_connection_extension_no_effect_on_P10_by_Pother",
            "formula": "D_i J_Aj=partial_i J_Aj+Omega_A^10_i J_10j",
            "hypothesis": (
                "i is one of nine P10 atoms, j is one of ninety principal atoms at fields 0..9, "
                "and Omega is the predecessor sparse B=10 repair extended by zero"
            ),
            "exact_fact": "J_10j=0 for all ninety registered right atoms and all twelve candidates",
            "conclusion": (
                "All 8910 connection corrections vanish per candidate. The resulting one-sided "
                "values equal the predecessor naive values, but cannot be admitted without the "
                "reverse ordered derivatives and a zero corrected-curl certificate."
            ),
        },
        "candidate_records": records,
        "exact_controls": {
            "infer_cross_slice_admission_from_one_sided_values": {
                "rejected": True,
                "missing_reverse_ordered_values": 8910,
                "corrected_curl_registered": False,
            },
            "infer_general_output_connection_no_go": {
                "rejected": True,
                "unregistered_input_output_rows_B0_through_B9": True,
                "unregistered_connection_on_other_principal_directions": True,
            },
            "reduce_principal_high_atom_missing_count": {
                "rejected": True,
                "entries_still_unadmitted": 106920,
            },
        },
        "first_blocker": FIRST_BLOCKER,
        "secondary_blockers": [
            "general_output_connection_rows_B0_through_B9_not_registered",
            "remaining_other_principal_by_other_principal_values_not_registered",
            "complete_high_atom_good_unknown_identity_not_registered",
            "induced_TC1_TC2_TC3_TC5_bounds_not_closed",
            "B7_global_H7_dyadic_summation_PDE_and_lifespan_not_closed",
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
            "candidate-bound exact audit of the predecessor B=10 sparse output connection, "
            "extended by zero, on the one-sided 11x9x90 P10-by-other-principal slice; no reverse "
            "values, cross-slice admission, general connection no-go, full D2F, high-atom identity, "
            "global H7, PDE, lifespan, candidate rejection, or observation"
        ),
    }


def _validate_source_bindings(value: Mapping[str, Any], root: Path) -> None:
    bindings = value.get("source_bindings")
    if not isinstance(bindings, Mapping):
        raise TypeError("principal connection source bindings missing")
    for label, relative in {"source": SOURCE_PATH, "config": CONFIG_PATH, "test": TEST_PATH}.items():
        binding = bindings.get(label)
        if not isinstance(binding, Mapping) or binding.get("path") != relative:
            raise ValueError("principal connection local binding changed")
        if binding.get("file_sha256") != _file_sha(_inside(root, relative)):
            raise ValueError("principal connection local binding hash changed")
    for label, expected in EXPECTED_PREDECESSORS.items():
        if bindings.get(label) != expected:
            raise ValueError("principal connection predecessor binding changed")


def _validate_result(value: Mapping[str, Any], *, root: Path | None = None) -> None:
    validation_root = (root or Path(__file__).resolve().parents[2]).resolve()
    if value.get("content_sha256") != _content_sha(value):
        raise ValueError("principal connection content hash changed")
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
        raise ValueError("principal connection result boundary changed")


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
