"""Classify every ordered D2F atom pair without promoting missing tensors."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .quartic_full_tensor_good_unknown_reconciliation_gate import (
    _validate_result as validate_full_tensor_reconciliation,
)
from .quartic_scalar_hessian_output_bundle_repair_gate import (
    _validate_result as validate_output_bundle_repair,
)

CONFIG_SCHEMA = "sigma-quartic-full-d2f-high-atom-coverage-config-1.0"
RESULT_SCHEMA = "sigma-quartic-full-d2f-high-atom-coverage-gate-1.0"
CAMPAIGN_ID = "quartic-full-d2f-high-atom-coverage-001"
CONFIG_PATH = "configs/backgrounds/quartic_full_d2f_high_atom_coverage_gate.json"
OUTPUT_PATH = "runs/physics-language/quartic-full-d2f-high-atom-coverage-gate/campaign.json"
SOURCE_PATH = "src/sigma_theory_compiler/quartic_full_d2f_high_atom_coverage_gate.py"
TEST_PATH = "tests/test_quartic_full_d2f_high_atom_coverage_gate.py"
FIRST_BLOCKER = (
    "candidate_bound_covariant_source_derivatives_and_output_bundle_connection_"
    "extension_for_remaining_106920_principal_high_atom_D2F_entries_not_registered"
)
REPAIRED_ATOMS = {
    f"{family}[10]"
    for family in ("s01", "s02", "s03", "s11", "s12", "s13", "s22", "s23", "s33")
}

EXPECTED_PREDECESSORS = {
    "full_source_jacobian": {
        "path": "runs/physics-language/quartic-full-source-jacobian-arithmetic-campaign/campaign.json",
        "file_sha256": "e893ebcaef464b958516279c557382fb76ecdb0fd542b3e3fed6a347076fcdae",
        "content_sha256": "1707b7258fd434f68b06c7af6bc447b4136624b9916992df8b412e048ab6538a",
    },
    "high_atom_obstruction": {
        "path": "runs/physics-language/quartic-high-atom-d2-good-unknown-campaign/campaign.json",
        "file_sha256": "5848e62c811baf4a005e821d73c3dcc6d29a285fa2be57cfbe6842b56dfd3513",
        "content_sha256": "5b6a5c43d9e22c2780f3987e3271b8c863c802129b3837777da246a5d635b466",
    },
    "full_tensor_reconciliation": {
        "path": "runs/physics-language/quartic-full-tensor-good-unknown-reconciliation-gate/campaign.json",
        "file_sha256": "cf7957c2efad52a1fa91761fc6259e17a58011cc6093365f9e86e8e7eea0dfd6",
        "content_sha256": "9994df86948a4419dd999b66610e9fea847dece6d5300f68152e942ffb2b87c8",
    },
    "scalar_hessian_output_bundle_repair": {
        "path": "runs/physics-language/quartic-scalar-hessian-output-bundle-repair-gate/campaign.json",
        "file_sha256": "e1ae98ebcb3c2739f7c84938d61ce9e7d2d209d4025f54a7d1d499a8495acfdb",
        "content_sha256": "688dcb478b86d44330f8a3623183e91c237bd91f31bd4e91bf5869098175973f",
    },
}
EXPECTED_COVERAGE_CONTRACT = {
    "atom_count": 153,
    "output_dimension": 11,
    "ordered_domain_pair_count": 23409,
    "ordered_D2F_entry_count": 257499,
    "high_atom_definition": "all_99_principal_scalar_hessian_atoms",
    "repaired_atom_definition": "nine_scalar_hessian_families_at_field_10",
}
EXPECTED_POLICIES = {
    "covariant_source_terms": "fail_closed_unless_registered",
    "output_connection_terms": "fail_closed_unless_registered",
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
    "complete_ordered_D2F_coverage_domain_classified": True,
    "corrected_scalar_hessian_high_field10_submanifest_admitted": True,
    "remaining_principal_high_atom_domain_exactly_classified": True,
    "remaining_ordered_D2F_values_materialized": False,
    "covariant_source_derivative_extension_registered": False,
    "output_bundle_connection_extension_registered": False,
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
        raise ValueError("full D2F coverage path escapes project root")
    return target


def _validate_config(value: Mapping[str, Any]) -> None:
    expected = {
        "schema_version": CONFIG_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "output_path": OUTPUT_PATH,
        "predecessors": EXPECTED_PREDECESSORS,
        "coverage_contract": EXPECTED_COVERAGE_CONTRACT,
        "policies": EXPECTED_POLICIES,
        "seals": EXPECTED_SEALS,
    }
    if value != expected:
        raise ValueError("full D2F coverage config boundary changed")


def _load_bound(root: Path, binding: Mapping[str, Any]) -> dict[str, Any]:
    if set(binding) != {"path", "file_sha256", "content_sha256"}:
        raise ValueError("full D2F coverage predecessor binding changed")
    path = _inside(root, str(binding["path"]))
    if not path.is_file() or _file_sha(path) != binding["file_sha256"]:
        raise ValueError("full D2F coverage predecessor file hash changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("content_sha256") != binding["content_sha256"]:
        raise ValueError("full D2F coverage predecessor content binding changed")
    if value.get("content_sha256") != _content_sha(value):
        raise ValueError("full D2F coverage predecessor content hash changed")
    return value


def _record_map(value: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    raw = value.get("candidate_records", value.get("certificates", []))
    if not isinstance(raw, list):
        raise TypeError("full D2F coverage candidate records missing")
    result = {str(row.get("candidate_id")): row for row in raw if isinstance(row, Mapping)}
    if len(raw) != 12 or len(result) != 12:
        raise ValueError("full D2F coverage candidate set changed")
    return result


def _atom_registry(source: Mapping[str, Any]) -> list[dict[str, Any]]:
    manifest = source.get("common_full_entry_manifest")
    if not isinstance(manifest, Mapping):
        raise TypeError("full source manifest missing")
    if manifest.get("shape") != [11, 153] or manifest.get("total_entry_count") != 1683:
        raise ValueError("full source shape changed")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or len(entries) != 1683:
        raise ValueError("full source entries changed")
    groups: dict[int, set[tuple[str, str, int]]] = {}
    for entry in entries:
        column = entry.get("coordinate_column")
        row = entry.get("source_row")
        if type(column) is not int or type(row) is not int:
            raise TypeError("full source coordinates must be integers")
        groups.setdefault(column, set()).add(
            (str(entry.get("coordinate_atom")), str(entry.get("family")), row)
        )
    if set(groups) != set(range(153)):
        raise ValueError("full source coordinate columns changed")
    registry = []
    for column in range(153):
        group = groups[column]
        atoms = {(atom, family) for atom, family, _ in group}
        rows = {row for _, _, row in group}
        if len(atoms) != 1 or rows != set(range(11)) or len(group) != 11:
            raise ValueError("full source atom/output coverage changed")
        atom, family = next(iter(atoms))
        if family not in {"lower", "principal"}:
            raise ValueError("full source atom family changed")
        atom_class = (
            "lower"
            if family == "lower"
            else "principal_high_field10"
            if atom in REPAIRED_ATOMS
            else "principal_other"
        )
        registry.append(
            {"coordinate_column": column, "coordinate_atom": atom, "atom_class": atom_class}
        )
    counts = Counter(row["atom_class"] for row in registry)
    if counts != {"lower": 54, "principal_high_field10": 9, "principal_other": 90}:
        raise ValueError("full source atom partition changed")
    if {row["coordinate_atom"] for row in registry if row["atom_class"] == "principal_high_field10"} != REPAIRED_ATOMS:
        raise ValueError("repaired atom registry changed")
    return registry


def _pair_status(left: str, right: str) -> str:
    if left == "principal_high_field10" and right == "principal_high_field10":
        return "corrected_admitted"
    if left == "principal_high_field10" and right == "principal_other":
        return "naive_evaluated_not_admitted"
    if left == "principal_other" and right == "principal_high_field10":
        return "reverse_principal_not_registered"
    if left == "principal_other" and right == "principal_other":
        return "other_principal_pair_not_registered"
    if left != "lower" and right == "lower":
        return "principal_lower_not_registered"
    if left == "lower" and right != "lower":
        return "lower_principal_not_registered"
    return "lower_lower_not_registered"


def _coverage_ledger(registry: list[dict[str, Any]]) -> dict[str, Any]:
    packets = []
    all_cells = []
    for left in registry:
        cells = [
            {
                "right_column": right["coordinate_column"],
                "status": _pair_status(left["atom_class"], right["atom_class"]),
            }
            for right in registry
        ]
        all_cells.extend(
            {"left_column": left["coordinate_column"], **cell} for cell in cells
        )
        packets.append(
            {
                "left_column": left["coordinate_column"],
                "left_atom": left["coordinate_atom"],
                "ordered_pair_cells": 153,
                "status_counts": dict(sorted(Counter(cell["status"] for cell in cells).items())),
                "row_classification_root_sha256": _sha(cells),
            }
        )
    counts = dict(sorted(Counter(cell["status"] for cell in all_cells).items()))
    expected = {
        "corrected_admitted": 81,
        "naive_evaluated_not_admitted": 810,
        "reverse_principal_not_registered": 810,
        "other_principal_pair_not_registered": 8100,
        "principal_lower_not_registered": 5346,
        "lower_principal_not_registered": 5346,
        "lower_lower_not_registered": 2916,
    }
    if counts != expected or sum(counts.values()) != 23409:
        raise ValueError("ordered D2F coverage partition changed")
    return {
        "shape": [11, 153, 153],
        "output_dimension": 11,
        "ordered_pair_cell_count": 23409,
        "ordered_D2F_entry_count": 257499,
        "atom_partition_counts": {
            "lower": 54,
            "principal": 99,
            "principal_high_field10": 9,
            "principal_other": 90,
        },
        "pair_status_counts": counts,
        "entry_status_counts": {key: count * 11 for key, count in counts.items()},
        "principal_high_atom_pair_cells": 9801,
        "principal_high_atom_entries": 107811,
        "principal_high_atom_entries_admitted": 891,
        "principal_high_atom_entries_missing": 106920,
        "full_ordered_D2F_entries_missing": 256608,
        "row_packets": packets,
        "ordered_pair_classification_root_sha256": _sha(all_cells),
    }


def _validate_predecessors(values: Mapping[str, Mapping[str, Any]], root: Path) -> None:
    if set(values) != set(EXPECTED_PREDECESSORS):
        raise ValueError("full D2F coverage predecessor set changed")
    validate_full_tensor_reconciliation(values["full_tensor_reconciliation"], root=root)
    validate_output_bundle_repair(values["scalar_hessian_output_bundle_repair"], root=root)
    source = values["full_source_jacobian"]
    high = values["high_atom_obstruction"]
    if source.get("status") != "pass_all_12_full_11x153_entrywise_arithmetic_mixed_tensors_fail_closed":
        raise ValueError("full source semantic status changed")
    if high.get("status") != "pass_all_12_exact_representative_D2_obstructions_named_good_unknown_cancellation_refuted_global_H7_fail_closed":
        raise ValueError("high atom obstruction semantic status changed")
    _atom_registry(source)
    maps = {label: _record_map(value) for label, value in values.items()}
    candidate_ids = set(maps["scalar_hessian_output_bundle_repair"])
    if any(set(rows) != candidate_ids for rows in maps.values()):
        raise ValueError("full D2F coverage predecessor candidates disagree")
    repair = maps["scalar_hessian_output_bundle_repair"]
    for candidate_id in candidate_ids:
        coefficients = repair[candidate_id].get("coefficients")
        for rows in maps.values():
            if rows[candidate_id].get("coefficients", coefficients) != coefficients:
                raise ValueError("full D2F coverage predecessor coefficients disagree")
        manifest = repair[candidate_id].get("corrected_D2_submanifest")
        if not isinstance(manifest, Mapping) or manifest.get("entry_count") != 891:
            raise ValueError("corrected D2 submanifest coverage changed")
        if manifest.get("content_sha256") != _content_sha(manifest):
            raise ValueError("corrected D2 submanifest content hash changed")


def _candidate_records(
    values: Mapping[str, Mapping[str, Any]],
    registry: list[dict[str, Any]],
    ledger: Mapping[str, Any],
) -> list[dict[str, Any]]:
    repair = _record_map(values["scalar_hessian_output_bundle_repair"])
    return [
        {
            "candidate_id": candidate_id,
            "coefficients": row["coefficients"],
            "corrected_D2_submanifest_content_sha256": row["corrected_D2_submanifest"]["content_sha256"],
            "atom_registry_content_sha256": _sha(registry),
            "ordered_pair_classification_root_sha256": ledger[
                "ordered_pair_classification_root_sha256"
            ],
            "corrected_admitted_entries": 891,
            "principal_high_atom_entries_missing": 106920,
            "full_ordered_D2F_entries_missing": 256608,
            "candidate_decision": "blocked",
            "candidate_rejection_authorized": False,
            "first_blocker": FIRST_BLOCKER,
        }
        for candidate_id, row in sorted(repair.items())
    ]


def _expected_body(
    root: Path, config_path: Path, values: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    registry = _atom_registry(values["full_source_jacobian"])
    ledger = _coverage_ledger(registry)
    return {
        "schema_version": RESULT_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "decision": "complete_ordered_D2F_domain_classified_values_and_full_high_atom_identity_blocked",
        "decision_counts": {"pass": 0, "reject": 0, "blocked": 12},
        "gate_counts": {
            "selected": 12,
            "coordinate_atoms": 153,
            "ordered_pair_cells_classified": 23409,
            "ordered_D2F_entries_in_domain": 257499,
            "corrected_entries_admitted_per_candidate": 891,
            "principal_high_atom_entries_missing_per_candidate": 106920,
            "full_ordered_D2F_entries_missing_per_candidate": 256608,
            "complete_ordered_D2F_tensors_registered": 0,
            "full_high_atom_good_unknown_identities_proved": 0,
            "global_H7_closures": 0,
            "nonlinear_PDE_closures": 0,
            "lifespans_proved": 0,
        },
        "coverage_theorem": {
            "name": "complete_ordered_D2F_domain_partition_after_scalar_high_field10_repair",
            "domain": "11 output rows by the ordered square of the bound 153-atom source-Jacobian basis",
            "admitted_subdomain": "nine high-field-10 scalar-Hessian atoms by themselves",
            "conclusion": (
                "Every ordered atom pair is classified. Exactly 81 pair cells, or 891 output "
                "entries, inherit corrected values. All other cells remain unregistered; "
                "classification is not materialization and proves no full good-unknown identity."
            ),
        },
        "atom_registry": registry,
        "ordered_coverage_ledger": ledger,
        "candidate_records": _candidate_records(values, registry, ledger),
        "exact_controls": {
            "promote_naive_P10_by_other_principal_values": {
                "rejected": True,
                "entries": 8910,
                "reason": "output_connection_extension_and_reverse_order_values_not_registered",
            },
            "promote_coverage_classification_to_D2F_values": {
                "rejected": True,
                "remaining_entries": 256608,
            },
            "promote_principal_domain_to_high_atom_identity": {
                "rejected": True,
                "remaining_entries": 106920,
            },
            "admit_unregistered_covariant_source_or_connection_term": {"rejected": True},
        },
        "first_blocker": FIRST_BLOCKER,
        "secondary_blockers": [
            "remaining_256608_ordered_D2F_values_per_candidate_not_registered",
            "ordered_symmetry_for_remaining_principal_pairs_not_registered",
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
            "candidate-bound closed-world classification of all ordered 153x153 D2F atom pairs "
            "and exact admission of only the predecessor 11x9x9 repaired submanifest; no claim "
            "of the remaining values, covariant origin, full high-atom identity, global H7, PDE, "
            "lifespan, candidate rejection, or observation"
        ),
    }


def _validate_source_bindings(value: Mapping[str, Any], root: Path) -> None:
    bindings = value.get("source_bindings")
    if not isinstance(bindings, Mapping):
        raise TypeError("full D2F coverage source bindings missing")
    for label, relative in {"source": SOURCE_PATH, "config": CONFIG_PATH, "test": TEST_PATH}.items():
        binding = bindings.get(label)
        if not isinstance(binding, Mapping) or binding.get("path") != relative:
            raise ValueError("full D2F coverage local binding changed")
        if binding.get("file_sha256") != _file_sha(_inside(root, relative)):
            raise ValueError("full D2F coverage local binding hash changed")
    for label, expected in EXPECTED_PREDECESSORS.items():
        if bindings.get(label) != expected:
            raise ValueError("full D2F coverage predecessor binding changed")


def _validate_result(value: Mapping[str, Any], *, root: Path | None = None) -> None:
    validation_root = (root or Path(__file__).resolve().parents[2]).resolve()
    if value.get("content_sha256") != _content_sha(value):
        raise ValueError("full D2F coverage content hash changed")
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
        raise ValueError("full D2F coverage result boundary changed")


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
