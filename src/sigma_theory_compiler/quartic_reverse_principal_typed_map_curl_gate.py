"""Register the flat typed source map and audit the reverse principal curl."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from functools import cache
from pathlib import Path
from typing import Any

import sympy as sp

from .quartic_principal_high_atom_connection_extension_gate import (
    _validate_result as validate_principal_connection_extension,
)
from .quartic_reverse_principal_source_map_identifiability_gate import (
    _validate_result as validate_reverse_identifiability,
)
from .quartic_scalar_hessian_d2_integrability_gate import FAMILY_SPECS
from .quartic_tc2_variable_sylvester_campaign import _coordinate_atom_to_jet_packet
from .quartic_unspecialized_source_jacobian_campaign import _unspecialized_principal_blocks

CONFIG_SCHEMA = "sigma-quartic-reverse-principal-typed-map-curl-config-1.0"
RESULT_SCHEMA = "sigma-quartic-reverse-principal-typed-map-curl-gate-1.0"
CAMPAIGN_ID = "quartic-reverse-principal-typed-map-curl-001"
CONFIG_PATH = "configs/backgrounds/quartic_reverse_principal_typed_map_curl_gate.json"
OUTPUT_PATH = "runs/physics-language/quartic-reverse-principal-typed-map-curl-gate/campaign.json"
SOURCE_PATH = "src/sigma_theory_compiler/quartic_reverse_principal_typed_map_curl_gate.py"
TEST_PATH = "tests/test_quartic_reverse_principal_typed_map_curl_gate.py"
FIRST_BLOCKER = (
    "registered_flat_typed_map_leaves_63_nonzero_cross_slice_curl_entries_per_candidate_"
    "requiring_general_output_connection_or_corrected_source_extension"
)

EXPECTED_PREDECESSORS = {
    "reverse_source_map_identifiability": {
        "path": (
            "runs/physics-language/quartic-reverse-principal-source-map-identifiability-gate/"
            "campaign.json"
        ),
        "file_sha256": "683b0e62afb0bbbcaf9cf8237749a7c97c48b1075e01832a5a73682657210cfd",
        "content_sha256": "6882a6899a83c9551a6a2443847480e254a00e26cf29539bab318966af26b9fd",
    },
    "principal_high_atom_connection_extension": {
        "path": (
            "runs/physics-language/quartic-principal-high-atom-connection-extension-gate/"
            "campaign.json"
        ),
        "file_sha256": "e4ffc8f0d82f3c4381703f338a03cc334e15268aaf5b0c7f0dd1305ee96f8b92",
        "content_sha256": "33942664dc481ae112650c1f9ad1c4834687161601b54348b55a82139816c028",
    },
}
EXPECTED_DIRECT_DEPENDENCIES = {
    "variable_sylvester_coordinate_map": {
        "source": {
            "path": "src/sigma_theory_compiler/quartic_tc2_variable_sylvester_campaign.py",
            "file_sha256": "5df63ca3084654198c7ca23e8e7ba6e171aadfeff0ab6c5f1d2709b16f20937f",
        },
        "test": {
            "path": "tests/test_quartic_tc2_variable_sylvester_campaign.py",
            "file_sha256": "0c455ec2cf911e080225570d8766d8ab85278988b2d5708c20eb8185c59c07fb",
        },
        "artifact": {
            "path": "runs/physics-language/quartic-tc2-variable-sylvester-campaign/campaign.json",
            "file_sha256": "b83041437d9dcd882459a5d1722af2f32222ae620ca2b4c3dd16d353380e845d",
            "content_sha256": "e38464400121b2a0fbfbf64453788273ad6f0e0eb2639e9899fd56b44f5881e8",
        },
    }
}
EXPECTED_SLICE_CONTRACT = {
    "typed_map": "flat_coordinate_153_to_covariant_24_Jacobian",
    "reverse_order": "Pother_by_P10",
    "forward_order": "P10_by_Pother",
    "output_dimension": 11,
    "left_reverse_atoms": 90,
    "right_reverse_atoms": 9,
    "reverse_ordered_pairs": 810,
}
EXPECTED_POLICIES = {
    "cross_slice_admission": "require_zero_corrected_ordered_curl",
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
    "flat_coordinate_to_covariant_jet_map_registered": True,
    "other_principal_to_Einstein_submap_registered": True,
    "reverse_Pother_by_P10_values_materialized": True,
    "corrected_cross_slice_curl_completely_materialized": True,
    "corrected_cross_slice_curl_zero": False,
    "cross_slice_D2F_entries_admitted": False,
    "general_covariant_typed_map_registered": False,
    "general_output_connection_registered": False,
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
        raise ValueError("reverse typed-map curl path escapes project root")
    return target


def _validate_config(value: Mapping[str, Any]) -> None:
    expected = {
        "schema_version": CONFIG_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "output_path": OUTPUT_PATH,
        "predecessors": EXPECTED_PREDECESSORS,
        "direct_dependencies": EXPECTED_DIRECT_DEPENDENCIES,
        "slice_contract": EXPECTED_SLICE_CONTRACT,
        "policies": EXPECTED_POLICIES,
        "seals": EXPECTED_SEALS,
    }
    if value != expected:
        raise ValueError("reverse typed-map curl config boundary changed")


def _load_bound(root: Path, binding: Mapping[str, Any]) -> dict[str, Any]:
    if set(binding) != {"path", "file_sha256", "content_sha256"}:
        raise ValueError("reverse typed-map curl artifact binding changed")
    path = _inside(root, str(binding["path"]))
    if not path.is_file() or _file_sha(path) != binding["file_sha256"]:
        raise ValueError("reverse typed-map curl artifact file hash changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("content_sha256") != binding["content_sha256"]:
        raise ValueError("reverse typed-map curl artifact content binding changed")
    if value.get("content_sha256") != _content_sha(value):
        raise ValueError("reverse typed-map curl artifact content hash changed")
    return value


def _records(value: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    raw = value.get("candidate_records", value.get("certificates", []))
    if not isinstance(raw, list):
        raise TypeError("reverse typed-map curl candidate records missing")
    result = {str(row.get("candidate_id")): row for row in raw if isinstance(row, Mapping)}
    if len(raw) != 12 or len(result) != 12:
        raise ValueError("reverse typed-map curl candidate set changed")
    return result


def _validate_direct_dependencies(root: Path, dependencies: Mapping[str, Any]) -> dict[str, Any]:
    if dependencies != EXPECTED_DIRECT_DEPENDENCIES:
        raise ValueError("reverse typed-map curl direct dependency boundary changed")
    bundle = dependencies.get("variable_sylvester_coordinate_map")
    if not isinstance(bundle, Mapping) or set(bundle) != {"source", "test", "artifact"}:
        raise ValueError("reverse typed-map curl direct dependency bundle changed")
    for label in ("source", "test"):
        binding = bundle[label]
        if not isinstance(binding, Mapping) or set(binding) != {"path", "file_sha256"}:
            raise ValueError("reverse typed-map curl direct file binding changed")
        path = _inside(root, str(binding["path"]))
        if not path.is_file() or _file_sha(path) != binding["file_sha256"]:
            raise ValueError(f"reverse typed-map curl direct {label} file hash changed")
    artifact = _load_bound(root, bundle["artifact"])
    packet = artifact.get("common_coordinate_to_covariant_jet_packet")
    live_packet = _coordinate_atom_to_jet_packet()["packet"]
    if (
        artifact.get("status")
        != "pass_all_12_first_order_variable_deltaK_extensions_higher_orders_global_H7_fail_closed"
        or artifact.get("counts", {}).get("selected") != 12
        or packet != live_packet
        or not isinstance(packet, Mapping)
        or packet.get("content_sha256") != "bbb9790adec7f1551945263bc6b7910204dcab3c51b0f6bc62e76553bf50246f"
    ):
        raise ValueError("reverse typed-map curl coordinate-map replay changed")
    return artifact


def _validate_predecessors(values: Mapping[str, Mapping[str, Any]], root: Path) -> None:
    if set(values) != set(EXPECTED_PREDECESSORS):
        raise ValueError("reverse typed-map curl predecessor set changed")
    validate_reverse_identifiability(values["reverse_source_map_identifiability"], root=root)
    validate_principal_connection_extension(
        values["principal_high_atom_connection_extension"], root=root
    )
    maps = {label: _records(value) for label, value in values.items()}
    ids = set(maps["reverse_source_map_identifiability"])
    if any(set(rows) != ids for rows in maps.values()):
        raise ValueError("reverse typed-map curl predecessor candidates disagree")
    for candidate_id in ids:
        coefficients = maps["reverse_source_map_identifiability"][candidate_id]["coefficients"]
        if maps["principal_high_atom_connection_extension"][candidate_id][
            "coefficients"
        ] != coefficients:
            raise ValueError("reverse typed-map curl predecessor coefficients disagree")


@cache
def _generic_reverse_vectors() -> dict[tuple[str, str], sp.MatrixBase]:
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
    einstein_symbols = {str(symbol): symbol for symbol in data["einstein_upper"].free_symbols}
    chunks = {}
    for family, _, _, kind, first, second, multiplicity in FAMILY_SPECS:
        base = blocks[kind][first] if kind == "B_i" else blocks[kind][first][second]
        chunks[family] = multiplicity * base
    derivatives = {
        (name, family): (
            inverse
            * sp.diff(blocks["A"], symbol).subs(zero)
            * inverse
            * chunk.subs(zero)
            - inverse * sp.diff(chunk, symbol).subs(zero)
        )[:, 10]
        for name, symbol in einstein_symbols.items()
        for family, chunk in chunks.items()
    }
    packet = _coordinate_atom_to_jet_packet()
    maps = dict(zip(packet["atoms"], packet["maps"], strict=True))
    result = {}
    for atom, mapping in maps.items():
        if atom.startswith("s") and not atom.endswith("[10]"):
            for family in chunks:
                result[(atom, f"{family}[10]")] = sum(
                    (
                        sp.sympify(value) * derivatives[(name, family)]
                        for name, value in mapping.items()
                        if name.startswith("G_")
                    ),
                    sp.zeros(11, 1),
                ).applyfunc(sp.factor)
    if len(result) != 810:
        raise ValueError("reverse typed-map curl generic pair coverage changed")
    return result


def _candidate_manifest(
    extension: Mapping[str, Any], coefficients: Mapping[str, Any]
) -> dict[str, Any]:
    blocks = _unspecialized_principal_blocks()
    data = blocks["data"]
    substitutions = {
        data["alpha"]: sp.sympify(coefficients["a10"]),
        data["c20"]: sp.sympify(coefficients["c20"]),
    }
    forward_sparse = {
        (entry["right_atom"], entry["left_atom"], int(entry["output_row"])): sp.sympify(
            entry["value"]
        )
        for entry in extension["one_sided_value_manifest"]["nonzero_entries"]
    }
    reverse_dense = []
    reverse_sparse = []
    curl_dense = []
    curl_sparse = []
    for (left_atom, right_atom), generic in _generic_reverse_vectors().items():
        vector = generic.subs(substitutions)
        for output_row in range(11):
            coordinate = {
                "output_row": output_row,
                "left_atom": left_atom,
                "right_atom": right_atom,
            }
            reverse = sp.factor(vector[output_row])
            forward = forward_sparse.get((left_atom, right_atom, output_row), 0)
            curl = sp.factor(reverse - forward)
            reverse_entry = {**coordinate, "value": str(reverse)}
            curl_entry = {**coordinate, "value": str(curl)}
            reverse_dense.append(reverse_entry)
            curl_dense.append(curl_entry)
            if reverse != 0:
                reverse_sparse.append(reverse_entry)
            if curl != 0:
                curl_sparse.append(curl_entry)
    if len(reverse_dense) != 8910 or len(reverse_sparse) != 75 or len(curl_sparse) != 63:
        raise ValueError("reverse typed-map curl candidate manifest changed")
    return {
        "reverse_value_manifest": {
            "shape": [11, 90, 9],
            "entry_count": 8910,
            "nonzero_entry_count": 75,
            "nonzero_entries": reverse_sparse,
            "dense_content_sha256": _sha(reverse_dense),
            "typed_map_content_sha256": (
                "bbb9790adec7f1551945263bc6b7910204dcab3c51b0f6bc62e76553bf50246f"
            ),
        },
        "corrected_ordered_curl_manifest": {
            "shape": [11, 90, 9],
            "entry_count": 8910,
            "nonzero_entry_count": 63,
            "nonzero_entries": curl_sparse,
            "dense_content_sha256": _sha(curl_dense),
            "orientation": "reverse_Pother_by_P10_minus_forward_P10_by_Pother",
        },
    }


def _candidate_records(values: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    reverse_prior = _records(values["reverse_source_map_identifiability"])
    extension = _records(values["principal_high_atom_connection_extension"])
    result = []
    for candidate_id in sorted(reverse_prior):
        coefficients = reverse_prior[candidate_id]["coefficients"]
        manifests = _candidate_manifest(extension[candidate_id], coefficients)
        result.append(
            {
                "candidate_id": candidate_id,
                "coefficients": coefficients,
                "predecessor_one_sided_dense_content_sha256": extension[candidate_id][
                    "one_sided_value_manifest"
                ]["dense_content_sha256"],
                **manifests,
                "restricted_reverse_connection_nonzero_corrections": 0,
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
    return {
        "schema_version": RESULT_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "decision": (
            "flat_typed_map_registers_reverse_values_nonzero_corrected_cross_slice_curl_"
            "blocks_admission_candidates_blocked"
        ),
        "decision_counts": {"pass": 0, "reject": 0, "blocked": 12},
        "gate_counts": {
            "selected": 12,
            "typed_coordinate_maps_registered": 1,
            "typed_map_coordinate_atoms": 153,
            "typed_map_covariant_jet_symbols": 24,
            "other_principal_atoms_mapped": 90,
            "reverse_ordered_pair_cells_per_candidate": 810,
            "reverse_entries_materialized_per_candidate": 8910,
            "reverse_nonzero_entries_per_candidate": 75,
            "corrected_curl_entries_checked_per_candidate": 8910,
            "corrected_curl_nonzero_entries_per_candidate": 63,
            "candidates_with_nonzero_corrected_curl": 12,
            "cross_slice_entries_admitted": 0,
            "principal_high_atom_entries_missing_per_candidate": 106920,
            "complete_ordered_D2F_tensors_registered": 0,
            "full_high_atom_good_unknown_identities_proved": 0,
            "global_H7_closures": 0,
            "nonlinear_PDE_closures": 0,
            "lifespans_proved": 0,
        },
        "typed_map_theorem": {
            "name": "flat_coordinate_second_metric_jet_to_linearized_Einstein_upper_map",
            "map_schema": "sigma-flat-coordinate-153-to-covariant-24-Jacobian-1.0",
            "map_content_sha256": "bbb9790adec7f1551945263bc6b7910204dcab3c51b0f6bc62e76553bf50246f",
            "registered_scope": (
                "exact flat-reference linearized Einstein map for the canonical orthonormal metric "
                "component convention; not a nonlinear or arbitrary-background chain rule"
            ),
            "reverse_source_formula": (
                "partial_g(-A^-1 C_i)="
                "A^-1(partial_g A)A^-1 C_i-A^-1(partial_g C_i)"
            ),
            "conclusion": (
                "The map determines all 8910 reverse entries per candidate. Comparison with the "
                "registered forward slice leaves exactly 63 nonzero corrected-curl entries, so "
                "none of this cross-slice is admitted as D2F."
            ),
        },
        "candidate_records": records,
        "exact_controls": {
            "promote_flat_map_to_general_covariant_map": {"rejected": True},
            "ignore_nonzero_corrected_curl": {
                "rejected": True,
                "nonzero_entries_per_candidate": 63,
            },
            "admit_reverse_values_without_integrability": {
                "rejected": True,
                "cross_slice_entries_admitted": 0,
            },
            "reject_candidates_from_restricted_curl": {
                "rejected": True,
                "broader_connection_or_source_repairs_unclassified": True,
            },
        },
        "first_blocker": FIRST_BLOCKER,
        "secondary_blockers": [
            "general_output_connection_rows_B0_through_B9_not_registered",
            "nonlinear_arbitrary_background_coordinate_to_Einstein_map_not_registered",
            "remaining_other_principal_by_other_principal_values_not_registered",
            "complete_high_atom_good_unknown_identity_not_registered",
            "induced_TC1_TC2_TC3_TC5_B7_global_H7_PDE_and_lifespan_not_closed",
        ],
        "claim_seals": CLAIM_SEALS,
        "data_seals": EXPECTED_SEALS,
        "source_bindings": {
            "source": {"path": SOURCE_PATH, "file_sha256": _file_sha(_inside(root, SOURCE_PATH))},
            "config": {"path": CONFIG_PATH, "file_sha256": _file_sha(config_path)},
            "test": {"path": TEST_PATH, "file_sha256": _file_sha(_inside(root, TEST_PATH))},
            "direct_dependencies": EXPECTED_DIRECT_DEPENDENCIES,
            **EXPECTED_PREDECESSORS,
        },
        "scope": (
            "candidate-bound flat-reference typed-map materialization of the complete reverse "
            "11x90x9 slice and exact corrected ordered curl against the registered forward slice; "
            "no cross-slice admission, general covariant map, general connection no-go, full D2F, "
            "high-atom identity, global H7, PDE, lifespan, candidate rejection, or observation"
        ),
    }


def _validate_source_bindings(value: Mapping[str, Any], root: Path) -> None:
    bindings = value.get("source_bindings")
    if not isinstance(bindings, Mapping):
        raise TypeError("reverse typed-map curl source bindings missing")
    if set(bindings) != {
        "source",
        "config",
        "test",
        "direct_dependencies",
        *EXPECTED_PREDECESSORS,
    }:
        raise ValueError("reverse typed-map curl source binding keys changed")
    for label, relative in {"source": SOURCE_PATH, "config": CONFIG_PATH, "test": TEST_PATH}.items():
        binding = bindings[label]
        if not isinstance(binding, Mapping) or binding.get("path") != relative:
            raise ValueError("reverse typed-map curl local binding changed")
        if binding.get("file_sha256") != _file_sha(_inside(root, relative)):
            raise ValueError("reverse typed-map curl local binding hash changed")
    if bindings["direct_dependencies"] != EXPECTED_DIRECT_DEPENDENCIES:
        raise ValueError("reverse typed-map curl direct dependency binding changed")
    _validate_direct_dependencies(root, bindings["direct_dependencies"])
    for label, expected in EXPECTED_PREDECESSORS.items():
        if bindings[label] != expected:
            raise ValueError("reverse typed-map curl predecessor binding changed")


def _validate_result(value: Mapping[str, Any], *, root: Path | None = None) -> None:
    validation_root = (root or Path(__file__).resolve().parents[2]).resolve()
    if value.get("content_sha256") != _content_sha(value):
        raise ValueError("reverse typed-map curl content hash changed")
    _validate_source_bindings(value, validation_root)
    config_path = _inside(validation_root, CONFIG_PATH)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    _validate_direct_dependencies(validation_root, config["direct_dependencies"])
    predecessors = {
        label: _load_bound(validation_root, binding)
        for label, binding in EXPECTED_PREDECESSORS.items()
    }
    _validate_predecessors(predecessors, validation_root)
    expected = _expected_body(validation_root, config_path, predecessors)
    if {key: item for key, item in value.items() if key != "content_sha256"} != expected:
        raise ValueError("reverse typed-map curl result boundary changed")


def build_gate(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    root = config_path.parents[2]
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    _validate_direct_dependencies(root, config["direct_dependencies"])
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
