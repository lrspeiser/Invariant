"""Evaluate the target source D2 values under the registered flat typed map."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import sympy as sp

from .quartic_reverse_principal_typed_map_curl_gate import (
    _coordinate_atom_to_jet_packet,
)
from .quartic_scalar_hessian_d2_integrability_gate import FAMILY_SPECS
from .quartic_unspecialized_source_jacobian_campaign import (
    _unspecialized_principal_blocks,
)

CONFIG_SCHEMA = "sigma-quartic-flat-factorized-leaf-jet-d2-specialization-config-1.0"
RESULT_SCHEMA = "sigma-quartic-flat-factorized-leaf-jet-d2-specialization-gate-1.0"
CAMPAIGN_ID = "quartic-flat-factorized-leaf-jet-d2-specialization-001"
CONFIG_PATH = "configs/backgrounds/quartic_flat_factorized_leaf_jet_d2_specialization_gate.json"
SOURCE_PATH = "src/sigma_theory_compiler/quartic_flat_factorized_leaf_jet_d2_specialization_gate.py"
TEST_PATH = "tests/test_quartic_flat_factorized_leaf_jet_d2_specialization_gate.py"
OUTPUT_PATH = (
    "runs/physics-language/quartic-flat-factorized-leaf-jet-d2-specialization-gate/campaign.json"
)
FIRST_BLOCKER = (
    "register_the_nonlinear_arbitrary_background_coordinate_to_covariant_jet_map_and_"
    "candidate_bound_A_B_C_leaf_derivatives"
)
MAP_SHA256 = "bbb9790adec7f1551945263bc6b7910204dcab3c51b0f6bc62e76553bf50246f"
BLOCK_SHA256 = "695ff2a5fd45fa3fba21d4ce25ab2f62bd168df187c8e931bc9b5803a9cd4aed"
EXPECTED_PREDECESSOR = {
    "source": {
        "path": (
            "src/sigma_theory_compiler/"
            "quartic_ordered_mixed_d2_arithmetic_dag_differentiability_gate.py"
        ),
        "file_sha256": "f8cd296240dba457caaf058a8cebabf8e728e35b9170360a96fe042574e08cc7",
    },
    "config": {
        "path": (
            "configs/backgrounds/"
            "quartic_ordered_mixed_d2_arithmetic_dag_differentiability_gate.json"
        ),
        "file_sha256": "30a51cf13934379c627fe557aec85607b419e80ba5708cde6867c53ac488dbf0",
    },
    "test": {
        "path": ("tests/test_quartic_ordered_mixed_d2_arithmetic_dag_differentiability_gate.py"),
        "file_sha256": "baf4316beccb1fc09232bb9354fbaa485f9ffcdb2c888210f434d83af9d53cd9",
    },
    "artifact": {
        "path": (
            "runs/physics-language/"
            "quartic-ordered-mixed-d2-arithmetic-dag-differentiability-gate/campaign.json"
        ),
        "file_sha256": "2992571c544846efc96142e2e4a74efe280a7bb025efadb1ff945ab9515bafcc",
        "content_sha256": "d8afd9f91c090ad1c07e4bb22257baa8c61c095f8d434e02a27082b5591abb6a",
    },
}
EXPECTED_EVIDENCE = {
    "flat_typed_coordinate_map": {
        "source": {
            "path": "src/sigma_theory_compiler/quartic_reverse_principal_typed_map_curl_gate.py",
            "file_sha256": "e63be1538452210ee432e45f9919793ea3077b855ae2b12a138bfb0a757d3516",
        },
        "config": {
            "path": "configs/backgrounds/quartic_reverse_principal_typed_map_curl_gate.json",
            "file_sha256": "016283509fcc921de382937db874b8e2f657076a64769fb3ecc8f64f99640121",
        },
        "test": {
            "path": "tests/test_quartic_reverse_principal_typed_map_curl_gate.py",
            "file_sha256": "426e52dd87b70f5a6587adb2b00ac68fb1bb19796414822a4f62cb09721986a3",
        },
        "artifact": {
            "path": (
                "runs/physics-language/quartic-reverse-principal-typed-map-curl-gate/campaign.json"
            ),
            "file_sha256": "4e432566b16e44b7d5ca05a2ce6e60b5ebd849e2fe8c88fa6523297f1fc111b4",
            "content_sha256": "79d06514c1dd8fd7933bdc36b19622fc3cce8ddcaf14712f0b908fbe6c9f2664",
        },
    },
    "unspecialized_source_blocks": {
        "source": {
            "path": "src/sigma_theory_compiler/quartic_unspecialized_source_jacobian_campaign.py",
            "file_sha256": "f5a8649b52bd7f2384ee9087d0f5f6d8850a5c5bc443fbe823c6b09655ce9616",
        },
        "config": {
            "path": "configs/backgrounds/quartic_unspecialized_source_jacobian_campaign.json",
            "file_sha256": "c2a49e33e425a6d1d04e49b998c211f45c1d8f0ec1cfdc11a43f80dc525412ae",
        },
        "test": {
            "path": "tests/test_quartic_unspecialized_source_jacobian_campaign.py",
            "file_sha256": "82040e82e63f23b2e03df9351a8049cc957bc27414cabc85e49ac01132c83181",
        },
        "artifact": {
            "path": (
                "runs/physics-language/quartic-unspecialized-source-jacobian-campaign/campaign.json"
            ),
            "file_sha256": "8ecae346f75ba5bbeb266e486b96f48a0c76387513ff92d20d1bc68d8ecef22b",
            "content_sha256": "b60dbbb191f43d84d3d9c9e44e4adf70e4e7d729143905561b695cfabcaa7c72",
        },
    },
}
EXPECTED_CONTRACT = {
    "candidate_count": 12,
    "target_records_per_candidate": 22,
    "unique_target_roots_per_candidate": 20,
    "general_root_count": 264,
    "coordinate_map_shape": [153, 24],
    "flat_reference": "zero_gradient_zero_hessian_zero_Einstein_m2_one",
}
EXPECTED_POLICIES = {
    "flat_D2_admission": "require_live_typed_map_block_formula_and_exact_value_replay",
    "general_background_D2": "fail_closed",
    "promote_flat_to_general": "forbidden",
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


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _content_sha(value: Mapping[str, Any]) -> str:
    return _sha({key: item for key, item in value.items() if key != "content_sha256"})


def _inside(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if path != root and root not in path.parents:
        raise ValueError("flat factorized D2 path escapes project root")
    return path


def _copy(value: Any) -> Any:
    return json.loads(_canonical(value))


def _validate_config(value: Mapping[str, Any]) -> None:
    if value != {
        "schema_version": CONFIG_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "output_path": OUTPUT_PATH,
        "predecessor": EXPECTED_PREDECESSOR,
        "direct_evidence": EXPECTED_EVIDENCE,
        "specialization_contract": EXPECTED_CONTRACT,
        "policies": EXPECTED_POLICIES,
        "seals": EXPECTED_SEALS,
    }:
        raise ValueError("flat factorized D2 config boundary changed")


def _load_bound(root: Path, binding: Mapping[str, Any]) -> dict[str, Any]:
    path = _inside(root, str(binding["path"]))
    if _file_sha(path) != binding["file_sha256"]:
        raise ValueError("flat factorized D2 artifact file binding changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("content_sha256") != binding["content_sha256"] or value.get(
        "content_sha256"
    ) != _content_sha(value):
        raise ValueError("flat factorized D2 artifact content binding changed")
    return value


def _load_bundle(root: Path, bundle: Mapping[str, Any]) -> dict[str, Any]:
    if set(bundle) != {"source", "config", "test", "artifact"}:
        raise ValueError("flat factorized D2 evidence bundle keys changed")
    for label in ("source", "config", "test"):
        binding = bundle[label]
        if _file_sha(_inside(root, binding["path"])) != binding["file_sha256"]:
            raise ValueError("flat factorized D2 evidence file binding changed")
    return _load_bound(root, bundle["artifact"])


def _load_inputs(root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    predecessor = _load_bound(root, EXPECTED_PREDECESSOR["artifact"])
    for label in ("source", "config", "test"):
        binding = EXPECTED_PREDECESSOR[label]
        if _file_sha(_inside(root, binding["path"])) != binding["file_sha256"]:
            raise ValueError("flat factorized D2 predecessor file binding changed")
    if (
        predecessor.get("gate_counts", {}).get("deduplicated_leaf_derivative_obligations") != 31680
        or predecessor.get("gate_counts", {}).get("registered_ordered_mixed_D2_roots") != 0
    ):
        raise ValueError("flat factorized D2 predecessor boundary changed")
    typed = _load_bundle(root, EXPECTED_EVIDENCE["flat_typed_coordinate_map"])
    unspecialized = _load_bundle(root, EXPECTED_EVIDENCE["unspecialized_source_blocks"])
    if (
        typed.get("typed_map_theorem", {}).get("map_content_sha256") != MAP_SHA256
        or typed.get("claim_seals", {}).get("flat_coordinate_to_covariant_jet_map_registered")
        is not True
        or unspecialized.get("status")
        != "pass_all_12_complete_unspecialized_principal_source_jacobians_remainder_fail_closed"
        or unspecialized.get("generic_unspecialized_source_jacobian_control", {})
        .get("unspecialized_block_extraction", {})
        .get("block_content_sha256")
        != BLOCK_SHA256
    ):
        raise ValueError("flat factorized D2 evidence semantic boundary changed")
    return predecessor, typed, unspecialized


def _generic_target_values(
    templates: list[Mapping[str, Any]],
) -> tuple[dict[str, sp.Expr], list[dict[str, Any]]]:
    blocks = _unspecialized_principal_blocks()
    if blocks["content_sha256"] != BLOCK_SHA256:
        raise ValueError("flat factorized D2 live block reconstruction changed")
    data = blocks["data"]
    groups = [data["gradient_lower"], data["hessian_lower"], data["einstein_upper"]]
    zero = {symbol: 0 for group in groups for symbol in group.free_symbols}
    zero[data["m2"]] = 1
    zero[data["c20"]] = data["c20"]
    inverse = blocks["A"].subs(zero).inv()
    symbols = {str(symbol): symbol for group in groups for symbol in group.free_symbols}
    packet = _coordinate_atom_to_jet_packet()
    if packet["packet"].get("content_sha256") != MAP_SHA256:
        raise ValueError("flat factorized D2 live typed map changed")
    maps = dict(zip(packet["atoms"], packet["maps"], strict=True))
    chunks = {
        family: multiplicity
        * (blocks[kind][first] if kind == "B_i" else blocks[kind][first][second])
        for family, _, _, kind, first, second, multiplicity in FAMILY_SPECS
    }
    result = {}
    factorization = []
    for template in templates:
        atom = str(template["coordinate_atom"])
        family = atom.split("[")[0]
        mapping = maps[atom]
        vector = sp.zeros(11, 1)
        for name, coefficient in mapping.items():
            symbol = symbols[name]
            chunk = chunks[family]
            derivative = (
                inverse * sp.diff(blocks["A"], symbol).subs(zero) * inverse * chunk.subs(zero)
                - inverse * sp.diff(chunk, symbol).subs(zero)
            )[:, 10]
            vector += sp.sympify(coefficient) * derivative
        result[atom] = sp.factor(vector[10])
        factorization.append(
            {
                "coordinate_atom": atom,
                "coordinate_column": template["coordinate_column"],
                "coordinate_ordinals": template["coordinate_ordinals"],
                "source_chunk_family": family,
                "typed_jet_sparse_map": [
                    {"jet_symbol": name, "coefficient": str(value)}
                    for name, value in sorted(mapping.items())
                ],
                "typed_jet_support_size": len(mapping),
                "generic_flat_D2_row10_value": str(result[atom]),
            }
        )
    if len(result) != 20:
        raise ValueError("flat factorized D2 target set changed")
    return result, factorization


def _candidate_coefficients(
    unspecialized: Mapping[str, Any], candidate_ids: list[str]
) -> dict[str, Mapping[str, Any]]:
    records = {
        str(row["candidate_id"]): row["coefficients"]
        for row in unspecialized.get("certificates", [])
    }
    if set(records) != set(candidate_ids) or any(row.get("m2") != "1" for row in records.values()):
        raise ValueError("flat factorized D2 candidate coefficients changed")
    return records


def _arithmetic_DAG(values: set[str]) -> tuple[dict[str, Any], dict[str, int]]:
    ordered = sorted(values, key=lambda value: (sp.sympify(value), value))
    nodes = [{"op": "exact_constant", "value": value} for value in ordered]
    body = {
        "schema_version": "sigma-flat-specialized-exact-constant-D2-DAG-1.0",
        "allowed_operations": ["exact_constant"],
        "node_count": len(nodes),
        "nodes": nodes,
    }
    return {**body, "content_sha256": _sha(body)}, {
        value: index for index, value in enumerate(ordered)
    }


def _expected_body(
    root: Path,
    config_path: Path,
    predecessor: Mapping[str, Any],
    typed: Mapping[str, Any],
    unspecialized: Mapping[str, Any],
) -> dict[str, Any]:
    templates = predecessor["target_root_templates"]
    generic, factorization = _generic_target_values(templates)
    candidate_ids = [row["candidate_id"] for row in predecessor["candidate_manifests"]]
    coefficients = _candidate_coefficients(unspecialized, candidate_ids)
    candidate_values: dict[str, dict[str, str]] = {}
    all_values = set()
    blocks = _unspecialized_principal_blocks()
    data = blocks["data"]
    for candidate_id in candidate_ids:
        subs = {
            data["alpha"]: sp.sympify(coefficients[candidate_id]["a10"]),
            data["c20"]: sp.sympify(coefficients[candidate_id]["c20"]),
        }
        candidate_values[candidate_id] = {
            atom: str(sp.factor(value.subs(subs))) for atom, value in generic.items()
        }
        all_values.update(candidate_values[candidate_id].values())
    dag, roots = _arithmetic_DAG(all_values)
    manifests = []
    total_nonzero = 0
    for candidate_id in candidate_ids:
        records = []
        by_atom = candidate_values[candidate_id]
        for template in templates:
            atom = template["coordinate_atom"]
            for ordinal in template["coordinate_ordinals"]:
                value = by_atom[atom]
                total_nonzero += value != "0"
                records.append(
                    {
                        "candidate_id": candidate_id,
                        "coordinate_ordinal": ordinal,
                        "coordinate_atom": atom,
                        "coordinate_column": template["coordinate_column"],
                        "D1_arithmetic_root": template["D1_arithmetic_root"],
                        "flat_D2_value": value,
                        "flat_D2_arithmetic_root": roots[value],
                        "flat_D2_arithmetic_dag_sha256": dag["content_sha256"],
                        "flat_typed_map_sha256": MAP_SHA256,
                        "unspecialized_block_sha256": BLOCK_SHA256,
                        "general_background_D2_root_registered": False,
                        "candidate_rejection_authorized": False,
                    }
                )
        if len(records) != 22:
            raise AssertionError("flat factorized D2 candidate projection changed")
        manifests.append(
            {
                "candidate_id": candidate_id,
                "coefficients": coefficients[candidate_id],
                "flat_D2_records": records,
                "flat_D2_roots_materialized": 22,
                "flat_D2_nonzero_roots": sum(row["flat_D2_value"] != "0" for row in records),
                "general_background_D2_roots_registered": 0,
                "manifest_sha256": _sha(records),
                "candidate_decision": "pass_flat_specialization_general_background_blocked",
                "candidate_rejection_authorized": False,
                "first_blocker": FIRST_BLOCKER,
            }
        )
    if (
        len(manifests) != 12
        or total_nonzero != 72
        or all_values
        != {
            "-1",
            "-1/2",
            "0",
            "1/2",
            "1",
        }
    ):
        raise ValueError("flat factorized D2 value census changed")
    return {
        "schema_version": RESULT_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "decision": "pass_264_flat_factorized_D2_roots_general_background_roots_blocked",
        "decision_counts": {"pass": 12, "blocked": 0, "reject": 0},
        "general_background_admission_counts": {"pass": 0, "blocked": 12, "reject": 0},
        "first_blocker": FIRST_BLOCKER,
        "factorization_theorem": {
            "name": "flat_coordinate_to_covariant_jet_factorization_of_target_leaf_derivatives",
            "exact_result": (
                "The registered flat 153-to-24 coordinate map contracts exact derivatives of "
                "the live A/B/C block formulas and discharges the predecessor leaf obligations "
                "without expanding 31,680 records. All 264 target values are exact: 192 are zero "
                "and the remaining 72 equal the candidate coefficient alpha, split evenly among "
                "-1, -1/2, 1/2, and 1."
            ),
            "boundary": (
                "These are candidate-bound flat-reference specializations. The registered map is "
                "not a nonlinear arbitrary-background chain rule, so no general D2 root, curl "
                "admission, or physical no-go follows."
            ),
        },
        "factorized_leaf_derivative_manifest": factorization,
        "factorized_manifest_sha256": _sha(factorization),
        "flat_specialized_D2_arithmetic_DAG": dag,
        "candidate_manifests": manifests,
        "gate_counts": {
            "selected_candidates": 12,
            "factorized_target_roots_per_candidate": 20,
            "flat_D2_roots_materialized": 264,
            "flat_D2_nonzero_roots": 72,
            "flat_D2_zero_roots": 192,
            "flat_unique_exact_values": 5,
            "predecessor_leaf_obligations_factorized": 31680,
            "general_background_leaf_derivative_roots_registered": 0,
            "general_background_D2_roots_registered": 0,
            "general_background_D2_roots_blocked": 264,
            "complete_ordered_D2F_tensors_registered": 0,
            "global_H7_closures": 0,
            "nonlinear_PDE_closures": 0,
            "lifespans_proved": 0,
        },
        "claim_seals": {
            "flat_typed_coordinate_map_replayed": True,
            "flat_A_B_C_leaf_derivative_factorization_replayed": True,
            "all_264_flat_D2_values_materialized": True,
            "general_background_coordinate_map_registered": False,
            "general_background_D2_values_registered": False,
            "physical_no_go_proved": False,
            "complete_ordered_D2F_tensor_registered": False,
            "global_H7_energy_closed": False,
            "nonlinear_PDE_closed": False,
            "nonlinear_lifespan_proved": False,
            "candidate_theory_rejected": False,
            "observational_claim_made": False,
        },
        "exact_controls": {
            "promote_flat_map_to_arbitrary_background": {"rejected": True},
            "promote_flat_value_to_general_D2_root": {"rejected": True},
            "treat_empty_sparse_typed_map_as_unknown_in_flat_scope": {"rejected": True},
            "infer_physical_no_go_from_missing_general_map": {"rejected": True},
            "reject_candidate_from_general_background_blocker": {"rejected": True},
        },
        "data_seals": dict(EXPECTED_SEALS),
        "source_bindings": {
            "source": {"path": SOURCE_PATH, "file_sha256": _file_sha(_inside(root, SOURCE_PATH))},
            "config": {"path": CONFIG_PATH, "file_sha256": _file_sha(config_path)},
            "test": {"path": TEST_PATH, "file_sha256": _file_sha(_inside(root, TEST_PATH))},
            "predecessor": _copy(EXPECTED_PREDECESSOR),
            "direct_evidence": _copy(EXPECTED_EVIDENCE),
        },
        "scope": (
            "candidate-bound exact flat-reference factorization and materialization of the 264 "
            "target D2 values only; no arbitrary-background chain rule or D2 root, covariant "
            "admission, physical no-go, full D2F, H7, PDE, lifespan, rejection, or observation"
        ),
    }


def _validate_bindings(value: Mapping[str, Any], root: Path) -> None:
    bindings = value.get("source_bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != {
        "source",
        "config",
        "test",
        "predecessor",
        "direct_evidence",
    }:
        raise ValueError("flat factorized D2 source binding keys changed")
    for label, relative in {
        "source": SOURCE_PATH,
        "config": CONFIG_PATH,
        "test": TEST_PATH,
    }.items():
        if bindings[label] != {
            "path": relative,
            "file_sha256": _file_sha(_inside(root, relative)),
        }:
            raise ValueError("flat factorized D2 local binding changed")
    if bindings["predecessor"] != EXPECTED_PREDECESSOR:
        raise ValueError("flat factorized D2 predecessor binding changed")
    if bindings["direct_evidence"] != EXPECTED_EVIDENCE:
        raise ValueError("flat factorized D2 evidence binding changed")


def _validate_result(value: Mapping[str, Any], *, root: Path | None = None) -> None:
    validation_root = (root or Path(__file__).resolve().parents[2]).resolve()
    if value.get("content_sha256") != _content_sha(value):
        raise ValueError("flat factorized D2 content hash changed")
    _validate_bindings(value, validation_root)
    config_path = _inside(validation_root, CONFIG_PATH)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    predecessor, typed, unspecialized = _load_inputs(validation_root)
    expected = _expected_body(validation_root, config_path, predecessor, typed, unspecialized)
    if {key: item for key, item in value.items() if key != "content_sha256"} != expected:
        raise ValueError("flat factorized D2 result boundary changed")


def build_gate(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    root = config_path.parents[2]
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    predecessor, typed, unspecialized = _load_inputs(root)
    body = _expected_body(root, config_path, predecessor, typed, unspecialized)
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
