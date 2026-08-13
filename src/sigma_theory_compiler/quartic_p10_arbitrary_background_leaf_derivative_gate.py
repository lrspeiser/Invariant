"""Register arbitrary-background A/B/C leaf derivatives for five target P10 atoms."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from functools import cache
from pathlib import Path
from typing import Any

import sympy as sp

from .quartic_scalar_hessian_d2_integrability_gate import FAMILY_SPECS
from .quartic_unspecialized_source_jacobian_campaign import (
    _unspecialized_principal_blocks,
)

CONFIG_SCHEMA = "sigma-quartic-p10-arbitrary-background-leaf-derivative-config-1.0"
RESULT_SCHEMA = "sigma-quartic-p10-arbitrary-background-leaf-derivative-gate-1.0"
CAMPAIGN_ID = "quartic-p10-arbitrary-background-leaf-derivative-001"
CONFIG_PATH = "configs/backgrounds/quartic_p10_arbitrary_background_leaf_derivative_gate.json"
SOURCE_PATH = "src/sigma_theory_compiler/quartic_p10_arbitrary_background_leaf_derivative_gate.py"
TEST_PATH = "tests/test_quartic_p10_arbitrary_background_leaf_derivative_gate.py"
OUTPUT_PATH = (
    "runs/physics-language/quartic-p10-arbitrary-background-leaf-derivative-gate/campaign.json"
)
FIRST_BLOCKER = (
    "differentiate_and_replay_the_bound_inverse_product_D1_DAG_using_the_7920_registered_"
    "P10_leaf_roots_then_register_Pother_leaf_derivatives"
)
FORMULA_SHA256 = "9d0a41e02f3a86b4f6351240d57078e859dd9b6ce047bcaf1b08b71e2296cb11"
BLOCK_SHA256 = "695ff2a5fd45fa3fba21d4ce25ab2f62bd168df187c8e931bc9b5803a9cd4aed"
EXPECTED_PREDECESSOR = {
    "source": {
        "path": (
            "src/sigma_theory_compiler/quartic_flat_factorized_leaf_jet_d2_specialization_gate.py"
        ),
        "file_sha256": "ad86b9611ac869b491bc83d9a4966209184e5bd7fd4eacb6a98739ec9191549e",
    },
    "config": {
        "path": (
            "configs/backgrounds/quartic_flat_factorized_leaf_jet_d2_specialization_gate.json"
        ),
        "file_sha256": "feb8da4e94aebdf8713b998383945d1540ef4cbd852ab4a6bdaed5fe67fa92fa",
    },
    "test": {
        "path": "tests/test_quartic_flat_factorized_leaf_jet_d2_specialization_gate.py",
        "file_sha256": "3348c7f8895851d65d220128ce46c20927bcc57f7c26923a697496bcbc023363",
    },
    "artifact": {
        "path": (
            "runs/physics-language/quartic-flat-factorized-leaf-jet-d2-specialization-gate/"
            "campaign.json"
        ),
        "file_sha256": "7f433906323391a8b84179d2abb63b0b107fadad2667a8f6350d3add357a7d1c",
        "content_sha256": "be94d39348864e642a0b4460c35f845e21f7cd093792f0ce97eab152505bfd2a",
    },
}
EXPECTED_EVIDENCE = {
    "nonlinear_geometric_map": {
        "source": {
            "path": "src/sigma_theory_compiler/quartic_geometric_jet_campaign.py",
            "file_sha256": "d0600d6475d32d06a00140ab230aa41b3c057aef7a968163989fc5028d6acd21",
        },
        "config": {
            "path": "configs/backgrounds/quartic_geometric_jet_campaign.json",
            "file_sha256": "95b58104969e5c3b2a93b382a7564af16b9278ff8151b1ba68d1fef4ac590f1b",
        },
        "test": {
            "path": "tests/test_quartic_geometric_jet_campaign.py",
            "file_sha256": "2c67be8eb34d43bb3e0b119d5cfe258a24388d1067845b9bea0e6ae0b83130c9",
        },
        "artifact": {
            "path": "runs/physics-language/quartic-geometric-jet-campaign/campaign.json",
            "file_sha256": "aa18e643877f4eb7224891e70e929b8d9574a83309aac9007e9f635689d82b65",
            "content_sha256": "3878728a11df567606c18d37cd683ff222a5de20f87248394f7ce75a618562a4",
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
    "P10_target_records_per_candidate": 7,
    "unique_P10_directions_per_candidate": 5,
    "A_leaf_entries_per_direction": 121,
    "source_chunk_column_leaf_entries_per_direction": 11,
    "leaf_derivative_roots_per_candidate": 660,
    "leaf_derivative_roots": 7920,
}
EXPECTED_POLICIES = {
    "leaf_derivative_admission": "require_exact_nonlinear_map_and_live_A_B_C_formula_replay",
    "ordered_D2_root_admission": "require_closed_arithmetic_DAG_derivative_replay",
    "full_D2F": "fail_closed",
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
TARGET_SYMBOLS = {
    "s11[10]": "H_11",
    "s12[10]": "H_12",
    "s13[10]": "H_13",
    "s22[10]": "H_22",
    "s23[10]": "H_23",
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
        raise ValueError("P10 arbitrary-background leaf path escapes project root")
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
        "leaf_contract": EXPECTED_CONTRACT,
        "policies": EXPECTED_POLICIES,
        "seals": EXPECTED_SEALS,
    }:
        raise ValueError("P10 arbitrary-background leaf config boundary changed")


def _load_bound(root: Path, binding: Mapping[str, Any]) -> dict[str, Any]:
    path = _inside(root, str(binding["path"]))
    if _file_sha(path) != binding["file_sha256"]:
        raise ValueError("P10 arbitrary-background leaf artifact file binding changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("content_sha256") != binding["content_sha256"] or value.get(
        "content_sha256"
    ) != _content_sha(value):
        raise ValueError("P10 arbitrary-background leaf artifact content binding changed")
    return value


def _load_bundle(root: Path, bundle: Mapping[str, Any]) -> dict[str, Any]:
    if set(bundle) != {"source", "config", "test", "artifact"}:
        raise ValueError("P10 arbitrary-background leaf evidence bundle changed")
    for label in ("source", "config", "test"):
        binding = bundle[label]
        if _file_sha(_inside(root, binding["path"])) != binding["file_sha256"]:
            raise ValueError("P10 arbitrary-background leaf evidence file changed")
    return _load_bound(root, bundle["artifact"])


def _load_inputs(root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    for label in ("source", "config", "test"):
        binding = EXPECTED_PREDECESSOR[label]
        if _file_sha(_inside(root, binding["path"])) != binding["file_sha256"]:
            raise ValueError("P10 arbitrary-background leaf predecessor file changed")
    predecessor = _load_bound(root, EXPECTED_PREDECESSOR["artifact"])
    geometric = _load_bundle(root, EXPECTED_EVIDENCE["nonlinear_geometric_map"])
    unspecialized = _load_bundle(root, EXPECTED_EVIDENCE["unspecialized_source_blocks"])
    if (
        predecessor.get("gate_counts", {}).get("flat_D2_roots_materialized") != 264
        or predecessor.get("gate_counts", {}).get("general_background_D2_roots_registered") != 0
        or geometric.get("status") != "pass_all_12_exact_nonlinear_geometric_state_to_jet_maps"
        or geometric.get("geometric_control", {}).get("formula_contract_sha256") != FORMULA_SHA256
        or geometric.get("geometric_control", {}).get("passed") is not True
        or unspecialized.get("status")
        != "pass_all_12_complete_unspecialized_principal_source_jacobians_remainder_fail_closed"
        or unspecialized.get("generic_unspecialized_source_jacobian_control", {})
        .get("unspecialized_block_extraction", {})
        .get("block_content_sha256")
        != BLOCK_SHA256
    ):
        raise ValueError("P10 arbitrary-background leaf evidence boundary changed")
    return predecessor, geometric, unspecialized


def _target_templates(predecessor: Mapping[str, Any]) -> list[dict[str, Any]]:
    records = predecessor.get("factorized_leaf_derivative_manifest")
    if not isinstance(records, list) or len(records) != 20:
        raise ValueError("P10 arbitrary-background leaf target inventory changed")
    targets = [row for row in records if row.get("coordinate_atom") in TARGET_SYMBOLS]
    if (
        len(targets) != 5
        or sum(len(row.get("coordinate_ordinals", [])) for row in targets) != 7
        or {row["coordinate_atom"] for row in targets} != set(TARGET_SYMBOLS)
    ):
        raise ValueError("P10 arbitrary-background leaf target subset changed")
    return targets


@cache
def _generic_sparse_derivatives() -> dict[str, dict[str, Any]]:
    blocks = _unspecialized_principal_blocks()
    if blocks["content_sha256"] != BLOCK_SHA256:
        raise ValueError("P10 arbitrary-background leaf live blocks changed")
    data = blocks["data"]
    hessian_symbols = {str(symbol): symbol for symbol in data["hessian_lower"].free_symbols}
    chunks = {
        family: multiplicity
        * (blocks[kind][first] if kind == "B_i" else blocks[kind][first][second])
        for family, _, _, kind, first, second, multiplicity in FAMILY_SPECS
    }
    derivatives = {}
    for atom, symbol_name in TARGET_SYMBOLS.items():
        family = atom.split("[")[0]
        symbol = hessian_symbols[symbol_name]
        derivative_A = blocks["A"].diff(symbol).applyfunc(sp.factor)
        derivative_chunk = chunks[family].diff(symbol).applyfunc(sp.factor)
        A_entries = [
            {"row": row, "column": column, "value": str(derivative_A[row, column])}
            for row in range(11)
            for column in range(11)
            if derivative_A[row, column] != 0
        ]
        chunk_entries = [
            {"row": row, "value": str(derivative_chunk[row, 10])}
            for row in range(11)
            if derivative_chunk[row, 10] != 0
        ]
        dense_values = [
            *[str(derivative_A[row, column]) for row in range(11) for column in range(11)],
            *[str(derivative_chunk[row, 10]) for row in range(11)],
        ]
        derivatives[atom] = {
            "covariant_tangent": {symbol_name: "1"},
            "covariant_tangent_background_independent": True,
            "source_chunk_family": family,
            "source_chunk_input_column": 10,
            "A_derivative_sparse_entries": A_entries,
            "source_chunk_column_derivative_sparse_entries": chunk_entries,
            "total_leaf_derivatives": 132,
            "nonzero_leaf_derivatives": len(A_entries) + len(chunk_entries),
            "zero_leaf_derivatives": 132 - len(A_entries) - len(chunk_entries),
            "generic_dense_values_sha256": _sha(dense_values),
        }
    return derivatives


def _generic_derivative_packets(
    targets: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    derivatives = _generic_sparse_derivatives()
    packets = [
        {
            "coordinate_atom": target["coordinate_atom"],
            "coordinate_column": target["coordinate_column"],
            "coordinate_ordinals": target["coordinate_ordinals"],
            **_copy(derivatives[str(target["coordinate_atom"])]),
        }
        for target in targets
    ]
    if (
        len(packets) != 5
        or sum(row["nonzero_leaf_derivatives"] for row in packets) != 20
        or sum(row["zero_leaf_derivatives"] for row in packets) != 640
    ):
        raise ValueError("P10 arbitrary-background leaf generic derivative census changed")
    return packets


def _candidate_coefficients(
    unspecialized: Mapping[str, Any], candidate_ids: list[str]
) -> dict[str, Mapping[str, Any]]:
    records = {
        str(row["candidate_id"]): row["coefficients"]
        for row in unspecialized.get("certificates", [])
    }
    if set(records) != set(candidate_ids):
        raise ValueError("P10 arbitrary-background leaf candidate set changed")
    return records


def _constant_DAG(values: set[str]) -> tuple[dict[str, Any], dict[str, int]]:
    ordered = sorted(values, key=lambda value: (sp.N(sp.sympify(value)), value))
    nodes = [{"op": "exact_constant", "value": value} for value in ordered]
    body = {
        "schema_version": "sigma-P10-leaf-derivative-exact-constant-DAG-1.0",
        "allowed_operations": ["exact_constant"],
        "node_count": len(nodes),
        "nodes": nodes,
    }
    return {**body, "content_sha256": _sha(body)}, {
        value: index for index, value in enumerate(ordered)
    }


def _candidate_manifests(
    candidate_ids: list[str],
    coefficients: Mapping[str, Mapping[str, Any]],
    generic: list[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    specialized = {}
    values = {"0"}
    alpha = _unspecialized_principal_blocks()["data"]["alpha"]
    for candidate_id in candidate_ids:
        substitution = {alpha: sp.sympify(coefficients[candidate_id]["a10"])}
        specialized[candidate_id] = []
        for packet in generic:
            A_entries = [
                {
                    **entry,
                    "value": str(
                        sp.factor(sp.sympify(entry["value"], locals={"alpha": substitution[alpha]}))
                    ),
                }
                for entry in packet["A_derivative_sparse_entries"]
            ]
            chunk_entries = [
                {
                    **entry,
                    "value": str(
                        sp.factor(sp.sympify(entry["value"], locals={"alpha": substitution[alpha]}))
                    ),
                }
                for entry in packet["source_chunk_column_derivative_sparse_entries"]
            ]
            values.update(entry["value"] for entry in [*A_entries, *chunk_entries])
            specialized[candidate_id].append((packet, A_entries, chunk_entries))
    dag, roots = _constant_DAG(values)
    manifests = []
    for candidate_id in candidate_ids:
        packets = []
        for generic_packet, A_entries, chunk_entries in specialized[candidate_id]:
            A_rooted = [{**entry, "arithmetic_root": roots[entry["value"]]} for entry in A_entries]
            chunk_rooted = [
                {**entry, "arithmetic_root": roots[entry["value"]]} for entry in chunk_entries
            ]
            dense_roots = [roots["0"]] * 132
            for entry in A_rooted:
                dense_roots[11 * entry["row"] + entry["column"]] = entry["arithmetic_root"]
            for entry in chunk_rooted:
                dense_roots[121 + entry["row"]] = entry["arithmetic_root"]
            packets.append(
                {
                    "coordinate_atom": generic_packet["coordinate_atom"],
                    "coordinate_column": generic_packet["coordinate_column"],
                    "coordinate_ordinals": generic_packet["coordinate_ordinals"],
                    "covariant_tangent": generic_packet["covariant_tangent"],
                    "source_chunk_family": generic_packet["source_chunk_family"],
                    "source_chunk_input_column": 10,
                    "A_derivative_shape": [11, 11],
                    "A_derivative_sparse_entries": A_rooted,
                    "source_chunk_column_shape": [11],
                    "source_chunk_column_derivative_sparse_entries": chunk_rooted,
                    "zero_default_arithmetic_root": roots["0"],
                    "arithmetic_dag_sha256": dag["content_sha256"],
                    "total_leaf_derivative_roots": 132,
                    "nonzero_leaf_derivative_roots": len(A_rooted) + len(chunk_rooted),
                    "dense_root_manifest_sha256": _sha(dense_roots),
                    "arbitrary_background_valid": True,
                }
            )
        manifests.append(
            {
                "candidate_id": candidate_id,
                "coefficients": coefficients[candidate_id],
                "direction_packets": packets,
                "unique_P10_directions": 5,
                "P10_target_records": 7,
                "registered_leaf_derivative_roots": 660,
                "nonzero_leaf_derivative_roots": 20,
                "zero_leaf_derivative_roots": 640,
                "P10_ordered_D2_roots_registered": 0,
                "manifest_sha256": _sha(packets),
                "candidate_decision": "pass_P10_leaf_derivatives_D2_propagation_blocked",
                "candidate_rejection_authorized": False,
                "first_blocker": FIRST_BLOCKER,
            }
        )
    return manifests, dag


def _expected_body(
    root: Path,
    config_path: Path,
    predecessor: Mapping[str, Any],
    geometric: Mapping[str, Any],
    unspecialized: Mapping[str, Any],
) -> dict[str, Any]:
    targets = _target_templates(predecessor)
    generic = _generic_derivative_packets(targets)
    candidate_ids = [row["candidate_id"] for row in predecessor["candidate_manifests"]]
    coefficients = _candidate_coefficients(unspecialized, candidate_ids)
    manifests, dag = _candidate_manifests(candidate_ids, coefficients, generic)
    return {
        "schema_version": RESULT_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "decision": "pass_7920_P10_arbitrary_background_leaf_derivative_roots_D2_propagation_blocked",
        "decision_counts": {"pass": 12, "blocked": 0, "reject": 0},
        "downstream_admission_counts": {"pass": 0, "blocked": 12, "reject": 0},
        "first_blocker": FIRST_BLOCKER,
        "leaf_derivative_theorem": {
            "name": "background_independent_scalar_Hessian_coordinate_tangents_and_A_B_C_leaf_derivatives",
            "exact_result": (
                "For the five target scalar second-partial atoms, the exact nonlinear geometric "
                "map gives dH_ij/ds_ij[10]=1 on every nonsingular background. Differentiating "
                "the live unspecialized A/B/C formulas registers all 660 reachable leaf roots per "
                "candidate: 20 nonzero and 640 zero, hence 7,920 roots total."
            ),
            "boundary": (
                "This registers arbitrary-background input-leaf derivatives for the P10 subset. "
                "It does not yet propagate them through the inverse/product D1 DAG, register any "
                "P10 ordered-D2 root, or provide the Pother nonlinear coordinate map."
            ),
        },
        "nonlinear_geometric_map_binding": {
            "formula_contract_sha256": FORMULA_SHA256,
            "status": geometric["status"],
            "P10_coordinate_rule": (
                "scalar_hessian_ij=partial_ij_phi-Gamma^rho_ij_partial_rho_phi_so_"
                "partial_scalar_hessian_ij_over_partial_sij_scalar_equals_one"
            ),
            "arbitrary_background_scope": True,
        },
        "generic_derivative_packets": generic,
        "leaf_derivative_arithmetic_DAG": dag,
        "candidate_manifests": manifests,
        "manifest_sha256": _sha([row["manifest_sha256"] for row in manifests]),
        "gate_counts": {
            "selected_candidates": 12,
            "P10_target_records": 84,
            "unique_P10_directions_per_candidate": 5,
            "registered_arbitrary_background_leaf_derivative_roots": 7920,
            "nonzero_leaf_derivative_roots": 240,
            "zero_leaf_derivative_roots": 7680,
            "P10_ordered_D2_roots_registered": 0,
            "P10_ordered_D2_roots_blocked": 84,
            "Pother_leaf_derivative_roots_registered": 0,
            "Pother_leaf_derivative_roots_remaining": 23760,
            "all_target_ordered_D2_roots_registered": 0,
            "all_target_ordered_D2_roots_blocked": 264,
            "complete_ordered_D2F_tensors_registered": 0,
            "global_H7_closures": 0,
            "nonlinear_PDE_closures": 0,
            "lifespans_proved": 0,
        },
        "claim_seals": {
            "nonlinear_geometric_map_directly_bound": True,
            "P10_coordinate_to_Hessian_tangents_arbitrary_background_registered": True,
            "all_7920_P10_leaf_derivative_roots_registered": True,
            "P10_ordered_D2_roots_registered": False,
            "Pother_leaf_derivative_roots_registered": False,
            "physical_no_go_proved": False,
            "complete_ordered_D2F_tensor_registered": False,
            "global_H7_energy_closed": False,
            "nonlinear_PDE_closed": False,
            "nonlinear_lifespan_proved": False,
            "candidate_theory_rejected": False,
            "observational_claim_made": False,
        },
        "exact_controls": {
            "omit_connection_from_nonlinear_Hessian_map": {"rejected": True},
            "promote_leaf_derivative_root_to_D2_root": {"rejected": True},
            "default_Pother_leaf_derivatives_to_zero": {"rejected": True},
            "promote_P10_subset_to_complete_D2F": {"rejected": True},
            "infer_physical_no_go_from_remaining_schema": {"rejected": True},
            "reject_candidate_from_D2_propagation_blocker": {"rejected": True},
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
            "candidate-bound arbitrary-background coordinate-to-Hessian and A/B/C input-leaf "
            "derivatives for the five unique P10 target directions only; no propagated D2 root, "
            "Pother leaf jet, physical no-go, full D2F, H7, PDE, lifespan, rejection, or observation"
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
        raise ValueError("P10 arbitrary-background leaf binding keys changed")
    for label, relative in {
        "source": SOURCE_PATH,
        "config": CONFIG_PATH,
        "test": TEST_PATH,
    }.items():
        if bindings[label] != {
            "path": relative,
            "file_sha256": _file_sha(_inside(root, relative)),
        }:
            raise ValueError("P10 arbitrary-background leaf local binding changed")
    if bindings["predecessor"] != EXPECTED_PREDECESSOR:
        raise ValueError("P10 arbitrary-background leaf predecessor binding changed")
    if bindings["direct_evidence"] != EXPECTED_EVIDENCE:
        raise ValueError("P10 arbitrary-background leaf evidence binding changed")


def _validate_result(value: Mapping[str, Any], *, root: Path | None = None) -> None:
    validation_root = (root or Path(__file__).resolve().parents[2]).resolve()
    if value.get("content_sha256") != _content_sha(value):
        raise ValueError("P10 arbitrary-background leaf content hash changed")
    _validate_bindings(value, validation_root)
    config_path = _inside(validation_root, CONFIG_PATH)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    predecessor, geometric, unspecialized = _load_inputs(validation_root)
    expected = _expected_body(validation_root, config_path, predecessor, geometric, unspecialized)
    if {key: item for key, item in value.items() if key != "content_sha256"} != expected:
        raise ValueError("P10 arbitrary-background leaf result boundary changed")


def build_gate(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    root = config_path.parents[2]
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    predecessor, geometric, unspecialized = _load_inputs(root)
    body = _expected_body(root, config_path, predecessor, geometric, unspecialized)
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
