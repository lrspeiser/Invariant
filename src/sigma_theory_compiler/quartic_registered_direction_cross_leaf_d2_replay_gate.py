"""Seal registered-direction cross derivatives from exact A/B/C leaf arithmetic."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable, Mapping
from functools import cache
from pathlib import Path
from typing import Any

import sympy as sp

from .quartic_full_d2f_typed_partition_row_extension_gate import (
    _closure,
    _ContentReplay,
    _replay,
)
from .quartic_p10_arbitrary_background_leaf_derivative_gate import TARGET_SYMBOLS
from .quartic_p10_inverse_product_d2_replay_gate import _load_D1
from .quartic_pother_arbitrary_background_leaf_derivative_gate import (
    BLOCK_SHA256,
    EXPECTED_POTHER,
    _background_symbols,
    _expression_DAG,
    _second_metric_tangent,
)
from .quartic_scalar_hessian_d2_integrability_gate import FAMILY_SPECS
from .quartic_unspecialized_source_jacobian_campaign import (
    _unspecialized_principal_blocks,
)

CONFIG_SCHEMA = "sigma-quartic-registered-direction-cross-leaf-d2-replay-config-1.0"
RESULT_SCHEMA = "sigma-quartic-registered-direction-cross-leaf-d2-replay-gate-1.0"
CAMPAIGN_ID = "quartic-registered-direction-cross-leaf-d2-replay-001"
CONFIG_PATH = "configs/backgrounds/quartic_registered_direction_cross_leaf_d2_replay_gate.json"
SOURCE_PATH = "src/sigma_theory_compiler/quartic_registered_direction_cross_leaf_d2_replay_gate.py"
TEST_PATH = "tests/test_quartic_registered_direction_cross_leaf_d2_replay_gate.py"
OUTPUT_PATH = (
    "runs/physics-language/quartic-registered-direction-cross-leaf-d2-replay-gate/campaign.json"
)
FIRST_BLOCKER = (
    "register_coordinate_to_covariant_tangents_for_the_131_unregistered_derivative_"
    "directions_before_attempting_the_31702_registered_D1_to_unregistered_derivative_"
    "entries_per_candidate"
)
PREDECESSOR = {
    "source": {
        "path": (
            "src/sigma_theory_compiler/quartic_full_d2f_typed_partition_row_extension_gate.py"
        ),
        "file_sha256": "8536f8eeda3d2326d2c796669a6f662ad322d82fc0c43e751fd677677cf9a99c",
    },
    "config": {
        "path": ("configs/backgrounds/quartic_full_d2f_typed_partition_row_extension_gate.json"),
        "file_sha256": "fa78b937e5c42385649dce056af5094c92dd3222fb2c61dcc02d0bd1198ba448",
    },
    "test": {
        "path": "tests/test_quartic_full_d2f_typed_partition_row_extension_gate.py",
        "file_sha256": "6a98a6db37f17207efd5d5bf994c6cd2a4719105f64cf4d5ceab6da996847967",
    },
    "artifact": {
        "path": (
            "runs/physics-language/quartic-full-d2f-typed-partition-row-extension-gate/"
            "campaign.json"
        ),
        "file_sha256": "9502843234509a4ddd21631acdfe412d0f17fe3552d7c9cac0daf7fb1475190a",
        "content_sha256": "76eff324a16396dfbeee91552220b26dd745b3c22aa5dd6fb9538fffa843bece",
    },
}
DIRECT_EVIDENCE = {
    "D1_artifact": {
        "path": (
            "runs/physics-language/quartic-full-source-jacobian-arithmetic-campaign/campaign.json"
        ),
        "file_sha256": "e893ebcaef464b958516279c557382fb76ecdb0fd542b3e3fed6a347076fcdae",
        "content_sha256": "1707b7258fd434f68b06c7af6bc447b4136624b9916992df8b412e048ab6538a",
    },
    "geometric_source": {
        "path": "src/sigma_theory_compiler/quartic_geometric_jet_campaign.py",
        "file_sha256": "d0600d6475d32d06a00140ab230aa41b3c057aef7a968163989fc5028d6acd21",
    },
    "p10_leaf_artifact": {
        "path": (
            "runs/physics-language/quartic-p10-arbitrary-background-leaf-derivative-gate/"
            "campaign.json"
        ),
        "file_sha256": "c74171c48d7fc4f80de8f0c51b2b2700a1ce33de8795c3a999cee7c957b35869",
        "content_sha256": "51f76fa7ebc81ab2f570bfe5ad920215420e005687d0c861b24ea6da766c37e0",
    },
    "p10_leaf_source": {
        "path": (
            "src/sigma_theory_compiler/quartic_p10_arbitrary_background_leaf_derivative_gate.py"
        ),
        "file_sha256": "6bb9ab72854d74e21a120f41fc9299712d5d64d0c9c9aeb243d5c3948b689239",
    },
    "pother_leaf_artifact": {
        "path": (
            "runs/physics-language/"
            "quartic-pother-arbitrary-background-leaf-derivative-gate/campaign.json"
        ),
        "file_sha256": "c687e15839628dcca3480740ea8ee568576461c200ce8937ab5499a71e9e49c2",
        "content_sha256": "b2f4eacd73026bc92a057be0ad5340d487ef0ff3b18353e3412d7aea5475b670",
    },
    "pother_leaf_source": {
        "path": (
            "src/sigma_theory_compiler/quartic_pother_arbitrary_background_leaf_derivative_gate.py"
        ),
        "file_sha256": "7e34d374df437075f040b40f7435cb0ab07a4ca60ccf696a1730f6153761f2a0",
    },
    "scalar_hessian_source": {
        "path": ("src/sigma_theory_compiler/quartic_scalar_hessian_d2_integrability_gate.py"),
        "file_sha256": "bb570371bdf3a33bc0bb492b48cf642fcf1f950cd6a9576eb9e85670c9321064",
    },
    "unspecialized_source": {
        "path": "src/sigma_theory_compiler/quartic_unspecialized_source_jacobian_campaign.py",
        "file_sha256": "f5a8649b52bd7f2384ee9087d0f5f6d8850a5c5bc443fbe823c6b09655ce9616",
    },
}
CONTRACT = {
    "candidate_count": 12,
    "registered_direction_slots": 22,
    "source_rows": 11,
    "cross_atom_leaf_packets_per_candidate": 380,
    "cross_atom_leaf_roots_per_candidate": 50160,
    "off_diagonal_entries_per_candidate": 5082,
}
POLICIES = {
    "cross_leaf_admission": (
        "require_live_exact_A_B_C_differentiation_under_registered_coordinate_tangent"
    ),
    "full_D2F": "fail_closed",
    "high_atom_identity": "fail_closed",
    "global_H7": "fail_closed",
    "nonlinear_PDE": "fail_closed",
    "physical_no_go": "forbidden",
    "candidate_rejection": "forbidden",
}
SEALS = {
    "observations_opened": False,
    "live_SQLite_opened": False,
    "GPU_execution_used": False,
}
_EXPECTED_CACHE: dict[tuple[str, ...], dict[str, Any]] = {}
_REPLAY_CACHE: dict[tuple[int, str, str], str] = {}


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _content_sha(value: Mapping[str, Any]) -> str:
    return _sha({key: item for key, item in value.items() if key != "content_sha256"})


def _copy(value: Any) -> Any:
    return json.loads(_canonical(value))


def _inside(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if path != root and root not in path.parents:
        raise ValueError("cross-direction replay path escapes project root")
    return path


def _load_bound(root: Path, binding: Mapping[str, Any]) -> dict[str, Any]:
    path = _inside(root, str(binding["path"]))
    if _file_sha(path) != binding["file_sha256"]:
        raise ValueError("cross-direction replay file binding changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("content_sha256") != binding["content_sha256"] or value.get(
        "content_sha256"
    ) != _content_sha(value):
        raise ValueError("cross-direction replay content binding changed")
    return value


def _validate_config(value: Mapping[str, Any]) -> None:
    if value != {
        "schema_version": CONFIG_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "output_path": OUTPUT_PATH,
        "predecessor": PREDECESSOR,
        "direct_evidence": DIRECT_EVIDENCE,
        "cross_direction_contract": CONTRACT,
        "policies": POLICIES,
        "seals": SEALS,
    }:
        raise ValueError("cross-direction replay config boundary changed")


def _load_inputs(
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    for binding in [
        *[PREDECESSOR[label] for label in ("source", "config", "test")],
        *[
            DIRECT_EVIDENCE[label]
            for label in (
                "geometric_source",
                "p10_leaf_source",
                "pother_leaf_source",
                "scalar_hessian_source",
                "unspecialized_source",
            )
        ],
    ]:
        if _file_sha(_inside(root, str(binding["path"]))) != binding["file_sha256"]:
            raise ValueError("cross-direction replay source evidence changed")
    predecessor = _load_bound(root, PREDECESSOR["artifact"])
    p10_leaf = _load_bound(root, DIRECT_EVIDENCE["p10_leaf_artifact"])
    pother_leaf = _load_bound(root, DIRECT_EVIDENCE["pother_leaf_artifact"])
    d1 = _load_bound(root, DIRECT_EVIDENCE["D1_artifact"])
    if (
        predecessor.get("gate_counts", {}).get("registered_per_candidate") != 242
        or predecessor.get("gate_counts", {}).get("remaining_per_candidate") != 257257
        or predecessor.get("first_blocker")
        != "register_cross_direction_leaf_derivatives_for_the_5082_registered_direction_off_diagonal_entries_per_candidate"
        or p10_leaf.get("gate_counts", {}).get(
            "registered_arbitrary_background_leaf_derivative_roots"
        )
        != 7920
        or pother_leaf.get("gate_counts", {}).get("registered_Pother_leaf_derivative_roots")
        != 23760
    ):
        raise ValueError("cross-direction replay predecessor boundary changed")
    return predecessor, p10_leaf, pother_leaf, d1


def _target_atoms() -> list[str]:
    atoms = [*TARGET_SYMBOLS, *(row[0] for row in EXPECTED_POTHER)]
    if len(atoms) != 20 or len(set(atoms)) != 20:
        raise ValueError("cross-direction replay target atom inventory changed")
    return atoms


def _chunks(blocks: Mapping[str, Any]) -> dict[str, sp.Matrix]:
    return {
        family: multiplicity
        * (blocks[kind][first] if kind == "B_i" else blocks[kind][first][second])
        for family, _, _, kind, first, second, multiplicity in FAMILY_SPECS
    }


@cache
def _generic_cross_packets() -> tuple[dict[str, Any], ...]:
    blocks = _unspecialized_principal_blocks()
    if blocks["content_sha256"] != BLOCK_SHA256:
        raise ValueError("cross-direction replay live A/B/C blocks changed")
    data = blocks["data"]
    hessian = {str(symbol): symbol for symbol in data["hessian_lower"].free_symbols}
    einstein = sorted(data["einstein_upper"].free_symbols, key=str)
    directions: list[tuple[str, str, Callable[[sp.Expr], sp.Expr]]] = []
    for atom, symbol_name in TARGET_SYMBOLS.items():
        symbol = hessian[symbol_name]
        directions.append(
            (
                atom,
                "P10_flat_covariant_Hessian_unit_tangent",
                lambda x, s=symbol: sp.factor(sp.diff(x, s)),
            )
        )
    for atom, _, _ in EXPECTED_POTHER:
        _, _, _, tangent = _second_metric_tangent(atom)
        directions.append(
            (
                atom,
                "Pother_coordinate_second_metric_to_G_upper_tangent",
                lambda x, t=tangent: sp.factor(
                    sum(sp.diff(x, symbol) * t[str(symbol)] for symbol in einstein)
                ),
            )
        )
    chunks = _chunks(blocks)
    packets = []
    for derivative_atom, tangent_kind, chain in directions:
        derivative_A = blocks["A"].applyfunc(chain)
        for target_atom in _target_atoms():
            if target_atom == derivative_atom:
                continue
            family, field_text = target_atom.split("[")
            field = int(field_text[:-1])
            derivative_chunk = chunks[family].applyfunc(chain)
            sparse_A = [
                {"row": row, "column": column, "value": str(derivative_A[row, column])}
                for row in range(11)
                for column in range(11)
                if derivative_A[row, column] != 0
            ]
            sparse_chunk = [
                {"row": row, "value": str(derivative_chunk[row, field])}
                for row in range(11)
                if derivative_chunk[row, field] != 0
            ]
            packets.append(
                {
                    "D1_target_atom": target_atom,
                    "derivative_atom": derivative_atom,
                    "derivative_tangent_kind": tangent_kind,
                    "source_chunk_family": family,
                    "source_chunk_input_column": field,
                    "A_derivative_sparse_entries": sparse_A,
                    "source_chunk_column_derivative_sparse_entries": sparse_chunk,
                    "total_leaf_derivative_roots": 132,
                    "nonzero_leaf_derivative_roots": len(sparse_A) + len(sparse_chunk),
                }
            )
    if (
        len(packets) != 380
        or sum(row["nonzero_leaf_derivative_roots"] for row in packets) != 602
        or sum(132 - row["nonzero_leaf_derivative_roots"] for row in packets) != 49558
    ):
        raise ValueError("cross-direction replay generic leaf census changed")
    return tuple(packets)


def _candidate_coefficients(p10_leaf: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    coefficients = {
        str(row["candidate_id"]): row["coefficients"] for row in p10_leaf["candidate_manifests"]
    }
    if len(coefficients) != 12:
        raise ValueError("cross-direction replay candidate coefficients changed")
    return coefficients


def _specialized_cross_packets(
    coefficients: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    generic = _generic_cross_packets()
    alpha = _unspecialized_principal_blocks()["data"]["alpha"]
    _, _, background_symbols = _background_symbols()
    locals_map = {str(symbol): symbol for symbol in background_symbols}
    locals_map["alpha"] = alpha
    staged: dict[str, list[dict[str, Any]]] = {}
    staged_classes: dict[str, list[dict[str, Any]]] = {}
    values = {"0"}
    for candidate_id, candidate_coefficients in coefficients.items():
        coefficient_key = str(sp.sympify(candidate_coefficients["a10"]))
        rows = staged_classes.get(coefficient_key)
        if rows is None:
            substitution = {alpha: sp.sympify(coefficient_key)}
            rows = []
            for packet in generic:
                a_entries = []
                for entry in packet["A_derivative_sparse_entries"]:
                    value = str(
                        sp.factor(sp.sympify(entry["value"], locals=locals_map).subs(substitution))
                    )
                    if value != "0":
                        a_entries.append({**entry, "value": value})
                        values.add(value)
                chunk_entries = []
                for entry in packet["source_chunk_column_derivative_sparse_entries"]:
                    value = str(
                        sp.factor(sp.sympify(entry["value"], locals=locals_map).subs(substitution))
                    )
                    if value != "0":
                        chunk_entries.append({**entry, "value": value})
                        values.add(value)
                rows.append(
                    {
                        **{
                            key: packet[key]
                            for key in (
                                "D1_target_atom",
                                "derivative_atom",
                                "derivative_tangent_kind",
                                "source_chunk_family",
                                "source_chunk_input_column",
                            )
                        },
                        "A_derivative_sparse_entries": a_entries,
                        "source_chunk_column_derivative_sparse_entries": chunk_entries,
                    }
                )
            staged_classes[coefficient_key] = rows
        staged[candidate_id] = rows
    dag, roots = _expression_DAG(values)

    def root_for(value: str) -> int:
        canonical = str(sp.sympify(value, locals=locals_map))
        return roots[canonical]

    specialized = {}
    for candidate_id, rows in staged.items():
        packets = []
        for row in rows:
            a_entries = [
                {**entry, "arithmetic_root": root_for(entry["value"])}
                for entry in row["A_derivative_sparse_entries"]
            ]
            chunk_entries = [
                {**entry, "arithmetic_root": root_for(entry["value"])}
                for entry in row["source_chunk_column_derivative_sparse_entries"]
            ]
            dense = [roots["0"]] * 132
            for entry in a_entries:
                dense[11 * entry["row"] + entry["column"]] = entry["arithmetic_root"]
            for entry in chunk_entries:
                dense[121 + entry["row"]] = entry["arithmetic_root"]
            packets.append(
                {
                    **{
                        key: row[key]
                        for key in (
                            "D1_target_atom",
                            "derivative_atom",
                            "derivative_tangent_kind",
                            "source_chunk_family",
                            "source_chunk_input_column",
                        )
                    },
                    "A_derivative_shape": [11, 11],
                    "A_derivative_sparse_entries": a_entries,
                    "source_chunk_column_shape": [11],
                    "source_chunk_column_derivative_sparse_entries": chunk_entries,
                    "zero_default_arithmetic_root": roots["0"],
                    "leaf_arithmetic_dag_sha256": dag["content_sha256"],
                    "total_leaf_derivative_roots": 132,
                    "nonzero_leaf_derivative_roots": len(a_entries) + len(chunk_entries),
                    "dense_root_manifest_sha256": _sha(dense),
                    "registered_symbolic_background_scope": True,
                }
            )
        specialized[candidate_id] = packets
    return specialized, dag


def _cross_leaf_roots(packet: Mapping[str, Any]) -> dict[str, tuple[int, str]]:
    zero = int(packet["zero_default_arithmetic_root"])
    values = {f"A[{row},{column}]": (zero, "0") for row in range(11) for column in range(11)}
    family = str(packet["source_chunk_family"])
    field = int(packet["source_chunk_input_column"])
    chunk = f"B_{family[2:]}" if family.startswith("s0") else f"C_{family[1:]}"
    values.update({f"{chunk}[{row},{field}]": (zero, "0") for row in range(11)})
    for entry in packet["A_derivative_sparse_entries"]:
        values[f"A[{entry['row']},{entry['column']}]"] = (
            int(entry["arithmetic_root"]),
            str(entry["value"]),
        )
    for entry in packet["source_chunk_column_derivative_sparse_entries"]:
        values[f"{chunk}[{entry['row']},{field}]"] = (
            int(entry["arithmetic_root"]),
            str(entry["value"]),
        )
    if len(values) != 132:
        raise ValueError("cross-direction replay dense leaf inventory changed")
    return values


def _replay_cross(
    packet: Mapping[str, Any], root_index: int, nodes: list[Mapping[str, Any]]
) -> str:
    key = (
        root_index,
        str(packet["dense_root_manifest_sha256"]),
        str(packet["derivative_atom"]),
    )
    cached = _REPLAY_CACHE.get(key)
    if cached is not None:
        return cached
    closure = _closure(root_index, nodes)
    leaves = _cross_leaf_roots(packet)
    labels = {
        str(nodes[index]["label"])
        for index in closure
        if nodes[index]["op"] == "exact_component_input"
    }
    if labels != set(leaves):
        raise ValueError("cross-direction replay D1 closure/leaf labels changed")
    replay = _ContentReplay(
        coordinate_atom=str(packet["derivative_atom"]),
        leaf_dag=str(packet["leaf_arithmetic_dag_sha256"]),
    )
    derivatives: dict[int, str] = {}
    for index in closure:
        node = nodes[index]
        op = node["op"]
        if op == "exact_constant":
            derivative = replay.zero
        elif op == "exact_component_input":
            leaf_root, value = leaves[str(node["label"])]
            derivative = replay.leaf(str(node["label"]), leaf_root, value)
        elif op == "exact_add":
            derivative = replay.add([derivatives[item] for item in node["arguments"]])
        elif op == "exact_negate":
            derivative = replay.negate(derivatives[int(node["argument"])])
        elif op == "exact_multiply":
            left, right = int(node["left"]), int(node["right"])
            derivative = replay.add(
                [
                    replay.multiply(derivatives[left], replay.primal(right)),
                    replay.multiply(replay.primal(left), derivatives[right]),
                ]
            )
        elif op == "exact_divide":
            numerator, denominator = int(node["numerator"]), int(node["denominator"])
            derivative = replay.divide(
                replay.add(
                    [
                        replay.multiply(derivatives[numerator], replay.primal(denominator)),
                        replay.negate(
                            replay.multiply(replay.primal(numerator), derivatives[denominator])
                        ),
                    ]
                ),
                replay.multiply(replay.primal(denominator), replay.primal(denominator)),
            )
        else:
            raise ValueError("cross-direction replay D1 operator changed")
        derivatives[index] = derivative
    result = derivatives[root_index]
    _REPLAY_CACHE[key] = result
    return result


def _direction_packets(
    candidate_id: str,
    p10_leaf: Mapping[str, Any],
    pother_leaf: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[tuple[int, str]]]:
    packets = []
    for artifact in (p10_leaf, pother_leaf):
        candidate = next(
            row for row in artifact["candidate_manifests"] if row["candidate_id"] == candidate_id
        )
        packets.extend(candidate["direction_packets"])
    slots = sorted(
        (int(slot), str(packet["coordinate_atom"]))
        for packet in packets
        for slot in packet["coordinate_ordinals"]
    )
    if (
        len(packets) != 20
        or slots != [(slot, atom) for slot, atom in slots]
        or {slot for slot, _ in slots} != set(range(22))
    ):
        raise ValueError("cross-direction replay registered slot inventory changed")
    return packets, slots


def _candidate_manifests(
    predecessor: Mapping[str, Any],
    p10_leaf: Mapping[str, Any],
    pother_leaf: Mapping[str, Any],
    cross_packets: Mapping[str, list[dict[str, Any]]],
    nodes: list[Mapping[str, Any]],
    entries: Mapping[tuple[int, str], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    manifests = []
    global_ids: set[str] = set()
    content_replays: dict[tuple[str, int, str, str], str] = {}
    for previous in predecessor["candidate_manifests"]:
        candidate_id = str(previous["candidate_id"])
        existing_packets, slots = _direction_packets(candidate_id, p10_leaf, pother_leaf)
        existing = {str(packet["coordinate_atom"]): packet for packet in existing_packets}
        crossed = {
            (str(packet["D1_target_atom"]), str(packet["derivative_atom"])): packet
            for packet in cross_packets[candidate_id]
        }
        if len(crossed) != 380:
            raise ValueError("cross-direction replay packet key census changed")
        records = []
        reused = 0
        for target_slot, target_atom in slots:
            for derivative_slot, derivative_atom in slots:
                if target_slot == derivative_slot:
                    continue
                same_atom = target_atom == derivative_atom
                if same_atom:
                    reused += 11
                packet = (
                    existing[derivative_atom]
                    if same_atom
                    else crossed[(target_atom, derivative_atom)]
                )
                for source_row in range(11):
                    root_index = int(entries[(source_row, target_atom)]["arithmetic_root"])
                    replay_key = (
                        "same" if same_atom else "cross",
                        root_index,
                        str(packet["dense_root_manifest_sha256"]),
                        derivative_atom,
                    )
                    merkle = content_replays.get(replay_key)
                    if merkle is None:
                        merkle = (
                            _replay(candidate_id, packet, root_index, nodes)
                            if same_atom
                            else _replay_cross(packet, root_index, nodes)
                        )
                        content_replays[replay_key] = merkle
                    identity = {
                        "candidate_id": candidate_id,
                        "source_row": source_row,
                        "D1_target_slot": target_slot,
                        "D1_target_atom": target_atom,
                        "derivative_slot": derivative_slot,
                        "derivative_atom": derivative_atom,
                        "D1_arithmetic_root": root_index,
                        "D2_merkle_root_sha256": merkle,
                    }
                    record_id = _sha(identity)
                    if record_id in global_ids:
                        raise AssertionError("cross-direction replay record collision")
                    global_ids.add(record_id)
                    records.append(
                        {
                            **identity,
                            "record_id": record_id,
                            "root_status": "sealed_exact_registered_symbolic_background_merkle_replay",
                            "leaf_jet_status": (
                                "reused_same_atom_registered_leaf_jet"
                                if same_atom
                                else "new_exact_cross_atom_leaf_jet"
                            ),
                        }
                    )
        if len(records) != 5082 or reused != 44:
            raise ValueError("cross-direction replay candidate record census changed")
        packets = cross_packets[candidate_id]
        manifests.append(
            {
                "candidate_id": candidate_id,
                "cross_atom_leaf_packets": packets,
                "cross_atom_leaf_packet_count": 380,
                "cross_atom_leaf_roots": 50160,
                "cross_atom_nonzero_leaf_roots": sum(
                    row["nonzero_leaf_derivative_roots"] for row in packets
                ),
                "same_atom_off_diagonal_records_reusing_prior_leaf_jets": 44,
                "new_cross_atom_records": 5038,
                "off_diagonal_records": records,
                "off_diagonal_records_registered": 5082,
                "registered_records_total": 5324,
                "manifest_sha256": _sha([row["record_id"] for row in records]),
                "candidate_decision": "pass_all_5082_registered_direction_off_diagonal_entries",
                "candidate_rejection_authorized": False,
            }
        )
    if len(global_ids) != 60984:
        raise AssertionError("cross-direction replay global record census changed")
    return manifests


def _expected_body(
    root: Path,
    config_path: Path,
    predecessor: Mapping[str, Any],
    p10_leaf: Mapping[str, Any],
    pother_leaf: Mapping[str, Any],
    d1: Mapping[str, Any],
) -> dict[str, Any]:
    cache_key = (
        str(root),
        _file_sha(_inside(root, SOURCE_PATH)),
        _file_sha(config_path),
        _file_sha(_inside(root, TEST_PATH)),
        str(predecessor["content_sha256"]),
        str(p10_leaf["content_sha256"]),
        str(pother_leaf["content_sha256"]),
        str(d1["content_sha256"]),
    )
    cached = _EXPECTED_CACHE.get(cache_key)
    if cached is not None:
        return _copy(cached)
    nodes, entries = _load_D1(d1)
    coefficients = _candidate_coefficients(p10_leaf)
    cross_packets, leaf_dag = _specialized_cross_packets(coefficients)
    manifests = _candidate_manifests(
        predecessor, p10_leaf, pother_leaf, cross_packets, nodes, entries
    )
    nonzero = [row["cross_atom_nonzero_leaf_roots"] for row in manifests]
    partition = [
        {
            "block": "registered_direction_same_direction_rows",
            "per_candidate": 242,
            "status": "registered",
        },
        {
            "block": "registered_direction_off_diagonal_pairs",
            "per_candidate": 5082,
            "status": "registered_here",
        },
        {
            "block": "registered_D1_unregistered_derivative",
            "per_candidate": 31702,
            "status": "blocked",
        },
        {
            "block": "unregistered_D1_registered_derivative",
            "per_candidate": 31702,
            "status": "blocked_unexamined",
        },
        {"block": "both_directions_unregistered", "per_candidate": 188771, "status": "blocked"},
    ]
    body = {
        "schema_version": RESULT_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "decision": "pass_all_5082_registered_direction_off_diagonal_entries_full_D2F_blocked",
        "decision_counts": {"pass": 12, "blocked": 0, "reject": 0},
        "downstream_admission_counts": {"pass": 0, "blocked": 12, "reject": 0},
        "first_blocker": FIRST_BLOCKER,
        "cross_direction_theorem": {
            "name": "registered_coordinate_tangent_cross_A_B_C_chain_and_closed_D1_replay",
            "exact_result": (
                "The live unspecialized A/B/C arithmetic and the 20 unique registered coordinate "
                "tangents determine 380 ordered cross-atom leaf packets per candidate. These "
                "packets seal 5,038 cross-atom entries, while 44 off-diagonal slot entries reuse "
                "the already registered same-atom leaf jets, thereby sealing all 5,082 entries "
                "in the registered-direction off-diagonal block."
            ),
            "boundary": (
                "This is an exact result only for the registered 22 direction slots, live "
                "principal A/B/C model, nonsingular symbolic metric background, and D1 quotient "
                "domain. It does not register any of the other 131 coordinate directions or "
                "prove the complete D2F tensor, high-atom identity, H7 estimate, PDE closure, "
                "physical no-go, or candidate rejection."
            ),
        },
        "typed_full_domain_partition": partition,
        "partition_sha256": _sha(partition),
        "cross_leaf_arithmetic_DAG": leaf_dag,
        "generic_cross_leaf_packet_count": 380,
        "generic_cross_leaf_nonzero_roots": 602,
        "generic_cross_leaf_zero_roots": 49558,
        "candidate_cross_leaf_nonzero_root_counts": nonzero,
        "candidate_manifests": manifests,
        "manifest_sha256": _sha([row["manifest_sha256"] for row in manifests]),
        "gate_counts": {
            "selected_candidates": 12,
            "registered_direction_slots": 22,
            "unique_registered_atoms": 20,
            "cross_atom_leaf_packets_per_candidate": 380,
            "cross_atom_leaf_roots_per_candidate": 50160,
            "same_atom_off_diagonal_records_reusing_prior_leaf_jets_per_candidate": 44,
            "new_cross_atom_records_per_candidate": 5038,
            "new_off_diagonal_records_per_candidate": 5082,
            "new_off_diagonal_records_all_candidates": 60984,
            "previously_registered_per_candidate": 242,
            "registered_per_candidate": 5324,
            "remaining_per_candidate": 252175,
            "full_entries_per_candidate": 257499,
            "complete_D2F_tensors": 0,
            "H7_closures": 0,
            "PDE_closures": 0,
        },
        "claim_seals": {
            "all_registered_direction_off_diagonal_entries_sealed": True,
            "complete_D2F": False,
            "full_high_atom_identity": False,
            "global_H7": False,
            "nonlinear_PDE": False,
            "physical_no_go": False,
            "candidate_rejected": False,
        },
        "source_bindings": {
            "source": {"path": SOURCE_PATH, "file_sha256": _file_sha(_inside(root, SOURCE_PATH))},
            "config": {"path": CONFIG_PATH, "file_sha256": _file_sha(config_path)},
            "test": {"path": TEST_PATH, "file_sha256": _file_sha(_inside(root, TEST_PATH))},
            "predecessor": _copy(PREDECESSOR),
            "direct_evidence": _copy(DIRECT_EVIDENCE),
            "D1_artifact": _copy(DIRECT_EVIDENCE["D1_artifact"]),
        },
        "data_seals": _copy(SEALS),
        "scope": (
            "registered 22-slot off-diagonal block only; no remaining D2F, high-atom, H7, "
            "PDE, lifespan, physical no-go, observation, or rejection claim"
        ),
    }
    _EXPECTED_CACHE[cache_key] = _copy(body)
    return body


def build_gate(config_path: Path) -> dict[str, Any]:
    root = config_path.resolve().parents[2]
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    predecessor, p10_leaf, pother_leaf, d1 = _load_inputs(root)
    body = _expected_body(root, config_path, predecessor, p10_leaf, pother_leaf, d1)
    return {**body, "content_sha256": _sha(body)}


def _validate_result(value: Mapping[str, Any], *, root: Path) -> None:
    if set(value) != {
        "schema_version",
        "campaign_id",
        "decision",
        "decision_counts",
        "downstream_admission_counts",
        "first_blocker",
        "cross_direction_theorem",
        "typed_full_domain_partition",
        "partition_sha256",
        "cross_leaf_arithmetic_DAG",
        "generic_cross_leaf_packet_count",
        "generic_cross_leaf_nonzero_roots",
        "generic_cross_leaf_zero_roots",
        "candidate_cross_leaf_nonzero_root_counts",
        "candidate_manifests",
        "manifest_sha256",
        "gate_counts",
        "claim_seals",
        "source_bindings",
        "data_seals",
        "scope",
        "content_sha256",
    }:
        raise ValueError("cross-direction replay result keys changed")
    if value.get("content_sha256") != _content_sha(value):
        raise ValueError("cross-direction replay result content seal changed")
    expected = build_gate(_inside(root, CONFIG_PATH))
    if value != expected:
        raise ValueError("cross-direction replay result differs from exact live replay")


def write_gate(config_path: Path) -> Path:
    result = build_gate(config_path)
    output = _inside(config_path.resolve().parents[2], OUTPUT_PATH)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(CONFIG_PATH))
    print(write_gate(parser.parse_args().config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
