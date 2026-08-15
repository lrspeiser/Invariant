"""Replay the bound inverse/product D1 DAG along the registered P10 leaf jets."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from functools import cache
from pathlib import Path
from typing import Any

CONFIG_SCHEMA = "sigma-quartic-p10-inverse-product-d2-replay-config-1.0"
RESULT_SCHEMA = "sigma-quartic-p10-inverse-product-d2-replay-gate-1.0"
CAMPAIGN_ID = "quartic-p10-inverse-product-d2-replay-001"
CONFIG_PATH = "configs/backgrounds/quartic_p10_inverse_product_d2_replay_gate.json"
SOURCE_PATH = "src/sigma_theory_compiler/quartic_p10_inverse_product_d2_replay_gate.py"
TEST_PATH = "tests/test_quartic_p10_inverse_product_d2_replay_gate.py"
OUTPUT_PATH = "runs/physics-language/quartic-p10-inverse-product-d2-replay-gate/campaign.json"
FIRST_BLOCKER = (
    "register_candidate_bound_Pother_A_B_C_leaf_derivatives_then_replay_the_remaining_"
    "180_ordered_mixed_D2_roots"
)
D1_DAG_SHA256 = "4a227fcf136d440c4dd55e4c5525eef8e5b73681339062d7ca44cb000944ec5c"
LEAF_DAG_SHA256 = "9ca74cce6a55717342c031e86b6dc7b6129dcbf22a38aa82b21cc049d18f8422"
INPUT_PROVENANCE_SHA256 = "695ff2a5fd45fa3fba21d4ce25ab2f62bd168df187c8e931bc9b5803a9cd4aed"
EXPECTED_PREDECESSOR = {
    "source": {
        "path": (
            "src/sigma_theory_compiler/quartic_p10_arbitrary_background_leaf_derivative_gate.py"
        ),
        "file_sha256": "6bb9ab72854d74e21a120f41fc9299712d5d64d0c9c9aeb243d5c3948b689239",
    },
    "config": {
        "path": ("configs/backgrounds/quartic_p10_arbitrary_background_leaf_derivative_gate.json"),
        "file_sha256": "2a769692f4b263c6d05cefa5b486e1e401a358be996b159c40a5634025bdf196",
    },
    "test": {
        "path": "tests/test_quartic_p10_arbitrary_background_leaf_derivative_gate.py",
        "file_sha256": "e909f5d6e9638a5c7b11268d4555b153598399ed4da22dccadc93cd9495a023c",
    },
    "artifact": {
        "path": (
            "runs/physics-language/quartic-p10-arbitrary-background-leaf-derivative-gate/"
            "campaign.json"
        ),
        "file_sha256": "c74171c48d7fc4f80de8f0c51b2b2700a1ce33de8795c3a999cee7c957b35869",
        "content_sha256": "51f76fa7ebc81ab2f570bfe5ad920215420e005687d0c861b24ea6da766c37e0",
    },
}
EXPECTED_D1 = {
    "path": (
        "runs/physics-language/quartic-full-source-jacobian-arithmetic-campaign/campaign.json"
    ),
    "file_sha256": "e893ebcaef464b958516279c557382fb76ecdb0fd542b3e3fed6a347076fcdae",
    "content_sha256": "1707b7258fd434f68b06c7af6bc447b4136624b9916992df8b412e048ab6538a",
}
EXPECTED_CONTRACT = {
    "candidate_count": 12,
    "registered_leaf_derivative_roots": 7920,
    "unique_P10_directions_per_candidate": 5,
    "ordered_P10_D2_records": 84,
    "unique_replay_roots": 60,
}
EXPECTED_POLICIES = {
    "derivative_replay": "closed_exact_forward_mode_over_bound_D1_DAG",
    "Pother_roots": "blocked_until_leaf_derivatives_registered",
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
EXPECTED_D1_OPERATIONS = [
    "exact_constant",
    "exact_component_input",
    "exact_add",
    "exact_negate",
    "exact_multiply",
    "exact_divide",
]


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
        raise ValueError("P10 inverse-product replay path escapes project root")
    return path


def _validate_config(value: Mapping[str, Any]) -> None:
    if value != {
        "schema_version": CONFIG_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "output_path": OUTPUT_PATH,
        "predecessor": EXPECTED_PREDECESSOR,
        "direct_D1_artifact": EXPECTED_D1,
        "derivative_contract": EXPECTED_CONTRACT,
        "policies": EXPECTED_POLICIES,
        "seals": EXPECTED_SEALS,
    }:
        raise ValueError("P10 inverse-product replay config boundary changed")


def _load_bound(root: Path, binding: Mapping[str, Any]) -> dict[str, Any]:
    path = _inside(root, str(binding["path"]))
    if _file_sha(path) != binding["file_sha256"]:
        raise ValueError("P10 inverse-product replay file binding changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("content_sha256") != binding["content_sha256"] or value.get(
        "content_sha256"
    ) != _content_sha(value):
        raise ValueError("P10 inverse-product replay content binding changed")
    return value


def _load_inputs(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    for label in ("source", "config", "test"):
        binding = EXPECTED_PREDECESSOR[label]
        if _file_sha(_inside(root, binding["path"])) != binding["file_sha256"]:
            raise ValueError("P10 inverse-product replay predecessor binding changed")
    predecessor = _load_bound(root, EXPECTED_PREDECESSOR["artifact"])
    d1 = _load_bound(root, EXPECTED_D1)
    if (
        predecessor.get("decision")
        != "pass_7920_P10_arbitrary_background_leaf_derivative_roots_D2_propagation_blocked"
        or predecessor.get("gate_counts", {}).get(
            "registered_arbitrary_background_leaf_derivative_roots"
        )
        != 7920
        or predecessor.get("gate_counts", {}).get("P10_ordered_D2_roots_registered") != 0
        or predecessor.get("leaf_derivative_arithmetic_DAG", {}).get("content_sha256")
        != LEAF_DAG_SHA256
        or len(predecessor.get("candidate_manifests", [])) != 12
        or d1.get("status")
        != "pass_all_12_full_11x153_entrywise_arithmetic_mixed_tensors_fail_closed"
    ):
        raise ValueError("P10 inverse-product replay input boundary changed")
    return predecessor, d1


def _load_D1(
    d1: Mapping[str, Any],
) -> tuple[list[Mapping[str, Any]], dict[tuple[int, str], Mapping[str, Any]]]:
    packet = d1.get("common_principal_arithmetic_packet")
    manifest = d1.get("common_full_entry_manifest")
    if not isinstance(packet, Mapping) or not isinstance(manifest, Mapping):
        raise TypeError("P10 inverse-product replay D1 packet missing")
    dag = packet.get("arithmetic_dag")
    nodes = dag.get("nodes") if isinstance(dag, Mapping) else None
    if (
        not isinstance(dag, Mapping)
        or dag.get("content_sha256") != D1_DAG_SHA256
        or dag.get("node_count") != 30658
        or dag.get("allowed_operations") != EXPECTED_D1_OPERATIONS
        or not isinstance(nodes, list)
        or len(nodes) != 30658
        or packet.get("inverse_evidence", {}).get("division_assumption")
        != "c11=(-1)^11 det(A) is nonzero"
    ):
        raise ValueError("P10 inverse-product replay D1 DAG changed")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or len(entries) != 1683:
        raise ValueError("P10 inverse-product replay D1 manifest changed")
    entry_map = {(int(row["source_row"]), str(row["coordinate_atom"])): row for row in entries}
    if len(entry_map) != 1683:
        raise ValueError("P10 inverse-product replay D1 manifest keys changed")
    return nodes, entry_map


def _children(node: Mapping[str, Any]) -> list[int]:
    op = node.get("op")
    if op in {"exact_constant", "exact_component_input"}:
        return []
    if op == "exact_add":
        children = node.get("arguments")
    elif op == "exact_negate":
        children = [node.get("argument")]
    elif op == "exact_multiply":
        children = [node.get("left"), node.get("right")]
    elif op == "exact_divide":
        children = [node.get("numerator"), node.get("denominator")]
    else:
        raise ValueError("P10 inverse-product replay unknown D1 operator")
    if not isinstance(children, list) or any(type(index) is not int for index in children):
        raise TypeError("P10 inverse-product replay D1 child schema changed")
    return children


@cache
def _closure(root_index: int, node_identity: int) -> tuple[int, ...]:
    nodes = _NODE_TABLES[node_identity]
    if not 0 <= root_index < len(nodes):
        raise ValueError("P10 inverse-product replay root outside D1 DAG")
    seen: set[int] = set()
    stack = [root_index]
    while stack:
        index = stack.pop()
        if index in seen:
            continue
        if not 0 <= index < len(nodes):
            raise ValueError("P10 inverse-product replay child outside D1 DAG")
        seen.add(index)
        stack.extend(_children(nodes[index]))
    return tuple(sorted(seen))


_NODE_TABLES: dict[int, list[Mapping[str, Any]]] = {}
_EXPECTED_CACHE: dict[tuple[str, ...], dict[str, Any]] = {}


class _MerkleReplay:
    """Hash-cons an exact forward derivative without serializing a 400k-node DAG."""

    def __init__(self, *, candidate_id: str, coordinate_atom: str) -> None:
        self.candidate_id = candidate_id
        self.coordinate_atom = coordinate_atom
        self.nodes: dict[str, dict[str, Any]] = {}
        self.zero = self.make({"op": "exact_constant", "value": "0"})

    def make(self, descriptor: dict[str, Any]) -> str:
        digest = _sha(descriptor)
        previous = self.nodes.setdefault(digest, descriptor)
        if previous != descriptor:
            raise AssertionError("P10 inverse-product replay SHA collision")
        return digest

    def primal(self, index: int) -> str:
        return self.make(
            {
                "op": "bound_D1_primal_node_reference",
                "D1_arithmetic_dag_sha256": D1_DAG_SHA256,
                "node_index": index,
            }
        )

    def leaf(self, label: str, root: int, value: str) -> str:
        if value == "0":
            return self.zero
        return self.make(
            {
                "op": "bound_leaf_derivative_root_reference",
                "candidate_id": self.candidate_id,
                "coordinate_atom": self.coordinate_atom,
                "component_input_label": label,
                "leaf_arithmetic_dag_sha256": LEAF_DAG_SHA256,
                "leaf_arithmetic_root": root,
                "exact_value": value,
            }
        )

    def add(self, arguments: list[str]) -> str:
        nonzero = [item for item in arguments if item != self.zero]
        if not nonzero:
            return self.zero
        if len(nonzero) == 1:
            return nonzero[0]
        return self.make({"op": "exact_add", "arguments": nonzero})

    def negate(self, argument: str) -> str:
        if argument == self.zero:
            return self.zero
        return self.make({"op": "exact_negate", "argument": argument})

    def multiply(self, left: str, right: str) -> str:
        if left == self.zero or right == self.zero:
            return self.zero
        return self.make({"op": "exact_multiply", "left": left, "right": right})

    def divide(self, numerator: str, denominator: str) -> str:
        if numerator == self.zero:
            return self.zero
        return self.make({"op": "exact_divide", "numerator": numerator, "denominator": denominator})


def _dense_leaf_roots(packet: Mapping[str, Any]) -> dict[str, tuple[int, str]]:
    if (
        packet.get("arithmetic_dag_sha256") != LEAF_DAG_SHA256
        or packet.get("A_derivative_shape") != [11, 11]
        or packet.get("source_chunk_column_shape") != [11]
        or packet.get("total_leaf_derivative_roots") != 132
        or packet.get("arbitrary_background_valid") is not True
    ):
        raise ValueError("P10 inverse-product replay leaf packet changed")
    zero = int(packet["zero_default_arithmetic_root"])
    values = {f"A[{row},{column}]": (zero, "0") for row in range(11) for column in range(11)}
    family = str(packet["source_chunk_family"])
    chunk = "C_" + family[1:]
    values.update({f"{chunk}[{row},10]": (zero, "0") for row in range(11)})
    for entry in packet["A_derivative_sparse_entries"]:
        label = f"A[{int(entry['row'])},{int(entry['column'])}]"
        values[label] = (int(entry["arithmetic_root"]), str(entry["value"]))
    for entry in packet["source_chunk_column_derivative_sparse_entries"]:
        label = f"{chunk}[{int(entry['row'])},10]"
        values[label] = (int(entry["arithmetic_root"]), str(entry["value"]))
    if len(values) != 132:
        raise ValueError("P10 inverse-product replay leaf inventory changed")
    return values


def _replay_packet(
    candidate_id: str,
    packet: Mapping[str, Any],
    nodes: list[Mapping[str, Any]],
    entry: Mapping[str, Any],
) -> dict[str, Any]:
    coordinate_atom = str(packet["coordinate_atom"])
    if (
        entry.get("coordinate_atom") != coordinate_atom
        or entry.get("source_row") != 10
        or entry.get("arithmetic_dag_sha256") != D1_DAG_SHA256
    ):
        raise ValueError("P10 inverse-product replay target entry changed")
    root_index = int(entry["arithmetic_root"])
    node_identity = id(nodes)
    _NODE_TABLES[node_identity] = nodes
    closure = _closure(root_index, node_identity)
    leaves = _dense_leaf_roots(packet)
    closure_labels = {
        str(nodes[index]["label"])
        for index in closure
        if nodes[index].get("op") == "exact_component_input"
    }
    if closure_labels != set(leaves):
        raise ValueError("P10 inverse-product replay closure/leaf binding mismatch")
    replay = _MerkleReplay(candidate_id=candidate_id, coordinate_atom=coordinate_atom)
    derivatives: dict[int, str] = {}
    trace = []
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
            left = int(node["left"])
            right = int(node["right"])
            derivative = replay.add(
                [
                    replay.multiply(derivatives[left], replay.primal(right)),
                    replay.multiply(replay.primal(left), derivatives[right]),
                ]
            )
        elif op == "exact_divide":
            numerator = int(node["numerator"])
            denominator = int(node["denominator"])
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
            raise ValueError("P10 inverse-product replay operator changed")
        derivatives[index] = derivative
        trace.append([index, derivative])
    root_sha = derivatives[root_index]
    if root_sha == replay.zero:
        raise ValueError("P10 inverse-product replay unexpectedly produced zero root")
    operation_counts = dict(sorted(Counter(row["op"] for row in replay.nodes.values()).items()))
    return {
        "coordinate_atom": coordinate_atom,
        "coordinate_column": int(packet["coordinate_column"]),
        "coordinate_ordinals": list(packet["coordinate_ordinals"]),
        "D1_arithmetic_root": root_index,
        "D1_arithmetic_dag_sha256": D1_DAG_SHA256,
        "leaf_derivative_arithmetic_dag_sha256": LEAF_DAG_SHA256,
        "bound_leaf_derivative_count": len(leaves),
        "nonzero_bound_leaf_derivative_count": sum(value != "0" for _, value in leaves.values()),
        "D1_dependency_closure_nodes": len(closure),
        "D2_merkle_replay_root_sha256": root_sha,
        "D2_merkle_replay_trace_sha256": _sha(trace),
        "D2_merkle_replay_node_count": len(replay.nodes),
        "D2_merkle_replay_operation_counts": operation_counts,
        "quotient_domain_assumption": "c11=(-1)^11 det(A) is nonzero",
        "packet_sha256": _sha(
            {
                "candidate_id": candidate_id,
                "coordinate_atom": coordinate_atom,
                "root": root_sha,
                "trace": _sha(trace),
            }
        ),
    }


def _candidate_manifests(
    predecessor: Mapping[str, Any],
    nodes: list[Mapping[str, Any]],
    entries: Mapping[tuple[int, str], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    manifests = []
    record_ids: set[str] = set()
    replay_roots: set[str] = set()
    for candidate in predecessor["candidate_manifests"]:
        candidate_id = str(candidate["candidate_id"])
        packets = []
        records = []
        for packet in candidate["direction_packets"]:
            atom = str(packet["coordinate_atom"])
            entry = entries.get((10, atom))
            if entry is None:
                raise ValueError("P10 inverse-product replay D1 target missing")
            replay_packet = _replay_packet(candidate_id, packet, nodes, entry)
            packets.append(replay_packet)
            replay_roots.add(replay_packet["D2_merkle_replay_root_sha256"])
            for ordinal in replay_packet["coordinate_ordinals"]:
                identity = {
                    "candidate_id": candidate_id,
                    "coordinate_ordinal": ordinal,
                    "coordinate_atom": atom,
                    "coordinate_column": replay_packet["coordinate_column"],
                    "D1_arithmetic_root": replay_packet["D1_arithmetic_root"],
                    "D2_merkle_replay_root_sha256": replay_packet["D2_merkle_replay_root_sha256"],
                }
                record_id = _sha(identity)
                if record_id in record_ids:
                    raise AssertionError("P10 inverse-product replay record collision")
                record_ids.add(record_id)
                records.append(
                    {
                        **identity,
                        "ordered_D2_record_id": record_id,
                        "root_status": "sealed_exact_arbitrary_background_merkle_replay",
                        "candidate_rejection_authorized": False,
                    }
                )
        if len(packets) != 5 or len(records) != 7:
            raise ValueError("P10 inverse-product replay candidate census changed")
        manifests.append(
            {
                "candidate_id": candidate_id,
                "replay_packets": packets,
                "ordered_P10_D2_records": records,
                "unique_P10_replay_roots": 5,
                "sealed_ordered_P10_D2_roots": 7,
                "blocked_ordered_Pother_D2_roots": 15,
                "candidate_manifest_sha256": _sha([row["ordered_D2_record_id"] for row in records]),
                "candidate_decision": "pass_7_P10_D2_roots_Pother_blocked",
                "candidate_rejection_authorized": False,
                "first_blocker": FIRST_BLOCKER,
            }
        )
    if len(record_ids) != 84 or len(replay_roots) != 60:
        raise AssertionError("P10 inverse-product replay global root census changed")
    return manifests


def _expected_body(
    root: Path,
    config_path: Path,
    predecessor: Mapping[str, Any],
    d1: Mapping[str, Any],
) -> dict[str, Any]:
    cache_key = (
        str(root),
        _file_sha(_inside(root, SOURCE_PATH)),
        _file_sha(config_path),
        _file_sha(_inside(root, TEST_PATH)),
        str(predecessor["content_sha256"]),
        str(d1["content_sha256"]),
    )
    cached = _EXPECTED_CACHE.get(cache_key)
    if cached is not None:
        return _copy(cached)
    nodes, entries = _load_D1(d1)
    manifests = _candidate_manifests(predecessor, nodes, entries)
    all_packets = [packet for row in manifests for packet in row["replay_packets"]]
    body = {
        "schema_version": RESULT_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "decision": "pass_all_84_P10_ordered_D2_roots_exactly_replayed_Pother_blocked",
        "decision_counts": {"pass": 12, "blocked": 0, "reject": 0},
        "downstream_admission_counts": {"pass": 0, "blocked": 12, "reject": 0},
        "first_blocker": FIRST_BLOCKER,
        "replay_theorem": {
            "name": "closed_forward_derivative_replay_of_bound_inverse_product_D1_DAG",
            "premises": (
                "The directly bound D1 DAG uses only exact constants, component inputs, sums, "
                "negations, products, and quotients on the declared det(A) nonzero domain. For "
                "each P10 direction, its complete 132-leaf closure is bound to registered exact "
                "arbitrary-background leaf derivative roots."
            ),
            "exact_result": (
                "Forward differentiation with the sum, negation, product, and quotient rules "
                "replays five candidate-bound D2 roots per candidate. Repeated coordinate "
                "ordinals expand these to seven ordered P10 records per candidate: all 84 P10 "
                "records are sealed by exact Merkle replay roots."
            ),
            "representation": (
                "Each replay root is the canonical Merkle root of the exact differentiated "
                "arithmetic DAG. The validator reconstructs every descriptor and the full "
                "per-primal-node trace from the bound live DAG and leaf roots; the large derived "
                "node list is not duplicated in the artifact."
            ),
            "boundary": (
                "This is only the P10 arbitrary-background subset. No Pother leaf derivative or "
                "remaining 180 ordered root is registered, so complete D2F and downstream "
                "analytic claims remain blocked."
            ),
        },
        "replay_contract": {
            "schema_version": "sigma-exact-forward-derivative-merkle-replay-1.0",
            "bound_D1_arithmetic_dag_sha256": D1_DAG_SHA256,
            "bound_leaf_derivative_arithmetic_dag_sha256": LEAF_DAG_SHA256,
            "closed_derivative_rules": [
                "D(constant)=0",
                "D(input)=bound_leaf_root",
                "D(add)=add(D(arguments))",
                "D(negate)=negate(D(argument))",
                "D(multiply)=add(multiply(D(left),right),multiply(left,D(right)))",
                "D(divide)=divide(add(multiply(D(numerator),denominator),negate(multiply(numerator,D(denominator)))),multiply(denominator,denominator))",
            ],
            "exact_zero_simplifications": [
                "additive_zero_elision",
                "zero_product_elision",
                "zero_numerator_division_elision",
            ],
            "quotient_domain_assumption": "c11=(-1)^11 det(A) is nonzero",
            "full_trace_recomputed_during_validation": True,
        },
        "candidate_manifests": manifests,
        "manifest_sha256": _sha([row["candidate_manifest_sha256"] for row in manifests]),
        "gate_counts": {
            "selected_candidates": 12,
            "registered_P10_leaf_derivative_roots_consumed": 7920,
            "unique_P10_replay_roots": 60,
            "sealed_P10_ordered_D2_roots": 84,
            "blocked_P10_ordered_D2_roots": 0,
            "D1_dependency_nodes_replayed_across_packets": sum(
                row["D1_dependency_closure_nodes"] for row in all_packets
            ),
            "derived_merkle_nodes_across_packets": sum(
                row["D2_merkle_replay_node_count"] for row in all_packets
            ),
            "Pother_leaf_derivative_roots_registered": 0,
            "Pother_ordered_D2_roots_registered": 0,
            "Pother_ordered_D2_roots_blocked": 180,
            "all_target_ordered_D2_roots_registered": 84,
            "all_target_ordered_D2_roots_blocked": 180,
            "complete_ordered_D2F_tensors_registered": 0,
            "global_H7_closures": 0,
            "nonlinear_PDE_closures": 0,
            "lifespans_proved": 0,
        },
        "claim_seals": {
            "all_84_P10_ordered_D2_roots_exactly_replayed": True,
            "Pother_leaf_derivative_roots_registered": False,
            "Pother_ordered_D2_roots_registered": False,
            "physical_no_go_proved": False,
            "complete_ordered_D2F_tensor_registered": False,
            "global_H7_energy_closed": False,
            "nonlinear_PDE_closed": False,
            "nonlinear_lifespan_proved": False,
            "candidate_theory_rejected": False,
            "observational_claim_made": False,
        },
        "exact_controls": {
            "default_unregistered_leaf_derivative_to_zero": {"rejected": True},
            "accept_unknown_D1_operator": {"rejected": True},
            "drop_quotient_denominator_derivative": {"rejected": True},
            "promote_P10_subset_to_complete_D2F": {"rejected": True},
            "infer_physical_no_go_from_Pother_blocker": {"rejected": True},
            "reject_candidate_from_Pother_blocker": {"rejected": True},
        },
        "data_seals": dict(EXPECTED_SEALS),
        "source_bindings": {
            "source": {"path": SOURCE_PATH, "file_sha256": _file_sha(_inside(root, SOURCE_PATH))},
            "config": {"path": CONFIG_PATH, "file_sha256": _file_sha(config_path)},
            "test": {"path": TEST_PATH, "file_sha256": _file_sha(_inside(root, TEST_PATH))},
            "predecessor": _copy(EXPECTED_PREDECESSOR),
            "direct_D1_artifact": _copy(EXPECTED_D1),
        },
        "scope": (
            "exact candidate-bound arbitrary-background forward derivative replay for the 84 "
            "ordered P10 records only; no Pother root, complete D2F, physical no-go, H7, PDE, "
            "lifespan, candidate rejection, or observational claim"
        ),
    }
    _EXPECTED_CACHE[cache_key] = body
    return _copy(body)


def _validate_bindings(value: Mapping[str, Any], root: Path) -> None:
    bindings = value.get("source_bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != {
        "source",
        "config",
        "test",
        "predecessor",
        "direct_D1_artifact",
    }:
        raise ValueError("P10 inverse-product replay binding keys changed")
    for label, relative in {
        "source": SOURCE_PATH,
        "config": CONFIG_PATH,
        "test": TEST_PATH,
    }.items():
        if bindings[label] != {
            "path": relative,
            "file_sha256": _file_sha(_inside(root, relative)),
        }:
            raise ValueError("P10 inverse-product replay local binding changed")
    if bindings["predecessor"] != EXPECTED_PREDECESSOR:
        raise ValueError("P10 inverse-product replay predecessor binding changed")
    if bindings["direct_D1_artifact"] != EXPECTED_D1:
        raise ValueError("P10 inverse-product replay D1 binding changed")


def _validate_result(value: Mapping[str, Any], *, root: Path | None = None) -> None:
    validation_root = (root or Path(__file__).resolve().parents[2]).resolve()
    if value.get("content_sha256") != _content_sha(value):
        raise ValueError("P10 inverse-product replay content hash changed")
    _validate_bindings(value, validation_root)
    config_path = _inside(validation_root, CONFIG_PATH)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    predecessor, d1 = _load_inputs(validation_root)
    expected = _expected_body(validation_root, config_path, predecessor, d1)
    if {key: item for key, item in value.items() if key != "content_sha256"} != expected:
        raise ValueError("P10 inverse-product replay result boundary changed")


def build_gate(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    root = config_path.parents[2]
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    predecessor, d1 = _load_inputs(root)
    body = _expected_body(root, config_path, predecessor, d1)
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
