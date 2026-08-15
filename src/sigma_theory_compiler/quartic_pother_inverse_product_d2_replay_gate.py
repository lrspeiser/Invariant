"""Audit exact leaf-label alignment before replaying the Pother D1 arithmetic roots."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .quartic_p10_inverse_product_d2_replay_gate import (
    _children,
    _load_D1,
)
from .quartic_p10_inverse_product_d2_replay_gate import (
    _MerkleReplay as _P10MerkleReplay,
)
from .quartic_pother_arbitrary_background_leaf_derivative_gate import (
    _validate_result as _validate_pother_leaf,
)

CONFIG_SCHEMA = "sigma-quartic-pother-inverse-product-d2-replay-config-1.0"
RESULT_SCHEMA = "sigma-quartic-pother-inverse-product-d2-replay-gate-1.0"
CAMPAIGN_ID = "quartic-pother-inverse-product-d2-replay-001"
CONFIG_PATH = "configs/backgrounds/quartic_pother_inverse_product_d2_replay_gate.json"
SOURCE_PATH = "src/sigma_theory_compiler/quartic_pother_inverse_product_d2_replay_gate.py"
TEST_PATH = "tests/test_quartic_pother_inverse_product_d2_replay_gate.py"
OUTPUT_PATH = "runs/physics-language/quartic-pother-inverse-product-d2-replay-gate/campaign.json"
FIRST_BLOCKER = (
    "extend_beyond_the_22_bounded_row10_targets_to_the_remaining_257477_ordered_"
    "D2F_entries_per_candidate_and_reconcile_the_full_high_atom_identity"
)
LEAF_DAG_SHA256 = "b91251c5cc660ec3444cd96750925eed814dba25d739b95cb618f97928ec1f5a"
EXPECTED_PREDECESSOR = {
    "source": {
        "path": (
            "src/sigma_theory_compiler/quartic_pother_arbitrary_background_leaf_derivative_gate.py"
        ),
        "file_sha256": "7e34d374df437075f040b40f7435cb0ab07a4ca60ccf696a1730f6153761f2a0",
    },
    "config": {
        "path": (
            "configs/backgrounds/quartic_pother_arbitrary_background_leaf_derivative_gate.json"
        ),
        "file_sha256": "d11e40b85836c2d858d03e242f3df50d8751e965b81f6b0d5a44399006c48af2",
    },
    "test": {
        "path": "tests/test_quartic_pother_arbitrary_background_leaf_derivative_gate.py",
        "file_sha256": "d51fa61d4bcd12defa6f5f84fb308890b378bb2e01ff6866aea2c05642e3df69",
    },
    "artifact": {
        "path": (
            "runs/physics-language/quartic-pother-arbitrary-background-leaf-derivative-"
            "gate/campaign.json"
        ),
        "file_sha256": "c687e15839628dcca3480740ea8ee568576461c200ce8937ab5499a71e9e49c2",
        "content_sha256": "b2f4eacd73026bc92a057be0ad5340d487ef0ff3b18353e3412d7aea5475b670",
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
    "candidate_direction_packets": 180,
    "presented_leaf_derivative_roots": 23760,
    "ordered_Pother_D2_records_targeted": 180,
}
EXPECTED_POLICIES = {
    "derivative_replay": "require_exact_reachable_leaf_label_alignment_then_closed_forward_replay",
    "mismatched_leaf_label": "block",
    "full_D2F": "fail_closed_outside_22_bounded_targets",
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
_EXPECTED_CACHE: dict[tuple[str, ...], dict[str, Any]] = {}


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
        raise ValueError("Pother inverse-product replay path escapes project root")
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
        raise ValueError("Pother inverse-product replay config boundary changed")


def _load_bound(root: Path, binding: Mapping[str, Any]) -> dict[str, Any]:
    path = _inside(root, str(binding["path"]))
    if _file_sha(path) != binding["file_sha256"]:
        raise ValueError("Pother inverse-product replay file binding changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("content_sha256") != binding["content_sha256"] or value.get(
        "content_sha256"
    ) != _content_sha(value):
        raise ValueError("Pother inverse-product replay content binding changed")
    return value


def _load_inputs(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    for label in ("source", "config", "test"):
        binding = EXPECTED_PREDECESSOR[label]
        if _file_sha(_inside(root, binding["path"])) != binding["file_sha256"]:
            raise ValueError("Pother inverse-product replay predecessor binding changed")
    predecessor = _load_bound(root, EXPECTED_PREDECESSOR["artifact"])
    _validate_pother_leaf(predecessor, root=root)
    d1 = _load_bound(root, EXPECTED_D1)
    if (
        predecessor.get("decision")
        != "pass_23760_Pother_leaf_roots_all_180_D2_records_replay_ready"
        or predecessor.get("gate_counts", {}).get("registered_Pother_leaf_derivative_roots")
        != 23760
        or predecessor.get("gate_counts", {}).get("Pother_ordered_D2_roots_replay_ready") != 180
        or predecessor.get("leaf_derivative_arithmetic_DAG", {}).get("content_sha256")
        != LEAF_DAG_SHA256
        or len(predecessor.get("candidate_manifests", [])) != 12
        or d1.get("status")
        != "pass_all_12_full_11x153_entrywise_arithmetic_mixed_tensors_fail_closed"
    ):
        raise ValueError("Pother inverse-product replay input boundary changed")
    return predecessor, d1


def _closure(root_index: int, nodes: list[Mapping[str, Any]]) -> tuple[int, ...]:
    if not 0 <= root_index < len(nodes):
        raise ValueError("Pother inverse-product replay root outside D1 DAG")
    seen: set[int] = set()
    stack = [root_index]
    while stack:
        index = stack.pop()
        if index in seen:
            continue
        if not 0 <= index < len(nodes):
            raise ValueError("Pother inverse-product replay child outside D1 DAG")
        seen.add(index)
        stack.extend(_children(nodes[index]))
    return tuple(sorted(seen))


def _bound_labels(packet: Mapping[str, Any]) -> set[str]:
    if (
        packet.get("arithmetic_dag_sha256") != LEAF_DAG_SHA256
        or packet.get("A_derivative_shape") != [11, 11]
        or packet.get("source_chunk_column_shape") != [11]
        or packet.get("total_leaf_derivative_roots") != 132
    ):
        raise ValueError("Pother inverse-product replay leaf packet changed")
    family = str(packet["source_chunk_family"])
    column = int(packet["source_chunk_input_column"])
    chunk = f"B_{family[2:]}" if family.startswith("s0") else "C_" + family[1:]
    return {
        *(f"A[{row},{col}]" for row in range(11) for col in range(11)),
        *(f"{chunk}[{row},{column}]" for row in range(11)),
    }


class _MerkleReplay(_P10MerkleReplay):
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
                "exact_expression": value,
            }
        )


def _dense_leaf_roots(packet: Mapping[str, Any]) -> dict[str, tuple[int, str]]:
    values = {
        label: (int(packet["zero_default_arithmetic_root"]), "0") for label in _bound_labels(packet)
    }
    for entry in packet["A_derivative_sparse_entries"]:
        values[f"A[{entry['row']},{entry['column']}]"] = (
            int(entry["arithmetic_root"]),
            str(entry["value"]),
        )
    family = str(packet["source_chunk_family"])
    column = int(packet["source_chunk_input_column"])
    chunk = f"B_{family[2:]}" if family.startswith("s0") else "C_" + family[1:]
    for entry in packet["source_chunk_column_derivative_sparse_entries"]:
        values[f"{chunk}[{entry['row']},{column}]"] = (
            int(entry["arithmetic_root"]),
            str(entry["value"]),
        )
    return values


def _replay_packet(
    candidate_id: str,
    packet: Mapping[str, Any],
    nodes: list[Mapping[str, Any]],
    entry: Mapping[str, Any],
) -> dict[str, Any]:
    atom = str(packet["coordinate_atom"])
    root_index = int(entry["arithmetic_root"])
    closure = _closure(root_index, nodes)
    leaves = _dense_leaf_roots(packet)
    reachable = {
        str(nodes[index]["label"])
        for index in closure
        if nodes[index].get("op") == "exact_component_input"
    }
    if reachable != set(leaves):
        raise ValueError("Pother inverse-product replay closure/leaf binding mismatch")
    replay = _MerkleReplay(candidate_id=candidate_id, coordinate_atom=atom)
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
            raise ValueError("Pother inverse-product replay operator changed")
        derivatives[index] = derivative
        trace.append([index, derivative])
    root_sha = derivatives[root_index]
    return {
        "coordinate_atom": atom,
        "coordinate_column": packet["coordinate_column"],
        "coordinate_ordinal": packet["coordinate_ordinals"][0],
        "D1_arithmetic_root": root_index,
        "D1_dependency_closure_nodes": len(closure),
        "D2_merkle_replay_root_sha256": root_sha,
        "D2_merkle_replay_trace_sha256": _sha(trace),
        "D2_merkle_replay_node_count": len(replay.nodes),
        "D2_root_exactly_zero": root_sha == replay.zero,
        "nonzero_bound_leaf_derivative_count": sum(value != "0" for _, value in leaves.values()),
        "D1_quotient_domain_assumption": "c11=(-1)^11 det(A) is nonzero",
        "leaf_background_domain_assumption": "g_is_nonsingular_and_gu_is_its_exact_inverse",
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
        packets, records = [], []
        for packet in candidate["direction_packets"]:
            atom = str(packet["coordinate_atom"])
            entry = entries.get((10, atom))
            if entry is None:
                raise ValueError("Pother inverse-product replay D1 target missing")
            replay = _replay_packet(candidate_id, packet, nodes, entry)
            packets.append(replay)
            replay_roots.add(replay["D2_merkle_replay_root_sha256"])
            identity = {
                "candidate_id": candidate_id,
                "coordinate_atom": atom,
                "coordinate_ordinal": replay["coordinate_ordinal"],
                "D2_merkle_replay_root_sha256": replay["D2_merkle_replay_root_sha256"],
            }
            record_id = _sha(identity)
            if record_id in record_ids:
                raise AssertionError("Pother inverse-product replay record collision")
            record_ids.add(record_id)
            records.append(
                {
                    **identity,
                    "ordered_D2_record_id": record_id,
                    "D2_root_exactly_zero": replay["D2_root_exactly_zero"],
                    "root_status": "sealed_exact_registered_symbolic_background_merkle_replay",
                }
            )
        if len(packets) != 15 or sum(row["D2_root_exactly_zero"] for row in packets) != 2:
            raise ValueError("Pother inverse-product replay candidate census changed")
        manifests.append(
            {
                "candidate_id": candidate_id,
                "replay_packets": packets,
                "ordered_Pother_D2_records": records,
                "Pother_ordered_D2_roots_registered": 15,
                "nonzero_Pother_ordered_D2_roots": 13,
                "zero_Pother_ordered_D2_roots": 2,
                "candidate_manifest_sha256": _sha([row["ordered_D2_record_id"] for row in records]),
                "candidate_decision": "pass_all_15_Pother_D2_roots_bounded_target_complete",
                "candidate_rejection_authorized": False,
                "first_blocker": FIRST_BLOCKER,
            }
        )
    if len(record_ids) != 180 or len(replay_roots) != 157:
        raise AssertionError("Pother inverse-product replay global census changed")
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
    body = {
        "schema_version": RESULT_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "decision": "pass_all_180_Pother_roots_all_264_bounded_targets_sealed",
        "decision_counts": {"pass": 12, "blocked": 0, "reject": 0},
        "downstream_admission_counts": {"pass": 0, "blocked": 12, "reject": 0},
        "first_blocker": FIRST_BLOCKER,
        "replay_theorem": {
            "name": "typed_B_i_C_ij_leaf_alignment_and_closed_forward_D1_replay",
            "exact_result": (
                "Every live Pother root closure aligns exactly with its 132 bound leaf labels: "
                "121 A entries plus 11 typed B_i or C_ij column entries. Closed exact forward "
                "differentiation seals all 180 Pother roots; 156 are nonzero and 24 are the "
                "canonical zero root. Together with P10, all 264 bounded targets are sealed."
            ),
            "boundary": (
                "This completes only 22 bounded row-10 targets per candidate, not the complete "
                "ordered 11x153x153 D2F tensor or full high-atom identity."
            ),
        },
        "candidate_manifests": manifests,
        "manifest_sha256": _sha([row["candidate_manifest_sha256"] for row in manifests]),
        "gate_counts": {
            "selected_candidates": 12,
            "presented_Pother_leaf_derivative_roots": 23760,
            "Pother_direction_packets_replayed": 180,
            "Pother_ordered_D2_roots_replayed": 180,
            "Pother_ordered_D2_roots_registered": 180,
            "nonzero_Pother_ordered_D2_roots": 156,
            "zero_Pother_ordered_D2_roots": 24,
            "P10_ordered_D2_roots_previously_sealed": 84,
            "all_bounded_target_ordered_D2_roots_registered": 264,
            "all_bounded_target_ordered_D2_roots_blocked": 0,
            "complete_ordered_D2F_tensors_registered": 0,
            "global_H7_closures": 0,
            "nonlinear_PDE_closures": 0,
            "lifespans_proved": 0,
        },
        "claim_seals": {
            "exact_leaf_label_alignment_proved": True,
            "Pother_ordered_D2_roots_registered": True,
            "all_264_bounded_target_ordered_D2_roots_registered": True,
            "complete_ordered_D2F_tensor_registered": False,
            "full_high_atom_identity_closed": False,
            "physical_no_go_proved": False,
            "global_H7_energy_closed": False,
            "nonlinear_PDE_closed": False,
            "nonlinear_lifespan_proved": False,
            "candidate_theory_rejected": False,
            "observational_claim_made": False,
        },
        "exact_controls": {
            "identify_B_i_and_C_ij_leaf_names": {"rejected": True},
            "default_unbound_leaf_root_to_zero": {"rejected": True},
            "replay_with_incomplete_leaf_closure": {"rejected": True},
            "promote_bounded_targets_to_complete_D2F": {"rejected": True},
            "infer_physical_no_go_from_full_tensor_gap": {"rejected": True},
            "reject_candidate_from_full_tensor_gap": {"rejected": True},
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
            "exact candidate-bound replay for all 180 Pother roots and completion of the 264 "
            "bounded targets; no complete D2F, full high-atom "
            "identity, physical no-go, H7, PDE, lifespan, candidate rejection, or observation"
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
        raise ValueError("Pother inverse-product replay binding keys changed")
    for label, relative in {
        "source": SOURCE_PATH,
        "config": CONFIG_PATH,
        "test": TEST_PATH,
    }.items():
        if bindings[label] != {
            "path": relative,
            "file_sha256": _file_sha(_inside(root, relative)),
        }:
            raise ValueError("Pother inverse-product replay local binding changed")
    if bindings["predecessor"] != EXPECTED_PREDECESSOR:
        raise ValueError("Pother inverse-product replay predecessor binding changed")
    if bindings["direct_D1_artifact"] != EXPECTED_D1:
        raise ValueError("Pother inverse-product replay D1 binding changed")


def _validate_result(value: Mapping[str, Any], *, root: Path | None = None) -> None:
    validation_root = (root or Path(__file__).resolve().parents[2]).resolve()
    if value.get("content_sha256") != _content_sha(value):
        raise ValueError("Pother inverse-product replay content hash changed")
    _validate_bindings(value, validation_root)
    config_path = _inside(validation_root, CONFIG_PATH)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    predecessor, d1 = _load_inputs(validation_root)
    expected = _expected_body(validation_root, config_path, predecessor, d1)
    if {key: item for key, item in value.items() if key != "content_sha256"} != expected:
        raise ValueError("Pother inverse-product replay result boundary changed")


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
