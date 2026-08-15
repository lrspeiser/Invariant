"""Partition full D2F and seal the maximal registered-direction source-row extension."""

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
    _dense_leaf_roots as _p10_dense_leaf_roots,
)
from .quartic_pother_inverse_product_d2_replay_gate import (
    _dense_leaf_roots,
    _MerkleReplay,
)
from .quartic_pother_inverse_product_d2_replay_gate import (
    _validate_result as _validate_predecessor,
)

CONFIG_PATH = "configs/backgrounds/quartic_full_d2f_typed_partition_row_extension_gate.json"
SOURCE_PATH = "src/sigma_theory_compiler/quartic_full_d2f_typed_partition_row_extension_gate.py"
TEST_PATH = "tests/test_quartic_full_d2f_typed_partition_row_extension_gate.py"
OUTPUT_PATH = (
    "runs/physics-language/quartic-full-d2f-typed-partition-row-extension-gate/campaign.json"
)
SCHEMA = "sigma-quartic-full-d2f-typed-partition-row-extension-gate-1.0"
CAMPAIGN = "quartic-full-d2f-typed-partition-row-extension-001"
FIRST_BLOCKER = "register_cross_direction_leaf_derivatives_for_the_5082_registered_direction_off_diagonal_entries_per_candidate"
PRE = {
    "source": {
        "path": "src/sigma_theory_compiler/quartic_pother_inverse_product_d2_replay_gate.py",
        "file_sha256": "d92c8ad16c677ce42a47ad2cf8fed09eb04b073f7667e7c3dc59fde208c50144",
    },
    "config": {
        "path": "configs/backgrounds/quartic_pother_inverse_product_d2_replay_gate.json",
        "file_sha256": "4cbc0c3806360b8dab384d6e9f94f9fd3ea0e15315b79199b0c93412b4a18718",
    },
    "test": {
        "path": "tests/test_quartic_pother_inverse_product_d2_replay_gate.py",
        "file_sha256": "d3300eec715b6ad1265c66d7b0aa85dd2185aa2f83494f7f983f3e86db7a54e3",
    },
    "artifact": {
        "path": "runs/physics-language/quartic-pother-inverse-product-d2-replay-gate/campaign.json",
        "file_sha256": "f29f10dacec37a235c3a8de5755876ab2a05b9bcb21f4e90a1b5d27f46fba6b8",
        "content_sha256": "8190747a51531a6debdbca68a63e9ebfc932ba2e5ebe6770d2a1d51f1514f472",
    },
}
_REPLAY_CACHE: dict[tuple[int, str], str] = {}


class _ContentReplay(_MerkleReplay):
    def __init__(self, *, coordinate_atom: str, leaf_dag: str) -> None:
        super().__init__(candidate_id="content_bound", coordinate_atom=coordinate_atom)
        self.leaf_dag = leaf_dag

    def leaf(self, label: str, root: int, value: str) -> str:
        if value == "0":
            return self.zero
        return self.make(
            {
                "op": "bound_leaf_derivative_root_reference",
                "coordinate_atom": self.coordinate_atom,
                "component_input_label": label,
                "leaf_arithmetic_dag_sha256": self.leaf_dag,
                "leaf_arithmetic_root": root,
                "exact_expression": value,
            }
        )


def _canon(x: Any) -> bytes:
    return json.dumps(x, sort_keys=True, separators=(",", ":")).encode()


def _sha(x: Any) -> str:
    return hashlib.sha256(_canon(x)).hexdigest()


def _fsha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _csha(x: Mapping[str, Any]) -> str:
    return _sha({k: v for k, v in x.items() if k != "content_sha256"})


def _inside(r: Path, p: str) -> Path:
    q = (r / p).resolve()
    if q != r and r not in q.parents:
        raise ValueError("row extension path escapes root")
    return q


def _bound(r: Path, b: Mapping[str, Any]) -> dict[str, Any]:
    p = _inside(r, b["path"])
    if _fsha(p) != b["file_sha256"]:
        raise ValueError("row extension binding changed")
    x = json.loads(p.read_text())
    if x.get("content_sha256") != b["content_sha256"] or x["content_sha256"] != _csha(x):
        raise ValueError("row extension content changed")
    return x


def _closure(root: int, nodes: list[Mapping[str, Any]]) -> list[int]:
    seen = set()
    stack = [root]
    while stack:
        i = stack.pop()
        if i in seen:
            continue
        seen.add(i)
        stack.extend(_children(nodes[i]))
    return sorted(seen)


def _replay(cid: str, packet: Mapping[str, Any], root: int, nodes: list[Mapping[str, Any]]) -> str:
    del cid
    cache_key = (root, str(packet["dense_root_manifest_sha256"]))
    if cache_key in _REPLAY_CACHE:
        return _REPLAY_CACHE[cache_key]
    closure = _closure(root, nodes)
    leaves = (
        _dense_leaf_roots(packet)
        if packet.get("registered_symbolic_background_scope") is True
        else _p10_dense_leaf_roots(packet)
    )
    labels = {nodes[i]["label"] for i in closure if nodes[i]["op"] == "exact_component_input"}
    if labels != set(leaves):
        raise ValueError("row extension leaf closure mismatch")
    r = _ContentReplay(
        coordinate_atom=str(packet["coordinate_atom"]),
        leaf_dag=str(packet["arithmetic_dag_sha256"]),
    )
    d = {}
    for i in closure:
        n = nodes[i]
        op = n["op"]
        if op == "exact_constant":
            v = r.zero
        elif op == "exact_component_input":
            a, b = leaves[n["label"]]
            v = r.leaf(n["label"], a, b)
        elif op == "exact_add":
            v = r.add([d[j] for j in n["arguments"]])
        elif op == "exact_negate":
            v = r.negate(d[n["argument"]])
        elif op == "exact_multiply":
            v = r.add(
                [
                    r.multiply(d[n["left"]], r.primal(n["right"])),
                    r.multiply(r.primal(n["left"]), d[n["right"]]),
                ]
            )
        elif op == "exact_divide":
            a, b = n["numerator"], n["denominator"]
            v = r.divide(
                r.add([r.multiply(d[a], r.primal(b)), r.negate(r.multiply(r.primal(a), d[b]))]),
                r.multiply(r.primal(b), r.primal(b)),
            )
        else:
            raise ValueError("row extension operator changed")
        d[i] = v
    _REPLAY_CACHE[cache_key] = d[root]
    return d[root]


def build_gate(config_path: Path) -> dict[str, Any]:
    root = config_path.resolve().parents[2]
    config = json.loads(config_path.read_text())
    if config != {
        "schema_version": "sigma-quartic-full-d2f-typed-partition-row-extension-config-1.0",
        "campaign_id": CAMPAIGN,
        "output_path": OUTPUT_PATH,
        "predecessor": PRE,
        "seals": {
            "GPU_execution_used": False,
            "live_SQLite_opened": False,
            "observations_opened": False,
        },
    }:
        raise ValueError("row extension config changed")
    for k in ("source", "config", "test"):
        if _fsha(_inside(root, PRE[k]["path"])) != PRE[k]["file_sha256"]:
            raise ValueError("row extension predecessor changed")
    pre = _bound(root, PRE["artifact"])
    _validate_predecessor(pre, root=root)
    d1 = _bound(root, pre["source_bindings"]["direct_D1_artifact"])
    nodes, entries = _load_D1(d1)
    pother_leaf = _bound(root, pre["source_bindings"]["predecessor"]["artifact"])
    p10_replay = _bound(root, pother_leaf["source_bindings"]["predecessor"]["artifact"])
    p10_leaf = _bound(root, p10_replay["source_bindings"]["predecessor"]["artifact"])
    leaf_maps = {m["candidate_id"]: m["direction_packets"] for m in p10_leaf["candidate_manifests"]}
    for m in pother_leaf["candidate_manifests"]:
        leaf_maps[m["candidate_id"]] += m["direction_packets"]
    manifests = []
    allids = set()
    for cid, packets in leaf_maps.items():
        records = []
        for packet in packets:
            atom = packet["coordinate_atom"]
            for row in range(10):
                root_index = int(entries[(row, atom)]["arithmetic_root"])
                merkle = _replay(cid, packet, root_index, nodes)
                for ordinal in packet["coordinate_ordinals"]:
                    identity = {
                        "candidate_id": cid,
                        "source_row": row,
                        "direction_slot": ordinal,
                        "coordinate_atom": atom,
                        "D1_arithmetic_root": root_index,
                        "D2_merkle_root_sha256": merkle,
                    }
                    rid = _sha(identity)
                    if rid in allids:
                        raise AssertionError("row extension collision")
                    allids.add(rid)
                    records.append({**identity, "record_id": rid})
        if len(records) != 220:
            raise ValueError("row extension record census changed")
        manifests.append(
            {
                "candidate_id": cid,
                "new_row_extension_records": records,
                "new_records": 220,
                "bounded_records_total": 242,
                "manifest_sha256": _sha(records),
                "candidate_rejection_authorized": False,
            }
        )
    if len(allids) != 2640:
        raise AssertionError("row extension global census changed")
    partition = [
        {"block": "previous_row10_diagonal_slots", "per_candidate": 22, "status": "registered"},
        {
            "block": "selected_rows0_to9_diagonal_slots",
            "per_candidate": 220,
            "status": "registered_here",
        },
        {
            "block": "registered_direction_off_diagonal_pairs",
            "per_candidate": 5082,
            "status": "blocked_missing_cross_direction_leaf_jets",
        },
        {
            "block": "registered_D1_unregistered_derivative",
            "per_candidate": 31702,
            "status": "blocked",
        },
        {
            "block": "unregistered_D1_registered_derivative",
            "per_candidate": 31702,
            "status": "blocked",
        },
        {"block": "both_directions_unregistered", "per_candidate": 188771, "status": "blocked"},
    ]
    body = {
        "schema_version": SCHEMA,
        "campaign_id": CAMPAIGN,
        "decision": "pass_220_additional_records_per_candidate_full_D2F_blocked",
        "decision_counts": {"pass": 12, "blocked": 0, "reject": 0},
        "downstream_admission_counts": {"pass": 0, "blocked": 12, "reject": 0},
        "first_blocker": FIRST_BLOCKER,
        "typed_full_domain_partition": partition,
        "partition_sha256": _sha(partition),
        "candidate_manifests": manifests,
        "manifest_sha256": _sha([m["manifest_sha256"] for m in manifests]),
        "gate_counts": {
            "full_entries_per_candidate": 257499,
            "previously_registered_per_candidate": 22,
            "newly_registered_per_candidate": 220,
            "registered_per_candidate": 242,
            "remaining_per_candidate": 257257,
            "newly_registered_all_candidates": 2640,
            "complete_D2F_tensors": 0,
            "H7_closures": 0,
            "PDE_closures": 0,
        },
        "claim_seals": {
            "typed_full_domain_partition_complete": True,
            "maximal_same_direction_row_extension_registered": True,
            "complete_D2F": False,
            "full_high_atom_identity": False,
            "physical_no_go": False,
            "candidate_rejected": False,
        },
        "source_bindings": {
            "source": {"path": SOURCE_PATH, "file_sha256": _fsha(_inside(root, SOURCE_PATH))},
            "config": {"path": CONFIG_PATH, "file_sha256": _fsha(config_path)},
            "test": {"path": TEST_PATH, "file_sha256": _fsha(_inside(root, TEST_PATH))},
            "predecessor": PRE,
        },
        "data_seals": config["seals"],
        "scope": "exact 11x153x153 typed partition and same-direction rows0-9 extension only; no cross-direction D2F, full tensor, high-atom, H7, PDE, lifespan, no-go, or rejection",
    }
    result = {**body, "content_sha256": _sha(body)}
    return result


def write_gate(p: Path) -> Path:
    x = build_gate(p)
    o = _inside(p.resolve().parents[2], OUTPUT_PATH)
    o.parent.mkdir(parents=True, exist_ok=True)
    o.write_text(json.dumps(x, indent=2, sort_keys=True) + "\n")
    return o


def main() -> int:
    a = argparse.ArgumentParser()
    a.add_argument("--config", type=Path, default=Path(CONFIG_PATH))
    print(write_gate(a.parse_args().config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
