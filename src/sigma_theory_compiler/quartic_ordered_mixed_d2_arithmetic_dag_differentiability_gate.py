"""Audit exact differentiation of the 264 target D1 arithmetic roots."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

CONFIG_SCHEMA = "sigma-quartic-ordered-mixed-d2-arithmetic-dag-differentiability-config-1.0"
RESULT_SCHEMA = "sigma-quartic-ordered-mixed-d2-arithmetic-dag-differentiability-gate-1.0"
CAMPAIGN_ID = "quartic-ordered-mixed-d2-arithmetic-dag-differentiability-001"
CONFIG_PATH = (
    "configs/backgrounds/quartic_ordered_mixed_d2_arithmetic_dag_differentiability_gate.json"
)
SOURCE_PATH = (
    "src/sigma_theory_compiler/quartic_ordered_mixed_d2_arithmetic_dag_differentiability_gate.py"
)
TEST_PATH = "tests/test_quartic_ordered_mixed_d2_arithmetic_dag_differentiability_gate.py"
OUTPUT_PATH = (
    "runs/physics-language/"
    "quartic-ordered-mixed-d2-arithmetic-dag-differentiability-gate/campaign.json"
)
FIRST_BLOCKER = (
    "register_candidate_bound_coordinate_derivatives_for_the_31680_reachable_A_B_C_"
    "component_input_leaf_obligations"
)
DAG_SHA256 = "4a227fcf136d440c4dd55e4c5525eef8e5b73681339062d7ca44cb000944ec5c"
INPUT_PROVENANCE_SHA256 = "695ff2a5fd45fa3fba21d4ce25ab2f62bd168df187c8e931bc9b5803a9cd4aed"
EXPECTED_PREDECESSOR = {
    "source": {
        "path": (
            "src/sigma_theory_compiler/quartic_p10_pother_coordinate_tangent_embedding_gate.py"
        ),
        "file_sha256": "9d42fee2340a1c284c8b4a17cb74ced5386e2cb41e50f54ee92f0c41cc5fe201",
    },
    "config": {
        "path": ("configs/backgrounds/quartic_p10_pother_coordinate_tangent_embedding_gate.json"),
        "file_sha256": "82362de865c1f13032c2cb346589262f2963408cf0794cc1fa6dfc3a37abc2d9",
    },
    "test": {
        "path": "tests/test_quartic_p10_pother_coordinate_tangent_embedding_gate.py",
        "file_sha256": "79fef8c850eff976fdd47347534c33e0635129c7fab9649b591fbb853329ed2a",
    },
    "artifact": {
        "path": (
            "runs/physics-language/quartic-p10-pother-coordinate-tangent-embedding-gate/"
            "campaign.json"
        ),
        "file_sha256": "d393050c41d58308a28a29a5b72b9da6f5ea0797b30ba97b885c3da8ca20efaa",
        "content_sha256": "fb4a55d74a8bfbe1009f13373883b5553c778441a04af143ad031f45de50e271",
    },
}
EXPECTED_CONTRACT = {
    "candidate_count": 12,
    "target_records_per_candidate": 22,
    "coordinate_obligations": 264,
    "unique_target_D1_roots_per_candidate": 20,
    "reachable_nodes": 13983,
    "reachable_component_input_labels": 341,
    "leaf_derivative_obligations_per_candidate": 2640,
    "deduplicated_leaf_derivative_obligations": 31680,
}
EXPECTED_POLICIES = {
    "derivative_root_admission": (
        "require_every_reachable_component_input_leaf_derivative_binding_and_exact_replay"
    ),
    "unbound_component_input_derivative": "block",
    "complete_ordered_D2F": "fail_closed",
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
EXPECTED_OPERATIONS = [
    "exact_constant",
    "exact_component_input",
    "exact_add",
    "exact_negate",
    "exact_multiply",
    "exact_divide",
]
EXPECTED_UNION_OPERATION_COUNTS = {
    "exact_add": 1241,
    "exact_component_input": 341,
    "exact_constant": 11,
    "exact_divide": 22,
    "exact_multiply": 12326,
    "exact_negate": 42,
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
        raise ValueError("D2 DAG differentiability path escapes project root")
    return path


def _copy(value: Any) -> Any:
    return json.loads(_canonical(value))


def _validate_config(value: Mapping[str, Any]) -> None:
    if value != {
        "schema_version": CONFIG_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "output_path": OUTPUT_PATH,
        "predecessor": EXPECTED_PREDECESSOR,
        "derivative_contract": EXPECTED_CONTRACT,
        "policies": EXPECTED_POLICIES,
        "seals": EXPECTED_SEALS,
    }:
        raise ValueError("D2 DAG differentiability config boundary changed")


def _load_bound(root: Path, binding: Mapping[str, Any]) -> dict[str, Any]:
    path = _inside(root, str(binding["path"]))
    if _file_sha(path) != binding["file_sha256"]:
        raise ValueError("D2 DAG differentiability artifact file binding changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("content_sha256") != binding["content_sha256"] or value.get(
        "content_sha256"
    ) != _content_sha(value):
        raise ValueError("D2 DAG differentiability artifact content binding changed")
    return value


def _load_predecessor(root: Path) -> dict[str, Any]:
    for label in ("source", "config", "test"):
        binding = EXPECTED_PREDECESSOR[label]
        if _file_sha(_inside(root, binding["path"])) != binding["file_sha256"]:
            raise ValueError("D2 DAG differentiability predecessor file binding changed")
    predecessor = _load_bound(root, EXPECTED_PREDECESSOR["artifact"])
    if (
        predecessor.get("decision")
        != "pass_all_264_candidate_bound_coordinate_unit_tangents_registered_D2_roots_blocked"
        or predecessor.get("gate_counts", {}).get("registered_coordinate_tangent_embeddings") != 264
        or predecessor.get("gate_counts", {}).get("registered_ordered_mixed_D2_roots") != 0
        or len(predecessor.get("candidate_manifests", [])) != 12
    ):
        raise ValueError("D2 DAG differentiability predecessor boundary changed")
    return predecessor


def _load_D1_DAG(
    root: Path, predecessor: Mapping[str, Any]
) -> tuple[list[Mapping[str, Any]], dict[tuple[int, str], Mapping[str, Any]], dict[str, Any]]:
    binding = predecessor.get("source_bindings", {}).get("full_source_D1_artifact")
    if not isinstance(binding, Mapping):
        raise TypeError("D2 DAG differentiability full D1 binding missing")
    source = _load_bound(root, binding)
    packet = source.get("common_principal_arithmetic_packet")
    manifest = source.get("common_full_entry_manifest")
    if not isinstance(packet, Mapping) or not isinstance(manifest, Mapping):
        raise TypeError("D2 DAG differentiability principal packet missing")
    dag = packet.get("arithmetic_dag")
    nodes = dag.get("nodes") if isinstance(dag, Mapping) else None
    if (
        not isinstance(dag, Mapping)
        or dag.get("content_sha256") != DAG_SHA256
        or dag.get("node_count") != 30658
        or dag.get("allowed_operations") != EXPECTED_OPERATIONS
        or not isinstance(nodes, list)
        or len(nodes) != 30658
        or packet.get("inverse_evidence", {}).get("division_assumption")
        != "c11=(-1)^11 det(A) is nonzero"
    ):
        raise ValueError("D2 DAG differentiability arithmetic DAG changed")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or len(entries) != 1683:
        raise ValueError("D2 DAG differentiability D1 manifest changed")
    entry_map = {(int(row["source_row"]), str(row["coordinate_atom"])): row for row in entries}
    if len(entry_map) != 1683:
        raise ValueError("D2 DAG differentiability D1 manifest keys changed")
    return nodes, entry_map, dict(binding)


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
        raise ValueError("D2 DAG differentiability unknown arithmetic operator")
    if not isinstance(children, list) or any(type(index) is not int for index in children):
        raise TypeError("D2 DAG differentiability child reference schema changed")
    return children


def _closure(root_index: int, nodes: list[Mapping[str, Any]]) -> set[int]:
    if not 0 <= root_index < len(nodes):
        raise ValueError("D2 DAG differentiability root outside arithmetic DAG")
    seen: set[int] = set()
    stack = [root_index]
    while stack:
        index = stack.pop()
        if index in seen:
            continue
        if not 0 <= index < len(nodes):
            raise ValueError("D2 DAG differentiability child outside arithmetic DAG")
        seen.add(index)
        stack.extend(_children(nodes[index]))
    return seen


def _target_templates(
    predecessor: Mapping[str, Any],
    nodes: list[Mapping[str, Any]],
    entries: Mapping[tuple[int, str], Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    first = predecessor["candidate_manifests"][0]["embedding_records"]
    by_key: dict[tuple[str, int], dict[str, Any]] = {}
    union: set[int] = set()
    for row in first:
        key = (str(row["coordinate_atom"]), int(row["coordinate_column"]))
        entry = entries.get((10, key[0]))
        if entry is None or int(entry["coordinate_column"]) != key[1]:
            raise ValueError("D2 DAG differentiability target D1 anchor changed")
        root_index = int(entry["arithmetic_root"])
        closure = _closure(root_index, nodes)
        leaves = sorted(
            str(nodes[index]["label"])
            for index in closure
            if nodes[index].get("op") == "exact_component_input"
        )
        if len(leaves) != 132 or len(set(leaves)) != 132:
            raise ValueError("D2 DAG differentiability target leaf closure changed")
        if any(
            nodes[index].get("provenance_sha256") != INPUT_PROVENANCE_SHA256
            for index in closure
            if nodes[index].get("op") == "exact_component_input"
        ):
            raise ValueError("D2 DAG differentiability component input provenance changed")
        union |= closure
        template = by_key.setdefault(
            key,
            {
                "coordinate_atom": key[0],
                "coordinate_column": key[1],
                "direction_label": row["direction_label"],
                "D1_arithmetic_root": root_index,
                "D1_arithmetic_dag_sha256": DAG_SHA256,
                "coordinate_ordinals": [],
                "reachable_nodes": len(closure),
                "reachable_component_input_labels": leaves,
                "reachable_component_input_count": 132,
                "operation_counts": dict(
                    sorted(Counter(str(nodes[index]["op"]) for index in closure).items())
                ),
            },
        )
        if template["D1_arithmetic_root"] != root_index:
            raise ValueError("D2 DAG differentiability duplicate target root changed")
        template["coordinate_ordinals"].append(int(row["coordinate_ordinal"]))
    operation_counts = dict(sorted(Counter(str(nodes[index]["op"]) for index in union).items()))
    union_leaves = {
        str(nodes[index]["label"])
        for index in union
        if nodes[index].get("op") == "exact_component_input"
    }
    if (
        len(by_key) != 20
        or len(union) != 13983
        or len(union_leaves) != 341
        or operation_counts != EXPECTED_UNION_OPERATION_COUNTS
    ):
        raise ValueError("D2 DAG differentiability union closure changed")
    return list(by_key.values()), {
        "reachable_nodes": len(union),
        "reachable_component_input_labels": len(union_leaves),
        "operation_counts": operation_counts,
        "component_input_family_counts": dict(
            sorted(Counter(label.split("[")[0] for label in union_leaves).items())
        ),
        "closure_sha256": _sha(sorted(union)),
    }


def _leaf_obligation(
    candidate_id: str, template: Mapping[str, Any], label: str, embedding_id: str
) -> dict[str, Any]:
    identity = {
        "candidate_id": candidate_id,
        "target_coordinate_atom": template["coordinate_atom"],
        "target_coordinate_column": template["coordinate_column"],
        "target_D1_arithmetic_root": template["D1_arithmetic_root"],
        "component_input_label": label,
        "component_input_provenance_sha256": INPUT_PROVENANCE_SHA256,
        "tangent_embedding_id": embedding_id,
    }
    return {
        **identity,
        "leaf_derivative_obligation_id": _sha(identity),
        "component_input_coordinate_derivative_root_registered": False,
        "component_input_coordinate_derivative_dag_sha256_registered": False,
        "obligation_status": "required_unregistered",
        "candidate_rejection_authorized": False,
    }


def _candidate_manifests(
    predecessor: Mapping[str, Any], templates: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    template_map = {(row["coordinate_atom"], row["coordinate_column"]): row for row in templates}
    manifests = []
    all_ids: set[str] = set()
    for candidate in predecessor["candidate_manifests"]:
        embedding_map = {}
        for row in candidate["embedding_records"]:
            key = (row["coordinate_atom"], row["coordinate_column"])
            vector_sha256 = _sha(
                {
                    "coordinate_atom_basis_sha256": row["coordinate_atom_basis_sha256"],
                    "basis_dimension": row["basis_dimension"],
                    "sparse_entries": row["sparse_entries"],
                }
            )
            embedding_map.setdefault(key, vector_sha256)
            if embedding_map[key] != vector_sha256:
                raise ValueError("D2 DAG differentiability duplicate unit tangent changed")
        packets = []
        candidate_ids: set[str] = set()
        for key, template in template_map.items():
            obligations = [
                _leaf_obligation(candidate["candidate_id"], template, label, embedding_map[key])
                for label in template["reachable_component_input_labels"]
            ]
            ids = {row["leaf_derivative_obligation_id"] for row in obligations}
            if len(ids) != 132 or ids & candidate_ids or ids & all_ids:
                raise AssertionError("D2 DAG differentiability leaf obligation collision")
            candidate_ids |= ids
            packets.append(
                {
                    "target_coordinate_atom": template["coordinate_atom"],
                    "target_coordinate_column": template["coordinate_column"],
                    "coordinate_ordinals": template["coordinate_ordinals"],
                    "D1_arithmetic_root": template["D1_arithmetic_root"],
                    "tangent_embedding_id": embedding_map[key],
                    "required_leaf_derivatives": 132,
                    "registered_leaf_derivatives": 0,
                    "leaf_derivative_obligations": obligations,
                    "packet_sha256": _sha(obligations),
                }
            )
        if len(candidate_ids) != 2640:
            raise AssertionError("D2 DAG differentiability candidate obligations incomplete")
        all_ids |= candidate_ids
        manifests.append(
            {
                "candidate_id": candidate["candidate_id"],
                "unique_target_D1_roots": 20,
                "required_leaf_derivative_obligations": 2640,
                "registered_leaf_derivative_roots": 0,
                "ordered_mixed_D2_roots_registered": 0,
                "derivative_packets": packets,
                "candidate_manifest_sha256": _sha(packets),
                "candidate_decision": "blocked_missing_component_input_leaf_derivatives",
                "candidate_rejection_authorized": False,
                "first_blocker": FIRST_BLOCKER,
            }
        )
    if len(all_ids) != 31680:
        raise AssertionError("D2 DAG differentiability global obligations incomplete")
    return manifests


def _expected_body(root: Path, config_path: Path, predecessor: Mapping[str, Any]) -> dict[str, Any]:
    nodes, entries, source_binding = _load_D1_DAG(root, predecessor)
    templates, union = _target_templates(predecessor, nodes, entries)
    manifests = _candidate_manifests(predecessor, templates)
    return {
        "schema_version": RESULT_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "decision": "pass_exact_D1_DAG_differentiability_boundary_31680_leaf_jets_missing_D2_blocked",
        "decision_counts": {"pass": 12, "blocked": 0, "reject": 0},
        "downstream_admission_counts": {"pass": 0, "blocked": 12, "reject": 0},
        "first_blocker": FIRST_BLOCKER,
        "differentiability_theorem": {
            "name": "closed_operator_calculus_with_unbound_component_input_leaf_jets",
            "premises": (
                "The bound principal D1 DAG has exact constant, input, sum, negation, product, "
                "and quotient nodes, with det(A) nonzero on its declared domain. The 20 distinct "
                "target roots reach 13,983 nodes and 341 A/B/C input labels. Every target root "
                "closure contains exactly 132 component-input leaves."
            ),
            "exact_result": (
                "The five non-input operations have closed exact derivative rules, but each "
                "exact_component_input node supplies only a value label and provenance hash. No "
                "registered root gives its derivative along any target coordinate tangent. "
                "Deduplication over repeated target roots leaves exactly 2,640 required leaf jets "
                "per candidate and 31,680 candidate-bound obligations."
            ),
            "boundary": (
                "No ordered mixed-D2 arithmetic root can be emitted until all reachable input "
                "leaf derivatives for its candidate and tangent are registered. This is an input-"
                "jet schema obstruction, not a physical no-go or evidence that any derivative is "
                "zero."
            ),
        },
        "operator_derivative_rules": {
            "exact_constant": "D(c)=0",
            "exact_component_input": "requires_registered_candidate_coordinate_derivative_leaf",
            "exact_add": "D(sum_i x_i)=sum_i D(x_i)",
            "exact_negate": "D(-x)=-D(x)",
            "exact_multiply": "D(x*y)=D(x)*y+x*D(y)",
            "exact_divide": "D(x/y)=(D(x)*y-x*D(y))/(y*y)_on_registered_nonzero_denominator_domain",
            "closed_noninput_operator_rules": 5,
            "unbound_input_operator_kinds": 1,
            "derivative_DAG_emitted": False,
        },
        "target_root_templates": templates,
        "union_dependency_closure": union,
        "candidate_manifests": manifests,
        "leaf_derivative_obligation_manifest_sha256": _sha(
            [row["candidate_manifest_sha256"] for row in manifests]
        ),
        "gate_counts": {
            "selected_candidates": 12,
            "target_coordinate_records": 264,
            "unique_target_D1_roots_per_candidate": 20,
            "reachable_D1_DAG_nodes": 13983,
            "reachable_component_input_labels": 341,
            "component_input_leaves_per_target_root": 132,
            "raw_leaf_derivative_references": 34848,
            "deduplicated_leaf_derivative_obligations": 31680,
            "registered_leaf_derivative_roots": 0,
            "registered_ordered_mixed_D2_roots": 0,
            "blocked_ordered_mixed_D2_roots": 264,
            "complete_ordered_D2F_tensors_registered": 0,
            "full_high_atom_good_unknown_identities_proved": 0,
            "global_H7_closures": 0,
            "nonlinear_PDE_closures": 0,
            "lifespans_proved": 0,
        },
        "claim_seals": {
            "target_D1_DAG_dependency_closure_replayed": True,
            "exact_noninput_operator_derivative_rules_closed": True,
            "minimal_31680_leaf_derivative_obligations_materialized": True,
            "component_input_leaf_derivatives_registered": False,
            "ordered_mixed_D2_values_registered": False,
            "physical_no_go_proved": False,
            "complete_ordered_D2F_tensor_registered": False,
            "full_high_atom_good_unknown_identity_proved": False,
            "global_H7_energy_closed": False,
            "nonlinear_PDE_closed": False,
            "nonlinear_lifespan_proved": False,
            "candidate_theory_rejected": False,
            "observational_claim_made": False,
        },
        "exact_controls": {
            "treat_component_input_value_as_its_coordinate_derivative": {"rejected": True},
            "default_unbound_input_derivative_to_zero": {"rejected": True},
            "emit_formal_derivative_skeleton_as_arithmetic_root": {"rejected": True},
            "differentiate_division_without_nonzero_domain": {"rejected": True},
            "promote_missing_leaf_jet_to_physical_no_go": {"rejected": True},
            "reject_candidate_for_missing_leaf_jet": {"rejected": True},
        },
        "data_seals": dict(EXPECTED_SEALS),
        "source_bindings": {
            "source": {"path": SOURCE_PATH, "file_sha256": _file_sha(_inside(root, SOURCE_PATH))},
            "config": {"path": CONFIG_PATH, "file_sha256": _file_sha(config_path)},
            "test": {"path": TEST_PATH, "file_sha256": _file_sha(_inside(root, TEST_PATH))},
            "predecessor": _copy(EXPECTED_PREDECESSOR),
            "full_source_D1_artifact": source_binding,
        },
        "scope": (
            "candidate-bound exact dependency and differentiability audit for the 264 target D1 "
            "roots along registered coordinate tangents, materializing only missing A/B/C input-"
            "leaf derivative obligations; no invented derivative root, D2 admission, physical no-"
            "go, full D2F, high-atom identity, H7, PDE, lifespan, rejection, or observation"
        ),
    }


def _validate_bindings(value: Mapping[str, Any], root: Path) -> None:
    bindings = value.get("source_bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != {
        "source",
        "config",
        "test",
        "predecessor",
        "full_source_D1_artifact",
    }:
        raise ValueError("D2 DAG differentiability source binding keys changed")
    for label, relative in {
        "source": SOURCE_PATH,
        "config": CONFIG_PATH,
        "test": TEST_PATH,
    }.items():
        if bindings[label] != {
            "path": relative,
            "file_sha256": _file_sha(_inside(root, relative)),
        }:
            raise ValueError("D2 DAG differentiability local binding changed")
    if bindings["predecessor"] != EXPECTED_PREDECESSOR:
        raise ValueError("D2 DAG differentiability predecessor binding changed")
    predecessor = _load_predecessor(root)
    if (
        bindings["full_source_D1_artifact"]
        != predecessor["source_bindings"]["full_source_D1_artifact"]
    ):
        raise ValueError("D2 DAG differentiability full D1 binding changed")


def _validate_result(value: Mapping[str, Any], *, root: Path | None = None) -> None:
    validation_root = (root or Path(__file__).resolve().parents[2]).resolve()
    if value.get("content_sha256") != _content_sha(value):
        raise ValueError("D2 DAG differentiability content hash changed")
    _validate_bindings(value, validation_root)
    config_path = _inside(validation_root, CONFIG_PATH)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    predecessor = _load_predecessor(validation_root)
    expected = _expected_body(validation_root, config_path, predecessor)
    if {key: item for key, item in value.items() if key != "content_sha256"} != expected:
        raise ValueError("D2 DAG differentiability result boundary changed")


def build_gate(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    root = config_path.parents[2]
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    predecessor = _load_predecessor(root)
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
