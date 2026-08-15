from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import Any


class System10GravityScalarAWReadinessError(RuntimeError):
    """Raised when the fixed-r gravity/scalar A/W readiness contract fails closed."""


def _canonical_sha(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode("ascii")).hexdigest()


def _canonical_lf_sha(path: Path) -> str:
    try:
        raw = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    except OSError as exc:
        raise System10GravityScalarAWReadinessError(f"cannot read bound file: {path}") from exc
    return hashlib.sha256(raw).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise System10GravityScalarAWReadinessError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise System10GravityScalarAWReadinessError("bound JSON root is not an object")
    return value


def _resolve(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if root.resolve() not in path.parents:
        raise System10GravityScalarAWReadinessError("bound path escapes repository root")
    return path


def _load_binding(root: Path, binding: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    path = _resolve(root, str(binding.get("path", "")))
    if _canonical_lf_sha(path) != binding.get("canonical_lf_sha256"):
        raise System10GravityScalarAWReadinessError(f"bound file hash mismatch: {path}")
    value = _load_json(path)
    if value.get("content_sha256") != binding.get("content_sha256"):
        raise System10GravityScalarAWReadinessError(f"bound content hash mismatch: {path}")
    return path, value


def _load_source(root: Path, binding: dict[str, Any]) -> Path:
    path = _resolve(root, str(binding.get("path", "")))
    if _canonical_lf_sha(path) != binding.get("canonical_lf_sha256"):
        raise System10GravityScalarAWReadinessError(f"bound source hash mismatch: {path}")
    return path


def _with_sha(body: dict[str, Any], key: str) -> dict[str, Any]:
    return {**body, key: _canonical_sha(body)}


def _candidate_map(value: dict[str, Any]) -> dict[str, dict[str, Any]]:
    items = value.get("certificates", [])
    try:
        result = {str(item["candidate_id"]): item for item in items}
    except (KeyError, TypeError) as exc:
        raise System10GravityScalarAWReadinessError("candidate manifest malformed") from exc
    if len(result) != 12 or len(items) != 12:
        raise System10GravityScalarAWReadinessError("candidate census changed")
    return result


def _root_shape(roots: Any, rows: int, columns: int | None = None) -> bool:
    if not isinstance(roots, list) or len(roots) != rows:
        return False
    if columns is None:
        return all(isinstance(item, int) for item in roots)
    return all(
        isinstance(row, list) and len(row) == columns and all(isinstance(item, int) for item in row)
        for row in roots
    )


def _validate_predecessors(
    bound: dict[str, tuple[Path, dict[str, Any]]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    blocker = bound["gravity_scalar_blocker"][1]
    tensor = bound["metric_tensor_dag"][1]
    arithmetic = bound["lower_row_arithmetic"][1]
    nonlinear = bound["nonlinear_evolution"][1]
    domain = bound["r_positive_domain"][1]
    if (
        blocker.get("decision") != "BLOCK_GRAVITY_SCALAR_REQUIRES_COUPLED_11X11_FIXED_R_SOLVE"
        or blocker.get("counts", {}).get("total_rhs_rows_closed_per_candidate") != 74
        or blocker.get("counts", {}).get("exact_candidate_ambiguity_witnesses") != 12
    ):
        raise System10GravityScalarAWReadinessError("gravity-scalar blocker changed")
    if (
        tensor.get("status")
        != "pass_all_12_all_Euler_rows_tensor_lowered_mixed_incomplete_fail_closed"
    ):
        raise System10GravityScalarAWReadinessError("tensor-DAG status changed")
    packet = tensor.get("common_explicit_tensor_dag_packet", {})
    roots = packet.get("root_packet", {})
    dag = packet.get("tensor_dag", {})
    if (
        not _root_shape(roots.get("time_block_A"), 11, 11)
        or not _root_shape(roots.get("acceleration_free_W"), 11)
        or not _root_shape(roots.get("Euler_E"), 11)
        or not _root_shape(roots.get("solved_source_F"), 11)
        or dag.get("node_count") != len(dag.get("nodes", []))
        or dag.get("content_sha256")
        != "045c5935e4350018364db7f51729a031add8488cde72fcf5380a43cf279d3d29"
        or roots.get("content_sha256")
        != "900f3d261e4f725b25f3375ae5064c9f2dffacbd26e4666a94be614f26331439"
    ):
        raise System10GravityScalarAWReadinessError("11x11 operational root packet changed")
    if arithmetic.get("status") != "pass_all_12_all_lower_rows_arithmetic_mixed_tensor_fail_closed":
        raise System10GravityScalarAWReadinessError("arithmetic checkpoint status changed")
    arithmetic_nodes = (
        arithmetic.get("common_rows5_10_arithmetic_packet", {})
        .get("arithmetic_dag", {})
        .get("nodes", [])
    )
    component_inputs = {
        str(node.get("label")): node
        for node in arithmetic_nodes
        if node.get("op") == "exact_component_input"
    }
    required_labels = {
        *{f"A[{row},{column}]" for row in range(11) for column in range(11)},
        *{f"W[{row}]" for row in range(11)},
    }
    if not required_labels <= set(component_inputs):
        raise System10GravityScalarAWReadinessError("A/W component placeholder set changed")
    root_sha = str(roots["content_sha256"])
    if any(
        component_inputs[label].get("provenance_sha256") != root_sha for label in required_labels
    ):
        raise System10GravityScalarAWReadinessError("A/W placeholder provenance changed")
    if (
        nonlinear.get("status")
        != "pass_all_12_exact_local_nonlinear_time_acceleration_eliminations"
    ):
        raise System10GravityScalarAWReadinessError("nonlinear source authority changed")
    if (
        domain.get("decision")
        != "BOUNDED_PASS_FIXED_CYLINDRICAL_R_POSITIVE_1010_JETS_AND_96_ROWS_NO_PROPAGATION_CLAIM"
        or domain.get("materialization", {}).get("domain_certificate", {}).get("domain") != "r>0"
    ):
        raise System10GravityScalarAWReadinessError("r-positive authority changed")
    tensor_candidates = _candidate_map(tensor)
    nonlinear_candidates = _candidate_map(nonlinear)
    if set(tensor_candidates) != set(nonlinear_candidates):
        raise System10GravityScalarAWReadinessError("candidate identity join changed")
    candidate_id = min(tensor_candidates)
    tensor_candidate = tensor_candidates[candidate_id]
    nonlinear_candidate = nonlinear_candidates[candidate_id]
    if tensor_candidate.get("coefficients") != nonlinear_candidate.get("coefficients"):
        raise System10GravityScalarAWReadinessError("representative coefficients changed")
    representative = {
        "candidate_id": candidate_id,
        "coefficients": tensor_candidate["coefficients"],
        "tensor_provenance": tensor_candidate["provenance"],
        "local_solve_certificate": {
            "time_block_determinant_nonzero": nonlinear_candidate["time_block_determinant_nonzero"],
            "acceleration_solution_residual_zero": nonlinear_candidate[
                "acceleration_solution_residual_zero"
            ],
            "certified_local_jet_bound": nonlinear_candidate["certified_local_jet_bound"],
        },
    }
    return representative, packet, {label: component_inputs[label] for label in required_labels}


def _function_node(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise System10GravityScalarAWReadinessError(f"source function missing: {name}")


def _call_name(call: ast.Call) -> str:
    function = call.func
    if isinstance(function, ast.Name):
        return function.id
    if isinstance(function, ast.Attribute):
        prefix = _call_name(ast.Call(func=function.value, args=[], keywords=[]))
        return f"{prefix}.{function.attr}" if prefix else function.attr
    return ""


def _source_api_audit(source_path: Path) -> dict[str, Any]:
    try:
        text = source_path.read_text(encoding="utf-8")
        tree = ast.parse(text)
    except (OSError, SyntaxError) as exc:
        raise System10GravityScalarAWReadinessError("cannot audit nonlinear source API") from exc
    action = _function_node(tree, "quartic_action_euler_tensors")
    adapter = _function_node(tree, "gauge_fixed_euler_from_state")
    assembler = _function_node(tree, "_assemble_equations")
    action_calls = [_call_name(node) for node in ast.walk(action) if isinstance(node, ast.Call)]
    assembler_calls = [
        _call_name(node) for node in ast.walk(assembler) if isinstance(node, ast.Call)
    ]
    adapter_parameters = [
        argument.arg for argument in (*adapter.args.args, *adapter.args.kwonlyargs)
    ]
    forbidden_checkpoint_parameters = {
        "row",
        "rows",
        "checkpoint",
        "callback",
        "budget",
        "output_path",
    }
    audit = {
        "source_path": "/".join(source_path.parts[-3:]),
        "source_canonical_lf_sha256": _canonical_lf_sha(source_path),
        "action_function": "quartic_action_euler_tensors",
        "action_factor_tensor_calls": action_calls.count("_factor_tensor"),
        "action_simplify_calls": action_calls.count("sp.simplify"),
        "assembler_factor_calls": assembler_calls.count("sp.factor"),
        "public_adapter": "gauge_fixed_euler_from_state",
        "adapter_parameters": adapter_parameters,
        "checkpoint_parameters_present": sorted(
            forbidden_checkpoint_parameters & set(adapter_parameters)
        ),
        "per_row_return_before_full_tensor_factor": False,
        "atomic_A_W_entry_serializer": False,
    }
    if (
        audit["action_factor_tensor_calls"] != 3
        or audit["action_simplify_calls"] != 1
        or audit["assembler_factor_calls"] != 2
        or audit["checkpoint_parameters_present"]
    ):
        raise System10GravityScalarAWReadinessError("nonlinear source API audit changed")
    return _with_sha(audit, "audit_sha256")


def _representative_manifest(
    representative: dict[str, Any], packet: dict[str, Any]
) -> dict[str, Any]:
    roots = packet["root_packet"]
    a_entries = []
    for row, row_roots in enumerate(roots["time_block_A"]):
        for column, root in enumerate(row_roots):
            body = {
                "label": f"A[{row},{column}]",
                "row": row,
                "column": column,
                "semantic_root": root,
                "common_tensor_dag_sha256": packet["tensor_dag"]["content_sha256"],
                "coefficient_substitution_sha256": _canonical_sha(representative["coefficients"]),
            }
            a_entries.append(_with_sha(body, "entry_sha256"))
    w_entries = []
    for row, root in enumerate(roots["acceleration_free_W"]):
        body = {
            "label": f"W[{row}]",
            "row": row,
            "semantic_root": root,
            "common_tensor_dag_sha256": packet["tensor_dag"]["content_sha256"],
            "coefficient_substitution_sha256": _canonical_sha(representative["coefficients"]),
        }
        w_entries.append(_with_sha(body, "entry_sha256"))
    body = {
        "candidate_id": representative["candidate_id"],
        "candidate_selection": "lexicographically_first_of_12",
        "coefficients": representative["coefficients"],
        "domain": "fixed cylindrical r>0 generic registered 85-state jet",
        "semantic_A_entries": a_entries,
        "semantic_W_entries": w_entries,
        "semantic_A_entry_count": len(a_entries),
        "semantic_W_entry_count": len(w_entries),
        "semantic_root_packet_sha256": roots["content_sha256"],
        "entrywise_coordinate_arithmetic_A_entries": 0,
        "entrywise_coordinate_arithmetic_W_entries": 0,
        "candidate_dynamic_rows_closed": 0,
        "status": "SEMANTIC_ROOT_MANIFEST_COMPLETE_ARITHMETIC_MATERIALIZATION_BLOCKED",
    }
    return _with_sha(body, "manifest_sha256")


def _materialize(
    bound: dict[str, tuple[Path, dict[str, Any]]],
    nonlinear_source: Path,
    frozen: dict[str, Any],
) -> dict[str, Any]:
    representative, packet, component_inputs = _validate_predecessors(bound)
    manifest = _representative_manifest(representative, packet)
    audit = _source_api_audit(nonlinear_source)
    component_input_labels = sorted(component_inputs)
    component_body = {
        "component_inputs": len(component_input_labels),
        "baseline_labels_sha256": _canonical_sha(component_input_labels),
        "operation": "exact_component_input",
        "semantic_root_packet_sha256": packet["root_packet"]["content_sha256"],
        "coordinate_arithmetic_expressions_embedded": 0,
    }
    component_boundary = _with_sha(component_body, "boundary_sha256")
    measured = {
        "representative_candidates": 1,
        "semantic_A_entries": manifest["semantic_A_entry_count"],
        "semantic_W_entries": manifest["semantic_W_entry_count"],
        "arithmetic_A_entries": manifest["entrywise_coordinate_arithmetic_A_entries"],
        "arithmetic_W_entries": manifest["entrywise_coordinate_arithmetic_W_entries"],
        "manifest_sha256": manifest["manifest_sha256"],
        "source_api_audit_sha256": audit["audit_sha256"],
        "component_boundary_sha256": component_boundary["boundary_sha256"],
    }
    if measured != frozen:
        raise System10GravityScalarAWReadinessError("frozen A/W readiness expectations changed")
    missing = {
        "primitive_id": (
            "checkpointable_unfactored_fixed_r_positive_coordinate_A_W_materializer_v1"
        ),
        "input": {
            "candidate_id": representative["candidate_id"],
            "candidate_coefficients": representative["coefficients"],
            "domain": "fixed cylindrical r>0 generic registered 85-state jet",
            "semantic_root_packet_sha256": packet["root_packet"]["content_sha256"],
        },
        "required_output": {
            "A_entries": 121,
            "W_entries": 11,
            "format": "exact checkpointed coordinate-arithmetic DAG or canonical expressions",
            "per_entry_immutable_seal": True,
        },
        "acceptance": [
            "emit each A[i,j] and W[i] without waiting for whole-tensor factorization",
            "prove E-A*a-W=0 for all eleven Euler rows in the registered acceleration basis",
            "prove every A and W expression is acceleration-free after the affine split",
            "certify every radial denominator and exclude all poles on the declared r>0 domain",
            "specialize r=1 and replay the registered local time-block authority exactly",
            "reject zero-fill, missing-entry, coefficient, row-order, sign, pole, and seal tamper",
        ],
        "current_source_obstruction": {
            "whole_tensor_factor_before_return": True,
            "per_row_checkpoint_api": False,
            "atomic_entry_serializer": False,
            "arithmetic_consumers_receive_component_placeholders": True,
        },
        "status": "BLOCK_FIRST_IMPLEMENTATION_PRIMITIVE_MISSING",
    }
    missing = _with_sha(missing, "block_sha256")
    negatives = {
        "promote_semantic_root_to_arithmetic_value": {
            "semantic_A_entries": 121,
            "arithmetic_A_entries": 0,
            "rejected": True,
        },
        "promote_component_placeholder": {
            "placeholder_operation": "exact_component_input",
            "embedded_coordinate_expression": False,
            "rejected": True,
        },
        "reuse_local_point_determinant": {
            "local_point_nonzero": representative["local_solve_certificate"][
                "time_block_determinant_nonzero"
            ],
            "symbolic_r_positive_domain_nonzero": False,
            "rejected": True,
        },
        "claim_dynamic_row": {
            "coordinate_A_W_entries": 0,
            "full_Euler_residual_replays": 0,
            "rejected": True,
        },
    }
    return {
        "representative_candidate": representative,
        "semantic_A_W_manifest": manifest,
        "arithmetic_component_boundary": component_boundary,
        "live_source_api_audit": audit,
        "bounded_probe_observation": {
            "candidate_id": representative["candidate_id"],
            "operation": "quartic_action_euler_tensors on generic fixed-r>0 coordinate jet",
            "wall_clock_cap_seconds": 300,
            "outcome": "TIMEOUT_BEFORE_FIRST_RETURNED_ACTION_PACKET",
            "scientific_role": "supplemental operational evidence; not used as an algebraic proof",
        },
        "first_missing_primitive": missing,
        "negative_controls": negatives,
    }


def build_receipt(config_path: Path, *, root: Path | None = None) -> dict[str, Any]:
    repository = (root or config_path.resolve().parents[1]).resolve()
    config = _load_json(config_path)
    if config.get("schema_version") != (
        "invariant-system10-cylindrical-r-positive-gravity-scalar-aw-readiness-config-1.0"
    ):
        raise System10GravityScalarAWReadinessError("unsupported config schema")
    expected_caps = {
        "candidates": 12,
        "representative_candidates": 1,
        "state_dimension": 85,
        "A_entries": 121,
        "W_entries": 11,
        "probe_wall_clock_seconds": 300,
        "maximum_output_bytes": 262144,
    }
    if config.get("caps") != expected_caps:
        raise System10GravityScalarAWReadinessError("caps changed")
    expected_claims = {
        "semantic_A_W_root_manifest": True,
        "coordinate_arithmetic_A_W": False,
        "solved_dynamic_rows": False,
        "all_twelve_candidates": False,
        "full_85_state_rhs": False,
        "constraint_propagation": False,
        "hyperbolicity": False,
        "promotion": False,
    }
    if config.get("claims_policy") != expected_claims:
        raise System10GravityScalarAWReadinessError("claims policy broadened")
    bound = {
        name: _load_binding(repository, binding)
        for name, binding in config.get("bindings", {}).items()
    }
    if set(bound) != {
        "gravity_scalar_blocker",
        "metric_tensor_dag",
        "lower_row_arithmetic",
        "nonlinear_evolution",
        "r_positive_domain",
    }:
        raise System10GravityScalarAWReadinessError("binding manifest changed")
    sources = {
        name: _load_source(repository, binding)
        for name, binding in config.get("source_evidence", {}).items()
    }
    if set(sources) != {"source", "test", "nonlinear_source"}:
        raise System10GravityScalarAWReadinessError("source evidence manifest changed")
    expected_test = repository / (
        "tests/test_system10_cylindrical_r_positive_gravity_scalar_aw_readiness.py"
    )
    expected_nonlinear = repository / (
        "src/sigma_theory_compiler/quartic_nonlinear_evolution_campaign.py"
    )
    if (
        sources["source"] != Path(__file__).resolve()
        or sources["test"] != expected_test
        or sources["nonlinear_source"] != expected_nonlinear
    ):
        raise System10GravityScalarAWReadinessError("self evidence path changed")
    materialization = _materialize(
        bound, sources["nonlinear_source"], config.get("frozen_expectations", {})
    )
    body = {
        "schema_version": (
            "invariant-system10-cylindrical-r-positive-gravity-scalar-aw-readiness-result-1.0"
        ),
        "campaign_id": config["campaign_id"],
        "decision": "BLOCK_COORDINATE_ARITHMETIC_A_W_MATERIALIZER_MISSING",
        "materialization": materialization,
        "counts": {
            "registered_candidates": 12,
            "representative_candidates_audited": 1,
            "semantic_A_entries_manifested": 121,
            "semantic_W_entries_manifested": 11,
            "coordinate_arithmetic_A_entries": 0,
            "coordinate_arithmetic_W_entries": 0,
            "candidate_dynamic_rows_closed": 0,
            "rhs_rows_closed_per_candidate": 74,
            "candidate_dynamic_rows_remaining": 132,
            "negative_controls": 4,
        },
        "claims": {
            "semantic_A_W_root_manifest_complete_for_representative": True,
            "coordinate_arithmetic_A_W_materialized": False,
            "representative_dynamic_rows_closed": False,
            "all_twelve_candidates_closed": False,
            "full_85_state_rhs_closed": False,
            "constraint_propagation_closed": False,
            "hyperbolicity_closed": False,
            "promotion_authorized": False,
        },
        "scope": (
            "Exact readiness audit for the remaining candidate gravity/scalar 11x11 "
            "acceleration block. For the deterministic representative candidate, all 121 "
            "A roots and 11 W roots are individually manifested and bound to the registered "
            "tensor DAG, coefficients, and fixed cylindrical r>0 domain. They are not "
            "coordinate-arithmetic expressions: every downstream arithmetic consumer still "
            "imports A[i,j] and W[i] as opaque exact_component_input nodes. Static source "
            "audit shows the live Euler adapter factors the complete tensor before returning "
            "and has no row/checkpoint serializer; a bounded 300-second generic-jet probe "
            "timed out before return. The first missing implementation primitive and its exact "
            "acceptance tests are sealed. No dynamic row, full RHS, propagation, "
            "hyperbolicity, or promotion claim is made."
        ),
        "source_bindings": {
            "config": {
                "path": config_path.relative_to(repository).as_posix(),
                "canonical_json_sha256": _canonical_sha(config),
            },
            **{
                name: {
                    "path": path.relative_to(repository).as_posix(),
                    "canonical_lf_sha256": _canonical_lf_sha(path),
                    "content_sha256": value["content_sha256"],
                }
                for name, (path, value) in bound.items()
            },
            **{
                name: {
                    "path": path.relative_to(repository).as_posix(),
                    "canonical_lf_sha256": _canonical_lf_sha(path),
                }
                for name, path in sources.items()
            },
        },
    }
    receipt = {**body, "content_sha256": _canonical_sha(body)}
    if len(json.dumps(receipt).encode("utf-8")) > expected_caps["maximum_output_bytes"]:
        raise System10GravityScalarAWReadinessError("output cap exceeded")
    return receipt


def write_receipt(
    config_path: Path, output_path: Path, *, root: Path | None = None
) -> dict[str, Any]:
    receipt = build_receipt(config_path, root=root)
    data = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if output_path.exists() and output_path.read_bytes() != data:
        raise System10GravityScalarAWReadinessError("immutable output conflict")
    if not output_path.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(data)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    write_receipt(arguments.config.resolve(), arguments.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
