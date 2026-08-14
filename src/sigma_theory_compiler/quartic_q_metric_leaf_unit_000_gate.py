"""Materialize the first bounded q-metric live A/B/C leaf unit exactly."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from functools import cache
from pathlib import Path
from typing import Any

import sympy as sp

from . import quartic_p0_metric_lower_abc_leaf_authority_gate as base
from . import quartic_q_metric_lower_abc_leaf_authority_gate as checkpoint
from .quartic_remaining_scalar_hessian_abc_leaf_authority_gate import (
    _content_sha,
    _production_sha,
    _sha,
)
from .quartic_unspecialized_source_jacobian_campaign import _unspecialized_principal_blocks

CONFIG_SCHEMA = "sigma-quartic-q-metric-leaf-unit-config-1.0"
RESULT_SCHEMA = "sigma-quartic-q-metric-leaf-unit-1.0"
CAMPAIGN_ID = "quartic-q-metric-leaf-unit-000"
STEM = "quartic_q_metric_leaf_unit_000_gate"
SLUG = "quartic-q-metric-leaf-unit-000-gate"
CONFIG_PATH = f"configs/backgrounds/{STEM}.json"
SOURCE_PATH = f"src/sigma_theory_compiler/{STEM}.py"
TEST_PATH = f"tests/test_{STEM}.py"
OUTPUT_PATH = f"runs/physics-language/{SLUG}/campaign.json"
CONFIG_PRODUCTION_SHA256 = "ea3e4ae1dcc2a6313dded5348d5ce53ac52d6f82d8e4d9c9f4bccd65bb87c95f"
TEST_PRODUCTION_SHA256 = "d21433c00dfeafa58cf99acef872455d65b5e177f64c78d62a0de652abbe2792"
UNIT_INDEX = 0
Q_ATOM = "q[0]"
Q_COLUMN = 0
CONTRACT = {
    "unit_index": 0,
    "total_checkpoint_units": 120,
    "q_atom": "q[0]",
    "q_column": 0,
    "candidate_ordinal": 0,
    "target_atoms": 20,
    "leaf_roots_per_target": 132,
    "materialized_leaf_roots": 2640,
    "planned_leaf_roots_all_candidates": 316800,
    "registered_D2_entries_per_candidate": 5324,
}
POLICIES = {
    "unit_order": "candidate_major_then_q_atom_ascending",
    "canonicalization": "exact_sympy_expand_mul_then_srepr_hash",
    "factor_terms": "forbidden",
    "root_admission": "only_materialized_sparse_live_A_B_C_composition",
    "q_family_promotion": "forbidden_until_all_120_units_replay",
    "D2_promotion": "forbidden_without_separate_closed_D1_arithmetic_DAG_replay",
    "global_H7": "fail_closed",
}
SEALS = dict(base.SEALS)


class QMetricLeafUnitError(ValueError):
    """The first bounded q leaf unit or one of its exact bindings changed."""


def _inside(root: Path, relative: str) -> Path:
    if not relative or "\\" in relative:
        raise QMetricLeafUnitError("q leaf unit path is not portable")
    path = (root / relative).resolve()
    if path != root and root not in path.parents:
        raise QMetricLeafUnitError("q leaf unit path escapes root")
    return path


def _copy(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, separators=(",", ":")))


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_config(value: Mapping[str, Any], path: Path) -> None:
    if _production_sha(path) != CONFIG_PRODUCTION_SHA256:
        raise QMetricLeafUnitError("q leaf unit config production bytes changed")
    if (
        value.get("schema_version") != CONFIG_SCHEMA
        or value.get("campaign_id") != CAMPAIGN_ID
        or value.get("output_path") != OUTPUT_PATH
        or value.get("self_bindings")
        != {"test_path": TEST_PATH, "test_sha256": TEST_PRODUCTION_SHA256}
        or value.get("unit_contract") != CONTRACT
        or value.get("policies") != POLICIES
        or value.get("seals") != SEALS
    ):
        raise QMetricLeafUnitError("q leaf unit config contract changed")


def _load_bundle(root: Path, bundle: Mapping[str, Any]) -> dict[str, Any]:
    try:
        value = base._load_bundle(root, bundle, artifact_hash=True)
    except base.P0MetricLowerLeafAuthorityError as error:
        raise QMetricLeafUnitError(str(error)) from error
    return value


def _load_inputs(root: Path, config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    checkpoint_value = _load_bundle(root, config["predecessor"])
    p3 = _load_bundle(root, config["candidate_authority"])
    if (
        checkpoint_value.get("decision")
        != "pass_q_metric_200_exact_tangent_scalars_leaf_composition_blocked"
        or checkpoint_value.get("gate_counts", {}).get("materialized_q_tangent_scalar_values")
        != 200
        or checkpoint_value.get("gate_counts", {}).get("materialized_leaf_roots_all_candidates")
        != 0
        or p3.get("decision")
        != "pass_p3_metric_10_column_nonlinear_tangents_316800_exact_leaf_roots_D2_blocked"
    ):
        raise QMetricLeafUnitError("q leaf unit predecessor boundary changed")
    return {"checkpoint": checkpoint_value, "p3": p3}


def _generic_packets(targets: tuple[str, ...]) -> tuple[dict[str, Any], ...]:
    old = base.P0_METRIC
    base._generic_packets.cache_clear()
    base.P0_METRIC = checkpoint.Q_METRIC
    try:
        packets = base._generic_packets(targets)
        selected = tuple(row for row in packets if row["derivative_atom"] == Q_ATOM)
        if len(selected) != 20:
            raise QMetricLeafUnitError("q leaf unit target-pair census changed")
        return selected
    finally:
        base._generic_packets.cache_clear()
        base.P0_METRIC = old


def _canonical(expression: sp.Expr) -> sp.Expr:
    return sp.expand_mul(expression)


@cache
def _materialize_unit(
    candidate_id: str,
    coefficient_blob: str,
    target_atoms: tuple[str, ...],
    tangent_blob: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    coefficient = json.loads(coefficient_blob)
    tangent = json.loads(tangent_blob)
    primitive_symbols = {
        str(symbol): symbol for symbol in checkpoint._coordinate_primitives()["symbols"]
    }
    tangent_symbols = {
        f"T_0_{kind}_{left}{right}": sp.Symbol(f"T_0_{kind}_{left}{right}", real=True)
        for kind in ("H", "G")
        for left, right in base.SYMMETRIC_PAIRS
    }
    alpha = _unspecialized_principal_blocks()["data"]["alpha"]
    locals_map = {**primitive_symbols, **tangent_symbols, "alpha": alpha}
    tangent_substitution = {}
    for label, value in {**tangent["delta_H"], **tangent["delta_G_upper"]}.items():
        tangent_substitution[tangent_symbols[f"T_0_{label}"]] = sp.sympify(
            value, locals=primitive_symbols
        )
    expressions: set[sp.Expr] = {sp.Integer(0)}
    staged = []
    for packet in _generic_packets(target_atoms):
        sparse_a = []
        for entry in packet["A_derivative_sparse_entries"]:
            expression = _canonical(
                sp.sympify(entry["value"], locals=locals_map)
                .subs({alpha: sp.sympify(coefficient["a10"])})
                .subs(tangent_substitution)
            )
            if expression != 0:
                expressions.add(expression)
                sparse_a.append({**entry, "expression": expression})
        sparse_chunk = []
        for entry in packet["source_chunk_column_derivative_sparse_entries"]:
            expression = _canonical(
                sp.sympify(entry["value"], locals=locals_map)
                .subs({alpha: sp.sympify(coefficient["a10"])})
                .subs(tangent_substitution)
            )
            if expression != 0:
                expressions.add(expression)
                sparse_chunk.append({**entry, "expression": expression})
        staged.append((packet, sparse_a, sparse_chunk))
    ordered = sorted(expressions, key=sp.srepr)
    roots = {expression: index for index, expression in enumerate(ordered)}
    nodes = [
        {
            "op": "exact_expand_mul_expression",
            "expression": str(expression),
            "srepr_sha256": hashlib.sha256(sp.srepr(expression).encode()).hexdigest(),
            "free_symbols": sorted(str(symbol) for symbol in expression.free_symbols),
        }
        for expression in ordered
    ]
    dag_body = {
        "schema_version": "sigma-q-leaf-unit-expression-DAG-1.0",
        "allowed_operation": "exact_expand_mul_expression",
        "factor_terms_used": False,
        "allowed_symbols": sorted(primitive_symbols),
        "node_count": len(nodes),
        "nodes": nodes,
    }
    dag = {**dag_body, "content_sha256": _sha(dag_body)}
    direction_packets = []
    for packet, sparse_a, sparse_chunk in staged:
        rooted_a = [
            {
                **{key: value for key, value in entry.items() if key != "expression"},
                "value": str(entry["expression"]),
                "arithmetic_root": roots[entry["expression"]],
            }
            for entry in sparse_a
        ]
        rooted_chunk = [
            {
                **{key: value for key, value in entry.items() if key != "expression"},
                "value": str(entry["expression"]),
                "arithmetic_root": roots[entry["expression"]],
            }
            for entry in sparse_chunk
        ]
        dense = [roots[sp.Integer(0)]] * 132
        for entry in rooted_a:
            dense[11 * entry["row"] + entry["column"]] = entry["arithmetic_root"]
        for entry in rooted_chunk:
            dense[121 + entry["row"]] = entry["arithmetic_root"]
        body = {
            "D1_target_atom": packet["D1_target_atom"],
            "derivative_atom": Q_ATOM,
            "derivative_coordinate_column": Q_COLUMN,
            "source_chunk_family": packet["source_chunk_family"],
            "source_chunk_input_column": packet["source_chunk_input_column"],
            "A_derivative_shape": [11, 11],
            "A_derivative_sparse_entries": rooted_a,
            "source_chunk_column_shape": [11],
            "source_chunk_column_derivative_sparse_entries": rooted_chunk,
            "zero_default_arithmetic_root": roots[sp.Integer(0)],
            "leaf_arithmetic_DAG_sha256": dag["content_sha256"],
            "total_leaf_derivative_roots": 132,
            "nonzero_leaf_derivative_roots": len(rooted_a) + len(rooted_chunk),
            "exact_zero_leaf_derivative_roots": 132 - len(rooted_a) - len(rooted_chunk),
            "dense_root_manifest_sha256": _sha(dense),
        }
        direction_packets.append({**body, "content_sha256": _sha(body)})
    nonzero = sum(row["nonzero_leaf_derivative_roots"] for row in direction_packets)
    unit_body = {
        "unit_index": UNIT_INDEX,
        "candidate_id": candidate_id,
        "candidate_ordinal": 0,
        "q_atom": Q_ATOM,
        "q_column": Q_COLUMN,
        "target_atoms": list(target_atoms),
        "target_direction_pairs": 20,
        "leaf_derivative_roots": 2640,
        "nonzero_leaf_derivative_roots": nonzero,
        "exact_zero_leaf_derivative_roots": 2640 - nonzero,
        "direction_packets": direction_packets,
        "leaf_arithmetic_DAG_sha256": dag["content_sha256"],
    }
    return {**unit_body, "content_sha256": _sha(unit_body)}, dag


def _expected_body(
    root: Path,
    config_path: Path,
    config: Mapping[str, Any],
    values: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    checkpoint_value = values["checkpoint"]
    p3 = values["p3"]
    candidate_id = checkpoint_value["resumable_leaf_composition_contract"]["candidate_ids"][0]
    coefficient_rows = {
        row["candidate_id"]: row["coefficients"] for row in p3["candidate_manifests"]
    }
    targets = tuple(checkpoint_value["resumable_leaf_composition_contract"]["target_atoms"])
    tangent = next(
        row
        for row in checkpoint_value["coordinate_tangent_packets"]
        if row["coordinate_atom"] == Q_ATOM
    )
    unit, dag = _materialize_unit(
        candidate_id,
        json.dumps(coefficient_rows[candidate_id], sort_keys=True, separators=(",", ":")),
        targets,
        json.dumps(tangent, sort_keys=True, separators=(",", ":")),
    )
    return {
        "schema_version": RESULT_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "decision": "pass_q_leaf_unit_000_2640_exact_roots_family_completion_blocked",
        "decision_counts": {"pass": 1, "blocked": 0, "reject": 0},
        "downstream_q_family_counts": {"pass": 0, "blocked": 12, "reject": 0},
        "first_blocker": "materialize_and_chain_units_001_through_119_before_q_family_promotion",
        "leaf_unit": unit,
        "leaf_arithmetic_DAG": dag,
        "gate_counts": {
            "completed_checkpoint_units": 1,
            "remaining_checkpoint_units": 119,
            "materialized_leaf_roots": 2640,
            "planned_leaf_roots_all_candidates": 316800,
            "remaining_leaf_roots": 314160,
            "nonzero_leaf_derivative_roots": unit["nonzero_leaf_derivative_roots"],
            "exact_zero_leaf_derivative_roots": unit["exact_zero_leaf_derivative_roots"],
            "unique_registered_coordinate_columns_after": 143,
            "remaining_unique_coordinate_columns_without_A_B_C_leaf_authority": 10,
            "registered_D2_entries_per_candidate_after": 5324,
        },
        "claim_seals": {
            "unit_000_exact_leaf_roots_materialized": True,
            "all_120_units_materialized": False,
            "complete_q_metric_leaf_family_registered": False,
            "all_153_unique_coordinate_leaf_authorities_registered": False,
            "D2_entry_count_advanced": False,
            "global_H7": False,
        },
        "source_bindings": {
            "predecessor": _copy(config["predecessor"]),
            "candidate_authority": _copy(config["candidate_authority"]),
            "source": {
                "path": SOURCE_PATH,
                "production_file_sha256": _production_sha(_inside(root, SOURCE_PATH)),
            },
            "config": {"path": CONFIG_PATH, "production_file_sha256": _production_sha(config_path)},
            "test": {
                "path": TEST_PATH,
                "production_file_sha256": _production_sha(_inside(root, TEST_PATH)),
            },
        },
        "data_seals": _copy(SEALS),
        "scope": "one exact candidate-by-q-atom leaf unit; remaining q family, D2, H7, and observations closed",
    }


def build_campaign(
    config_path: Path | str = CONFIG_PATH, *, root: Path | str | None = None
) -> dict[str, Any]:
    project_root = Path(root or Path.cwd()).resolve()
    path = _inside(project_root, str(config_path))
    config = json.loads(path.read_text(encoding="utf-8"))
    _validate_config(config, path)
    values = _load_inputs(project_root, config)
    body = _expected_body(project_root, path, config, values)
    return {**body, "content_sha256": _sha(body)}


def validate_campaign(value: Mapping[str, Any], *, root: Path | str | None = None) -> None:
    expected = build_campaign(root=Path(root or Path.cwd()).resolve())
    if value.get("content_sha256") != _content_sha(value) or value != expected:
        raise QMetricLeafUnitError("checked result changed")


def write_campaign(
    output_path: Path | str = OUTPUT_PATH, *, root: Path | str | None = None
) -> dict[str, Any]:
    project_root = Path(root or Path.cwd()).resolve()
    result = build_campaign(root=project_root)
    path = _inside(project_root, str(output_path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=OUTPUT_PATH)
    parser.add_argument("--validate-checked", action="store_true")
    args = parser.parse_args(argv)
    if args.validate_checked:
        validate_campaign(json.loads(Path(args.output).read_text(encoding="utf-8")))
    else:
        write_campaign(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
