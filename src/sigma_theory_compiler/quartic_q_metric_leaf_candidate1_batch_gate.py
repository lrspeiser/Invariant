"""Materialize candidate-one q leaf units 010 through 019 with one DAG."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from . import quartic_p0_metric_lower_abc_leaf_authority_gate as base
from . import quartic_q_metric_leaf_unit_000_gate as kernel
from .quartic_remaining_scalar_hessian_abc_leaf_authority_gate import (
    _content_sha,
    _production_sha,
    _sha,
)

SCHEMA = "sigma-quartic-q-metric-leaf-candidate1-batch-1.0"
CAMPAIGN_ID = "quartic-q-metric-leaf-candidate1-batch-001"
STEM = "quartic_q_metric_leaf_candidate1_batch_gate"
SLUG = "quartic-q-metric-leaf-candidate1-batch-gate"
CONFIG_PATH = f"configs/backgrounds/{STEM}.json"
SOURCE_PATH = f"src/sigma_theory_compiler/{STEM}.py"
TEST_PATH = f"tests/test_{STEM}.py"
OUTPUT_PATH = f"runs/physics-language/{SLUG}/campaign.json"
CONFIG_SHA = "7c8c424f76798fa02ba0ae3e1265176dd8dd83784ee70ff7303b10f1177d16d5"
TEST_SHA = "381829903b66ac341d779cec5b6ee2d6ad63159ea67178de4c31974853c0cdc8"


class QMetricLeafCandidate1BatchError(ValueError):
    """A candidate-one q leaf unit, chain, or authority changed."""


def _inside(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not relative or "\\" in relative or (path != root and root not in path.parents):
        raise QMetricLeafCandidate1BatchError("candidate1 batch path changed")
    return path


def _copy(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, separators=(",", ":")))


def _load(root: Path, bundle: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return base._load_bundle(root, bundle, artifact_hash=True)
    except base.P0MetricLowerLeafAuthorityError as error:
        raise QMetricLeafCandidate1BatchError(str(error)) from error


def _validate_config(config: Mapping[str, Any], path: Path) -> None:
    if (
        _production_sha(path) != CONFIG_SHA
        or config.get("schema_version") != SCHEMA
        or config.get("campaign_id") != CAMPAIGN_ID
        or config.get("output_path") != OUTPUT_PATH
        or config.get("self_bindings") != {"test_path": TEST_PATH, "test_sha256": TEST_SHA}
        or set(config)
        != {
            "schema_version",
            "campaign_id",
            "output_path",
            "self_bindings",
            "candidate0_batch",
            "q_checkpoint",
            "candidate_authority",
        }
    ):
        raise QMetricLeafCandidate1BatchError("candidate1 batch config changed")


def _dense(packet: Mapping[str, Any], zero: int) -> list[int]:
    roots = [zero] * 132
    for entry in packet["A_derivative_sparse_entries"]:
        roots[11 * entry["row"] + entry["column"]] = entry["arithmetic_root"]
    for entry in packet["source_chunk_column_derivative_sparse_entries"]:
        roots[121 + entry["row"]] = entry["arithmetic_root"]
    return roots


def _merge_local_dags(
    local: list[tuple[int, dict[str, Any], dict[str, Any]]],
) -> tuple[dict[str, Any], dict[str, int]]:
    by_expression = {node["expression"]: node for _, _, dag in local for node in dag["nodes"]}
    expressions = sorted(
        by_expression,
        key=lambda expression: (by_expression[expression]["srepr_sha256"], expression),
    )
    roots = {expression: index for index, expression in enumerate(expressions)}
    body = {
        "schema_version": "sigma-q-leaf-candidate1-shared-DAG-1.0",
        "allowed_operation": "exact_expand_mul_expression",
        "factor_terms_used": False,
        "allowed_symbols": local[0][2]["allowed_symbols"],
        "node_count": len(expressions),
        "nodes": [by_expression[expression] for expression in expressions],
    }
    return {**body, "content_sha256": _sha(body)}, roots


def _reroot_unit(
    unit: dict[str, Any],
    old_dag: Mapping[str, Any],
    shared_dag: Mapping[str, Any],
    roots: Mapping[str, int],
    *,
    field: int,
    predecessor: str,
) -> dict[str, Any]:
    old_expressions = {index: node["expression"] for index, node in enumerate(old_dag["nodes"])}
    for packet in unit["direction_packets"]:
        for group in (
            packet["A_derivative_sparse_entries"],
            packet["source_chunk_column_derivative_sparse_entries"],
        ):
            for entry in group:
                entry["arithmetic_root"] = roots[old_expressions[entry["arithmetic_root"]]]
        zero = roots[old_expressions[packet["zero_default_arithmetic_root"]]]
        packet.update(
            {
                "zero_default_arithmetic_root": zero,
                "derivative_atom": f"q[{field}]",
                "derivative_coordinate_column": field,
                "leaf_arithmetic_DAG_sha256": shared_dag["content_sha256"],
                "dense_root_manifest_sha256": _sha(_dense(packet, zero)),
            }
        )
        packet["content_sha256"] = _content_sha(packet)
    unit.update(
        {
            "unit_index": 10 + field,
            "candidate_ordinal": 1,
            "q_atom": f"q[{field}]",
            "q_column": field,
            "leaf_arithmetic_DAG_sha256": shared_dag["content_sha256"],
            "predecessor_unit_content_sha256": predecessor,
        }
    )
    unit["content_sha256"] = _content_sha(unit)
    return unit


def _expected(root: Path, config_path: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    candidate0 = _load(root, config["candidate0_batch"])
    checkpoint = _load(root, config["q_checkpoint"])
    candidates = _load(root, config["candidate_authority"])
    if (
        candidate0.get("decision")
        != "pass_candidate0_q_units_001_009_23760_exact_roots_family_blocked"
        or candidate0.get("gate_counts", {}).get("cumulative_completed_units") != 10
        or checkpoint.get("decision")
        != "pass_q_metric_200_exact_tangent_scalars_leaf_composition_blocked"
    ):
        raise QMetricLeafCandidate1BatchError("candidate1 predecessor changed")
    candidate_id = checkpoint["resumable_leaf_composition_contract"]["candidate_ids"][1]
    targets = tuple(checkpoint["resumable_leaf_composition_contract"]["target_atoms"])
    coefficient = next(
        row["coefficients"]
        for row in candidates["candidate_manifests"]
        if row["candidate_id"] == candidate_id
    )
    coefficient_blob = json.dumps(coefficient, sort_keys=True, separators=(",", ":"))
    local = []
    for field in range(10):
        tangent = next(
            row
            for row in checkpoint["coordinate_tangent_packets"]
            if row["coordinate_atom"] == f"q[{field}]"
        )
        unit, dag = kernel._materialize_unit(
            candidate_id,
            coefficient_blob,
            targets,
            json.dumps(tangent, sort_keys=True, separators=(",", ":")),
        )
        local.append((field, _copy(unit), _copy(dag)))
    shared_dag, roots = _merge_local_dags(local)
    units = []
    predecessor = candidate0["unit_chain_head_sha256"]
    for field, unit, old_dag in local:
        unit = _reroot_unit(
            unit,
            old_dag,
            shared_dag,
            roots,
            field=field,
            predecessor=predecessor,
        )
        predecessor = unit["content_sha256"]
        units.append(unit)
    nonzero = sum(unit["nonzero_leaf_derivative_roots"] for unit in units)
    body = {
        "schema_version": SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "decision": "pass_candidate1_q_units_010_019_26400_exact_roots_family_blocked",
        "units": units,
        "shared_leaf_arithmetic_DAG": shared_dag,
        "unit_chain_predecessor_sha256": candidate0["unit_chain_head_sha256"],
        "unit_chain_head_sha256": predecessor,
        "gate_counts": {
            "new_completed_units": 10,
            "cumulative_completed_units": 20,
            "remaining_units": 100,
            "new_leaf_roots": 26400,
            "cumulative_leaf_roots": 52800,
            "remaining_leaf_roots": 264000,
            "nonzero_leaf_roots": nonzero,
            "exact_zero_leaf_roots": 26400 - nonzero,
            "unique_registered_coordinate_columns_after": 143,
            "remaining_unique_coordinate_columns": 10,
            "registered_D2_entries_per_candidate_after": 5324,
        },
        "claim_seals": {
            "candidate1_all_q_units_complete": True,
            "all_120_units_complete": False,
            "complete_q_metric_leaf_family_registered": False,
            "all_153_unique_coordinate_leaf_authorities_registered": False,
            "D2_entry_count_advanced": False,
            "global_H7": False,
        },
        "source_bindings": {
            "candidate0_batch": _copy(config["candidate0_batch"]),
            "q_checkpoint": _copy(config["q_checkpoint"]),
            "candidate_authority": _copy(config["candidate_authority"]),
            "source": {
                "path": SOURCE_PATH,
                "production_file_sha256": _production_sha(_inside(root, SOURCE_PATH)),
            },
            "config": {
                "path": CONFIG_PATH,
                "production_file_sha256": _production_sha(config_path),
            },
            "test": {
                "path": TEST_PATH,
                "production_file_sha256": _production_sha(_inside(root, TEST_PATH)),
            },
        },
        "scope": ("candidate1 q units010-019 only; remaining candidates, q closure, D2, H7 closed"),
    }
    return body


def build_campaign(
    config_path: Path | str = CONFIG_PATH, *, root: Path | str | None = None
) -> dict[str, Any]:
    project_root = Path(root or Path.cwd()).resolve()
    path = _inside(project_root, str(config_path))
    config = json.loads(path.read_text(encoding="utf-8"))
    _validate_config(config, path)
    body = _expected(project_root, path, config)
    return {**body, "content_sha256": _sha(body)}


def validate_campaign(value: Mapping[str, Any], *, root: Path | str | None = None) -> None:
    if value.get("content_sha256") != _content_sha(value) or value != build_campaign(root=root):
        raise QMetricLeafCandidate1BatchError("checked result changed")


def write_campaign(
    output_path: Path | str = OUTPUT_PATH, *, root: Path | str | None = None
) -> dict[str, Any]:
    project_root = Path(root or Path.cwd()).resolve()
    value = build_campaign(root=project_root)
    path = _inside(project_root, str(output_path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=OUTPUT_PATH)
    args = parser.parse_args(argv)
    write_campaign(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
