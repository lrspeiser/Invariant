"""Materialize candidate-zero q leaf units 001 through 009 with one DAG."""

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

SCHEMA = "sigma-quartic-q-metric-leaf-candidate0-batch-1.0"
CAMPAIGN_ID = "quartic-q-metric-leaf-candidate0-batch-001"
STEM = "quartic_q_metric_leaf_candidate0_batch_gate"
SLUG = "quartic-q-metric-leaf-candidate0-batch-gate"
CONFIG_PATH = f"configs/backgrounds/{STEM}.json"
SOURCE_PATH = f"src/sigma_theory_compiler/{STEM}.py"
TEST_PATH = f"tests/test_{STEM}.py"
OUTPUT_PATH = f"runs/physics-language/{SLUG}/campaign.json"
CONFIG_SHA = "45b7811264698213777ec604e59b1e95ba72096610caa815c85c9cb7284cebd7"
TEST_SHA = "6220f950756ff812cd71a2d1fa607cbed377e189a5e204cfbc6add60e62403bd"


class QMetricLeafBatchError(ValueError):
    pass


def _inside(root: Path, rel: str) -> Path:
    p = (root / rel).resolve()
    if not rel or "\\" in rel or (p != root and root not in p.parents):
        raise QMetricLeafBatchError("batch path changed")
    return p


def _copy(v: Any) -> Any:
    return json.loads(json.dumps(v, sort_keys=True, separators=(",", ":")))


def _load(root: Path, b: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return base._load_bundle(root, b, artifact_hash=True)
    except base.P0MetricLowerLeafAuthorityError as e:
        raise QMetricLeafBatchError(str(e)) from e


def _validate_config(c: Mapping[str, Any], p: Path) -> None:
    if (
        _production_sha(p) != CONFIG_SHA
        or c.get("schema_version") != SCHEMA
        or c.get("campaign_id") != CAMPAIGN_ID
        or c.get("output_path") != OUTPUT_PATH
        or c.get("self_bindings") != {"test_path": TEST_PATH, "test_sha256": TEST_SHA}
    ):
        raise QMetricLeafBatchError("batch config changed")


def _dense(packet: Mapping[str, Any], zero: int) -> list[int]:
    out = [zero] * 132
    for e in packet["A_derivative_sparse_entries"]:
        out[11 * e["row"] + e["column"]] = e["arithmetic_root"]
    for e in packet["source_chunk_column_derivative_sparse_entries"]:
        out[121 + e["row"]] = e["arithmetic_root"]
    return out


def _expected(root: Path, path: Path, c: Mapping[str, Any]) -> dict[str, Any]:
    u0 = _load(root, c["unit000"])
    q = _load(root, c["q_checkpoint"])
    p3 = _load(root, c["candidate_authority"])
    if (
        u0.get("decision") != "pass_q_leaf_unit_000_2640_exact_roots_family_completion_blocked"
        or q.get("decision") != "pass_q_metric_200_exact_tangent_scalars_leaf_composition_blocked"
    ):
        raise QMetricLeafBatchError("batch predecessor changed")
    cid = q["resumable_leaf_composition_contract"]["candidate_ids"][0]
    targets = tuple(q["resumable_leaf_composition_contract"]["target_atoms"])
    coeff = next(r["coefficients"] for r in p3["candidate_manifests"] if r["candidate_id"] == cid)
    coeff_blob = json.dumps(coeff, sort_keys=True, separators=(",", ":"))
    local = []
    for field in range(1, 10):
        tangent = next(
            r for r in q["coordinate_tangent_packets"] if r["coordinate_atom"] == f"q[{field}]"
        )
        unit, dag = kernel._materialize_unit(
            cid, coeff_blob, targets, json.dumps(tangent, sort_keys=True, separators=(",", ":"))
        )
        unit, dag = _copy(unit), _copy(dag)
        local.append((field, unit, dag))
    by_expr = {n["expression"]: n for _, _, d in local for n in d["nodes"]}
    expressions = sorted(by_expr, key=lambda x: (by_expr[x]["srepr_sha256"], x))
    roots = {x: i for i, x in enumerate(expressions)}
    nodes = [by_expr[x] for x in expressions]
    dag_body = {
        "schema_version": "sigma-q-leaf-candidate0-shared-DAG-1.0",
        "allowed_operation": "exact_expand_mul_expression",
        "factor_terms_used": False,
        "allowed_symbols": local[0][2]["allowed_symbols"],
        "node_count": len(nodes),
        "nodes": nodes,
    }
    dag = {**dag_body, "content_sha256": _sha(dag_body)}
    units = []
    predecessor = u0["content_sha256"]
    for field, unit, old in local:
        old_expr = {i: n["expression"] for i, n in enumerate(old["nodes"])}
        for packet in unit["direction_packets"]:
            for group in (
                packet["A_derivative_sparse_entries"],
                packet["source_chunk_column_derivative_sparse_entries"],
            ):
                for e in group:
                    e["arithmetic_root"] = roots[old_expr[e["arithmetic_root"]]]
            zero = roots[old_expr[packet["zero_default_arithmetic_root"]]]
            packet["zero_default_arithmetic_root"] = zero
            packet["derivative_atom"] = f"q[{field}]"
            packet["derivative_coordinate_column"] = field
            packet["leaf_arithmetic_DAG_sha256"] = dag["content_sha256"]
            packet["dense_root_manifest_sha256"] = _sha(_dense(packet, zero))
            packet["content_sha256"] = _content_sha(packet)
        unit.update(
            {
                "unit_index": field,
                "q_atom": f"q[{field}]",
                "q_column": field,
                "leaf_arithmetic_DAG_sha256": dag["content_sha256"],
                "predecessor_unit_content_sha256": predecessor,
            }
        )
        unit["content_sha256"] = _content_sha(unit)
        predecessor = unit["content_sha256"]
        units.append(unit)
    nz = sum(u["nonzero_leaf_derivative_roots"] for u in units)
    body = {
        "schema_version": SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "decision": "pass_candidate0_q_units_001_009_23760_exact_roots_family_blocked",
        "units": units,
        "shared_leaf_arithmetic_DAG": dag,
        "unit_chain_head_sha256": predecessor,
        "gate_counts": {
            "new_completed_units": 9,
            "cumulative_completed_units": 10,
            "remaining_units": 110,
            "new_leaf_roots": 23760,
            "cumulative_leaf_roots": 26400,
            "remaining_leaf_roots": 290400,
            "nonzero_leaf_roots": nz,
            "exact_zero_leaf_roots": 23760 - nz,
            "unique_registered_coordinate_columns_after": 143,
            "remaining_unique_coordinate_columns": 10,
            "registered_D2_entries_per_candidate_after": 5324,
        },
        "claim_seals": {
            "candidate0_all_q_units_complete": True,
            "all_120_units_complete": False,
            "complete_q_metric_leaf_family_registered": False,
            "all_153_unique_coordinate_leaf_authorities_registered": False,
            "D2_entry_count_advanced": False,
            "global_H7": False,
        },
        "source_bindings": {
            "unit000": _copy(c["unit000"]),
            "q_checkpoint": _copy(c["q_checkpoint"]),
            "candidate_authority": _copy(c["candidate_authority"]),
            "source": {
                "path": SOURCE_PATH,
                "production_file_sha256": _production_sha(_inside(root, SOURCE_PATH)),
            },
            "config": {"path": CONFIG_PATH, "production_file_sha256": _production_sha(path)},
            "test": {
                "path": TEST_PATH,
                "production_file_sha256": _production_sha(_inside(root, TEST_PATH)),
            },
        },
        "scope": "candidate0 q units001-009 only; remaining candidates, q closure, D2, H7 closed",
    }
    return body


def build_campaign(
    config_path: Path | str = CONFIG_PATH, *, root: Path | str | None = None
) -> dict[str, Any]:
    r = Path(root or Path.cwd()).resolve()
    p = _inside(r, str(config_path))
    c = json.loads(p.read_text(encoding="utf-8"))
    _validate_config(c, p)
    b = _expected(r, p, c)
    return {**b, "content_sha256": _sha(b)}


def validate_campaign(v: Mapping[str, Any], *, root: Path | str | None = None) -> None:
    if v.get("content_sha256") != _content_sha(v) or v != build_campaign(root=root):
        raise QMetricLeafBatchError("checked result changed")


def write_campaign(
    output_path: Path | str = OUTPUT_PATH, *, root: Path | str | None = None
) -> dict[str, Any]:
    r = Path(root or Path.cwd()).resolve()
    v = build_campaign(root=r)
    p = _inside(r, str(output_path))
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(v, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return v


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--output", default=OUTPUT_PATH)
    a = p.parse_args(argv)
    write_campaign(a.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
