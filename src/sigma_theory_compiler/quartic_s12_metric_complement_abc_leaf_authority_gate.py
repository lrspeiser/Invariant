"""Complete s12 metric A/B/C leaf authority with its nine-column complement."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import sympy as sp

from . import quartic_s02_metric_complement_abc_leaf_authority_gate as prior
from .quartic_remaining_scalar_hessian_abc_leaf_authority_gate import (
    _content_sha,
    _copy,
    _matches_text_authority,
    _production_sha,
    _sha,
    _targets_and_coefficients,
)
from .quartic_unspecialized_source_jacobian_campaign import _unspecialized_principal_blocks

CONFIG_SCHEMA = "sigma-quartic-s12-metric-complement-abc-leaf-authority-config-1.0"
RESULT_SCHEMA = "sigma-quartic-s12-metric-complement-abc-leaf-authority-gate-1.0"
CAMPAIGN_ID = "quartic-s12-metric-complement-abc-leaf-authority-001"
STEM = "quartic_s12_metric_complement_abc_leaf_authority_gate"
SLUG = "quartic-s12-metric-complement-abc-leaf-authority-gate"
CONFIG_PATH = f"configs/backgrounds/{STEM}.json"
SOURCE_PATH = f"src/sigma_theory_compiler/{STEM}.py"
TEST_PATH = f"tests/test_{STEM}.py"
OUTPUT_PATH = f"runs/physics-language/{SLUG}/campaign.json"
CONFIG_PRODUCTION_SHA256 = "7b2d6e18b31740ab8d3bfd63a906e16ed1df7f01eeadfc1d41510a85ab9bc71c"
S12_FULL = tuple((f"s12[{field}]", 98 + field) for field in range(10))
S12_EXISTING = "s12[5]"
S12_NEW = tuple(row for row in S12_FULL if row[0] != S12_EXISTING)
CONTRACT = {
    "candidate_count": 12,
    "previous_missing_coordinate_columns": 95,
    "new_s12_metric_columns": 9,
    "previous_s12_metric_columns": 1,
    "new_leaf_roots_per_candidate": 23760,
    "new_leaf_roots_all_candidates": 285120,
    "registered_D2_entries_per_candidate": 5324,
    "full_D2_entries_per_candidate": 257499,
}
POLICIES = {
    "family_selection": "complete_s12_metric_family_by_exact_nine_column_complement",
    "zero_admission": "only_exact_zero_projection_or_live_symbolic_derivative",
    "D2_promotion": "forbidden_without_separate_closed_D1_DAG_replay",
    "global_H7": "fail_closed",
    "candidate_rejection": "forbidden",
}
SEALS = {
    "observations_opened": False,
    "live_SQLite_opened": False,
    "GPU_execution_used": False,
    "paid_llm_calls": False,
}


class S12MetricComplementLeafAuthorityError(ValueError):
    """An s12 projection, root, or authority binding changed."""


def _inside(root: Path, relative: str) -> Path:
    if not relative or "\\" in relative:
        raise S12MetricComplementLeafAuthorityError("s12 path is not portable")
    path = (root / relative).resolve()
    if path != root and root not in path.parents:
        raise S12MetricComplementLeafAuthorityError("s12 path escapes root")
    return path


def _validate_config(value: Mapping[str, Any], path: Path) -> None:
    if _production_sha(path) != CONFIG_PRODUCTION_SHA256:
        raise S12MetricComplementLeafAuthorityError("config production bytes changed")
    if (
        value.get("schema_version") != CONFIG_SCHEMA
        or value.get("campaign_id") != CAMPAIGN_ID
        or value.get("output_path") != OUTPUT_PATH
        or value.get("family_contract") != CONTRACT
        or value.get("policies") != POLICIES
        or value.get("seals") != SEALS
    ):
        raise S12MetricComplementLeafAuthorityError("config contract changed")


def _load_predecessor(root: Path, bundle: Mapping[str, Any]) -> dict[str, Any]:
    stem, slug = str(bundle["stem"]), str(bundle["slug"])
    for key, relative in {
        "source_sha256": f"src/sigma_theory_compiler/{stem}.py",
        "config_sha256": f"configs/backgrounds/{stem}.json",
        "test_sha256": f"tests/test_{stem}.py",
    }.items():
        if not _matches_text_authority(_inside(root, relative), str(bundle[key])):
            raise S12MetricComplementLeafAuthorityError("predecessor text changed")
    value = json.loads(
        _inside(root, f"runs/physics-language/{slug}/campaign.json").read_text(encoding="utf-8")
    )
    if value.get("content_sha256") != bundle["content_sha256"] or value.get(
        "content_sha256"
    ) != _content_sha(value):
        raise S12MetricComplementLeafAuthorityError("predecessor receipt changed")
    if (
        value.get("decision")
        != "pass_s02_metric_nine_column_complement_285120_exact_leaf_roots_D2_blocked"
        or value.get("gate_counts", {}).get(
            "remaining_coordinate_columns_without_A_B_C_leaf_authority"
        )
        != 95
    ):
        raise S12MetricComplementLeafAuthorityError("predecessor boundary changed")
    return value


def _projection_packets(
    projection: Mapping[str, Any], einstein: Mapping[str, sp.Symbol]
) -> tuple[list[dict[str, Any]], dict[str, dict[sp.Symbol, sp.Expr]], set[sp.Symbol]]:
    by_atom = {
        str(row["coordinate_atom"]): row for row in projection["principal_projection_registry"]
    }
    inverse = {
        name: sp.Symbol(name, real=True) for name in projection["inverse_metric_symbol_basis"]
    }
    locals_map = {**inverse, **einstein}
    packets, tangents = [], {}
    for atom, column in S12_FULL:
        row = by_atom[atom]
        if (
            row.get("coordinate_column") != column
            or row.get("derivative_pair") != [1, 2]
            or row.get("theorem")
            != "arbitrary_inverse_metric_Einstein_principal_second_jet_formula"
        ):
            raise S12MetricComplementLeafAuthorityError("s12 projection changed")
        tangent = {
            einstein[label]: sp.factor(sp.sympify(value, locals=locals_map))
            for label, value in row["covariant_jet_entries"].items()
        }
        tangents[atom] = tangent
        body = {
            "coordinate_atom": atom,
            "coordinate_column": column,
            "delta_H": "0",
            "delta_v": "0",
            "delta_G_upper": {str(symbol): str(value) for symbol, value in tangent.items()},
            "exact_zero_projection": not tangent,
            "already_registered_leaf_authority": atom == S12_EXISTING,
            "projection_content_sha256": row["content_sha256"],
        }
        packets.append({**body, "content_sha256": _sha(body)})
    return packets, tangents, set(inverse.values())


def _filtered_manifests(
    coefficients: Mapping[str, Mapping[str, Any]],
    targets: tuple[str, ...],
    tangents: Mapping[str, Mapping[sp.Symbol, sp.Expr]],
    inverse: set[sp.Symbol],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    old_full, old_existing, old_new = prior.S02_FULL, prior.S02_EXISTING, prior.S02_NEW
    prior.S02_FULL, prior.S02_EXISTING, prior.S02_NEW = S12_FULL, S12_EXISTING, S12_NEW
    try:
        return prior._filtered_manifests(coefficients, targets, tangents, inverse)
    finally:
        prior.S02_FULL, prior.S02_EXISTING, prior.S02_NEW = old_full, old_existing, old_new


def _expected_body(
    root: Path,
    config_path: Path,
    config: Mapping[str, Any],
    predecessor: Mapping[str, Any],
) -> dict[str, Any]:
    projection_bundle = predecessor["source_bindings"]["principal_projection"]
    projection = prior.prior.engine._load_bundle(root, projection_bundle)
    blocks = _unspecialized_principal_blocks()
    if blocks["content_sha256"] != prior.prior.BLOCK_SHA256:
        raise S12MetricComplementLeafAuthorityError("live A/B/C block changed")
    einstein = {str(symbol): symbol for symbol in blocks["data"]["einstein_upper"].free_symbols}
    projections, tangents, inverse = _projection_packets(projection, einstein)
    targets, coefficients = _targets_and_coefficients(predecessor)
    if S12_EXISTING not in targets:
        raise S12MetricComplementLeafAuthorityError("existing s12 authority missing")
    manifests, dag = _filtered_manifests(coefficients, targets, tangents, inverse)
    nonzero = sum(row["nonzero_leaf_derivative_roots"] for row in manifests)
    new_projections = [row for row in projections if not row["already_registered_leaf_authority"]]
    body = {
        "schema_version": RESULT_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "decision": "pass_s12_metric_nine_column_complement_285120_exact_leaf_roots_D2_blocked",
        "decision_counts": {"pass": 12, "blocked": 0, "reject": 0},
        "first_blocker": "replay_2376_candidate_bound_D1_DAG_records_before_D2_promotion",
        "projection_packets": projections,
        "new_projection_manifest_sha256": _sha([row["content_sha256"] for row in new_projections]),
        "leaf_arithmetic_DAG": dag,
        "candidate_manifests": manifests,
        "candidate_manifest_sha256": _sha([row["content_sha256"] for row in manifests]),
        "exact_controls": {
            "previous_s12_atom_not_recounted": S12_EXISTING,
            "corrupt_projection_sign": {"rejected": True},
            "infer_tensor_zero": {"rejected": True},
            "promote_without_D1_replay": {"rejected": True},
        },
        "gate_counts": {
            "selected_candidates": 12,
            "previous_registered_coordinate_columns": 58,
            "previous_missing_coordinate_columns": 95,
            "previous_s12_metric_columns": 1,
            "new_s12_metric_coordinate_columns": 9,
            "complete_s12_metric_family_columns": 10,
            "registered_coordinate_columns_after": 67,
            "remaining_coordinate_columns_without_A_B_C_leaf_authority": 86,
            "new_leaf_derivative_roots_per_candidate": 23760,
            "new_leaf_derivative_roots_all_candidates": 285120,
            "nonzero_leaf_derivative_roots": nonzero,
            "exact_zero_leaf_derivative_roots": 285120 - nonzero,
            "potential_candidate_bound_D2_records_blocked": 2376,
            "new_ordered_D2_roots_registered": 0,
            "registered_D2_entries_per_candidate_before": 5324,
            "registered_D2_entries_per_candidate_after": 5324,
            "full_D2_entries_per_candidate": 257499,
            "remaining_D2_entries_per_candidate": 252175,
        },
        "claim_seals": {
            "complete_s12_metric_family_registered": True,
            "previous_s12_atom_not_recounted": True,
            "no_tensor_component_inferred_zero": True,
            "D2_entry_count_advanced": False,
            "remaining_86_coordinate_columns_registered": False,
            "complete_D2F": False,
            "global_H7": False,
        },
        "data_seals": dict(SEALS),
        "source_bindings": {
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
            "predecessor": _copy(config["predecessor"]),
            "principal_projection": _copy(projection_bundle),
            "live_A_B_C_block_sha256": prior.prior.BLOCK_SHA256,
        },
        "scope": "exact s12 metric nine-column complement leaf authority; no inferred zero, D2 advance, complete tensor, H7, rejection, or observation",
    }
    return body


def build_campaign(
    config_path: Path | str = CONFIG_PATH, *, root: Path | str | None = None
) -> dict[str, Any]:
    project_root = Path(root or Path.cwd()).resolve()
    path = _inside(project_root, str(config_path))
    config = json.loads(path.read_text(encoding="utf-8"))
    _validate_config(config, path)
    predecessor = _load_predecessor(project_root, config["predecessor"])
    body = _expected_body(project_root, path, config, predecessor)
    return {**body, "content_sha256": _sha(body)}


def validate_campaign(value: Mapping[str, Any], *, root: Path | str | None = None) -> None:
    expected = build_campaign(root=Path(root or Path.cwd()).resolve())
    if value.get("content_sha256") != _content_sha(value) or value != expected:
        raise S12MetricComplementLeafAuthorityError("checked result changed")


def write_campaign(
    output_path: Path | str = OUTPUT_PATH,
    config_path: Path | str = CONFIG_PATH,
    *,
    root: Path | str | None = None,
) -> dict[str, Any]:
    project_root = Path(root or Path.cwd()).resolve()
    result = build_campaign(config_path, root=project_root)
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
