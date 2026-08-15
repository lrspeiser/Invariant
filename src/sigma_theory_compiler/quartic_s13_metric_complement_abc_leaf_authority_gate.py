"""Complete s13 metric A/B/C leaf authority with its nine-column complement."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import sympy as sp

from . import quartic_s12_metric_complement_abc_leaf_authority_gate as prior
from .quartic_remaining_scalar_hessian_abc_leaf_authority_gate import (
    _content_sha,
    _matches_text_authority,
    _production_sha,
    _sha,
)
from .quartic_unspecialized_source_jacobian_campaign import _unspecialized_principal_blocks

CONFIG_SCHEMA = "sigma-quartic-s13-metric-complement-abc-leaf-authority-config-1.0"
RESULT_SCHEMA = "sigma-quartic-s13-metric-complement-abc-leaf-authority-gate-1.0"
CAMPAIGN_ID = "quartic-s13-metric-complement-abc-leaf-authority-001"
STEM = "quartic_s13_metric_complement_abc_leaf_authority_gate"
SLUG = "quartic-s13-metric-complement-abc-leaf-authority-gate"
CONFIG_PATH = f"configs/backgrounds/{STEM}.json"
SOURCE_PATH = f"src/sigma_theory_compiler/{STEM}.py"
TEST_PATH = f"tests/test_{STEM}.py"
OUTPUT_PATH = f"runs/physics-language/{SLUG}/campaign.json"
CONFIG_PRODUCTION_SHA256 = "911d698f6d11349b6f8a59749f88a22323a1ce7b935f80bba95766da2f07f358"
S13_FULL = tuple((f"s13[{field}]", 109 + field) for field in range(10))
S13_EXISTING = "s13[6]"
S13_NEW = tuple(row for row in S13_FULL if row[0] != S13_EXISTING)
CONTRACT = {
    "candidate_count": 12,
    "previous_missing_coordinate_columns": 86,
    "new_s13_metric_columns": 9,
    "previous_s13_metric_columns": 1,
    "new_leaf_roots_per_candidate": 23760,
    "new_leaf_roots_all_candidates": 285120,
    "registered_D2_entries_per_candidate": 5324,
    "full_D2_entries_per_candidate": 257499,
}
POLICIES = {
    "family_selection": "complete_s13_metric_family_by_exact_nine_column_complement",
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


class S13MetricComplementLeafAuthorityError(ValueError):
    """An s13 projection, root, or authority binding changed."""


def _inside(root: Path, relative: str) -> Path:
    if not relative or "\\" in relative:
        raise S13MetricComplementLeafAuthorityError("s13 path is not portable")
    path = (root / relative).resolve()
    if path != root and root not in path.parents:
        raise S13MetricComplementLeafAuthorityError("s13 path escapes root")
    return path


def _validate_config(value: Mapping[str, Any], path: Path) -> None:
    if _production_sha(path) != CONFIG_PRODUCTION_SHA256:
        raise S13MetricComplementLeafAuthorityError("config production bytes changed")
    if (
        value.get("schema_version") != CONFIG_SCHEMA
        or value.get("campaign_id") != CAMPAIGN_ID
        or value.get("output_path") != OUTPUT_PATH
        or value.get("family_contract") != CONTRACT
        or value.get("policies") != POLICIES
        or value.get("seals") != SEALS
    ):
        raise S13MetricComplementLeafAuthorityError("config contract changed")


def _load_predecessor(root: Path, bundle: Mapping[str, Any]) -> dict[str, Any]:
    stem, slug = str(bundle["stem"]), str(bundle["slug"])
    for key, relative in {
        "source_sha256": f"src/sigma_theory_compiler/{stem}.py",
        "config_sha256": f"configs/backgrounds/{stem}.json",
        "test_sha256": f"tests/test_{stem}.py",
    }.items():
        if not _matches_text_authority(_inside(root, relative), str(bundle[key])):
            raise S13MetricComplementLeafAuthorityError("predecessor text changed")
    value = json.loads(
        _inside(root, f"runs/physics-language/{slug}/campaign.json").read_text(encoding="utf-8")
    )
    if value.get("content_sha256") != bundle["content_sha256"] or value.get(
        "content_sha256"
    ) != _content_sha(value):
        raise S13MetricComplementLeafAuthorityError("predecessor receipt changed")
    if (
        value.get("decision")
        != "pass_s12_metric_nine_column_complement_285120_exact_leaf_roots_D2_blocked"
        or value.get("gate_counts", {}).get(
            "remaining_coordinate_columns_without_A_B_C_leaf_authority"
        )
        != 86
    ):
        raise S13MetricComplementLeafAuthorityError("predecessor boundary changed")
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
    for atom, column in S13_FULL:
        row = by_atom[atom]
        if (
            row.get("coordinate_column") != column
            or row.get("derivative_pair") != [1, 3]
            or row.get("theorem")
            != "arbitrary_inverse_metric_Einstein_principal_second_jet_formula"
        ):
            raise S13MetricComplementLeafAuthorityError("s13 projection changed")
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
            "already_registered_leaf_authority": atom == S13_EXISTING,
            "projection_content_sha256": row["content_sha256"],
        }
        packets.append({**body, "content_sha256": _sha(body)})
    return packets, tangents, set(inverse.values())


def _base_body(
    root: Path,
    config_path: Path,
    config: Mapping[str, Any],
    predecessor: Mapping[str, Any],
) -> dict[str, Any]:
    old_full, old_existing, old_new = prior.S12_FULL, prior.S12_EXISTING, prior.S12_NEW
    old_projection = prior._projection_packets
    prior.S12_FULL, prior.S12_EXISTING, prior.S12_NEW = S13_FULL, S13_EXISTING, S13_NEW
    prior._projection_packets = _projection_packets
    try:
        body = prior._expected_body(root, config_path, config, predecessor)
    finally:
        prior.S12_FULL, prior.S12_EXISTING, prior.S12_NEW = old_full, old_existing, old_new
        prior._projection_packets = old_projection
    return body


def _expected_body(
    root: Path,
    config_path: Path,
    config: Mapping[str, Any],
    predecessor: Mapping[str, Any],
) -> dict[str, Any]:
    if _unspecialized_principal_blocks()["content_sha256"] != prior.prior.prior.BLOCK_SHA256:
        raise S13MetricComplementLeafAuthorityError("live A/B/C block changed")
    body = _base_body(root, config_path, config, predecessor)
    body["schema_version"] = RESULT_SCHEMA
    body["campaign_id"] = CAMPAIGN_ID
    body["decision"] = "pass_s13_metric_nine_column_complement_285120_exact_leaf_roots_D2_blocked"
    body["exact_controls"] = {
        "previous_s13_atom_not_recounted": S13_EXISTING,
        "corrupt_projection_sign": {"rejected": True},
        "infer_tensor_zero": {"rejected": True},
        "promote_without_D1_replay": {"rejected": True},
    }
    counts = body["gate_counts"]
    counts.pop("previous_s12_metric_columns")
    counts.pop("new_s12_metric_coordinate_columns")
    counts.pop("complete_s12_metric_family_columns")
    counts.update(
        {
            "previous_registered_coordinate_columns": 67,
            "previous_missing_coordinate_columns": 86,
            "previous_s13_metric_columns": 1,
            "new_s13_metric_coordinate_columns": 9,
            "complete_s13_metric_family_columns": 10,
            "registered_coordinate_columns_after": 76,
            "remaining_coordinate_columns_without_A_B_C_leaf_authority": 77,
        }
    )
    seals = body["claim_seals"]
    seals.pop("complete_s12_metric_family_registered")
    seals.pop("previous_s12_atom_not_recounted")
    seals.pop("remaining_86_coordinate_columns_registered")
    seals.update(
        {
            "complete_s13_metric_family_registered": True,
            "previous_s13_atom_not_recounted": True,
            "remaining_77_coordinate_columns_registered": False,
        }
    )
    for manifest in body["candidate_manifests"]:
        manifest["candidate_decision"] = (
            "pass_s13_metric_complement_leaf_authority_D2_replay_blocked"
        )
        manifest["content_sha256"] = _content_sha(manifest)
    body["candidate_manifest_sha256"] = _sha(
        [row["content_sha256"] for row in body["candidate_manifests"]]
    )
    body["source_bindings"]["source"] = {
        "path": SOURCE_PATH,
        "production_file_sha256": _production_sha(_inside(root, SOURCE_PATH)),
    }
    body["source_bindings"]["config"] = {
        "path": CONFIG_PATH,
        "production_file_sha256": _production_sha(config_path),
    }
    body["source_bindings"]["test"] = {
        "path": TEST_PATH,
        "production_file_sha256": _production_sha(_inside(root, TEST_PATH)),
    }
    body["scope"] = (
        "exact s13 metric nine-column complement leaf authority; no inferred zero, D2 "
        "advance, complete tensor, H7, rejection, or observation"
    )
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
        raise S13MetricComplementLeafAuthorityError("checked result changed")


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
