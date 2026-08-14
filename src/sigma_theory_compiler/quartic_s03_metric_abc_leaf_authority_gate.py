"""Register exact A/B/C leaf roots for the complete s03 metric family."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping, Sequence
from functools import cache
from pathlib import Path
from typing import Any

import sympy as sp

from .quartic_remaining_scalar_hessian_abc_leaf_authority_gate import (
    _chunks,
    _content_sha,
    _copy,
    _expression_dag,
    _matches_text_authority,
    _production_sha,
    _sha,
    _targets_and_coefficients,
)
from .quartic_unspecialized_source_jacobian_campaign import _unspecialized_principal_blocks

CONFIG_SCHEMA = "sigma-quartic-s03-metric-abc-leaf-authority-config-1.0"
RESULT_SCHEMA = "sigma-quartic-s03-metric-abc-leaf-authority-gate-1.0"
CAMPAIGN_ID = "quartic-s03-metric-abc-leaf-authority-001"
STEM = "quartic_s03_metric_abc_leaf_authority_gate"
SLUG = "quartic-s03-metric-abc-leaf-authority-gate"
CONFIG_PATH = f"configs/backgrounds/{STEM}.json"
SOURCE_PATH = f"src/sigma_theory_compiler/{STEM}.py"
TEST_PATH = f"tests/test_{STEM}.py"
OUTPUT_PATH = f"runs/physics-language/{SLUG}/campaign.json"
CONFIG_PRODUCTION_SHA256 = "e2febff5139fecf9b05cadcf2bfcb0145cd084cc709982431ef3a6012cb2296e"
BLOCK_SHA256 = "695ff2a5fd45fa3fba21d4ce25ab2f62bd168df187c8e931bc9b5803a9cd4aed"
BUNDLE_ROLES = ("leaf4_predecessor", "principal_projection")
CONTRACT = {
    "candidate_count": 12,
    "previous_missing_coordinate_columns": 123,
    "new_s03_metric_columns": 10,
    "registered_target_atoms": 20,
    "leaf_roots_per_target_direction_pair": 132,
    "new_leaf_roots_per_candidate": 26400,
    "new_leaf_roots_all_candidates": 316800,
    "registered_D2_entries_per_candidate": 5324,
    "full_D2_entries_per_candidate": 257499,
}
POLICIES = {
    "family_selection": "complete_s03_metric_second_jet_tensor_family",
    "zero_admission": "only_exact_zero_projection_or_live_symbolic_derivative",
    "background_domain": "arbitrary_symmetric_nonsingular_inverse_metric_at_one_point",
    "D2_promotion": "forbidden_without_separate_closed_D1_DAG_replay",
    "complete_D2F": "fail_closed",
    "global_H7": "fail_closed",
    "candidate_rejection": "forbidden",
}
SEALS = {
    "observations_opened": False,
    "live_SQLite_opened": False,
    "GPU_execution_used": False,
    "paid_llm_calls": False,
}
S03_METRIC = tuple((f"s03[{field}]", 76 + field) for field in range(10))
FIRST_BLOCKER = (
    "differentiate_and_replay_the_registered_D1_arithmetic_DAG_for_the_2640_"
    "new_candidate_bound_target_direction_records_before_advancing_any_D2_count"
)


class S03MetricLeafAuthorityError(ValueError):
    """An s03 projection, live derivative, or authority binding changed."""


def _inside(root: Path, relative: str) -> Path:
    if not relative or "\\" in relative:
        raise S03MetricLeafAuthorityError("s03 path is not portable")
    path = (root / relative).resolve()
    if path != root and root not in path.parents:
        raise S03MetricLeafAuthorityError("s03 path escapes root")
    return path


def _validate_config(value: Mapping[str, Any], path: Path) -> None:
    if _production_sha(path) != CONFIG_PRODUCTION_SHA256:
        raise S03MetricLeafAuthorityError("config production bytes changed")
    if (
        value.get("schema_version") != CONFIG_SCHEMA
        or value.get("campaign_id") != CAMPAIGN_ID
        or value.get("output_path") != OUTPUT_PATH
        or tuple(value.get("source_bundles", {})) != BUNDLE_ROLES
        or value.get("family_contract") != CONTRACT
        or value.get("policies") != POLICIES
        or value.get("seals") != SEALS
    ):
        raise S03MetricLeafAuthorityError("config contract changed")
    for bundle in value["source_bundles"].values():
        if set(bundle) != {
            "stem",
            "slug",
            "source_sha256",
            "config_sha256",
            "test_sha256",
            "content_sha256",
        } or any(
            not re.fullmatch(r"[0-9a-f]{64}", bundle[key])
            for key in bundle
            if key.endswith("sha256")
        ):
            raise S03MetricLeafAuthorityError("source bundle changed")


def _load_bundle(root: Path, bundle: Mapping[str, Any]) -> dict[str, Any]:
    stem, slug = str(bundle["stem"]), str(bundle["slug"])
    paths = {
        "source_sha256": f"src/sigma_theory_compiler/{stem}.py",
        "config_sha256": f"configs/backgrounds/{stem}.json",
        "test_sha256": f"tests/test_{stem}.py",
    }
    for key, relative in paths.items():
        path = _inside(root, relative)
        if not path.is_file() or not _matches_text_authority(path, str(bundle[key])):
            raise S03MetricLeafAuthorityError("text authority changed")
    artifact = json.loads(
        _inside(root, f"runs/physics-language/{slug}/campaign.json").read_text(encoding="utf-8")
    )
    if artifact.get("content_sha256") != bundle["content_sha256"] or artifact.get(
        "content_sha256"
    ) != _content_sha(artifact):
        raise S03MetricLeafAuthorityError("receipt authority changed")
    return artifact


def _load_inputs(root: Path, config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    values = {role: _load_bundle(root, config["source_bundles"][role]) for role in BUNDLE_ROLES}
    leaf4, projection = values["leaf4_predecessor"], values["principal_projection"]
    if (
        leaf4.get("decision")
        != "pass_4_remaining_scalar_Hessian_columns_126720_exact_leaf_roots_D2_blocked"
        or leaf4.get("gate_counts", {}).get(
            "remaining_coordinate_columns_without_A_B_C_leaf_authority"
        )
        != 123
        or leaf4.get("gate_counts", {}).get("registered_D2_entries_per_candidate_after") != 5324
        or projection.get("decision")
        != "pass_all_99_principal_second_jet_covariant_projections_lower_54_blocked"
    ):
        raise S03MetricLeafAuthorityError("predecessor boundary changed")
    if _unspecialized_principal_blocks()["content_sha256"] != BLOCK_SHA256:
        raise S03MetricLeafAuthorityError("live A/B/C block changed")
    return values


def _projection_packets(
    projection: Mapping[str, Any], einstein: Mapping[str, sp.Symbol]
) -> tuple[list[dict[str, Any]], dict[str, dict[sp.Symbol, sp.Expr]], set[sp.Symbol]]:
    by_atom = {
        str(row["coordinate_atom"]): row
        for row in projection.get("principal_projection_registry", [])
    }
    inverse = {
        name: sp.Symbol(name, real=True) for name in projection["inverse_metric_symbol_basis"]
    }
    locals_map = {**inverse, **einstein}
    packets, tangents = [], {}
    for atom, column in S03_METRIC:
        row = by_atom.get(atom)
        if (
            row is None
            or row.get("coordinate_column") != column
            or row.get("field_index") != column - 76
            or row.get("derivative_pair") != [0, 3]
            or row.get("theorem")
            != "arbitrary_inverse_metric_Einstein_principal_second_jet_formula"
        ):
            raise S03MetricLeafAuthorityError("s03 projection changed")
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
            "projection_content_sha256": row["content_sha256"],
            "domain": "arbitrary_symmetric_nonsingular_inverse_metric_at_one_point",
        }
        packets.append({**body, "content_sha256": _sha(body)})
    if len(packets) != 10 or sum(row["exact_zero_projection"] for row in packets) != 2:
        raise S03MetricLeafAuthorityError("s03 projection census changed")
    return packets, tangents, set(inverse.values())


@cache
def _generic_packets(
    target_atoms: tuple[str, ...], tangent_key: tuple[tuple[str, tuple[tuple[str, str], ...]], ...]
) -> tuple[dict[str, Any], ...]:
    blocks = _unspecialized_principal_blocks()
    data = blocks["data"]
    einstein = {str(symbol): symbol for symbol in data["einstein_upper"].free_symbols}
    locals_map = {
        name: sp.Symbol(name, real=True)
        for name in {
            item
            for _, rows in tangent_key
            for _, value in rows
            for item in re.findall(r"u\d\d", value)
        }
    }
    locals_map.update(einstein)
    tangents = {
        atom: {einstein[label]: sp.sympify(value, locals=locals_map) for label, value in rows}
        for atom, rows in tangent_key
    }
    chunks = _chunks(blocks)
    packets = []
    for derivative_atom, derivative_column in S03_METRIC:
        tangent = tangents[derivative_atom]

        def chain(expression: sp.Expr, local: Mapping[sp.Symbol, sp.Expr] = tangent) -> sp.Expr:
            return sp.factor(
                sum(
                    (sp.diff(expression, symbol) * value for symbol, value in local.items()),
                    sp.S.Zero,
                )
            )

        derivative_a = blocks["A"].applyfunc(chain)
        for target_atom in target_atoms:
            family, field_text = target_atom.split("[")
            field = int(field_text[:-1])
            derivative_chunk = chunks[family].applyfunc(chain)
            packets.append(
                {
                    "D1_target_atom": target_atom,
                    "derivative_atom": derivative_atom,
                    "derivative_coordinate_column": derivative_column,
                    "source_chunk_family": family,
                    "source_chunk_input_column": field,
                    "A_derivative_sparse_entries": [
                        {"row": row, "column": column, "value": str(derivative_a[row, column])}
                        for row in range(11)
                        for column in range(11)
                        if derivative_a[row, column] != 0
                    ],
                    "source_chunk_column_derivative_sparse_entries": [
                        {"row": row, "value": str(derivative_chunk[row, field])}
                        for row in range(11)
                        if derivative_chunk[row, field] != 0
                    ],
                }
            )
    if len(packets) != 200:
        raise S03MetricLeafAuthorityError("generic packet census changed")
    return tuple(packets)


def _candidate_manifests(
    coefficients: Mapping[str, Mapping[str, Any]],
    target_atoms: tuple[str, ...],
    tangents: Mapping[str, Mapping[sp.Symbol, sp.Expr]],
    inverse_symbols: set[sp.Symbol],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tangent_key = tuple(
        (atom, tuple(sorted((str(symbol), str(value)) for symbol, value in tangent.items())))
        for atom, tangent in tangents.items()
    )
    generic = _generic_packets(target_atoms, tangent_key)
    data = _unspecialized_principal_blocks()["data"]
    coefficient_symbols = {data["m2"], data["alpha"], data["c20"]}
    allowed = (
        set(data["gradient_lower"])
        | set(data["hessian_lower"].free_symbols)
        | set(data["einstein_upper"].free_symbols)
        | inverse_symbols
    )
    locals_map = {str(symbol): symbol for symbol in allowed | coefficient_symbols}
    staged, values = {}, {"0"}
    for candidate_id, row_coefficients in coefficients.items():
        substitution = {
            data["m2"]: sp.sympify(row_coefficients["m2"]),
            data["alpha"]: sp.sympify(row_coefficients["a10"]),
            data["c20"]: sp.sympify(row_coefficients["c20"]),
        }
        rows = []
        for packet in generic:
            groups = []
            for key in (
                "A_derivative_sparse_entries",
                "source_chunk_column_derivative_sparse_entries",
            ):
                sparse = []
                for entry in packet[key]:
                    value = str(
                        sp.factor(sp.sympify(entry["value"], locals=locals_map).subs(substitution))
                    )
                    if value != "0":
                        sparse.append({**entry, "value": value})
                        values.add(value)
                groups.append(sparse)
            rows.append((packet, groups[0], groups[1]))
        staged[candidate_id] = rows
    dag, roots = _expression_dag(values, allowed)
    manifests = []
    for candidate_id, rows in staged.items():
        packets = []
        for packet, sparse_a, sparse_chunk in rows:
            for entries in (sparse_a, sparse_chunk):
                for entry in entries:
                    entry["arithmetic_root"] = roots[entry["value"]]
            dense = [roots["0"]] * 132
            for entry in sparse_a:
                dense[11 * entry["row"] + entry["column"]] = entry["arithmetic_root"]
            for entry in sparse_chunk:
                dense[121 + entry["row"]] = entry["arithmetic_root"]
            nonzero = len(sparse_a) + len(sparse_chunk)
            body = {
                **{
                    key: packet[key]
                    for key in (
                        "D1_target_atom",
                        "derivative_atom",
                        "derivative_coordinate_column",
                        "source_chunk_family",
                        "source_chunk_input_column",
                    )
                },
                "A_derivative_shape": [11, 11],
                "A_derivative_sparse_entries": sparse_a,
                "source_chunk_column_shape": [11],
                "source_chunk_column_derivative_sparse_entries": sparse_chunk,
                "zero_default_arithmetic_root": roots["0"],
                "leaf_arithmetic_DAG_sha256": dag["content_sha256"],
                "total_leaf_derivative_roots": 132,
                "nonzero_leaf_derivative_roots": nonzero,
                "exact_zero_leaf_derivative_roots": 132 - nonzero,
                "dense_root_manifest_sha256": _sha(dense),
            }
            packets.append({**body, "content_sha256": _sha(body)})
        nonzero = sum(row["nonzero_leaf_derivative_roots"] for row in packets)
        body = {
            "candidate_id": candidate_id,
            "coefficients": _copy(coefficients[candidate_id]),
            "derivative_coordinate_columns": list(range(76, 86)),
            "target_atoms": list(target_atoms),
            "target_direction_pairs": 200,
            "leaf_derivative_roots": 26400,
            "nonzero_leaf_derivative_roots": nonzero,
            "exact_zero_leaf_derivative_roots": 26400 - nonzero,
            "direction_packets": packets,
            "candidate_decision": "pass_complete_s03_metric_leaf_authority_D2_replay_blocked",
            "candidate_rejection_authorized": False,
        }
        manifests.append({**body, "content_sha256": _sha(body)})
    return manifests, dag


def _expected_body(
    root: Path,
    config_path: Path,
    config: Mapping[str, Any],
    values: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    data = _unspecialized_principal_blocks()["data"]
    einstein = {str(symbol): symbol for symbol in data["einstein_upper"].free_symbols}
    projections, tangents, inverse = _projection_packets(values["principal_projection"], einstein)
    targets, coefficients = _targets_and_coefficients(values["leaf4_predecessor"])
    manifests, dag = _candidate_manifests(coefficients, targets, tangents, inverse)
    nonzero = sum(row["nonzero_leaf_derivative_roots"] for row in manifests)
    body = {
        "schema_version": RESULT_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "decision": "pass_complete_s03_metric_family_316800_exact_leaf_roots_D2_blocked",
        "decision_counts": {"pass": 12, "blocked": 0, "reject": 0},
        "downstream_D2_counts": {"pass": 0, "blocked": 12, "reject": 0},
        "first_blocker": FIRST_BLOCKER,
        "next_family_blocker": "register_exact_A_B_C_leaf_authority_for_the_remaining_113_coordinate_columns",
        "projection_packets": projections,
        "projection_manifest_sha256": _sha([row["content_sha256"] for row in projections]),
        "leaf_arithmetic_DAG": dag,
        "candidate_manifests": manifests,
        "candidate_manifest_sha256": _sha([row["content_sha256"] for row in manifests]),
        "exact_controls": {
            "exact_zero_projection_atoms": ["s03[0]", "s03[9]"],
            "corrupt_nonzero_Einstein_tangent_sign": {"rejected": True},
            "infer_uncomputed_tensor_zero": {"rejected": True},
            "promote_without_D1_replay": {"rejected": True},
        },
        "gate_counts": {
            "selected_candidates": 12,
            "previous_registered_coordinate_columns": 30,
            "previous_missing_coordinate_columns": 123,
            "new_s03_metric_coordinate_columns": 10,
            "registered_coordinate_columns_after": 40,
            "remaining_coordinate_columns_without_A_B_C_leaf_authority": 113,
            "exact_zero_covariant_projection_columns": 2,
            "registered_target_atoms": 20,
            "target_direction_pairs_per_candidate": 200,
            "candidate_bound_target_direction_pairs": 2400,
            "new_leaf_derivative_roots_per_candidate": 26400,
            "new_leaf_derivative_roots_all_candidates": 316800,
            "nonzero_leaf_derivative_roots": nonzero,
            "exact_zero_leaf_derivative_roots": 316800 - nonzero,
            "potential_alias_expanded_D2_records_per_candidate": 220,
            "potential_candidate_bound_D2_records_blocked": 2640,
            "new_ordered_D2_roots_registered": 0,
            "registered_D2_entries_per_candidate_before": 5324,
            "registered_D2_entries_per_candidate_after": 5324,
            "full_D2_entries_per_candidate": 257499,
            "remaining_D2_entries_per_candidate": 252175,
            "complete_D2F_tensors": 0,
            "global_H7_closures": 0,
        },
        "claim_seals": {
            "complete_s03_metric_family_registered": True,
            "all_316800_candidate_bound_leaf_roots_registered": True,
            "no_tensor_component_inferred_zero": True,
            "new_ordered_D2_roots_registered": False,
            "D2_entry_count_advanced": False,
            "remaining_113_coordinate_columns_registered": False,
            "complete_D2F": False,
            "global_H7": False,
            "candidate_theory_rejected": False,
        },
        "data_seals": dict(SEALS),
        "source_bindings": {
            "source": {
                "path": SOURCE_PATH,
                "production_file_sha256": _production_sha(_inside(root, SOURCE_PATH)),
            },
            "config": {"path": CONFIG_PATH, "production_file_sha256": _production_sha(config_path)},
            "test": {
                "path": TEST_PATH,
                "production_file_sha256": _production_sha(_inside(root, TEST_PATH)),
            },
            "evidence": _copy(config["source_bundles"]),
            "live_A_B_C_block_sha256": BLOCK_SHA256,
        },
        "scope": "exact arbitrary-background s03 metric-family A/B/C leaf authority; no inferred zeros, D2 advance, complete tensor, H7, rejection, or observation",
    }
    return body


def build_campaign(
    config_path: Path | str = CONFIG_PATH, *, root: Path | str | None = None
) -> dict[str, Any]:
    project_root = Path(root or Path.cwd()).resolve()
    path = _inside(project_root, str(config_path))
    config = json.loads(path.read_text(encoding="utf-8"))
    _validate_config(config, path)
    body = _expected_body(project_root, path, config, _load_inputs(project_root, config))
    return {**body, "content_sha256": _sha(body)}


def validate_campaign(value: Mapping[str, Any], *, root: Path | str | None = None) -> None:
    expected = build_campaign(root=Path(root or Path.cwd()).resolve())
    if value.get("content_sha256") != _content_sha(value) or value != expected:
        raise S03MetricLeafAuthorityError("checked result changed")


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
    parser.add_argument("--config", default=CONFIG_PATH)
    parser.add_argument("--output", default=OUTPUT_PATH)
    parser.add_argument("--validate-checked", action="store_true")
    args = parser.parse_args(argv)
    if args.validate_checked:
        validate_campaign(json.loads(Path(args.output).read_text(encoding="utf-8")))
    else:
        write_campaign(args.output, args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
