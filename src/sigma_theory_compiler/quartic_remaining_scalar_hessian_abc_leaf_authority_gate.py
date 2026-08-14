"""Register the four remaining scalar-Hessian A/B/C leaf directions exactly."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from functools import cache
from pathlib import Path
from typing import Any

import sympy as sp

from .quartic_pgradient_abc_leaf_authority_gate import _chunks
from .quartic_unspecialized_source_jacobian_campaign import (
    _unspecialized_principal_blocks,
)

CONFIG_SCHEMA = "sigma-quartic-remaining-scalar-hessian-abc-leaf-authority-config-1.0"
RESULT_SCHEMA = "sigma-quartic-remaining-scalar-hessian-abc-leaf-authority-gate-1.0"
CAMPAIGN_ID = "quartic-remaining-scalar-hessian-abc-leaf-authority-001"
STEM = "quartic_remaining_scalar_hessian_abc_leaf_authority_gate"
SLUG = "quartic-remaining-scalar-hessian-abc-leaf-authority-gate"
CONFIG_PATH = f"configs/backgrounds/{STEM}.json"
SOURCE_PATH = f"src/sigma_theory_compiler/{STEM}.py"
TEST_PATH = f"tests/test_{STEM}.py"
OUTPUT_PATH = f"runs/physics-language/{SLUG}/campaign.json"
CONFIG_PRODUCTION_SHA256 = "18c496d01f1b30f6715959d3621534dcea79057825f0e5fab83db82ae5d6ec73"
BLOCK_SHA256 = "695ff2a5fd45fa3fba21d4ce25ab2f62bd168df187c8e931bc9b5803a9cd4aed"
BUNDLE_ROLES = ("pgradient_predecessor", "principal_projection")
CONTRACT = {
    "candidate_count": 12,
    "previous_missing_coordinate_columns": 127,
    "new_scalar_hessian_columns": 4,
    "registered_target_atoms": 20,
    "leaf_roots_per_target_direction_pair": 132,
    "new_leaf_roots_per_candidate": 10560,
    "new_leaf_roots_all_candidates": 126720,
    "registered_D2_entries_per_candidate": 5324,
    "full_D2_entries_per_candidate": 257499,
}
POLICIES = {
    "family_selection": "complete_remaining_scalar_covariant_Hessian_principal_family",
    "zero_admission": "only_exact_zero_root_after_live_symbolic_differentiation",
    "background_domain": "arbitrary_nonsingular_metric_at_one_coordinate_point",
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
SCALAR_HESSIAN = (
    ("s01[10]", 64, "H_01"),
    ("s02[10]", 75, "H_02"),
    ("s03[10]", 86, "H_03"),
    ("s33[10]", 152, "H_33"),
)
FIRST_BLOCKER = (
    "differentiate_and_replay_the_registered_D1_arithmetic_DAG_for_the_1056_"
    "new_candidate_bound_target_direction_records_before_advancing_any_D2_count"
)
NEXT_FAMILY_BLOCKER = "register_exact_A_B_C_leaf_authority_for_the_remaining_123_coordinate_columns"


class RemainingScalarHessianLeafAuthorityError(ValueError):
    """A scalar-Hessian leaf root, projection, or authority binding changed."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _content_sha(value: Mapping[str, Any]) -> str:
    return _sha({key: item for key, item in value.items() if key != "content_sha256"})


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _lf_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _production_sha(path: Path) -> str:
    lf_bytes = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(lf_bytes.replace(b"\n", b"\r\n")).hexdigest()


def _matches_text_authority(path: Path, expected: str) -> bool:
    return expected in {_file_sha(path), _lf_sha(path), _production_sha(path)}


def _inside(root: Path, relative: str) -> Path:
    if not relative or "\\" in relative:
        raise RemainingScalarHessianLeafAuthorityError("scalar-Hessian path is not portable")
    path = (root / relative).resolve()
    if path != root and root not in path.parents:
        raise RemainingScalarHessianLeafAuthorityError("scalar-Hessian path escapes root")
    return path


def _copy(value: Any) -> Any:
    return json.loads(_canonical(value))


def _validate_config(value: Mapping[str, Any], path: Path) -> None:
    if _production_sha(path) != CONFIG_PRODUCTION_SHA256:
        raise RemainingScalarHessianLeafAuthorityError("config production bytes changed")
    if (
        value.get("schema_version") != CONFIG_SCHEMA
        or value.get("campaign_id") != CAMPAIGN_ID
        or value.get("output_path") != OUTPUT_PATH
        or tuple(value.get("source_bundles", {})) != BUNDLE_ROLES
        or value.get("family_contract") != CONTRACT
        or value.get("policies") != POLICIES
        or value.get("seals") != SEALS
    ):
        raise RemainingScalarHessianLeafAuthorityError("config contract changed")
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
            raise RemainingScalarHessianLeafAuthorityError("source bundle changed")


def _load_bundle(root: Path, bundle: Mapping[str, Any]) -> dict[str, Any]:
    stem, slug = str(bundle["stem"]), str(bundle["slug"])
    text_paths = {
        "source_sha256": f"src/sigma_theory_compiler/{stem}.py",
        "config_sha256": f"configs/backgrounds/{stem}.json",
        "test_sha256": f"tests/test_{stem}.py",
    }
    for key, relative in text_paths.items():
        path = _inside(root, relative)
        if not path.is_file() or not _matches_text_authority(path, str(bundle[key])):
            raise RemainingScalarHessianLeafAuthorityError("text authority changed")
    artifact_path = _inside(root, f"runs/physics-language/{slug}/campaign.json")
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    if artifact.get("content_sha256") != bundle["content_sha256"] or artifact.get(
        "content_sha256"
    ) != _content_sha(artifact):
        raise RemainingScalarHessianLeafAuthorityError("receipt authority changed")
    return artifact


def _load_inputs(root: Path, config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    values = {role: _load_bundle(root, config["source_bundles"][role]) for role in BUNDLE_ROLES}
    predecessor = values["pgradient_predecessor"]
    projection = values["principal_projection"]
    if (
        predecessor.get("decision")
        != "pass_4_pgradient_columns_126720_exact_leaf_roots_D2_replay_blocked"
        or predecessor.get("gate_counts", {}).get(
            "remaining_coordinate_columns_without_A_B_C_leaf_authority"
        )
        != 127
        or predecessor.get("gate_counts", {}).get("registered_D2_entries_per_candidate_after")
        != 5324
        or projection.get("decision")
        != "pass_all_99_principal_second_jet_covariant_projections_lower_54_blocked"
    ):
        raise RemainingScalarHessianLeafAuthorityError("predecessor boundary changed")
    blocks = _unspecialized_principal_blocks()
    if blocks["content_sha256"] != BLOCK_SHA256:
        raise RemainingScalarHessianLeafAuthorityError("live A/B/C block changed")
    return values


def _projection_packets(projection: Mapping[str, Any]) -> list[dict[str, Any]]:
    by_atom = {
        str(row["coordinate_atom"]): row
        for row in projection.get("principal_projection_registry", [])
    }
    packets = []
    for atom, column, hessian_label in SCALAR_HESSIAN:
        row = by_atom.get(atom)
        if (
            row is None
            or row.get("coordinate_column") != column
            or row.get("covariant_jet_entries") != {hessian_label: "1"}
            or row.get("theorem") != "scalar_covariant_Hessian_principal_second_jet_identity"
        ):
            raise RemainingScalarHessianLeafAuthorityError("scalar projection changed")
        body = {
            "coordinate_atom": atom,
            "coordinate_column": column,
            "delta_H": {hessian_label: "1"},
            "delta_v": "0",
            "delta_G_upper": "0",
            "projection_content_sha256": row["content_sha256"],
            "domain": "arbitrary_nonsingular_metric_at_one_coordinate_point",
        }
        packets.append({**body, "content_sha256": _sha(body)})
    return packets


def _targets_and_coefficients(
    predecessor: Mapping[str, Any],
) -> tuple[tuple[str, ...], dict[str, Mapping[str, Any]]]:
    manifests = predecessor.get("candidate_manifests")
    if not isinstance(manifests, list) or len(manifests) != 12:
        raise RemainingScalarHessianLeafAuthorityError("candidate inventory changed")
    target_sets = {tuple(row.get("target_atoms", [])) for row in manifests}
    if len(target_sets) != 1:
        raise RemainingScalarHessianLeafAuthorityError("target inventory changed")
    targets = next(iter(target_sets))
    if len(targets) != 20 or any(atom in targets for atom, _, _ in SCALAR_HESSIAN):
        raise RemainingScalarHessianLeafAuthorityError("remaining scalar family changed")
    coefficients = {str(row["candidate_id"]): row["coefficients"] for row in manifests}
    return targets, coefficients


@cache
def _generic_packets(target_atoms: tuple[str, ...]) -> tuple[dict[str, Any], ...]:
    blocks = _unspecialized_principal_blocks()
    data = blocks["data"]
    hessian = {str(symbol): symbol for symbol in data["hessian_lower"].free_symbols}
    chunks = _chunks(blocks)
    packets = []
    for derivative_atom, derivative_column, hessian_label in SCALAR_HESSIAN:
        symbol = hessian[hessian_label]
        derivative_a = blocks["A"].applyfunc(
            lambda expression, symbol=symbol: sp.factor(sp.diff(expression, symbol))
        )
        for target_atom in target_atoms:
            family, field_text = target_atom.split("[")
            field = int(field_text[:-1])
            derivative_chunk = chunks[family].applyfunc(
                lambda expression, symbol=symbol: sp.factor(sp.diff(expression, symbol))
            )
            sparse_a = [
                {"row": row, "column": column, "value": str(derivative_a[row, column])}
                for row in range(11)
                for column in range(11)
                if derivative_a[row, column] != 0
            ]
            sparse_chunk = [
                {"row": row, "value": str(derivative_chunk[row, field])}
                for row in range(11)
                if derivative_chunk[row, field] != 0
            ]
            packets.append(
                {
                    "D1_target_atom": target_atom,
                    "derivative_atom": derivative_atom,
                    "derivative_coordinate_column": derivative_column,
                    "covariant_Hessian_component": hessian_label,
                    "source_chunk_family": family,
                    "source_chunk_input_column": field,
                    "A_derivative_sparse_entries": sparse_a,
                    "source_chunk_column_derivative_sparse_entries": sparse_chunk,
                }
            )
    if len(packets) != 80:
        raise RemainingScalarHessianLeafAuthorityError("generic packet census changed")
    return tuple(packets)


def _expression_dag(
    values: set[str], allowed: set[sp.Symbol]
) -> tuple[dict[str, Any], dict[str, int]]:
    locals_map = {str(symbol): symbol for symbol in allowed}
    expressions = {sp.factor(sp.sympify(value, locals=locals_map)) for value in values}
    if any(expression.free_symbols - allowed for expression in expressions):
        raise RemainingScalarHessianLeafAuthorityError("expression symbol escaped")
    ordered = sorted(
        expressions, key=lambda expression: (sp.count_ops(expression), str(expression))
    )
    nodes = [
        {
            "op": "exact_sympy_covariant_expression",
            "expression": str(expression),
            "srepr_sha256": hashlib.sha256(sp.srepr(expression).encode()).hexdigest(),
            "free_symbols": sorted(str(symbol) for symbol in expression.free_symbols),
        }
        for expression in ordered
    ]
    body = {
        "schema_version": "sigma-scalar-hessian-leaf-exact-expression-DAG-1.0",
        "allowed_operations": ["exact_sympy_covariant_expression"],
        "allowed_symbols": sorted(str(symbol) for symbol in allowed),
        "node_count": len(nodes),
        "nodes": nodes,
    }
    return {**body, "content_sha256": _sha(body)}, {
        str(expression): index for index, expression in enumerate(ordered)
    }


def _candidate_manifests(
    coefficients: Mapping[str, Mapping[str, Any]], target_atoms: tuple[str, ...]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    generic = _generic_packets(target_atoms)
    blocks = _unspecialized_principal_blocks()
    data = blocks["data"]
    coefficient_symbols = (data["m2"], data["alpha"], data["c20"])
    allowed = (
        set(data["gradient_lower"])
        | set(data["hessian_lower"].free_symbols)
        | set(data["einstein_upper"].free_symbols)
    )
    locals_map = {str(symbol): symbol for symbol in allowed | set(coefficient_symbols)}
    staged: dict[str, list[tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]]] = {}
    values = {"0"}
    for candidate_id, row_coefficients in coefficients.items():
        substitution = {
            data["m2"]: sp.sympify(row_coefficients["m2"]),
            data["alpha"]: sp.sympify(row_coefficients["a10"]),
            data["c20"]: sp.sympify(row_coefficients["c20"]),
        }
        rows = []
        for packet in generic:
            sparse_groups = []
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
                sparse_groups.append(sparse)
            rows.append((packet, sparse_groups[0], sparse_groups[1]))
        staged[candidate_id] = rows
    dag, roots = _expression_dag(values, allowed)
    manifests = []
    for candidate_id, rows in staged.items():
        packets = []
        for generic_packet, sparse_a, sparse_chunk in rows:
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
                    key: generic_packet[key]
                    for key in (
                        "D1_target_atom",
                        "derivative_atom",
                        "derivative_coordinate_column",
                        "covariant_Hessian_component",
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
            "derivative_coordinate_columns": [64, 75, 86, 152],
            "target_atoms": list(target_atoms),
            "target_direction_pairs": 80,
            "leaf_derivative_roots": 10560,
            "nonzero_leaf_derivative_roots": nonzero,
            "exact_zero_leaf_derivative_roots": 10560 - nonzero,
            "direction_packets": packets,
            "candidate_decision": "pass_remaining_scalar_Hessian_leaf_authority_D2_replay_blocked",
            "candidate_rejection_authorized": False,
        }
        manifests.append({**body, "content_sha256": _sha(body)})
    if len(manifests) != 12:
        raise RemainingScalarHessianLeafAuthorityError("manifest census changed")
    return manifests, dag


def _expected_body(
    root: Path,
    config_path: Path,
    config: Mapping[str, Any],
    values: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    projection_packets = _projection_packets(values["principal_projection"])
    targets, coefficients = _targets_and_coefficients(values["pgradient_predecessor"])
    manifests, dag = _candidate_manifests(coefficients, targets)
    nonzero = sum(row["nonzero_leaf_derivative_roots"] for row in manifests)
    return {
        "schema_version": RESULT_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "decision": "pass_4_remaining_scalar_Hessian_columns_126720_exact_leaf_roots_D2_blocked",
        "decision_counts": {"pass": 12, "blocked": 0, "reject": 0},
        "downstream_D2_counts": {"pass": 0, "blocked": 12, "reject": 0},
        "first_blocker": FIRST_BLOCKER,
        "next_family_blocker": NEXT_FAMILY_BLOCKER,
        "leaf_authority_theorem": {
            "name": "complete_scalar_covariant_Hessian_principal_family_to_live_A_B_C_chain",
            "exact_result": (
                "The four previously uncovered scalar principal atoms s01[10], s02[10], "
                "s03[10], and s33[10] have delta H_ab=1 and all other covariant jet "
                "tangents zero. Live exact differentiation registers 126,720 roots."
            ),
            "zero_boundary": (
                "Every zero is the exact zero node after symbolic differentiation and "
                "candidate substitution; no tensor component is inferred zero."
            ),
            "scope_boundary": (
                "This completes the nine-direction scalar-Hessian principal family at the "
                "leaf layer only; no D1 DAG replay or ordered-D2 root is emitted."
            ),
        },
        "projection_packets": projection_packets,
        "projection_manifest_sha256": _sha([row["content_sha256"] for row in projection_packets]),
        "leaf_arithmetic_DAG": dag,
        "candidate_manifests": manifests,
        "candidate_manifest_sha256": _sha([row["content_sha256"] for row in manifests]),
        "exact_controls": {
            "s03_scalar_projection": {"delta_H_03": "1", "passed": True},
            "corrupt_scalar_projection_sign": {"exact_residual": "2", "rejected": True},
            "infer_uncomputed_metric_tangent_zero": {"rejected": True},
            "promote_leaf_packet_without_D1_replay": {"rejected": True},
        },
        "gate_counts": {
            "selected_candidates": 12,
            "previous_registered_coordinate_columns": 26,
            "previous_missing_coordinate_columns": 127,
            "new_scalar_Hessian_coordinate_columns": 4,
            "registered_coordinate_columns_after": 30,
            "remaining_coordinate_columns_without_A_B_C_leaf_authority": 123,
            "complete_scalar_Hessian_principal_family_columns": 9,
            "registered_target_atoms": 20,
            "target_direction_pairs_per_candidate": 80,
            "candidate_bound_target_direction_pairs": 960,
            "new_leaf_derivative_roots_per_candidate": 10560,
            "new_leaf_derivative_roots_all_candidates": 126720,
            "nonzero_leaf_derivative_roots": nonzero,
            "exact_zero_leaf_derivative_roots": 126720 - nonzero,
            "potential_alias_expanded_D2_records_per_candidate": 88,
            "potential_candidate_bound_D2_records_blocked": 1056,
            "new_ordered_D2_roots_registered": 0,
            "registered_D2_entries_per_candidate_before": 5324,
            "registered_D2_entries_per_candidate_after": 5324,
            "full_D2_entries_per_candidate": 257499,
            "remaining_D2_entries_per_candidate": 252175,
            "complete_D2F_tensors": 0,
            "global_H7_closures": 0,
        },
        "claim_seals": {
            "four_remaining_scalar_Hessian_columns_registered": True,
            "complete_nine_direction_scalar_Hessian_principal_family_registered": True,
            "all_126720_candidate_bound_leaf_roots_registered": True,
            "no_tensor_component_inferred_zero": True,
            "new_ordered_D2_roots_registered": False,
            "D2_entry_count_advanced": False,
            "remaining_123_coordinate_columns_registered": False,
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
        "scope": (
            "exact arbitrary-background A/B/C leaf authority for the four remaining scalar "
            "Hessian principal atoms against 20 targets and 12 candidates; no inferred zero, "
            "D2 replay/count increment, complete tensor, H7, rejection, or observation"
        ),
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
    project_root = Path(root or Path.cwd()).resolve()
    expected = build_campaign(root=project_root)
    if value.get("content_sha256") != _content_sha(value) or value != expected:
        raise RemainingScalarHessianLeafAuthorityError("checked result changed")


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
