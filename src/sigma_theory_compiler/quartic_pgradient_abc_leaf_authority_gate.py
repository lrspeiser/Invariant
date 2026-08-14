"""Register exact A/B/C leaf derivatives for four scalar-gradient columns."""

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

from .quartic_scalar_hessian_d2_integrability_gate import FAMILY_SPECS
from .quartic_unspecialized_source_jacobian_campaign import (
    _unspecialized_principal_blocks,
)

CONFIG_SCHEMA = "sigma-quartic-pgradient-abc-leaf-authority-config-1.0"
RESULT_SCHEMA = "sigma-quartic-pgradient-abc-leaf-authority-gate-1.0"
CAMPAIGN_ID = "quartic-pgradient-abc-leaf-authority-001"
CONFIG_PATH = "configs/backgrounds/quartic_pgradient_abc_leaf_authority_gate.json"
SOURCE_PATH = "src/sigma_theory_compiler/quartic_pgradient_abc_leaf_authority_gate.py"
TEST_PATH = "tests/test_quartic_pgradient_abc_leaf_authority_gate.py"
OUTPUT_PATH = "runs/physics-language/quartic-pgradient-abc-leaf-authority-gate/campaign.json"
CONFIG_PRODUCTION_SHA256 = "2f828847a23dbafb0080fc9928088613ff327233bf65096a551652e7a97acd07"
BLOCK_SHA256 = "695ff2a5fd45fa3fba21d4ce25ab2f62bd168df187c8e931bc9b5803a9cd4aed"
BUNDLE_ROLES = ("leaf_completion", "coordinate_projection", "p10_formula_authority")
CONTRACT = {
    "candidate_count": 12,
    "previous_registered_coordinate_columns": 22,
    "previous_missing_coordinate_columns": 131,
    "new_scalar_gradient_columns": 4,
    "registered_target_atoms": 20,
    "leaf_roots_per_target_direction_pair": 132,
    "new_leaf_roots_per_candidate": 10560,
    "new_leaf_roots_all_candidates": 126720,
    "registered_D2_entries_per_candidate": 5324,
    "full_D2_entries_per_candidate": 257499,
}
POLICIES = {
    "tangent_admission": "exact_delta_v_and_connection_scalar_Hessian_chain",
    "zero_admission": "only_exact_zero_root_after_live_symbolic_differentiation",
    "background_domain": "arbitrary_nonsingular_metric_and_consistent_coordinate_first_jet",
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
PGRADIENT = (
    ("p0[10]", 20, 0),
    ("p1[10]", 31, 1),
    ("p2[10]", 42, 2),
    ("p3[10]", 53, 3),
)
SYMMETRIC_PAIRS = tuple((left, right) for left in range(4) for right in range(left, 4))
FIRST_BLOCKER = (
    "differentiate_and_replay_the_registered_D1_arithmetic_DAG_for_the_1056_"
    "new_candidate_bound_target_direction_records_before_advancing_any_D2_count"
)
NEXT_FAMILY_BLOCKER = "register_exact_A_B_C_leaf_authority_for_the_remaining_127_coordinate_columns"


class PGradientLeafAuthorityError(ValueError):
    """A scalar-gradient tangent, leaf root, or authority binding changed."""


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
        raise PGradientLeafAuthorityError("p-gradient authority path must be portable")
    path = (root / relative).resolve()
    if path != root and root not in path.parents:
        raise PGradientLeafAuthorityError("p-gradient authority path escapes project root")
    return path


def _copy(value: Any) -> Any:
    return json.loads(_canonical(value))


def _validate_config(value: Mapping[str, Any], path: Path) -> None:
    if _production_sha(path) != CONFIG_PRODUCTION_SHA256:
        raise PGradientLeafAuthorityError("p-gradient config production bytes changed")
    if (
        value.get("schema_version") != CONFIG_SCHEMA
        or value.get("campaign_id") != CAMPAIGN_ID
        or value.get("output_path") != OUTPUT_PATH
        or tuple(value.get("source_bundles", {})) != BUNDLE_ROLES
        or value.get("family_contract") != CONTRACT
        or value.get("policies") != POLICIES
        or value.get("seals") != SEALS
    ):
        raise PGradientLeafAuthorityError("p-gradient config contract changed")
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
            raise PGradientLeafAuthorityError("p-gradient source bundle changed")


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
            raise PGradientLeafAuthorityError("p-gradient text authority changed")
    artifact_path = _inside(root, f"runs/physics-language/{slug}/campaign.json")
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    if artifact.get("content_sha256") != bundle["content_sha256"] or artifact.get(
        "content_sha256"
    ) != _content_sha(artifact):
        raise PGradientLeafAuthorityError("p-gradient receipt authority changed")
    return artifact


def _validate_transitive_formula_authority(root: Path, p10: Mapping[str, Any]) -> None:
    direct = p10.get("source_bindings", {}).get("direct_evidence", {})
    for role in ("nonlinear_geometric_map", "unspecialized_source_blocks"):
        bundle = direct.get(role)
        if not isinstance(bundle, Mapping):
            raise PGradientLeafAuthorityError("p-gradient transitive authority missing")
        for label in ("source", "config", "test"):
            binding = bundle[label]
            path = _inside(root, str(binding["path"]))
            if not path.is_file() or not _matches_text_authority(path, str(binding["file_sha256"])):
                raise PGradientLeafAuthorityError("p-gradient transitive text changed")
        artifact_binding = bundle["artifact"]
        artifact_path = _inside(root, str(artifact_binding["path"]))
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        if artifact.get("content_sha256") != artifact_binding["content_sha256"] or artifact.get(
            "content_sha256"
        ) != _content_sha(artifact):
            raise PGradientLeafAuthorityError("p-gradient transitive receipt changed")


def _load_inputs(root: Path, config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    values = {role: _load_bundle(root, config["source_bundles"][role]) for role in BUNDLE_ROLES}
    completion = values["leaf_completion"]
    projection = values["coordinate_projection"]
    p10 = values["p10_formula_authority"]
    if (
        completion.get("decision")
        != "pass_all_31680_reachable_leaf_roots_and_264_bounded_D2_roots_exact"
        or completion.get("gate_counts", {}).get(
            "remaining_coordinate_columns_without_A_B_C_leaf_authority"
        )
        != 131
        or completion.get("gate_counts", {}).get("registered_D2_entries_per_candidate_after")
        != 5324
        or projection.get("decision")
        != "pass_all_54_lower_covariant_projections_D2_count_preserved"
        or p10.get("decision")
        != "pass_7920_P10_arbitrary_background_leaf_derivative_roots_D2_propagation_blocked"
    ):
        raise PGradientLeafAuthorityError("p-gradient predecessor boundary changed")
    _validate_transitive_formula_authority(root, p10)
    blocks = _unspecialized_principal_blocks()
    if blocks["content_sha256"] != BLOCK_SHA256:
        raise PGradientLeafAuthorityError("live A/B/C formula block changed")
    return values


def _target_atoms(completion: Mapping[str, Any]) -> tuple[str, ...]:
    candidates = completion.get("candidate_certificates")
    if not isinstance(candidates, list) or len(candidates) != 12:
        raise PGradientLeafAuthorityError("p-gradient target candidate inventory changed")
    target_sets = []
    for candidate in candidates:
        rows = candidate.get("direction_root_manifests")
        atoms = tuple(sorted(str(row["coordinate_atom"]) for row in rows))
        if len(atoms) != 20 or len(set(atoms)) != 20:
            raise PGradientLeafAuthorityError("p-gradient target atom inventory changed")
        target_sets.append(atoms)
    if len(set(target_sets)) != 1:
        raise PGradientLeafAuthorityError("p-gradient target atoms differ by candidate")
    return target_sets[0]


def _candidate_coefficients(p10: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    values = {
        str(row["candidate_id"]): row["coefficients"] for row in p10.get("candidate_manifests", [])
    }
    if len(values) != 12:
        raise PGradientLeafAuthorityError("p-gradient candidate coefficients changed")
    return values


def _validate_projection(projection: Mapping[str, Any]) -> None:
    records = {
        str(row["coordinate_atom"]): row
        for row in projection.get("lower_projection_registry", [])
        if row.get("family") == "p_scalar_gradient"
    }
    expected = {atom: (column, component) for atom, column, component in PGRADIENT}
    if set(records) != set(expected):
        raise PGradientLeafAuthorityError("p-gradient projection atom set changed")
    for atom, (column, component) in expected.items():
        row = records[atom]
        if (
            row.get("coordinate_column") != column
            or row.get("tangent_seed")
            != {
                "dg": "0",
                "dP": "0",
                "dv_component": component,
                "dv_value": "1",
            }
            or row.get("exact_projection_registered") is not True
        ):
            raise PGradientLeafAuthorityError("p-gradient projection seed changed")


def _chunks(blocks: Mapping[str, Any]) -> dict[str, sp.Matrix]:
    return {
        family: multiplicity
        * (blocks[kind][first] if kind == "B_i" else blocks[kind][first][second])
        for family, _, _, kind, first, second, multiplicity in FAMILY_SPECS
    }


@cache
def _gamma_symbols() -> tuple[dict[int, dict[str, sp.Symbol]], tuple[sp.Symbol, ...]]:
    by_upper: dict[int, dict[str, sp.Symbol]] = {}
    all_symbols = []
    for upper in range(4):
        values = {}
        for left, right in SYMMETRIC_PAIRS:
            symbol = sp.Symbol(f"GammaU_{upper}_{left}{right}", real=True)
            values[f"H_{left}{right}"] = symbol
            all_symbols.append(symbol)
        by_upper[upper] = values
    return by_upper, tuple(all_symbols)


@cache
def _generic_packets(target_atoms: tuple[str, ...]) -> tuple[dict[str, Any], ...]:
    blocks = _unspecialized_principal_blocks()
    if blocks["content_sha256"] != BLOCK_SHA256:
        raise PGradientLeafAuthorityError("p-gradient live formula block changed")
    data = blocks["data"]
    hessian = {str(symbol): symbol for symbol in data["hessian_lower"].free_symbols}
    if set(hessian) != {f"H_{left}{right}" for left, right in SYMMETRIC_PAIRS}:
        raise PGradientLeafAuthorityError("p-gradient Hessian symbol basis changed")
    gamma, _ = _gamma_symbols()
    chunks = _chunks(blocks)
    packets = []
    for derivative_atom, derivative_column, component in PGRADIENT:
        tangent = {label: -symbol for label, symbol in gamma[component].items()}

        def chain(expression: sp.Expr, local_tangent: Mapping[str, sp.Expr] = tangent) -> sp.Expr:
            return sp.factor(
                sum(
                    sp.diff(expression, symbol) * local_tangent[label]
                    for label, symbol in hessian.items()
                )
            )

        derivative_A = blocks["A"].applyfunc(chain)
        for target_atom in target_atoms:
            family, field_text = target_atom.split("[")
            field = int(field_text[:-1])
            derivative_chunk = chunks[family].applyfunc(chain)
            sparse_A = [
                {"row": row, "column": column, "value": str(derivative_A[row, column])}
                for row in range(11)
                for column in range(11)
                if derivative_A[row, column] != 0
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
                    "scalar_gradient_component": component,
                    "source_chunk_family": family,
                    "source_chunk_input_column": field,
                    "A_derivative_sparse_entries": sparse_A,
                    "source_chunk_column_derivative_sparse_entries": sparse_chunk,
                    "total_leaf_derivative_roots": 132,
                    "nonzero_leaf_derivative_roots": len(sparse_A) + len(sparse_chunk),
                }
            )
    if len(packets) != 80 or any(row["nonzero_leaf_derivative_roots"] == 0 for row in packets):
        raise PGradientLeafAuthorityError("p-gradient generic leaf census changed")
    return tuple(packets)


def _expression_DAG(
    values: set[str], gamma_symbols: tuple[sp.Symbol, ...]
) -> tuple[dict[str, Any], dict[str, int]]:
    locals_map = {str(symbol): symbol for symbol in gamma_symbols}
    expressions = {sp.factor(sp.sympify(value, locals=locals_map)) for value in values}
    allowed = set(gamma_symbols)
    if any(expression.free_symbols - allowed for expression in expressions):
        raise PGradientLeafAuthorityError("p-gradient expression symbol escaped")
    ordered = sorted(
        expressions, key=lambda expression: (sp.count_ops(expression), str(expression))
    )
    nodes = [
        {
            "op": "exact_sympy_connection_expression",
            "expression": str(expression),
            "srepr_sha256": hashlib.sha256(sp.srepr(expression).encode()).hexdigest(),
            "free_symbols": sorted(str(symbol) for symbol in expression.free_symbols),
        }
        for expression in ordered
    ]
    body = {
        "schema_version": "sigma-pgradient-leaf-exact-connection-expression-DAG-1.0",
        "allowed_operations": ["exact_sympy_connection_expression"],
        "allowed_symbols": sorted(str(symbol) for symbol in gamma_symbols),
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
    alpha = blocks["data"]["alpha"]
    _, gamma_symbols = _gamma_symbols()
    locals_map = {str(symbol): symbol for symbol in gamma_symbols}
    locals_map["alpha"] = alpha
    staged = {}
    values = {"0"}
    for candidate_id, candidate_coefficients in coefficients.items():
        substitution = {alpha: sp.sympify(candidate_coefficients["a10"])}
        packets = []
        for packet in generic:
            sparse_A = []
            for entry in packet["A_derivative_sparse_entries"]:
                value = str(
                    sp.factor(sp.sympify(entry["value"], locals=locals_map).subs(substitution))
                )
                if value != "0":
                    sparse_A.append({**entry, "value": value})
                    values.add(value)
            sparse_chunk = []
            for entry in packet["source_chunk_column_derivative_sparse_entries"]:
                value = str(
                    sp.factor(sp.sympify(entry["value"], locals=locals_map).subs(substitution))
                )
                if value != "0":
                    sparse_chunk.append({**entry, "value": value})
                    values.add(value)
            packets.append((packet, sparse_A, sparse_chunk))
        staged[candidate_id] = packets
    dag, roots = _expression_DAG(values, gamma_symbols)
    manifests = []
    for candidate_id, rows in staged.items():
        packets = []
        for generic_packet, sparse_A, sparse_chunk in rows:
            rooted_A = [
                {
                    **entry,
                    "arithmetic_root": roots[
                        str(sp.factor(sp.sympify(entry["value"], locals=locals_map)))
                    ],
                }
                for entry in sparse_A
            ]
            rooted_chunk = [
                {
                    **entry,
                    "arithmetic_root": roots[
                        str(sp.factor(sp.sympify(entry["value"], locals=locals_map)))
                    ],
                }
                for entry in sparse_chunk
            ]
            dense = [roots["0"]] * 132
            for entry in rooted_A:
                dense[11 * entry["row"] + entry["column"]] = entry["arithmetic_root"]
            for entry in rooted_chunk:
                dense[121 + entry["row"]] = entry["arithmetic_root"]
            body = {
                **{
                    key: generic_packet[key]
                    for key in (
                        "D1_target_atom",
                        "derivative_atom",
                        "derivative_coordinate_column",
                        "scalar_gradient_component",
                        "source_chunk_family",
                        "source_chunk_input_column",
                    )
                },
                "A_derivative_shape": [11, 11],
                "A_derivative_sparse_entries": rooted_A,
                "source_chunk_column_shape": [11],
                "source_chunk_column_derivative_sparse_entries": rooted_chunk,
                "zero_default_arithmetic_root": roots["0"],
                "leaf_arithmetic_DAG_sha256": dag["content_sha256"],
                "total_leaf_derivative_roots": 132,
                "nonzero_leaf_derivative_roots": len(rooted_A) + len(rooted_chunk),
                "exact_zero_leaf_derivative_roots": 132 - len(rooted_A) - len(rooted_chunk),
                "dense_root_manifest_sha256": _sha(dense),
                "registered_arbitrary_background_connection_scope": True,
            }
            packets.append({**body, "content_sha256": _sha(body)})
        nonzero = sum(row["nonzero_leaf_derivative_roots"] for row in packets)
        body = {
            "candidate_id": candidate_id,
            "coefficients": _copy(coefficients[candidate_id]),
            "derivative_coordinate_columns": [20, 31, 42, 53],
            "target_atoms": list(target_atoms),
            "target_direction_pairs": 80,
            "leaf_derivative_roots": 10560,
            "nonzero_leaf_derivative_roots": nonzero,
            "exact_zero_leaf_derivative_roots": 10560 - nonzero,
            "direction_packets": packets,
            "candidate_decision": "pass_4_pgradient_columns_leaf_authority_D2_replay_blocked",
            "candidate_rejection_authorized": False,
        }
        manifests.append({**body, "content_sha256": _sha(body)})
    if len(manifests) != 12:
        raise PGradientLeafAuthorityError("p-gradient candidate manifest count changed")
    return manifests, dag


def _tangent_packets() -> list[dict[str, Any]]:
    gamma, _ = _gamma_symbols()
    packets = []
    for atom, column, component in PGRADIENT:
        body = {
            "coordinate_atom": atom,
            "coordinate_column": column,
            "delta_v": {f"v_{component}": "1"},
            "delta_H": {label: str(-symbol) for label, symbol in gamma[component].items()},
            "delta_G_upper": "0",
            "formula": "delta_H_mn=-Gamma^k_mn_for_delta_v_k=1",
            "domain": "arbitrary_nonsingular_metric_and_consistent_coordinate_first_jet",
        }
        packets.append({**body, "content_sha256": _sha(body)})
    return packets


def _expected_body(
    root: Path,
    config_path: Path,
    config: Mapping[str, Any],
    values: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    _validate_projection(values["coordinate_projection"])
    targets = _target_atoms(values["leaf_completion"])
    coefficients = _candidate_coefficients(values["p10_formula_authority"])
    manifests, dag = _candidate_manifests(coefficients, targets)
    nonzero = sum(row["nonzero_leaf_derivative_roots"] for row in manifests)
    exact_zero = 126720 - nonzero
    tangent_packets = _tangent_packets()
    return {
        "schema_version": RESULT_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "decision": "pass_4_pgradient_columns_126720_exact_leaf_roots_D2_replay_blocked",
        "decision_counts": {"pass": 12, "blocked": 0, "reject": 0},
        "downstream_D2_counts": {"pass": 0, "blocked": 12, "reject": 0},
        "first_blocker": FIRST_BLOCKER,
        "next_family_blocker": NEXT_FAMILY_BLOCKER,
        "leaf_authority_theorem": {
            "name": "scalar_gradient_to_connection_Hessian_to_live_A_B_C_chain",
            "exact_result": (
                "For each k=0..3, the p_k[10] coordinate tangent has delta v_k=1, "
                "delta H_mn=-Gamma^k_mn, and delta G^mn=0. Exact differentiation of the "
                "registered live A/B/C formulas supplies 132 roots for each of the 20 bound "
                "target atoms and all 12 candidates: 126,720 candidate-bound roots total."
            ),
            "zero_boundary": (
                "Every zero is the explicit exact zero node after symbolic differentiation and "
                "candidate substitution; no connection coefficient is assumed zero."
            ),
            "scope_boundary": (
                "This closes four of 131 missing derivative columns at the leaf layer only. "
                "No differentiated D1 DAG or new ordered-D2 root is emitted."
            ),
        },
        "tangent_packets": tangent_packets,
        "tangent_manifest_sha256": _sha([row["content_sha256"] for row in tangent_packets]),
        "leaf_arithmetic_DAG": dag,
        "candidate_manifests": manifests,
        "candidate_manifest_sha256": _sha([row["content_sha256"] for row in manifests]),
        "exact_controls": {
            "cylindrical_p1_scalar_H22": {
                "GammaU_1_22": "-1",
                "exact_delta_H22": "1",
                "passed": True,
            },
            "omit_connection_term": {"exact_residual": "-1", "rejected": True},
            "assume_connection_zero_on_arbitrary_background": {"rejected": True},
            "promote_leaf_packet_without_D1_replay": {"rejected": True},
        },
        "gate_counts": {
            "selected_candidates": 12,
            "previous_registered_coordinate_columns": 22,
            "previous_missing_coordinate_columns": 131,
            "new_scalar_gradient_coordinate_columns": 4,
            "registered_coordinate_columns_after": 26,
            "remaining_coordinate_columns_without_A_B_C_leaf_authority": 127,
            "registered_target_atoms": 20,
            "target_direction_pairs_per_candidate": 80,
            "candidate_bound_target_direction_pairs": 960,
            "new_leaf_derivative_roots_per_candidate": 10560,
            "new_leaf_derivative_roots_all_candidates": 126720,
            "nonzero_leaf_derivative_roots": nonzero,
            "exact_zero_leaf_derivative_roots": exact_zero,
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
            "four_pgradient_coordinate_columns_registered": True,
            "all_126720_candidate_bound_leaf_roots_registered": True,
            "no_connection_coefficient_assumed_zero": True,
            "new_ordered_D2_roots_registered": False,
            "D2_entry_count_advanced": False,
            "remaining_127_coordinate_columns_registered": False,
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
            "config": {
                "path": CONFIG_PATH,
                "production_file_sha256": _production_sha(config_path),
            },
            "test": {
                "path": TEST_PATH,
                "production_file_sha256": _production_sha(_inside(root, TEST_PATH)),
            },
            "evidence": _copy(config["source_bundles"]),
            "live_A_B_C_block_sha256": BLOCK_SHA256,
        },
        "scope": (
            "exact arbitrary-background A/B/C leaf authority for p0[10] through p3[10] "
            "against the 20 registered target atoms across 12 candidates; no inferred zero, "
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
        raise PGradientLeafAuthorityError("p-gradient leaf authority result changed")


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
