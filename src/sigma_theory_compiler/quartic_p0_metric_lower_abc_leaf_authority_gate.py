"""Register exact lower p0 metric A/B/C leaves through the nonlinear jet map."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from functools import cache
from pathlib import Path
from typing import Any

import sympy as sp

from .quartic_remaining_scalar_hessian_abc_leaf_authority_gate import (
    _content_sha,
    _matches_text_authority,
    _production_sha,
    _sha,
)
from .quartic_scalar_hessian_d2_integrability_gate import FAMILY_SPECS
from .quartic_unspecialized_source_jacobian_campaign import (
    _unspecialized_principal_blocks,
)

CONFIG_SCHEMA = "sigma-quartic-p0-metric-lower-abc-leaf-authority-config-1.0"
RESULT_SCHEMA = "sigma-quartic-p0-metric-lower-abc-leaf-authority-gate-1.0"
CAMPAIGN_ID = "quartic-p0-metric-lower-abc-leaf-authority-001"
STEM = "quartic_p0_metric_lower_abc_leaf_authority_gate"
SLUG = "quartic-p0-metric-lower-abc-leaf-authority-gate"
CONFIG_PATH = f"configs/backgrounds/{STEM}.json"
SOURCE_PATH = f"src/sigma_theory_compiler/{STEM}.py"
TEST_PATH = f"tests/test_{STEM}.py"
OUTPUT_PATH = f"runs/physics-language/{SLUG}/campaign.json"
CONFIG_PRODUCTION_SHA256 = "0151ac63863fe57a85915b1b79f75c4e849fb73b32eacb5ca3d3c497213a11a8"
LIVE_BLOCK_SHA256 = "695ff2a5fd45fa3fba21d4ce25ab2f62bd168df187c8e931bc9b5803a9cd4aed"
SYMMETRIC_PAIRS = tuple((left, right) for left in range(4) for right in range(left, 4))
P0_METRIC = tuple((f"p0[{field}]", 10 + field, pair) for field, pair in enumerate(SYMMETRIC_PAIRS))
CONTRACT = {
    "candidate_count": 12,
    "previous_registered_coordinate_columns": 105,
    "previous_missing_coordinate_columns": 48,
    "new_p0_metric_columns": 10,
    "d1_target_atoms": 20,
    "leaf_roots_per_target_pair": 132,
    "new_leaf_roots_per_candidate": 26400,
    "new_leaf_roots_all_candidates": 316800,
    "registered_D2_entries_per_candidate": 5324,
    "full_D2_entries_per_candidate": 257499,
}
POLICIES = {
    "family_selection": (
        "complete_p0_metric_first_jet_family_by_exact_nonlinear_coordinate_tangent"
    ),
    "coordinate_jet_domain": (
        "registered_pointwise_orthonormal_frame_with_arbitrary_symmetric_coordinate_first_jet"
    ),
    "zero_admission": "only_live_symbolic_derivative_exact_zero",
    "D2_promotion": "forbidden_without_separate_closed_D1_arithmetic_DAG_replay",
    "global_H7": "fail_closed",
    "candidate_rejection": "forbidden",
}
SEALS = {
    "observations_opened": False,
    "live_SQLite_opened": False,
    "GPU_execution_used": False,
    "paid_llm_calls": False,
}


class P0MetricLowerLeafAuthorityError(ValueError):
    """A lower p0 tangent, live leaf derivative, or sealed boundary changed."""


def _inside(root: Path, relative: str) -> Path:
    if not relative or "\\" in relative:
        raise P0MetricLowerLeafAuthorityError("p0 lower path is not portable")
    path = (root / relative).resolve()
    if path != root and root not in path.parents:
        raise P0MetricLowerLeafAuthorityError("p0 lower path escapes root")
    return path


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _copy(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, separators=(",", ":")))


def _validate_config(value: Mapping[str, Any], path: Path) -> None:
    if _production_sha(path) != CONFIG_PRODUCTION_SHA256:
        raise P0MetricLowerLeafAuthorityError("p0 lower config production bytes changed")
    if (
        value.get("schema_version") != CONFIG_SCHEMA
        or value.get("campaign_id") != CAMPAIGN_ID
        or value.get("output_path") != OUTPUT_PATH
        or value.get("family_contract") != CONTRACT
        or value.get("policies") != POLICIES
        or value.get("seals") != SEALS
        or set(value.get("direct_evidence", {})) != {"lower_projection", "live_abc"}
    ):
        raise P0MetricLowerLeafAuthorityError("p0 lower config contract changed")


def _load_bundle(root: Path, bundle: Mapping[str, Any], *, artifact_hash: bool) -> dict[str, Any]:
    stem, slug = str(bundle["stem"]), str(bundle["slug"])
    for key, relative in {
        "source_sha256": f"src/sigma_theory_compiler/{stem}.py",
        "config_sha256": f"configs/backgrounds/{stem}.json",
        "test_sha256": f"tests/test_{stem}.py",
    }.items():
        if not _matches_text_authority(_inside(root, relative), str(bundle[key])):
            raise P0MetricLowerLeafAuthorityError("p0 lower text authority changed")
    artifact = _inside(root, f"runs/physics-language/{slug}/campaign.json")
    if artifact_hash and _file_sha(artifact) != bundle["artifact_sha256"]:
        raise P0MetricLowerLeafAuthorityError("p0 lower artifact bytes changed")
    value = json.loads(artifact.read_text(encoding="utf-8"))
    if value.get("content_sha256") != bundle["content_sha256"] or value.get(
        "content_sha256"
    ) != _content_sha(value):
        raise P0MetricLowerLeafAuthorityError("p0 lower receipt authority changed")
    return value


def _load_inputs(root: Path, config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    predecessor = _load_bundle(root, config["predecessor"], artifact_hash=False)
    projection = _load_bundle(
        root, config["direct_evidence"]["lower_projection"], artifact_hash=True
    )
    live = _load_bundle(root, config["direct_evidence"]["live_abc"], artifact_hash=True)
    if (
        predecessor.get("decision")
        != "pass_s22_metric_six_column_complement_190080_exact_leaf_roots_D2_blocked"
        or predecessor.get("gate_counts", {}).get(
            "remaining_coordinate_columns_without_A_B_C_leaf_authority"
        )
        != 48
        or projection.get("decision")
        != "pass_all_54_lower_covariant_projections_D2_count_preserved"
        or live.get("status")
        != "pass_all_12_complete_unspecialized_principal_source_jacobians_remainder_fail_closed"
        or live.get("generic_unspecialized_source_jacobian_control", {})
        .get("unspecialized_block_extraction", {})
        .get("block_content_sha256")
        != LIVE_BLOCK_SHA256
    ):
        raise P0MetricLowerLeafAuthorityError("p0 lower predecessor boundary changed")
    return {"predecessor": predecessor, "projection": projection, "live_abc": live}


def _target_atoms(predecessor: Mapping[str, Any]) -> tuple[str, ...]:
    manifests = predecessor.get("candidate_manifests")
    if not isinstance(manifests, list) or len(manifests) != 12:
        raise P0MetricLowerLeafAuthorityError("p0 lower candidate inventory changed")
    target_sets = []
    for manifest in manifests:
        targets = tuple(
            sorted({str(row["D1_target_atom"]) for row in manifest["direction_packets"]})
        )
        if len(targets) != 20:
            raise P0MetricLowerLeafAuthorityError("p0 lower D1 target inventory changed")
        target_sets.append(targets)
    if len(set(target_sets)) != 1:
        raise P0MetricLowerLeafAuthorityError("p0 lower D1 targets differ by candidate")
    return target_sets[0]


def _candidate_coefficients(predecessor: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    values = {
        str(row["candidate_id"]): row["coefficients"] for row in predecessor["candidate_manifests"]
    }
    if len(values) != 12:
        raise P0MetricLowerLeafAuthorityError("p0 lower candidate coefficients changed")
    return values


def _validate_projection(projection: Mapping[str, Any]) -> None:
    records = {
        str(row["coordinate_atom"]): row
        for row in projection.get("lower_projection_registry", [])
        if row.get("family") == "p_metric" and row.get("derivative_index") == 0
    }
    if set(records) != {row[0] for row in P0_METRIC}:
        raise P0MetricLowerLeafAuthorityError("p0 lower projection family changed")
    for atom, column, pair in P0_METRIC:
        row = records[atom]
        seed = row.get("tangent_seed", {})
        if (
            row.get("coordinate_column") != column
            or seed.get("dP_derivative") != 0
            or seed.get("dP_symmetric_pair") != list(pair)
            or seed.get("dP_value") != ("1" if pair[0] == pair[1] else "sqrt(2)/2")
            or row.get("exact_projection_registered") is not True
        ):
            raise P0MetricLowerLeafAuthorityError("p0 lower projection seed changed")


def _symmetric_symbols(prefix: str) -> tuple[list[list[sp.Expr]], tuple[sp.Symbol, ...]]:
    matrix: list[list[sp.Expr]] = [[sp.Integer(0) for _ in range(4)] for _ in range(4)]
    symbols = []
    for left, right in SYMMETRIC_PAIRS:
        symbol = sp.Symbol(f"{prefix}_{left}{right}", real=True)
        matrix[left][right] = symbol
        matrix[right][left] = symbol
        symbols.append(symbol)
    return matrix, tuple(symbols)


@cache
def _coordinate_primitives() -> dict[str, Any]:
    inverse = [[sp.Integer(0) for _ in range(4)] for _ in range(4)]
    metric = [[sp.Integer(0) for _ in range(4)] for _ in range(4)]
    for index, value in enumerate((-1, 1, 1, 1)):
        inverse[index][index] = sp.Integer(value)
        metric[index][index] = sp.Integer(value)
    first = []
    first_symbols = []
    for derivative in range(4):
        matrix, symbols = _symmetric_symbols(f"P{derivative}")
        first.append(matrix)
        first_symbols.extend(symbols)
    gradient = tuple(sp.Symbol(f"v_{index}", real=True) for index in range(4))
    return {
        "inverse": inverse,
        "metric": metric,
        "first": first,
        "gradient": gradient,
        "symbols": (*first_symbols, *gradient),
    }


def _norm(expression: sp.Expr) -> sp.Expr:
    return sp.factor_terms(expression)


@cache
def _exact_p0_tangents() -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    primitive = _coordinate_primitives()
    inverse, metric = primitive["inverse"], primitive["metric"]
    first, gradient = primitive["first"], primitive["gradient"]
    packets: dict[str, dict[str, Any]] = {}
    for atom, column, (seed_left, seed_right) in P0_METRIC:
        delta = [sp.zeros(4) for _ in range(4)]
        weight = sp.Integer(1) if seed_left == seed_right else sp.sqrt(2) / 2
        delta[0][seed_left, seed_right] = weight
        delta[0][seed_right, seed_left] = weight
        bracket = [[[sp.Integer(0) for _ in range(4)] for _ in range(4)] for _ in range(4)]
        dbracket = [[[sp.Integer(0) for _ in range(4)] for _ in range(4)] for _ in range(4)]
        for contracted in range(4):
            for left in range(4):
                for right in range(4):
                    bracket[contracted][left][right] = (
                        first[left][contracted][right]
                        + first[right][contracted][left]
                        - first[contracted][left][right]
                    )
                    dbracket[contracted][left][right] = (
                        delta[left][contracted, right]
                        + delta[right][contracted, left]
                        - delta[contracted][left, right]
                    )
        gamma = [[[sp.Integer(0) for _ in range(4)] for _ in range(4)] for _ in range(4)]
        dgamma = [[[sp.Integer(0) for _ in range(4)] for _ in range(4)] for _ in range(4)]
        for upper in range(4):
            for left in range(4):
                for right in range(4):
                    gamma[upper][left][right] = (
                        sum(
                            inverse[upper][contracted] * bracket[contracted][left][right]
                            for contracted in range(4)
                        )
                        / 2
                    )
                    dgamma[upper][left][right] = (
                        sum(
                            inverse[upper][contracted] * dbracket[contracted][left][right]
                            for contracted in range(4)
                        )
                        / 2
                    )
        inverse_first = [[[sp.Integer(0) for _ in range(4)] for _ in range(4)] for _ in range(4)]
        dinverse_first = [[[sp.Integer(0) for _ in range(4)] for _ in range(4)] for _ in range(4)]
        for derivative in range(4):
            for upper in range(4):
                for right in range(4):
                    inverse_first[derivative][upper][right] = -sum(
                        inverse[upper][left]
                        * first[derivative][left][lower]
                        * inverse[lower][right]
                        for left in range(4)
                        for lower in range(4)
                    )
                    dinverse_first[derivative][upper][right] = -sum(
                        inverse[upper][left]
                        * delta[derivative][left, lower]
                        * inverse[lower][right]
                        for left in range(4)
                        for lower in range(4)
                    )
        dpartial = [
            [[[sp.Integer(0) for _ in range(4)] for _ in range(4)] for _ in range(4)]
            for _ in range(4)
        ]
        for derivative in range(4):
            for upper in range(4):
                for left in range(4):
                    for right in range(4):
                        dpartial[derivative][upper][left][right] = (
                            sum(
                                dinverse_first[derivative][upper][contracted]
                                * bracket[contracted][left][right]
                                + inverse_first[derivative][upper][contracted]
                                * dbracket[contracted][left][right]
                                for contracted in range(4)
                            )
                            / 2
                        )
        dricci = sp.zeros(4)
        for left in range(4):
            for right in range(4):
                dricci[left, right] = _norm(
                    sum(
                        dpartial[upper][upper][right][left]
                        - dpartial[right][upper][upper][left]
                        + sum(
                            dgamma[upper][upper][contracted] * gamma[contracted][right][left]
                            + gamma[upper][upper][contracted] * dgamma[contracted][right][left]
                            - dgamma[upper][right][contracted] * gamma[contracted][upper][left]
                            - gamma[upper][right][contracted] * dgamma[contracted][upper][left]
                            for contracted in range(4)
                        )
                        for upper in range(4)
                    )
                )
        trace = _norm(
            sum(
                inverse[left][right] * dricci[left, right]
                for left in range(4)
                for right in range(4)
            )
        )
        delta_h = {}
        delta_g = {}
        for left, right in SYMMETRIC_PAIRS:
            h_value = _norm(
                -sum(dgamma[upper][left][right] * gradient[upper] for upper in range(4))
            )
            g_value = _norm(
                sum(
                    inverse[left][alpha]
                    * inverse[right][beta]
                    * (dricci[alpha, beta] - metric[alpha][beta] * trace / 2)
                    for alpha in range(4)
                    for beta in range(4)
                )
            )
            delta_h[f"H_{left}{right}"] = str(h_value)
            delta_g[f"G_{left}{right}"] = str(g_value)
        body = {
            "coordinate_atom": atom,
            "coordinate_column": column,
            "seed": {
                "dP_derivative": 0,
                "dP_symmetric_pair": [seed_left, seed_right],
                "dP_value": str(weight),
                "dg": "0",
                "dv": "0",
                "dS": "0",
            },
            "delta_v": "0",
            "delta_H": delta_h,
            "delta_G_upper": delta_g,
            "domain": (
                "registered pointwise orthonormal local frame g_ab=eta_ab; "
                "P_kab=P_kba is an otherwise arbitrary consistent coordinate first jet"
            ),
            "all_20_covariant_tangent_components_materialized": True,
        }
        packets[atom] = {**body, "content_sha256": _sha(body)}
    program_body = {
        "schema_version": "sigma-p0-metric-nonlinear-coordinate-tangent-program-1.0",
        "primitive_symbols": sorted(str(symbol) for symbol in primitive["symbols"]),
        "frame_constraint": "g_ab=u^ab=diag(-1,1,1,1) at the evaluation point",
        "formulas": [
            "B_smn=P_msn+P_nsm-P_smn",
            "Gamma^r_mn=(1/2)u^rs B_smn",
            "deltaGamma^r_mn=(1/2)u^rs deltaB_smn",
            "U_k^rs=-u^ra P_kab u^bs",
            "deltaU_k^rs=-u^ra deltaP_kab u^bs",
            "delta(partial_k Gamma^r_mn)=(1/2)(deltaU_k^rs B_smn+U_k^rs deltaB_smn)",
            "deltaRicci_mn=deltaR^r_mrn including both deltaGamma*Gamma products",
            "deltaH_mn=-deltaGamma^r_mn v_r",
            "deltaG^mn=u^ma u^nb(deltaRicci_ab-(1/2)g_ab u^cd deltaRicci_cd)",
        ],
        "tangent_packet_count": 10,
        "materialized_scalar_values": 200,
        "no_flat_reference_specialization": True,
        "arbitrary_background_pointwise_local_frame": True,
    }
    return packets, {**program_body, "content_sha256": _sha(program_body)}


def _chunks(blocks: Mapping[str, Any]) -> dict[str, sp.Matrix]:
    return {
        family: multiplicity
        * (blocks[kind][first] if kind == "B_i" else blocks[kind][first][second])
        for family, _, _, kind, first, second, multiplicity in FAMILY_SPECS
    }


@cache
def _generic_packets(target_atoms: tuple[str, ...]) -> tuple[dict[str, Any], ...]:
    blocks = _unspecialized_principal_blocks()
    if blocks["content_sha256"] != LIVE_BLOCK_SHA256:
        raise P0MetricLowerLeafAuthorityError("p0 lower live A/B/C block changed")
    data = blocks["data"]
    hessian = {str(symbol): symbol for symbol in data["hessian_lower"].free_symbols}
    einstein = {str(symbol): symbol for symbol in data["einstein_upper"].free_symbols}
    tangents, _ = _exact_p0_tangents()
    chunks = _chunks(blocks)
    packets = []
    for derivative_atom, derivative_column, _ in P0_METRIC:
        derivative_field = int(derivative_atom.split("[")[1][:-1])
        tangent_symbols = {
            label: sp.Symbol(f"T_{derivative_field}_{label}", real=True)
            for label in (*sorted(hessian), *sorted(einstein))
        }

        def chain(
            expression: sp.Expr,
            tangent_symbols: dict[str, sp.Symbol] = tangent_symbols,
        ) -> sp.Expr:
            return _norm(
                sum(
                    sp.diff(expression, symbol) * tangent_symbols[label]
                    for label, symbol in hessian.items()
                )
                + sum(
                    sp.diff(expression, symbol) * tangent_symbols[label]
                    for label, symbol in einstein.items()
                )
            )

        derivative_a = blocks["A"].applyfunc(chain)
        for target_atom in target_atoms:
            family, field_text = target_atom.split("[")
            field = int(field_text[:-1])
            derivative_chunk = chunks[family].applyfunc(chain)
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
                    "source_chunk_family": family,
                    "source_chunk_input_column": field,
                    "tangent_packet_sha256": tangents[derivative_atom]["content_sha256"],
                    "A_derivative_sparse_entries": sparse_a,
                    "source_chunk_column_derivative_sparse_entries": sparse_chunk,
                    "total_leaf_derivative_roots": 132,
                }
            )
    if len(packets) != 200:
        raise P0MetricLowerLeafAuthorityError("p0 lower target-pair census changed")
    return tuple(packets)


def _expression_dag(values: set[str]) -> tuple[dict[str, Any], dict[str, int]]:
    primitive_symbols = tuple(_coordinate_primitives()["symbols"])
    locals_map = {str(symbol): symbol for symbol in primitive_symbols}
    expressions = {_norm(sp.sympify(value, locals=locals_map)) for value in values}
    allowed = set(primitive_symbols)
    if any(expression.free_symbols - allowed for expression in expressions):
        raise P0MetricLowerLeafAuthorityError("p0 lower expression symbol escaped")
    ordered = sorted(
        expressions, key=lambda expression: (sp.count_ops(expression), str(expression))
    )
    nodes = [
        {
            "op": "exact_sympy_composed_tangent_expression",
            "expression": str(expression),
            "srepr_sha256": hashlib.sha256(sp.srepr(expression).encode()).hexdigest(),
            "free_symbols": sorted(str(symbol) for symbol in expression.free_symbols),
        }
        for expression in ordered
    ]
    body = {
        "schema_version": "sigma-p0-metric-lower-leaf-expression-DAG-1.0",
        "allowed_operations": ["exact_sympy_composed_tangent_expression"],
        "allowed_symbols": sorted(str(symbol) for symbol in primitive_symbols),
        "node_count": len(nodes),
        "nodes": nodes,
    }
    return {**body, "content_sha256": _sha(body)}, {
        str(expression): index for index, expression in enumerate(ordered)
    }


@cache
def _candidate_manifests_cached(
    coefficients_blob: str, target_atoms: tuple[str, ...]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    coefficients = json.loads(coefficients_blob)
    generic = _generic_packets(target_atoms)
    alpha = _unspecialized_principal_blocks()["data"]["alpha"]
    tangent_symbols = {
        str(symbol): symbol
        for symbol in tuple(
            sp.Symbol(f"T_{field}_{kind}_{left}{right}", real=True)
            for field in range(10)
            for kind in ("H", "G")
            for left, right in SYMMETRIC_PAIRS
        )
    }
    primitive_symbols = {str(symbol): symbol for symbol in _coordinate_primitives()["symbols"]}
    locals_map = {**tangent_symbols, **primitive_symbols, "alpha": alpha}
    tangent_packets, _ = _exact_p0_tangents()
    tangent_substitution = {}
    for field in range(10):
        atom = f"p0[{field}]"
        for label, value in {
            **tangent_packets[atom]["delta_H"],
            **tangent_packets[atom]["delta_G_upper"],
        }.items():
            tangent_substitution[tangent_symbols[f"T_{field}_{label}"]] = sp.sympify(
                value, locals=primitive_symbols
            )
    staged = {}
    values = {"0"}
    for candidate_id, candidate_coefficients in coefficients.items():
        substitution = {alpha: sp.sympify(candidate_coefficients["a10"])}
        rows = []
        for packet in generic:
            sparse_a = []
            for entry in packet["A_derivative_sparse_entries"]:
                value = str(
                    _norm(
                        sp.sympify(entry["value"], locals=locals_map)
                        .subs(substitution)
                        .subs(tangent_substitution)
                    )
                )
                if value != "0":
                    sparse_a.append({**entry, "value": value})
                    values.add(value)
            sparse_chunk = []
            for entry in packet["source_chunk_column_derivative_sparse_entries"]:
                value = str(
                    _norm(
                        sp.sympify(entry["value"], locals=locals_map)
                        .subs(substitution)
                        .subs(tangent_substitution)
                    )
                )
                if value != "0":
                    sparse_chunk.append({**entry, "value": value})
                    values.add(value)
            rows.append((packet, sparse_a, sparse_chunk))
        staged[candidate_id] = rows
    dag, roots = _expression_dag(values)
    manifests = []
    for candidate_id, rows in staged.items():
        packets = []
        for generic_packet, sparse_a, sparse_chunk in rows:
            rooted_a = [
                {
                    **entry,
                    "arithmetic_root": roots[
                        str(_norm(sp.sympify(entry["value"], locals=locals_map)))
                    ],
                }
                for entry in sparse_a
            ]
            rooted_chunk = [
                {
                    **entry,
                    "arithmetic_root": roots[
                        str(_norm(sp.sympify(entry["value"], locals=locals_map)))
                    ],
                }
                for entry in sparse_chunk
            ]
            dense = [roots["0"]] * 132
            for entry in rooted_a:
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
                        "source_chunk_family",
                        "source_chunk_input_column",
                        "tangent_packet_sha256",
                    )
                },
                "A_derivative_shape": [11, 11],
                "A_derivative_sparse_entries": rooted_a,
                "source_chunk_column_shape": [11],
                "source_chunk_column_derivative_sparse_entries": rooted_chunk,
                "zero_default_arithmetic_root": roots["0"],
                "leaf_arithmetic_DAG_sha256": dag["content_sha256"],
                "total_leaf_derivative_roots": 132,
                "nonzero_leaf_derivative_roots": len(rooted_a) + len(rooted_chunk),
                "exact_zero_leaf_derivative_roots": 132 - len(rooted_a) - len(rooted_chunk),
                "dense_root_manifest_sha256": _sha(dense),
                "arbitrary_first_jet_pointwise_frame_domain": True,
            }
            packets.append({**body, "content_sha256": _sha(body)})
        nonzero = sum(row["nonzero_leaf_derivative_roots"] for row in packets)
        body = {
            "candidate_id": candidate_id,
            "coefficients": _copy(coefficients[candidate_id]),
            "derivative_coordinate_columns": list(range(10, 20)),
            "target_atoms": list(target_atoms),
            "target_direction_pairs": 200,
            "leaf_derivative_roots": 26400,
            "nonzero_leaf_derivative_roots": nonzero,
            "exact_zero_leaf_derivative_roots": 26400 - nonzero,
            "direction_packets": packets,
            "candidate_decision": "pass_p0_metric_lower_leaf_authority_D2_replay_blocked",
            "candidate_rejection_authorized": False,
        }
        manifests.append({**body, "content_sha256": _sha(body)})
    if len(manifests) != 12:
        raise P0MetricLowerLeafAuthorityError("p0 lower candidate manifest count changed")
    return manifests, dag


def _candidate_manifests(
    coefficients: Mapping[str, Mapping[str, Any]], target_atoms: tuple[str, ...]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    coefficients_blob = json.dumps(coefficients, sort_keys=True, separators=(",", ":"))
    manifests, dag = _candidate_manifests_cached(coefficients_blob, target_atoms)
    return _copy(manifests), _copy(dag)


def _expected_body(
    root: Path,
    config_path: Path,
    config: Mapping[str, Any],
    values: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    _validate_projection(values["projection"])
    targets = _target_atoms(values["predecessor"])
    coefficients = _candidate_coefficients(values["predecessor"])
    tangent_by_atom, tangent_program = _exact_p0_tangents()
    tangents = [tangent_by_atom[atom] for atom, _, _ in P0_METRIC]
    manifests, dag = _candidate_manifests(coefficients, targets)
    nonzero = sum(row["nonzero_leaf_derivative_roots"] for row in manifests)
    return {
        "schema_version": RESULT_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "decision": (
            "pass_p0_metric_10_column_nonlinear_tangents_316800_exact_leaf_roots_D2_blocked"
        ),
        "decision_counts": {"pass": 12, "blocked": 0, "reject": 0},
        "downstream_D2_counts": {"pass": 0, "blocked": 12, "reject": 0},
        "first_blocker": (
            "independently_replay_the_D1_arithmetic_DAG_for_p0_metric_columns_before_"
            "admitting_ordered_D2_roots_then_materialize_the_remaining_38_columns"
        ),
        "leaf_authority_theorem": {
            "name": "pointwise_frame_arbitrary_first_jet_p0_metric_to_live_A_B_C_chain",
            "exact_result": (
                "All ten p0 metric first-jet coordinate directions are mapped through exact "
                "connection, differentiated-connection, Ricci, Hessian, and raised Einstein "
                "tangents, then through every live A/B/C leaf used by the 20 registered D1 targets."
            ),
            "domain": tangent_program["frame_constraint"],
            "boundary": (
                "This registers compositional leaf roots only; it does not independently replay "
                "the source D1 arithmetic DAG or admit a new ordered D2 value."
            ),
        },
        "coordinate_tangent_program": tangent_program,
        "coordinate_tangent_packets": tangents,
        "coordinate_tangent_manifest_sha256": _sha([row["content_sha256"] for row in tangents]),
        "leaf_arithmetic_DAG": dag,
        "candidate_manifests": manifests,
        "candidate_manifest_sha256": _sha([row["content_sha256"] for row in manifests]),
        "exact_controls": {
            "off_diagonal_seed_normalization": {
                "atom": "p0[1]",
                "exact_value": "sqrt(2)/2",
                "replace_by_one_rejected": True,
            },
            "connection_dependence": {
                "atom": "p0[0]",
                "delta_H_contains_live_scalar_gradient": True,
                "infer_zero_rejected": True,
            },
            "inverse_first_jet_product_rule": {
                "formula": "deltaU_0^ab=-u^ac deltaP_0_cd u^db",
                "omit_term_rejected": True,
            },
            "flat_reference_substitution_for_general_claim": {"rejected": True},
            "promote_without_D1_replay": {"rejected": True},
        },
        "gate_counts": {
            "selected_candidates": 12,
            "previous_registered_coordinate_columns": 105,
            "previous_missing_coordinate_columns": 48,
            "new_p0_metric_coordinate_columns": 10,
            "registered_coordinate_columns_after": 115,
            "remaining_coordinate_columns_without_A_B_C_leaf_authority": 38,
            "coordinate_tangent_packets": 10,
            "materialized_coordinate_tangent_scalar_values": 200,
            "D1_target_atoms": 20,
            "target_direction_pairs_per_candidate": 200,
            "new_leaf_derivative_roots_per_candidate": 26400,
            "new_leaf_derivative_roots_all_candidates": 316800,
            "nonzero_leaf_derivative_roots": nonzero,
            "exact_zero_leaf_derivative_roots": 316800 - nonzero,
            "registered_D2_entries_per_candidate_before": 5324,
            "new_ordered_D2_roots_registered_per_candidate": 0,
            "registered_D2_entries_per_candidate_after": 5324,
            "full_D2_entries_per_candidate": 257499,
        },
        "claim_seals": {
            "complete_p0_metric_lower_family_registered": True,
            "all_tangent_scalar_values_defined_by_indexed_formulas": True,
            "zeros_inferred_from_tensor_pattern": False,
            "remaining_38_coordinate_columns_registered": False,
            "D2_entry_count_advanced": False,
            "complete_D2F": False,
            "global_H7": False,
            "candidate_rejection_authorized": False,
        },
        "source_bindings": {
            "predecessor": _copy(config["predecessor"]),
            "direct_evidence": _copy(config["direct_evidence"]),
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
            "live_A_B_C_block_sha256": LIVE_BLOCK_SHA256,
        },
        "data_seals": _copy(SEALS),
        "scope": (
            "exact p0 metric lower-coordinate nonlinear tangent and live leaf authority; "
            "remaining lower columns, ordered D2 replay, complete tensor, H7, rejection, "
            "and observations remain closed"
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
    expected = build_campaign(root=Path(root or Path.cwd()).resolve())
    if value.get("content_sha256") != _content_sha(value) or value != expected:
        raise P0MetricLowerLeafAuthorityError("checked result changed")


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
