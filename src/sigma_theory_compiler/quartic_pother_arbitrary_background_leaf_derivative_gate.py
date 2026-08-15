"""Register Pother A/B/C leaf derivatives from the exact geometric G-upper tangent."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Mapping
from functools import cache
from pathlib import Path
from typing import Any

import sympy as sp

from .quartic_geometric_jet_campaign import (
    SYMMETRIC_METRIC_PAIRS,
    SYMMETRIC_METRIC_WEIGHTS,
)
from .quartic_p10_inverse_product_d2_replay_gate import (
    _validate_result as _validate_p10_replay,
)
from .quartic_scalar_hessian_d2_integrability_gate import FAMILY_SPECS
from .quartic_unspecialized_source_jacobian_campaign import (
    _unspecialized_principal_blocks,
)

CONFIG_SCHEMA = "sigma-quartic-pother-arbitrary-background-leaf-derivative-config-1.0"
RESULT_SCHEMA = "sigma-quartic-pother-arbitrary-background-leaf-derivative-gate-1.0"
CAMPAIGN_ID = "quartic-pother-arbitrary-background-leaf-derivative-001"
CONFIG_PATH = "configs/backgrounds/quartic_pother_arbitrary_background_leaf_derivative_gate.json"
SOURCE_PATH = (
    "src/sigma_theory_compiler/quartic_pother_arbitrary_background_leaf_derivative_gate.py"
)
TEST_PATH = "tests/test_quartic_pother_arbitrary_background_leaf_derivative_gate.py"
OUTPUT_PATH = (
    "runs/physics-language/quartic-pother-arbitrary-background-leaf-derivative-gate/campaign.json"
)
FIRST_BLOCKER = (
    "replay_the_bound_inverse_product_D1_DAG_along_the_23760_registered_Pother_"
    "leaf_roots_to_seal_the_remaining_180_ordered_D2_roots"
)
GEOMETRIC_FORMULA_SHA256 = "9d0a41e02f3a86b4f6351240d57078e859dd9b6ce047bcaf1b08b71e2296cb11"
EVOLUTION_FORMULA_SHA256 = "53397a05fafb8728e716a57ac484b5b9e720b193daf8295bc9db1c56b4831500"
BLOCK_SHA256 = "695ff2a5fd45fa3fba21d4ce25ab2f62bd168df187c8e931bc9b5803a9cd4aed"
EXPECTED_PREDECESSOR = {
    "source": {
        "path": "src/sigma_theory_compiler/quartic_p10_inverse_product_d2_replay_gate.py",
        "file_sha256": "1da3a2d205d44331024ebe839301b50e99b93e8f1bb859ebe988039af553562a",
    },
    "config": {
        "path": "configs/backgrounds/quartic_p10_inverse_product_d2_replay_gate.json",
        "file_sha256": "3c4090b2cfd7b8d2f0cee68a2ce40c09dcf0d487c4be1a79e77942b1dedc394e",
    },
    "test": {
        "path": "tests/test_quartic_p10_inverse_product_d2_replay_gate.py",
        "file_sha256": "451762264627a58ad1bd8e5a5e8c92b981717a270326408fbed2723abdf8792b",
    },
    "artifact": {
        "path": ("runs/physics-language/quartic-p10-inverse-product-d2-replay-gate/campaign.json"),
        "file_sha256": "2a9814a27123099b9e942bde72fa45fe8783e3ddde0743d080b17008dbb9318c",
        "content_sha256": "e02949cb28f43851483d2b0b6cb06c6710ac53a16f210150449d85ceb0ec92ba",
    },
}
EXPECTED_EVOLUTION = {
    "source": {
        "path": "src/sigma_theory_compiler/quartic_nonlinear_evolution_campaign.py",
        "file_sha256": "5ffa57365f9bb74a7699719d3b6548d2d9a79155be028757940375c2b91659a8",
    },
    "config": {
        "path": "configs/backgrounds/quartic_nonlinear_evolution_campaign.json",
        "file_sha256": "68ec854157aa9107e698e1d61668aa455ef9cf7a4c0708ca443663f037f17dcf",
    },
    "test": {
        "path": "tests/test_quartic_nonlinear_evolution_campaign.py",
        "file_sha256": "1fc782722ecbb3aaac1ec9adc5917005232e28de10a530b03bbc5164b89016e6",
    },
    "artifact": {
        "path": "runs/physics-language/quartic-nonlinear-evolution-campaign/campaign.json",
        "file_sha256": "9553e58c8f4a9b0676b20331ca5bec58719bdeb495e2273084a2e5349fca65f0",
        "content_sha256": "426c7407a1839a63ebd022ee17676a202ad31a22ea5c8eb017f8ca625698f40c",
    },
}
EXPECTED_CONTRACT = {
    "candidate_count": 12,
    "Pother_target_records_per_candidate": 15,
    "unique_Pother_directions_per_candidate": 15,
    "reachable_leaf_entries_per_direction": 132,
    "leaf_derivative_roots_per_candidate": 1980,
    "leaf_derivative_roots": 23760,
}
EXPECTED_POLICIES = {
    "leaf_derivative_admission": (
        "require_exact_registered_G_upper_tangent_and_live_A_B_C_formula_replay"
    ),
    "ordered_D2_root_admission": "require_separate_closed_D1_DAG_derivative_replay",
    "full_D2F": "fail_closed",
    "global_H7": "fail_closed",
    "nonlinear_PDE": "fail_closed",
    "lifespan": "fail_closed",
    "candidate_rejection": "forbidden",
}
EXPECTED_SEALS = {
    "observations_opened": False,
    "solar_system_inputs_opened": False,
    "cosmology_inputs_opened": False,
    "paid_llm_calls": False,
    "live_SQLite_opened": False,
    "GPU_execution_used": False,
}
EXPECTED_POTHER = (
    ("s01[1]", 55, 7),
    ("s02[2]", 67, 8),
    ("s11[0]", 87, 9),
    ("s11[4]", 91, 10),
    ("s11[7]", 94, 11),
    ("s11[9]", 96, 12),
    ("s12[5]", 103, 13),
    ("s13[6]", 115, 14),
    ("s22[0]", 120, 15),
    ("s22[4]", 124, 16),
    ("s22[7]", 127, 17),
    ("s22[9]", 129, 18),
    ("s23[8]", 139, 19),
    ("s33[4]", 146, 20),
    ("s33[7]", 149, 21),
)
_EXPECTED_CACHE: dict[tuple[str, ...], dict[str, Any]] = {}


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _content_sha(value: Mapping[str, Any]) -> str:
    return _sha({key: item for key, item in value.items() if key != "content_sha256"})


def _copy(value: Any) -> Any:
    return json.loads(_canonical(value))


def _inside(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if path != root and root not in path.parents:
        raise ValueError("Pother arbitrary-background leaf path escapes project root")
    return path


def _validate_config(value: Mapping[str, Any]) -> None:
    if value != {
        "schema_version": CONFIG_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "output_path": OUTPUT_PATH,
        "predecessor": EXPECTED_PREDECESSOR,
        "direct_evidence": {"nonlinear_evolution": EXPECTED_EVOLUTION},
        "leaf_contract": EXPECTED_CONTRACT,
        "policies": EXPECTED_POLICIES,
        "seals": EXPECTED_SEALS,
    }:
        raise ValueError("Pother arbitrary-background leaf config boundary changed")


def _load_bound(root: Path, binding: Mapping[str, Any]) -> dict[str, Any]:
    path = _inside(root, str(binding["path"]))
    if _file_sha(path) != binding["file_sha256"]:
        raise ValueError("Pother arbitrary-background leaf artifact file binding changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("content_sha256") != binding["content_sha256"] or value.get(
        "content_sha256"
    ) != _content_sha(value):
        raise ValueError("Pother arbitrary-background leaf artifact content binding changed")
    return value


def _load_bundle(root: Path, bundle: Mapping[str, Any]) -> dict[str, Any]:
    if set(bundle) != {"source", "config", "test", "artifact"}:
        raise ValueError("Pother arbitrary-background leaf evidence bundle changed")
    for label in ("source", "config", "test"):
        binding = bundle[label]
        if _file_sha(_inside(root, binding["path"])) != binding["file_sha256"]:
            raise ValueError("Pother arbitrary-background leaf evidence file changed")
    return _load_bound(root, bundle["artifact"])


def _load_inputs(
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    for label in ("source", "config", "test"):
        binding = EXPECTED_PREDECESSOR[label]
        if _file_sha(_inside(root, binding["path"])) != binding["file_sha256"]:
            raise ValueError("Pother arbitrary-background leaf predecessor changed")
    predecessor = _load_bound(root, EXPECTED_PREDECESSOR["artifact"])
    _validate_p10_replay(predecessor, root=root)
    evolution = _load_bundle(root, EXPECTED_EVOLUTION)
    if (
        predecessor.get("decision")
        != "pass_all_84_P10_ordered_D2_roots_exactly_replayed_Pother_blocked"
        or predecessor.get("gate_counts", {}).get("sealed_P10_ordered_D2_roots") != 84
        or predecessor.get("gate_counts", {}).get("Pother_ordered_D2_roots_registered") != 0
        or evolution.get("status")
        != "pass_all_12_exact_local_nonlinear_time_acceleration_eliminations"
        or evolution.get("nonlinear_evolution_control", {}).get("formula_contract_sha256")
        != EVOLUTION_FORMULA_SHA256
        or evolution.get("nonlinear_evolution_control", {}).get("passed") is not True
    ):
        raise ValueError("Pother arbitrary-background leaf evidence boundary changed")
    p10_leaf_binding = predecessor["source_bindings"]["predecessor"]["artifact"]
    p10_leaf = _load_bound(root, p10_leaf_binding)
    flat_binding = p10_leaf["source_bindings"]["predecessor"]["artifact"]
    flat = _load_bound(root, flat_binding)
    if (
        p10_leaf.get("nonlinear_geometric_map_binding", {}).get("formula_contract_sha256")
        != GEOMETRIC_FORMULA_SHA256
        or flat.get("gate_counts", {}).get("factorized_target_roots_per_candidate") != 20
    ):
        raise ValueError("Pother arbitrary-background leaf transitive boundary changed")
    return predecessor, p10_leaf, flat, evolution


def _target_templates(flat: Mapping[str, Any]) -> list[dict[str, Any]]:
    records = flat.get("factorized_leaf_derivative_manifest")
    if not isinstance(records, list) or len(records) != 20:
        raise ValueError("Pother arbitrary-background leaf target inventory changed")
    by_atom = {str(row["coordinate_atom"]): row for row in records}
    targets = []
    for atom, column, ordinal in EXPECTED_POTHER:
        row = by_atom.get(atom)
        if (
            row is None
            or row.get("coordinate_column") != column
            or row.get("coordinate_ordinals") != [ordinal]
        ):
            raise ValueError("Pother arbitrary-background leaf target record changed")
        targets.append(row)
    return targets


@cache
def _background_symbols() -> tuple[sp.Matrix, sp.Matrix, tuple[sp.Symbol, ...]]:
    lower = sp.zeros(4)
    inverse = sp.zeros(4)
    symbols = []
    for left in range(4):
        for right in range(left, 4):
            lower_symbol = sp.Symbol(f"g_{left}{right}", real=True)
            inverse_symbol = sp.Symbol(f"gu_{left}{right}", real=True)
            lower[left, right] = lower[right, left] = lower_symbol
            inverse[left, right] = inverse[right, left] = inverse_symbol
            symbols.extend((lower_symbol, inverse_symbol))
    return lower, inverse, tuple(symbols)


def _second_metric_tangent(
    atom: str,
) -> tuple[int, int, int, dict[str, sp.Expr]]:
    match = re.fullmatch(r"s(\d)(\d)\[(\d+)]", atom)
    if match is None:
        raise ValueError("Pother arbitrary-background leaf atom schema changed")
    derivative_left, derivative_right, field = map(int, match.groups())
    metric_left, metric_right = SYMMETRIC_METRIC_PAIRS[field]
    weight = SYMMETRIC_METRIC_WEIGHTS[field]
    lower, inverse, _ = _background_symbols()

    def partial2(mu: int, nu: int, left: int, right: int) -> sp.Expr:
        derivative_match = int(
            (mu == derivative_left and nu == derivative_right)
            or (
                derivative_left != derivative_right
                and mu == derivative_right
                and nu == derivative_left
            )
        )
        metric_match = (
            int(left == metric_left and right == metric_right)
            if metric_left == metric_right
            else int(left == metric_left and right == metric_right)
            + int(left == metric_right and right == metric_left)
        )
        return sp.Rational(derivative_match * metric_match, 1) / weight

    ricci = sp.Matrix(
        4,
        4,
        lambda mu, nu: sp.factor(
            sum(
                inverse[rho, sigma]
                * (
                    partial2(rho, mu, sigma, nu)
                    + partial2(rho, nu, sigma, mu)
                    - partial2(rho, sigma, mu, nu)
                    - partial2(mu, nu, rho, sigma)
                )
                / 2
                for rho in range(4)
                for sigma in range(4)
            )
        ),
    )
    scalar = sp.factor(sum(inverse[mu, nu] * ricci[mu, nu] for mu in range(4) for nu in range(4)))
    einstein_lower = sp.Matrix(
        4,
        4,
        lambda mu, nu: sp.factor(ricci[mu, nu] - lower[mu, nu] * scalar / 2),
    )
    einstein_upper = (inverse * einstein_lower * inverse).applyfunc(sp.factor)
    return (
        derivative_left,
        derivative_right,
        field,
        {f"G_{mu}{nu}": einstein_upper[mu, nu] for mu in range(4) for nu in range(mu, 4)},
    )


@cache
def _generic_derivatives() -> dict[str, dict[str, Any]]:
    blocks = _unspecialized_principal_blocks()
    if blocks["content_sha256"] != BLOCK_SHA256:
        raise ValueError("Pother arbitrary-background leaf live blocks changed")
    data = blocks["data"]
    einstein_symbols = sorted(data["einstein_upper"].free_symbols, key=str)
    chunks = {
        family: multiplicity
        * (blocks[kind][first] if kind == "B_i" else blocks[kind][first][second])
        for family, _, _, kind, first, second, multiplicity in FAMILY_SPECS
    }
    packets = {}
    for atom, _, _ in EXPECTED_POTHER:
        left, right, field, tangent = _second_metric_tangent(atom)

        def chain(expression: sp.Expr, local_tangent: Mapping[str, sp.Expr] = tangent) -> sp.Expr:
            return sp.factor(
                sum(
                    sp.diff(expression, symbol) * local_tangent[str(symbol)]
                    for symbol in einstein_symbols
                )
            )

        derivative_A = blocks["A"].applyfunc(chain)
        family = atom.split("[")[0]
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
        tangent_values = [tangent[f"G_{mu}{nu}"] for mu in range(4) for nu in range(mu, 4)]
        packets[atom] = {
            "coordinate_second_derivative_indices": [left, right],
            "metric_field": field,
            "metric_component": list(SYMMETRIC_METRIC_PAIRS[field]),
            "metric_component_weight": str(SYMMETRIC_METRIC_WEIGHTS[field]),
            "G_upper_tangent": {
                label: str(value) for label, value in tangent.items() if value != 0
            },
            "G_upper_tangent_nonzero_components": sum(value != 0 for value in tangent_values),
            "G_upper_tangent_sha256": _sha([str(value) for value in tangent_values]),
            "source_chunk_family": family,
            "source_chunk_input_column": field,
            "A_derivative_sparse_entries": sparse_A,
            "source_chunk_column_derivative_sparse_entries": sparse_chunk,
            "total_leaf_derivatives": 132,
            "nonzero_leaf_derivatives": len(sparse_A) + len(sparse_chunk),
            "zero_leaf_derivatives": 132 - len(sparse_A) - len(sparse_chunk),
        }
    if (
        len(packets) != 15
        or sum(row["nonzero_leaf_derivatives"] for row in packets.values()) != 13
        or sum(row["zero_leaf_derivatives"] for row in packets.values()) != 1967
        or sum(row["G_upper_tangent_nonzero_components"] == 0 for row in packets.values()) != 2
    ):
        raise ValueError("Pother arbitrary-background leaf generic census changed")
    return packets


def _expression_DAG(values: set[str]) -> tuple[dict[str, Any], dict[str, int]]:
    lower, inverse, background_symbols = _background_symbols()
    del lower, inverse
    allowed_symbols = {str(symbol) for symbol in background_symbols} | {"alpha"}
    locals_map = {str(symbol): symbol for symbol in background_symbols}
    locals_map["alpha"] = sp.Symbol("alpha", real=True)
    expressions = []
    for value in values:
        expression = sp.sympify(value, locals=locals_map)
        if {str(symbol) for symbol in expression.free_symbols} - allowed_symbols:
            raise ValueError("Pother arbitrary-background leaf expression symbol escaped")
        expressions.append(expression)
    ordered = sorted(
        expressions, key=lambda expression: (sp.count_ops(expression), str(expression))
    )
    nodes = [
        {
            "op": "exact_sympy_rational_expression",
            "expression": str(expression),
            "srepr_sha256": hashlib.sha256(sp.srepr(expression).encode()).hexdigest(),
            "free_symbols": sorted(str(symbol) for symbol in expression.free_symbols),
        }
        for expression in ordered
    ]
    body = {
        "schema_version": "sigma-Pother-leaf-exact-rational-expression-DAG-1.0",
        "allowed_operations": ["exact_sympy_rational_expression"],
        "allowed_expression_primitives": [
            "Integer",
            "Rational",
            "Add",
            "Mul",
            "Pow(integer_exponent)",
            "sqrt(2)",
        ],
        "background_symbol_contract": {
            "lower_metric_symbols": [f"g_{i}{j}" for i in range(4) for j in range(i, 4)],
            "inverse_metric_symbols": [f"gu_{i}{j}" for i in range(4) for j in range(i, 4)],
            "coefficient_symbols": ["alpha"],
            "domain": "g_is_nonsingular_and_gu_is_its_exact_inverse",
        },
        "node_count": len(nodes),
        "nodes": nodes,
    }
    return {**body, "content_sha256": _sha(body)}, {
        str(expression): index for index, expression in enumerate(ordered)
    }


def _candidate_coefficients(p10_leaf: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    records = {
        str(row["candidate_id"]): row["coefficients"]
        for row in p10_leaf.get("candidate_manifests", [])
    }
    if len(records) != 12:
        raise ValueError("Pother arbitrary-background leaf candidate set changed")
    return records


def _candidate_manifests(
    coefficients: Mapping[str, Mapping[str, Any]],
    targets: list[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    generic = _generic_derivatives()
    specialized: dict[
        str, list[tuple[Mapping[str, Any], Mapping[str, Any], list[dict[str, Any]]]]
    ] = {}
    values = {"0"}
    alpha = _unspecialized_principal_blocks()["data"]["alpha"]
    for candidate_id, candidate_coefficients in coefficients.items():
        substitution = {alpha: sp.sympify(candidate_coefficients["a10"])}
        specialized[candidate_id] = []
        for target in targets:
            packet = generic[str(target["coordinate_atom"])]
            sparse = []
            for entry in packet["A_derivative_sparse_entries"]:
                value = str(sp.factor(sp.sympify(entry["value"]).subs(substitution)))
                sparse.append({**entry, "value": value})
                values.add(value)
            for entry in packet["source_chunk_column_derivative_sparse_entries"]:
                value = str(sp.factor(sp.sympify(entry["value"]).subs(substitution)))
                sparse.append({"row": entry["row"], "chunk": True, "value": value})
                values.add(value)
            specialized[candidate_id].append((target, packet, sparse))
    dag, roots = _expression_DAG(values)
    manifests = []
    for candidate_id, rows in specialized.items():
        packets = []
        for target, generic_packet, sparse in rows:
            A_entries = [
                {**entry, "arithmetic_root": roots[entry["value"]]}
                for entry in sparse
                if not entry.get("chunk")
            ]
            chunk_entries = [
                {
                    "row": entry["row"],
                    "value": entry["value"],
                    "arithmetic_root": roots[entry["value"]],
                }
                for entry in sparse
                if entry.get("chunk")
            ]
            dense_roots = [roots["0"]] * 132
            for entry in A_entries:
                dense_roots[11 * entry["row"] + entry["column"]] = entry["arithmetic_root"]
            for entry in chunk_entries:
                dense_roots[121 + entry["row"]] = entry["arithmetic_root"]
            packets.append(
                {
                    "coordinate_atom": target["coordinate_atom"],
                    "coordinate_column": target["coordinate_column"],
                    "coordinate_ordinals": target["coordinate_ordinals"],
                    "coordinate_second_derivative_indices": generic_packet[
                        "coordinate_second_derivative_indices"
                    ],
                    "metric_field": generic_packet["metric_field"],
                    "metric_component": generic_packet["metric_component"],
                    "metric_component_weight": generic_packet["metric_component_weight"],
                    "G_upper_tangent_sha256": generic_packet["G_upper_tangent_sha256"],
                    "G_upper_tangent_nonzero_components": generic_packet[
                        "G_upper_tangent_nonzero_components"
                    ],
                    "source_chunk_family": generic_packet["source_chunk_family"],
                    "source_chunk_input_column": generic_packet["source_chunk_input_column"],
                    "A_derivative_shape": [11, 11],
                    "A_derivative_sparse_entries": A_entries,
                    "source_chunk_column_shape": [11],
                    "source_chunk_column_derivative_sparse_entries": chunk_entries,
                    "zero_default_arithmetic_root": roots["0"],
                    "arithmetic_dag_sha256": dag["content_sha256"],
                    "total_leaf_derivative_roots": 132,
                    "nonzero_leaf_derivative_roots": len(sparse),
                    "dense_root_manifest_sha256": _sha(dense_roots),
                    "registered_symbolic_background_scope": True,
                }
            )
        if len(packets) != 15 or sum(row["nonzero_leaf_derivative_roots"] for row in packets) != 13:
            raise ValueError("Pother arbitrary-background leaf candidate census changed")
        manifests.append(
            {
                "candidate_id": candidate_id,
                "coefficients": coefficients[candidate_id],
                "direction_packets": packets,
                "unique_Pother_directions": 15,
                "Pother_target_records": 15,
                "registered_leaf_derivative_roots": 1980,
                "nonzero_leaf_derivative_roots": 13,
                "zero_leaf_derivative_roots": 1967,
                "Pother_ordered_D2_roots_registered": 0,
                "Pother_ordered_D2_roots_replay_ready": 15,
                "manifest_sha256": _sha(packets),
                "candidate_decision": "pass_Pother_leaf_derivatives_D2_replay_ready",
                "candidate_rejection_authorized": False,
                "first_blocker": FIRST_BLOCKER,
            }
        )
    return manifests, dag


def _expected_body(
    root: Path,
    config_path: Path,
    predecessor: Mapping[str, Any],
    p10_leaf: Mapping[str, Any],
    flat: Mapping[str, Any],
    evolution: Mapping[str, Any],
) -> dict[str, Any]:
    cache_key = (
        str(root),
        _file_sha(_inside(root, SOURCE_PATH)),
        _file_sha(config_path),
        _file_sha(_inside(root, TEST_PATH)),
        str(predecessor["content_sha256"]),
        str(p10_leaf["content_sha256"]),
        str(flat["content_sha256"]),
        str(evolution["content_sha256"]),
    )
    cached = _EXPECTED_CACHE.get(cache_key)
    if cached is not None:
        return _copy(cached)
    targets = _target_templates(flat)
    coefficients = _candidate_coefficients(p10_leaf)
    manifests, dag = _candidate_manifests(coefficients, targets)
    generic = _generic_derivatives()
    body = {
        "schema_version": RESULT_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "decision": "pass_23760_Pother_leaf_roots_all_180_D2_records_replay_ready",
        "decision_counts": {"pass": 12, "blocked": 0, "reject": 0},
        "downstream_admission_counts": {"pass": 0, "blocked": 12, "reject": 0},
        "first_blocker": FIRST_BLOCKER,
        "leaf_derivative_theorem": {
            "name": "coordinate_second_metric_to_Einstein_upper_to_registered_A_B_C_leaf_chain",
            "exact_result": (
                "For each of the 15 Pother coordinate-second-metric directions, the registered "
                "nonlinear geometric formulas give the exact tangent of Ricci, Einstein-lower, "
                "and Einstein-upper on a nonsingular symbolic metric background. Chain-rule "
                "differentiation of the live registered A/B/C formulas registers all 1,980 "
                "reachable leaf roots per candidate: 13 nonzero and 1,967 zero."
            ),
            "zero_directions": [
                atom for atom, packet in generic.items() if packet["nonzero_leaf_derivatives"] == 0
            ],
            "boundary": (
                "The result is exact for the registered symbolic principal-block formula "
                "composed with the registered coordinate-to-G-upper map, with gu constrained "
                "to be the inverse of g. It does not claim a separately covariantized A/B/C "
                "formula beyond that registered model, and it does not yet emit the 180 D2 "
                "replay roots."
            ),
        },
        "registered_formula_bindings": {
            "geometric_formula_contract_sha256": GEOMETRIC_FORMULA_SHA256,
            "nonlinear_evolution_formula_contract_sha256": EVOLUTION_FORMULA_SHA256,
            "unspecialized_A_B_C_block_sha256": BLOCK_SHA256,
            "metric_inverse_domain": "g_is_nonsingular_and_gu_is_its_exact_inverse",
            "registered_symbolic_principal_block_scope": True,
        },
        "generic_tangent_packets": [
            {"coordinate_atom": atom, **_copy(generic[atom])} for atom, _, _ in EXPECTED_POTHER
        ],
        "leaf_derivative_arithmetic_DAG": dag,
        "candidate_manifests": manifests,
        "manifest_sha256": _sha([row["manifest_sha256"] for row in manifests]),
        "gate_counts": {
            "selected_candidates": 12,
            "P10_ordered_D2_roots_previously_sealed": 84,
            "Pother_target_records": 180,
            "unique_Pother_directions_per_candidate": 15,
            "registered_Pother_leaf_derivative_roots": 23760,
            "nonzero_Pother_leaf_derivative_roots": 156,
            "zero_Pother_leaf_derivative_roots": 23604,
            "Pother_ordered_D2_roots_replay_ready": 180,
            "Pother_ordered_D2_roots_registered": 0,
            "Pother_ordered_D2_roots_blocked_on_replay": 180,
            "all_target_ordered_D2_roots_registered": 84,
            "all_target_ordered_D2_roots_blocked": 180,
            "complete_ordered_D2F_tensors_registered": 0,
            "global_H7_closures": 0,
            "nonlinear_PDE_closures": 0,
            "lifespans_proved": 0,
        },
        "claim_seals": {
            "all_23760_Pother_leaf_derivative_roots_registered": True,
            "all_180_Pother_ordered_D2_records_replay_ready": True,
            "Pother_ordered_D2_roots_registered": False,
            "physical_no_go_proved": False,
            "complete_ordered_D2F_tensor_registered": False,
            "global_H7_energy_closed": False,
            "nonlinear_PDE_closed": False,
            "nonlinear_lifespan_proved": False,
            "candidate_theory_rejected": False,
            "observational_claim_made": False,
        },
        "exact_controls": {
            "identify_lower_and_upper_Einstein_components": {"rejected": True},
            "omit_metric_component_sqrt2_weight": {"rejected": True},
            "treat_gu_as_independent_of_inverse_metric_domain": {"rejected": True},
            "promote_replay_ready_record_to_D2_root": {"rejected": True},
            "promote_Pother_leaf_registration_to_complete_D2F": {"rejected": True},
            "infer_physical_no_go_from_replay_boundary": {"rejected": True},
            "reject_candidate_from_replay_boundary": {"rejected": True},
        },
        "data_seals": dict(EXPECTED_SEALS),
        "source_bindings": {
            "source": {"path": SOURCE_PATH, "file_sha256": _file_sha(_inside(root, SOURCE_PATH))},
            "config": {"path": CONFIG_PATH, "file_sha256": _file_sha(config_path)},
            "test": {"path": TEST_PATH, "file_sha256": _file_sha(_inside(root, TEST_PATH))},
            "predecessor": _copy(EXPECTED_PREDECESSOR),
            "direct_evidence": {"nonlinear_evolution": _copy(EXPECTED_EVOLUTION)},
        },
        "scope": (
            "candidate-bound registered-symbolic-background A/B/C input-leaf derivatives for "
            "the 15 Pother target directions only; no Pother D2 replay root, separately "
            "covariantized principal block, physical no-go, complete D2F, H7, PDE, lifespan, "
            "candidate rejection, or observational claim"
        ),
    }
    _EXPECTED_CACHE[cache_key] = body
    return _copy(body)


def _validate_bindings(value: Mapping[str, Any], root: Path) -> None:
    bindings = value.get("source_bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != {
        "source",
        "config",
        "test",
        "predecessor",
        "direct_evidence",
    }:
        raise ValueError("Pother arbitrary-background leaf binding keys changed")
    for label, relative in {
        "source": SOURCE_PATH,
        "config": CONFIG_PATH,
        "test": TEST_PATH,
    }.items():
        if bindings[label] != {
            "path": relative,
            "file_sha256": _file_sha(_inside(root, relative)),
        }:
            raise ValueError("Pother arbitrary-background leaf local binding changed")
    if bindings["predecessor"] != EXPECTED_PREDECESSOR:
        raise ValueError("Pother arbitrary-background leaf predecessor binding changed")
    if bindings["direct_evidence"] != {"nonlinear_evolution": EXPECTED_EVOLUTION}:
        raise ValueError("Pother arbitrary-background leaf evidence binding changed")


def _validate_result(value: Mapping[str, Any], *, root: Path | None = None) -> None:
    validation_root = (root or Path(__file__).resolve().parents[2]).resolve()
    if value.get("content_sha256") != _content_sha(value):
        raise ValueError("Pother arbitrary-background leaf content hash changed")
    _validate_bindings(value, validation_root)
    config_path = _inside(validation_root, CONFIG_PATH)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    predecessor, p10_leaf, flat, evolution = _load_inputs(validation_root)
    expected = _expected_body(validation_root, config_path, predecessor, p10_leaf, flat, evolution)
    if {key: item for key, item in value.items() if key != "content_sha256"} != expected:
        raise ValueError("Pother arbitrary-background leaf result boundary changed")


def build_gate(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    root = config_path.parents[2]
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    predecessor, p10_leaf, flat, evolution = _load_inputs(root)
    body = _expected_body(root, config_path, predecessor, p10_leaf, flat, evolution)
    result = {**body, "content_sha256": _sha(body)}
    _validate_result(result, root=root)
    return result


def write_gate(config_path: Path) -> Path:
    result = build_gate(config_path)
    root = config_path.resolve().parents[2]
    output = _inside(root, OUTPUT_PATH)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(CONFIG_PATH))
    args = parser.parse_args()
    print(write_gate(args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
