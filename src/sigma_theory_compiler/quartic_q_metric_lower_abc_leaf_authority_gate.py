"""Close the unique lower q-metric leaf family with exact two-jet tangents."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from functools import cache
from pathlib import Path
from typing import Any

import sympy as sp

from . import quartic_p0_metric_lower_abc_leaf_authority_gate as base
from .quartic_remaining_scalar_hessian_abc_leaf_authority_gate import (
    _content_sha,
    _production_sha,
    _sha,
)

CONFIG_SCHEMA = "sigma-quartic-q-metric-lower-tangent-checkpoint-config-1.0"
RESULT_SCHEMA = "sigma-quartic-q-metric-lower-tangent-checkpoint-1.0"
CAMPAIGN_ID = "quartic-q-metric-lower-abc-leaf-authority-001"
STEM = "quartic_q_metric_lower_abc_leaf_authority_gate"
SLUG = "quartic-q-metric-lower-abc-leaf-authority-gate"
CONFIG_PATH = f"configs/backgrounds/{STEM}.json"
SOURCE_PATH = f"src/sigma_theory_compiler/{STEM}.py"
TEST_PATH = f"tests/test_{STEM}.py"
OUTPUT_PATH = f"runs/physics-language/{SLUG}/campaign.json"
CONFIG_PRODUCTION_SHA256 = "10824a48596334e2fc35c3d45b4f6918d6bf61285bee5d13ae4af753bbd37c81"
TEST_PRODUCTION_SHA256 = "48bb9283da77dfb8d9223cea1cad39f7bb30fd3ab8303e17372a2672bbaf5b8a"
Q_METRIC = tuple((f"q[{field}]", field, pair) for field, pair in enumerate(base.SYMMETRIC_PAIRS))
CONTRACT = {
    "candidate_count": 12,
    "predecessor_formal_registered_slots": 145,
    "predecessor_duplicate_formal_slot_excess": 2,
    "predecessor_unique_registered_coordinate_columns": 143,
    "predecessor_missing_unique_coordinate_columns": 10,
    "new_q_metric_columns": 10,
    "unique_coordinate_columns_after": 153,
    "d1_target_atoms": 20,
    "planned_leaf_roots_per_target_pair": 132,
    "planned_leaf_roots_per_candidate": 26400,
    "planned_leaf_roots_all_candidates": 316800,
    "materialized_leaf_roots_all_candidates": 0,
    "registered_D2_entries_per_candidate": 5324,
    "full_D2_entries_per_candidate": 257499,
}
POLICIES = {
    "family_selection": "checkpoint_complete_q_metric_value_indexed_two_jet_tangents",
    "coordinate_jet_domain": (
        "pointwise_orthonormal_metric_with_arbitrary_consistent_first_and_second_metric_jets"
    ),
    "alias_accounting": "22_formal_predecessor_slots_equal_20_unique_coordinate_vectors",
    "normalization": "deterministic_unexpanded_exact_sympy_expression_tree",
    "D2_promotion": "forbidden_without_separate_closed_D1_arithmetic_DAG_replay",
    "global_H7": "fail_closed",
    "candidate_rejection": "forbidden",
}
SEALS = dict(base.SEALS)
_BODY_CACHE: dict[tuple[str, ...], dict[str, Any]] = {}


class QMetricLowerLeafAuthorityError(ValueError):
    """The q tangent, unique-coordinate census, or sealed authority changed."""


def _inside(root: Path, relative: str) -> Path:
    if not relative or "\\" in relative:
        raise QMetricLowerLeafAuthorityError("q lower path is not portable")
    path = (root / relative).resolve()
    if path != root and root not in path.parents:
        raise QMetricLowerLeafAuthorityError("q lower path escapes root")
    return path


def _copy(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, separators=(",", ":")))


def _validate_config(value: Mapping[str, Any], path: Path) -> None:
    if _production_sha(path) != CONFIG_PRODUCTION_SHA256:
        raise QMetricLowerLeafAuthorityError("q lower config production bytes changed")
    if (
        value.get("schema_version") != CONFIG_SCHEMA
        or value.get("campaign_id") != CAMPAIGN_ID
        or value.get("output_path") != OUTPUT_PATH
        or value.get("self_bindings")
        != {"test_path": TEST_PATH, "test_sha256": TEST_PRODUCTION_SHA256}
        or value.get("family_contract") != CONTRACT
        or value.get("policies") != POLICIES
        or value.get("seals") != SEALS
        or set(value.get("direct_evidence", {})) != {"lower_projection", "live_abc"}
    ):
        raise QMetricLowerLeafAuthorityError("q lower config contract changed")


def _load_inputs(root: Path, config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    try:
        predecessor = base._load_bundle(root, config["predecessor"], artifact_hash=False)
        projection = base._load_bundle(
            root, config["direct_evidence"]["lower_projection"], artifact_hash=True
        )
        live = base._load_bundle(root, config["direct_evidence"]["live_abc"], artifact_hash=True)
    except base.P0MetricLowerLeafAuthorityError as error:
        raise QMetricLowerLeafAuthorityError(str(error)) from error
    if (
        predecessor.get("decision")
        != "pass_p3_metric_10_column_nonlinear_tangents_316800_exact_leaf_roots_D2_blocked"
        or predecessor.get("gate_counts", {}).get(
            "remaining_coordinate_columns_without_A_B_C_leaf_authority"
        )
        != 8
        or predecessor.get("gate_counts", {}).get("registered_D2_entries_per_candidate_after")
        != 5324
        or projection.get("decision")
        != "pass_all_54_lower_covariant_projections_D2_count_preserved"
        or projection.get("gate_counts", {}).get("lower_projection_directions_registered") != 54
        or projection.get("gate_counts", {}).get("unique_coordinate_directions_projected") != 153
        or live.get("status")
        != "pass_all_12_complete_unspecialized_principal_source_jacobians_remainder_fail_closed"
    ):
        raise QMetricLowerLeafAuthorityError("q lower predecessor boundary changed")
    return {"predecessor": predecessor, "projection": projection, "live_abc": live}


def _validate_projection(projection: Mapping[str, Any]) -> None:
    records = {
        str(row["coordinate_atom"]): row
        for row in projection.get("lower_projection_registry", [])
        if row.get("family") == "q_metric"
    }
    if set(records) != {row[0] for row in Q_METRIC}:
        raise QMetricLowerLeafAuthorityError("q lower projection family changed")
    for atom, column, pair in Q_METRIC:
        row = records[atom]
        seed = row.get("tangent_seed", {})
        if (
            row.get("coordinate_column") != column
            or row.get("derivative_index") is not None
            or seed.get("dg_symmetric_pair") != list(pair)
            or seed.get("dg_value") != ("1" if pair[0] == pair[1] else "sqrt(2)/2")
            or seed.get("dP") != "0"
            or row.get("exact_projection_registered") is not True
        ):
            raise QMetricLowerLeafAuthorityError("q lower projection seed changed")


@cache
def _coordinate_primitives() -> dict[str, Any]:
    inverse = [[sp.Integer(0) for _ in range(4)] for _ in range(4)]
    metric = [[sp.Integer(0) for _ in range(4)] for _ in range(4)]
    for index, value in enumerate((-1, 1, 1, 1)):
        inverse[index][index] = sp.Integer(value)
        metric[index][index] = sp.Integer(value)
    first: list[list[list[sp.Expr]]] = []
    first_symbols: list[sp.Symbol] = []
    for derivative in range(4):
        matrix, symbols = base._symmetric_symbols(f"P{derivative}")
        first.append(matrix)
        first_symbols.extend(symbols)
    second = [
        [[[sp.Integer(0) for _ in range(4)] for _ in range(4)] for _ in range(4)] for _ in range(4)
    ]
    second_symbols: list[sp.Symbol] = []
    for dleft, dright in base.SYMMETRIC_PAIRS:
        for mleft, mright in base.SYMMETRIC_PAIRS:
            symbol = sp.Symbol(f"S{dleft}{dright}_{mleft}{mright}", real=True)
            second_symbols.append(symbol)
            for a, b in ((dleft, dright), (dright, dleft)):
                second[a][b][mleft][mright] = symbol
                second[a][b][mright][mleft] = symbol
    gradient = tuple(sp.Symbol(f"v_{index}", real=True) for index in range(4))
    return {
        "inverse": inverse,
        "metric": metric,
        "first": first,
        "second": second,
        "gradient": gradient,
        "symbols": (*first_symbols, *second_symbols, *gradient),
    }


@cache
def _exact_q_tangents() -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    primitive = _coordinate_primitives()
    inverse = primitive["inverse"]
    metric = primitive["metric"]
    first = primitive["first"]
    second = primitive["second"]
    gradient = primitive["gradient"]
    bracket = [
        [
            [
                first[left][contracted][right]
                + first[right][contracted][left]
                - first[contracted][left][right]
                for right in range(4)
            ]
            for left in range(4)
        ]
        for contracted in range(4)
    ]
    cbracket = [
        [
            [
                [
                    second[derivative][left][contracted][right]
                    + second[derivative][right][contracted][left]
                    - second[derivative][contracted][left][right]
                    for right in range(4)
                ]
                for left in range(4)
            ]
            for contracted in range(4)
        ]
        for derivative in range(4)
    ]
    gamma = [
        [
            [
                sum(
                    inverse[upper][contracted] * bracket[contracted][left][right]
                    for contracted in range(4)
                )
                / 2
                for right in range(4)
            ]
            for left in range(4)
        ]
        for upper in range(4)
    ]
    inverse_first = [
        [
            [
                -sum(
                    inverse[upper][left] * first[derivative][left][lower] * inverse[lower][right]
                    for left in range(4)
                    for lower in range(4)
                )
                for right in range(4)
            ]
            for upper in range(4)
        ]
        for derivative in range(4)
    ]
    partial_gamma = [
        [
            [
                [
                    sum(
                        inverse_first[derivative][upper][contracted]
                        * bracket[contracted][left][right]
                        + inverse[upper][contracted] * cbracket[derivative][contracted][left][right]
                        for contracted in range(4)
                    )
                    / 2
                    for right in range(4)
                ]
                for left in range(4)
            ]
            for upper in range(4)
        ]
        for derivative in range(4)
    ]
    ricci = sp.zeros(4)
    for left in range(4):
        for right in range(4):
            ricci[left, right] = sum(
                partial_gamma[upper][upper][right][left]
                - partial_gamma[right][upper][upper][left]
                + sum(
                    gamma[upper][upper][contracted] * gamma[contracted][right][left]
                    - gamma[upper][right][contracted] * gamma[contracted][upper][left]
                    for contracted in range(4)
                )
                for upper in range(4)
            )
    scalar = sum(
        inverse[left][right] * ricci[left, right] for left in range(4) for right in range(4)
    )
    einstein = sp.Matrix(
        4,
        4,
        lambda left, right: ricci[left, right] - metric[left][right] * scalar / 2,
    )
    packets: dict[str, dict[str, Any]] = {}
    for atom, column, (seed_left, seed_right) in Q_METRIC:
        delta_metric = sp.zeros(4)
        weight = sp.Integer(1) if seed_left == seed_right else sp.sqrt(2) / 2
        delta_metric[seed_left, seed_right] = weight
        delta_metric[seed_right, seed_left] = weight
        delta_inverse = sp.Matrix(inverse) * delta_metric * sp.Matrix(inverse) * -1
        delta_gamma = [
            [
                [
                    sum(
                        delta_inverse[upper, contracted] * bracket[contracted][left][right]
                        for contracted in range(4)
                    )
                    / 2
                    for right in range(4)
                ]
                for left in range(4)
            ]
            for upper in range(4)
        ]
        delta_inverse_first = [
            [
                [
                    -sum(
                        (
                            delta_inverse[upper, a] * first[derivative][a][b] * inverse[b][right]
                            + inverse[upper][a] * first[derivative][a][b] * delta_inverse[b, right]
                        )
                        for a in range(4)
                        for b in range(4)
                    )
                    for right in range(4)
                ]
                for upper in range(4)
            ]
            for derivative in range(4)
        ]
        delta_partial = [
            [
                [
                    [
                        sum(
                            delta_inverse_first[derivative][upper][contracted]
                            * bracket[contracted][left][right]
                            + delta_inverse[upper, contracted]
                            * cbracket[derivative][contracted][left][right]
                            for contracted in range(4)
                        )
                        / 2
                        for right in range(4)
                    ]
                    for left in range(4)
                ]
                for upper in range(4)
            ]
            for derivative in range(4)
        ]
        delta_ricci = sp.zeros(4)
        for left in range(4):
            for right in range(4):
                delta_ricci[left, right] = sum(
                    delta_partial[upper][upper][right][left]
                    - delta_partial[right][upper][upper][left]
                    + sum(
                        delta_gamma[upper][upper][contracted] * gamma[contracted][right][left]
                        + gamma[upper][upper][contracted] * delta_gamma[contracted][right][left]
                        - delta_gamma[upper][right][contracted] * gamma[contracted][upper][left]
                        - gamma[upper][right][contracted] * delta_gamma[contracted][upper][left]
                        for contracted in range(4)
                    )
                    for upper in range(4)
                )
        delta_scalar = sum(
            delta_inverse[left, right] * ricci[left, right]
            + inverse[left][right] * delta_ricci[left, right]
            for left in range(4)
            for right in range(4)
        )
        delta_einstein = sp.Matrix(
            4,
            4,
            lambda left, right, dr=delta_ricci, dm=delta_metric, ds=delta_scalar: (
                dr[left, right] - (dm[left, right] * scalar + metric[left][right] * ds) / 2
            ),
        )
        delta_h = {}
        delta_g = {}
        for left, right in base.SYMMETRIC_PAIRS:
            delta_h[f"H_{left}{right}"] = str(
                -sum(delta_gamma[upper][left][right] * gradient[upper] for upper in range(4))
            )
            raised = sum(
                delta_inverse[left, a] * inverse[right][b] * einstein[a, b]
                + inverse[left][a] * delta_inverse[right, b] * einstein[a, b]
                + inverse[left][a] * inverse[right][b] * delta_einstein[a, b]
                for a in range(4)
                for b in range(4)
            )
            delta_g[f"G_{left}{right}"] = str(raised)
        body = {
            "coordinate_atom": atom,
            "coordinate_column": column,
            "seed": {
                "dg_symmetric_pair": [seed_left, seed_right],
                "dg_value": str(weight),
                "dP": "0",
                "dS": "0",
                "dv": "0",
            },
            "delta_v": "0",
            "delta_H": delta_h,
            "delta_G_upper": delta_g,
            "domain": (
                "pointwise orthonormal g_ab=eta_ab with arbitrary symmetric P_kab and "
                "commuting symmetric S_klab coordinate jets"
            ),
            "all_20_covariant_tangent_components_materialized": True,
        }
        packets[atom] = {**body, "content_sha256": _sha(body)}
    for field in range(10):
        packets[f"p0[{field}]"] = packets[f"q[{field}]"]
    program_body = {
        "schema_version": "sigma-q-metric-indexed-two-jet-tangent-program-1.0",
        "primitive_symbols": sorted(str(symbol) for symbol in primitive["symbols"]),
        "primitive_symbol_count": 144,
        "frame_constraint": "g_ab=u^ab=diag(-1,1,1,1) at the evaluation point",
        "fixed_tangent_slots": ["dP=0", "dS=0", "dv=0"],
        "materialized_unexpanded_exact_scalar_values": 200,
        "tangent_packet_count": 10,
        "no_flat_reference_jet_specialization": True,
        "arbitrary_consistent_first_and_second_metric_jets": True,
        "factor_terms_or_expand_applied": False,
    }
    return packets, {**program_body, "content_sha256": _sha(program_body)}


def _base_body(
    root: Path,
    config_path: Path,
    config: Mapping[str, Any],
    values: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    names = (
        "P0_METRIC",
        "_coordinate_primitives",
        "_exact_p0_tangents",
        "_validate_projection",
        "SOURCE_PATH",
        "CONFIG_PATH",
        "TEST_PATH",
        "RESULT_SCHEMA",
        "CAMPAIGN_ID",
    )
    old = {name: getattr(base, name) for name in names}
    updates = {
        "P0_METRIC": Q_METRIC,
        "_coordinate_primitives": _coordinate_primitives,
        "_exact_p0_tangents": _exact_q_tangents,
        "_validate_projection": _validate_projection,
        "SOURCE_PATH": SOURCE_PATH,
        "CONFIG_PATH": CONFIG_PATH,
        "TEST_PATH": TEST_PATH,
        "RESULT_SCHEMA": RESULT_SCHEMA,
        "CAMPAIGN_ID": CAMPAIGN_ID,
    }
    base._generic_packets.cache_clear()
    base._candidate_manifests_cached.cache_clear()
    try:
        for name, value in updates.items():
            setattr(base, name, value)
        return base._expected_body(root, config_path, config, values)
    finally:
        base._generic_packets.cache_clear()
        base._candidate_manifests_cached.cache_clear()
        for name, value in old.items():
            setattr(base, name, value)


def _expected_body_uncached(
    root: Path,
    config_path: Path,
    config: Mapping[str, Any],
    values: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    _validate_projection(values["projection"])
    tangent_by_atom, tangent_program = _exact_q_tangents()
    tangents = [tangent_by_atom[atom] for atom, _, _ in Q_METRIC]
    targets = base._target_atoms(values["predecessor"])
    coefficients = base._candidate_coefficients(values["predecessor"])
    return {
        "schema_version": RESULT_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "decision": "pass_q_metric_200_exact_tangent_scalars_leaf_composition_blocked",
        "decision_counts": {"pass": 12, "blocked": 0, "reject": 0},
        "downstream_leaf_counts": {"pass": 0, "blocked": 12, "reject": 0},
        "first_blocker": (
            "compose_the_200_unexpanded_q_tangent_scalars_through_the_20_live_D1_targets_"
            "and_canonicalize_316800_exact_leaf_roots_with_bounded_checkpoints"
        ),
        "tangent_checkpoint_theorem": {
            "name": "pointwise_frame_arbitrary_two_jet_q_metric_tangent_checkpoint",
            "exact_result": (
                "All ten metric-value directions are expanded through the registered indexed "
                "inverse, connection, curvature, Hessian, and raised-Einstein product rules. "
                "The resulting 200 exact scalar expressions retain all 144 registered P/S/v "
                "primitives without flat or first-jet specialization."
            ),
            "domain": tangent_program["frame_constraint"],
            "boundary": (
                "No live A/B/C leaf root is admitted by this checkpoint. The predecessor's "
                "eight-slot remainder is also not promoted to a unique-coordinate closure."
            ),
        },
        "coordinate_tangent_program": tangent_program,
        "coordinate_tangent_packets": tangents,
        "coordinate_tangent_manifest_sha256": _sha([row["content_sha256"] for row in tangents]),
        "resumable_leaf_composition_contract": {
            "target_atoms": list(targets),
            "target_atom_manifest_sha256": _sha(list(targets)),
            "candidate_ids": sorted(coefficients),
            "candidate_coefficient_manifest_sha256": _sha(coefficients),
            "q_tangent_atoms": [row[0] for row in Q_METRIC],
            "target_direction_pairs_per_candidate": 200,
            "leaf_roots_per_target_pair": 132,
            "planned_leaf_roots_per_candidate": 26400,
            "planned_leaf_roots_all_candidates": 316800,
            "checkpoint_unit": "one_candidate_by_one_q_atom_equals_2640_exact_leaf_roots",
            "checkpoint_units": 120,
            "canonicalization": "exact_sympy_srepr_hash_after_sparse_live_leaf_composition",
        },
        "exact_controls": {
            "formal_slot_alias_correction": {
                "formal_slots": 145,
                "duplicate_excess": 2,
                "unique_columns_before": 143,
                "duplicate_atoms": ["s11[10]", "s22[10]"],
            },
            "off_diagonal_seed_normalization": {
                "atom": "q[1]",
                "exact_value": "sqrt(2)/2",
                "replace_by_one_rejected": True,
            },
            "inverse_metric_product_rule": {
                "formula": "du=-u*dg*u",
                "omit_term_rejected": True,
            },
            "second_jet_dependence": {
                "primitive_symbols": 144,
                "flat_or_first_jet_only_substitution_rejected": True,
            },
            "leaf_promotion_without_composition": {"rejected": True},
        },
        "gate_counts": {
            "selected_candidates": 12,
            "predecessor_formal_registered_slots": 145,
            "predecessor_duplicate_formal_slot_excess": 2,
            "predecessor_unique_registered_coordinate_columns": 143,
            "predecessor_missing_unique_coordinate_columns": 10,
            "q_metric_tangent_packets": 10,
            "materialized_q_tangent_scalar_values": 200,
            "coordinate_jet_primitive_symbols": 144,
            "D1_target_atoms_bound_for_resume": 20,
            "planned_leaf_roots_all_candidates": 316800,
            "materialized_leaf_roots_all_candidates": 0,
            "unique_registered_coordinate_columns_after": 143,
            "remaining_unique_coordinate_columns_without_A_B_C_leaf_authority": 10,
            "registered_D2_entries_per_candidate_before": 5324,
            "new_ordered_D2_roots_registered_per_candidate": 0,
            "registered_D2_entries_per_candidate_after": 5324,
            "full_D2_entries_per_candidate": 257499,
        },
        "claim_seals": {
            "all_153_unique_coordinate_leaf_authorities_registered": False,
            "145_predecessor_slots_are_145_unique_columns": False,
            "two_duplicate_formal_slots_reconciled_for_accounting": True,
            "complete_q_metric_tangent_family_materialized": True,
            "complete_q_metric_leaf_family_registered": False,
            "all_316800_leaf_roots_materialized": False,
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
        },
        "data_seals": _copy(SEALS),
        "scope": (
            "exact q-metric two-jet tangent checkpoint and resumable leaf plan; q leaf closure, "
            "153-unique-column closure, D2 replay, H7, rejection, and observations remain closed"
        ),
    }


def _expected_body(
    root: Path,
    config_path: Path,
    config: Mapping[str, Any],
    values: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    key = (
        str(root),
        _production_sha(config_path),
        _production_sha(_inside(root, SOURCE_PATH)),
        _production_sha(_inside(root, TEST_PATH)),
        str(values["predecessor"]["content_sha256"]),
        str(values["projection"]["content_sha256"]),
        str(values["live_abc"]["content_sha256"]),
    )
    if key not in _BODY_CACHE:
        _BODY_CACHE[key] = _expected_body_uncached(root, config_path, config, values)
    return _copy(_BODY_CACHE[key])


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
        raise QMetricLowerLeafAuthorityError("checked result changed")


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
