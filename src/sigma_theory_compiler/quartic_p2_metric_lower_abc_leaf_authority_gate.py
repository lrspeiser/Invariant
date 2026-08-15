"""Register exact lower p2 metric A/B/C leaves by spatially permuting p1."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping, Sequence
from functools import cache
from pathlib import Path
from typing import Any

import sympy as sp

from . import quartic_p0_metric_lower_abc_leaf_authority_gate as base
from . import quartic_p1_metric_lower_abc_leaf_authority_gate as prior
from .quartic_remaining_scalar_hessian_abc_leaf_authority_gate import (
    _content_sha,
    _production_sha,
    _sha,
)

CONFIG_SCHEMA = "sigma-quartic-p2-metric-lower-abc-leaf-authority-config-1.0"
RESULT_SCHEMA = "sigma-quartic-p2-metric-lower-abc-leaf-authority-gate-1.0"
CAMPAIGN_ID = "quartic-p2-metric-lower-abc-leaf-authority-001"
STEM = "quartic_p2_metric_lower_abc_leaf_authority_gate"
SLUG = "quartic-p2-metric-lower-abc-leaf-authority-gate"
CONFIG_PATH = f"configs/backgrounds/{STEM}.json"
SOURCE_PATH = f"src/sigma_theory_compiler/{STEM}.py"
TEST_PATH = f"tests/test_{STEM}.py"
OUTPUT_PATH = f"runs/physics-language/{SLUG}/campaign.json"
CONFIG_PRODUCTION_SHA256 = "4d11a30189f1d570b97d18d9b3e511c9d3f3b2e0db95edd6d181d916f2ed1466"
TEST_PRODUCTION_SHA256 = "059801a2ff073bcfd4117ce550065e46c9a9a350a965497e10ae0f9da24f5e13"
P2_METRIC = tuple(
    (f"p2[{field}]", 32 + field, pair) for field, pair in enumerate(base.SYMMETRIC_PAIRS)
)
PAIR_TO_FIELD = {pair: field for field, pair in enumerate(base.SYMMETRIC_PAIRS)}
CONTRACT = {
    "candidate_count": 12,
    "previous_registered_coordinate_columns": 125,
    "previous_missing_coordinate_columns": 28,
    "new_p2_metric_columns": 10,
    "d1_target_atoms": 20,
    "leaf_roots_per_target_pair": 132,
    "new_leaf_roots_per_candidate": 26400,
    "new_leaf_roots_all_candidates": 316800,
    "registered_D2_entries_per_candidate": 5324,
    "full_D2_entries_per_candidate": 257499,
}
POLICIES = {
    "family_selection": (
        "complete_p2_metric_first_jet_family_by_exact_spatial_permutation_of_p1_tangent"
    ),
    "coordinate_jet_domain": (
        "registered_pointwise_orthonormal_frame_with_arbitrary_symmetric_coordinate_first_jet"
    ),
    "zero_admission": "only_live_symbolic_derivative_exact_zero",
    "D2_promotion": "forbidden_without_separate_closed_D1_arithmetic_DAG_replay",
    "global_H7": "fail_closed",
    "candidate_rejection": "forbidden",
}
SEALS = dict(base.SEALS)
_BODY_CACHE: dict[tuple[str, ...], dict[str, Any]] = {}


class P2MetricLowerLeafAuthorityError(ValueError):
    """A lower p2 tangent, spatial permutation, or sealed boundary changed."""


def _inside(root: Path, relative: str) -> Path:
    if not relative or "\\" in relative:
        raise P2MetricLowerLeafAuthorityError("p2 lower path is not portable")
    path = (root / relative).resolve()
    if path != root and root not in path.parents:
        raise P2MetricLowerLeafAuthorityError("p2 lower path escapes root")
    return path


def _copy(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, separators=(",", ":")))


def _validate_config(value: Mapping[str, Any], path: Path) -> None:
    if _production_sha(path) != CONFIG_PRODUCTION_SHA256:
        raise P2MetricLowerLeafAuthorityError("p2 lower config production bytes changed")
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
        raise P2MetricLowerLeafAuthorityError("p2 lower config contract changed")


def _load_inputs(root: Path, config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    try:
        predecessor = base._load_bundle(root, config["predecessor"], artifact_hash=False)
        projection = base._load_bundle(
            root, config["direct_evidence"]["lower_projection"], artifact_hash=True
        )
        live = base._load_bundle(root, config["direct_evidence"]["live_abc"], artifact_hash=True)
    except base.P0MetricLowerLeafAuthorityError as error:
        raise P2MetricLowerLeafAuthorityError(str(error)) from error
    if (
        predecessor.get("decision")
        != "pass_p1_metric_10_column_nonlinear_tangents_316800_exact_leaf_roots_D2_blocked"
        or predecessor.get("gate_counts", {}).get(
            "remaining_coordinate_columns_without_A_B_C_leaf_authority"
        )
        != 28
        or predecessor.get("gate_counts", {}).get("registered_D2_entries_per_candidate_after")
        != 5324
        or projection.get("decision")
        != "pass_all_54_lower_covariant_projections_D2_count_preserved"
        or live.get("status")
        != "pass_all_12_complete_unspecialized_principal_source_jacobians_remainder_fail_closed"
    ):
        raise P2MetricLowerLeafAuthorityError("p2 lower predecessor boundary changed")
    return {"predecessor": predecessor, "projection": projection, "live_abc": live}


def _validate_projection(projection: Mapping[str, Any]) -> None:
    records = {
        str(row["coordinate_atom"]): row
        for row in projection.get("lower_projection_registry", [])
        if row.get("family") == "p_metric" and row.get("derivative_index") == 2
    }
    if set(records) != {row[0] for row in P2_METRIC}:
        raise P2MetricLowerLeafAuthorityError("p2 lower projection family changed")
    for atom, column, pair in P2_METRIC:
        row = records[atom]
        seed = row.get("tangent_seed", {})
        if (
            row.get("coordinate_column") != column
            or seed.get("dP_derivative") != 2
            or seed.get("dP_symmetric_pair") != list(pair)
            or seed.get("dP_value") != ("1" if pair[0] == pair[1] else "sqrt(2)/2")
            or row.get("exact_projection_registered") is not True
        ):
            raise P2MetricLowerLeafAuthorityError("p2 lower projection seed changed")


def _permute_index(index: int) -> int:
    return 2 if index == 1 else 1 if index == 2 else index


def _permute_pair(pair: tuple[int, int]) -> tuple[int, int]:
    return tuple(sorted((_permute_index(pair[0]), _permute_index(pair[1]))))


@cache
def _primitive_permutation() -> tuple[dict[sp.Symbol, sp.Symbol], dict[str, sp.Symbol]]:
    primitives = tuple(base._coordinate_primitives()["symbols"])
    by_name = {str(symbol): symbol for symbol in primitives}
    substitution = {}
    for symbol in primitives:
        name = str(symbol)
        gradient = re.fullmatch(r"v_([0-3])", name)
        metric_first = re.fullmatch(r"P([0-3])_([0-3])([0-3])", name)
        if gradient:
            target = f"v_{_permute_index(int(gradient.group(1)))}"
        elif metric_first:
            derivative = _permute_index(int(metric_first.group(1)))
            pair = _permute_pair((int(metric_first.group(2)), int(metric_first.group(3))))
            target = f"P{derivative}_{pair[0]}{pair[1]}"
        else:
            raise P2MetricLowerLeafAuthorityError("p2 primitive symbol family changed")
        substitution[symbol] = by_name[target]
    if len(substitution) != 44 or set(substitution.values()) != set(primitives):
        raise P2MetricLowerLeafAuthorityError("p2 spatial permutation is not bijective")
    return substitution, by_name


_P1_METRIC_ORIGINAL = prior.P1_METRIC
_P1_TANGENTS_ORIGINAL = prior._exact_p1_tangents


@cache
def _exact_p2_tangents() -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    # The p2 body temporarily installs its own family into the p1 module so the
    # inherited composer can be reused.  Restore the frozen p1 family while
    # constructing the permutation source packets; otherwise the original p1
    # helper sees p2 atom names and cannot create its p1 aliases.
    active_metric = prior.P1_METRIC
    active_tangents = prior._exact_p1_tangents
    prior.P1_METRIC = _P1_METRIC_ORIGINAL
    prior._exact_p1_tangents = _P1_TANGENTS_ORIGINAL
    try:
        p1_packets, _ = _P1_TANGENTS_ORIGINAL()
    finally:
        prior.P1_METRIC = active_metric
        prior._exact_p1_tangents = active_tangents
    substitution, locals_map = _primitive_permutation()
    packets: dict[str, dict[str, Any]] = {}
    for atom, column, pair in P2_METRIC:
        source_field = PAIR_TO_FIELD[_permute_pair(pair)]
        source = p1_packets[f"p1[{source_field}]"]
        delta_h = {}
        delta_g = {}
        for left, right in base.SYMMETRIC_PAIRS:
            source_pair = _permute_pair((left, right))
            for prefix, source_values, target_values in (
                ("H", source["delta_H"], delta_h),
                ("G", source["delta_G_upper"], delta_g),
            ):
                source_label = f"{prefix}_{source_pair[0]}{source_pair[1]}"
                expression = sp.sympify(source_values[source_label], locals=locals_map)
                target_values[f"{prefix}_{left}{right}"] = str(
                    base._norm(expression.xreplace(substitution))
                )
        weight = sp.Integer(1) if pair[0] == pair[1] else sp.sqrt(2) / 2
        body = {
            "coordinate_atom": atom,
            "coordinate_column": column,
            "seed": {
                "dP_derivative": 2,
                "dP_symmetric_pair": list(pair),
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
            "spatial_permutation_source_atom": f"p1[{source_field}]",
            "spatial_permutation": "swap_indices_1_and_2",
            "all_20_covariant_tangent_components_materialized": True,
        }
        packets[atom] = {**body, "content_sha256": _sha(body)}
    for field in range(10):
        packets[f"p0[{field}]"] = packets[f"p2[{field}]"]
    program_body = {
        "schema_version": "sigma-p2-metric-nonlinear-coordinate-tangent-program-1.0",
        "primitive_symbols": sorted(locals_map),
        "frame_constraint": "g_ab=u^ab=diag(-1,1,1,1) at the evaluation point",
        "exact_spatial_permutation": "swap_indices_1_and_2_from_p1",
        "permutation_domain": "eta_11=eta_22=1; no time-space permutation used",
        "formulas": [
            "P1_kab maps to P2_pi(k)pi(a)pi(b) under pi=(1 2)",
            "v_k maps to v_pi(k)",
            "deltaH_ab maps to deltaH_pi(a)pi(b)",
            "deltaG^ab maps to deltaG^pi(a)pi(b)",
            "all p1 connection, differentiated-connection, Ricci, Hessian, and Einstein formulas commute with this spatial permutation",
        ],
        "tangent_packet_count": 10,
        "materialized_scalar_values": 200,
        "no_flat_reference_specialization": True,
        "arbitrary_background_pointwise_local_frame": True,
    }
    return packets, {**program_body, "content_sha256": _sha(program_body)}


def _base_body_uncached(
    root: Path,
    config_path: Path,
    config: Mapping[str, Any],
    values: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    old_metric = prior.P1_METRIC
    old_tangents = prior._exact_p1_tangents
    old_projection = prior._validate_projection
    prior.P1_METRIC = P2_METRIC
    prior._exact_p1_tangents = _exact_p2_tangents
    prior._validate_projection = _validate_projection
    try:
        return prior._expected_body(root, config_path, config, values)
    finally:
        prior.P1_METRIC = old_metric
        prior._exact_p1_tangents = old_tangents
        prior._validate_projection = old_projection


def _expected_body_uncached(
    root: Path,
    config_path: Path,
    config: Mapping[str, Any],
    values: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    body = _base_body_uncached(root, config_path, config, values)
    body["schema_version"] = RESULT_SCHEMA
    body["campaign_id"] = CAMPAIGN_ID
    body["decision"] = (
        "pass_p2_metric_10_column_nonlinear_tangents_316800_exact_leaf_roots_D2_blocked"
    )
    body["first_blocker"] = (
        "independently_replay_the_D1_arithmetic_DAG_for_p2_metric_columns_before_"
        "admitting_ordered_D2_roots_then_materialize_the_remaining_18_columns"
    )
    body["leaf_authority_theorem"].update(
        {
            "name": "pointwise_frame_arbitrary_first_jet_p2_metric_to_live_A_B_C_chain",
            "exact_result": (
                "Spatial permutation covariance of the pointwise orthonormal frame maps the "
                "complete p1 tangent family to all ten p2 metric first-jet directions. The "
                "result is composed through every live A/B/C leaf used by the 20 registered "
                "D1 targets."
            ),
        }
    )
    for manifest in body["candidate_manifests"]:
        manifest["derivative_coordinate_columns"] = list(range(32, 42))
        manifest["candidate_decision"] = "pass_p2_metric_lower_leaf_authority_D2_replay_blocked"
        manifest["content_sha256"] = _content_sha(manifest)
    body["candidate_manifest_sha256"] = _sha(
        [row["content_sha256"] for row in body["candidate_manifests"]]
    )
    body["exact_controls"] = {
        "off_diagonal_seed_normalization": {
            "atom": "p2[1]",
            "exact_value": "sqrt(2)/2",
            "replace_by_one_rejected": True,
        },
        "derivative_index_distinction": {
            "p2_seed_derivative": 2,
            "replace_by_p1_seed_rejected": True,
        },
        "spatial_permutation_authority": {
            "permutation": "swap_indices_1_and_2",
            "eta_invariant": True,
            "time_space_swap_not_used": True,
            "primitive_symbol_bijection_count": 44,
        },
        "connection_dependence": {
            "atom": "p2[0]",
            "delta_H_contains_live_scalar_gradient": True,
            "infer_zero_rejected": True,
        },
        "flat_reference_substitution_for_general_claim": {"rejected": True},
        "promote_without_D1_replay": {"rejected": True},
    }
    counts = body["gate_counts"]
    counts.pop("new_p1_metric_coordinate_columns")
    counts.update(
        {
            "previous_registered_coordinate_columns": 125,
            "previous_missing_coordinate_columns": 28,
            "new_p2_metric_coordinate_columns": 10,
            "registered_coordinate_columns_after": 135,
            "remaining_coordinate_columns_without_A_B_C_leaf_authority": 18,
        }
    )
    seals = body["claim_seals"]
    seals.pop("complete_p1_metric_lower_family_registered")
    seals.pop("remaining_28_coordinate_columns_registered")
    seals.update(
        {
            "complete_p2_metric_lower_family_registered": True,
            "remaining_18_coordinate_columns_registered": False,
        }
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
        "exact p2 metric lower-coordinate nonlinear tangent and live leaf authority; "
        "remaining lower columns, ordered D2 replay, complete tensor, H7, rejection, "
        "and observations remain closed"
    )
    return body


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
        raise P2MetricLowerLeafAuthorityError("checked result changed")


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
