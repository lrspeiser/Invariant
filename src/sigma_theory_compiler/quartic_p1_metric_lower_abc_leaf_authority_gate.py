"""Register exact lower p1 metric A/B/C leaves through the nonlinear jet map."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from functools import cache
from pathlib import Path
from typing import Any

import sympy as sp

from . import quartic_p0_metric_lower_abc_leaf_authority_gate as prior
from .quartic_remaining_scalar_hessian_abc_leaf_authority_gate import (
    _content_sha,
    _production_sha,
    _sha,
)

CONFIG_SCHEMA = "sigma-quartic-p1-metric-lower-abc-leaf-authority-config-1.0"
RESULT_SCHEMA = "sigma-quartic-p1-metric-lower-abc-leaf-authority-gate-1.0"
CAMPAIGN_ID = "quartic-p1-metric-lower-abc-leaf-authority-001"
STEM = "quartic_p1_metric_lower_abc_leaf_authority_gate"
SLUG = "quartic-p1-metric-lower-abc-leaf-authority-gate"
CONFIG_PATH = f"configs/backgrounds/{STEM}.json"
SOURCE_PATH = f"src/sigma_theory_compiler/{STEM}.py"
TEST_PATH = f"tests/test_{STEM}.py"
OUTPUT_PATH = f"runs/physics-language/{SLUG}/campaign.json"
CONFIG_PRODUCTION_SHA256 = "899114fe6559c5fd10d307fd10502a154df4de86775a520b7f0455a9ef61a03b"
TEST_PRODUCTION_SHA256 = "db1fbd65fc9ffac195897a19d2ed69796e240506245a92907496b6198eac0a14"
P1_METRIC = tuple(
    (f"p1[{field}]", 21 + field, pair) for field, pair in enumerate(prior.SYMMETRIC_PAIRS)
)
CONTRACT = {
    "candidate_count": 12,
    "previous_registered_coordinate_columns": 115,
    "previous_missing_coordinate_columns": 38,
    "new_p1_metric_columns": 10,
    "d1_target_atoms": 20,
    "leaf_roots_per_target_pair": 132,
    "new_leaf_roots_per_candidate": 26400,
    "new_leaf_roots_all_candidates": 316800,
    "registered_D2_entries_per_candidate": 5324,
    "full_D2_entries_per_candidate": 257499,
}
POLICIES = {
    "family_selection": (
        "complete_p1_metric_first_jet_family_by_exact_nonlinear_coordinate_tangent"
    ),
    "coordinate_jet_domain": (
        "registered_pointwise_orthonormal_frame_with_arbitrary_symmetric_coordinate_first_jet"
    ),
    "zero_admission": "only_live_symbolic_derivative_exact_zero",
    "D2_promotion": "forbidden_without_separate_closed_D1_arithmetic_DAG_replay",
    "global_H7": "fail_closed",
    "candidate_rejection": "forbidden",
}
SEALS = dict(prior.SEALS)


class P1MetricLowerLeafAuthorityError(ValueError):
    """A lower p1 tangent, live leaf derivative, or sealed boundary changed."""


def _inside(root: Path, relative: str) -> Path:
    if not relative or "\\" in relative:
        raise P1MetricLowerLeafAuthorityError("p1 lower path is not portable")
    path = (root / relative).resolve()
    if path != root and root not in path.parents:
        raise P1MetricLowerLeafAuthorityError("p1 lower path escapes root")
    return path


def _copy(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, separators=(",", ":")))


def _validate_config(value: Mapping[str, Any], path: Path) -> None:
    if _production_sha(path) != CONFIG_PRODUCTION_SHA256:
        raise P1MetricLowerLeafAuthorityError("p1 lower config production bytes changed")
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
        raise P1MetricLowerLeafAuthorityError("p1 lower config contract changed")


def _load_inputs(root: Path, config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    try:
        predecessor = prior._load_bundle(root, config["predecessor"], artifact_hash=False)
        projection = prior._load_bundle(
            root, config["direct_evidence"]["lower_projection"], artifact_hash=True
        )
        live = prior._load_bundle(root, config["direct_evidence"]["live_abc"], artifact_hash=True)
    except prior.P0MetricLowerLeafAuthorityError as error:
        raise P1MetricLowerLeafAuthorityError(str(error)) from error
    if (
        predecessor.get("decision")
        != "pass_p0_metric_10_column_nonlinear_tangents_316800_exact_leaf_roots_D2_blocked"
        or predecessor.get("gate_counts", {}).get(
            "remaining_coordinate_columns_without_A_B_C_leaf_authority"
        )
        != 38
        or predecessor.get("gate_counts", {}).get("registered_D2_entries_per_candidate_after")
        != 5324
        or projection.get("decision")
        != "pass_all_54_lower_covariant_projections_D2_count_preserved"
        or live.get("status")
        != "pass_all_12_complete_unspecialized_principal_source_jacobians_remainder_fail_closed"
    ):
        raise P1MetricLowerLeafAuthorityError("p1 lower predecessor boundary changed")
    return {"predecessor": predecessor, "projection": projection, "live_abc": live}


def _validate_projection(projection: Mapping[str, Any]) -> None:
    records = {
        str(row["coordinate_atom"]): row
        for row in projection.get("lower_projection_registry", [])
        if row.get("family") == "p_metric" and row.get("derivative_index") == 1
    }
    if set(records) != {row[0] for row in P1_METRIC}:
        raise P1MetricLowerLeafAuthorityError("p1 lower projection family changed")
    for atom, column, pair in P1_METRIC:
        row = records[atom]
        seed = row.get("tangent_seed", {})
        if (
            row.get("coordinate_column") != column
            or seed.get("dP_derivative") != 1
            or seed.get("dP_symmetric_pair") != list(pair)
            or seed.get("dP_value") != ("1" if pair[0] == pair[1] else "sqrt(2)/2")
            or row.get("exact_projection_registered") is not True
        ):
            raise P1MetricLowerLeafAuthorityError("p1 lower projection seed changed")


@cache
def _exact_p1_tangents() -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    primitive = prior._coordinate_primitives()
    inverse, metric = primitive["inverse"], primitive["metric"]
    first, gradient = primitive["first"], primitive["gradient"]
    packets: dict[str, dict[str, Any]] = {}
    for atom, column, (seed_left, seed_right) in P1_METRIC:
        delta = [sp.zeros(4) for _ in range(4)]
        weight = sp.Integer(1) if seed_left == seed_right else sp.sqrt(2) / 2
        delta[1][seed_left, seed_right] = weight
        delta[1][seed_right, seed_left] = weight
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
                dricci[left, right] = prior._norm(
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
        trace = prior._norm(
            sum(
                inverse[left][right] * dricci[left, right]
                for left in range(4)
                for right in range(4)
            )
        )
        delta_h = {}
        delta_g = {}
        for left, right in prior.SYMMETRIC_PAIRS:
            h_value = prior._norm(
                -sum(dgamma[upper][left][right] * gradient[upper] for upper in range(4))
            )
            g_value = prior._norm(
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
                "dP_derivative": 1,
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
    for field in range(10):
        packets[f"p0[{field}]"] = packets[f"p1[{field}]"]
    program_body = {
        "schema_version": "sigma-p1-metric-nonlinear-coordinate-tangent-program-1.0",
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


def _base_body(
    root: Path,
    config_path: Path,
    config: Mapping[str, Any],
    values: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    old_metric = prior.P0_METRIC
    old_tangents = prior._exact_p0_tangents
    old_projection = prior._validate_projection
    prior.P0_METRIC = P1_METRIC
    prior._exact_p0_tangents = _exact_p1_tangents
    prior._validate_projection = _validate_projection
    prior._generic_packets.cache_clear()
    prior._candidate_manifests_cached.cache_clear()
    try:
        return prior._expected_body(root, config_path, config, values)
    finally:
        prior.P0_METRIC = old_metric
        prior._exact_p0_tangents = old_tangents
        prior._validate_projection = old_projection
        prior._generic_packets.cache_clear()
        prior._candidate_manifests_cached.cache_clear()


def _expected_body(
    root: Path,
    config_path: Path,
    config: Mapping[str, Any],
    values: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    body = _base_body(root, config_path, config, values)
    body["schema_version"] = RESULT_SCHEMA
    body["campaign_id"] = CAMPAIGN_ID
    body["decision"] = (
        "pass_p1_metric_10_column_nonlinear_tangents_316800_exact_leaf_roots_D2_blocked"
    )
    body["first_blocker"] = (
        "independently_replay_the_D1_arithmetic_DAG_for_p1_metric_columns_before_"
        "admitting_ordered_D2_roots_then_materialize_the_remaining_28_columns"
    )
    body["leaf_authority_theorem"].update(
        {
            "name": "pointwise_frame_arbitrary_first_jet_p1_metric_to_live_A_B_C_chain",
            "exact_result": (
                "All ten p1 metric first-jet coordinate directions are mapped through exact "
                "connection, differentiated-connection, Ricci, Hessian, and raised Einstein "
                "tangents, then through every live A/B/C leaf used by the 20 registered D1 "
                "targets."
            ),
        }
    )
    for manifest in body["candidate_manifests"]:
        manifest["derivative_coordinate_columns"] = list(range(21, 31))
        manifest["candidate_decision"] = "pass_p1_metric_lower_leaf_authority_D2_replay_blocked"
        manifest["content_sha256"] = _content_sha(manifest)
    body["candidate_manifest_sha256"] = _sha(
        [row["content_sha256"] for row in body["candidate_manifests"]]
    )
    body["exact_controls"] = {
        "off_diagonal_seed_normalization": {
            "atom": "p1[1]",
            "exact_value": "sqrt(2)/2",
            "replace_by_one_rejected": True,
        },
        "derivative_index_distinction": {
            "p1_seed_derivative": 1,
            "replace_by_p0_seed_rejected": True,
        },
        "connection_dependence": {
            "atom": "p1[0]",
            "delta_H_contains_live_scalar_gradient": True,
            "infer_zero_rejected": True,
        },
        "inverse_first_jet_product_rule": {
            "formula": "deltaU_1^ab=-u^ac deltaP_1_cd u^db",
            "omit_term_rejected": True,
        },
        "flat_reference_substitution_for_general_claim": {"rejected": True},
        "promote_without_D1_replay": {"rejected": True},
    }
    counts = body["gate_counts"]
    counts.pop("new_p0_metric_coordinate_columns")
    counts.update(
        {
            "previous_registered_coordinate_columns": 115,
            "previous_missing_coordinate_columns": 38,
            "new_p1_metric_coordinate_columns": 10,
            "registered_coordinate_columns_after": 125,
            "remaining_coordinate_columns_without_A_B_C_leaf_authority": 28,
        }
    )
    seals = body["claim_seals"]
    seals.pop("complete_p0_metric_lower_family_registered")
    seals.pop("remaining_38_coordinate_columns_registered")
    seals.update(
        {
            "complete_p1_metric_lower_family_registered": True,
            "remaining_28_coordinate_columns_registered": False,
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
        "exact p1 metric lower-coordinate nonlinear tangent and live leaf authority; "
        "remaining lower columns, ordered D2 replay, complete tensor, H7, rejection, "
        "and observations remain closed"
    )
    return body


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
        raise P1MetricLowerLeafAuthorityError("checked result changed")


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
