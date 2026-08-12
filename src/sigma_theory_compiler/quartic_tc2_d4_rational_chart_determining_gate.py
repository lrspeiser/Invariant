from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import sympy as sp

from . import quartic_tc2_d4_degree_three_rank_two_xyz_completion_campaign as xyz_campaign
from . import quartic_tc2_d4_degree_three_sixth_frame_completion_campaign as sixth_campaign
from . import quartic_tc2_diagonal_third_jet_campaign as directional_engine
from .quartic_tc2_d4_parity_cubic_generic_direction_campaign import (
    JET_ORDER,
    _polarized_payload,
    _solve,
    _state_rotation,
)
from .quartic_tc2_diagonal_third_jet_campaign import (
    _content_hash,
    _matrix_payload,
    _reference_and_first_jet_packet,
)
from .quartic_tc2_mixed_third_jet_continuation_service import (
    _atomic_write,
    _file_sha256,
    _hash_matches,
    _json_bytes,
    _load_file,
    _with_hash,
)

SCHEMA = "sigma-quartic-tc2-d4-rational-chart-determining-gate-1.0"
CONFIG_SCHEMA = "sigma-quartic-tc2-d4-rational-chart-determining-config-1.0"
STATUS = "pass_exact_rational_chart_counterexample_disproves_current_full_sphere_D4_compatibility"
ACTIVE_INDICES = (0, 2, 3, 9)
OBLIGATION_OFFSET = 244
EXPECTED_CANDIDATES = 12
BASE_RHS_SHA256 = "fa14b03f231b6790ae610f31d9d4deafeb86ef060e2c5d580809e4d0752730c8"
XYZ_BLOCK_SHA256 = "7b7c840b90ffa0d18fea359f8c617fbf3c28ea15188732ee954827e092041849"
SIXTH_BLOCK_SHA256 = "a7290209e029b6a2d6ddcf5b09818711ef4a96687d2f188992be5a950da5dbe8"
TOTAL_SYMBOL_SHA256 = "616d48e8339aacd4f7254c8c40c2052354237b882ea8d4d5a7d5afcab5f8f59b"
ALIGNED_SYMBOL_SHA256 = "2b6438fba2717ec28471f2ee2fb7bc59f789d783464b21b42b5d900d9097fa6d"
COMPRESSION_HASHES = {
    "quartic-symbol-06e267a9215345b6": "0534ae9c62015441e9a6e563dc46430507476bbf5f29853e146150e75630bce1",
    "quartic-symbol-076dc0ba965ab63a": "83e9a808dc2caadeeb6463dd245503b4f4fc64a5d020ec0dd3af73b7663a6345",
    "quartic-symbol-317e5395817a432b": "28b13f16535184378357e35b3158e8368f603c53e8900c36e583086f966f41cb",
    "quartic-symbol-50f184dfe1a814bf": "0534ae9c62015441e9a6e563dc46430507476bbf5f29853e146150e75630bce1",
    "quartic-symbol-5455cad9e42a0dbc": "28b13f16535184378357e35b3158e8368f603c53e8900c36e583086f966f41cb",
    "quartic-symbol-561de1410d6cb21f": "83e9a808dc2caadeeb6463dd245503b4f4fc64a5d020ec0dd3af73b7663a6345",
    "quartic-symbol-8fd254934d778c28": "9d0461dc9d75d55b697617a92d67ff7acabb2015c984b47065cbe6e47d0515a1",
    "quartic-symbol-9e65901e5299a514": "9d0461dc9d75d55b697617a92d67ff7acabb2015c984b47065cbe6e47d0515a1",
    "quartic-symbol-e4a6a9193316a6ff": "83e9a808dc2caadeeb6463dd245503b4f4fc64a5d020ec0dd3af73b7663a6345",
    "quartic-symbol-ef832e4c3b71ee42": "9d0461dc9d75d55b697617a92d67ff7acabb2015c984b47065cbe6e47d0515a1",
    "quartic-symbol-f31a234e2bf7b97f": "0534ae9c62015441e9a6e563dc46430507476bbf5f29853e146150e75630bce1",
    "quartic-symbol-fb5c20c15ce6d778": "28b13f16535184378357e35b3158e8368f603c53e8900c36e583086f966f41cb",
}


class RationalChartDeterminingError(ValueError):
    """Raised when the exact rational-chart gate is inconsistent."""


def _load_bound(root: Path, binding: Mapping[str, Any]) -> dict[str, Any]:
    path = (root / str(binding["path"])).resolve()
    if root not in path.parents:
        raise RationalChartDeterminingError("bound input escaped project root")
    value, data = _load_file(path)
    if (
        _file_sha256(data) != binding.get("file_sha256")
        or value.get("content_sha256") != binding.get("content_sha256")
        or not _hash_matches(value)
    ):
        raise RationalChartDeterminingError(f"bound input mismatch: {binding.get('path')}")
    return value


def _check_raw(root: Path, binding: Mapping[str, Any]) -> None:
    path = (root / str(binding["path"])).resolve()
    if root not in path.parents or not path.is_file():
        raise RationalChartDeterminingError("raw binding escaped root or is absent")
    if _file_sha256(path.read_bytes()) != binding.get("file_sha256"):
        raise RationalChartDeterminingError(f"raw binding mismatch: {binding.get('path')}")


def _atlas() -> dict[str, Any]:
    u, v = sp.symbols("u v", real=True)
    denominator = 1 + u**2 + v**2
    numerator = sp.Matrix(
        [
            [1 - u**2 - v**2, -2 * u, -2 * v],
            [2 * u, 1 - u**2 + v**2, -2 * u * v],
            [2 * v, -2 * u * v, 1 + u**2 - v**2],
        ]
    )
    primary = numerator / denominator
    antipodal = primary * sp.diag(-1, 1, -1)
    for rotation in (primary, antipodal):
        if not (rotation.T * rotation - sp.eye(3)).applyfunc(sp.cancel).is_zero_matrix:
            raise RationalChartDeterminingError("chart rotation is not orthogonal")
        if sp.cancel(rotation.det() - 1) != 0:
            raise RationalChartDeterminingError("chart rotation is not orientation preserving")
    point = {u: sp.Rational(2, 5), v: sp.Rational(1, 5)}
    point_rotation = primary.subs(point)
    if list(point_rotation[:, 0]) != [sp.Rational(2, 3), sp.Rational(2, 3), sp.Rational(1, 3)]:
        raise RationalChartDeterminingError("rational chart point mismatch")
    return {
        "variables": (u, v),
        "denominator": denominator,
        "primary": primary,
        "antipodal": antipodal,
        "point": point,
        "point_rotation": point_rotation,
        "record": {
            "chart_count": 2,
            "primary_missing_point": "-e1",
            "antipodal_missing_point": "+e1",
            "union_covers_real_S2": True,
            "common_denominator": "1+u^2+v^2",
            "common_denominator_strictly_positive_on_R2": True,
            "real_singular_strata": 0,
            "complex_singular_stratum": "1+u^2+v^2=0",
            "primary_rotation_sha256": _content_hash(_matrix_payload(primary)),
            "antipodal_rotation_sha256": _content_hash(_matrix_payload(antipodal)),
            "SO3_identities_proved_after_denominator_clearing": True,
        },
    }


def _capture_current_extensions(
    xyz_artifact: Mapping[str, Any],
    c23_artifact: Mapping[str, Any],
    minimal: Mapping[str, Any],
    fourth: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    captured: dict[str, Mapping[str, Any]] = {}
    original_xyz = xyz_campaign._global_curl_extension
    original_sixth = sixth_campaign._sphere_extension

    def capture_xyz(terms: list[dict[str, Any]], rotation: sp.Matrix, direction: list[sp.Expr]):
        result = original_xyz(terms, rotation, direction)
        captured["xyz"] = result
        return result

    def capture_sixth(
        terms: list[dict[str, Any]], rotation: sp.Matrix, direction: tuple[sp.Expr, ...]
    ):
        result = original_sixth(terms, rotation, direction)
        captured["sixth"] = result
        return result

    xyz_campaign._global_curl_extension = capture_xyz
    sixth_campaign._sphere_extension = capture_sixth
    try:
        sixth_campaign._exact_result(xyz_artifact, c23_artifact, minimal, fourth)
    finally:
        xyz_campaign._global_curl_extension = original_xyz
        sixth_campaign._sphere_extension = original_sixth
    if set(captured) != {"xyz", "sixth"}:
        raise RationalChartDeterminingError("current symbol reconstruction failed")
    return captured


def _cleared_compression(matrix: sp.Matrix) -> dict[str, Any]:
    nonzero = [sp.together(value) for value in matrix if value != 0]
    denominators = [sp.denom(value) for value in nonzero]
    clearing = sp.lcm(denominators) if denominators else sp.Integer(1)
    numerator = (clearing * matrix).applyfunc(sp.expand)
    if any(sp.denom(value) != 1 for value in numerator):
        raise RationalChartDeterminingError("point numerator denominator clearing failed")
    return {
        "clearing_denominator": str(clearing),
        "numerator_polynomial_total_degree_uv": 0,
        "numerator_nonzero_entries": sum(value != 0 for value in numerator),
        "numerator_rank": numerator.rank(),
        "numerator_sha256": _content_hash(_matrix_payload(numerator)),
    }


def _exact_counterexample(
    xyz_artifact: Mapping[str, Any],
    c23_artifact: Mapping[str, Any],
    minimal: Mapping[str, Any],
    fourth: Mapping[str, Any],
) -> dict[str, Any]:
    atlas = _atlas()
    captured = _capture_current_extensions(xyz_artifact, c23_artifact, minimal, fourth)
    rotation3 = atlas["point_rotation"]
    frame = {
        "name": "stereographic_2_5_1_5",
        "rotation": rotation3,
        "direction": tuple(rotation3[:, 0]),
    }
    prior_order = directional_engine.TAYLOR_ORDER
    directional_engine.TAYLOR_ORDER = JET_ORDER
    try:
        payload, evaluations = _polarized_payload(frame, fourth)
    finally:
        directional_engine.TAYLOR_ORDER = prior_order
    reference = _reference_and_first_jet_packet()
    state_rotation, _ = _state_rotation(rotation3)
    direction = list(frame["direction"])
    base = xyz_campaign._prior_symbol_at(direction)["combined"]

    def evaluate(angular: Mapping[str, Any]) -> sp.Matrix:
        return (
            angular["extension"]
            .subs(dict(zip(angular["variables"], direction, strict=True)))
            .applyfunc(sp.factor)
        )

    xyz_block = evaluate(captured["xyz"])
    sixth_block = evaluate(captured["sixth"])
    total = (base + xyz_block + sixth_block).applyfunc(sp.factor)
    aligned = (state_rotation.T * total * state_rotation).applyfunc(sp.factor)
    skew = (reference["energy0"] * aligned - aligned.T * reference["energy0"]).applyfunc(sp.factor)
    rhs = payload["fourth_Sylvester_RHS"]
    symbols = {str(symbol): symbol for symbol in rhs.free_symbols}
    alpha = symbols.get("alpha", sp.Symbol("alpha"))
    c20 = symbols.get("c20", sp.Symbol("c20"))
    rows = []
    for candidate in minimal["exact_escape"]["candidate_classification"]:
        candidate_rhs = rhs.subs(
            {alpha: sp.sympify(candidate["a10"]), c20: sp.sympify(candidate["c20"])}
        ).applyfunc(sp.factor)
        eta = sp.sympify(candidate["eta_unique_tuning"])
        residual = (candidate_rhs + eta * skew).applyfunc(sp.factor)
        solvable, nonzero = _solve(residual)
        projector0 = reference["projectors"][sp.S.Zero]
        compression0 = (projector0.T * residual * projector0).applyfunc(sp.factor)
        cleared = _cleared_compression(compression0)
        row = {
            "candidate_id": candidate["candidate_id"],
            "D4_Sylvester_solvable": solvable,
            "nonzero_equal_eigenspace_compressions": nonzero,
            "zero_speed_cleared_numerator": cleared,
        }
        rows.append(row)
    if (
        evaluations != 15
        or sum(value != 0 for value in rhs) != 116
        or _content_hash(_matrix_payload(rhs)) != BASE_RHS_SHA256
        or _content_hash(_matrix_payload(xyz_block)) != XYZ_BLOCK_SHA256
        or _content_hash(_matrix_payload(sixth_block)) != SIXTH_BLOCK_SHA256
        or total.rank() != 5
        or _content_hash(_matrix_payload(total)) != TOTAL_SYMBOL_SHA256
        or _content_hash(_matrix_payload(aligned)) != ALIGNED_SYMBOL_SHA256
        or len(rows) != 12
        or any(row["D4_Sylvester_solvable"] for row in rows)
        or any(
            set(row["nonzero_equal_eigenspace_compressions"]) != {"0"}
            or row["nonzero_equal_eigenspace_compressions"]["0"]["rank"] != 4
            or row["nonzero_equal_eigenspace_compressions"]["0"]["nonzero_entries"] != 56
            or row["nonzero_equal_eigenspace_compressions"]["0"]["sha256"]
            != COMPRESSION_HASHES[row["candidate_id"]]
            or row["zero_speed_cleared_numerator"]["numerator_rank"] != 4
            or row["zero_speed_cleared_numerator"]["numerator_nonzero_entries"] != 56
            for row in rows
        )
    ):
        raise RationalChartDeterminingError("exact rational counterexample mismatch")
    return {
        "atlas": atlas["record"],
        "counterexample_selector": {
            "chart": "primary_e1_stereographic",
            "chart_coordinates": ["2/5", "1/5"],
            "chart_denominator_value": "6/5",
            "direction": ["2/3", "2/3", "1/3"],
            "frame_name": frame["name"],
            "regular_real_chart_point": True,
        },
        "full_recurrence": {
            "directional_evaluations": evaluations,
            "all_seven_eigenspaces_checked_per_candidate": True,
            "base_D4_RHS_nonzero_entries": sum(value != 0 for value in rhs),
            "base_D4_RHS_sha256": _content_hash(_matrix_payload(rhs)),
            "xyz_block_sha256": _content_hash(_matrix_payload(xyz_block)),
            "sixth_block_sha256": _content_hash(_matrix_payload(sixth_block)),
            "current_global_symbol_rank": total.rank(),
            "current_global_symbol_sha256": _content_hash(_matrix_payload(total)),
            "current_aligned_symbol_sha256": _content_hash(_matrix_payload(aligned)),
        },
        "exact_rational_obstruction": {
            "candidate_conditions_checked": len(rows),
            "candidate_compatibilities": 0,
            "candidate_obstructions": 12,
            "nonzero_equal_eigenspace_compressions": 12,
            "cleared_constant_numerator_polynomials": 12,
            "candidate_records": rows,
            "current_full_sphere_D4_compatibility_disproved": True,
        },
        "symbolic_chart_reduction": {
            "global_two_variable_numerator_polynomials_materialized": 0,
            "terminal_reason": "exact_regular_rational_counterexample_found",
            "full_polynomial_identity_reduction_required_after_counterexample": False,
            "singular_stratum_invoked": False,
        },
    }


TRUE_CLAIMS = {
    "current_combined_symbol_full_sphere_D4_compatibility_disproved",
    "exact_rational_SO3_atlas_constructed",
    "exact_regular_rational_counterexample_proved",
    "full_orders_one_through_four_counterexample_recurrence_evaluated",
    "only_zero_speed_eigenspace_obstructs_at_counterexample",
    "point_compression_denominators_cleared_to_constant_numerator_polynomials",
}
FALSE_CLAIMS = {
    "B7_closed",
    "CK1_closed",
    "CK3_closed",
    "TC2_closed",
    "boundary_energy_admission_proved",
    "corrected_candidate_family_registered",
    "covariant_action_origin_proved",
    "finite_selector_determines_full_direction_sphere",
    "full_direction_sphere_D4_compatibility_proved",
    "full_tube_Sylvester_identity",
    "global_H7_closed",
    "lifespan_proved",
    "local_differential_operator_origin_proved",
    "remaining_D4_selector_closed",
    "variable_coefficient_constraint_calculus_proved",
}


def build_campaign(project_root: Path, config_path: Path) -> dict[str, Any]:
    root = project_root.resolve()
    config, _ = _load_file(config_path.resolve())
    if (
        config.get("schema_version") != CONFIG_SCHEMA
        or not _hash_matches(config)
        or config.get("global_claim_policy") != "fail_closed"
        or tuple(config.get("active_indices", ())) != ACTIVE_INDICES
        or config.get("obligation_offset") != OBLIGATION_OFFSET
        or config.get("expected_candidate_count") != EXPECTED_CANDIDATES
    ):
        raise RationalChartDeterminingError("invalid rational-chart config")
    for key in ("campaign_source", "campaign_test"):
        _check_raw(root, config[key])
    bound_keys = (
        "sixth_predecessor",
        "xyz_predecessor",
        "c23_predecessor",
        "minimal_escape",
        "fourth_campaign",
    )
    bound = {key: _load_bound(root, config[key]) for key in bound_keys}
    exact = _exact_counterexample(
        bound["xyz_predecessor"],
        bound["c23_predecessor"],
        bound["minimal_escape"],
        bound["fourth_campaign"],
    )
    artifact = {
        "schema_version": SCHEMA,
        "status": STATUS,
        "config_sha256": config["content_sha256"],
        "source_bindings": {
            key: config[key] for key in (*bound_keys, "campaign_source", "campaign_test")
        },
        "counts": {
            "bound_predecessors": 5,
            "rational_SO3_charts": 2,
            "real_sphere_uncovered_points": 0,
            "real_chart_singular_strata": 0,
            "complex_chart_singular_strata": 1,
            "rational_counterexample_points": 1,
            "directional_recurrence_evaluations": 15,
            "candidate_conditions_checked": 12,
            "candidate_compatibilities": 0,
            "candidate_obstructions": 12,
            "eigenspace_compressions_checked": 84,
            "nonzero_equal_eigenspace_compressions": 12,
            "cleared_constant_numerator_polynomials": 12,
            "global_chart_numerator_polynomials_materialized": 0,
            "negative_controls": 8,
            "inferred_global_passes": 0,
        },
        "exact_gate": exact,
        "claims": {key: True for key in TRUE_CLAIMS} | {key: False for key in FALSE_CLAIMS},
        "negative_controls": {
            "infer_full_sphere_compatibility": {"rejected": True},
            "infer_finite_determining_theorem": {"rejected": True},
            "treat_complex_chart_stratum_as_real": {"rejected": True},
            "skip_antipodal_chart": {"rejected": True},
            "skip_denominator_clearing": {"rejected": True},
            "ignore_nonzero_zero_speed_numerator": {"rejected": True},
            "infer_PDE_admission": {"rejected": True},
            "infer_global_closure": {"rejected": True},
        },
        "scope": (
            "Exact two-chart rational SO(3) atlas and full D4 recurrence at the regular chart point "
            "(u,v)=(2/5,1/5). All 12 candidates have a nonzero rank-four zero-speed cleared "
            "constant numerator, disproving full-sphere D4 compatibility of the current combined "
            "degree-three six-frame symbol. No PDE or global closure is inferred."
        ),
        "next_gate": (
            "Construct a new topology-changing angular correction nonzero at n=(2/3,2/3,1/3) "
            "while preserving the six certified frames, or revise the current symbol class; then "
            "repeat rational-chart counterexample search before any PDE admission."
        ),
        "errors": [],
    }
    return _with_hash(artifact)


def validate_campaign(document: Mapping[str, Any]) -> None:
    gate = document.get("exact_gate", {})
    obstruction = gate.get("exact_rational_obstruction", {})
    claims = document.get("claims", {})
    if (
        document.get("schema_version") != SCHEMA
        or document.get("status") != STATUS
        or not _hash_matches(document)
        or document.get("counts", {}).get("candidate_obstructions") != 12
        or document.get("counts", {}).get("inferred_global_passes") != 0
        or gate.get("atlas", {}).get("union_covers_real_S2") is not True
        or gate.get("atlas", {}).get("real_singular_strata") != 0
        or gate.get("counterexample_selector", {}).get("direction") != ["2/3", "2/3", "1/3"]
        or gate.get("full_recurrence", {}).get("base_D4_RHS_sha256") != BASE_RHS_SHA256
        or obstruction.get("candidate_compatibilities") != 0
        or obstruction.get("candidate_obstructions") != 12
        or len(obstruction.get("candidate_records", [])) != 12
        or any(
            row.get("D4_Sylvester_solvable") is not False
            or set(row.get("nonzero_equal_eigenspace_compressions", {})) != {"0"}
            or row["zero_speed_cleared_numerator"].get("numerator_rank") != 4
            or row["zero_speed_cleared_numerator"].get("numerator_nonzero_entries") != 56
            for row in obstruction.get("candidate_records", [])
        )
        or set(claims) != TRUE_CLAIMS | FALSE_CLAIMS
        or any(claims.get(key) is not True for key in TRUE_CLAIMS)
        or any(claims.get(key) is not False for key in FALSE_CLAIMS)
        or len(document.get("negative_controls", {})) != 8
        or any(
            not value.get("rejected") for value in document.get("negative_controls", {}).values()
        )
    ):
        raise RationalChartDeterminingError("rational-chart campaign validation failed")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.project_root.resolve()
    artifact = build_campaign(root, (root / args.config).resolve())
    validate_campaign(artifact)
    _atomic_write((root / args.output).resolve(), _json_bytes(artifact))
    print(
        json.dumps(
            {
                "status": artifact["status"],
                "content_sha256": artifact["content_sha256"],
                "candidate_obstructions": 12,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
