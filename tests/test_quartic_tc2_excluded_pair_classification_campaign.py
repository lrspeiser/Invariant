import json
from pathlib import Path

import sympy as sp

from sigma_theory_compiler.quartic_geometric_jet_campaign import (
    reconstruct_covariant_geometry,
)
from sigma_theory_compiler.quartic_tc2_excluded_pair_classification_campaign import (
    ETA,
    _atom_variation,
    _second_coordinate_jet_direction,
    _zeros,
    run_quartic_tc2_excluded_pair_classification_campaign,
)
from sigma_theory_compiler.quartic_tc2_variable_sylvester_campaign import (
    _content_hash_matches,
)

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "physics-language"
VARIABLE = RUNS / "quartic-tc2-variable-sylvester-campaign" / "campaign.json"
SERVICE = RUNS / "quartic-tc2-continuous-service"
CHECKPOINT = SERVICE / "checkpoint.json"
TAIL = SERVICE / "chunks" / "offset-000832.json"
CONFIG = (
    ROOT
    / "configs"
    / "backgrounds"
    / "quartic_tc2_excluded_pair_classification_campaign.json"
)
ARTIFACT = (
    RUNS / "quartic-tc2-excluded-pair-classification-campaign" / "campaign.json"
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _independent_exact_mixed_map(left_atom: str, right_atom: str) -> dict[str, sp.Expr]:
    x, y = sp.symbols("x y")
    left = _atom_variation(left_atom)
    right = _atom_variation(right_atom)
    metric = ETA + x * left["metric"] + y * right["metric"]
    metric_first = _zeros((4, 4, 4))
    metric_second = _zeros((4, 4, 4, 4))
    scalar_first = [sp.Integer(0) for _ in range(4)]
    scalar_second = _zeros((4, 4))
    for mu in range(4):
        scalar_first[mu] = (
            x * left["scalar_first"][mu] + y * right["scalar_first"][mu]
        )
        for nu in range(4):
            scalar_second[mu][nu] = (
                x * left["scalar_second"][mu][nu]
                + y * right["scalar_second"][mu][nu]
            )
            for rho in range(4):
                metric_first[mu][nu][rho] = (
                    x * left["metric_first"][mu][nu][rho]
                    + y * right["metric_first"][mu][nu][rho]
                )
                for sigma in range(4):
                    metric_second[mu][nu][rho][sigma] = (
                        x * left["metric_second"][mu][nu][rho][sigma]
                        + y * right["metric_second"][mu][nu][rho][sigma]
                    )
    geometry = reconstruct_covariant_geometry(
        metric, metric_first, metric_second, scalar_first, scalar_second
    )
    inverse = geometry["inverse_metric"]
    einstein_upper = inverse * sp.Matrix(geometry["einstein"]) * inverse
    result: dict[str, sp.Expr] = {}
    for mu in range(4):
        for nu in range(mu, 4):
            hessian = sp.factor(
                sp.diff(geometry["scalar_hessian"][mu][nu], x, y).subs(
                    {x: 0, y: 0}
                )
            )
            einstein = sp.factor(
                sp.diff(einstein_upper[mu, nu], x, y).subs({x: 0, y: 0})
            )
            if hessian != 0:
                result[f"H_{mu}{nu}"] = hessian
            if einstein != 0:
                result[f"G_{mu}{nu}"] = einstein
    return result


def test_campaign_replays_exact_artifact() -> None:
    result = run_quartic_tc2_excluded_pair_classification_campaign(
        _load(VARIABLE), _load(CHECKPOINT), _load(TAIL), _load(CONFIG)
    )
    assert result == _load(ARTIFACT)
    assert _content_hash_matches(result)


def test_exact_partition_counts_and_smallest_next_selector() -> None:
    result = _load(ARTIFACT)
    assert result["status"] == (
        "pass_exact_excluded_pair_partition_with_zero_subfamily_"
        "remaining_obligations_fail_closed"
    )
    assert result["counts"] == {
        "full_unordered_coordinate_atom_pairs": 11781,
        "completed_canonical_active_pairs": 861,
        "excluded_pairs_classified": 10920,
        "first_direction_zero_atoms": 80,
        "first_direction_nonzero_atoms": 73,
        "first_Sylvester_active_atoms": 41,
        "both_first_directions_zero_pairs": 3240,
        "exactly_one_first_direction_zero_pairs": 5840,
        "both_first_directions_nonzero_excluded_pairs": 1840,
        "structurally_possible_second_coordinate_direction_pairs": 1880,
        "exact_nonzero_second_coordinate_direction_pairs": 835,
        "rigorously_discharged_entrywise_zero_pairs": 8245,
        "remaining_exact_second_Sylvester_obligations": 2675,
        "TC2_closures": 0,
        "global_H7_closures": 0,
        "lifespans_proved": 0,
    }
    assert result["requirement_counts"] == {
        "coordinate_D2_pushforward_D2P_D2K_D2TC2": 835,
        "entrywise_zero_chain_rule_discharged": 8245,
        "intrinsic_jet_D2P_D2K_D2TC2": 1840,
    }
    assert result["first_direction_by_second_direction_matrix"] == {
        "both_first_directions_zero": {
            "pairs": 3240,
            "second_coordinate_direction_nonzero": 414,
            "second_coordinate_direction_zero": 2826,
            "rigorously_discharged": 2826,
            "remaining_obligations": 414,
        },
        "exactly_one_first_direction_zero": {
            "pairs": 5840,
            "second_coordinate_direction_nonzero": 421,
            "second_coordinate_direction_zero": 5419,
            "rigorously_discharged": 5419,
            "remaining_obligations": 421,
        },
        "both_first_directions_nonzero": {
            "pairs": 1840,
            "second_coordinate_direction_nonzero": 0,
            "second_coordinate_direction_zero": 1840,
            "rigorously_discharged": 0,
            "remaining_obligations": 1840,
        },
    }
    assert result["nonlinear_coordinate_map_support"]["family_counts"] == {
        "metric_value_x_metric_second_to_Einstein": {
            "structurally_possible_pairs": 900,
            "exact_nonzero_pairs": 381,
            "exact_zero_pairs": 519,
        },
        "metric_first_x_metric_first_to_Einstein": {
            "structurally_possible_pairs": 820,
            "exact_nonzero_pairs": 378,
            "exact_zero_pairs": 442,
        },
        "metric_first_x_scalar_first_to_Hessian": {
            "structurally_possible_pairs": 160,
            "exact_nonzero_pairs": 76,
            "exact_zero_pairs": 84,
        },
    }
    next_selector = result["next_exact_selector"]
    assert next_selector["total_pairs"] == 2675
    assert next_selector["chunk_size"] == 64
    assert next_selector["first_global_pair_index"] == 55
    assert next_selector["first_pair"] == {
        "left_atom": "q[0]",
        "right_atom": "s01[1]",
        "requirement": "coordinate_D2_pushforward_D2P_D2K_D2TC2",
    }


def test_discharged_manifest_is_entrywise_chain_rule_zero() -> None:
    result = _load(ARTIFACT)
    manifest = result["excluded_pair_manifest"]
    assert len(manifest) == 10920
    assert len({item["global_pair_index"] for item in manifest}) == 10920
    discharged = [item for item in manifest if item["rigorously_discharged"]]
    remaining = [item for item in manifest if not item["rigorously_discharged"]]
    assert len(discharged) == 8245 and len(remaining) == 2675
    assert all(
        (
            not item["left_first_jet_direction_nonzero"]
            or not item["right_first_jet_direction_nonzero"]
        )
        and not item["exact_second_coordinate_direction_nonzero"]
        and item["requirement"] == "entrywise_zero_chain_rule_discharged"
        for item in discharged
    )
    assert all(
        item["exact_second_coordinate_direction_nonzero"]
        or (
            item["left_first_jet_direction_nonzero"]
            and item["right_first_jet_direction_nonzero"]
        )
        for item in remaining
    )
    control = result["entrywise_zero_chain_rule_control"]
    assert control["passed"]
    assert all(
        item["rejected"] for item in control["negative_controls"].values()
    )


def test_three_nonlinear_map_families_match_independent_exact_geometry() -> None:
    pairs = (
        ("q[0]", "s11[0]"),
        ("p1[0]", "p1[0]"),
        ("p1[0]", "p1[10]"),
    )
    for left, right in pairs:
        assert _second_coordinate_jet_direction(
            left, right
        ) == _independent_exact_mixed_map(left, right)
        assert _second_coordinate_jet_direction(left, right)


def test_tampered_checkpoint_and_false_promotion_reject_before_work() -> None:
    variable, checkpoint, tail, config = (
        _load(VARIABLE),
        _load(CHECKPOINT),
        _load(TAIL),
        _load(CONFIG),
    )
    corrupt = dict(checkpoint)
    corrupt["next_offset"] = 860
    result = run_quartic_tc2_excluded_pair_classification_campaign(
        variable, corrupt, tail, config
    )
    assert result["status"] == "reject"
    promoted = dict(config)
    promoted["global_H7_policy"] = "pass"
    result = run_quartic_tc2_excluded_pair_classification_campaign(
        variable, checkpoint, tail, promoted
    )
    assert result["status"] == "reject"
