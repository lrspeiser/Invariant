from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import sympy as sp

from .quartic_geometric_jet_campaign import (
    SYMMETRIC_METRIC_PAIRS,
    SYMMETRIC_METRIC_WEIGHTS,
)
from .quartic_tc2_continuous_service import _checkpoint_hash_matches
from .quartic_tc2_second_atom_chunk_campaign import (
    _canonical_active_affine_pairs,
    _global_unordered_pair_index,
)
from .quartic_tc2_variable_sylvester_campaign import (
    ATOM_DIMENSION,
    STATE_DIMENSION,
    _content_hash,
    _content_hash_matches,
    _coordinate_atom_to_jet_packet,
    _reference_and_first_jet_packet,
)

SCHEMA_VERSION = "sigma-quartic-tc2-excluded-pair-classification-1.0"
TOTAL_PAIRS = ATOM_DIMENSION * (ATOM_DIMENSION + 1) // 2
ETA = sp.diag(-1, 1, 1, 1)


class QuarticTC2ExcludedPairClassificationError(ValueError):
    """Raised when an excluded second-atom obligation is inferred unsafely."""


def _zeros(shape: tuple[int, ...]) -> Any:
    if len(shape) == 1:
        return [sp.Integer(0) for _ in range(shape[0])]
    return [_zeros(shape[1:]) for _ in range(shape[0])]


def _matrix_component(matrix: sp.Matrix, row: int, column: int) -> sp.Expr:
    return sp.factor(matrix[row, column])


def _atom_variation(atom: str) -> dict[str, Any]:
    metric = sp.zeros(4)
    metric_first = _zeros((4, 4, 4))
    metric_second = _zeros((4, 4, 4, 4))
    scalar_first = [sp.Integer(0) for _ in range(4)]
    scalar_second = _zeros((4, 4))
    family, field_text = atom.split("[")
    field = int(field_text[:-1])
    if family == "q":
        left, right = SYMMETRIC_METRIC_PAIRS[field]
        value = 1 / SYMMETRIC_METRIC_WEIGHTS[field]
        metric[left, right] = value
        metric[right, left] = value
    elif family.startswith("p"):
        derivative = int(family[1])
        if field == 10:
            scalar_first[derivative] = sp.Integer(1)
        else:
            left, right = SYMMETRIC_METRIC_PAIRS[field]
            value = 1 / SYMMETRIC_METRIC_WEIGHTS[field]
            metric_first[derivative][left][right] = value
            metric_first[derivative][right][left] = value
    elif family.startswith("s"):
        first, second = int(family[1]), int(family[2])
        if field == 10:
            scalar_second[first][second] = sp.Integer(1)
            scalar_second[second][first] = sp.Integer(1)
        else:
            left, right = SYMMETRIC_METRIC_PAIRS[field]
            value = 1 / SYMMETRIC_METRIC_WEIGHTS[field]
            for derivative_left, derivative_right in (
                (first, second),
                (second, first),
            ):
                metric_second[derivative_left][derivative_right][left][right] = value
                metric_second[derivative_left][derivative_right][right][left] = value
    return {
        "metric": metric,
        "metric_first": metric_first,
        "metric_second": metric_second,
        "scalar_first": scalar_first,
        "scalar_second": scalar_second,
    }


def _connection_first_variation(variation: dict[str, Any]) -> Any:
    result = _zeros((4, 4, 4))
    metric_first = variation["metric_first"]
    for upper in range(4):
        for left in range(4):
            for right in range(4):
                result[upper][left][right] = sp.factor(
                    sum(
                        ETA[upper, contracted]
                        * (
                            metric_first[left][contracted][right]
                            + metric_first[right][contracted][left]
                            - metric_first[contracted][left][right]
                        )
                        for contracted in range(4)
                    )
                    / 2
                )
    return result


def _connection_derivative_first_variation(variation: dict[str, Any]) -> Any:
    result = _zeros((4, 4, 4, 4))
    metric_second = variation["metric_second"]
    for derivative in range(4):
        for upper in range(4):
            for left in range(4):
                for right in range(4):
                    result[derivative][upper][left][right] = sp.factor(
                        sum(
                            ETA[upper, contracted]
                            * (
                                metric_second[derivative][left][contracted][right]
                                + metric_second[derivative][right][contracted][left]
                                - metric_second[derivative][contracted][left][right]
                            )
                            for contracted in range(4)
                        )
                        / 2
                    )
    return result


def _inverse_first(metric_variation: sp.Matrix) -> sp.Matrix:
    return (-ETA * metric_variation * ETA).applyfunc(sp.factor)


def _connection_derivative_mixed(
    left: dict[str, Any], right: dict[str, Any]
) -> Any:
    result = _zeros((4, 4, 4, 4))
    inverse_left = _inverse_first(left["metric"])
    inverse_right = _inverse_first(right["metric"])
    for derivative in range(4):
        left_metric_first_matrix = sp.Matrix(
            left["metric_first"][derivative]
        )
        right_metric_first_matrix = sp.Matrix(
            right["metric_first"][derivative]
        )
        inverse_spacetime_left = (
            -ETA * left_metric_first_matrix * ETA
        ).applyfunc(sp.factor)
        inverse_spacetime_right = (
            -ETA * right_metric_first_matrix * ETA
        ).applyfunc(sp.factor)
        for upper in range(4):
            for lower_left in range(4):
                for lower_right in range(4):
                    value = sp.Integer(0)
                    for contracted in range(4):
                        right_bracket = (
                            right["metric_second"][derivative][lower_left][contracted][lower_right]
                            + right["metric_second"][derivative][lower_right][contracted][lower_left]
                            - right["metric_second"][derivative][contracted][lower_left][lower_right]
                        )
                        left_bracket = (
                            left["metric_second"][derivative][lower_left][contracted][lower_right]
                            + left["metric_second"][derivative][lower_right][contracted][lower_left]
                            - left["metric_second"][derivative][contracted][lower_left][lower_right]
                        )
                        right_metric_first_bracket = (
                            right["metric_first"][lower_left][contracted][lower_right]
                            + right["metric_first"][lower_right][contracted][lower_left]
                            - right["metric_first"][contracted][lower_left][lower_right]
                        )
                        left_metric_first_bracket = (
                            left["metric_first"][lower_left][contracted][lower_right]
                            + left["metric_first"][lower_right][contracted][lower_left]
                            - left["metric_first"][contracted][lower_left][lower_right]
                        )
                        value += (
                            inverse_left[upper, contracted] * right_bracket
                            + inverse_right[upper, contracted] * left_bracket
                            + inverse_spacetime_left[upper, contracted]
                            * right_metric_first_bracket
                            + inverse_spacetime_right[upper, contracted]
                            * left_metric_first_bracket
                        ) / 2
                    result[derivative][upper][lower_left][lower_right] = sp.factor(value)
    return result


def _ricci_first(connection_derivative: Any) -> sp.Matrix:
    ricci = sp.zeros(4)
    for left in range(4):
        for right in range(4):
            ricci[left, right] = sp.factor(
                sum(
                    connection_derivative[upper][upper][right][left]
                    - connection_derivative[right][upper][upper][left]
                    for upper in range(4)
                )
            )
    return ricci


def _ricci_mixed(
    left_connection: Any,
    right_connection: Any,
    mixed_connection_derivative: Any,
) -> sp.Matrix:
    ricci = sp.zeros(4)
    for left in range(4):
        for right in range(4):
            value = sp.Integer(0)
            for upper in range(4):
                value += (
                    mixed_connection_derivative[upper][upper][right][left]
                    - mixed_connection_derivative[right][upper][upper][left]
                )
                for contracted in range(4):
                    value += (
                        left_connection[upper][upper][contracted]
                        * right_connection[contracted][right][left]
                        + right_connection[upper][upper][contracted]
                        * left_connection[contracted][right][left]
                        - left_connection[upper][right][contracted]
                        * right_connection[contracted][upper][left]
                        - right_connection[upper][right][contracted]
                        * left_connection[contracted][upper][left]
                    )
            ricci[left, right] = sp.factor(value)
    return ricci


def _einstein_upper_first(variation: dict[str, Any]) -> sp.Matrix:
    ricci = _ricci_first(_connection_derivative_first_variation(variation))
    scalar = sp.factor(sum(ETA[mu, nu] * ricci[mu, nu] for mu in range(4) for nu in range(4)))
    lower = (ricci - ETA * scalar / 2).applyfunc(sp.factor)
    return (ETA * lower * ETA).applyfunc(sp.factor)


def _einstein_upper_mixed(left: dict[str, Any], right: dict[str, Any]) -> sp.Matrix:
    connection_left = _connection_first_variation(left)
    connection_right = _connection_first_variation(right)
    ricci_left = _ricci_first(_connection_derivative_first_variation(left))
    ricci_right = _ricci_first(_connection_derivative_first_variation(right))
    ricci_mixed = _ricci_mixed(
        connection_left,
        connection_right,
        _connection_derivative_mixed(left, right),
    )
    inverse_left = _inverse_first(left["metric"])
    inverse_right = _inverse_first(right["metric"])
    scalar_left = sp.factor(sum(ETA[mu, nu] * ricci_left[mu, nu] for mu in range(4) for nu in range(4)))
    scalar_right = sp.factor(sum(ETA[mu, nu] * ricci_right[mu, nu] for mu in range(4) for nu in range(4)))
    scalar_mixed = sp.factor(
        sum(
            ETA[mu, nu] * ricci_mixed[mu, nu]
            + inverse_left[mu, nu] * ricci_right[mu, nu]
            + inverse_right[mu, nu] * ricci_left[mu, nu]
            for mu in range(4)
            for nu in range(4)
        )
    )
    lower_left = (ricci_left - ETA * scalar_left / 2).applyfunc(sp.factor)
    lower_right = (ricci_right - ETA * scalar_right / 2).applyfunc(sp.factor)
    lower_mixed = (
        ricci_mixed
        - (
            ETA * scalar_mixed
            + left["metric"] * scalar_right
            + right["metric"] * scalar_left
        )
        / 2
    ).applyfunc(sp.factor)
    return (
        ETA * lower_mixed * ETA
        + inverse_left * lower_right * ETA
        + inverse_right * lower_left * ETA
        + ETA * lower_left * inverse_right
        + ETA * lower_right * inverse_left
    ).applyfunc(sp.factor)


def _second_coordinate_jet_direction(left_atom: str, right_atom: str) -> dict[str, sp.Expr]:
    left = _atom_variation(left_atom)
    right = _atom_variation(right_atom)
    result: dict[str, sp.Expr] = {}
    left_connection = _connection_first_variation(left)
    right_connection = _connection_first_variation(right)
    for mu in range(4):
        for nu in range(mu, 4):
            hessian = -sum(
                left_connection[upper][mu][nu] * right["scalar_first"][upper]
                + right_connection[upper][mu][nu] * left["scalar_first"][upper]
                for upper in range(4)
            )
            hessian = sp.factor(hessian)
            if hessian != 0:
                result[f"H_{mu}{nu}"] = hessian
    einstein = _einstein_upper_mixed(left, right)
    for mu in range(4):
        for nu in range(mu, 4):
            value = _matrix_component(einstein, mu, nu)
            if value != 0:
                result[f"G_{mu}{nu}"] = value
    return result


def _structurally_possible_second_direction(left_atom: str, right_atom: str) -> str | None:
    atoms = (left_atom, right_atom)
    families = [atom.split("[")[0] for atom in atoms]
    fields = [int(atom.split("[")[1][:-1]) for atom in atoms]
    if (
        families[0] == "q"
        and families[1].startswith("s")
        and fields[1] < 10
    ) or (
        families[1] == "q"
        and families[0].startswith("s")
        and fields[0] < 10
    ):
        return "metric_value_x_metric_second_to_Einstein"
    if all(family.startswith("p") for family in families) and all(
        field < 10 for field in fields
    ):
        return "metric_first_x_metric_first_to_Einstein"
    if all(family.startswith("p") for family in families) and sorted(
        field == 10 for field in fields
    ) == [False, True]:
        return "metric_first_x_scalar_first_to_Hessian"
    return None


def _chain_rule_zero_control() -> tuple[bool, dict[str, Any]]:
    fa, fab, ja, jb, jab = sp.symbols("FA FAB JA JB JAB")
    chain = fab * ja * jb + fa * jab
    one_first_zero_and_mixed_zero = sp.expand(chain.subs({ja: 0, jab: 0}))
    corrupted_mixed = sp.expand(chain.subs({ja: 0, jab: 1}))
    corrupted_both_first = sp.expand(chain.subs({ja: 1, jb: 1, jab: 0}))
    passed = bool(
        one_first_zero_and_mixed_zero == 0
        and corrupted_mixed != 0
        and corrupted_both_first != 0
    )
    return passed, {
        "identity": "D2(F o J)[A,B]=D2F[DJ_A,DJ_B]+DF[D2J_AB]",
        "one_first_zero_and_mixed_zero_residual": str(
            one_first_zero_and_mixed_zero
        ),
        "applies_entrywise_to": ["P55", "K55", "TC2"],
        "negative_controls": {
            "drop_nonzero_second_coordinate_direction": {
                "residual": str(corrupted_mixed),
                "rejected": corrupted_mixed != 0,
            },
            "infer_zero_with_two_nonzero_first_directions": {
                "residual": str(corrupted_both_first),
                "rejected": corrupted_both_first != 0,
            },
        },
        "passed": passed,
    }


def run_quartic_tc2_excluded_pair_classification_campaign(
    variable: dict[str, Any],
    checkpoint: dict[str, Any],
    tail_artifact: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        if (
            config.get("schema_version") != SCHEMA_VERSION
            or int(config.get("coordinate_atom_dimension", 0)) != ATOM_DIMENSION
            or int(config.get("full_unordered_pair_denominator", 0)) != TOTAL_PAIRS
            or int(config.get("completed_canonical_active_pairs", 0)) != 861
            or config.get("variable_campaign_sha256") != variable.get("content_sha256")
            or config.get("checkpoint_sha256") != checkpoint.get("content_sha256")
            or config.get("tail_artifact_sha256")
            != tail_artifact.get("content_sha256")
            or config.get("global_TC2_policy") != "fail_closed"
            or config.get("B7_policy") != "fail_closed"
            or config.get("global_H7_policy") != "fail_closed"
            or config.get("lifespan_policy") != "fail_closed"
            or not _content_hash_matches(variable)
            or not _checkpoint_hash_matches(checkpoint)
            or not _content_hash_matches(tail_artifact)
            or checkpoint.get("next_offset") != 861
            or tail_artifact.get("status")
            != "pass_cumulative_861_second_atom_pairs_no_obstruction_remaining_fail_closed"
        ):
            raise QuarticTC2ExcludedPairClassificationError(
                "unsupported excluded-pair classification contract"
            )
        coordinate = _coordinate_atom_to_jet_packet()
        reference = _reference_and_first_jet_packet()
        if (
            coordinate["packet"]["content_sha256"]
            != variable["common_coordinate_to_covariant_jet_packet"][
                "content_sha256"
            ]
        ):
            raise QuarticTC2ExcludedPairClassificationError(
                "coordinate-to-jet provenance mismatch"
            )
        completed = {
            pair["global_pair_index"] for pair in _canonical_active_affine_pairs()
        }
        if len(completed) != 861:
            raise QuarticTC2ExcludedPairClassificationError(
                "canonical-active selector count mismatch"
            )
        first_nonzero = [bool(direction) for direction in coordinate["maps"]]
        sylvester_active: list[bool] = []
        for direction in coordinate["maps"]:
            derivative = sum(
                (
                    coefficient * reference["delta_derivatives"][name]
                    for name, coefficient in direction.items()
                ),
                sp.zeros(STATE_DIMENSION),
            )
            sylvester_active.append(not derivative.is_zero_matrix)

        manifest: list[dict[str, Any]] = []
        second_packets: dict[str, dict[str, Any]] = {}
        family_witnesses: dict[str, dict[str, Any]] = {}
        for left in range(ATOM_DIMENSION):
            for right in range(left, ATOM_DIMENSION):
                global_index = _global_unordered_pair_index(left, right)
                if global_index in completed:
                    continue
                left_atom = coordinate["atoms"][left]
                right_atom = coordinate["atoms"][right]
                structural_family = _structurally_possible_second_direction(
                    left_atom, right_atom
                )
                second_direction = (
                    _second_coordinate_jet_direction(left_atom, right_atom)
                    if structural_family is not None
                    else {}
                )
                serialized_second = {
                    name: str(value) for name, value in second_direction.items()
                }
                second_hash = _content_hash(serialized_second)
                if second_direction:
                    second_packets[second_hash] = {
                        "content_sha256": second_hash,
                        "jet_entries": serialized_second,
                    }
                    family_witnesses.setdefault(
                        str(structural_family),
                        {
                            "global_pair_index": global_index,
                            "left_atom": left_atom,
                            "right_atom": right_atom,
                            "second_direction_sha256": second_hash,
                            "jet_entries": serialized_second,
                        },
                    )
                either_first_zero = not first_nonzero[left] or not first_nonzero[right]
                discharged = either_first_zero and not second_direction
                if discharged:
                    requirement = "entrywise_zero_chain_rule_discharged"
                elif second_direction and first_nonzero[left] and first_nonzero[right]:
                    requirement = "intrinsic_D2_and_coordinate_D2_pushforward"
                elif second_direction:
                    requirement = "coordinate_D2_pushforward_D2P_D2K_D2TC2"
                else:
                    requirement = "intrinsic_jet_D2P_D2K_D2TC2"
                manifest.append(
                    {
                        "global_pair_index": global_index,
                        "left_atom_index": left,
                        "right_atom_index": right,
                        "left_atom": left_atom,
                        "right_atom": right_atom,
                        "left_first_jet_direction_nonzero": first_nonzero[left],
                        "right_first_jet_direction_nonzero": first_nonzero[right],
                        "left_first_Sylvester_direction_active": sylvester_active[left],
                        "right_first_Sylvester_direction_active": sylvester_active[right],
                        "structural_second_map_family": structural_family,
                        "exact_second_coordinate_direction_nonzero": bool(
                            second_direction
                        ),
                        "second_coordinate_direction_sha256": second_hash,
                        "requirement": requirement,
                        "rigorously_discharged": discharged,
                    }
                )

        if len(manifest) != TOTAL_PAIRS - len(completed):
            raise QuarticTC2ExcludedPairClassificationError(
                "excluded-pair manifest count mismatch"
            )
        chain_passed, chain_control = _chain_rule_zero_control()
        if not chain_passed:
            raise QuarticTC2ExcludedPairClassificationError(
                "entrywise chain-rule zero control failed"
            )
        counts: dict[str, int] = {
            "full_unordered_coordinate_atom_pairs": TOTAL_PAIRS,
            "completed_canonical_active_pairs": len(completed),
            "excluded_pairs_classified": len(manifest),
            "first_direction_zero_atoms": sum(not value for value in first_nonzero),
            "first_direction_nonzero_atoms": sum(first_nonzero),
            "first_Sylvester_active_atoms": sum(sylvester_active),
            "both_first_directions_zero_pairs": sum(
                not item["left_first_jet_direction_nonzero"]
                and not item["right_first_jet_direction_nonzero"]
                for item in manifest
            ),
            "exactly_one_first_direction_zero_pairs": sum(
                item["left_first_jet_direction_nonzero"]
                != item["right_first_jet_direction_nonzero"]
                for item in manifest
            ),
            "both_first_directions_nonzero_excluded_pairs": sum(
                item["left_first_jet_direction_nonzero"]
                and item["right_first_jet_direction_nonzero"]
                for item in manifest
            ),
            "structurally_possible_second_coordinate_direction_pairs": sum(
                item["structural_second_map_family"] is not None for item in manifest
            ),
            "exact_nonzero_second_coordinate_direction_pairs": sum(
                item["exact_second_coordinate_direction_nonzero"]
                for item in manifest
            ),
            "rigorously_discharged_entrywise_zero_pairs": sum(
                item["rigorously_discharged"] for item in manifest
            ),
            "remaining_exact_second_Sylvester_obligations": sum(
                not item["rigorously_discharged"] for item in manifest
            ),
            "TC2_closures": 0,
            "global_H7_closures": 0,
            "lifespans_proved": 0,
        }
        requirements = {
            name: sum(item["requirement"] == name for item in manifest)
            for name in sorted({item["requirement"] for item in manifest})
        }
        first_category = {
            "both_first_directions_zero": lambda item: (
                not item["left_first_jet_direction_nonzero"]
                and not item["right_first_jet_direction_nonzero"]
            ),
            "exactly_one_first_direction_zero": lambda item: (
                item["left_first_jet_direction_nonzero"]
                != item["right_first_jet_direction_nonzero"]
            ),
            "both_first_directions_nonzero": lambda item: (
                item["left_first_jet_direction_nonzero"]
                and item["right_first_jet_direction_nonzero"]
            ),
        }
        classification_matrix = {
            name: {
                "pairs": sum(predicate(item) for item in manifest),
                "second_coordinate_direction_nonzero": sum(
                    predicate(item)
                    and item["exact_second_coordinate_direction_nonzero"]
                    for item in manifest
                ),
                "second_coordinate_direction_zero": sum(
                    predicate(item)
                    and not item["exact_second_coordinate_direction_nonzero"]
                    for item in manifest
                ),
                "rigorously_discharged": sum(
                    predicate(item) and item["rigorously_discharged"]
                    for item in manifest
                ),
                "remaining_obligations": sum(
                    predicate(item) and not item["rigorously_discharged"]
                    for item in manifest
                ),
            }
            for name, predicate in first_category.items()
        }
        structural_family_counts = {
            family: {
                "structurally_possible_pairs": sum(
                    item["structural_second_map_family"] == family
                    for item in manifest
                ),
                "exact_nonzero_pairs": sum(
                    item["structural_second_map_family"] == family
                    and item["exact_second_coordinate_direction_nonzero"]
                    for item in manifest
                ),
                "exact_zero_pairs": sum(
                    item["structural_second_map_family"] == family
                    and not item["exact_second_coordinate_direction_nonzero"]
                    for item in manifest
                ),
            }
            for family in (
                "metric_value_x_metric_second_to_Einstein",
                "metric_first_x_metric_first_to_Einstein",
                "metric_first_x_scalar_first_to_Hessian",
            )
        }
        next_selector = [
            item for item in manifest if not item["rigorously_discharged"]
        ]
        next_chunk = next_selector[: int(config["next_selector_chunk_size"])]
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": (
                "pass_exact_excluded_pair_partition_with_zero_subfamily_"
                "remaining_obligations_fail_closed"
            ),
            "errors": [],
            "upstream_sha256": {
                "variable_campaign": variable["content_sha256"],
                "coordinate_to_jet_packet": coordinate["packet"][
                    "content_sha256"
                ],
                "continuous_checkpoint": checkpoint["content_sha256"],
                "completed_tail_artifact": tail_artifact["content_sha256"],
            },
            "config_sha256": _content_hash(config),
            "counts": counts,
            "requirement_counts": requirements,
            "first_direction_by_second_direction_matrix": classification_matrix,
            "entrywise_zero_chain_rule_control": chain_control,
            "nonlinear_coordinate_map_support": {
                "families": {
                    "metric_value_x_metric_second_to_Einstein": (
                        "D(g^{-1}) times coordinate metric second derivatives"
                    ),
                    "metric_first_x_metric_first_to_Einstein": (
                        "Christoffel-square curvature terms"
                    ),
                    "metric_first_x_scalar_first_to_Hessian": (
                        "minus Christoffel times scalar gradient"
                    ),
                },
                "exact_nonzero_family_witnesses": family_witnesses,
                "family_counts": structural_family_counts,
                "all_other_coordinate_mixed_second_directions_entrywise_zero": True,
            },
            "second_coordinate_direction_packets": [
                second_packets[key] for key in sorted(second_packets)
            ],
            "excluded_pair_manifest": manifest,
            "next_exact_selector": {
                "ordering": "ascending global unordered coordinate-pair index",
                "total_pairs": len(next_selector),
                "chunk_size": len(next_chunk),
                "first_global_pair_index": next_chunk[0]["global_pair_index"],
                "first_pair": {
                    "left_atom": next_chunk[0]["left_atom"],
                    "right_atom": next_chunk[0]["right_atom"],
                    "requirement": next_chunk[0]["requirement"],
                },
                "pair_global_indices": [
                    item["global_pair_index"] for item in next_chunk
                ],
                "selector_sha256": _content_hash(next_chunk),
            },
            "claim": (
                "Every coordinate pair outside the completed 861-pair canonical-active "
                "selector is classified from exact first and mixed-second coordinate-to-jet "
                "directions. Only chain-rule-identically-zero pairs are discharged."
            ),
            "scope": (
                "The remaining work manifest requires componentwise D2P55/D2K55/D2TC2 "
                "and Sylvester compression evaluation. TC2, B7, global H7, and lifespan "
                "remain fail-closed."
            ),
        }
    except (KeyError, TypeError, ValueError, QuarticTC2ExcludedPairClassificationError) as error:
        errors.append(str(error))
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": "reject",
            "errors": errors,
            "counts": {
                "full_unordered_coordinate_atom_pairs": TOTAL_PAIRS,
                "completed_canonical_active_pairs": 0,
                "excluded_pairs_classified": 0,
                "remaining_exact_second_Sylvester_obligations": TOTAL_PAIRS,
                "TC2_closures": 0,
                "global_H7_closures": 0,
                "lifespans_proved": 0,
            },
        }
    return {**body, "content_sha256": _content_hash(body)}


def write_quartic_tc2_excluded_pair_classification_campaign(
    result: dict[str, Any], output: Path
) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    path = output / "campaign.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
