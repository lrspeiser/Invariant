"""Development-only probe: score eight converged RG cells on one common row mask."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from sigma_theory_compiler import (
    open_gravity_refracted_gravity_phangs_sparc_development_score_v1 as fixed_score,
)
from sigma_theory_compiler import (
    open_gravity_refracted_gravity_published_prior_development_scan_v1 as prior,
)


BLOCKED_CELL = "PRIOR_CORNER_E0.1_Q2_R-23"
CONTROLS = [
    "NEWTON_3D_DST",
    "RAR_2016_ON_NEWTON_3D",
    "MOND_STANDARD_MU_ON_NEWTON_3D",
]


def score_object(
    config: Mapping[str, Any],
    source_row: Mapping[str, Any],
    response_rows: Sequence[Mapping[str, Any]],
    *,
    asymmetric: bool,
    admitted: Sequence[str],
) -> dict[str, Any]:
    fine_radius, fine = prior._candidate_profiles(config, source_row, "fine")
    coarse_radius, coarse = prior._candidate_profiles(config, source_row, "convergence")
    if not np.array_equal(fine_radius, coarse_radius):
        raise RuntimeError("radial grids changed")
    candidate_ids = CONTROLS + list(admitted)
    threshold = float(config["grid_contract"]["fine_vs_convergence_maximum_relative_difference"])
    used: list[Mapping[str, Any]] = []
    predictions: dict[str, list[float]] = {candidate_id: [] for candidate_id in candidate_ids}
    for response_row in response_rows:
        radius = float(response_row["radius_kpc"])
        in_range = (
            float(config["grid_contract"]["radial_min_kpc"])
            <= radius
            <= float(config["grid_contract"]["radial_max_kpc"])
        )
        enough_cells = radius / float(config["grid_contract"]["fine_spacing_kpc"]) >= float(
            config["grid_contract"]["minimum_fine_cells_per_radius"]
        )
        values: dict[str, float] = {}
        differences: dict[str, float] = {}
        if in_range:
            for candidate_id in candidate_ids:
                fine_value = float(np.interp(radius, fine_radius, fine[candidate_id]))
                coarse_value = float(np.interp(radius, coarse_radius, coarse[candidate_id]))
                values[candidate_id] = fine_value
                differences[candidate_id] = abs(fine_value - coarse_value) / max(
                    abs(fine_value), abs(coarse_value), 1.0e-30
                )
        eligible = (
            in_range
            and enough_cells
            and all(value <= threshold for value in differences.values())
            and all(math.isfinite(value) and value > 0.0 for value in values.values())
        )
        if eligible:
            used.append(response_row)
            for candidate_id in candidate_ids:
                predictions[candidate_id].append(values[candidate_id])
    if len(used) < int(config["scoring_contract"]["minimum_rows_per_object"]):
        raise RuntimeError(f"insufficient rows for {source_row['object_id']}: {len(used)}")
    radius = np.asarray([float(row["radius_kpc"]) for row in used])
    observed = np.asarray([float(row["velocity_km_s"]) for row in used])
    scores: dict[str, Any] = {}
    for candidate_id in candidate_ids:
        predicted = (
            np.sqrt(np.asarray(predictions[candidate_id]) * radius * 3.085677581491367e19)
            / 1000.0
        )
        if asymmetric:
            upper = np.asarray([float(row["upper_error_km_s"]) for row in used])
            lower = np.asarray([float(row["lower_error_km_s"]) for row in used])
            sigma = np.where(predicted >= observed, upper, lower)
        else:
            sigma = np.asarray([float(row["error_km_s"]) for row in used])
        residual = (predicted - observed) / sigma
        scores[candidate_id] = {"loss": float(np.mean(residual * residual))}
    return {
        "object_id": source_row["object_id"],
        "rows_available": len(response_rows),
        "rows_scored_common": len(used),
        "scores": scores,
    }


def aggregate(rows: Sequence[Mapping[str, Any]], candidate_id: str) -> float:
    return float(np.mean([float(row["scores"][candidate_id]["loss"]) for row in rows]))


def main() -> None:
    config = prior.load_config()
    receipt = json.loads(
        Path(
            "runs/gravity/open-gravity-refracted-gravity-published-prior-development-scan-v1/receipt.json"
        ).read_text(encoding="utf-8")
    )
    source_rows = {row["object_id"]: row for row in receipt["source_field_rows"]}
    all_cells = [str(row["id"]) for row in prior.parameter_cells(config)]
    admitted = [cell for cell in all_cells if cell != BLOCKED_CELL]
    fixed_config = fixed_score.load_config()
    fixed_evidence = fixed_score.validate_predecessors(fixed_config)
    response_config = fixed_evidence["response_config"]
    phangs_response, phangs_access = fixed_score.responses._load_phangs_responses(response_config)
    sparc_response, sparc_access = fixed_score.responses._load_sparc_responses(response_config)
    phangs = [
        score_object(
            config,
            source_rows[object_id],
            phangs_response[object_id],
            asymmetric=True,
            admitted=admitted,
        )
        for object_id in config["objects"]
    ]
    sparc = [
        score_object(
            config,
            source_rows["NGC2903"],
            fixed_score.responses._sparc_rows(sparc_response["NGC2903"]),
            asymmetric=False,
            admitted=admitted,
        )
    ]
    phangs_ngc2903 = list(phangs_response["NGC2903"])
    sparc_ngc2903 = list(fixed_score.responses._sparc_rows(sparc_response["NGC2903"]))
    overlap_min = max(
        min(float(row["radius_kpc"]) for row in phangs_ngc2903),
        min(float(row["radius_kpc"]) for row in sparc_ngc2903),
    )
    overlap_max = min(
        max(float(row["radius_kpc"]) for row in phangs_ngc2903),
        max(float(row["radius_kpc"]) for row in sparc_ngc2903),
    )
    phangs_overlap = [
        row for row in phangs_ngc2903 if overlap_min <= float(row["radius_kpc"]) <= overlap_max
    ]
    sparc_overlap = [
        row for row in sparc_ngc2903 if overlap_min <= float(row["radius_kpc"]) <= overlap_max
    ]
    phangs_overlap_score = score_object(
        config,
        source_rows["NGC2903"],
        phangs_overlap,
        asymmetric=True,
        admitted=admitted,
    )
    sparc_overlap_score = score_object(
        config,
        source_rows["NGC2903"],
        sparc_overlap,
        asymmetric=False,
        admitted=admitted,
    )
    sparc_outer = [row for row in sparc_ngc2903 if float(row["radius_kpc"]) > overlap_max]
    sparc_outer_score = score_object(
        config,
        source_rows["NGC2903"],
        sparc_outer,
        asymmetric=False,
        admitted=admitted,
    )
    phangs_radius = np.asarray([float(row["radius_kpc"]) for row in phangs_overlap])
    phangs_velocity = np.asarray([float(row["velocity_km_s"]) for row in phangs_overlap])
    phangs_sigma = np.asarray(
        [
            0.5 * (float(row["upper_error_km_s"]) + float(row["lower_error_km_s"]))
            for row in phangs_overlap
        ]
    )
    sparc_radius = np.asarray([float(row["radius_kpc"]) for row in sparc_overlap])
    sparc_velocity = np.asarray([float(row["velocity_km_s"]) for row in sparc_overlap])
    sparc_sigma = np.asarray([float(row["error_km_s"]) for row in sparc_overlap])
    interpolated_phangs_velocity = np.interp(sparc_radius, phangs_radius, phangs_velocity)
    interpolated_phangs_sigma = np.interp(sparc_radius, phangs_radius, phangs_sigma)
    tracer_fractional_difference = np.abs(interpolated_phangs_velocity - sparc_velocity) / np.maximum(
        0.5 * (np.abs(interpolated_phangs_velocity) + np.abs(sparc_velocity)), 1.0e-12
    )
    tracer_standardized_difference = (
        interpolated_phangs_velocity - sparc_velocity
    ) / np.sqrt(interpolated_phangs_sigma**2 + sparc_sigma**2)
    overlap_candidate_rows = []
    for candidate in admitted:
        inner_phangs_control = min(
            phangs_overlap_score["scores"][control]["loss"] for control in CONTROLS
        )
        inner_sparc_control = min(
            sparc_overlap_score["scores"][control]["loss"] for control in CONTROLS
        )
        outer_sparc_control = min(
            sparc_outer_score["scores"][control]["loss"] for control in CONTROLS
        )
        overlap_candidate_rows.append(
            {
                "candidate": candidate,
                "phangs_inner_fractional_improvement": (
                    inner_phangs_control - phangs_overlap_score["scores"][candidate]["loss"]
                )
                / inner_phangs_control,
                "sparc_inner_fractional_improvement": (
                    inner_sparc_control - sparc_overlap_score["scores"][candidate]["loss"]
                )
                / inner_sparc_control,
                "sparc_outer_fractional_improvement": (
                    outer_sparc_control - sparc_outer_score["scores"][candidate]["loss"]
                )
                / outer_sparc_control,
                "sparc_outer_loss": sparc_outer_score["scores"][candidate]["loss"],
            }
        )
    phangs_loss = {candidate: aggregate(phangs, candidate) for candidate in CONTROLS + admitted}
    sparc_loss = {candidate: aggregate(sparc, candidate) for candidate in CONTROLS + admitted}
    ranking = sorted(admitted, key=lambda candidate: (phangs_loss[candidate], candidate))
    best = ranking[0]
    best_phangs_control = min(CONTROLS, key=phangs_loss.__getitem__)
    best_sparc_control = min(CONTROLS, key=sparc_loss.__getitem__)
    phangs_improvement = (phangs_loss[best_phangs_control] - phangs_loss[best]) / phangs_loss[
        best_phangs_control
    ]
    sparc_improvement = (sparc_loss[best_sparc_control] - sparc_loss[best]) / sparc_loss[
        best_sparc_control
    ]
    cross_tracer_rows = [
        {
            "candidate": candidate,
            "phangs_fractional_improvement": (
                phangs_loss[best_phangs_control] - phangs_loss[candidate]
            )
            / phangs_loss[best_phangs_control],
            "sparc_fractional_improvement": (
                sparc_loss[best_sparc_control] - sparc_loss[candidate]
            )
            / sparc_loss[best_sparc_control],
        }
        for candidate in admitted
    ]
    support = []
    for row in phangs:
        control = min(CONTROLS, key=lambda candidate: row["scores"][candidate]["loss"])
        rg_loss = float(row["scores"][best]["loss"])
        control_loss = float(row["scores"][control]["loss"])
        support.append(
            {
                "object_id": row["object_id"],
                "rows_scored_common": row["rows_scored_common"],
                "best_comparator": control,
                "fractional_improvement": (control_loss - rg_loss) / control_loss,
                "supports": rg_loss < control_loss,
            }
        )
    print(
        json.dumps(
            {
                "scope": "DEVELOPMENT_ONLY_NO_TUNING_NO_CONFIRMATION",
                "registered_multiplicity": 9,
                "source_admitted_cells": admitted,
                "numerically_blocked_cells": [BLOCKED_CELL],
                "phangs_rows": [
                    {key: row[key] for key in ("object_id", "rows_available", "rows_scored_common")}
                    for row in phangs
                ],
                "sparc_rows": [
                    {key: row[key] for key in ("object_id", "rows_available", "rows_scored_common")}
                    for row in sparc
                ],
                "ranking": [
                    {
                        "rank": index + 1,
                        "candidate": candidate,
                        "phangs_loss": phangs_loss[candidate],
                        "sparc_loss": sparc_loss[candidate],
                    }
                    for index, candidate in enumerate(ranking)
                ],
                "best_candidate": best,
                "best_phangs_control": best_phangs_control,
                "best_sparc_control": best_sparc_control,
                "control_losses": {
                    "phangs": {candidate: phangs_loss[candidate] for candidate in CONTROLS},
                    "sparc": {candidate: sparc_loss[candidate] for candidate in CONTROLS},
                },
                "cross_tracer_improvements": cross_tracer_rows,
                "ngc2903_tracer_overlap": {
                    "radius_min_kpc": overlap_min,
                    "radius_max_kpc": overlap_max,
                    "phangs_rows_available": len(phangs_overlap),
                    "sparc_rows_available": len(sparc_overlap),
                    "phangs_rows_scored": phangs_overlap_score["rows_scored_common"],
                    "sparc_rows_scored": sparc_overlap_score["rows_scored_common"],
                    "sparc_outer_rows_available": len(sparc_outer),
                    "sparc_outer_rows_scored": sparc_outer_score["rows_scored_common"],
                    "median_absolute_fractional_velocity_difference": float(
                        np.median(tracer_fractional_difference)
                    ),
                    "maximum_absolute_fractional_velocity_difference": float(
                        np.max(tracer_fractional_difference)
                    ),
                    "rms_standardized_velocity_difference": float(
                        np.sqrt(np.mean(tracer_standardized_difference**2))
                    ),
                    "best_candidate_overlap_losses": {
                        "phangs": phangs_overlap_score["scores"][best]["loss"],
                        "sparc": sparc_overlap_score["scores"][best]["loss"],
                    },
                    "best_control_overlap_losses": {
                        "phangs": min(
                            phangs_overlap_score["scores"][candidate]["loss"]
                            for candidate in CONTROLS
                        ),
                        "sparc": min(
                            sparc_overlap_score["scores"][candidate]["loss"]
                            for candidate in CONTROLS
                        ),
                    },
                    "per_candidate_inner_outer_improvements": overlap_candidate_rows,
                },
                "phangs_fractional_improvement": phangs_improvement,
                "sparc_fractional_improvement": sparc_improvement,
                "phangs_support": support,
                "development_signal_checks": {
                    "phangs_improvement_over_2_percent": phangs_improvement > 0.02,
                    "at_least_two_phangs_objects_support": sum(row["supports"] for row in support)
                    >= 2,
                    "sparc_same_direction": sparc_improvement > 0.0,
                },
                "access": {
                    "phangs": phangs_access,
                    "sparc": sparc_access,
                    "confirmation_rows": 0,
                    "network_calls": 0,
                    "model_calls": 0,
                    "paid_calls": 0,
                    "tuning_calls": 0,
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
