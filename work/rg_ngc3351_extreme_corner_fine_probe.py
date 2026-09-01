"""Source-only 0.20 kpc convergence probe for the sole limiting RG corner."""

from __future__ import annotations

import gc
import json
import math

import numpy as np

from sigma_theory_compiler import open_gravity_refracted_gravity_3d_primary_benchmark_v1 as rg
from sigma_theory_compiler import (
    open_gravity_refracted_gravity_phangs_things_scoring_resolution_v1 as resolution,
)
from sigma_theory_compiler import (
    open_gravity_refracted_gravity_published_prior_development_scan_v1 as prior,
)


OBJECT_ID = "NGC3351"
CELL_ID = "PRIOR_CORNER_E0.1_Q2_R-23"
NODES = 301


def main() -> None:
    config = prior.load_config()
    resolution_config = resolution.load_config()
    source_config, acquisition, expected, bridge_config = resolution._source_evidence()
    maps, exponential_scale_pc = resolution._primary_maps(
        source_config, acquisition, OBJECT_ID
    )
    built = resolution._build_density(
        resolution_config,
        bridge_config,
        maps,
        exponential_scale_pc=exponential_scale_pc,
        nodes=NODES,
    )
    rhs = 4.0 * math.pi * built["density_dimensionless"]
    newton, newton_residual = resolution.solve_poisson_dst(
        rhs, built["newton_boundary"], built["grid"].spacing
    )
    cell = next(row for row in prior.parameter_cells(config) if row["id"] == CELL_ID)
    epsilon = rg.published_permittivity(
        built["density_g_cm3"],
        epsilon_0=float(cell["epsilon_0"]),
        rho_c=10.0 ** float(cell["log10_rho_c_g_cm3"]),
        q_slope=float(cell["Q"]),
    )
    operator = resolution_config["operator_contract"]
    potential, metrics = resolution.solve_variable_pcg(
        rhs,
        built["newton_boundary"] / float(cell["epsilon_0"]),
        epsilon,
        built["grid"].spacing,
        relative_tolerance=float(operator["pcg_relative_tolerance"]),
        absolute_tolerance=float(operator["pcg_absolute_tolerance"]),
        max_iterations=int(operator["pcg_max_iterations"]),
        initial_potential=newton / float(cell["epsilon_0"]),
    )
    profile_020 = prior._profile(potential, built, config, bridge_config)

    receipt = prior._read_json(prior._output_path(), "prior scan receipt")
    row = next(row for row in receipt["source_field_rows"] if row["object_id"] == OBJECT_ID)
    profile_025 = row["fine"]["profiles"][CELL_ID]
    profile_03125 = row["convergence"]["profiles"][CELL_ID]
    radius_020 = np.asarray([float(row["radius_kpc"]) for row in profile_020])
    radius_025 = np.asarray([float(row["radius_kpc"]) for row in profile_025])
    if not np.array_equal(radius_020, radius_025):
        raise RuntimeError("radial grids changed")
    value_020 = np.asarray([float(row["radial_acceleration_m_s2"]) for row in profile_020])
    value_025 = np.asarray([float(row["radial_acceleration_m_s2"]) for row in profile_025])
    value_03125 = np.asarray(
        [float(row["radial_acceleration_m_s2"]) for row in profile_03125]
    )
    relative = np.abs(value_020 - value_025) / np.maximum.reduce(
        [np.abs(value_020), np.abs(value_025), np.full_like(value_020, 1.0e-30)]
    )
    prior_relative = np.abs(value_025 - value_03125) / np.maximum.reduce(
        [np.abs(value_025), np.abs(value_03125), np.full_like(value_025, 1.0e-30)]
    )
    failure = relative > 0.05
    failure_indices = np.flatnonzero(failure)
    bands: list[dict[str, float | int]] = []
    if failure_indices.size:
        starts = [int(failure_indices[0])]
        ends: list[int] = []
        for previous, current in zip(failure_indices[:-1], failure_indices[1:], strict=True):
            if int(current) != int(previous) + 1:
                ends.append(int(previous))
                starts.append(int(current))
        ends.append(int(failure_indices[-1]))
        bands = [
            {
                "radius_min_kpc": float(radius_020[start]),
                "radius_max_kpc": float(radius_020[end]),
                "points": end - start + 1,
                "maximum_relative_difference": float(np.max(relative[start : end + 1])),
            }
            for start, end in zip(starts, ends, strict=True)
        ]
    converging = prior_relative > relative
    valid_order = (prior_relative > 1.0e-14) & (relative > 1.0e-14)
    observed_order = np.log(prior_relative[valid_order] / relative[valid_order]) / math.log(
        1.25
    )
    expected_source = expected[(OBJECT_ID, resolution_config["source_cell"]["id"])]
    result = {
        "scope": "SOURCE_ONLY_NO_VELOCITY_VALUES",
        "object_id": OBJECT_ID,
        "parameter_id": CELL_ID,
        "nodes": NODES,
        "spacing_kpc": built["grid"].spacing
        * float(config["grid_contract"]["solver_half_box_kpc"]),
        "newton_relative_residual": newton_residual,
        "rg_solver_metrics": metrics,
        "source_mass_relative_error": resolution.source_systematics._mass_error(
            built["masses"], expected_source
        ),
        "dimensionless_mass_relative_error": built["dimensionless_mass_relative_error"],
        "profile_points": int(relative.size),
        "points_within_5_percent": int(np.sum(~failure)),
        "points_over_5_percent": int(np.sum(failure)),
        "prior_points_over_5_percent_025_vs_03125": int(np.sum(prior_relative > 0.05)),
        "points_with_shrinking_difference": int(np.sum(converging)),
        "points_with_nonshrinking_difference": int(np.sum(~converging)),
        "median_relative_difference_020_vs_025": float(np.median(relative)),
        "median_relative_difference_025_vs_03125": float(np.median(prior_relative)),
        "median_observed_convergence_order": float(np.median(observed_order)),
        "maximum_relative_difference_020_vs_025": float(np.max(relative)),
        "maximum_difference_radius_kpc": float(radius_020[int(np.argmax(relative))]),
        "failure_radius_min_kpc": float(np.min(radius_020[failure])) if np.any(failure) else None,
        "failure_radius_max_kpc": float(np.max(radius_020[failure])) if np.any(failure) else None,
        "failure_bands": bands,
        "density_sha256": resolution.array_sha256(built["density_dimensionless"]),
        "potential_sha256": resolution.array_sha256(potential),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    del maps, built, rhs, newton, epsilon, potential
    gc.collect()


if __name__ == "__main__":
    main()
