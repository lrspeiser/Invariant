"""Isolated scalar source transfer and independent discretization controls."""
from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import subprocess
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import numpy as np
import scipy
from scipy.interpolate import RegularGridInterpolator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from invariant_gravity_extensions.isolated_axisymmetric import (
    MassComponent,
    MultipoleGrid,
    anomalous_source,
    solve_isolated,
    solve_poisson,
    total_newtonian,
)
from invariant_gravity_extensions.saturated_actions import SaturatedActionSpec


def relative_vector_error(a, b):
    return float(np.max(np.linalg.norm(a-b, axis=0)/np.linalg.norm(b, axis=0)))


def finite_volume_check(config, progress):
    """Reuse the historical FV implementation without changing its code/state."""
    module_spec = importlib.util.spec_from_file_location("legacy_axisym", ROOT/config["source"])
    legacy = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(legacy)
    components = tuple(MassComponent(**c) for c in config["components"])
    spec = SaturatedActionSpec("qumond", shape=config["shape"])
    reference = solve_isolated(components, spec, config["a0"], MultipoleGrid(**config["reference_grid"]))
    probe = np.array([[.7, .2], [1.2, .4], [2, .6], [3, .9], [4, 1.2]])
    expected = reference.anomaly.evaluate(probe[:, 0], probe[:, 1])["acceleration"]
    rows = []
    for n in config["resolutions"]:
        progress(f"Independent finite-volume check {n}x{n}")
        # The legacy Grid multiplies its input lengths by KPC. Cancel that
        # unit wrapper to use the declared dimensionless control coordinates.
        grid = legacy.Grid(n, n, config["domain_R_z"]/legacy.KPC, config["domain_R_z"]/legacy.KPC)
        R, z = np.meshgrid(grid.Rc, grid.zc, indexing="ij")
        source = anomalous_source(components, spec, config["a0"], R, z)
        boundary = reference.anomaly.evaluate(R, z)["potential"]
        boundary -= boundary[-1, -1]  # irrelevant gauge, shared throughout solve
        # This input is the effective Poisson source, not baryonic density.
        potential, iterations, residual = legacy.solve_axi(
            source/(4*np.pi*legacy.G), legacy.isotropic_A(source.shape), grid,
            boundary, tol=config["tolerance"], maxiter=config["maxiter"])
        if residual > config["tolerance"]:
            raise RuntimeError(f"finite-volume solve unresolved: residual={residual}")
        gradients = np.gradient(potential, grid.dR, grid.dz, edge_order=2)
        measured = -np.array([RegularGridInterpolator((grid.Rc, grid.zc), g)(probe) for g in gradients])
        rows.append({"resolution": n, "iterations": iterations, "relative_residual": residual,
                     "anomalous_acceleration": measured.tolist(),
                     "relative_RMS_force_difference": float(np.linalg.norm(measured-expected)/np.linalg.norm(expected))})
    return {"config": config, "probe_R_z": probe.tolist(), "reference_acceleration": expected.tolist(), "rows": rows,
            "passes": rows[-1]["relative_RMS_force_difference"] < config["max_finest_relative_RMS_force_difference"] and
                      rows[-1]["relative_RMS_force_difference"] < rows[0]["relative_RMS_force_difference"],
            "independent_boundary_validation": False}


def campaign(config, progress):
    units = config["units"]
    gm_unit = units["gm_sun_m3_s2"]/(units["length_m"]*units["speed_m_s"]**2)
    a0 = config["a0_m_s2"]*units["length_m"]/units["speed_m_s"]**2
    grid_config = config["grid"]
    rows, newtonian = [], []
    for scene in config["scenes"]:
        components = tuple(MassComponent(c["name"], c["mass_msun"]*gm_unit, c["a_kpc"], c["b_kpc"])
                           for c in scene["components"])
        spherical = all(c.a == 0 for c in components)
        r = np.asarray(scene["radii_kpc"])
        R, z = np.concatenate([r, r]), np.concatenate([np.zeros_like(r), r*scene["offplane_z_over_R"]])
        grids = []
        for definition in grid_config["refinement"]:
            values = definition.copy()
            if spherical:
                values.update(angular_nodes=grid_config["spherical_angular_nodes"], l_max=grid_config["spherical_l_max"])
            grids.append(MultipoleGrid(scene["radial_scale_kpc"]*grid_config["r_min_over_scale"],
                                       scene["radial_scale_kpc"]*grid_config["r_max_over_scale"],
                                       **values, plane_scale=None if spherical else min(c.b for c in components)))
        wide = asdict(grids[-1])
        wide["r_min"] *= grid_config["boundary_inner_factor"]
        wide["r_max"] *= grid_config["boundary_outer_factor"]
        grids.append(MultipoleGrid(**wide))
        exact_newtonian = -total_newtonian(components, R, z)["gradient"]
        for grid in grids:
            reconstructed = solve_poisson(grid, lambda R, z: total_newtonian(components, R, z)["laplacian"])
            newtonian.append({"scene": scene["id"], "grid": asdict(grid),
                              "max_relative_force_error": relative_vector_error(reconstructed.evaluate(R, z)["acceleration"], exact_newtonian)})
        for shape in config["shapes"]:
            spec = SaturatedActionSpec("qumond", shape=shape, epsilon=config["epsilon"])
            solutions = []
            for grid in grids:
                progress(f"{scene['id']} m={shape}, radial={grid.radial_nodes}, angular={grid.angular_nodes}, l_max={grid.l_max}, r_max={grid.r_max}")
                solution = solve_isolated(components, spec, a0, grid)
                solutions.append(solution.evaluate(R, z))
            fine, wide_solution = solutions[-2], solutions[-1]
            magnitude = np.linalg.norm(exact_newtonian, axis=0)
            algebraic = exact_newtonian*(1+spec.delta_nu(magnitude/a0))[None, :]
            v2 = -r*wide_solution["acceleration"][0, :len(r)]
            row = {"scene": scene["id"], "shape": shape, "a0_m_s2": config["a0_m_s2"],
                   "card_sha256": spec.card()["content_sha256"], "components": [asdict(c) for c in components],
                   "probe_R_z_kpc": np.array([R, z]).T.tolist(),
                   "grids": [asdict(g) for g in grids],
                   "accelerations_kms2_per_kpc": [s["acceleration"].tolist() for s in solutions],
                   "newtonian_acceleration_kms2_per_kpc": exact_newtonian.tolist(),
                   "max_refinement_relative_force_change": relative_vector_error(fine["acceleration"], solutions[0]["acceleration"]),
                   "max_boundary_relative_force_change": relative_vector_error(wide_solution["acceleration"], fine["acceleration"]),
                   "algebraic_shortcut_relative_force_difference_by_point": (np.linalg.norm(algebraic-wide_solution["acceleration"], axis=0)/np.linalg.norm(wide_solution["acceleration"], axis=0)).tolist(),
                   "spherical_exact_relative_force_error": relative_vector_error(wide_solution["acceleration"], algebraic) if spherical else None,
                   "midplane_circular_v_squared_kms2": v2.tolist(),
                   "midplane_circular_v_km_s": [float(np.sqrt(v)) if v >= 0 else None for v in v2],
                   "midplane_acceleration_ratio_to_newtonian": (wide_solution["acceleration"][0, :len(r)]/exact_newtonian[0, :len(r)]).tolist(),
                   "inward_midplane_radial_force": bool(np.all(v2 > 0)),
                   "scope": scene["role"], "empirical_validation": False}
            rows.append(row)
    controls = config["controls"]
    worst_refinement = max(r["max_refinement_relative_force_change"] for r in rows)
    worst_boundary = max(r["max_boundary_relative_force_change"] for r in rows)
    worst_newtonian = max(r["max_relative_force_error"] for r in newtonian if r["grid"]["radial_nodes"] == grid_config["refinement"][-1]["radial_nodes"])
    worst_spherical = max(r["spherical_exact_relative_force_error"] for r in rows if r["spherical_exact_relative_force_error"] is not None)
    summary = {"candidate_scenes": len(rows), "candidate_joint_solves": 3*len(rows),
               "worst_refinement": worst_refinement, "worst_boundary_change": worst_boundary,
               "worst_fine_newtonian_reconstruction": worst_newtonian, "worst_spherical_exact_error": worst_spherical,
               "numerical_targets_met": (worst_refinement < controls["max_force_refinement_relative"] and
                                         worst_boundary < controls["max_boundary_force_change_relative"] and
                                         worst_newtonian < controls["max_newtonian_reconstruction_relative"] and
                                         worst_spherical < controls["max_spherical_relative_error"])}
    return {"summary": summary, "rows": rows, "newtonian_reconstruction": newtonian,
            "converted_units": {"gm_per_solar_mass_kpc_kms2": gm_unit, "a0_kms2_per_kpc": a0}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config_path = ROOT/"configs/gravity_isolated_scalar_v1.json"
    config = json.loads(config_path.read_bytes())
    paths = [Path(__file__), config_path, ROOT/config["finite_volume_check"]["source"], ROOT/config["predecessor"],
             *sorted((ROOT/"src/invariant_gravity_extensions").glob("*.py"))]

    def hashes():
        return {p.relative_to(ROOT).as_posix(): sha256(p.read_bytes()).hexdigest() for p in paths}

    def write(name, value):
        with (args.output/name).open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")

    before = hashes()
    provenance = {"input_hashes": before, "started_utc": datetime.now(UTC).isoformat(),
                  "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  "source_hashes_authoritative": True, "python": platform.python_version(),
                  "numpy": np.__version__, "scipy": scipy.__version__}
    write("started.json", {"config": config, **provenance})
    try:
        progress = lambda message: print(message, flush=True)
        result = campaign(config, progress)
        independent = finite_volume_check(config["finite_volume_check"], progress)
        previous = json.loads((ROOT/config["predecessor"]).read_bytes())
        cassini = [{k: r[k] for k in ("shape", "a0_m_s2", "physical_external_m_s2", "Q2_s_minus2", "status")}
                   for r in previous["rows"] if r["a0_m_s2"] == config["a0_m_s2"]]
        if hashes() != before:
            raise RuntimeError("inputs changed during run")
        write("result.json", {"config": config, **provenance, **result,
                              "independent_finite_volume": independent,
                              "prior_cassini_summary_scenarios": cassini,
                              "empirical_three_regime_validation": False, "discovery_claim": False})
        if not result["summary"]["numerical_targets_met"] or not independent["passes"]:
            raise RuntimeError("implementation targets unresolved; result retained, not a physical rejection")
        write("receipt.json", {"status": "COMPLETED_AT_DECLARED_SCOPE",
                               "result_sha256": sha256((args.output/"result.json").read_bytes()).hexdigest()})
        print(json.dumps({**result["summary"], "finite_volume_passes": independent["passes"]}))
    except Exception as exc:
        write("failure.json", {"status": "EXECUTION_FAILURE_NOT_PHYSICAL_REJECTION", "error": str(exc)})
        raise


if __name__ == "__main__":
    main()
