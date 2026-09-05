"""Bounded synthetic coupling scan using, and testing, exact quadratic scaling."""
from __future__ import annotations

import argparse
import json
import sys
from hashlib import sha256
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from run_gravity_saturated_actions import features, scene_components

from invariant_gravity_extensions.fields import (
    FieldSolution,
    PeriodicGrid,
    joint_density,
    solve_fields,
)
from invariant_gravity_extensions.observables import member_relative_acceleration
from invariant_gravity_extensions.saturated_actions import SaturatedActionSpec


def signed_confinement(solution, components, definition):
    """Mass-weighted -r_relative dot a_relative; positive means inward virial.

    Positions are shortest periodic displacements around each declared centre,
    then centered by their mass-weighted mean. This instantaneous diagnostic
    makes no equilibrium or velocity-dispersion claim.
    """
    grid = solution.grid
    coords = grid.coordinates()
    result = {}
    for component in definition["components"]:
        name = component["id"]
        if name == "gas":
            continue
        rho = components[name]
        centre = np.array(component["centre"])[:, None, None, None]
        position = (coords-centre+grid.length/2) % grid.length-grid.length/2
        mean = np.sum(position*rho[None], axis=(1, 2, 3))/rho.sum()
        position -= mean[:, None, None, None]
        relative, _ = member_relative_acceleration(solution, rho)
        result[name] = float(-np.sum(rho*np.sum(position*relative, axis=0))/rho.sum())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config_path = ROOT / "configs/gravity_saturated_actions_v1.json"
    paths = [Path(__file__), ROOT / "scripts/run_gravity_saturated_actions.py", config_path,
             *sorted((ROOT / "src/invariant_gravity_extensions").glob("*.py"))]

    def hashes():
        return {p.relative_to(ROOT).as_posix(): sha256(p.read_bytes()).hexdigest() for p in paths}

    def write(name, value):
        with (args.output / name).open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")

    before = hashes()
    settings = {"grid": 33, "kernel_shape": 1, "beta": 2, "powers": [1, 2],
                "mass_scales": [.2, 2, 20], "mixing": [0, .75, 3, 10, 30],
                "direct_check_mixing": [10, 30],
                "scope": "synthetic amplitude feasibility; no physical amplitude selected or observed response fit"}
    write("started.json", {"settings": settings, "input_hashes": before})
    try:
        config = json.loads(config_path.read_bytes())
        grid = PeriodicGrid(settings["grid"], config["periodic_scene"]["box_length"])
        rows, checks = [], []
        for scale in settings["mass_scales"]:
            components = scene_components(grid, config["periodic_scene"], scale)
            rho, _ = joint_density(grid, components, subtract_background=True)
            base_spec = SaturatedActionSpec("qumond", shape=1)
            base = solve_fields(grid, rho, base_spec)
            base_features = features(base, components)
            base_virial = signed_confinement(base, components, config["periodic_scene"])
            for power in settings["powers"]:
                unit = solve_fields(grid, rho, SaturatedActionSpec("trimond_alignment", 1, 2, power=power, shape=1))
                delta = unit.physical-base.physical
                for mixing in settings["mixing"]:
                    extrapolated = FieldSolution(base_spec, grid, base.newtonian,
                                                 base.physical+mixing**2*delta, None, {})
                    predicted = features(extrapolated, components)
                    virial = signed_confinement(extrapolated, components, config["periodic_scene"])
                    rows.append({"mass_scale": scale, "power": power, "mixing": mixing,
                                 "features": predicted,
                                 "signed_confinement": virial,
                                 "confinement_fractional_change_from_scalar": {k: virial[k]/base_virial[k]-1 for k in virial},
                                 "all_component_virials_inward": all(v > 0 for v in virial.values()),
                                 "fractional_change_from_scalar": {k: predicted[k]/base_features[k]-1 for k in predicted}})
                    if mixing in settings["direct_check_mixing"]:
                        direct = solve_fields(grid, rho, SaturatedActionSpec("trimond_alignment", mixing, 2, power=power, shape=1))
                        accel = direct.acceleration
                        checks.append({"mass_scale": scale, "power": power, "mixing": mixing,
                                       "relative_field_prediction_error": float(np.linalg.norm(accel-extrapolated.acceleration)/np.linalg.norm(accel))})
        if hashes() != before:
            raise RuntimeError("inputs changed during run")
        if max(r["relative_field_prediction_error"] for r in checks) > 1e-7:
            raise RuntimeError("direct amplitude checks failed")
        write("result.json", {"settings": settings, "input_hashes": before,
                              "rows": rows, "independent_amplitude_solves": checks,
                              "mechanism": "chi proportional to mixing; physical field correction proportional to mixing squared",
                              "universal_amplitude_fitted": False, "galaxy_compatibility_claim": False,
                              "cluster_success_claim": False, "stability_claim": False})
        write("receipt.json", {"status": "COMPLETED_AT_DECLARED_SCOPE",
                               "result_sha256": sha256((args.output/"result.json").read_bytes()).hexdigest()})
        print(json.dumps({"scan_points": len(rows), "direct_checks": len(checks),
                          "max_relative_prediction_error": max(r["relative_field_prediction_error"] for r in checks)}))
    except Exception as exc:
        write("failure.json", {"status": "EXECUTION_FAILURE_NOT_PHYSICAL_REJECTION", "error": str(exc)})
        raise


if __name__ == "__main__":
    main()
