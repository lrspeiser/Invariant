"""Evaluate bounded-action successors without loading astronomical responses."""
from __future__ import annotations

import argparse
import json
import platform
import sys
from hashlib import sha256
from pathlib import Path

import numpy as np
import scipy
import sympy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from invariant_gravity_extensions.actions import ActionSpec
from invariant_gravity_extensions.fields import PeriodicGrid, joint_density, solve_fields
from invariant_gravity_extensions.local_limits import Orbit, mas_per_century, perihelion_first_order
from invariant_gravity_extensions.observables import member_relative_acceleration
from invariant_gravity_extensions.saturated_actions import (
    SaturatedActionSpec,
    generate_saturated_specs,
    saturated_certificates,
)


def scene_components(grid: PeriodicGrid, definition: dict, scale: float) -> dict:
    coords = grid.coordinates()
    components = {}
    for row in definition["components"]:
        rho = np.ones(grid.shape)
        for axis in range(3):
            displacement = coords[axis] - row["centre"][axis]
            wrapped = sum(np.exp(-0.5*((displacement+k*grid.length)/row["widths"][axis])**2)
                          for k in (-1, 0, 1))
            rho *= wrapped
        components[row["id"]] = rho * (scale*row["mass"]/(rho.sum()*grid.dx**3))
    return components


def features(solution, components: dict) -> dict:
    a = solution.acceleration
    result = {"volume_acceleration_rms": float(np.sqrt(np.mean(np.sum(a*a, axis=0))))}
    for name in ("central", "member-a", "member-b"):
        density = components[name]
        relative, com = member_relative_acceleration(solution, density)
        result[name+"_relative_rms"] = float(np.sqrt(np.sum(density*np.sum(relative*relative, axis=0))/density.sum()))
        result[name+"_com_norm"] = float(np.linalg.norm(com))
    return result


def run(config: dict, local: dict, output: Path) -> dict:
    if config["access"] != {"network": False, "raw_observations": False, "sealed_products": False}:
        raise ValueError("this campaign cannot authorize observational access")
    specs = generate_saturated_specs(config["grammar"])
    cards = [s.card() for s in specs]
    certificates = [saturated_certificates(m) for m in config["grammar"]["shapes"]]
    if not all(c["all_pass"] for c in certificates):
        raise RuntimeError("symbolic action certificate failed")
    curves, local_rows = [], []
    for m in config["grammar"]["shapes"]:
        spec = SaturatedActionSpec("qumond", shape=m, epsilon=config["grammar"]["epsilon"])
        y = np.logspace(-4, 8, 121)
        curves.append({"shape": m, "gN_over_a0": y.tolist(), "delta_nu": spec.delta_nu(y).tolist()})
        for planet in local["planets"]:
            orbit = Orbit(planet["a_au"]*local["au_m"], planet["e"], local["gm_sun_m3_s2"])
            lo = planet["interval_center_mas_cy"]-planet["interval_halfwidth_mas_cy"]
            hi = planet["interval_center_mas_cy"]+planet["interval_halfwidth_mas_cy"]
            for a0 in local["a0_m_s2"]:
                angle = perihelion_first_order(orbit, a0, spec.delta_nu, nodes=128)
                rate = mas_per_century(angle, orbit, local["century_s"])
                local_rows.append({"shape": m, "planet": planet["name"], "a0_m_s2": a0,
                                   "mas_per_century": rate, "inside_monopole_interval": lo <= rate <= hi,
                                   "full_solar_system_pass": False})
    field_rows = []
    definition = config["periodic_scene"]
    # Retain completed rows even if a later solve fails; never convert a solver
    # failure into a physical rejection or overwrite a completed run.
    with (output / "field-ledger.jsonl").open("x", encoding="utf-8", newline="\n") as ledger:
        for n in definition["grids"]:
            grid = PeriodicGrid(n, definition["box_length"])
            for scale in definition["mass_scales"]:
                components = scene_components(grid, definition, scale)
                rho, source = joint_density(grid, components, subtract_background=True)
                legacy = solve_fields(grid, rho, ActionSpec("qumond", epsilon=config["grammar"]["epsilon"]), **config["solver"])
                legacy_features = features(legacy, components)
                newtonian_g = np.sqrt(np.sum(grid.gradient(legacy.newtonian)**2, axis=0))
                matched = {}
                for index, spec in enumerate(specs):
                    solved = solve_fields(grid, rho, spec, **config["solver"])
                    f = features(solved, components)
                    if spec.family == "qumond":
                        matched[spec.shape] = f
                    row = {"grid_n": n, "mass_scale": scale, "candidate": index,
                           "shape": spec.shape, "family": spec.family, "source": source,
                           "newtonian_g_percentiles": np.percentile(newtonian_g, [0, 50, 95, 100]).tolist(),
                           "features": f, "diagnostics": solved.diagnostics,
                           "fractional_change_from_same_kernel_scalar": {k: f[k]/matched[spec.shape][k]-1 for k in f},
                           "fractional_change_from_legacy_scalar": {k: f[k]/legacy_features[k]-1 for k in f}}
                    ledger.write(json.dumps(row, sort_keys=True, allow_nan=False)+"\n")
                    ledger.flush()
                    field_rows.append(row)
                print(f"Completed grid={n}, mass_scale={scale}: {len(specs)} successor solves", flush=True)
    coarse, fine = definition["grids"]
    convergence = []
    for scale in definition["mass_scales"]:
        for i in range(len(specs)):
            a, b = [next(r for r in field_rows if r["mass_scale"] == scale and r["candidate"] == i and r["grid_n"] == n)
                    for n in (coarse, fine)]
            convergence.append({"mass_scale": scale, "candidate": i,
                                "fine_over_coarse_minus_one": {k: b["features"][k]/a["features"][k]-1 for k in b["features"]},
                                "status": "TWO_GRID_DIAGNOSTIC_NOT_CONTINUUM_CERTIFICATION"})
    return {"schema": config["schema"], "claim_ceiling": config["scope"], "configuration": config,
            "cards": cards, "symbolic_certificates": certificates, "spherical_curves": curves,
            "local_monopole_screen": local_rows, "field_solves": len(field_rows),
            "legacy_control_solves": len(definition["grids"])*len(definition["mass_scales"]),
            "field_ledger_sha256": sha256((output/"field-ledger.jsonl").read_bytes()).hexdigest(),
            "convergence": convergence,
            "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                        "scipy": scipy.__version__, "sympy": sympy.__version__},
            "raw_observations_opened": 0, "sealed_products_opened": 0,
            "full_solar_system_pass": False, "galaxy_compatibility_tested": False,
            "isolated_cluster_tested": False, "derived_photon_sector": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT/"configs/gravity_saturated_actions_v1.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.config = args.config.resolve()
    args.output.mkdir(parents=True, exist_ok=False)

    def write(name, value):
        with (args.output/name).open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")

    try:
        config_bytes = args.config.read_bytes()
        config = json.loads(config_bytes)
        local_path = ROOT/config["local_reference"]
        paths = [args.config, local_path, Path(__file__), *sorted((ROOT/"src/invariant_gravity_extensions").glob("*.py"))]

        def hashes():
            return {p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else str(p): sha256(p.read_bytes()).hexdigest() for p in paths}

        before = hashes()
        config_key = args.config.relative_to(ROOT).as_posix() if args.config.is_relative_to(ROOT) else str(args.config)
        if sha256(config_bytes).hexdigest() != before[config_key]:
            raise RuntimeError("config changed before run")
        write("started.json", {"input_hashes": before})
        result = run(config, json.loads(local_path.read_bytes()), args.output)
        if hashes() != before:
            raise RuntimeError("input changed during run; outputs quarantined")
        result["input_hashes"] = before
        write("result.json", result)
        digest = sha256((args.output/"result.json").read_bytes()).hexdigest()
        write("receipt.json", {"status": "COMPLETED_AT_DECLARED_SCOPE", "result_sha256": digest,
                               "discovery_claim": False})
        print(json.dumps({"result_sha256": digest, "field_solves": result["field_solves"]}))
    except Exception as exc:
        write("failure.json", {"status": "EXECUTION_FAILURE_NOT_THEORY_REJECTION", "error": str(exc)})
        raise


if __name__ == "__main__":
    main()
