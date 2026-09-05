"""Sealed controls and conditional external-quadrupole TRIMOND development scan."""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from invariant_gravity_extensions.external_multifield import (
    FluxPoissonSolver,
    beta_zero_source,
    physical_auxiliary_flux,
    point_external_gradient,
    solve_external_auxiliary,
)
from invariant_gravity_extensions.external_quadrupole import (
    quadrupole_integrals,
    saturated_nu_derivative,
)
from invariant_gravity_extensions.isolated_axisymmetric import MultipoleGrid, solve_poisson
from invariant_gravity_extensions.saturated_actions import SaturatedActionSpec


def controls(config):
    grid = MultipoleGrid(**config["grids"][-1])
    solver = FluxPoissonSolver(grid)
    r, mu, sine = solver.radius[:, None], solver.mu, solver.sine
    e = np.exp(-r*r)
    flux = np.array([e*(-2*r*(1+.3*r*mu)+.3*mu), -.3*e*sine+np.zeros_like(r)])
    reconstructed = solver.gradient(solver.solve(flux))
    select = (solver.radius > .03) & (solver.radius < 3)
    manufactured_field = float(np.linalg.norm((reconstructed-flux)[:, select])/np.linalg.norm(flux[:, select]))
    qflux = np.array([r**3*e*(3*mu*mu-1)/2, np.zeros((len(r), mu.size))])
    manufactured_q2 = abs(solver.quadrupole(qflux)["Q2_volume"]-.9)
    c = config["theory_controls"]
    aux0 = solve_external_auxiliary(solver, c["eta_newtonian"], c["mixing"], 0, c["power"], **config["solver"])
    source = solve_poisson(grid, lambda R, z: beta_zero_source(c["eta_newtonian"], c["mixing"], c["power"], R, z))
    probes = np.asarray(config["probes_R_z"])
    a = aux0.potential.evaluate(*probes.T)["acceleration"]
    b = source.evaluate(*probes.T)["acceleration"]
    beta_zero_error = float(np.linalg.norm(a-b)/np.linalg.norm(b))
    unit = solve_external_auxiliary(solver, c["eta_newtonian"], 1, c["beta"], c["power"], **config["solver"])
    half = solve_external_auxiliary(solver, c["eta_newtonian"], -.5, c["beta"], c["power"], **config["solver"])
    fq = physical_auxiliary_flux(unit.p, unit.q, 1, c["beta"], c["power"])
    fh = physical_auxiliary_flux(half.p, half.q, -.5, c["beta"], c["power"])
    scale_error = float(max(np.linalg.norm(half.q+.5*unit.q)/np.linalg.norm(unit.q),
                            np.linalg.norm(fh-.25*fq)/np.linalg.norm(fq)))
    scalar = SaturatedActionSpec("qumond", shape=1)
    p = point_external_gradient(solver, c["eta_newtonian"])
    scalar_flux = scalar.delta_nu(np.linalg.norm(p, axis=0))*p
    exact_q = quadrupole_integrals(c["eta_newtonian"], scalar.delta_nu,
                                 lambda y: saturated_nu_derivative(scalar, y), nodes=1024)
    scalar_Q2 = solver.quadrupole(scalar_flux)
    scalar_error = abs(scalar_Q2["Q2_volume"]+1.5*exact_q["q_milgrom"])
    checks = {"max_manufactured_relative_field_error": manufactured_field,
              "max_manufactured_absolute_Q2_error": manufactured_q2,
              "max_beta_zero_relative_field_disagreement": beta_zero_error,
              "max_scaling_relative_disagreement": scale_error,
              "max_scalar_absolute_Q2_disagreement": scalar_error}
    passes = {name: value < config["controls"][name] for name, value in checks.items()}
    return {"checks": checks, "passes": passes, "all_pass": all(passes.values()),
            "scalar_reference_integrals": exact_q, "scalar_flux_quadrupole": scalar_Q2,
            "beta_zero_flux_probe_acceleration": a.tolist(), "beta_zero_source_probe_acceleration": b.tolist(),
            "unit_auxiliary_iterations": unit.iterations, "unit_auxiliary_history": unit.history}


def one_auxiliary(config, solver, eta, beta, power):
    aux = solve_external_auxiliary(solver, eta, 1, beta, power, **config["solver"])
    flux = physical_auxiliary_flux(aux.p, aux.q, 1, beta, power)
    quad = solver.quadrupole(flux)
    probes = np.asarray(config["probes_R_z"])
    q_probes = -aux.potential.evaluate(*probes.T)["acceleration"]
    q_probes[1] -= eta/(1+eta**2)**power
    return {"beta": beta, "power": power, "eta_newtonian": eta, **quad,
            "iterations": aux.iterations, "relative_update_energy": aux.relative_update_energy,
            "max_absolute_update": aux.max_absolute_update,
            "probe_q_R_z": q_probes.T.tolist(), "history": aux.history}


def campaign(config, previous, common_controls):
    old = previous["rows"]
    solvers = [FluxPoissonSolver(MultipoleGrid(**g)) for g in config["grids"]]
    rows, auxiliary = [], []
    legacy_config = json.loads((ROOT/config["historical_summary_config"]).read_bytes())
    summary = legacy_config["cassini_summary"]
    low = summary["mean_Q2_s_minus2"]-summary["screen_sigma_multiplier"]*summary["one_sigma_s_minus2"]
    high = summary["mean_Q2_s_minus2"]+summary["screen_sigma_multiplier"]*summary["one_sigma_s_minus2"]
    grammar = config["grammar"]
    combinations = [(beta, power) for beta in grammar["beta"] for power in grammar["powers"]]
    for index, prior in enumerate(old):
        eta = prior["quadrature"][-1]["eta_newtonian"]
        print(f"Scenario {index+1}/{len(old)}: m={prior['shape']}, a0={prior['a0_m_s2']}, external={prior['physical_external_m_s2']}", flush=True)
        scalar_spec = SaturatedActionSpec("qumond", shape=prior["shape"], epsilon=grammar["epsilon"])
        scalar_flux_values, grid_results = [], []
        for solver in solvers:
            p = point_external_gradient(solver, eta)
            scalar_flux_values.append(solver.quadrupole(scalar_spec.delta_nu(np.linalg.norm(p, axis=0))*p))
            with ThreadPoolExecutor(max_workers=3) as pool:
                jobs = [pool.submit(one_auxiliary, config, solver, eta, b, power) for b, power in combinations]
                grid_results.append([job.result() for job in jobs])
            print(f"  completed {solver.grid.radial_nodes} x {solver.grid.angular_nodes}, l={solver.grid.l_max}", flush=True)
        scalar_dimless = -1.5*prior["quadrature"][-1]["q_milgrom"]
        scalar_check = abs(scalar_flux_values[-1]["Q2_volume"]-scalar_dimless)
        conversion = prior["a0_m_s2"]**1.5/np.sqrt(legacy_config["gm_sun_m3_s2"])
        for j, (beta, power) in enumerate(combinations):
            solves = [results[j] for results in grid_results]
            values = [s["Q2_volume"] for s in solves]
            checks = {"max_auxiliary_refinement_Q2_error": abs(values[1]-values[0]),
                      "max_auxiliary_boundary_Q2_error": abs(values[2]-values[1]),
                      "max_auxiliary_surface_Q2": abs(solves[-1]["Q2_surface"]),
                      "max_scalar_absolute_Q2_disagreement": scalar_check}
            passes = {k: v < config["controls"][k] for k, v in checks.items()}
            audit = {"scenario_index": index, "shape": prior["shape"], "a0_m_s2": prior["a0_m_s2"],
                     "physical_external_m_s2": prior["physical_external_m_s2"], "beta": beta, "power": power,
                     "unit_mixing_solves": solves, "checks": checks, "checks_passed": passes,
                     "scalar_flux_quadrupoles": scalar_flux_values}
            auxiliary.append(audit)
            for mixing in grammar["mixing"]:
                spec = SaturatedActionSpec("trimond_alignment", shape=prior["shape"], mixing=mixing,
                                           beta=beta, power=power, epsilon=grammar["epsilon"])
                correction = mixing**2*values[-1]*conversion
                total = prior["Q2_s_minus2"]+correction
                spread = max(checks["max_auxiliary_refinement_Q2_error"], checks["max_auxiliary_boundary_Q2_error"],
                             checks["max_auxiliary_surface_Q2"])*mixing**2*conversion+prior["empirical_Q2_spread_not_error_bound"]
                numerical = common_controls["all_pass"] and all(passes.values())
                classification = ("UNRESOLVED_NUMERICAL_CONTROLS" if not numerical else
                                  "NUMERICALLY_NEAR_SUMMARY_BOUNDARY" if min(abs(total-low), abs(total-high)) <= spread else
                                  "WITHIN_CONDITIONAL_HISTORICAL_SUMMARY_SCREEN" if low <= total <= high else
                                  "OUTSIDE_CONDITIONAL_HISTORICAL_SUMMARY_SCREEN")
                rows.append({"scenario_index": index, "card_sha256": spec.card()["content_sha256"],
                             "shape": spec.shape, "mixing": mixing, "beta": beta, "power": power,
                             "a0_m_s2": prior["a0_m_s2"], "physical_external_m_s2": prior["physical_external_m_s2"],
                             "eta_newtonian": eta, "scalar_Q2_s_minus2": prior["Q2_s_minus2"],
                             "auxiliary_Q2_s_minus2": correction, "total_Q2_s_minus2": total,
                             "numerical_spread_not_certified_error_bound": spread,
                             "numerical_controls_pass": numerical, "status": classification,
                             "full_solar_system_pass": False, "theory_falsified": False})
    return {"rows": rows, "auxiliary_scenarios": auxiliary,
            "summary_interval_Q2_s_minus2": [low, high],
            "summary": {"rows": len(rows), "auxiliary_solves": len(auxiliary)*len(solvers),
                        "status_counts": {status: sum(r["status"] == status for r in rows) for status in sorted({r["status"] for r in rows})},
                        "discovery_claim": False, "full_solar_system_pass": False}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT/"configs/gravity_multifield_external_v1.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--controls-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads(args.config.read_bytes())
    paths = [Path(__file__).resolve(), args.config.resolve(), ROOT/config["scalar_predecessor"],
             ROOT/config["historical_summary_config"], *sorted((ROOT/"src/invariant_gravity_extensions").glob("*.py"))]

    def hashes():
        return {p.relative_to(ROOT).as_posix(): sha256(p.read_bytes()).hexdigest() for p in paths}

    def write(name, obj):
        with (args.output/name).open("x", encoding="utf-8", newline="\n") as file:
            json.dump(obj, file, indent=2, sort_keys=True, allow_nan=False)
            file.write("\n")

    before = hashes()
    for source in paths:
        target = args.output/"input-snapshots"/source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    provenance = {"started_utc": datetime.now(UTC).isoformat(), "input_hashes": before,
                  "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                  "controls_only": args.controls_only, "thread_workers": 3, "raw_observations_accessed": False}
    write("started.json", {"config": config, **provenance})
    try:
        print("Independent analytic and action controls", flush=True)
        checks = controls(config)
        write("controls.json", checks)
        print(json.dumps(checks["checks"]), flush=True)
        if not checks["all_pass"]:
            raise RuntimeError("Preflight numerical controls unresolved; scenario scan not started")
        result = {} if args.controls_only else campaign(config, json.loads((ROOT/config["scalar_predecessor"]).read_bytes()), checks)
        if hashes() != before:
            raise RuntimeError("Inputs changed during run")
        write("result.json", {"config": config, **provenance, "controls": checks, **result})
        write("receipt.json", {"status": "COMPLETED_AT_DECLARED_SCOPE", "result_sha256": sha256((args.output/"result.json").read_bytes()).hexdigest()})
        print(json.dumps(result.get("summary", {"controls_pass": True})), flush=True)
    except Exception as exc:
        write("failure.json", {"status": "EXECUTION_OR_NUMERICAL_FAILURE_NOT_PHYSICAL_REJECTION", "error": str(exc)})
        raise


if __name__ == "__main__":
    main()
