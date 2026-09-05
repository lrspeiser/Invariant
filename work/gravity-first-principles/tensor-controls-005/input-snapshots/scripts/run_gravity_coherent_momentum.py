"""Freeze and run a synthetic internal-force audit of Sigma's P0696 base."""
from __future__ import annotations

import argparse
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from invariant_gravity_extensions.coherent_momentum import audit_scene
from invariant_gravity_extensions.isolated_axisymmetric import (
    MassComponent,
    MultipoleGrid,
    total_newtonian,
)


def legacy_replay(config, parts):
    imported = ROOT/config["source_import"]
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(imported/"source-snapshots/frontiers/src"))
    from voidscreen.coherent_monopole import coherent_monopole_potential
    from voidscreen.field_solvers import cell_coordinates
    # Refuse an unrelated previously imported package with the same name.
    module_path = Path(sys.modules[coherent_monopole_potential.__module__].__file__).resolve()
    if not module_path.is_relative_to(imported.resolve()):
        raise RuntimeError("legacy import did not resolve to the frozen source snapshot")
    rows = []
    for n in config["grid_nodes"]:
        print(f"Unchanged legacy Cartesian replay {n}^3", flush=True)
        step = config["box_width"]/(n-1)
        x, y, z = cell_coordinates((n, n, n), step)
        R = np.hypot(x, y)
        fields = total_newtonian(parts, R, z)
        rho = fields["laplacian"]/(4*np.pi)
        radial = np.divide(-fields["gradient"][0], R, out=np.zeros_like(R), where=R > 0)
        acceleration = (radial*x, radial*y, -fields["gradient"][1])
        solution = coherent_monopole_potential(rho, fields["potential"], acceleration, step, a0=config["a0"])
        mass = np.sum(rho)*step**3
        force = np.array([np.sum(rho*g)*step**3 for g in solution.correction_acceleration])
        newton = np.array([np.sum(rho*g)*step**3 for g in acceleration])
        scale = np.sum(rho*np.sqrt(sum(g*g for g in acceleration)))*step**3
        rows.append({"nodes": n, "step": step, "mass_in_cube": float(mass),
                     "center_of_mass": list(solution.center_of_mass),
                     "correction_net_force": force.tolist(), "analytic_newtonian_net_force": newton.tolist(),
                     "force_normalizer": float(scale),
                     "normalized_correction_net_force": float(np.linalg.norm(force)/scale),
                     "scope": config["role"]})
    return rows


def judge(config, rows):
    by_id = {s["id"]: [r for r in rows if r["scene"] == s["id"]] for s in config["scenes"]}
    a, reflected = by_id["asymmetric"], by_id["reflected"]
    net = [r["correction_net_force_z"] for r in a]
    checks = {
        "max_normalized_newtonian_net_force": max(r["normalized_newtonian_net_force"] for r in rows),
        "max_normalized_symmetric_force": max(r["normalized_correction_net_force"] for r in rows if r["scene"] in ("symmetric", "concentric")),
        "max_relative_reflection_mismatch": max(abs(x["correction_net_force_z"]+y["correction_net_force_z"])/max(abs(x["correction_net_force_z"]), 1e-300) for x, y in zip(a, reflected, strict=True)),
        "max_relative_force_refinement": abs(net[1]-net[0])/max(abs(net[1]), 1e-300),
        "max_relative_force_boundary_change": abs(net[2]-net[1])/max(abs(net[2]), 1e-300),
        "max_relative_shell_gauss_error_for_r_at_least_0_01": max(r["max_relative_shell_gauss_error_for_r_at_least_0_01"] for r in rows),
        "max_relative_fine_mass_deficit": max(abs(r["relative_mass_deficit"]) for r in rows if r["grid_index"] > 0),
    }
    passed = {k: v < config["controls"][k] for k, v in checks.items()}
    numerical = all(passed.values())
    witness = numerical and a[-1]["normalized_correction_net_force"] > config["controls"]["min_normalized_candidate_force_for_resolved_witness"]
    return {"checks": checks, "checks_passed": passed, "numerical_controls_pass": numerical,
            "resolved_nonzero_internal_force": witness,
            "status": "EXACT_BASE_FAILS_CLOSED_MOMENTUM_BALANCE" if witness else
            "UNRESOLVED_NUMERICAL_CONTROLS" if not numerical else "NO_RESOLVED_WITNESS_IN_THIS_SCENE",
            "discovery_claim": False, "all_coherence_families_rejected": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT/"configs/gravity_coherent_momentum_v1.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_bytes())
    args.output.mkdir(parents=True, exist_ok=False)
    imported = ROOT/config["legacy_replay"]["source_import"]
    manifest_path = imported/"manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    paths = [Path(__file__).resolve(), args.config.resolve(), manifest_path,
             *[ROOT/"src/invariant_gravity_extensions"/name for name in
               ("coherent_momentum.py", "isolated_axisymmetric.py", "saturated_actions.py")],
             *[imported/e["snapshot"] for e in manifest["files"]]]

    def hashes():
        return {p.relative_to(ROOT).as_posix(): sha256(p.read_bytes()).hexdigest() for p in paths}

    def write(name, value):
        with (args.output/name).open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")

    before = hashes()
    for e in manifest["files"]:
        if sha256((imported/e["snapshot"]).read_bytes()).hexdigest() != e["sha256"]:
            raise RuntimeError("Sigma snapshot hash differs from import manifest")
    # Seal the exact running bytes, in addition to hashes and repository HEAD.
    for path in paths[:6]:
        target = args.output/"source-snapshots"/path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    (args.output/".gitattributes").write_text("source-snapshots/** -text\n", encoding="utf-8")
    provenance = {"input_hashes": before, "started_utc": datetime.now(UTC).isoformat(),
                  "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                  "config_frozen_before_metrics": True, "raw_observations_accessed": False}
    write("started.json", {"config": config, **provenance})
    try:
        rows = []
        for scene in config["scenes"]:
            parts = tuple(MassComponent(**c) for c in scene["components"])
            for i, definition in enumerate(config["grids"]):
                grid = MultipoleGrid(**definition)
                print(f"Continuum {scene['id']} radial={grid.radial_nodes} angular={grid.angular_nodes}", flush=True)
                rows.append({"scene": scene["id"], "components": scene["components"],
                             "grid_index": i, "grid": asdict(grid), **audit_scene(parts, config["a0"], grid)})
        summary = judge(config, rows)
        parts = tuple(MassComponent(**c) for c in next(s for s in config["scenes"] if s["id"] == config["legacy_replay"]["scene"])["components"])
        replay = legacy_replay({**config["legacy_replay"], "a0": config["a0"]}, parts)
        if hashes() != before:
            raise RuntimeError("inputs changed during audit")
        write("result.json", {"config": config, **provenance, "rows": rows, "summary": summary,
                              "legacy_replay": replay, "source_snapshots_verified": True,
                              "empirical_three_regime_validation": False})
        write("receipt.json", {"status": summary["status"],
                               "result_sha256": sha256((args.output/"result.json").read_bytes()).hexdigest()})
        print(json.dumps(summary))
        if not summary["numerical_controls_pass"]:
            raise RuntimeError("numerical controls unresolved; no physical rejection")
    except Exception as exc:
        write("failure.json", {"status": "EXECUTION_OR_CONTROL_FAILURE_RETAINED", "error": str(exc)})
        raise


if __name__ == "__main__":
    main()
