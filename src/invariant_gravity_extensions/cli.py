"""Opt-in synthetic extension campaign; never invokes a legacy data loader."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import platform
import sys
from typing import Any

import numpy as np
import scipy
import sympy

from .actions import action_certificates, generate_specs
from .dynamics import InertiaMemory, evolve_auxiliary
from .fields import PeriodicGrid, joint_density, solve_fields
from .observables import assumed_metric, born_lensing, member_relative_acceleration
from .policy import CompatibilityPolicy, assess_compatibility, next_stage, rank_experiments

SCHEMA = "invariant-gravity-extensions-1.0"


def read_config(path: Path) -> dict[str, Any]:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if cfg.get("schema_version") != SCHEMA:
        raise ValueError("unknown extension protocol")
    if cfg.get("access") != {"mode": "synthetic_only", "network": False,
                             "observational_inputs": False, "confirmation_access": False}:
        raise ValueError("this runner cannot authorize observational or network access")
    CompatibilityPolicy(**cfg["compatibility"])
    PeriodicGrid(cfg["synthetic"]["grid_n"], float(cfg["synthetic"]["box_length"]))
    if (type(cfg["synthetic"]["scene_count"]) is not int or
            cfg["synthetic"]["scene_count"] < 3):
        raise ValueError("at least three synthetic scenes required")
    if type(cfg["synthetic"]["seed"]) is not int or cfg["synthetic"]["seed"] < 0:
        raise ValueError("synthetic seed must be a nonnegative integer")
    for key in ("log_noise_dex", "design_noise"):
        if not np.isfinite(cfg["synthetic"][key]) or cfg["synthetic"][key] <= 0:
            raise ValueError(f"{key} must be positive")
    generate_specs(cfg["grammar"])
    return cfg


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def _scene(grid: PeriodicGrid, rng: np.random.Generator, index: int) -> dict[str, np.ndarray]:
    xyz = grid.coordinates()
    widths = np.array([1.3, 0.8+0.4*rng.random(), 0.4+0.3*rng.random()])
    centre = np.exp(-np.sum((xyz/widths[:, None, None, None])**2, axis=0)/2)
    gas = 0.04*np.exp(-np.sum((xyz/2.3)**2, axis=0)/2)
    members = {}
    for j in range(2):
        position = rng.uniform(-2, 2, 3)
        members[f"member-{j}"] = 0.12*np.exp(
            -np.sum((xyz-position[:, None, None, None])**2, axis=0)/(2*0.65**2))
    return {"central-galaxy": centre*(0.6+0.15*index), "gas": gas, **members}


def _features(solution) -> np.ndarray:
    c = solution.grid.n//2
    a = solution.acceleration
    # Exact declared mock sample locations, not reconstructed real accelerations.
    return np.array([np.linalg.norm(a[:, c+1, c, c]),
                     np.linalg.norm(a[:, c+2, c, c]),
                     np.linalg.norm(a[:, c, c, c+1]),
                     np.linalg.norm(a[:, c, c+2, c])])


def run_demo(config: dict[str, Any], *, include_lensing: bool = False) -> dict[str, Any]:
    """Exercise generation -> field solve -> compatibility -> active design.

    The synthetic observations are generated from the baseline on purpose.
    This is a regression/smoke campaign, NOT an independently calibrated
    alternate-universe power study and NOT empirical evidence.
    """
    sc = config["synthetic"]
    grid = PeriodicGrid(int(sc["grid_n"]), float(sc["box_length"]))
    specs = generate_specs(config["grammar"])
    rng = np.random.default_rng(int(sc["seed"]))
    nscene = int(sc["scene_count"])
    predictions = np.empty((len(specs), nscene, 4))
    diagnostics, source_records, light = [], [], []
    for i in range(nscene):
        components = _scene(grid, rng, i)
        rho, source_record = joint_density(grid, components, subtract_background=True)
        source_records.append(source_record)
        for k, spec in enumerate(specs):
            solved = solve_fields(grid, rho, spec, **config["solver"])
            predictions[k, i] = _features(solved)
            relative, com = member_relative_acceleration(solved, components["member-0"])
            row = {"candidate": k, "scene": i, **solved.diagnostics,
                   "member_com_acceleration": com.tolist(),
                   "member_weighted_relative_acceleration": (
                       np.sum(relative*components["member-0"][None, ...], axis=(1, 2, 3)) /
                       components["member-0"].sum()).tolist()}
            diagnostics.append(row)
            if include_lensing:
                metric = assumed_metric(solved, closure="assumed_no_slip", speed_of_light=100.0)
                lens = born_lensing(metric, distance_factor=1.0)
                light.append({"candidate": k, "scene": i, "metadata": lens["metadata"],
                              "shear_rms": float(np.sqrt(np.mean(
                                  lens["shear_1"]**2+lens["shear_2"]**2)))})
    if np.any(predictions <= 0) or not np.all(np.isfinite(predictions)):
        raise RuntimeError("mock features must be positive and finite before taking logs")
    # Only radial mock features enter compatibility. Orthogonal features remain
    # separate design predictions. No candidate parameters are fit in this demo.
    baseline = np.log10(predictions[0, :, :2]).ravel()
    observed = baseline+rng.normal(0, sc["log_noise_dex"], baseline.size)
    ids = [f"synthetic-{i}" for i in range(nscene) for _ in range(2)]
    policy = CompatibilityPolicy(**config["compatibility"])
    candidates = []
    for k, spec in enumerate(specs):
        assessment = assess_compatibility(observed, baseline,
                                          np.log10(predictions[k, :, :2]).ravel(), ids,
                                          policy, role="synthetic")
        difference = float(np.max(np.abs(np.log10(predictions[k]/predictions[0]))))
        candidates.append({"card": spec.card(), "compatibility": assessment,
                           "max_mock_log10_prediction_difference": difference,
                           "next_stage": next_stage(assessment["status"], difference > 1e-8)})
    designs = {f"scene-{i}": np.log10(predictions[:, i, :]) for i in range(nscene)}
    # Common normalization is a nuisance; use all four mock channels to see shape.
    design = rank_experiments(designs, np.eye(4)*sc["design_noise"]**2, np.ones((4, 1)))
    t = np.linspace(0, 6, 61)
    memory = InertiaMemory(0.3, 1.0, 2.0)
    trajectory = memory.integrate(lambda time, x: -x, t, np.array([1., 0.]),
                                  np.array([0., 0.8]), np.zeros(2), np.zeros(2))
    energy = memory.energy(trajectory, lambda x: float(np.dot(x, x)/2))
    sources = np.zeros((len(t), *grid.shape))
    sources[30:, ...] = 0.01
    q, _ = evolve_auxiliary(grid, t, sources, np.zeros(grid.shape), np.zeros(grid.shape))
    return {
        "schema_version": SCHEMA,
        "claim_ceiling": "SYNTHETIC_METHOD_EXERCISE_NOT_A_DISCOVERY",
        "synthetic_truth": "same-adapter qumond baseline; smoke test only, not independent recovery",
        "observational_inputs_opened": 0,
        "network_calls": 0,
        "confirmation_accesses": 0,
        "frozen_predecessors_modified": False,
        "units": "dimensionless synthetic units: a0=1, 4*pi*G=1",
        "config": config,
        "action_certificates": action_certificates(),
        "source_scenes": source_records,
        "candidates": candidates,
        "solver_diagnostics": diagnostics,
        "prediction_array": predictions.tolist(),
        "active_design": design,
        "lensing": light if include_lensing else {"status": "UNSUPPORTED_WITHOUT_EXPLICIT_CLOSURE"},
        "dynamics_controls": {
            "kinetic_matrix": [[1.0, -memory.coupling], [-memory.coupling, memory.mu]],
            "relative_energy_drift": float(np.max(np.abs(energy-energy[0]))/abs(energy[0])),
            "auxiliary_pre_drive_max": float(np.max(np.abs(q[:31]))),
            "auxiliary_final_mean": float(q[-1].mean()),
            "scope": "conservative_worldline_and_linear_wave_controls_not_MOND_or_covariant_proof",
        },
        "remaining_adapters": ["isolated_nested_boundary_solver", "actual_RAR_data_adapter",
                               "covariant_photon_closure", "vector_tensor_evolution",
                               "independent_synthetic_generator", "real_observation_forward_likelihood"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["catalog", "demo"])
    parser.add_argument("--config", type=Path, default=Path("configs/gravity_extension_discovery_v1.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--assume-no-slip", action="store_true",
                        help="explicitly enable synthetic Born lensing with an ASSUMED metric")
    args = parser.parse_args(argv)
    created = False
    try:
        config_bytes = args.config.read_bytes()
        config = read_config(args.config)
        if args.config.read_bytes() != config_bytes:
            raise RuntimeError("config changed while loading")
        source_hashes = {p.name: sha256(p.read_bytes()).hexdigest()
                         for p in sorted(Path(__file__).parent.glob("*.py"))}
        config_hash = sha256(config_bytes).hexdigest()
        if args.output.exists():
            raise ValueError("output already exists; choose a new directory (append-only runs)")
        args.output.mkdir(parents=True, exist_ok=False)
        created = True
        _write_json(args.output / "started.json", {
            "schema_version": SCHEMA, "command": args.command,
            "config_sha256": config_hash, "source_hashes": source_hashes})
        if args.command == "catalog":
            result = {"schema_version": SCHEMA, "claim_ceiling": "SYMBOLIC_STATIC_CONTROLS_ONLY",
                      "action_certificates": action_certificates(),
                      "candidates": [s.card() for s in generate_specs(config["grammar"])],
                      "observational_inputs_opened": 0, "confirmation_accesses": 0}
        else:
            result = run_demo(config, include_lensing=args.assume_no_slip)
        result["runtime"] = {"python": platform.python_version(), "numpy": np.__version__,
                             "scipy": scipy.__version__, "sympy": sympy.__version__}
        current_sources = {p.name: sha256(p.read_bytes()).hexdigest()
                           for p in sorted(Path(__file__).parent.glob("*.py"))}
        if current_sources != source_hashes or sha256(args.config.read_bytes()).hexdigest() != config_hash:
            raise RuntimeError("source or config changed during execution; result quarantined")
        result["source_hashes"] = source_hashes
        result["config_sha256"] = config_hash
        _write_json(args.output / "result.json", result)
        digest = sha256((args.output / "result.json").read_bytes()).hexdigest()
        _write_json(args.output / "receipt.json", {"status": "COMPLETED_AT_DECLARED_SCOPE",
                    "result_sha256": digest, "claim_ceiling": result["claim_ceiling"],
                    "confirmation_access_authorized": False})
        print(json.dumps({"output": str(args.output), "result_sha256": digest,
                          "claim_ceiling": result["claim_ceiling"]}, sort_keys=True))
        return 0
    except (ValueError, RuntimeError, OSError, KeyError, TypeError, FloatingPointError) as exc:
        # Preserve the failed directory rather than turn a numerical failure into
        # a physical rejection or overwrite a previous successful receipt.
        if created and (args.output / "started.json").exists():
            failure = args.output / "failure.json"
            if not failure.exists() and not (args.output / "receipt.json").exists():
                _write_json(failure, {"status": "NUMERICAL_OR_INPUT_BLOCK_NOT_THEORY_REJECTION",
                                     "error": str(exc)})
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 2
