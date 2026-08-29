"""Record and optionally replay failed development-only cluster nuisance samplers."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import differential_evolution

from sigma_theory_compiler import gravity_cluster_comparator_suite as comparators
from sigma_theory_compiler import gravity_cluster_uncertainty_program as uncertainty
from sigma_theory_compiler import gravity_item59_xcop_forward_observable_gate as item59

CONFIG_PATH = Path("configs/gravity_cluster_nuisance_sampler_diagnostic_v1.json")
OUTPUT_PATH = Path("runs/gravity/publication-readiness/nuisance-sampler-diagnostic-v1.json")
CONFIG_SCHEMA = "invariant-gravity-cluster-nuisance-sampler-diagnostic-1.0"
RECEIPT_SCHEMA = "invariant-gravity-cluster-nuisance-sampler-diagnostic-receipt-1.0"
RUN_IDS = (
    "SOBOL_PRIOR_IMPORTANCE_4096",
    "COMPONENTWISE_LONG_500",
    "AFFINE_4X36_400_400",
    "AFFINE_4X36_1200_800",
)


class GravityClusterNuisanceDiagnosticError(RuntimeError):
    """Raised when a diagnostic, source, seal, or failed-gate boundary changes."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    ) + b"\n"


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise GravityClusterNuisanceDiagnosticError(f"expected JSON object: {path}")
    return value


def _content_sha(value: Mapping[str, Any]) -> str:
    body = dict(value)
    expected = body.pop("content_sha256", None)
    actual = _sha(body)
    if expected != actual:
        raise GravityClusterNuisanceDiagnosticError("bound source content hash changed")
    return actual


def _strict(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    if set(value) != keys:
        raise GravityClusterNuisanceDiagnosticError(f"{label} keys changed")


def _under(root: Path, relative: str, label: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise GravityClusterNuisanceDiagnosticError(f"{label} escaped root") from error
    return path


def load_config(root: Path) -> dict[str, Any]:
    config = _read_json(root.resolve() / CONFIG_PATH)
    validate_config(config, root.resolve())
    return config


def validate_config(config: Mapping[str, Any], root: Path) -> None:
    _strict(
        config,
        {
            "schema_version",
            "status",
            "diagnostic_id",
            "purpose",
            "source_bindings",
            "sample_seal",
            "unchanged_completion_thresholds",
            "diagnostic_runs",
            "adjudication",
            "required_next_actions",
            "output_path",
        },
        "nuisance diagnostic config",
    )
    if (
        config["schema_version"] != CONFIG_SCHEMA
        or config["status"] != "frozen_development_diagnostic_no_target_access"
        or config["diagnostic_id"] != "gravity-cluster-nuisance-sampler-diagnostic-v1"
        or config["output_path"] != OUTPUT_PATH.as_posix()
    ):
        raise GravityClusterNuisanceDiagnosticError("nuisance diagnostic identity changed")
    bindings = config["source_bindings"]
    if tuple(row.get("source_id") for row in bindings) != (
        "uncertainty_config",
        "uncertainty_receipt",
    ):
        raise GravityClusterNuisanceDiagnosticError("diagnostic source order changed")
    for binding in bindings:
        _strict(
            binding,
            {"source_id", "path", "file_sha256", "content_sha256"},
            "diagnostic source binding",
        )
        path = _under(root, str(binding["path"]), "diagnostic source")
        if not path.is_file() or _file_sha(path) != binding["file_sha256"]:
            raise GravityClusterNuisanceDiagnosticError(
                f"diagnostic source file changed: {binding['source_id']}"
            )
        value = _read_json(path)
        if binding["content_sha256"] is not None and _content_sha(value) != binding[
            "content_sha256"
        ]:
            raise GravityClusterNuisanceDiagnosticError(
                f"diagnostic source content changed: {binding['source_id']}"
            )
    if config["sample_seal"] != {
        "likelihood_split": "development_train",
        "predictive_split": "development_holdout",
        "same_release_confirmation_rows_used": False,
        "independent_source_rows_used": False,
        "target_rows_opened": 0,
        "paid_model_calls": 0,
    }:
        raise GravityClusterNuisanceDiagnosticError("diagnostic sample seal changed")
    thresholds = config["unchanged_completion_thresholds"]
    if thresholds != {
        "maximum_rhat": 1.2,
        "minimum_effective_samples": 50,
        "maximum_standardized_between_ensemble_median_spread": 0.25,
        "all_parameters_must_pass": True,
    }:
        raise GravityClusterNuisanceDiagnosticError("completion threshold changed")
    runs = config["diagnostic_runs"]
    if tuple(row.get("run_id") for row in runs) != RUN_IDS:
        raise GravityClusterNuisanceDiagnosticError("diagnostic run inventory changed")
    for run in runs:
        _strict(run, {"run_id", "engine", "settings", "result"}, "diagnostic run")
        if run["result"]["converged"] is not False:
            raise GravityClusterNuisanceDiagnosticError("failed diagnostic promoted")
    qmc_run, component, affine_short, affine_long = runs
    if (
        qmc_run["result"]["importance_effective_samples"] != 1.0
        or qmc_run["result"]["maximum_normalized_weight"] != 1.0
        or component["result"]["maximum_rhat"] <= thresholds["maximum_rhat"]
        or component["result"]["minimum_effective_samples"]
        >= thresholds["minimum_effective_samples"]
        or affine_short["result"]["maximum_rhat"] <= thresholds["maximum_rhat"]
        or affine_long["result"]["maximum_rhat"] <= thresholds["maximum_rhat"]
        or affine_long["result"]["maximum_standardized_between_ensemble_median_spread"]
        <= thresholds["maximum_standardized_between_ensemble_median_spread"]
        or affine_long["result"]["minimum_effective_samples"]
        <= affine_short["result"]["minimum_effective_samples"]
        or affine_long["result"]["maximum_rhat"]
        >= affine_short["result"]["maximum_rhat"]
        or set(affine_long["result"]["parameters_above_rhat_threshold"])
        != set(uncertainty.PARAMETERS)
    ):
        raise GravityClusterNuisanceDiagnosticError("diagnostic result boundary changed")
    if config["adjudication"] != {
        "decision": "CORRELATION_AWARE_MIXING_IMPROVED_NOT_CONVERGED_REQUIRES_REPARAMETERIZATION_OR_INDEPENDENT_PRIORS",
        "more_componentwise_brute_force_supported": False,
        "correlation_aware_sampler_materially_improved_ess": True,
        "correlation_aware_sampler_met_completion_thresholds": False,
        "CP5_7_through_CP5_10_complete": False,
        "do_not_weaken_thresholds": True,
        "do_not_use_holdout_for_sampler_selection": True,
        "do_not_open_confirmation_or_independent_rows": True,
    }:
        raise GravityClusterNuisanceDiagnosticError("diagnostic adjudication changed")
    if len(config["required_next_actions"]) != 5:
        raise GravityClusterNuisanceDiagnosticError("diagnostic next actions changed")


def build_receipt(root: Path) -> dict[str, Any]:
    root = root.resolve()
    config = load_config(root)
    runs = config["diagnostic_runs"]
    total_evaluations = sum(
        int(
            run["result"].get(
                "evaluations",
                run["settings"].get(
                    "evaluations", run["settings"].get("proposal_evaluations", 0)
                ),
            )
        )
        for run in runs
    )
    body = {
        "schema_version": RECEIPT_SCHEMA,
        "diagnostic_id": config["diagnostic_id"],
        "decision": config["adjudication"]["decision"],
        "config_binding": {"path": CONFIG_PATH.as_posix(), "content_sha256": _sha(config)},
        "source_bindings": config["source_bindings"],
        "sample_seal": config["sample_seal"],
        "unchanged_completion_thresholds": config["unchanged_completion_thresholds"],
        "diagnostic_runs": runs,
        "completed_goal_evidence": {},
        "blocked_goal_evidence": {
            "CP5.7": "nonthermal_and_calibration_posterior_not_converged",
            "CP5.8": "stellar_latent_factor_posterior_not_converged",
            "CP5.9": "outer_boundary_posterior_not_converged",
            "CP5.10": "density_clumping_centering_projection_triaxial_and_spherical_posterior_not_converged",
        },
        "claims": {
            "componentwise_lengthening_solved_mixing": False,
            "correlation_aware_sampler_materially_improved_mixing": True,
            "posterior_sampler_converged": False,
            "development_nuisance_marginalization_complete": False,
            "CP5_7_through_CP5_10_complete": False,
            "independent_replication": False,
            "physical_mechanism_identified": False,
        },
        "counts": {
            "diagnostic_runs": len(runs),
            "candidate_forward_evaluations": total_evaluations,
            "largest_affine_posterior_draws": 4 * 36 * 800,
            "nuisance_dimensions": len(uncertainty.PARAMETERS),
            "parameters_passing_extended_affine_rhat": 0,
            "target_rows_opened": 0,
            "paid_model_calls": 0,
        },
        "required_next_actions": config["required_next_actions"],
        "reproduction": {
            "write_command": "python -m sigma_theory_compiler.gravity_cluster_nuisance_sampler_diagnostic write",
            "check_command": "python -m sigma_theory_compiler.gravity_cluster_nuisance_sampler_diagnostic check",
            "expensive_replay_command": "python -m sigma_theory_compiler.gravity_cluster_nuisance_sampler_diagnostic replay --run-id RUN_ID",
            "check_reexecutes_expensive_numeric_runs": False,
            "replay_compares_against_frozen_observation": True,
        },
        "limitations": [
            "These diagnostics use X-COP development training likelihood only and do not provide source covariance.",
            "The affine sampler improves effective sample size but does not meet the unchanged all-parameter convergence thresholds.",
            "The stored receipt is cheap to check; exact numerical replays are explicit expensive commands.",
        ],
        "next_action": config["required_next_actions"][0],
    }
    return {**body, "content_sha256": _sha(body)}


def write_receipt(root: Path) -> Path:
    path = root.resolve() / OUTPUT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_bytes(build_receipt(root)))
    return path


def validate_receipt(receipt: Mapping[str, Any], root: Path) -> None:
    body = dict(receipt)
    expected = body.pop("content_sha256", None)
    if expected != _sha(body) or dict(receipt) != build_receipt(root):
        raise GravityClusterNuisanceDiagnosticError("nuisance diagnostic receipt changed")


def _logit(unit: np.ndarray) -> np.ndarray:
    clipped = np.clip(unit, 1e-12, 1.0 - 1e-12)
    return np.log(clipped) - np.log1p(-clipped)


def _logistic(values: np.ndarray) -> np.ndarray:
    result = np.empty_like(values)
    positive = values >= 0.0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def _runtime(root: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    config = uncertainty.load_config(root)
    config59 = item59.load_config(root)
    packets = comparators._development_packets(root, config59)
    family = config["candidate_and_control_families"][0]
    return config, config59, packets, family


def _replay_qmc(root: Path, run: Mapping[str, Any]) -> dict[str, Any]:
    config, config59, packets, family = _runtime(root)
    exponent = round(math.log2(int(run["settings"]["samples"])))
    config["quasi_monte_carlo"]["power_of_two_exponent"] = exponent
    config["quasi_monte_carlo"]["samples"] = 2**exponent
    unit, _ = uncertainty._samples(config)
    log_likelihood = np.asarray(
        [
            uncertainty._evaluate_unit(value, packets, family, config, config59)[0]
            for value in unit
        ]
    )
    shifted = np.exp(log_likelihood - float(np.max(log_likelihood)))
    normalized = shifted / float(np.sum(shifted))
    return {
        "evaluations": len(unit),
        "importance_effective_samples": float(
            float(np.sum(shifted)) ** 2 / float(shifted @ shifted)
        ),
        "maximum_normalized_weight": float(np.max(normalized)),
        "log_likelihood_minimum": float(np.min(log_likelihood)),
        "log_likelihood_maximum": float(np.max(log_likelihood)),
        "converged": False,
    }


def _replay_component(root: Path, run: Mapping[str, Any]) -> dict[str, Any]:
    config, config59, packets, family = _runtime(root)
    settings = run["settings"]
    config["posterior_sampler"].update(
        {
            "burn_in_sweeps": settings["burn_sweeps"],
            "retained_samples_per_chain": settings["retained_samples_per_chain"],
            "adapt_interval_during_burn_in_sweeps": 20,
        }
    )
    initial, _ = uncertainty._samples(config)
    result = uncertainty._family_marginalization(
        packets, initial, family, config, config59
    )
    sampler = result["posterior_sampler"]
    per_parameter = {row["parameter"]: row for row in sampler["per_parameter"]}
    return {
        "maximum_rhat": sampler["maximum_rhat"],
        "minimum_effective_samples": sampler["minimum_effective_samples"],
        "acceptance_rates": sampler["acceptance_rates"],
        "holdout_score_median": result["posterior_holdout_score"]["median"],
        "holdout_score_q05": result["posterior_holdout_score"]["q05"],
        "holdout_score_q95": result["posterior_holdout_score"]["q95"],
        "worst_rhat_parameter": max(per_parameter, key=lambda key: per_parameter[key]["rhat"]),
        "worst_ess_parameter": min(
            per_parameter, key=lambda key: per_parameter[key]["effective_samples"]
        ),
        "converged": sampler["converged"],
    }


def _replay_affine(root: Path, run: Mapping[str, Any]) -> dict[str, Any]:
    config, config59, packets, family = _runtime(root)
    settings = run["settings"]
    dimensions = int(settings["dimensions"])
    ensemble_count = int(settings["ensembles"])
    walkers = int(settings["walkers_per_ensemble"])
    burn = int(settings["burn_sweeps"])
    retained = int(settings["retained_sweeps"])
    traces = np.empty((ensemble_count, walkers, retained, dimensions), dtype=float)
    acceptance = []
    map_log_likelihoods = []
    evaluations = 0

    def evaluate(unit: np.ndarray) -> float:
        nonlocal evaluations
        evaluations += 1
        return uncertainty._evaluate_unit(unit, packets, family, config, config59)[0]

    for ensemble_index in range(ensemble_count):
        optimum = differential_evolution(
            lambda unit: -evaluate(np.asarray(unit, dtype=float)),
            bounds=[(1e-8, 1.0 - 1e-8)] * dimensions,
            seed=591_000 + ensemble_index,
            popsize=4,
            maxiter=45,
            tol=1e-7,
            polish=False,
            updating="immediate",
            workers=1,
        )
        order = np.argsort(optimum.population_energies)
        initial = np.asarray(optimum.population[order[:walkers]], dtype=float)
        positions = _logit(initial)
        log_likelihood = np.asarray([evaluate(unit) for unit in initial])
        units = _logistic(positions)
        log_target = log_likelihood + np.sum(np.log(units) + np.log1p(-units), axis=1)
        rng = np.random.default_rng(592_000 + ensemble_index)
        accepted = 0
        proposals = 0
        recorded = 0
        groups = (np.arange(0, walkers, 2), np.arange(1, walkers, 2))
        for sweep in range(burn + retained):
            for active, complement in ((groups[0], groups[1]), (groups[1], groups[0])):
                for walker in active:
                    partner = int(rng.choice(complement))
                    uniform = rng.random()
                    z_scale = (
                        1.0 / math.sqrt(2.0)
                        + uniform * (math.sqrt(2.0) - 1.0 / math.sqrt(2.0))
                    ) ** 2
                    proposal = positions[partner] + z_scale * (
                        positions[walker] - positions[partner]
                    )
                    proposal_unit = _logistic(proposal)
                    proposal_log_likelihood = evaluate(proposal_unit)
                    proposal_log_target = proposal_log_likelihood + float(
                        np.sum(np.log(proposal_unit) + np.log1p(-proposal_unit))
                    )
                    log_acceptance = (dimensions - 1) * math.log(z_scale)
                    log_acceptance += proposal_log_target - log_target[walker]
                    proposals += 1
                    if math.log(max(rng.random(), np.finfo(float).tiny)) < min(
                        0.0, log_acceptance
                    ):
                        positions[walker] = proposal
                        log_likelihood[walker] = proposal_log_likelihood
                        log_target[walker] = proposal_log_target
                        accepted += 1
            if sweep >= burn:
                traces[ensemble_index, :, recorded, :] = _logistic(positions)
                recorded += 1
        acceptance.append(accepted / proposals)
        map_log_likelihoods.append(float(np.max(log_likelihood)))
    chains = traces.reshape(ensemble_count * walkers, retained, dimensions)
    rhat, ess = uncertainty._rhat_and_ess(chains)
    flattened = traces.reshape(-1, dimensions)
    standard_deviation = np.std(flattened, axis=0, ddof=1)
    median_spread = np.ptp(np.median(traces, axis=(1, 2)), axis=0)
    standardized_spread = np.divide(
        median_spread,
        standard_deviation,
        out=np.zeros_like(median_spread),
        where=standard_deviation > np.finfo(float).tiny,
    )
    threshold = 1.2
    result = {
        "maximum_rhat": float(np.max(rhat)),
        "minimum_effective_samples": float(np.min(ess)),
        "maximum_standardized_between_ensemble_median_spread": float(
            np.max(standardized_spread)
        ),
        "acceptance_rates": acceptance,
        "converged": bool(np.max(rhat) <= threshold and np.min(ess) >= 50),
    }
    if "map_log_likelihoods" in run["result"]:
        result["map_log_likelihoods"] = map_log_likelihoods
        result["parameters_above_rhat_threshold"] = [
            parameter
            for parameter, value in zip(uncertainty.PARAMETERS, rhat, strict=True)
            if value > threshold
        ]
        result["largest_between_ensemble_disagreement_parameter"] = uncertainty.PARAMETERS[
            int(np.argmax(standardized_spread))
        ]
    if evaluations != int(settings["evaluations"]):
        raise GravityClusterNuisanceDiagnosticError("affine evaluation count changed")
    return result


def _compare_observed(actual: Any, expected: Any, path: str = "result") -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise GravityClusterNuisanceDiagnosticError(f"replay keys changed at {path}")
        for key in expected:
            _compare_observed(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise GravityClusterNuisanceDiagnosticError(f"replay list changed at {path}")
        for index, value in enumerate(expected):
            _compare_observed(actual[index], value, f"{path}[{index}]")
    elif isinstance(expected, float):
        if not math.isclose(float(actual), expected, rel_tol=1e-10, abs_tol=1e-10):
            raise GravityClusterNuisanceDiagnosticError(f"replay value changed at {path}")
    elif actual != expected:
        raise GravityClusterNuisanceDiagnosticError(f"replay value changed at {path}")


def replay(root: Path, run_id: str) -> dict[str, Any]:
    config = load_config(root)
    by_id = {run["run_id"]: run for run in config["diagnostic_runs"]}
    if run_id not in by_id:
        raise GravityClusterNuisanceDiagnosticError("unknown diagnostic run id")
    run = by_id[run_id]
    if run_id == RUN_IDS[0]:
        actual = _replay_qmc(root.resolve(), run)
    elif run_id == RUN_IDS[1]:
        actual = _replay_component(root.resolve(), run)
    else:
        actual = _replay_affine(root.resolve(), run)
    _compare_observed(actual, run["result"])
    return actual


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("write", "check", "status", "replay"))
    parser.add_argument("--run-id", choices=RUN_IDS)
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.command == "write":
        output: Any = str(write_receipt(root))
    elif args.command == "replay":
        if args.run_id is None:
            parser.error("replay requires --run-id")
        output = {"status": "PASS", "run_id": args.run_id, "result": replay(root, args.run_id)}
    else:
        receipt = _read_json(root / OUTPUT_PATH)
        validate_receipt(receipt, root)
        if args.command == "check":
            output = {"status": "PASS", "content_sha256": receipt["content_sha256"]}
        else:
            output = {
                "decision": receipt["decision"],
                "counts": receipt["counts"],
                "claims": receipt["claims"],
                "next_action": receipt["next_action"],
            }
    print(json.dumps(output, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
