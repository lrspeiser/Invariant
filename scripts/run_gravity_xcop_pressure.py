"""Fixed scalar gravity pressure diagnostics on already-exposed X-COP clusters."""
from __future__ import annotations

import argparse
import itertools
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import astropy
import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from invariant_gravity_extensions.cluster_pressure import (
    DEVELOPMENT_CLUSTERS,
    boundary_residual_covariance,
    covariance_loss,
    load_development_packet,
    predict_pressure,
    pressure_indices,
)
from invariant_gravity_extensions.saturated_actions import SaturatedActionSpec


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(item) for item in value]
    return value


def definitions(config):
    models = [{"id": "newtonian_baryons", "family": "newtonian"},
              {"id": "empirical_RAR_a0_1.2e-10", "family": "rar_comparator", "a0": 1.2e-10}]
    for shape, a0 in itertools.product(config["shapes"], config["a0_m_s2"]):
        spec = SaturatedActionSpec("qumond", shape=shape, epsilon=config["epsilon"])
        models.append({"id": f"saturated_m{shape:g}_a0_{a0:.1e}", "family": "saturated_qumond",
                       "shape": shape, "a0": a0, "epsilon": config["epsilon"],
                       "card_sha256": spec.card()["content_sha256"]})
    nominal = config["nominal_nuisance"]
    scenarios = [{"id": "nominal", "values": nominal.copy()}]
    for name, choices in config["single_sensitivities"].items():
        for value in choices:
            scenarios.append({"id": f"single_{name}_{value:g}", "values": {**nominal, name: value}})
    corners = config["joint_corner_sensitivities"]
    for i, values in enumerate(itertools.product(*corners.values())):
        scenarios.append({"id": f"joint_corner_{i+1}", "values": {**nominal, **dict(zip(corners, values))}})
    return models, scenarios


def evaluate(packet, model, nuisance, nodes):
    answer = predict_pressure(packet, model, nuisance, nodes=nodes)
    residual = answer["prediction"]-answer["observed"]
    log_ratio = np.log10(answer["prediction"]/answer["observed"])
    choices = {"transferred_correlation": packet["covariance"],
               "native_scaled": packet["native_scaled_covariance"],
               "diagonal_quoted": np.diag(packet["pressure_error"]**2)}
    covariance_scores, standardized = {}, {}
    for name, covariance in choices.items():
        transformed = boundary_residual_covariance(
            covariance*answer["pressure_scale"]**2, answer["indices"], answer["anchor"], answer["boundary_coefficients"])
        covariance_scores[name] = covariance_loss(residual, transformed)
        standardized[name] = residual/np.sqrt(np.diag(transformed))
    return {"cluster": packet["cluster"], "stellar_profile_present": packet["stellar"] is not None,
            **answer, "residual": residual, "log10_ratio": log_ratio,
            "mse_log10_ratio": float(np.mean(log_ratio**2)), "mean_absolute_dex": float(np.mean(abs(log_ratio))),
            "median_pressure_ratio": float(np.median(answer["prediction"]/answer["observed"])),
            "whitened_mean_squared_residual": covariance_scores,
            "marginal_standardized_residual_conditional_not_significance": standardized}


def aggregate(rows):
    return {"clusters": len(rows), "targets": sum(len(row["indices"]) for row in rows),
            "equal_cluster_mse_log10_ratio": float(np.mean([row["mse_log10_ratio"] for row in rows])),
            "equal_cluster_mean_absolute_dex": float(np.mean([row["mean_absolute_dex"] for row in rows])),
            "median_cluster_median_pressure_ratio": float(np.median([row["median_pressure_ratio"] for row in rows])),
            "equal_cluster_whitened_mean_squared_residual": {
                key: float(np.mean([row["whitened_mean_squared_residual"][key] for row in rows]))
                for key in rows[0]["whitened_mean_squared_residual"]}}


def comparison(rows, baseline):
    """Object-level descriptive robustness; no terminal family rejection."""
    lookup = {row["cluster"]: row for row in baseline}
    contributions = [{"cluster": row["cluster"], "stellar_profile_present": row["stellar_profile_present"],
                      "candidate_minus_baseline_mse_dex2": row["mse_log10_ratio"]-lookup[row["cluster"]]["mse_log10_ratio"]}
                     for row in rows]
    values = np.array([row["candidate_minus_baseline_mse_dex2"] for row in contributions])
    order = np.argsort(values)
    influential = int(np.argmax(abs(values)))
    return {"object_contributions": contributions, "mean_difference": float(values.mean()),
            "raw_comparative_loss_count": int(np.count_nonzero(values > 0)),
            "raw_comparative_win_count": int(np.count_nonzero(values < 0)),
            "raw_tie_count": int(np.count_nonzero(values == 0)),
            "quality_verified_counterexample_count": 0, "uncertainty_resolved_counterexample_count": 0,
            "counts_scope": "raw counts compare descriptive loss, not physical falsifications; zero qualified counts means incomplete audit, not zero true counterexamples",
            "leave_most_influential_out": {"omitted": contributions[influential]["cluster"],
                                           "mean_difference": float(np.delete(values, influential).mean())},
            "symmetric_trim": {"omitted": [contributions[int(i)]["cluster"] for i in [order[0], order[-1]]],
                               "mean_difference": float(values[order[1:-1]].mean()), "fraction_removed": 2/len(values)},
            "strata": [{"stellar_profile_present": present,
                        "clusters": sum(r["stellar_profile_present"] == present for r in contributions),
                        "mean_difference": float(np.mean([r["candidate_minus_baseline_mse_dex2"] for r in contributions
                                                          if r["stellar_profile_present"] == present]))}
                       for present in [True, False]],
            "independent_replication_count": 0, "family_pruned": False}


def campaign(config, write):
    if set(config["clusters"]) != DEVELOPMENT_CLUSTERS or len(config["clusters"]) != len(DEVELOPMENT_CLUSTERS):
        raise ValueError("development population must equal the registered eight clusters")
    source_contract = json.loads((ROOT/config["source_contract"]).read_bytes())
    covariance_manifest = json.loads((ROOT/config["covariance_manifest"]).read_bytes())
    packets, failures = [], []
    for cluster in config["clusters"]:
        try:
            packet = load_development_packet(ROOT, cluster, source_contract, covariance_manifest)
            pressure_indices(packet)
            packets.append(packet)
        except (OSError, ValueError, KeyError, IndexError, StopIteration) as exc:
            failures.append({"cluster": cluster, "status": "SOURCE_ADAPTER_UNRESOLVED_RETAINED", "error": str(exc)})
    write("source_preflight.json", {"packets": packets, "failures": failures,
                                    "reserved_clusters_accessed": 0, "temperature_profiles_scored": 0,
                                    "inferred_total_mass_columns_parsed": False})
    if failures:
        raise RuntimeError("source preflight incomplete; all failures retained, no reduced-population scoring")
    models, scenarios = definitions(config)
    entries, refinement = [], []
    for model in models:
        print(f"Pressure development: {model['id']}, {len(scenarios)} global scenarios", flush=True)
        for scenario in scenarios:
            rows = [evaluate(packet, model, scenario["values"], config["numerical_control"]["nodes"]) for packet in packets]
            entries.append({"model": model["id"], "scenario": scenario["id"], "summary": aggregate(rows), "rows": rows})
            if scenario["id"] == "nominal":
                for packet, row in zip(packets, rows):
                    refined = predict_pressure(packet, model, scenario["values"], nodes=config["numerical_control"]["refined_nodes"])
                    refinement.append({"model": model["id"], "cluster": packet["cluster"],
                                       "maximum_relative_pressure_change": float(np.max(abs(refined["prediction"]/row["prediction"]-1)))})
        write(f"model_{model['id']}.json", {"model": model, "entries": [e for e in entries if e["model"] == model["id"]]})
    nominal = {entry["model"]: entry for entry in entries if entry["scenario"] == "nominal"}
    stronger_baseline = min(config["comparators"], key=lambda key: nominal[key]["summary"]["equal_cluster_mse_log10_ratio"])
    comparisons = [{"model": model["id"], "baseline": baseline,
                    **comparison(nominal[model["id"]]["rows"], nominal[baseline]["rows"])}
                   for model in models if model["family"] == "saturated_qumond" for baseline in config["comparators"]]
    sensitivity = []
    for model in models:
        subset = [entry for entry in entries if entry["model"] == model["id"]]
        losses = [entry["summary"]["equal_cluster_mse_log10_ratio"] for entry in subset]
        changes = []
        for baseline in config["comparators"]:
            if baseline == model["id"]:
                continue
            baseline_lookup = {entry["scenario"]: entry for entry in entries if entry["model"] == baseline}
            differences = [entry["summary"]["equal_cluster_mse_log10_ratio"]-
                           baseline_lookup[entry["scenario"]]["summary"]["equal_cluster_mse_log10_ratio"] for entry in subset]
            changes.append({"baseline": baseline, "minimum_matched_scenario_difference": min(differences),
                            "maximum_matched_scenario_difference": max(differences),
                            "scenarios_with_lower_loss": sum(value < 0 for value in differences)})
        sensitivity.append({"model": model["id"], "minimum_mse_dex2": min(losses), "maximum_mse_dex2": max(losses),
                            "matched_scenario_comparisons": changes,
                            "scope": "fixed global sensitivity range, not a confidence interval or fitted nuisance posterior"})
    previous = json.loads((ROOT/config["predecessor"]).read_bytes())
    solar = [{"model": model["id"], "scenarios": [
        {key: row[key] for key in ("shape", "a0_m_s2", "physical_external_m_s2", "Q2_s_minus2", "status")}
        for row in previous["rows"] if row["shape"] == model["shape"] and row["a0_m_s2"] == model["a0"]],
        "full_solar_system_pass": False}
        for model in models if model["family"] == "saturated_qumond"]
    maximum = max(row["maximum_relative_pressure_change"] for row in refinement)
    return {"models": models, "scenarios": scenarios, "entries": entries, "nominal_comparisons": comparisons,
            "global_sensitivity": sensitivity, "prior_cassini_summary_screen": solar,
            "numerical_refinement": refinement,
            "summary": {"models": len(models), "global_scenarios": len(scenarios), "clusters": len(packets),
                        "profile_predictions": len(entries)*len(packets), "stronger_predeclared_nominal_baseline": stronger_baseline,
                        "maximum_numerical_relative_pressure_change": maximum,
                        "numerical_target_met": maximum <= config["numerical_control"]["maximum_relative_pressure_change"],
                        "nominal_models": {key: value["summary"] for key, value in nominal.items()}},
            "status": "QUALITY_LIMITED_EVIDENCE_RETAINED",
            "missing_quality_records": ["validated native-to-high-level pressure covariance mapping",
                                        "joint density and stellar radial covariance",
                                        "distance, calibration and deprojection joint covariance",
                                        "triaxiality, clumping, time dependence and nonthermal profile response",
                                        "selection function and independent instrument/systematics replication",
                                        "stellar profile outer continuation and inner source mass uncertainty"],
            "temperature_profiles_scored": 0, "galaxy_profiles_scored": 0, "lensing_profiles_scored": 0,
            "reserved_clusters_accessed": 0, "empirical_three_regime_validation": False,
            "discovery_claim": False, "family_pruning_authorized": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config_path = ROOT/"configs/gravity_xcop_pressure_development_v1.json"
    config = json.loads(config_path.read_bytes())
    paths = [Path(__file__), config_path, ROOT/config["source_contract"], ROOT/config["covariance_manifest"],
             ROOT/config["predecessor"], *sorted((ROOT/"src/invariant_gravity_extensions").glob("*.py"))]

    def hashes():
        return {path.relative_to(ROOT).as_posix(): sha256(path.read_bytes()).hexdigest() for path in paths}

    def write(name, value):
        with (args.output/name).open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(jsonable(value), handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")

    before = hashes()
    provenance = {"input_hashes": before, "started_utc": datetime.now(UTC).isoformat(),
                  "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  "source_hashes_authoritative": True, "python": platform.python_version(),
                  "numpy": np.__version__, "scipy": scipy.__version__, "astropy": astropy.__version__}
    write("started.json", {"config": config, **provenance})
    try:
        result = campaign(config, write)
        if hashes() != before:
            raise RuntimeError("inputs changed during run")
        write("result.json", {"config": config, **provenance, **result})
        if not result["summary"]["numerical_target_met"]:
            raise RuntimeError("numerical target unresolved; result retained, not a physical rejection")
        write("receipt.json", {"status": "COMPLETED_AT_DECLARED_DEVELOPMENT_SCOPE",
                               "scientific_status": result["status"],
                               "result_sha256": sha256((args.output/"result.json").read_bytes()).hexdigest()})
        print(json.dumps(result["summary"]))
    except Exception as exc:
        write("failure.json", {"status": "EXECUTION_FAILURE_NOT_PHYSICAL_REJECTION", "error": str(exc)})
        raise


if __name__ == "__main__":
    main()
