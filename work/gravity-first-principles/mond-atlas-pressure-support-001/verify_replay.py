"""Read-only replay of the frozen pressure study; writes one new verification file."""
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from threadpoolctl import threadpool_limits

PUBLIC = Path(__file__).resolve().parent
ROOT = PUBLIC.parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from mond_atlas_pressure_support import case_balance, fit_amplitude, independent_truth


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="run-002")
    parser.add_argument("--output-name", default="verification.json")
    args = parser.parse_args()
    if not args.run_id.startswith("run-") or not args.run_id[4:].isdigit():
        raise ValueError("Invalid run ID")
    if Path(args.output_name).name != args.output_name or not args.output_name.endswith(".json"):
        raise ValueError("Output must be a JSON filename within this package")
    output = PUBLIC / args.output_name
    if output.exists():
        raise FileExistsError("Verification exists; choose a new output name")
    freeze = json.loads((PUBLIC / "freeze.json").read_text())
    config_path = ROOT / freeze["config_path"]
    assert digest(config_path) == freeze["config_sha256"]
    assert digest(PUBLIC / "PREFLIGHT.md") == freeze["preflight_sha256"]
    for name, expected in freeze["prior_immutable_files"].items():
        assert digest(ROOT / name) == expected, name
    count = 0
    for manifest_path in sorted(PUBLIC.glob("run-*/manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        for mapping in (manifest["public"], manifest["private"]):
            for name, expected in mapping.items():
                assert digest(ROOT / name) == expected, name
                count += 1
    run = PUBLIC / args.run_id
    private = ROOT / "work/private" / PUBLIC.name / args.run_id
    receipt = json.loads((run / "receipt.json").read_text())
    for name, expected in receipt["implementation_sha256"].items():
        assert digest(ROOT / name) == expected, name
    assert freeze["frozen_utc"] < receipt["started_utc"]
    assert receipt["controls_admitted_utc"] < receipt["first_study_response_generated_utc"]
    assert receipt["controls_passed"] == receipt["controls_total"] == 42
    assert receipt["failed_fits"] == 0
    config = json.loads(config_path.read_text())
    study = config["study"]
    rows = json.loads((run / "fits.json").read_text())["fits"]
    summary = json.loads((run / "summary.json").read_text())
    cases = {case["id"]: case for case in study["cases"]}
    with np.load(private / "design.npz") as design:
        r = design["radius"].copy()
        train, test = design["train"].copy(), design["heldout"].copy()
    np.testing.assert_array_equal(r, np.linspace(study["radius_min_kpc"], study["radius_max_kpc"], study["radius_count"]))
    np.testing.assert_array_equal(train, study["train_indices"])
    np.testing.assert_array_equal(test, study["heldout_indices"])
    noise_arrays = prediction_arrays = truth_packets = diagnostic_values = 0
    with threadpool_limits(limits=1):
        for index, case in enumerate(study["cases"]):
            cid = case["id"]
            expected_v2, expected_vc2, expected_support = independent_truth(r, case)
            with np.load(private / (cid+"-truth.npz")) as packet:
                np.testing.assert_array_equal(packet["rotation_squared"], expected_v2)
                np.testing.assert_array_equal(packet["circular_squared"], expected_vc2)
                np.testing.assert_array_equal(packet["support"], expected_support)
                truth = packet["los_truth"].copy()
                np.testing.assert_array_equal(truth, study["systemic_km_s"]+np.sin(np.deg2rad(study["inclination_deg"]))*np.sqrt(expected_v2))
            truth_packets += 1
            for model in study["fit_models"]:
                fit, pred = fit_amplitude(r, truth, train, case, model, study)
                with np.load(private / f"{cid}-{model}-noiseless-fit.npz") as packet:
                    np.testing.assert_array_equal(packet["prediction"], pred)
                saved_fit = next(c for c in summary["cases"] if c["case"] == cid)["noiseless_fits"][model]
                assert fit["amplitude"] == saved_fit["amplitude"]
                prediction_arrays += 1
            for base_seed in study["seeds"]:
                seed = base_seed+index*study["case_seed_stride"]
                fresh_seed = seed+study["fresh_seed_offset"]
                with np.load(private / f"{cid}-seed-{seed}.npz") as packet:
                    np.testing.assert_array_equal(packet["noise"], np.random.default_rng(seed).normal(0, study["noise_sigma_km_s"], len(r)))
                    np.testing.assert_array_equal(packet["fresh_noise"], np.random.default_rng(fresh_seed).normal(0, study["noise_sigma_km_s"], len(r)))
                    np.testing.assert_array_equal(packet["observed"], truth+packet["noise"])
                    np.testing.assert_array_equal(packet["fresh"], truth+packet["fresh_noise"])
                    noise_arrays += 2
                    for model in study["fit_models"]:
                        row = next(x for x in rows if x["case"] == cid and x["seed"] == seed and x["model"] == model)
                        fit, pred = fit_amplitude(r, packet["observed"], train, case, model, study)
                        assert fit["success"] and fit["amplitude"] == row["amplitude"]
                        np.testing.assert_array_equal(packet[model+"_prediction"], pred)
                        assert float(packet[model+"_amplitude"]) == fit["amplitude"]
                        sigma = study["noise_sigma_km_s"]
                        metrics = {
                            "relative_force_bias": fit["amplitude"]/case["amplitude"]-1,
                            "train_q_per_sample": np.mean(((pred[train]-packet["observed"][train])/sigma)**2),
                            "heldout_q_per_sample": np.mean(((pred[test]-packet["observed"][test])/sigma)**2),
                            "fresh_heldout_q_per_sample": np.mean(((pred[test]-packet["fresh"][test])/sigma)**2),
                            "noiseless_signal_rmse_km_s": np.sqrt(np.mean((pred-truth)**2)),
                        }
                        for name, value in metrics.items():
                            assert float(value) == row[name], (cid, seed, model, name)
                            diagnostic_values += 1
                        prediction_arrays += 1
        rejected = case_balance(r, study["impossible_case"])
        with np.load(private / "impossible-signed-equilibrium.npz") as packet:
            np.testing.assert_array_equal(packet["rotation_squared"], rejected.rotation_squared)
            np.testing.assert_array_equal(packet["feasible"], rejected.feasible)
            assert not np.any(packet["feasible"])
    result = {"status": "PASS", "admission": "THEORY_BENCHMARK_ONLY", "verified_utc": datetime.now(timezone.utc).isoformat(),
              "run_id": args.run_id, "prior_immutable_files_checked": len(freeze["prior_immutable_files"]),
              "run_manifest_entries_checked": count, "noise_arrays_exact": noise_arrays,
              "prediction_arrays_exact": prediction_arrays, "truth_packets_exact": truth_packets,
              "fit_diagnostic_values_exact": diagnostic_values, "invalid_signed_case_exact": True,
              "preflight_and_gate_timestamps_verified": True,
              "config_sha256": digest(config_path), "verifier_sha256": digest(Path(__file__)),
              "observational_source_or_velocity_files_opened": 0,
              "note": "Exact replay validates saved-byte reproducibility; independent analytic/finite-difference/quadrature controls provide the correctness checks."}
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
