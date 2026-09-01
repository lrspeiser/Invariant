"""Source-free independent probes for Lane 7 V5.

This script deliberately does not call verify_v4_preservation_and_sources,
build_artifacts, build_receipt, validate_receipt, or any FITS/SLACS reader.
It opens only code/config/test and the already-produced V5 receipt artifacts.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.fft import irfftn, rfftn

from sigma_theory_compiler import (
    open_gravity_same_law_eso325_extended_source_v5 as subject,
)


ROOT = Path(__file__).resolve().parents[1]
V5 = ROOT / "runs/gravity/open-gravity-same-law-eso325-extended-source-v5"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def relative_rms(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.linalg.norm(left - right) / max(np.linalg.norm(left), 1e-300))


def raw_channels(state, points: np.ndarray, image_axis: np.ndarray, config: dict):
    return {
        "matter": subject.matter_acceleration(state, points).ravel(),
        "lensing": subject.reduced_photon_deflection(state, points[:, :2], config).ravel(),
        "extended_image": subject.extended_source_image(state, image_axis, config).ravel(),
    }


def accepted_by_validate(config: dict) -> tuple[bool, str | None]:
    try:
        subject.validate_config(config)
    except Exception as exc:  # audit records exact fail-closed behavior
        return False, f"{type(exc).__name__}: {exc}"
    return True, None


def main() -> None:
    config = subject.load_config()
    receipt_path = V5 / "receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt_without_self_hash = dict(receipt)
    recorded_self_hash = receipt_without_self_hash.pop("content_sha256")

    package_files = {
        "config": ROOT / receipt["artifact_bindings"]["config"]["path"],
        "module": ROOT / receipt["artifact_bindings"]["module"]["path"],
        "test": ROOT / receipt["artifact_bindings"]["test"]["path"],
        "receipt": receipt_path,
    }
    package_hashes = {
        name: {
            "path": path.relative_to(ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "matches_v5_binding": (
                sha256(path) == receipt["artifact_bindings"][name]["sha256"]
                if name in receipt["artifact_bindings"]
                else None
            ),
        }
        for name, path in package_files.items()
    }
    artifact_hashes = {}
    for name, binding in sorted(receipt["artifact_manifest"].items()):
        path = V5 / "artifacts" / name
        artifact_hashes[name] = {
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "matches_v5_binding": (
                path.stat().st_size == binding["bytes"] and sha256(path) == binding["sha256"]
            ),
        }

    # Run the exact source-free synthetic gate once, independently.
    exact_gate = subject.target_free_gate(config)

    gate = config["target_free_gate"]
    extent = float(gate["physical_extent_kpc"])
    mass = float(gate["total_mass_Msun"])
    injected_g = float(gate["injected_g"])
    injected_range = float(gate["injected_range_kpc"])
    coarse_coordinates, coarse_density, coarse_cell = subject.asymmetric_density(
        int(gate["coarse_cells_per_axis"]), extent, mass
    )
    fine_coordinates, fine_density, fine_cell = subject.asymmetric_density(
        int(gate["fine_cells_per_axis"]), extent, mass
    )
    coarse = subject.solve_physical_state(
        coarse_density,
        coarse_coordinates,
        coarse_cell,
        g=injected_g,
        range_kpc=injected_range,
        padding_factor=2,
        config=config,
    )
    coarse_pad4 = subject.solve_physical_state(
        coarse_density.copy(),
        coarse_coordinates.copy(),
        coarse_cell,
        g=injected_g,
        range_kpc=injected_range,
        padding_factor=4,
        config=config,
    )
    fine_truth = subject.solve_physical_state(
        fine_density,
        fine_coordinates,
        fine_cell,
        g=injected_g,
        range_kpc=injected_range,
        padding_factor=2,
        config=config,
    )
    points = np.array(
        [
            [0.8, 0.0, 0.0],
            [1.2, 0.3, 0.0],
            [1.8, -0.4, 0.2],
            [2.4, 0.5, -0.3],
            [-1.1, 0.7, 0.4],
            [0.4, -1.9, -0.5],
        ]
    )
    image_axis = np.linspace(-4.0, 4.0, 33)
    coarse_raw = raw_channels(coarse, points, image_axis, config)
    pad4_raw = raw_channels(coarse_pad4, points, image_axis, config)
    fine_raw = raw_channels(fine_truth, points, image_axis, config)
    raw_convergence = {
        "padding_pad2_reference_vs_pad4": {
            channel: relative_rms(coarse_raw[channel], pad4_raw[channel])
            for channel in coarse_raw
        },
        "resolution_fine_reference_vs_coarse": {
            channel: relative_rms(fine_raw[channel], coarse_raw[channel])
            for channel in fine_raw
        },
        "reported_separately_normalized_vector": {
            "padding": exact_gate["metrics"]["doubled_padding_observable_relative_rms"],
            "resolution": exact_gate["metrics"]["halved_cell_observable_relative_rms"],
        },
        "scale_blindness_probe": {
            "multiplicative_amplitude_change": 1.25,
            "raw_relative_rms": relative_rms(
                fine_raw["matter"], 1.25 * fine_raw["matter"]
            ),
            "separately_rms_normalized_relative_rms": relative_rms(
                fine_raw["matter"]
                / max(float(np.sqrt(np.mean(fine_raw["matter"] ** 2))), 1e-300),
                (1.25 * fine_raw["matter"])
                / max(
                    float(np.sqrt(np.mean((1.25 * fine_raw["matter"]) ** 2))),
                    1e-300,
                ),
            ),
            "interpretation": "A 25% amplitude error is mathematically removed by the per-run normalization used by the gate.",
        },
        "implementation_lines": {
            "per_state_channel_normalization": [551, 552, 553, 555],
            "convergence_comparison": [701, 702, 703, 704, 705],
        },
    }

    # Reproduce the holdout contamination exactly.  The fitted slice is 0:30,
    # whereas every fifth datum beginning at 4 is later called held out.
    observed, sigma, channel_slices = subject._synthetic_measurements(
        fine_truth, points, image_axis, config
    )
    linear_stop = channel_slices["lensing"].stop
    holdout = np.arange(4, len(observed), 5, dtype=int)
    fit_indices = np.arange(linear_stop, dtype=int)
    leaked = np.intersect1d(fit_indices, holdout)
    training_linear = np.setdiff1d(fit_indices, leaked)

    fine_gr = subject.solve_physical_state(
        fine_density.copy(),
        fine_coordinates.copy(),
        fine_cell,
        g=0.0,
        range_kpc=injected_range,
        padding_factor=2,
        config=config,
    )
    unit_state = subject.solve_physical_state(
        fine_density.copy(),
        fine_coordinates.copy(),
        fine_cell,
        g=1.0,
        range_kpc=injected_range,
        padding_factor=2,
        config=config,
    )
    gr_raw = raw_channels(fine_gr, points, image_axis, config)
    unit_raw = raw_channels(unit_state, points, image_axis, config)
    gr_prediction = np.concatenate(tuple(gr_raw.values()))
    unit_prediction = np.concatenate(tuple(unit_raw.values()))
    basis = unit_prediction[:linear_stop] - gr_prediction[:linear_stop]
    weights = 1.0 / sigma[:linear_stop] ** 2

    def fit_g(indices: np.ndarray) -> float:
        return float(
            np.sum(
                weights[indices]
                * basis[indices]
                * (observed[indices] - gr_prediction[indices])
            )
            / np.sum(weights[indices] * basis[indices] ** 2)
        )

    contaminated_g = fit_g(fit_indices)
    clean_g = fit_g(training_linear)

    def prediction_for(g_value: float) -> np.ndarray:
        state = subject.solve_physical_state(
            fine_density.copy(),
            fine_coordinates.copy(),
            fine_cell,
            g=g_value,
            range_kpc=injected_range,
            padding_factor=2,
            config=config,
        )
        return np.concatenate(tuple(raw_channels(state, points, image_axis, config).values()))

    contaminated_prediction = prediction_for(contaminated_g)
    clean_prediction = prediction_for(clean_g)
    holdout_leakage = {
        "total_observations": int(len(observed)),
        "linear_fit_slice": [0, int(linear_stop)],
        "holdout_rule": "arange(4, N, 5)",
        "holdout_count": int(len(holdout)),
        "fit_count": int(len(fit_indices)),
        "leaked_holdout_count": int(len(leaked)),
        "leaked_indices": leaked.tolist(),
        "reported_recovered_g": float(exact_gate["metrics"]["recovered_g"]),
        "recomputed_contaminated_g": contaminated_g,
        "clean_training_only_g": clean_g,
        "reported_contaminated_holdout_lpd": subject.gaussian_log_predictive_density(
            observed, contaminated_prediction, sigma, holdout
        ),
        "clean_training_only_holdout_lpd": subject.gaussian_log_predictive_density(
            observed, clean_prediction, sigma, holdout
        ),
        "implementation_lines": {
            "fit_before_split": [756, 757, 758, 759, 760, 761, 762, 763],
            "holdout_defined_after_fit": [784, 785, 786, 787],
        },
    }

    # Direct spectral-integral probe of the implemented Helmholtz zero mode.
    probe_n = 9
    probe_coordinates, probe_density, probe_cell = subject.asymmetric_density(
        probe_n, 4.0, 1.0e9
    )
    padded_n = 2 * (probe_n - 1) + 1
    start = (padded_n - probe_n) // 2
    padded = np.zeros((padded_n,) * 3)
    padded[start : start + probe_n, start : start + probe_n, start : start + probe_n] = (
        probe_density
    )
    transformed = rfftn(padded)
    wave_xy = 2.0 * np.pi * np.fft.fftfreq(padded_n, d=probe_cell)
    wave_z = 2.0 * np.pi * np.fft.rfftfreq(padded_n, d=probe_cell)
    k_squared = (
        wave_xy[:, None, None] ** 2
        + wave_xy[None, :, None] ** 2
        + wave_z[None, None, :] ** 2
    )
    gravitational_constant = float(config["law_binding"]["constants"]["G_kpc_km2_s2_Msun"])
    mu_squared = 1.0 / injected_range**2
    mathematically_required_kernel_zero = -4.0 * math.pi * gravitational_constant / mu_squared
    implemented_kernel = -4.0 * math.pi * gravitational_constant / (k_squared + mu_squared)
    implemented_kernel[0, 0, 0] = 0.0
    y_full = irfftn(transformed * implemented_kernel, s=padded.shape).real
    lhs_mean = -mu_squared * float(np.mean(y_full))
    rhs_mean = 4.0 * math.pi * gravitational_constant * float(np.mean(padded))
    helmholtz_probe = {
        "implemented_k0_kernel": float(implemented_kernel[0, 0, 0]),
        "required_k0_kernel_for_declared_equation": mathematically_required_kernel_zero,
        "mean_Y_over_periodic_padded_box": float(np.mean(y_full)),
        "integrated_equation_lhs_mean_minus_mu2Y": lhs_mean,
        "integrated_equation_rhs_mean_4piG_rho": rhs_mean,
        "relative_integrated_residual": abs(lhs_mean - rhs_mean) / abs(rhs_mean),
        "implementation_lines": [447, 448, 449, 450, 451],
    }

    module_text = subject.MODULE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(module_text)
    function_names = sorted(
        node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    )
    name_identifiers = sorted(
        {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    )
    execution_completeness = {
        "pseudo_nfw_function_names": [name for name in function_names if "nfw" in name.lower()],
        "pseudo_nfw_runtime_identifiers": [name for name in name_identifiers if "nfw" in name.lower()],
        "logsumexp_runtime_identifier_present": "logsumexp" in name_identifiers,
        "posterior_draw_runtime_identifiers": [
            name for name in name_identifiers if any(token in name.lower() for token in ("walker", "rhat", "posterior", "draw"))
        ],
        "implemented_lpd_function": "gaussian_log_predictive_density",
        "implemented_lpd_signature": ["observed", "predicted", "sigma", "indices"],
        "implemented_lpd_is_diagonal_single_prediction": True,
        "contract_requires_4096_draw_logsumexp_and_full_covariance": True,
        "tests_only_assert_pseudo_nfw_and_lpd_contract_strings": True,
        "implementation_lines": {
            "diagonal_lpd": [559, 560, 561, 562, 563, 564, 565, 566, 567, 568, 569, 570, 571, 572, 573],
            "contract_string_test": [183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194],
        },
    }

    mutations = {}
    forged = copy.deepcopy(config)
    forged["access_accounting"] = {"fabricated_zero_counter": 0}
    mutations["replace_required_access_ledger_with_unrelated_zero_key"] = accepted_by_validate(forged)
    forged = copy.deepcopy(config)
    forged["slacs_seal"]["sample_manifest"] = {
        "path": "configs/open_gravity_same_law_eso325_extended_source_v5.json",
        "sha256": package_hashes["config"]["sha256"],
    }
    mutations["repoint_slacs_sample_manifest_binding"] = accepted_by_validate(forged)
    forged = copy.deepcopy(config)
    forged["v4_preservation"]["path"] = "arbitrary-preservation.json"
    forged["v4_preservation"]["sha256"] = "0" * 64
    mutations["repoint_v4_preservation_binding"] = accepted_by_validate(forged)
    forged = copy.deepcopy(config)
    forged["outputs"]["receipt"] = "arbitrary/receipt.json"
    forged["outputs"]["artifact_directory"] = "arbitrary/artifacts"
    mutations["mutate_declared_output_paths"] = accepted_by_validate(forged)

    result = {
        "schema": "invariant-lane7-v5-independent-audit-probes-1.0",
        "scientific_payload_access": {
            "hst_array_elements_decoded": 0,
            "muse_array_elements_decoded": 0,
            "slacs_rows_or_response_values_opened": 0,
            "scientific_payload_files_opened_or_rehashed_by_this_audit": 0,
            "prohibition": "No scientific payload path is opened; only frozen package code/config/test/receipt artifacts and synthetic arrays are used.",
        },
        "package_hashes": package_hashes,
        "artifact_hashes": artifact_hashes,
        "v5_receipt_self_hash": {
            "recorded": recorded_self_hash,
            "recomputed_excluding_content_sha256": canonical_sha256(receipt_without_self_hash),
            "pass": recorded_self_hash == canonical_sha256(receipt_without_self_hash),
        },
        "exact_source_free_gate_reproduction": {
            "status": exact_gate["status"],
            "pass": bool(exact_gate["pass"]),
            "metrics": exact_gate["metrics"],
        },
        "holdout_leakage": holdout_leakage,
        "raw_convergence": raw_convergence,
        "helmholtz_zero_mode": helmholtz_probe,
        "execution_completeness": execution_completeness,
        "mutation_closure": {
            key: {"accepted": value[0], "error": value[1]} for key, value in mutations.items()
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
