"""Freeze and validate the coherent GW170817 Lane-2 successor.

The ``freeze`` command is deliberately target-free: it hashes but never opens the
HDF5 strain datasets.  It validates the pinned LALSuite runtime and performs the
synthetic GR/branch injection gates.  Real-data execution belongs to an
append-only successor which must inherit this receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import tarfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import differential_evolution

CONFIG_PATH = Path("configs/open_gravity_differential_propagation_gw170817_coherent_v4.json")
MODULE_PATH = Path(
    "src/sigma_theory_compiler/open_gravity_differential_propagation_gw170817_coherent_v4.py"
)
TEST_PATH = Path("tests/test_open_gravity_differential_propagation_gw170817_coherent_v4.py")
RUN_DIR = Path("runs/gravity/open-gravity-differential-propagation-gw170817-coherent-v4")
PREDICTION_PATH = RUN_DIR / "prediction-receipt.json"
ARTIFACT_DIR = RUN_DIR / "artifacts"

CONFIG_SCHEMA = "invariant-open-gravity-differential-propagation-gw170817-coherent-config-4.0"
PREDICTION_SCHEMA = (
    "invariant-open-gravity-differential-propagation-gw170817-coherent-prediction-receipt-4.0"
)

COMMON_PARAMETER_NAMES = (
    "chirp_mass_detector_solar",
    "mass_ratio_m2_over_m1",
    "spin_1z",
    "spin_2z",
    "lambda_1",
    "lambda_2",
    "cos_inclination",
    "polarization_radians",
    "coalescence_phase_radians",
    "geocentric_time_offset_seconds",
    "luminosity_distance_mpc",
    "calibration_amplitude_H1",
    "calibration_phase_H1",
    "calibration_amplitude_L1",
    "calibration_phase_L1",
    "calibration_amplitude_V1",
    "calibration_phase_V1",
)


class CoherentV4Error(RuntimeError):
    """Fail-closed error for the frozen successor."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CoherentV4Error(message)


def _repo_root(root: Path | None = None) -> Path:
    return (root or Path.cwd()).resolve()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _self_hash(value: Mapping[str, Any]) -> str:
    body = dict(value)
    body.pop("content_sha256", None)
    return _sha256_bytes(_canonical(body))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical(value) + b"\n")


def load_config(root: Path | None = None) -> dict[str, Any]:
    base = _repo_root(root)
    config = _read_json(base / CONFIG_PATH)
    validate_config(config)
    return config


def validate_config(config: Mapping[str, Any]) -> None:
    _require(config.get("schema_version") == CONFIG_SCHEMA, "wrong config schema")
    _require(
        config.get("analysis_id") == "open-gravity-differential-propagation-gw170817-coherent-v4",
        "wrong analysis id",
    )
    _require(
        config.get("status") == "BLOCKED_PRE_RESPONSE_END_TAPER_AUDIT",
        "config is not the retained blocked draft",
    )
    freeze = config["freeze_boundary"]
    _require(freeze["v4_strain_values_read_before_freeze"] == 0, "response leak")
    _require(freeze["v4_likelihood_values_computed_before_freeze"] == 0, "score leak")
    _require(
        freeze["gw190425_status"] == "SEALED_NOT_ACQUIRED_NOT_OPENED",
        "GW190425 is not sealed",
    )
    _require(not freeze["post_freeze_scientific_retuning_allowed"], "retuning allowed")
    _require(config["event"]["detectors"] == ["H1", "L1", "V1"], "detectors changed")
    _require(
        config["sources"]["lalsuite"]["version"] == "7.26.15",
        "LALSuite version changed",
    )
    _require(
        config["sources"]["lalsuite"]["primary_approximant"] == "IMRPhenomPv2_NRTidalv2",
        "waveform changed",
    )
    branch_ids = [row["id"] for row in config["transfer_branches"]]
    _require(
        branch_ids
        == [
            "GR",
            "DYNAMIC_PHASE",
            "ATTENUATION",
            "RESERVOIR",
            "NONLINEAR_PHASE",
            "SCREENED_PHASE",
        ],
        "branch set/order changed",
    )
    _require(config["multiple_comparisons"]["time_slides"]["count"] == 31, "slides changed")
    _require(config["optimizer"]["gr_seeds"] == [260831, 260832, 260833], "GR seeds changed")
    _require(
        config["optimizer"]["branch_seeds"] == [260841, 260842, 260843],
        "branch seeds changed",
    )
    _require(
        config["claim_boundary"]["no_paid_or_model_calls"],
        "paid/model calls not prohibited",
    )


def transfer_function(
    branch_id: str,
    frequency_hz: np.ndarray,
    parameters: Mapping[str, float],
    reference_frequency_hz: float = 100.0,
) -> np.ndarray:
    """Evaluate one dimensionally closed frozen propagation transfer law."""

    f = np.asarray(frequency_hz, dtype=float)
    _require(np.all(f > 0.0), "transfer frequencies must be positive")
    x = f / reference_frequency_hz
    if branch_id == "GR":
        return np.ones(f.shape, dtype=complex)
    if branch_id == "DYNAMIC_PHASE":
        phase = float(parameters["beta"]) * (x**-1 - 1.0)
        return np.exp(1j * phase)
    if branch_id == "ATTENUATION":
        return np.exp(-float(parameters["alpha"]) * np.log(x)).astype(complex)
    if branch_id == "RESERVOIR":
        strength = float(parameters["r"])
        f_res = 10.0 ** float(parameters["log10_f_res_hz"])
        numerator = 1.0 + strength / (1.0 + 1j * f / f_res)
        denominator = 1.0 + strength / (1.0 + 1j * reference_frequency_hz / f_res)
        return numerator / denominator
    if branch_id == "NONLINEAR_PHASE":
        phase = float(parameters["gamma"]) * (x**3 - 1.0)
        return np.exp(1j * phase)
    if branch_id == "SCREENED_PHASE":
        beta = float(parameters["beta_s"])
        f_screen = 10.0 ** float(parameters["log10_f_screen_hz"])
        shape = x**-1 / (1.0 + (f_screen / f) ** 4)
        reference_shape = 1.0 / (1.0 + (f_screen / reference_frequency_hz) ** 4)
        return np.exp(1j * beta * (shape - reference_shape))
    raise CoherentV4Error(f"unknown branch {branch_id}")


def _source_audit(config: Mapping[str, Any], base: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    source_specs: list[Mapping[str, Any]] = list(config["sources"]["strain"])
    source_specs.extend(
        [
            config["sources"]["published_psd"],
            config["sources"]["calibration_envelopes"],
            {
                "path": config["sources"]["lalsuite"]["release_source_path"],
                "bytes": config["sources"]["lalsuite"]["release_source_bytes"],
                "sha256": config["sources"]["lalsuite"]["release_source_sha256"],
            },
        ]
    )
    for spec in source_specs:
        path = base / spec["path"]
        _require(path.is_file(), f"missing source {spec['path']}")
        observed_bytes = path.stat().st_size
        observed_sha256 = _sha256_file(path)
        _require(observed_bytes == spec["bytes"], f"byte mismatch {spec['path']}")
        _require(observed_sha256 == spec["sha256"], f"hash mismatch {spec['path']}")
        rows.append(
            {
                "path": spec["path"],
                "bytes": observed_bytes,
                "sha256": observed_sha256,
            }
        )
    return {
        "status": "PASS_EXACT_SOURCE_BYTES_HASHED_WITHOUT_HDF5_DATASET_ACCESS",
        "files": rows,
        "hdf5_files_opened": 0,
        "strain_values_read": 0,
        "gw190425_opened": False,
    }


def _runtime_audit(config: Mapping[str, Any]) -> dict[str, Any]:
    import lal
    import lalsimulation

    expected = config["sources"]["lalsuite"]
    _require(metadata.version("lalsuite") == expected["version"], "wheel mismatch")
    _require(lal.__version__ == expected["lal_version"], "LAL mismatch")
    _require(
        lalsimulation.__version__ == expected["lalsimulation_version"],
        "LALSimulation mismatch",
    )
    approximants: dict[str, dict[str, Any]] = {}
    for name in [expected["primary_approximant"], expected["published_countermodel_approximant"]]:
        approximant = lalsimulation.GetApproximantFromString(name)
        implemented = bool(lalsimulation.SimInspiralImplementedFDApproximants(approximant))
        _require(implemented, f"FD approximant unavailable: {name}")
        approximants[name] = {"enum": int(approximant), "fd_implemented": implemented}
    packages = {name: metadata.version(name) for name in ["lalsuite", "numpy", "scipy", "h5py"]}
    return {
        "status": "PASS_PINNED_LALSUITE_RUNTIME_AND_APPROXIMANTS",
        "packages": packages,
        "lal_version": lal.__version__,
        "lalsimulation_version": lalsimulation.__version__,
        "approximants": approximants,
    }


def _load_published_psd(config: Mapping[str, Any], base: Path) -> dict[str, np.ndarray]:
    table = np.loadtxt(base / config["sources"]["published_psd"]["path"])
    _require(table.ndim == 2 and table.shape[1] == 4, "published PSD columns changed")
    _require(np.all(np.diff(table[:, 0]) > 0.0), "PSD frequency not monotonic")
    return {"frequency": table[:, 0], "H1": table[:, 1], "L1": table[:, 2], "V1": table[:, 3]}


def _load_calibration(config: Mapping[str, Any], base: Path) -> dict[str, np.ndarray]:
    path = base / config["sources"]["calibration_envelopes"]["path"]
    result: dict[str, np.ndarray] = {}
    with tarfile.open(path, "r:gz") as archive:
        for detector, member_name in config["sources"]["calibration_envelopes"]["members"].items():
            member = archive.extractfile(member_name)
            _require(member is not None, f"calibration member missing: {detector}")
            result[detector] = np.loadtxt(io.BytesIO(member.read()))
    return result


def _frequency_inputs(
    config: Mapping[str, Any], base: Path
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, dict[str, np.ndarray]]]:
    low, high = config["preprocessing"]["frequency_band_hz"]
    delta_f = config["preprocessing"]["likelihood_delta_f_hz"]
    frequency = np.arange(low, high + delta_f / 2.0, delta_f)
    published = _load_published_psd(config, base)
    psds = {
        detector: np.interp(frequency, published["frequency"], published[detector])
        for detector in config["event"]["detectors"]
    }
    raw_calibration = _load_calibration(config, base)
    calibration: dict[str, dict[str, np.ndarray]] = {}
    for detector, table in raw_calibration.items():
        median_magnitude = np.interp(frequency, table[:, 0], table[:, 1])
        median_phase = np.interp(frequency, table[:, 0], table[:, 2])
        lower_magnitude = np.interp(frequency, table[:, 0], table[:, 3])
        lower_phase = np.interp(frequency, table[:, 0], table[:, 4])
        upper_magnitude = np.interp(frequency, table[:, 0], table[:, 5])
        upper_phase = np.interp(frequency, table[:, 0], table[:, 6])
        calibration[detector] = {
            "median": median_magnitude * np.exp(1j * median_phase),
            "sigma_log_magnitude": (np.log(upper_magnitude) - np.log(lower_magnitude)) / 2.0,
            "sigma_phase": (upper_phase - lower_phase) / 2.0,
        }
    return frequency, psds, calibration


def _masses_from_chirp_q(chirp_mass: float, q: float) -> tuple[float, float]:
    eta = q / (1.0 + q) ** 2
    total = chirp_mass / eta ** (3.0 / 5.0)
    return total / (1.0 + q), total * q / (1.0 + q)


@dataclass(frozen=True)
class SyntheticContext:
    config: Mapping[str, Any]
    frequency: np.ndarray
    psds: Mapping[str, np.ndarray]
    calibration: Mapping[str, Mapping[str, np.ndarray]]
    delta_f: float
    antenna: Mapping[str, tuple[float, float]]
    delays: Mapping[str, float]


def _synthetic_context(config: Mapping[str, Any], base: Path) -> SyntheticContext:
    import lal

    frequency, psds, calibration = _frequency_inputs(config, base)
    event = config["event"]
    gps = lal.LIGOTimeGPS(event["event_gps"])
    ra = math.radians(event["right_ascension_degrees"])
    dec = math.radians(event["declination_degrees"])
    gmst = lal.GreenwichMeanSiderealTime(gps)
    detector_indices = {
        "H1": lal.LALDetectorIndexLHODIFF,
        "L1": lal.LALDetectorIndexLLODIFF,
        "V1": lal.LALDetectorIndexVIRGODIFF,
    }
    antenna: dict[str, tuple[float, float]] = {}
    delays: dict[str, float] = {}
    # psi=0 response is stored; arbitrary psi is evaluated in _network_waveform.
    for detector, index in detector_indices.items():
        lal_detector = lal.CachedDetectors[index]
        fp, fc = lal.ComputeDetAMResponse(lal_detector.response, ra, dec, 0.0, gmst)
        antenna[detector] = (float(fp), float(fc))
        delays[detector] = float(lal.TimeDelayFromEarthCenter(lal_detector.location, ra, dec, gps))
    return SyntheticContext(
        config=config,
        frequency=frequency,
        psds=psds,
        calibration=calibration,
        delta_f=float(config["preprocessing"]["likelihood_delta_f_hz"]),
        antenna=antenna,
        delays=delays,
    )


def _antenna_at_psi(fp0: float, fc0: float, psi: float) -> tuple[float, float]:
    # Polarization rotation by 2 psi.
    c = math.cos(2.0 * psi)
    s = math.sin(2.0 * psi)
    return fp0 * c + fc0 * s, -fp0 * s + fc0 * c


def _network_waveform(
    context: SyntheticContext,
    common: Mapping[str, float],
    branch_id: str = "GR",
    branch_parameters: Mapping[str, float] | None = None,
    approximant_name: str | None = None,
) -> dict[str, np.ndarray]:
    import lal
    import lalsimulation

    config = context.config
    approximant_name = approximant_name or config["sources"]["lalsuite"]["primary_approximant"]
    approximant = lalsimulation.GetApproximantFromString(approximant_name)
    m1, m2 = _masses_from_chirp_q(
        float(common["chirp_mass_detector_solar"]),
        float(common["mass_ratio_m2_over_m1"]),
    )
    params = lal.CreateDict()
    lalsimulation.SimInspiralWaveformParamsInsertTidalLambda1(params, float(common["lambda_1"]))
    lalsimulation.SimInspiralWaveformParamsInsertTidalLambda2(params, float(common["lambda_2"]))
    hp, hc = lalsimulation.SimInspiralChooseFDWaveform(
        m1 * lal.MSUN_SI,
        m2 * lal.MSUN_SI,
        0.0,
        0.0,
        float(common["spin_1z"]),
        0.0,
        0.0,
        float(common["spin_2z"]),
        float(common["luminosity_distance_mpc"]) * 1.0e6 * lal.PC_SI,
        math.acos(float(common["cos_inclination"])),
        float(common["coalescence_phase_radians"]),
        0.0,
        0.0,
        0.0,
        context.delta_f,
        float(config["preprocessing"]["frequency_band_hz"][0]),
        float(config["preprocessing"]["frequency_band_hz"][1]),
        float(config["event"]["reference_frequency_hz"]),
        params,
        approximant,
    )
    start_index = round(context.frequency[0] / context.delta_f)
    plus = np.asarray(
        hp.data.data[start_index : start_index + len(context.frequency)], dtype=complex
    )
    cross = np.asarray(
        hc.data.data[start_index : start_index + len(context.frequency)], dtype=complex
    )
    transfer = transfer_function(
        branch_id,
        context.frequency,
        branch_parameters or {},
        float(config["event"]["reference_frequency_hz"]),
    )
    psi = float(common["polarization_radians"])
    time_offset = float(common["geocentric_time_offset_seconds"])
    result: dict[str, np.ndarray] = {}
    for detector in config["event"]["detectors"]:
        fp, fc = _antenna_at_psi(*context.antenna[detector], psi)
        arrival = time_offset + context.delays[detector]
        time_phase = np.exp(-2j * np.pi * context.frequency * arrival)
        amp_coeff = float(common[f"calibration_amplitude_{detector}"])
        phase_coeff = float(common[f"calibration_phase_{detector}"])
        cal = context.calibration[detector]
        calibration_factor = cal["median"] * np.exp(
            amp_coeff * cal["sigma_log_magnitude"] + 1j * phase_coeff * cal["sigma_phase"]
        )
        result[detector] = (fp * plus + fc * cross) * time_phase * calibration_factor * transfer
    return result


def _inner(context: SyntheticContext, detector: str, left: np.ndarray, right: np.ndarray) -> float:
    return float(4.0 * context.delta_f * np.real(np.vdot(left, right / context.psds[detector])))


def _network_snr(context: SyntheticContext, waveforms: Mapping[str, np.ndarray]) -> float:
    return math.sqrt(
        sum(
            _inner(context, detector, waveform, waveform)
            for detector, waveform in waveforms.items()
        )
    )


def _log_likelihood_ratio(
    context: SyntheticContext,
    data: Mapping[str, np.ndarray],
    waveforms: Mapping[str, np.ndarray],
) -> tuple[float, dict[str, float]]:
    by_detector = {
        detector: _inner(context, detector, data[detector], waveforms[detector])
        - 0.5 * _inner(context, detector, waveforms[detector], waveforms[detector])
        for detector in context.config["event"]["detectors"]
    }
    return float(sum(by_detector.values())), by_detector


def _common_bounds(config: Mapping[str, Any]) -> list[tuple[float, float]]:
    priors = config["nuisance_priors"]
    bounds = [
        tuple(priors["chirp_mass_detector_solar"]["bounds"]),
        tuple(priors["mass_ratio_m2_over_m1"]["bounds"]),
        tuple(priors["spin_1z"]["bounds"]),
        tuple(priors["spin_2z"]["bounds"]),
        tuple(priors["lambda_1"]["bounds"]),
        tuple(priors["lambda_2"]["bounds"]),
        tuple(priors["cos_inclination"]["bounds"]),
        tuple(priors["polarization_radians"]["bounds"]),
        tuple(priors["coalescence_phase_radians"]["bounds"]),
        tuple(priors["geocentric_time_offset_seconds"]["bounds"]),
        tuple(priors["luminosity_distance_mpc"]["bounds"]),
    ]
    bounds.extend([(-3.0, 3.0)] * 6)
    return [(float(a), float(b)) for a, b in bounds]


def _vector_to_common(vector: Sequence[float]) -> dict[str, float]:
    return {name: float(value) for name, value in zip(COMMON_PARAMETER_NAMES, vector, strict=True)}


def _log_prior(common: Mapping[str, float]) -> float:
    # Constants shared by models are omitted.  Volume distance and Gaussian
    # calibration are the only non-uniform densities in the frozen parameterization.
    distance = float(common["luminosity_distance_mpc"])
    calibration_coefficients = [
        float(common[name]) for name in COMMON_PARAMETER_NAMES if name.startswith("calibration_")
    ]
    return 2.0 * math.log(distance) - 0.5 * sum(value * value for value in calibration_coefficients)


def _branch_spec(config: Mapping[str, Any], branch_id: str) -> Mapping[str, Any]:
    return next(row for row in config["transfer_branches"] if row["id"] == branch_id)


def _branch_bounds(config: Mapping[str, Any], branch_id: str) -> list[tuple[float, float]]:
    return [
        tuple(map(float, row["bounds"])) for row in _branch_spec(config, branch_id)["parameters"]
    ]


def _vector_to_branch(
    config: Mapping[str, Any], branch_id: str, vector: Sequence[float]
) -> dict[str, float]:
    names = [row["name"] for row in _branch_spec(config, branch_id)["parameters"]]
    return {name: float(value) for name, value in zip(names, vector, strict=True)}


def _fit_full_synthetic_gr(
    context: SyntheticContext, data: Mapping[str, np.ndarray]
) -> dict[str, Any]:
    config = context.config
    optimizer = config["optimizer"]
    results: list[dict[str, Any]] = []

    def objective(vector: np.ndarray) -> float:
        common = _vector_to_common(vector)
        try:
            waveform = _network_waveform(context, common)
            log_likelihood, _ = _log_likelihood_ratio(context, data, waveform)
        except (RuntimeError, ValueError, FloatingPointError):
            return 1.0e100
        return -(log_likelihood + _log_prior(common))

    for seed in optimizer["gr_seeds"]:
        fitted = differential_evolution(
            objective,
            _common_bounds(config),
            seed=int(seed),
            popsize=int(optimizer["population_multiplier"]),
            maxiter=int(optimizer["gr_max_iterations"]),
            tol=float(optimizer["relative_tolerance"]),
            atol=float(optimizer["absolute_tolerance"]),
            polish=bool(optimizer["polish"]),
            workers=int(optimizer["workers"]),
            updating="immediate",
        )
        common = _vector_to_common(fitted.x)
        waveform = _network_waveform(context, common)
        log_likelihood, by_detector = _log_likelihood_ratio(context, data, waveform)
        results.append(
            {
                "seed": int(seed),
                "success": bool(fitted.success),
                "message": str(fitted.message),
                "evaluations": int(fitted.nfev),
                "parameters": common,
                "log_likelihood": log_likelihood,
                "log_posterior_without_shared_constants": log_likelihood + _log_prior(common),
                "detector_log_likelihood": by_detector,
                "network_model_snr": _network_snr(context, waveform),
            }
        )
    best = max(results, key=lambda row: row["log_posterior_without_shared_constants"])
    spread = 2.0 * (
        max(row["log_likelihood"] for row in results)
        - min(row["log_likelihood"] for row in results)
    )
    return {"seeds": results, "best": best, "delta_2_log_likelihood_spread": spread}


def _normalized_synthetic_data(
    context: SyntheticContext,
    common: Mapping[str, float],
    target_snr: float,
    branch_id: str = "GR",
    branch_parameters: Mapping[str, float] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    initial = _network_waveform(context, common, branch_id, branch_parameters)
    initial_snr = _network_snr(context, initial)
    scaled = dict(common)
    scaled["luminosity_distance_mpc"] = (
        float(common["luminosity_distance_mpc"]) * initial_snr / target_snr
    )
    _require(
        10.0 <= scaled["luminosity_distance_mpc"] <= 200.0,
        "scaled injection distance outside prior",
    )
    return _network_waveform(context, scaled, branch_id, branch_parameters), scaled


def _fixed_branch_fit(
    context: SyntheticContext,
    data: Mapping[str, np.ndarray],
    common: Mapping[str, float],
    branch_id: str,
    seed: int,
) -> dict[str, Any]:
    bounds = _branch_bounds(context.config, branch_id)
    if not bounds:
        waveform = _network_waveform(context, common, "GR", {})
        log_likelihood, by_detector = _log_likelihood_ratio(context, data, waveform)
        return {
            "branch": branch_id,
            "parameters": {},
            "log_likelihood": log_likelihood,
            "detector_log_likelihood": by_detector,
            "evaluations": 1,
        }

    def objective(vector: np.ndarray) -> float:
        branch = _vector_to_branch(context.config, branch_id, vector)
        waveform = _network_waveform(context, common, branch_id, branch)
        log_likelihood, _ = _log_likelihood_ratio(context, data, waveform)
        return -log_likelihood

    fitted = differential_evolution(
        objective,
        bounds,
        seed=seed,
        popsize=8,
        maxiter=32,
        tol=1.0e-6,
        atol=1.0e-6,
        polish=True,
        workers=1,
        updating="immediate",
    )
    branch = _vector_to_branch(context.config, branch_id, fitted.x)
    waveform = _network_waveform(context, common, branch_id, branch)
    log_likelihood, by_detector = _log_likelihood_ratio(context, data, waveform)
    return {
        "branch": branch_id,
        "parameters": branch,
        "log_likelihood": log_likelihood,
        "detector_log_likelihood": by_detector,
        "evaluations": int(fitted.nfev),
        "success": bool(fitted.success),
    }


def _target_free_controls(config: Mapping[str, Any], base: Path) -> dict[str, Any]:
    context = _synthetic_context(config, base)
    injection = config["target_free_gates"]["zero_noise_gr_control"]
    injected_common = dict(injection["parameters"])
    injected_common["luminosity_distance_mpc"] = 40.0
    for detector in config["event"]["detectors"]:
        injected_common[f"calibration_amplitude_{detector}"] = 0.0
        injected_common[f"calibration_phase_{detector}"] = 0.0
    injected_common.pop("calibration_coefficients", None)
    gr_data, gr_common = _normalized_synthetic_data(
        context, injected_common, float(injection["network_snr"])
    )
    gr_fit = _fit_full_synthetic_gr(context, gr_data)
    gr_best = gr_fit["best"]
    gr_checks = {
        "chirp_mass": abs(
            gr_best["parameters"]["chirp_mass_detector_solar"]
            - gr_common["chirp_mass_detector_solar"]
        )
        <= 0.0015,
        "geocentric_time": abs(
            gr_best["parameters"]["geocentric_time_offset_seconds"]
            - gr_common["geocentric_time_offset_seconds"]
        )
        <= 0.002,
        "network_snr_fraction": gr_best["network_model_snr"] / float(injection["network_snr"])
        >= 0.90,
        "convergence": gr_fit["delta_2_log_likelihood_spread"] <= 2.0,
    }

    n = 2 * len(context.frequency) * len(config["event"]["detectors"])
    candidate_ids = [row["id"] for row in config["transfer_branches"]]
    injection_rows: list[dict[str, Any]] = []
    for injection_index, branch_injection in enumerate(
        config["target_free_gates"]["branch_injections"]
    ):
        branch_data, branch_common = _normalized_synthetic_data(
            context,
            injected_common,
            float(injection["network_snr"]),
            branch_injection["branch"],
            branch_injection["parameters"],
        )
        fits = [
            _fixed_branch_fit(
                context,
                branch_data,
                branch_common,
                candidate_id,
                260900 + 10 * injection_index + candidate_index,
            )
            for candidate_index, candidate_id in enumerate(candidate_ids)
        ]
        for fit in fits:
            extras = int(_branch_spec(config, fit["branch"])["extra_parameter_count"])
            fit["bic_relative_common"] = -2.0 * fit["log_likelihood"] + extras * math.log(n)
        winner = min(fits, key=lambda row: row["bic_relative_common"])
        amplitude_name = next(
            row["name"] for row in _branch_spec(config, branch_injection["branch"])["parameters"]
        )
        injected_amplitude = float(branch_injection["parameters"][amplitude_name])
        fitted_amplitude = float(
            next(row for row in fits if row["branch"] == branch_injection["branch"])["parameters"][
                amplitude_name
            ]
        )
        tolerance = max(0.25 * abs(injected_amplitude), 0.05)
        checks = {
            "branch_winner": winner["branch"] == branch_injection["branch"],
            "amplitude_recovery": abs(fitted_amplitude - injected_amplitude) <= tolerance,
        }
        injection_rows.append(
            {
                "injection_id": branch_injection["id"],
                "injected_branch": branch_injection["branch"],
                "injected_parameters": branch_injection["parameters"],
                "scaled_distance_mpc": branch_common["luminosity_distance_mpc"],
                "fits": fits,
                "winner": winner["branch"],
                "fitted_amplitude": fitted_amplitude,
                "amplitude_tolerance": tolerance,
                "checks": checks,
                "passed": all(checks.values()),
            }
        )
    all_branch_passed = all(row["passed"] for row in injection_rows)
    gr_passed = all(gr_checks.values())
    return {
        "status": "PASS" if gr_passed and all_branch_passed else "FAIL_IDENTIFIABILITY_GATE",
        "frequency_bins_per_detector": len(context.frequency),
        "effective_real_samples": n,
        "gr_injection": {
            "injected_parameters": gr_common,
            "fit": gr_fit,
            "checks": gr_checks,
            "passed": gr_passed,
        },
        "branch_injections": injection_rows,
        "all_branch_injections_passed": all_branch_passed,
        "strain_values_read": 0,
        "real_likelihood_values_computed": 0,
    }


def _prediction_payload(
    config: Mapping[str, Any],
    base: Path,
    source: Mapping[str, Any],
    runtime: Mapping[str, Any],
    controls: Mapping[str, Any],
) -> dict[str, Any]:
    package_hashes = {
        "config_raw_sha256": _sha256_file(base / CONFIG_PATH),
        "module_raw_sha256": _sha256_file(base / MODULE_PATH),
        "test_raw_sha256": _sha256_file(base / TEST_PATH),
    }
    formula_hashes = {
        row["id"]: _sha256_bytes(_canonical({"law": row["law"], "parameters": row["parameters"]}))
        for row in config["transfer_branches"]
    }
    artifacts = {
        "source-and-runtime-audit.json": {"source": source, "runtime": runtime},
        "target-free-injection-gates.json": controls,
    }
    for name, value in artifacts.items():
        _write_json(base / ARTIFACT_DIR / name, value)
    artifact_hashes = {name: _sha256_file(base / ARTIFACT_DIR / name) for name in artifacts}
    decision = (
        "FROZEN_PRE_RESPONSE_IDENTIFIABILITY_PASS_REAL_EXECUTOR_MAY_PROCEED"
        if controls["status"] == "PASS"
        else "FROZEN_PRE_RESPONSE_IDENTIFIABILITY_FAIL_REAL_EXECUTOR_DIAGNOSTIC_ONLY"
    )
    payload: dict[str, Any] = {
        "schema_version": PREDICTION_SCHEMA,
        "analysis_id": config["analysis_id"],
        "decision": decision,
        "package_hashes": package_hashes,
        "formula_hashes": formula_hashes,
        "artifact_sha256": artifact_hashes,
        "source_status": source["status"],
        "runtime_status": runtime["status"],
        "target_free_status": controls["status"],
        "freeze_boundary": config["freeze_boundary"],
        "access_ledger": {
            "source_files_hashed": len(source["files"]),
            "hdf5_files_opened": 0,
            "strain_values_read": 0,
            "real_likelihood_values_computed": 0,
            "gw190425_opened": 0,
            "model_calls": 0,
            "paid_calls": 0,
        },
        "claim_boundary": config["claim_boundary"],
    }
    payload["content_sha256"] = _self_hash(payload)
    return payload


def freeze(root: Path | None = None) -> str:
    base = _repo_root(root)
    config = load_config(base)
    package_hashes = {
        "config_raw_sha256": _sha256_file(base / CONFIG_PATH),
        "module_raw_sha256": _sha256_file(base / MODULE_PATH),
        "test_raw_sha256": _sha256_file(base / TEST_PATH),
    }
    payload: dict[str, Any] = {
        "schema_version": PREDICTION_SCHEMA,
        "analysis_id": config["analysis_id"],
        "decision": "BLOCKED_PRE_RESPONSE_END_TAPER_AUDIT_NO_INJECTIONS_NO_STRAIN",
        "package_hashes": package_hashes,
        "pre_response_blocker": config["pre_response_blocker"],
        "access_ledger": {
            "source_files_hashed": 0,
            "hdf5_files_opened": 0,
            "dq_values_read": 0,
            "strain_values_read": 0,
            "synthetic_injections_run": 0,
            "real_likelihood_values_computed": 0,
            "gw190425_opened": 0,
            "model_calls": 0,
            "paid_calls": 0,
        },
        "claim_boundary": config["claim_boundary"],
    }
    payload["content_sha256"] = _self_hash(payload)
    _write_json(base / PREDICTION_PATH, payload)
    return payload["decision"]


def check(root: Path | None = None) -> str:
    base = _repo_root(root)
    load_config(base)
    receipt = _read_json(base / PREDICTION_PATH)
    _require(receipt["schema_version"] == PREDICTION_SCHEMA, "prediction schema mismatch")
    _require(receipt["content_sha256"] == _self_hash(receipt), "prediction content hash mismatch")
    expected_hashes = {
        "config_raw_sha256": _sha256_file(base / CONFIG_PATH),
        "module_raw_sha256": _sha256_file(base / MODULE_PATH),
        "test_raw_sha256": _sha256_file(base / TEST_PATH),
    }
    _require(receipt["package_hashes"] == expected_hashes, "frozen package hash mismatch")
    _require(receipt["access_ledger"]["strain_values_read"] == 0, "prediction opened strain")
    _require(receipt["access_ledger"]["gw190425_opened"] == 0, "holdout opened")
    for name, expected in receipt.get("artifact_sha256", {}).items():
        _require(_sha256_file(base / ARTIFACT_DIR / name) == expected, f"artifact drift: {name}")
    return receipt["decision"]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("freeze")
    subparsers.add_parser("check")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "freeze":
        print(freeze())
    else:
        print(check())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
