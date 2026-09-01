"""Append-only DQ-cadence repair for the frozen GW170817 response run."""

from __future__ import annotations

import argparse
import csv
import io
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy import signal

from sigma_theory_compiler import (
    open_gravity_differential_propagation_gw170817_response_v2 as frozen,
)

CONFIG_PATH = Path("configs/open_gravity_differential_propagation_gw170817_response_v3.json")
MODULE_PATH = Path(
    "src/sigma_theory_compiler/open_gravity_differential_propagation_gw170817_response_v3.py"
)
TEST_PATH = Path("tests/test_open_gravity_differential_propagation_gw170817_response_v3.py")
PREDICTION_PATH = Path(
    "runs/gravity/open-gravity-differential-propagation-gw170817-response-v3/"
    "prediction-receipt.json"
)
ACQUISITION_PATH = Path(
    "runs/gravity/open-gravity-differential-propagation-gw170817-response-v3/"
    "source/acquisition.json"
)
OUTPUT_PATH = Path(
    "runs/gravity/open-gravity-differential-propagation-gw170817-response-v3/receipt.json"
)
ARTIFACT_DIR = OUTPUT_PATH.parent / "artifacts"
CONFIG_SCHEMA = "invariant-open-gravity-differential-propagation-gw170817-response-config-3.0"
PREDICTION_SCHEMA = "invariant-open-gravity-gw170817-response-prediction-receipt-3.0"
ACQUISITION_SCHEMA = "invariant-open-gravity-gw170817-response-acquisition-receipt-3.0"
RECEIPT_SCHEMA = "invariant-open-gravity-gw170817-response-receipt-3.0"


class GW170817ResponseV3Error(RuntimeError):
    """Raised when the append-only schema repair or frozen run fails."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GW170817ResponseV3Error(message)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _canonical(value: Any) -> bytes:
    return frozen._canonical(value)


def _sha256_bytes(value: bytes) -> str:
    return frozen._sha256_bytes(value)


def _sha256_file(path: Path) -> str:
    return frozen._sha256_file(path)


def _md5_file(path: Path) -> str:
    return frozen._md5_file(path)


def _self_hash(value: Mapping[str, Any]) -> str:
    return frozen._self_hash(value)


def _read_json(path: Path) -> dict[str, Any]:
    return frozen._read_json(path)


def _package_hashes(base: Path) -> dict[str, str]:
    return {
        "config_raw_sha256": _sha256_file(base / CONFIG_PATH),
        "module_raw_sha256": _sha256_file(base / MODULE_PATH),
        "test_raw_sha256": _sha256_file(base / TEST_PATH),
    }


def _science_config(config: Mapping[str, Any], base: Path) -> dict[str, Any]:
    predecessor = config["predecessor_v2"]
    science = frozen.load_config(base)
    section_keys = {
        "preprocessing": "preprocessing_sha256",
        "gr_control": "gr_control_sha256",
        "response_likelihood": "response_likelihood_sha256",
        "target_free_controls": "target_free_controls_config_sha256",
    }
    for section, hash_key in section_keys.items():
        _require(
            _sha256_bytes(_canonical(science[section])) == config["science_freeze"][hash_key],
            f"frozen science section changed: {section}",
        )
    prediction = _read_json(base / predecessor["prediction_path"])
    _require(
        prediction.get("content_sha256") == predecessor["prediction_content_sha256"],
        "v2 prediction content changed",
    )
    _require(
        prediction["target_free_controls"]["control_sha256"]
        == predecessor["target_free_control_sha256"],
        "v2 target-free controls changed",
    )
    return science


def validate_config(config: Mapping[str, Any], base: Path | None = None) -> None:
    _require(config.get("schema_version") == CONFIG_SCHEMA, "config schema changed")
    _require(
        config.get("analysis_id") == "open-gravity-differential-propagation-gw170817-response-v3",
        "analysis ID changed",
    )
    _require(
        config.get("status") == "FROZEN_RESPONSE_BLIND_DQ_CADENCE_REPAIR_SCIENCE_UNCHANGED",
        "status changed",
    )
    _require(
        config.get("package")
        == {
            "module_path": MODULE_PATH.as_posix(),
            "test_path": TEST_PATH.as_posix(),
            "prediction_receipt_path": PREDICTION_PATH.as_posix(),
            "acquisition_receipt_path": ACQUISITION_PATH.as_posix(),
            "output_path": OUTPUT_PATH.as_posix(),
            "artifact_directory": ARTIFACT_DIR.as_posix(),
        },
        "package paths changed",
    )
    predecessor = config.get("predecessor_v2")
    _require(isinstance(predecessor, dict), "v2 predecessor missing")
    if base is not None:
        for role in ("config", "module", "test", "prediction"):
            path = base / predecessor[f"{role}_path"]
            _require(path.is_file(), f"v2 predecessor missing: {path}")
            _require(
                _sha256_file(path) == predecessor[f"{role}_raw_sha256"],
                f"v2 predecessor changed: {path}",
            )
        _science_config(config, base)
    repair = config.get("repair")
    _require(
        repair
        == {
            "v2_failure": "DQ sample count changed",
            "observed_hdf_fact": (
                "strain is 4096 Hz with 16777216 samples; simple DQmask is 1 Hz "
                "with 4096 samples and Xspacing=1.0 s"
            ),
            "strain_rate_hz": 4096,
            "dq_rate_hz": 1,
            "dq_sample_count": 4096,
            "dq_slice_rule": (
                "index DQmask by integer GPS seconds independently of strain indices; "
                "a strain interval is admissible only when every covered DQ second passes "
                "DATA and CBC_CAT1"
            ),
            "scientific_formula_or_grid_changed": False,
        },
        "DQ cadence repair changed",
    )
    products = config.get("products")
    _require(
        isinstance(products, list)
        and [row.get("detector") for row in products] == ["H1", "L1", "V1"],
        "product inventory changed",
    )
    expected = {
        "H1": (125217658, "1a1cca3fb28686d5798539468a99dbae"),
        "L1": (124266501, "dbbde824db6df6a9f653db374fc5c88c"),
        "V1": (129470892, "8ea80f93257a292d82f0af497e2a4cff"),
    }
    for row in products:
        _require(
            (row["expected_bytes"], row["published_md5"]) == expected[row["detector"]],
            "product metadata changed",
        )
        _require("response-v3/source" in row["local_path"], "v3 source path changed")
    access = config.get("access_before_v3_freeze")
    _require(
        access
        == {
            "payload_files_downloaded": 1,
            "payload_bytes_downloaded": 125217658,
            "hdf5_headers_opened": 1,
            "dq_values_read": 0,
            "strain_values_read": 0,
            "real_likelihood_scores": 0,
            "h1_sha256": "9e3f8a3adb966f6d70eeade0bc44bea2344f85b2af5233a3cba34a723984c9e2",
            "response_blind_science_grid": True,
        },
        "pre-v3 access ledger changed",
    )
    boundary = config.get("claim_boundary")
    _require(
        isinstance(boundary, dict)
        and boundary.get("v2_preserved") is True
        and boundary.get("theorem_unchanged") is True
        and boundary.get("science_grid_unchanged") is True
        and boundary.get("repair_only_dq_cadence") is True
        and boundary.get("empirical_support_claim") is False
        and boundary.get("publication_ready") is False,
        "claim boundary widened",
    )


def load_config(root: Path | None = None) -> dict[str, Any]:
    base = (root or _repo_root()).resolve()
    config = _read_json(base / CONFIG_PATH)
    validate_config(config, base)
    return config


def build_prediction_receipt(config: Mapping[str, Any], base: Path) -> dict[str, Any]:
    science = _science_config(config, base)
    controls = frozen.target_free_controls(science)
    _require(
        controls["control_sha256"] == config["predecessor_v2"]["target_free_control_sha256"],
        "target-free control replay changed",
    )
    receipt: dict[str, Any] = {
        "schema_version": PREDICTION_SCHEMA,
        "analysis_id": config["analysis_id"],
        "status": "FROZEN_AFTER_HEADER_ONLY_FAILURE_BEFORE_ANY_STRAIN_OR_DQ_VALUE_ACCESS",
        "content_sha256": "",
        "package_hashes": _package_hashes(base),
        "config_content_sha256": _sha256_bytes(_canonical(config)),
        "predecessor_v2": config["predecessor_v2"],
        "science_freeze": config["science_freeze"],
        "repair": config["repair"],
        "products": config["products"],
        "target_free_controls": controls,
        "access_at_freeze": config["access_before_v3_freeze"],
        "claim_boundary": config["claim_boundary"],
    }
    receipt["content_sha256"] = _self_hash(receipt)
    return receipt


def freeze(root: Path | None = None) -> str:
    base = (root or _repo_root()).resolve()
    config = load_config(base)
    payload = _canonical(build_prediction_receipt(config, base)) + b"\n"
    path = base / PREDICTION_PATH
    if path.exists():
        _require(path.read_bytes() == payload, "v3 prediction receipt differs")
        return "EXISTING_IDENTICAL"
    frozen._atomic_write(path, payload)
    return "CREATED"


def validate_prediction(config: Mapping[str, Any], base: Path) -> dict[str, Any]:
    path = base / PREDICTION_PATH
    _require(path.is_file(), "v3 prediction receipt missing")
    observed = _read_json(path)
    _require(observed.get("content_sha256") == _self_hash(observed), "v3 prediction hash invalid")
    _require(observed == build_prediction_receipt(config, base), "v3 prediction changed")
    return observed


def _decode_strings(dataset: h5py.Dataset) -> list[str]:
    return frozen._decode_strings(dataset)


def _hdf_header_and_dq(
    path: Path, science: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    contract = science["hdf5_contract"]
    repair = config["repair"]
    with h5py.File(path, "r") as handle:
        strain = handle[contract["strain_dataset"]]
        dq = handle[contract["dq_dataset"]]
        names = _decode_strings(handle[contract["dq_shortnames_dataset"]])
        _require(strain.shape == (contract["sample_count"],), "strain sample count changed")
        _require(dq.shape == (repair["dq_sample_count"],), "DQ one-Hz sample count changed")
        _require(strain.dtype == np.dtype("float64"), "strain dtype changed")
        _require(np.issubdtype(dq.dtype, np.integer), "DQ dtype changed")
        _require(float(dq.attrs["Xspacing"]) == 1.0, "DQ cadence changed")
        _require(
            float(strain.attrs["Xspacing"]) == 1.0 / repair["strain_rate_hz"],
            "strain cadence changed",
        )
        for required in contract["required_dq_flags"]:
            _require(required in names, f"required DQ flag absent: {required}")
        unique, counts = np.unique(dq[()], return_counts=True)
        required_mask = sum(1 << names.index(name) for name in contract["required_dq_flags"])
        required_pass = int(
            sum(
                count
                for value, count in zip(unique, counts, strict=True)
                if int(value) & required_mask == required_mask
            )
        )
        return {
            "groups": sorted(handle.keys()),
            "strain_shape": list(strain.shape),
            "strain_dtype": str(strain.dtype),
            "strain_xspacing_seconds": float(strain.attrs["Xspacing"]),
            "dq_shape": list(dq.shape),
            "dq_dtype": str(dq.dtype),
            "dq_xspacing_seconds": float(dq.attrs["Xspacing"]),
            "dq_shortnames": names,
            "dq_unique_counts": {
                str(int(value)): int(count) for value, count in zip(unique, counts, strict=True)
            },
            "required_dq_mask": required_mask,
            "required_dq_pass_seconds": required_pass,
            "required_dq_pass_fraction": required_pass / int(dq.shape[0]),
        }


def acquire(root: Path | None = None) -> str:
    base = (root or _repo_root()).resolve()
    config = load_config(base)
    science = _science_config(config, base)
    prediction = validate_prediction(config, base)
    rows = []
    for product in config["products"]:
        row = frozen._download(product, base / product["local_path"])
        row["hdf5_and_dq"] = _hdf_header_and_dq(base / product["local_path"], science, config)
        rows.append(row)
    receipt: dict[str, Any] = {
        "schema_version": ACQUISITION_SCHEMA,
        "analysis_id": config["analysis_id"],
        "status": "EXACT_THREE_C00_PRODUCTS_ACQUIRED_HASHED_DQ_CADENCE_VALIDATED",
        "content_sha256": "",
        "prediction_receipt_raw_sha256": _sha256_file(base / PREDICTION_PATH),
        "prediction_receipt_content_sha256": prediction["content_sha256"],
        "products": rows,
        "counts": {
            "files": 3,
            "payload_bytes": sum(row["bytes"] for row in rows),
            "strain_values_read": 0,
            "dq_values_read": 3 * int(config["repair"]["dq_sample_count"]),
        },
    }
    receipt["content_sha256"] = _self_hash(receipt)
    payload = _canonical(receipt) + b"\n"
    path = base / ACQUISITION_PATH
    if path.exists():
        _require(path.read_bytes() == payload, "v3 acquisition receipt differs")
        return "EXISTING_IDENTICAL"
    frozen._atomic_write(path, payload)
    return "CREATED"


def validate_acquisition(config: Mapping[str, Any], base: Path) -> dict[str, Any]:
    prediction = validate_prediction(config, base)
    path = base / ACQUISITION_PATH
    _require(path.is_file(), "v3 acquisition receipt missing")
    receipt = _read_json(path)
    _require(receipt.get("content_sha256") == _self_hash(receipt), "v3 acquisition hash invalid")
    _require(
        receipt.get("prediction_receipt_content_sha256") == prediction["content_sha256"],
        "v3 prediction/acquisition binding changed",
    )
    _require(len(receipt.get("products", [])) == 3, "v3 acquisition inventory changed")
    for expected, observed in zip(config["products"], receipt["products"], strict=True):
        path_value = base / expected["local_path"]
        _require(path_value.is_file(), f"payload missing: {expected['detector']}")
        _require(_sha256_file(path_value) == observed["sha256"], "payload SHA changed")
        _require(_md5_file(path_value) == expected["published_md5"], "payload MD5 changed")
    return receipt


def _slice_dq(
    dataset: h5py.Dataset, gps_start: int, target_start: int, duration: int
) -> np.ndarray:
    start = target_start - gps_start
    stop = start + duration
    _require(0 <= start < stop <= dataset.shape[0], "requested DQ interval outside payload")
    return np.asarray(dataset[start:stop])


def _required_dq_mask(handle: h5py.File, science: Mapping[str, Any]) -> int:
    contract = science["hdf5_contract"]
    names = _decode_strings(handle[contract["dq_shortnames_dataset"]])
    return sum(1 << names.index(name) for name in contract["required_dq_flags"])


def _read_and_preprocess_detector(
    product: Mapping[str, Any],
    science: Mapping[str, Any],
    config: Mapping[str, Any],
    base: Path,
) -> dict[str, Any]:
    preprocessing = science["preprocessing"]
    contract = science["hdf5_contract"]
    rate = int(config["repair"]["strain_rate_hz"])
    gps_start = int(contract["gps_start"])
    with h5py.File(base / product["local_path"], "r") as handle:
        strain_dataset = handle[contract["strain_dataset"]]
        dq_dataset = handle[contract["dq_dataset"]]
        required_mask = _required_dq_mask(handle, science)
        analysis = frozen._slice(
            strain_dataset,
            gps_start,
            int(preprocessing["analysis_gps_start"]),
            int(preprocessing["analysis_duration_seconds"]),
            rate,
        ).astype(float)
        analysis_dq = _slice_dq(
            dq_dataset,
            gps_start,
            int(preprocessing["analysis_gps_start"]),
            int(preprocessing["analysis_duration_seconds"]),
        )
        _require(
            np.all((analysis_dq & required_mask) == required_mask),
            f"analysis DQ failed: {product['detector']}",
        )
        psd_arrays = []
        psd_dq_fractions = []
        for interval in preprocessing["psd_intervals_gps"]:
            values = frozen._slice(
                strain_dataset,
                gps_start,
                int(interval[0]),
                int(preprocessing["psd_interval_seconds"]),
                rate,
            ).astype(float)
            dq_values = _slice_dq(
                dq_dataset,
                gps_start,
                int(interval[0]),
                int(preprocessing["psd_interval_seconds"]),
            )
            good = (dq_values & required_mask) == required_mask
            psd_dq_fractions.append(float(np.mean(good)))
            _require(np.all(good), f"PSD DQ failed: {product['detector']}:{interval[0]}")
            frequencies_psd, psd = signal.welch(
                values,
                fs=rate,
                window="hann",
                nperseg=int(preprocessing["psd_segment_seconds"]) * rate,
                noverlap=int(
                    preprocessing["psd_segment_seconds"]
                    * rate
                    * preprocessing["psd_overlap_fraction"]
                ),
                detrend="constant",
                scaling="density",
                average="median",
            )
            psd_arrays.append(psd)
    times = int(preprocessing["analysis_gps_start"]) + np.arange(analysis.size) / rate
    if product["detector"] in preprocessing["glitch_gate_detectors"]:
        glitch_window = frozen._glitch_gate(times, preprocessing["l1_glitch_gate"])
    else:
        glitch_window = np.ones(analysis.size)
    tapered = signal.detrend(analysis, type="linear") * glitch_window
    tapered *= signal.windows.tukey(analysis.size, alpha=0.1)
    spectrum = np.fft.rfft(tapered) / rate
    frequencies = np.fft.rfftfreq(tapered.size, d=1.0 / rate)
    psd_mean = np.mean(np.vstack(psd_arrays), axis=0)
    _require(np.all(np.isfinite(psd_mean)) and np.all(psd_mean > 0.0), "invalid PSD")
    interpolated_psd = np.exp(np.interp(frequencies, frequencies_psd, np.log(psd_mean)))
    band = preprocessing["frequency_band_hz"]
    indices = np.flatnonzero((frequencies >= band[0]) & (frequencies <= band[1]))[
        :: int(preprocessing["frequency_stride"])
    ]
    _require(indices.size > 100, "analysis frequency grid too small")
    return {
        "detector": product["detector"],
        "frequencies": frequencies[indices],
        "spectrum": spectrum[indices],
        "psd": interpolated_psd[indices],
        "delta_f": float(frequencies[1] - frequencies[0]) * int(preprocessing["frequency_stride"]),
        "analysis_required_dq_fraction": float(
            np.mean((analysis_dq & required_mask) == required_mask)
        ),
        "analysis_dq_seconds_read": int(analysis_dq.size),
        "psd_required_dq_fractions": psd_dq_fractions,
        "psd_dq_seconds_read": int(
            len(preprocessing["psd_intervals_gps"]) * int(preprocessing["psd_interval_seconds"])
        ),
        "glitch_gate_zero_samples": int(np.count_nonzero(glitch_window == 0.0)),
        "strain_samples_read": int(
            analysis.size
            + len(preprocessing["psd_intervals_gps"])
            * int(preprocessing["psd_interval_seconds"])
            * rate
        ),
        "analysis_spectrum_sha256": _sha256_bytes(
            np.column_stack((spectrum[indices].real, spectrum[indices].imag))
            .astype("<f8")
            .tobytes()
        ),
        "psd_sha256": _sha256_bytes(psd_mean.astype("<f8").tobytes()),
    }


def real_analysis(
    config: Mapping[str, Any], science: Mapping[str, Any], base: Path
) -> dict[str, Any]:
    acquisition = validate_acquisition(config, base)
    prediction = validate_prediction(config, base)
    processed = [
        _read_and_preprocess_detector(product, science, config, base)
        for product in config["products"]
    ]
    gr = frozen._gr_recovery(science, processed)
    response_fit = frozen._real_response_fit(science, processed, gr, prediction)
    summaries = [
        {key: value for key, value in row.items() if key not in {"frequencies", "spectrum", "psd"}}
        for row in processed
    ]
    return {
        "source_receipt_content_sha256": acquisition["content_sha256"],
        "prediction_receipt_content_sha256": prediction["content_sha256"],
        "processed_detectors": summaries,
        "gr_recovery": gr,
        "response_likelihood": response_fit,
        "access": {
            "payload_files_opened": 3,
            "strain_samples_read": sum(row["strain_samples_read"] for row in summaries),
            "dq_seconds_read_during_analysis": sum(
                row["analysis_dq_seconds_read"] + row["psd_dq_seconds_read"] for row in summaries
            ),
            "real_data_scores_computed": 1,
            "model_calls": 0,
            "paid_calls": 0,
            "post_response_grid_changes": 0,
        },
    }


def _csv_bytes(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def artifact_payloads(
    config: Mapping[str, Any], science: Mapping[str, Any], base: Path
) -> tuple[dict[str, bytes], dict[str, Any]]:
    prediction = validate_prediction(config, base)
    acquisition = validate_acquisition(config, base)
    analysis = real_analysis(config, science, base)
    theorem = {
        "schema_version": "invariant-open-gravity-theorem-binding-3.0",
        "predecessor": science["predecessors"]["theorem"],
        "theorem_modified": False,
    }
    source = {
        "schema_version": "invariant-open-gravity-gw170817-source-result-3.0",
        "official_metadata": science["official_metadata"],
        "v3_repair": config["repair"],
        "acquisition": acquisition,
    }
    real = {
        "schema_version": "invariant-open-gravity-gw170817-real-data-result-3.0",
        "analysis": analysis,
        "claim_boundary": config["claim_boundary"],
    }
    rows = [
        [
            row["family_id"],
            row["best_coefficient"],
            row["delta_2_log_likelihood"],
            row["target_free_null_max_threshold"],
            row["exceeds_target_free_threshold"],
            row["interpretation_status"],
        ]
        for row in analysis["response_likelihood"]["families"]
    ]
    report = f"""# GW170817 differential-propagation response v3

## Theorem result

The predecessor theorem is preserved unchanged.  The stationary massless force remains independent of a bare propagation-speed change.

## Source result

The v2 prediction receipt and all science grids are preserved.  V2 failed closed because it expected a strain-rate DQ mask.  V3 repairs only the observed GWOSC schema fact: strain is 4096 Hz while DQmask is 1 Hz.  Exact H1/L1/V1 C00 products pass published MD5, computed SHA-256, HDF5, cadence, and DQ checks.

## Real-data result

The frozen approximate TaylorF2 GR control pass is `{analysis["gr_recovery"]["passed"]}`; network SNR is `{analysis["gr_recovery"]["network_snr"]}` and recovered chirp mass is `{analysis["gr_recovery"]["best"]["chirp_mass_solar"]}` solar masses.  Each response-family result is reported with the unchanged target-free threshold.  A GR-control failure invalidates all response interpretation.

This is an approximate one-event response-shape diagnostic, not published tidal parameter estimation, not a fundamental-parameter posterior, and not an empirical-support or publication-ready claim.
"""
    payloads = {
        "theorem-result.json": _canonical(theorem) + b"\n",
        "source-metadata-and-acquisition-result.json": _canonical(source) + b"\n",
        "real-data-result.json": _canonical(real) + b"\n",
        "real-data-response-families.csv": _csv_bytes(
            [
                "family_id",
                "best_coefficient",
                "delta_2_log_likelihood",
                "target_free_null_max_threshold",
                "exceeds_target_free_threshold",
                "interpretation_status",
            ],
            rows,
        ),
        "target-free-controls.json": _canonical(prediction["target_free_controls"]) + b"\n",
        "report.md": report.encode("utf-8"),
    }
    return payloads, analysis


def build_receipt(
    config: Mapping[str, Any], science: Mapping[str, Any], base: Path
) -> tuple[dict[str, Any], dict[str, bytes]]:
    payloads, analysis = artifact_payloads(config, science, base)
    gr_pass = bool(analysis["gr_recovery"]["passed"])
    decision = (
        "REAL_DATA_RESPONSE_RUN_COMPLETE_DQ_CADENCE_REPAIRED__"
        f"{'GR_CONTROL_PASS' if gr_pass else 'GR_CONTROL_FAIL'}__NO_EMPIRICAL_SUPPORT_CLAIM"
    )
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "analysis_id": config["analysis_id"],
        "decision": decision,
        "content_sha256": "",
        "package_hashes": _package_hashes(base),
        "config_content_sha256": _sha256_bytes(_canonical(config)),
        "v2_prediction_raw_sha256": config["predecessor_v2"]["prediction_raw_sha256"],
        "v3_prediction_raw_sha256": _sha256_file(base / PREDICTION_PATH),
        "acquisition_raw_sha256": _sha256_file(base / ACQUISITION_PATH),
        "artifact_sha256": {
            name: _sha256_bytes(payload) for name, payload in sorted(payloads.items())
        },
        "theorem_result": "PRESERVED_UNCHANGED",
        "source_result": "EXACT_THREE_C00_PRODUCTS_ACQUIRED_HASHED_AND_VALIDATED",
        "real_data_result": {
            "gr_control_passed": gr_pass,
            "network_snr": analysis["gr_recovery"]["network_snr"],
            "recovered_chirp_mass_solar": analysis["gr_recovery"]["best"]["chirp_mass_solar"],
            "response_family_statuses": {
                row["family_id"]: row["interpretation_status"]
                for row in analysis["response_likelihood"]["families"]
            },
            "any_empirical_support_claim": False,
        },
        "access_ledger": analysis["access"],
        "claim_boundary": config["claim_boundary"],
    }
    receipt["content_sha256"] = _self_hash(receipt)
    return receipt, payloads


def build(root: Path | None = None) -> str:
    base = (root or _repo_root()).resolve()
    config = load_config(base)
    science = _science_config(config, base)
    receipt, payloads = build_receipt(config, science, base)
    targets = {base / ARTIFACT_DIR / name: payload for name, payload in payloads.items()}
    targets[base / OUTPUT_PATH] = _canonical(receipt) + b"\n"
    existing = [path for path in targets if path.exists()]
    if existing:
        _require(len(existing) == len(targets), "partial v3 output package exists")
        for path, payload in targets.items():
            _require(path.read_bytes() == payload, f"existing output differs: {path}")
        return "EXISTING_IDENTICAL"
    for path, payload in targets.items():
        frozen._atomic_write(path, payload)
    return "CREATED"


def check(root: Path | None = None) -> str:
    base = (root or _repo_root()).resolve()
    config = load_config(base)
    science = _science_config(config, base)
    expected, payloads = build_receipt(config, science, base)
    observed = _read_json(base / OUTPUT_PATH)
    _require(observed.get("content_sha256") == _self_hash(observed), "v3 receipt hash invalid")
    _require(observed == expected, "v3 receipt differs from deterministic rebuild")
    for name, payload in payloads.items():
        path = base / ARTIFACT_DIR / name
        _require(path.is_file(), f"missing artifact: {name}")
        _require(path.read_bytes() == payload, f"artifact differs: {name}")
    return "VALID"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("freeze")
    subparsers.add_parser("acquire")
    subparsers.add_parser("build")
    subparsers.add_parser("check")
    subparsers.add_parser("status")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "freeze":
        print(freeze())
        return 0
    if args.command == "acquire":
        print(acquire())
        return 0
    if args.command == "build":
        print(build())
        return 0
    if args.command == "check":
        print(check())
        return 0
    config = load_config()
    print(
        json.dumps(
            {
                "analysis_id": config["analysis_id"],
                "status": config["status"],
                "prediction_frozen": PREDICTION_PATH.is_file(),
                "source_acquired": ACQUISITION_PATH.is_file(),
                "real_result_built": OUTPUT_PATH.is_file(),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
