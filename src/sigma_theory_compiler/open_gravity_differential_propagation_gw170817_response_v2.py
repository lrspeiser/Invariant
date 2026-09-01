"""Frozen GW170817 C00 acquisition and differential-propagation response test."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import tempfile
import urllib.request
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy import signal

CONFIG_PATH = Path("configs/open_gravity_differential_propagation_gw170817_response_v2.json")
MODULE_PATH = Path(
    "src/sigma_theory_compiler/open_gravity_differential_propagation_gw170817_response_v2.py"
)
TEST_PATH = Path("tests/test_open_gravity_differential_propagation_gw170817_response_v2.py")
PREDICTION_PATH = Path(
    "runs/gravity/open-gravity-differential-propagation-gw170817-response-v2/"
    "prediction-receipt.json"
)
ACQUISITION_PATH = Path(
    "runs/gravity/open-gravity-differential-propagation-gw170817-response-v2/"
    "source/acquisition.json"
)
OUTPUT_PATH = Path(
    "runs/gravity/open-gravity-differential-propagation-gw170817-response-v2/receipt.json"
)
ARTIFACT_DIR = OUTPUT_PATH.parent / "artifacts"
CONFIG_SCHEMA = "invariant-open-gravity-differential-propagation-gw170817-response-config-2.0"
PREDICTION_SCHEMA = "invariant-open-gravity-gw170817-response-prediction-receipt-2.0"
ACQUISITION_SCHEMA = "invariant-open-gravity-gw170817-response-acquisition-receipt-2.0"
RECEIPT_SCHEMA = "invariant-open-gravity-gw170817-response-receipt-2.0"
DECISION_PREFIX = "REAL_DATA_RESPONSE_RUN_COMPLETE"
SOLAR_MASS_TIME_SECONDS = 4.9254909476412675e-6


class GW170817ResponseError(RuntimeError):
    """Raised when a frozen gate, source, likelihood, or seal fails."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GW170817ResponseError(message)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _md5_file(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _self_hash(value: Mapping[str, Any]) -> str:
    body = dict(value)
    body["content_sha256"] = ""
    return _sha256_bytes(_canonical(body))


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GW170817ResponseError(f"invalid JSON: {path}") from exc
    _require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def _inclusive_grid(specification: Mapping[str, Any]) -> np.ndarray:
    minimum = float(specification["min"])
    maximum = float(specification["max"])
    step = float(specification["step"])
    _require(step > 0.0 and maximum >= minimum, "invalid frozen grid")
    count = round((maximum - minimum) / step)
    grid = minimum + step * np.arange(count + 1, dtype=float)
    _require(
        math.isclose(float(grid[-1]), maximum, rel_tol=0.0, abs_tol=1e-12), "grid does not close"
    )
    return grid


def validate_config(config: Mapping[str, Any], base: Path | None = None) -> None:
    _require(config.get("schema_version") == CONFIG_SCHEMA, "config schema changed")
    _require(
        config.get("analysis_id") == "open-gravity-differential-propagation-gw170817-response-v2",
        "analysis ID changed",
    )
    _require(
        config.get("status") == "FROZEN_PRE_RESPONSE_EXACT_C00_PRODUCTS_LIKELIHOOD_AND_INJECTIONS",
        "status changed",
    )
    package = config.get("package")
    _require(
        package
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
    predecessors = config.get("predecessors")
    _require(
        isinstance(predecessors, dict) and set(predecessors) == {"theorem", "source_preflight"},
        "predecessors changed",
    )
    if base is not None:
        for predecessor in predecessors.values():
            for role in ("config", "module", "test", "receipt"):
                path = base / predecessor[f"{role}_path"]
                _require(path.is_file(), f"missing predecessor: {path}")
                _require(
                    _sha256_file(path) == predecessor[f"{role}_raw_sha256"],
                    f"predecessor changed: {path}",
                )
            receipt = _read_json(base / predecessor["receipt_path"])
            _require(
                receipt.get("content_sha256") == predecessor["receipt_content_sha256"],
                "predecessor receipt content changed",
            )
    metadata = config.get("official_metadata")
    _require(
        isinstance(metadata, dict)
        and metadata.get("event_uid") == "GW170817-v1"
        and metadata.get("event_gps") == 1187008882.43
        and metadata.get("calibration") == "C00 online calibration"
        and metadata.get("noise_subtraction") is False,
        "event metadata changed",
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
    for product in products:
        detector = product["detector"]
        _require(
            (product["expected_bytes"], product["published_md5"]) == expected[detector],
            f"official product metadata changed: {detector}",
        )
        _require(product["filename"] in product["url"], "product URL/filename mismatch")
        _require(product["local_path"].endswith(product["filename"]), "local product mismatch")
    hdf = config.get("hdf5_contract")
    _require(
        isinstance(hdf, dict)
        and hdf.get("gps_start") == 1187006834
        and hdf.get("duration_seconds") == 4096
        and hdf.get("sample_rate_hz") == 4096
        and hdf.get("sample_count") == 16777216
        and hdf.get("required_dq_flags") == ["DATA", "CBC_CAT1"],
        "HDF5 contract changed",
    )
    preprocessing = config.get("preprocessing")
    _require(
        isinstance(preprocessing, dict)
        and preprocessing.get("analysis_gps_start") == 1187008756
        and preprocessing.get("analysis_duration_seconds") == 128
        and preprocessing.get("analysis_sample_count") == 524288
        and preprocessing.get("frequency_band_hz") == [30, 300]
        and preprocessing.get("frequency_stride") == 8
        and preprocessing.get("notches") == []
        and preprocessing.get("resampling") == "none",
        "preprocessing changed",
    )
    gate = preprocessing.get("l1_glitch_gate")
    _require(
        gate
        == {
            "center_gps": 1187008881.389,
            "zero_half_width_seconds": 0.1,
            "rolloff_seconds_each_side": 0.5,
            "source": "official GWOSC GW170817 page",
        },
        "official L1 glitch gate changed",
    )
    gr = config.get("gr_control")
    _require(
        isinstance(gr, dict)
        and gr.get("fourier_phase_signs") == [-1, 1]
        and len(_inclusive_grid(gr["chirp_mass_solar_grid"])) == 31
        and len(gr.get("symmetric_mass_ratio_grid", [])) == 5
        and len(_inclusive_grid(gr["coalescence_offset_seconds_grid"])) == 123,
        "GR control grid changed",
    )
    response = config.get("response_likelihood")
    _require(
        isinstance(response, dict)
        and [row.get("id") for row in response.get("families", [])]
        == [
            "PHASE_INVERSE_FREQUENCY",
            "PHASE_CUBIC_FREQUENCY",
            "ATTENUATION_LINEAR_FREQUENCY",
        ]
        and "no c_g" in response.get("interpretation_boundary", ""),
        "response family or boundary changed",
    )
    for family in response["families"]:
        _require(0.0 in _inclusive_grid(family["grid"]), f"null absent: {family['id']}")
    controls = config.get("target_free_controls")
    _require(
        isinstance(controls, dict)
        and len(controls.get("zero_noise_injections", [])) == 4
        and controls.get("null_noise", {}).get("realizations") == 64,
        "target-free controls changed",
    )
    audit = config.get("official_metadata_audit")
    _require(
        isinstance(audit, dict)
        and audit.get("status") == "PASS_STRICT_PREDECESSOR_AND_OFFICIAL_METADATA_AUDIT"
        and audit.get("observational_payload_bytes_opened_during_audit") == 0,
        "metadata audit changed",
    )
    access = config.get("access_contract")
    _require(
        access
        == {
            "metadata_requests_before_freeze": 3,
            "observational_payload_files_before_freeze": 0,
            "observational_payload_bytes_before_freeze": 0,
            "model_calls": 0,
            "paid_calls": 0,
            "post_response_grid_changes_allowed": 0,
        },
        "access contract changed",
    )
    boundary = config.get("claim_boundary")
    _require(
        isinstance(boundary, dict)
        and boundary.get("theorem_modified") is False
        and boundary.get("real_data_result_separate") is True
        and boundary.get("approximate_GR_control_only") is True
        and boundary.get("published_GR_parameter_estimation_reproduced") is False
        and boundary.get("fundamental_parameter_posterior") is False
        and boundary.get("publication_ready") is False,
        "claim boundary widened",
    )


def load_config(root: Path | None = None) -> dict[str, Any]:
    base = (root or _repo_root()).resolve()
    config = _read_json(base / CONFIG_PATH)
    validate_config(config, base)
    return config


def _taylorf2(
    frequencies: np.ndarray, chirp_mass: float, eta: float, phase_sign: float
) -> np.ndarray:
    _require(chirp_mass > 0.0 and 0.0 < eta <= 0.25, "invalid TaylorF2 masses")
    total_mass = chirp_mass / eta ** (3.0 / 5.0)
    v = (np.pi * total_mass * SOLAR_MASS_TIME_SECONDS * frequencies) ** (1.0 / 3.0)
    coefficient_2pn = 15293365.0 / 508032.0 + 27145.0 * eta / 504.0 + 3085.0 * eta**2 / 72.0
    phase = (
        (3.0 / (128.0 * eta))
        * v ** (-5.0)
        * (
            1.0
            + (3715.0 / 756.0 + 55.0 * eta / 9.0) * v**2
            - 16.0 * np.pi * v**3
            + coefficient_2pn * v**4
        )
    )
    return frequencies ** (-7.0 / 6.0) * np.exp(1j * phase_sign * phase)


def _phase_basis(frequencies: np.ndarray, family_id: str, reference: float) -> np.ndarray:
    if family_id == "PHASE_INVERSE_FREQUENCY":
        raw = reference / frequencies
    elif family_id == "PHASE_CUBIC_FREQUENCY":
        raw = (frequencies / reference) ** 3
    else:
        raise GW170817ResponseError(f"no phase basis: {family_id}")
    design = np.column_stack((np.ones(frequencies.size), frequencies / reference))
    weights = frequencies ** (-7.0 / 3.0)
    normal = design.T @ (weights[:, None] * design)
    rhs = design.T @ (weights * raw)
    projected = raw - design @ np.linalg.solve(normal, rhs)
    scale = float(np.max(np.abs(projected)))
    _require(scale > 0.0, "phase basis projection collapsed")
    return projected / scale


def _modifier(
    frequencies: np.ndarray, family_id: str, coefficient: float, reference: float
) -> np.ndarray:
    if family_id.startswith("PHASE_"):
        return np.exp(1j * coefficient * _phase_basis(frequencies, family_id, reference))
    if family_id == "ATTENUATION_LINEAR_FREQUENCY":
        return np.exp(-coefficient * (frequencies / reference - 1.0))
    if family_id == "GR":
        return np.ones(frequencies.size, dtype=complex)
    raise GW170817ResponseError(f"unknown response family: {family_id}")


def _profile_rho2(
    data: np.ndarray, template: np.ndarray, inverse_variance: np.ndarray | None = None
) -> float:
    weight = np.ones(data.size, dtype=float) if inverse_variance is None else inverse_variance
    numerator = np.sum(data * np.conj(template) * weight)
    denominator = float(np.sum(np.abs(template) ** 2 * weight))
    _require(denominator > 0.0 and math.isfinite(denominator), "invalid template norm")
    return float(abs(numerator) ** 2 / denominator)


def _family_map(config: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {row["id"]: row for row in config["response_likelihood"]["families"]}


def _fit_response_families(
    config: Mapping[str, Any],
    frequencies: np.ndarray,
    detector_data: Sequence[np.ndarray],
    base_template: np.ndarray,
    inverse_variances: Sequence[np.ndarray] | None = None,
) -> dict[str, Any]:
    if inverse_variances is None:
        inverse_variances = [np.ones(frequencies.size) for _ in detector_data]
    gr_rho2 = sum(
        _profile_rho2(data, base_template, weight)
        for data, weight in zip(detector_data, inverse_variances, strict=True)
    )
    reference = float(config["response_likelihood"]["reference_frequency_hz"])
    rows = []
    for family in config["response_likelihood"]["families"]:
        candidates = []
        for coefficient in _inclusive_grid(family["grid"]):
            template = base_template * _modifier(
                frequencies, family["id"], float(coefficient), reference
            )
            rho2 = sum(
                _profile_rho2(data, template, weight)
                for data, weight in zip(detector_data, inverse_variances, strict=True)
            )
            candidates.append((rho2, float(coefficient)))
        best_rho2, best_coefficient = max(candidates, key=lambda row: (row[0], -abs(row[1])))
        rows.append(
            {
                "family_id": family["id"],
                "best_coefficient": best_coefficient,
                "best_network_rho2": best_rho2,
                "delta_2_log_likelihood": best_rho2 - gr_rho2,
                "grid_points": len(candidates),
            }
        )
    return {"gr_network_rho2": gr_rho2, "families": rows}


def target_free_controls(config: Mapping[str, Any]) -> dict[str, Any]:
    controls = config["target_free_controls"]
    frequencies = _inclusive_grid(controls["frequency_grid_hz"])
    chirp_mass = 1.1975
    eta = 0.245
    base = _taylorf2(frequencies, chirp_mass, eta, 1.0)
    detectors = int(controls["detectors"])
    norm = float(np.sum(np.abs(base) ** 2))
    amplitude = float(controls["network_snr"]) / math.sqrt(detectors * norm)
    reference = float(config["response_likelihood"]["reference_frequency_hz"])
    zero_noise_rows = []
    for injection in controls["zero_noise_injections"]:
        truth_modifier = _modifier(
            frequencies, injection["family"], float(injection["coefficient"]), reference
        )
        data = [amplitude * base * truth_modifier for _ in range(detectors)]
        fit = _fit_response_families(config, frequencies, data, base)
        if injection["family"] == "GR":
            recovered = {row["family_id"]: row["best_coefficient"] for row in fit["families"]}
            passed = all(value == 0.0 for value in recovered.values())
        else:
            row = next(item for item in fit["families"] if item["family_id"] == injection["family"])
            recovered = row["best_coefficient"]
            passed = math.isclose(
                recovered,
                float(injection["coefficient"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        zero_noise_rows.append(
            {
                "injection_id": injection["id"],
                "family": injection["family"],
                "truth": injection["coefficient"],
                "recovered": recovered,
                "passed": passed,
            }
        )
    _require(all(row["passed"] for row in zero_noise_rows), "zero-noise control failed")
    null = controls["null_noise"]
    null_amplitude = float(null["network_snr"]) / math.sqrt(detectors * norm)
    maxima: dict[str, list[float]] = {
        row["id"]: [] for row in config["response_likelihood"]["families"]
    }
    for offset in range(int(null["realizations"])):
        rng = np.random.default_rng(int(null["seed_start"]) + offset)
        data = []
        for _ in range(detectors):
            noise = (
                rng.normal(size=frequencies.size) + 1j * rng.normal(size=frequencies.size)
            ) / math.sqrt(2.0)
            data.append(null_amplitude * base + noise)
        fit = _fit_response_families(config, frequencies, data, base)
        for row in fit["families"]:
            maxima[row["family_id"]].append(max(0.0, row["delta_2_log_likelihood"]))
    thresholds = {family: max(values) for family, values in maxima.items()}
    _require(
        all(math.isfinite(value) and value >= 0.0 for value in thresholds.values()),
        "null threshold invalid",
    )
    replay = {
        "zero_noise": zero_noise_rows,
        "null_realizations": int(null["realizations"]),
        "null_seed_start": int(null["seed_start"]),
        "family_delta_2_log_likelihood_thresholds": thresholds,
    }
    replay["control_sha256"] = _sha256_bytes(_canonical(replay))
    return replay


def _package_hashes(base: Path) -> dict[str, str]:
    return {
        "config_raw_sha256": _sha256_file(base / CONFIG_PATH),
        "module_raw_sha256": _sha256_file(base / MODULE_PATH),
        "test_raw_sha256": _sha256_file(base / TEST_PATH),
    }


def build_prediction_receipt(config: Mapping[str, Any], base: Path) -> dict[str, Any]:
    receipt: dict[str, Any] = {
        "schema_version": PREDICTION_SCHEMA,
        "analysis_id": config["analysis_id"],
        "status": "FROZEN_BEFORE_OBSERVATIONAL_PAYLOAD_ACCESS",
        "content_sha256": "",
        "package_hashes": _package_hashes(base),
        "config_content_sha256": _sha256_bytes(_canonical(config)),
        "predecessors": config["predecessors"],
        "products": config["products"],
        "preprocessing_sha256": _sha256_bytes(_canonical(config["preprocessing"])),
        "gr_control_sha256": _sha256_bytes(_canonical(config["gr_control"])),
        "response_likelihood_sha256": _sha256_bytes(_canonical(config["response_likelihood"])),
        "target_free_controls": target_free_controls(config),
        "official_metadata_audit": config["official_metadata_audit"],
        "access_at_freeze": config["access_contract"],
        "claim_boundary": config["claim_boundary"],
    }
    receipt["content_sha256"] = _self_hash(receipt)
    return receipt


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def freeze(root: Path | None = None) -> str:
    base = (root or _repo_root()).resolve()
    config = load_config(base)
    expected = _canonical(build_prediction_receipt(config, base)) + b"\n"
    path = base / PREDICTION_PATH
    if path.exists():
        _require(path.read_bytes() == expected, "prediction receipt differs")
        return "EXISTING_IDENTICAL"
    _atomic_write(path, expected)
    return "CREATED"


def validate_prediction(config: Mapping[str, Any], base: Path) -> dict[str, Any]:
    path = base / PREDICTION_PATH
    _require(path.is_file(), "prediction receipt missing")
    observed = _read_json(path)
    _require(observed.get("content_sha256") == _self_hash(observed), "prediction self-hash invalid")
    _require(observed == build_prediction_receipt(config, base), "prediction receipt changed")
    return observed


def _decode_strings(dataset: h5py.Dataset) -> list[str]:
    values = dataset[()]
    return [
        value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value)
        for value in np.atleast_1d(values)
    ]


def _json_scalar(value: Any) -> Any:
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_json_scalar(item) for item in value.tolist()]
    return value


def _hdf_header_and_dq(path: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    contract = config["hdf5_contract"]
    with h5py.File(path, "r") as handle:
        for dataset_path in (
            contract["strain_dataset"],
            contract["dq_dataset"],
            contract["dq_shortnames_dataset"],
        ):
            _require(dataset_path in handle, f"missing HDF5 dataset: {dataset_path}")
        strain = handle[contract["strain_dataset"]]
        dq = handle[contract["dq_dataset"]]
        names = _decode_strings(handle[contract["dq_shortnames_dataset"]])
        _require(strain.shape == (contract["sample_count"],), "strain sample count changed")
        _require(dq.shape == strain.shape, "DQ sample count changed")
        _require(strain.dtype == np.dtype("float64"), "strain dtype changed")
        _require(np.issubdtype(dq.dtype, np.integer), "DQ dtype changed")
        for required in contract["required_dq_flags"]:
            _require(required in names, f"required DQ flag absent: {required}")
        unique, counts = np.unique(dq[()], return_counts=True)
        required_mask = sum(
            1 << names.index(required) for required in contract["required_dq_flags"]
        )
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
            "strain_attributes": {
                key: _json_scalar(value) for key, value in sorted(strain.attrs.items())
            },
            "dq_shape": list(dq.shape),
            "dq_dtype": str(dq.dtype),
            "dq_shortnames": names,
            "dq_unique_counts": {
                str(int(value)): int(count) for value, count in zip(unique, counts, strict=True)
            },
            "required_dq_mask": required_mask,
            "required_dq_pass_samples": required_pass,
            "required_dq_pass_fraction": required_pass / int(dq.shape[0]),
        }


def _download(product: Mapping[str, Any], path: Path) -> dict[str, Any]:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        sha = hashlib.sha256()
        md5 = hashlib.md5(usedforsecurity=False)
        size = 0
        try:
            with os.fdopen(handle, "wb") as output:
                with urllib.request.urlopen(str(product["url"]), timeout=120) as response:
                    while True:
                        chunk = response.read(8 * 1024 * 1024)
                        if not chunk:
                            break
                        output.write(chunk)
                        sha.update(chunk)
                        md5.update(chunk)
                        size += len(chunk)
                output.flush()
                os.fsync(output.fileno())
            _require(
                size == product["expected_bytes"], f"download size changed: {product['detector']}"
            )
            _require(
                md5.hexdigest() == product["published_md5"],
                f"published MD5 failed: {product['detector']}",
            )
            os.replace(temporary_name, path)
        except BaseException:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
            raise
    _require(
        path.stat().st_size == product["expected_bytes"],
        f"local size changed: {product['detector']}",
    )
    md5_value = _md5_file(path)
    _require(md5_value == product["published_md5"], f"local MD5 failed: {product['detector']}")
    return {
        "detector": product["detector"],
        "filename": product["filename"],
        "url": product["url"],
        "local_path": product["local_path"],
        "bytes": path.stat().st_size,
        "published_md5": product["published_md5"],
        "observed_md5": md5_value,
        "sha256": _sha256_file(path),
    }


def acquire(root: Path | None = None) -> str:
    base = (root or _repo_root()).resolve()
    config = load_config(base)
    prediction = validate_prediction(config, base)
    rows = []
    for product in config["products"]:
        row = _download(product, base / product["local_path"])
        row["hdf5_and_dq"] = _hdf_header_and_dq(base / product["local_path"], config)
        rows.append(row)
    receipt: dict[str, Any] = {
        "schema_version": ACQUISITION_SCHEMA,
        "analysis_id": config["analysis_id"],
        "status": "EXACT_THREE_C00_PRODUCTS_ACQUIRED_HASHED_AND_STRUCTURALLY_VALIDATED",
        "content_sha256": "",
        "prediction_receipt_raw_sha256": _sha256_file(base / PREDICTION_PATH),
        "prediction_receipt_content_sha256": prediction["content_sha256"],
        "products": rows,
        "counts": {
            "files": 3,
            "payload_bytes": sum(row["bytes"] for row in rows),
            "strain_values_read": 0,
            "dq_values_read": 3 * int(config["hdf5_contract"]["sample_count"]),
        },
    }
    receipt["content_sha256"] = _self_hash(receipt)
    payload = _canonical(receipt) + b"\n"
    path = base / ACQUISITION_PATH
    if path.exists():
        _require(path.read_bytes() == payload, "acquisition receipt differs")
        return "EXISTING_IDENTICAL"
    _atomic_write(path, payload)
    return "CREATED"


def validate_acquisition(config: Mapping[str, Any], base: Path) -> dict[str, Any]:
    prediction = validate_prediction(config, base)
    path = base / ACQUISITION_PATH
    _require(path.is_file(), "acquisition receipt missing")
    receipt = _read_json(path)
    _require(receipt.get("content_sha256") == _self_hash(receipt), "acquisition self-hash invalid")
    _require(
        receipt.get("prediction_receipt_content_sha256") == prediction["content_sha256"],
        "prediction/acquisition binding changed",
    )
    rows = receipt.get("products")
    _require(isinstance(rows, list) and len(rows) == 3, "acquisition products changed")
    for expected, observed in zip(config["products"], rows, strict=True):
        path_value = base / expected["local_path"]
        _require(path_value.is_file(), f"acquired payload missing: {expected['detector']}")
        _require(path_value.stat().st_size == observed["bytes"], "acquired payload size changed")
        _require(_sha256_file(path_value) == observed["sha256"], "acquired payload SHA changed")
        _require(_md5_file(path_value) == expected["published_md5"], "acquired payload MD5 changed")
    return receipt


def _glitch_gate(times: np.ndarray, specification: Mapping[str, Any]) -> np.ndarray:
    center = float(specification["center_gps"])
    half_width = float(specification["zero_half_width_seconds"])
    rolloff = float(specification["rolloff_seconds_each_side"])
    distance = np.abs(times - center)
    result = np.ones(times.size, dtype=float)
    result[distance <= half_width] = 0.0
    transition = (distance > half_width) & (distance < half_width + rolloff)
    x_value = (distance[transition] - half_width) / rolloff
    result[transition] = 0.5 * (1.0 - np.cos(np.pi * x_value))
    return result


def _slice(
    dataset: h5py.Dataset, gps_start: int, target_start: int, duration: int, rate: int
) -> np.ndarray:
    start = (target_start - gps_start) * rate
    stop = start + duration * rate
    _require(0 <= start < stop <= dataset.shape[0], "requested HDF5 slice outside payload")
    return np.asarray(dataset[start:stop])


def _required_dq_mask(handle: h5py.File, config: Mapping[str, Any]) -> int:
    names = _decode_strings(handle[config["hdf5_contract"]["dq_shortnames_dataset"]])
    return sum(1 << names.index(name) for name in config["hdf5_contract"]["required_dq_flags"])


def _read_and_preprocess_detector(
    product: Mapping[str, Any], config: Mapping[str, Any], base: Path
) -> dict[str, Any]:
    preprocessing = config["preprocessing"]
    contract = config["hdf5_contract"]
    rate = int(contract["sample_rate_hz"])
    gps_start = int(contract["gps_start"])
    with h5py.File(base / product["local_path"], "r") as handle:
        strain_dataset = handle[contract["strain_dataset"]]
        dq_dataset = handle[contract["dq_dataset"]]
        required_mask = _required_dq_mask(handle, config)
        analysis = _slice(
            strain_dataset,
            gps_start,
            int(preprocessing["analysis_gps_start"]),
            int(preprocessing["analysis_duration_seconds"]),
            rate,
        ).astype(float)
        analysis_dq = _slice(
            dq_dataset,
            gps_start,
            int(preprocessing["analysis_gps_start"]),
            int(preprocessing["analysis_duration_seconds"]),
            rate,
        )
        _require(
            np.all((analysis_dq & required_mask) == required_mask),
            f"analysis DQ failed: {product['detector']}",
        )
        psd_arrays = []
        psd_dq_fractions = []
        for interval in preprocessing["psd_intervals_gps"]:
            values = _slice(
                strain_dataset,
                gps_start,
                int(interval[0]),
                int(preprocessing["psd_interval_seconds"]),
                rate,
            ).astype(float)
            dq_values = _slice(
                dq_dataset,
                gps_start,
                int(interval[0]),
                int(preprocessing["psd_interval_seconds"]),
                rate,
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
        glitch_window = _glitch_gate(times, preprocessing["l1_glitch_gate"])
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
        "psd_required_dq_fractions": psd_dq_fractions,
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


def _gr_recovery(
    config: Mapping[str, Any], processed: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    gr = config["gr_control"]
    masses = _inclusive_grid(gr["chirp_mass_solar_grid"])
    etas = [float(value) for value in gr["symmetric_mass_ratio_grid"]]
    offsets = _inclusive_grid(gr["coalescence_offset_seconds_grid"])
    base_time = float(config["official_metadata"]["event_gps"]) - float(
        config["preprocessing"]["analysis_gps_start"]
    )
    detector_matrices = []
    for row in processed:
        frequencies = row["frequencies"]
        phase_matrix = np.exp(2j * np.pi * np.outer(offsets, frequencies))
        detector_matrices.append(phase_matrix)
    candidates = []
    for phase_sign in gr["fourier_phase_signs"]:
        for chirp_mass in masses:
            for eta in etas:
                detector_rows = []
                network_rho2 = 0.0
                for row, phase_matrix in zip(processed, detector_matrices, strict=True):
                    frequencies = row["frequencies"]
                    intrinsic = _taylorf2(frequencies, float(chirp_mass), eta, float(phase_sign))
                    base_template = intrinsic * np.exp(-2j * np.pi * frequencies * base_time)
                    inverse_variance = 4.0 * float(row["delta_f"]) / row["psd"]
                    correlations = phase_matrix @ (
                        row["spectrum"] * np.conj(base_template) * inverse_variance
                    )
                    norm = float(np.sum(np.abs(base_template) ** 2 * inverse_variance))
                    rho2_values = np.abs(correlations) ** 2 / norm
                    best_index = int(np.argmax(rho2_values))
                    rho2 = float(rho2_values[best_index])
                    network_rho2 += rho2
                    detector_rows.append(
                        {
                            "detector": row["detector"],
                            "rho2": rho2,
                            "snr": math.sqrt(max(0.0, rho2)),
                            "coalescence_offset_seconds": float(offsets[best_index]),
                            "coalescence_gps": float(config["official_metadata"]["event_gps"])
                            + float(offsets[best_index]),
                        }
                    )
                candidates.append(
                    {
                        "chirp_mass_solar": float(chirp_mass),
                        "symmetric_mass_ratio": eta,
                        "fourier_phase_sign": int(phase_sign),
                        "network_rho2": network_rho2,
                        "detectors": detector_rows,
                    }
                )
    best = max(candidates, key=lambda row: row["network_rho2"])
    detector_map = {row["detector"]: row for row in best["detectors"]}
    thresholds = gr["pass_thresholds"]
    checks = {
        "network_snr": math.sqrt(best["network_rho2"]) >= thresholds["network_snr_min"],
        "h1_snr": detector_map["H1"]["snr"] >= thresholds["h1_snr_min"],
        "l1_snr": detector_map["L1"]["snr"] >= thresholds["l1_snr_min"],
        "chirp_mass": abs(best["chirp_mass_solar"] - gr["published_control_chirp_mass_solar"])
        <= thresholds["chirp_mass_abs_error_max"],
        "h1_l1_time": abs(
            detector_map["H1"]["coalescence_gps"] - detector_map["L1"]["coalescence_gps"]
        )
        <= thresholds["h1_l1_time_difference_abs_max_seconds"],
    }
    return {
        "waveform_boundary": gr["waveform"],
        "candidate_count": len(candidates),
        "best": best,
        "network_snr": math.sqrt(best["network_rho2"]),
        "checks": checks,
        "passed": all(checks.values()),
    }


def _real_response_fit(
    config: Mapping[str, Any],
    processed: Sequence[Mapping[str, Any]],
    gr_recovery: Mapping[str, Any],
    prediction: Mapping[str, Any],
) -> dict[str, Any]:
    best = gr_recovery["best"]
    detector_best = {row["detector"]: row for row in best["detectors"]}
    reference = float(config["response_likelihood"]["reference_frequency_hz"])
    gr_network_rho2 = 0.0
    family_network: dict[str, list[tuple[float, float]]] = {
        family["id"]: [] for family in config["response_likelihood"]["families"]
    }
    for row in processed:
        frequencies = row["frequencies"]
        intrinsic = _taylorf2(
            frequencies,
            float(best["chirp_mass_solar"]),
            float(best["symmetric_mass_ratio"]),
            float(best["fourier_phase_sign"]),
        )
        relative_time = (
            float(config["official_metadata"]["event_gps"])
            - float(config["preprocessing"]["analysis_gps_start"])
            + float(detector_best[row["detector"]]["coalescence_offset_seconds"])
        )
        base_template = intrinsic * np.exp(-2j * np.pi * frequencies * relative_time)
        weight = 4.0 * float(row["delta_f"]) / row["psd"]
        gr_network_rho2 += _profile_rho2(row["spectrum"], base_template, weight)
        for family in config["response_likelihood"]["families"]:
            for coefficient in _inclusive_grid(family["grid"]):
                modifier = _modifier(frequencies, family["id"], float(coefficient), reference)
                rho2 = _profile_rho2(row["spectrum"], base_template * modifier, weight)
                family_network[family["id"]].append((float(coefficient), rho2))
    rows = []
    thresholds = prediction["target_free_controls"]["family_delta_2_log_likelihood_thresholds"]
    detector_count = len(processed)
    for family in config["response_likelihood"]["families"]:
        grouped: dict[float, float] = {}
        for coefficient, rho2 in family_network[family["id"]]:
            grouped[coefficient] = grouped.get(coefficient, 0.0) + rho2
        _require(
            all(
                sum(1 for coefficient, _ in family_network[family["id"]] if coefficient == key)
                == detector_count
                for key in grouped
            ),
            "response grid detector coverage failed",
        )
        best_coefficient, best_rho2 = max(
            grouped.items(), key=lambda item: (item[1], -abs(item[0]))
        )
        delta = best_rho2 - gr_network_rho2
        threshold = float(thresholds[family["id"]])
        rows.append(
            {
                "family_id": family["id"],
                "best_coefficient": best_coefficient,
                "best_network_rho2": best_rho2,
                "gr_network_rho2": gr_network_rho2,
                "delta_2_log_likelihood": delta,
                "target_free_null_max_threshold": threshold,
                "exceeds_target_free_threshold": delta > threshold,
                "interpretation_status": (
                    "EXPLORATORY_SHAPE_EXCESS_GR_CONTROL_PASSED"
                    if gr_recovery["passed"] and delta > threshold
                    else (
                        "NO_FROZEN_SHAPE_EXCESS_GR_CONTROL_PASSED"
                        if gr_recovery["passed"]
                        else "INVALID_GR_CONTROL"
                    )
                ),
            }
        )
    return {
        "gr_network_rho2_fixed_recovery": gr_network_rho2,
        "families": rows,
        "interpretation_boundary": config["response_likelihood"]["interpretation_boundary"],
        "any_empirical_support_claim": False,
    }


def real_analysis(config: Mapping[str, Any], base: Path) -> dict[str, Any]:
    acquisition = validate_acquisition(config, base)
    prediction = validate_prediction(config, base)
    processed = [
        _read_and_preprocess_detector(product, config, base) for product in config["products"]
    ]
    gr = _gr_recovery(config, processed)
    response = _real_response_fit(config, processed, gr, prediction)
    summaries = [
        {key: value for key, value in row.items() if key not in {"frequencies", "spectrum", "psd"}}
        for row in processed
    ]
    return {
        "source_receipt_content_sha256": acquisition["content_sha256"],
        "prediction_receipt_content_sha256": prediction["content_sha256"],
        "processed_detectors": summaries,
        "gr_recovery": gr,
        "response_likelihood": response,
        "access": {
            "payload_files_opened": 3,
            "strain_samples_read": sum(row["strain_samples_read"] for row in summaries),
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
    config: Mapping[str, Any], base: Path
) -> tuple[dict[str, bytes], dict[str, Any]]:
    prediction = validate_prediction(config, base)
    acquisition = validate_acquisition(config, base)
    analysis = real_analysis(config, base)
    theorem = {
        "schema_version": "invariant-open-gravity-theorem-binding-2.0",
        "predecessor": config["predecessors"]["theorem"],
        "theorem_modified": False,
        "real_data_does_not_reprove_or_modify_static_no_enhancement_theorem": True,
    }
    source = {
        "schema_version": "invariant-open-gravity-gw170817-source-result-2.0",
        "official_metadata": config["official_metadata"],
        "metadata_audit": config["official_metadata_audit"],
        "acquisition": acquisition,
    }
    real = {
        "schema_version": "invariant-open-gravity-gw170817-real-data-result-2.0",
        "analysis": analysis,
        "claim_boundary": config["claim_boundary"],
    }
    family_rows = [
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
    report = f"""# GW170817 differential-propagation response run v2

## Theorem result

The predecessor's static theorem is unchanged: changing propagation speed alone does not enhance the stationary massless force.  This real-data run neither reproves nor modifies that result.

## Source result

Exactly three preregistered C00 4096-second, 4096-Hz HDF5 products were acquired.  Published MD5, computed SHA-256, byte count, HDF5 structure, and DQ receipts pass for H1, L1, and V1.  No cleaned or later-release product was substituted.

## Real-data result

The frozen approximate TaylorF2 GR recovery control status is `{analysis["gr_recovery"]["passed"]}` with network SNR `{analysis["gr_recovery"]["network_snr"]}` and recovered chirp mass `{analysis["gr_recovery"]["best"]["chirp_mass_solar"]}` solar masses.  Response-family rows retain their raw profiled likelihood improvements and target-free thresholds.  Their interpretation is invalid if the GR control failed.

This is a one-event response-shape diagnostic using an approximate 2PN control, not reproduction of the published tidal parameter estimation and not a posterior on c_g, graviton mass, Gamma, zeta, distance, cosmology, or a fundamental gravity theory.  No empirical-support or publication-ready claim is made.
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
            family_rows,
        ),
        "target-free-controls.json": _canonical(prediction["target_free_controls"]) + b"\n",
        "report.md": report.encode("utf-8"),
    }
    return payloads, analysis


def build_receipt(config: Mapping[str, Any], base: Path) -> tuple[dict[str, Any], dict[str, bytes]]:
    payloads, analysis = artifact_payloads(config, base)
    gr_pass = bool(analysis["gr_recovery"]["passed"])
    decision = f"{DECISION_PREFIX}__{'GR_CONTROL_PASS' if gr_pass else 'GR_CONTROL_FAIL'}__NO_EMPIRICAL_SUPPORT_CLAIM"
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "analysis_id": config["analysis_id"],
        "decision": decision,
        "content_sha256": "",
        "package_hashes": _package_hashes(base),
        "config_content_sha256": _sha256_bytes(_canonical(config)),
        "prediction_receipt_raw_sha256": _sha256_file(base / PREDICTION_PATH),
        "acquisition_receipt_raw_sha256": _sha256_file(base / ACQUISITION_PATH),
        "artifact_sha256": {
            name: _sha256_bytes(payload) for name, payload in sorted(payloads.items())
        },
        "theorem_result": "PRESERVED_UNCHANGED",
        "source_result": "EXACT_THREE_C00_PRODUCTS_ACQUIRED_AND_VALIDATED",
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
    receipt, payloads = build_receipt(config, base)
    targets = {base / ARTIFACT_DIR / name: payload for name, payload in payloads.items()}
    targets[base / OUTPUT_PATH] = _canonical(receipt) + b"\n"
    existing = [path for path in targets if path.exists()]
    if existing:
        _require(len(existing) == len(targets), "partial output package exists")
        for path, payload in targets.items():
            _require(path.read_bytes() == payload, f"existing output differs: {path}")
        return "EXISTING_IDENTICAL"
    for path, payload in targets.items():
        _atomic_write(path, payload)
    return "CREATED"


def check(root: Path | None = None) -> str:
    base = (root or _repo_root()).resolve()
    config = load_config(base)
    expected, payloads = build_receipt(config, base)
    observed = _read_json(base / OUTPUT_PATH)
    _require(observed.get("content_sha256") == _self_hash(observed), "receipt self-hash invalid")
    _require(observed == expected, "receipt differs from deterministic rebuild")
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
