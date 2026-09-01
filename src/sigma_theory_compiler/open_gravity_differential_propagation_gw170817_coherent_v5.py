"""Append-only pre-response repair and synthetic gates for coherent GW170817.

This package inherits every scientific choice from the retained blocked v4
draft, replaces only the preprocessing window, validates source schemas and DQ
without reading strain samples, and then runs target-free injections.  It does
not authorize or implement real-data scoring.
"""

from __future__ import annotations

import argparse
import copy
import io
import json
import math
import tarfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from sigma_theory_compiler import (
    open_gravity_differential_propagation_gw170817_coherent_v4 as v4,
)

CONFIG_PATH = Path("configs/open_gravity_differential_propagation_gw170817_coherent_v5.json")
MODULE_PATH = Path(
    "src/sigma_theory_compiler/open_gravity_differential_propagation_gw170817_coherent_v5.py"
)
TEST_PATH = Path("tests/test_open_gravity_differential_propagation_gw170817_coherent_v5.py")
RUN_DIR = Path("runs/gravity/open-gravity-differential-propagation-gw170817-coherent-v5")
PREDICTION_PATH = RUN_DIR / "prediction-receipt.json"
ARTIFACT_DIR = RUN_DIR / "artifacts"

CONFIG_SCHEMA = "invariant-open-gravity-differential-propagation-gw170817-coherent-config-5.0"
PREDICTION_SCHEMA = (
    "invariant-open-gravity-differential-propagation-gw170817-coherent-prediction-receipt-5.0"
)


class CoherentV5Error(RuntimeError):
    """Fail-closed v5 pre-response error."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CoherentV5Error(message)


def _base(root: Path | None = None) -> Path:
    return (root or Path.cwd()).resolve()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(v4._canonical(value) + b"\n")


def load_config(root: Path | None = None) -> dict[str, Any]:
    config = _read_json(_base(root) / CONFIG_PATH)
    validate_config(config)
    return config


def validate_config(config: Mapping[str, Any]) -> None:
    _require(config.get("schema_version") == CONFIG_SCHEMA, "wrong schema")
    _require(
        config.get("analysis_id") == "open-gravity-differential-propagation-gw170817-coherent-v5",
        "wrong analysis id",
    )
    _require(
        config.get("status") == "FROZEN_APPEND_ONLY_PRE_RESPONSE_WINDOW_AND_SCHEMA_REPAIR",
        "v5 is not frozen",
    )
    predecessor = config["blocked_predecessor_v4"]
    _require(
        predecessor["decision"] == "BLOCKED_PRE_RESPONSE_END_TAPER_AUDIT_NO_INJECTIONS_NO_STRAIN",
        "v4 blocker not inherited",
    )
    inheritance = config["science_inheritance"]
    _require(not inheritance["post_freeze_scientific_retuning_allowed"], "retuning allowed")
    preprocessing = config["corrected_preprocessing"]
    _require(preprocessing["analysis_duration_seconds"] == 256, "duration changed")
    _require(preprocessing["frequency_stride_from_256_second_fft"] == 32, "stride changed")
    _require(preprocessing["likelihood_delta_f_hz"] == 0.125, "delta f changed")
    _require(preprocessing["frequency_band_hz"] == [23.0, 2047.875], "band changed")
    _require(preprocessing["nyquist_rule"].startswith("exclude"), "Nyquist not excluded")
    _require(config["pre_freeze_dq_audit"]["strain_values_read"] == 0, "strain leak")
    freeze = config["freeze_boundary"]
    _require(freeze["v5_strain_values_read_before_freeze"] == 0, "strain leak")
    _require(freeze["gw190425_status"] == "SEALED_NOT_ACQUIRED_NOT_OPENED", "holdout opened")
    _require(config["execution_gate"]["independent_audit_required"], "audit disabled")
    _require(
        not config["execution_gate"]["real_response_allowed_by_v5_prediction_alone"],
        "v5 incorrectly authorizes response",
    )


def _validate_predecessor(config: Mapping[str, Any], base: Path) -> dict[str, Any]:
    predecessor = config["blocked_predecessor_v4"]
    path_fields = {
        "config": ("config_path", "config_raw_sha256"),
        "module": ("module_path", "module_raw_sha256"),
        "test": ("test_path", "test_raw_sha256"),
        "prediction": ("prediction_path", "prediction_raw_sha256"),
    }
    observed: dict[str, str] = {}
    for label, (path_key, hash_key) in path_fields.items():
        path = base / predecessor[path_key]
        _require(path.is_file(), f"missing v4 {label}")
        digest = v4._sha256_file(path)
        _require(digest == predecessor[hash_key], f"v4 {label} drift")
        observed[label] = digest
    receipt = _read_json(base / predecessor["prediction_path"])
    _require(
        receipt["content_sha256"] == predecessor["prediction_content_sha256"],
        "v4 receipt content drift",
    )
    _require(receipt["decision"] == predecessor["decision"], "v4 decision drift")
    _require(v4.check(base) == predecessor["decision"], "v4 replay failed")
    return {"status": "PASS_BLOCKED_V4_REPLAYED_EXACT", "hashes": observed}


def compose_science_config(config: Mapping[str, Any], base: Path) -> dict[str, Any]:
    blocked = _read_json(base / config["blocked_predecessor_v4"]["config_path"])
    composed = copy.deepcopy(blocked)
    composed["analysis_id"] = config["analysis_id"]
    composed["status"] = config["status"]
    composed["preprocessing"] = copy.deepcopy(config["corrected_preprocessing"])
    composed["package"] = copy.deepcopy(config["package"])
    composed["freeze_boundary"] = copy.deepcopy(config["freeze_boundary"])
    composed.pop("pre_response_blocker", None)
    return composed


def _inheritance_audit(config: Mapping[str, Any], base: Path) -> dict[str, Any]:
    blocked = _read_json(base / config["blocked_predecessor_v4"]["config_path"])
    composed = compose_science_config(config, base)
    hashes: dict[str, str] = {}
    for section in config["science_inheritance"]["unchanged_sections"]:
        before = v4._sha256_bytes(v4._canonical(blocked[section]))
        after = v4._sha256_bytes(v4._canonical(composed[section]))
        _require(before == after, f"science inheritance drift: {section}")
        hashes[section] = before
    return {
        "status": "PASS_ONLY_PREPROCESSING_CHANGED",
        "unchanged_section_sha256": hashes,
        "corrected_preprocessing_sha256": v4._sha256_bytes(
            v4._canonical(composed["preprocessing"])
        ),
    }


def _source_schema_audit(
    config: Mapping[str, Any], science: Mapping[str, Any], base: Path
) -> dict[str, Any]:
    psd_contract = config["source_schema_contract"]["published_psd"]
    psd_path = base / science["sources"]["published_psd"]["path"]
    with psd_path.open("rb") as handle:
        psd_header = handle.readline().decode("ascii").rstrip("\r\n")
    _require(psd_header == psd_contract["first_line_exact"], "PSD header/schema drift")
    psd = np.loadtxt(psd_path)
    _require(psd.ndim == 2 and psd.shape[1] == 4, "PSD numeric schema drift")
    _require(np.all(np.isfinite(psd)), "PSD nonfinite")
    _require(np.all(np.diff(psd[:, 0]) > 0.0), "PSD frequencies not increasing")
    _require(np.all(psd[:, 1:] > 0.0), "PSD values must be positive power/Hz")
    low, high = science["preprocessing"]["frequency_band_hz"]
    _require(psd[0, 0] <= low and psd[-1, 0] >= high, "PSD does not cover band")

    calibration_contract = config["source_schema_contract"]["calibration_envelopes"]
    calibration_path = base / science["sources"]["calibration_envelopes"]["path"]
    calibration_rows: dict[str, Any] = {}
    with tarfile.open(calibration_path, "r:gz") as archive:
        for detector, member_name in science["sources"]["calibration_envelopes"]["members"].items():
            member = archive.extractfile(member_name)
            _require(member is not None, f"missing calibration member {detector}")
            payload = member.read()
            header = payload.splitlines()[0].decode("ascii")
            _require(
                header == calibration_contract["first_line_exact"],
                f"calibration header drift {detector}",
            )
            table = np.loadtxt(io.BytesIO(payload))
            _require(table.ndim == 2 and table.shape[1] == 7, f"calibration columns {detector}")
            _require(np.all(np.isfinite(table)), f"calibration nonfinite {detector}")
            _require(np.all(np.diff(table[:, 0]) > 0.0), f"calibration frequency order {detector}")
            _require(np.all(table[:, [1, 3, 5]] > 0.0), f"calibration magnitude {detector}")
            _require(
                table[0, 0] <= low and table[-1, 0] >= high, f"calibration coverage {detector}"
            )
            calibration_rows[detector] = {
                "rows": int(table.shape[0]),
                "frequency_min_hz": float(table[0, 0]),
                "frequency_max_hz": float(table[-1, 0]),
                "header": header,
            }
    return {
        "status": "PASS_PSD_IS_POWER_PER_HZ_AND_CALIBRATION_PHASE_IS_RADIANS",
        "published_psd": {
            "rows": int(psd.shape[0]),
            "frequency_min_hz": float(psd[0, 0]),
            "frequency_max_hz": float(psd[-1, 0]),
            "header": psd_header,
            "value_semantics": psd_contract["values_are"],
        },
        "calibration": calibration_rows,
    }


def _normalize_attr(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _hdf_and_dq_audit(
    config: Mapping[str, Any], science: Mapping[str, Any], base: Path
) -> dict[str, Any]:
    contract = config["hdf5_contract"]
    prep = science["preprocessing"]
    intervals = {
        "analysis_256": [prep["analysis_gps_start"], prep["analysis_gps_end_exclusive"]],
        "psd_early": prep["psd_robustness"][0]["gps"],
        "psd_late": prep["psd_robustness"][1]["gps"],
    }
    rows: list[dict[str, Any]] = []
    dq_values_read = 0
    for product in science["sources"]["strain"]:
        with h5py.File(base / product["path"], "r") as handle:
            strain = handle[contract["strain_dataset"]]
            _require(list(strain.shape) == contract["strain_shape"], "strain shape drift")
            _require(str(strain.dtype) == contract["strain_dtype"], "strain dtype drift")
            for name, expected in contract["strain_attributes"].items():
                _require(
                    _normalize_attr(strain.attrs[name]) == expected, f"strain attr drift {name}"
                )
            dq = handle[contract["dq_dataset"]]
            _require(list(dq.shape) == contract["dq_shape"], "DQ shape drift")
            _require(str(dq.dtype) == contract["dq_dtype"], "DQ dtype drift")
            for name, expected in contract["dq_attributes"].items():
                _require(_normalize_attr(dq.attrs[name]) == expected, f"DQ attr drift {name}")
            names_dataset = handle["/quality/simple/DQShortnames"][:]
            names = [_normalize_attr(value) for value in names_dataset]
            _require(names == contract["dq_shortnames_exact"], "DQ names drift")
            start = int(dq.attrs["Xstart"])
            coverage: dict[str, list[int]] = {}
            for interval_name, (interval_start, interval_end) in intervals.items():
                values = dq[int(interval_start) - start : int(interval_end) - start]
                dq_values_read += len(values)
                passing = int(
                    np.count_nonzero(
                        (values & contract["required_bitmask"]) == contract["required_bitmask"]
                    )
                )
                coverage[interval_name] = [passing, len(values)]
                _require(passing == len(values), f"DQ failed {product['detector']} {interval_name}")
            expected_coverage = config["pre_freeze_dq_audit"]["per_detector"][product["detector"]]
            _require(coverage == expected_coverage, f"DQ audit drift {product['detector']}")
            rows.append({"detector": product["detector"], "coverage": coverage})
    _require(
        dq_values_read == config["pre_freeze_dq_audit"]["dq_values_read"], "DQ access count drift"
    )
    return {
        "status": "PASS_EXACT_HDF_HEADERS_AND_DQ_WITHOUT_STRAIN_VALUES",
        "detectors": rows,
        "hdf5_files_opened": 3,
        "dq_values_read": dq_values_read,
        "strain_values_read": 0,
    }


def _support_and_nyquist_audit(science: Mapping[str, Any]) -> dict[str, Any]:
    prep = science["preprocessing"]
    flat_start, flat_end = prep["flat_interior_gps"]
    signal_start = prep["estimated_23hz_signal_start_gps"]
    coalescence = prep["coalescence_gps"]
    support_checks = {
        "signal_start_after_flat_start": signal_start >= flat_start,
        "coalescence_before_flat_end": coalescence <= flat_end,
        "post_coalescence_flat_margin": flat_end - coalescence >= 16.0,
    }
    base_delta = 1.0 / prep["analysis_duration_seconds"]
    likelihood_delta = base_delta * prep["frequency_stride_from_256_second_fft"]
    low, high = prep["frequency_band_hz"]
    bins = round((high - low) / likelihood_delta) + 1
    last = low + (bins - 1) * likelihood_delta
    nyquist_checks = {
        "base_delta_f": math.isclose(base_delta, prep["base_fft_delta_f_hz"], abs_tol=0.0),
        "likelihood_delta_f": math.isclose(
            likelihood_delta, prep["likelihood_delta_f_hz"], abs_tol=0.0
        ),
        "last_frequency_exact": math.isclose(last, high, abs_tol=1.0e-12),
        "nyquist_excluded": last < prep["nyquist_hz"],
        "generator_excludes_nyquist": prep["generator_f_max_hz"] == high,
    }
    _require(all(support_checks.values()), "waveform support/taper gate failed")
    _require(all(nyquist_checks.values()), "Nyquist/frequency gate failed")
    return {
        "status": "PASS_SIGNAL_IN_FLAT_INTERIOR_AND_NYQUIST_EXCLUDED",
        "support_checks": support_checks,
        "nyquist_checks": nyquist_checks,
        "frequency_bins_per_detector": bins,
        "last_frequency_hz": last,
        "flat_start_margin_seconds": signal_start - flat_start,
        "flat_end_margin_seconds": flat_end - coalescence,
    }


def _package_hashes(config: Mapping[str, Any], base: Path) -> dict[str, str]:
    return {
        "config_raw_sha256": v4._sha256_file(base / CONFIG_PATH),
        "module_raw_sha256": v4._sha256_file(base / MODULE_PATH),
        "test_raw_sha256": v4._sha256_file(base / TEST_PATH),
        "blocked_v4_config_raw_sha256": config["blocked_predecessor_v4"]["config_raw_sha256"],
        "blocked_v4_prediction_raw_sha256": config["blocked_predecessor_v4"][
            "prediction_raw_sha256"
        ],
    }


def freeze(root: Path | None = None) -> str:
    base = _base(root)
    config = load_config(base)
    predecessor = _validate_predecessor(config, base)
    inheritance = _inheritance_audit(config, base)
    science = compose_science_config(config, base)
    sources = v4._source_audit(science, base)
    schemas = _source_schema_audit(config, science, base)
    hdf = _hdf_and_dq_audit(config, science, base)
    support = _support_and_nyquist_audit(science)
    runtime = v4._runtime_audit(science)
    controls = v4._target_free_controls(science, base)
    artifacts = {
        "predecessor-and-inheritance-audit.json": {
            "predecessor": predecessor,
            "inheritance": inheritance,
        },
        "source-schema-hdf-dq-support-audit.json": {
            "sources": sources,
            "schemas": schemas,
            "hdf_and_dq": hdf,
            "support_and_nyquist": support,
        },
        "runtime-audit.json": runtime,
        "target-free-injection-gates.json": controls,
    }
    for name, value in artifacts.items():
        _write_json(base / ARTIFACT_DIR / name, value)
    decision = (
        "FROZEN_PRE_RESPONSE_GATES_PASS_PENDING_INDEPENDENT_AUDIT"
        if controls["status"] == "PASS"
        else "FROZEN_PRE_RESPONSE_IDENTIFIABILITY_FAIL_DIAGNOSTIC_ONLY_PENDING_AUDIT"
    )
    receipt: dict[str, Any] = {
        "schema_version": PREDICTION_SCHEMA,
        "analysis_id": config["analysis_id"],
        "decision": decision,
        "package_hashes": _package_hashes(config, base),
        "artifact_sha256": {
            name: v4._sha256_file(base / ARTIFACT_DIR / name) for name in artifacts
        },
        "source_status": sources["status"],
        "schema_status": schemas["status"],
        "hdf_dq_status": hdf["status"],
        "support_nyquist_status": support["status"],
        "runtime_status": runtime["status"],
        "target_free_status": controls["status"],
        "independent_audit_required": True,
        "real_response_authorized": False,
        "access_ledger": {
            "source_files_hashed": len(sources["files"]),
            "hdf5_headers_opened": hdf["hdf5_files_opened"],
            "dq_values_read": hdf["dq_values_read"],
            "strain_values_read": 0,
            "real_likelihood_values_computed": 0,
            "gw190425_opened": 0,
            "model_calls": 0,
            "paid_calls": 0,
        },
    }
    receipt["content_sha256"] = v4._self_hash(receipt)
    _write_json(base / PREDICTION_PATH, receipt)
    return decision


def check(root: Path | None = None) -> str:
    base = _base(root)
    config = load_config(base)
    receipt = _read_json(base / PREDICTION_PATH)
    _require(receipt["schema_version"] == PREDICTION_SCHEMA, "receipt schema drift")
    _require(receipt["content_sha256"] == v4._self_hash(receipt), "receipt content drift")
    _require(receipt["package_hashes"] == _package_hashes(config, base), "package drift")
    _require(receipt["access_ledger"]["strain_values_read"] == 0, "strain access recorded")
    _require(receipt["access_ledger"]["gw190425_opened"] == 0, "holdout opened")
    _require(not receipt["real_response_authorized"], "audit bypass")
    for name, digest in receipt["artifact_sha256"].items():
        _require(v4._sha256_file(base / ARTIFACT_DIR / name) == digest, f"artifact drift {name}")
    return receipt["decision"]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("freeze")
    subparsers.add_parser("check")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    print(freeze() if arguments.command == "freeze" else check())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
