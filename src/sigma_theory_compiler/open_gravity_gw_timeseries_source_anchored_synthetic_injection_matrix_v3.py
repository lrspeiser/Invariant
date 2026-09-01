"""Semantics-repaired response-blind GW/time-series synthetic matrix v3."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from sigma_theory_compiler import (
    open_gravity_differential_propagation_gw170817_coherent_v4 as lane2,
)
from sigma_theory_compiler import open_gravity_dynamic_source_memory_kernels_v1 as lane5
from sigma_theory_compiler.open_gravity_data_element_ontology_v1 import (
    Availability,
    DataElement,
    DataRole,
    ExperimentRole,
    UncertaintyKind,
    catalogue_from_elements,
)
from sigma_theory_compiler.open_gravity_formula_adapter_registry_v1 import (
    AdapterRegistration,
    validate_adapter_registry,
)
from sigma_theory_compiler.open_gravity_formula_execution_protocol_v1 import (
    BindingStatus,
    FormulaExecutionBinding,
    ResourceBounds,
)
from sigma_theory_compiler.open_gravity_observation_operators_v1 import SeedLineage
from sigma_theory_compiler.open_gravity_synthetic_discovery_runner_v1 import (
    ObservableComparison,
    ParameterCell,
    ScenarioRuntimeValues,
)
from sigma_theory_compiler.open_gravity_synthetic_discovery_runner_v2 import (
    run_discovery_matrix_v2,
)
from sigma_theory_compiler.open_gravity_synthetic_replay_ledger_v1 import (
    SyntheticSuiteRelease,
)
from sigma_theory_compiler.open_gravity_synthetic_scenario_packet_v1 import (
    AnchorBinding,
    AxisSpec,
    EmittedPredictionSpec,
    FeatureValueRef,
    ScenarioDescriptor,
    UncertaintyRef,
    array_sha256,
    validate_scenario_values,
)
from sigma_theory_compiler.sigma_core import SchemaViolation, canonical_sha256

CONFIG_PATH = Path(
    "configs/open_gravity_gw_timeseries_source_anchored_synthetic_injection_matrix_v3.json"
)
PARAMETER_SCHEMA_PATH = Path(
    "configs/open_gravity_gw_timeseries_source_anchored_synthetic_injection_matrix_v3.parameters.schema.json"
)
TEST_PATH = Path(
    "tests/test_open_gravity_gw_timeseries_source_anchored_synthetic_injection_matrix_v3.py"
)
OUTPUT_DIR = Path(
    "runs/gravity/open-gravity-gw-timeseries-source-anchored-synthetic-injection-matrix-v3"
)
VALUES_PATH = OUTPUT_DIR / "values.npz"
SCENARIOS_PATH = OUTPUT_DIR / "scenarios.jsonl"
MATRIX_PATH = OUTPUT_DIR / "matrix-result.json"
LEDGER_PATH = OUTPUT_DIR / "ledger.json"
CONFUSION_PATH = OUTPUT_DIR / "confusion-matrix.json"
DIAGNOSTICS_PATH = OUTPUT_DIR / "invariance-degeneracy-and-failure-ledger.json"
RECEIPT_PATH = OUTPUT_DIR / "receipt.json"
_ROOT = Path(__file__).resolve().parents[2]

_EXPECTED_CONFIG_CONTENT_SHA256 = "e71eb6e3a7d1bdf99fb592acadf45b23b918643d00f053767cfc89c59c57f3ac"
_EXPECTED_CONFIG_RAW_SHA256 = "ccfdd28dbab743d17fef7e85444acf4b03c66f66bebd762690bd4406ea60a693"

_FEATURE_SPECS: dict[str, tuple[tuple[str, ...], str, int]] = {
    "source.matrix.base-frequency-imag": (("detector", "frequency"), "1", 0),
    "source.matrix.base-frequency-real": (("detector", "frequency"), "1", 0),
    "source.matrix.base-time-strain": (("detector", "time"), "1", 0),
    "source.matrix.calibration-imag": (("detector", "frequency"), "1", 0),
    "source.matrix.calibration-real": (("detector", "frequency"), "1", 0),
    "source.matrix.psd-sigma": (("detector", "frequency"), "1", 0),
    "source.scalar.chirp-mass-solar": (("object",), "solar mass", 0),
    "source.scalar.distance-mpc": (("object",), "Mpc", 0),
    "source.scalar.event-code": (("object",), "integer code", 0),
    "source.scalar.lane5-time-scale-seconds": (("object",), "s", 0),
    "source.scalar.source-variant-code": (("object",), "integer code", 0),
    "source.vector.active-detector-mask": (("detector",), "integer code", 0),
    "source.vector.antenna-weight": (("detector",), "1", 0),
    "source.vector.conditional-variance-sigma": (("detector",), "1", 0),
    "source.vector.detector-delay-seconds": (("detector",), "s", 0),
    "source.vector.frequency-hz": (("frequency",), "Hz", 0),
    "source.vector.time-noise-sigma": (("detector",), "1", 0),
    "source.vector.time-seconds": (("time",), "s", 0),
}
_FEATURES = tuple(sorted(_FEATURE_SPECS))
_OUTPUTS = (
    "prediction.matrix.conditional-variance",
    "prediction.matrix.frequency-imag",
    "prediction.matrix.frequency-real",
    "prediction.matrix.time-strain",
)
_SCORING_OUTPUTS = (
    "prediction.matrix.conditional-variance",
    "prediction.matrix.frequency-imag",
    "prediction.matrix.frequency-real",
)
_OUTPUT_AXES = {
    "prediction.matrix.conditional-variance": ("detector", "time"),
    "prediction.matrix.frequency-imag": ("detector", "frequency"),
    "prediction.matrix.frequency-real": ("detector", "frequency"),
    "prediction.matrix.time-strain": ("detector", "time"),
}
_RESPONSE_FOR = {
    value: value.replace("prediction.", "response.synthetic-", 1) for value in _OUTPUTS
}

_ENTRYPOINTS = {
    "CONTROL_FREE_DELAY": "control_free_delay_adapter",
    "CONTROL_OU_NOISE": "control_ou_noise_adapter",
    "CONTROL_SINGLE_LTI": "control_single_lti_adapter",
    "CONTROL_SOURCE_RINGDOWN": "control_source_ringdown_adapter",
    "CONTROL_TWO_POLE_LTI": "control_two_pole_lti_adapter",
    "GR_NETWORK_CONTROL": "gr_network_control_adapter",
    "LANE2_ATTENUATION": "lane2_attenuation_adapter",
    "LANE2_DYNAMIC_PHASE": "lane2_dynamic_phase_adapter",
    "LANE2_NONLINEAR_PHASE": "lane2_nonlinear_phase_adapter",
    "LANE2_RESERVOIR": "lane2_reservoir_adapter",
    "LANE2_SCREENED_PHASE": "lane2_screened_phase_adapter",
    "LANE5_BIEXPONENTIAL_MEMORY": "lane5_biexponential_memory_adapter",
    "LANE5_DELAY_MEMORY": "lane5_delay_memory_adapter",
    "LANE5_EXPONENTIAL_MEMORY": "lane5_exponential_memory_adapter",
    "LANE5_HYSTERETIC_MEMORY": "lane5_hysteretic_memory_adapter",
    "LANE5_RESONANCE_MEMORY": "lane5_resonance_memory_adapter",
    "LANE5_STOCHASTIC_OU_MEMORY": "lane5_stochastic_ou_memory_adapter",
}
_BLOCK_STATUS = {
    "LANE2_EXACT_LALSUITE_WAVEFORM": BindingStatus.UNADAPTED,
    "NON_GAUSSIAN_JUMP_MEMORY": BindingStatus.UNADAPTED,
    "POLARIZATION_BIREFRINGENCE": BindingStatus.SOURCE_BLOCKED,
    "REAL_CALIBRATION_LEARNING": BindingStatus.SOURCE_BLOCKED,
    "REAL_OFFSOURCE_PSD_REFIT": BindingStatus.SOURCE_BLOCKED,
    "REAL_STRAIN_LIKELIHOOD": BindingStatus.SOURCE_BLOCKED,
}
_KERNEL_PARAMETERS: dict[str, dict[str, float]] = {
    "K01_RETARDED": {"delay": 0.244140625},
    "K02_EXPONENTIAL": {"tau": 0.35},
    "K03_BIEXPONENTIAL": {"tau1": 0.08, "tau2": 1.5, "weight": 0.65},
    "K04_DAMPED_RESONANCE": {"omega0": 7.0, "zeta": 0.18},
    "K05_HYSTERETIC": {"threshold": 0.25, "offset": 0.15, "tau": 0.03},
    "K06_STOCHASTIC_OU": {"tau": 0.35, "sigma": 0.05},
}
_RINGDOWN = {"onset": 6.0, "amplitude": 0.65, "decay": 1.2, "omega": 7.0}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SchemaViolation(message)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: Any, *, indent: int | None = None) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":") if indent is None else None,
            indent=indent,
            allow_nan=False,
        )
        + ("\n" if indent is not None else "")
    ).encode("utf-8")


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _repo_path(value: str | Path) -> Path:
    parsed = PurePosixPath(str(value).replace("\\", "/"))
    if parsed.is_absolute() or any(part in {"", ".", ".."} for part in parsed.parts):
        raise SchemaViolation("GW synthetic path escaped repository")
    result = (_ROOT / parsed.as_posix()).resolve()
    if not result.is_relative_to(_ROOT):
        raise SchemaViolation("GW synthetic path escaped repository")
    return result


def load_config() -> dict[str, Any]:
    return json.loads((_ROOT / CONFIG_PATH).read_text(encoding="utf-8"))


def validate_config(config: Mapping[str, Any], *, verify_hashes: bool = True) -> None:
    if verify_hashes:
        _require(
            _file_sha256(_ROOT / CONFIG_PATH) == _EXPECTED_CONFIG_RAW_SHA256,
            "GW synthetic frozen config bytes changed",
        )
    _require(
        _json_sha256(config) == _EXPECTED_CONFIG_CONTENT_SHA256,
        "GW synthetic frozen config changed",
    )
    _require(
        config["status"] == "FROZEN_SYNTHETIC_ONLY_SEMANTICS_REPAIR_PRE_AUDIT"
        and config["claim_class"] == "SYNTHETIC_DIRECTIONAL_SIGNAL",
        "GW synthetic boundary changed",
    )
    _require(config["truth_mechanisms"] == sorted(_ENTRYPOINTS), "truth inventory changed")
    _require(
        [row["formula_id"] for row in config["adapter_blocks"]] == sorted(_BLOCK_STATUS),
        "blocked inventory changed",
    )
    _require(
        _repo_path(config["output_directory"]) == (_ROOT / OUTPUT_DIR).resolve(),
        "GW output path changed",
    )
    access = config["access_contract"]
    _require(
        access["source_anchor_files_opened"] == 19
        and access["source_anchor_bytes_opened"] == 1_022_661
        and access["infrastructure_files_hashed"] == 15
        and access["infrastructure_bytes_hashed"] == 132_771,
        "GW source accounting changed",
    )
    _require(
        config["representation_contract"]
        == {
            "canonical_domain": "positive-frequency-rfft-bins-excluding-dc",
            "dc_rule": "DC_IS_EXACT_REAL_ZERO_AND_NOT_STORED",
            "nyquist_rule": "NYQUIST_IMAGINARY_IS_EXACT_ZERO",
            "time_domain_rule": "DERIVE_ONLY_BY_NUMPY_IRFFT_FROM_CANONICAL_FREQUENCY",
            "stored_frequency_rule": "RECOMPUTE_BY_NUMPY_RFFT_FROM_DERIVED_REAL_TIME",
            "noise_rule": "DRAW_ONCE_IN_CANONICAL_FREQUENCY_AND_TRANSFORM_TO_TIME",
            "metric_rule": "SCORE_CANONICAL_FREQUENCY_PLUS_NONDUPLICATE_CONDITIONAL_VARIANCE_ONLY",
            "numpy_version": "2.2.6",
            "consistency_absolute_tolerance": 1.0e-12,
        }
        and tuple(config["scoring"]["scored_outputs"]) == _SCORING_OUTPUTS
        and config["scoring"]["derived_unscored_outputs"] == ["prediction.matrix.time-strain"],
        "GW canonical representation contract changed",
    )
    _require(
        all(
            access[key] == 0
            for key in (
                "strain_files_opened",
                "strain_samples_opened",
                "real_likelihood_responses_opened",
                "real_likelihood_values_computed",
                "psd_payload_arrays_opened",
                "calibration_payload_archives_opened",
                "theory_or_nuisance_tuning_events",
                "network_calls",
                "model_calls",
                "paid_calls",
            )
        ),
        "GW response seal changed",
    )
    for group in (config["source_anchors"], config["infrastructure_bindings"]):
        ids = [row["id"] for row in group]
        _require(ids == sorted(set(ids)), "GW bindings must be sorted unique")
        for row in group:
            path = _repo_path(row["path"])
            if verify_hashes:
                _require(
                    path.is_file() and _file_sha256(path) == row["sha256"],
                    f"GW binding drift: {row['id']}",
                )
                if "bytes" in row:
                    _require(path.stat().st_size == row["bytes"], f"GW size drift: {row['id']}")


def _anchor_map(config: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(row["id"]): row for row in config["source_anchors"]}


def _source_inventory(config: Mapping[str, Any]) -> dict[str, Any]:
    anchors = _anchor_map(config)

    def read_json(anchor_id: str) -> dict[str, Any]:
        return json.loads(_repo_path(anchors[anchor_id]["path"]).read_text(encoding="utf-8"))

    lane2_preflight = read_json("LANE2_SOURCE_PREFLIGHT_RECEIPT")
    lane2_prediction = read_json("LANE2_COHERENT_V6_PREDICTION")
    lane2_gates = read_json("LANE2_COHERENT_V6_TARGET_FREE_GATES")
    lane5_receipt = read_json("LANE5_V2_RECEIPT")
    lane5_gate = read_json("LANE5_V2_SOURCE_GATE")
    lane5_drivers = read_json("LANE5_DRIVER_EXECUTIONS")
    runner_audit_row = next(
        row for row in config["infrastructure_bindings"] if row["id"] == "GENERIC_RUNNER_V2_AUDIT"
    )
    runner_audit = json.loads(_repo_path(runner_audit_row["path"]).read_text(encoding="utf-8"))
    blocked_audit_row = next(
        row
        for row in config["infrastructure_bindings"]
        if row["id"] == "GW_V2_INDEPENDENT_AUDIT_BLOCK"
    )
    blocked_audit = json.loads(_repo_path(blocked_audit_row["path"]).read_text(encoding="utf-8"))
    v2_receipt_row = next(
        row for row in config["infrastructure_bindings"] if row["id"] == "GW_V2_RECEIPT"
    )
    v2_receipt = json.loads(_repo_path(v2_receipt_row["path"]).read_text(encoding="utf-8"))
    for name, payload, convention in (
        ("Lane2 source receipt", lane2_preflight, "omit"),
        ("Lane2 prediction receipt", lane2_prediction, "omit"),
        ("Lane5 receipt", lane5_receipt, "blank"),
        ("generic runner audit", runner_audit, "omit"),
    ):
        body = dict(payload)
        if convention == "blank":
            body["content_sha256"] = ""
        else:
            body.pop("content_sha256")
        _require(
            payload["content_sha256"] == _json_sha256(body),
            f"{name} self hash changed",
        )
    _require(
        lane2_preflight["decision"] == "PASS_SOURCE_METADATA_ONLY__BLOCK_PAYLOAD_ACCESS_AND_SCORING"
        and lane2_preflight["counts"]["payload_rows_opened"] == 0,
        "Lane2 source preflight boundary changed",
    )
    _require(
        lane2_prediction["real_response_authorized"] is False
        and lane2_prediction["method_passed"] is False,
        "Lane2 retained method failure changed",
    )
    _require(
        lane2_gates["strain_values_read"] == 0 and lane2_gates["status"] == "FAIL_METHOD_GATES",
        "Lane2 target-free gate boundary changed",
    )
    _require(
        lane5_receipt["decision"]
        == "PASS_EXECUTABLE_DIMENSIONED_DRIVER_PIPELINES_STRUCTURAL_TRIAGE_SOURCE_BLOCKED_NO_RESPONSE_ACCESS"
        and lane5_receipt["counts"]["observational_response_rows"] == 0,
        "Lane5 source boundary changed",
    )
    _require(
        lane5_gate["access_contract"]["observational_response_rows_read"] == 0
        and lane5_gate["preflight"]["status"]
        == "SOURCE_BLOCKED_MISSING_PAYLOAD_HASHES_AND_SCHEMA_RECEIPTS",
        "Lane5 response gate changed",
    )
    _require(
        runner_audit["status"] == "PASS"
        and runner_audit["decision"]
        == "GENERIC_RUNNER_V2_CODE_CONTRACT_ADMISSIBLE_FOR_RESPONSE_BLIND_SYNTHETIC_DISCOVERY",
        "generic runner v2 audit changed",
    )
    _require(
        blocked_audit["decision"] == "BLOCK"
        and blocked_audit["blocking_finding"]["code"]
        == "B01_FALSE_FFT_ROUNDTRIP_PASS_AND_INCONSISTENT_DUAL_DOMAIN_OBSERVATIONS"
        and blocked_audit["required_repair_before_pass"]
        == [
            "Enforce the real-signal Hermitian constraints at both DC and Nyquist after every phase transfer and calibration operation, or construct a valid two-sided spectrum and derive both stored domains from it.",
            "Generate one stochastic observation per detector and derive its other domain by FFT, or use an explicitly registered joint time-frequency covariance without treating duplicate representations as independent evidence.",
            "Replace the one-way inverse check with forward and inverse round-trip checks on every candidate prediction and every synthetic response, including all noise families and the calibration envelope.",
            "Regenerate and rescore the scientific artifacts under a semantics-changing successor version. This cannot be represented as a test-expectation-only correction.",
        ],
        "GW v2 blocking audit changed",
    )
    v2_receipt_body = dict(v2_receipt)
    v2_receipt_body.pop("content_sha256")
    _require(
        v2_receipt["content_sha256"] == _json_sha256(v2_receipt_body)
        and v2_receipt["status"]
        == "FROZEN_SYNTHETIC_ONLY_TEST_CORRECTION_COMPLETE_AWAITING_DISTINCT_AUDIT",
        "GW v2 predecessor receipt changed",
    )
    _require(
        len(lane5_drivers["drivers"]) == len(lane5_drivers["executions"]) == 20,
        "Lane5 target-free driver count changed",
    )
    waveform_path = _repo_path(anchors["LANE5_NR_WAVEFORM"]["path"])
    waveform = np.loadtxt(waveform_path, dtype=np.float64)
    _require(
        waveform.shape == (2769, 2)
        and np.all(np.diff(waveform[:, 0]) > 0.0)
        and np.all(np.isfinite(waveform)),
        "target-free GW150914 NR source changed",
    )
    return {
        "lane2_preflight": lane2_preflight,
        "lane2_prediction": lane2_prediction,
        "lane2_gates": lane2_gates,
        "lane5_receipt": lane5_receipt,
        "lane5_gate": lane5_gate,
        "runner_audit": runner_audit,
        "blocked_audit": blocked_audit,
        "v2_receipt": v2_receipt,
        "waveform": waveform,
    }


def _catalogue(config: Mapping[str, Any]):
    provenance = canonical_sha256({row["id"]: row["sha256"] for row in config["source_anchors"]})
    dimensions = {
        "1": (0, 0, 0, 0, 0, 0, 0),
        "integer code": (0, 0, 0, 0, 0, 0, 0),
        "Hz": (0, 0, -1, 0, 0, 0, 0),
        "s": (0, 0, 1, 0, 0, 0, 0),
        "Mpc": (0, 1, 0, 0, 0, 0, 0),
        "solar mass": (1, 0, 0, 0, 0, 0, 0),
    }
    specs = [
        (element_id, axes, unit, rank) for element_id, (axes, unit, rank) in _FEATURE_SPECS.items()
    ]
    for output in _OUTPUTS:
        specs.append((output, _OUTPUT_AXES[output], "1", 0))
        specs.append((_RESPONSE_FOR[output], _OUTPUT_AXES[output], "1", 0))
    specs.append(("truth.scalar.injection-id", ("object",), "integer code", 0))
    elements = []
    for element_id, axes, unit, rank in specs:
        if element_id.startswith("response."):
            role, availability = DataRole.SCORING_ONLY_RESPONSE, Availability.SYNTHETIC_ONLY
        elif element_id.startswith("prediction."):
            role, availability = DataRole.DERIVED, Availability.SYNTHETIC_ONLY
        elif element_id.startswith("truth."):
            role, availability = DataRole.LATENT_SYNTHETIC_TRUTH, Availability.SYNTHETIC_ONLY
        else:
            role, availability = DataRole.FORMULA_INPUT, Availability.PUBLIC_SOURCE
        elements.append(
            DataElement(
                element_id=element_id,
                namespace=element_id.rsplit(".", 1)[0],
                physical_quantity=element_id,
                tensor_rank=rank,
                si_dimension=dimensions[unit],
                canonical_unit=unit,
                frame="latent" if element_id.startswith("truth.") else config["coordinate_frame"],
                support="response-blind GW170817/GW150914 source-shaped synthetic network",
                axes=axes,
                component="total",
                derivation_parents=(),
                uncertainty=(
                    UncertaintyKind.COVARIANCE
                    if element_id.startswith(("prediction.", "response."))
                    else UncertaintyKind.NONE
                ),
                availability=availability,
                experiment_roles=(ExperimentRole(config["experiment_id"], role),),
                provenance_sha256=provenance,
            )
        )
    return catalogue_from_elements("open-gravity-gw-timeseries-source-anchored", "v3.0.0", elements)


def _bindings(config: Mapping[str, Any]) -> tuple[FormulaExecutionBinding, ...]:
    block_reason = {row["formula_id"]: row["reason"] for row in config["adapter_blocks"]}
    schema_sha = _file_sha256(_ROOT / PARAMETER_SCHEMA_PATH)
    upstream = {
        row["id"]: row["sha256"]
        for row in (*config["source_anchors"], *config["infrastructure_bindings"])
    }
    rows = []
    for formula_id in sorted((*_ENTRYPOINTS, *_BLOCK_STATUS)):
        executable = formula_id in _ENTRYPOINTS
        rows.append(
            FormulaExecutionBinding(
                binding_id=f"binding.gw-timeseries.{formula_id.lower().replace('_', '-')}.v3",
                formula_id=formula_id,
                formula_version="v3.0.0-canonical-frequency-synthetic",
                formula_sha256=canonical_sha256(
                    {
                        "formula_id": formula_id,
                        "lane2_module": upstream["LANE2_COHERENT_V4_MODULE"],
                        "lane5_module": upstream["LANE5_V1_MODULE"],
                        "blocked_reason": block_reason.get(formula_id),
                    }
                ),
                status=BindingStatus.EXECUTABLE if executable else _BLOCK_STATUS[formula_id],
                entrypoint=(
                    "sigma_theory_compiler."
                    "open_gravity_gw_timeseries_source_anchored_synthetic_injection_matrix_v3:"
                    f"{_ENTRYPOINTS[formula_id]}"
                    if executable
                    else None
                ),
                required_features=_FEATURES if executable else ("source.matrix.real-strain",),
                optional_features=(),
                emitted_features=_OUTPUTS,
                domains=(("gw-network",) if executable else ("gw-real-response",)),
                geometry_support=((config["geometry_mode"],) if executable else ("real-detector",)),
                time_support=((config["time_mode"],) if executable else ("empirical",)),
                parameter_schema_path=config["parameter_schema_path"],
                parameter_schema_sha256=schema_sha,
                approximation_ceiling=(
                    "source-shaped synthetic canonical-frequency network with time as an unscored derived view; no real strain, PSD payload, calibration archive, or likelihood"
                    if executable
                    else block_reason[formula_id]
                ),
                health_gates=(
                    "canonical-forward-backward-dc-nyquist-consistency",
                    "determinism",
                    "finite-output",
                    "limit",
                    "source-hash",
                    "typed-output",
                    "unit",
                ),
                resource_bounds=ResourceBounds(30, 500_000_000, 2_000_000),
            )
        )
    return tuple(rows)


def _base_complex(features: Mapping[str, Any]) -> np.ndarray:
    return np.asarray(
        features["source.matrix.base-frequency-real"], dtype=np.float64
    ) + 1j * np.asarray(features["source.matrix.base-frequency-imag"], dtype=np.float64)


def _calibration_complex(features: Mapping[str, Any]) -> np.ndarray:
    return np.asarray(
        features["source.matrix.calibration-real"], dtype=np.float64
    ) + 1j * np.asarray(features["source.matrix.calibration-imag"], dtype=np.float64)


def _canonical_pair_from_positive_frequency(
    frequency_positive: np.ndarray, sample_count: int
) -> tuple[np.ndarray, np.ndarray]:
    spectrum = np.array(frequency_positive, dtype=np.complex128, copy=True)
    _require(
        spectrum.ndim == 2 and spectrum.shape[1] == sample_count // 2,
        "canonical GW spectrum shape changed",
    )
    spectrum[:, -1] = spectrum[:, -1].real + 0.0j
    full = np.concatenate([np.zeros((spectrum.shape[0], 1), dtype=np.complex128), spectrum], axis=1)
    time = np.fft.irfft(full, n=sample_count, axis=1)
    canonical = np.fft.rfft(time, n=sample_count, axis=1)[:, 1:]
    canonical[:, -1] = canonical[:, -1].real + 0.0j
    return np.asarray(time, dtype=np.float64), np.asarray(canonical, dtype=np.complex128)


def _fft_pair_errors(
    time: np.ndarray, frequency_positive: np.ndarray, *, tolerance: float = 1.0e-12
) -> dict[str, float]:
    time_array = np.asarray(time, dtype=np.float64)
    spectrum = np.asarray(frequency_positive, dtype=np.complex128)
    _require(
        time_array.ndim == spectrum.ndim == 2
        and time_array.shape[0] == spectrum.shape[0]
        and spectrum.shape[1] == time_array.shape[1] // 2,
        "GW FFT pair shape changed",
    )
    forward_full = np.fft.rfft(time_array, axis=1)
    forward = forward_full[:, 1:]
    backward = np.fft.irfft(
        np.concatenate([np.zeros((spectrum.shape[0], 1), dtype=np.complex128), spectrum], axis=1),
        n=time_array.shape[1],
        axis=1,
    )
    errors = {
        "forward": float(np.max(np.abs(forward - spectrum))),
        "backward": float(np.max(np.abs(backward - time_array))),
        "dc_magnitude": float(np.max(np.abs(forward_full[:, 0]))),
        "nyquist_imaginary": float(np.max(np.abs(spectrum[:, -1].imag))),
    }
    _require(all(value <= tolerance for value in errors.values()), "incoherent GW FFT pair")
    return errors


def _finish_prediction(
    features: Mapping[str, Any],
    frequency_positive: np.ndarray,
    conditional_variance: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    transformed = np.asarray(frequency_positive, dtype=np.complex128) * _calibration_complex(
        features
    )
    sample_count = np.asarray(features["source.matrix.base-time-strain"]).shape[1]
    time, spectrum = _canonical_pair_from_positive_frequency(transformed, sample_count)
    _fft_pair_errors(time, spectrum)
    if conditional_variance is None:
        conditional_variance = np.zeros(time.shape, dtype=np.float64)
    result = {
        "prediction.matrix.conditional-variance": np.asarray(
            conditional_variance, dtype=np.float64
        ),
        "prediction.matrix.frequency-imag": np.asarray(spectrum.imag, dtype=np.float64),
        "prediction.matrix.frequency-real": np.asarray(spectrum.real, dtype=np.float64),
        "prediction.matrix.time-strain": np.asarray(time, dtype=np.float64),
    }
    _require(
        all(value.shape == _expected_output_shape(features, key) for key, value in result.items()),
        "GW prediction shape changed",
    )
    _require(
        all(np.all(np.isfinite(value)) for value in result.values()), "nonfinite GW prediction"
    )
    return result


def _expected_output_shape(features: Mapping[str, Any], output_id: str) -> tuple[int, ...]:
    base_time = np.asarray(features["source.matrix.base-time-strain"])
    base_frequency = np.asarray(features["source.matrix.base-frequency-real"])
    return base_frequency.shape if "frequency-" in output_id else base_time.shape


def _lane2_prediction(
    features: Mapping[str, Any], branch_id: str, parameters: Mapping[str, float]
) -> dict[str, np.ndarray]:
    _require(set(features) == set(_FEATURES), "GW feature projection changed")
    frequency = np.asarray(features["source.vector.frequency-hz"], dtype=np.float64)
    transfer = lane2.transfer_function(branch_id, frequency, parameters, 100.0)
    return _finish_prediction(features, _base_complex(features) * transfer[None, :])


def _memory_prediction(
    features: Mapping[str, Any], kernel_id: str, parameters: Mapping[str, float]
) -> dict[str, np.ndarray]:
    _require(set(features) == set(_FEATURES), "GW feature projection changed")
    time = np.asarray(features["source.vector.time-seconds"], dtype=np.float64)
    scale = float(np.asarray(features["source.scalar.lane5-time-scale-seconds"])[0])
    normalized_time = time / scale
    source = np.asarray(features["source.matrix.base-time-strain"], dtype=np.float64)
    response = np.vstack(
        [lane5.simulate_kernel(kernel_id, normalized_time, row, parameters) for row in source]
    )
    spectrum = np.fft.rfft(response, axis=1)[:, 1:]
    conditional = np.zeros(response.shape, dtype=np.float64)
    if kernel_id == "K06_STOCHASTIC_OU":
        sigma = float(parameters["sigma"])
        tau = float(parameters["tau"])
        mask = np.asarray(features["source.vector.active-detector-mask"], dtype=np.float64)
        rms2 = np.mean(source * source, axis=1)
        level = mask * sigma * sigma * tau * rms2 / 2.0
        conditional[:] = level[:, None]
    return _finish_prediction(features, spectrum, conditional)


def _source_ringdown_prediction(features: Mapping[str, Any]) -> dict[str, np.ndarray]:
    _require(set(features) == set(_FEATURES), "GW feature projection changed")
    time = np.asarray(features["source.vector.time-seconds"], dtype=np.float64)
    scale = float(np.asarray(features["source.scalar.lane5-time-scale-seconds"])[0])
    normalized = time / scale
    elapsed = normalized - _RINGDOWN["onset"]
    ring = np.where(
        elapsed >= 0.0,
        np.exp(-elapsed / _RINGDOWN["decay"]) * np.sin(_RINGDOWN["omega"] * elapsed),
        0.0,
    )
    source = np.asarray(features["source.matrix.base-time-strain"], dtype=np.float64)
    amplitude = np.max(np.abs(source), axis=1)
    response = source + _RINGDOWN["amplitude"] * amplitude[:, None] * ring[None, :]
    return _finish_prediction(features, np.fft.rfft(response, axis=1)[:, 1:])


def gr_network_control_adapter(
    features: Mapping[str, Any], parameters: Mapping[str, Any]
) -> Mapping[str, Any]:
    _require(not parameters, "GR control has no free parameter")
    return _lane2_prediction(features, "GR", {})


def lane2_dynamic_phase_adapter(
    features: Mapping[str, Any], parameters: Mapping[str, Any]
) -> Mapping[str, Any]:
    _require(set(parameters) == {"beta_milliradian"}, "dynamic phase cell changed")
    return _lane2_prediction(
        features, "DYNAMIC_PHASE", {"beta": int(parameters["beta_milliradian"]) / 1000.0}
    )


def lane2_attenuation_adapter(
    features: Mapping[str, Any], parameters: Mapping[str, Any]
) -> Mapping[str, Any]:
    _require(set(parameters) == {"alpha_milli"}, "attenuation cell changed")
    return _lane2_prediction(
        features, "ATTENUATION", {"alpha": int(parameters["alpha_milli"]) / 1000.0}
    )


def lane2_reservoir_adapter(
    features: Mapping[str, Any], parameters: Mapping[str, Any]
) -> Mapping[str, Any]:
    _require(
        set(parameters) == {"r_milli", "resonance_frequency_hz"},
        "reservoir cell changed",
    )
    return _lane2_prediction(
        features,
        "RESERVOIR",
        {
            "r": int(parameters["r_milli"]) / 1000.0,
            "log10_f_res_hz": math.log10(int(parameters["resonance_frequency_hz"])),
        },
    )


def lane2_nonlinear_phase_adapter(
    features: Mapping[str, Any], parameters: Mapping[str, Any]
) -> Mapping[str, Any]:
    _require(set(parameters) == {"gamma_milliradian"}, "nonlinear phase cell changed")
    return _lane2_prediction(
        features,
        "NONLINEAR_PHASE",
        {"gamma": int(parameters["gamma_milliradian"]) / 1000.0},
    )


def lane2_screened_phase_adapter(
    features: Mapping[str, Any], parameters: Mapping[str, Any]
) -> Mapping[str, Any]:
    _require(
        set(parameters) == {"screen_beta_milliradian", "screen_frequency_hz"},
        "screened phase cell changed",
    )
    return _lane2_prediction(
        features,
        "SCREENED_PHASE",
        {
            "beta_s": int(parameters["screen_beta_milliradian"]) / 1000.0,
            "log10_f_screen_hz": math.log10(int(parameters["screen_frequency_hz"])),
        },
    )


def lane5_delay_memory_adapter(
    features: Mapping[str, Any], parameters: Mapping[str, Any]
) -> Mapping[str, Any]:
    _require(not parameters, "delay memory cell changed")
    return _memory_prediction(features, "K01_RETARDED", _KERNEL_PARAMETERS["K01_RETARDED"])


def lane5_exponential_memory_adapter(
    features: Mapping[str, Any], parameters: Mapping[str, Any]
) -> Mapping[str, Any]:
    _require(not parameters, "exponential memory cell changed")
    return _memory_prediction(features, "K02_EXPONENTIAL", _KERNEL_PARAMETERS["K02_EXPONENTIAL"])


def lane5_biexponential_memory_adapter(
    features: Mapping[str, Any], parameters: Mapping[str, Any]
) -> Mapping[str, Any]:
    _require(not parameters, "bi-exponential memory cell changed")
    return _memory_prediction(
        features, "K03_BIEXPONENTIAL", _KERNEL_PARAMETERS["K03_BIEXPONENTIAL"]
    )


def lane5_resonance_memory_adapter(
    features: Mapping[str, Any], parameters: Mapping[str, Any]
) -> Mapping[str, Any]:
    _require(set(parameters) in (set(), {"zeta_milli"}), "resonance memory cell changed")
    values = dict(_KERNEL_PARAMETERS["K04_DAMPED_RESONANCE"])
    if "zeta_milli" in parameters:
        values["zeta"] = int(parameters["zeta_milli"]) / 1000.0
    return _memory_prediction(features, "K04_DAMPED_RESONANCE", values)


def lane5_hysteretic_memory_adapter(
    features: Mapping[str, Any], parameters: Mapping[str, Any]
) -> Mapping[str, Any]:
    _require(not parameters, "hysteretic memory cell changed")
    return _memory_prediction(features, "K05_HYSTERETIC", _KERNEL_PARAMETERS["K05_HYSTERETIC"])


def lane5_stochastic_ou_memory_adapter(
    features: Mapping[str, Any], parameters: Mapping[str, Any]
) -> Mapping[str, Any]:
    _require(not parameters, "stochastic OU memory cell changed")
    return _memory_prediction(
        features, "K06_STOCHASTIC_OU", _KERNEL_PARAMETERS["K06_STOCHASTIC_OU"]
    )


def control_free_delay_adapter(
    features: Mapping[str, Any], parameters: Mapping[str, Any]
) -> Mapping[str, Any]:
    return lane5_delay_memory_adapter(features, parameters)


def control_single_lti_adapter(
    features: Mapping[str, Any], parameters: Mapping[str, Any]
) -> Mapping[str, Any]:
    return lane5_exponential_memory_adapter(features, parameters)


def control_two_pole_lti_adapter(
    features: Mapping[str, Any], parameters: Mapping[str, Any]
) -> Mapping[str, Any]:
    return lane5_biexponential_memory_adapter(features, parameters)


def control_ou_noise_adapter(
    features: Mapping[str, Any], parameters: Mapping[str, Any]
) -> Mapping[str, Any]:
    return lane5_stochastic_ou_memory_adapter(features, parameters)


def control_source_ringdown_adapter(
    features: Mapping[str, Any], parameters: Mapping[str, Any]
) -> Mapping[str, Any]:
    _require(not parameters, "source ringdown cell changed")
    return _source_ringdown_prediction(features)


def _parameter_cells(
    bindings: Sequence[FormulaExecutionBinding], config: Mapping[str, Any]
) -> dict[str, tuple[ParameterCell, ...]]:
    values_by_formula: dict[str, tuple[ParameterCell, ...]] = {}
    for formula_id in sorted(_ENTRYPOINTS):
        values: Mapping[str, Any] = config["lane2_injection_parameters"].get(formula_id, {})
        cells = (ParameterCell(f"{formula_id.lower()}.fixed", values),)
        if formula_id == "LANE5_RESONANCE_MEMORY":
            cells = tuple(
                sorted(
                    (
                        ParameterCell(
                            "lane5-resonance.invalid-negative-zeta", {"zeta_milli": -180}
                        ),
                        ParameterCell("lane5-resonance.valid", {}),
                    ),
                    key=lambda row: row.parameter_cell_id,
                )
            )
        values_by_formula[formula_id] = cells
    return {
        binding.binding_id: (
            values_by_formula[binding.formula_id]
            if binding.status is BindingStatus.EXECUTABLE
            else ()
        )
        for binding in bindings
    }


def _truth_parameters(formula_id: str, config: Mapping[str, Any]) -> Mapping[str, Any]:
    return config["lane2_injection_parameters"].get(formula_id, {})


def _adapter_callable(formula_id: str):
    return globals()[_ENTRYPOINTS[formula_id]]


def _array_key(*parts: str) -> str:
    return "__".join(
        part.lower().replace("-", "_").replace(".", "_").replace(" ", "_").replace("+", "p")
        for part in parts
    )


def _intrinsic_signal(
    event: Mapping[str, Any], variant: Mapping[str, Any], time: np.ndarray, waveform: np.ndarray
) -> np.ndarray:
    stretch = float(variant["time_stretch"])
    if event["id"] == "GW150914":
        source_coordinate = (time - 0.195) / stretch
        signal = np.interp(
            source_coordinate,
            waveform[:, 0],
            waveform[:, 1],
            left=0.0,
            right=0.0,
        )
    else:
        nominal_mass = float(event["source_variants"][0]["chirp_mass_solar"])
        mass_ratio = float(variant["chirp_mass_solar"]) / nominal_mass
        warped = time / stretch
        merger = 0.215
        progress = np.clip(warped / merger, 0.0, 1.0)
        frequency = 24.0 + 330.0 * progress**3 * mass_ratio ** (5.0 / 3.0)
        phase = 2.0 * np.pi * np.cumsum(frequency) / 1024.0
        inspiral = (0.03 + progress) ** 2 * np.sin(phase)
        post = np.maximum(warped - merger, 0.0)
        ring = np.exp(-post / 0.018) * np.sin(
            phase[np.searchsorted(time, merger, side="left") - 1] + 2.0 * np.pi * 360.0 * post
        )
        signal = np.where(warped <= merger, inspiral, ring)
    maximum = float(np.max(np.abs(signal)))
    _require(math.isfinite(maximum) and maximum > 0.0, "invalid target-free intrinsic signal")
    return np.asarray(signal / maximum, dtype=np.float64)


def _source_slots(config: Mapping[str, Any], inventory: Mapping[str, Any]) -> list[dict[str, Any]]:
    samples = int(config["sample_grid"]["samples"])
    sample_rate = int(config["sample_grid"]["sample_rate_hz"])
    time = np.arange(samples, dtype=np.float64) / float(sample_rate)
    frequency = np.fft.rfftfreq(samples, d=1.0 / sample_rate)[1:]
    _require(frequency.size == config["sample_grid"]["positive_frequency_bins"], "frequency grid")
    noise_config = config["noise"]
    results = []
    for event in config["events"]:
        reference_distance = float(event["distance_mpc"][0])
        active = np.asarray(
            [int(detector in event["active_detectors"]) for detector in config["detectors"]],
            dtype=np.int64,
        )
        weights = np.asarray(event["antenna_weight"], dtype=np.float64)
        delays = np.asarray(event["detector_delay_seconds"], dtype=np.float64)
        for variant_code, variant in enumerate(event["source_variants"]):
            intrinsic = _intrinsic_signal(event, variant, time, inventory["waveform"])
            for distance in event["distance_mpc"]:
                scale = reference_distance / float(distance)
                network = np.vstack(
                    [
                        weights[index]
                        * scale
                        * np.interp(time - delays[index], time, intrinsic, left=0.0, right=0.0)
                        for index in range(len(config["detectors"]))
                    ]
                )
                network *= active[:, None]
                network -= np.mean(network, axis=1, keepdims=True)
                network *= active[:, None]
                base_frequency = np.fft.rfft(network, axis=1)[:, 1:]
                _fft_pair_errors(network, base_frequency)
                for family in config["noise_families"]:
                    calibration = np.ones(base_frequency.shape, dtype=np.complex128)
                    if family == "published-psd-calibration-envelope":
                        amplitude = np.asarray(
                            noise_config["calibration_amplitude"], dtype=np.float64
                        )
                        phase = np.asarray(
                            noise_config["calibration_phase_radians"], dtype=np.float64
                        )
                        calibration = np.exp(amplitude + 1j * phase)[:, None] * np.ones(
                            base_frequency.shape, dtype=np.complex128
                        )
                    family_scale = float(noise_config["family_sigma_scale"][family])
                    uncertainty_scale = max(family_scale, 0.25)
                    time_sigma_active = (
                        float(noise_config["base_fraction_of_reference_signal"]) * uncertainty_scale
                    )
                    shape = np.sqrt(
                        1.0
                        + (
                            float(noise_config["frequency_shape_corner_hz"])
                            / np.maximum(frequency, 20.0)
                        )
                        ** 4
                        + (frequency / 350.0) ** 2
                    )
                    frequency_sigma = np.vstack(
                        [
                            time_sigma_active * math.sqrt(samples / 2.0) * shape
                            if flag
                            else np.full(
                                frequency.shape,
                                math.sqrt(float(noise_config["inactive_detector_variance"])),
                            )
                            for flag in active
                        ]
                    )
                    time_sigma = np.asarray(
                        [
                            time_sigma_active
                            if flag
                            else math.sqrt(float(noise_config["inactive_detector_variance"]))
                            for flag in active
                        ],
                        dtype=np.float64,
                    )
                    conditional_sigma = np.asarray(
                        [
                            max(time_sigma_active * time_sigma_active, 1.0e-8) if flag else 1.0e6
                            for flag in active
                        ],
                        dtype=np.float64,
                    )
                    values = {
                        "source.matrix.base-frequency-imag": np.asarray(
                            base_frequency.imag, dtype=np.float64
                        ),
                        "source.matrix.base-frequency-real": np.asarray(
                            base_frequency.real, dtype=np.float64
                        ),
                        "source.matrix.base-time-strain": np.asarray(network, dtype=np.float64),
                        "source.matrix.calibration-imag": np.asarray(
                            calibration.imag, dtype=np.float64
                        ),
                        "source.matrix.calibration-real": np.asarray(
                            calibration.real, dtype=np.float64
                        ),
                        "source.matrix.psd-sigma": np.asarray(frequency_sigma, dtype=np.float64),
                        "source.scalar.chirp-mass-solar": np.asarray(
                            [float(variant["chirp_mass_solar"])], dtype=np.float64
                        ),
                        "source.scalar.distance-mpc": np.asarray(
                            [float(distance)], dtype=np.float64
                        ),
                        "source.scalar.event-code": np.asarray(
                            [int(event["event_code"])], dtype=np.int64
                        ),
                        "source.scalar.lane5-time-scale-seconds": np.asarray(
                            [float(config["sample_grid"]["lane5_time_scale_seconds"])],
                            dtype=np.float64,
                        ),
                        "source.scalar.source-variant-code": np.asarray(
                            [variant_code], dtype=np.int64
                        ),
                        "source.vector.active-detector-mask": active,
                        "source.vector.antenna-weight": weights,
                        "source.vector.conditional-variance-sigma": conditional_sigma,
                        "source.vector.detector-delay-seconds": delays,
                        "source.vector.frequency-hz": frequency,
                        "source.vector.time-noise-sigma": time_sigma,
                        "source.vector.time-seconds": time,
                    }
                    _require(set(values) == set(_FEATURES), "GW source feature set changed")
                    _require(
                        all(np.all(np.isfinite(value)) for value in values.values()),
                        "nonfinite GW source feature",
                    )
                    slot_id = _array_key(
                        event["id"],
                        variant["id"],
                        f"distance-{distance:g}",
                        family,
                    )
                    results.append(
                        {
                            "slot_id": slot_id,
                            "event_id": event["id"],
                            "variant_id": variant["id"],
                            "variant_code": variant_code,
                            "distance_mpc": float(distance),
                            "noise_family": family,
                            "noise_scale": family_scale,
                            "values": values,
                        }
                    )
    _require(len(results) == 40, "GW source population slot count changed")
    return sorted(results, key=lambda row: row["slot_id"])


def _noise_response(
    truth: Mapping[str, np.ndarray],
    slot: Mapping[str, Any],
    lineage: SeedLineage,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    values = slot["values"]
    rng = np.random.default_rng(lineage.derived_seed)
    active = np.asarray(values["source.vector.active-detector-mask"], dtype=np.float64)
    frequency_sigma = np.asarray(values["source.matrix.psd-sigma"], dtype=np.float64)
    conditional_sigma = np.asarray(
        values["source.vector.conditional-variance-sigma"], dtype=np.float64
    )
    family = str(slot["noise_family"])
    truth_frequency = np.asarray(
        truth["prediction.matrix.frequency-real"], dtype=np.float64
    ) + 1j * np.asarray(truth["prediction.matrix.frequency-imag"], dtype=np.float64)
    if family == "zero-noise":
        frequency_noise = np.zeros(truth_frequency.shape, dtype=np.complex128)
        response_frequency = np.array(truth_frequency, copy=True)
        response_time = np.array(
            truth["prediction.matrix.time-strain"], dtype=np.float64, copy=True
        )
    else:
        draw = rng.normal(size=(2, *truth_frequency.shape))
        frequency_noise = (draw[0] + 1j * draw[1]) * frequency_sigma * active[:, None]
        frequency_noise[:, -1] = frequency_noise[:, -1].real + 0.0j
        response_time, response_frequency = _canonical_pair_from_positive_frequency(
            truth_frequency + frequency_noise,
            np.asarray(truth["prediction.matrix.time-strain"]).shape[1],
        )
    frequency_noise = response_frequency - truth_frequency
    noise_time = response_time - np.asarray(
        truth["prediction.matrix.time-strain"], dtype=np.float64
    )
    _fft_pair_errors(response_time, response_frequency)
    _fft_pair_errors(noise_time, frequency_noise)
    conditional = np.asarray(truth["prediction.matrix.conditional-variance"], dtype=np.float64)
    responses = {
        "prediction.matrix.conditional-variance": np.array(conditional, copy=True),
        "prediction.matrix.frequency-imag": np.asarray(response_frequency.imag, dtype=np.float64),
        "prediction.matrix.frequency-real": np.asarray(response_frequency.real, dtype=np.float64),
        "prediction.matrix.time-strain": np.asarray(response_time, dtype=np.float64),
    }
    frequency_variance = np.square(frequency_sigma, dtype=np.float64)
    sample_count = response_time.shape[1]
    time_variance_by_detector = (
        4.0 * np.sum(frequency_variance[:, :-1], axis=1) + frequency_variance[:, -1]
    ) / float(sample_count * sample_count)
    variances = {
        "prediction.matrix.conditional-variance": np.repeat(
            np.square(conditional_sigma, dtype=np.float64)[:, None],
            sample_count,
            axis=1,
        ),
        "prediction.matrix.frequency-imag": np.array(frequency_variance, copy=True),
        "prediction.matrix.frequency-real": np.array(frequency_variance, copy=True),
        "prediction.matrix.time-strain": np.repeat(
            time_variance_by_detector[:, None], sample_count, axis=1
        ),
    }
    for output_id in _OUTPUTS:
        _require(np.all(variances[output_id] > 0.0), "nonpositive GW variance")
    return (
        responses,
        variances,
        {
            "family": family,
            "derived_seed": lineage.derived_seed,
            "response_noise_scale": float(slot["noise_scale"]),
            "canonical_noise_domain": "positive-frequency-rfft-bins-excluding-dc",
            "canonical_noise_draw_count": 0 if family == "zero-noise" else 1,
            "time_noise_is_derived": True,
            "calibration_envelope_applied": family == "published-psd-calibration-envelope",
            "real_strain_used": False,
            "real_likelihood_used": False,
        },
    )


def _scenario(
    config: Mapping[str, Any],
    slot: Mapping[str, Any],
    scenario_id: str,
    truth_world_id: str,
    truth_index: int,
    lineage: SeedLineage,
    responses: Mapping[str, np.ndarray],
    variances: Mapping[str, np.ndarray],
) -> ScenarioDescriptor:
    anchors = _anchor_map(config)
    values = slot["values"]
    truth = np.asarray([truth_index], dtype=np.int64)
    return ScenarioDescriptor(
        scenario_id=scenario_id,
        object_id=str(slot["slot_id"]),
        experiment_id=config["experiment_id"],
        domain="gw-network",
        geometry_mode=config["geometry_mode"],
        time_mode=config["time_mode"],
        coordinate_frame=config["coordinate_frame"],
        axes=(
            AxisSpec("detector", 3, None, None),
            AxisSpec("frequency", 128, None, None),
            AxisSpec("object", 1, None, None),
            AxisSpec("time", 256, None, None),
        ),
        formula_features=tuple(
            FeatureValueRef(
                feature_id,
                VALUES_PATH.as_posix(),
                array_sha256(values[feature_id]),
                values[feature_id].dtype.name,
                values[feature_id].shape,
                _FEATURE_SPECS[feature_id][0],
                _FEATURE_SPECS[feature_id][1],
                config["coordinate_frame"],
            )
            for feature_id in _FEATURES
        ),
        scoring_responses=tuple(
            FeatureValueRef(
                _RESPONSE_FOR[output_id],
                VALUES_PATH.as_posix(),
                array_sha256(responses[output_id]),
                "float64",
                responses[output_id].shape,
                _OUTPUT_AXES[output_id],
                "1",
                config["coordinate_frame"],
            )
            for output_id in _SCORING_OUTPUTS
        ),
        hidden_truth=(
            FeatureValueRef(
                "truth.scalar.injection-id",
                VALUES_PATH.as_posix(),
                array_sha256(truth),
                "int64",
                truth.shape,
                ("object",),
                "integer code",
                "latent",
            ),
        ),
        expected_predictions=tuple(
            EmittedPredictionSpec(
                output_id,
                VALUES_PATH.as_posix(),
                "float64",
                responses[output_id].shape,
                _OUTPUT_AXES[output_id],
                "1",
                config["coordinate_frame"],
            )
            for output_id in _OUTPUTS
        ),
        uncertainties=tuple(
            UncertaintyRef(
                f"uncertainty.{_RESPONSE_FOR[output_id]}",
                _RESPONSE_FOR[output_id],
                "diagonal-covariance",
                VALUES_PATH.as_posix(),
                array_sha256(variances[output_id]),
            )
            for output_id in _SCORING_OUTPUTS
        ),
        anchors=(
            AnchorBinding(
                "anchor.generic-runner-v2-audit",
                next(
                    row["path"]
                    for row in config["infrastructure_bindings"]
                    if row["id"] == "GENERIC_RUNNER_V2_AUDIT"
                ),
                next(
                    row["sha256"]
                    for row in config["infrastructure_bindings"]
                    if row["id"] == "GENERIC_RUNNER_V2_AUDIT"
                ),
            ),
            AnchorBinding(
                "anchor.lane2-source-preflight",
                anchors["LANE2_SOURCE_PREFLIGHT_RECEIPT"]["path"],
                anchors["LANE2_SOURCE_PREFLIGHT_RECEIPT"]["sha256"],
            ),
            AnchorBinding(
                "anchor.lane2-target-free-gates",
                anchors["LANE2_COHERENT_V6_TARGET_FREE_GATES"]["path"],
                anchors["LANE2_COHERENT_V6_TARGET_FREE_GATES"]["sha256"],
            ),
            AnchorBinding(
                "anchor.lane5-memory-receipt",
                anchors["LANE5_V2_RECEIPT"]["path"],
                anchors["LANE5_V2_RECEIPT"]["sha256"],
            ),
        ),
        seed_lineage=lineage,
    )


def _comparisons() -> tuple[ObservableComparison, ...]:
    return tuple(
        ObservableComparison(
            output_id,
            _RESPONSE_FOR[output_id],
            f"uncertainty.{_RESPONSE_FOR[output_id]}",
        )
        for output_id in _SCORING_OUTPUTS
    )


def _max_output_error(
    left: Mapping[str, np.ndarray],
    right: Mapping[str, np.ndarray],
    outputs: Sequence[str] = _OUTPUTS,
) -> float:
    return max(
        float(np.max(np.abs(np.asarray(left[key]) - np.asarray(right[key])))) for key in outputs
    )


def _diagnostics(
    config: Mapping[str, Any],
    inventory: Mapping[str, Any],
    slots: Sequence[Mapping[str, Any]],
    predictions: Mapping[tuple[str, str], Mapping[str, np.ndarray]],
    response_records: Sequence[Mapping[str, Any]],
    numerical_invalid_cell_count: int,
) -> dict[str, Any]:
    alias_rows = []
    for left, right in config["scoring"]["exact_alias_degeneracies"]:
        maximum = 0.0
        for slot in slots:
            maximum = max(
                maximum,
                _max_output_error(
                    predictions[(slot["slot_id"], left)],
                    predictions[(slot["slot_id"], right)],
                ),
            )
        alias_rows.append(
            {"left_formula_id": left, "right_formula_id": right, "maximum_error": maximum}
        )

    prediction_fft_maxima = {
        "forward": 0.0,
        "backward": 0.0,
        "dc_magnitude": 0.0,
        "nyquist_imaginary": 0.0,
    }
    inactive_error = 0.0
    prediction_check_count = 0
    for slot in slots:
        for formula_id in config["truth_mechanisms"]:
            prediction = predictions[(slot["slot_id"], formula_id)]
            spectrum = (
                prediction["prediction.matrix.frequency-real"]
                + 1j * prediction["prediction.matrix.frequency-imag"]
            )
            errors = _fft_pair_errors(prediction["prediction.matrix.time-strain"], spectrum)
            prediction_check_count += 1
            for key, value in errors.items():
                prediction_fft_maxima[key] = max(prediction_fft_maxima[key], value)
            if slot["event_id"] == "GW150914":
                inactive_error = max(
                    inactive_error,
                    max(float(np.max(np.abs(prediction[key][2]))) for key in _OUTPUTS),
                )

    response_fft_maxima = dict.fromkeys(prediction_fft_maxima, 0.0)
    noise_fft_maxima = dict.fromkeys(prediction_fft_maxima, 0.0)
    zero_noise_check_count = 0
    nonzero_noise_check_count = 0
    for record in response_records:
        response = record["response"]
        truth_prediction = record["truth_prediction"]
        response_frequency = (
            response["prediction.matrix.frequency-real"]
            + 1j * response["prediction.matrix.frequency-imag"]
        )
        truth_frequency = (
            truth_prediction["prediction.matrix.frequency-real"]
            + 1j * truth_prediction["prediction.matrix.frequency-imag"]
        )
        response_errors = _fft_pair_errors(
            response["prediction.matrix.time-strain"], response_frequency
        )
        noise_errors = _fft_pair_errors(
            response["prediction.matrix.time-strain"]
            - truth_prediction["prediction.matrix.time-strain"],
            response_frequency - truth_frequency,
        )
        for key, value in response_errors.items():
            response_fft_maxima[key] = max(response_fft_maxima[key], value)
        for key, value in noise_errors.items():
            noise_fft_maxima[key] = max(noise_fft_maxima[key], value)
        if record["noise_family"] == "zero-noise":
            zero_noise_check_count += 1
        else:
            nonzero_noise_check_count += 1

    first = slots[0]["values"]
    gr = _lane2_prediction(first, "GR", {})
    lane2_limits = {
        "dynamic_phase_beta_zero": _max_output_error(
            gr, _lane2_prediction(first, "DYNAMIC_PHASE", {"beta": 0.0})
        ),
        "attenuation_alpha_zero": _max_output_error(
            gr, _lane2_prediction(first, "ATTENUATION", {"alpha": 0.0})
        ),
        "reservoir_r_zero": _max_output_error(
            gr,
            _lane2_prediction(first, "RESERVOIR", {"r": 0.0, "log10_f_res_hz": 2.0}),
        ),
        "nonlinear_gamma_zero": _max_output_error(
            gr, _lane2_prediction(first, "NONLINEAR_PHASE", {"gamma": 0.0})
        ),
        "screened_beta_zero": _max_output_error(
            gr,
            _lane2_prediction(
                first,
                "SCREENED_PHASE",
                {"beta_s": 0.0, "log10_f_screen_hz": 2.0},
            ),
        ),
    }
    memory_delay_zero = _memory_prediction(first, "K01_RETARDED", {"delay": 0.0})
    memory_exp_zero = _memory_prediction(first, "K02_EXPONENTIAL", {"tau": 1.0e-12})
    memory_biexp_zero = _memory_prediction(
        first, "K03_BIEXPONENTIAL", {"tau1": 1.0e-12, "tau2": 1.0e-12, "weight": 0.65}
    )
    memory_limits = {
        "delay_zero": _max_output_error(gr, memory_delay_zero),
        "exponential_tau_zero": _max_output_error(gr, memory_exp_zero),
        "biexponential_tau_zero": _max_output_error(gr, memory_biexp_zero),
        "ou_mean_equals_exponential": _max_output_error(
            _memory_prediction(first, "K02_EXPONENTIAL", {"tau": 0.35}),
            _memory_prediction(first, "K06_STOCHASTIC_OU", {"tau": 0.35, "sigma": 0.05}),
            (
                "prediction.matrix.frequency-imag",
                "prediction.matrix.frequency-real",
                "prediction.matrix.time-strain",
            ),
        ),
    }

    distance_rows = []
    groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for slot in slots:
        groups[(slot["event_id"], slot["variant_id"], slot["noise_family"])].append(slot)
    for key, group in sorted(groups.items()):
        ordered = sorted(group, key=lambda row: row["distance_mpc"])
        near = predictions[(ordered[0]["slot_id"], "GR_NETWORK_CONTROL")]
        far = predictions[(ordered[1]["slot_id"], "GR_NETWORK_CONTROL")]
        ratios = []
        for output_id in (
            "prediction.matrix.frequency-imag",
            "prediction.matrix.frequency-real",
            "prediction.matrix.time-strain",
        ):
            denominator = float(np.linalg.norm(far[output_id]))
            ratios.append(float(np.linalg.norm(near[output_id])) / denominator)
        expected = ordered[1]["distance_mpc"] / ordered[0]["distance_mpc"]
        distance_rows.append(
            {
                "event_id": key[0],
                "variant_id": key[1],
                "noise_family": key[2],
                "expected_ratio": expected,
                "maximum_ratio_error": max(abs(value - expected) for value in ratios),
            }
        )

    variation_rows = []
    for event_id in ("GW150914", "GW170817"):
        selected = [
            slot
            for slot in slots
            if slot["event_id"] == event_id
            and slot["noise_family"] == "zero-noise"
            and slot["distance_mpc"]
            == min(row["distance_mpc"] for row in slots if row["event_id"] == event_id)
        ]
        difference = float(
            np.linalg.norm(
                selected[0]["values"]["source.matrix.base-time-strain"]
                - selected[1]["values"]["source.matrix.base-time-strain"]
            )
        )
        variation_rows.append({"event_id": event_id, "variant_signal_difference": difference})

    maximum_alias = max(row["maximum_error"] for row in alias_rows)
    maximum_lane2_limit = max(lane2_limits.values())
    maximum_memory_limit = max(memory_limits.values())
    maximum_distance_error = max(row["maximum_ratio_error"] for row in distance_rows)
    minimum_variant_difference = min(row["variant_signal_difference"] for row in variation_rows)
    fft_error = max(
        *prediction_fft_maxima.values(),
        *response_fft_maxima.values(),
        *noise_fft_maxima.values(),
    )
    passed = (
        maximum_alias == 0.0
        and fft_error <= 1.0e-12
        and prediction_check_count == 680
        and len(response_records) == 680
        and zero_noise_check_count == 136
        and nonzero_noise_check_count == 544
        and inactive_error <= 1.0e-12
        and maximum_lane2_limit <= 1.0e-12
        and maximum_memory_limit <= 1.0e-12
        and maximum_distance_error <= 1.0e-12
        and minimum_variant_difference > 0.0
        and numerical_invalid_cell_count > 0
    )
    return {
        "schema": "open-gravity-gw-timeseries-invariance-degeneracy-failure-ledger-3.0",
        "exact_alias_degeneracies": alias_rows,
        "lane2_gr_limits": lane2_limits,
        "lane5_memory_limits": memory_limits,
        "distance_scaling": distance_rows,
        "source_parameter_variation": variation_rows,
        "maximum_exact_alias_error": maximum_alias,
        "maximum_fft_roundtrip_error": fft_error,
        "fft_consistency": {
            "tolerance": 1.0e-12,
            "prediction_truth_and_candidate_check_count": prediction_check_count,
            "response_check_count": len(response_records),
            "noise_realization_check_count": len(response_records),
            "zero_noise_check_count": zero_noise_check_count,
            "nonzero_noise_check_count": nonzero_noise_check_count,
            "prediction_maxima": prediction_fft_maxima,
            "response_maxima": response_fft_maxima,
            "noise_realization_maxima": noise_fft_maxima,
            "all_forward_backward_dc_nyquist_pass": fft_error <= 1.0e-12,
        },
        "maximum_inactive_detector_signal": inactive_error,
        "maximum_lane2_gr_limit_error": maximum_lane2_limit,
        "maximum_lane5_zero_memory_limit_error": maximum_memory_limit,
        "maximum_distance_scaling_error": maximum_distance_error,
        "minimum_source_variant_signal_difference": minimum_variant_difference,
        "retained_upstream_optimizer_failures": {
            "method_passed": inventory["lane2_prediction"]["method_passed"],
            "optimizer_recovery_status": inventory["lane2_prediction"]["optimizer_recovery_status"],
            "registered_branch_power_status": inventory["lane2_prediction"][
                "registered_branch_power_status"
            ],
            "reservoir_power_status": inventory["lane2_prediction"]["reservoir_power_status"],
        },
        "retained_current_numerical_invalid_cell_count": numerical_invalid_cell_count,
        "pass": passed,
    }


def _npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **{key: arrays[key] for key in sorted(arrays)})
    return buffer.getvalue()


def derive_release() -> tuple[dict[str, Any], bytes, bytes, bytes, bytes, bytes, bytes]:
    config = load_config()
    validate_config(config)
    inventory = _source_inventory(config)
    slots = _source_slots(config, inventory)
    catalogue = _catalogue(config)
    bindings = _bindings(config)
    executable = tuple(row for row in bindings if row.status is BindingStatus.EXECUTABLE)
    registrations = tuple(
        AdapterRegistration.create(f"adapter.gw-timeseries.{row.formula_id.lower()}.v3", row)
        for row in executable
    )
    validate_adapter_registry(registrations)
    parameter_cells = _parameter_cells(bindings, config)
    module_sha = _file_sha256(Path(__file__))
    release = SyntheticSuiteRelease(
        suite_id=config["package_id"],
        version=config["version"],
        release_sha256=canonical_sha256(
            {
                "config_raw_sha256": _file_sha256(_ROOT / CONFIG_PATH),
                "generator_raw_sha256": module_sha,
                "runner_v2_audit_sha256": next(
                    row["sha256"]
                    for row in config["infrastructure_bindings"]
                    if row["id"] == "GENERIC_RUNNER_V2_AUDIT"
                ),
            }
        ),
        ontology_sha256=catalogue.content_sha256,
        generator_sha256=module_sha,
        observation_operator_sha256=_json_sha256(
            {
                "noise": config["noise"],
                "noise_families": config["noise_families"],
                "sample_grid": config["sample_grid"],
                "scoring": config["scoring"],
            }
        ),
        changed_feature_ids=(
            *_OUTPUTS,
            *(_RESPONSE_FOR[output_id] for output_id in _OUTPUTS),
            "truth.scalar.injection-id",
        ),
        change_level="MAJOR",
        response_calibrated=False,
        prediction_semantics_changed=True,
    )

    arrays: dict[str, np.ndarray] = {}
    predictions: dict[tuple[str, str], Mapping[str, np.ndarray]] = {}
    for slot in slots:
        for feature_id, value in slot["values"].items():
            arrays[_array_key("source", slot["slot_id"], feature_id)] = value
        for formula_id in config["truth_mechanisms"]:
            prediction = {
                key: np.asarray(value, dtype=np.float64)
                for key, value in _adapter_callable(formula_id)(
                    slot["values"], _truth_parameters(formula_id, config)
                ).items()
            }
            _require(set(prediction) == set(_OUTPUTS), "truth generator output changed")
            predictions[(slot["slot_id"], formula_id)] = prediction
            for output_id, value in prediction.items():
                arrays[_array_key("truth-prediction", slot["slot_id"], formula_id, output_id)] = (
                    value
                )

    scenarios: list[ScenarioDescriptor] = []
    runtime_values: dict[str, ScenarioRuntimeValues] = {}
    truth_formula_by_scenario: dict[str, str] = {}
    comparisons_by_scenario: dict[str, tuple[ObservableComparison, ...]] = {}
    scenario_payload: dict[str, dict[str, Any]] = {}
    response_records: list[dict[str, Any]] = []
    family_index = {value: index for index, value in enumerate(config["noise_families"])}
    for slot in slots:
        for truth_index, truth_formula_id in enumerate(config["truth_mechanisms"]):
            truth_world_id = f"truth.{truth_formula_id.lower()}"
            scenario_id = f"gw.{slot['slot_id']}.{truth_formula_id.lower()}.v3"
            lineage = SeedLineage(
                config["suite_seed"],
                scenario_id,
                slot["slot_id"],
                truth_world_id,
                family_index[slot["noise_family"]],
                0,
            )
            truth_prediction = predictions[(slot["slot_id"], truth_formula_id)]
            responses, variances, noise = _noise_response(truth_prediction, slot, lineage)
            response_records.append(
                {
                    "scenario_id": scenario_id,
                    "noise_family": slot["noise_family"],
                    "truth_prediction": truth_prediction,
                    "response": responses,
                }
            )
            scenario = _scenario(
                config,
                slot,
                scenario_id,
                truth_world_id,
                truth_index,
                lineage,
                responses,
                variances,
            )
            truth_value = np.asarray([truth_index], dtype=np.int64)
            response_runtime = {
                _RESPONSE_FOR[output_id]: responses[output_id] for output_id in _SCORING_OUTPUTS
            }
            uncertainty_runtime = {
                f"uncertainty.{_RESPONSE_FOR[output_id]}": variances[output_id]
                for output_id in _SCORING_OUTPUTS
            }
            validate_scenario_values(
                scenario,
                formula_values=slot["values"],
                response_values=response_runtime,
                truth_values={"truth.scalar.injection-id": truth_value},
                uncertainty_values=uncertainty_runtime,
            )
            response_locators = {}
            variance_locators = {}
            prediction_locators = {}
            for output_id in _OUTPUTS:
                response_key = _array_key("response", scenario_id, output_id)
                variance_key = _array_key("variance", scenario_id, output_id)
                prediction_key = _array_key(
                    "truth-prediction", slot["slot_id"], truth_formula_id, output_id
                )
                arrays[response_key] = responses[output_id]
                arrays[variance_key] = variances[output_id]
                response_locators[output_id] = {
                    "key": response_key,
                    "sha256": array_sha256(responses[output_id]),
                }
                variance_locators[output_id] = {
                    "key": variance_key,
                    "sha256": array_sha256(variances[output_id]),
                }
                prediction_locators[output_id] = {
                    "key": prediction_key,
                    "sha256": array_sha256(truth_prediction[output_id]),
                }
            truth_key = _array_key("truth", scenario_id)
            arrays[truth_key] = truth_value
            scenarios.append(scenario)
            runtime_values[scenario_id] = ScenarioRuntimeValues(
                slot["values"],
                response_runtime,
                {"truth.scalar.injection-id": truth_value},
                uncertainty_runtime,
            )
            truth_formula_by_scenario[scenario_id] = truth_formula_id
            comparisons_by_scenario[scenario_id] = _comparisons()
            scenario_payload[scenario_id] = {
                "scenario": scenario.to_dict(),
                "scenario_sha256": scenario.content_sha256,
                "event_id": slot["event_id"],
                "source_variant_id": slot["variant_id"],
                "distance_mpc": slot["distance_mpc"],
                "truth_formula_id": truth_formula_id,
                "truth_world_id": truth_world_id,
                "noise": noise,
                "value_locators": {
                    "path": VALUES_PATH.as_posix(),
                    "truth_prediction": prediction_locators,
                    "responses": response_locators,
                    "variances": variance_locators,
                    "truth": {"key": truth_key, "sha256": array_sha256(truth_value)},
                },
            }

    ordered_scenarios = tuple(sorted(scenarios, key=lambda row: row.scenario_id))
    _require(len(ordered_scenarios) == 680, "GW scenario count changed")
    result = run_discovery_matrix_v2(
        catalogue=catalogue,
        release=release,
        scenarios=ordered_scenarios,
        scenario_values=runtime_values,
        truth_formula_by_scenario=truth_formula_by_scenario,
        bindings=bindings,
        adapters=registrations,
        parameter_cells=parameter_cells,
        comparisons=comparisons_by_scenario,
        distinct_gap=float(config["scoring"]["distinct_gap"]),
        ledger_id="gravity.synthetic.gw-timeseries-source-anchored-matrix.v3",
    )
    cells_by_scenario: dict[str, list[Any]] = defaultdict(list)
    for cell in result.cells:
        cells_by_scenario[cell.scenario_id].append(cell)
    confusion_counts = {
        truth: {candidate: 0 for candidate in config["truth_mechanisms"]}
        for truth in config["truth_mechanisms"]
    }
    recovery_by_truth = {
        truth: {"scenarios": 0, "recovered": 0, "distinct": 0}
        for truth in config["truth_mechanisms"]
    }
    scenario_rows = []
    for scenario in ordered_scenarios:
        cells = cells_by_scenario[scenario.scenario_id]
        winners = sorted({cell.formula_id for cell in cells if cell.winner})
        valid = [cell for cell in cells if cell.whitened_rmse is not None]
        truth_formula_id = truth_formula_by_scenario[scenario.scenario_id]
        truth_recovered = any(cell.truth_recovered for cell in cells)
        distinct = any(cell.distinct for cell in valid)
        recovery_by_truth[truth_formula_id]["scenarios"] += 1
        recovery_by_truth[truth_formula_id]["recovered"] += int(truth_recovered)
        recovery_by_truth[truth_formula_id]["distinct"] += int(truth_recovered and distinct)
        for winner in winners:
            if winner in confusion_counts[truth_formula_id]:
                confusion_counts[truth_formula_id][winner] += 1
        scenario_rows.append(
            {
                **scenario_payload[scenario.scenario_id],
                "injection_recovery": {
                    "winner_formula_ids": winners,
                    "truth_recovered": truth_recovered,
                    "truth_distinctly_recovered": truth_recovered and distinct,
                    "distinct_by_generic_runner_v2": distinct,
                    "matrix_cell_count": len(cells),
                    "valid_scored_cell_count": len(valid),
                },
            }
        )

    numerical_invalid_count = sum(
        cell.discovery_status == "NUMERICAL_INVALID" for cell in result.cells
    )
    source_blocked_count = sum(cell.discovery_status == "SOURCE_BLOCKED" for cell in result.cells)
    unadapted_count = sum(cell.discovery_status == "UNADAPTED" for cell in result.cells)
    _require(result.attempted_cell_count == 16_320, "GW attempted matrix count changed")
    _require(result.scored_cell_count == 11_560, "GW scored matrix count changed")
    _require(len(result.ledger.entries) == 28_560, "GW replay count changed")
    _require(numerical_invalid_count == 680, "GW numerical invalid retention changed")
    _require(source_blocked_count == 2_720, "GW source-blocked count changed")
    _require(unadapted_count == 1_360, "GW unadapted count changed")

    diagnostics = _diagnostics(
        config,
        inventory,
        slots,
        predictions,
        response_records,
        numerical_invalid_count,
    )
    _require(diagnostics["pass"], "GW invariant/limit gates failed")
    confusion = {
        "schema": "open-gravity-gw-timeseries-confusion-matrix-3.0",
        "truth_formula_ids": config["truth_mechanisms"],
        "candidate_formula_ids": [row.formula_id for row in executable],
        "winner_membership_counts": confusion_counts,
        "recovery_by_truth": recovery_by_truth,
        "scenario_count": len(ordered_scenarios),
        "attempted_matrix_cell_count": result.attempted_cell_count,
        "scored_matrix_cell_count": result.scored_cell_count,
        "truth_recovery_count": result.truth_recovery_count,
        "distinct_truth_recovery_count": result.distinct_truth_recovery_count,
        "numerical_invalid_cell_count": numerical_invalid_count,
        "source_blocked_cell_count": source_blocked_count,
        "unadapted_cell_count": unadapted_count,
        "no_hand_ranking": True,
    }

    values_bytes = _npz_bytes(arrays)
    _require(values_bytes == _npz_bytes(arrays), "GW NPZ serialization nondeterministic")
    scenarios_bytes = b"".join(_json_bytes(row) + b"\n" for row in scenario_rows)
    matrix_bytes = _json_bytes(result.to_dict(), indent=2)
    ledger_bytes = _json_bytes(result.ledger.to_dict(), indent=2)
    confusion_bytes = _json_bytes(confusion, indent=2)
    diagnostics_bytes = _json_bytes(diagnostics, indent=2)
    source_hashes = {row["id"]: row["sha256"] for row in config["source_anchors"]}
    receipt_body = {
        "schema": "open-gravity-gw-timeseries-source-anchored-synthetic-injection-matrix-receipt-3.0",
        "package_id": config["package_id"],
        "version": config["version"],
        "status": "FROZEN_SYNTHETIC_ONLY_COMPLETE_AWAITING_DISTINCT_AUDIT",
        "claim_class": config["claim_class"],
        "scientific_claim": "NONE_SYNTHETIC_ONLY_NOT_SUPPORT_OR_REJECTION",
        "independent_audit_completed": False,
        "distinct_independent_audit_required": True,
        "blocked_predecessor_audit": {
            "path": next(
                row["path"]
                for row in config["infrastructure_bindings"]
                if row["id"] == "GW_V2_INDEPENDENT_AUDIT_BLOCK"
            ),
            "raw_sha256": next(
                row["sha256"]
                for row in config["infrastructure_bindings"]
                if row["id"] == "GW_V2_INDEPENDENT_AUDIT_BLOCK"
            ),
            "decision": inventory["blocked_audit"]["decision"],
            "finding_code": inventory["blocked_audit"]["blocking_finding"]["code"],
        },
        "representation_contract": config["representation_contract"],
        "scored_output_ids": list(_SCORING_OUTPUTS),
        "derived_unscored_output_ids": config["scoring"]["derived_unscored_outputs"],
        "event_count": len(config["events"]),
        "source_population_slot_count": len(slots),
        "truth_mechanism_count": len(config["truth_mechanisms"]),
        "noise_family_count": len(config["noise_families"]),
        "scenario_count": len(ordered_scenarios),
        "executable_binding_count": len(executable),
        "blocked_binding_count": len(bindings) - len(executable),
        "attempted_matrix_cell_count": result.attempted_cell_count,
        "scored_matrix_cell_count": result.scored_cell_count,
        "numerical_invalid_cell_count": numerical_invalid_count,
        "source_blocked_cell_count": source_blocked_count,
        "unadapted_cell_count": unadapted_count,
        "replay_entry_count": len(result.ledger.entries),
        "truth_recovery_count": result.truth_recovery_count,
        "distinct_truth_recovery_count": result.distinct_truth_recovery_count,
        "recovery_by_truth": recovery_by_truth,
        "mechanism_ids": [row.formula_id for row in executable],
        "formula_bindings": {row.formula_id: row.to_dict() for row in bindings},
        "formula_binding_sha256": {row.formula_id: row.content_sha256 for row in bindings},
        "adapter_sha256": {
            row.formula_binding.formula_id: row.adapter_sha256 for row in registrations
        },
        "adapter_blocks": config["adapter_blocks"],
        "source_anchor_sha256": source_hashes,
        "generic_runner_v2": {
            "result_content_sha256": result.content_sha256,
            "ledger_content_sha256": result.ledger.content_sha256,
            "audit_status": inventory["runner_audit"]["status"],
            "audit_content_sha256": inventory["runner_audit"]["content_sha256"],
        },
        "invariance_gates": {
            key: value
            for key, value in diagnostics.items()
            if key.startswith(("maximum_", "minimum_")) or key == "pass"
        },
        "fft_consistency": diagnostics["fft_consistency"],
        "retained_upstream_optimizer_failures": diagnostics["retained_upstream_optimizer_failures"],
        "package_hashes": {
            "config_raw_sha256": _file_sha256(_ROOT / CONFIG_PATH),
            "parameter_schema_raw_sha256": _file_sha256(_ROOT / PARAMETER_SCHEMA_PATH),
            "module_raw_sha256": module_sha,
            "test_raw_sha256": _file_sha256(_ROOT / TEST_PATH),
        },
        "artifact_sha256": {
            "values.npz": hashlib.sha256(values_bytes).hexdigest(),
            "scenarios.jsonl": hashlib.sha256(scenarios_bytes).hexdigest(),
            "matrix-result.json": hashlib.sha256(matrix_bytes).hexdigest(),
            "ledger.json": hashlib.sha256(ledger_bytes).hexdigest(),
            "confusion-matrix.json": hashlib.sha256(confusion_bytes).hexdigest(),
            "invariance-degeneracy-and-failure-ledger.json": hashlib.sha256(
                diagnostics_bytes
            ).hexdigest(),
        },
        "access_accounting": config["access_contract"],
        "limitations": [
            "All scoring responses are synthetic; no empirical support or rejection is authorized.",
            "GW150914 uses the exact frozen public NR source waveform, while GW170817 uses a source-parameter-anchored analytic chirp surrogate rather than the platform-pinned LALSuite runtime.",
            "Published PSD and calibration payload arrays remain unopened; their manifests define synthetic nuisance families only.",
            "Exact delay, one-pole, two-pole, and OU-mean aliases are retained as non-identifiable countermodel pairs.",
            "Frequency is the only scored waveform domain; the stored time series is a deterministic unscored inverse-FFT view and is not counted as independent evidence.",
            "The upstream Lane2 optimizer and branch-power failures remain failures and are not repaired or reinterpreted by this matrix.",
            "GW170817/GW150914 strain samples, real likelihoods, real off-source PSD fits, and empirical calibration learning remain sealed.",
        ],
    }
    receipt = {**receipt_body, "content_sha256": _json_sha256(receipt_body)}
    return (
        receipt,
        values_bytes,
        scenarios_bytes,
        matrix_bytes,
        ledger_bytes,
        confusion_bytes,
        diagnostics_bytes,
    )


def _write_once(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        _require(path.read_bytes() == payload, f"refusing changed GW artifact: {path}")
        return "EXISTING_IDENTICAL"
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            _require(path.read_bytes() == payload, f"concurrent changed GW artifact: {path}")
            return "EXISTING_IDENTICAL"
        return "CREATED"
    finally:
        temporary.unlink(missing_ok=True)


def freeze() -> str:
    receipt, values, scenarios, matrix, ledger, confusion, diagnostics = derive_release()
    payloads = (
        (VALUES_PATH, values),
        (SCENARIOS_PATH, scenarios),
        (MATRIX_PATH, matrix),
        (LEDGER_PATH, ledger),
        (CONFUSION_PATH, confusion),
        (DIAGNOSTICS_PATH, diagnostics),
        (RECEIPT_PATH, _json_bytes(receipt, indent=2)),
    )
    return ":".join(_write_once(_ROOT / path, payload) for path, payload in payloads)


def check() -> dict[str, Any]:
    receipt, values, scenarios, matrix, ledger, confusion, diagnostics = derive_release()
    payloads = (
        (VALUES_PATH, values),
        (SCENARIOS_PATH, scenarios),
        (MATRIX_PATH, matrix),
        (LEDGER_PATH, ledger),
        (CONFUSION_PATH, confusion),
        (DIAGNOSTICS_PATH, diagnostics),
        (RECEIPT_PATH, _json_bytes(receipt, indent=2)),
    )
    for path, payload in payloads:
        if not (_ROOT / path).is_file() or (_ROOT / path).read_bytes() != payload:
            raise SystemExit(f"stored GW synthetic artifact differs: {path}")
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "check"))
    args = parser.parse_args(argv)
    if args.command == "freeze":
        print(freeze())
    else:
        receipt = check()
        print(receipt["status"], receipt["content_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
