"""Bounded cross-domain target-blind evaluator scaffold.

This module deliberately separates a predictor-only generation packet from every
domain adapter.  Its only empirical activity is a finite/schema smoke on already
exposed development data; all formula recovery scores are injected controls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

CONFIG_PATH = Path("configs/gravity_shared_target_blind_evaluator_v1.json")
OUTPUT_PATH = Path("runs/gravity/shared-target-blind-evaluator-v1.json")
MODULE_PATH = Path("src/sigma_theory_compiler/gravity_shared_target_blind_evaluator.py")
TEST_PATH = Path("tests/test_gravity_shared_target_blind_evaluator.py")
DOC_PATH = Path("docs/GRAVITY_SHARED_TARGET_BLIND_EVALUATOR_V1.md")

DOMAINS = (
    "galaxy_rotation",
    "xcop_thermodynamic",
    "group_bridge",
    "lensing_metric",
)
CONTROL_IDS = ("newtonian", "known_law", "wrong_law")
INJECTION_IDS = ("newtonian", "known_law")
EXPECTED_CONTROL_FORMULAS = {
    "galaxy_rotation": {
        "newtonian": "x_source",
        "known_law": "x_source/(1-exp(-sqrt(x_source)))",
        "wrong_law": "x_source*(1+0.75*x_radial)",
    },
    "xcop_thermodynamic": {
        "newtonian": "x_source",
        "known_law": "x_source+0.4*x_state/(1+x_radial)",
        "wrong_law": "x_source-0.4*x_state/(1+x_radial)",
    },
    "group_bridge": {
        "newtonian": "x_source",
        "known_law": "sqrt(x_source*x_source+0.25*x_source)+0.1*x_state",
        "wrong_law": "x_source/(1+x_state)",
    },
    "lensing_metric": {
        "newtonian": "2*x_source",
        "known_law": "x_source+x_state",
        "wrong_law": "abs(x_source-x_state)",
    },
}


class SharedTargetBlindEvaluatorError(RuntimeError):
    """Raised when a target-blind boundary or frozen binding changes."""


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _content_hashed(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result["content_sha256"] = _sha256_bytes(_canonical(result).encode())
    return result


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SharedTargetBlindEvaluatorError(f"JSON object required: {path}")
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_canonical(payload) + "\n", encoding="utf-8")


def _without_contract_hash(adapter: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(adapter)
    value.pop("contract_sha256", None)
    return value


def adapter_contract_sha256(adapter: Mapping[str, Any]) -> str:
    return _sha256_bytes(_canonical(_without_contract_hash(adapter)).encode())


def config_contract_sha256(config: Mapping[str, Any]) -> str:
    value = dict(config)
    value.pop("contract_sha256", None)
    return _sha256_bytes(_canonical(value).encode())


def load_config(root: Path) -> dict[str, Any]:
    config = _read_json(root / CONFIG_PATH)
    validate_config(root, config)
    return config


def _validate_binding(root: Path, binding: Mapping[str, Any]) -> dict[str, Any]:
    path = root / str(binding["path"])
    if not path.is_file() or _sha256_file(path) != str(binding["file_sha256"]):
        raise SharedTargetBlindEvaluatorError(f"source binding changed: {path}")
    payload = _read_json(path)
    expected_content = binding.get("content_sha256")
    if expected_content is not None and payload.get("content_sha256") != expected_content:
        raise SharedTargetBlindEvaluatorError(f"content binding changed: {path}")
    return payload


def validate_config(root: Path, config: Mapping[str, Any]) -> None:
    if (
        config.get("schema_version") != "invariant-gravity-shared-target-blind-evaluator-config-1.0"
        or config.get("evaluator_id") != "gravity-shared-target-blind-evaluator-v1"
        or config.get("status") != "bounded_development_scaffold_not_scientific_execution"
    ):
        raise SharedTargetBlindEvaluatorError("unsupported shared evaluator config")
    if config_contract_sha256(config) != config.get("contract_sha256"):
        raise SharedTargetBlindEvaluatorError("top-level config contract changed")
    if config.get("single_counterexample_is_universal_veto") is not False:
        raise SharedTargetBlindEvaluatorError("single counterexample may not be a universal veto")
    if int(config.get("rows_per_injected_packet", 0)) != 64:
        raise SharedTargetBlindEvaluatorError("bounded injected row count changed")
    if set(config.get("adapters", {})) != set(DOMAINS):
        raise SharedTargetBlindEvaluatorError("adapter set changed")
    if set(config.get("control_formulas", {})) != set(DOMAINS):
        raise SharedTargetBlindEvaluatorError("control domain set changed")
    if config.get("control_formulas") != EXPECTED_CONTROL_FORMULAS:
        raise SharedTargetBlindEvaluatorError("control expression registry changed")
    for domain in DOMAINS:
        adapter = config["adapters"][domain]
        if adapter_contract_sha256(adapter) != adapter.get("contract_sha256"):
            raise SharedTargetBlindEvaluatorError(f"adapter contract changed: {domain}")
        if set(config["control_formulas"][domain]) != set(CONTROL_IDS):
            raise SharedTargetBlindEvaluatorError(f"control set changed: {domain}")
        if adapter.get("target_passed_to_generation") is not False:
            raise SharedTargetBlindEvaluatorError(f"target leak enabled: {domain}")
    if config["adapters"]["group_bridge"].get("empirical_score_allowed") is not False:
        raise SharedTargetBlindEvaluatorError("group empirical scoring must be fail-closed")
    if config["adapters"]["lensing_metric"].get("empirical_score_allowed") is not False:
        raise SharedTargetBlindEvaluatorError("lensing empirical scoring must be fail-closed")
    if config["claims"] != {
        "ben_child_empirically_works": False,
        "historical_novelty_established": False,
        "publication_ready": False,
        "gr_replaced": False,
        "synthetic_recovery_is_scientific_evidence": False,
    }:
        raise SharedTargetBlindEvaluatorError("claim ceiling changed")

    bindings = config["source_bindings"]
    parent = _validate_binding(root, bindings["parent_registry"])
    if parent.get("decision") != bindings["parent_registry"]["required_decision"]:
        raise SharedTargetBlindEvaluatorError("parent registry decision changed")
    recombination = _validate_binding(root, bindings["recombination"])
    top = recombination.get("top_architecture", {})
    if (
        recombination.get("decision") != bindings["recombination"]["required_decision"]
        or top.get("architecture_id") != bindings["recombination"]["required_architecture_id"]
        or top.get("children_empirically_work") is not False
        or top.get("structural_descendants_only") is not True
        or recombination.get("claim_boundary", {}).get("children_empirically_work") is not False
    ):
        raise SharedTargetBlindEvaluatorError("B+E+N structural boundary changed")
    # Empirical bindings are deliberately not opened in this pre-generation validation.
    # They are validated only by ``validate_empirical_bindings`` after the generation
    # packet and candidate registry have been frozen.


def validate_empirical_bindings(root: Path, config: Mapping[str, Any]) -> None:
    """Validate exposed-data bindings after predictor generation is complete."""

    bindings = config["source_bindings"]
    galaxy = _validate_binding(root, bindings["galaxy_exposure_receipt"])
    counts = galaxy.get("counts", {})
    if (
        counts.get("exploration_points_fitted")
        != bindings["galaxy_exposure_receipt"]["required_exploration_rows"]
        or counts.get("exploration_galaxies_fitted")
        != bindings["galaxy_exposure_receipt"]["required_exploration_objects"]
        or galaxy.get("claims", {}).get("confirmation_set_fitted") is not False
    ):
        raise SharedTargetBlindEvaluatorError("galaxy exposure boundary changed")
    _validate_binding(root, bindings["xcop_config"])
    _validate_binding(root, bindings["xcop_result"])
    _validate_binding(root, bindings["xcop_source_receipt"])


def build_generation_packet(config: Mapping[str, Any]) -> dict[str, Any]:
    """Build and validate the predictor-only packet before any response access."""

    contract = config["generation_contract"]
    architecture_sha = config["source_bindings"]["recombination"]["content_sha256"]
    packet = _content_hashed(
        {
            "schema_version": "invariant-predictor-generation-packet-1.0",
            "packet_id": "shared-predictor-schema-v1",
            "architecture_binding_sha256": architecture_sha,
            "typed_dimensionless_variables": contract["typed_dimensionless_variables"],
            "operator_schema": contract["operator_schema"],
        }
    )
    allowed = set(contract["allowed_top_level_fields"])
    if set(packet) != allowed:
        raise SharedTargetBlindEvaluatorError("generation packet fields are not exact")
    encoded = _canonical(packet).lower()
    for token in map(str, contract["forbidden_tokens"]):
        if token.lower() in encoded:
            raise SharedTargetBlindEvaluatorError(
                f"forbidden generation-packet token present: {token}"
            )
    for row in packet["typed_dimensionless_variables"]:
        if row.get("dimension") != "1":
            raise SharedTargetBlindEvaluatorError("non-dimensionless generation variable")
    return packet


def _ensure_matrix(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != 4 or np.any(~np.isfinite(matrix)):
        raise SharedTargetBlindEvaluatorError("adapter did not produce finite Nx4 predictors")
    if np.any(matrix[:, 0] <= 0.0) or np.any(matrix[:, 1] <= 0.0):
        raise SharedTargetBlindEvaluatorError("positive predictor contract failed")
    return matrix


def adapt_galaxy_rotation_rows(rows: Mapping[str, Sequence[float]]) -> np.ndarray:
    """Map one label-free rotation packet to four dimensionless predictors."""

    radius = np.asarray(rows["rad_kpc"], dtype=np.float64)
    gas = np.asarray(rows["vgas_km_s"], dtype=np.float64)
    disk = np.asarray(rows["vdisk_km_s"], dtype=np.float64)
    bulge = np.asarray(rows["vbul_km_s"], dtype=np.float64)
    if not (len(radius) == len(gas) == len(disk) == len(bulge)) or len(radius) == 0:
        raise SharedTargetBlindEvaluatorError("galaxy adapter column length mismatch")
    components = np.abs(gas * gas) + 0.5 * disk * disk + 0.7 * bulge * bulge
    signed = gas * np.abs(gas) + 0.5 * disk * disk + 0.7 * bulge * bulge
    if np.any(radius <= 0.0) or np.any(signed <= 0.0) or np.any(components <= 0.0):
        raise SharedTargetBlindEvaluatorError("galaxy adapter domain failure")
    a0_km2_s2_kpc = 3702.81458
    stellar = 0.5 * disk * disk + 0.7 * bulge * bulge
    morphology = np.divide(
        0.7 * bulge * bulge,
        stellar,
        out=np.zeros_like(stellar),
        where=stellar > 0.0,
    )
    return _ensure_matrix(
        np.column_stack(
            (
                signed / radius / a0_km2_s2_kpc,
                radius / np.max(radius),
                np.abs(gas * gas) / components,
                morphology,
            )
        )
    )


def adapt_xcop_arrays(packet: Mapping[str, Any]) -> np.ndarray:
    """Map density-only X-COP predictors; thermodynamic responses remain separate."""

    radius = np.asarray(packet["density_radius_kpc"], dtype=np.float64)
    density = np.asarray(packet["ne_cm3"], dtype=np.float64)
    r500 = float(packet["r500_kpc"])
    if len(radius) != len(density) or len(radius) < 2 or r500 <= 0.0:
        raise SharedTargetBlindEvaluatorError("X-COP adapter shape failure")
    if np.any(radius <= 0.0) or np.any(density <= 0.0):
        raise SharedTargetBlindEvaluatorError("X-COP adapter domain failure")
    log_slope = np.abs(np.gradient(np.log(density), np.log(radius)))
    return _ensure_matrix(
        np.column_stack(
            (
                density / np.max(density),
                radius / r500,
                log_slope,
                np.ones_like(radius),
            )
        )
    )


def adapt_group_rows(rows: Mapping[str, Sequence[float]]) -> np.ndarray:
    return _ensure_matrix(
        np.column_stack(
            tuple(
                np.asarray(rows[key], dtype=np.float64)
                for key in (
                    "baryonic_acceleration_ratio",
                    "radius_scale_ratio",
                    "environment_ratio",
                    "geometry_ratio",
                )
            )
        )
    )


def adapt_lensing_rows(rows: Mapping[str, Sequence[float]]) -> np.ndarray:
    return _ensure_matrix(
        np.column_stack(
            tuple(
                np.asarray(rows[key], dtype=np.float64)
                for key in (
                    "potential_ratio_phi",
                    "impact_parameter_ratio",
                    "potential_ratio_psi",
                    "geometry_ratio",
                )
            )
        )
    )


def _synthetic_predictors(domain: str, rows: int) -> np.ndarray:
    t = np.linspace(0.05, 0.95, rows, dtype=np.float64)
    if domain == "galaxy_rotation":
        radius = 0.2 + 19.8 * t
        return adapt_galaxy_rotation_rows(
            {
                "rad_kpc": radius,
                "vgas_km_s": 18.0 + 7.0 * np.sin(2.0 * math.pi * t),
                "vdisk_km_s": 35.0 + 55.0 * t,
                "vbul_km_s": 4.0 + 18.0 * t * t,
            }
        )
    if domain == "xcop_thermodynamic":
        radius = 50.0 + 1450.0 * t
        return adapt_xcop_arrays(
            {
                "density_radius_kpc": radius,
                "ne_cm3": 0.02 * (1.0 + radius / 180.0) ** -1.8,
                "r500_kpc": 1500.0,
            }
        )
    if domain == "group_bridge":
        return adapt_group_rows(
            {
                "baryonic_acceleration_ratio": 0.03 + 1.5 * t * t,
                "radius_scale_ratio": 0.08 + 1.8 * t,
                "environment_ratio": 0.1 + 0.8 * t,
                "geometry_ratio": 0.2 + 0.6 * (1.0 - t),
            }
        )
    if domain == "lensing_metric":
        phi = 0.08 + 0.9 * t
        return adapt_lensing_rows(
            {
                "potential_ratio_phi": phi,
                "potential_ratio_psi": phi * (0.65 + 0.2 * t),
                "impact_parameter_ratio": 0.1 + 2.0 * t,
                "geometry_ratio": 0.25 + 0.5 * t,
            }
        )
    raise SharedTargetBlindEvaluatorError(f"unknown domain: {domain}")


def evaluate_control_vector(domain: str, control_id: str, x: np.ndarray) -> np.ndarray:
    x = _ensure_matrix(x)
    source, radial, state, _geometry = x.T
    if domain == "galaxy_rotation":
        if control_id == "newtonian":
            return source
        if control_id == "known_law":
            return source / (-np.expm1(-np.sqrt(source)))
        if control_id == "wrong_law":
            return source * (1.0 + 0.75 * radial)
    elif domain == "xcop_thermodynamic":
        term = 0.4 * state / (1.0 + radial)
        if control_id == "newtonian":
            return source
        if control_id == "known_law":
            return source + term
        if control_id == "wrong_law":
            return source - term
    elif domain == "group_bridge":
        if control_id == "newtonian":
            return source
        if control_id == "known_law":
            return np.sqrt(source * source + 0.25 * source) + 0.1 * state
        if control_id == "wrong_law":
            return source / (1.0 + state)
    elif domain == "lensing_metric":
        if control_id == "newtonian":
            return 2.0 * source
        if control_id == "known_law":
            return source + state
        if control_id == "wrong_law":
            return np.abs(source - state)
    raise SharedTargetBlindEvaluatorError(f"unknown control: {domain}:{control_id}")


def evaluate_control_scalar(domain: str, control_id: str, row: Sequence[float]) -> float:
    source, radial, state, _geometry = map(float, row)
    if domain == "galaxy_rotation":
        values = {
            "newtonian": source,
            "known_law": source / (1.0 - math.exp(-math.sqrt(source))),
            "wrong_law": source * (1.0 + 0.75 * radial),
        }
    elif domain == "xcop_thermodynamic":
        term = 0.4 * state / (1.0 + radial)
        values = {
            "newtonian": source,
            "known_law": source + term,
            "wrong_law": source - term,
        }
    elif domain == "group_bridge":
        values = {
            "newtonian": source,
            "known_law": math.sqrt(source * source + 0.25 * source) + 0.1 * state,
            "wrong_law": source / (1.0 + state),
        }
    elif domain == "lensing_metric":
        values = {
            "newtonian": 2.0 * source,
            "known_law": source + state,
            "wrong_law": abs(source - state),
        }
    else:
        raise SharedTargetBlindEvaluatorError(f"unknown domain: {domain}")
    try:
        return values[control_id]
    except KeyError as exc:
        raise SharedTargetBlindEvaluatorError(f"unknown control: {control_id}") from exc


def _score(truth: np.ndarray, prediction: np.ndarray) -> str:
    return f"{float(np.mean((truth - prediction) ** 2)):.12e}"


def _synthetic_domain_result(domain: str, rows: int) -> dict[str, Any]:
    x = _synthetic_predictors(domain, rows)
    parity: dict[str, str] = {}
    for control_id in CONTROL_IDS:
        vector = evaluate_control_vector(domain, control_id, x)
        scalar = np.asarray(
            [evaluate_control_scalar(domain, control_id, row) for row in x],
            dtype=np.float64,
        )
        parity[control_id] = f"{float(np.max(np.abs(vector - scalar))):.12e}"
    recovery: dict[str, Any] = {}
    for injection_id in INJECTION_IDS:
        truth = evaluate_control_vector(domain, injection_id, x)
        scores = {
            control_id: _score(truth, evaluate_control_vector(domain, control_id, x))
            for control_id in CONTROL_IDS
        }
        winner = min(CONTROL_IDS, key=lambda item: (float(scores[item]), item))
        recovery[injection_id] = {
            "injected_control": injection_id,
            "winner": winner,
            "scores_mse": scores,
            "recovered": winner == injection_id,
            "wrong_law_rejected": (
                float(scores["wrong_law"]) > float(scores[injection_id]) and winner != "wrong_law"
            ),
        }
    return {
        "domain": domain,
        "rows_generated": rows,
        "adapter_output_sha256": _sha256_bytes(x.astype("<f8", copy=False).tobytes()),
        "parity_max_abs_difference": parity,
        "parity_pass": max(float(value) for value in parity.values()) <= 1.0e-12,
        "recovery": recovery,
        "recovery_pass": all(value["recovered"] for value in recovery.values()),
        "wrong_law_control_pass": all(value["wrong_law_rejected"] for value in recovery.values()),
    }


def build_synthetic_controls(config: Mapping[str, Any]) -> dict[str, Any]:
    rows = int(config["rows_per_injected_packet"])
    with ThreadPoolExecutor(max_workers=len(DOMAINS)) as pool:
        futures = {
            domain: pool.submit(_synthetic_domain_result, domain, rows) for domain in DOMAINS
        }
        results = {domain: futures[domain].result() for domain in DOMAINS}
    return {
        "execution_mode": "bounded_cpu_parallel_four_domain_workers",
        "domains": results,
        "all_parity_pass": all(row["parity_pass"] for row in results.values()),
        "all_recovery_pass": all(row["recovery_pass"] for row in results.values()),
        "all_wrong_law_controls_pass": all(
            row["wrong_law_control_pass"] for row in results.values()
        ),
    }


def build_candidate_registry(
    config: Mapping[str, Any], generation_packet_sha256: str
) -> dict[str, Any]:
    """Freeze evaluation-only controls and the unscored structural child."""

    controls = [
        {
            "registry_id": f"injected-control.{domain}.{control_id}",
            "domain_adapter": domain,
            "control_role": control_id,
            "frozen_expression": config["control_formulas"][domain][control_id],
            "scientific_candidate": False,
            "response_selected_or_retuned": False,
        }
        for domain in DOMAINS
        for control_id in CONTROL_IDS
    ]
    expressions = [str(row["frozen_expression"]) for row in controls]
    return _content_hashed(
        {
            "schema_version": "invariant-shared-evaluator-candidate-registry-1.0",
            "generation_packet_sha256": generation_packet_sha256,
            "ben_structural_candidate": {
                "registry_id": "structural.BEN-additive-cross-scale-v1",
                "registered": True,
                "executed": False,
                "outcome_scores_computed": 0,
                "status": "PREFLIGHT_ONLY_CHILD_NOT_EXECUTED",
                "recombination_content_sha256": config["source_bindings"]["recombination"][
                    "content_sha256"
                ],
            },
            "injected_controls": controls,
            "raw_registry_entries": len(controls) + 1,
            "symbolic_equivalence_classes": len(set(expressions)) + 1,
            "equivalence_rule": (
                "exact frozen expression string for controls; the structurally bound B+E+N "
                "child is one separate unexecuted class"
            ),
        }
    )


def _galaxy_metadata_smoke(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    binding = config["source_bindings"]["galaxy_exposure_receipt"]
    receipt = _validate_binding(root, binding)
    counts = receipt["counts"]
    return {
        "mode": "exposure_receipt_metadata_only",
        "source_artifact_records_read": 1,
        "raw_data_rows_read": 0,
        "raw_target_rows_read": 0,
        "raw_predictor_fields_read": [],
        "raw_target_fields_read": [],
        "reported_already_exposed_objects": counts["exploration_galaxies_fitted"],
        "reported_already_exposed_rows": counts["exploration_points_fitted"],
        "confirmation_rows_opened": 0,
        "new_formula_scores_computed": 0,
        "new_formula_selection_events": 0,
        "limitation": (
            "No row-level galaxy smoke ran because the available consolidated raw packet also "
            "contains later-designated confirmation rows; the adapter itself is exercised only "
            "by injected packets in V1."
        ),
    }


def _xcop_development_smoke(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    from astropy.io import fits

    xcop_config = _read_json(root / str(config["source_bindings"]["xcop_config"]["path"]))
    development = tuple(map(str, xcop_config["population"]["development_clusters_already_exposed"]))
    confirmation = set(
        map(
            str,
            xcop_config["population"]["independent_confirmation_clusters_sealed_until_freeze"],
        )
    )
    if len(development) != 8 or set(development) & confirmation:
        raise SharedTargetBlindEvaluatorError("X-COP development seal changed")
    source_receipt = _validate_binding(root, config["source_bindings"]["xcop_source_receipt"])
    receipt_files = {
        (str(row["cluster"]), str(row["role"])): row for row in source_receipt["files"]
    }
    source_dir = root / str(xcop_config["paths"]["source_dir"])
    raw_dir = source_dir / str(xcop_config["paths"]["raw_dir"])
    density_rows = pressure_rows = temperature_rows = 0
    predictor_digests: list[str] = []
    raw_file_reads = 0
    for object_key in development:
        records = [
            receipt_files.get((object_key, role)) for role in ("density", "pressure", "temperature")
        ]
        if any(row is None for row in records):
            raise SharedTargetBlindEvaluatorError("X-COP development receipt incomplete")
        arrays: dict[str, Any] = {}
        headers: dict[str, Mapping[str, Any]] = {}
        for row in records:
            assert row is not None
            path = raw_dir / str(row["member"])
            if not path.is_file() or _sha256_file(path) != str(row["sha256"]):
                raise SharedTargetBlindEvaluatorError("X-COP development file changed")
            raw_file_reads += 1
            role = str(row["role"])
            hdu_key = {
                "density": "density_hdu",
                "pressure": "sz_pressure_hdu",
                "temperature": "xray_temperature_hdu",
            }[role]
            with fits.open(path, memmap=False) as handle:
                hdu = handle[int(xcop_config["source"][hdu_key])]
                arrays[role] = hdu.data.copy()
                headers[role] = dict(hdu.header)
        density = arrays["density"]
        pressure = arrays["pressure"]
        temperature = arrays["temperature"]
        if list(density.dtype.names or ()) != ["RW_X", "NE", "ERR_NE_LO", "ERR_NE_HI"]:
            raise SharedTargetBlindEvaluatorError("X-COP density schema changed")
        if list(pressure.dtype.names or ()) != ["RW_SZ", "P_SZ", "eP_SZ"]:
            raise SharedTargetBlindEvaluatorError("X-COP pressure schema changed")
        if list(temperature.dtype.names or ()) != ["RW_X", "T_X", "eT_X"]:
            raise SharedTargetBlindEvaluatorError("X-COP temperature schema changed")
        r500 = float(headers["density"]["R500"])
        if not math.isclose(float(headers["pressure"]["R500"]), r500) or not math.isclose(
            float(headers["temperature"]["R500"]), r500
        ):
            raise SharedTargetBlindEvaluatorError("X-COP radial scale changed")
        packet = {
            "density_radius_kpc": np.asarray(density["RW_X"], dtype=np.float64) * r500,
            "ne_cm3": np.asarray(density["NE"], dtype=np.float64),
            "r500_kpc": r500,
        }
        target_arrays = (
            np.asarray(pressure["P_SZ"], dtype=np.float64) * float(headers["pressure"]["P500"]),
            np.asarray(pressure["eP_SZ"], dtype=np.float64) * float(headers["pressure"]["P500"]),
            np.asarray(temperature["T_X"], dtype=np.float64)
            * float(headers["temperature"]["T500"]),
            np.asarray(temperature["eT_X"], dtype=np.float64)
            * float(headers["temperature"]["T500"]),
        )
        if any(np.any(~np.isfinite(values)) or np.any(values <= 0.0) for values in target_arrays):
            raise SharedTargetBlindEvaluatorError("X-COP target finite-schema check failed")
        predictors = adapt_xcop_arrays(packet)
        predictor_digests.append(_sha256_bytes(predictors.astype("<f8").tobytes()))
        density_rows += len(density)
        pressure_rows += len(pressure)
        temperature_rows += len(temperature)
    return {
        "mode": "development_only_finite_schema_smoke",
        "objects_read": len(development),
        "raw_files_read": raw_file_reads,
        "source_table_rows_read": density_rows + pressure_rows + temperature_rows,
        "predictor_source_rows_parsed": density_rows,
        "predictor_rows_consumed_by_adapter": density_rows,
        "raw_target_rows_read": pressure_rows + temperature_rows,
        "row_counts_by_table": {
            "density": density_rows,
            "pressure": pressure_rows,
            "temperature": temperature_rows,
        },
        "raw_predictor_fields_read": [
            "RW_X",
            "NE",
            "ERR_NE_LO",
            "ERR_NE_HI",
            "R500",
        ],
        "raw_target_fields_read": [
            "RW_SZ",
            "P_SZ",
            "eP_SZ",
            "RW_X",
            "T_X",
            "eT_X",
            "P500",
            "T500",
        ],
        "predictor_packet_set_sha256": _sha256_bytes(
            _canonical(sorted(predictor_digests)).encode()
        ),
        "confirmation_objects_opened": 0,
        "forbidden_item59_holdout_rows_opened": 0,
        "inferred_total_mass_fields_read": [],
        "new_formula_scores_computed": 0,
        "new_formula_selection_events": 0,
    }


def _blocked_empirical_domain(mode: str, blocker: str) -> dict[str, Any]:
    payload = {
        "mode": mode,
        "blocker": blocker,
        "eligible_real_packet_exists": False,
        "empirical_rows_read": 0,
        "empirical_target_fields_read": [],
        "empirical_score_key_present": False,
        "new_formula_scores_computed": 0,
        "new_formula_selection_events": 0,
    }
    if "score" in payload or "scores" in payload:
        raise SharedTargetBlindEvaluatorError("blocked empirical score unexpectedly present")
    return payload


def build_real_adapter_smokes(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    validate_empirical_bindings(root, config)
    domains = {
        "galaxy_rotation": _galaxy_metadata_smoke(root, config),
        "xcop_thermodynamic": _xcop_development_smoke(root, config),
        "group_bridge": _blocked_empirical_domain(
            "schema_plus_injected_control_only", "no_eligible_real_group_packet"
        ),
        "lensing_metric": _blocked_empirical_domain(
            "schema_plus_injected_metric_potential_control_only",
            "no_frozen_photon_law_or_direct_lensing_authorization",
        ),
    }
    blocked_pass = all(
        domains[name]["empirical_rows_read"] == 0
        and domains[name]["empirical_score_key_present"] is False
        and "score" not in domains[name]
        and "scores" not in domains[name]
        for name in ("group_bridge", "lensing_metric")
    )
    if not blocked_pass:
        raise SharedTargetBlindEvaluatorError("blocked empirical-domain assertion failed")
    return {
        "domains": domains,
        "group_and_lensing_empirical_scores_absent": blocked_pass,
        "real_formula_scores_computed": 0,
        "real_formula_selection_events": 0,
        "sealed_rows_opened": 0,
        "forbidden_or_independent_rows_opened": 0,
    }


def _source_bindings_for_receipt(root: Path) -> dict[str, Any]:
    files = {
        "config": CONFIG_PATH,
        "implementation": MODULE_PATH,
        "test": TEST_PATH,
        "documentation": DOC_PATH,
    }
    return {
        key: {"path": str(path).replace("\\", "/"), "file_sha256": _sha256_file(root / path)}
        for key, path in files.items()
    }


def build_receipt(root: Path) -> dict[str, Any]:
    root = root.resolve()
    config = load_config(root)

    # This packet is frozen before either injected responses or exposed target fields exist.
    generation_packet = build_generation_packet(config)
    generation_sha = generation_packet["content_sha256"]
    candidate_registry = build_candidate_registry(config, generation_sha)

    synthetic = build_synthetic_controls(config)
    real = build_real_adapter_smokes(root, config)
    if not (
        synthetic["all_parity_pass"]
        and synthetic["all_recovery_pass"]
        and synthetic["all_wrong_law_controls_pass"]
        and real["group_and_lensing_empirical_scores_absent"]
    ):
        raise SharedTargetBlindEvaluatorError("bounded adjudication failed")

    rows = int(config["rows_per_injected_packet"])
    raw_control_count = len(DOMAINS) * len(CONTROL_IDS)
    adapter_execution_bindings = {
        domain: _content_hashed(
            {
                "schema_version": "invariant-adapter-execution-binding-1.0",
                "generation_packet_sha256": generation_sha,
                "adapter_contract_sha256": config["adapters"][domain]["contract_sha256"],
                "synthetic_adapter_output_sha256": synthetic["domains"][domain][
                    "adapter_output_sha256"
                ],
                "real_smoke_record_sha256": _sha256_bytes(
                    _canonical(real["domains"][domain]).encode()
                ),
            }
        )
        for domain in DOMAINS
    }
    payload = {
        "schema_version": "invariant-gravity-shared-target-blind-evaluator-receipt-1.0",
        "evaluator_id": config["evaluator_id"],
        "decision": "PASS_BOUNDED_SCAFFOLD_CONTROLS_ONLY_CHILD_NOT_EXECUTED",
        "execution_scope": {
            "four_scale_infrastructure_exercised": True,
            "real_scientific_cross_scale_evaluation_completed": False,
            "galaxy": "committed_exposure_metadata_only_no_rows_no_scores",
            "cluster": "eight_xcop_development_objects_finite_schema_only_no_scores",
            "group": "schema_and_injected_controls_only_no_empirical_packet",
            "lensing": "schema_and_injected_metric_potential_controls_only_no_empirical_packet",
        },
        "source_bindings": _source_bindings_for_receipt(root),
        "architecture_binding": {
            "parent_registry_content_sha256": config["source_bindings"]["parent_registry"][
                "content_sha256"
            ],
            "recombination_content_sha256": config["source_bindings"]["recombination"][
                "content_sha256"
            ],
            "architecture_id": config["source_bindings"]["recombination"][
                "required_architecture_id"
            ],
            "structural_child_executed": False,
            "structural_child_outcome_scores_computed": 0,
            "structural_child_status": "PREFLIGHT_ONLY_CHILD_NOT_EXECUTED",
        },
        "two_stage_separation": {
            "generation_completed_before_response_access": True,
            "response_fields_read_before_generation": [],
            "phase_order": [
                "structural_binding_validation",
                "predictor_only_generation_packet_freeze",
                "evaluation_candidate_registry_freeze",
                "injected_control_evaluation",
                "exposed_empirical_binding_validation_and_adapter_smoke",
            ],
            "generation_packet": generation_packet,
            "generation_packet_sha256": generation_sha,
            "formula_candidates_generated": 0,
            "post_response_generation_calls": 0,
            "adapters_separate_from_generation": True,
            "adapter_contract_sha256": {
                domain: config["adapters"][domain]["contract_sha256"] for domain in DOMAINS
            },
            "adapter_inputs_hash_bound_to_generation_packet": True,
            "adapter_execution_bindings": adapter_execution_bindings,
        },
        "candidate_registry": candidate_registry,
        "candidate_registry_sha256": candidate_registry["content_sha256"],
        "candidate_registry_frozen_before_response_access": True,
        "candidate_accounting": {
            "ben_structural_architectures_bound": 1,
            "ben_structural_candidates_executed": 0,
            "scientific_formula_candidates_generated": 0,
            "scientific_formula_candidates_evaluated": 0,
            "predeclared_injected_control_raw_candidates": raw_control_count,
            "predeclared_injected_control_equivalence_classes": (
                candidate_registry["symbolic_equivalence_classes"] - 1
            ),
            "candidate_registry_raw_entries": candidate_registry["raw_registry_entries"],
            "candidate_registry_equivalence_classes": candidate_registry[
                "symbolic_equivalence_classes"
            ],
            "equivalence_rule": "exact frozen symbolic expression string across adapters",
            "control_candidates_selected_or_retuned_on_responses": 0,
        },
        "synthetic_injected_controls": synthetic,
        "retrospective_exposed_adapter_smokes": real,
        "per_domain_counts": {
            domain: {
                "synthetic_rows_generated": synthetic["domains"][domain]["rows_generated"],
                "synthetic_injections": len(INJECTION_IDS),
                "synthetic_fixed_controls_per_injection": len(CONTROL_IDS),
                "real_source_rows_read": (
                    real["domains"][domain].get("source_table_rows_read", 0)
                    or real["domains"][domain].get("raw_data_rows_read", 0)
                    or real["domains"][domain].get("empirical_rows_read", 0)
                ),
                "real_target_rows_read": (
                    real["domains"][domain].get("raw_target_rows_read", 0)
                    or real["domains"][domain].get("empirical_rows_read", 0)
                ),
                "real_target_fields_read": (
                    real["domains"][domain].get("raw_target_fields_read", [])
                    or real["domains"][domain].get("empirical_target_fields_read", [])
                ),
                "real_formula_scores_computed": 0,
            }
            for domain in DOMAINS
        },
        "data_seals": {
            "galaxy_row_level_smoke_blocked_to_avoid_consolidated_confirmation_rows": True,
            "xcop_development_objects_only": True,
            "xcop_formerly_sealed_item59_holdout_objects_reused": 0,
            "group_empirical_rows_opened": 0,
            "lensing_empirical_rows_opened": 0,
            "sealed_rows_opened": 0,
            "forbidden_or_independent_rows_opened": 0,
            "inferred_total_mass_fields_read": [],
        },
        "counterexample_policy": {
            "single_counterexample_is_universal_veto": False,
            "failures_are_retained_with_scope_and_data_quality_context": True,
        },
        "compute_accounting": {
            "cpu_parallel_workers_maximum": len(DOMAINS),
            "synthetic_dataset_packets": len(DOMAINS) * len(INJECTION_IDS),
            "synthetic_control_vector_evaluation_calls": len(DOMAINS)
            * len(INJECTION_IDS)
            * len(CONTROL_IDS),
            "synthetic_control_row_predictions": len(DOMAINS)
            * len(INJECTION_IDS)
            * len(CONTROL_IDS)
            * rows,
            "parity_vector_evaluation_calls": len(DOMAINS) * len(CONTROL_IDS),
            "parity_scalar_evaluation_calls": len(DOMAINS) * len(CONTROL_IDS) * rows,
            "real_adapter_calls": len(DOMAINS),
            "real_formula_evaluation_calls": 0,
            "gpu_calls": 0,
            "network_calls": 0,
            "model_calls": 0,
            "paid_calls": 0,
            "production_sweep_launched": False,
        },
        "claims": config["claims"],
        "limitations": [
            "Injected recovery validates plumbing, not any gravity formula.",
            "The galaxy V1 smoke binds exposure metadata only because no exploration-only row packet is available without also opening later-designated confirmation rows.",
            "The X-COP smoke checks already-exposed development schemas and finite values; it performs no formula scoring or selection.",
            "No eligible real group packet is registered.",
            "No frozen photon law or direct-lensing authorization exists.",
            "B+E+N remains a structural preflight child and was neither selected nor retuned here.",
        ],
    }
    return _content_hashed(payload)


def write_receipt(root: Path) -> Path:
    receipt = build_receipt(root)
    path = root / OUTPUT_PATH
    _write_json(path, receipt)
    return path


def check_receipt(root: Path) -> Path:
    path = root / OUTPUT_PATH
    if not path.is_file() or _read_json(path) != build_receipt(root):
        raise SharedTargetBlindEvaluatorError("shared evaluator receipt is absent or changed")
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("write", "check", "status"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    root = args.root.resolve()
    path = write_receipt(root) if args.command == "write" else check_receipt(root)
    receipt = _read_json(path)
    if args.command == "status":
        print(
            _canonical(
                {
                    "decision": receipt["decision"],
                    "content_sha256": receipt["content_sha256"],
                    "output_path": str(OUTPUT_PATH).replace("\\", "/"),
                }
            )
        )
    else:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
