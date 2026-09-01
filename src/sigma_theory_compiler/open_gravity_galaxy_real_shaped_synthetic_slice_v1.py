"""First end-to-end real-source-shaped synthetic discovery slice.

This is deliberately a known-answer infrastructure control.  It converts the
three response-blind PHANGS/THINGS model-lifted primary radial source profiles
into typed scenario packets, injects scale laws and noise without consulting a
velocity response, executes a registered adapter, scores the synthetic
response, and records every matrix cell in the replay ledger.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from sigma_theory_compiler.open_gravity_formula_adapter_registry_v1 import (
    AdapterRegistration,
)
from sigma_theory_compiler.open_gravity_formula_execution_protocol_v1 import (
    BindingStatus,
    EligibilityStatus,
    FormulaExecutionBinding,
    ResourceBounds,
    validate_binding_catalogue,
)
from sigma_theory_compiler.open_gravity_observation_operators_v1 import SeedLineage
from sigma_theory_compiler.open_gravity_real_shaped_synthetic_universe_v2 import (
    build_catalogue,
)
from sigma_theory_compiler.open_gravity_real_shaped_synthetic_universe_v2 import (
    load_config as load_universe_config,
)
from sigma_theory_compiler.open_gravity_synthetic_replay_ledger_v1 import (
    DiscoveryStatus,
    SyntheticReplayLedger,
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
    decide_scenario_eligibility,
    execute_binding_in_process,
    validate_scenario_catalogue,
    validate_scenario_values,
)
from sigma_theory_compiler.sigma_core import SchemaViolation, canonical_sha256

CONFIG_PATH = Path("configs/open_gravity_galaxy_real_shaped_synthetic_slice_v1.json")
OUTPUT_PATH = Path("runs/gravity/open-gravity-galaxy-real-shaped-synthetic-slice-v1/receipt.json")
POPULATION_PATH = Path(
    "runs/gravity/open-gravity-galaxy-real-shaped-synthetic-slice-v1/population.jsonl"
)
_ROOT = Path(__file__).resolve().parents[2]
_PROFILE_SCHEMA = "invariant-open-gravity-phangs-things-model-lifted-3d-source-profiles-1.0"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _repo_path(value: str) -> Path:
    parsed = PurePosixPath(value)
    if parsed.is_absolute() or any(part in {"", ".", ".."} for part in parsed.parts):
        raise SchemaViolation("synthetic slice path escaped repository")
    path = (_ROOT / parsed.as_posix()).resolve()
    if not path.is_relative_to(_ROOT):
        raise SchemaViolation("synthetic slice path escaped repository")
    return path


def load_config() -> dict[str, Any]:
    return json.loads((_ROOT / CONFIG_PATH).read_text(encoding="utf-8"))


def validate_config(config: Mapping[str, Any], *, verify_sources: bool = True) -> None:
    expected = {
        "schema",
        "suite_id",
        "version",
        "claim_class",
        "status",
        "source_profile",
        "source_receipt",
        "object_ids",
        "truth_worlds",
        "candidate_cells",
        "noise_sigma_m_s",
        "likelihood_sigma_floor_m_s",
        "suite_seed",
        "kpc_m",
        "output_directory",
    }
    if set(config) != expected:
        raise SchemaViolation("galaxy synthetic slice config keys changed")
    if config["schema"] != "open-gravity-galaxy-real-shaped-synthetic-slice-1.0":
        raise SchemaViolation("galaxy synthetic slice schema changed")
    if config["claim_class"] != "SYNTHETIC_DIRECTIONAL_SIGNAL":
        raise SchemaViolation("synthetic slice claim ceiling changed")
    if tuple(config["object_ids"]) != ("NGC2903", "NGC3351", "NGC3627"):
        raise SchemaViolation("source-shaped galaxy inventory changed")
    for label in ("source_profile", "source_receipt"):
        binding = config[label]
        if not isinstance(binding, Mapping) or set(binding) != {"path", "sha256"}:
            raise SchemaViolation(f"{label} binding changed")
        path = _repo_path(str(binding["path"]))
        if verify_sources and (not path.is_file() or _file_sha256(path) != binding["sha256"]):
            raise SchemaViolation(f"{label} bytes changed")
    worlds = config["truth_worlds"]
    cells = config["candidate_cells"]
    required_keys = {"truth_world_id", "scale_numerator", "scale_denominator"}
    if any(set(row) != required_keys for row in worlds):
        raise SchemaViolation("truth-world schema changed")
    cell_keys = {"parameter_cell_id", "scale_numerator", "scale_denominator"}
    if any(set(row) != cell_keys for row in cells):
        raise SchemaViolation("parameter-cell schema changed")
    truth_ids = tuple(row["truth_world_id"] for row in worlds)
    cell_ids = tuple(row["parameter_cell_id"] for row in cells)
    if truth_ids != tuple(sorted(set(truth_ids))) or cell_ids != tuple(sorted(set(cell_ids))):
        raise SchemaViolation("truth worlds and parameter cells must be ordered and unique")
    for row in (*worlds, *cells):
        if type(row["scale_numerator"]) is not int or type(row["scale_denominator"]) is not int:
            raise SchemaViolation("scale parameters must be exact integers")
        if row["scale_denominator"] == 0:
            raise SchemaViolation("scale denominator cannot be zero")
    noise = tuple(config["noise_sigma_m_s"])
    if noise != tuple(sorted(set(noise))) or any(
        not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0
        for value in noise
    ):
        raise SchemaViolation("noise grid must be finite, nonnegative, ordered, and unique")
    if config["likelihood_sigma_floor_m_s"] <= 0 or config["kpc_m"] <= 0:
        raise SchemaViolation("physical constants and likelihood floor must be positive")
    if type(config["suite_seed"]) is not int or config["suite_seed"] < 0:
        raise SchemaViolation("suite seed must be a nonnegative integer")
    if _repo_path(str(config["output_directory"])) != (_ROOT / OUTPUT_PATH.parent).resolve():
        raise SchemaViolation("synthetic slice output path changed")


def _binding() -> FormulaExecutionBinding:
    parameter_path = _ROOT / "configs/open_gravity_vector_scale_control_parameters_v1.schema.json"
    adapter_path = _ROOT / "src/sigma_theory_compiler/open_gravity_formula_adapter_registry_v1.py"
    return FormulaExecutionBinding(
        binding_id="binding.vector-scale-control.galaxy.v1",
        formula_id="vector-scale-control",
        formula_version="v1.0.0",
        formula_sha256=_file_sha256(adapter_path),
        status=BindingStatus.EXECUTABLE,
        entrypoint=(
            "sigma_theory_compiler.open_gravity_formula_adapter_registry_v1:vector_scale_control"
        ),
        required_features=("source.vector.acceleration",),
        optional_features=(),
        emitted_features=("prediction.vector.acceleration",),
        domains=("galaxy",),
        geometry_support=("radial-source-slice",),
        time_support=("static",),
        parameter_schema_path="configs/open_gravity_vector_scale_control_parameters_v1.schema.json",
        parameter_schema_sha256=_file_sha256(parameter_path),
        approximation_ceiling="known-answer real-source-shaped synthetic infrastructure control",
        health_gates=("determinism", "dimension", "typed-output"),
        resource_bounds=ResourceBounds(10, 64_000_000, 1_000_000),
    )


def _primary_profile(source: Mapping[str, Any], object_id: str) -> list[dict[str, float]]:
    selected = next((row for row in source["objects"] if row["object_id"] == object_id), None)
    if selected is None:
        raise SchemaViolation("registered galaxy is absent from source profile")
    profile = next(
        (
            row["radial_profile"]
            for row in selected["cell_profiles"]
            if row["cell_id"] == selected["primary_cell_id"]
        ),
        None,
    )
    if profile is None or len(profile) != 60:
        raise SchemaViolation("registered primary radial profile changed")
    return profile


def _source_arrays(profile: list[dict[str, float]], kpc_m: float) -> tuple[np.ndarray, np.ndarray]:
    radius = np.asarray([row["radius_kpc"] * kpc_m for row in profile], dtype=np.float64)
    acceleration = np.zeros((len(profile), 3), dtype=np.float64)
    for index, row in enumerate(profile):
        # Preserve the measured source-profile asymmetry as an in-plane direction,
        # while the exact norm remains the source-only baryonic acceleration.
        angle = 0.5 * float(row["radial_force_rms_asymmetry"])
        magnitude = float(row["g_b_m_s2"])
        acceleration[index, 0] = -magnitude * np.cos(angle)
        acceleration[index, 1] = magnitude * np.sin(angle)
    if not np.all(np.diff(radius) > 0) or not np.all(np.isfinite(acceleration)):
        raise SchemaViolation("source radial profile is not finite and ordered")
    return radius, acceleration


def _scenario(
    *,
    config: Mapping[str, Any],
    object_id: str,
    truth_world: Mapping[str, Any],
    truth_index: int,
    nuisance_draw: int,
    radius: np.ndarray,
    acceleration: np.ndarray,
    response: np.ndarray,
    covariance: np.ndarray,
) -> ScenarioDescriptor:
    object_key = object_id.lower()
    scenario_id = f"galaxy.{object_key}.{truth_world['truth_world_id']}.noise.{nuisance_draw}"
    seed = SeedLineage(
        int(config["suite_seed"]),
        scenario_id,
        object_key,
        str(truth_world["truth_world_id"]),
        nuisance_draw,
        0,
    )
    truth = np.asarray([truth_index], dtype=np.int64)
    artifact = "population.jsonl"
    return ScenarioDescriptor(
        scenario_id=scenario_id,
        object_id=object_key,
        experiment_id="galaxy.synthetic.v2",
        domain="galaxy",
        geometry_mode="radial-source-slice",
        time_mode="static",
        coordinate_frame="source",
        axes=(
            AxisSpec("component", 3, None, None),
            AxisSpec("object", 1, None, None),
            AxisSpec("radial_bin", len(radius), "source.scalar.radius", array_sha256(radius)),
        ),
        formula_features=(
            FeatureValueRef(
                "source.scalar.radius",
                artifact,
                array_sha256(radius),
                "float64",
                radius.shape,
                ("radial_bin",),
                "m",
                "source",
            ),
            FeatureValueRef(
                "source.vector.acceleration",
                artifact,
                array_sha256(acceleration),
                "float64",
                acceleration.shape,
                ("radial_bin", "component"),
                "m s^-2",
                "source",
            ),
        ),
        scoring_responses=(
            FeatureValueRef(
                "response.scalar.circular-speed",
                artifact,
                array_sha256(response),
                "float64",
                response.shape,
                ("radial_bin",),
                "m s^-1",
                "source",
            ),
        ),
        hidden_truth=(
            FeatureValueRef(
                "truth.scalar.injection-id",
                artifact,
                array_sha256(truth),
                "int64",
                truth.shape,
                ("object",),
                "typed hidden value",
                "latent",
            ),
        ),
        expected_predictions=(
            EmittedPredictionSpec(
                "prediction.vector.acceleration",
                artifact,
                "float64",
                acceleration.shape,
                ("radial_bin", "component"),
                "m s^-2",
                "source",
            ),
        ),
        uncertainties=(
            UncertaintyRef(
                "circular-speed.covariance",
                "response.scalar.circular-speed",
                "diagonal-covariance",
                artifact,
                array_sha256(covariance),
            ),
        ),
        anchors=(
            AnchorBinding(
                "galaxy.source-profile.v1",
                str(config["source_profile"]["path"]),
                str(config["source_profile"]["sha256"]),
            ),
            AnchorBinding(
                "galaxy.source-receipt.v1",
                str(config["source_receipt"]["path"]),
                str(config["source_receipt"]["sha256"]),
            ),
        ),
        seed_lineage=seed,
    )


def derive_release() -> tuple[dict[str, Any], bytes]:
    config = load_config()
    validate_config(config)
    source = json.loads(_repo_path(config["source_profile"]["path"]).read_text(encoding="utf-8"))
    if source.get("schema") != _PROFILE_SCHEMA:
        raise SchemaViolation("source profile schema changed")
    if tuple(row["object_id"] for row in source["objects"]) != tuple(config["object_ids"]):
        raise SchemaViolation("source profile object ordering changed")

    universe_config = load_universe_config()
    catalogue = build_catalogue(universe_config)
    binding = _binding()
    validate_binding_catalogue((binding,), catalogue)
    registration = AdapterRegistration.create("adapter.vector-scale-control.galaxy.v1", binding)
    module_sha = _file_sha256(Path(__file__))
    operator_sha = _file_sha256(
        _ROOT / "src/sigma_theory_compiler/open_gravity_observation_operators_v1.py"
    )
    release = SyntheticSuiteRelease(
        suite_id=str(config["suite_id"]),
        version=str(config["version"]),
        release_sha256=canonical_sha256(
            {
                "config_raw_sha256": _file_sha256(_ROOT / CONFIG_PATH),
                "source_profile_sha256": config["source_profile"]["sha256"],
                "generator_sha256": module_sha,
            }
        ),
        ontology_sha256=catalogue.content_sha256,
        generator_sha256=module_sha,
        observation_operator_sha256=operator_sha,
        changed_feature_ids=(
            "prediction.vector.acceleration",
            "response.scalar.circular-speed",
            "source.scalar.radius",
            "source.vector.acceleration",
        ),
        change_level="MINOR",
        response_calibrated=False,
        prediction_semantics_changed=True,
    )

    ledger = SyntheticReplayLedger("gravity.synthetic.galaxy-slice.replays", ())
    rows: list[dict[str, Any]] = []
    recoveries: dict[str, list[bool]] = {
        str(index): [] for index, _ in enumerate(config["noise_sigma_m_s"])
    }
    for object_id in config["object_ids"]:
        radius, acceleration = _source_arrays(
            _primary_profile(source, object_id), float(config["kpc_m"])
        )
        acceleration_norm = np.linalg.norm(acceleration, axis=1)
        for truth_index, truth_world in enumerate(config["truth_worlds"]):
            truth_scale = truth_world["scale_numerator"] / truth_world["scale_denominator"]
            noiseless = np.sqrt(radius * acceleration_norm * truth_scale)
            for nuisance_draw, noise_sigma in enumerate(config["noise_sigma_m_s"]):
                object_key = object_id.lower()
                scenario_id = (
                    f"galaxy.{object_key}.{truth_world['truth_world_id']}.noise.{nuisance_draw}"
                )
                lineage = SeedLineage(
                    int(config["suite_seed"]),
                    scenario_id,
                    object_key,
                    str(truth_world["truth_world_id"]),
                    nuisance_draw,
                    0,
                )
                noise = np.random.default_rng(lineage.derived_seed).normal(
                    0.0, float(noise_sigma), size=noiseless.shape
                )
                response = np.maximum(noiseless + noise, 0.0).astype(np.float64)
                likelihood_sigma = max(
                    float(noise_sigma), float(config["likelihood_sigma_floor_m_s"])
                )
                covariance = np.full(response.shape, likelihood_sigma**2, dtype=np.float64)
                scenario = _scenario(
                    config=config,
                    object_id=object_id,
                    truth_world=truth_world,
                    truth_index=truth_index,
                    nuisance_draw=nuisance_draw,
                    radius=radius,
                    acceleration=acceleration,
                    response=response,
                    covariance=covariance,
                )
                validate_scenario_catalogue(scenario, catalogue)
                truth_value = np.asarray([truth_index], dtype=np.int64)
                validate_scenario_values(
                    scenario,
                    formula_values={
                        "source.scalar.radius": radius,
                        "source.vector.acceleration": acceleration,
                    },
                    response_values={"response.scalar.circular-speed": response},
                    truth_values={"truth.scalar.injection-id": truth_value},
                    uncertainty_values={"circular-speed.covariance": covariance},
                )
                decision = decide_scenario_eligibility(binding, catalogue, scenario)
                if decision.status is not EligibilityStatus.ELIGIBLE:
                    raise SchemaViolation("known-answer control unexpectedly became ineligible")
                candidate_rows: list[dict[str, Any]] = []
                execution_by_cell: dict[str, Any] = {}
                for cell in config["candidate_cells"]:
                    result = execute_binding_in_process(
                        binding,
                        catalogue,
                        scenario,
                        {"source.vector.acceleration": acceleration},
                        {
                            "scale_denominator": cell["scale_denominator"],
                            "scale_numerator": cell["scale_numerator"],
                        },
                    )
                    predicted_acceleration = result.output_values["prediction.vector.acceleration"]
                    predicted_speed = np.sqrt(
                        radius * np.linalg.norm(predicted_acceleration, axis=1)
                    )
                    residual = response - predicted_speed
                    candidate_rows.append(
                        {
                            "parameter_cell_id": cell["parameter_cell_id"],
                            "rmse_m_s": float(np.sqrt(np.mean(residual**2))),
                            "chi2": float(np.sum(residual**2 / covariance)),
                            "prediction": {
                                "artifact": result.output_predictions[
                                    "prediction.vector.acceleration"
                                ].to_dict(),
                                "value": predicted_acceleration.tolist(),
                            },
                            "prediction_root_sha256": result.output_sha256,
                        }
                    )
                    execution_by_cell[cell["parameter_cell_id"]] = result
                candidate_rows.sort(key=lambda row: (row["rmse_m_s"], row["parameter_cell_id"]))
                best = candidate_rows[0]
                recovered = best["parameter_cell_id"] == truth_world["truth_world_id"]
                recoveries[str(nuisance_draw)].append(recovered)
                diagnostics = {
                    "candidate_scores": candidate_rows,
                    "self_injection_recovered": recovered,
                    "known_answer_control": True,
                    "response_calibrated": False,
                }
                metrics = {
                    "best_parameter_cell_id": best["parameter_cell_id"],
                    "best_rmse_m_s": best["rmse_m_s"],
                    "best_chi2": best["chi2"],
                    "truth_parameter_cell_id": truth_world["truth_world_id"],
                }
                ledger = ledger.append(
                    release=release,
                    binding=binding,
                    eligibility=decision,
                    adapter_sha256=registration.adapter_sha256,
                    domain="galaxy",
                    experiment_id="galaxy.synthetic.v2",
                )
                selected_result = execution_by_cell[best["parameter_cell_id"]]
                ledger = ledger.complete_last_eligible(
                    release=release,
                    binding=binding,
                    adapter_sha256=registration.adapter_sha256,
                    domain="galaxy",
                    experiment_id="galaxy.synthetic.v2",
                    status=DiscoveryStatus.AMBIGUOUS_WITH_COMPARATOR,
                    scenario_id=scenario.scenario_id,
                    object_id=scenario.object_id,
                    truth_world_id=str(truth_world["truth_world_id"]),
                    seed_lineage_sha256=canonical_sha256(lineage.to_dict()),
                    nuisance_draw=nuisance_draw,
                    parameter_cell_id=str(best["parameter_cell_id"]),
                    observable_ids=("response.scalar.circular-speed",),
                    result_sha256=selected_result.output_sha256,
                    metrics_sha256=_json_sha256(metrics),
                    diagnostics_sha256=_json_sha256(diagnostics),
                    reason_codes=("known-answer-control", "no-scientific-formula-comparison"),
                )
                rows.append(
                    {
                        "scenario": scenario.to_dict(),
                        "values": {
                            "source.scalar.radius": radius.tolist(),
                            "source.vector.acceleration": acceleration.tolist(),
                            "response.scalar.circular-speed": response.tolist(),
                            "truth.scalar.injection-id": truth_value.tolist(),
                            "uncertainty.circular-speed-variance": covariance.tolist(),
                        },
                        "metrics": metrics,
                        "diagnostics": diagnostics,
                    }
                )

    population_bytes = b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode(
            "utf-8"
        )
        for row in rows
    )
    recovery_by_noise = {
        str(config["noise_sigma_m_s"][int(index)]): {
            "recovered": sum(values),
            "total": len(values),
            "fraction": sum(values) / len(values),
        }
        for index, values in recoveries.items()
    }
    receipt_body = {
        "schema": "open-gravity-galaxy-real-shaped-synthetic-slice-receipt-1.0",
        "suite_id": config["suite_id"],
        "version": config["version"],
        "status": "PASS_ONE_GALAXY_VERTICAL_SLICE_KNOWN_ANSWER_CONTROL",
        "claim_class": "SYNTHETIC_DIRECTIONAL_SIGNAL",
        "scientific_claim": "NONE_INFRASTRUCTURE_AND_POWER_STEERING_ONLY",
        "config_sha256": _file_sha256(_ROOT / CONFIG_PATH),
        "source_profile_sha256": config["source_profile"]["sha256"],
        "source_receipt_sha256": config["source_receipt"]["sha256"],
        "catalogue_sha256": catalogue.content_sha256,
        "binding_sha256": binding.content_sha256,
        "adapter_sha256": registration.adapter_sha256,
        "release": release.to_dict(),
        "population_jsonl_sha256": hashlib.sha256(population_bytes).hexdigest(),
        "population_rows": len(rows),
        "object_count": len(config["object_ids"]),
        "truth_world_count": len(config["truth_worlds"]),
        "noise_level_count": len(config["noise_sigma_m_s"]),
        "parameter_cell_count": len(config["candidate_cells"]),
        "formula_executions": len(rows) * len(config["candidate_cells"]),
        "recovery_by_noise_sigma_m_s": recovery_by_noise,
        "replay_ledger_sha256": ledger.content_sha256,
        "replay_entry_count": len(ledger.entries),
        "source_only_profile_rows_opened": 3 * 60,
        "scientific_response_rows_opened": 0,
        "real_scores_computed": 0,
        "response_calibrated": False,
        "limitations": [
            "known-answer scalar control only",
            "radial source slice rather than full 3D population",
            "no novel gravity formula adapted",
            "synthetic response cannot support or reject a theory",
        ],
    }
    return {
        **receipt_body,
        "content_sha256": _json_sha256(receipt_body),
    }, population_bytes


def _write_identical(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise SchemaViolation(f"existing synthetic artifact differs: {path.name}")
        return "EXISTING_IDENTICAL"
    path.write_bytes(payload)
    return "CREATED"


def build() -> str:
    receipt, population = derive_release()
    population_status = _write_identical(_ROOT / POPULATION_PATH, population)
    receipt_bytes = (json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    receipt_status = _write_identical(_ROOT / OUTPUT_PATH, receipt_bytes)
    return f"{population_status}:{receipt_status}"


def check() -> str:
    expected_receipt, expected_population = derive_release()
    if (_ROOT / POPULATION_PATH).read_bytes() != expected_population:
        raise SchemaViolation("stored synthetic population differs from deterministic replay")
    stored_receipt = json.loads((_ROOT / OUTPUT_PATH).read_text(encoding="utf-8"))
    if stored_receipt != expected_receipt:
        raise SchemaViolation("stored synthetic receipt differs from deterministic replay")
    return "VALID"


__all__ = [
    "CONFIG_PATH",
    "OUTPUT_PATH",
    "POPULATION_PATH",
    "build",
    "check",
    "derive_release",
    "load_config",
    "validate_config",
]
