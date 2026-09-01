"""Mechanism-specific successor to the Lane6 full-3D adapter preflight.

The predecessor's three Newton-only scenarios remain immutable.  This
successor emits one response-blind known-answer scenario for every
object/mechanism pair and executes every frozen candidate through the common
typed ABI in every scenario.  Synthetic response generation is kept distinct
from candidate execution and no scientific response is opened.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from sigma_theory_compiler import (
    open_gravity_full3d_phangs_synthetic_adapter_preflight_v1 as predecessor,
)
from sigma_theory_compiler.open_gravity_formula_execution_protocol_v1 import (
    EligibilityStatus,
)
from sigma_theory_compiler.open_gravity_observation_operators_v1 import SeedLineage
from sigma_theory_compiler.open_gravity_synthetic_replay_ledger_v1 import (
    DiscoveryStatus,
    SyntheticReplayLedger,
    SyntheticSuiteRelease,
)
from sigma_theory_compiler.open_gravity_synthetic_scenario_packet_v1 import (
    AnchorBinding,
    ScenarioDescriptor,
    array_sha256,
    decide_scenario_eligibility,
    execute_binding_in_process,
    validate_scenario_catalogue,
    validate_scenario_values,
)
from sigma_theory_compiler.sigma_core import SchemaViolation, canonical_sha256

CONFIG_PATH = Path("configs/open_gravity_full3d_phangs_synthetic_injection_matrix_v2.json")
OUTPUT_DIR = Path("runs/gravity/open-gravity-full3d-phangs-synthetic-injection-matrix-v2")
VALUES_PATH = OUTPUT_DIR / "values.jsonl"
SCENARIOS_PATH = OUTPUT_DIR / "scenarios.jsonl"
LEDGER_PATH = OUTPUT_DIR / "ledger.json"
RECEIPT_PATH = OUTPUT_DIR / "receipt.json"
_ROOT = Path(__file__).resolve().parents[2]


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
        raise SchemaViolation("injection-matrix path escaped repository")
    path = (_ROOT / parsed.as_posix()).resolve()
    if not path.is_relative_to(_ROOT):
        raise SchemaViolation("injection-matrix path escaped repository")
    return path


def load_config() -> dict[str, Any]:
    return json.loads((_ROOT / CONFIG_PATH).read_text(encoding="utf-8"))


def validate_config(config: Mapping[str, Any], *, verify_predecessor: bool = True) -> None:
    expected = {
        "schema",
        "package_id",
        "version",
        "status",
        "claim_class",
        "experiment_id",
        "suite_seed",
        "predecessor",
        "scenario_count",
        "candidate_execution_count",
        "mechanism_count",
        "object_count",
        "output_directory",
        "access_contract",
    }
    if set(config) != expected:
        raise SchemaViolation("injection-matrix config keys changed")
    if config["schema"] != "open-gravity-full3d-phangs-synthetic-injection-matrix-2.0":
        raise SchemaViolation("injection-matrix schema changed")
    if config["package_id"] != "open-gravity-full3d-phangs-synthetic-injection-matrix-v2":
        raise SchemaViolation("injection-matrix package changed")
    if config["version"] != "v2.0.0":
        raise SchemaViolation("injection-matrix version changed")
    if config["claim_class"] != "SYNTHETIC_DIRECTIONAL_SIGNAL":
        raise SchemaViolation("injection-matrix claim ceiling changed")
    if config["experiment_id"] != predecessor._EXPERIMENT:
        raise SchemaViolation("injection-matrix experiment changed")
    if type(config["suite_seed"]) is not int or config["suite_seed"] < 0:
        raise SchemaViolation("injection-matrix suite seed changed")
    if (
        config["scenario_count"] != 27
        or config["candidate_execution_count"] != 243
        or config["mechanism_count"] != 9
        or config["object_count"] != 3
    ):
        raise SchemaViolation("injection-matrix cardinality contract changed")
    if _repo_path(config["output_directory"]) != (_ROOT / OUTPUT_DIR).resolve():
        raise SchemaViolation("injection-matrix output directory changed")
    if any(config["access_contract"].values()):
        raise SchemaViolation("injection-matrix response-blind boundary changed")
    if set(config["predecessor"]) != {"config", "module", "receipt", "packets", "ledger"}:
        raise SchemaViolation("injection-matrix predecessor binding changed")
    for label, binding in config["predecessor"].items():
        if set(binding) != {"path", "sha256"}:
            raise SchemaViolation("injection-matrix predecessor binding schema changed")
        if verify_predecessor:
            path = _repo_path(binding["path"])
            if not path.is_file() or _file_sha256(path) != binding["sha256"]:
                raise SchemaViolation(f"injection-matrix predecessor {label} changed")


def _predecessor_evidence(config: Mapping[str, Any]):
    source_config = predecessor.load_config()
    predecessor.validate_config(source_config, verify_sources=False)
    receipt = json.loads(_repo_path(config["predecessor"]["receipt"]["path"]).read_text())
    if (
        receipt["status"]
        != "PASS_SOURCE_AND_NINE_COMMON_ABI_ADAPTER_PREFLIGHT_TARGET_FREE_CONFUSION_ONLY"
        or receipt["object_count"] != 3
        or receipt["mechanism_count"] != 9
        or receipt["predecessor_source_gate_failures_retained"] != 1
    ):
        raise SchemaViolation("injection-matrix predecessor receipt semantics changed")
    packet_rows = tuple(
        json.loads(line)
        for line in _repo_path(config["predecessor"]["packets"]["path"])
        .read_text(encoding="utf-8")
        .splitlines()
    )
    if len(packet_rows) != 3:
        raise SchemaViolation("injection-matrix predecessor packet count changed")
    return source_config, receipt, packet_rows


def _source_values(packet: Mapping[str, Any]) -> dict[str, np.ndarray]:
    references = {row["element_id"]: row for row in packet["scenario"]["formula_features"]}
    if set(references) != set(predecessor._FORMULA_FEATURES):
        raise SchemaViolation("injection-matrix source feature inventory changed")
    values = {}
    for element_id in predecessor._FORMULA_FEATURES:
        reference = references[element_id]
        value = np.asarray(packet["values"][element_id], dtype=reference["dtype"])
        if (
            list(value.shape) != reference["shape"]
            or array_sha256(value) != reference["value_sha256"]
        ):
            raise SchemaViolation("injection-matrix predecessor source value changed")
        values[element_id] = value
    return values


def _truth_world_id(mechanism_id: str) -> str:
    return f"truth.{mechanism_id.lower()}"


def _scenario_id(object_id: str, mechanism_id: str) -> str:
    return f"galaxy.full3d.{object_id.lower()}.primary.{_truth_world_id(mechanism_id)}.v2"


def _truth_response(mechanism_id: str, values: Mapping[str, np.ndarray]) -> np.ndarray:
    entrypoint = getattr(predecessor, predecessor._MECHANISM_ENTRYPOINTS[mechanism_id])
    response = np.asarray(
        entrypoint(values, {})["prediction.vector.acceleration"], dtype=np.float64
    )
    if response.shape != (17, 17, 17, 3) or not np.all(np.isfinite(response)):
        raise SchemaViolation("mechanism-specific truth generator emitted an invalid response")
    return response


def _mechanism_scenario(
    *,
    packet: Mapping[str, Any],
    source_values: Mapping[str, np.ndarray],
    response: np.ndarray,
    variance: np.ndarray,
    mechanism_id: str,
    injection_id: int,
    source_config: Mapping[str, Any],
    suite_seed: int,
) -> ScenarioDescriptor:
    object_id = packet["object_id"]
    scenario_id = _scenario_id(object_id, mechanism_id)
    base_scenario = predecessor._scenario(
        {"object_id": object_id},
        source_values,
        response,
        variance,
        injection_id,
        source_config,
    )
    feature_refs = tuple(
        replace(
            reference,
            artifact_path=predecessor.PACKETS_PATH.as_posix(),
        )
        for reference in base_scenario.formula_features
    )
    response_refs = tuple(
        replace(reference, artifact_path=VALUES_PATH.as_posix())
        for reference in base_scenario.scoring_responses
    )
    truth_refs = tuple(
        replace(reference, artifact_path=SCENARIOS_PATH.as_posix())
        for reference in base_scenario.hidden_truth
    )
    prediction_specs = tuple(
        replace(specification, artifact_path=VALUES_PATH.as_posix())
        for specification in base_scenario.expected_predictions
    )
    uncertainty_refs = tuple(
        replace(reference, artifact_path=VALUES_PATH.as_posix())
        for reference in base_scenario.uncertainties
    )
    anchors = tuple(
        sorted(
            (
                *base_scenario.anchors,
                AnchorBinding(
                    "full3d.predecessor-packets.v1",
                    predecessor.PACKETS_PATH.as_posix(),
                    "fa3645648bc5ac01217b042d3e190f8f2687b869c44678bc12fb3069a4d6e076",
                ),
            ),
            key=lambda row: row.anchor_id,
        )
    )
    return replace(
        base_scenario,
        scenario_id=scenario_id,
        formula_features=feature_refs,
        scoring_responses=response_refs,
        hidden_truth=truth_refs,
        expected_predictions=prediction_specs,
        uncertainties=uncertainty_refs,
        anchors=anchors,
        seed_lineage=SeedLineage(
            suite_seed,
            scenario_id,
            object_id.lower(),
            _truth_world_id(mechanism_id),
            0,
            0,
        ),
    )


def _typed_value_artifacts(
    mechanism_id: str, response: np.ndarray
) -> tuple[dict[str, Any], dict[str, Any]]:
    common = {
        "artifact_path": VALUES_PATH.as_posix(),
        "dtype": "float64",
        "shape": [17, 17, 17, 3],
        "axes": ["x", "y", "z", "component"],
        "unit": "m s^-2",
        "frame": "solver-source",
        "value_sha256": array_sha256(response),
    }
    response_artifact = {
        "element_id": "response.vector.synthetic-acceleration",
        **common,
    }
    prediction_artifact = {
        "element_id": "prediction.vector.acceleration",
        **common,
    }
    if mechanism_id not in predecessor._MECHANISM_ENTRYPOINTS:
        raise SchemaViolation("typed value artifact has an unknown mechanism")
    return response_artifact, prediction_artifact


def _distance(candidate: np.ndarray, response: np.ndarray) -> float:
    scale = max(float(np.sqrt(np.mean(response * response))), 1.0e-30)
    return float(np.sqrt(np.mean((candidate - response) ** 2)) / scale)


def derive_release() -> tuple[dict[str, Any], bytes, bytes, bytes]:
    config = load_config()
    validate_config(config)
    source_config, predecessor_receipt, source_packets = _predecessor_evidence(config)
    catalogue = predecessor._catalogue(source_config)
    bindings = predecessor._bindings(source_config)
    registrations = tuple(
        predecessor.AdapterRegistration.create(
            f"adapter.full3d.{binding.formula_id.lower()}.v1", binding
        )
        for binding in bindings
    )
    predecessor.validate_binding_catalogue(bindings, catalogue)
    predecessor.validate_adapter_registry(registrations)
    registration_by_formula = {row.formula_binding.formula_id: row for row in registrations}
    mechanisms = tuple(sorted(predecessor._MECHANISM_ENTRYPOINTS))
    module_sha = _file_sha256(Path(__file__))
    release = SyntheticSuiteRelease(
        suite_id=config["package_id"],
        version=config["version"],
        release_sha256=canonical_sha256(
            {
                "config_raw_sha256": _file_sha256(_ROOT / CONFIG_PATH),
                "generator_sha256": module_sha,
                "predecessor_receipt_sha256": config["predecessor"]["receipt"]["sha256"],
            }
        ),
        ontology_sha256=catalogue.content_sha256,
        generator_sha256=module_sha,
        observation_operator_sha256=canonical_sha256(
            {
                "operator": "mechanism-specific-zero-noise-known-answer",
                "truth_worlds": list(mechanisms),
            }
        ),
        changed_feature_ids=(
            "prediction.vector.acceleration",
            "response.vector.synthetic-acceleration",
            "truth.scalar.injection-id",
        ),
        change_level="MAJOR",
        response_calibrated=False,
        prediction_semantics_changed=True,
    )
    ledger = SyntheticReplayLedger("gravity.synthetic.full3d-phangs-injection-matrix.v2", ())
    value_rows = []
    scenario_rows = []
    scenario_hashes: dict[str, str] = {}
    response_hashes: dict[str, str] = {}
    recovery_count = 0
    execution_cell_hashes: set[str] = set()

    for source_packet in source_packets:
        object_id = source_packet["object_id"]
        source_values = _source_values(source_packet)
        truth_responses = {
            mechanism_id: _truth_response(mechanism_id, source_values)
            for mechanism_id in mechanisms
        }
        variance = np.full((17, 17, 17, 3), (predecessor._A0 * 1.0e-3) ** 2, dtype=np.float64)
        truth_value_rows = []
        for injection_id, mechanism_id in enumerate(mechanisms):
            response = truth_responses[mechanism_id]
            response_artifact, prediction_artifact = _typed_value_artifacts(mechanism_id, response)
            truth_value_rows.append(
                {
                    "mechanism_id": mechanism_id,
                    "truth_world_id": _truth_world_id(mechanism_id),
                    "injection_id": injection_id,
                    "response_artifact": response_artifact,
                    "prediction_artifact": prediction_artifact,
                    "value": response.tolist(),
                }
            )
        uncertainty_artifact = {
            "uncertainty_id": "synthetic-acceleration.diagonal-covariance",
            "applies_to_element_id": "response.vector.synthetic-acceleration",
            "representation": "diagonal-covariance",
            "artifact_path": VALUES_PATH.as_posix(),
            "dtype": "float64",
            "shape": [17, 17, 17, 3],
            "axes": ["x", "y", "z", "component"],
            "unit": "m2 s^-4",
            "frame": "solver-source",
            "value_sha256": array_sha256(variance),
        }
        value_rows.append(
            {
                "object_id": object_id,
                "source_packet": {
                    "path": predecessor.PACKETS_PATH.as_posix(),
                    "scenario_id": source_packet["scenario"]["scenario_id"],
                    "sealed_density_sha256": source_packet["sealed_density_sha256"],
                },
                "uncertainty_artifact": uncertainty_artifact,
                "uncertainty_value": variance.tolist(),
                "truth_responses": truth_value_rows,
            }
        )

        for injection_id, truth_mechanism_id in enumerate(mechanisms):
            response = truth_responses[truth_mechanism_id]
            truth_value = np.asarray([injection_id], dtype=np.int64)
            scenario = _mechanism_scenario(
                packet=source_packet,
                source_values=source_values,
                response=response,
                variance=variance,
                mechanism_id=truth_mechanism_id,
                injection_id=injection_id,
                source_config=source_config,
                suite_seed=config["suite_seed"],
            )
            validate_scenario_catalogue(scenario, catalogue)
            validate_scenario_values(
                scenario,
                formula_values=source_values,
                response_values={"response.vector.synthetic-acceleration": response},
                truth_values={"truth.scalar.injection-id": truth_value},
                uncertainty_values={"synthetic-acceleration.diagonal-covariance": variance},
            )
            scenario_hashes[scenario.scenario_id] = scenario.content_sha256
            response_hashes[scenario.scenario_id] = array_sha256(response)
            candidate_results = []
            candidate_values: dict[str, np.ndarray] = {}
            for binding in bindings:
                decision = decide_scenario_eligibility(binding, catalogue, scenario)
                if decision.status is not EligibilityStatus.ELIGIBLE:
                    raise SchemaViolation("mechanism-specific scenario became ineligible")
                result = execute_binding_in_process(
                    binding,
                    catalogue,
                    scenario,
                    {
                        element_id: source_values[element_id]
                        for element_id in predecessor._FORMULA_FEATURES
                    },
                    {},
                )
                prediction = result.output_values["prediction.vector.acceleration"]
                expected_prediction = truth_responses[binding.formula_id]
                if array_sha256(prediction) != array_sha256(
                    expected_prediction
                ) or not np.array_equal(prediction, expected_prediction):
                    raise SchemaViolation("candidate execution differs from frozen value bank")
                candidate_values[binding.formula_id] = prediction
                metric = {
                    "relative_rmse_to_mechanism_generated_response": _distance(prediction, response)
                }
                diagnostics = {
                    "candidate_formula_id": binding.formula_id,
                    "mechanism_specific_synthetic_response": True,
                    "real_response_used": False,
                    "scenario_id": scenario.scenario_id,
                    "truth_world_id": _truth_world_id(truth_mechanism_id),
                }
                ledger = ledger.append(
                    release=release,
                    binding=binding,
                    eligibility=decision,
                    adapter_sha256=registration_by_formula[binding.formula_id].adapter_sha256,
                    domain="galaxy",
                    experiment_id=config["experiment_id"],
                )
                ledger = ledger.complete_last_eligible(
                    release=release,
                    binding=binding,
                    adapter_sha256=registration_by_formula[binding.formula_id].adapter_sha256,
                    domain="galaxy",
                    experiment_id=config["experiment_id"],
                    status=DiscoveryStatus.AMBIGUOUS_WITH_COMPARATOR,
                    scenario_id=scenario.scenario_id,
                    object_id=scenario.object_id,
                    truth_world_id=_truth_world_id(truth_mechanism_id),
                    seed_lineage_sha256=canonical_sha256(scenario.seed_lineage.to_dict()),
                    nuisance_draw=0,
                    parameter_cell_id="frozen-no-free-parameters",
                    observable_ids=("response.vector.synthetic-acceleration",),
                    result_sha256=result.output_sha256,
                    metrics_sha256=_json_sha256(metric),
                    diagnostics_sha256=_json_sha256(diagnostics),
                    reason_codes=(
                        "mechanism-specific-synthetic-response",
                        "response-blind",
                    ),
                )
                completed_entry = ledger.entries[-1]
                execution_cell_sha256 = canonical_sha256(
                    {
                        "scenario_sha256": scenario.content_sha256,
                        "truth_world_id": _truth_world_id(truth_mechanism_id),
                        "candidate_formula_id": binding.formula_id,
                        "binding_sha256": binding.content_sha256,
                        "result_sha256": result.output_sha256,
                        "response_value_sha256": array_sha256(response),
                        "metrics_sha256": _json_sha256(metric),
                        "diagnostics_sha256": _json_sha256(diagnostics),
                        "completed_ledger_entry_sha256": completed_entry.entry_sha256,
                    }
                )
                if execution_cell_sha256 in execution_cell_hashes:
                    raise SchemaViolation("mechanism-specific execution cell hash repeated")
                execution_cell_hashes.add(execution_cell_sha256)
                candidate_results.append(
                    {
                        "candidate_formula_id": binding.formula_id,
                        "binding_sha256": binding.content_sha256,
                        "adapter_sha256": registration_by_formula[
                            binding.formula_id
                        ].adapter_sha256,
                        "artifact": result.output_predictions[
                            "prediction.vector.acceleration"
                        ].to_dict(),
                        "value_locator": {
                            "path": VALUES_PATH.as_posix(),
                            "object_id": object_id,
                            "mechanism_id": binding.formula_id,
                        },
                        "value_sha256": array_sha256(prediction),
                        "output_sha256": result.output_sha256,
                        "metrics": metric,
                        "diagnostics": diagnostics,
                        "completed_ledger_sequence": completed_entry.sequence,
                        "completed_ledger_entry_sha256": completed_entry.entry_sha256,
                        "execution_cell_sha256": execution_cell_sha256,
                    }
                )
            distances = [
                {
                    "candidate_formula_id": candidate_id,
                    "relative_rmse": _distance(candidate_values[candidate_id], response),
                }
                for candidate_id in sorted(candidate_values)
            ]
            minimum = min(row["relative_rmse"] for row in distances)
            winners = sorted(
                row["candidate_formula_id"]
                for row in distances
                if math.isclose(row["relative_rmse"], minimum, abs_tol=1.0e-15, rel_tol=0.0)
            )
            truth_recovered = truth_mechanism_id in winners
            recovery_count += int(truth_recovered)
            scenario_rows.append(
                {
                    "object_id": object_id,
                    "scenario_sha256": scenario.content_sha256,
                    "scenario": scenario.to_dict(),
                    "truth_world": {
                        "mechanism_id": truth_mechanism_id,
                        "truth_world_id": _truth_world_id(truth_mechanism_id),
                        "injection_id": injection_id,
                        "truth_value": truth_value.tolist(),
                        "truth_value_sha256": array_sha256(truth_value),
                        "response_value_locator": {
                            "path": VALUES_PATH.as_posix(),
                            "object_id": object_id,
                            "mechanism_id": truth_mechanism_id,
                        },
                        "response_value_sha256": array_sha256(response),
                        "uncertainty_value_sha256": array_sha256(variance),
                    },
                    "candidate_executions": candidate_results,
                    "injection_recovery": {
                        "comparison": "candidate-field-to-mechanism-generated-response",
                        "candidate_distances": distances,
                        "winner_formula_ids": winners,
                        "truth_formula_id": truth_mechanism_id,
                        "truth_recovered": truth_recovered,
                    },
                }
            )

    if (
        len(value_rows) != config["object_count"]
        or len(scenario_rows) != config["scenario_count"]
        or len(execution_cell_hashes) != config["candidate_execution_count"]
        or len(ledger.entries) != 2 * config["candidate_execution_count"]
    ):
        raise SchemaViolation("mechanism-specific injection matrix is incomplete")
    values_bytes = b"".join(_json_bytes(row) + b"\n" for row in value_rows)
    scenarios_bytes = b"".join(_json_bytes(row) + b"\n" for row in scenario_rows)
    ledger_bytes = _json_bytes(ledger.to_dict(), indent=2)
    receipt_body = {
        "schema": "open-gravity-full3d-phangs-synthetic-injection-matrix-receipt-2.0",
        "package_id": config["package_id"],
        "version": config["version"],
        "status": "SEALED_MECHANISM_SPECIFIC_MATRIX_AWAITING_INDEPENDENT_AUDIT",
        "claim_class": config["claim_class"],
        "scientific_claim": "NONE_SYNTHETIC_POWER_AND_CONFUSION_STEERING_ONLY",
        "object_count": len(value_rows),
        "mechanism_count": len(mechanisms),
        "scenario_count": len(scenario_rows),
        "truth_generation_evaluations": len(scenario_rows),
        "candidate_common_abi_executions": len(execution_cell_hashes),
        "completed_replay_cells": len(execution_cell_hashes),
        "replay_entry_count": len(ledger.entries),
        "confusion_matrix_cells": sum(
            len(row["injection_recovery"]["candidate_distances"]) for row in scenario_rows
        ),
        "recovery_row_count": len(scenario_rows),
        "truth_recovered_count": recovery_count,
        "distinct_scenario_hash_count": len(set(scenario_hashes.values())),
        "distinct_execution_cell_hash_count": len(execution_cell_hashes),
        "mechanism_ids": list(mechanisms),
        "scenario_hashes": scenario_hashes,
        "response_value_hashes": response_hashes,
        "config_raw_sha256": _file_sha256(_ROOT / CONFIG_PATH),
        "module_raw_sha256": module_sha,
        "catalogue_sha256": catalogue.content_sha256,
        "release": release.to_dict(),
        "predecessor": config["predecessor"],
        "transitive_source_bindings": predecessor_receipt["source_bindings"],
        "raw_fits_inventory": predecessor_receipt["raw_fits_inventory"],
        "source_density_hashes": predecessor_receipt["source_density_hashes"],
        "predecessor_source_gate_failures_retained": predecessor_receipt[
            "predecessor_source_gate_failures_retained"
        ],
        "values_jsonl_sha256": hashlib.sha256(values_bytes).hexdigest(),
        "scenarios_jsonl_sha256": hashlib.sha256(scenarios_bytes).hexdigest(),
        "ledger_json_sha256": hashlib.sha256(ledger_bytes).hexdigest(),
        "access_accounting": {
            "scientific_response_files_opened": 0,
            "scientific_response_rows_opened": 0,
            "lensing_response_files_opened": 0,
            "real_scores_computed": 0,
            "response_calibrated": False,
            "predecessor_synthetic_packet_files_opened": 1,
            "source_fits_files_opened_by_successor": 0,
            "source_fits_files_bound_transitively": predecessor_receipt["raw_fits_inventory"][
                "count"
            ],
        },
        "blocks": [
            "no velocity or lensing response opened",
            "model-lifted 2.5D source in a full-3D field grid is not measured 3D mass",
            "DPEL01 retains its target-free periodic boundary; isolated empirical boundary remains blocked",
            "the retained NGC2903 predecessor primary-source numerical gate failure remains visible",
            "synthetic mechanism recovery cannot support or reject a gravity theory",
        ],
    }
    receipt = {
        **receipt_body,
        "content_sha256": canonical_sha256(receipt_body),
    }
    return receipt, values_bytes, scenarios_bytes, ledger_bytes


def _write_once(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise SchemaViolation(
                f"refusing to overwrite changed injection-matrix artifact: {path}"
            )
        return "EXISTING_IDENTICAL"
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
    return "CREATED"


def build() -> str:
    receipt, values, scenarios, ledger = derive_release()
    statuses = (
        _write_once(_ROOT / VALUES_PATH, values),
        _write_once(_ROOT / SCENARIOS_PATH, scenarios),
        _write_once(_ROOT / LEDGER_PATH, ledger),
        _write_once(_ROOT / RECEIPT_PATH, _json_bytes(receipt, indent=2)),
    )
    return ":".join(statuses)


def check() -> None:
    receipt, values, scenarios, ledger = derive_release()
    expected = (
        (_ROOT / VALUES_PATH, values),
        (_ROOT / SCENARIOS_PATH, scenarios),
        (_ROOT / LEDGER_PATH, ledger),
        (_ROOT / RECEIPT_PATH, _json_bytes(receipt, indent=2)),
    )
    for path, payload in expected:
        if not path.is_file() or path.read_bytes() != payload:
            raise SystemExit(f"stored injection-matrix artifact differs: {path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build", "check"))
    arguments = parser.parse_args()
    if arguments.command == "build":
        print(build())
    else:
        check()
        print("OK")
