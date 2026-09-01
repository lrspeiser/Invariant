"""Source-coverage successor to the response-blind void/cosmology synthetic matrix."""

from __future__ import annotations

import gzip
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

import sigma_theory_compiler.open_gravity_void_cosmology_source_shaped_synthetic_injection_matrix_v1 as base
from sigma_theory_compiler.open_gravity_formula_adapter_registry_v1 import (
    AdapterRegistration,
    validate_adapter_registry,
)
from sigma_theory_compiler.open_gravity_formula_execution_protocol_v1 import (
    BindingStatus,
    validate_binding_catalogue,
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
from sigma_theory_compiler.open_gravity_synthetic_scenario_packet_v1 import array_sha256
from sigma_theory_compiler.sigma_core import SchemaViolation, canonical_sha256

CONFIG_PATH = Path(
    "configs/open_gravity_void_cosmology_source_shaped_synthetic_injection_matrix_v2.json"
)
TEST_PATH = Path(
    "tests/test_open_gravity_void_cosmology_source_shaped_synthetic_injection_matrix_v2.py"
)
OUTPUT_DIR = Path(
    "runs/gravity/open-gravity-void-cosmology-source-shaped-synthetic-injection-matrix-v2"
)
VALUES_PATH = OUTPUT_DIR / "values.npz"
SCENARIOS_PATH = OUTPUT_DIR / "scenarios.jsonl"
LEDGER_PATH = OUTPUT_DIR / "ledger.json"
CONFUSION_PATH = OUTPUT_DIR / "confusion-matrix.json"
DIAGNOSTICS_PATH = OUTPUT_DIR / "geometry-and-identifiability.json"
RECEIPT_PATH = OUTPUT_DIR / "receipt.json"
_ROOT = Path(__file__).resolve().parents[2]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SchemaViolation(message)


def load_config() -> dict[str, Any]:
    return json.loads((_ROOT / CONFIG_PATH).read_text(encoding="utf-8"))


def validate_config(config: Mapping[str, Any], *, verify_hashes: bool = True) -> None:
    _require(
        set(config)
        == {
            "schema",
            "package_id",
            "version",
            "status",
            "claim_class",
            "output_directory",
            "base_config",
            "predecessor",
            "repair",
            "access_contract",
        },
        "void synthetic v2 config keys changed",
    )
    _require(
        config["schema"]
        == "open-gravity-void-cosmology-source-shaped-synthetic-injection-matrix-2.0"
        and config["package_id"]
        == "open-gravity-void-cosmology-source-shaped-synthetic-injection-matrix-v2"
        and config["version"] == "v1.0.1",
        "void synthetic v2 identity changed",
    )
    _require(
        config["status"] == "FROZEN_SYNTHETIC_ONLY_PRE_RESPONSE"
        and config["claim_class"] == "SYNTHETIC_DIRECTIONAL_SIGNAL",
        "void synthetic v2 boundary changed",
    )
    _require(
        base._repo_path(config["output_directory"]) == (_ROOT / OUTPUT_DIR).resolve(),
        "void synthetic v2 output path changed",
    )
    _require(
        config["repair"]["classification"] == "V1_FAIL_SOURCE_DOMAIN_ALL_ZERO_VOID_EXPOSURE"
        and config["repair"]["v1_zero_exposure_object_count"] == 8
        and config["repair"]["absolute_length_permutation_used"] is False,
        "void synthetic v2 repair changed",
    )
    _require(
        all(value == 0 for value in config["access_contract"].values()),
        "void synthetic v2 response barrier changed",
    )
    base_row = config["base_config"]
    predecessor = config["predecessor"]
    if verify_hashes:
        base_path = base._repo_path(base_row["path"])
        receipt_path = base._repo_path(predecessor["receipt_path"])
        _require(
            base._file_sha256(base_path)
            == base_row["raw_sha256"]
            == predecessor["config_raw_sha256"],
            "void synthetic v1 config drift",
        )
        _require(
            base._file_sha256(base.Path(base.__file__)) == predecessor["module_raw_sha256"],
            "void synthetic v1 module drift",
        )
        _require(
            base._file_sha256(_ROOT / base.TEST_PATH) == predecessor["test_raw_sha256"],
            "void synthetic v1 test drift",
        )
        _require(
            base._file_sha256(receipt_path) == predecessor["receipt_raw_sha256"],
            "void synthetic v1 receipt drift",
        )
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        _require(
            receipt["content_sha256"] == predecessor["receipt_content_sha256"]
            and receipt["distinct_truth_recovery_count"] == 0,
            "void synthetic v1 failure evidence drift",
        )
        base.validate_config(base.load_config())


def _select_nonzero_source_objects(
    base_config: Mapping[str, Any], geometry: Mapping[str, Any], successor: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    anchors = {row["id"]: row for row in base_config["source_anchors"]}
    mask = base._repo_path(anchors["CANONICAL_VAST_ANGULAR_MASK"]["path"]).read_bytes()
    _require(len(mask) == 64800 and set(mask) <= {0, 1}, "canonical mask invalid")
    ledger = base._load_identifier_ledger(base_config)
    source_path = base._repo_path(anchors["CF4_TABLE4_OPAQUE_ROW_CONTAINER"]["path"])
    selected: list[dict[str, Any]] = []
    source_fields_decoded = 0
    raw_rows_read = 0
    stream_offset = 0
    radial_limit = float(base_config["law_constants"]["radial_mask_limit_h_inverse_mpc"])
    h = float(base_config["law_constants"]["planck_h"])
    with gzip.open(source_path, "rb") as handle:
        for source_index, raw in enumerate(handle):
            entry = ledger[source_index]
            raw_rows_read += 1
            payload = raw[:-1] if raw.endswith(b"\n") else raw
            if payload.endswith(b"\r"):
                payload = payload[:-1]
            _require(len(payload) == 157, "invalid CF4 payload length")
            _require(int(entry["framed_start"]) == stream_offset, "CF4 ledger offset mismatch")
            _require(
                hashlib.sha256(raw).hexdigest() == entry["framed_raw_sha256"]
                and hashlib.sha256(payload).hexdigest() == entry["payload_raw_sha256"],
                "CF4 ledger row hash mismatch",
            )
            stream_offset += len(raw)
            identifier = int(payload[:7].decode("ascii").strip())
            bucket, role = base._split_role(identifier)
            _require(
                identifier == int(entry["identifier"])
                and bucket == int(entry["bucket"])
                and role == entry["role"],
                "CF4 identifier partition mismatch",
            )
            if role != "development":
                continue
            source = base._parse_permitted_cf4_source(payload, identifier)
            source_fields_decoded += 1
            ra, dec = float(source["RAdeg"]), float(source["DEdeg"])
            if not base.mask_contains(mask, ra, dec):
                continue
            _, distance_hinv = base.luminosity_to_comoving_hinv(float(source["Dist"]))
            if not 0.0 < distance_hinv <= radial_limit:
                continue
            distance = distance_hinv / h
            direction = base.radec_to_xyz(ra, dec, 1.0)
            planck = base._interval_summary(direction, distance, geometry["spheres"]["Planck2018"])
            wmap = base._interval_summary(direction, distance, geometry["spheres"]["WMAP5"])
            if float(planck["void_length_mpc"]) == 0.0 and float(wmap["void_length_mpc"]) == 0.0:
                continue
            selected.append(
                {
                    **source,
                    "source_index": source_index,
                    "bucket": bucket,
                    "role": role,
                    "distance_path_mpc": distance,
                    "direction": tuple(float(value) for value in direction),
                    "mask_neighborhood_fraction": base._mask_neighborhood_fraction(mask, ra, dec),
                    "planck": planck,
                    "wmap": wmap,
                }
            )
            if len(selected) == int(base_config["object_count"]):
                break
    identifiers = [int(row["identifier"]) for row in selected]
    _require(
        identifiers == successor["repair"]["expected_identifiers"],
        "source-only nonzero selection changed",
    )
    _require(
        int(selected[-1]["source_index"]) == int(successor["repair"]["expected_last_source_index"]),
        "source-only selection stop changed",
    )
    return selected, {
        "cf4_raw_rows_read": raw_rows_read,
        "cf4_development_source_rows_decoded": source_fields_decoded,
        "cf4_measured_velocity_fields_decoded": 0,
        "cf4_published_peculiar_velocity_fields_decoded": 0,
        "validation_source_fields_decoded": 0,
        "confirmation_source_fields_decoded": 0,
    }


def _scenario_v2(*args: Any, **kwargs: Any):
    scenario = base._scenario(*args, **kwargs)
    replace_path = lambda row: replace(row, artifact_path=VALUES_PATH.as_posix())
    return replace(
        scenario,
        formula_features=tuple(replace_path(row) for row in scenario.formula_features),
        scoring_responses=tuple(replace_path(row) for row in scenario.scoring_responses),
        hidden_truth=tuple(replace_path(row) for row in scenario.hidden_truth),
        expected_predictions=tuple(replace_path(row) for row in scenario.expected_predictions),
        uncertainties=tuple(replace_path(row) for row in scenario.uncertainties),
    )


def _signature_diagnostics(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    exposure_for = {
        "C01_OBSERVER_ENDPOINT_LOCAL_VOID": "source.scalar.observer-endpoint-chord-mpc",
        "C02_TARGET_ENDPOINT_LOCAL_VOID": "source.scalar.target-endpoint-chord-mpc",
        "C03_SINGLE_DOMINANT_VOID": "source.scalar.maximum-chord-mpc",
        "C04_BOUNDED_FRACTION_NULL": "source.scalar.null-void-length-mpc",
        "VQ00_STANDARD_FLRW_FLOW_CONTROL": None,
        "VQ08_TWO_PHASE_VOID_FRACTION": "source.scalar.void-length-mpc",
    }
    signatures = {
        formula_id: np.asarray(
            [float(base._prediction(item["values"], exposure)[base._OUTPUT][0]) for item in items],
            dtype=np.float64,
        )
        for formula_id, exposure in exposure_for.items()
    }
    pairs = []
    formulas = sorted(signatures)
    for left_index, left in enumerate(formulas):
        for right in formulas[left_index + 1 :]:
            difference = signatures[left] - signatures[right]
            maximum = float(np.max(np.abs(difference)))
            rmse = float(np.sqrt(np.mean(difference * difference)))
            pairs.append(
                {
                    "left": left,
                    "right": right,
                    "maximum_absolute_difference_hex": maximum.hex(),
                    "rmse_hex": rmse.hex(),
                    "exactly_degenerate": bool(np.array_equal(signatures[left], signatures[right])),
                }
            )
    return {
        "pair_count": len(pairs),
        "exact_degenerate_pair_count": sum(row["exactly_degenerate"] for row in pairs),
        "pairs": pairs,
    }


def derive_release() -> tuple[dict[str, Any], bytes, bytes, bytes, bytes, bytes]:
    successor = load_config()
    validate_config(successor)
    config = base.load_config()
    catalogue = base._catalogue(config)
    bindings = base._bindings(config)
    validate_binding_catalogue(bindings, catalogue)
    registrations = tuple(
        AdapterRegistration.create(f"adapter.void-cosmology.{row.formula_id.lower()}.v1", row)
        for row in bindings
        if row.status is BindingStatus.EXECUTABLE
    )
    validate_adapter_registry(registrations)
    geometry = base._parse_vast_geometry(config)
    source_rows, source_access = _select_nonzero_source_objects(config, geometry, successor)
    items = base._variant_items(source_rows, config)
    arrays: dict[str, np.ndarray] = {}
    scenarios = []
    scenario_values: dict[str, ScenarioRuntimeValues] = {}
    truths: dict[str, str] = {}
    comparisons: dict[str, tuple[ObservableComparison, ...]] = {}
    scenario_rows = []
    truth_ids = list(config["truth_formula_ids"])
    exposure_for = {
        "C01_OBSERVER_ENDPOINT_LOCAL_VOID": "source.scalar.observer-endpoint-chord-mpc",
        "C02_TARGET_ENDPOINT_LOCAL_VOID": "source.scalar.target-endpoint-chord-mpc",
        "C03_SINGLE_DOMINANT_VOID": "source.scalar.maximum-chord-mpc",
        "C04_BOUNDED_FRACTION_NULL": "source.scalar.null-void-length-mpc",
        "VQ00_STANDARD_FLRW_FLOW_CONTROL": None,
        "VQ08_TWO_PHASE_VOID_FRACTION": "source.scalar.void-length-mpc",
    }
    for item in items:
        for truth_index, truth_formula_id in enumerate(truth_ids):
            truth_prediction = base._prediction(item["values"], exposure_for[truth_formula_id])[
                base._OUTPUT
            ]
            for nuisance_draw, family in enumerate(config["noise_families"]):
                scenario_id = (
                    f"void.cf4-{item['identifier']}.{item['variant']}."
                    f"truth-{truth_formula_id.lower()}.noise-{family}.v2"
                )
                truth_world_id = f"truth.{truth_formula_id.lower()}"
                lineage = SeedLineage(
                    int(config["suite_seed"]),
                    scenario_id,
                    f"cf4-{item['identifier']}",
                    truth_world_id,
                    nuisance_draw,
                    0,
                )
                response, variance, noise = base._noise_response(
                    truth_prediction, item["values"], family, lineage, config
                )
                scenario = _scenario_v2(
                    config,
                    item,
                    scenario_id,
                    truth_world_id,
                    nuisance_draw,
                    response,
                    variance,
                    truth_index,
                )
                truth_value = np.asarray([truth_index], dtype=np.int64)
                scenarios.append(scenario)
                scenario_values[scenario_id] = ScenarioRuntimeValues(
                    formula_values=item["values"],
                    response_values={base._RESPONSE: response},
                    truth_values={base._TRUTH: truth_value},
                    uncertainty_values={base._UNCERTAINTY: variance},
                )
                truths[scenario_id] = truth_formula_id
                comparisons[scenario_id] = (
                    ObservableComparison(base._OUTPUT, base._RESPONSE, base._UNCERTAINTY),
                )
                locators: dict[str, dict[str, str]] = {}
                for feature, value in item["values"].items():
                    key = base._array_key(
                        "feature", str(item["identifier"]), str(item["variant"]), feature
                    )
                    arrays[key] = value
                    locators[feature] = {"key": key, "sha256": array_sha256(value)}
                for label, value in (
                    ("response", response),
                    ("variance", variance),
                    ("truth", truth_value),
                ):
                    key = base._array_key(label, scenario_id)
                    arrays[key] = value
                    locators[label] = {"key": key, "sha256": array_sha256(value)}
                scenario_rows.append(
                    {
                        "scenario": scenario.to_dict(),
                        "scenario_sha256": scenario.content_sha256,
                        "truth_formula_id": truth_formula_id,
                        "geometry_variant": item["variant"],
                        "source_geometry": item["source_geometry"],
                        "source_index": item["source_index"],
                        "noise": noise,
                        "geometry_values": {
                            key: item[key]
                            for key in (
                                "distance_mpc_hex",
                                "void_length_mpc_hex",
                                "null_void_length_mpc_hex",
                                "void_fraction_hex",
                            )
                        },
                        "value_locators": locators,
                    }
                )
    scenarios = sorted(scenarios, key=lambda row: row.scenario_id)
    scenario_rows = sorted(scenario_rows, key=lambda row: row["scenario"]["scenario_id"])
    release = SyntheticSuiteRelease(
        suite_id="gravity.synthetic.void-cosmology-source-shaped-matrix.v2",
        version=successor["version"],
        release_sha256=canonical_sha256(
            {
                "config": base._file_sha256(_ROOT / CONFIG_PATH),
                "scenario_ids": [row.scenario_id for row in scenarios],
            }
        ),
        ontology_sha256=catalogue.content_sha256,
        generator_sha256=base._file_sha256(Path(__file__)),
        observation_operator_sha256=base._json_sha256(config["noise"]),
        changed_feature_ids=(),
        change_level="PATCH",
        response_calibrated=False,
        prediction_semantics_changed=False,
    )
    parameter_cells = {
        row.binding_id: (
            (ParameterCell("fixed-source-contract", {}),)
            if row.status is BindingStatus.EXECUTABLE
            else ()
        )
        for row in bindings
    }
    result = run_discovery_matrix_v2(
        catalogue=catalogue,
        release=release,
        scenarios=scenarios,
        scenario_values=scenario_values,
        truth_formula_by_scenario=truths,
        bindings=bindings,
        adapters=registrations,
        parameter_cells=parameter_cells,
        comparisons=comparisons,
        distinct_gap=float(config["scoring"]["distinct_gap"]),
        ledger_id="gravity.synthetic.void-cosmology-source-shaped-matrix.v2.ledger",
    )
    cells_by_scenario: dict[str, list[Any]] = {}
    for cell in result.cells:
        cells_by_scenario.setdefault(cell.scenario_id, []).append(cell)
    confusion = {
        truth: {candidate: 0 for candidate in sorted(base._EXECUTABLE)} for truth in truth_ids
    }
    recovery = {truth: {"scenarios": 0, "recovered": 0, "distinct": 0} for truth in truth_ids}
    for scenario_id in sorted(cells_by_scenario):
        cells = cells_by_scenario[scenario_id]
        truth = truths[scenario_id]
        for winner in (cell.formula_id for cell in cells if cell.winner):
            confusion[truth][winner] += 1
        recovery[truth]["scenarios"] += 1
        recovery[truth]["recovered"] += int(any(cell.truth_recovered for cell in cells))
        recovery[truth]["distinct"] += int(
            any(cell.truth_recovered and cell.distinct for cell in cells)
        )
    values_bytes = base._npz_bytes(arrays)
    _require(values_bytes == base._npz_bytes(arrays), "NPZ serialization nondeterministic")
    scenarios_bytes = b"".join(base._json_bytes(row) + b"\n" for row in scenario_rows)
    ledger_bytes = base._json_bytes(result.ledger.to_dict(), indent=2)
    confusion_payload = {
        "schema": "open-gravity-void-cosmology-confusion-matrix-2.0",
        "truth_formula_ids": truth_ids,
        "candidate_formula_ids": sorted(base._EXECUTABLE),
        "winner_membership_counts": confusion,
        "recovery_by_truth": recovery,
        "scenario_count": result.scenario_count,
        "attempted_cell_count": result.attempted_cell_count,
        "scored_cell_count": result.scored_cell_count,
        "truth_recovery_count": result.truth_recovery_count,
        "distinct_truth_recovery_count": result.distinct_truth_recovery_count,
        "runner_result_content_sha256": result.content_sha256,
        "no_hand_ranking": True,
    }
    confusion_bytes = base._json_bytes(confusion_payload, indent=2)
    geometry_valid = all(
        0.0
        <= float(item["values"]["source.scalar.void-length-mpc"][0])
        <= float(item["values"]["source.scalar.distance-mpc"][0])
        and 0.0
        <= float(item["values"]["source.scalar.null-void-length-mpc"][0])
        <= float(item["values"]["source.scalar.distance-mpc"][0])
        for item in items
    )
    diagnostics = {
        "schema": "open-gravity-void-cosmology-geometry-identifiability-2.0",
        "geometry_valid": geometry_valid,
        "absolute_length_permutation_used": False,
        "permutation_rule": "permute dimensionless void fraction within four-object distance strata; reconstruct L'=f_perm*D for each target",
        "source_selection_rule": successor["repair"]["selection_rule"],
        "source_selection_response_blind": True,
        "object_count": len(source_rows),
        "variant_item_count": len(items),
        "signature_identifiability": _signature_diagnostics(items),
        "selected_cf4": [
            {
                "identifier": str(row["identifier"]),
                "source_index": int(row["source_index"]),
                "bucket": int(row["bucket"]),
                "role": row["role"],
                "distance_path_mpc_hex": float(row["distance_path_mpc"]).hex(),
                "planck_void_length_mpc_hex": float(row["planck"]["void_length_mpc"]).hex(),
                "wmap_void_length_mpc_hex": float(row["wmap"]["void_length_mpc"]).hex(),
            }
            for row in source_rows
        ],
        "blocked_formula_ids": sorted(base._BLOCKED),
        "claim_class": successor["claim_class"],
    }
    _require(geometry_valid, "geometry-valid null invariant failed")
    diagnostics_bytes = base._json_bytes(diagnostics, indent=2)
    receipt_body = {
        "schema": "open-gravity-void-cosmology-source-shaped-synthetic-injection-matrix-receipt-2.0",
        "package_id": successor["package_id"],
        "version": successor["version"],
        "status": "FROZEN_SYNTHETIC_ONLY_COMPLETE_AWAITING_DISTINCT_AUDIT",
        "claim_class": successor["claim_class"],
        "scientific_claim": "NONE_SYNTHETIC_ONLY_NOT_SUPPORT_OR_REJECTION",
        "independent_audit_completed": False,
        "distinct_independent_audit_required": True,
        "predecessor": successor["predecessor"],
        "predecessor_failure_retained": successor["repair"],
        "object_count": len(source_rows),
        "geometry_variant_count": len(config["geometry_variants"]),
        "noise_family_count": len(config["noise_families"]),
        "truth_formula_count": len(truth_ids),
        "scenario_count": result.scenario_count,
        "attempted_cell_count": result.attempted_cell_count,
        "scored_cell_count": result.scored_cell_count,
        "replay_entry_count": len(result.ledger.entries),
        "truth_recovery_count": result.truth_recovery_count,
        "distinct_truth_recovery_count": result.distinct_truth_recovery_count,
        "recovery_by_truth": recovery,
        "executable_formula_ids": sorted(base._EXECUTABLE),
        "blocked_formula_ids": sorted(base._BLOCKED),
        "formula_binding_sha256": {row.formula_id: row.content_sha256 for row in bindings},
        "adapter_sha256": {
            row.formula_binding.formula_id: row.adapter_sha256 for row in registrations
        },
        "geometry_gates": {
            "zero_le_l_void_and_null_le_distance": geometry_valid,
            "nonzero_source_coverage_per_selected_object": all(
                float(row["planck"]["void_length_mpc"]) > 0.0
                or float(row["wmap"]["void_length_mpc"]) > 0.0
                for row in source_rows
            ),
            "absolute_length_permutation_used": False,
        },
        "package_hashes": {
            "config_raw_sha256": base._file_sha256(_ROOT / CONFIG_PATH),
            "module_raw_sha256": base._file_sha256(Path(__file__)),
            "test_raw_sha256": base._file_sha256(_ROOT / TEST_PATH),
        },
        "artifact_sha256": {
            "values.npz": hashlib.sha256(values_bytes).hexdigest(),
            "scenarios.jsonl": hashlib.sha256(scenarios_bytes).hexdigest(),
            "ledger.json": hashlib.sha256(ledger_bytes).hexdigest(),
            "confusion-matrix.json": hashlib.sha256(confusion_bytes).hexdigest(),
            "geometry-and-identifiability.json": hashlib.sha256(diagnostics_bytes).hexdigest(),
        },
        "access_accounting": {
            **successor["access_contract"],
            **source_access,
            "vast1_source_rows_decoded": geometry["table1_rows"],
            "vast2_source_rows_decoded": geometry["table2_rows"],
            "synthetic_response_values_generated": result.scenario_count,
            "real_scores": 0,
        },
        "limitations": [
            "All response vectors are synthetic; no empirical support or rejection is authorized.",
            "The v2 coverage gate uses source geometry only and does not inspect a response.",
            "V3k, peculiar velocities, redshift residuals, validation/confirmation fields, and Pantheon remain unopened and undecoded.",
            "VQ01-VQ07 and VQ09-VQ10 remain SOURCE_BLOCKED rather than receiving surrogate inputs.",
            "The v1 all-zero exposure selection and its complete formula degeneracy remain frozen counterevidence.",
        ],
    }
    receipt = {**receipt_body, "content_sha256": base._json_sha256(receipt_body)}
    return receipt, values_bytes, scenarios_bytes, ledger_bytes, confusion_bytes, diagnostics_bytes


def freeze() -> str:
    receipt, values, scenarios, ledger, confusion, diagnostics = derive_release()
    payloads = (
        (VALUES_PATH, values),
        (SCENARIOS_PATH, scenarios),
        (LEDGER_PATH, ledger),
        (CONFUSION_PATH, confusion),
        (DIAGNOSTICS_PATH, diagnostics),
        (RECEIPT_PATH, base._json_bytes(receipt, indent=2)),
    )
    return ":".join(base._write_once(_ROOT / path, payload) for path, payload in payloads)


def check() -> dict[str, Any]:
    receipt, values, scenarios, ledger, confusion, diagnostics = derive_release()
    payloads = (
        (VALUES_PATH, values),
        (SCENARIOS_PATH, scenarios),
        (LEDGER_PATH, ledger),
        (CONFUSION_PATH, confusion),
        (DIAGNOSTICS_PATH, diagnostics),
        (RECEIPT_PATH, base._json_bytes(receipt, indent=2)),
    )
    for path, payload in payloads:
        _require(
            (_ROOT / path).is_file() and (_ROOT / path).read_bytes() == payload,
            f"stored void synthetic v2 artifact differs: {path}",
        )
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "check"))
    args = parser.parse_args(argv)
    if args.command == "freeze":
        print(freeze())
    else:
        print(json.dumps(check(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
