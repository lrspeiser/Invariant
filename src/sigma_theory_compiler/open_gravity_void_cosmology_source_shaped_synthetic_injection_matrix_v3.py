"""Typed-metadata repair for the response-blind void/cosmology synthetic matrix."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

import sigma_theory_compiler.open_gravity_void_cosmology_source_shaped_synthetic_injection_matrix_v1 as base
import sigma_theory_compiler.open_gravity_void_cosmology_source_shaped_synthetic_injection_matrix_v2 as v2
from sigma_theory_compiler.open_gravity_data_element_ontology_v1 import catalogue_from_elements
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
    "configs/open_gravity_void_cosmology_source_shaped_synthetic_injection_matrix_v3.json"
)
TEST_PATH = Path(
    "tests/test_open_gravity_void_cosmology_source_shaped_synthetic_injection_matrix_v3.py"
)
OUTPUT_DIR = Path(
    "runs/gravity/open-gravity-void-cosmology-source-shaped-synthetic-injection-matrix-v3"
)
VALUES_PATH = OUTPUT_DIR / "values.npz"
SCENARIOS_PATH = OUTPUT_DIR / "scenarios.jsonl"
LEDGER_PATH = OUTPUT_DIR / "ledger.json"
CONFUSION_PATH = OUTPUT_DIR / "confusion-matrix.json"
DIAGNOSTICS_PATH = OUTPUT_DIR / "geometry-and-identifiability.json"
TYPED_DIFF_PATH = OUTPUT_DIR / "typed-contract-diff.json"
RECEIPT_PATH = OUTPUT_DIR / "receipt.json"
_ROOT = Path(__file__).resolve().parents[2]
_RATE_FEATURES = (
    "source.scalar.delta-h-km-s-mpc",
    "source.scalar.h-m-km-s-mpc",
)
_RATE_UNIT = "km s^-1 Mpc^-1"
_RATE_DIMENSION = (0, 0, -1, 0, 0, 0, 0)

# This full map deliberately avoids suffix precedence and makes every feature type explicit.
_FEATURE_METADATA = {
    "source.scalar.delta-h-km-s-mpc": (_RATE_UNIT, ("object",)),
    "source.scalar.distance-modulus-mag": ("mag", ("object",)),
    "source.scalar.distance-modulus-uncertainty-mag": ("mag", ("object",)),
    "source.scalar.distance-mpc": ("Mpc", ("object",)),
    "source.scalar.h-m-km-s-mpc": (_RATE_UNIT, ("object",)),
    "source.scalar.mask-neighborhood-fraction": ("1", ("object",)),
    "source.scalar.maximum-chord-mpc": ("Mpc", ("object",)),
    "source.scalar.null-void-length-mpc": ("Mpc", ("object",)),
    "source.scalar.observer-endpoint-chord-mpc": ("Mpc", ("object",)),
    "source.scalar.target-endpoint-chord-mpc": ("Mpc", ("object",)),
    "source.scalar.void-fraction": ("1", ("object",)),
    "source.scalar.void-length-mpc": ("Mpc", ("object",)),
    "source.vector.direction-cartesian": ("1", ("cartesian",)),
    "source.vector.flow-shear-design": ("1", ("nuisance",)),
}


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
            "predecessor",
            "blocked_audit",
            "repair",
            "expected_unchanged",
            "access_contract",
        },
        "void synthetic v3 config keys changed",
    )
    _require(
        config["schema"]
        == "open-gravity-void-cosmology-source-shaped-synthetic-injection-matrix-3.0"
        and config["package_id"]
        == "open-gravity-void-cosmology-source-shaped-synthetic-injection-matrix-v3"
        and config["version"] == "v1.1.0",
        "void synthetic v3 identity changed",
    )
    _require(
        config["status"] == "FROZEN_SYNTHETIC_ONLY_PRE_RESPONSE"
        and config["claim_class"] == "SYNTHETIC_DIRECTIONAL_SIGNAL",
        "void synthetic v3 claim boundary changed",
    )
    _require(
        base._repo_path(config["output_directory"]) == (_ROOT / OUTPUT_DIR).resolve(),
        "void synthetic v3 output path changed",
    )
    repair = config["repair"]
    _require(
        repair["affected_feature_ids"] == list(_RATE_FEATURES)
        and repair["canonical_unit"] == _RATE_UNIT
        and tuple(repair["si_dimension"]) == _RATE_DIMENSION
        and repair["expected_scenario_reference_occurrences"] == 1440
        and repair["explicit_feature_map_required"] is True,
        "void synthetic v3 repair contract changed",
    )
    _require(
        set(_FEATURE_METADATA) == set(base._FEATURES),
        "explicit void feature metadata map incomplete",
    )
    _require(
        all(value == 0 for value in config["access_contract"].values()),
        "void synthetic v3 response barrier changed",
    )
    if not verify_hashes:
        return
    predecessor = config["predecessor"]
    for prefix in ("config", "module", "test", "receipt"):
        path = base._repo_path(predecessor[f"{prefix}_path"])
        _require(
            base._file_sha256(path) == predecessor[f"{prefix}_raw_sha256"],
            f"void synthetic v2 {prefix} drift",
        )
    predecessor_receipt = json.loads(
        base._repo_path(predecessor["receipt_path"]).read_text(encoding="utf-8")
    )
    _require(
        predecessor_receipt["content_sha256"] == predecessor["receipt_content_sha256"],
        "void synthetic v2 receipt content drift",
    )
    audit = config["blocked_audit"]
    audit_path = base._repo_path(audit["path"])
    _require(base._file_sha256(audit_path) == audit["raw_sha256"], "blocked audit raw drift")
    audit_receipt = json.loads(audit_path.read_text(encoding="utf-8"))
    _require(
        audit_receipt["content_sha256"] == audit["content_sha256"]
        and audit_receipt["decision"] == audit["decision"],
        "blocked audit evidence drift",
    )
    v2.validate_config(v2.load_config())


def _catalogue(config: Mapping[str, Any]):
    predecessor = base._catalogue(config)
    elements = tuple(
        replace(element, canonical_unit=_RATE_UNIT, si_dimension=_RATE_DIMENSION)
        if element.element_id in _RATE_FEATURES
        else element
        for element in predecessor.elements
    )
    catalogue = catalogue_from_elements(predecessor.catalogue_id, "v1.1.0", elements)
    by_id = catalogue.by_id()
    _require(
        all(
            by_id[feature].canonical_unit == _RATE_UNIT
            and by_id[feature].si_dimension == _RATE_DIMENSION
            for feature in _RATE_FEATURES
        ),
        "Hubble-rate catalogue typing repair failed",
    )
    return catalogue


def _scenario_v3(*args: Any, **kwargs: Any):
    scenario = v2._scenario_v2(*args, **kwargs)

    def formula_ref(row):
        unit, axes = _FEATURE_METADATA[row.element_id]
        _require(row.axes == axes, "scenario feature axes differ from explicit map")
        return replace(row, artifact_path=VALUES_PATH.as_posix(), unit=unit)

    replace_path = lambda row: replace(row, artifact_path=VALUES_PATH.as_posix())
    repaired = replace(
        scenario,
        formula_features=tuple(formula_ref(row) for row in scenario.formula_features),
        scoring_responses=tuple(replace_path(row) for row in scenario.scoring_responses),
        hidden_truth=tuple(replace_path(row) for row in scenario.hidden_truth),
        expected_predictions=tuple(replace_path(row) for row in scenario.expected_predictions),
        uncertainties=tuple(replace_path(row) for row in scenario.uncertainties),
    )
    _require(
        all(
            next(row for row in repaired.formula_features if row.element_id == feature).unit
            == _RATE_UNIT
            for feature in _RATE_FEATURES
        ),
        "Hubble-rate scenario typing repair failed",
    )
    return repaired


def _normalized_scenario(value: Mapping[str, Any]) -> dict[str, Any]:
    result = json.loads(json.dumps(value))
    for group in (
        "formula_features",
        "scoring_responses",
        "hidden_truth",
        "expected_predictions",
        "uncertainties",
    ):
        for row in result[group]:
            row["artifact_path"] = "<PACKAGE_VALUES_NPZ>"
            if group == "formula_features" and row.get("element_id") in _RATE_FEATURES:
                row["unit"] = "<HUBBLE_RATE_UNIT>"
    return result


def _typed_difference_proof(
    *,
    catalogue: Any,
    scenario_rows: Sequence[Mapping[str, Any]],
    values_bytes: bytes,
    confusion: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    predecessor_scenarios = [
        json.loads(line)
        for line in (_ROOT / v2.SCENARIOS_PATH).read_text(encoding="utf-8").splitlines()
    ]
    _require(len(predecessor_scenarios) == len(scenario_rows), "scenario count changed in v3")
    unexpected = 0
    rate_occurrences = 0
    for old, new in zip(predecessor_scenarios, scenario_rows, strict=True):
        _require(
            old["scenario"]["scenario_id"] == new["scenario"]["scenario_id"],
            "scenario identity changed",
        )
        old_outer = {
            key: value for key, value in old.items() if key not in {"scenario", "scenario_sha256"}
        }
        new_outer = {
            key: value for key, value in new.items() if key not in {"scenario", "scenario_sha256"}
        }
        _require(old_outer == new_outer, "non-scenario numerical/provenance row changed")
        if _normalized_scenario(old["scenario"]) != _normalized_scenario(new["scenario"]):
            unexpected += 1
        for feature in new["scenario"]["formula_features"]:
            if feature["element_id"] in _RATE_FEATURES:
                rate_occurrences += 1
                _require(feature["unit"] == _RATE_UNIT, "scenario rate unit not repaired")
    expected = config["expected_unchanged"]
    values_hash = hashlib.sha256(values_bytes).hexdigest()
    _require(values_hash == expected["values_npz_raw_sha256"], "numerical NPZ changed")
    predecessor_confusion = json.loads((_ROOT / v2.CONFUSION_PATH).read_text(encoding="utf-8"))
    numerical_keys = (
        "truth_formula_ids",
        "candidate_formula_ids",
        "winner_membership_counts",
        "recovery_by_truth",
        "scenario_count",
        "attempted_cell_count",
        "scored_cell_count",
        "truth_recovery_count",
        "distinct_truth_recovery_count",
        "no_hand_ranking",
    )
    _require(
        all(confusion[key] == predecessor_confusion[key] for key in numerical_keys),
        "numerical confusion matrix changed",
    )
    by_id = catalogue.by_id()
    return {
        "schema": "open-gravity-void-cosmology-typed-contract-diff-1.0",
        "blocked_audit_raw_sha256": config["blocked_audit"]["raw_sha256"],
        "blocked_audit_content_sha256": config["blocked_audit"]["content_sha256"],
        "affected_feature_ids": list(_RATE_FEATURES),
        "catalogue": {
            feature: {
                "v2_unit": "Mpc",
                "v2_si_dimension": [0, 1, 0, 0, 0, 0, 0],
                "v3_unit": by_id[feature].canonical_unit,
                "v3_si_dimension": list(by_id[feature].si_dimension),
            }
            for feature in _RATE_FEATURES
        },
        "scenario_reference_occurrences": rate_occurrences,
        "expected_scenario_reference_occurrences": config["repair"][
            "expected_scenario_reference_occurrences"
        ],
        "unexpected_normalized_scenario_differences": unexpected,
        "allowed_scenario_differences": [
            "all artifact_path values change from v2 package to v3 package",
            "the two affected formula-feature units change from Mpc to km s^-1 Mpc^-1",
            "scenario_sha256 changes as a consequence of the two preceding metadata changes",
        ],
        "values_npz_byte_identical_to_v2": True,
        "values_npz_raw_sha256": values_hash,
        "geometry_diagnostics_byte_identical_to_v2": True,
        "numerical_confusion_fields_identical_to_v2": True,
        "source_selection_changed": False,
        "noise_draws_changed": False,
        "numerical_law_changed": False,
        "response_access_changed": False,
    }


def derive_release() -> tuple[dict[str, Any], bytes, bytes, bytes, bytes, bytes, bytes]:
    successor = load_config()
    validate_config(successor)
    config = base.load_config()
    catalogue = _catalogue(config)
    bindings = base._bindings(config)
    validate_binding_catalogue(bindings, catalogue)
    registrations = tuple(
        AdapterRegistration.create(f"adapter.void-cosmology.{row.formula_id.lower()}.v1", row)
        for row in bindings
        if row.status is BindingStatus.EXECUTABLE
    )
    validate_adapter_registry(registrations)
    geometry = base._parse_vast_geometry(config)
    source_rows, source_access = v2._select_nonzero_source_objects(
        config, geometry, v2.load_config()
    )
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
                # Preserve the v2 identity so the typed-only patch cannot change a noise seed.
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
                scenario = _scenario_v3(
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
        suite_id="gravity.synthetic.void-cosmology-source-shaped-matrix.v3",
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
        changed_feature_ids=_RATE_FEATURES,
        change_level="MINOR",
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
        ledger_id="gravity.synthetic.void-cosmology-source-shaped-matrix.v3.ledger",
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
    scenarios_bytes = b"".join(base._json_bytes(row) + b"\n" for row in scenario_rows)
    ledger_bytes = base._json_bytes(result.ledger.to_dict(), indent=2)
    confusion_payload = {
        "schema": "open-gravity-void-cosmology-confusion-matrix-3.0",
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
    expected = successor["expected_unchanged"]
    for key in (
        "scenario_count",
        "attempted_cell_count",
        "scored_cell_count",
        "truth_recovery_count",
        "distinct_truth_recovery_count",
    ):
        _require(confusion_payload[key] == expected[key], f"v3 numerical count changed: {key}")
    _require(
        len(result.ledger.entries) == expected["replay_entry_count"], "v3 ledger count changed"
    )
    confusion_bytes = base._json_bytes(confusion_payload, indent=2)
    diagnostics_bytes = (_ROOT / v2.DIAGNOSTICS_PATH).read_bytes()
    _require(
        hashlib.sha256(diagnostics_bytes).hexdigest()
        == expected["geometry_diagnostics_raw_sha256"],
        "v2 geometry diagnostics drift",
    )
    typed_diff = _typed_difference_proof(
        catalogue=catalogue,
        scenario_rows=scenario_rows,
        values_bytes=values_bytes,
        confusion=confusion_payload,
        config=successor,
    )
    _require(
        typed_diff["scenario_reference_occurrences"] == 1440
        and typed_diff["unexpected_normalized_scenario_differences"] == 0,
        "typed-only scenario difference proof failed",
    )
    typed_diff_bytes = base._json_bytes(typed_diff, indent=2)
    receipt_body = {
        "schema": "open-gravity-void-cosmology-source-shaped-synthetic-injection-matrix-receipt-3.0",
        "package_id": successor["package_id"],
        "version": successor["version"],
        "status": "FROZEN_SYNTHETIC_ONLY_COMPLETE_AWAITING_DISTINCT_REAUDIT",
        "claim_class": successor["claim_class"],
        "scientific_claim": "NONE_SYNTHETIC_ONLY_NOT_SUPPORT_OR_REJECTION",
        "independent_audit_completed": False,
        "distinct_independent_reaudit_required": True,
        "predecessor": successor["predecessor"],
        "blocked_audit": successor["blocked_audit"],
        "typed_repair": successor["repair"],
        "catalogue_sha256": catalogue.content_sha256,
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
        "formula_binding_sha256": {row.formula_id: row.content_sha256 for row in bindings},
        "adapter_sha256": {
            row.formula_binding.formula_id: row.adapter_sha256 for row in registrations
        },
        "typed_difference_proof": {
            "scenario_reference_occurrences": typed_diff["scenario_reference_occurrences"],
            "unexpected_normalized_scenario_differences": typed_diff[
                "unexpected_normalized_scenario_differences"
            ],
            "values_npz_byte_identical_to_v2": True,
            "geometry_diagnostics_byte_identical_to_v2": True,
            "numerical_confusion_fields_identical_to_v2": True,
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
            "typed-contract-diff.json": hashlib.sha256(typed_diff_bytes).hexdigest(),
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
            "V3 repairs dimensional metadata only; all v2 numerical arrays, noise draws, source selection, and recovery counts are unchanged.",
            "The v2 independent audit remains a BLOCK and is bound as counterevidence until a distinct v3 re-audit passes.",
            "All responses are synthetic; no empirical support or rejection is authorized.",
            "V3k, peculiar velocities, redshift residuals, validation/confirmation fields, and Pantheon remain unopened and undecoded.",
        ],
    }
    receipt = {**receipt_body, "content_sha256": base._json_sha256(receipt_body)}
    return (
        receipt,
        values_bytes,
        scenarios_bytes,
        ledger_bytes,
        confusion_bytes,
        diagnostics_bytes,
        typed_diff_bytes,
    )


def freeze() -> str:
    receipt, values, scenarios, ledger, confusion, diagnostics, typed_diff = derive_release()
    payloads = (
        (VALUES_PATH, values),
        (SCENARIOS_PATH, scenarios),
        (LEDGER_PATH, ledger),
        (CONFUSION_PATH, confusion),
        (DIAGNOSTICS_PATH, diagnostics),
        (TYPED_DIFF_PATH, typed_diff),
        (RECEIPT_PATH, base._json_bytes(receipt, indent=2)),
    )
    return ":".join(base._write_once(_ROOT / path, payload) for path, payload in payloads)


def check() -> dict[str, Any]:
    receipt, values, scenarios, ledger, confusion, diagnostics, typed_diff = derive_release()
    payloads = (
        (VALUES_PATH, values),
        (SCENARIOS_PATH, scenarios),
        (LEDGER_PATH, ledger),
        (CONFUSION_PATH, confusion),
        (DIAGNOSTICS_PATH, diagnostics),
        (TYPED_DIFF_PATH, typed_diff),
        (RECEIPT_PATH, base._json_bytes(receipt, indent=2)),
    )
    for path, payload in payloads:
        _require(
            (_ROOT / path).is_file() and (_ROOT / path).read_bytes() == payload,
            f"stored void synthetic v3 artifact differs: {path}",
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
