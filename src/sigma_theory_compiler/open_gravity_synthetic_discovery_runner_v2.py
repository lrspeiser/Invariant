"""Corrected Cartesian runner for response-blind synthetic discovery.

This append-only successor preserves the v1 audit failure while closing its
tie, blocked-cell, truth-identity, calibration, covariance-target, and retained
adapter-failure gaps.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from sigma_theory_compiler.open_gravity_data_element_ontology_v1 import (
    DataElementCatalogue,
)
from sigma_theory_compiler.open_gravity_formula_adapter_registry_v1 import (
    AdapterRegistration,
)
from sigma_theory_compiler.open_gravity_formula_execution_protocol_v1 import (
    BindingStatus,
    EligibilityStatus,
    FormulaExecutionBinding,
    validate_binding_catalogue,
)
from sigma_theory_compiler.open_gravity_synthetic_discovery_runner_v1 import (
    ObservableComparison,
    ParameterCell,
    ScenarioRuntimeValues,
    _score_candidate,
)
from sigma_theory_compiler.open_gravity_synthetic_replay_ledger_v1 import (
    DiscoveryStatus,
    SyntheticReplayLedger,
    SyntheticSuiteRelease,
)
from sigma_theory_compiler.open_gravity_synthetic_scenario_packet_v1 import (
    ScenarioDescriptor,
    decide_scenario_eligibility,
    execute_binding_in_process,
    validate_scenario_catalogue,
)
from sigma_theory_compiler.sigma_core import SchemaViolation, canonical_sha256

SCHEMA_VERSION = "open-gravity-synthetic-discovery-runner-2.0"


@dataclass(frozen=True, slots=True)
class DiscoveryMatrixCellV2:
    scenario_id: str
    object_id: str
    truth_formula_id: str
    formula_id: str
    binding_id: str
    parameter_cell_id: str | None
    eligibility: str
    discovery_status: str
    whitened_rmse: float | None
    result_sha256: str | None
    metrics_sha256: str | None
    diagnostics_sha256: str | None
    ledger_entry_sha256: str
    winner: bool
    truth_recovered: bool
    distinct: bool
    reason_codes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "object_id": self.object_id,
            "truth_formula_id": self.truth_formula_id,
            "formula_id": self.formula_id,
            "binding_id": self.binding_id,
            "parameter_cell_id": self.parameter_cell_id,
            "eligibility": self.eligibility,
            "discovery_status": self.discovery_status,
            "whitened_rmse_hex": (None if self.whitened_rmse is None else self.whitened_rmse.hex()),
            "result_sha256": self.result_sha256,
            "metrics_sha256": self.metrics_sha256,
            "diagnostics_sha256": self.diagnostics_sha256,
            "ledger_entry_sha256": self.ledger_entry_sha256,
            "winner": self.winner,
            "truth_recovered": self.truth_recovered,
            "distinct": self.distinct,
            "reason_codes": list(self.reason_codes),
        }


@dataclass(frozen=True, slots=True)
class SyntheticDiscoveryResultV2:
    cells: tuple[DiscoveryMatrixCellV2, ...]
    ledger: SyntheticReplayLedger
    scenario_count: int
    attempted_cell_count: int
    scored_cell_count: int
    truth_recovery_count: int
    distinct_truth_recovery_count: int
    content_sha256: str
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise SchemaViolation("synthetic discovery runner v2 schema changed")
        if self.attempted_cell_count != len(self.cells):
            raise SchemaViolation("attempted discovery cell count changed")
        ledger_hashes = {entry.entry_sha256 for entry in self.ledger.entries}
        if any(cell.ledger_entry_sha256 not in ledger_hashes for cell in self.cells):
            raise SchemaViolation("matrix cell lacks an exact replay-ledger entry")
        if self.content_sha256 != canonical_sha256(self._body()):
            raise SchemaViolation("synthetic discovery v2 result hash changed")

    def _body(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "cells": [cell.to_dict() for cell in self.cells],
            "ledger_sha256": self.ledger.content_sha256,
            "scenario_count": self.scenario_count,
            "attempted_cell_count": self.attempted_cell_count,
            "scored_cell_count": self.scored_cell_count,
            "truth_recovery_count": self.truth_recovery_count,
            "distinct_truth_recovery_count": self.distinct_truth_recovery_count,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._body(), "content_sha256": self.content_sha256}


@dataclass(frozen=True, slots=True)
class _Candidate:
    binding: FormulaExecutionBinding
    adapter_sha256: str
    parameter_cell: ParameterCell
    result_sha256: str
    metrics_sha256: str
    diagnostics_sha256: str
    observable_ids: tuple[str, ...]
    distance: float | None
    invalid: bool
    reason_codes: tuple[str, ...]


def _decision_reasons(decision: Any) -> tuple[str, ...]:
    return tuple(
        sorted(
            [
                *(f"missing.{value}" for value in decision.missing_features),
                *(f"forbidden.{value}" for value in decision.forbidden_features),
            ]
        )
    )


def _validate_comparisons(
    scenario: ScenarioDescriptor, comparisons: Sequence[ObservableComparison]
) -> None:
    rows = tuple(comparisons)
    keys = tuple(
        (row.prediction_element_id, row.response_element_id, row.uncertainty_id) for row in rows
    )
    if not rows or keys != tuple(sorted(set(keys))):
        raise SchemaViolation("scenario comparisons must be nonempty, unique, and sorted")
    prediction_ids = {row.element_id for row in scenario.expected_predictions}
    response_ids = {row.element_id for row in scenario.scoring_responses}
    uncertainty_targets = {
        row.uncertainty_id: row.applies_to_element_id for row in scenario.uncertainties
    }
    for row in rows:
        if row.prediction_element_id not in prediction_ids:
            raise SchemaViolation("comparison prediction is absent from scenario contract")
        if row.response_element_id not in response_ids:
            raise SchemaViolation("comparison response is absent from scenario contract")
        if uncertainty_targets.get(row.uncertainty_id) != row.response_element_id:
            raise SchemaViolation("comparison covariance targets a different data element")


def _failed_candidate(
    *,
    binding: FormulaExecutionBinding,
    adapter_sha256: str,
    parameter_cell: ParameterCell,
    scenario: ScenarioDescriptor,
    comparisons: Sequence[ObservableComparison],
    error: Exception,
) -> _Candidate:
    observable_ids = tuple(
        sorted(
            {
                *(row.prediction_element_id for row in comparisons),
                *(row.response_element_id for row in comparisons),
            }
        )
    )
    failure = {
        "error_type": type(error).__name__,
        "error_message": str(error),
        "binding_id": binding.binding_id,
        "scenario_id": scenario.scenario_id,
        "parameter_cell_id": parameter_cell.parameter_cell_id,
    }
    diagnostics = {
        **failure,
        "parameter_cell_sha256": parameter_cell.content_sha256,
        "real_response_used": False,
    }
    return _Candidate(
        binding,
        adapter_sha256,
        parameter_cell,
        canonical_sha256(failure),
        canonical_sha256({"metric": "NOT_EVALUATED_NUMERICAL_INVALID"}),
        canonical_sha256(diagnostics),
        observable_ids,
        None,
        True,
        ("execution_or_scoring_invalid",),
    )


def run_discovery_matrix_v2(
    *,
    catalogue: DataElementCatalogue,
    release: SyntheticSuiteRelease,
    scenarios: Sequence[ScenarioDescriptor],
    scenario_values: Mapping[str, ScenarioRuntimeValues],
    truth_formula_by_scenario: Mapping[str, str],
    bindings: Sequence[FormulaExecutionBinding],
    adapters: Sequence[AdapterRegistration],
    parameter_cells: Mapping[str, Sequence[ParameterCell]],
    comparisons: Mapping[str, Sequence[ObservableComparison]],
    distinct_gap: float,
    ledger_id: str,
) -> SyntheticDiscoveryResultV2:
    """Run a complete response-blind synthetic candidate matrix."""

    if release.response_calibrated:
        raise SchemaViolation("response-calibrated release cannot use response-blind runner")
    if not math.isfinite(distinct_gap) or distinct_gap < 0:
        raise SchemaViolation("distinct gap must be finite and nonnegative")
    validate_binding_catalogue(bindings, catalogue)
    ordered_scenarios = tuple(scenarios)
    scenario_ids = tuple(row.scenario_id for row in ordered_scenarios)
    if not scenario_ids or scenario_ids != tuple(sorted(set(scenario_ids))):
        raise SchemaViolation("scenarios must be nonempty, unique, and sorted")
    binding_ids = tuple(row.binding_id for row in bindings)
    formula_ids = tuple(row.formula_id for row in bindings)
    if not binding_ids or binding_ids != tuple(sorted(set(binding_ids))):
        raise SchemaViolation("bindings must be nonempty, unique, and sorted")
    if len(set(formula_ids)) != len(formula_ids):
        raise SchemaViolation("candidate formula IDs must be unique")
    candidate_formula_ids = set(formula_ids)
    adapter_by_binding = {row.formula_binding.binding_id: row for row in adapters}
    if len(adapter_by_binding) != len(adapters):
        raise SchemaViolation("active adapters must be unique by binding")
    executable_ids = {row.binding_id for row in bindings if row.status is BindingStatus.EXECUTABLE}
    if set(adapter_by_binding) != executable_ids:
        raise SchemaViolation("every executable binding needs exactly one active adapter")
    for registration in adapters:
        if registration.formula_binding not in bindings:
            raise SchemaViolation("adapter binding is absent from discovery candidates")
    expected_maps = set(scenario_ids)
    if (
        set(scenario_values) != expected_maps
        or set(truth_formula_by_scenario) != expected_maps
        or set(comparisons) != expected_maps
    ):
        raise SchemaViolation("scenario runtime/truth/comparison maps differ from matrix")
    if any(value not in candidate_formula_ids for value in truth_formula_by_scenario.values()):
        raise SchemaViolation("truth formula is absent from candidate bindings")
    if set(parameter_cells) != set(binding_ids):
        raise SchemaViolation("parameter-cell map differs from bindings")
    for binding in bindings:
        cells = tuple(parameter_cells[binding.binding_id])
        if binding.status is BindingStatus.EXECUTABLE:
            ids = tuple(row.parameter_cell_id for row in cells)
            if not cells or ids != tuple(sorted(set(ids))):
                raise SchemaViolation("executable parameter cells must be unique and sorted")
        elif cells:
            raise SchemaViolation("non-executable binding cannot have parameter cells")

    ledger = SyntheticReplayLedger(ledger_id, ())
    output_cells: list[DiscoveryMatrixCellV2] = []
    recovery_count = 0
    distinct_recovery_count = 0
    scored_count = 0

    for scenario in ordered_scenarios:
        validate_scenario_catalogue(scenario, catalogue)
        _validate_comparisons(scenario, comparisons[scenario.scenario_id])
        validated = scenario_values[scenario.scenario_id].validate(scenario)
        truth_formula_id = truth_formula_by_scenario[scenario.scenario_id]
        decisions = {
            binding.binding_id: decide_scenario_eligibility(binding, catalogue, scenario)
            for binding in bindings
        }
        candidates: dict[tuple[str, str], _Candidate] = {}

        for binding in bindings:
            decision = decisions[binding.binding_id]
            if decision.status is not EligibilityStatus.ELIGIBLE:
                continue
            registration = adapter_by_binding[binding.binding_id]
            selected_ids = tuple(
                sorted(
                    set(binding.required_features)
                    | (set(binding.optional_features) & set(validated.formula_features))
                )
            )
            projected = {key: validated.formula_features[key] for key in selected_ids}
            for parameter_cell in parameter_cells[binding.binding_id]:
                try:
                    execution = execute_binding_in_process(
                        binding,
                        catalogue,
                        scenario,
                        projected,
                        parameter_cell.values,
                    )
                    distance, observable_ids = _score_candidate(
                        output_values=execution.output_values,
                        validated=validated,
                        comparisons=comparisons[scenario.scenario_id],
                    )
                    metric = {
                        "metric": "whitened_rmse",
                        "value_hex": distance.hex(),
                        "comparisons": [
                            {
                                "prediction": row.prediction_element_id,
                                "response": row.response_element_id,
                                "uncertainty": row.uncertainty_id,
                            }
                            for row in comparisons[scenario.scenario_id]
                        ],
                    }
                    diagnostics = {
                        "deterministic_replay": execution.deterministic_replay,
                        "scenario_sha256": execution.scenario_sha256,
                        "binding_sha256": execution.binding_sha256,
                        "parameter_values_sha256": execution.parameter_values_sha256,
                        "parameter_cell_sha256": parameter_cell.content_sha256,
                        "real_response_used": False,
                    }
                    candidate = _Candidate(
                        binding,
                        registration.adapter_sha256,
                        parameter_cell,
                        execution.output_sha256,
                        canonical_sha256(metric),
                        canonical_sha256(diagnostics),
                        observable_ids,
                        distance,
                        False,
                        (),
                    )
                except Exception as error:  # noqa: BLE001 -- retain cell failure, continue matrix
                    candidate = _failed_candidate(
                        binding=binding,
                        adapter_sha256=registration.adapter_sha256,
                        parameter_cell=parameter_cell,
                        scenario=scenario,
                        comparisons=comparisons[scenario.scenario_id],
                        error=error,
                    )
                candidates[(binding.binding_id, parameter_cell.parameter_cell_id)] = candidate

        valid = [row for row in candidates.values() if not row.invalid]
        minimum = min((row.distance for row in valid if row.distance is not None), default=None)
        joint_winners = (
            [row for row in valid if row.distance == minimum] if minimum is not None else []
        )
        larger = (
            sorted(
                row.distance for row in valid if row.distance is not None and row.distance > minimum
            )
            if minimum is not None
            else []
        )
        runner_up = larger[0] if larger else None
        gap = runner_up - minimum if runner_up is not None and minimum is not None else None
        distinct = bool(
            len(joint_winners) == 1
            and len(valid) >= 2
            and gap is not None
            and gap > 0
            and gap >= distinct_gap
        )
        truth_recovered = any(row.binding.formula_id == truth_formula_id for row in joint_winners)
        recovery_count += int(truth_recovered)
        distinct_recovery_count += int(truth_recovered and distinct)

        for binding in bindings:
            decision = decisions[binding.binding_id]
            if decision.status is not EligibilityStatus.ELIGIBLE:
                blocked_parameters: tuple[ParameterCell | None, ...] = (
                    tuple(parameter_cells[binding.binding_id])
                    if binding.status is BindingStatus.EXECUTABLE
                    else (None,)
                )
                for parameter_cell in blocked_parameters:
                    ledger = ledger.append(
                        release=release,
                        binding=binding,
                        eligibility=decision,
                        adapter_sha256=(
                            adapter_by_binding[binding.binding_id].adapter_sha256
                            if binding.status is BindingStatus.EXECUTABLE
                            else None
                        ),
                        domain=scenario.domain,
                        experiment_id=scenario.experiment_id,
                    )
                    ledger_entry = ledger.entries[-1]
                    output_cells.append(
                        DiscoveryMatrixCellV2(
                            scenario.scenario_id,
                            scenario.object_id,
                            truth_formula_id,
                            binding.formula_id,
                            binding.binding_id,
                            (
                                parameter_cell.parameter_cell_id
                                if parameter_cell is not None
                                else None
                            ),
                            decision.status.value,
                            decision.status.value,
                            None,
                            None,
                            None,
                            None,
                            ledger_entry.entry_sha256,
                            False,
                            truth_recovered,
                            False,
                            _decision_reasons(decision),
                        )
                    )
                continue

            for parameter_cell in parameter_cells[binding.binding_id]:
                candidate = candidates[(binding.binding_id, parameter_cell.parameter_cell_id)]
                is_winner = candidate in joint_winners
                if candidate.invalid:
                    status = DiscoveryStatus.NUMERICAL_INVALID
                    reasons = candidate.reason_codes
                elif not distinct:
                    status = DiscoveryStatus.UNDERPOWERED
                    reasons = ("candidate_gap_below_threshold",)
                elif is_winner and truth_recovered:
                    status = DiscoveryStatus.PROMISING_DISTINCT_SIGNATURE
                    reasons = ()
                else:
                    status = DiscoveryStatus.AMBIGUOUS_WITH_COMPARATOR
                    reasons = () if truth_recovered else ("truth_generator_not_recovered",)
                ledger = ledger.append(
                    release=release,
                    binding=binding,
                    eligibility=decision,
                    adapter_sha256=candidate.adapter_sha256,
                    domain=scenario.domain,
                    experiment_id=scenario.experiment_id,
                )
                ledger = ledger.complete_last_eligible(
                    release=release,
                    binding=binding,
                    adapter_sha256=candidate.adapter_sha256,
                    domain=scenario.domain,
                    experiment_id=scenario.experiment_id,
                    status=status,
                    scenario_id=scenario.scenario_id,
                    object_id=scenario.object_id,
                    truth_world_id=scenario.seed_lineage.truth_world_id,
                    seed_lineage_sha256=canonical_sha256(scenario.seed_lineage.to_dict()),
                    nuisance_draw=scenario.seed_lineage.nuisance_draw,
                    parameter_cell_id=parameter_cell.parameter_cell_id,
                    observable_ids=candidate.observable_ids,
                    result_sha256=candidate.result_sha256,
                    metrics_sha256=candidate.metrics_sha256,
                    diagnostics_sha256=candidate.diagnostics_sha256,
                    reason_codes=reasons,
                )
                ledger_entry = ledger.entries[-1]
                output_cells.append(
                    DiscoveryMatrixCellV2(
                        scenario.scenario_id,
                        scenario.object_id,
                        truth_formula_id,
                        binding.formula_id,
                        binding.binding_id,
                        parameter_cell.parameter_cell_id,
                        EligibilityStatus.ELIGIBLE.value,
                        status.value,
                        candidate.distance,
                        candidate.result_sha256,
                        candidate.metrics_sha256,
                        candidate.diagnostics_sha256,
                        ledger_entry.entry_sha256,
                        is_winner,
                        truth_recovered,
                        distinct,
                        reasons,
                    )
                )
                scored_count += int(not candidate.invalid)

    cells = tuple(output_cells)
    body = {
        "schema_version": SCHEMA_VERSION,
        "cells": [cell.to_dict() for cell in cells],
        "ledger_sha256": ledger.content_sha256,
        "scenario_count": len(ordered_scenarios),
        "attempted_cell_count": len(cells),
        "scored_cell_count": scored_count,
        "truth_recovery_count": recovery_count,
        "distinct_truth_recovery_count": distinct_recovery_count,
    }
    return SyntheticDiscoveryResultV2(
        cells,
        ledger,
        len(ordered_scenarios),
        len(cells),
        scored_count,
        recovery_count,
        distinct_recovery_count,
        canonical_sha256(body),
    )


__all__ = [
    "SCHEMA_VERSION",
    "DiscoveryMatrixCellV2",
    "SyntheticDiscoveryResultV2",
    "run_discovery_matrix_v2",
]
