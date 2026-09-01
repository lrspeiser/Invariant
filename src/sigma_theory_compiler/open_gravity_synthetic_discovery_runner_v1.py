"""Common matrix runner for response-blind Open-Gravity synthetic discovery.

The runner is intentionally an in-memory foundation primitive.  It executes
only scenario-declared formula inputs, keeps scoring responses outside the
formula boundary, retains non-executable candidates without pseudo-scores, and
emits an append-only replay chain for every attempted matrix cell.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import NDArray

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
from sigma_theory_compiler.open_gravity_synthetic_replay_ledger_v1 import (
    DiscoveryStatus,
    SyntheticReplayLedger,
    SyntheticSuiteRelease,
)
from sigma_theory_compiler.open_gravity_synthetic_scenario_packet_v1 import (
    ScenarioDescriptor,
    ValidatedScenarioValues,
    decide_scenario_eligibility,
    execute_binding_in_process,
    validate_scenario_catalogue,
    validate_scenario_values,
)
from sigma_theory_compiler.sigma_core import SchemaViolation, canonical_sha256

SCHEMA_VERSION = "open-gravity-synthetic-discovery-runner-1.0"


@dataclass(frozen=True, slots=True)
class ParameterCell:
    parameter_cell_id: str
    values: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.parameter_cell_id or not isinstance(self.parameter_cell_id, str):
            raise SchemaViolation("parameter cell ID is required")
        object.__setattr__(self, "values", MappingProxyType(dict(self.values)))

    @property
    def content_sha256(self) -> str:
        return canonical_sha256(
            {"parameter_cell_id": self.parameter_cell_id, "values": dict(self.values)}
        )


@dataclass(frozen=True, slots=True)
class ObservableComparison:
    prediction_element_id: str
    response_element_id: str
    uncertainty_id: str

    def __post_init__(self) -> None:
        if not all(
            isinstance(value, str) and value
            for value in (
                self.prediction_element_id,
                self.response_element_id,
                self.uncertainty_id,
            )
        ):
            raise SchemaViolation("comparison identifiers are required")


@dataclass(frozen=True, slots=True)
class ScenarioRuntimeValues:
    formula_values: Mapping[str, Any]
    response_values: Mapping[str, Any]
    truth_values: Mapping[str, Any]
    uncertainty_values: Mapping[str, Any]

    def validate(self, scenario: ScenarioDescriptor) -> ValidatedScenarioValues:
        return validate_scenario_values(
            scenario,
            formula_values=self.formula_values,
            response_values=self.response_values,
            truth_values=self.truth_values,
            uncertainty_values=self.uncertainty_values,
        )


@dataclass(frozen=True, slots=True)
class DiscoveryMatrixCell:
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
            "winner": self.winner,
            "truth_recovered": self.truth_recovered,
            "distinct": self.distinct,
            "reason_codes": list(self.reason_codes),
        }


@dataclass(frozen=True, slots=True)
class SyntheticDiscoveryResult:
    cells: tuple[DiscoveryMatrixCell, ...]
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
            raise SchemaViolation("synthetic discovery runner schema changed")
        if self.attempted_cell_count != len(self.cells):
            raise SchemaViolation("attempted discovery cell count changed")
        if self.content_sha256 != canonical_sha256(self._body()):
            raise SchemaViolation("synthetic discovery result hash changed")

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
class _ScoredCandidate:
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


def _whitened_squared_residuals(
    prediction: NDArray[Any],
    response: NDArray[Any],
    uncertainty: NDArray[Any],
) -> NDArray[np.float64]:
    delta = np.asarray(prediction, dtype=np.float64).reshape(-1) - np.asarray(
        response, dtype=np.float64
    ).reshape(-1)
    covariance = np.asarray(uncertainty, dtype=np.float64)
    if covariance.shape == prediction.shape:
        variance = covariance.reshape(-1)
        if np.any(variance < 0):
            raise SchemaViolation("diagonal covariance cannot be negative")
        positive = variance > 0
        scale = max(
            1.0,
            float(np.linalg.norm(np.asarray(prediction, dtype=np.float64))),
            float(np.linalg.norm(np.asarray(response, dtype=np.float64))),
        )
        if np.any(np.abs(delta[~positive]) > 1e-12 * scale):
            raise SchemaViolation("residual violates zero-variance diagonal support")
        result = np.zeros(delta.shape, dtype=np.float64)
        result[positive] = np.square(delta[positive]) / variance[positive]
        return result
    if covariance.shape != (delta.size, delta.size):
        raise SchemaViolation("comparison covariance shape changed")
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    tolerance = max(float(np.max(eigenvalues)), 1.0) * 1e-12
    positive = eigenvalues > tolerance
    null_projection = eigenvectors[:, ~positive].T @ delta
    scale = max(
        1.0,
        float(np.linalg.norm(np.asarray(prediction, dtype=np.float64))),
        float(np.linalg.norm(np.asarray(response, dtype=np.float64))),
    )
    if np.any(np.abs(null_projection) > 1e-12 * scale):
        raise SchemaViolation("residual violates zero-variance covariance support")
    if not np.any(positive):
        return np.zeros(1, dtype=np.float64)
    projected = eigenvectors[:, positive].T @ delta
    return np.square(projected) / eigenvalues[positive]


def _score_candidate(
    *,
    output_values: Mapping[str, NDArray[Any]],
    validated: ValidatedScenarioValues,
    comparisons: Sequence[ObservableComparison],
) -> tuple[float, tuple[str, ...]]:
    residuals: list[NDArray[np.float64]] = []
    observables: list[str] = []
    for comparison in comparisons:
        if comparison.prediction_element_id not in output_values:
            raise SchemaViolation("comparison prediction is absent from formula output")
        if comparison.response_element_id not in validated.scoring_responses:
            raise SchemaViolation("comparison response is absent from trusted packet")
        if comparison.uncertainty_id not in validated.uncertainties:
            raise SchemaViolation("comparison uncertainty is absent from trusted packet")
        prediction = output_values[comparison.prediction_element_id]
        response = validated.scoring_responses[comparison.response_element_id]
        if prediction.shape != response.shape:
            raise SchemaViolation("prediction and response shapes differ")
        residuals.append(
            _whitened_squared_residuals(
                prediction,
                response,
                validated.uncertainties[comparison.uncertainty_id],
            )
        )
        observables.extend((comparison.prediction_element_id, comparison.response_element_id))
    if not residuals:
        raise SchemaViolation("at least one observable comparison is required")
    squared = np.concatenate(residuals)
    distance = math.sqrt(math.fsum(float(value) for value in squared) / squared.size)
    if not math.isfinite(distance):
        raise SchemaViolation("candidate distance is not finite")
    return distance, tuple(sorted(set(observables)))


def run_discovery_matrix(
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
) -> SyntheticDiscoveryResult:
    """Execute the registered Cartesian scenario × formula × parameter matrix.

    Results are steering diagnostics only.  The function has no real-data
    access surface and cannot emit an empirical support or rejection claim.
    """

    if not math.isfinite(distinct_gap) or distinct_gap < 0:
        raise SchemaViolation("distinct gap must be finite and nonnegative")
    validate_binding_catalogue(bindings, catalogue)
    ordered_scenarios = tuple(scenarios)
    scenario_ids = tuple(row.scenario_id for row in ordered_scenarios)
    if scenario_ids != tuple(sorted(set(scenario_ids))):
        raise SchemaViolation("scenarios must be unique and sorted")
    binding_ids = tuple(row.binding_id for row in bindings)
    if not binding_ids or binding_ids != tuple(sorted(set(binding_ids))):
        raise SchemaViolation("bindings must be unique and sorted")
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
    final_cells: list[DiscoveryMatrixCell] = []
    recovery_count = 0
    distinct_recovery_count = 0
    scored_count = 0

    for scenario in ordered_scenarios:
        validate_scenario_catalogue(scenario, catalogue)
        validated = scenario_values[scenario.scenario_id].validate(scenario)
        truth_formula_id = truth_formula_by_scenario[scenario.scenario_id]
        candidates: list[_ScoredCandidate] = []
        blocked_rows: dict[str, tuple[str, tuple[str, ...]]] = {}

        for binding in bindings:
            decision = decide_scenario_eligibility(binding, catalogue, scenario)
            if decision.status is not EligibilityStatus.ELIGIBLE:
                reasons = tuple(
                    sorted(
                        [
                            *(f"missing.{value}" for value in decision.missing_features),
                            *(f"forbidden.{value}" for value in decision.forbidden_features),
                        ]
                    )
                )
                ledger = ledger.append(
                    release=release,
                    binding=binding,
                    eligibility=decision,
                    adapter_sha256=None,
                    domain=scenario.domain,
                    experiment_id=scenario.experiment_id,
                )
                blocked_rows[binding.binding_id] = (decision.status.value, reasons)
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
                    metric_payload = {
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
                    diagnostics_payload = {
                        "deterministic_replay": execution.deterministic_replay,
                        "scenario_sha256": execution.scenario_sha256,
                        "binding_sha256": execution.binding_sha256,
                        "parameter_values_sha256": execution.parameter_values_sha256,
                        "parameter_cell_sha256": parameter_cell.content_sha256,
                        "real_response_used": False,
                    }
                    result_sha256 = execution.output_sha256
                    invalid = False
                    reason_codes: tuple[str, ...] = ()
                except (
                    SchemaViolation,
                    ArithmeticError,
                    RuntimeError,
                    TypeError,
                    ValueError,
                ) as error:  # retain each invalid cell; never pseudo-score it
                    distance = None
                    observable_ids = tuple(
                        sorted(
                            {
                                *(
                                    row.prediction_element_id
                                    for row in comparisons[scenario.scenario_id]
                                ),
                                *(
                                    row.response_element_id
                                    for row in comparisons[scenario.scenario_id]
                                ),
                            }
                        )
                    )
                    failure_payload = {
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                        "binding_id": binding.binding_id,
                        "scenario_id": scenario.scenario_id,
                        "parameter_cell_id": parameter_cell.parameter_cell_id,
                    }
                    result_sha256 = canonical_sha256(failure_payload)
                    metric_payload = {"metric": "NOT_EVALUATED_NUMERICAL_INVALID"}
                    diagnostics_payload = {
                        **failure_payload,
                        "parameter_cell_sha256": parameter_cell.content_sha256,
                        "real_response_used": False,
                    }
                    invalid = True
                    reason_codes = ("execution_or_scoring_invalid",)
                candidates.append(
                    _ScoredCandidate(
                        binding=binding,
                        adapter_sha256=registration.adapter_sha256,
                        parameter_cell=parameter_cell,
                        result_sha256=result_sha256,
                        metrics_sha256=canonical_sha256(metric_payload),
                        diagnostics_sha256=canonical_sha256(diagnostics_payload),
                        observable_ids=observable_ids,
                        distance=distance,
                        invalid=invalid,
                        reason_codes=reason_codes,
                    )
                )

        valid_candidates = [row for row in candidates if not row.invalid]
        ordered = sorted(
            valid_candidates,
            key=lambda row: (
                row.distance if row.distance is not None else math.inf,
                row.binding.formula_id,
                row.binding.binding_id,
                row.parameter_cell.parameter_cell_id,
            ),
        )
        winner = ordered[0] if ordered else None
        winner_distance = winner.distance if winner is not None else None
        runner_up_distance = ordered[1].distance if len(ordered) > 1 else math.inf
        gap = (
            runner_up_distance - winner_distance
            if runner_up_distance is not None and winner_distance is not None
            else -math.inf
        )
        distinct = bool(winner is not None and gap >= distinct_gap)
        truth_recovered = bool(winner is not None and winner.binding.formula_id == truth_formula_id)
        recovery_count += int(truth_recovered)
        distinct_recovery_count += int(truth_recovered and distinct)

        by_identity = {
            (row.binding.binding_id, row.parameter_cell.parameter_cell_id): row
            for row in candidates
        }
        for binding in bindings:
            if binding.binding_id in blocked_rows:
                eligibility, blocked_reasons = blocked_rows[binding.binding_id]
                final_cells.append(
                    DiscoveryMatrixCell(
                        scenario.scenario_id,
                        scenario.object_id,
                        truth_formula_id,
                        binding.formula_id,
                        binding.binding_id,
                        None,
                        eligibility,
                        eligibility,
                        None,
                        None,
                        None,
                        None,
                        False,
                        False,
                        False,
                        blocked_reasons,
                    )
                )
                continue
            for parameter_cell in parameter_cells[binding.binding_id]:
                candidate = by_identity[(binding.binding_id, parameter_cell.parameter_cell_id)]
                is_winner = winner is not None and candidate is winner
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
                decision = decide_scenario_eligibility(binding, catalogue, scenario)
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
                final_cells.append(
                    DiscoveryMatrixCell(
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
                        is_winner,
                        truth_recovered,
                        distinct,
                        reasons,
                    )
                )
                scored_count += int(not candidate.invalid)

    cells = tuple(final_cells)
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
    return SyntheticDiscoveryResult(
        cells=cells,
        ledger=ledger,
        scenario_count=len(ordered_scenarios),
        attempted_cell_count=len(cells),
        scored_cell_count=scored_count,
        truth_recovery_count=recovery_count,
        distinct_truth_recovery_count=distinct_recovery_count,
        content_sha256=canonical_sha256(body),
    )


__all__ = [
    "SCHEMA_VERSION",
    "DiscoveryMatrixCell",
    "ObservableComparison",
    "ParameterCell",
    "ScenarioRuntimeValues",
    "SyntheticDiscoveryResult",
    "run_discovery_matrix",
]
