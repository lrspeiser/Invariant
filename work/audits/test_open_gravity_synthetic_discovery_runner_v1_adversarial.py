"""Independent adversarial audit probes; not part of the subject test suite."""

from __future__ import annotations

import importlib.util
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import sigma_theory_compiler.open_gravity_formula_adapter_registry_v1 as adapter_registry
from sigma_theory_compiler.open_gravity_formula_adapter_registry_v1 import AdapterRegistration
from sigma_theory_compiler.open_gravity_formula_execution_protocol_v1 import BindingStatus
from sigma_theory_compiler.open_gravity_synthetic_discovery_runner_v1 import (
    ObservableComparison,
    ParameterCell,
    ScenarioRuntimeValues,
    run_discovery_matrix,
)
from sigma_theory_compiler.open_gravity_synthetic_scenario_packet_v1 import (
    UncertaintyRef,
    array_sha256,
)
from sigma_theory_compiler.sigma_core import SchemaViolation

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "runner_subject_tests",
    ROOT / "tests/test_open_gravity_synthetic_discovery_runner_v1.py",
)
assert SPEC is not None and SPEC.loader is not None
fixture = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fixture)


def _invoke(
    *,
    bindings,
    parameter_cells,
    truth_formula="scale-control",
    distinct_gap=0.1,
    scenario=None,
    values=None,
    comparisons=None,
    release=None,
):
    if scenario is None or values is None:
        scenario, values = fixture._scenario_and_values()
    if comparisons is None:
        comparisons = (
            ObservableComparison(
                "prediction.vector.acceleration",
                "response.vector.acceleration",
                "response.diagonal-covariance",
            ),
        )
    executable = [row for row in bindings if row.status is BindingStatus.EXECUTABLE]
    return run_discovery_matrix(
        catalogue=fixture._catalogue(),
        release=release or fixture._release(),
        scenarios=(scenario,),
        scenario_values={scenario.scenario_id: values},
        truth_formula_by_scenario={scenario.scenario_id: truth_formula},
        bindings=bindings,
        adapters=tuple(
            AdapterRegistration.create(f"adapter.audit.{index}.v1", binding)
            for index, binding in enumerate(executable)
        ),
        parameter_cells=parameter_cells,
        comparisons={scenario.scenario_id: comparisons},
        distinct_gap=distinct_gap,
        ledger_id="gravity.synthetic.runner.audit.ledger",
    )


def _two_bindings():
    comparator = fixture._binding(
        "binding.comparator.v1", "scale-comparator", BindingStatus.EXECUTABLE
    )
    control = fixture._binding(
        "binding.control.v1", "scale-control", BindingStatus.EXECUTABLE
    )
    return comparator, control


def test_exact_tie_at_zero_threshold_cannot_be_a_distinct_unique_win() -> None:
    comparator, control = _two_bindings()
    two = ParameterCell(
        "scale.two", {"scale_denominator": 1, "scale_numerator": 2}
    )
    result = _invoke(
        bindings=(comparator, control),
        parameter_cells={comparator.binding_id: (two,), control.binding_id: (two,)},
        distinct_gap=0.0,
    )
    assert result.distinct_truth_recovery_count == 0
    assert sum(cell.winner for cell in result.cells) != 1


def test_one_candidate_has_no_comparator_and_cannot_be_distinct() -> None:
    control = fixture._binding(
        "binding.control.v1", "scale-control", BindingStatus.EXECUTABLE
    )
    result = _invoke(
        bindings=(control,),
        parameter_cells={
            control.binding_id: (
                ParameterCell(
                    "scale.two", {"scale_denominator": 1, "scale_numerator": 2}
                ),
            )
        },
    )
    assert result.distinct_truth_recovery_count == 0
    assert result.cells[0].distinct is False


def test_scenario_ineligible_executable_retains_each_parameter_cell() -> None:
    comparator, control = _two_bindings()
    cluster_only = replace(
        comparator,
        binding_id="binding.cluster-only.v1",
        formula_id="cluster-only",
        domains=("cluster",),
    )
    one = ParameterCell("scale.one", {"scale_denominator": 1, "scale_numerator": 1})
    two = ParameterCell("scale.two", {"scale_denominator": 1, "scale_numerator": 2})
    result = _invoke(
        bindings=(cluster_only, control),
        parameter_cells={cluster_only.binding_id: (one, two), control.binding_id: (two,)},
    )
    blocked = [cell for cell in result.cells if cell.binding_id == cluster_only.binding_id]
    assert len(blocked) == 2
    assert {cell.parameter_cell_id for cell in blocked} == {"scale.one", "scale.two"}


def test_truth_formula_identity_must_name_a_registered_candidate() -> None:
    comparator, control = _two_bindings()
    one = ParameterCell("scale.one", {"scale_denominator": 1, "scale_numerator": 1})
    two = ParameterCell("scale.two", {"scale_denominator": 1, "scale_numerator": 2})
    with pytest.raises(SchemaViolation, match="truth"):
        _invoke(
            bindings=(comparator, control),
            parameter_cells={comparator.binding_id: (one,), control.binding_id: (two,)},
            truth_formula="not-a-candidate",
        )


def test_response_calibrated_release_is_rejected_by_response_blind_runner() -> None:
    comparator, control = _two_bindings()
    one = ParameterCell("scale.one", {"scale_denominator": 1, "scale_numerator": 1})
    two = ParameterCell("scale.two", {"scale_denominator": 1, "scale_numerator": 2})
    with pytest.raises(SchemaViolation, match="response"):
        _invoke(
            bindings=(comparator, control),
            parameter_cells={comparator.binding_id: (one,), control.binding_id: (two,)},
            release=replace(fixture._release(), response_calibrated=True),
        )


def test_uncertainty_must_apply_to_the_selected_response() -> None:
    scenario, values = fixture._scenario_and_values()
    source_variance = np.ones((2, 3), dtype=np.float64) * 9.0
    source_uncertainty = UncertaintyRef(
        "source.diagonal-covariance",
        "source.vector.acceleration",
        "diagonal-covariance",
        "uncertainty/source-variance.npy",
        array_sha256(source_variance),
    )
    scenario = replace(
        scenario,
        uncertainties=tuple(sorted((*scenario.uncertainties, source_uncertainty), key=lambda row: row.uncertainty_id)),
    )
    values = ScenarioRuntimeValues(
        values.formula_values,
        values.response_values,
        values.truth_values,
        {
            **values.uncertainty_values,
            "source.diagonal-covariance": source_variance,
        },
    )
    control = fixture._binding(
        "binding.control.v1", "scale-control", BindingStatus.EXECUTABLE
    )
    with pytest.raises(SchemaViolation, match="uncertainty"):
        _invoke(
            bindings=(control,),
            parameter_cells={
                control.binding_id: (
                    ParameterCell(
                        "scale.two", {"scale_denominator": 1, "scale_numerator": 2}
                    ),
                )
            },
            scenario=scenario,
            values=values,
            comparisons=(
                ObservableComparison(
                    "prediction.vector.acceleration",
                    "response.vector.acceleration",
                    "source.diagonal-covariance",
                ),
            ),
        )


def test_all_schema_invalid_candidates_are_retained() -> None:
    comparator, control = _two_bindings()
    original = adapter_registry.vector_scale_control

    def fail_all(features, parameters):
        del features, parameters
        raise SchemaViolation("audit-all-invalid")

    adapter_registry.vector_scale_control = fail_all
    try:
        one = ParameterCell("scale.one", {"scale_denominator": 1, "scale_numerator": 1})
        two = ParameterCell("scale.two", {"scale_denominator": 1, "scale_numerator": 2})
        result = _invoke(
            bindings=(comparator, control),
            parameter_cells={comparator.binding_id: (one,), control.binding_id: (two,)},
        )
    finally:
        adapter_registry.vector_scale_control = original
    assert result.scored_cell_count == 0
    assert result.truth_recovery_count == 0
    assert not any(cell.winner for cell in result.cells)
    assert {cell.discovery_status for cell in result.cells} == {"NUMERICAL_INVALID"}


def test_eligible_cartesian_parameter_cells_are_complete() -> None:
    comparator, control = _two_bindings()
    one = ParameterCell("scale.one", {"scale_denominator": 1, "scale_numerator": 1})
    three = ParameterCell("scale.three", {"scale_denominator": 1, "scale_numerator": 3})
    two = ParameterCell("scale.two", {"scale_denominator": 1, "scale_numerator": 2})
    result = _invoke(
        bindings=(comparator, control),
        parameter_cells={
            comparator.binding_id: (one, three),
            control.binding_id: (two,),
        },
    )
    assert result.attempted_cell_count == 3
    assert len(result.ledger.entries) == 6
    assert {
        (cell.binding_id, cell.parameter_cell_id) for cell in result.cells
    } == {
        (comparator.binding_id, "scale.one"),
        (comparator.binding_id, "scale.three"),
        (control.binding_id, "scale.two"),
    }


def test_valid_full_psd_covariance_is_whitened() -> None:
    scenario, values = fixture._scenario_and_values()
    covariance = np.diag(np.asarray((1.0, 2.0, 3.0, 4.0, 5.0, 6.0)))
    full = UncertaintyRef(
        "response.full-covariance",
        "response.vector.acceleration",
        "covariance",
        "uncertainty/full-covariance.npy",
        array_sha256(covariance),
    )
    scenario = replace(scenario, uncertainties=(full,))
    values = ScenarioRuntimeValues(
        values.formula_values,
        values.response_values,
        values.truth_values,
        {"response.full-covariance": covariance},
    )
    comparator, control = _two_bindings()
    one = ParameterCell("scale.one", {"scale_denominator": 1, "scale_numerator": 1})
    two = ParameterCell("scale.two", {"scale_denominator": 1, "scale_numerator": 2})
    result = _invoke(
        bindings=(comparator, control),
        parameter_cells={comparator.binding_id: (one,), control.binding_id: (two,)},
        scenario=scenario,
        values=values,
        comparisons=(
            ObservableComparison(
                "prediction.vector.acceleration",
                "response.vector.acceleration",
                "response.full-covariance",
            ),
        ),
    )
    winner = next(cell for cell in result.cells if cell.winner)
    assert winner.formula_id == "scale-control"
    assert winner.whitened_rmse == 0.0


def test_tiny_negative_covariance_mode_cannot_hide_a_residual() -> None:
    scenario, values = fixture._scenario_and_values()
    source = values.formula_values["source.vector.acceleration"]
    response = np.array(source, copy=True)
    response[-1, -1] += 1.0
    response_ref = replace(
        scenario.scoring_responses[0], value_sha256=array_sha256(response)
    )
    covariance = np.diag(np.asarray((1.0, 1.0, 1.0, 1.0, 1.0, -5e-13)))
    full = UncertaintyRef(
        "response.full-covariance",
        "response.vector.acceleration",
        "covariance",
        "uncertainty/full-covariance.npy",
        array_sha256(covariance),
    )
    scenario = replace(
        scenario, scoring_responses=(response_ref,), uncertainties=(full,)
    )
    values = ScenarioRuntimeValues(
        values.formula_values,
        {"response.vector.acceleration": response},
        values.truth_values,
        {"response.full-covariance": covariance},
    )
    control = fixture._binding(
        "binding.control.v1", "scale-control", BindingStatus.EXECUTABLE
    )
    result = _invoke(
        bindings=(control,),
        parameter_cells={
            control.binding_id: (
                ParameterCell(
                    "scale.one", {"scale_denominator": 1, "scale_numerator": 1}
                ),
            )
        },
        scenario=scenario,
        values=values,
        comparisons=(
            ObservableComparison(
                "prediction.vector.acceleration",
                "response.vector.acceleration",
                "response.full-covariance",
            ),
        ),
    )
    assert result.scored_cell_count == 0
    assert result.cells[0].discovery_status == "NUMERICAL_INVALID"


def test_uncaught_adapter_exception_is_retained_as_numerical_invalid() -> None:
    comparator, control = _two_bindings()
    original = adapter_registry.vector_scale_control

    def fail_with_io_error(features, parameters):
        del features, parameters
        raise OSError("audit-adapter-io-failure")

    adapter_registry.vector_scale_control = fail_with_io_error
    try:
        one = ParameterCell("scale.one", {"scale_denominator": 1, "scale_numerator": 1})
        two = ParameterCell("scale.two", {"scale_denominator": 1, "scale_numerator": 2})
        result = _invoke(
            bindings=(comparator, control),
            parameter_cells={comparator.binding_id: (one,), control.binding_id: (two,)},
        )
    finally:
        adapter_registry.vector_scale_control = original
    assert {cell.discovery_status for cell in result.cells} == {"NUMERICAL_INVALID"}


def test_result_hash_body_contains_no_native_float() -> None:
    result = fixture._run()

    def contains_float(value):
        if isinstance(value, float):
            return True
        if isinstance(value, dict):
            return any(contains_float(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return any(contains_float(item) for item in value)
        return False

    assert not contains_float(result.to_dict())


def test_ledger_eligible_entries_are_immediately_completed() -> None:
    result = fixture._run()
    entries = result.ledger.entries
    for index, entry in enumerate(entries):
        if entry.status.value == "ELIGIBLE_NOT_RUN":
            assert index + 1 < len(entries)
            completed = entries[index + 1]
            assert completed.binding_sha256 == entry.binding_sha256
            assert completed.prior_entry_sha256 == entry.entry_sha256
            assert completed.scenario_id is not None
