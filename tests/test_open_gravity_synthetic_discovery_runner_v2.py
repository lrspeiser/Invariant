from __future__ import annotations

import importlib.util
from dataclasses import replace
from pathlib import Path

import pytest

import sigma_theory_compiler.open_gravity_formula_adapter_registry_v1 as adapter_registry
from sigma_theory_compiler.open_gravity_formula_adapter_registry_v1 import (
    AdapterRegistration,
)
from sigma_theory_compiler.open_gravity_formula_execution_protocol_v1 import (
    BindingStatus,
)
from sigma_theory_compiler.open_gravity_synthetic_discovery_runner_v1 import (
    ObservableComparison,
    ParameterCell,
)
from sigma_theory_compiler.open_gravity_synthetic_discovery_runner_v2 import (
    run_discovery_matrix_v2,
)
from sigma_theory_compiler.open_gravity_synthetic_replay_ledger_v1 import (
    DiscoveryStatus,
)
from sigma_theory_compiler.open_gravity_synthetic_scenario_packet_v1 import (
    UncertaintyRef,
)
from sigma_theory_compiler.sigma_core import SchemaViolation

_FIXTURE_PATH = Path(__file__).with_name("test_open_gravity_synthetic_discovery_runner_v1.py")
_SPEC = importlib.util.spec_from_file_location("runner_v1_fixtures", _FIXTURE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_FIXTURES = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_FIXTURES)


def _comparison() -> tuple[ObservableComparison, ...]:
    return (
        ObservableComparison(
            "prediction.vector.acceleration",
            "response.vector.acceleration",
            "response.diagonal-covariance",
        ),
    )


def _two_binding_run(
    *, comparator_scale: int = 1, control_scale: int = 2, distinct_gap: float = 0.1
):
    scenario, values = _FIXTURES._scenario_and_values()
    comparator = _FIXTURES._binding(
        "binding.comparator.v1", "scale-comparator", BindingStatus.EXECUTABLE
    )
    control = _FIXTURES._binding("binding.control.v1", "scale-control", BindingStatus.EXECUTABLE)
    return run_discovery_matrix_v2(
        catalogue=_FIXTURES._catalogue(),
        release=_FIXTURES._release(),
        scenarios=(scenario,),
        scenario_values={scenario.scenario_id: values},
        truth_formula_by_scenario={scenario.scenario_id: "scale-control"},
        bindings=(comparator, control),
        adapters=(
            AdapterRegistration.create("adapter.comparator.v1", comparator),
            AdapterRegistration.create("adapter.control.v1", control),
        ),
        parameter_cells={
            comparator.binding_id: (
                ParameterCell(
                    "scale.comparator",
                    {"scale_denominator": 1, "scale_numerator": comparator_scale},
                ),
            ),
            control.binding_id: (
                ParameterCell(
                    "scale.control",
                    {"scale_denominator": 1, "scale_numerator": control_scale},
                ),
            ),
        },
        comparisons={scenario.scenario_id: _comparison()},
        distinct_gap=distinct_gap,
        ledger_id="gravity.synthetic.runner.v2.ledger",
    )


def test_normal_matrix_is_complete_and_each_cell_links_to_ledger() -> None:
    result = _two_binding_run()
    assert result.attempted_cell_count == 2
    assert result.scored_cell_count == 2
    assert result.truth_recovery_count == 1
    assert result.distinct_truth_recovery_count == 1
    assert len(result.ledger.entries) == 4
    assert len({cell.ledger_entry_sha256 for cell in result.cells}) == 2
    winner = next(cell for cell in result.cells if cell.winner)
    assert winner.formula_id == "scale-control"
    assert winner.discovery_status == DiscoveryStatus.PROMISING_DISTINCT_SIGNATURE.value


def test_exact_tie_and_single_candidate_are_never_distinct() -> None:
    tied = _two_binding_run(comparator_scale=2, control_scale=2, distinct_gap=0.0)
    assert sum(cell.winner for cell in tied.cells) == 2
    assert tied.truth_recovery_count == 1
    assert tied.distinct_truth_recovery_count == 0
    assert {cell.discovery_status for cell in tied.cells} == {DiscoveryStatus.UNDERPOWERED.value}

    scenario, values = _FIXTURES._scenario_and_values()
    control = _FIXTURES._binding("binding.control.v1", "scale-control", BindingStatus.EXECUTABLE)
    single = run_discovery_matrix_v2(
        catalogue=_FIXTURES._catalogue(),
        release=_FIXTURES._release(),
        scenarios=(scenario,),
        scenario_values={scenario.scenario_id: values},
        truth_formula_by_scenario={scenario.scenario_id: "scale-control"},
        bindings=(control,),
        adapters=(AdapterRegistration.create("adapter.control.v1", control),),
        parameter_cells={
            control.binding_id: (
                ParameterCell("scale.control", {"scale_denominator": 1, "scale_numerator": 2}),
            )
        },
        comparisons={scenario.scenario_id: _comparison()},
        distinct_gap=0.0,
        ledger_id="gravity.synthetic.runner.v2.single",
    )
    assert single.truth_recovery_count == 1
    assert single.distinct_truth_recovery_count == 0
    assert single.cells[0].discovery_status == DiscoveryStatus.UNDERPOWERED.value


def test_ineligible_executable_keeps_every_parameter_cell() -> None:
    scenario, values = _FIXTURES._scenario_and_values()
    incompatible = replace(
        _FIXTURES._binding("binding.incompatible.v1", "incompatible-law", BindingStatus.EXECUTABLE),
        geometry_support=("spherical1d",),
    )
    result = run_discovery_matrix_v2(
        catalogue=_FIXTURES._catalogue(),
        release=_FIXTURES._release(),
        scenarios=(scenario,),
        scenario_values={scenario.scenario_id: values},
        truth_formula_by_scenario={scenario.scenario_id: "incompatible-law"},
        bindings=(incompatible,),
        adapters=(AdapterRegistration.create("adapter.incompatible.v1", incompatible),),
        parameter_cells={
            incompatible.binding_id: (
                ParameterCell("scale.one", {"scale_denominator": 1, "scale_numerator": 1}),
                ParameterCell("scale.two", {"scale_denominator": 1, "scale_numerator": 2}),
            )
        },
        comparisons={scenario.scenario_id: _comparison()},
        distinct_gap=0.1,
        ledger_id="gravity.synthetic.runner.v2.incompatible",
    )
    assert result.attempted_cell_count == 2
    assert result.scored_cell_count == 0
    assert len(result.ledger.entries) == 2
    assert {cell.parameter_cell_id for cell in result.cells} == {"scale.one", "scale.two"}
    assert {cell.eligibility for cell in result.cells} == {"INCOMPATIBLE_FEATURE_SET"}
    assert len({cell.ledger_entry_sha256 for cell in result.cells}) == 2


def test_truth_formula_must_be_a_registered_candidate() -> None:
    scenario, values = _FIXTURES._scenario_and_values()
    control = _FIXTURES._binding("binding.control.v1", "scale-control", BindingStatus.EXECUTABLE)
    with pytest.raises(SchemaViolation, match="truth formula is absent"):
        run_discovery_matrix_v2(
            catalogue=_FIXTURES._catalogue(),
            release=_FIXTURES._release(),
            scenarios=(scenario,),
            scenario_values={scenario.scenario_id: values},
            truth_formula_by_scenario={scenario.scenario_id: "not-a-candidate"},
            bindings=(control,),
            adapters=(AdapterRegistration.create("adapter.control.v1", control),),
            parameter_cells={
                control.binding_id: (
                    ParameterCell(
                        "scale.control",
                        {"scale_denominator": 1, "scale_numerator": 2},
                    ),
                )
            },
            comparisons={scenario.scenario_id: _comparison()},
            distinct_gap=0.1,
            ledger_id="gravity.synthetic.runner.v2.badtruth",
        )


def test_response_calibrated_release_is_rejected() -> None:
    scenario, values = _FIXTURES._scenario_and_values()
    control = _FIXTURES._binding("binding.control.v1", "scale-control", BindingStatus.EXECUTABLE)
    with pytest.raises(SchemaViolation, match="response-calibrated"):
        run_discovery_matrix_v2(
            catalogue=_FIXTURES._catalogue(),
            release=replace(_FIXTURES._release(), response_calibrated=True),
            scenarios=(scenario,),
            scenario_values={scenario.scenario_id: values},
            truth_formula_by_scenario={scenario.scenario_id: "scale-control"},
            bindings=(control,),
            adapters=(AdapterRegistration.create("adapter.control.v1", control),),
            parameter_cells={
                control.binding_id: (
                    ParameterCell(
                        "scale.control",
                        {"scale_denominator": 1, "scale_numerator": 2},
                    ),
                )
            },
            comparisons={scenario.scenario_id: _comparison()},
            distinct_gap=0.1,
            ledger_id="gravity.synthetic.runner.v2.calibrated",
        )


def test_covariance_target_must_match_scoring_response() -> None:
    scenario, values = _FIXTURES._scenario_and_values()
    wrong = UncertaintyRef(
        "response.diagonal-covariance",
        "source.vector.acceleration",
        "diagonal-covariance",
        scenario.uncertainties[0].artifact_path,
        scenario.uncertainties[0].artifact_sha256,
    )
    forged = replace(scenario, uncertainties=(wrong,))
    control = _FIXTURES._binding("binding.control.v1", "scale-control", BindingStatus.EXECUTABLE)
    with pytest.raises(SchemaViolation, match="covariance targets"):
        run_discovery_matrix_v2(
            catalogue=_FIXTURES._catalogue(),
            release=_FIXTURES._release(),
            scenarios=(forged,),
            scenario_values={forged.scenario_id: values},
            truth_formula_by_scenario={forged.scenario_id: "scale-control"},
            bindings=(control,),
            adapters=(AdapterRegistration.create("adapter.control.v1", control),),
            parameter_cells={
                control.binding_id: (
                    ParameterCell(
                        "scale.control",
                        {"scale_denominator": 1, "scale_numerator": 2},
                    ),
                )
            },
            comparisons={forged.scenario_id: _comparison()},
            distinct_gap=0.1,
            ledger_id="gravity.synthetic.runner.v2.badcovariance",
        )


def test_oserror_is_retained_per_cell(monkeypatch) -> None:
    def unavailable_adapter(_features, _parameters):
        raise OSError("synthetic adapter unavailable")

    monkeypatch.setattr(adapter_registry, "vector_scale_control", unavailable_adapter)
    result = _two_binding_run()
    assert result.scored_cell_count == 0
    assert result.truth_recovery_count == 0
    assert {cell.discovery_status for cell in result.cells} == {
        DiscoveryStatus.NUMERICAL_INVALID.value
    }
    assert all(cell.result_sha256 is not None for cell in result.cells)
    assert len(result.ledger.entries) == 4
