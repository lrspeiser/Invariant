from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import sigma_theory_compiler.open_gravity_formula_adapter_registry_v1 as adapter_registry
import sigma_theory_compiler.open_gravity_synthetic_discovery_runner_v1 as discovery_runner
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
    run_discovery_matrix,
)
from sigma_theory_compiler.open_gravity_synthetic_replay_ledger_v1 import (
    DiscoveryStatus,
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
)
from sigma_theory_compiler.sigma_core import SchemaViolation

HASH = "7" * 64
EXPERIMENT = "galaxy.synthetic.runner.v1"
PARAMETER_SCHEMA_SHA256 = "7f14379df933fa469a78189b69371a84e9ca295080f86b48a1814cef5e902e9f"


def _element(element_id: str, role: DataRole, rank: int) -> DataElement:
    return DataElement(
        element_id=element_id,
        namespace=element_id.rsplit(".", 1)[0],
        physical_quantity="acceleration" if rank else "identifier",
        tensor_rank=rank,
        si_dimension=(0, 1, -2, 0, 0, 0, 0) if rank else (0, 0, 0, 0, 0, 0, 0),
        canonical_unit="m s^-2" if rank else "1",
        frame="source",
        support="samples",
        axes=("sample", "component") if rank else ("sample",),
        component="total",
        derivation_parents=(),
        uncertainty=UncertaintyKind.COVARIANCE,
        availability=Availability.SYNTHETIC_ONLY,
        experiment_roles=(ExperimentRole(EXPERIMENT, role),),
        provenance_sha256=HASH,
    )


def _catalogue():
    return catalogue_from_elements(
        "gravity.synthetic.runner.elements",
        "v1.0.0",
        [
            _element("prediction.vector.acceleration", DataRole.DERIVED, 1),
            _element("response.vector.acceleration", DataRole.SCORING_ONLY_RESPONSE, 1),
            _element("source.vector.acceleration", DataRole.FORMULA_INPUT, 1),
            _element("truth.scalar.injection-id", DataRole.LATENT_SYNTHETIC_TRUTH, 0),
        ],
    )


def _scenario_and_values() -> tuple[ScenarioDescriptor, ScenarioRuntimeValues]:
    source = np.arange(1, 7, dtype=np.float64).reshape(2, 3)
    response = 2.0 * source
    truth = np.array([2, 2], dtype=np.int64)
    variance = np.ones((2, 3), dtype=np.float64)
    scenario = ScenarioDescriptor(
        scenario_id="galaxy.runner.fixture.v1",
        object_id="synthetic.object.001",
        experiment_id=EXPERIMENT,
        domain="galaxy",
        geometry_mode="nonspherical3d",
        time_mode="static",
        coordinate_frame="source",
        axes=(
            AxisSpec("component", 3, None, None),
            AxisSpec("sample", 2, None, None),
        ),
        formula_features=(
            FeatureValueRef(
                "source.vector.acceleration",
                "features/source.npy",
                array_sha256(source),
                "float64",
                (2, 3),
                ("sample", "component"),
                "m s^-2",
                "source",
            ),
        ),
        scoring_responses=(
            FeatureValueRef(
                "response.vector.acceleration",
                "responses/acceleration.npy",
                array_sha256(response),
                "float64",
                (2, 3),
                ("sample", "component"),
                "m s^-2",
                "source",
            ),
        ),
        hidden_truth=(
            FeatureValueRef(
                "truth.scalar.injection-id",
                "truth/injection.npy",
                array_sha256(truth),
                "int64",
                (2,),
                ("sample",),
                "1",
                "source",
            ),
        ),
        expected_predictions=(
            EmittedPredictionSpec(
                "prediction.vector.acceleration",
                "predictions/acceleration.npy",
                "float64",
                (2, 3),
                ("sample", "component"),
                "m s^-2",
                "source",
            ),
        ),
        uncertainties=(
            UncertaintyRef(
                "response.diagonal-covariance",
                "response.vector.acceleration",
                "diagonal-covariance",
                "uncertainty/variance.npy",
                array_sha256(variance),
            ),
        ),
        anchors=(AnchorBinding("galaxy.source.v1", "anchors/source.json", HASH),),
        seed_lineage=SeedLineage(
            11,
            "galaxy.runner.fixture.v1",
            "synthetic.object.001",
            "scale-control",
            0,
            0,
        ),
    )
    values = ScenarioRuntimeValues(
        formula_values={"source.vector.acceleration": source},
        response_values={"response.vector.acceleration": response},
        truth_values={"truth.scalar.injection-id": truth},
        uncertainty_values={"response.diagonal-covariance": variance},
    )
    return scenario, values


def _binding(binding_id: str, formula_id: str, status: BindingStatus) -> FormulaExecutionBinding:
    executable = status is BindingStatus.EXECUTABLE
    return FormulaExecutionBinding(
        binding_id=binding_id,
        formula_id=formula_id,
        formula_version="v1.0.0",
        formula_sha256=HASH,
        status=status,
        entrypoint=(
            "sigma_theory_compiler.open_gravity_formula_adapter_registry_v1:vector_scale_control"
            if executable
            else None
        ),
        required_features=("source.vector.acceleration",) if executable else (),
        optional_features=(),
        emitted_features=("prediction.vector.acceleration",) if executable else (),
        domains=("galaxy",),
        geometry_support=("nonspherical3d",),
        time_support=("static",),
        parameter_schema_path="configs/open_gravity_vector_scale_control_parameters_v1.schema.json",
        parameter_schema_sha256=PARAMETER_SCHEMA_SHA256,
        approximation_ceiling="synthetic runner fixture",
        health_gates=("dimension",),
        resource_bounds=ResourceBounds(10, 1_000_000, 100_000),
    )


def _release() -> SyntheticSuiteRelease:
    return SyntheticSuiteRelease(
        suite_id="gravity.synthetic.runner",
        version="v1.0.0",
        release_sha256=HASH,
        ontology_sha256=HASH,
        generator_sha256=HASH,
        observation_operator_sha256=HASH,
        changed_feature_ids=(),
        change_level="MAJOR",
        response_calibrated=False,
    )


def _run(distinct_gap: float = 0.1):
    scenario, values = _scenario_and_values()
    comparator = _binding("binding.comparator.v1", "scale-comparator", BindingStatus.EXECUTABLE)
    control = _binding("binding.control.v1", "scale-control", BindingStatus.EXECUTABLE)
    unadapted = _binding("binding.unadapted.v1", "unadapted-law", BindingStatus.UNADAPTED)
    bindings = (comparator, control, unadapted)
    return run_discovery_matrix(
        catalogue=_catalogue(),
        release=_release(),
        scenarios=(scenario,),
        scenario_values={scenario.scenario_id: values},
        truth_formula_by_scenario={scenario.scenario_id: "scale-control"},
        bindings=bindings,
        adapters=(
            AdapterRegistration.create("adapter.comparator.v1", comparator),
            AdapterRegistration.create("adapter.control.v1", control),
        ),
        parameter_cells={
            comparator.binding_id: (
                ParameterCell("scale.one", {"scale_denominator": 1, "scale_numerator": 1}),
            ),
            control.binding_id: (
                ParameterCell("scale.two", {"scale_denominator": 1, "scale_numerator": 2}),
            ),
            unadapted.binding_id: (),
        },
        comparisons={
            scenario.scenario_id: (
                ObservableComparison(
                    "prediction.vector.acceleration",
                    "response.vector.acceleration",
                    "response.diagonal-covariance",
                ),
            )
        },
        distinct_gap=distinct_gap,
        ledger_id="gravity.synthetic.runner.ledger",
    )


def test_matrix_executes_truth_and_retains_unadapted_without_pseudo_score() -> None:
    result = _run()
    assert result.scenario_count == 1
    assert result.attempted_cell_count == 3
    assert result.scored_cell_count == 2
    assert result.truth_recovery_count == 1
    assert result.distinct_truth_recovery_count == 1
    assert len(result.ledger.entries) == 5
    winner = next(row for row in result.cells if row.winner)
    assert winner.formula_id == "scale-control"
    assert winner.whitened_rmse == 0.0
    assert winner.discovery_status == DiscoveryStatus.PROMISING_DISTINCT_SIGNATURE.value
    blocked = next(row for row in result.cells if row.formula_id == "unadapted-law")
    assert blocked.eligibility == "UNADAPTED"
    assert blocked.whitened_rmse is None
    assert blocked.result_sha256 is None


def test_large_gap_threshold_reports_underpowered_not_discovery() -> None:
    result = _run(distinct_gap=100.0)
    assert result.truth_recovery_count == 1
    assert result.distinct_truth_recovery_count == 0
    assert {row.discovery_status for row in result.cells if row.whitened_rmse is not None} == {
        DiscoveryStatus.UNDERPOWERED.value
    }


def test_response_and_truth_maps_are_required_but_never_formula_inputs() -> None:
    scenario, values = _scenario_and_values()
    comparator = _binding("binding.comparator.v1", "scale-comparator", BindingStatus.EXECUTABLE)
    with pytest.raises(SchemaViolation, match="runtime/truth/comparison"):
        run_discovery_matrix(
            catalogue=_catalogue(),
            release=_release(),
            scenarios=(scenario,),
            scenario_values={},
            truth_formula_by_scenario={scenario.scenario_id: "scale-comparator"},
            bindings=(comparator,),
            adapters=(AdapterRegistration.create("adapter.comparator.v1", comparator),),
            parameter_cells={
                comparator.binding_id: (
                    ParameterCell("scale.one", {"scale_denominator": 1, "scale_numerator": 1}),
                )
            },
            comparisons={scenario.scenario_id: ()},
            distinct_gap=0.1,
            ledger_id="gravity.synthetic.runner.ledger",
        )
    poisoned = replace(
        values,
        formula_values={
            "source.vector.acceleration": values.formula_values["source.vector.acceleration"],
            "response.vector.acceleration": values.response_values["response.vector.acceleration"],
        },
    )
    with pytest.raises(SchemaViolation, match="formula values differ"):
        poisoned.validate(scenario)


def test_result_hash_and_replay_are_deterministic() -> None:
    first = _run()
    second = _run()
    assert first.content_sha256 == second.content_sha256
    assert first.to_dict() == second.to_dict()


def test_numerical_failure_is_retained_and_other_cells_continue(monkeypatch) -> None:
    original = adapter_registry.vector_scale_control

    def fail_one_cell(features, parameters):
        if parameters["scale_numerator"] == 1:
            raise SchemaViolation("registered synthetic failure")
        return original(features, parameters)

    monkeypatch.setattr(adapter_registry, "vector_scale_control", fail_one_cell)
    result = _run()
    invalid = next(row for row in result.cells if row.formula_id == "scale-comparator")
    assert invalid.discovery_status == DiscoveryStatus.NUMERICAL_INVALID.value
    assert invalid.whitened_rmse is None
    assert invalid.result_sha256 is not None
    assert invalid.reason_codes == ("execution_or_scoring_invalid",)
    assert result.scored_cell_count == 1
    assert result.truth_recovery_count == 1


def test_singular_covariance_enforces_zero_variance_support() -> None:
    diagonal = discovery_runner._whitened_squared_residuals(
        np.array([1.0, 2.0]),
        np.array([1.0, 1.0]),
        np.array([0.0, 1.0]),
    )
    assert np.array_equal(diagonal, np.array([0.0, 1.0]))
    with pytest.raises(SchemaViolation, match="zero-variance diagonal support"):
        discovery_runner._whitened_squared_residuals(
            np.array([2.0, 2.0]),
            np.array([1.0, 1.0]),
            np.array([0.0, 1.0]),
        )
    covariance = np.array([[1.0, 1.0], [1.0, 1.0]])
    supported = discovery_runner._whitened_squared_residuals(
        np.array([1.0, 1.0]), np.zeros(2), covariance
    )
    assert np.all(np.isfinite(supported))
    with pytest.raises(SchemaViolation, match="zero-variance covariance support"):
        discovery_runner._whitened_squared_residuals(np.array([1.0, -1.0]), np.zeros(2), covariance)


def test_executable_but_incompatible_formula_is_retained_without_execution() -> None:
    scenario, values = _scenario_and_values()
    incompatible = replace(
        _binding("binding.incompatible.v1", "incompatible-law", BindingStatus.EXECUTABLE),
        geometry_support=("spherical1d",),
    )
    result = run_discovery_matrix(
        catalogue=_catalogue(),
        release=_release(),
        scenarios=(scenario,),
        scenario_values={scenario.scenario_id: values},
        truth_formula_by_scenario={scenario.scenario_id: "incompatible-law"},
        bindings=(incompatible,),
        adapters=(AdapterRegistration.create("adapter.incompatible.v1", incompatible),),
        parameter_cells={
            incompatible.binding_id: (
                ParameterCell("scale.one", {"scale_denominator": 1, "scale_numerator": 1}),
            )
        },
        comparisons={
            scenario.scenario_id: (
                ObservableComparison(
                    "prediction.vector.acceleration",
                    "response.vector.acceleration",
                    "response.diagonal-covariance",
                ),
            )
        },
        distinct_gap=0.1,
        ledger_id="gravity.synthetic.runner.incompatible-ledger",
    )
    assert result.attempted_cell_count == 1
    assert result.scored_cell_count == 0
    assert len(result.ledger.entries) == 1
    assert result.cells[0].eligibility == "INCOMPATIBLE_FEATURE_SET"
    assert result.cells[0].whitened_rmse is None
