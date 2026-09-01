from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from jsonschema import Draft202012Validator

import sigma_theory_compiler.open_gravity_formula_adapter_registry_v1 as adapter_registry
from sigma_theory_compiler.open_gravity_data_element_ontology_v1 import (
    Availability,
    DataElement,
    DataRole,
    ExperimentRole,
    UncertaintyKind,
    catalogue_from_elements,
)
from sigma_theory_compiler.open_gravity_formula_execution_protocol_v1 import (
    BindingStatus,
    EligibilityStatus,
    FormulaExecutionBinding,
    ResourceBounds,
)
from sigma_theory_compiler.open_gravity_observation_operators_v1 import SeedLineage
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
from sigma_theory_compiler.sigma_core import SchemaViolation

HASH = "6" * 64
EXPERIMENT = "galaxy.synthetic.v2"
ROOT = Path(__file__).resolve().parents[1]


def element(element_id: str, role: DataRole, rank: int, unit: str) -> DataElement:
    return DataElement(
        element_id=element_id,
        namespace=element_id.rsplit(".", 1)[0],
        physical_quantity="test quantity",
        tensor_rank=rank,
        si_dimension=(0, 1, -2, 0, 0, 0, 0),
        canonical_unit=unit,
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


def catalogue():
    return catalogue_from_elements(
        "gravity.synthetic.elements",
        "v1.0.0",
        [
            element("prediction.vector.acceleration", DataRole.DERIVED, 1, "m s^-2"),
            element("response.scalar.velocity", DataRole.SCORING_ONLY_RESPONSE, 0, "m s^-1"),
            element("source.vector.acceleration", DataRole.FORMULA_INPUT, 1, "m s^-2"),
            element("truth.scalar.injection-id", DataRole.LATENT_SYNTHETIC_TRUTH, 0, "1"),
        ],
    )


def scenario() -> tuple[ScenarioDescriptor, np.ndarray]:
    source = np.arange(6, dtype=np.float64).reshape(2, 3)
    response = np.array([1.0, 2.0], dtype=np.float64)
    truth = np.array([1, 1], dtype=np.int64)
    packet = ScenarioDescriptor(
        scenario_id="galaxy.disk.fixture.v1",
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
                "features/source-acceleration.npy",
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
                "response.scalar.velocity",
                "responses/velocity.npy",
                array_sha256(response),
                "float64",
                (2,),
                ("sample",),
                "m s^-1",
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
                "predictions/vector-acceleration.npy",
                "float64",
                (2, 3),
                ("sample", "component"),
                "m s^-2",
                "source",
            ),
        ),
        uncertainties=(
            UncertaintyRef(
                "velocity.covariance",
                "response.scalar.velocity",
                "covariance",
                "uncertainty/velocity-covariance.npy",
                HASH,
            ),
        ),
        anchors=(AnchorBinding("galaxy.source.v1", "anchors/source-receipt.json", HASH),),
        seed_lineage=SeedLineage(
            7,
            "galaxy.disk.fixture.v1",
            "synthetic.object.001",
            "vector-scale.control",
            0,
            0,
        ),
    )
    return packet, source


def binding() -> FormulaExecutionBinding:
    return FormulaExecutionBinding(
        binding_id="binding.vector-scale.v1",
        formula_id="vector-scale-control",
        formula_version="v1.0.0",
        formula_sha256=HASH,
        status=BindingStatus.EXECUTABLE,
        entrypoint=(
            "sigma_theory_compiler.open_gravity_formula_adapter_registry_v1:vector_scale_control"
        ),
        required_features=("source.vector.acceleration",),
        optional_features=(),
        emitted_features=("prediction.vector.acceleration",),
        domains=("galaxy",),
        geometry_support=("nonspherical3d",),
        time_support=("static",),
        parameter_schema_path="configs/open_gravity_vector_scale_control_parameters_v1.schema.json",
        parameter_schema_sha256="7f14379df933fa469a78189b69371a84e9ca295080f86b48a1814cef5e902e9f",
        approximation_ceiling="known-answer synthetic control",
        health_gates=("dimension",),
        resource_bounds=ResourceBounds(10, 1_000_000, 100_000),
    )


def test_scenario_visibility_and_geometry_time_eligibility() -> None:
    packet, _ = scenario()
    validate_scenario_catalogue(packet, catalogue())
    assert decide_scenario_eligibility(binding(), catalogue(), packet).status is (
        EligibilityStatus.ELIGIBLE
    )
    incompatible = FormulaExecutionBinding.from_dict(
        {**binding().to_dict(), "geometry_support": ["spherical1d"]}
    )
    decision = decide_scenario_eligibility(incompatible, catalogue(), packet)
    assert decision.status is EligibilityStatus.INCOMPATIBLE_FEATURE_SET
    assert decision.missing_features == ("geometry.nonspherical3d",)


def test_exact_feature_projection_and_deterministic_execution() -> None:
    packet, source = scenario()
    result = execute_binding_in_process(
        binding(),
        catalogue(),
        packet,
        {"source.vector.acceleration": source},
        {"scale_denominator": 2, "scale_numerator": 3},
    )
    assert result.deterministic_replay
    assert set(result.output_predictions) == {"prediction.vector.acceleration"}
    assert np.array_equal(result.output_values["prediction.vector.acceleration"], source * 1.5)
    assert not result.output_values["prediction.vector.acceleration"].flags.writeable

    with pytest.raises(SchemaViolation, match="exact binding projection"):
        execute_binding_in_process(
            binding(),
            catalogue(),
            packet,
            {"source.vector.acceleration": source, "response.scalar.velocity": [1.0, 2.0]},
            {"scale_denominator": 2, "scale_numerator": 3},
        )


def test_value_hash_unit_frame_and_seed_identity_fail_closed() -> None:
    packet, source = scenario()
    with pytest.raises(SchemaViolation, match="content"):
        execute_binding_in_process(
            binding(),
            catalogue(),
            packet,
            {"source.vector.acceleration": source + 1.0},
            {"scale_denominator": 2, "scale_numerator": 3},
        )
    broken = packet.formula_features[0].to_dict()
    broken["unit"] = "km s^-2"
    forged = FeatureValueRef(
        broken["element_id"],
        broken["artifact_path"],
        broken["value_sha256"],
        broken["dtype"],
        tuple(broken["shape"]),
        tuple(broken["axes"]),
        broken["unit"],
        broken["frame"],
    )
    with pytest.raises(SchemaViolation, match="unit, frame, or axes"):
        validate_scenario_catalogue(
            replace(packet, formula_features=(forged,)),
            catalogue(),
        )


def test_all_visibility_partitions_and_axis_order_match_ontology() -> None:
    packet, _ = scenario()
    response = packet.scoring_responses[0]
    wrong_response = FeatureValueRef(
        response.element_id,
        response.artifact_path,
        response.value_sha256,
        response.dtype,
        response.shape,
        response.axes,
        "km s^-1",
        "observer",
    )
    with pytest.raises(SchemaViolation, match="scoring response unit, frame, or axes"):
        validate_scenario_catalogue(
            replace(packet, scoring_responses=(wrong_response,)), catalogue()
        )

    source = packet.formula_features[0]
    transposed = FeatureValueRef(
        source.element_id,
        source.artifact_path,
        source.value_sha256,
        source.dtype,
        (3, 2),
        ("component", "sample"),
        source.unit,
        source.frame,
    )
    with pytest.raises(SchemaViolation, match="formula feature unit, frame, or axes"):
        validate_scenario_catalogue(replace(packet, formula_features=(transposed,)), catalogue())


def test_parameter_schema_and_typed_prediction_fail_closed(monkeypatch) -> None:
    packet, source = scenario()
    with pytest.raises(SchemaViolation, match="parameters violate registered schema"):
        execute_binding_in_process(
            binding(),
            catalogue(),
            packet,
            {"source.vector.acceleration": source},
            {"undeclared_parameter": 99},
        )
    monkeypatch.setattr(
        adapter_registry,
        "vector_scale_control",
        lambda _features, _parameters: {
            "prediction.vector.acceleration": np.array([1.0], dtype=np.float64)
        },
    )
    with pytest.raises(SchemaViolation, match="output dtype or shape"):
        execute_binding_in_process(
            binding(),
            catalogue(),
            packet,
            {"source.vector.acceleration": source},
            {"scale_denominator": 2, "scale_numerator": 3},
        )


def test_coordinate_reference_and_whole_packet_values_fail_closed() -> None:
    packet, source = scenario()
    component = AxisSpec(
        "component",
        3,
        "source.vector.acceleration",
        packet.formula_features[0].value_sha256,
    )
    with pytest.raises(SchemaViolation, match="formula-visible scalar"):
        validate_scenario_catalogue(replace(packet, axes=(component, packet.axes[1])), catalogue())

    response = np.array([1.0, 2.0], dtype=np.float64)
    truth = np.array([1, 1], dtype=np.int64)
    covariance = np.eye(2, dtype=np.float64)
    uncertainty = UncertaintyRef(
        "velocity.covariance",
        "response.scalar.velocity",
        "covariance",
        "uncertainty/velocity-covariance.npy",
        array_sha256(covariance),
    )
    typed = validate_scenario_values(
        replace(packet, uncertainties=(uncertainty,)),
        formula_values={"source.vector.acceleration": source},
        response_values={"response.scalar.velocity": response},
        truth_values={"truth.scalar.injection-id": truth},
        uncertainty_values={"velocity.covariance": covariance},
    )
    assert not typed.hidden_truth["truth.scalar.injection-id"].flags.writeable
    with pytest.raises(SchemaViolation, match="loaded feature content"):
        validate_scenario_values(
            replace(packet, uncertainties=(uncertainty,)),
            formula_values={"source.vector.acceleration": source},
            response_values={"response.scalar.velocity": response},
            truth_values={"truth.scalar.injection-id": np.array([0, 0], dtype=np.int64)},
            uncertainty_values={"velocity.covariance": covariance},
        )


def test_scenario_runtime_payload_matches_json_schema() -> None:
    packet, _ = scenario()
    schema = json.loads(
        (ROOT / "configs/open_gravity_synthetic_scenario_packet_v1.schema.json").read_text(
            encoding="utf-8"
        )
    )
    assert list(Draft202012Validator(schema).iter_errors(packet.to_dict())) == []
