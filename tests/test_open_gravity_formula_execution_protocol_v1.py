from __future__ import annotations

import pytest

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
    decide_eligibility,
    validate_binding_catalogue,
)
from sigma_theory_compiler.sigma_core import SchemaViolation

HASH = "2" * 64


def element(element_id: str, role: DataRole) -> DataElement:
    return DataElement(
        element_id=element_id,
        namespace=element_id.rsplit(".", 1)[0],
        physical_quantity="test",
        tensor_rank=0,
        si_dimension=(0, 0, 0, 0, 0, 0, 0),
        canonical_unit="1",
        frame="local",
        support="samples",
        axes=(),
        component="total",
        derivation_parents=(),
        uncertainty=UncertaintyKind.NONE,
        availability=Availability.SYNTHETIC_ONLY,
        experiment_roles=(ExperimentRole("galaxy.synthetic.v1", role),),
        provenance_sha256=HASH,
        derivation_sha256=None,
    )


def binding(*required: str, status: BindingStatus = BindingStatus.EXECUTABLE):
    return FormulaExecutionBinding(
        binding_id="binding.DPEL01.v1",
        formula_id="DPEL01",
        formula_version="v1.0.0",
        formula_sha256=HASH,
        status=status,
        entrypoint=(
            "sigma_theory_compiler.open_gravity_formula_execution_protocol_v1:decide_eligibility"
            if status is BindingStatus.EXECUTABLE
            else None
        ),
        required_features=tuple(sorted(required)),
        optional_features=(),
        emitted_features=("prediction.vector.acceleration",) if required else (),
        domains=("galaxy",),
        geometry_support=("nonspherical3d",),
        time_support=("static",),
        parameter_schema_path="configs/open_gravity_vector_scale_control_parameters_v1.schema.json",
        parameter_schema_sha256=HASH,
        approximation_ceiling="synthetic direction finding only",
        health_gates=("dimension",),
        resource_bounds=ResourceBounds(60, 1_000_000, 100_000),
    )


def test_eligibility_uses_experiment_relative_roles() -> None:
    source = element("source.scalar.density", DataRole.FORMULA_INPUT)
    response = element("response.scalar.velocity", DataRole.SCORING_ONLY_RESPONSE)
    output = element("prediction.vector.acceleration", DataRole.DERIVED)
    catalogue = catalogue_from_elements(
        "gravity.synthetic.elements", "v1.0.0", [source, response, output]
    )
    good = binding(source.element_id)
    forbidden = binding(response.element_id)
    assert decide_eligibility(good, catalogue, "galaxy.synthetic.v1", "galaxy").status is (
        EligibilityStatus.ELIGIBLE
    )
    decision = decide_eligibility(forbidden, catalogue, "galaxy.synthetic.v1", "galaxy")
    assert decision.status is EligibilityStatus.FORBIDDEN_RESPONSE_ACCESS
    assert decision.forbidden_features == (response.element_id,)


def test_nonexecutables_are_classified_without_pseudo_scores() -> None:
    output = element("prediction.vector.acceleration", DataRole.DERIVED)
    catalogue = catalogue_from_elements("gravity.synthetic.elements", "v1.0.0", [output])
    theory = binding(status=BindingStatus.THEORY_ONLY)
    assert decide_eligibility(theory, catalogue, "galaxy.synthetic.v1", "galaxy").status is (
        EligibilityStatus.THEORY_ONLY
    )


def test_unknown_executable_features_fail_catalogue_validation() -> None:
    output = element("prediction.vector.acceleration", DataRole.DERIVED)
    catalogue = catalogue_from_elements("gravity.synthetic.elements", "v1.0.0", [output])
    candidate = binding("source.scalar.missing")
    with pytest.raises(SchemaViolation, match="unknown"):
        validate_binding_catalogue((candidate,), catalogue)
