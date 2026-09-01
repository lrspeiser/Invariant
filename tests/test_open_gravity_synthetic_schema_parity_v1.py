from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator

from sigma_theory_compiler.open_gravity_data_element_ontology_v1 import (
    Availability,
    DataElement,
    DataElementCatalogue,
    DataRole,
    ExperimentRole,
    UncertaintyKind,
)
from sigma_theory_compiler.open_gravity_formula_execution_protocol_v1 import (
    BindingStatus,
    FormulaExecutionBinding,
    ResourceBounds,
)

ROOT = Path(__file__).resolve().parents[1]
HASH = "5" * 64


def schema(name: str) -> dict:
    return json.loads((ROOT / "configs" / name).read_text(encoding="utf-8"))


def catalogue_payload() -> dict:
    element = DataElement(
        element_id="source.scalar.density",
        namespace="source.scalar",
        physical_quantity="density",
        tensor_rank=0,
        si_dimension=(1, -3, 0, 0, 0, 0, 0),
        canonical_unit="kg m^-3",
        frame="source",
        support="voxels",
        axes=("time", "x", "y", "z"),
        component="total",
        derivation_parents=(),
        uncertainty=UncertaintyKind.COVARIANCE,
        availability=Availability.SYNTHETIC_ONLY,
        experiment_roles=(ExperimentRole("galaxy.synthetic.v2", DataRole.FORMULA_INPUT),),
        provenance_sha256=HASH,
    )
    return DataElementCatalogue("gravity.synthetic.elements", "v1.0.0", (element,)).to_dict()


def binding_payload() -> dict:
    return FormulaExecutionBinding(
        binding_id="binding.DPEL01.v1",
        formula_id="DPEL01",
        formula_version="v1.0.0",
        formula_sha256=HASH,
        status=BindingStatus.EXECUTABLE,
        entrypoint=(
            "sigma_theory_compiler.open_gravity_formula_execution_protocol_v1:decide_eligibility"
        ),
        required_features=("source.scalar.density",),
        optional_features=(),
        emitted_features=("prediction.vector.acceleration",),
        domains=("galaxy",),
        geometry_support=("nonspherical3d",),
        time_support=("static",),
        parameter_schema_path="configs/open_gravity_vector_scale_control_parameters_v1.schema.json",
        parameter_schema_sha256=HASH,
        approximation_ceiling="synthetic direction finding only",
        health_gates=("dimension",),
        resource_bounds=ResourceBounds(60, 1_000_000, 100_000),
    ).to_dict()


def test_runtime_catalogue_and_binding_are_schema_valid() -> None:
    ontology = Draft202012Validator(schema("open_gravity_data_element_ontology_v1.schema.json"))
    formula = Draft202012Validator(schema("open_gravity_formula_execution_binding_v1.schema.json"))
    assert list(ontology.iter_errors(catalogue_payload())) == []
    assert list(formula.iter_errors(binding_payload())) == []


def test_schema_rejects_the_previous_executable_and_derivation_forgeries() -> None:
    formula = Draft202012Validator(schema("open_gravity_formula_execution_binding_v1.schema.json"))
    forged_binding = binding_payload()
    forged_binding["entrypoint"] = None
    forged_binding["required_features"] = []
    forged_binding["emitted_features"] = []
    assert list(formula.iter_errors(forged_binding))

    ontology = Draft202012Validator(schema("open_gravity_data_element_ontology_v1.schema.json"))
    forged_catalogue = catalogue_payload()
    forged_catalogue["elements"][0]["derivation_parents"] = ["response.scalar.velocity"]
    forged_catalogue["elements"][0]["derivation_sha256"] = None
    assert list(ontology.iter_errors(forged_catalogue))


def test_discovery_manifest_rejects_empty_release_and_real_confirmation_role() -> None:
    validator = Draft202012Validator(
        schema("open_gravity_synthetic_discovery_manifest_v1.schema.json")
    )
    forged = {
        "schema_version": "open-gravity-synthetic-discovery-manifest-1.0",
        "suite_release": {},
        "experiment_id": "galaxy.synthetic.v2",
        "domain": "galaxy",
        "data_role": "REAL_CONFIRMATION",
        "formula_binding_sha256": HASH,
        "adapter_sha256": None,
        "seeds": [1],
        "countermodel_ids": [],
        "metrics": ["self-injection"],
        "claim_class": "SYNTHETIC_DIRECTIONAL_SIGNAL",
    }
    assert list(validator.iter_errors(forged))
