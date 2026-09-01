from __future__ import annotations

import pytest

from sigma_theory_compiler.open_gravity_data_element_ontology_v1 import (
    Availability,
    DataElement,
    DataElementCatalogue,
    DataRole,
    ExperimentRole,
    UncertaintyKind,
    catalogue_from_elements,
)
from sigma_theory_compiler.sigma_core import SchemaViolation

HASH = "1" * 64


def element(
    element_id: str,
    *,
    parents: tuple[str, ...] = (),
    roles: tuple[ExperimentRole, ...] | None = None,
) -> DataElement:
    namespace = element_id.rsplit(".", 1)[0]
    return DataElement(
        element_id=element_id,
        namespace=namespace,
        physical_quantity="test quantity",
        tensor_rank=0,
        si_dimension=(0, 1, -2, 0, 0, 0, 0),
        canonical_unit="m s^-2",
        frame="barycentric",
        support="sample points",
        axes=("sample",),
        component="total",
        derivation_parents=parents,
        uncertainty=UncertaintyKind.COVARIANCE,
        availability=Availability.SYNTHETIC_ONLY,
        experiment_roles=roles
        or (
            ExperimentRole(
                "galaxy.synthetic.v1",
                DataRole.SOURCE_DERIVED if parents else DataRole.FORMULA_INPUT,
            ),
        ),
        provenance_sha256=HASH,
        derivation_sha256=HASH if parents else None,
    )


def test_round_trip_and_experiment_relative_visibility() -> None:
    item = element(
        "response.rotation.velocity",
        roles=(
            ExperimentRole("galaxy.confirmation.v1", DataRole.SCORING_ONLY_RESPONSE),
            ExperimentRole("galaxy.synthetic.v1", DataRole.FORMULA_INPUT),
        ),
    )
    rebuilt = DataElement.from_dict(item.to_dict())
    assert rebuilt == item
    assert rebuilt.visible_to_formula("galaxy.synthetic.v1")
    assert not rebuilt.visible_to_formula("galaxy.confirmation.v1")


def test_catalogue_hash_is_order_independent_at_construction_boundary() -> None:
    primitive = element("source.scalar.density")
    derived = element("derived.scalar.potential", parents=(primitive.element_id,))
    first = catalogue_from_elements("gravity.synthetic.elements", "v1.0.0", [derived, primitive])
    second = catalogue_from_elements("gravity.synthetic.elements", "v1.0.0", [primitive, derived])
    assert first.content_sha256 == second.content_sha256
    assert DataElementCatalogue.from_dict(first.to_dict()) == first


def test_derivation_cycle_and_truth_leak_fail_closed() -> None:
    a = element("derived.scalar.a", parents=("derived.scalar.b",))
    b = element("derived.scalar.b", parents=("derived.scalar.a",))
    with pytest.raises(SchemaViolation, match="cycle"):
        catalogue_from_elements("gravity.synthetic.elements", "v1.0.0", [a, b])

    truth = element(
        "truth.scalar.injection",
        roles=(ExperimentRole("galaxy.synthetic.v1", DataRole.LATENT_SYNTHETIC_TRUTH),),
    )
    catalogue = catalogue_from_elements("gravity.synthetic.elements", "v1.0.0", [truth])
    assert not catalogue.visible_features("galaxy.synthetic.v1")


def test_response_derived_feature_cannot_be_relabelled_as_formula_input() -> None:
    response = element(
        "response.scalar.velocity",
        roles=(ExperimentRole("galaxy.synthetic.v1", DataRole.SCORING_ONLY_RESPONSE),),
    )
    derived = element(
        "derived.scalar.velocity-summary",
        parents=(response.element_id,),
        roles=(ExperimentRole("galaxy.synthetic.v1", DataRole.SOURCE_DERIVED),),
    )
    with pytest.raises(SchemaViolation, match="response or truth"):
        catalogue_from_elements("gravity.synthetic.elements", "v1.0.0", [response, derived])


def test_axis_order_is_semantic_not_alphabetical() -> None:
    item = DataElement(
        element_id="history.tensor.field",
        namespace="history.tensor",
        physical_quantity="ordered field",
        tensor_rank=2,
        si_dimension=(0, 0, 0, 0, 0, 0, 0),
        canonical_unit="1",
        frame="source",
        support="grid",
        axes=("time", "x", "y", "z"),
        component="total",
        derivation_parents=(),
        uncertainty=UncertaintyKind.NONE,
        availability=Availability.SYNTHETIC_ONLY,
        experiment_roles=(ExperimentRole("galaxy.synthetic.v1", DataRole.FORMULA_INPUT),),
        provenance_sha256=HASH,
    )
    assert DataElement.from_dict(item.to_dict()).axes == ("time", "x", "y", "z")
