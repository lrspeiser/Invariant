"""Typed, experiment-relative data elements for Open-Gravity synthetic discovery.

This module is deliberately domain-neutral.  It describes what a value means and
who may see it; it does not contain scientific response values or formula scores.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from sigma_theory_compiler.sigma_core import SchemaViolation, canonical_sha256

SCHEMA_VERSION = "open-gravity-data-element-ontology-1.0"
_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SI_BASES = ("kg", "m", "s", "A", "K", "mol", "cd")


class DataRole(str, Enum):
    PRIMITIVE_CAUSE = "PRIMITIVE_CAUSE"
    LATENT_SYNTHETIC_TRUTH = "LATENT_SYNTHETIC_TRUTH"
    FORMULA_INPUT = "FORMULA_INPUT"
    SCORING_ONLY_RESPONSE = "SCORING_ONLY_RESPONSE"
    INSTRUMENT = "INSTRUMENT"
    SELECTION_MASK = "SELECTION_MASK"
    SOURCE_DERIVED = "SOURCE_DERIVED"
    DERIVED = "DERIVED"


class Availability(str, Enum):
    ANALYTIC = "ANALYTIC"
    PUBLIC_SOURCE = "PUBLIC_SOURCE"
    PUBLIC_RESPONSE = "PUBLIC_RESPONSE"
    SYNTHETIC_ONLY = "SYNTHETIC_ONLY"
    SOURCE_BLOCKED = "SOURCE_BLOCKED"
    NOT_AVAILABLE = "NOT_AVAILABLE"


class UncertaintyKind(str, Enum):
    NONE = "NONE"
    PER_VALUE = "PER_VALUE"
    COVARIANCE = "COVARIANCE"
    POSTERIOR_DRAWS = "POSTERIOR_DRAWS"
    HIERARCHICAL = "HIERARCHICAL"
    CENSORING = "CENSORING"
    MIXED = "MIXED"


def _identifier(value: str, label: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise SchemaViolation(f"{label} is not a canonical identifier")
    return value


def _nonempty(value: str, label: str) -> str:
    if not isinstance(value, str) or value != value.strip() or not value:
        raise SchemaViolation(f"{label} must be nonempty and stripped")
    return value


def _hash(value: str, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise SchemaViolation(f"{label} must be a lowercase SHA-256")
    return value


@dataclass(frozen=True, slots=True)
class ExperimentRole:
    experiment_id: str
    role: DataRole

    def __post_init__(self) -> None:
        _identifier(self.experiment_id, "experiment_id")

    def to_dict(self) -> dict[str, str]:
        return {"experiment_id": self.experiment_id, "role": self.role.value}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ExperimentRole:
        if set(value) != {"experiment_id", "role"}:
            raise SchemaViolation("experiment role keys changed")
        return cls(str(value["experiment_id"]), DataRole(str(value["role"])))


@dataclass(frozen=True, slots=True)
class DataElement:
    element_id: str
    namespace: str
    physical_quantity: str
    tensor_rank: int
    si_dimension: tuple[int, int, int, int, int, int, int]
    canonical_unit: str
    frame: str
    support: str
    axes: tuple[str, ...]
    component: str
    derivation_parents: tuple[str, ...]
    uncertainty: UncertaintyKind
    availability: Availability
    experiment_roles: tuple[ExperimentRole, ...]
    provenance_sha256: str
    derivation_sha256: str | None = None
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise SchemaViolation("data-element schema changed")
        _identifier(self.element_id, "element_id")
        _identifier(self.namespace, "namespace")
        if not self.element_id.startswith(f"{self.namespace}."):
            raise SchemaViolation("element_id must be inside its namespace")
        _nonempty(self.physical_quantity, "physical_quantity")
        if type(self.tensor_rank) is not int or not 0 <= self.tensor_rank <= 4:
            raise SchemaViolation("tensor_rank must be an integer from zero through four")
        if len(self.si_dimension) != len(_SI_BASES) or any(
            type(value) is not int or not -12 <= value <= 12 for value in self.si_dimension
        ):
            raise SchemaViolation("si_dimension must contain seven bounded integer exponents")
        for label, value in (
            ("canonical_unit", self.canonical_unit),
            ("frame", self.frame),
            ("support", self.support),
            ("component", self.component),
        ):
            _nonempty(value, label)
        if len(set(self.axes)) != len(self.axes):
            raise SchemaViolation("axes must be unique and preserve physical ordering")
        for axis in self.axes:
            _identifier(axis, "axis")
        if self.derivation_parents != tuple(sorted(set(self.derivation_parents))):
            raise SchemaViolation("derivation parents must be unique and sorted")
        for parent in self.derivation_parents:
            _identifier(parent, "derivation parent")
        if self.element_id in self.derivation_parents:
            raise SchemaViolation("data element cannot derive from itself")
        if not self.experiment_roles:
            raise SchemaViolation("at least one experiment-relative role is required")
        experiments = tuple(role.experiment_id for role in self.experiment_roles)
        if experiments != tuple(sorted(set(experiments))):
            raise SchemaViolation("experiment roles must be unique and sorted")
        _hash(self.provenance_sha256, "provenance_sha256")
        if self.derivation_sha256 is not None:
            _hash(self.derivation_sha256, "derivation_sha256")
        if bool(self.derivation_parents) != (self.derivation_sha256 is not None):
            raise SchemaViolation(
                "derived elements require exactly one derivation implementation hash"
            )
        if self.derivation_parents and any(
            row.role
            in {
                DataRole.PRIMITIVE_CAUSE,
                DataRole.FORMULA_INPUT,
                DataRole.INSTRUMENT,
                DataRole.SELECTION_MASK,
            }
            for row in self.experiment_roles
        ):
            raise SchemaViolation("formula-visible derived elements must use SOURCE_DERIVED")

    def role_for(self, experiment_id: str) -> DataRole | None:
        _identifier(experiment_id, "experiment_id")
        return next(
            (row.role for row in self.experiment_roles if row.experiment_id == experiment_id),
            None,
        )

    def visible_to_formula(self, experiment_id: str) -> bool:
        return self.role_for(experiment_id) in {
            DataRole.PRIMITIVE_CAUSE,
            DataRole.FORMULA_INPUT,
            DataRole.INSTRUMENT,
            DataRole.SELECTION_MASK,
            DataRole.SOURCE_DERIVED,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "element_id": self.element_id,
            "namespace": self.namespace,
            "physical_quantity": self.physical_quantity,
            "tensor_rank": self.tensor_rank,
            "si_dimension": dict(zip(_SI_BASES, self.si_dimension, strict=True)),
            "canonical_unit": self.canonical_unit,
            "frame": self.frame,
            "support": self.support,
            "axes": list(self.axes),
            "component": self.component,
            "derivation_parents": list(self.derivation_parents),
            "uncertainty": self.uncertainty.value,
            "availability": self.availability.value,
            "experiment_roles": [row.to_dict() for row in self.experiment_roles],
            "provenance_sha256": self.provenance_sha256,
            "derivation_sha256": self.derivation_sha256,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DataElement:
        expected = {
            "schema_version",
            "element_id",
            "namespace",
            "physical_quantity",
            "tensor_rank",
            "si_dimension",
            "canonical_unit",
            "frame",
            "support",
            "axes",
            "component",
            "derivation_parents",
            "uncertainty",
            "availability",
            "experiment_roles",
            "provenance_sha256",
            "derivation_sha256",
        }
        if set(value) != expected:
            raise SchemaViolation("data-element keys changed")
        dimension = value["si_dimension"]
        if not isinstance(dimension, Mapping) or tuple(dimension) != _SI_BASES:
            raise SchemaViolation("SI dimension keys or ordering changed")
        if (
            not isinstance(value["axes"], list)
            or not isinstance(value["derivation_parents"], list)
            or not isinstance(value["experiment_roles"], list)
        ):
            raise SchemaViolation("data-element arrays changed")
        return cls(
            element_id=str(value["element_id"]),
            namespace=str(value["namespace"]),
            physical_quantity=str(value["physical_quantity"]),
            tensor_rank=value["tensor_rank"],
            si_dimension=tuple(dimension[name] for name in _SI_BASES),  # type: ignore[arg-type]
            canonical_unit=str(value["canonical_unit"]),
            frame=str(value["frame"]),
            support=str(value["support"]),
            axes=tuple(str(item) for item in value["axes"]),
            component=str(value["component"]),
            derivation_parents=tuple(str(item) for item in value["derivation_parents"]),
            uncertainty=UncertaintyKind(str(value["uncertainty"])),
            availability=Availability(str(value["availability"])),
            experiment_roles=tuple(
                ExperimentRole.from_dict(item) for item in value["experiment_roles"]
            ),
            provenance_sha256=str(value["provenance_sha256"]),
            derivation_sha256=value["derivation_sha256"],
            schema_version=str(value["schema_version"]),
        )


@dataclass(frozen=True, slots=True)
class DataElementCatalogue:
    catalogue_id: str
    version: str
    elements: tuple[DataElement, ...]

    def __post_init__(self) -> None:
        _identifier(self.catalogue_id, "catalogue_id")
        _nonempty(self.version, "catalogue version")
        identifiers = tuple(item.element_id for item in self.elements)
        if identifiers != tuple(sorted(set(identifiers))):
            raise SchemaViolation("catalogue elements must be unique and sorted")
        known = set(identifiers)
        for element in self.elements:
            missing = set(element.derivation_parents) - known
            if missing:
                raise SchemaViolation(f"unknown derivation parents: {sorted(missing)}")
        self._check_acyclic()
        self._check_information_flow()

    def _check_acyclic(self) -> None:
        parents = {item.element_id: item.derivation_parents for item in self.elements}
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(element_id: str) -> None:
            if element_id in visiting:
                raise SchemaViolation("data-element derivation cycle")
            if element_id in visited:
                return
            visiting.add(element_id)
            for parent in parents[element_id]:
                visit(parent)
            visiting.remove(element_id)
            visited.add(element_id)

        for identifier in parents:
            visit(identifier)

    def _check_information_flow(self) -> None:
        by_id = self.by_id()

        def clean(element_id: str, experiment_id: str) -> bool:
            element = by_id[element_id]
            role = element.role_for(experiment_id)
            if role not in {
                DataRole.PRIMITIVE_CAUSE,
                DataRole.FORMULA_INPUT,
                DataRole.INSTRUMENT,
                DataRole.SELECTION_MASK,
                DataRole.SOURCE_DERIVED,
            }:
                return False
            if role is not DataRole.SOURCE_DERIVED:
                return True
            return bool(element.derivation_parents) and all(
                clean(parent, experiment_id) for parent in element.derivation_parents
            )

        for element in self.elements:
            for row in element.experiment_roles:
                if row.role is DataRole.SOURCE_DERIVED and not clean(
                    element.element_id, row.experiment_id
                ):
                    raise SchemaViolation(
                        "source-derived element crosses a response or truth boundary"
                    )

    @property
    def content_sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    def by_id(self) -> dict[str, DataElement]:
        return {item.element_id: item for item in self.elements}

    def visible_features(self, experiment_id: str) -> frozenset[str]:
        by_id = self.by_id()

        def visible(element: DataElement) -> bool:
            role = element.role_for(experiment_id)
            if role not in {
                DataRole.PRIMITIVE_CAUSE,
                DataRole.FORMULA_INPUT,
                DataRole.INSTRUMENT,
                DataRole.SELECTION_MASK,
                DataRole.SOURCE_DERIVED,
            }:
                return False
            if role is not DataRole.SOURCE_DERIVED:
                return True
            return bool(element.derivation_parents) and all(
                visible(by_id[parent]) for parent in element.derivation_parents
            )

        return frozenset(item.element_id for item in self.elements if visible(item))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "open-gravity-data-element-catalogue-1.0",
            "catalogue_id": self.catalogue_id,
            "version": self.version,
            "elements": [item.to_dict() for item in self.elements],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DataElementCatalogue:
        if set(value) != {"schema_version", "catalogue_id", "version", "elements"}:
            raise SchemaViolation("catalogue keys changed")
        if value["schema_version"] != "open-gravity-data-element-catalogue-1.0":
            raise SchemaViolation("catalogue schema changed")
        if not isinstance(value["elements"], list):
            raise SchemaViolation("catalogue elements must be an array")
        return cls(
            str(value["catalogue_id"]),
            str(value["version"]),
            tuple(DataElement.from_dict(item) for item in value["elements"]),
        )


def catalogue_from_elements(
    catalogue_id: str, version: str, elements: Sequence[DataElement]
) -> DataElementCatalogue:
    return DataElementCatalogue(
        catalogue_id, version, tuple(sorted(elements, key=lambda x: x.element_id))
    )


__all__ = [
    "SCHEMA_VERSION",
    "Availability",
    "DataElement",
    "DataElementCatalogue",
    "DataRole",
    "ExperimentRole",
    "UncertaintyKind",
    "catalogue_from_elements",
]
