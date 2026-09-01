"""Common capability and eligibility protocol for Open-Gravity formula adapters."""

from __future__ import annotations

import importlib
import inspect
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import PurePosixPath
from typing import Any

from sigma_theory_compiler.open_gravity_data_element_ontology_v1 import (
    DataElementCatalogue,
)
from sigma_theory_compiler.sigma_core import SchemaViolation, canonical_sha256

ABI_VERSION = "open-gravity-formula-execution-1.0"
_IDENTIFIER = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,159}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ENTRYPOINT = re.compile(r"^sigma_theory_compiler\.[a-z][a-z0-9_]*:[A-Za-z_][A-Za-z0-9_]*$")


class BindingStatus(str, Enum):
    EXECUTABLE = "EXECUTABLE"
    UNADAPTED = "UNADAPTED"
    THEORY_ONLY = "THEORY_ONLY"
    SOURCE_BLOCKED = "SOURCE_BLOCKED"
    QUARANTINED = "QUARANTINED"


class EligibilityStatus(str, Enum):
    ELIGIBLE = "ELIGIBLE"
    UNADAPTED = "UNADAPTED"
    THEORY_ONLY = "THEORY_ONLY"
    SOURCE_BLOCKED = "SOURCE_BLOCKED"
    QUARANTINED = "QUARANTINED"
    INCOMPATIBLE_FEATURE_SET = "INCOMPATIBLE_FEATURE_SET"
    FORBIDDEN_RESPONSE_ACCESS = "FORBIDDEN_RESPONSE_ACCESS"
    OUT_OF_REGISTERED_DOMAIN = "OUT_OF_REGISTERED_DOMAIN"


def _identifier(value: str, label: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise SchemaViolation(f"{label} is not a canonical identifier")
    return value


def _hash(value: str, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise SchemaViolation(f"{label} must be a lowercase SHA-256")
    return value


def _sorted_unique(values: Sequence[str], label: str) -> tuple[str, ...]:
    result = tuple(values)
    if result != tuple(sorted(set(result))):
        raise SchemaViolation(f"{label} must be unique and sorted")
    return result


def _relative_path(value: str, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise SchemaViolation(f"{label} must be a canonical relative POSIX path")
    parsed = PurePosixPath(value)
    if (
        parsed.is_absolute()
        or parsed.as_posix() != value
        or any(part in {"", ".", ".."} for part in parsed.parts)
    ):
        raise SchemaViolation(f"{label} escaped the repository")
    return value


@dataclass(frozen=True, slots=True)
class ResourceBounds:
    max_wall_seconds: int
    max_memory_bytes: int
    max_output_bytes: int

    def __post_init__(self) -> None:
        for label, value in (
            ("max_wall_seconds", self.max_wall_seconds),
            ("max_memory_bytes", self.max_memory_bytes),
            ("max_output_bytes", self.max_output_bytes),
        ):
            if type(value) is not int or value <= 0:
                raise SchemaViolation(f"{label} must be a positive integer")

    def to_dict(self) -> dict[str, int]:
        return {
            "max_wall_seconds": self.max_wall_seconds,
            "max_memory_bytes": self.max_memory_bytes,
            "max_output_bytes": self.max_output_bytes,
        }


@dataclass(frozen=True, slots=True)
class FormulaExecutionBinding:
    binding_id: str
    formula_id: str
    formula_version: str
    formula_sha256: str
    status: BindingStatus
    entrypoint: str | None
    required_features: tuple[str, ...]
    optional_features: tuple[str, ...]
    emitted_features: tuple[str, ...]
    domains: tuple[str, ...]
    geometry_support: tuple[str, ...]
    time_support: tuple[str, ...]
    parameter_schema_path: str
    parameter_schema_sha256: str
    approximation_ceiling: str
    health_gates: tuple[str, ...]
    resource_bounds: ResourceBounds
    abi_version: str = ABI_VERSION

    def __post_init__(self) -> None:
        for label, value in (
            ("binding_id", self.binding_id),
            ("formula_id", self.formula_id),
            ("formula_version", self.formula_version),
        ):
            _identifier(value, label)
        _hash(self.formula_sha256, "formula_sha256")
        _relative_path(self.parameter_schema_path, "parameter_schema_path")
        _hash(self.parameter_schema_sha256, "parameter_schema_sha256")
        if self.abi_version != ABI_VERSION:
            raise SchemaViolation("formula execution ABI changed")
        if self.status is BindingStatus.EXECUTABLE:
            if self.entrypoint is None or _ENTRYPOINT.fullmatch(self.entrypoint) is None:
                raise SchemaViolation("executable binding requires a package-local entrypoint")
            if not self.required_features or not self.emitted_features:
                raise SchemaViolation("executable binding requires input and output features")
        elif self.entrypoint is not None:
            raise SchemaViolation("non-executable binding cannot expose an entrypoint")
        required = _sorted_unique(self.required_features, "required features")
        optional = _sorted_unique(self.optional_features, "optional features")
        emitted = _sorted_unique(self.emitted_features, "emitted features")
        if set(required) & set(optional):
            raise SchemaViolation("required and optional features overlap")
        for feature in (*required, *optional, *emitted):
            _identifier(feature, "feature ID")
        for label, values in (
            ("domains", self.domains),
            ("geometry support", self.geometry_support),
            ("time support", self.time_support),
            ("health gates", self.health_gates),
        ):
            _sorted_unique(values, label)
            if not values:
                raise SchemaViolation(f"{label} cannot be empty")
            for value in values:
                _identifier(value, label)
        if (
            not isinstance(self.approximation_ceiling, str)
            or not self.approximation_ceiling.strip()
        ):
            raise SchemaViolation("approximation ceiling is required")

    @property
    def content_sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "abi_version": self.abi_version,
            "binding_id": self.binding_id,
            "formula_id": self.formula_id,
            "formula_version": self.formula_version,
            "formula_sha256": self.formula_sha256,
            "status": self.status.value,
            "entrypoint": self.entrypoint,
            "required_features": list(self.required_features),
            "optional_features": list(self.optional_features),
            "emitted_features": list(self.emitted_features),
            "domains": list(self.domains),
            "geometry_support": list(self.geometry_support),
            "time_support": list(self.time_support),
            "parameter_schema_path": self.parameter_schema_path,
            "parameter_schema_sha256": self.parameter_schema_sha256,
            "approximation_ceiling": self.approximation_ceiling,
            "health_gates": list(self.health_gates),
            "resource_bounds": self.resource_bounds.to_dict(),
        }

    def resolve(self) -> Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]]:
        if self.status is not BindingStatus.EXECUTABLE or self.entrypoint is None:
            raise SchemaViolation("formula binding is not executable")
        module_name, callable_name = self.entrypoint.split(":", 1)
        candidate = getattr(importlib.import_module(module_name), callable_name)
        if not callable(candidate):
            raise SchemaViolation("formula entrypoint is not callable")
        signature = inspect.signature(candidate)
        if len(signature.parameters) != 2:
            raise SchemaViolation("formula entrypoint must take feature and parameter mappings")
        return candidate

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> FormulaExecutionBinding:
        expected = {
            "abi_version",
            "binding_id",
            "formula_id",
            "formula_version",
            "formula_sha256",
            "status",
            "entrypoint",
            "required_features",
            "optional_features",
            "emitted_features",
            "domains",
            "geometry_support",
            "time_support",
            "parameter_schema_path",
            "parameter_schema_sha256",
            "approximation_ceiling",
            "health_gates",
            "resource_bounds",
        }
        if set(value) != expected:
            raise SchemaViolation("formula binding keys changed")
        arrays = (
            "required_features",
            "optional_features",
            "emitted_features",
            "domains",
            "geometry_support",
            "time_support",
            "health_gates",
        )
        if any(not isinstance(value[name], list) for name in arrays):
            raise SchemaViolation("formula binding arrays changed")
        bounds = value["resource_bounds"]
        if not isinstance(bounds, Mapping) or set(bounds) != {
            "max_wall_seconds",
            "max_memory_bytes",
            "max_output_bytes",
        }:
            raise SchemaViolation("resource bounds changed")
        return cls(
            binding_id=str(value["binding_id"]),
            formula_id=str(value["formula_id"]),
            formula_version=str(value["formula_version"]),
            formula_sha256=str(value["formula_sha256"]),
            status=BindingStatus(str(value["status"])),
            entrypoint=value["entrypoint"],
            required_features=tuple(str(item) for item in value["required_features"]),
            optional_features=tuple(str(item) for item in value["optional_features"]),
            emitted_features=tuple(str(item) for item in value["emitted_features"]),
            domains=tuple(str(item) for item in value["domains"]),
            geometry_support=tuple(str(item) for item in value["geometry_support"]),
            time_support=tuple(str(item) for item in value["time_support"]),
            parameter_schema_path=str(value["parameter_schema_path"]),
            parameter_schema_sha256=str(value["parameter_schema_sha256"]),
            approximation_ceiling=str(value["approximation_ceiling"]),
            health_gates=tuple(str(item) for item in value["health_gates"]),
            resource_bounds=ResourceBounds(
                bounds["max_wall_seconds"],
                bounds["max_memory_bytes"],
                bounds["max_output_bytes"],
            ),
            abi_version=str(value["abi_version"]),
        )


@dataclass(frozen=True, slots=True)
class EligibilityDecision:
    status: EligibilityStatus
    missing_features: tuple[str, ...]
    forbidden_features: tuple[str, ...]


def decide_eligibility(
    binding: FormulaExecutionBinding,
    catalogue: DataElementCatalogue,
    experiment_id: str,
    domain: str,
) -> EligibilityDecision:
    status_map = {
        BindingStatus.UNADAPTED: EligibilityStatus.UNADAPTED,
        BindingStatus.THEORY_ONLY: EligibilityStatus.THEORY_ONLY,
        BindingStatus.SOURCE_BLOCKED: EligibilityStatus.SOURCE_BLOCKED,
        BindingStatus.QUARANTINED: EligibilityStatus.QUARANTINED,
    }
    if binding.status in status_map:
        return EligibilityDecision(status_map[binding.status], (), ())
    if domain not in binding.domains:
        return EligibilityDecision(EligibilityStatus.OUT_OF_REGISTERED_DOMAIN, (), ())
    known = catalogue.by_id()
    unknown = sorted(set(binding.required_features) - set(known))
    if unknown:
        return EligibilityDecision(EligibilityStatus.INCOMPATIBLE_FEATURE_SET, tuple(unknown), ())
    forbidden = tuple(
        sorted(
            feature
            for feature in binding.required_features
            if not known[feature].visible_to_formula(experiment_id)
        )
    )
    if forbidden:
        return EligibilityDecision(EligibilityStatus.FORBIDDEN_RESPONSE_ACCESS, (), forbidden)
    visible = catalogue.visible_features(experiment_id)
    missing = tuple(sorted(set(binding.required_features) - visible))
    if missing:
        return EligibilityDecision(EligibilityStatus.INCOMPATIBLE_FEATURE_SET, missing, ())
    return EligibilityDecision(EligibilityStatus.ELIGIBLE, (), ())


def validate_binding_catalogue(
    bindings: Sequence[FormulaExecutionBinding], catalogue: DataElementCatalogue
) -> None:
    ids = tuple(binding.binding_id for binding in bindings)
    if ids != tuple(sorted(set(ids))):
        raise SchemaViolation("formula bindings must be unique and sorted")
    known = set(catalogue.by_id())
    for binding in bindings:
        referenced = set(
            binding.required_features + binding.optional_features + binding.emitted_features
        )
        if binding.status is BindingStatus.EXECUTABLE and not referenced <= known:
            raise SchemaViolation("executable binding references unknown data elements")


__all__ = [
    "ABI_VERSION",
    "BindingStatus",
    "EligibilityDecision",
    "EligibilityStatus",
    "FormulaExecutionBinding",
    "ResourceBounds",
    "decide_eligibility",
    "validate_binding_catalogue",
]
