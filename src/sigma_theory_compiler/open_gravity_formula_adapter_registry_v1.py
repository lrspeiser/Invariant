"""Small initial registry of synthetic formula adapters.

Only known-answer controls live here initially.  Historical scientific formulas
must be added through explicit versioned bindings; absence is reported as
UNADAPTED rather than silently replaced by these controls.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from sigma_theory_compiler.open_gravity_formula_execution_protocol_v1 import (
    FormulaExecutionBinding,
)
from sigma_theory_compiler.sigma_core import SchemaViolation, canonical_sha256


def vector_scale_control(
    features: Mapping[str, Any], parameters: Mapping[str, Any]
) -> Mapping[str, Any]:
    if set(features) != {"source.vector.acceleration"}:
        raise SchemaViolation("vector-scale control input changed")
    if set(parameters) != {"scale_denominator", "scale_numerator"}:
        raise SchemaViolation("vector-scale control parameters changed")
    numerator = parameters["scale_numerator"]
    denominator = parameters["scale_denominator"]
    if type(numerator) is not int or type(denominator) is not int or denominator == 0:
        raise SchemaViolation("vector-scale control parameters must be exact integers")
    values = np.asarray(features["source.vector.acceleration"], dtype=np.float64)
    return {"prediction.vector.acceleration": values * (numerator / denominator)}


@dataclass(frozen=True, slots=True)
class AdapterRegistration:
    adapter_id: str
    adapter_sha256: str
    formula_binding: FormulaExecutionBinding

    def __post_init__(self) -> None:
        expected = canonical_sha256(
            {
                "adapter_id": self.adapter_id,
                "entrypoint": self.formula_binding.entrypoint,
                "binding_sha256": self.formula_binding.content_sha256,
            }
        )
        if self.adapter_sha256 != expected:
            raise SchemaViolation("adapter registration hash changed")

    @classmethod
    def create(
        cls, adapter_id: str, formula_binding: FormulaExecutionBinding
    ) -> AdapterRegistration:
        body = {
            "adapter_id": adapter_id,
            "entrypoint": formula_binding.entrypoint,
            "binding_sha256": formula_binding.content_sha256,
        }
        return cls(adapter_id, canonical_sha256(body), formula_binding)


def validate_adapter_registry(registrations: Sequence[AdapterRegistration]) -> None:
    adapter_ids = tuple(row.adapter_id for row in registrations)
    formula_ids = tuple(row.formula_binding.formula_id for row in registrations)
    if adapter_ids != tuple(sorted(set(adapter_ids))):
        raise SchemaViolation("adapter IDs must be unique and sorted")
    if len(set(formula_ids)) != len(formula_ids):
        raise SchemaViolation("formula has more than one active synthetic adapter")
    for row in registrations:
        row.formula_binding.resolve()


__all__ = [
    "AdapterRegistration",
    "validate_adapter_registry",
    "vector_scale_control",
]
