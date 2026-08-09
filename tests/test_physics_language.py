from __future__ import annotations

import json
from pathlib import Path

from sigma_theory_compiler.physics_language import (
    SCHEMA_VERSION,
    compile_physics_program,
)

ROOT = Path(__file__).resolve().parents[1]


def test_coherence_gravity_is_expressible_but_fails_closed_on_missing_adapters() -> None:
    program = json.loads(
        (ROOT / "configs/physics_programs/coherence_gravity_example.json").read_text()
    )
    result = compile_physics_program(program)
    assert result["status"] == "unresolved_missing_adapters"
    assert result["errors"] == []
    assert result["universal_metric"] == "g"
    assert result["inferred_kinds"]["G1"] == "observable"
    assert result["inferred_kinds"]["total_action"] == "action"
    assert result["mutation_space"]["declared_cardinality"] == 12
    missing = {(item["primitive"], item["stage"]) for item in result["missing_adapters"]}
    assert ("state.coherent", "covariance") in missing
    assert ("operator.complex_scalar_kinetic", "variation") in missing
    assert ("observable.first_order_coherence", "principal") in missing


def test_current_einstein_scalar_composition_is_ready_for_formal_routing() -> None:
    program = {
        "schema_version": SCHEMA_VERSION,
        "fields": [
            {"id": "g", "kind": "metric"},
            {"id": "phi", "kind": "real_scalar", "matter": False},
        ],
        "matter_coupling": {"universal_metric": "g", "exceptions": []},
        "concepts": [
            {"id": "eh", "primitive": "operator.einstein_hilbert", "inputs": ["g"]},
            {
                "id": "kinetic",
                "primitive": "operator.real_scalar_kinetic",
                "inputs": ["g", "phi"],
            },
            {"id": "action", "primitive": "action.sum", "inputs": ["eh", "kinetic"]},
        ],
        "required_verification_stages": [
            "syntax",
            "type",
            "dimension",
            "covariance",
            "variation",
            "noether",
            "adm",
            "dirac",
            "hamiltonian",
            "principal",
        ],
    }
    result = compile_physics_program(program)
    assert result["status"] == "ready", result
    assert result["missing_adapters"] == []


def test_unknown_concept_requires_a_semantic_declaration() -> None:
    program = {
        "schema_version": SCHEMA_VERSION,
        "fields": [{"id": "g", "kind": "metric"}],
        "matter_coupling": {"universal_metric": "g", "exceptions": []},
        "concepts": [{"id": "mystery", "primitive": "concept.analogy", "inputs": ["g"]}],
    }
    result = compile_physics_program(program)
    assert result["status"] == "reject"
    assert "add a typed primitive_declaration" in " ".join(result["errors"])


def test_nonuniversal_matter_metric_is_rejected() -> None:
    program = {
        "schema_version": SCHEMA_VERSION,
        "fields": [
            {"id": "g", "kind": "metric"},
            {"id": "psi", "kind": "complex_scalar", "matter": True},
        ],
        "matter_coupling": {"universal_metric": "g", "exceptions": ["psi"]},
        "concepts": [],
    }
    result = compile_physics_program(program)
    assert result["status"] == "reject"
    assert "cannot declare exceptions" in " ".join(result["errors"])
