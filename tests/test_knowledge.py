from pathlib import Path

from sigma_theory_compiler.knowledge import (
    GateOntology,
    _infer_outcome,
    pareto_fronts,
)

ROOT = Path(__file__).resolve().parents[1]
ONTOLOGY = ROOT / "configs" / "gate_ontology.json"


def test_gate_ontology_enforces_hard_before_observational() -> None:
    ontology = GateOntology.from_path(ONTOLOGY)
    gates = {row["id"]: row for row in ontology.gates}
    assert gates["hamiltonian_boundedness"]["hard"] is True
    assert gates["hamiltonian_boundedness"]["stage"] < gates["measured_galaxy_rotation"]["stage"]


def test_prohibited_evidence_is_retained_as_history_only() -> None:
    ontology = GateOntology.from_path(ONTOLOGY)
    status, reason = ontology.evidence_admissibility("GR/NFW-derived acceleration target")
    assert status == "historical_only"
    assert "gr/nfw-derived" in reason


def test_outcome_inference_does_not_turn_mixed_language_into_pass() -> None:
    assert _infer_outcome("exact branch rejected") == "reject"
    assert _infer_outcome("all hard gates passed") == "pass"
    assert _infer_outcome("one gate passed but final action rejected") == "mixed"


def test_pareto_front_does_not_collapse_axes_to_one_score() -> None:
    rows = [
        {"formula_id": "A", "integrity": 1.0, "simplicity": 0.5},
        {"formula_id": "B", "integrity": 0.5, "simplicity": 1.0},
        {"formula_id": "C", "integrity": 0.4, "simplicity": 0.4},
    ]
    fronts = pareto_fronts(rows, ["integrity", "simplicity"])
    assert {row["formula_id"] for row in fronts[0]} == {"A", "B"}
    assert [row["formula_id"] for row in fronts[1]] == ["C"]
