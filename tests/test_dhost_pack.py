from __future__ import annotations

import json
from pathlib import Path

from sigma_theory_compiler.dhost_pack import compile_reduced_dhost_pack

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "configs/operator_packs/dhost_rank_one_family.json"


def test_rank_one_dhost_family_generates_degeneracy_before_enumeration() -> None:
    result = compile_reduced_dhost_pack(json.loads(PACK.read_text()))
    assert result["status"] == "compiled_full_covariant_adapters_unresolved", result
    assert result["errors"] == []
    assert result["generated_coefficients"]["c"] == "C*alpha**2"
    assert result["determinant_residual"] == "0"
    assert result["null_vector_residual"] == ["0", "0"]
    assert result["c_override_residual"] == "0"
    assert result["mutation_space"]["declared_cardinality"] == 15
    assert not result["mutation_space"]["enumerated"]
    assert result["known_dirac_mechanism_control"]["passed"]
    assert result["capability_status"]["reduced_adm_kinetic_degeneracy"] == "pass"
    assert result["capability_status"]["generic_secondary_constraint"] == "unresolved"


def test_inconsistent_dhost_completion_is_rejected() -> None:
    spec = json.loads(PACK.read_text())
    spec["kinetic"]["c_override"] = "C*alpha^2 + 1"
    result = compile_reduced_dhost_pack(spec)
    assert result["status"] == "reject"
    assert result["c_override_residual"] == "1"
    assert "violates the generated degeneracy relation" in " ".join(result["errors"])


def test_zero_a_branch_is_not_misclassified_as_the_regular_family() -> None:
    spec = json.loads(PACK.read_text())
    spec["kinetic"]["a"] = "0"
    result = compile_reduced_dhost_pack(spec)
    assert result["status"] == "reject"
    assert "requires a!=0" in " ".join(result["errors"])
