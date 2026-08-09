from __future__ import annotations

import copy
from pathlib import Path

from sigma_theory_compiler.action_ir import (
    compile_action_file,
    compile_action_spec,
    load_action_grammar,
)
from sigma_theory_compiler.adm_ir import compile_adm_ir
from sigma_theory_compiler.formal_backend import load_field_contract
from sigma_theory_compiler.legendre_ir import compile_legendre_ir

ROOT = Path(__file__).resolve().parents[1]
GRAMMAR_PATH = ROOT / "configs" / "covariant_action_grammar.json"
CONTRACT_PATH = ROOT / "configs" / "covariant_field_contract.json"


def _controls() -> dict[str, bool]:
    return {
        "cadabra_adm_spatial_curvature_variation": True,
        "nonlinear_adm_hamiltonian_constraint_algebra": True,
        "canonical_scalar": True,
        "proca_adm_dirac": True,
        "einstein_aether_generic_3plus1_legendre": True,
        "unit_timelike_vector_dirac_chain": True,
    }


def _compile(name: str) -> tuple[dict, dict, dict]:
    action = compile_action_file(
        ROOT / "configs" / "actions" / name,
        GRAMMAR_PATH,
        CONTRACT_PATH,
    )
    adm = compile_adm_ir(action, _controls())
    return action, adm, compile_legendre_ir(action, adm)


def test_known_control_legendre_maps_are_exact_hash_bound_and_deterministic() -> None:
    expected = {
        "einstein_hilbert_control.json": (6, 6),
        "canonical_scalar_control.json": (7, 7),
        "proca_control.json": (9, 9),
        "einstein_aether_control.json": (9, 9),
    }
    for name, (velocity_count, rank) in expected.items():
        action, adm, legendre = _compile(name)
        assert legendre == compile_legendre_ir(action, adm)
        assert legendre["status"] == "pass"
        assert legendre["input_action_sha256"] == action["content_sha256"]
        assert legendre["input_adm_ir_sha256"] == adm["content_sha256"]
        assert legendre["velocity_count"] == velocity_count
        assert legendre["generic_hessian_rank"] == rank
        assert legendre["generic_hessian_nullity"] == 0
        assert legendre["legendre_status"] == "regular_generic"
        assert len(legendre["content_sha256"]) == 64


def test_scalar_proca_and_aether_sector_blocks_are_extracted() -> None:
    _, _, scalar = _compile("canonical_scalar_control.json")
    _, _, proca = _compile("proca_control.json")
    _, _, aether = _compile("einstein_aether_control.json")
    assert scalar["sector_blocks"]["scalar_phi"]["determinant"] == "1"
    assert proca["sector_blocks"]["proca_spatial_vector"]["determinant"] == "1"
    assert (
        aether["sector_blocks"]["unit_aether_spatial_vector"]["determinant"]
        == "M_Pl**6*(c1 + c4)**3"
    )
    factors = {
        (item["factor"], item["multiplicity"])
        for item in aether["regularity_factors"]
    }
    assert ("c1 + c4", 3) in factors
    assert ("c1 + c3 - 1", 5) in factors
    assert ("c1 + 3*c2 + c3 + 2", 1) in factors


def test_singular_aether_subset_emits_three_kinetic_primary_constraints() -> None:
    grammar = load_action_grammar(GRAMMAR_PATH)
    contract = load_field_contract(CONTRACT_PATH)
    action = compile_action_spec(
        {
            "schema_version": "sigma-action-spec-1.0",
            "role": "candidate",
            "fields": ["g_mu_nu", "u_mu", "lambda_u"],
            "matter_metric": "g_mu_nu",
            "terms": ["EH_R", "AETHER_K2", "UNIT_VECTOR_CONSTRAINT"],
            "coefficients": {},
            "universal_constants": ["M_Pl", "c2"],
            "static_dictionary_status": "derived",
        },
        grammar,
        contract,
    )
    adm = compile_adm_ir(action, _controls())
    legendre = compile_legendre_ir(action, adm)
    assert action["valid"]
    assert adm["status"] == "pass"
    assert legendre["status"] == "unresolved"
    assert legendre["generic_hessian_rank"] == 6
    assert legendre["generic_hessian_nullity"] == 3
    assert legendre["hessian_determinant"] == "0"
    assert legendre["kinetic_primary_constraints"] == [
        "p_V_u0",
        "p_V_u1",
        "p_V_u2",
    ]


def test_legendre_ir_rejects_an_adm_ir_from_another_action_hash() -> None:
    action, adm, _ = _compile("einstein_hilbert_control.json")
    wrong = copy.deepcopy(adm)
    wrong["input_action_sha256"] = "0" * 64
    result = compile_legendre_ir(action, wrong)
    assert result["status"] == "reject"
    assert result["errors"] == ["ADM IR belongs to a different action hash"]
