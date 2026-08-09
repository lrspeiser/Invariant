from __future__ import annotations

import copy
import json
from pathlib import Path

from sigma_theory_compiler.action_ir import (
    compile_action_file,
    compile_action_spec,
    load_action_grammar,
)
from sigma_theory_compiler.adm_ir import compile_adm_ir
from sigma_theory_compiler.formal_backend import load_field_contract

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
        "projected_aether_q_generic_3plus1_decomposition": True,
    }


def _action(name: str) -> dict:
    return compile_action_file(
        ROOT / "configs" / "actions" / name,
        GRAMMAR_PATH,
        CONTRACT_PATH,
    )


def test_einstein_hilbert_adm_ir_is_hash_bound_and_extracts_lapse_shift_seeds() -> None:
    action = _action("einstein_hilbert_control.json")
    first = compile_adm_ir(action, _controls())
    second = compile_adm_ir(action, _controls())
    assert first == second
    assert first["status"] == "pass"
    assert first["input_action_sha256"] == action["content_sha256"]
    assert first["term_templates_complete"]
    assert first["velocity_channels"] == [
        "K_ij=(dot(h)_ij-L_shift(h)_ij)/(2N)"
    ]
    assert first["primary_constraint_seeds"] == [
        "p_N=0",
        "p_(N^i)=0 (three components)",
    ]
    assert first["secondary_constraint_seeds"] == [
        "H_perp=0",
        "H_i=0 (three components)",
    ]
    assert first["boundary_contract"]
    assert len(first["content_sha256"]) == 64


def test_scalar_and_proca_adm_ir_extract_field_specific_channels() -> None:
    scalar = compile_adm_ir(_action("canonical_scalar_control.json"), _controls())
    proca = compile_adm_ir(_action("proca_control.json"), _controls())
    assert scalar["status"] == "pass"
    assert "Pi_phi" in scalar["velocity_channels"]
    assert "D_i phi" in scalar["spatial_jets"]
    assert proca["status"] == "pass"
    assert "E_i" in proca["velocity_channels"]
    assert "p_(A_perp)=0" in proca["primary_constraint_seeds"]
    assert any("Gauss/mass" in item for item in proca["secondary_constraint_seeds"])


def test_generated_aether_subset_gets_complete_decomposition_without_false_closure() -> None:
    grammar = load_action_grammar(GRAMMAR_PATH)
    contract = load_field_contract(CONTRACT_PATH)
    spec = {
        "schema_version": "sigma-action-spec-1.0",
        "role": "candidate",
        "fields": ["g_mu_nu", "u_mu", "lambda_u"],
        "matter_metric": "g_mu_nu",
        "terms": ["EH_R", "AETHER_K1", "UNIT_VECTOR_CONSTRAINT"],
        "coefficients": {"AETHER_K1": "-M_Pl^2*c1/2"},
        "universal_constants": ["M_Pl", "c1"],
        "static_dictionary_status": "derived",
    }
    action = compile_action_spec(spec, grammar, contract)
    adm = compile_adm_ir(action, _controls())
    assert action["valid"]
    assert adm["status"] == "pass"
    assert adm["source_role"] == "candidate"
    assert adm["promotion_allowed"] is False
    assert {item["term_id"] for item in adm["terms"]} == {
        "EH_R",
        "AETHER_K1",
        "UNIT_VECTOR_CONSTRAINT",
    }
    assert "p_(lambda_u)=0" in adm["primary_constraint_seeds"]
    assert "-chi^2+A_i A^i+1=0" in adm["secondary_constraint_seeds"]
    assert adm["unit_aether_reduction"]["branch"].endswith(">0")
    assert "does not by itself prove" in adm["proof_scope"]


def test_adm_ir_fails_closed_without_executed_controls_or_complete_templates() -> None:
    action = _action("einstein_hilbert_control.json")
    unverified = compile_adm_ir(action)
    assert unverified["status"] == "unresolved"
    assert unverified["missing_or_failed_controls"]

    corrupted = copy.deepcopy(action)
    corrupted["canonical"]["terms"].append(
        {
            "id": "UNKNOWN_NONMINIMAL_TERM",
            "coefficient": "1",
            "density": "sqrt(-g) mystery",
            "fields": ["g_mu_nu"],
            "invariant": None,
            "maximum_derivatives_per_field": 2,
        }
    )
    unsupported = compile_adm_ir(corrupted, _controls())
    assert unsupported["status"] == "unresolved"
    assert unsupported["missing_templates"] == ["UNKNOWN_NONMINIMAL_TERM"]
    assert not unsupported["term_templates_complete"]


def test_adm_ir_json_is_serializable() -> None:
    adm = compile_adm_ir(_action("einstein_aether_control.json"), _controls())
    encoded = json.dumps(adm, sort_keys=True)
    assert '"status": "pass"' in encoded
    assert len(adm["terms"]) == 6


def test_q_candidate_gets_exact_generic_tilt_templates_but_seals_constraint_seeds() -> None:
    action = _action("generated_gf_cb4ebf3da5a74582_q_q2_candidate.json")
    adm = compile_adm_ir(action, _controls())
    assert action["valid"], action["errors"]
    assert adm["status"] == "pass"
    assert adm["missing_templates"] == []
    assert adm["higher_time_derivative_channels"] == ["L_n a_i", "L_n a_perp"]
    assert "unresolved" in adm["primary_constraint_seeds"][0]
    assert "higher-jet Dirac" in adm["secondary_constraint_seeds"][0]
    assert "P3^{nn}=A_i A^i" in adm["lapse_shift_statement"]
    assert not adm["promotion_allowed"]
