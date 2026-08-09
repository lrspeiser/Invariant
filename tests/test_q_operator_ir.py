from pathlib import Path

from sigma_theory_compiler.action_ir import (
    compile_action_file,
    compile_action_spec,
    load_action_grammar,
)
from sigma_theory_compiler.formal_backend import load_field_contract
from sigma_theory_compiler.q_operator_ir import compile_q_operator_ir

ROOT = Path(__file__).resolve().parents[1]
GRAMMAR_PATH = ROOT / "configs" / "covariant_action_grammar.json"
CONTRACT_PATH = ROOT / "configs" / "covariant_field_contract.json"


def _action(name: str) -> dict:
    return compile_action_file(
        ROOT / "configs" / "actions" / name, GRAMMAR_PATH, CONTRACT_PATH
    )


def test_q_q2_survivor_is_rejected_by_exact_zero_mode_rank_change() -> None:
    result = compile_q_operator_ir(
        _action("generated_gf_cb4ebf3da5a74582_q_q2_candidate.json")
    )
    assert result["status"] == "reject"
    assert result["perturbative_order_in_vector_amplitude"] == {
        "AETHER_Q1": 2,
        "AETHER_Q2": 4,
    }
    assert result["quadratic_lagrangian_coefficients"]["AETHER_Q2"] == "0"
    assert result["rank_certificate"] == {
        "spatial_vector_components": 3,
        "rank_at_k_nonzero_without_local_regularizer": 3,
        "rank_at_k_zero_without_local_regularizer": 0,
        "constant_rank": False,
    }
    assert result["homogeneous_velocity_hessian_k0"] == "0"
    assert "underdetermined" in result["conclusion"]
    assert not result["observational_data_opened"]


def test_non_q_controls_pass_as_not_applicable() -> None:
    result = compile_q_operator_ir(_action("einstein_hilbert_control.json"))
    assert result["status"] == "pass"
    assert not result["applicable"]


def test_q_with_only_local_acceleration_regularizer_still_fails_gradient_gate() -> None:
    grammar = load_action_grammar(GRAMMAR_PATH)
    contract = load_field_contract(CONTRACT_PATH)
    action = compile_action_spec(
        {
            "schema_version": "sigma-action-spec-1.0",
            "role": "candidate",
            "fields": ["g_mu_nu", "u_mu", "lambda_u"],
            "matter_metric": "g_mu_nu",
            "terms": ["EH_R", "AETHER_K4", "AETHER_Q1", "UNIT_VECTOR_CONSTRAINT"],
            "coefficients": {
                "AETHER_K4": "M_Pl^2*c4/2",
                "AETHER_Q1": "epsilon*M_Pl^2*a_sigma^2",
            },
            "universal_constants": [
                "M_Pl",
                "c4",
                "epsilon",
                "L_sigma",
                "a_sigma",
            ],
            "parameter_domain": {"positive": ["M_Pl", "c4", "epsilon", "L_sigma", "a_sigma"]},
            "static_dictionary_status": "derived",
        },
        grammar,
        contract,
    )
    assert action["valid"], action["errors"]
    result = compile_q_operator_ir(action)
    assert result["status"] == "reject"
    assert result["rank_certificate"]["constant_rank"]
    assert result["gradient_sign_certificates"]["transverse"]["status"] == "reject"


def test_generated_mixed_q_sqrt_x_survivor_repairs_zero_mode_rank_only() -> None:
    result = compile_q_operator_ir(
        _action("generated_gf_5df8715b319f54cb_q_sqrtx_candidate.json")
    )
    assert result["status"] == "reject"
    assert result["rank_certificate"]["constant_rank"]
    assert result["homogeneous_velocity_hessian_k0"] == "M_Pl**2*epsilon"
    assert result["gradient_energy_coefficients"] == {
        "transverse": "0",
        "longitudinal": "0",
    }
    assert "arbitrary static spatial vector patterns" in result["conclusion"]


def test_negative_sqrt_x_regularizer_is_rejected_as_a_ghost() -> None:
    result = compile_q_operator_ir(
        compile_action_file(
            ROOT / "runs" / "covariant-export-v1" / "specs" / "GF-d4f5d179057437bd.json",
            GRAMMAR_PATH,
            CONTRACT_PATH,
        )
    )
    assert result["status"] == "reject"
    assert result["local_acceleration_sign_certificate"]["status"] == "reject"
    assert "ghost" in result["conclusion"]


def test_static_null_k1_k4_completion_repairs_both_necessary_symbols() -> None:
    action = _action("generated_gf_5df8715b319f54cb_static_null_completion.json")
    result = compile_q_operator_ir(action)
    assert action["valid"], action["errors"]
    assert result["status"] == "reject"
    assert result["rank_certificate"]["constant_rank"]
    assert result["gradient_energy_coefficients"] == {
        "transverse": "M_Pl**2*gamma/2",
        "longitudinal": "M_Pl**2*gamma/2",
    }
    assert all(
        item["status"] == "pass"
        for item in result["gradient_sign_certificates"].values()
    )
    assert result["q_kinetic_sign_certificate"]["status"] == "pass"
    assert result["nonlinear_x_convexity"]["status"] == "pass"
    covariant = result["constant_background_covariant_principal"]
    assert covariant["status"] == "reject"
    assert covariant["low_frequency_speed_squared"] == "gamma/epsilon"
    assert covariant["matter_cone_certificate"]["status"] == "pass"
    assert covariant["complete_tilt_root_audit"]["expanded_quartic_nonreal_root_count"] == 2
    assert "nonreal" in result["conclusion"]
