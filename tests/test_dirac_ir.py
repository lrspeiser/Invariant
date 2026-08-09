from __future__ import annotations

import copy
from pathlib import Path

from sigma_theory_compiler.action_ir import (
    compile_action_file,
    compile_action_spec,
    load_action_grammar,
)
from sigma_theory_compiler.adm_ir import compile_adm_ir
from sigma_theory_compiler.dirac_ir import (
    canonical_scalar_spatial_density_certificate,
    compile_dirac_ir,
)
from sigma_theory_compiler.formal_backend import load_field_contract
from sigma_theory_compiler.legendre_ir import compile_legendre_ir

ROOT = Path(__file__).resolve().parents[1]
GRAMMAR_PATH = ROOT / "configs" / "covariant_action_grammar.json"
CONTRACT_PATH = ROOT / "configs" / "covariant_field_contract.json"


def _controls() -> dict[str, bool]:
    return {
        name: True
        for name in (
            "cadabra_adm_spatial_curvature_variation",
            "nonlinear_adm_hamiltonian_constraint_algebra",
            "canonical_metric_diffeomorphism_algebra",
            "canonical_metric_dewitt_kinetic_covariance",
            "spatial_curvature_density_diffeomorphism_covariance",
            "canonical_scalar",
            "canonical_scalar_noether_identity",
            "canonical_scalar_gravity_cross_constraint_identities",
            "three_spatial_dimensional_smeared_brackets",
            "proca_adm_dirac",
            "proca_divergence_identity",
            "proca_stress_noether_identity",
            "proca_reduced_smeared_constraint_algebra",
            "einstein_aether_generic_3plus1_legendre",
            "einstein_aether_generic_lapse_shift_constraint_seeds",
            "einstein_aether_spatial_diffeomorphism_algebra",
            "einstein_aether_generic_dh_covariance",
            "einstein_aether_generic_hh_deformation_kinematics",
            "einstein_aether_arbitrary_background_4d_noether",
            "regular_holonomic_multiplier_dirac_theorem",
            "unit_timelike_vector_dirac_chain",
        )
    }


def _compile(name: str) -> tuple[dict, dict, dict, dict]:
    action = compile_action_file(
        ROOT / "configs" / "actions" / name,
        GRAMMAR_PATH,
        CONTRACT_PATH,
    )
    adm = compile_adm_ir(action, _controls())
    legendre = compile_legendre_ir(action, adm)
    return action, adm, legendre, compile_dirac_ir(action, adm, legendre, _controls())


def test_pure_gr_dirac_ir_derives_canonical_transform_and_two_modes() -> None:
    action, adm, legendre, result = _compile("einstein_hilbert_control.json")
    assert result == compile_dirac_ir(action, adm, legendre, _controls())
    assert result["status"] == "pass"
    assert result["input_action_sha256"] == action["content_sha256"]
    assert result["input_adm_ir_sha256"] == adm["content_sha256"]
    assert result["input_legendre_ir_sha256"] == legendre["content_sha256"]
    local = result["local_canonical_transform"]
    assert local["hessian_rank"] == 6
    assert local["hessian_nullity"] == 0
    assert (
        local["hessian_chain_residual"]
        == "Matrix([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]])"
    )
    assert local["channel_map"][0] == "K11=dot_h11/2 at N=1,N^i=0"
    closure = result["distributed_constraint_closure"]
    assert closure["status"] == "pass"
    assert closure["constraint_surface_rank"]["physical_dof"] == 2
    assert closure["constraint_surface_rank"]["extended_first_class_constraints"] == 8


def test_scalar_gravity_certificate_closes_and_counts_three_modes() -> None:
    certificate = canonical_scalar_spatial_density_certificate()
    assert certificate["passed"]
    assert certificate["local_frame_density_weight_residual"] == "0"
    assert certificate["gravity_matter_cross_hh_antisymmetry_residual"] == "0"
    _, _, _, result = _compile("canonical_scalar_control.json")
    assert result["status"] == "pass"
    assert result["local_canonical_transform"]["hessian_rank"] == 7
    closure = result["distributed_constraint_closure"]
    assert closure["family"] == "canonical_scalar_gravity"
    assert closure["constraint_surface_rank"]["physical_dof"] == 3
    assert closure["poisson_algebra"]["status"] == "pass"


def test_regular_aether_specialization_inherits_generic_closure_and_five_modes() -> None:
    _, _, _, result = _compile("einstein_aether_control.json")
    assert result["status"] == "pass"
    assert result["local_canonical_transform"]["hessian_rank"] == 9
    closure = result["distributed_constraint_closure"]
    assert closure["family"] == "einstein_aether"
    assert closure["regular_local_legendre_patch"]
    assert closure["constraint_surface_rank"]["physical_dof"] == 5
    assert "coefficient specializations" in closure["scope"]


def test_singular_aether_branch_stays_unresolved_with_primary_constraints() -> None:
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
    result = compile_dirac_ir(action, adm, legendre, _controls())
    assert action["valid"]
    assert result["status"] == "unresolved"
    local = result["local_canonical_transform"]
    assert local["status"] == "pass"
    assert local["hessian_rank"] == 6
    assert local["hessian_nullity"] == 3
    assert local["primary_constraints"] == ["P_u0", "P_u1", "P_u2"]
    assert result["distributed_constraint_closure"]["status"] == "unresolved"
    assert (
        result["distributed_constraint_closure"]["constraint_surface_rank"]["physical_dof"] is None
    )


def test_proca_reduced_smeared_algebra_closes_and_counts_five_combined_modes() -> None:
    _, _, _, result = _compile("proca_control.json")
    assert result["local_canonical_transform"]["status"] == "pass"
    assert result["local_canonical_transform"]["hessian_rank"] == 9
    assert result["status"] == "pass"
    closure = result["distributed_constraint_closure"]
    assert closure["family"] == "proca_gravity"
    assert closure["poisson_algebra"]["status"] == "pass"
    assert closure["constraint_surface_rank"]["physical_dof"] == 5
    assert closure["constraint_surface_rank"]["extended_pairs"] == 14
    assert closure["constraint_surface_rank"]["extended_second_class_constraints"] == 2


def test_dirac_ir_rejects_broken_hash_chain_and_missing_controls() -> None:
    action, adm, legendre, _ = _compile("einstein_hilbert_control.json")
    wrong = copy.deepcopy(legendre)
    wrong["input_adm_ir_sha256"] = "0" * 64
    rejected = compile_dirac_ir(action, adm, wrong, _controls())
    assert rejected["status"] == "reject"
    assert rejected["errors"] == ["Legendre IR belongs to a different ADM hash"]

    unresolved = compile_dirac_ir(action, adm, legendre, {})
    assert unresolved["status"] == "unresolved"
    assert unresolved["distributed_constraint_closure"]["missing_or_failed_controls"]
