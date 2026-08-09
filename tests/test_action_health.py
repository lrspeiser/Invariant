from __future__ import annotations

import json
from pathlib import Path

from sigma_theory_compiler.action_health import analyze_action_health

ROOT = Path(__file__).resolve().parents[1]
GRAMMAR = ROOT / "configs" / "covariant_action_grammar.json"
CONTRACT = ROOT / "configs" / "covariant_field_contract.json"


def _run(name: str, tmp_path: Path):
    return analyze_action_health(
        ROOT / "configs" / "actions" / name,
        GRAMMAR,
        CONTRACT,
        tmp_path / name.replace(".json", ""),
        project_root=ROOT,
    )


def test_einstein_hilbert_control_passes_all_formal_gates(tmp_path) -> None:
    result = _run("einstein_hilbert_control.json", tmp_path)
    assert result["status"] == "pass"
    assert result["promotion_allowed"]
    assert result["physical_dof"] == 2
    assert all(gate["status"] == "pass" for gate in result["gates"].values())
    assert not result["observational_gates_unsealed"]
    assert result["gates"]["adm_decomposition"]["status"] == "pass"
    assert result["gates"]["static_dictionary_derivation"]["status"] == "pass"
    assert result["gates"]["legendre_map"]["status"] == "pass"
    assert result["gates"]["generated_dirac_closure"]["status"] == "pass"
    assert result["gates"]["parameter_domain"]["status"] == "pass"
    adm_path = Path(result["generated_adm_ir"]["path"])
    adm = json.loads(adm_path.read_text(encoding="utf-8"))
    assert adm["input_action_sha256"] == result["input_action_sha256"]
    assert adm["primary_constraint_seeds"] == [
        "p_N=0",
        "p_(N^i)=0 (three components)",
    ]
    legendre_path = Path(result["generated_legendre_ir"]["path"])
    legendre = json.loads(legendre_path.read_text(encoding="utf-8"))
    assert legendre["input_action_sha256"] == result["input_action_sha256"]
    assert legendre["input_adm_ir_sha256"] == adm["content_sha256"]
    assert legendre["generic_hessian_rank"] == 6
    dirac = json.loads(Path(result["generated_dirac_ir"]["path"]).read_text(encoding="utf-8"))
    assert dirac["input_legendre_ir_sha256"] == legendre["content_sha256"]
    assert dirac["distributed_constraint_closure"]["constraint_surface_rank"]["physical_dof"] == 2
    stability = json.loads(
        Path(result["generated_stability_ir"]["path"]).read_text(encoding="utf-8")
    )
    assert stability["input_action_sha256"] == result["input_action_sha256"]
    assert stability["input_dirac_ir_sha256"] == dirac["content_sha256"]
    assert stability["condition_certificate"]["status"] == "pass"
    principal = json.loads(
        Path(result["generated_principal_ir"]["path"]).read_text(encoding="utf-8")
    )
    assert principal["input_stability_ir_sha256"] == stability["content_sha256"]
    assert principal["gauge_reduction_certificate"]["retained_mode_count"] == 2
    hamiltonian = json.loads(
        Path(result["generated_hamiltonian_ir"]["path"]).read_text(encoding="utf-8")
    )
    assert hamiltonian["input_principal_ir_sha256"] == principal["content_sha256"]
    assert hamiltonian["positivity_certificate"]["status"] == "pass"
    static_dictionary = json.loads(
        Path(result["generated_static_dictionary_ir"]["path"]).read_text(encoding="utf-8")
    )
    assert static_dictionary["input_action_sha256"] == result["input_action_sha256"]
    assert static_dictionary["static_reductions"]["baryonic_diagnostic"]["z_b"] == (
        "n_b**2/n_0**2"
    )


def test_canonical_scalar_control_passes_combined_action_gates(tmp_path) -> None:
    result = _run("canonical_scalar_control.json", tmp_path)
    assert result["status"] == "pass"
    assert result["promotion_allowed"]
    assert result["physical_dof"] == 3
    assert result["generated_legendre_ir"]["generic_hessian_rank"] == 7
    assert result["generated_dirac_ir"]["physical_dof"] == 3


def test_proca_control_passes_combined_action_gates(tmp_path) -> None:
    result = _run("proca_control.json", tmp_path)
    assert result["status"] == "pass"
    assert result["promotion_allowed"]
    assert result["physical_dof"] == 5
    assert not result["discovery_blockers"]
    assert result["generated_legendre_ir"]["generic_hessian_rank"] == 9
    assert result["gates"]["adm_dirac"]["status"] == "pass"
    assert result["gates"]["generated_dirac_closure"]["status"] == "pass"
    assert result["generated_dirac_ir"]["physical_dof"] == 5


def test_einstein_aether_passes_patchwise_dirac_but_remains_unpromoted(tmp_path) -> None:
    result = _run("einstein_aether_control.json", tmp_path)
    assert result["status"] == "unresolved"
    assert not result["promotion_allowed"]
    assert result["gates"]["covariant_variation"]["status"] == "pass"
    assert result["gates"]["covariant_identity"]["status"] == "pass"
    assert result["gates"]["adm_dirac"]["status"] == "pass"
    assert result["gates"]["adm_dirac"]["physical_dof"] == 5
    assert result["gates"]["legendre_map"]["status"] == "pass"
    assert result["gates"]["generated_dirac_closure"]["status"] == "pass"
    assert result["gates"]["hamiltonian_stability"]["status"] == "unresolved"
    hamiltonian = json.loads(
        Path(result["generated_hamiltonian_ir"]["path"]).read_text(encoding="utf-8")
    )
    assert hamiltonian["physical_mode_count"] == 5
    assert hamiltonian["gauge_reduction_status"] == "pass"
    assert hamiltonian["legendre_transform_residual"] == "0"
    assert hamiltonian["positivity_certificate"]["status"] == "pass"
    assert (
        hamiltonian["positivity_certificate"]["declared_background_patch_status"]
        == "unresolved"
    )
    assert hamiltonian["generic_nonlinear_total_energy"]["status"] == "unresolved"
    assert result["gates"]["hamiltonian_stability"]["evidence"] == [
        "einstein_aether_linearized_physical_energy",
        "einstein_aether_restricted_nonlinear_total_energy",
        result["generated_hamiltonian_ir"]["path"],
    ]
    assert result["gates"]["principal_symbol"]["status"] == "pass"
    assert result["gates"]["principal_symbol"]["evidence"] == [
        "einstein_aether_reduced_five_mode_principal_domain",
        "einstein_aether_global_tilt_legendre_strata",
        "einstein_aether_covariant_arbitrary_background_hyperbolicity",
        result["generated_principal_ir"]["path"],
    ]
    adm = json.loads(Path(result["generated_adm_ir"]["path"]).read_text(encoding="utf-8"))
    assert adm["status"] == "pass"
    assert "p_(lambda_u)=0" in adm["primary_constraint_seeds"]
    assert adm["unit_aether_reduction"]["branch"] == "chi=sqrt(1+A_i A^i)>0"
    legendre = json.loads(Path(result["generated_legendre_ir"]["path"]).read_text(encoding="utf-8"))
    assert legendre["generic_hessian_rank"] == 9
    assert legendre["generic_hessian_nullity"] == 0
    assert result["generated_dirac_ir"]["physical_dof"] == 5
    assert result["generated_stability_ir"]["condition_status"] == "pass"
    assert result["generated_stability_ir"]["physical_hamiltonian_status"] == "unresolved"
    assert result["generated_stability_ir"]["principal_symbol_status"] == "pass"
    assert result["generated_principal_ir"]["status"] == "pass"
    assert result["generated_principal_ir"]["retained_mode_count"] == 5
    assert result["generated_hamiltonian_ir"]["status"] == "unresolved"
    assert result["generated_hamiltonian_ir"]["positivity_status"] == "pass"
    assert (
        result["generated_hamiltonian_ir"]["generic_nonlinear_total_energy"]["status"]
        == "unresolved"
    )


def test_quartic_horndeski_passes_patchwise_dirac_and_hamiltonian_subgates(
    tmp_path,
) -> None:
    result = _run("quartic_horndeski_control.json", tmp_path)
    assert result["status"] == "reject"
    assert not result["promotion_allowed"]
    assert result["gates"]["covariant_variation"]["status"] == "pass"
    assert result["gates"]["covariant_identity"]["status"] == "pass"
    assert result["gates"]["adm_decomposition"]["status"] == "pass"
    assert result["gates"]["legendre_map"]["status"] == "pass"
    assert result["generated_legendre_ir"]["generic_hessian_rank"] == 6
    assert result["generated_legendre_ir"]["generic_hessian_nullity"] == 1
    assert result["gates"]["generated_dirac_closure"]["status"] == "pass"
    assert result["gates"]["adm_dirac"]["status"] == "pass"
    assert result["gates"]["adm_dirac"]["physical_dof"] == 3
    assert result["generated_dirac_ir"]["family"] == "quartic_horndeski"
    dirac = json.loads(
        Path(result["generated_dirac_ir"]["path"]).read_text(encoding="utf-8")
    )
    assert dirac["local_canonical_transform"]["status"] == "pass"
    assert dirac["unitary_gauge_local_dirac_control"]["passed"]
    assert result["gates"]["parameter_domain"]["status"] == "reject"
    assert result["gates"]["parameter_domain"]["pointwise_status"] == "pass"
    assert result["gates"]["parameter_domain"]["background_domain"]["variables"][
        0
    ]["id"] == "A_star_squared"
    assert result["gates"]["parameter_domain"]["background_domain_preservation"][
        "status"
    ] == "reject"
    assert result["gates"]["parameter_domain"]["background_domain_preservation"][
        "missing_or_failed_controls"
    ] == []
    assert result["gates"]["parameter_domain"]["global_all_timelike_amplitudes"] == (
        "reject"
    )
    assert "A_star^2" in result["gates"]["parameter_domain"][
        "background_domain_required"
    ]
    assert result["gates"]["principal_symbol"]["status"] == "unresolved"
    assert result["generated_stability_ir"]["family"] == "quartic_horndeski"
    principal = json.loads(
        Path(result["generated_principal_ir"]["path"]).read_text(encoding="utf-8")
    )
    assert principal["family"] == "quartic_horndeski"
    assert principal["gauge_reduction_certificate"]["retained_mode_count"] == 3
    assert principal["exact_diagonal_physical_eigenbasis"]
    assert principal["status"] == "unresolved"
    assert principal["gauge_reduction_certificate"]["status"] == "pass"
    assert principal["declared_background_patch_certificate"]["status"] == "pass"
    assert "A_star" in principal["characteristic_speed_squared"][
        "tensor_speed_squared"
    ]
    assert result["gates"]["hamiltonian_stability"]["status"] == "unresolved"


def test_wrong_sign_candidate_is_rejected_by_generated_stability_gate(tmp_path) -> None:
    spec = json.loads(
        (ROOT / "configs" / "actions" / "canonical_scalar_control.json").read_text(
            encoding="utf-8"
        )
    )
    spec["role"] = "candidate"
    spec["coefficients"]["SCALAR_X"] = "-1"
    spec_path = tmp_path / "wrong-sign-scalar.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    result = analyze_action_health(
        spec_path,
        GRAMMAR,
        CONTRACT,
        tmp_path / "wrong-sign-health",
        project_root=ROOT,
    )
    assert result["status"] == "reject"
    assert not result["promotion_allowed"]
    assert result["gates"]["parameter_domain"]["status"] == "reject"
    assert result["gates"]["hamiltonian_stability"]["status"] == "reject"
    assert result["gates"]["principal_symbol"]["status"] == "reject"


def test_generated_q_candidate_exports_full_fail_closed_health_chain(tmp_path) -> None:
    result = _run("generated_gf_cb4ebf3da5a74582_q_q2_candidate.json", tmp_path)
    assert result["status"] == "reject"
    assert not result["promotion_allowed"]
    assert result["family"] == "unsupported"
    assert result["gates"]["static_dictionary_derivation"]["status"] == "pass"
    assert result["gates"]["higher_jet_regularity"]["status"] == "reject"
    assert result["gates"]["adm_decomposition"]["status"] == "pass"
    assert result["generated_adm_ir"]["status"] == "pass"
    assert result["gates"]["legendre_map"]["status"] == "unresolved"
    assert result["gates"]["generated_dirac_closure"]["status"] == "unresolved"
    assert result["gates"]["covariant_variation"]["status"] == "unresolved"
    assert result["gates"]["principal_symbol"]["status"] == "unresolved"
    assert result["gates"]["hamiltonian_stability"]["status"] == "unresolved"
    assert not result["observational_gates_unsealed"]
    legendre = json.loads(
        Path(result["generated_legendre_ir"]["path"]).read_text(encoding="utf-8")
    )
    assert legendre["unsupported_kinetic_terms"] == ["AETHER_Q1", "AETHER_Q2"]
    assert result["generated_q_operator_ir"]["rank_certificate"]["constant_rank"] is False
