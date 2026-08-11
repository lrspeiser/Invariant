from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from pathlib import Path
from typing import Any

import sympy as sp

from .promotion_orchestrator import ELIGIBILITY
from .scalar_tensor_pack import generic_g3_variation_noether_control

ARTIFACT_SCHEMA = "sigma-future-g3-radial-conformal-constraint-reduction-campaign-1.0"
FIRST_BLOCKER = (
    "candidate_specific_positive_global_solution_of_radial_Lichnerowicz_BVP_"
    "with_psi_to_one"
)


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_bound(root: Path, descriptor: dict[str, Any], *, content: bool) -> dict[str, Any]:
    path = root / descriptor["path"]
    if _file_sha(path) != descriptor["file_sha256"]:
        raise ValueError(f"bound file hash mismatch: {descriptor['path']}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if content:
        body = {key: item for key, item in value.items() if key != "content_sha256"}
        if (
            value.get("content_sha256") != descriptor["content_sha256"]
            or _sha(body) != descriptor["content_sha256"]
        ):
            raise ValueError(f"bound content hash mismatch: {descriptor['path']}")
    return value


def _validate_sources(root: Path, config: dict[str, Any]) -> None:
    if _file_sha(root / config["adapter_source"]["path"]) != config["adapter_source"][
        "file_sha256"
    ]:
        raise ValueError("campaign source hash mismatch")
    for label, descriptor in config["source_bindings"].items():
        if _file_sha(root / descriptor["path"]) != descriptor["file_sha256"]:
            raise ValueError(f"{label} source hash mismatch")


def _validate_ansatz(ansatz: dict[str, Any]) -> None:
    expected = {
        "initial_slice": "R3_with_isotropic_radial_coordinate_r",
        "spatial_metric": "h_ij=psi(r)^4*delta_ij",
        "extrinsic_curvature": "K_ij=(K(r)/3)*h_ij",
        "extrinsic_curvature_convention": "K_ij=(dot(h)_ij-L_shift(h)_ij)/(2*N)",
        "scalar_position": "phi_at_t0=0",
        "scalar_normal_gradient": "Pi=v(r)=1/sqrt(1+(r/L)^4)",
        "scalar_spatial_gradient": "D_i(phi)=0",
        "transition_length_L": "100",
        "lapse": "ordinary_ADM_lapse_not_scalar_clock",
        "shift": "0_on_initial_slice",
        "conformal_boundary_conditions": ["psi(r)>0", "psi_prime(0)=0", "psi(infinity)=1"],
        "falloff_target": "psi-1=O_2(r^-1),K_ij=O_1(r^-2)",
    }
    if ansatz != expected:
        raise ValueError("radial conformal ansatz contract changed")


def _symbolic_constraint_reduction_control() -> dict[str, Any]:
    beta, v, d_v, h_00, mean_k, d_mean_k, scalar_r = sp.symbols(
        "beta v d_v H_00 K d_K R3", real=True
    )
    theta = -h_00 - v * mean_k
    q_0 = v * h_00
    q_r = v * d_v
    grad_g_0 = beta * q_0
    grad_g_r = beta * q_r
    grad_g_dot_p = -grad_g_0 * v
    cubic_rho = sp.factor(
        -beta * theta * v**2 - 2 * grad_g_0 * v - grad_g_dot_p
    )
    cubic_t_nr = sp.factor(-grad_g_r * v)
    canonical_rho = v**2 / 2
    solved_k = -beta * v**3 / 2
    momentum_lhs = -sp.Rational(2, 3) * d_mean_k
    momentum_rhs = -cubic_t_nr
    momentum_residual = sp.factor(
        (momentum_lhs - momentum_rhs).subs(
            d_mean_k, sp.diff(solved_k, v) * d_v
        )
    )
    rho = sp.factor(canonical_rho + cubic_rho.subs(mean_k, solved_k))
    hamiltonian_residual = sp.factor(
        scalar_r
        + sp.Rational(2, 3) * solved_k**2
        - 2 * rho
    )
    required_scalar_curvature = sp.solve(
        sp.Eq(hamiltonian_residual, 0), scalar_r
    )[0]
    expected_curvature = v**2 * (1 - sp.Rational(7, 6) * beta**2 * v**4)
    if (
        cubic_rho != beta * v**3 * mean_k
        or cubic_t_nr != -beta * v**2 * d_v
        or momentum_residual != 0
        or sp.factor(required_scalar_curvature - expected_curvature) != 0
    ):
        raise ValueError("symbolic radial constraint reduction failed")
    body = {
        "status": "pass",
        "signature": "(-,+,+,+)",
        "normal_frame": "n^mu=(1,0,0,0)_on_the_initial_slice",
        "extrinsic_curvature_convention": "K_ij=(dot(h)_ij-L_shift(h)_ij)/(2*N)",
        "scalar_hessian_trace": "theta=-H_00-Pi*K",
        "stress_projections": {
            "rho_G2": str(canonical_rho),
            "rho_G3": str(cubic_rho),
            "T_n_r_G3": str(cubic_t_nr),
        },
        "constraint_normalization": {
            "Hamiltonian": "R3+K^2-K_ij*K^ij=2*rho",
            "momentum_in_declared_K_convention": (
                "D_j(K^j_i-delta^j_i*K)=-T_n_i"
            ),
        },
        "pure_trace_momentum_solution": "K(r)=-(beta/2)*v(r)^3",
        "momentum_residual_after_solution": str(momentum_residual),
        "rho_after_momentum_solution": str(rho),
        "Hamiltonian_residual": str(hamiltonian_residual),
        "required_scalar_curvature": str(required_scalar_curvature),
        "scope": (
            "Exact candidate-independent tensor reduction for G2=X and G3=beta*X in the "
            "declared radial pure-trace initial-data ansatz. Candidate beta and action hashes "
            "are specialized separately; no existence theorem is asserted."
        ),
    }
    return {**body, "content_sha256": _sha(body)}


def _candidate_reduction(
    prior: dict[str, Any], profile: dict[str, Any], beta: Fraction, ansatz: dict[str, Any]
) -> dict[str, Any]:
    margin = 1 - Fraction(7, 6) * beta**2
    if margin <= 0:
        raise ValueError("candidate scalar-curvature coefficient is not positive")
    beta_squared = beta**2
    body = {
        "candidate_id": prior["candidate_id"],
        "action_sha256": prior["action_sha256"],
        "beta": str(beta),
        "ansatz": ansatz,
        "direct_action_binding": True,
        "family_label_used_as_constraint_evidence": False,
        "registered_profile_sha256": profile["radial_profile_certificate"]["content_sha256"],
        "nonunitary_principal_certificate_sha256": prior[
            "nonunitary_AF_principal_certificate"
        ]["content_sha256"],
        "exact_momentum_constraint": {
            "matter_flux_T_n_r": "-beta*v(r)^2*v_prime(r)",
            "equation": "-(2/3)*K_prime(r)=beta*v(r)^2*v_prime(r)",
            "AF_integration_constant": "0",
            "unique_pure_trace_solution": "K(r)=-(beta/2)*v(r)^3",
            "K_at_center": str(-beta / 2),
            "K_falloff": "-(beta/2)*L^6/r^6+O(r^-10)",
            "residual": "0",
            "status": "pass_exact_radial_momentum_constraint_reduction",
        },
        "exact_Hamiltonian_constraint": {
            "rho": "v(r)^2/2-(beta^2/2)*v(r)^6",
            "pure_trace_K_squared_minus_Kij_squared": "(beta^2/6)*v(r)^6",
            "required_scalar_curvature": "v(r)^2-(7/6)*beta^2*v(r)^6",
            "factorized_required_scalar_curvature": (
                "v(r)^2*(1-(7/6)*beta^2*v(r)^4)"
            ),
            "global_positive_factor_lower": str(margin),
            "beta_squared": str(beta_squared),
            "required_scalar_curvature_at_center": str(margin),
            "required_scalar_curvature_falloff": "L^4/r^4+O(r^-8)",
            "flat_spatial_metric_residual": (
                "-v(r)^2*(1-(7/6)*beta^2*v(r)^4)"
            ),
            "flat_spatial_metric_residual_strictly_negative_at_finite_r": True,
            "flat_pure_trace_completion_status": "reject_exact_ansatz",
        },
        "radial_Lichnerowicz_BVP": {
            "conformal_scalar_curvature_identity": (
                "R(h)=-8*psi(r)^(-5)*(psi_double_prime(r)+2*psi_prime(r)/r)"
            ),
            "equation": (
                "psi_double_prime(r)+2*psi_prime(r)/r+Q_beta(r)*psi(r)^5=0"
            ),
            "Q_beta": "(1/8)*v(r)^2*(1-(7/6)*beta^2*v(r)^4)",
            "Q_beta_strictly_positive_at_finite_r": True,
            "Q_beta_over_v_squared_lower": str(margin / 8),
            "boundary_conditions": ["psi(r)>0", "psi_prime(0)=0", "psi(infinity)=1"],
            "source_falloff": "Q_beta=L^4/(8*r^4)+O(r^-8)",
            "source_integrable_on_R3": True,
            "if_positive_global_solution_exists": {
                "Hamiltonian_constraint_residual": "0",
                "momentum_constraint_residual": "0",
                "metric_falloff": "psi-1=O_2(r^-1)",
                "extrinsic_curvature_falloff": "O_1(r^-6)",
                "nontrivial_AF_Einstein_constraint_datum": True,
            },
            "positive_global_solution_proved": False,
            "status": "blocked_at_positive_global_BVP_solution",
        },
        "candidate_nontrivial_AF_Einstein_constraint_solution_available": False,
        "theory_rejected": False,
        "decision": "blocked",
        "first_blocker": FIRST_BLOCKER,
        "scope": (
            "The exact momentum constraint is solved within the candidate-bound radial "
            "conformal/pure-trace ansatz, and the Hamiltonian constraint is reduced to one "
            "positive-coefficient radial Lichnerowicz boundary-value problem. The flat member "
            "of the class is ruled out exactly. A positive global solution with psi->1 has not "
            "been constructed or proved to exist, so no nontrivial AF constraint pass follows."
        ),
    }
    return {**body, "content_sha256": _sha(body)}


def build_future_g3_radial_conformal_constraint_reduction_campaign(
    config: dict[str, Any], project_root: str | Path
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    if config.get("data_eligibility") != ELIGIBILITY:
        raise ValueError("campaign eligibility is not fail-closed")
    _validate_sources(root, config)
    _validate_ansatz(config["radial_conformal_ansatz"])
    predecessor = _load_bound(root, config["bindings"]["predecessor"], content=True)
    profile_source = _load_bound(root, config["bindings"]["AF_profile_source"], content=True)
    if (
        predecessor.get("source_bindings", {}).get("predecessor", {}).get("content_sha256")
        != profile_source.get("content_sha256")
    ):
        raise ValueError("predecessor AF profile chain changed")
    stress_passed, stress_evidence = generic_g3_variation_noether_control()
    if (
        not stress_passed
        or stress_evidence["metric_stress_tensor"]
        != (
            "T_mu_nu=-G3_X theta p_mu p_nu-2 nabla_(mu(G3) p_nu)"
            "+g_mu_nu nabla_rho(G3)p^rho"
        )
    ):
        raise ValueError("generic G3 stress control failed")
    symbolic_control = _symbolic_constraint_reduction_control()
    predecessor_records = {
        item["candidate_id"]: item for item in predecessor["candidate_records"]
    }
    profile_records = {
        item["candidate_id"]: item for item in profile_source["candidate_records"]
    }
    records = []
    for target in config["targets"]:
        prior = predecessor_records.get(target["candidate_id"])
        profile = profile_records.get(target["candidate_id"])
        if (
            prior is None
            or profile is None
            or prior["action_sha256"] != target["action_sha256"]
            or profile["action_sha256"] != target["action_sha256"]
            or prior["beta"] != target["beta"]
            or profile["beta"] != target["beta"]
            or prior["content_sha256"] != target["predecessor_record_content_sha256"]
            or profile["content_sha256"] != target["AF_profile_record_content_sha256"]
            or prior["decision"] != "blocked"
            or prior["theory_rejected"] is not False
        ):
            raise ValueError("target binding changed")
        certificate = _candidate_reduction(
            prior, profile, Fraction(target["beta"]), config["radial_conformal_ansatz"]
        )
        gates = {
            "nonunitary_AF_principal_predecessor": {"status": "pass"},
            "radial_pure_trace_momentum_constraint_reduction": {"status": "pass"},
            "positive_Hamiltonian_source_registration": {"status": "pass"},
            "flat_pure_trace_completion": {"status": "reject_ansatz"},
            "positive_global_radial_Lichnerowicz_solution": {"status": "blocked"},
            "candidate_nontrivial_AF_Einstein_constraint_solution": {"status": "blocked"},
            "global_hamiltonian_energy": {"status": "blocked"},
            "full_formal": {"status": "blocked"},
        }
        provenance_body = {
            "predecessor_content_sha256": predecessor["content_sha256"],
            "predecessor_record_content_sha256": prior["content_sha256"],
            "AF_profile_source_content_sha256": profile_source["content_sha256"],
            "AF_profile_record_content_sha256": profile["content_sha256"],
            "action_sha256": prior["action_sha256"],
            "ansatz_sha256": _sha(config["radial_conformal_ansatz"]),
            "symbolic_control_sha256": symbolic_control["content_sha256"],
            "G3_stress_control_sha256": _sha(stress_evidence),
            "constraint_reduction_sha256": certificate["content_sha256"],
            "data_eligibility": dict(ELIGIBILITY),
        }
        record_body = {
            "candidate_id": prior["candidate_id"],
            "action_sha256": prior["action_sha256"],
            "beta": prior["beta"],
            "decision": "blocked",
            "first_blocker": FIRST_BLOCKER,
            "constraint_reduction_certificate": certificate,
            "gate_ledger": gates,
            "theory_rejected": False,
            "global_energy_pass": False,
            "full_formal_pass": False,
            "observational_data_opened": False,
            "data_eligibility": dict(ELIGIBILITY),
            "provenance": {**provenance_body, "binding_sha256": _sha(provenance_body)},
        }
        records.append({**record_body, "content_sha256": _sha(record_body)})
    if len(records) != 3:
        raise ValueError("expected exactly three candidate-bound reductions")
    body = {
        "schema_version": ARTIFACT_SCHEMA,
        "campaign_id": config["campaign_id"],
        "campaign_adapter_source": config["adapter_source"],
        "config_content_sha256": _sha(config),
        "source_bindings": {**config["bindings"], **config["source_bindings"]},
        "symbolic_constraint_reduction_control": symbolic_control,
        "G3_stress_control_evidence_sha256": _sha(stress_evidence),
        "candidate_count": 3,
        "decision_counts": {"blocked": 3},
        "candidate_records": records,
        "radial_pure_trace_momentum_constraint_reduction_pass_count": 3,
        "positive_Hamiltonian_source_registration_pass_count": 3,
        "flat_pure_trace_completion_ansatz_reject_count": 3,
        "radial_Lichnerowicz_BVP_registration_pass_count": 3,
        "positive_global_radial_Lichnerowicz_solution_pass_count": 0,
        "candidate_nontrivial_AF_Einstein_constraint_solution_pass_count": 0,
        "global_hamiltonian_energy_pass_count": 0,
        "full_formal_pass_count": 0,
        "first_blocker_counts": {FIRST_BLOCKER: 3},
        "observational_data_opened": False,
        "dark_matter_or_halo_inputs": False,
        "redshift_distance_inputs": False,
        "paid_llm_spend_usd": 0.0,
        "synthetic_fixture_role": "none_used",
        "data_eligibility": dict(ELIGIBILITY),
        "interpretation": (
            "For all three actions the candidate-bound nontrivial scalar profile now admits an "
            "exact radial conformal/pure-trace constraint reduction. The momentum constraint "
            "fixes K=-(beta/2)*v^3, the flat member is ruled out, and the Hamiltonian constraint "
            "becomes an explicit positive-coefficient radial Lichnerowicz boundary-value problem. "
            "No positive global solution with psi->1 is yet proved, so nontrivial AF constraint, "
            "global-energy, and full-formal counts remain zero."
        ),
    }
    return {**body, "content_sha256": _sha(body)}


def write_future_g3_radial_conformal_constraint_reduction_campaign(
    config_path: str | Path, project_root: str | Path
) -> Path:
    root = Path(project_root).resolve()
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    artifact = build_future_g3_radial_conformal_constraint_reduction_campaign(config, root)
    output = root / config["output_path"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_canonical(artifact) + "\n", encoding="utf-8")
    return output
