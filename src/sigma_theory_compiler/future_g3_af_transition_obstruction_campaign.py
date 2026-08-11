"""Candidate-specific AF transition profiles and exact unitary-lapse obstructions."""

from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from pathlib import Path
from typing import Any

import sympy as sp

from .cubic_bssn_domain import cubic_scalar_effective_metric
from .g3_full_lapse_dirac_operator_audit import _derive_full_delta
from .promotion_orchestrator import ELIGIBILITY
from .scalar_tensor_pack import generic_g3_variation_noether_control

CONFIG_SCHEMA = "sigma-future-g3-af-transition-obstruction-config-1.0"
ARTIFACT_SCHEMA = "sigma-future-g3-af-transition-obstruction-campaign-1.0"
FIRST_BLOCKER = "bounded_global_unitary_Delta_N_inverse_on_candidate_AF_transition_profile"


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_bound(root: Path, descriptor: dict[str, Any], label: str) -> dict[str, Any]:
    path = (root / descriptor["path"]).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} path escapes repository") from error
    if not path.is_file() or _file_sha(path) != descriptor["file_sha256"]:
        raise ValueError(f"{label} file hash mismatch")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{label} must contain an object")
    if "content_sha256" in descriptor:
        body = {key: item for key, item in value.items() if key != "content_sha256"}
        if (
            value.get("content_sha256") != descriptor["content_sha256"]
            or _sha(body) != (descriptor["content_sha256"])
        ):
            raise ValueError(f"{label} content hash mismatch")
    return value


def _validate_profile_domain(domain: dict[str, Any]) -> None:
    required = {
        "contract_kind": (
            "candidate_action_bound_asymptotically_flat_transition_reference_"
            "not_constraint_solution"
        ),
        "initial_slice": "R3_with_radial_coordinate_r",
        "transition_length_L": "100",
        "scalar_profile": {
            "phi_at_t0": "0",
            "normal_gradient_v": "1/sqrt(1+(r/L)^4)",
            "spatial_gradient_at_t0": "0",
            "X": "1/(2*(1+(r/L)^4))",
            "mixed_hessian": "phi_0i=D_i(v)",
            "other_hessian_components_at_t0": "0",
        },
        "reference_gravity_data": {
            "h_ij": "delta_ij",
            "K_ij": "0",
            "ADM_lapse": "1",
            "shift": "0",
            "role": "constraint_ansatz_and_principal_reference_only",
        },
        "function_space": {
            "lapse_multiplier": "L2(R3)",
            "test_core": "C_c_infinity(R3)",
            "asymptotic_metric_target": "h-delta=O_2(r^-1),K=O_1(r^-2)",
            "scalar_falloff": "v=O(r^-2),X=O(r^-4),D_i(v)=O(r^-3)",
        },
        "constraint_normalization": {
            "M_Pl": "1",
            "Hamiltonian_constraint": "R3+K^2-K_ij*K^ij=2*rho",
        },
        "direction_domain": "all_unit_spatial_covectors_no_sampling",
    }
    if domain != required:
        raise ValueError("future G3 AF transition domain changed")


def _symbolic_profile_identity_control() -> dict[str, Any]:
    velocity, radial_derivative, beta = sp.symbols("v d beta", real=True)
    x = velocity**2 / 2
    gradient = [velocity, 0, 0, 0]
    hessian = [
        [0, radial_derivative, 0, 0],
        [radial_derivative, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    matrix = cubic_scalar_effective_metric(
        x=x,
        gradient_covariant=gradient,
        hessian_covariant=hessian,
        g2_x=0,
        g2_xx=0,
        g3_phi=0,
        g3_x_phi=0,
        g3_x=-beta,
        g3_xx=0,
    )
    expected = [
        [-(1 + 3 * beta**2 * x**2), -2 * beta * radial_derivative, 0, 0],
        [-2 * beta * radial_derivative, 1 - beta**2 * x**2, 0, 0],
        [0, 0, 1 - beta**2 * x**2, 0],
        [0, 0, 0, 1 - beta**2 * x**2],
    ]
    residuals = [
        str(sp.factor(matrix[row][column] - expected[row][column]))
        for row in range(4)
        for column in range(4)
    ]
    if set(residuals) != {"0"}:
        raise ValueError("future G3 AF effective-metric identity failed")
    body = {
        "control": "direct symbolic cubic effective metric on the radial transition jet",
        "source_normalized_G3_X": "-beta",
        "effective_metric": {
            "P00": "-(1+3*beta^2*X^2)",
            "P0r": "-2*beta*d_v_d_r",
            "Pij": "(1-beta^2*X^2)*delta_ij",
        },
        "exact_residuals": residuals,
        "status": "pass",
    }
    return {**body, "content_sha256": _sha(body)}


def _symbolic_g3_stress_center_control() -> dict[str, Any]:
    passed, evidence = generic_g3_variation_noether_control()
    expected_stress = (
        "T_mu_nu=-G3_X theta p_mu p_nu-2 nabla_(mu(G3) p_nu)+g_mu_nu nabla_rho(G3)p^rho"
    )
    if (
        not passed
        or evidence.get("status") != "pass"
        or evidence.get("metric_stress_tensor") != expected_stress
    ):
        raise ValueError("future G3 AF cubic stress identity control failed")
    body = {
        "control": "generic exact cubic-Horndeski metric stress specialized at the center jet",
        "generic_control_evidence_sha256": _sha(evidence),
        "metric_stress_tensor": expected_stress,
        "center_substitution": {
            "theta_equals_box_phi": "0",
            "nabla_mu_X": "0",
            "nabla_mu_G3_equals_G3_X_nabla_mu_X": "0",
            "T_mu_nu_G3": "0",
        },
        "status": "pass",
    }
    return {**body, "content_sha256": _sha(body)}


def _radial_profile(length: Fraction) -> dict[str, Any]:
    radius = sp.Symbol("r", nonnegative=True, finite=True)
    scale = sp.Symbol("L", positive=True, finite=True)
    velocity = 1 / sp.sqrt(1 + (radius / scale) ** 4)
    x = sp.factor(velocity**2 / 2)
    derivative = sp.factor(sp.diff(velocity, radius))
    derivative_squared_derivative = sp.factor(sp.diff(derivative**2, radius))
    maximum = sp.factor(abs(derivative.subs(radius, scale)))
    dimensionless_radius = sp.Symbol("s", positive=True)
    canonical_energy = sp.factor(
        4
        * sp.pi
        * scale**3
        * sp.integrate(
            dimensionless_radius**2 / (2 * (1 + dimensionless_radius**4)),
            (dimensionless_radius, 0, sp.oo),
        )
    )
    if maximum != 1 / (sp.sqrt(2) * scale) or canonical_energy != (
        sp.sqrt(2) * sp.pi**2 * scale**3 / 2
    ):
        raise ValueError("future G3 AF radial profile control failed")
    body = {
        "normal_gradient_v": str(velocity),
        "X": str(x),
        "d_v_d_r": str(derivative),
        "d_dr_of_d_v_d_r_squared": str(derivative_squared_derivative),
        "global_derivative_maximum_proof": {
            "sign_for_0_less_r_less_L": "positive",
            "unique_positive_stationary_point": "r=L",
            "sign_for_r_greater_L": "negative",
            "maximum_abs_d_v_d_r": "1/(sqrt(2)*L)",
            "at_L_100": "sqrt(2)/200",
        },
        "endpoint_and_falloff": {
            "X_at_r_zero": "1/2",
            "X_strictly_positive_at_finite_r": True,
            "X_limit_at_infinity": "0",
            "v": "L^2/r^2+O(r^-6)",
            "X": "L^4/(2*r^4)+O(r^-8)",
            "d_v_d_r": "-2*L^2/r^3+O(r^-7)",
        },
        "canonical_G2_energy_diagnostic": {
            "density": "rho_G2=v^2/2",
            "radial_integral": "4*pi*integral_0^infinity r^2*rho_G2 dr",
            "exact": str(
                canonical_energy.subs(scale, sp.Rational(length.numerator, length.denominator))
            ),
            "finite": True,
            "scope": "canonical term only; not the complete constrained cubic-G3 Hamiltonian energy",
        },
        "length_specialization": str(length),
        "status": "pass_smooth_decaying_gradient_reference_profile",
    }
    return {**body, "content_sha256": _sha(body)}


def _principal_certificate(beta: Fraction, length: Fraction) -> dict[str, Any]:
    spatial_lower = 1 - beta**2 / 4
    time_space_norm_squared = 2 * beta**2 / length**2
    cone_upper = -1 + 4 * beta / length
    if spatial_lower <= 0 or cone_upper >= 0 or time_space_norm_squared <= 0:
        raise ValueError("future G3 AF principal/common-cone bound failed")
    body = {
        "beta": str(beta),
        "length": str(length),
        "direct_candidate_recompute": True,
        "prior_candidate_bound_reused": False,
        "uniform_bounds_on_zero_less_equal_X_less_equal_one_half": {
            "P00_upper": "-1",
            "common_time_covector_margin": "1",
            "spatial_eigenvalue_lower": str(spatial_lower),
            "time_space_norm_upper_squared": str(time_space_norm_squared),
            "characteristic_discriminant_lower": str(spatial_lower),
            "BSSN_sigma": "1",
            "slicing_cone_polynomial_upper": str(cone_upper),
            "slicing_cone_separation": str(-cone_upper),
        },
        "direction_sphere_method": (
            "isotropic spatial block plus exact radial time-space norm; no sampling"
        ),
        "status": "pass_on_complete_AF_reference_profile_including_X_limit_zero",
        "scope": (
            "candidate-specific principal/common-cone theorem on the reference profile; "
            "not an Einstein-constraint solution or evolution-invariant domain"
        ),
    }
    return {**body, "content_sha256": _sha(body)}


def _flat_reference_constraint_audit(
    beta: Fraction, stress_control: dict[str, Any]
) -> dict[str, Any]:
    body = {
        "ansatz": "h_ij=delta_ij,K_ij=0 with the declared scalar transition jet",
        "center_jet_r_equals_zero": {
            "X": "1/2",
            "symmetric_hessian": "0",
            "R3_plus_K_squared_minus_KijKij": "0",
            "rho_canonical_G2": "1/2",
            "rho_cubic_G3": "0",
            "Hamiltonian_constraint_residual_LHS_minus_2rho": "-1",
        },
        "cubic_zero_reason": (
            "the exact generic cubic stress identity specializes to zero when box(phi)=0 and "
            "nabla_mu(X)=0 on the zero-Hessian center jet"
        ),
        "cubic_stress_center_control_sha256": stress_control["content_sha256"],
        "beta_specialization": str(beta),
        "status": "reject_flat_reference_as_constraint_datum",
        "theory_rejected": False,
        "scope": (
            "necessary center check rejects only this flat reference ansatz; it does not rule "
            "out a conformally deformed AF constraint solution"
        ),
    }
    return {**body, "content_sha256": _sha(body)}


def _lapse_obstruction(beta: Fraction, length: Fraction) -> dict[str, Any]:
    derivation = _derive_full_delta(beta)
    velocity = sp.Symbol("v", positive=True, finite=True)
    beta_symbol = sp.Symbol("beta", real=True)
    lapse = sp.Symbol("N", positive=True, finite=True)
    curvature = sp.Symbol("K", real=True)
    expression = (
        lapse**-3
        + 2 * beta_symbol * curvature * lapse**-4
        + sp.Rational(3, 2) * beta_symbol**2 * lapse**-7
    )
    specialized = sp.factor(expression.subs({lapse: 1 / velocity, curvature: 0}))
    expected = velocity**3 + sp.Rational(3, 2) * beta_symbol**2 * velocity**7
    residual = sp.factor(specialized - expected)
    if residual != 0 or derivation["full_Delta_N"]["beta_specialization"] != str(beta):
        raise ValueError("future G3 AF lapse specialization failed")
    n_two_bound = Fraction(1, 2**6) + Fraction(3, 2) * beta**2 / 2**14
    body = {
        "beta": str(beta),
        "length": str(length),
        "full_Delta_N_derivation": derivation,
        "direct_candidate_specialization": True,
        "periodic_inverse_reused": False,
        "unitary_clock_lapse": "N_phi=1/sqrt(2X)=1/v=sqrt(1+(r/L)^4)",
        "unitary_trace_curvature": "K_phi=0 on the t=0 reference jet",
        "candidate_multiplier": "Delta_N(r)=v(r)^3+(3/2)*beta^2*v(r)^7",
        "specialization_residual": str(residual),
        "pointwise_properties": {
            "positive_at_every_finite_r": True,
            "pointwise_kernel": "none",
            "limit_at_infinity": "0",
            "asymptotic": "L^6/r^6+O(r^-14)",
        },
        "normalized_annulus_sequence": {
            "test_functions": "f_n in C_c_infinity, support nL<r<2nL, ||f_n||_L2=1",
            "exact_norm_bound": "||Delta_N*f_n||_L2<=n^-6+(3/2)*beta^2*n^-14",
            "bound_at_n_equals_2": str(n_two_bound),
            "bound_limit": "0",
            "conclusion": "zero_in_approximate_spectrum_and_no_bounded_L2_inverse",
        },
        "boundary_robustness": (
            "compactly supported annulus modes satisfy every homogeneous AF boundary condition"
        ),
        "status": "exact_obstruction_to_bounded_global_unitary_Delta_N_inverse",
        "scope": (
            "obstructs this global unitary-clock L2 formulation on the declared AF profile; "
            "not a ghost, negative-energy mode, or obstruction to every nonunitary formulation"
        ),
    }
    return {**body, "content_sha256": _sha(body)}


def _validate_config(config: dict[str, Any]) -> None:
    if (
        config.get("schema_version") != CONFIG_SCHEMA
        or config.get("data_eligibility") != ELIGIBILITY
        or len(config.get("targets", [])) != 3
        or not config.get("output_path", "").startswith("runs/engine/")
    ):
        raise ValueError("future G3 AF transition config is invalid")
    if set(config.get("bindings", {})) != {
        "predecessor",
        "method_control",
        "method_control_source",
        "effective_metric_source",
        "lapse_source",
        "g3_variation_source",
    }:
        raise ValueError("future G3 AF transition bindings changed")


def build_future_g3_af_transition_obstruction_campaign(
    config: dict[str, Any], project_root: str | Path
) -> dict[str, Any]:
    """Audit three AF reference profiles without promoting constraint or energy gates."""

    _validate_config(config)
    root = Path(project_root).resolve()
    source_path = (root / config["adapter_source"]["path"]).resolve()
    if _file_sha(source_path) != config["adapter_source"]["file_sha256"]:
        raise ValueError("future G3 AF transition campaign source hash mismatch")
    predecessor = _load_bound(root, config["bindings"]["predecessor"], "predecessor")
    method = _load_bound(root, config["bindings"]["method_control"], "method control")
    for name in (
        "method_control_source",
        "effective_metric_source",
        "lapse_source",
        "g3_variation_source",
    ):
        binding = config["bindings"][name]
        path = (root / binding["path"]).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError(f"{name} path escapes repository") from error
        if not path.is_file() or _file_sha(path) != binding["file_sha256"]:
            raise ValueError(f"{name} file hash mismatch")
    if (
        predecessor.get("candidate_count") != 3
        or predecessor.get("periodic_distributed_Dirac_pass_count") != 3
        or predecessor.get("asymptotically_flat_Dirac_pass_count") != 0
        or predecessor.get("global_energy_pass_count") != 0
        or predecessor.get("full_formal_pass_count") != 0
        or method.get("AF_principal_common_cone_profile_pass_count") != 1
        or method.get("AF_lapse_Dirac_pass_count") != 0
    ):
        raise ValueError("future G3 AF transition predecessor boundary changed")
    identity = _symbolic_profile_identity_control()
    stress_control = _symbolic_g3_stress_center_control()
    records = []
    for target in config["targets"]:
        matches = [
            item
            for item in predecessor["candidate_records"]
            if item["candidate_id"] == target["candidate_id"]
        ]
        if len(matches) != 1:
            raise ValueError("future G3 AF target is not unique")
        prior = matches[0]
        expected = {
            "action_sha256": target["action_sha256"],
            "beta": target["beta"],
            "decision": "blocked",
            "first_blocker": "asymptotically_flat_or_global_energy_domain_missing",
            "content_sha256": target["predecessor_record_content_sha256"],
        }
        if any(prior.get(key) != value for key, value in expected.items()):
            raise ValueError("future G3 AF target binding changed")
        domain = target["transition_domain"]
        _validate_profile_domain(domain)
        beta = Fraction(target["beta"])
        length = Fraction(domain["transition_length_L"])
        profile = _radial_profile(length)
        principal = _principal_certificate(beta, length)
        constraint = _flat_reference_constraint_audit(beta, stress_control)
        obstruction = _lapse_obstruction(beta, length)
        domain_body = {
            "profile_id": target["profile_id"],
            "candidate_id": target["candidate_id"],
            "action_sha256": target["action_sha256"],
            "beta": str(beta),
            "transition_domain": domain,
            "direct_action_bound_registration": True,
            "method_control_domain_reused": False,
            "status": "pass_candidate_action_bound_AF_reference_domain_registered",
        }
        domain_certificate = {**domain_body, "content_sha256": _sha(domain_body)}
        gates = {
            "predecessor_action_and_periodic_Dirac": "pass",
            "candidate_action_bound_AF_transition_domain": "pass",
            "smooth_decaying_gradient_profile": "pass",
            "finite_canonical_G2_energy_tail": "pass_diagnostic_only",
            "uniform_AF_profile_principal_common_cone": "pass",
            "flat_reference_Einstein_constraint_ansatz": "reject_ansatz_not_theory",
            "bounded_global_unitary_Delta_N_inverse": "blocked_exact_obstruction",
            "alternative_nonunitary_AF_constraint_formulation": "blocked_not_registered",
            "candidate_specific_AF_Einstein_constraint_solution": "blocked",
            "candidate_specific_global_hamiltonian_energy": "blocked",
            "full_formal_pass": "blocked",
            "observational_data_seal": "pass",
        }
        provenance_body = {
            "predecessor_content_sha256": predecessor["content_sha256"],
            "predecessor_record_content_sha256": prior["content_sha256"],
            "candidate_id": target["candidate_id"],
            "action_sha256": target["action_sha256"],
            "profile_domain_sha256": domain_certificate["content_sha256"],
            "profile_sha256": profile["content_sha256"],
            "principal_sha256": principal["content_sha256"],
            "constraint_ansatz_sha256": constraint["content_sha256"],
            "lapse_obstruction_sha256": obstruction["content_sha256"],
            "cubic_stress_center_control_sha256": stress_control["content_sha256"],
            "data_eligibility": dict(ELIGIBILITY),
        }
        record_body = {
            "candidate_id": target["candidate_id"],
            "action_sha256": target["action_sha256"],
            "beta": str(beta),
            "decision": "blocked",
            "first_blocker": FIRST_BLOCKER,
            "AF_transition_domain_certificate": domain_certificate,
            "radial_profile_certificate": profile,
            "principal_common_cone_certificate": principal,
            "flat_reference_constraint_audit": constraint,
            "unitary_lapse_obstruction": obstruction,
            "gate_ledger": gates,
            "negative_energy_counterexample_found": False,
            "theory_rejected": False,
            "full_formal_pass": False,
            "observational_data_opened": False,
            "data_eligibility": dict(ELIGIBILITY),
            "provenance": {**provenance_body, "binding_sha256": _sha(provenance_body)},
        }
        records.append({**record_body, "content_sha256": _sha(record_body)})
    body = {
        "schema_version": ARTIFACT_SCHEMA,
        "campaign_id": config["campaign_id"],
        "campaign_adapter_source": config["adapter_source"],
        "config_content_sha256": _sha(config),
        "source_bindings": config["bindings"],
        "symbolic_profile_identity_control": identity,
        "symbolic_g3_stress_center_control": stress_control,
        "candidate_count": 3,
        "decision_counts": {"blocked": 3},
        "candidate_records": records,
        "AF_decaying_gradient_profile_pass_count": 3,
        "AF_principal_common_cone_profile_pass_count": 3,
        "flat_reference_constraint_ansatz_reject_count": 3,
        "AF_unitary_lapse_Dirac_pass_count": 0,
        "AF_Einstein_constraint_solution_pass_count": 0,
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
            "Each action now has a direct smooth AF decaying-gradient reference profile with a "
            "finite canonical tail and uniform principal/common-cone margins through X->0. The "
            "flat gravitational reference fails the Hamiltonian constraint at the center, and "
            "normalized annulus modes exactly obstruct a bounded global unitary-lapse inverse. "
            "No alternative AF constraint solution, global energy theorem, theory rejection, or "
            "full-formal pass is inferred."
        ),
    }
    return {**body, "content_sha256": _sha(body)}


def write_future_g3_af_transition_obstruction_campaign(
    config_path: str | Path, project_root: str | Path
) -> Path:
    root = Path(project_root).resolve()
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    artifact = build_future_g3_af_transition_obstruction_campaign(config, root)
    output = root / config["output_path"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_canonical(artifact) + "\n", encoding="utf-8")
    return output
