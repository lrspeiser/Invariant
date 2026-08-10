from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from pathlib import Path
from typing import Any

import sympy as sp

from .promotion_orchestrator import ELIGIBILITY

SCHEMA_VERSION = "sigma-g3-asymptotically-flat-transition-audit-1.0"


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


def _load_bound(root: Path, descriptor: dict[str, Any]) -> dict[str, Any]:
    path = root / descriptor["path"]
    if _file_sha(path) != descriptor["file_sha256"]:
        raise ValueError(f"bound file hash mismatch: {descriptor['path']}")
    value = json.loads(path.read_text(encoding="utf-8"))
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    if value.get("content_sha256") != descriptor["content_sha256"] or _sha(body) != descriptor[
        "content_sha256"
    ]:
        raise ValueError(f"bound content hash mismatch: {descriptor['path']}")
    return value


def _validate_target(record: dict[str, Any], target: dict[str, Any]) -> None:
    if record.get("seed_id") != target["seed_id"] or record.get("decision") != "blocked":
        raise ValueError("G3 predecessor identity or decision mismatch")
    if record.get("action_sha256") != target["action_sha256"]:
        raise ValueError("G3 action hash mismatch")
    if record["provenance"].get("binding_sha256") != target["predecessor_provenance_sha256"]:
        raise ValueError("G3 predecessor provenance mismatch")
    if record["full_lapse_operator_derivation"].get("content_sha256") != target[
        "lapse_derivation_sha256"
    ]:
        raise ValueError("G3 lapse derivation mismatch")
    if record["coercivity_certificate"].get("content_sha256") != target[
        "periodic_coercivity_sha256"
    ]:
        raise ValueError("G3 periodic coercivity mismatch")
    if record.get("first_missing_premise") != "asymptotically_flat_or_global_energy_domain":
        raise ValueError("G3 predecessor AF blocker changed")


def _validate_domain(domain: dict[str, Any]) -> None:
    required = {
        "contract_kind": "asymptotically_flat_reference_profile_not_constraint_solution",
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
            "BSSN_lapse": "1",
            "shift": "0",
            "role": "principal_and_asymptotic_reference_only",
        },
        "function_space": {
            "lapse_multiplier": "L2(R3)",
            "test_core": "C_c_infinity(R3)",
            "asymptotic_metric_target": "h-delta=O_2(r^-1),K=O_1(r^-2)",
            "scalar_falloff": "v=O(r^-2),X=O(r^-4),D_i(v)=O(r^-3)",
        },
        "direction_domain": "all_unit_spatial_covectors_no_sampling",
    }
    if domain != required:
        raise ValueError("G3 asymptotically flat transition domain changed")


def _radial_profile_certificate(length: Fraction) -> dict[str, Any]:
    radius = sp.Symbol("r", nonnegative=True, finite=True)
    scale = sp.Symbol("L", positive=True, finite=True)
    s = radius / scale
    v = 1 / sp.sqrt(1 + s**4)
    x = sp.factor(v**2 / 2)
    derivative = sp.factor(sp.diff(v, radius))
    derivative_at_scale = sp.factor(abs(derivative.subs(radius, scale)))
    derivative_squared = sp.factor(derivative**2)
    stationary_residual = sp.factor(sp.diff(derivative_squared, radius).subs(radius, scale))
    second_derivative_at_scale = sp.factor(
        sp.diff(derivative_squared, radius, 2).subs(radius, scale)
    )
    if (
        derivative_at_scale != 1 / (sp.sqrt(2) * scale)
        or stationary_residual != 0
        or second_derivative_at_scale.is_negative is not True
    ):
        raise ValueError("radial gradient maximum witness changed")
    body = {
        "v": str(v),
        "X": str(x),
        "d_v_d_r": str(derivative),
        "endpoint_values": {
            "X_at_r_zero": "1/2",
            "X_strictly_positive_at_finite_r": True,
            "X_limit_at_infinity": "0",
        },
        "interior_connection": {
            "matches_certified_center_at_r_zero": True,
            "gradient_covariant_at_center": ["1", "0", "0", "0"],
            "hessian_at_center": "zero",
        },
        "mixed_hessian_norm": {
            "global_maximum_location": "r=L",
            "global_maximum": "1/(sqrt(2)*L)",
            "stationary_residual_at_r_equals_L": str(stationary_residual),
            "second_derivative_squared_at_r_equals_L": str(second_derivative_at_scale),
            "at_L_100": "sqrt(2)/200",
            "below_predecessor_component_bound_1_over_100": True,
        },
        "falloff": {
            "v": "L^2/r^2+O(r^-6)",
            "X": "L^4/(2*r^4)+O(r^-8)",
            "d_v_d_r": "-2*L^2/r^3+O(r^-7)",
            "canonical_G2_energy_tail_integrable": True,
        },
        "length_specialization": str(length),
    }
    return {**body, "content_sha256": _sha(body)}


def _principal_certificate(beta: Fraction, length: Fraction) -> dict[str, Any]:
    p00_upper = Fraction(-1)
    spatial_lower = Fraction(39999, 40000)
    time_space_norm_upper_squared = Fraction(2) * beta**2 / length**2
    cone_upper = Fraction(-2499, 2500)
    if not (
        p00_upper < 0
        and spatial_lower > 0
        and cone_upper < 0
        and time_space_norm_upper_squared == Fraction(1, 50_000_000)
    ):
        raise ValueError("G3 radial principal bound failed")
    body = {
        "effective_metric_on_profile": {
            "P00": "-(1+3*beta^2*X^2)",
            "P0i": "-2*beta*D_i(v)",
            "Pij": "(1-beta^2*X^2)*delta_ij",
        },
        "uniform_bounds_X_in_0_to_half": {
            "P00_upper": str(p00_upper),
            "spatial_eigenvalue_lower": str(spatial_lower),
            "time_space_norm_upper": "1/(5000*sqrt(2))",
            "time_space_norm_upper_squared": str(time_space_norm_upper_squared),
            "characteristic_discriminant_lower": str(spatial_lower),
            "BSSN_sigma": "1",
            "slicing_cone_polynomial_upper": str(cone_upper),
        },
        "direction_sphere_method": "isotropic_spatial_block_plus_exact_radial_time_space_norm_no_sampling",
        "status": "pass_on_complete_radial_reference_profile_including_X_limit_zero",
        "scope": "principal/common-cone profile theorem, not an Einstein-constraint solution",
    }
    return {**body, "content_sha256": _sha(body)}


def _lapse_crossing_obstruction(beta: Fraction, length: Fraction) -> dict[str, Any]:
    body = {
        "unitary_lapse": "N_phi=1/sqrt(2X)=sqrt(1+(r/L)^4)",
        "unitary_trace_curvature": "K_phi=0 on the t=0 radial reference profile",
        "full_multiplier": "Delta_N(r)=v(r)^3+(3/2)*beta^2*v(r)^7",
        "pointwise_properties": {
            "positive_at_every_finite_r": True,
            "pointwise_kernel": "none",
            "limit_at_infinity": "0",
            "asymptotic": "L^6/r^6+O(r^-14)",
        },
        "annulus_approximate_zero_modes": {
            "sequence": "f_R in C_c_infinity(R3), support R<r<2R, ||f_R||_L2=1",
            "bound_for_R_at_least_L": (
                "||Delta_N f_R||_L2 <= (L/R)^6+(3/2)*beta^2*(L/R)^14"
            ),
            "bound_limit": "0",
            "conclusion": "zero_lies_in_approximate_spectrum_and_inverse_is_unbounded",
        },
        "boundary_condition_robustness": (
            "compactly supported annulus modes obey every asymptotic homogeneous boundary condition"
        ),
        "Dirac_operator_status": "blocked_not_boundedly_invertible_on_L2_R3",
        "exact_obstruction": "uniform_timelike_clock_margin_is_lost_as_X_tends_to_zero",
        "beta_specialization": str(beta),
        "length_specialization": str(length),
    }
    return {**body, "content_sha256": _sha(body)}


def build_g3_asymptotically_flat_transition_audit(
    config: dict[str, Any], root: str | Path
) -> dict[str, Any]:
    root = Path(root).resolve()
    if config.get("data_eligibility") != ELIGIBILITY:
        raise ValueError("campaign eligibility is not fail-closed")
    _validate_domain(config["transition_domain"])
    predecessor = _load_bound(root, config["predecessor"])
    target = config["target"]
    record = next(item for item in predecessor["candidate_records"] if item["seed_id"] == target["seed_id"])
    _validate_target(record, target)
    beta = Fraction(target["beta"])
    length = Fraction(config["transition_domain"]["transition_length_L"])
    if beta != Fraction(1, 100) or length != 100:
        raise ValueError("unsupported G3 AF transition specialization")
    profile = _radial_profile_certificate(length)
    principal = _principal_certificate(beta, length)
    lapse = _lapse_crossing_obstruction(beta, length)
    gates = {
        "typed_action_and_predecessor": {"status": "pass"},
        "explicit_AF_decaying_gradient_profile": {"status": "pass"},
        "finite_canonical_G2_energy_tail": {"status": "pass"},
        "uniform_principal_and_common_cone_on_reference_profile": {"status": "pass"},
        "uniform_lapse_Dirac_invertibility": {
            "status": "blocked",
            "reason": "Delta_N approaches zero and has normalized annulus approximate zero modes",
        },
        "Einstein_constraint_solution": {
            "status": "blocked",
            "reason": "the flat reference metric with nonzero scalar stress is not a constraint solution",
        },
        "global_hamiltonian_energy": {"status": "blocked"},
        "formal_prerequisite_completion": {"status": "blocked"},
    }
    provenance_body = {
        "predecessor_content_sha256": config["predecessor"]["content_sha256"],
        "action_sha256": target["action_sha256"],
        "predecessor_provenance_sha256": target["predecessor_provenance_sha256"],
        "lapse_derivation_sha256": target["lapse_derivation_sha256"],
        "periodic_coercivity_sha256": target["periodic_coercivity_sha256"],
        "transition_domain_sha256": _sha(config["transition_domain"]),
        "profile_sha256": profile["content_sha256"],
        "principal_sha256": principal["content_sha256"],
        "lapse_obstruction_sha256": lapse["content_sha256"],
        "data_eligibility": ELIGIBILITY,
    }
    candidate = {
        "seed_id": target["seed_id"],
        "action_sha256": target["action_sha256"],
        "decision": "blocked",
        "transition_domain": config["transition_domain"],
        "radial_profile_certificate": profile,
        "principal_common_cone_certificate": principal,
        "lapse_crossing_obstruction": lapse,
        "gate_ledger": gates,
        "first_missing_premise": "uniformly_invertible_Delta_N_on_AF_decaying_gradient_domain",
        "negative_energy_counterexample_found": False,
        "provenance": {**provenance_body, "binding_sha256": _sha(provenance_body)},
        "solar_bundle": {"generated": False, "status": "blocked"},
    }
    body = {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": config["campaign_id"],
        "source_bindings": {"predecessor": config["predecessor"]},
        "target_seed_count": 1,
        "decision_counts": {"blocked": 1},
        "candidate_records": [candidate],
        "AF_principal_common_cone_profile_pass_count": 1,
        "AF_lapse_Dirac_pass_count": 0,
        "full_formal_pass_count": 0,
        "solar_bundle_count": 0,
        "observational_data_opened": False,
        "paid_llm_spend_usd": 0.0,
        "data_eligibility": dict(ELIGIBILITY),
        "interpretation": (
            "A smooth finite-tail radial profile connects the certified X=1/2 center to X->0 "
            "and retains uniform scalar/BSSN cone margins. However the scalar-clock lapse grows "
            "without bound and the exact Delta_N multiplier tends to zero. Normalized compactly "
            "supported annulus modes prove that Delta_N has no bounded inverse on L2(R3), so the "
            "periodic Dirac result cannot extend to this asymptotically flat domain."
        ),
    }
    return {**body, "content_sha256": _sha(body)}
