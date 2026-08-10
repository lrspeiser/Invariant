from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from pathlib import Path
from typing import Any

import sympy as sp

from .promotion_orchestrator import ELIGIBILITY

SCHEMA_VERSION = "sigma-g4-source-class-scalar-uniqueness-audit-1.0"


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
    if (
        record.get("seed_id") != target["seed_id"]
        or record.get("action_sha256") != target["action_sha256"]
        or record.get("decision") != "blocked"
        or record.get("candidate_analytic_prediction_status")
        != "pass_on_declared_scalar_free_background"
    ):
        raise ValueError("G4 Solar predecessor target or decision mismatch")
    if record["provenance"].get("binding_sha256") != target[
        "predecessor_provenance_sha256"
    ]:
        raise ValueError("G4 Solar predecessor provenance mismatch")
    if record["coupling_and_PPN_certificate"].get("content_sha256") != target[
        "coupling_certificate_sha256"
    ]:
        raise ValueError("G4 coupling certificate mismatch")
    if record["exact_scalar_free_branch_certificate"].get("content_sha256") != target[
        "scalar_free_branch_sha256"
    ]:
        raise ValueError("G4 scalar-free branch certificate mismatch")
    if record["real_solar_admissibility"].get("content_sha256") != target[
        "prior_admissibility_sha256"
    ]:
        raise ValueError("G4 prior Solar admissibility mismatch")


def _validate_source_class(source_class: dict[str, Any]) -> None:
    required = {
        "role": "candidate_specific_static_weak_compact_source_theorem_not_real_Sun_registration",
        "slice": {
            "topology": "R3_one_asymptotically_Euclidean_end",
            "inner_boundary": "none",
            "ordinary_lapse_interval": ["99/100", "101/100"],
            "inverse_metric_ellipticity_lower": "99/100",
            "coordinate_volume_density_interval": ["99/100", "101/100"],
        },
        "scalar": {
            "function_space": "D^{1,2}(R3)",
            "boundary": "chi->0_at_spatial_infinity",
            "static": True,
        },
        "matter": {
            "support": "arbitrary_shape_contained_in_coordinate_ball_B_R",
            "trace_mass_density": "tau=(-T_E)/c^2=(epsilon-sum_i_p_i)/c^2",
            "trace_interval": "0<=tau<=rho_trace_max",
            "allows_anisotropic_pressure": True,
            "dimensionless_trace_radius_bound": (
                "G_star*rho_trace_max*R^2/c^2<=1/1000"
            ),
        },
        "coefficient_bounds_apply_to": (
            "every_self_consistent_static_candidate_solution_in_the_class"
        ),
    }
    if source_class != required:
        raise ValueError("G4 source-class contract changed")


def _global_coupling_lipschitz_certificate() -> dict[str, Any]:
    phi = sp.Symbol("phi", real=True)
    y = sp.Symbol("y", nonnegative=True)
    alpha = -phi / sp.sqrt(2500 + 56 * phi**2)
    kinetic = 4 * (14 * phi**2 + 625) / (phi**2 + 50) ** 2
    d_alpha_d_chi = sp.factor(sp.diff(alpha, phi) / sp.sqrt(kinetic))
    derivative_magnitude_y = sp.factor((-d_alpha_d_chi).subs(phi**2, y))
    lipschitz_gap = sp.factor(sp.Rational(1, 50) - derivative_magnitude_y)
    expected_gap = y * (392 * y + 19375) / (100 * (14 * y + 625) ** 2)
    if sp.factor(lipschitz_gap - expected_gap) != 0 or expected_gap.is_nonnegative is not True:
        raise ValueError("global scalar coupling Lipschitz bound failed")
    body = {
        "alpha(phi)": str(alpha),
        "d_alpha_d_chi": str(d_alpha_d_chi),
        "one_over_50_minus_abs_derivative": str(lipschitz_gap),
        "global_result": "abs(d_alpha/d_chi)<=1/50_for_all_real_phi",
        "integrated_result": "abs(alpha(chi))<=abs(chi)/50_with_alpha(0)=0",
        "sign_result": "alpha(chi)*chi<=0",
        "role": "candidate_specific_global_nonlinear_bound_not_only_beta0",
    }
    return {**body, "content_sha256": _sha(body)}


def _source_class_coercivity_certificate(source_class: dict[str, Any]) -> dict[str, Any]:
    source_slice = source_class["slice"]
    lapse_lower, lapse_upper = map(Fraction, source_slice["ordinary_lapse_interval"])
    volume_lower, volume_upper = map(
        Fraction, source_slice["coordinate_volume_density_interval"]
    )
    ellipticity_lower = Fraction(source_slice["inverse_metric_ellipticity_lower"])
    trace_radius = Fraction(1, 1000)
    geometry_ratio = lapse_upper * volume_upper / (
        lapse_lower * volume_lower * ellipticity_lower
    )
    pi_rational_upper = Fraction(22, 7)
    eta_upper = (
        Fraction(16, 50) * pi_rational_upper * trace_radius * geometry_ratio
    )
    coercive_margin = 1 - eta_upper
    if (
        geometry_ratio != Fraction(1_020_100, 970_299)
        or eta_upper != Fraction(81_608, 77_182_875)
        or coercive_margin != Fraction(77_101_267, 77_182_875)
        or eta_upper >= 1
    ):
        raise ValueError("source-class scalar coercivity arithmetic failed")
    body = {
        "static_Einstein_frame_scalar_equation": (
            "D_i(N*D^i(chi))=(4*pi*G_star/c^2)*N*alpha(chi)*tau"
        ),
        "integrated_identity": (
            "integral N*sqrt(h)*|Dchi|_h^2="
            "-(4*pi*G_star/c^2)*integral N*sqrt(h)*alpha(chi)*chi*tau"
        ),
        "function_space_and_boundary": {
            "space": source_class["scalar"]["function_space"],
            "infinity": source_class["scalar"]["boundary"],
            "inner_boundary": source_slice["inner_boundary"],
            "boundary_flux": "zero",
        },
        "inequality_chain": {
            "candidate_coupling": "-alpha(chi)*chi<=chi^2/50",
            "compact_support": "integral_support chi^2<=R^2*integral chi^2/r^2",
            "Hardy_R3": "integral chi^2/r^2<=4*integral |grad_delta chi|^2",
            "kinetic_lower": (
                "N_min*sqrt(h)_min*lambda_min*integral|grad_delta chi|^2"
            ),
        },
        "resolved_profile_Birman_Schwinger_route": {
            "free_static_operator": "L0=-D_i(N*D^i) on D^{1,2}",
            "potential": "W=(4*pi*G_star/(50*c^2))*N*tau",
            "operator": "B=sqrt(W)*L0^(-1)*sqrt(W)",
            "Schur_bound": (
                "kappa=sup_x integral G_L0(x,y)*W(y)*dmu_h(y)"
            ),
            "coercivity_rule": "kappa<1 implies Q>=(1-kappa)*Q0",
            "flat_space_specialization": (
                "kappa=(G_star/(50*c^2))*sup_x integral tau(y)/|x-y| d^3y"
            ),
            "advantage": (
                "a registered resolved trace profile may pass without a pointwise rho_trace_max"
            ),
            "status": "exact_conditional_profile_criterion",
        },
        "dimensionless_relative_form_bound": {
            "eta_formula": (
                "(16*pi/50)*(N_max*sqrt(h)_max/(N_min*sqrt(h)_min*lambda_min))*"
                "(G_star*rho_trace_max*R^2/c^2)"
            ),
            "geometry_ratio": str(geometry_ratio),
            "pi_upper_used": "22/7",
            "eta_strict_upper": str(eta_upper),
            "one_minus_eta_lower": str(coercive_margin),
            "eta_below_one": True,
        },
        "nonlinear_static_uniqueness": {
            "result": "chi=0_is_the_only_static_D1,2_solution_in_the_entire_source_class",
            "uses_linearization_only": False,
            "allows_arbitrary_source_shape_inside_B_R": True,
            "allows_self_consistent_metric_and_trace": (
                "yes_if_the_declared_intervals_hold_for_that_solution"
            ),
        },
        "linear_scalar_stability": {
            "linear_operator": (
                "-D_i(N*D^i)-(4*pi*G_star/(50*c^2))*N*tau"
            ),
            "negative_eigenvalue": "excluded",
            "D1,2_zero_mode": "excluded",
            "tachyonic_scalarization_mode": "excluded_on_the_source_class",
            "metric_and_material_perturbation_decoupling": "pass_because_alpha_0=0",
            "scope": (
                "scalar sector on a static source; not nonlinear dynamical stability of the "
                "coupled material body"
            ),
        },
        "status": "pass_source_class_nonlinear_static_uniqueness_and_linear_scalar_stability",
    }
    return {**body, "content_sha256": _sha(body)}


def _mass_radius_only_negative_control() -> dict[str, Any]:
    epsilon, mass = sp.symbols("epsilon M", positive=True)
    core_density = 3 * mass / (4 * sp.pi * epsilon**3)
    newton_trace_potential = sp.factor(2 * sp.pi * core_density * epsilon**2)
    expected = 3 * mass / (2 * epsilon)
    if sp.factor(newton_trace_potential - expected) != 0:
        raise ValueError("mass-radius concentration negative control failed")
    body = {
        "family": "fixed_total_trace_mass_M_in_uniform_core_of_radius_epsilon_inside_fixed_R",
        "core_trace_density": str(core_density),
        "center_integral_tau_over_distance": str(newton_trace_potential),
        "epsilon_limit": "infinity_as_epsilon->0",
        "lesson": (
            "total mass and outer radius alone cannot bound the scalar Birman-Schwinger/Kato "
            "norm; a trace-density, concentration, or resolved material-profile bound is necessary"
        ),
        "candidate_rejection": False,
    }
    return {**body, "content_sha256": _sha(body)}


def _minimal_real_source_instantiation_contract() -> dict[str, Any]:
    body = {
        "required_registered_facts": [
            {
                "id": "source_support_radius_upper",
                "quantity": "R_max in a declared Solar-system length standard",
                "purpose": "bind support inside B_R without redshift distance",
            },
            {
                "id": "total_mass_and_compactness",
                "quantity": "M_upper and 2*G_star*M_upper/(R_min*c^2)",
                "purpose": "support an independently audited weak static metric/lapse solution",
            },
            {
                "id": "trace_density_or_concentration_upper",
                "quantity": "rho_trace_max or a stronger resolved bound on (-T_E)/c^2",
                "purpose": "mass and radius alone are insufficient; instantiate rho_trace_max*R^2",
            },
            {
                "id": "pressure_trace_sign",
                "quantity": "0<=epsilon-sum_i(p_i)<=rho_trace_max*c^2",
                "purpose": "establish tau>=0 and the pointwise trace upper bound",
            },
            {
                "id": "static_geometry_intervals",
                "quantity": "N_min,N_max,lambda_min,sqrt(h)_min,sqrt(h)_max",
                "purpose": "instantiate the exact elliptic/Hardy coercivity margin",
            },
            {
                "id": "scalar_boundary_and_topology",
                "quantity": "chi_infinity=0, one end, no inner boundary, D1,2 falloff",
                "purpose": "remove scalar and inner-boundary flux terms",
            },
        ],
        "facts_not_sufficient_by_themselves": [
            "total mass plus visible radius without a concentration bound",
            "PPN gamma/beta fitted under another gravity model",
            "a GR ephemeris residual",
            "an unregistered interior solar model",
        ],
        "current_registration_status": "missing_no_real_source_facts_opened",
        "automatic_pass_rule": (
            "a future hash-bound source certificate passes the physics-side branch gate if all "
            "listed facts imply the exact source-class intervals and eta<1"
        ),
    }
    return {**body, "content_sha256": _sha(body)}


def build_g4_source_class_scalar_uniqueness_audit(
    config: dict[str, Any], root: str | Path
) -> dict[str, Any]:
    root = Path(root).resolve()
    if config.get("data_eligibility") != ELIGIBILITY:
        raise ValueError("campaign eligibility is not fail-closed")
    _validate_source_class(config["source_class"])
    predecessor = _load_bound(root, config["predecessor"])
    records = [
        item
        for item in predecessor["candidate_records"]
        if item.get("seed_id") == config["target"]["seed_id"]
    ]
    if len(records) != 1:
        raise ValueError("G4 predecessor target is not unique")
    _validate_target(records[0], config["target"])
    coupling = _global_coupling_lipschitz_certificate()
    coercivity = _source_class_coercivity_certificate(config["source_class"])
    negative = _mass_radius_only_negative_control()
    facts = _minimal_real_source_instantiation_contract()
    provenance_body = {
        "predecessor_content_sha256": config["predecessor"]["content_sha256"],
        "predecessor_provenance_sha256": config["target"][
            "predecessor_provenance_sha256"
        ],
        "action_sha256": config["target"]["action_sha256"],
        "source_class_sha256": _sha(config["source_class"]),
        "global_coupling_sha256": coupling["content_sha256"],
        "coercivity_sha256": coercivity["content_sha256"],
        "mass_radius_negative_sha256": negative["content_sha256"],
        "instantiation_contract_sha256": facts["content_sha256"],
        "data_eligibility": ELIGIBILITY,
    }
    record = {
        "seed_id": config["target"]["seed_id"],
        "action_sha256": config["target"]["action_sha256"],
        "decision": "blocked",
        "source_class_theorem_decision": "pass",
        "global_coupling_lipschitz_certificate": coupling,
        "source_class_coercivity_certificate": coercivity,
        "mass_radius_only_negative_control": negative,
        "minimal_real_source_instantiation_contract": facts,
        "gate_ledger": {
            "exact_candidate_and_predecessor": {"status": "pass"},
            "global_nonlinear_coupling_bound": {"status": "pass"},
            "source_class_static_nonlinear_uniqueness": {"status": "pass"},
            "source_class_linear_scalar_stability": {"status": "pass"},
            "mass_radius_sufficiency": {"status": "reject"},
            "registered_real_Sun_instantiation": {"status": "blocked"},
        },
        "physics_side_real_source_blocker": {
            "theorem_side": "closed_for_every_source_satisfying_the_explicit_class",
            "real_Sun_instantiation": "blocked_until_registered_facts_instantiate_the_class",
        },
        "first_missing_premise": "registered_real_source_interval_certificate",
        "candidate_rejection_found": False,
        "provenance": {**provenance_body, "binding_sha256": _sha(provenance_body)},
        "real_solar_bundle_admissible": False,
    }
    body = {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": config["campaign_id"],
        "source_bindings": {"predecessor": config["predecessor"]},
        "target_seed_count": 1,
        "decision_counts": {"blocked": 1},
        "gate_status_counts": {"pass": 4, "reject": 1, "blocked": 1},
        "source_class_theorem_pass_count": 1,
        "real_source_instantiation_pass_count": 0,
        "candidate_records": [record],
        "observational_data_opened": False,
        "paid_llm_spend_usd": 0.0,
        "data_eligibility": dict(ELIGIBILITY),
        "interpretation": (
            "A global candidate-specific coupling bound and an explicit Hardy coercivity margin "
            "prove nonlinear static scalar-branch uniqueness and exclude linear tachyonic scalar "
            "modes for a broad nonspherical weak compact source class. This closes the theorem-side "
            "branch blocker conditionally. It does not claim the Sun belongs to the class: a "
            "registered radius, trace-density/concentration, pressure-sign, mass/geometry, and "
            "boundary certificate is still required, and no observational data was opened."
        ),
    }
    return {**body, "content_sha256": _sha(body)}
