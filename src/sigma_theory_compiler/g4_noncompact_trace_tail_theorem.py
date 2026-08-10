from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from pathlib import Path
from typing import Any

from .promotion_orchestrator import ELIGIBILITY

SCHEMA_VERSION = "sigma-g4-noncompact-trace-tail-theorem-1.0"


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
    if (
        value.get("content_sha256") != descriptor["content_sha256"]
        or _sha(body) != descriptor["content_sha256"]
    ):
        raise ValueError(f"bound content hash mismatch: {descriptor['path']}")
    return value


def _validate_predecessor(value: dict[str, Any], descriptor: dict[str, Any]) -> None:
    if (
        value.get("schema_version")
        != "sigma-g4-real-sun-interval-source-audit-1.0"
        or value.get("decision") != "blocked"
        or value.get("first_missing_premise")
        != "registered_finite_trace_support_or_resolved_exterior_tail_Kato_bound"
        or value.get("observational_authorization") is not False
        or value.get("tracking_target_values_opened") is not False
    ):
        raise ValueError("real-Sun predecessor decision or seals changed")
    if value["provenance"].get("binding_sha256") != descriptor[
        "provenance_binding_sha256"
    ]:
        raise ValueError("real-Sun predecessor provenance mismatch")
    if value["authoritative_fact_registry"].get("content_sha256") != descriptor[
        "fact_registry_sha256"
    ]:
        raise ValueError("real-Sun predecessor fact registry mismatch")


def _validate_tail_class(tail_class: dict[str, Any]) -> None:
    expected = {
        "role": "candidate_specific_conditional_tail_theorem_not_real_Sun_registration",
        "coordinate_domain": {
            "space": "R3_one_asymptotically_Euclidean_end",
            "radius": "r=Euclidean_coordinate_distance_from_registered_source_center",
            "inner_boundary": "none",
        },
        "trace_envelope": {
            "trace": "tau=(-T_E)/c^2>=0",
            "interior": "tau(x)<=rho_in_for_r<=R",
            "exterior": "tau(x)<=rho_tail*(R/r)^p_for_r>R",
            "angular_dependence": "arbitrary_below_pointwise_envelope",
            "p_lower": "4",
        },
        "dimensionless_bounds": {
            "G_star*rho_in*R^2/c^2": "<=1/1000",
            "G_star*rho_tail*R^2/c^2": "<=1/1000",
        },
        "geometry": {
            "ordinary_lapse_interval": ["99/100", "101/100"],
            "inverse_metric_ellipticity_lower": "99/100",
            "coordinate_volume_density_interval": ["99/100", "101/100"],
        },
        "scalar": {
            "function_space": "D^{1,2}(R3)",
            "boundary": "chi->0_at_spatial_infinity",
            "static": True,
        },
    }
    if tail_class != expected:
        raise ValueError("noncompact trace-tail class changed")


def _hardy_tail_certificate(tail_class: dict[str, Any]) -> dict[str, Any]:
    geometry = tail_class["geometry"]
    lapse_lower, lapse_upper = map(Fraction, geometry["ordinary_lapse_interval"])
    volume_lower, volume_upper = map(
        Fraction, geometry["coordinate_volume_density_interval"]
    )
    ellipticity_lower = Fraction(geometry["inverse_metric_ellipticity_lower"])
    geometry_ratio = lapse_upper * volume_upper / (
        lapse_lower * volume_lower * ellipticity_lower
    )
    interior_strength = Fraction(1, 1000)
    exterior_strength = Fraction(1, 1000)
    pi_upper = Fraction(22, 7)
    prefactor = Fraction(16, 50) * pi_upper * geometry_ratio
    eta_in = prefactor * interior_strength
    eta_out = prefactor * exterior_strength
    eta_total = eta_in + eta_out
    margin = 1 - eta_total
    if (
        geometry_ratio != Fraction(1_020_100, 970_299)
        or eta_in != Fraction(81_608, 77_182_875)
        or eta_total != Fraction(163_216, 77_182_875)
        or margin != Fraction(77_019_659, 77_182_875)
        or not 0 < eta_total < 1
    ):
        raise ValueError("noncompact Hardy margin arithmetic failed")
    body = {
        "tail_implication": (
            "p>=2_and_r>R_implies_tau<=rho_tail*R^2/r^2_even_with_angular_structure"
        ),
        "interior_form_bound": (
            "integral_r<=R tau*chi^2<=4*rho_in*R^2*integral|grad_delta_chi|^2"
        ),
        "exterior_form_bound": (
            "integral_r>R tau*chi^2<=4*rho_tail*R^2*integral|grad_delta_chi|^2"
        ),
        "combined_relative_bound": {
            "formula": (
                "eta_H=(16*pi/50)*geometry_ratio*(G_star*R^2/c^2)*"
                "(rho_in+rho_tail)"
            ),
            "geometry_ratio": str(geometry_ratio),
            "pi_upper_used": "22/7",
            "eta_interior_upper": str(eta_in),
            "eta_exterior_upper": str(eta_out),
            "eta_total_upper": str(eta_total),
            "coercive_margin_lower": str(margin),
            "strictly_below_one": True,
        },
        "scope": (
            "pointwise anisotropic tails; scalar static uniqueness and scalar linear-mode "
            "coercivity on every self-consistent geometry satisfying the registered intervals"
        ),
        "status": "pass_conditional_noncompact_tail_class",
    }
    return {**body, "content_sha256": _sha(body)}


def _kato_tail_certificate() -> dict[str, Any]:
    u = Fraction(1, 1000)
    v = Fraction(1, 1000)
    p_lower = Fraction(4)
    pi_upper = Fraction(22, 7)
    kappa_in = Fraction(2, 50) * pi_upper * u
    kappa_out = Fraction(4, 50) * pi_upper * v / (p_lower - 2)
    kappa_flat = kappa_in + kappa_out
    margin_flat = 1 - kappa_flat
    green_constant = Fraction(101, 99)
    kappa_dominated = green_constant * kappa_flat
    margin_dominated = 1 - kappa_dominated
    if (
        kappa_in != Fraction(11, 87_500)
        or kappa_out != Fraction(11, 87_500)
        or kappa_flat != Fraction(11, 43_750)
        or margin_flat != Fraction(43_739, 43_750)
        or kappa_dominated != Fraction(1_111, 4_331_250)
        or margin_dominated != Fraction(4_330_139, 4_331_250)
    ):
        raise ValueError("noncompact Kato margin arithmetic failed")
    body = {
        "envelope_convolution_bound": {
            "interior": "sup_x integral_r<=R rho_in/|x-y| d3y<=2*pi*rho_in*R^2",
            "exterior": (
                "sup_x integral_r>R rho_tail*(R/r)^p/|x-y| d3y"
                "<=4*pi*rho_tail*R^2/(p-2), p>2"
            ),
            "anisotropic_extension": (
                "pointwise domination by the radial nonnegative envelope plus the Newton shell "
                "bound implies the same supremum bound"
            ),
        },
        "Birman_Schwinger_bound": {
            "flat_formula": (
                "kappa<=(1/50)*(2*pi*u+4*pi*v/(p-2)); "
                "u=G_star*rho_in*R^2/c^2, v=G_star*rho_tail*R^2/c^2"
            ),
            "kappa_interior_upper": str(kappa_in),
            "kappa_exterior_upper": str(kappa_out),
            "kappa_flat_upper": str(kappa_flat),
            "flat_coercive_margin_lower": str(margin_flat),
        },
        "general_geometry_route": {
            "extra_registered_premise": (
                "G_L0(x,y)*N(y)*sqrt(h(y))<=C_G/(4*pi*|x-y|)"
            ),
            "C_G_upper_control": str(green_constant),
            "kappa_upper": str(kappa_dominated),
            "coercive_margin_lower": str(margin_dominated),
            "status": "conditional_on_global_Green_kernel_domination",
        },
        "integrability_thresholds": {
            "Kato_trace_tail": "p>2",
            "finite_total_trace_mass_necessary_for_AF": "p>3",
            "registered_class_uses": "p>=4",
            "warning": (
                "a steady untruncated r^-2 wind can satisfy a Hardy form envelope when small "
                "but has divergent total trace mass; p>3 makes the trace envelope integrable "
                "but does not by itself certify total stress-energy falloff or an AF solution"
            ),
        },
        "status": "pass_exact_sufficient_conditions_not_real_source_instantiation",
    }
    return {**body, "content_sha256": _sha(body)}


def _minimal_tail_fact_contract() -> dict[str, Any]:
    facts = [
        {
            "id": "registered_reference_radius_and_center",
            "class": "calibrated",
            "quantity": "R interval and source-centered coordinate transform with uncertainty",
        },
        {
            "id": "interior_trace_density_upper",
            "class": "calibrated_or_model_dependent_explicitly_labeled",
            "quantity": "global rho_in upper or resolved interior Kato integral with covariance",
        },
        {
            "id": "exterior_trace_amplitude_upper",
            "class": "calibrated",
            "quantity": "rho_tail upper at R for tau=(-T_E)/c^2, not number density alone",
        },
        {
            "id": "exterior_decay_exponent_lower",
            "class": "calibrated",
            "quantity": "p_min>3 over every exterior shell, or a piecewise integral envelope",
        },
        {
            "id": "composition_pressure_trace_transform",
            "class": "calibrated",
            "quantity": (
                "species/composition, rest-energy and anisotropic pressure transform proving "
                "0<=tau and propagating uncertainty"
            ),
        },
        {
            "id": "angular_or_resolved_tail_coverage",
            "class": "calibrated",
            "quantity": (
                "all-angle pointwise envelope or a resolved three-dimensional Kato integral; "
                "single-line samples are insufficient"
            ),
        },
        {
            "id": "outer_transition_or_cutoff",
            "class": "calibrated",
            "quantity": (
                "heliospheric transition/cutoff or faster-decay certificate excluding an "
                "unbounded steady r^-2 mass tail"
            ),
        },
        {
            "id": "geometry_and_boundary_domain",
            "class": "model_dependent_explicitly_labeled",
            "quantity": (
                "candidate-independent or jointly solved lapse/metric intervals, one AF end, "
                "D1,2 scalar falloff, and zero boundary flux"
            ),
        },
    ]
    body = {
        "required_registered_facts": facts,
        "allowed_routes": [
            "pointwise anisotropic Hardy envelope",
            "radial or pointwise-dominated Kato envelope",
            "resolved profile Birman-Schwinger integral with a registered Green-kernel bound",
        ],
        "insufficient_controls": [
            "photospheric radius alone",
            "one-point wind density or speed",
            "number density without composition and pressure",
            "a steady r^-2 fit extrapolated to infinity",
            "standard-solar-model profile silently treated as raw fact",
            "GR-fitted residual or PPN parameter",
        ],
        "current_status": "missing_no_real_Sun_tail_profile_opened",
    }
    return {**body, "content_sha256": _sha(body)}


def build_g4_noncompact_trace_tail_theorem(
    config: dict[str, Any], root: str | Path
) -> dict[str, Any]:
    root = Path(root).resolve()
    if config.get("data_eligibility") != ELIGIBILITY:
        raise ValueError("campaign eligibility is not fail-closed")
    if config.get("observational_authorization") is not False:
        raise ValueError("observational authorization must remain false")
    _validate_tail_class(config["tail_class"])
    predecessor = _load_bound(root, config["predecessor"])
    _validate_predecessor(predecessor, config["predecessor"])

    hardy = _hardy_tail_certificate(config["tail_class"])
    kato = _kato_tail_certificate()
    facts = _minimal_tail_fact_contract()
    tail_class_sha = _sha(config["tail_class"])
    provenance_body = {
        "predecessor_content_sha256": config["predecessor"]["content_sha256"],
        "predecessor_provenance_sha256": config["predecessor"][
            "provenance_binding_sha256"
        ],
        "predecessor_fact_registry_sha256": config["predecessor"][
            "fact_registry_sha256"
        ],
        "tail_class_sha256": tail_class_sha,
        "hardy_certificate_sha256": hardy["content_sha256"],
        "kato_certificate_sha256": kato["content_sha256"],
        "tail_fact_contract_sha256": facts["content_sha256"],
        "data_eligibility": ELIGIBILITY,
    }
    record = {
        "theorem_decision": "pass",
        "real_Sun_instantiation_decision": "blocked",
        "overall_decision": "blocked",
        "tail_class_sha256": tail_class_sha,
        "hardy_anisotropic_tail_certificate": hardy,
        "kato_birman_schwinger_tail_certificate": kato,
        "minimal_real_source_tail_fact_contract": facts,
        "gate_ledger": {
            "exact_predecessor_and_seals": {"status": "pass"},
            "anisotropic_noncompact_Hardy_coercivity": {"status": "pass"},
            "radial_envelope_Kato_coercivity": {"status": "pass"},
            "finite_trace_mass_on_p_at_least_4_class": {"status": "pass"},
            "registered_real_Sun_tail_instantiation": {"status": "blocked"},
        },
        "first_missing_premise": "registered_trace_tail_amplitude_decay_and_outer_transition",
        "candidate_rejection_found": False,
        "real_solar_bundle_admissible": False,
        "provenance": {**provenance_body, "binding_sha256": _sha(provenance_body)},
    }
    body = {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": config["campaign_id"],
        "source_bindings": {"predecessor": config["predecessor"]},
        "tail_class": config["tail_class"],
        "theorem_pass_count": 1,
        "real_source_instantiation_pass_count": 0,
        "decision_counts": {"blocked": 1},
        "gate_status_counts": {"pass": 4, "blocked": 1},
        "candidate_records": [record],
        "observational_authorization": False,
        "observational_data_opened": False,
        "tracking_target_values_opened": False,
        "paid_llm_spend_usd": 0.0,
        "data_eligibility": dict(ELIGIBILITY),
        "interpretation": (
            "The G4 scalar uniqueness/coercivity theorem now covers a noncompact, angularly "
            "anisotropic trace tail dominated by rho_tail*(R/r)^p with p>=4. Exact Hardy and "
            "Kato sufficient conditions separate interior and exterior contributions and retain "
            "large positive margins. This is a conditional theorem, not evidence that the real "
            "Sun satisfies the envelope: no Solar tail values or tracking targets were opened."
        ),
    }
    return {**body, "content_sha256": _sha(body)}
