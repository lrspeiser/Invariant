from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from pathlib import Path
from typing import Any

import sympy as sp

from .promotion_orchestrator import ELIGIBILITY

SCHEMA_VERSION = "sigma-g3-full-lapse-dirac-operator-audit-1.0"


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
    if (
        record["principal_common_cone_certificate"].get("content_sha256")
        != target["principal_certificate_sha256"]
    ):
        raise ValueError("G3 principal certificate mismatch")
    if record["lapse_prerequisite"].get("content_sha256") != target[
        "prior_lapse_certificate_sha256"
    ]:
        raise ValueError("G3 prior lapse certificate mismatch")
    if record.get("first_missing_premise") != "candidate_specific_full_Delta_N_operator":
        raise ValueError("G3 predecessor lapse blocker changed")


def _validate_domain(domain: dict[str, Any]) -> None:
    required = {
        "contract_kind": "global_periodic_lapse_operator_domain_not_asymptotically_flat_energy_domain",
        "spatial_manifold": "connected_three_torus_with_periodic_coordinate_boundaries",
        "regularity": {
            "sobolev_index": "s>5/2",
            "canonical_fields": "H^s_periodic",
            "canonical_momenta": "H^{s-1}_periodic",
            "lapse_multiplier_domain": "L2_periodic",
            "lapse_multiplier_codomain": "L2_periodic",
        },
        "pointwise_cell": "the predecessor componentwise domain holds almost everywhere",
        "temporal_endpoint_variations": "zero",
        "spatial_boundary_terms": "zero_by_periodicity",
        "unitary_gauge": "phi=t on the future-timelike-gradient branch",
        "nonempty_witness": "constant center fields with N=1, K_ij=0, and vanishing Hessian",
    }
    if domain != required:
        raise ValueError("G3 lapse function-space or boundary domain changed")


def _derive_full_delta(beta_value: Fraction) -> dict[str, Any]:
    lapse = sp.Symbol("N", positive=True, finite=True)
    beta = sp.Symbol("beta", real=True)
    trace_momentum = sp.Symbol("q", real=True)
    trace_curvature = sp.Symbol("K", real=True)
    dewitt_momentum = sp.Symbol("D_pi", real=True)
    spatial_ricci = sp.Symbol("R3", real=True)
    a_star = 1 / lapse
    g3 = beta * a_star**2 / 2
    primitive = beta * a_star**3 / 6
    cubic_adm_coefficient = sp.factor(g3 - primitive / a_star)
    momentum_shift = beta / (6 * lapse**3)
    hamiltonian = sp.factor(
        2 * lapse * dewitt_momentum
        + beta * trace_momentum / (3 * lapse**2)
        - beta**2 / (12 * lapse**5)
        - lapse * spatial_ricci / 2
        - 1 / (2 * lapse)
    )
    delta_fixed_momentum = sp.factor(-sp.diff(hamiltonian, lapse, 2))
    expected_fixed_momentum = sp.factor(
        lapse**-3
        - 2 * beta * trace_momentum * lapse**-4
        + sp.Rational(5, 2) * beta**2 * lapse**-7
    )
    momentum_curvature_relation = -trace_curvature + beta / (2 * lapse**3)
    delta_curvature = sp.factor(
        delta_fixed_momentum.subs(trace_momentum, momentum_curvature_relation)
    )
    expected_curvature = sp.factor(
        lapse**-3
        + 2 * beta * trace_curvature * lapse**-4
        + sp.Rational(3, 2) * beta**2 * lapse**-7
    )
    if (
        cubic_adm_coefficient != beta / (3 * lapse**2)
        or sp.factor(delta_fixed_momentum - expected_fixed_momentum) != 0
        or sp.factor(delta_curvature - expected_curvature) != 0
    ):
        raise ValueError("G3 full lapse derivation residual is nonzero")
    body = {
        "covariant_density": "sqrt(-g)[R/2+X-beta*X*box(phi)]",
        "unitary_dictionary": {
            "phi": "t",
            "A_star": "1/N",
            "X": "A_star^2/2",
            "box_phi": "-n(A_star)-A_star*K",
        },
        "cubic_integration_by_parts": {
            "primitive_dF_dA_star_equals_G3": str(primitive),
            "periodic_spatial_and_fixed_time_endpoint_boundary": "zero",
            "reduced_ADM_density_over_sqrt_h": str(cubic_adm_coefficient * sp.Symbol("K")),
            "spatial_lapse_derivative_terms": "none",
        },
        "canonical_reduction": {
            "momentum_shift_per_metric_component": str(momentum_shift),
            "trace_momentum_relation": "q=pi/sqrt(h)=-K+beta/(2*N^3)",
            "Hamiltonian_density_over_sqrt_h": str(hamiltonian),
            "differentiation_contract": "Delta_N=-d^2 H/dN^2 at fixed canonical momentum",
        },
        "full_Delta_N": {
            "fixed_momentum": str(delta_fixed_momentum),
            "on_velocity_cell": str(delta_curvature),
            "differential_order": 0,
            "operator_type": "real_multiplication_operator",
            "beta_specialization": str(beta_value),
        },
        "exact_residuals": {
            "fixed_momentum_Delta": "0",
            "momentum_to_curvature_Delta": "0",
        },
    }
    return {**body, "content_sha256": _sha(body)}


def _coercivity_certificate(
    predecessor_record: dict[str, Any], beta: Fraction, operator_domain: dict[str, Any]
) -> dict[str, Any]:
    principal_domain = predecessor_record["componentwise_domain"]
    if (
        principal_domain["symmetric_hessian_component_abs"] != "1/100"
        or principal_domain["spatial_gradient_component_abs"] != "1/200"
        or principal_domain["lapse_interval"] != ["99/100", "101/100"]
    ):
        raise ValueError("G3 certified componentwise bounds changed")
    hessian_abs = Fraction(1, 100)
    scalar_norm_lower = Fraction(49, 50)
    scalar_norm_upper = Fraction(51, 50)
    gradient_l1_upper = Fraction(207, 200)
    raw_k_bound = (
        4 * hessian_abs / scalar_norm_lower
        + hessian_abs * gradient_l1_upper**2 / scalar_norm_lower**3
    )
    chosen_k_bound = Fraction(3, 50)
    if raw_k_bound >= chosen_k_bound:
        raise ValueError("derived unitary trace-curvature bound exceeds chosen envelope")
    # The Dirac calculation uses the scalar's unitary lapse N_phi=1/sqrt(2X),
    # not the predecessor BSSN gauge lapse.  The gradient box gives
    # 49/50 <= sqrt(2X) <= 51/50, hence the following conservative reciprocal box.
    lapse_min = 1 / scalar_norm_upper
    lapse_max = 1 / scalar_norm_lower
    monotonicity_gap = 3 * lapse_min - 8 * beta * chosen_k_bound
    delta_lower = (
        lapse_max**-3
        - 2 * beta * chosen_k_bound * lapse_max**-4
        + Fraction(3, 2) * beta**2 * lapse_max**-7
    )
    delta_upper = (
        lapse_min**-3
        + 2 * beta * chosen_k_bound * lapse_min**-4
        + Fraction(3, 2) * beta**2 * lapse_min**-7
    )
    if monotonicity_gap <= 0 or delta_lower <= 0:
        raise ValueError("G3 lapse coercivity bound failed")
    body = {
        "unitary_foliation_trace_bound": {
            "identity": "K_phi=-box(phi)/sqrt(2X)-p^mu*p^nu*phi_munu/(2X)^(3/2)",
            "certified_subbounds": {
                "sqrt_2X_lower": str(scalar_norm_lower),
                "sqrt_2X_upper": str(scalar_norm_upper),
                "sum_abs_p_upper": str(gradient_l1_upper),
                "each_hessian_component_abs": str(hessian_abs),
            },
            "raw_abs_K_upper": str(raw_k_bound),
            "chosen_abs_K_envelope": str(chosen_k_bound),
        },
        "Delta_N_lower_bound": {
            "unitary_lapse_interval": [str(lapse_min), str(lapse_max)],
            "BSSN_lapse_not_identified_with_unitary_lapse": True,
            "monotonicity": "Delta_N increases with K and decreases with N on the bound box",
            "N_derivative_sufficient_gap": str(monotonicity_gap),
            "attained_bound_endpoint": "N=50/49,K=-3/50",
            "exact": str(delta_lower),
            "decimal": float(delta_lower),
            "strictly_positive": True,
            "upper_at_N_50_over_51_K_3_over_50": str(delta_upper),
        },
        "function_space_result": {
            "domain_contract_sha256": _sha(operator_domain),
            "operator": "M_Delta:L2(T3)->L2(T3)",
            "operator_norm_upper": str(delta_upper),
            "coercive_inequality": f"<f,M_Delta f> >= ({delta_lower})*||f||_L2^2",
            "kernel": "{0}",
            "inverse_norm_upper": str(1 / delta_lower),
            "boundary_zero_modes": "excluded_by_strict_pointwise_lower_bound",
            "status": "pass",
        },
    }
    return {**body, "content_sha256": _sha(body)}


def build_g3_full_lapse_dirac_operator_audit(
    config: dict[str, Any], root: str | Path
) -> dict[str, Any]:
    root = Path(root).resolve()
    if config.get("data_eligibility") != ELIGIBILITY:
        raise ValueError("campaign eligibility is not fail-closed")
    _validate_domain(config["operator_domain"])
    predecessor = _load_bound(root, config["predecessor"])
    target = config["target"]
    record = next(item for item in predecessor["candidate_records"] if item["seed_id"] == target["seed_id"])
    _validate_target(record, target)
    beta = Fraction(target["beta"])
    if (
        beta != Fraction(1, 100)
        or target["G2"] != "X"
        or target["G3"] != "beta*X"
        or target["G4"] != "1/2"
    ):
        raise ValueError("unsupported G3 lapse specialization")
    derivation = _derive_full_delta(beta)
    coercivity = _coercivity_certificate(record, beta, config["operator_domain"])
    gates = {
        "typed_action_and_predecessor": {"status": "pass"},
        "full_candidate_Delta_N_derivation": {"status": "pass"},
        "bound_function_space_and_boundary_domain": {"status": "pass"},
        "Delta_N_coercivity_and_zero_mode_exclusion": {"status": "pass"},
        "distributed_Dirac_on_periodic_cell": {"status": "pass"},
        "principal_and_common_cone_predecessor": {"status": "pass"},
        "asymptotically_flat_extension": {
            "status": "blocked",
            "reason": (
                "the certified X>=0.488 cell has nondecaying canonical G2=X stress and cannot "
                "serve as an asymptotically flat finite-ADM-energy end"
            ),
        },
        "global_hamiltonian_energy": {"status": "blocked"},
        "formal_prerequisite_completion": {"status": "blocked"},
    }
    provenance_body = {
        "predecessor_content_sha256": config["predecessor"]["content_sha256"],
        "action_sha256": target["action_sha256"],
        "predecessor_provenance_sha256": target["predecessor_provenance_sha256"],
        "principal_certificate_sha256": target["principal_certificate_sha256"],
        "prior_lapse_certificate_sha256": target["prior_lapse_certificate_sha256"],
        "operator_domain_sha256": _sha(config["operator_domain"]),
        "derivation_sha256": derivation["content_sha256"],
        "coercivity_sha256": coercivity["content_sha256"],
        "data_eligibility": ELIGIBILITY,
    }
    candidate = {
        "seed_id": target["seed_id"],
        "action_sha256": target["action_sha256"],
        "decision": "blocked",
        "operator_domain": config["operator_domain"],
        "full_lapse_operator_derivation": derivation,
        "coercivity_certificate": coercivity,
        "gate_ledger": gates,
        "resolved_predecessor_blocker": "candidate_specific_full_Delta_N_operator",
        "first_missing_premise": "asymptotically_flat_or_global_energy_domain",
        "necessary_condition_rejection_found": False,
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
        "full_lapse_dirac_pass_count": 1,
        "full_formal_pass_count": 0,
        "solar_bundle_count": 0,
        "observational_data_opened": False,
        "paid_llm_spend_usd": 0.0,
        "data_eligibility": dict(ELIGIBILITY),
        "interpretation": (
            "For the exact cubic seed the unitary-gauge G3 term reduces to beta*K/(3N^2) "
            "after bound total derivatives. Its fixed-momentum Legendre transform yields a "
            "zeroth-order full Delta_N with a strict positive lower bound on the predecessor "
            "box, proving periodic L2 invertibility and closing the conditional Dirac gate. "
            "The positive-X box is not an asymptotically flat energy domain, so global energy "
            "and promotion remain blocked."
        ),
    }
    return {**body, "content_sha256": _sha(body)}
