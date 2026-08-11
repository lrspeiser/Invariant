from __future__ import annotations

import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path
from typing import Any

import sympy as sp

from .promotion_orchestrator import ELIGIBILITY

ARTIFACT_SCHEMA = "sigma-future-g3-york-analytic-mean-curvature-threshold-campaign-1.0"
FIRST_BLOCKER = (
    "candidate_specific_nontrivial_AF_Einstein_constraint_solution_beyond_"
    "analytic_conformally_flat_York_threshold"
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


def _validate_source(root: Path, config: dict[str, Any]) -> None:
    source = config["adapter_source"]
    if _file_sha(root / source["path"]) != source["file_sha256"]:
        raise ValueError("campaign source hash mismatch")


def _validate_threshold_contract(contract: dict[str, Any]) -> None:
    expected = {
        "retained_geometry_class": "h_ij=psi(x)^4*delta_ij_nonradial_psi_allowed",
        "retained_York_tensor_scope": (
            "arbitrary_smooth_nonradial_TT_or_longitudinal_or_mixed_A_ij"
        ),
        "retained_scalar_profile": "Pi=v(r)=1/sqrt(1+(r/L)^4)",
        "transition_length_L": "100",
        "source_factor": "c_beta(kappa)=1-2*beta*kappa-(2/3)*kappa^2",
        "Green_coefficient": "B_beta(kappa)=L^2*c_beta(kappa)/96",
        "universal_maximum": "256/3125",
        "threshold_definition": ("unique_positive_root_of_B_beta(kappa)=256/3125"),
        "endpoint_semantics": ("excluded_by_strict_interior_ball_and_Newton_kernel_inequality"),
        "above_threshold_semantics": ("comparison_inconclusive_not_AF_solution_or_action_pass"),
        "AF_conformal_factor_class": ("C2_positive_psi_minus_1_equals_O2_r_minus_1"),
        "no_momentum_solution_assumed": True,
        "no_observation_or_numerical_solver_used": True,
    }
    if contract != expected:
        raise ValueError("York analytic threshold contract changed")


def _primitive_threshold_polynomial(beta: Fraction) -> list[int]:
    kappa = sp.Symbol("kappa")
    c_star = sp.Rational(1536, 1953125)
    polynomial = sp.Poly(
        kappa**2
        + 3 * sp.Rational(beta.numerator, beta.denominator) * kappa
        + sp.Rational(3, 2) * (c_star - 1),
        kappa,
    )
    _, integer_polynomial = polynomial.clear_denoms(convert=True)
    _, primitive = integer_polynomial.primitive()
    return [int(value) for value in primitive.all_coeffs()]


def _candidate_threshold(
    prior: dict[str, Any], target: dict[str, Any], contract: dict[str, Any]
) -> dict[str, Any]:
    beta = Fraction(target["beta"])
    length = Fraction(contract["transition_length_L"])
    universal_maximum = Fraction(256, 3125)
    c_star = universal_maximum * 96 / length**2
    radicand = 9 * beta**2 + 6 * (1 - c_star)
    coefficients = _primitive_threshold_polynomial(beta)
    lower = Fraction(target["isolating_lower"])
    upper = Fraction(target["isolating_upper"])
    above = Fraction(target["above_threshold_control"])

    def source_factor(cap: Fraction) -> Fraction:
        return 1 - 2 * beta * cap - Fraction(2, 3) * cap**2

    def green_excess(cap: Fraction) -> Fraction:
        return source_factor(cap) * length**2 / 96 - universal_maximum

    lower_excess = green_excess(lower)
    upper_excess = green_excess(upper)
    above_excess = green_excess(above)
    if (
        target["radicand"] != str(radicand)
        or target["primitive_polynomial_coefficients"] != coefficients
        or lower < 0
        or lower >= upper
        or above < upper
        or lower_excess <= 0
        or upper_excess >= 0
        or above_excess >= 0
    ):
        raise ValueError("candidate analytic threshold certificate changed")
    if math.isqrt(radicand.numerator) ** 2 == radicand.numerator:
        raise ValueError("threshold radicand unexpectedly became a rational square")
    beta_sp = sp.Rational(beta.numerator, beta.denominator)
    c_star_sp = sp.Rational(c_star.numerator, c_star.denominator)
    root = (-3 * beta_sp + sp.sqrt(9 * beta_sp**2 + 6 * (1 - c_star_sp))) / 2
    root_expression = str(sp.simplify(root))
    if root_expression != target["root_expression"]:
        raise ValueError("candidate analytic root expression changed")
    if sp.simplify(1 - 2 * beta_sp * root - sp.Rational(2, 3) * root**2 - c_star_sp) != 0:
        raise ValueError("analytic root identity did not close")
    predecessor = prior["York_mean_curvature_frontier_certificate"]["exact_frontier"]
    if (
        prior["decision"] != "blocked"
        or prior["theory_rejected"] is not False
        or Fraction(predecessor["candidate_kappa_cap"]) != lower
        or predecessor["next_grid_cap_comparison_status"] != "inconclusive"
    ):
        raise ValueError("predecessor frontier decision changed")
    body = {
        "candidate_id": prior["candidate_id"],
        "action_sha256": prior["action_sha256"],
        "beta": str(beta),
        "predecessor_grid_frontier_sha256": prior["York_mean_curvature_frontier_certificate"][
            "content_sha256"
        ],
        "threshold_contract": contract,
        "direct_action_binding": True,
        "family_label_used_as_threshold_evidence": False,
        "exact_algebraic_threshold": {
            "kappa_star_definition": ("(-3*beta+sqrt(9*beta^2+6*(1-1536/1953125)))/2"),
            "candidate_root_expression": root_expression,
            "radicand": str(radicand),
            "radicand_is_not_a_rational_square": True,
            "primitive_integer_polynomial_coefficients_descending": coefficients,
            "primitive_integer_polynomial": (
                f"{coefficients[0]}*kappa^2+{coefficients[1]}*kappa{coefficients[2]:+d}=0"
            ),
            "isolating_interval": [str(lower), str(upper)],
            "source_factor_at_threshold": str(c_star),
            "Green_coefficient_at_threshold": str(universal_maximum),
            "source_factor_monotonicity": ("dc_beta/d_kappa=-2*beta-(4/3)*kappa<0_for_kappa>=0"),
            "unique_positive_root": True,
            "analytic_supremum_of_closed_no_go_caps": root_expression,
        },
        "endpoint_certificate": {
            "coarse_coefficient_relation": "B_beta(kappa_star)=256/3125",
            "strict_source_fact": "v(r)^2>1/2_for_|x|<L",
            "strict_kernel_fact": ("1/|x-z|>1/(2*L)_almost_everywhere_on_B_L_times_B_L"),
            "strict_endpoint_inequality": ("m-1>(256/3125)*m^5_for_m=min_B_L(psi)>=1"),
            "contradiction": ("(m-1)/m^5<=256/3125_for_all_m>=1"),
            "threshold_endpoint_excluded": True,
        },
        "above_threshold_negative_control": {
            "kappa": str(above),
            "strictly_above_kappa_star": True,
            "Green_excess": str(above_excess),
            "status": "comparison_inconclusive",
            "AF_solution_or_action_pass_inferred": False,
        },
        "decision": "reject_closed_analytic_threshold_York_class",
        "excluded_class": (
            "conformally_flat_AF_nonradial_psi_arbitrary_tracefree_York_A_ij_"
            "and_abs_K_less_than_or_equal_to_kappa_star_times_v"
        ),
        "candidate_nontrivial_AF_Einstein_constraint_solution_available": False,
        "theory_rejected": False,
        "first_remaining_blocker": FIRST_BLOCKER,
        "scope": (
            "The exact analytic root replaces the grid approximation and excludes the closed "
            "candidate-specific class |K|<=kappa_star*v, including its endpoint. Arbitrary "
            "smooth nonradial TT, longitudinal, or mixed trace-free York tensors remain in "
            "scope. The comparison is inconclusive for larger mean curvature and does not "
            "address non-conformally-flat metrics, different scalar data, or action viability."
        ),
    }
    return {**body, "content_sha256": _sha(body)}


def build_future_g3_york_analytic_mean_curvature_threshold_campaign(
    config: dict[str, Any], project_root: str | Path
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    if config.get("data_eligibility") != ELIGIBILITY:
        raise ValueError("campaign eligibility is not fail-closed")
    _validate_source(root, config)
    _validate_threshold_contract(config["threshold_contract"])
    predecessors = {
        key: _load_bound(root, descriptor) for key, descriptor in config["bindings"].items()
    }
    immediate = predecessors["predecessor"]
    york_no_go = predecessors["nonradial_York_no_go_source"]
    if immediate.get("source_bindings", {}).get("predecessor", {}).get(
        "content_sha256"
    ) != york_no_go.get("content_sha256"):
        raise ValueError("predecessor chain changed")
    records_by_id = {item["candidate_id"]: item for item in immediate["candidate_records"]}
    records = []
    for target in config["targets"]:
        prior = records_by_id.get(target["candidate_id"])
        if (
            prior is None
            or prior["action_sha256"] != target["action_sha256"]
            or prior["beta"] != target["beta"]
            or prior["content_sha256"] != target["predecessor_record_content_sha256"]
            or prior["York_mean_curvature_frontier_certificate"]["content_sha256"]
            != target["predecessor_frontier_content_sha256"]
        ):
            raise ValueError("target binding changed")
        certificate = _candidate_threshold(prior, target, config["threshold_contract"])
        provenance_body = {
            "predecessor_content_sha256": immediate["content_sha256"],
            "predecessor_record_content_sha256": prior["content_sha256"],
            "nonradial_York_no_go_source_content_sha256": york_no_go["content_sha256"],
            "action_sha256": prior["action_sha256"],
            "threshold_contract_sha256": _sha(config["threshold_contract"]),
            "threshold_certificate_sha256": certificate["content_sha256"],
            "data_eligibility": dict(ELIGIBILITY),
        }
        record_body = {
            "candidate_id": prior["candidate_id"],
            "action_sha256": prior["action_sha256"],
            "beta": prior["beta"],
            "decision": "blocked",
            "first_blocker": FIRST_BLOCKER,
            "York_analytic_threshold_certificate": certificate,
            "gate_ledger": {
                "York_grid_frontier_predecessor": {"status": "pass"},
                "exact_algebraic_threshold": {"status": "pass"},
                "closed_threshold_endpoint": {"status": "reject_ansatz_class"},
                "above_threshold_control": {"status": "inconclusive"},
                "candidate_nontrivial_AF_Einstein_constraint_solution_beyond_threshold": {
                    "status": "blocked"
                },
                "global_hamiltonian_energy": {"status": "blocked"},
                "full_formal": {"status": "blocked"},
            },
            "candidate_nontrivial_AF_Einstein_constraint_solution_available": False,
            "theory_rejected": False,
            "global_energy_pass": False,
            "full_formal_pass": False,
            "observational_data_opened": False,
            "data_eligibility": dict(ELIGIBILITY),
            "provenance": {**provenance_body, "binding_sha256": _sha(provenance_body)},
        }
        records.append({**record_body, "content_sha256": _sha(record_body)})
    if len(records) != 3:
        raise ValueError("expected exactly three candidate analytic threshold records")
    body = {
        "schema_version": ARTIFACT_SCHEMA,
        "campaign_id": config["campaign_id"],
        "campaign_adapter_source": config["adapter_source"],
        "config_content_sha256": _sha(config),
        "source_bindings": config["bindings"],
        "candidate_count": 3,
        "decision_counts": {"blocked": 3},
        "candidate_records": records,
        "exact_algebraic_threshold_pass_count": 3,
        "closed_threshold_endpoint_reject_count": 3,
        "above_threshold_negative_control_inconclusive_count": 3,
        "candidate_nontrivial_AF_Einstein_constraint_solution_pass_count": 0,
        "theory_reject_count": 0,
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
            "The 1/1000 York frontier is replaced by each candidate's exact positive algebraic "
            "threshold. The closed class |K|<=kappa_star*v is excluded, including the endpoint "
            "by strict interior source and Newton-kernel inequalities. Rational controls above "
            "each threshold are inconclusive. No AF solution, action rejection, global-energy "
            "result, or full-formal pass is inferred."
        ),
    }
    return {**body, "content_sha256": _sha(body)}


def write_future_g3_york_analytic_mean_curvature_threshold_campaign(
    config_path: str | Path, project_root: str | Path
) -> Path:
    root = Path(project_root).resolve()
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    artifact = build_future_g3_york_analytic_mean_curvature_threshold_campaign(config, root)
    output = root / config["output_path"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_canonical(artifact) + "\n", encoding="utf-8")
    return output
