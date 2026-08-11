from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from pathlib import Path
from typing import Any

from .promotion_orchestrator import ELIGIBILITY

ARTIFACT_SCHEMA = (
    "sigma-future-g3-general-geometry-curvature-shortfall-no-go-campaign-1.0"
)
FIRST_BLOCKER = (
    "candidate_specific_AF_Einstein_constraint_datum_outside_general_geometry_"
    "curvature_shortfall_class"
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


def _validate_geometry_contract(contract: dict[str, Any]) -> None:
    expected = {
        "geometry_class": "arbitrary_smooth_complete_AF_three_metric_h_ij",
        "conformal_flatness_required": False,
        "Cotton_tensor_scope": "unrestricted_may_be_nonzero",
        "York_decomposition": "K_ij=A_ij+(K/3)*h_ij_with_h_trace(A)=0",
        "mean_curvature_scope": "arbitrary_smooth_AF_K_with_no_pointwise_cap",
        "scalar_profile": "Pi=v(r)=1/sqrt(1+(r/L)^4)_in_registered_AF_chart",
        "source_factor_threshold": "c_star=1536/1953125",
        "compensation_condition": (
            "A_ij*A^ij>=max(0,(2/3)*K^2-2*beta*K*v^3-(1-c_star)*v^2)"
        ),
        "curvature_shortfall_domain": (
            "exists_finite_x_star_with_R3(x_star)/v(x_star)^2<c_star"
        ),
        "endpoint_semantics": "R3/v^2=c_star_is_inconclusive_if_source_also_saturates",
        "above_threshold_semantics": "not_excluded_by_this_pointwise_theorem",
        "momentum_semantics": (
            "Hamiltonian_no_go_independent_of_whether_momentum_equation_closes"
        ),
        "no_observation_or_numerical_solver_used": True,
    }
    if contract != expected:
        raise ValueError("general geometry curvature-shortfall contract changed")


def _candidate_theorem(
    prior: dict[str, Any], target: dict[str, Any], contract: dict[str, Any]
) -> dict[str, Any]:
    beta = Fraction(target["beta"])
    c_star = Fraction(1536, 1953125)
    below_ratio = Fraction(target["below_threshold_control_ratio"])
    endpoint_ratio = Fraction(target["endpoint_control_ratio"])
    above_ratio = Fraction(target["above_threshold_control_ratio"])
    below_margin = c_star - below_ratio
    if (
        below_ratio < 0
        or below_margin <= 0
        or endpoint_ratio != c_star
        or above_ratio <= c_star
    ):
        raise ValueError("curvature-ratio controls changed")
    compensation = prior["York_tracefree_compensation_certificate"]
    if (
        prior["decision"] != "blocked"
        or prior["theory_rejected"] is not False
        or compensation["exact_compensated_source_certificate"]["global_conclusion"]
        != "S_beta>=c_star*v^2"
        or compensation["candidate_nontrivial_AF_Einstein_constraint_solution_available"]
        is not False
    ):
        raise ValueError("tracefree compensation predecessor changed")
    body = {
        "candidate_id": prior["candidate_id"],
        "action_sha256": prior["action_sha256"],
        "beta": str(beta),
        "predecessor_tracefree_compensation_sha256": compensation["content_sha256"],
        "general_geometry_contract": contract,
        "direct_action_binding": True,
        "family_label_used_as_geometry_evidence": False,
        "exact_pointwise_theorem": {
            "Hamiltonian_constraint": "R3=S_beta",
            "candidate_source": (
                "S_beta=v^2+2*beta*K*v^3-(2/3)*K^2+A_ij*A^ij"
            ),
            "compensated_source_lower": "S_beta>=c_star*v^2",
            "finite_profile_positivity": "v(x_star)^2>0_for_every_finite_x_star",
            "curvature_shortfall_hypothesis": (
                "R3(x_star)<c_star*v(x_star)^2_at_some_finite_x_star"
            ),
            "strict_residual_sign": "R3(x_star)-S_beta(x_star)<0",
            "Hamiltonian_constraint_solution_exists_in_declared_class": False,
            "conformal_flatness_used": False,
            "Cotton_tensor_restricted": False,
        },
        "exact_below_threshold_control": {
            "role": "deterministic_symbolic_curvature_ratio_control",
            "R3_over_v_squared": str(below_ratio),
            "source_lower_ratio": str(c_star),
            "strict_residual_upper_ratio": str(-below_margin),
            "decision": "reject_by_pointwise_Hamiltonian_residual",
            "metric_constructed": False,
            "AF_constraint_solution_inferred": False,
        },
        "sharp_endpoint_control": {
            "R3_over_v_squared": str(endpoint_ratio),
            "source_ratio_if_compensation_saturates": str(c_star),
            "Hamiltonian_residual_if_both_saturate": "0",
            "status": "conditional_theorem_inconclusive_at_exact_endpoint",
            "AF_constraint_solution_inferred": False,
        },
        "above_threshold_negative_control": {
            "R3_over_v_squared": str(above_ratio),
            "difference_from_threshold": str(above_ratio - c_star),
            "status": "not_excluded_by_pointwise_lower_bound",
            "AF_constraint_solution_or_action_pass_inferred": False,
        },
        "decision": "reject_general_AF_geometry_curvature_shortfall_class",
        "excluded_class": (
            "arbitrary_AF_three_geometry_including_nonconformally_flat_with_compensated_"
            "York_data_and_a_finite_point_of_scalar_curvature_ratio_below_c_star"
        ),
        "candidate_nontrivial_AF_Einstein_constraint_solution_available": False,
        "theory_rejected": False,
        "first_remaining_blocker": FIRST_BLOCKER,
        "scope": (
            "The theorem removes conformal flatness and places no restriction on the Cotton "
            "tensor. It excludes any compensated candidate-bound AF geometry with even one "
            "finite scalar-curvature shortfall point. It does not construct a nonconformally-flat "
            "metric, solve momentum, or decide geometries with R3>=c_star*v^2 everywhere; the "
            "exact saturated endpoint remains inconclusive."
        ),
    }
    return {**body, "content_sha256": _sha(body)}


def build_future_g3_general_geometry_curvature_shortfall_no_go_campaign(
    config: dict[str, Any], project_root: str | Path
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    if config.get("data_eligibility") != ELIGIBILITY:
        raise ValueError("campaign eligibility is not fail-closed")
    _validate_source(root, config)
    _validate_geometry_contract(config["general_geometry_contract"])
    predecessors = {
        key: _load_bound(root, descriptor) for key, descriptor in config["bindings"].items()
    }
    immediate = predecessors["predecessor"]
    analytic_threshold = predecessors["analytic_threshold_source"]
    if (
        immediate.get("source_bindings", {}).get("predecessor", {}).get(
            "content_sha256"
        )
        != analytic_threshold.get("content_sha256")
    ):
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
            or prior["York_tracefree_compensation_certificate"]["content_sha256"]
            != target["predecessor_compensation_content_sha256"]
        ):
            raise ValueError("target binding changed")
        theorem = _candidate_theorem(prior, target, config["general_geometry_contract"])
        provenance_body = {
            "predecessor_content_sha256": immediate["content_sha256"],
            "predecessor_record_content_sha256": prior["content_sha256"],
            "analytic_threshold_source_content_sha256": analytic_threshold["content_sha256"],
            "action_sha256": prior["action_sha256"],
            "geometry_contract_sha256": _sha(config["general_geometry_contract"]),
            "pointwise_theorem_sha256": theorem["content_sha256"],
            "data_eligibility": dict(ELIGIBILITY),
        }
        record_body = {
            "candidate_id": prior["candidate_id"],
            "action_sha256": prior["action_sha256"],
            "beta": prior["beta"],
            "decision": "blocked",
            "first_blocker": FIRST_BLOCKER,
            "general_geometry_curvature_shortfall_certificate": theorem,
            "gate_ledger": {
                "tracefree_compensated_York_predecessor": {"status": "pass"},
                "general_geometry_pointwise_Hamiltonian_theorem": {"status": "pass"},
                "curvature_shortfall_class": {"status": "reject_constraint_class"},
                "exact_curvature_endpoint": {"status": "inconclusive"},
                "above_threshold_control": {"status": "not_excluded"},
                "nonconformally_flat_metric_construction": {"status": "not_attempted"},
                "candidate_nontrivial_AF_Einstein_constraint_solution": {
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
        raise ValueError("expected exactly three candidate general-geometry records")
    body = {
        "schema_version": ARTIFACT_SCHEMA,
        "campaign_id": config["campaign_id"],
        "campaign_adapter_source": config["adapter_source"],
        "config_content_sha256": _sha(config),
        "source_bindings": config["bindings"],
        "candidate_count": 3,
        "decision_counts": {"blocked": 3},
        "candidate_records": records,
        "general_geometry_pointwise_theorem_pass_count": 3,
        "curvature_shortfall_constraint_class_reject_count": 3,
        "exact_curvature_endpoint_inconclusive_count": 3,
        "above_threshold_not_excluded_control_count": 3,
        "nonconformally_flat_metric_construction_pass_count": 0,
        "momentum_constraint_solution_pass_count": 0,
        "candidate_nontrivial_AF_Einstein_constraint_solution_pass_count": 0,
        "theory_reject_count": 0,
        "global_hamiltonian_energy_pass_count": 0,
        "full_formal_pass_count": 0,
        "first_blocker_counts": {FIRST_BLOCKER: 3},
        "observational_data_opened": False,
        "dark_matter_or_halo_inputs": False,
        "redshift_distance_inputs": False,
        "paid_llm_spend_usd": 0.0,
        "synthetic_fixture_role": "deterministic_symbolic_controls_only",
        "data_eligibility": dict(ELIGIBILITY),
        "interpretation": (
            "The candidate-bound no-go now applies to arbitrary AF three-geometries, including "
            "non-conformally-flat metrics, whenever compensated York data meet a finite point "
            "where R3/v^2<c_star. The result is a sharp conditional pointwise theorem: equality "
            "and above-threshold geometries remain inconclusive, no metric or momentum solution "
            "is constructed, and no action, energy, or full-formal claim advances."
        ),
    }
    return {**body, "content_sha256": _sha(body)}


def write_future_g3_general_geometry_curvature_shortfall_no_go_campaign(
    config_path: str | Path, project_root: str | Path
) -> Path:
    root = Path(project_root).resolve()
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    artifact = build_future_g3_general_geometry_curvature_shortfall_no_go_campaign(
        config, root
    )
    output = root / config["output_path"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_canonical(artifact) + "\n", encoding="utf-8")
    return output
