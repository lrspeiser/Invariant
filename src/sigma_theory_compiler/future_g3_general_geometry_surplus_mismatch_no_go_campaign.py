from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from pathlib import Path
from typing import Any

from .promotion_orchestrator import ELIGIBILITY

ARTIFACT_SCHEMA = (
    "sigma-future-g3-general-geometry-surplus-mismatch-no-go-campaign-1.0"
)
FIRST_BLOCKER = (
    "candidate_specific_AF_metric_and_York_data_with_pointwise_curvature_"
    "surplus_matching_and_momentum_solution"
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


def _validate_surplus_contract(contract: dict[str, Any]) -> None:
    expected = {
        "geometry_class": "arbitrary_smooth_complete_AF_three_metric_h_ij",
        "conformal_flatness_required": False,
        "Cotton_tensor_scope": "unrestricted_may_be_nonzero",
        "York_decomposition": "K_ij=A_ij+(K/3)*h_ij_with_h_trace(A)=0",
        "scalar_profile": "Pi=v(r)=1/sqrt(1+(r/L)^4)_in_registered_AF_chart",
        "threshold": "c_star=1536/1953125",
        "deficit": "D_beta=(2/3)*K^2-2*beta*K*v^3-(1-c_star)*v^2",
        "compensation_condition": "A_ij*A^ij>=max(0,D_beta)",
        "curvature_surplus": "C=R3-c_star*v^2",
        "York_source_surplus": "Y=A_ij*A^ij-D_beta=S_beta-c_star*v^2",
        "no_go_domain": "exists_finite_x_star_with_Y(x_star)>C(x_star)",
        "matching_semantics": "C=Y_is_Hamiltonian_necessary_not_AF_solution",
        "overcurvature_semantics": "C>Y_not_excluded_by_this_pointwise_theorem",
        "momentum_semantics": (
            "Hamiltonian_no_go_independent_of_whether_momentum_equation_closes"
        ),
        "no_observation_or_numerical_solver_used": True,
    }
    if contract != expected:
        raise ValueError("general geometry surplus-mismatch contract changed")


def _candidate_theorem(
    prior: dict[str, Any], target: dict[str, Any], contract: dict[str, Any]
) -> dict[str, Any]:
    beta = Fraction(target["beta"])
    c_star = Fraction(1536, 1953125)
    unit = Fraction(target["control_unit"])
    mismatch_c = Fraction(target["mismatch_curvature_surplus_ratio"])
    mismatch_y = Fraction(target["mismatch_York_surplus_ratio"])
    matched = Fraction(target["matched_surplus_ratio"])
    over_c = Fraction(target["overcurvature_surplus_ratio"])
    over_y = Fraction(target["overcurvature_York_surplus_ratio"])
    if (
        unit <= 0
        or mismatch_c != unit
        or mismatch_y != 2 * unit
        or matched != unit
        or over_c != 3 * unit
        or over_y != 2 * unit
    ):
        raise ValueError("surplus controls changed")
    predecessor = prior["general_geometry_curvature_shortfall_certificate"]
    if (
        prior["decision"] != "blocked"
        or prior["theory_rejected"] is not False
        or predecessor["exact_pointwise_theorem"]["compensated_source_lower"]
        != "S_beta>=c_star*v^2"
        or predecessor["sharp_endpoint_control"]["status"]
        != "conditional_theorem_inconclusive_at_exact_endpoint"
    ):
        raise ValueError("general geometry predecessor changed")
    body = {
        "candidate_id": prior["candidate_id"],
        "action_sha256": prior["action_sha256"],
        "beta": str(beta),
        "predecessor_curvature_shortfall_sha256": predecessor["content_sha256"],
        "surplus_contract": contract,
        "direct_action_binding": True,
        "family_label_used_as_surplus_evidence": False,
        "exact_surplus_identity": {
            "Hamiltonian_source": (
                "S_beta=v^2+2*beta*K*v^3-(2/3)*K^2+A_ij*A^ij"
            ),
            "deficit": (
                "D_beta=(2/3)*K^2-2*beta*K*v^3-(1-c_star)*v^2"
            ),
            "York_source_surplus": "Y=A_ij*A^ij-D_beta=S_beta-c_star*v^2",
            "curvature_surplus": "C=R3-c_star*v^2",
            "Hamiltonian_residual": "R3-S_beta=C-Y",
            "compensation_implies": "Y>=0",
            "pointwise_no_go_condition": "Y(x_star)>C(x_star)",
            "conformal_flatness_used": False,
            "Cotton_tensor_restricted": False,
        },
        "above_threshold_mismatch_control": {
            "role": "deterministic_symbolic_endpoint_extension_control",
            "curvature_ratio_R3_over_v_squared": str(c_star + mismatch_c),
            "source_ratio_S_beta_over_v_squared": str(c_star + mismatch_y),
            "curvature_surplus_ratio": str(mismatch_c),
            "York_source_surplus_ratio": str(mismatch_y),
            "Hamiltonian_residual_ratio": str(mismatch_c - mismatch_y),
            "decision": "reject_by_exact_negative_Hamiltonian_residual",
            "metric_or_York_fields_constructed": False,
            "AF_constraint_solution_inferred": False,
        },
        "matched_surplus_control": {
            "curvature_surplus_ratio": str(matched),
            "York_source_surplus_ratio": str(matched),
            "Hamiltonian_residual_ratio": "0",
            "status": "pointwise_Hamiltonian_match_only",
            "momentum_constraint_solved": False,
            "AF_constraint_solution_inferred": False,
        },
        "overcurvature_negative_control": {
            "curvature_surplus_ratio": str(over_c),
            "York_source_surplus_ratio": str(over_y),
            "Hamiltonian_residual_ratio": str(over_c - over_y),
            "status": "not_excluded_by_surplus_lower_bound",
            "AF_constraint_solution_or_action_pass_inferred": False,
        },
        "decision": "reject_general_AF_geometry_surplus_mismatch_class",
        "excluded_class": (
            "arbitrary_AF_three_geometry_including_nonconformally_flat_with_compensated_"
            "York_data_and_a_finite_point_where_York_source_surplus_exceeds_curvature_surplus"
        ),
        "candidate_nontrivial_AF_Einstein_constraint_solution_available": False,
        "theory_rejected": False,
        "first_remaining_blocker": FIRST_BLOCKER,
        "scope": (
            "This theorem strictly extends the curvature-shortfall result into endpoint and "
            "above-threshold geometry: positive curvature surplus is allowed, but must match the "
            "candidate York-source surplus pointwise. It does not construct h, K, or A, solve "
            "momentum, or infer an AF solution from a matched point; C=Y everywhere is only a "
            "necessary Hamiltonian condition."
        ),
    }
    return {**body, "content_sha256": _sha(body)}


def build_future_g3_general_geometry_surplus_mismatch_no_go_campaign(
    config: dict[str, Any], project_root: str | Path
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    if config.get("data_eligibility") != ELIGIBILITY:
        raise ValueError("campaign eligibility is not fail-closed")
    _validate_source(root, config)
    _validate_surplus_contract(config["surplus_contract"])
    predecessors = {
        key: _load_bound(root, descriptor) for key, descriptor in config["bindings"].items()
    }
    immediate = predecessors["predecessor"]
    compensation = predecessors["tracefree_compensation_source"]
    if (
        immediate.get("source_bindings", {}).get("predecessor", {}).get(
            "content_sha256"
        )
        != compensation.get("content_sha256")
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
            or prior["general_geometry_curvature_shortfall_certificate"]["content_sha256"]
            != target["predecessor_theorem_content_sha256"]
        ):
            raise ValueError("target binding changed")
        theorem = _candidate_theorem(prior, target, config["surplus_contract"])
        provenance_body = {
            "predecessor_content_sha256": immediate["content_sha256"],
            "predecessor_record_content_sha256": prior["content_sha256"],
            "tracefree_compensation_source_content_sha256": compensation["content_sha256"],
            "action_sha256": prior["action_sha256"],
            "surplus_contract_sha256": _sha(config["surplus_contract"]),
            "surplus_theorem_sha256": theorem["content_sha256"],
            "data_eligibility": dict(ELIGIBILITY),
        }
        record_body = {
            "candidate_id": prior["candidate_id"],
            "action_sha256": prior["action_sha256"],
            "beta": prior["beta"],
            "decision": "blocked",
            "first_blocker": FIRST_BLOCKER,
            "general_geometry_surplus_mismatch_certificate": theorem,
            "gate_ledger": {
                "general_geometry_curvature_shortfall_predecessor": {"status": "pass"},
                "exact_surplus_identity": {"status": "pass"},
                "above_threshold_surplus_mismatch_class": {
                    "status": "reject_constraint_class"
                },
                "matched_surplus_control": {"status": "necessary_only"},
                "overcurvature_control": {"status": "not_excluded"},
                "candidate_AF_metric_York_matching_and_momentum": {"status": "blocked"},
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
        raise ValueError("expected exactly three candidate surplus-mismatch records")
    body = {
        "schema_version": ARTIFACT_SCHEMA,
        "campaign_id": config["campaign_id"],
        "campaign_adapter_source": config["adapter_source"],
        "config_content_sha256": _sha(config),
        "source_bindings": config["bindings"],
        "candidate_count": 3,
        "decision_counts": {"blocked": 3},
        "candidate_records": records,
        "exact_surplus_identity_pass_count": 3,
        "above_threshold_surplus_mismatch_class_reject_count": 3,
        "matched_surplus_necessary_control_count": 3,
        "overcurvature_not_excluded_control_count": 3,
        "registered_AF_metric_York_datum_pass_count": 0,
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
            "The exact candidate theorem now covers endpoint and above-threshold curvature when "
            "the compensated York-source surplus exceeds the geometry's curvature surplus at "
            "any finite point. Exact matching is recorded only as a necessary Hamiltonian "
            "condition, and overcurvature is not excluded. No AF datum, momentum solution, "
            "action pass, energy result, or full-formal result is inferred."
        ),
    }
    return {**body, "content_sha256": _sha(body)}


def write_future_g3_general_geometry_surplus_mismatch_no_go_campaign(
    config_path: str | Path, project_root: str | Path
) -> Path:
    root = Path(project_root).resolve()
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    artifact = build_future_g3_general_geometry_surplus_mismatch_no_go_campaign(
        config, root
    )
    output = root / config["output_path"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_canonical(artifact) + "\n", encoding="utf-8")
    return output
