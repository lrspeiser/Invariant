from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from pathlib import Path
from typing import Any

from .promotion_orchestrator import ELIGIBILITY

ARTIFACT_SCHEMA = "sigma-future-g3-york-mean-curvature-frontier-campaign-1.0"
FIRST_BLOCKER = (
    "candidate_specific_nontrivial_AF_Einstein_constraint_solution_beyond_"
    "registered_millicap_conformally_flat_York_class"
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


def _validate_frontier_contract(contract: dict[str, Any]) -> None:
    expected = {
        "retained_geometry_class": "h_ij=psi(x)^4*delta_ij_nonradial_psi_allowed",
        "retained_York_tensor_scope": (
            "arbitrary_smooth_nonradial_TT_or_longitudinal_or_mixed_A_ij"
        ),
        "retained_scalar_profile": "Pi=v(r)=1/sqrt(1+(r/L)^4)",
        "transition_length_L": "100",
        "predecessor_common_kappa_cap": "6/5",
        "kappa_grid_step": "1/1000",
        "cap_selection_rule": (
            "largest_nonnegative_kappa_grid_point_with_strict_Green_excess"
        ),
        "next_grid_point_semantics": (
            "comparison_inconclusive_not_solution_or_theory_pass"
        ),
        "AF_conformal_factor_class": (
            "C2_positive_psi_minus_1_equals_O2_r_minus_1"
        ),
        "no_momentum_solution_assumed": True,
        "no_observation_or_numerical_solver_used": True,
    }
    if contract != expected:
        raise ValueError("York mean-curvature frontier contract changed")


def _frontier_certificate(
    prior: dict[str, Any], target: dict[str, Any], contract: dict[str, Any]
) -> dict[str, Any]:
    beta = Fraction(target["beta"])
    kappa = Fraction(target["kappa_cap"])
    next_kappa = Fraction(target["next_grid_cap"])
    predecessor_cap = Fraction(contract["predecessor_common_kappa_cap"])
    grid_step = Fraction(contract["kappa_grid_step"])
    length = Fraction(contract["transition_length_L"])
    universal_maximum = Fraction(256, 3125)
    required_source_factor = universal_maximum * 96 / length**2

    def evaluate(cap: Fraction) -> tuple[Fraction, Fraction, Fraction]:
        source_factor = 1 - 2 * beta * cap - Fraction(2, 3) * cap**2
        green_coefficient = source_factor * length**2 / 96
        return (
            source_factor,
            green_coefficient,
            green_coefficient - universal_maximum,
        )

    factor, coefficient, excess = evaluate(kappa)
    next_factor, next_coefficient, next_excess = evaluate(next_kappa)
    if (
        kappa <= predecessor_cap
        or next_kappa != kappa + grid_step
        or factor <= required_source_factor
        or excess <= 0
        or next_excess > 0
    ):
        raise ValueError("candidate York millicap frontier did not certify")
    if (
        prior["decision"] != "blocked"
        or prior["theory_rejected"] is not False
        or prior["nonradial_York_no_go_certificate"]["theory_rejected"] is not False
    ):
        raise ValueError("predecessor York no-go decision changed")
    body = {
        "candidate_id": prior["candidate_id"],
        "action_sha256": prior["action_sha256"],
        "beta": str(beta),
        "predecessor_York_no_go_sha256": prior[
            "nonradial_York_no_go_certificate"
        ]["content_sha256"],
        "frontier_contract": contract,
        "direct_action_binding": True,
        "family_label_used_as_frontier_evidence": False,
        "exact_frontier": {
            "predecessor_common_kappa_cap": str(predecessor_cap),
            "candidate_kappa_cap": str(kappa),
            "strict_extension_beyond_predecessor": str(kappa - predecessor_cap),
            "grid_step": str(grid_step),
            "next_grid_cap": str(next_kappa),
            "source_factor_at_cap": str(factor),
            "required_source_factor_for_strict_no_go": str(required_source_factor),
            "green_ball_coefficient_at_cap": str(coefficient),
            "universal_allowed_green_coefficient": str(universal_maximum),
            "strict_green_excess_at_cap": str(excess),
            "source_factor_at_next_grid_cap": str(next_factor),
            "green_ball_coefficient_at_next_grid_cap": str(next_coefficient),
            "green_excess_at_next_grid_cap": str(next_excess),
            "source_factor_monotonicity_for_nonnegative_kappa": (
                "d_source_factor/d_kappa=-2*beta-(4/3)*kappa<0"
            ),
            "cap_is_largest_certified_point_on_declared_grid": True,
            "next_grid_cap_comparison_status": "inconclusive",
        },
        "retained_exact_bound": (
            "8*q>=v^2*(1-2*beta*kappa-(2/3)*kappa^2)"
        ),
        "retained_nonradial_Green_inequality": "m-1>=B_L*m^5",
        "decision": "reject_expanded_candidate_millicap_York_class",
        "candidate_nontrivial_AF_Einstein_constraint_solution_available": False,
        "theory_rejected": False,
        "first_remaining_blocker": FIRST_BLOCKER,
        "scope": (
            "This extends the exact conformally-flat nonradial York obstruction beyond "
            "|K|<=(6/5)*v to the largest candidate-specific 1/1000-grid cap closed by the "
            "same sharp Green comparison. Failure at the next grid point is only failure of "
            "this sufficient bound; it is not evidence for existence or an action pass. "
            "Non-conformally-flat metrics and larger mean curvature remain open."
        ),
    }
    return {**body, "content_sha256": _sha(body)}


def build_future_g3_york_mean_curvature_frontier_campaign(
    config: dict[str, Any], project_root: str | Path
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    if config.get("data_eligibility") != ELIGIBILITY:
        raise ValueError("campaign eligibility is not fail-closed")
    _validate_source(root, config)
    _validate_frontier_contract(config["frontier_contract"])
    predecessors = {
        key: _load_bound(root, descriptor) for key, descriptor in config["bindings"].items()
    }
    immediate = predecessors["predecessor"]
    radial_no_go = predecessors["radial_no_go_source"]
    radial_reduction = predecessors["radial_reduction_source"]
    if (
        immediate.get("source_bindings", {}).get("predecessor", {}).get(
            "content_sha256"
        )
        != radial_no_go.get("content_sha256")
        or immediate.get("source_bindings", {}).get("radial_reduction_source", {}).get(
            "content_sha256"
        )
        != radial_reduction.get("content_sha256")
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
            or prior["nonradial_York_no_go_certificate"]["content_sha256"]
            != target["predecessor_no_go_content_sha256"]
        ):
            raise ValueError("target binding changed")
        certificate = _frontier_certificate(prior, target, config["frontier_contract"])
        provenance_body = {
            "predecessor_content_sha256": immediate["content_sha256"],
            "predecessor_record_content_sha256": prior["content_sha256"],
            "radial_no_go_source_content_sha256": radial_no_go["content_sha256"],
            "radial_reduction_source_content_sha256": radial_reduction["content_sha256"],
            "action_sha256": prior["action_sha256"],
            "frontier_contract_sha256": _sha(config["frontier_contract"]),
            "frontier_certificate_sha256": certificate["content_sha256"],
            "data_eligibility": dict(ELIGIBILITY),
        }
        record_body = {
            "candidate_id": prior["candidate_id"],
            "action_sha256": prior["action_sha256"],
            "beta": prior["beta"],
            "decision": "blocked",
            "first_blocker": FIRST_BLOCKER,
            "York_mean_curvature_frontier_certificate": certificate,
            "gate_ledger": {
                "nonradial_York_no_go_predecessor": {"status": "pass"},
                "candidate_millicap_frontier_registration": {"status": "pass"},
                "strict_extension_beyond_kappa_6_over_5": {"status": "pass"},
                "expanded_nonradial_York_class": {"status": "reject_ansatz_class"},
                "next_grid_cap": {"status": "inconclusive"},
                "candidate_nontrivial_AF_Einstein_constraint_solution_beyond_frontier": {
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
        raise ValueError("expected exactly three candidate York frontier records")
    body = {
        "schema_version": ARTIFACT_SCHEMA,
        "campaign_id": config["campaign_id"],
        "campaign_adapter_source": config["adapter_source"],
        "config_content_sha256": _sha(config),
        "source_bindings": config["bindings"],
        "candidate_count": 3,
        "decision_counts": {"blocked": 3},
        "candidate_records": records,
        "candidate_millicap_frontier_registration_pass_count": 3,
        "strict_extension_beyond_kappa_6_over_5_pass_count": 3,
        "expanded_nonradial_York_class_reject_count": 3,
        "next_grid_cap_inconclusive_count": 3,
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
            "All three candidate-specific nonradial conformally-flat York no-go domains now "
            "extend strictly beyond |K|<=(6/5)*v. Exact 1/1000-grid frontier checks identify "
            "the largest cap closed by the existing sharp Green comparison and record the next "
            "grid point as inconclusive. No AF constraint solution, action rejection, global "
            "energy result, or full-formal pass is inferred."
        ),
    }
    return {**body, "content_sha256": _sha(body)}


def write_future_g3_york_mean_curvature_frontier_campaign(
    config_path: str | Path, project_root: str | Path
) -> Path:
    root = Path(project_root).resolve()
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    artifact = build_future_g3_york_mean_curvature_frontier_campaign(config, root)
    output = root / config["output_path"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_canonical(artifact) + "\n", encoding="utf-8")
    return output
