from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from pathlib import Path
from typing import Any

import sympy as sp

from .promotion_orchestrator import ELIGIBILITY

ARTIFACT_SCHEMA = (
    "sigma-future-g3-york-tracefree-compensation-no-go-campaign-1.0"
)
FIRST_BLOCKER = (
    "candidate_specific_nontrivial_AF_Einstein_constraint_solution_beyond_"
    "conformally_flat_tracefree_compensated_York_class"
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


def _validate_compensation_contract(contract: dict[str, Any]) -> None:
    expected = {
        "geometry_class": "h_ij=psi(x)^4*delta_ij_nonradial_psi_allowed",
        "York_decomposition": "K_ij=A_ij+(K/3)*h_ij_with_h_trace(A)=0",
        "York_tensor_scope": (
            "arbitrary_smooth_nonradial_TT_or_longitudinal_or_mixed_A_ij"
        ),
        "scalar_profile": "Pi=v(r)=1/sqrt(1+(r/L)^4)",
        "transition_length_L": "100",
        "mean_curvature_scope": "arbitrary_smooth_AF_K_with_no_pointwise_cap",
        "source_factor_threshold": "c_star=1536/1953125",
        "compensation_deficit": (
            "D_beta=(2/3)*K^2-2*beta*K*v^3-(1-c_star)*v^2"
        ),
        "compensation_condition": "A_ij*A^ij>=max(0,D_beta)",
        "endpoint_semantics": (
            "excluded_by_strict_interior_ball_and_Newton_kernel_inequality"
        ),
        "momentum_semantics": (
            "Hamiltonian_no_go_independent_of_whether_momentum_equation_closes"
        ),
        "no_observation_or_numerical_solver_used": True,
    }
    if contract != expected:
        raise ValueError("York tracefree compensation contract changed")


def _candidate_compensation(
    prior: dict[str, Any], target: dict[str, Any], contract: dict[str, Any]
) -> dict[str, Any]:
    beta = Fraction(target["beta"])
    c_star = Fraction(1536, 1953125)
    outside_cap = Fraction(target["outside_cap_control"])
    expected_center_a_sq = Fraction(target["witness_A_squared_at_center"])
    derived_center_a_sq = (
        Fraction(2, 3) * outside_cap**2 + 2 * beta * outside_cap
    )
    deficit_at_center = (
        Fraction(2, 3) * outside_cap**2
        + 2 * beta * outside_cap
        - (1 - c_star)
    )
    uncompensated_source = (
        1 - 2 * beta * outside_cap - Fraction(2, 3) * outside_cap**2
    )
    if (
        derived_center_a_sq != expected_center_a_sq
        or deficit_at_center <= 0
        or uncompensated_source >= c_star
        or uncompensated_source + derived_center_a_sq != 1
    ):
        raise ValueError("tracefree compensation witness changed")
    threshold = prior["York_analytic_threshold_certificate"]
    if (
        prior["decision"] != "blocked"
        or prior["theory_rejected"] is not False
        or threshold["endpoint_certificate"]["threshold_endpoint_excluded"] is not True
        or threshold["above_threshold_negative_control"]["kappa"]
        != str(outside_cap)
        or threshold["above_threshold_negative_control"]["status"]
        != "comparison_inconclusive"
    ):
        raise ValueError("analytic threshold predecessor changed")
    beta_sp = sp.Rational(beta.numerator, beta.denominator)
    kappa_star = sp.sympify(
        threshold["exact_algebraic_threshold"]["candidate_root_expression"]
    )
    inclusion_residual = sp.simplify(
        sp.Rational(2, 3) * kappa_star**2
        + 2 * beta_sp * kappa_star
        - (1 - sp.Rational(c_star.numerator, c_star.denominator))
    )
    if inclusion_residual != 0:
        raise ValueError("analytic threshold inclusion identity changed")
    body = {
        "candidate_id": prior["candidate_id"],
        "action_sha256": prior["action_sha256"],
        "beta": str(beta),
        "predecessor_analytic_threshold_sha256": threshold["content_sha256"],
        "compensation_contract": contract,
        "direct_action_binding": True,
        "family_label_used_as_compensation_evidence": False,
        "exact_compensated_source_certificate": {
            "Hamiltonian_source": (
                "S_beta=v^2+2*beta*K*v^3-(2/3)*K^2+A_ij*A^ij"
            ),
            "compensation_deficit": (
                "D_beta=(2/3)*K^2-2*beta*K*v^3-(1-c_star)*v^2"
            ),
            "case_D_nonpositive": "A_squared>=0_implies_S_beta>=c_star*v^2",
            "case_D_positive": "A_squared>=D_beta_implies_S_beta>=c_star*v^2",
            "global_conclusion": "S_beta>=c_star*v^2",
            "Green_coefficient": "L^2*c_star/96=256/3125",
            "strict_endpoint_reason": (
                "v^2>1/2_in_ball_interior_and_Newton_kernel_bound_strict_a.e."
            ),
            "positive_AF_conformal_factor_exists_in_compensated_class": False,
        },
        "analytic_threshold_class_inclusion": {
            "kappa_star": threshold["exact_algebraic_threshold"][
                "candidate_root_expression"
            ],
            "root_identity": (
                "(2/3)*kappa_star^2+2*beta*kappa_star=1-c_star"
            ),
            "proof": (
                "if_abs_K<=kappa_star*v_then_D_beta<=0_so_A_squared>=0_suffices"
            ),
            "entire_predecessor_closed_class_included": True,
        },
        "strict_extension_witness_control": {
            "role": "algebraic_domain_nonemptiness_control_not_constraint_solution",
            "kappa": str(outside_cap),
            "K_profile": f"K=-({outside_cap})*v",
            "outside_predecessor_threshold": True,
            "A_frame": "h_orthonormal_fixed_tracefree_frame_diag(2,-1,-1)/sqrt(6)",
            "A_squared_profile": (
                f"(2/3)*({outside_cap})^2*v^2+2*({beta})*({outside_cap})*v^4"
            ),
            "A_squared_at_center": str(derived_center_a_sq),
            "compensation_deficit_at_center": str(deficit_at_center),
            "resulting_Hamiltonian_source": "S_beta=v^2",
            "momentum_constraint_solved": False,
            "AF_constraint_solution_inferred": False,
        },
        "undercompensated_negative_control": {
            "K_at_center": str(-outside_cap),
            "A_squared_at_center": "0",
            "source_at_center": str(uncompensated_source),
            "source_below_c_star": True,
            "status": "outside_certificate_comparison_inconclusive",
            "AF_solution_or_action_pass_inferred": False,
        },
        "decision": "reject_tracefree_compensated_conformally_flat_York_class",
        "excluded_class": (
            "conformally_flat_AF_nonradial_psi_arbitrary_smooth_AF_K_and_"
            "tracefree_A_satisfying_A_squared_at_least_positive_part_D_beta"
        ),
        "candidate_nontrivial_AF_Einstein_constraint_solution_available": False,
        "theory_rejected": False,
        "first_remaining_blocker": FIRST_BLOCKER,
        "scope": (
            "This exact class removes the mean-curvature cap: K may be arbitrarily large "
            "relative to v if the trace-free York norm supplies the candidate-specific deficit. "
            "It strictly contains the closed analytic-threshold class. The algebraic witness "
            "shows the expanded domain is nonempty but is not a momentum-constraint solution. "
            "Undercompensated data and non-conformally-flat metrics remain open."
        ),
    }
    return {**body, "content_sha256": _sha(body)}


def build_future_g3_york_tracefree_compensation_no_go_campaign(
    config: dict[str, Any], project_root: str | Path
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    if config.get("data_eligibility") != ELIGIBILITY:
        raise ValueError("campaign eligibility is not fail-closed")
    _validate_source(root, config)
    _validate_compensation_contract(config["compensation_contract"])
    predecessors = {
        key: _load_bound(root, descriptor) for key, descriptor in config["bindings"].items()
    }
    immediate = predecessors["predecessor"]
    grid_frontier = predecessors["grid_frontier_source"]
    if (
        immediate.get("source_bindings", {}).get("predecessor", {}).get(
            "content_sha256"
        )
        != grid_frontier.get("content_sha256")
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
            or prior["York_analytic_threshold_certificate"]["content_sha256"]
            != target["predecessor_threshold_content_sha256"]
        ):
            raise ValueError("target binding changed")
        certificate = _candidate_compensation(
            prior, target, config["compensation_contract"]
        )
        provenance_body = {
            "predecessor_content_sha256": immediate["content_sha256"],
            "predecessor_record_content_sha256": prior["content_sha256"],
            "grid_frontier_source_content_sha256": grid_frontier["content_sha256"],
            "action_sha256": prior["action_sha256"],
            "compensation_contract_sha256": _sha(config["compensation_contract"]),
            "compensation_certificate_sha256": certificate["content_sha256"],
            "data_eligibility": dict(ELIGIBILITY),
        }
        record_body = {
            "candidate_id": prior["candidate_id"],
            "action_sha256": prior["action_sha256"],
            "beta": prior["beta"],
            "decision": "blocked",
            "first_blocker": FIRST_BLOCKER,
            "York_tracefree_compensation_certificate": certificate,
            "gate_ledger": {
                "analytic_York_threshold_predecessor": {"status": "pass"},
                "exact_tracefree_compensation_bound": {"status": "pass"},
                "strict_domain_extension_witness": {"status": "pass_control_only"},
                "tracefree_compensated_York_class": {"status": "reject_ansatz_class"},
                "undercompensated_control": {"status": "inconclusive"},
                "candidate_nontrivial_AF_Einstein_constraint_solution_beyond_class": {
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
        raise ValueError("expected exactly three candidate tracefree compensation records")
    body = {
        "schema_version": ARTIFACT_SCHEMA,
        "campaign_id": config["campaign_id"],
        "campaign_adapter_source": config["adapter_source"],
        "config_content_sha256": _sha(config),
        "source_bindings": config["bindings"],
        "candidate_count": 3,
        "decision_counts": {"blocked": 3},
        "candidate_records": records,
        "exact_tracefree_compensation_bound_pass_count": 3,
        "strict_domain_extension_control_pass_count": 3,
        "tracefree_compensated_York_class_reject_count": 3,
        "undercompensated_negative_control_inconclusive_count": 3,
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
            "The exact G3 York no-go domain now has no pointwise mean-curvature cap. Arbitrary "
            "smooth AF K is covered whenever the trace-free York norm compensates the exact "
            "candidate deficit, and this class strictly contains the prior analytic-threshold "
            "class. Hamiltonian nonexistence follows from the same strict endpoint comparison. "
            "The extension witness is algebraic only, not a momentum or AF solution; "
            "undercompensated and non-conformally-flat data remain open."
        ),
    }
    return {**body, "content_sha256": _sha(body)}


def write_future_g3_york_tracefree_compensation_no_go_campaign(
    config_path: str | Path, project_root: str | Path
) -> Path:
    root = Path(project_root).resolve()
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    artifact = build_future_g3_york_tracefree_compensation_no_go_campaign(
        config, root
    )
    output = root / config["output_path"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_canonical(artifact) + "\n", encoding="utf-8")
    return output
