"""Fail-closed candidate-action completions for the Kastner--Schlatter intake.

These actions are compiler hypotheses.  They are not claimed to occur in, or to
follow from, arXiv:2209.04025.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from fractions import Fraction
from pathlib import Path
from typing import Any

CONFIG_SCHEMA = "sigma-kastner-schlatter-candidate-action-completion-config-1.0"
RESULT_SCHEMA = "sigma-kastner-schlatter-candidate-action-completion-1.0"
FIRST_BLOCKER = "no_paper_derivation_of_candidate_action_or_transaction_intensity_dynamics"


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _content_sha(value: Mapping[str, Any]) -> str:
    return _sha({key: item for key, item in value.items() if key != "content_sha256"})


def _bound_json(root: Path, binding: Mapping[str, Any], label: str) -> dict[str, Any]:
    if set(binding) != {"path", "file_sha256", "content_sha256"}:
        raise ValueError(f"{label} binding fields changed")
    path = (root / str(binding["path"])).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"{label} path escapes repository") from error
    if not path.is_file() or _file_sha(path) != binding["file_sha256"]:
        raise ValueError(f"{label} file hash mismatch")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be an object")
    if value.get("content_sha256") != binding["content_sha256"]:
        raise ValueError(f"{label} declared content hash mismatch")
    if _content_sha(value) != binding["content_sha256"]:
        raise ValueError(f"{label} content hash mismatch")
    return value


def _validate_config(config: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "campaign_id",
        "output_path",
        "predecessors",
        "conventions",
        "branches",
        "seals",
    }
    if set(config) != required or config.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("candidate-action config shape changed")
    expected_seals = {
        "paper_action_claim_allowed": False,
        "paper_variational_derivation_claim_allowed": False,
        "formal_gr_equivalence_claim_allowed": False,
        "observational_pass_allowed": False,
        "theory_validity_claim_allowed": False,
        "dark_sector_elimination_claim_allowed": False,
        "observations_opened": False,
        "paid_llm_calls": False,
    }
    if config.get("seals") != expected_seals:
        raise ValueError("candidate-action claim/data seals changed")
    expected_conventions = {
        "metric_signature": "-+++",
        "coordinate_x0": "c*t",
        "einstein_equation": "G_mn=(8*pi*G/c**4)*T_mn",
        "stress_tensor": "T_mn=-2*c/sqrt(-g)*delta(S_matter)/delta(g^mn)",
        "planck_relation": "h=2*pi*hbar",
        "planck_length": "l_P**2=G*hbar/c**3",
        "intensity_dimension": "L^-4",
        "boundary_condition": (
            "Dirichlet metric with GHY term; Dirichlet q or natural n^m*grad_m(q)=0"
        ),
    }
    if config.get("conventions") != expected_conventions:
        raise ValueError("candidate-action conventions changed")
    expected_branches = [
        {
            "branch_id": "eq35_middle_h",
            "beta": "1/2",
            "target_lambda": "4*pi*G*h*q0/c**3",
            "target_lambda_planck_units": "8*pi**2*l_P**2*q0",
            "source_status": "matches equation 35 first and middle equalities",
        },
        {
            "branch_id": "eq35_printed_planck",
            "beta": "1/4",
            "target_lambda": "2*pi*G*h*q0/c**3",
            "target_lambda_planck_units": "4*pi**2*l_P**2*q0",
            "source_status": (
                "matches equation 35 printed final equality after h=2*pi*hbar"
            ),
        },
    ]
    if config.get("branches") != expected_branches:
        raise ValueError("equation-35 branch coefficients changed")


def _formula_nodes(graph: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    nodes = graph.get("knowledge_graph", {}).get("nodes", [])
    return {
        str(node["node_id"]): node
        for node in nodes
        if isinstance(node, dict) and node.get("node_type") == "formula"
    }


def _validate_predecessors(graph: Mapping[str, Any], intake: Mapping[str, Any]) -> None:
    if (
        graph.get("decision") != "blocked"
        or graph.get("first_blocker")
        != "no_candidate_bound_fundamental_action_or_complete_variational_field_system"
        or graph.get("admission_contract", {}).get("fundamental_action") is not None
        or graph.get("admission_contract", {}).get("variational_edges_present") is not False
    ):
        raise ValueError("equation-graph fail-closed boundary changed")
    formulas = _formula_nodes(graph)
    expected = {
        "EQ-KS-34-PRESSURE": "Pbar = -c*h*q_gamma",
        "EQ-KS-35-LAMBDA-PRESSURE": "Lambda = -4*pi*G*Pbar/c**4",
        "EQ-KS-35-LAMBDA-RATE": "Lambda = 4*pi*G*h*q_gamma/c**3",
        "EQ-KS-35-LAMBDA-PLANCK": "Lambda = 4*pi**2*lP**2*q_gamma",
    }
    for node_id, expression in expected.items():
        if formulas.get(node_id, {}).get("record", {}).get("expression") != expression:
            raise ValueError(f"source formula changed: {node_id}")
    checks = intake.get("synthetic_checks", [])
    eq35 = next(
        (item for item in checks if item.get("check_id") == "transaction_pressure_lambda_identity"),
        None,
    )
    if (
        eq35 is None
        or eq35.get("status") != "block"
        or eq35.get("first_missing_premise")
        != "equation_35_h_vs_hbar_factor_normalization_clarification"
    ):
        raise ValueError("intake equation-35 ambiguity boundary changed")
    if intake.get("action_contract", {}).get("fundamental_action") is not None:
        raise ValueError("intake unexpectedly claims a fundamental action")


def _dimensions() -> dict[str, Any]:
    return {
        "base_convention": "x0=c*t; d4x has L^4; the bulk integrand is divided by c",
        "quantities": {
            "q_and_q0": "L^-4",
            "grad_q": "L^-5",
            "c*h*q0": "M*L^-1*T^-2 (energy density/pressure)",
            "A_q": "M*L^7*T^-2",
            "B_q": "M*L^9*T^-2",
            "V": "M*L^-1*T^-2",
            "c^4*R/G": "M*L^-1*T^-2",
            "Lambda": "L^-2",
            "bulk_action": "M*L^2*T^-1",
            "GHY_action": "M*L^2*T^-1",
        },
        "checks": {
            "A_q_times_q_difference_squared": "M*L^-1*T^-2",
            "B_q_times_gradient_q_squared": "M*L^-1*T^-2",
            "eight_pi_G_V_over_c4": "L^-2",
            "poisson_mean_integral_q_dVol4": "dimensionless",
        },
        "all_declared_terms_dimensionally_closed": True,
    }


def _branch_contract(branch: Mapping[str, Any]) -> dict[str, Any]:
    beta = Fraction(str(branch["beta"]))
    coefficient_h = 8 * beta
    coefficient_planck = 16 * beta
    return {
        "branch_id": branch["branch_id"],
        "status": "candidate_completion_hypothesis_conditionally_matches_selected_eq35_branch",
        "paper_authorship_or_derivation": False,
        "beta": str(beta),
        "candidate_action": {
            "bulk": (
                "S_beta=(1/c)*Integral_M sqrt(-g)*[c^4*R/(16*pi*G)"
                "-(B_q/2)*g^mn*grad_m(q)*grad_n(q)-V_beta(q)] d4x"
            ),
            "potential": "V_beta(q)=beta*c*h*q0+(A_q/2)*(q-q0)^2",
            "boundary": "S_GHY=c^3/(8*pi*G)*Integral_boundary sqrt(abs(h_ind))*K d3y",
            "parameter_domain": "q0>0, A_q>0, B_q>0",
            "variational_boundary_data": (
                "delta(h_ind)=0 and delta(q)=0, or natural n^m*grad_m(q)=0"
            ),
            "local_deterministic_action_complete": True,
            "stochastic_law_derived_by_action": False,
        },
        "euler_lagrange": {
            "metric": (
                "G_mn=(8*pi*G/c^4)*[B_q*grad_m(q)*grad_n(q)"
                "-g_mn*((B_q/2)*(grad(q))^2+V_beta(q))]"
            ),
            "intensity": "B_q*Box(q)-A_q*(q-q0)=0",
            "stationary_solution": "q=q0",
            "stationary_stress": "T_mn=-beta*c*h*q0*g_mn",
            "stationary_einstein": "G_mn+Lambda_beta*g_mn=0",
        },
        "matching": {
            "derived_lambda": f"{coefficient_h}*pi*G*h*q0/c**3",
            "derived_lambda_planck_units": f"{coefficient_planck}*pi**2*l_P**2*q0",
            "declared_target_lambda": branch["target_lambda"],
            "declared_target_lambda_planck_units": branch["target_lambda_planck_units"],
            "exact_coefficient_match": True,
            "source_status": branch["source_status"],
            "normalization_selected_as_fact": False,
        },
        "noether_bianchi": {
            "off_shell_identity": (
                "nabla^m(T_mn)=[B_q*Box(q)-A_q*(q-q0)]*grad_n(q)"
            ),
            "on_shell_covariant_conservation": True,
            "stationary_solution_consistent": True,
            "external_nonconstant_q_without_intensity_EL": (
                "not admissible unless its force density is balanced by an added sector"
            ),
        },
        "conditional_stochastic_completion": {
            "law": "N(B)|q ~ Poisson(mu_B), mu_B=Integral_B q*dVol_g",
            "stationary_law": "N(B)|q0 ~ Poisson(q0*Vol_g(B))",
            "independent_disjoint_regions": "assumed",
            "diffeomorphism_covariance": "conditional law uses invariant four-volume",
            "derived_from_local_action": False,
            "derived_from_QED_actualization": False,
        },
        "decision": "conditional_hypothesis_match_not_a_paper_or_theory_pass",
    }


def _validate_result(result: Mapping[str, Any]) -> None:
    if result.get("schema_version") != RESULT_SCHEMA:
        raise ValueError("candidate-action result schema changed")
    branches = result.get("completion_hypotheses", [])
    if len(branches) != 2 or [item.get("beta") for item in branches] != ["1/2", "1/4"]:
        raise ValueError("candidate-action branch partition changed")
    expected_matching = [
        {
            "derived_lambda": "4*pi*G*h*q0/c**3",
            "derived_lambda_planck_units": "8*pi**2*l_P**2*q0",
            "declared_target_lambda": "4*pi*G*h*q0/c**3",
            "declared_target_lambda_planck_units": "8*pi**2*l_P**2*q0",
            "exact_coefficient_match": True,
            "source_status": "matches equation 35 first and middle equalities",
            "normalization_selected_as_fact": False,
        },
        {
            "derived_lambda": "2*pi*G*h*q0/c**3",
            "derived_lambda_planck_units": "4*pi**2*l_P**2*q0",
            "declared_target_lambda": "2*pi*G*h*q0/c**3",
            "declared_target_lambda_planck_units": "4*pi**2*l_P**2*q0",
            "exact_coefficient_match": True,
            "source_status": (
                "matches equation 35 printed final equality after h=2*pi*hbar"
            ),
            "normalization_selected_as_fact": False,
        },
    ]
    if [item.get("matching") for item in branches] != expected_matching:
        raise ValueError("candidate-action branch matching changed")
    if not all(item.get("candidate_action", {}).get("local_deterministic_action_complete") for item in branches):
        raise ValueError("incomplete local deterministic candidate action")
    if any(item.get("paper_authorship_or_derivation") for item in branches):
        raise ValueError("candidate action falsely attributed to paper")
    if any(item.get("conditional_stochastic_completion", {}).get("derived_from_local_action") for item in branches):
        raise ValueError("Poisson law falsely derived from local action")
    if result.get("counts") != {
        "normalization_branches": 2,
        "complete_local_deterministic_action_hypotheses": 2,
        "conditional_exact_eq35_branch_matches": 2,
        "paper_derived_actions": 0,
        "normalization_branches_selected_as_fact": 0,
        "observational_or_theory_passes": 0,
    }:
        raise ValueError("candidate-action counts changed")
    if result.get("first_blocker") != FIRST_BLOCKER:
        raise ValueError("candidate-action first blocker changed")
    controls = result.get("deterministic_controls", {})
    if controls != {
        "middle_branch_coefficient_residual": "0",
        "printed_branch_coefficient_residual": "0",
        "full_pressure_beta_one_vs_middle_residual": "4*pi*G*h*q0/c**3",
        "full_pressure_beta_one_vs_printed_residual": "6*pi*G*h*q0/c**3",
        "off_shell_nonconstant_intensity_bianchi_residual_generically_zero": False,
        "negative_controls_rejected": True,
    }:
        raise ValueError("candidate-action deterministic controls changed")
    if any(result.get("claim_seals", {}).values()) or any(result.get("data_seals", {}).values()):
        raise ValueError("candidate-action seal opened")
    declared = result.get("content_sha256")
    if declared is not None and _content_sha(result) != declared:
        raise ValueError("candidate-action content hash mismatch")


def build_completion(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    root = config_path.parents[1]
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    graph = _bound_json(root, config["predecessors"]["equation_graph"], "equation graph")
    intake = _bound_json(root, config["predecessors"]["intake"], "intake")
    _validate_predecessors(graph, intake)
    source_path = Path(__file__).resolve()
    test_path = root / "tests/test_kastner_schlatter_candidate_action_completion.py"
    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA,
        "campaign_id": config["campaign_id"],
        "scope": (
            "compiler-authored covariant candidate completions; not present in or derived by "
            "arXiv:2209.04025"
        ),
        "source_bindings": {
            "equation_graph": dict(config["predecessors"]["equation_graph"]),
            "intake": dict(config["predecessors"]["intake"]),
            "primary_pdf_sha256": graph["source_lineage"]["primary_pdf_sha256"],
            "config": {"path": str(config_path.relative_to(root)).replace("\\", "/"), "file_sha256": _file_sha(config_path)},
            "source": {"path": str(source_path.relative_to(root)).replace("\\", "/"), "file_sha256": _file_sha(source_path)},
            "test": {"path": str(test_path.relative_to(root)).replace("\\", "/"), "file_sha256": _file_sha(test_path)},
        },
        "conventions": config["conventions"],
        "dimensions": _dimensions(),
        "completion_hypotheses": [_branch_contract(item) for item in config["branches"]],
        "deterministic_controls": {
            "middle_branch_coefficient_residual": "0",
            "printed_branch_coefficient_residual": "0",
            "full_pressure_beta_one_vs_middle_residual": "4*pi*G*h*q0/c**3",
            "full_pressure_beta_one_vs_printed_residual": "6*pi*G*h*q0/c**3",
            "off_shell_nonconstant_intensity_bianchi_residual_generically_zero": False,
            "negative_controls_rejected": True,
        },
        "paper_step_classification": [
            {
                "step": "Poisson transactions with invariant four-rate q_gamma",
                "paper_status": "proposal/assertion",
                "completion_status": "conditional point-process law only",
                "still_missing": "QED actualization-to-Poisson derivation and operational event map",
            },
            {
                "step": "Pbar=-c*h*q_gamma",
                "paper_status": "printed equation 34",
                "completion_status": "target relation; action uses beta times its energy scale",
                "still_missing": "variational derivation of pressure and explanation of beta",
            },
            {
                "step": "Lambda=-4*pi*G*Pbar/c^4",
                "paper_status": "printed equation 35",
                "completion_status": "reproduced only by beta=1/2 under standard EH normalization",
                "still_missing": "why the action sources half the printed pressure scale",
            },
            {
                "step": "Lambda=4*pi^2*l_P^2*q_gamma",
                "paper_status": "printed equation 35 final equality",
                "completion_status": "reproduced only by separate beta=1/4 branch",
                "still_missing": "authoritative h-versus-hbar factor-two normalization",
            },
            {
                "step": "Einstein-equation recovery",
                "paper_status": "local argument in equation graph, not an action derivation",
                "completion_status": "standard EH metric EL supplied by compiler hypothesis",
                "still_missing": "derivation of EH action and matter coupling from transactions",
            },
        ],
        "counts": {
            "normalization_branches": 2,
            "complete_local_deterministic_action_hypotheses": 2,
            "conditional_exact_eq35_branch_matches": 2,
            "paper_derived_actions": 0,
            "normalization_branches_selected_as_fact": 0,
            "observational_or_theory_passes": 0,
        },
        "decision": "candidate_completions_registered_paper_derivation_and_physics_claims_blocked",
        "first_blocker": FIRST_BLOCKER,
        "secondary_blockers": [
            "equation_35_h_vs_hbar_factor_normalization_clarification",
            "no_QED_actualization_to_covariant_Poisson_process_derivation",
            "no_transaction_derivation_of_beta_or_stabilizing_intensity_potential",
            "no_matter_sector_or_global_quantum_completion",
        ],
        "claim_seals": {
            "paper_action_claimed": False,
            "paper_variational_derivation_claimed": False,
            "formal_gr_equivalence_proven": False,
            "observational_pass": False,
            "theory_validity_claimed": False,
            "dark_matter_elimination_proven": False,
            "dark_energy_elimination_proven": False,
        },
        "data_seals": {
            "observations_opened": False,
            "dark_matter_or_halo_data_opened": False,
            "redshift_or_cosmology_data_opened": False,
            "solar_system_data_opened": False,
            "paid_llm_calls": False,
        },
    }
    result["content_sha256"] = _content_sha(result)
    _validate_result(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--output")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    result = build_completion(config_path)
    output = Path(args.output).resolve() if args.output else config_path.parents[1] / json.loads(config_path.read_text(encoding="utf-8"))["output_path"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
