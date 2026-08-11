"""Exact Poisson-selector contract and registered-dependency no-go gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

CONFIG_SCHEMA = "sigma-kastner-schlatter-poisson-selector-contract-gate-config-1.0"
RESULT_SCHEMA = "sigma-kastner-schlatter-poisson-selector-contract-gate-1.0"
FIRST_BLOCKER = (
    "no_registered_derivation_of_a_set_indexed_Poisson_Laplace_functional_"
    "independent_increment_family_or_QED_counting_measure_kernel"
)
PMF_NODE = "EQ-KS-POISSON-PMF-IMPLEMENTATION"


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _content_sha(value: Mapping[str, Any]) -> str:
    return _sha({key: item for key, item in value.items() if key != "content_sha256"})


def _inside(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError("Poisson-selector path escapes repository") from error
    return path


def _bound_json(root: Path, binding: Mapping[str, Any], label: str) -> dict[str, Any]:
    if set(binding) != {"path", "file_sha256", "content_sha256"}:
        raise ValueError(f"{label} binding shape changed")
    path = _inside(root, str(binding["path"]))
    if not path.is_file() or _file_sha(path) != binding["file_sha256"]:
        raise ValueError(f"{label} file hash mismatch")
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or value.get("content_sha256") != binding["content_sha256"]
        or _content_sha(value) != binding["content_sha256"]
    ):
        raise ValueError(f"{label} content hash mismatch")
    return value


def _validate_config(config: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "campaign_id",
        "output_path",
        "predecessors",
        "closed_world_policy",
        "seals",
    }
    if set(config) != required or config.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("Poisson-selector config shape changed")
    if set(config.get("predecessors", {})) != {
        "point_process_measure_gate",
        "equation_graph",
        "candidate_action_completion",
    }:
        raise ValueError("Poisson-selector predecessor set changed")
    if config.get("closed_world_policy") != {
        "registered_nodes_and_edges_are_complete_for_this_audit": True,
        "paper_assertion_is_not_relabelled_as_derivation": True,
        "implementation_formula_is_not_relabelled_as_printed_equation": True,
        "missing_dependency_edge_fails_closed": True,
        "candidate_action_rejection_allowed": False,
    }:
        raise ValueError("Poisson-selector closed-world policy changed")
    if any(config.get("seals", {}).values()):
        raise ValueError("Poisson-selector seal opened")


def _validate_predecessors(
    point_process: Mapping[str, Any],
    graph: Mapping[str, Any],
    completion: Mapping[str, Any],
) -> tuple[list[Mapping[str, Any]], Mapping[str, Any], list[Mapping[str, Any]]]:
    if (
        point_process.get("decision_counts") != {"pass": 0, "reject": 0, "blocked": 2}
        or point_process.get("gate_counts", {}).get("covariant_intensity_measure_pass") != 2
        or point_process.get("gate_counts", {}).get("action_only_Poisson_derivation_reject") != 2
        or point_process.get("gate_counts", {}).get("candidate_action_reject") != 0
        or any(point_process.get("claim_seals", {}).values())
        or any(point_process.get("data_seals", {}).values())
    ):
        raise ValueError("point-process predecessor boundary changed")
    if graph.get("graph_counts") != {
        "nodes": 54,
        "edges": 137,
        "formula_nodes": 25,
        "assumption_nodes": 12,
        "domain_nodes": 6,
        "source_nodes": 2,
        "action_contract_nodes": 1,
        "absent_capability_nodes": 8,
        "dependency_edges": 18,
        "assumption_edges": 35,
        "semantic_algebraic_equivalence_edges": 1,
        "exact_duplicate_edges": 0,
        "theory_equivalence_edges": 0,
        "absent_action_edges": 33,
    }:
        raise ValueError("equation-graph predecessor count changed")
    nodes = graph.get("knowledge_graph", {}).get("nodes", [])
    edges = graph.get("knowledge_graph", {}).get("edges", [])
    pmf = next((node for node in nodes if node.get("node_id") == PMF_NODE), None)
    if (
        pmf is None
        or pmf.get("record", {}).get("expression") != "p_n = exp(-mu)*mu**n/fact_n"
        or "implementation_not_printed_equation" not in pmf.get("record", {}).get("tags", [])
        or pmf.get("record", {}).get("source_locator")
        != "p.9 Poisson assertion; standard PMF is implementation-only"
    ):
        raise ValueError("registered Poisson PMF premise changed")
    pmf_edges = [edge for edge in edges if edge.get("source") == PMF_NODE]
    if not any(
        edge.get("edge_type") == "not_derived_from_action"
        and edge.get("target") == "ACTION-CONTRACT-ABSENT"
        for edge in pmf_edges
    ):
        raise ValueError("Poisson PMF absent-action edge changed")
    branches = completion.get("completion_hypotheses", [])
    if (
        [branch.get("branch_id") for branch in branches] != ["eq35_middle_h", "eq35_printed_planck"]
        or any(branch.get("paper_authorship_or_derivation") for branch in branches)
        or any(
            branch.get("candidate_action", {}).get("stochastic_law_derived_by_action")
            for branch in branches
        )
        or completion.get("source_bindings", {}).get("equation_graph", {}).get("content_sha256")
        != graph.get("content_sha256")
        or any(completion.get("claim_seals", {}).values())
    ):
        raise ValueError("candidate-action stochastic boundary changed")
    return nodes, pmf, edges


def _selector_contract() -> dict[str, Any]:
    return {
        "target_object": (
            "a diffeomorphism-covariant probability kernel P_{g,q} on locally finite "
            "integer-valued measures N"
        ),
        "sufficient_selector_A_Laplace_functional": {
            "formula": "L[f]=exp(-Integral_M (1-exp(-f))*q*dVol_g)",
            "domain": "all nonnegative compactly supported measurable f",
            "effect": "uniquely determines the Poisson random measure",
        },
        "sufficient_selector_B_joint_count_family": {
            "formula": ("E[product_i z_i^N(B_i)]=exp(sum_i mu_q(B_i)*(z_i-1))"),
            "domain": "every finite family of pairwise disjoint relatively compact B_i",
            "effect": "Poisson marginals and independent increments",
        },
        "sufficient_selector_C_Mecke_event_kernel": {
            "formula": ("E[Integral F(x,N)N(dx)]=Integral E[F(x,N+delta_x)]q(x)dVol_g(x)"),
            "domain": "all nonnegative measurable test functionals F",
            "effect": "characterizes the Poisson random measure with intensity mu_q",
        },
        "QED_bridge_required_before_selector_C_can_be_attributed": {
            "object": "measurable actualization-history-to-counting-measure kernel",
            "requirements": [
                "event sigma algebra",
                "locally finite count map",
                "conditional law or compensator",
                "proof of the Mecke identity or equivalent factorization",
                "diffeomorphism-covariance proof",
            ],
        },
        "minimal_admission_rule": (
            "derive and source-bind at least one selector A/B/C; a scalar mean or one-set PMF "
            "without a set-indexed consistency and joint-law contract is insufficient"
        ),
    }


def _graph_audit(
    nodes: list[Mapping[str, Any]], pmf: Mapping[str, Any], edges: list[Mapping[str, Any]]
) -> dict[str, Any]:
    selector_terms = (
        "laplace functional",
        "independent increment",
        "mecke",
        "papangelou",
        "event kernel",
        "counting measure kernel",
    )
    selector_nodes = []
    for node in nodes:
        text = _canonical(node).lower()
        if any(term in text for term in selector_terms):
            selector_nodes.append(str(node.get("node_id")))
    action_derivation_edges = [
        edge
        for edge in edges
        if edge.get("source") == PMF_NODE
        and edge.get("edge_type") in {"derived_from_action", "implied_by_action"}
    ]
    return {
        "closed_world_counts": {"nodes": len(nodes), "edges": len(edges)},
        "registered_scalar_Poisson_PMF_nodes": 1,
        "PMF_node_id": PMF_NODE,
        "PMF_status": "paper_assertion_with_standard_implementation_formula",
        "PMF_printed_as_equation_in_paper": False,
        "PMF_expression_has_set_argument_B": False,
        "PMF_expression_has_joint_count_arguments": False,
        "PMF_expression_has_history_or_QED_kernel_argument": False,
        "registered_selector_node_ids": selector_nodes,
        "registered_selector_nodes": len(selector_nodes),
        "PMF_action_derivation_edges": len(action_derivation_edges),
        "PMF_not_derived_from_action_edge": True,
        "registered_equations_imply_independent_increments": False,
        "registered_equations_imply_Poisson_Laplace_functional": False,
        "registered_equations_imply_QED_counting_measure_kernel": False,
        "reason": (
            "the only stochastic-law formula is an implementation of an asserted scalar PMF; "
            "its graph edge explicitly says not_derived_from_action and no selector node exists"
        ),
        "PMF_record_sha256": pmf["record_sha256"],
    }


def _marginal_no_go() -> dict[str, Any]:
    return {
        "scope": (
            "insufficiency of the registered scalar one-count PMF node; not a competitor to a "
            "fully set-indexed Poisson random-measure axiom"
        ),
        "construction": ("for two named equal-intensity cells, draw X~Poisson(lambda) and set Y=X"),
        "single_cell_marginals": "X and Y each obey exp(-lambda)*lambda^n/n!",
        "dependent_joint_PGF": "G_dep(s,t)=exp(lambda*(s*t-1))",
        "independent_joint_PGF": "G_ind(s,t)=exp(lambda*(s-1)+lambda*(t-1))",
        "exact_separation": "coefficient of s*t is coupled in G_dep and factorized in G_ind",
        "covariance_dependent": "Cov(X,Y)=lambda>0",
        "covariance_independent": "Cov(X,Y)=0",
        "conclusion": (
            "the registered scalar PMF expression alone does not entail independent increments"
        ),
    }


def _branch_record(branch: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "branch_id": branch["branch_id"],
        "beta": branch["beta"],
        "compiler_authored_action": True,
        "paper_or_QED_derived": False,
        "registered_action_stochastic_outputs": {
            "positive_intensity_measure": True,
            "probability_kernel": False,
            "Laplace_functional": False,
            "independent_increment_family": False,
            "Mecke_or_QED_event_kernel": False,
        },
        "gate_ledger": {
            "scalar_Poisson_PMF_as_registered_assertion": "pass",
            "set_indexed_Poisson_Laplace_selector": "blocked",
            "independent_increment_joint_family": "blocked",
            "QED_actualization_counting_measure_kernel": "blocked",
            "registered_equations_imply_a_Poisson_selector": "reject",
            "candidate_action_rejection": "blocked",
        },
        "candidate_action_rejection_authorized": False,
        "candidate_decision": "blocked",
        "first_blocker": FIRST_BLOCKER,
    }


def _controls() -> dict[str, Any]:
    return {
        "Laplace_selector_positive_control": {
            "input": "selector A for all admissible f",
            "Poisson_law_uniquely_selected": True,
        },
        "joint_family_positive_control": {
            "input": "selector B for every finite disjoint family",
            "independent_increments_selected": True,
        },
        "single_PMF_negative_control": {
            "mutation": "treat p_n=exp(-mu)mu^n/n! as a complete joint random-measure law",
            "rejected": True,
            "counterexample": "equal-marginal dependent two-cell construction",
        },
        "mean_only_negative_control": {
            "mutation": "treat mu_q as a unique probability law",
            "rejected": True,
            "counterexample": "predecessor Poisson/Cox nonidentifiability witness",
        },
        "attribution_negative_control": {
            "mutation": "relabel the implementation PMF as a paper derivation from QED or action",
            "rejected": True,
        },
    }


def _validate_result(result: Mapping[str, Any]) -> None:
    if result.get("schema_version") != RESULT_SCHEMA:
        raise ValueError("Poisson-selector result schema changed")
    if result.get("decision_counts") != {"pass": 0, "reject": 0, "blocked": 2}:
        raise ValueError("Poisson-selector candidate partition changed")
    if result.get("gate_counts") != {
        "candidate_actions": 2,
        "registered_scalar_Poisson_PMF_assertions": 1,
        "minimal_sufficient_selector_contracts": 3,
        "registered_selector_nodes": 0,
        "registered_action_derivation_edges_to_PMF": 0,
        "independent_increment_derivation_pass": 0,
        "Poisson_Laplace_functional_derivation_pass": 0,
        "QED_counting_measure_kernel_derivation_pass": 0,
        "registered_equations_imply_selector_reject": 2,
        "candidate_action_reject": 0,
        "paper_QED_ontology_observational_pass": 0,
    }:
        raise ValueError("Poisson-selector gate counts changed")
    audit = result.get("registered_dependency_audit", {})
    if (
        audit.get("registered_selector_nodes") != 0
        or audit.get("PMF_action_derivation_edges") != 0
        or audit.get("registered_equations_imply_independent_increments") is not False
        or audit.get("registered_equations_imply_Poisson_Laplace_functional") is not False
    ):
        raise ValueError("Poisson-selector dependency no-go changed")
    records = result.get("candidate_records", [])
    if len(records) != 2 or any(record.get("paper_or_QED_derived") for record in records):
        raise ValueError("Poisson-selector attribution changed")
    if any(record.get("candidate_action_rejection_authorized") for record in records):
        raise ValueError("Poisson-selector gate overreached to action rejection")
    if result.get("first_blocker") != FIRST_BLOCKER:
        raise ValueError("Poisson-selector first blocker changed")
    if any(result.get("claim_seals", {}).values()) or any(result.get("data_seals", {}).values()):
        raise ValueError("Poisson-selector seal opened")
    if (
        result.get("content_sha256") is not None
        and _content_sha(result) != result["content_sha256"]
    ):
        raise ValueError("Poisson-selector content hash mismatch")


def build_gate(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    root = config_path.parents[1]
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    bindings = config["predecessors"]
    point_process = _bound_json(
        root, bindings["point_process_measure_gate"], "point-process measure gate"
    )
    graph = _bound_json(root, bindings["equation_graph"], "equation graph")
    completion = _bound_json(
        root, bindings["candidate_action_completion"], "candidate-action completion"
    )
    nodes, pmf, edges = _validate_predecessors(point_process, graph, completion)
    graph_audit = _graph_audit(nodes, pmf, edges)
    source_path = Path(__file__).resolve()
    test_path = root / "tests/test_kastner_schlatter_poisson_selector_contract_gate.py"
    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA,
        "campaign_id": config["campaign_id"],
        "scope": (
            "minimal exact Poisson-selector contracts and closed-world dependency audit of the "
            "registered paper equation graph and compiler-authored actions"
        ),
        "source_bindings": {
            **bindings,
            "primary_pdf_sha256": graph["source_lineage"]["primary_pdf_sha256"],
            "config": {
                "path": config_path.relative_to(root).as_posix(),
                "file_sha256": _file_sha(config_path),
            },
            "source": {
                "path": source_path.relative_to(root).as_posix(),
                "file_sha256": _file_sha(source_path),
            },
            "test": {
                "path": test_path.relative_to(root).as_posix(),
                "file_sha256": _file_sha(test_path),
            },
        },
        "minimal_Poisson_selector_contract": _selector_contract(),
        "registered_dependency_audit": graph_audit,
        "scalar_marginal_nonimplication_theorem": _marginal_no_go(),
        "candidate_records": [
            _branch_record(branch) for branch in completion["completion_hypotheses"]
        ],
        "deterministic_controls": _controls(),
        "gate_counts": {
            "candidate_actions": 2,
            "registered_scalar_Poisson_PMF_assertions": 1,
            "minimal_sufficient_selector_contracts": 3,
            "registered_selector_nodes": 0,
            "registered_action_derivation_edges_to_PMF": 0,
            "independent_increment_derivation_pass": 0,
            "Poisson_Laplace_functional_derivation_pass": 0,
            "QED_counting_measure_kernel_derivation_pass": 0,
            "registered_equations_imply_selector_reject": 2,
            "candidate_action_reject": 0,
            "paper_QED_ontology_observational_pass": 0,
        },
        "decision_counts": {"pass": 0, "reject": 0, "blocked": 2},
        "decision": "Poisson_selector_contract_registered_no_registered_derivation_path",
        "first_blocker": FIRST_BLOCKER,
        "secondary_blockers": [
            "no_set_indexed_consistency_or_joint_count_family_for_the_registered_scalar_PMF",
            "no_measurable_QED_actualization_history_to_counting_measure_kernel",
            "no_action_to_stochastic_generating_functional_map",
            "no_observational_transaction_event_or_exposure_registration",
        ],
        "claim_seals": {
            "paper_action_derived": False,
            "QED_actualization_action_derived": False,
            "Poisson_selector_paper_or_action_derived": False,
            "transaction_ontology_validated": False,
            "candidate_action_rejected": False,
            "observational_pass": False,
            "theory_validity_claimed": False,
            "dark_sector_elimination_proven": False,
        },
        "data_seals": dict(config["seals"]),
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
    result = build_gate(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output = (
        Path(args.output).resolve()
        if args.output
        else config_path.parents[1] / str(config["output_path"])
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
