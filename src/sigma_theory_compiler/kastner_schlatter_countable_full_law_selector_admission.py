"""Countable full-law Poisson-selector admission for KS candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

CONFIG_SCHEMA = "sigma-kastner-schlatter-countable-full-law-selector-admission-config-1.0"
RESULT_SCHEMA = "sigma-kastner-schlatter-countable-full-law-selector-admission-1.0"
PDF_SHA256 = "c2f671293d07b21397e745da00a3ce1a2193c00da647a2ebf4147612b76c1780"
FIRST_BLOCKER = (
    "no_source_bound_countable_determining_Laplace_or_Mecke_certificate_for_"
    "the_actualization_random_measure"
)
BRANCHES = [("eq35_middle_h", "1/2"), ("eq35_printed_planck", "1/4")]

EXPECTED_COUNTS = {
    "candidate_actions": 2,
    "candidate_blocked": 2,
    "candidate_pass": 0,
    "candidate_reject": 0,
    "mathematically_sufficient_countable_routes": 2,
    "compiler_route_replays": 4,
    "source_or_QED_route_certificates": 0,
    "typed_obligations": 12,
    "closed_by_compiler_mathematics": 6,
    "absent_from_source_QED_or_action": 6,
    "registered_selector_nodes": 0,
    "registered_action_derivation_edges_to_PMF": 0,
    "paper_QED_ontology_observational_pass": 0,
}

EXPECTED_CLAIM_SEALS = {
    "paper_QED_countable_selector_registered": False,
    "candidate_action_stochastic_law_derived": False,
    "candidate_action_selects_Poisson": False,
    "candidate_action_rejected": False,
    "compiler_replay_attributed_to_source": False,
    "scalar_PMF_claimed_as_full_law_certificate": False,
    "detector_event_equals_transaction_proven": False,
    "transaction_ontology_validated": False,
    "observational_pass": False,
    "scientific_test_pass": False,
    "theory_validity_claimed": False,
    "dark_sector_elimination_proven": False,
}


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
        raise ValueError("countable-selector path escapes repository") from error
    return path


def _bound_artifact(root: Path, binding: Mapping[str, Any]) -> dict[str, Any]:
    if set(binding) != {"path", "file_sha256", "content_sha256"}:
        raise ValueError("countable-selector predecessor binding shape changed")
    path = _inside(root, str(binding["path"]))
    if not path.is_file() or _file_sha(path) != binding["file_sha256"]:
        raise ValueError("countable-selector predecessor file hash mismatch")
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or value.get("content_sha256") != binding["content_sha256"]
        or _content_sha(value) != binding["content_sha256"]
    ):
        raise ValueError("countable-selector predecessor content hash mismatch")
    return value


def _validate_config(config: Mapping[str, Any]) -> None:
    if (
        set(config)
        != {
            "schema_version",
            "campaign_id",
            "output_path",
            "predecessors",
            "admission_domain",
            "admission_policy",
            "seals",
        }
        or config.get("schema_version") != CONFIG_SCHEMA
    ):
        raise ValueError("countable-selector config shape changed")
    if set(config["predecessors"]) != {
        "finite_factorial_hierarchy_no_go",
        "poisson_selector_contract",
        "canonical_probability_space",
        "equation_graph",
        "qed_actualization_audit",
        "actualization_history_map",
    }:
        raise ValueError("countable-selector predecessor set changed")
    if config.get("admission_domain") != {
        "spacetime": "second-countable regular patch W",
        "intensity": "diffuse Radon mu_g_phi(dx)=q0*exp(phi(x))*dVol_g(x)",
        "configuration_space": "N_lf(W) with evaluation sigma algebra",
        "determining_ring": (
            "countable ring R of relatively compact mu-continuity sets generating Borel(W)"
        ),
        "laplace_core": (
            "nonnegative rational simple functions over finite disjoint families in R"
        ),
        "mecke_core": ("nonnegative rational cylinder functions times indicators of sets in R"),
    }:
        raise ValueError("countable-selector domain changed")
    if config.get("admission_policy") != {
        "either_countable_core_route_is_mathematically_sufficient": True,
        "compiler_replay_counts_as_source_or_QED_derivation": False,
        "scalar_one_set_PMF_counts_as_a_core_certificate": False,
        "candidate_action_rejection_allowed": False,
    }:
        raise ValueError("countable-selector admission policy changed")
    if not config.get("seals") or any(config["seals"].values()):
        raise ValueError("countable-selector seal opened")


def _validate_predecessors(predecessors: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    finite = predecessors["finite_factorial_hierarchy_no_go"]
    records = finite.get("candidate_records", [])
    if (
        finite.get("decision_counts") != {"blocked": 2, "pass": 0, "reject": 0}
        or [(row.get("branch_id"), row.get("beta")) for row in records] != BRANCHES
        or any(
            row.get("arbitrary_finite_factorial_hierarchy_no_go") != "pass"
            or row.get("registered_nonfinite_selector") is not False
            or row.get("candidate_action_rejection_authorized") is not False
            for row in records
        )
    ):
        raise ValueError("finite-hierarchy predecessor boundary changed")
    selector = predecessors["poisson_selector_contract"].get("registered_dependency_audit", {})
    if (
        selector.get("registered_selector_nodes") != 0
        or selector.get("PMF_action_derivation_edges") != 0
        or selector.get("registered_equations_imply_Poisson_Laplace_functional") is not False
        or selector.get("registered_equations_imply_QED_counting_measure_kernel") is not False
        or selector.get("closed_world_counts") != {"edges": 137, "nodes": 54}
    ):
        raise ValueError("registered selector dependency audit changed")
    canonical = predecessors["canonical_probability_space"].get(
        "canonical_conditional_construction", {}
    )
    if (
        canonical.get("paper_or_QED_supplies_this_probability_space") is not False
        or canonical.get("candidate_action_selects_this_probability_space") is not False
    ):
        raise ValueError("canonical probability attribution changed")
    graph_counts = predecessors["equation_graph"].get("graph_counts", {})
    if graph_counts.get("nodes") != 54 or graph_counts.get("edges") != 137:
        raise ValueError("equation graph closed-world counts changed")
    qed_records = predecessors["qed_actualization_audit"].get("candidate_records", [])
    history_records = predecessors["actualization_history_map"].get("candidate_records", [])
    if (
        [(row.get("branch_id"), row.get("beta")) for row in qed_records] != BRANCHES
        or any(
            row.get("paper_or_QED_channel_kernel_registered") is not False for row in qed_records
        )
        or [(row.get("branch_id"), row.get("beta")) for row in history_records] != BRANCHES
        or any(row.get("paper_or_QED_kernel_selected") is not False for row in history_records)
    ):
        raise ValueError("QED/history selector evidence changed")
    return records


def _admission_theorem() -> dict[str, Any]:
    return {
        "laplace_core_route": {
            "certificate": (
                "for every f=sum_i r_i*1_A_i with rational r_i>=0 and disjoint A_i in R, "
                "E[exp(-Integral f*dN)]=exp(-Integral(1-exp(-f))*dmu)"
            ),
            "extension": (
                "continuity in rational coefficients, ring approximation on mu-continuity "
                "sets, and monotone convergence extend the identity to all nonnegative "
                "compactly supported measurable f"
            ),
            "selection": (
                "simple functions recover every finite disjoint-family joint Laplace transform; "
                "these determine the law on the evaluation sigma algebra, hence N~PRM(mu)"
            ),
            "mathematically_sufficient": True,
        },
        "mecke_core_route": {
            "certificate": (
                "E[Integral H(x,N)N(dx)]=Integral E[H(x,N+delta_x)]mu(dx) for every "
                "nonnegative rational cylinder H on the countable core"
            ),
            "extension": (
                "a functional monotone-class argument extends the identity to all nonnegative "
                "measurable H"
            ),
            "selection": (
                "with H(x,N)=f(x)exp(-t*Integral f*dN), the identity gives the Laplace ODE; "
                "its unique solution is the PRM Laplace functional"
            ),
            "mathematically_sufficient": True,
        },
        "countability_role": (
            "R, rational coefficients, and rational cylinder functions make each certificate "
            "a countable, hashable compiler object without weakening full-law determination"
        ),
        "scope_limit": (
            "the extension assumes the declared Radon/diffuse regularity and locally finite law; "
            "a compiler-authored certificate is not a paper, QED, or action derivation"
        ),
    }


def _evidence_ledger() -> list[dict[str, str]]:
    return [
        {"obligation": "locally_finite_configuration_space", "status": "closed_by_compiler"},
        {"obligation": "diffuse_Radon_candidate_intensity", "status": "closed_by_compiler"},
        {"obligation": "countable_continuity_ring", "status": "closed_by_compiler"},
        {"obligation": "rational_simple_Laplace_core", "status": "closed_by_compiler"},
        {"obligation": "rational_cylinder_Mecke_core", "status": "closed_by_compiler"},
        {"obligation": "monotone_class_full_law_extension", "status": "closed_by_compiler"},
        {"obligation": "paper_typed_counting_random_measure", "status": "absent"},
        {"obligation": "paper_source_quantification_over_a_determining_core", "status": "absent"},
        {"obligation": "source_bound_Laplace_core_identity", "status": "absent"},
        {"obligation": "source_bound_add_one_event_kernel", "status": "absent"},
        {"obligation": "QED_Mecke_core_identity", "status": "absent"},
        {"obligation": "action_to_full_law_selector_edge", "status": "absent"},
    ]


def _exact_controls() -> dict[str, Any]:
    return {
        "two_cell_Laplace_core_positive_control": {
            "cell_measures": ["2", "3"],
            "rational_coefficients": ["1", "2"],
            "joint_exponent": "-2*(1-exp(-1))-3*(1-exp(-2))",
            "factorized_exponent": "-2*(1-exp(-1))-3*(1-exp(-2))",
            "pass": True,
        },
        "Mecke_to_Laplace_ODE_positive_control": {
            "cell_measure": "2",
            "test_function": "f=1_B",
            "ODE": "L'(t)=-2*exp(-t)*L(t)",
            "initial_condition": "L(0)=1",
            "unique_solution": "L(t)=exp(-2*(1-exp(-t)))",
            "pass": True,
        },
        "scalar_PMF_negative_control": {
            "registered_node": "EQ-KS-POISSON-PMF-IMPLEMENTATION",
            "has_set_argument": False,
            "has_joint_count_arguments": False,
            "has_history_or_QED_kernel_argument": False,
            "qualifies_as_Laplace_core_certificate": False,
            "qualifies_as_Mecke_core_certificate": False,
        },
        "compiler_attribution_negative_control": {
            "canonical_PRM_construction_exists": True,
            "paper_or_QED_supplies_certificate": False,
            "candidate_action_selects_certificate": False,
        },
    }


def _candidate_record(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "branch_id": record["branch_id"],
        "beta": record["beta"],
        "compiler_countable_Laplace_route_replay": "pass",
        "compiler_countable_Mecke_route_replay": "pass",
        "source_bound_countable_selector_certificate": False,
        "paper_or_QED_selector_derived": False,
        "candidate_action_selects_Poisson": False,
        "candidate_action_rejection_authorized": False,
        "candidate_decision": "blocked",
        "first_blocker": FIRST_BLOCKER,
    }


def _validate_result(result: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "campaign_id",
        "scope",
        "source_bindings",
        "admission_domain",
        "countable_selector_admission_theorem",
        "evidence_ledger",
        "closed_world_source_audit",
        "exact_controls",
        "candidate_records",
        "gate_counts",
        "decision_counts",
        "decision",
        "first_blocker",
        "secondary_blockers",
        "claim_seals",
        "data_seals",
        "content_sha256",
    }
    if set(result) != required or result.get("schema_version") != RESULT_SCHEMA:
        raise ValueError("countable-selector result shape changed")
    if result.get("decision_counts") != {"pass": 0, "reject": 0, "blocked": 2}:
        raise ValueError("countable-selector candidate partition changed")
    if result.get("gate_counts") != EXPECTED_COUNTS:
        raise ValueError("countable-selector gate counts changed")
    if result.get("countable_selector_admission_theorem") != _admission_theorem():
        raise ValueError("countable-selector theorem changed")
    if result.get("evidence_ledger") != _evidence_ledger():
        raise ValueError("countable-selector evidence ledger changed")
    if result.get("exact_controls") != _exact_controls():
        raise ValueError("countable-selector controls changed")
    audit = result.get("closed_world_source_audit", {})
    if audit != {
        "equation_graph_nodes": 54,
        "equation_graph_edges": 137,
        "registered_scalar_PMF_nodes": 1,
        "registered_selector_nodes": 0,
        "action_derivation_edges_to_PMF": 0,
        "paper_typed_history_maps": 0,
        "QED_channel_kernels": 0,
        "Laplace_core_certificates": 0,
        "Mecke_core_certificates": 0,
        "audit_conclusion": "neither countable determining route has a source-bound premise",
    }:
        raise ValueError("countable-selector closed-world audit changed")
    records = result.get("candidate_records", [])
    if [(row.get("branch_id"), row.get("beta")) for row in records] != BRANCHES or any(
        row.get("compiler_countable_Laplace_route_replay") != "pass"
        or row.get("compiler_countable_Mecke_route_replay") != "pass"
        or row.get("source_bound_countable_selector_certificate") is not False
        or row.get("candidate_action_selects_Poisson") is not False
        or row.get("candidate_action_rejection_authorized") is not False
        or row.get("candidate_decision") != "blocked"
        or row.get("first_blocker") != FIRST_BLOCKER
        for row in records
    ):
        raise ValueError("countable-selector candidate boundary changed")
    if result.get("first_blocker") != FIRST_BLOCKER:
        raise ValueError("countable-selector blocker changed")
    if result.get("claim_seals") != EXPECTED_CLAIM_SEALS or any(result["data_seals"].values()):
        raise ValueError("countable-selector seal changed")
    if (
        result.get("content_sha256") is not None
        and _content_sha(result) != result["content_sha256"]
    ):
        raise ValueError("countable-selector content hash mismatch")


def build_gate(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    root = config_path.parents[1]
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    predecessors = {
        label: _bound_artifact(root, binding) for label, binding in config["predecessors"].items()
    }
    records = _validate_predecessors(predecessors)
    source_path = Path(__file__).resolve()
    test_path = root / "tests/test_kastner_schlatter_countable_full_law_selector_admission.py"
    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA,
        "campaign_id": config["campaign_id"],
        "scope": (
            "candidate-bound countable determining Laplace/Mecke admission contract and "
            "closed-world source audit; no paper/QED law, rejection, ontology, or observation inferred"
        ),
        "source_bindings": {
            **config["predecessors"],
            "primary_pdf_sha256": PDF_SHA256,
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
        "admission_domain": config["admission_domain"],
        "countable_selector_admission_theorem": _admission_theorem(),
        "evidence_ledger": _evidence_ledger(),
        "closed_world_source_audit": {
            "equation_graph_nodes": 54,
            "equation_graph_edges": 137,
            "registered_scalar_PMF_nodes": 1,
            "registered_selector_nodes": 0,
            "action_derivation_edges_to_PMF": 0,
            "paper_typed_history_maps": 0,
            "QED_channel_kernels": 0,
            "Laplace_core_certificates": 0,
            "Mecke_core_certificates": 0,
            "audit_conclusion": "neither countable determining route has a source-bound premise",
        },
        "exact_controls": _exact_controls(),
        "candidate_records": [_candidate_record(record) for record in records],
        "gate_counts": dict(EXPECTED_COUNTS),
        "decision_counts": {"pass": 0, "reject": 0, "blocked": 2},
        "decision": "countable_full_law_selector_contract_closed_source_certificate_blocked",
        "first_blocker": FIRST_BLOCKER,
        "secondary_blockers": [
            "no_paper_typed_actualization_random_measure_on_the_declared_configuration_space",
            "no_QED_add_one_event_kernel_or_Mecke_identity",
            "no_action_to_countable_Laplace_core_derivation_edge",
            "no_operational_transaction_event_filtration_or_exposure_bundle",
        ],
        "claim_seals": dict(EXPECTED_CLAIM_SEALS),
        "data_seals": dict(config["seals"]),
        "content_sha256": None,
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
        else config_path.parents[1] / config["output_path"]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
