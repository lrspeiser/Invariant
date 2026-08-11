"""Audit the missing QED actualization-to-Poisson derivation contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

CONFIG_SCHEMA = "sigma-kastner-schlatter-qed-actualization-poisson-derivation-config-1.0"
RESULT_SCHEMA = "sigma-kastner-schlatter-qed-actualization-poisson-derivation-1.0"
FIRST_BLOCKER = (
    "no_registered_QED_actualization_channel_probability_array_or_predictable_hazard_kernel"
)
SECOND_BLOCKER = (
    "no_source_bound_independence_or_no_clustering_theorem_for_disjoint_actualization_channels"
)
PDF_SHA256 = "c2f671293d07b21397e745da00a3ce1a2193c00da647a2ebf4147612b76c1780"


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
        raise ValueError("QED actualization audit path escapes repository") from error
    return path


def _bound_artifact(root: Path, binding: Mapping[str, Any]) -> dict[str, Any]:
    if set(binding) != {"path", "file_sha256", "content_sha256"}:
        raise ValueError("QED actualization predecessor binding shape changed")
    path = _inside(root, str(binding["path"]))
    if not path.is_file() or _file_sha(path) != binding["file_sha256"]:
        raise ValueError("QED actualization predecessor file hash mismatch")
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or value.get("content_sha256") != binding["content_sha256"]
        or _content_sha(value) != binding["content_sha256"]
    ):
        raise ValueError("QED actualization predecessor content hash mismatch")
    return value


def _validate_config(config: Mapping[str, Any]) -> None:
    if (
        set(config)
        != {
            "schema_version",
            "campaign_id",
            "output_path",
            "predecessors",
            "audit_domain",
            "admission_policy",
            "seals",
        }
        or config.get("schema_version") != CONFIG_SCHEMA
    ):
        raise ValueError("QED actualization config shape changed")
    if set(config.get("predecessors", {})) != {
        "operational_event_exposure",
        "conditional_poisson_kernel",
        "set_indexed_synthetic_campaign",
        "paper_intake",
        "equation_graph",
    }:
        raise ValueError("QED actualization predecessor set changed")
    if config.get("audit_domain") != {
        "paper_source": "arXiv:2209.04025v1 as hash-bound by the registered intake",
        "paper_pdf_sha256": PDF_SHA256,
        "target": "QED actualization channels -> locally finite set-indexed Poisson counting kernel",
        "admissible_sufficient_route": (
            "independent asymptotically negligible exclusive channel array with convergent cell intensities"
        ),
    }:
        raise ValueError("QED actualization audit domain changed")
    if config.get("admission_policy") != {
        "paper_poisson_sentence_counts_as_derivation": False,
        "external_QED_reference_counts_as_registered_kernel": False,
        "compiler_authored_conditional_limit_theorem_allowed": True,
        "same_mean_rate_counts_as_poisson_identification": False,
        "candidate_action_rejection_allowed": False,
    }:
        raise ValueError("QED actualization admission policy changed")
    if any(config.get("seals", {}).values()):
        raise ValueError("QED actualization seal opened")


def _validate_predecessors(values: Mapping[str, Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    operational = values["operational_event_exposure"]
    if (
        operational.get("decision_counts") != {"pass": 0, "reject": 0, "blocked": 2}
        or operational.get("first_blocker")
        != "no_registered_detector_level_transaction_event_schema_exposure_response_background_or_calibration_manifest"
        or operational.get("synthetic_only") is not True
        or any(operational.get("claim_seals", {}).values())
        or any(operational.get("data_seals", {}).values())
    ):
        raise ValueError("operational-event predecessor boundary changed")
    records = operational.get("candidate_records", [])
    if [record.get("branch_id") for record in records] != [
        "eq35_middle_h",
        "eq35_printed_planck",
    ]:
        raise ValueError("operational-event candidate ordering changed")

    conditional = values["conditional_poisson_kernel"]
    if (
        conditional.get("gate_counts", {}).get("conditional_Laplace_selector_pass") != 2
        or conditional.get("gate_counts", {}).get("conditional_independent_increment_pass") != 2
        or conditional.get("gate_counts", {}).get("paper_or_QED_actualization_derivation_pass") != 0
    ):
        raise ValueError("conditional-Poisson predecessor boundary changed")

    synthetic = values["set_indexed_synthetic_campaign"]
    if (
        synthetic.get("synthetic_only") is not True
        or synthetic.get("counts", {}).get("observational_records_accessed") != 0
        or synthetic.get("counts", {}).get("paper_or_qed_inferences") != 0
        or synthetic.get("scientific_test_pass") is not False
        or synthetic.get("paper_pass") is not False
        or synthetic.get("qed_pass") is not False
        or synthetic.get("theory_pass") is not False
        or synthetic.get("ontology_pass") is not False
    ):
        raise ValueError("set-indexed synthetic predecessor boundary changed")

    intake = values["paper_intake"]
    primary = intake.get("primary_source", {})
    rate = next(
        (
            item
            for item in intake.get("formula_contracts", [])
            if item.get("contract_id") == "transaction_poisson_rate_and_pressure"
        ),
        None,
    )
    if (
        primary.get("arxiv_id") != "2209.04025"
        or primary.get("version") != "v1"
        or primary.get("pdf_sha256") != PDF_SHA256
        or not isinstance(rate, dict)
        or rate.get("paper_equations") != [33, 34]
        or rate.get("classification") != "paper_proposal_with_qed_and_lorentz_invariance_assertions"
        or "the paper's cited Poisson/Lorentz-invariance premise applies"
        not in rate.get("assumptions", [])
    ):
        raise ValueError("paper intake QED/Poisson evidence changed")

    graph = values["equation_graph"]
    nodes = graph.get("knowledge_graph", {}).get("nodes", [])
    by_id = {node.get("node_id"): node for node in nodes}
    rate_node = by_id.get("EQ-KS-33-FOUR-RATE", {}).get("record", {})
    pmf_node = by_id.get("EQ-KS-POISSON-PMF-IMPLEMENTATION", {}).get("record", {})
    if (
        rate_node.get("source_locator") != "p.9 equation (33) following text"
        or pmf_node.get("source_locator")
        != "p.9 Poisson assertion; standard PMF is implementation-only"
        or "implementation_not_printed_equation" not in pmf_node.get("tags", [])
    ):
        raise ValueError("equation-graph Poisson evidence changed")
    return records


def _source_evidence() -> list[dict[str, Any]]:
    return [
        {
            "evidence_id": "actualization_endpoint_semantics",
            "source": "paper sections 1.1 and 1.3 as registered by the history-map audit",
            "contribution": "one actualized transaction has a localized absorption endpoint",
            "status": "semantic_only",
        },
        {
            "evidence_id": "transaction_count_density",
            "source": "paper equation (33)",
            "contribution": "N_R(x0)/V_R and a constant mean rate q_gamma",
            "status": "first_moment_or_density_only",
        },
        {
            "evidence_id": "qed_poisson_sentence",
            "source": "paper text following equation (33)",
            "contribution": "Poisson is asserted to follow from QED",
            "status": "assertion_without_registered_channel_kernel",
        },
        {
            "evidence_id": "standard_poisson_pmf",
            "source": "compiler equation graph node EQ-KS-POISSON-PMF-IMPLEMENTATION",
            "contribution": "one-cell PMF implementation",
            "status": "implementation_not_paper_equation_and_not_derivation",
        },
        {
            "evidence_id": "set_indexed_synthetic_discrimination",
            "source": "hash-bound synthetic CUDA campaign",
            "contribution": "synthetic diagnostics distinguish dependence with Poisson marginals",
            "status": "synthetic_only_no_QED_inference",
        },
    ]


def _microscopic_obligations() -> list[dict[str, Any]]:
    return [
        {"obligation": "QED_channel_index_set_and_state_domain", "status": "absent"},
        {"obligation": "normalized_channel_actualization_probabilities", "status": "absent"},
        {"obligation": "probabilities_derived_from_registered_QED_amplitudes", "status": "absent"},
        {"obligation": "exclusive_zero_or_one_actualization_per_channel_trial", "status": "absent"},
        {"obligation": "independence_or_factorization_across_channel_trials", "status": "absent"},
        {
            "obligation": "endpoint_localization_map_to_each_measurable_cell",
            "status": "semantic_partial",
        },
        {
            "obligation": "asymptotic_negligibility_max_channel_probability_to_zero",
            "status": "absent",
        },
        {"obligation": "cellwise_probability_sums_converge_to_mu", "status": "rate_partial"},
        {"obligation": "squared_probability_remainder_sum_to_zero", "status": "absent"},
        {"obligation": "nonexplosion_and_local_finiteness", "status": "absent"},
        {"obligation": "covariant_refinement_consistency_of_cell_limits", "status": "absent"},
        {"obligation": "candidate_q0_exp_phi_intensity_matching", "status": "absent"},
    ]


def _rare_channel_theorem() -> dict[str, Any]:
    return {
        "attribution": "compiler_authored_conditional_sufficient_theorem_not_derived_by_the_paper_or_QED",
        "setup": (
            "for disjoint cells B_i and row m, independent channels k choose at most one cell i "
            "with probabilities p_mki or choose no event"
        ),
        "conditions": {
            "exclusive_channel_outcome": "sum_i p_mki<=1",
            "asymptotic_negligibility": "max_k sum_i p_mki -> 0",
            "cell_intensity_convergence": "sum_k p_mki -> mu_i for every i",
            "quadratic_remainder": "sum_k (sum_i p_mki)^2 -> 0",
        },
        "finite_row_joint_PGF": "G_m(z)=product_k[1+sum_i p_mki*(z_i-1)]",
        "log_remainder_bound": ("abs(log(1+u)-u)<=u^2/(2*(1-abs(u))) for abs(u)<1"),
        "limit_joint_PGF": "exp(sum_i mu_i*(z_i-1))",
        "conclusion": "cell counts converge jointly to independent Poisson(mu_i)",
        "set_indexed_completion_requires": (
            "a projectively consistent locally finite family over all finite measurable partitions"
        ),
        "paper_or_registered_QED_closes_conditions": False,
    }


def _exact_controls() -> dict[str, Any]:
    return {
        "independent_rare_channel_positive_control": {
            "row": "m independent Bernoulli channels with p=mu/m",
            "PGF": "(1+mu*(z-1)/m)^m",
            "limit_PGF": "exp(mu*(z-1))",
            "limit_mean": "mu",
            "limit_variance": "mu",
            "pass": True,
        },
        "paired_cluster_same_rate_no_go": {
            "row": "m independent cluster channels; each creates two transactions with probability mu/(2*m)",
            "PGF": "(1+mu*(z^2-1)/(2*m))^m",
            "limit_PGF": "exp((mu/2)*(z^2-1))",
            "limit_mean": "mu",
            "limit_variance": "2*mu",
            "same_mean_rate_as_Poisson": True,
            "Poisson_conclusion_rejected": True,
        },
        "two_cell_common_shock_no_go": {
            "marginal_means": ["1", "1"],
            "common_shock_mean": "1/2",
            "idiosyncratic_means": ["1/2", "1/2"],
            "marginals": ["Poisson(1)", "Poisson(1)"],
            "cross_covariance": "1/2",
            "independent_increment_conclusion": False,
        },
        "mean_and_one_cell_PMF_insufficiency": {
            "premises": ["mean rate q_gamma", "one-cell Poisson marginals"],
            "not_identified": [
                "joint Laplace functional",
                "independent increments",
                "Mecke identity",
            ],
            "pass": True,
        },
        "external_reference_negative_control": {
            "mutation": "treat an uncaptured QED/RTI reference as a hash-bound derivation",
            "rejected": True,
        },
        "attribution_negative_control": {
            "mutation": "attribute the rare-channel limit theorem to the paper",
            "rejected": True,
        },
    }


def _candidate_record(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "branch_id": record["branch_id"],
        "beta": record["beta"],
        "QED_actualization_audit_is_branch_independent": True,
        "compiler_conditional_rare_channel_theorem": True,
        "paper_or_QED_channel_kernel_registered": False,
        "paper_or_QED_Poisson_derivation_pass": False,
        "candidate_action_rejection_authorized": False,
        "candidate_decision": "blocked",
        "first_blocker": FIRST_BLOCKER,
    }


def _validate_result(result: Mapping[str, Any]) -> None:
    if result.get("schema_version") != RESULT_SCHEMA:
        raise ValueError("QED actualization result schema changed")
    if result.get("decision_counts") != {"pass": 0, "reject": 0, "blocked": 2}:
        raise ValueError("QED actualization candidate partition changed")
    if result.get("gate_counts") != {
        "candidate_actions": 2,
        "source_evidence_clauses": 5,
        "microscopic_derivation_obligations": 12,
        "microscopic_obligations_closed": 0,
        "microscopic_obligations_partial": 2,
        "microscopic_obligations_absent": 10,
        "compiler_conditional_sufficient_theorems": 1,
        "exact_same_rate_non_Poisson_witnesses": 2,
        "paper_or_QED_channel_kernels_registered": 0,
        "paper_or_QED_Poisson_derivation_pass": 0,
        "candidate_action_reject": 0,
        "theory_ontology_observational_pass": 0,
    }:
        raise ValueError("QED actualization gate counts changed")
    theorem = result.get("independent_rare_channel_Poisson_limit", {})
    if (
        theorem.get("paper_or_registered_QED_closes_conditions") is not False
        or theorem.get("conclusion") != "cell counts converge jointly to independent Poisson(mu_i)"
    ):
        raise ValueError("QED actualization theorem boundary changed")
    records = result.get("candidate_records", [])
    if len(records) != 2 or any(
        record.get("paper_or_QED_channel_kernel_registered") for record in records
    ):
        raise ValueError("QED actualization channel-kernel overclaim")
    if any(record.get("paper_or_QED_Poisson_derivation_pass") for record in records):
        raise ValueError("QED actualization derivation overclaim")
    if any(record.get("candidate_action_rejection_authorized") for record in records):
        raise ValueError("QED actualization gate overreached to action rejection")
    if result.get("first_blocker") != FIRST_BLOCKER:
        raise ValueError("QED actualization first blocker changed")
    if any(result.get("claim_seals", {}).values()) or any(result.get("data_seals", {}).values()):
        raise ValueError("QED actualization seal opened")
    if (
        result.get("content_sha256") is not None
        and _content_sha(result) != result["content_sha256"]
    ):
        raise ValueError("QED actualization content hash mismatch")


def build_audit(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    root = config_path.parents[1]
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    predecessors = {
        label: _bound_artifact(root, binding) for label, binding in config["predecessors"].items()
    }
    records = _validate_predecessors(predecessors)
    source_path = Path(__file__).resolve()
    test_path = root / "tests/test_kastner_schlatter_qed_actualization_poisson_derivation_audit.py"
    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA,
        "campaign_id": config["campaign_id"],
        "scope": (
            "source-bound QED actualization-to-counting-kernel derivation audit with a "
            "compiler-authored conditional rare-channel theorem and no physical attribution"
        ),
        "synthetic_only": True,
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
        "audit_domain": config["audit_domain"],
        "primary_source_evidence": _source_evidence(),
        "microscopic_derivation_obligations": _microscopic_obligations(),
        "independent_rare_channel_Poisson_limit": _rare_channel_theorem(),
        "exact_controls": _exact_controls(),
        "candidate_records": [_candidate_record(record) for record in records],
        "gate_counts": {
            "candidate_actions": 2,
            "source_evidence_clauses": 5,
            "microscopic_derivation_obligations": 12,
            "microscopic_obligations_closed": 0,
            "microscopic_obligations_partial": 2,
            "microscopic_obligations_absent": 10,
            "compiler_conditional_sufficient_theorems": 1,
            "exact_same_rate_non_Poisson_witnesses": 2,
            "paper_or_QED_channel_kernels_registered": 0,
            "paper_or_QED_Poisson_derivation_pass": 0,
            "candidate_action_reject": 0,
            "theory_ontology_observational_pass": 0,
        },
        "decision_counts": {"pass": 0, "reject": 0, "blocked": 2},
        "decision": "conditional_rare_channel_theorem_closed_paper_QED_derivation_blocked",
        "first_blocker": FIRST_BLOCKER,
        "secondary_blockers": [
            SECOND_BLOCKER,
            "no_registered_nonexplosion_covariant_refinement_or_candidate_intensity_matching_proof",
            "no_operational_transaction_event_calibration_or_observation_bundle",
        ],
        "claim_seals": {
            "paper_QED_channel_kernel_registered": False,
            "paper_QED_Poisson_derivation_proven": False,
            "rare_channel_theorem_attributed_to_paper": False,
            "detector_event_equals_transaction_proven": False,
            "transaction_ontology_validated": False,
            "candidate_action_rejected": False,
            "observational_pass": False,
            "scientific_test_pass": False,
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
    result = build_audit(config_path)
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
