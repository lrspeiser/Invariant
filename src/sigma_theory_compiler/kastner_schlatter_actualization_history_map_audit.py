"""Source-bound audit of transactional prose as a history-to-counting-measure map."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

CONFIG_SCHEMA = "sigma-kastner-schlatter-actualization-history-map-audit-config-1.0"
RESULT_SCHEMA = "sigma-kastner-schlatter-actualization-history-map-audit-1.0"
FIRST_BLOCKER = (
    "no_paper_registered_locally_finite_measurable_actualization_history_space_or_"
    "set_indexed_counting_map"
)
SECOND_BLOCKER = (
    "no_registered_QED_probability_law_on_actualization_histories_selecting_the_"
    "conditional_Poisson_kernel"
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
        raise ValueError("actualization-history audit path escapes repository") from error
    return path


def _bound_artifact(root: Path, binding: Mapping[str, Any]) -> dict[str, Any]:
    if set(binding) != {"path", "file_sha256", "content_sha256"}:
        raise ValueError("actualization-history predecessor binding shape changed")
    path = _inside(root, str(binding["path"]))
    if not path.is_file() or _file_sha(path) != binding["file_sha256"]:
        raise ValueError("actualization-history predecessor file hash mismatch")
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or value.get("content_sha256") != binding["content_sha256"]
        or _content_sha(value) != binding["content_sha256"]
    ):
        raise ValueError("actualization-history predecessor content hash mismatch")
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
        raise ValueError("actualization-history config shape changed")
    if set(config.get("predecessors", {})) != {
        "conditional_poisson_kernel",
        "paper_intake",
        "equation_graph",
    }:
        raise ValueError("actualization-history predecessor set changed")
    if config.get("audit_domain") != {
        "paper_source": "arXiv:2209.04025v1 as hash-bound by the registered intake",
        "paper_pdf_sha256": PDF_SHA256,
        "target_map": "H -> N_H with N_H(B)=number of actualized absorption endpoints in B",
        "target_codomain": "locally finite integer-valued Borel counting measures on M",
        "target_kernel": "K[(g,phi),dN]=PRM(q0*exp(phi)*dVol_g)(dN)",
    }:
        raise ValueError("actualization-history audit domain changed")
    if config.get("admission_policy") != {
        "paper_prose_semantics_may_close_only_explicitly_stated_fields": True,
        "compiler_authored_conditional_map_may_be_registered": True,
        "poisson_assertion_counts_as_probability_law_derivation": False,
        "external_reference_counts_as_registered_QED_derivation": False,
        "candidate_action_rejection_allowed": False,
    }:
        raise ValueError("actualization-history admission policy changed")
    if any(config.get("seals", {}).values()):
        raise ValueError("actualization-history seal opened")


def _validate_sources(
    conditional: Mapping[str, Any], intake: Mapping[str, Any], graph: Mapping[str, Any]
) -> list[Mapping[str, Any]]:
    if (
        conditional.get("decision_counts") != {"pass": 0, "reject": 0, "blocked": 2}
        or conditional.get("first_blocker")
        != "no_paper_or_QED_derived_actualization_history_to_counting_measure_map_or_principle_selecting_the_compiler_authored_conditional_Poisson_kernel"
        or any(conditional.get("claim_seals", {}).values())
        or any(conditional.get("data_seals", {}).values())
    ):
        raise ValueError("conditional-Poisson predecessor boundary changed")
    records = conditional.get("candidate_records", [])
    if [record.get("branch_id") for record in records] != [
        "eq35_middle_h",
        "eq35_printed_planck",
    ]:
        raise ValueError("conditional-Poisson candidate ordering changed")

    primary = intake.get("primary_source", {})
    contracts = intake.get("formula_contracts", [])
    rate = next(
        (
            item
            for item in contracts
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
        or rate.get("equations_latex")
        != [
            r"\bar\lambda(x_0)=\lim_{R\to0}N_R(x_0)/V_R",
            r"\Delta\bar\lambda/\Delta x_0=q_\gamma",
            r"\bar P_\gamma=-c h q_\gamma",
        ]
        or "independence or stationarity beyond the stated constant-rate model"
        not in rate.get("not_claimed", [])
    ):
        raise ValueError("paper intake rate contract changed")

    equations = graph.get("knowledge_graph", {}).get("nodes", [])
    rate_node = next(
        (
            item
            for item in equations
            if item.get("record", {}).get("equation_id") == "EQ-KS-33-FOUR-RATE"
        ),
        None,
    )
    if (
        not isinstance(rate_node, dict)
        or rate_node.get("record", {}).get("source_locator") != "p.9 equation (33) following text"
        or rate_node.get("record", {}).get("assumptions") != ["constant_poisson_four_rate"]
    ):
        raise ValueError("equation-graph rate node changed")
    return records


def _paper_evidence() -> list[dict[str, Any]]:
    return [
        {
            "evidence_id": "actualized_transaction_record_semantics",
            "source_locator": "section 1 definitions, item 9; section 1.1",
            "faithful_paraphrase": (
                "one actualized transaction transfers one real photon from one emitter to one "
                "absorber and produces emission and absorption events joined by a null interval"
            ),
            "typed_contribution": "transaction record with two endpoint roles and one link",
            "status": "paper_prose_semantics_only",
        },
        {
            "evidence_id": "event_carrier_semantics",
            "source_locator": "sections 1.1 and 1.3",
            "faithful_paraphrase": (
                "actual events and their metric relations are modeled as points and relations of "
                "a four-dimensional manifold"
            ),
            "typed_contribution": "candidate carrier M for localized endpoints",
            "status": "paper_prose_semantics_only",
        },
        {
            "evidence_id": "absorption_endpoint_count_convention",
            "source_locator": "equation (33) context and footnote 19",
            "faithful_paraphrase": (
                "the transaction rate is described as the number of transactions, parenthetically "
                "absorptions, per four-volume"
            ),
            "typed_contribution": "absorption endpoint is the paper-indicated one-count representative",
            "status": "paper_prose_semantics_only",
        },
        {
            "evidence_id": "shrinking_region_count_notation",
            "source_locator": "equation (33) and immediately following text",
            "faithful_paraphrase": (
                "N_R(x0) denotes transaction counts in V_R and enters a shrinking-volume density limit"
            ),
            "typed_contribution": "one local family of count symbols, not an arbitrary set-indexed map",
            "status": "partial_formula_semantics",
        },
        {
            "evidence_id": "poisson_and_QED_statement",
            "source_locator": "text following equation (33)",
            "faithful_paraphrase": (
                "a Poisson stochastic process is asserted to follow from QED, and constant average "
                "rate is asserted to be Lorentz invariant"
            ),
            "typed_contribution": "law label and scalar-rate premise only",
            "status": "assertion_without_registered_QED_kernel_derivation",
        },
    ]


def _typed_obligations() -> list[dict[str, Any]]:
    return [
        {
            "obligation": "transaction_record_kind",
            "required": "record tau with emitter, absorber, transferred photon, and null link",
            "status": "closed_by_paper_semantics",
        },
        {
            "obligation": "localized_absorption_endpoint",
            "required": "a(tau) in M",
            "status": "closed_by_paper_semantics",
        },
        {
            "obligation": "one_count_per_transaction_convention",
            "required": "count the absorption endpoint once",
            "status": "closed_by_paper_semantics",
        },
        {
            "obligation": "event_carrier",
            "required": "manifold-like M carrying localized events",
            "status": "closed_by_paper_semantics",
        },
        {
            "obligation": "measurable_region_class",
            "required": "Borel sigma-algebra B(M)",
            "status": "partial_manifold_and_region_prose_but_no_sigma_algebra",
        },
        {
            "obligation": "set_indexed_count_evaluation",
            "required": "N_h(B)=sum_tau 1_B(a_h(tau)) for every B in B(M)",
            "status": "partial_N_R_for_shrinking_regions_only",
        },
        {
            "obligation": "history_sample_space",
            "required": "Omega_H of admissible actualization histories",
            "status": "absent",
        },
        {
            "obligation": "history_sigma_algebra_and_map_measurability",
            "required": "F_H and measurable h -> N_h",
            "status": "absent",
        },
        {
            "obligation": "local_finiteness_or_nonexplosion",
            "required": "N_h(K)<infinity for every relatively compact K",
            "status": "absent",
        },
        {
            "obligation": "probability_law_on_histories",
            "required": "P_H or conditional K[(g,phi),dh]",
            "status": "absent",
        },
        {
            "obligation": "QED_actualization_kernel",
            "required": "registered amplitudes/rates -> normalized history law derivation",
            "status": "asserted_or_externally_referenced_not_registered",
        },
        {
            "obligation": "Poisson_selector_identity",
            "required": "Laplace functional, independent increments, or Mecke identity",
            "status": "absent_from_paper_equation_graph",
        },
    ]


def _conditional_count_map() -> dict[str, Any]:
    return {
        "attribution": "compiler_authored_formal_completion_not_printed_or_derived_in_the_paper",
        "history_domain": (
            "H_lf={(T_h,a_h): a_h:T_h->M and #a_h^{-1}(K)<infinity for every relatively compact K}"
        ),
        "formula": "N_h(B)=sum_{tau in T_h} 1_B(a_h(tau))",
        "endpoint": "a_h(tau) is the absorption endpoint suggested by the paper's convention",
        "theorem": {
            "integer_valued": True,
            "countably_additive": True,
            "locally_finite_by_declared_domain": True,
            "diffeomorphism_covariance": "N_{psi_*h}(B)=N_h(psi^{-1}(B))",
        },
        "paper_supplies_H_lf_or_its_measurable_structure": False,
        "selects_a_probability_law": False,
        "selects_the_conditional_Poisson_kernel": False,
    }


def _nonidentifiability_controls() -> dict[str, Any]:
    return {
        "same_scalar_rate_different_counting_measure": {
            "two_equal_volume_cells": ["B1", "B2"],
            "history_A_counts": [2, 0],
            "history_B_counts": [1, 1],
            "same_total_count": 2,
            "different_set_indexed_measures": True,
            "conclusion": "a scalar total count or rate does not reconstruct N_h on subsets",
        },
        "same_mean_measure_different_history_laws": {
            "test_region_mean": "mu(B)=2",
            "Poisson_void_probability": "exp(-2)",
            "Cox_mixing": "Z in {1/2,3/2} with probabilities {1/2,1/2}",
            "Cox_void_probability": "(exp(-1)+exp(-3))/2",
            "Poisson_second_factorial_moment": "4",
            "Cox_second_factorial_moment": "5",
            "same_first_moment": True,
            "different_laws": True,
            "conclusion": "rate plus the deterministic count map does not select Poisson",
        },
        "locally_infinite_history_negative_control": {
            "history": "infinitely many absorption endpoints accumulating inside compact K",
            "N_h_K": "infinity",
            "rejected_from_H_lf": True,
            "paper_nonexplosion_premise_found": False,
        },
        "endpoint_double_count_negative_control": {
            "mutation": "count both emission and absorption endpoints as separate transactions",
            "rejected_by_absorption_count_convention": True,
        },
        "assertion_as_derivation_negative_control": {
            "mutation": "promote the paper's Poisson/QED sentence to a registered QED kernel proof",
            "rejected": True,
        },
    }


def _candidate_record(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "branch_id": record["branch_id"],
        "beta": record["beta"],
        "paper_semantics_audit_is_branch_independent": True,
        "paper_typed_history_map_complete": False,
        "paper_or_QED_kernel_selected": False,
        "compiler_conditional_count_map_available": True,
        "candidate_action_rejection_authorized": False,
        "candidate_decision": "blocked",
        "first_blocker": FIRST_BLOCKER,
    }


def _validate_result(result: Mapping[str, Any]) -> None:
    if result.get("schema_version") != RESULT_SCHEMA:
        raise ValueError("actualization-history result schema changed")
    if result.get("decision_counts") != {"pass": 0, "reject": 0, "blocked": 2}:
        raise ValueError("actualization-history candidate partition changed")
    if result.get("gate_counts") != {
        "candidate_actions": 2,
        "paper_source_clauses_audited": 5,
        "typed_map_obligations": 12,
        "closed_by_paper_semantics": 4,
        "partially_specified": 2,
        "blocked_or_absent": 6,
        "paper_complete_history_to_counting_measure_maps": 0,
        "compiler_conditional_count_maps": 1,
        "paper_or_QED_Poisson_kernel_selections": 0,
        "candidate_action_reject": 0,
        "theory_ontology_observational_pass": 0,
    }:
        raise ValueError("actualization-history gate counts changed")
    contract = result.get("compiler_conditional_count_map", {})
    if (
        contract.get("paper_supplies_H_lf_or_its_measurable_structure") is not False
        or contract.get("selects_a_probability_law") is not False
        or contract.get("theorem", {}).get("countably_additive") is not True
    ):
        raise ValueError("actualization-history conditional map boundary changed")
    records = result.get("candidate_records", [])
    if len(records) != 2 or any(
        record.get("paper_typed_history_map_complete") for record in records
    ):
        raise ValueError("actualization-history paper-map overclaim")
    if any(record.get("paper_or_QED_kernel_selected") for record in records):
        raise ValueError("actualization-history kernel-selection overclaim")
    if any(record.get("candidate_action_rejection_authorized") for record in records):
        raise ValueError("actualization-history gate overreached to action rejection")
    if result.get("first_blocker") != FIRST_BLOCKER:
        raise ValueError("actualization-history first blocker changed")
    if any(result.get("claim_seals", {}).values()) or any(result.get("data_seals", {}).values()):
        raise ValueError("actualization-history seal opened")
    if (
        result.get("content_sha256") is not None
        and _content_sha(result) != result["content_sha256"]
    ):
        raise ValueError("actualization-history content hash mismatch")


def build_audit(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    root = config_path.parents[1]
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    predecessors = {
        label: _bound_artifact(root, binding) for label, binding in config["predecessors"].items()
    }
    records = _validate_sources(
        predecessors["conditional_poisson_kernel"],
        predecessors["paper_intake"],
        predecessors["equation_graph"],
    )
    source_path = Path(__file__).resolve()
    test_path = root / "tests/test_kastner_schlatter_actualization_history_map_audit.py"
    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA,
        "campaign_id": config["campaign_id"],
        "scope": (
            "source-bound typed audit of whether transactional actualization prose defines a "
            "history-to-counting-measure map or selects a conditional Poisson law"
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
        "audit_domain": config["audit_domain"],
        "paper_evidence_ledger": _paper_evidence(),
        "typed_map_obligations": _typed_obligations(),
        "compiler_conditional_count_map": _conditional_count_map(),
        "exact_nonidentifiability_and_negative_controls": _nonidentifiability_controls(),
        "candidate_records": [_candidate_record(record) for record in records],
        "gate_counts": {
            "candidate_actions": 2,
            "paper_source_clauses_audited": 5,
            "typed_map_obligations": 12,
            "closed_by_paper_semantics": 4,
            "partially_specified": 2,
            "blocked_or_absent": 6,
            "paper_complete_history_to_counting_measure_maps": 0,
            "compiler_conditional_count_maps": 1,
            "paper_or_QED_Poisson_kernel_selections": 0,
            "candidate_action_reject": 0,
            "theory_ontology_observational_pass": 0,
        },
        "decision_counts": {"pass": 0, "reject": 0, "blocked": 2},
        "decision": "paper_supplies_partial_event_semantics_but_no_typed_history_map_or_kernel_selection",
        "first_blocker": FIRST_BLOCKER,
        "secondary_blockers": [
            SECOND_BLOCKER,
            "no_registered_nonexplosion_or_local_finiteness_theorem_for_actualization_histories",
            "no_deterministic_candidate_action_to_actualization_history_law_derivation",
            "no_observational_actualization_history_or_exposure_registration",
        ],
        "claim_seals": {
            "paper_complete_history_map_registered": False,
            "paper_or_QED_Poisson_kernel_derived": False,
            "conditional_kernel_action_derived": False,
            "unconditional_Poisson_law_derived": False,
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
