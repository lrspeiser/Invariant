"""Typed source-inhabitation audit for KS Poisson selector certificates."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

CONFIG_SCHEMA = "sigma-kastner-schlatter-source-selector-type-inhabitation-audit-config-1.0"
RESULT_SCHEMA = "sigma-kastner-schlatter-source-selector-type-inhabitation-audit-1.0"
PDF_SHA256 = "c2f671293d07b21397e745da00a3ce1a2193c00da647a2ebf4147612b76c1780"
FIRST_BLOCKER = (
    "paper_QED_actualization_language_has_no_typed_probability_expectation_and_set_"
    "indexed_count_map_needed_to_inhabit_a_countable_selector_certificate"
)
BRANCHES = [("eq35_middle_h", "1/2"), ("eq35_printed_planck", "1/4")]

EXPECTED_COUNTS = {
    "candidate_actions": 2,
    "candidate_blocked": 2,
    "candidate_pass": 0,
    "candidate_reject": 0,
    "registered_source_evidence_clauses": 5,
    "Laplace_required_typed_slots": 10,
    "Laplace_source_complete_slots": 0,
    "Laplace_source_partial_slots": 3,
    "Laplace_source_absent_slots": 7,
    "Mecke_required_typed_slots": 10,
    "Mecke_source_complete_slots": 0,
    "Mecke_source_partial_slots": 2,
    "Mecke_source_absent_slots": 8,
    "source_bound_countable_certificates": 0,
    "registered_scalar_PMF_nodes": 1,
    "compiler_scalar_transform_replays": 1,
    "paper_QED_ontology_observational_pass": 0,
}

EXPECTED_CLAIM_SEALS = {
    "paper_QED_countable_selector_registered": False,
    "paper_scalar_PMF_equation_claimed": False,
    "candidate_action_stochastic_law_derived": False,
    "candidate_action_selects_Poisson": False,
    "candidate_action_rejected": False,
    "semantic_actualization_prose_promoted_to_probability_kernel": False,
    "compiler_scalar_replay_promoted_to_set_indexed_certificate": False,
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
        raise ValueError("source-selector audit path escapes repository") from error
    return path


def _bound_artifact(root: Path, binding: Mapping[str, Any]) -> dict[str, Any]:
    if set(binding) != {"path", "file_sha256", "content_sha256"}:
        raise ValueError("source-selector predecessor binding shape changed")
    path = _inside(root, str(binding["path"]))
    if not path.is_file() or _file_sha(path) != binding["file_sha256"]:
        raise ValueError("source-selector predecessor file hash mismatch")
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or value.get("content_sha256") != binding["content_sha256"]
        or _content_sha(value) != binding["content_sha256"]
    ):
        raise ValueError("source-selector predecessor content hash mismatch")
    return value


def _validate_config(config: Mapping[str, Any]) -> None:
    if (
        set(config)
        != {
            "schema_version",
            "campaign_id",
            "output_path",
            "predecessors",
            "typed_certificate_policy",
            "seals",
        }
        or config.get("schema_version") != CONFIG_SCHEMA
    ):
        raise ValueError("source-selector config shape changed")
    if set(config["predecessors"]) != {
        "countable_full_law_admission",
        "paper_intake",
        "poisson_selector_contract",
        "qed_actualization_audit",
        "actualization_history_map",
        "equation_graph",
    }:
        raise ValueError("source-selector predecessor set changed")
    if config.get("typed_certificate_policy") != {
        "paper_prose_Poisson_assertion_is_a_complete_certificate": False,
        "compiler_standard_scalar_PMF_is_a_source_equation": False,
        "scalar_transform_replay_is_a_set_indexed_core_certificate": False,
        "missing_typed_arguments_may_be_inferred_from_semantics": False,
        "candidate_action_rejection_allowed": False,
    }:
        raise ValueError("source-selector typed policy changed")
    if not config.get("seals") or any(config["seals"].values()):
        raise ValueError("source-selector seal opened")


def _validate_predecessors(predecessors: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    admission = predecessors["countable_full_law_admission"]
    records = admission.get("candidate_records", [])
    if (
        admission.get("decision_counts") != {"blocked": 2, "pass": 0, "reject": 0}
        or [(row.get("branch_id"), row.get("beta")) for row in records] != BRANCHES
        or admission.get("closed_world_source_audit", {}).get("Laplace_core_certificates") != 0
        or admission.get("closed_world_source_audit", {}).get("Mecke_core_certificates") != 0
    ):
        raise ValueError("countable-admission boundary changed")
    intake_contracts = predecessors["paper_intake"].get("formula_contracts", [])
    rate = next(
        (
            row
            for row in intake_contracts
            if row.get("contract_id") == "transaction_poisson_rate_and_pressure"
        ),
        None,
    )
    pmf = next(
        (
            row
            for row in intake_contracts
            if row.get("contract_id") == "standard_poisson_cuda_reference"
        ),
        None,
    )
    if (
        rate is None
        or rate.get("paper_equations") != [33, 34]
        or "independence or stationarity beyond the stated constant-rate model"
        not in rate.get("not_claimed", [])
        or pmf is None
        or pmf.get("paper_equations") != []
        or pmf.get("classification")
        != "standard_implementation_of_paper_poisson_assertion_not_printed_equation"
    ):
        raise ValueError("paper intake Poisson boundary changed")
    dependency = predecessors["poisson_selector_contract"].get("registered_dependency_audit", {})
    if (
        dependency.get("registered_scalar_Poisson_PMF_nodes") != 1
        or dependency.get("registered_selector_nodes") != 0
        or dependency.get("PMF_expression_has_set_argument_B") is not False
        or dependency.get("PMF_expression_has_joint_count_arguments") is not False
        or dependency.get("PMF_expression_has_history_or_QED_kernel_argument") is not False
        or dependency.get("PMF_printed_as_equation_in_paper") is not False
    ):
        raise ValueError("selector dependency type boundary changed")
    evidence = predecessors["qed_actualization_audit"].get("primary_source_evidence", [])
    if len(evidence) != 5 or {row.get("status") for row in evidence} != {
        "semantic_only",
        "first_moment_or_density_only",
        "assertion_without_registered_channel_kernel",
        "implementation_not_paper_equation_and_not_derivation",
        "synthetic_only_no_QED_inference",
    }:
        raise ValueError("QED source evidence classification changed")
    history_records = predecessors["actualization_history_map"].get("candidate_records", [])
    if [(row.get("branch_id"), row.get("beta")) for row in history_records] != BRANCHES or any(
        row.get("paper_typed_history_map_complete") is not False for row in history_records
    ):
        raise ValueError("history-map type boundary changed")
    graph_counts = predecessors["equation_graph"].get("graph_counts", {})
    if graph_counts.get("nodes") != 54 or graph_counts.get("edges") != 137:
        raise ValueError("equation graph closed-world counts changed")
    return records


def _laplace_slots() -> list[dict[str, str]]:
    return [
        {
            "slot": "measurable_region_or_continuity_ring",
            "status": "partial",
            "evidence": "N_R(x0)/V_R refers to shrinking region scale R, not a Borel ring",
        },
        {
            "slot": "set_indexed_counting_random_measure_N_of_A",
            "status": "partial",
            "evidence": "N_R(x0) is a localized count symbol without a registered random-measure type",
        },
        {
            "slot": "intensity_measure_mu",
            "status": "partial",
            "evidence": "q_gamma is a constant average rate per four-volume, not a probability kernel",
        },
        {"slot": "configuration_space_N_lf_W", "status": "absent", "evidence": "no source type"},
        {
            "slot": "probability_measure_on_configurations",
            "status": "absent",
            "evidence": "no source type",
        },
        {"slot": "test_function_f", "status": "absent", "evidence": "no source argument"},
        {"slot": "expectation_operator", "status": "absent", "evidence": "no source operator"},
        {
            "slot": "Laplace_functional_identity",
            "status": "absent",
            "evidence": "no source equation",
        },
        {
            "slot": "countable_core_quantifier",
            "status": "absent",
            "evidence": "no source quantifier",
        },
        {"slot": "set_indexed_consistency", "status": "absent", "evidence": "no source relation"},
    ]


def _mecke_slots() -> list[dict[str, str]]:
    return [
        {
            "slot": "count_or_actualization_object",
            "status": "partial",
            "evidence": "localized absorption endpoints and N_R(x0) are semantic/count fragments",
        },
        {
            "slot": "intensity_measure_mu",
            "status": "partial",
            "evidence": "q_gamma is a first-moment rate fragment",
        },
        {"slot": "configuration_space_N_lf_W", "status": "absent", "evidence": "no source type"},
        {
            "slot": "probability_measure_on_configurations",
            "status": "absent",
            "evidence": "no source type",
        },
        {
            "slot": "add_one_map_N_plus_delta_x",
            "status": "absent",
            "evidence": "no source operator",
        },
        {"slot": "test_functional_H_x_N", "status": "absent", "evidence": "no source argument"},
        {"slot": "expectation_operator", "status": "absent", "evidence": "no source operator"},
        {"slot": "Mecke_identity", "status": "absent", "evidence": "no source equation"},
        {
            "slot": "countable_cylinder_core_quantifier",
            "status": "absent",
            "evidence": "no source quantifier",
        },
        {
            "slot": "QED_channel_or_history_kernel",
            "status": "absent",
            "evidence": "audit reports none",
        },
    ]


def _type_inhabitation_theorem() -> dict[str, Any]:
    return {
        "theorem_name": "registered_source_domain_countable_selector_noninhabitation",
        "typing_rule": (
            "a certificate instance is admitted only when every required slot has a source-bound "
            "typed term and the asserted equality has a registered derivation or explicit source status"
        ),
        "Laplace_conclusion": (
            "three source fragments are partial and seven required slots are absent; therefore "
            "no nontrivial rational-simple-function core instance is inhabited"
        ),
        "Mecke_conclusion": (
            "two source fragments are partial and eight required slots are absent; therefore no "
            "nontrivial rational-cylinder/add-one core instance is inhabited"
        ),
        "scalar_PMF_boundary": (
            "the compiler can algebraically sum the standard scalar PMF to one scalar Laplace "
            "transform, but the PMF is not printed in the paper and has no region, random-measure, "
            "history-kernel, or joint-consistency argument, so it inhabits neither countable core"
        ),
        "scope_limit": (
            "this is a closed-world audit of registered source content, not a claim that no future "
            "microscopic QED model could supply the missing typed objects"
        ),
    }


def _exact_controls() -> dict[str, Any]:
    return {
        "compiler_scalar_transform_partial_replay": {
            "input": "p(n|mu)=exp(-mu)*mu^n/n!",
            "derived_scalar_identity": ("sum_n exp(-t*n)*p(n|mu)=exp(-mu*(1-exp(-t)))"),
            "paper_printed_equation": False,
            "has_named_region_A": False,
            "has_counting_random_measure_N_of_A": False,
            "has_countable_core_quantifier": False,
            "qualifies_as_source_bound_core_certificate": False,
        },
        "Poisson_prose_promotion_negative_control": {
            "source_status": "assertion_without_registered_channel_kernel",
            "mutation": "infer all missing probability-space and set-indexed arguments",
            "rejected": True,
        },
        "rate_to_compensator_negative_control": {
            "source_quantity": "constant average q_gamma per four-volume",
            "mutation": "identify q_gamma with a predictable compensator without a filtration",
            "rejected": True,
        },
    }


def _candidate_record(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "branch_id": record["branch_id"],
        "beta": record["beta"],
        "source_Laplace_core_instances": 0,
        "source_Mecke_core_instances": 0,
        "compiler_scalar_transform_partial_replay": True,
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
        "source_evidence",
        "type_inhabitation_theorem",
        "Laplace_required_slots",
        "Mecke_required_slots",
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
        raise ValueError("source-selector result shape changed")
    if result.get("decision_counts") != {"pass": 0, "reject": 0, "blocked": 2}:
        raise ValueError("source-selector candidate partition changed")
    if result.get("gate_counts") != EXPECTED_COUNTS:
        raise ValueError("source-selector gate counts changed")
    if result.get("type_inhabitation_theorem") != _type_inhabitation_theorem():
        raise ValueError("source-selector theorem changed")
    if result.get("Laplace_required_slots") != _laplace_slots():
        raise ValueError("source-selector Laplace slots changed")
    if result.get("Mecke_required_slots") != _mecke_slots():
        raise ValueError("source-selector Mecke slots changed")
    if result.get("exact_controls") != _exact_controls():
        raise ValueError("source-selector controls changed")
    records = result.get("candidate_records", [])
    if [(row.get("branch_id"), row.get("beta")) for row in records] != BRANCHES or any(
        row.get("source_Laplace_core_instances") != 0
        or row.get("source_Mecke_core_instances") != 0
        or row.get("paper_or_QED_selector_derived") is not False
        or row.get("candidate_action_selects_Poisson") is not False
        or row.get("candidate_action_rejection_authorized") is not False
        or row.get("candidate_decision") != "blocked"
        or row.get("first_blocker") != FIRST_BLOCKER
        for row in records
    ):
        raise ValueError("source-selector candidate boundary changed")
    if result.get("first_blocker") != FIRST_BLOCKER:
        raise ValueError("source-selector blocker changed")
    if result.get("claim_seals") != EXPECTED_CLAIM_SEALS or any(result["data_seals"].values()):
        raise ValueError("source-selector seal changed")
    if (
        result.get("content_sha256") is not None
        and _content_sha(result) != result["content_sha256"]
    ):
        raise ValueError("source-selector content hash mismatch")


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
    test_path = root / "tests/test_kastner_schlatter_source_selector_type_inhabitation_audit.py"
    source_evidence = predecessors["qed_actualization_audit"]["primary_source_evidence"]
    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA,
        "campaign_id": config["campaign_id"],
        "scope": (
            "candidate-bound typed audit of whether registered paper/QED actualization content "
            "inhabits a countable Laplace or Mecke core; no missing arguments are inferred"
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
        "source_evidence": source_evidence,
        "type_inhabitation_theorem": _type_inhabitation_theorem(),
        "Laplace_required_slots": _laplace_slots(),
        "Mecke_required_slots": _mecke_slots(),
        "exact_controls": _exact_controls(),
        "candidate_records": [_candidate_record(record) for record in records],
        "gate_counts": dict(EXPECTED_COUNTS),
        "decision_counts": {"pass": 0, "reject": 0, "blocked": 2},
        "decision": "registered_source_inhabits_no_countable_selector_certificate",
        "first_blocker": FIRST_BLOCKER,
        "secondary_blockers": [
            "no_source_probability_law_on_actualization_histories_or_counting_measures",
            "no_source_test_function_expectation_or_add_one_operator",
            "no_QED_channel_kernel_or_set_indexed_consistency_derivation",
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
