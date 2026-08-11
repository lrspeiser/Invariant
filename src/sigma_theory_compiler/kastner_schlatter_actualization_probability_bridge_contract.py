"""Minimal actualization-probability bridge contract for KS candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

CONFIG_SCHEMA = "sigma-kastner-schlatter-actualization-probability-bridge-config-1.0"
RESULT_SCHEMA = "sigma-kastner-schlatter-actualization-probability-bridge-contract-1.0"
PDF_SHA256 = "c2f671293d07b21397e745da00a3ce1a2193c00da647a2ebf4147612b76c1780"
FIRST_BLOCKER = (
    "no_source_bound_candidate_conditioned_QED_history_probability_kernel_measurable_"
    "locally_finite_count_pushforward_and_countable_core_identity"
)
BRANCHES = [("eq35_middle_h", "1/2"), ("eq35_printed_planck", "1/4")]

EXPECTED_COUNTS = {
    "candidate_actions": 2,
    "candidate_blocked": 2,
    "candidate_pass": 0,
    "candidate_reject": 0,
    "primitive_source_registrations": 3,
    "source_complete_primitives": 0,
    "source_partial_primitives": 1,
    "source_absent_primitives": 2,
    "compiler_derived_bridge_objects": 2,
    "compiler_identity_fixtures": 2,
    "exact_missing_primitive_controls": 4,
    "complete_paper_or_QED_bridges": 0,
    "theory_ontology_observational_pass": 0,
}

EXPECTED_CLAIM_SEALS = {
    "paper_QED_history_probability_kernel_registered": False,
    "paper_QED_measurable_count_pushforward_registered": False,
    "paper_QED_core_identity_registered": False,
    "candidate_action_stochastic_law_derived": False,
    "candidate_action_selects_Poisson": False,
    "candidate_action_rejected": False,
    "compiler_identity_fixture_attributed_to_source": False,
    "semantic_endpoint_promoted_to_measurable_count_map": False,
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
        raise ValueError("actualization-probability bridge path escapes repository") from error
    return path


def _bound_artifact(root: Path, binding: Mapping[str, Any]) -> dict[str, Any]:
    if set(binding) != {"path", "file_sha256", "content_sha256"}:
        raise ValueError("actualization-probability predecessor binding shape changed")
    path = _inside(root, str(binding["path"]))
    if not path.is_file() or _file_sha(path) != binding["file_sha256"]:
        raise ValueError("actualization-probability predecessor file hash mismatch")
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or value.get("content_sha256") != binding["content_sha256"]
        or _content_sha(value) != binding["content_sha256"]
    ):
        raise ValueError("actualization-probability predecessor content hash mismatch")
    return value


def _validate_config(config: Mapping[str, Any]) -> None:
    if (
        set(config)
        != {
            "schema_version",
            "campaign_id",
            "output_path",
            "predecessors",
            "bridge_domain",
            "admission_policy",
            "seals",
        }
        or config.get("schema_version") != CONFIG_SCHEMA
    ):
        raise ValueError("actualization-probability config shape changed")
    if set(config["predecessors"]) != {
        "source_type_audit",
        "countable_full_law_admission",
        "actualization_history_map",
        "canonical_probability_space",
        "candidate_action_completion",
        "qed_actualization_audit",
    }:
        raise ValueError("actualization-probability predecessor set changed")
    if config.get("bridge_domain") != {
        "candidate_parameters": "fixed regular (g,phi) with mu_g_phi locally finite",
        "history_target": "a measurable history space (H,Sigma_H)",
        "count_target": "N_lf(W) with evaluation sigma algebra",
        "determining_core": "the registered countable Laplace or Mecke core",
    }:
        raise ValueError("actualization-probability bridge domain changed")
    if config.get("admission_policy") != {
        "three_primitive_source_registrations_are_jointly_sufficient": True,
        "pushforward_law_and_expectation_are_compiler_derived": True,
        "compiler_identity_fixture_counts_as_paper_QED_bridge": False,
        "candidate_action_rejection_allowed": False,
    }:
        raise ValueError("actualization-probability admission policy changed")
    if not config.get("seals") or any(config["seals"].values()):
        raise ValueError("actualization-probability seal opened")


def _validate_predecessors(predecessors: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    audit = predecessors["source_type_audit"]
    records = audit.get("candidate_records", [])
    if (
        audit.get("decision_counts") != {"blocked": 2, "pass": 0, "reject": 0}
        or [(row.get("branch_id"), row.get("beta")) for row in records] != BRANCHES
        or any(
            row.get("source_Laplace_core_instances") != 0
            or row.get("source_Mecke_core_instances") != 0
            for row in records
        )
    ):
        raise ValueError("source-type predecessor boundary changed")
    admission = predecessors["countable_full_law_admission"]
    theorem = admission.get("countable_selector_admission_theorem", {})
    if (
        theorem.get("laplace_core_route", {}).get("mathematically_sufficient") is not True
        or theorem.get("mecke_core_route", {}).get("mathematically_sufficient") is not True
        or admission.get("gate_counts", {}).get("source_or_QED_route_certificates") != 0
    ):
        raise ValueError("countable selector admission boundary changed")
    history_records = predecessors["actualization_history_map"].get("candidate_records", [])
    if [(row.get("branch_id"), row.get("beta")) for row in history_records] != BRANCHES or any(
        row.get("paper_typed_history_map_complete") is not False for row in history_records
    ):
        raise ValueError("history-map predecessor boundary changed")
    canonical = predecessors["canonical_probability_space"].get(
        "canonical_conditional_construction", {}
    )
    if (
        canonical.get("paper_or_QED_supplies_this_probability_space") is not False
        or canonical.get("candidate_action_selects_this_probability_space") is not False
    ):
        raise ValueError("canonical probability attribution changed")
    actions = predecessors["candidate_action_completion"].get("completion_hypotheses", [])
    if [(row.get("branch_id"), row.get("beta")) for row in actions] != BRANCHES or any(
        row.get("candidate_action", {}).get("stochastic_law_derived_by_action") is not False
        for row in actions
    ):
        raise ValueError("candidate-action stochastic boundary changed")
    qed_records = predecessors["qed_actualization_audit"].get("candidate_records", [])
    if [(row.get("branch_id"), row.get("beta")) for row in qed_records] != BRANCHES or any(
        row.get("paper_or_QED_channel_kernel_registered") is not False for row in qed_records
    ):
        raise ValueError("QED kernel boundary changed")
    return records


def _primitive_interface() -> list[dict[str, Any]]:
    return [
        {
            "primitive": "candidate_conditioned_history_probability_kernel",
            "signature": "Q:(g,phi)->Prob(H,Sigma_H)",
            "source_status": "absent",
            "required_properties": [
                "probability normalization",
                "measurability in candidate fields",
                "declared conditioning and causal domain",
            ],
        },
        {
            "primitive": "measurable_locally_finite_count_map",
            "signature": "C:(H,Sigma_H)->(N_lf(W),Sigma_eval)",
            "source_status": "partial_semantics_only",
            "required_properties": [
                "evaluation measurability h->C(h)(A)",
                "local finiteness",
                "typed relation between actualization endpoints and counted atoms",
            ],
        },
        {
            "primitive": "source_bound_countable_core_identity",
            "signature": "Cert(Q,C,mu_g_phi,R)=LaplaceCore or MeckeCore",
            "source_status": "absent",
            "required_properties": [
                "all rational core instances",
                "registered derivation or explicit source axiom",
                "candidate and source provenance binding",
            ],
        },
    ]


def _derived_objects() -> list[dict[str, str]]:
    return [
        {
            "object": "pushforward_probability_law",
            "formula": "P_g_phi=C_*Q_g_phi on N_lf(W)",
            "derivation": "Q plus measurable C",
        },
        {
            "object": "counting_measure_expectation",
            "formula": "E_P[F(N)]=Integral_H F(C(h))*Q_g_phi(dh)",
            "derivation": "pushforward change of variables",
        },
    ]


def _composition_theorem() -> dict[str, Any]:
    return {
        "theorem_name": "minimal_actualization_probability_bridge_composition",
        "premises": [
            "Q_g_phi is a source-bound probability kernel on a measurable history space",
            "C is a source-bound measurable locally finite history-to-counting-measure map",
            "the pushforward satisfies one registered countable Laplace or Mecke core identity",
        ],
        "derived_steps": [
            "P_g_phi=C_*Q_g_phi is a probability law on the evaluation sigma algebra",
            "expectations of counting-measure functionals pull back to history expectations",
            "the countable-core admission theorem uniquely gives P_g_phi=PRM(mu_g_phi)",
        ],
        "conclusion": (
            "the three primitive registrations are jointly sufficient for a candidate-bound "
            "source-derived Poisson selector"
        ),
        "minimality": {
            "without_Q": "history expectation and pushforward probability are undefined",
            "without_C": "N(A), local finiteness, and pushforward law are undefined",
            "without_core_identity": "the law is defined but Poisson versus Cox remains unresolved",
        },
        "scope_limit": (
            "minimality is relative to the registered history-pushforward route; an independently "
            "source-derived counting-measure law could bypass histories but is also absent"
        ),
    }


def _exact_controls() -> dict[str, Any]:
    return {
        "compiler_identity_fixture_positive_control": {
            "history_space": "H=N_lf(W)",
            "history_kernel": "Q_g_phi=PRM(mu_g_phi)",
            "count_map": "C=identity",
            "pushforward": "C_*Q_g_phi=PRM(mu_g_phi)",
            "countable_Laplace_core": "pass",
            "source_or_QED_attribution": False,
            "purpose": "schema satisfiability only",
        },
        "missing_Q_negative_control": {
            "present": ["semantic endpoint map", "mean intensity"],
            "missing": "probability kernel Q_g_phi",
            "failure": "expectation and pushforward probability undefined",
            "rejected": True,
        },
        "missing_C_negative_control": {
            "present": ["abstract history probability", "mean intensity"],
            "missing": "measurable locally finite count map C",
            "failure": "set-indexed counts and pushforward law undefined",
            "rejected": True,
        },
        "missing_certificate_negative_control": {
            "present": ["Q_g_phi", "measurable C", "mean intensity"],
            "missing": "Laplace or Mecke core identity",
            "failure": "same-mean Poisson and Cox pushforwards remain nonidentified",
            "rejected": True,
        },
        "semantic_promotion_negative_control": {
            "present": "localized absorption endpoint prose",
            "mutation": "treat prose as a measurable locally finite count map",
            "rejected": True,
        },
    }


def _candidate_record(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "branch_id": record["branch_id"],
        "beta": record["beta"],
        "minimal_bridge_interface_registered_by_compiler": True,
        "primitive_source_registrations_complete": 0,
        "compiler_identity_fixture_pass": True,
        "paper_or_QED_bridge_complete": False,
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
        "bridge_domain",
        "primitive_interface",
        "compiler_derived_objects",
        "composition_theorem",
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
        raise ValueError("actualization-probability result shape changed")
    if result.get("decision_counts") != {"pass": 0, "reject": 0, "blocked": 2}:
        raise ValueError("actualization-probability candidate partition changed")
    if result.get("gate_counts") != EXPECTED_COUNTS:
        raise ValueError("actualization-probability gate counts changed")
    if result.get("primitive_interface") != _primitive_interface():
        raise ValueError("actualization-probability primitive interface changed")
    if result.get("compiler_derived_objects") != _derived_objects():
        raise ValueError("actualization-probability derived objects changed")
    if result.get("composition_theorem") != _composition_theorem():
        raise ValueError("actualization-probability composition theorem changed")
    if result.get("exact_controls") != _exact_controls():
        raise ValueError("actualization-probability controls changed")
    records = result.get("candidate_records", [])
    if [(row.get("branch_id"), row.get("beta")) for row in records] != BRANCHES or any(
        row.get("primitive_source_registrations_complete") != 0
        or row.get("paper_or_QED_bridge_complete") is not False
        or row.get("candidate_action_selects_Poisson") is not False
        or row.get("candidate_action_rejection_authorized") is not False
        or row.get("candidate_decision") != "blocked"
        or row.get("first_blocker") != FIRST_BLOCKER
        for row in records
    ):
        raise ValueError("actualization-probability candidate boundary changed")
    if result.get("first_blocker") != FIRST_BLOCKER:
        raise ValueError("actualization-probability blocker changed")
    if result.get("claim_seals") != EXPECTED_CLAIM_SEALS or any(result["data_seals"].values()):
        raise ValueError("actualization-probability seal changed")
    if (
        result.get("content_sha256") is not None
        and _content_sha(result) != result["content_sha256"]
    ):
        raise ValueError("actualization-probability content hash mismatch")


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
    test_path = root / "tests/test_kastner_schlatter_actualization_probability_bridge_contract.py"
    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA,
        "campaign_id": config["campaign_id"],
        "scope": (
            "smallest candidate-bound history-probability/count-pushforward/core-identity bridge "
            "relative to the registered route; no source primitives or physical selection inferred"
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
        "bridge_domain": config["bridge_domain"],
        "primitive_interface": _primitive_interface(),
        "compiler_derived_objects": _derived_objects(),
        "composition_theorem": _composition_theorem(),
        "exact_controls": _exact_controls(),
        "candidate_records": [_candidate_record(record) for record in records],
        "gate_counts": dict(EXPECTED_COUNTS),
        "decision_counts": {"pass": 0, "reject": 0, "blocked": 2},
        "decision": "minimal_actualization_probability_bridge_registered_source_primitives_blocked",
        "first_blocker": FIRST_BLOCKER,
        "secondary_blockers": [
            "no_QED_probability_law_on_measurable_actualization_histories",
            "no_source_measurability_or_local_finiteness_proof_for_the_endpoint_count_map",
            "no_source_Laplace_or_Mecke_core_identity_on_the_pushforward",
            "no_diffeomorphism_naturality_proof_for_the_history_probability_bridge",
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
