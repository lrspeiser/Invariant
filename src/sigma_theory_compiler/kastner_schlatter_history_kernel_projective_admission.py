"""Projective-cylinder admission gate for a KS actualization history kernel."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

CONFIG_SCHEMA = "sigma-kastner-schlatter-history-kernel-projective-config-1.0"
RESULT_SCHEMA = "sigma-kastner-schlatter-history-kernel-projective-admission-1.0"
PDF_SHA256 = "c2f671293d07b21397e745da00a3ce1a2193c00da647a2ebf4147612b76c1780"
BRANCHES = [("eq35_middle_h", "1/2"), ("eq35_printed_planck", "1/4")]
FIRST_BLOCKER = (
    "no_source_bound_candidate_conditioned_projectively_consistent_finite_partition_"
    "probability_family_on_a_declared_countable_history_generator"
)

EXPECTED_COUNTS = {
    "candidate_actions": 2,
    "candidate_blocked": 2,
    "candidate_pass": 0,
    "candidate_reject": 0,
    "history_kernel_admission_obligations": 6,
    "source_complete_obligations": 0,
    "source_partial_obligations": 1,
    "source_absent_obligations": 5,
    "compiler_projective_positive_controls": 1,
    "exact_projective_negative_controls": 1,
    "exact_same_input_distinct_history_law_witnesses": 2,
    "paper_or_QED_history_kernels": 0,
    "theory_ontology_observational_pass": 0,
}

EXPECTED_CLAIM_SEALS = {
    "paper_QED_history_kernel_registered": False,
    "paper_QED_finite_partition_family_registered": False,
    "paper_QED_projective_consistency_proven": False,
    "paper_QED_candidate_measurability_proven": False,
    "paper_QED_nonexplosion_proven": False,
    "candidate_action_stochastic_law_derived": False,
    "candidate_action_selects_Poisson": False,
    "candidate_action_rejected": False,
    "compiler_projective_fixture_attributed_to_source": False,
    "compiler_nonidentifiability_witness_attributed_to_source": False,
    "transaction_ontology_validated": False,
    "observational_pass": False,
    "scientific_test_pass": False,
    "theory_validity_claimed": False,
    "dark_sector_elimination_proven": False,
}
EXPECTED_DATA_SEALS = {
    "observations_opened": False,
    "transaction_event_observations_opened": False,
    "dark_matter_or_halo_inputs": False,
    "redshift_or_cosmology_inputs": False,
    "solar_system_inputs": False,
    "QED_actualization_derivation_opened": False,
    "transaction_ontology_validated": False,
    "paper_action_attribution_allowed": False,
    "scientific_test_pass_allowed": False,
    "paid_llm_calls": False,
}
EXPECTED_DOMAIN = {
    "candidate_parameters": "fixed regular (g,phi)",
    "generator": "declared countable ring R with finite partitions",
    "history_coordinates": "locally finite count vectors on partitions of R",
    "target": "Q_g_phi on the cylinder sigma algebra of N_lf(W)",
}
EXPECTED_SECONDARY_BLOCKERS = [
    "no_source_declared_standard_Borel_actualization_history_space",
    "no_source_projective_coarsening_identities_for_finite_partition_counts",
    "no_source_candidate_measurability_or_nonexplosion_theorem",
    "no_source_bound_countable_Laplace_or_Mecke_selector_after_Q",
]
EXPECTED_SCOPE = (
    "candidate-bound projective-cylinder prerequisite for Q_g_phi and exact two-cell "
    "nonidentifiability; no paper/QED stochastic law or physical claim inferred"
)
EXPECTED_PREDECESSORS = {
    "actualization_probability_bridge": {
        "path": "runs/engine/kastner-schlatter-actualization-probability-bridge-contract.json",
        "file_sha256": "2a04cc415dbfe7a3fc3b37b8d74ae274fb66623984bfeb5633c86f29b8ec2a18",
        "content_sha256": "fa959c92c275e1094570e54684ac35edaab29fd0f3cd6ebab6140fd634a7fcc4",
    },
    "source_type_audit": {
        "path": "runs/engine/kastner-schlatter-source-selector-type-inhabitation-audit.json",
        "file_sha256": "ea6ce6d05573fde8fc9164b9372fd3b5d9404bc34a69de6101f2261134f25620",
        "content_sha256": "ba2f97b87e094c1262017ce9e2deded0a27185dd39ee23c85acd8c7c8ae7085e",
    },
    "qed_actualization_audit": {
        "path": "runs/engine/kastner-schlatter-qed-actualization-poisson-derivation-audit.json",
        "file_sha256": "b7891a450b43abc913c1d360c5f38edb117925a39939c629578c358f5931b92b",
        "content_sha256": "cec941a0455608f437c2fb8a79fee42dec6d614b9a8f23ef327300ea4d72e024",
    },
}
EXPECTED_RESULT_KEYS = {
    "admission_domain",
    "admission_obligations",
    "campaign_id",
    "candidate_records",
    "claim_seals",
    "content_sha256",
    "data_seals",
    "decision",
    "decision_counts",
    "exact_controls",
    "exact_nonidentifiability_witness",
    "extension_theorem",
    "first_blocker",
    "gate_counts",
    "schema_version",
    "scope",
    "secondary_blockers",
    "source_bindings",
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
        raise ValueError("history-kernel projective path escapes repository") from error
    return path


def _bound_artifact(root: Path, binding: Mapping[str, Any]) -> dict[str, Any]:
    if set(binding) != {"path", "file_sha256", "content_sha256"}:
        raise ValueError("history-kernel predecessor binding shape changed")
    path = _inside(root, str(binding["path"]))
    if not path.is_file() or _file_sha(path) != binding["file_sha256"]:
        raise ValueError("history-kernel predecessor file hash mismatch")
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or value.get("content_sha256") != binding["content_sha256"]
        or _content_sha(value) != binding["content_sha256"]
    ):
        raise ValueError("history-kernel predecessor content hash mismatch")
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
        or set(config.get("predecessors", {}))
        != {"actualization_probability_bridge", "source_type_audit", "qed_actualization_audit"}
    ):
        raise ValueError("history-kernel projective config shape changed")
    if config.get("admission_domain") != EXPECTED_DOMAIN:
        raise ValueError("history-kernel projective admission domain changed")
    if config.get("admission_policy") != {
        "projective_family_is_a_prerequisite_for_history_kernel": True,
        "scalar_total_count_PMF_and_mean_are_sufficient": False,
        "compiler_witness_counts_as_source_or_QED_derivation": False,
        "candidate_action_rejection_allowed": False,
    }:
        raise ValueError("history-kernel projective admission policy changed")
    if not config.get("seals") or any(config["seals"].values()):
        raise ValueError("history-kernel projective seal opened")


def _validate_predecessors(predecessors: Mapping[str, Mapping[str, Any]]) -> None:
    bridge = predecessors["actualization_probability_bridge"]
    if (
        bridge.get("decision_counts") != {"blocked": 2, "pass": 0, "reject": 0}
        or bridge.get("first_blocker")
        != "no_source_bound_candidate_conditioned_QED_history_probability_kernel_measurable_locally_finite_count_pushforward_and_countable_core_identity"
        or bridge.get("gate_counts", {}).get("source_complete_primitives") != 0
    ):
        raise ValueError("actualization-probability bridge boundary changed")
    source_audit = predecessors["source_type_audit"]
    if (
        source_audit.get("gate_counts", {}).get("registered_scalar_PMF_nodes") != 1
        or source_audit.get("claim_seals", {}).get("paper_scalar_PMF_equation_claimed")
        is not False
    ):
        raise ValueError("source-type scalar-PMF boundary changed")
    qed = predecessors["qed_actualization_audit"]
    records = qed.get("candidate_records", [])
    if [(row.get("branch_id"), row.get("beta")) for row in records] != BRANCHES or any(
        row.get("paper_or_QED_channel_kernel_registered") is not False for row in records
    ):
        raise ValueError("QED actualization-kernel boundary changed")


def _obligations() -> list[dict[str, str]]:
    return [
        {
            "obligation": "declared_history_space_and_countable_generator",
            "source_status": "absent",
            "required": "standard Borel history space and countable generating ring R",
        },
        {
            "obligation": "normalized_finite_partition_probability_kernels",
            "source_status": "absent",
            "required": "q^P_g_phi(n_1,...,n_m) for every registered finite partition P",
        },
        {
            "obligation": "coarsening_projective_consistency",
            "source_status": "absent",
            "required": "pushforward of q^P under every partition coarsening equals q^P_prime",
        },
        {
            "obligation": "candidate_parameter_measurability",
            "source_status": "partial_intensity_only",
            "required": "(g,phi)->q^P_g_phi(B) measurable for every cylinder B",
        },
        {
            "obligation": "local_tightness_and_nonexplosion",
            "source_status": "absent",
            "required": "finite counts on every relatively compact generator set almost surely",
        },
        {
            "obligation": "source_or_QED_derivation_of_the_family",
            "source_status": "absent",
            "required": "paper/QED rule fixing every q^P rather than a compiler completion",
        },
    ]


def _extension_theorem() -> dict[str, Any]:
    return {
        "theorem_name": "countable_projective_history_kernel_admission",
        "premises": [
            "a declared standard Borel locally finite history target with countable generator R",
            "normalized candidate-measurable finite-partition kernels q^P_g_phi",
            "projective consistency under every registered coarsening",
            "local tightness/nonexplosion on the generator",
        ],
        "conclusion": (
            "the cylinder family determines a unique candidate-conditioned probability kernel "
            "Q_g_phi on the generated history sigma algebra"
        ),
        "current_source_closure": "zero complete, one partial, five absent",
        "scope_limit": (
            "conditional admission theorem only; it neither derives the family from the paper/QED "
            "nor selects Poisson without a determining core identity"
        ),
    }


def _same_input_witness() -> dict[str, Any]:
    return {
        "witness_name": "same_total_Poisson_and_equal_cell_mean_distinct_history_laws",
        "space": "two-point locally finite counting space W={A,B}",
        "shared_inputs": {
            "total_count": "K=N_A+N_B~Poisson(2)",
            "cell_means": {"E[N_A]": "1", "E[N_B]": "1"},
            "total_count_PMF": "P(K=k)=exp(-2)*2^k/k!",
        },
        "law_independent_split": {
            "construction": "N_A,N_B independent Poisson(1)",
            "P_N_B_zero": "exp(-1)",
        },
        "law_coherent_allocation": {
            "construction": (
                "draw K~Poisson(2), then with probability 1/2 place all K in A and otherwise all K in B"
            ),
            "P_N_B_zero": "(1+exp(-2))/2",
        },
        "exact_separation": "(1+exp(-2))/2-exp(-1)=(1-exp(-1))^2/2>0",
        "conclusion": (
            "the scalar total-count PMF plus the complete two-cell mean vector does not determine "
            "the finite-partition law, hence cannot determine Q_g_phi"
        ),
        "source_or_QED_attribution": False,
    }


def _controls() -> dict[str, Any]:
    return {
        "compiler_independent_PRM_positive_control": {
            "fine_partition": "N_A,N_B independent Poisson(1)",
            "coarse_partition": "K=N_A+N_B~Poisson(2)",
            "coarsening_identity": "sum of independent Poisson(1) variables is Poisson(2)",
            "projective_consistency": "pass",
            "source_or_QED_attribution": False,
        },
        "inconsistent_coarsening_negative_control": {
            "fine_partition": "q^{A,B}=delta_(1,0)",
            "declared_coarse_partition": "q^{A_union_B}=delta_0",
            "actual_fine_pushforward": "delta_1",
            "rejected": True,
            "reason": "coarse law differs from the pushforward of the fine law",
        },
        "scalar_PMF_promotion_negative_control": {
            "present": ["one scalar total-count PMF node", "candidate mean intensity fragment"],
            "missing": [
                "finite-partition kernels",
                "coarsening consistency",
                "candidate measurability of cylinder probabilities",
                "nonexplosion",
            ],
            "rejected": True,
        },
    }


def _candidate_records() -> list[dict[str, Any]]:
    return [
        {
            "branch_id": branch,
            "beta": beta,
            "projective_interface_registered_by_compiler": True,
            "source_complete_obligations": 0,
            "same_input_distinct_history_law_witness": True,
            "paper_or_QED_history_kernel_registered": False,
            "candidate_action_selects_history_law": False,
            "candidate_action_rejection_authorized": False,
            "candidate_decision": "blocked",
            "first_blocker": FIRST_BLOCKER,
        }
        for branch, beta in BRANCHES
    ]


def _validate_source_bindings(result: Mapping[str, Any], root: Path) -> None:
    bindings = result.get("source_bindings", {})
    if (
        not isinstance(bindings, Mapping)
        or set(bindings)
        != {
            "actualization_probability_bridge",
            "source_type_audit",
            "qed_actualization_audit",
            "primary_pdf_sha256",
            "config",
            "source",
            "test",
        }
        or bindings.get("primary_pdf_sha256") != PDF_SHA256
    ):
        raise ValueError("history-kernel projective source binding set changed")
    for label, expected in EXPECTED_PREDECESSORS.items():
        if bindings.get(label) != expected:
            raise ValueError("history-kernel projective predecessor binding changed")
        _bound_artifact(root, expected)
    expected_paths = {
        "config": "configs/kastner_schlatter_history_kernel_projective_admission.json",
        "source": (
            "src/sigma_theory_compiler/"
            "kastner_schlatter_history_kernel_projective_admission.py"
        ),
        "test": "tests/test_kastner_schlatter_history_kernel_projective_admission.py",
    }
    for label, relative in expected_paths.items():
        binding = bindings.get(label, {})
        if (
            not isinstance(binding, Mapping)
            or set(binding) != {"path", "file_sha256"}
            or binding.get("path") != relative
            or _file_sha(_inside(root, relative)) != binding.get("file_sha256")
        ):
            raise ValueError("history-kernel projective local source binding changed")


def _validate_result(result: Mapping[str, Any], *, root: Path | None = None) -> None:
    body = {key: value for key, value in result.items() if key != "content_sha256"}
    if result.get("content_sha256") != _sha(body):
        raise ValueError("history-kernel projective content hash changed")
    validation_root = (root or Path(__file__).resolve().parents[2]).resolve()
    _validate_source_bindings(result, validation_root)
    if (
        set(result) != EXPECTED_RESULT_KEYS
        or result.get("schema_version") != RESULT_SCHEMA
        or result.get("campaign_id") != "kastner-schlatter-history-kernel-projective-001"
        or result.get("decision")
        != "projective_history_kernel_contract_registered_scalar_inputs_nonidentifying"
        or result.get("decision_counts") != {"pass": 0, "reject": 0, "blocked": 2}
        or result.get("gate_counts") != EXPECTED_COUNTS
        or result.get("first_blocker") != FIRST_BLOCKER
        or result.get("admission_domain") != EXPECTED_DOMAIN
        or result.get("admission_obligations") != _obligations()
        or result.get("extension_theorem") != _extension_theorem()
        or result.get("exact_nonidentifiability_witness") != _same_input_witness()
        or result.get("exact_controls") != _controls()
        or result.get("candidate_records") != _candidate_records()
        or result.get("secondary_blockers") != EXPECTED_SECONDARY_BLOCKERS
        or result.get("claim_seals") != EXPECTED_CLAIM_SEALS
        or any(result.get("claim_seals", {}).values())
        or result.get("data_seals") != EXPECTED_DATA_SEALS
        or result.get("scope") != EXPECTED_SCOPE
    ):
        raise ValueError("history-kernel projective result boundary changed")


def build_gate(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    root = config_path.parents[1]
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    predecessors = {
        label: _bound_artifact(root, binding)
        for label, binding in config["predecessors"].items()
    }
    _validate_predecessors(predecessors)
    source_path = Path(__file__).resolve()
    test_path = root / "tests/test_kastner_schlatter_history_kernel_projective_admission.py"
    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA,
        "campaign_id": config["campaign_id"],
        "decision": "projective_history_kernel_contract_registered_scalar_inputs_nonidentifying",
        "decision_counts": {"pass": 0, "reject": 0, "blocked": 2},
        "gate_counts": EXPECTED_COUNTS,
        "first_blocker": FIRST_BLOCKER,
        "admission_domain": config["admission_domain"],
        "admission_obligations": _obligations(),
        "extension_theorem": _extension_theorem(),
        "exact_nonidentifiability_witness": _same_input_witness(),
        "exact_controls": _controls(),
        "candidate_records": _candidate_records(),
        "secondary_blockers": EXPECTED_SECONDARY_BLOCKERS,
        "claim_seals": EXPECTED_CLAIM_SEALS,
        "data_seals": config["seals"],
        "scope": EXPECTED_SCOPE,
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
    }
    result["content_sha256"] = _sha(result)
    _validate_result(result, root=root)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = build_gate(args.config)
    output = args.output or (args.config.resolve().parents[1] / json.loads(
        args.config.read_text(encoding="utf-8")
    )["output_path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
