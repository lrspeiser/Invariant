"""Compiler-authored conditional Poisson-kernel completion for positive intensity fields."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

CONFIG_SCHEMA = "sigma-kastner-schlatter-conditional-poisson-kernel-completion-config-1.0"
RESULT_SCHEMA = "sigma-kastner-schlatter-conditional-poisson-kernel-completion-1.0"
FIRST_BLOCKER = (
    "no_paper_or_QED_derived_actualization_history_to_counting_measure_map_or_"
    "principle_selecting_the_compiler_authored_conditional_Poisson_kernel"
)


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
        raise ValueError("conditional Poisson-kernel path escapes repository") from error
    return path


def _bound_predecessor(root: Path, binding: Mapping[str, Any]) -> dict[str, Any]:
    if set(binding) != {"path", "file_sha256", "content_sha256"}:
        raise ValueError("conditional Poisson-kernel predecessor binding shape changed")
    path = _inside(root, str(binding["path"]))
    if not path.is_file() or _file_sha(path) != binding["file_sha256"]:
        raise ValueError("conditional Poisson-kernel predecessor file hash mismatch")
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or value.get("content_sha256") != binding["content_sha256"]
        or _content_sha(value) != binding["content_sha256"]
    ):
        raise ValueError("conditional Poisson-kernel predecessor content hash mismatch")
    return value


def _validate_config(config: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "campaign_id",
        "output_path",
        "predecessor",
        "conditional_kernel_domain",
        "admission_policy",
        "seals",
    }
    if set(config) != required or config.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("conditional Poisson-kernel config shape changed")
    if config.get("conditional_kernel_domain") != {
        "classical_input": "smooth oriented Lorentzian (M,g) and finite real phi",
        "positive_intensity": "q=q0*exp(phi)>0 with q0>0",
        "intensity_measure": "mu_g_phi(B)=Integral_B q0*exp(phi)*dVol_g",
        "local_finiteness_scope": "relatively compact measurable B on each regular field patch",
        "random_output": "locally finite integer-valued counting measure N",
        "conditioning": "conditional on the complete classical pair (g,phi)",
    }:
        raise ValueError("conditional Poisson-kernel domain changed")
    if config.get("admission_policy") != {
        "compiler_authored_external_stochastic_completion_allowed": True,
        "paper_or_QED_attribution_allowed": False,
        "deterministic_action_derivation_inferred": False,
        "conditional_Poisson_is_not_unconditional_Poisson_after_random_field_marginalization": True,
        "candidate_action_rejection_allowed": False,
    }:
        raise ValueError("conditional Poisson-kernel admission policy changed")
    if any(config.get("seals", {}).values()):
        raise ValueError("conditional Poisson-kernel seal opened")


def _validate_predecessor(value: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    counts = value.get("gate_counts", {})
    if (
        value.get("decision_counts") != {"pass": 0, "reject": 0, "blocked": 2}
        or counts.get("minimal_sufficient_selector_contracts") != 3
        or counts.get("registered_selector_nodes") != 0
        or counts.get("registered_action_derivation_edges_to_PMF") != 0
        or counts.get("candidate_action_reject") != 0
        or any(value.get("claim_seals", {}).values())
        or any(value.get("data_seals", {}).values())
    ):
        raise ValueError("Poisson-selector predecessor boundary changed")
    records = value.get("candidate_records", [])
    if (
        [record.get("branch_id") for record in records] != ["eq35_middle_h", "eq35_printed_planck"]
        or any(record.get("paper_or_QED_derived") for record in records)
        or any(record.get("candidate_action_rejection_authorized") for record in records)
    ):
        raise ValueError("Poisson-selector predecessor candidate boundary changed")
    return records


def _kernel_contract() -> dict[str, Any]:
    return {
        "status": "compiler_authored_external_conditional_stochastic_completion",
        "kernel": "K[(g,phi),dN]=PRM(mu_g_phi)(dN)",
        "intensity": "mu_g_phi(dx)=q0*exp(phi(x))*dVol_g(x)",
        "Laplace_functional": {
            "formula": (
                "E[exp(-Integral f*dN)|g,phi]=exp(-Integral_M (1-exp(-f))*q0*exp(phi)*dVol_g)"
            ),
            "domain": "all nonnegative compactly supported measurable f",
            "normalization_f_equals_zero": "1",
            "uniquely_selects_conditional_Poisson_law": True,
        },
        "joint_disjoint_set_PGF": {
            "formula": ("E[product_i z_i^N(B_i)|g,phi]=exp(sum_i mu_g_phi(B_i)*(z_i-1))"),
            "domain": "every finite pairwise-disjoint relatively compact family B_i",
            "independent_increment_factorization": True,
            "marginal": "N(B)|g,phi~Poisson(mu_g_phi(B))",
        },
        "Mecke_identity": {
            "formula": (
                "E[Integral F(x,N)N(dx)|g,phi]=Integral E[F(x,N+delta_x)|g,phi]mu_g_phi(dx)"
            ),
            "domain": "all nonnegative measurable F",
            "characterizes_same_conditional_Poisson_law": True,
        },
        "diffeomorphism_covariance": {
            "formula": "K[(psi_*g,psi_*phi),d(psi_*N)]=K[(g,phi),dN]",
            "measure_identity": "mu_{psi_*g,psi_*phi}(psi(B))=mu_{g,phi}(B)",
            "reason": "phi is a scalar and dVol_g is the metric volume density",
            "pass": True,
        },
        "existence_uniqueness_scope": {
            "premise": "mu_g_phi is sigma-finite and finite on relatively compact sets",
            "conclusion": "a unique Poisson random measure law with this Laplace functional exists",
            "global_singular_spacetime_or_divergent_phi_covered": False,
        },
    }


def _branch_record(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "branch_id": record["branch_id"],
        "beta": record["beta"],
        "compiler_authored_action": True,
        "compiler_authored_conditional_kernel": True,
        "paper_or_QED_derived": False,
        "action_derived": False,
        "candidate_binding": {
            "positive_field": "q=q0*exp(phi)",
            "conditional_kernel": "N|(g,phi)~PRM(q0*exp(phi)*dVol_g)",
            "stationary_reduction": "phi=0 gives N(B)|g,q0~Poisson(q0*Vol_g(B))",
        },
        "gate_ledger": {
            "positive_locally_finite_intensity_measure": "pass",
            "conditional_Laplace_functional_selector": "pass",
            "conditional_independent_increment_family": "pass",
            "conditional_Mecke_identity": "pass",
            "diffeomorphism_covariance": "pass",
            "stationary_scalar_PMF_recovery": "pass",
            "derivation_from_deterministic_candidate_action": "blocked",
            "paper_or_QED_actualization_derivation": "blocked",
            "candidate_action_rejection": "blocked",
        },
        "candidate_action_rejection_authorized": False,
        "candidate_decision": "blocked",
        "first_blocker": FIRST_BLOCKER,
    }


def _controls() -> dict[str, Any]:
    return {
        "stationary_PMF_positive_control": {
            "phi": "0",
            "mu_B": "q0*Vol_g(B)",
            "probability": "P[N(B)=n|g,phi=0]=exp(-mu_B)*mu_B^n/n!",
            "pass": True,
        },
        "two_cell_factorization_positive_control": {
            "mu_1": "2",
            "mu_2": "3",
            "z_1": "1/2",
            "z_2": "1/3",
            "joint_PGF": "exp(-3)",
            "product_of_marginal_PGFs": "exp(-3)",
            "pass": True,
        },
        "dependent_equal_cell_negative_control": {
            "construction": "X~Poisson(2), Y=X",
            "single_cell_marginals": "Poisson(2)",
            "joint_PGF_at_half_half": "exp(-3/2)",
            "required_independent_PGF_at_half_half": "exp(-2)",
            "rejected": True,
        },
        "Cox_same_mean_negative_control": {
            "mu": "2",
            "mixing_epsilon": "1/2",
            "Poisson_second_factorial_moment": "4",
            "Cox_second_factorial_moment": "5",
            "rejected_by_Laplace_Mecke_selector": True,
        },
        "random_field_marginalization_negative_control": {
            "identity": "Var(N(B))=E[mu_B]+Var(mu_B)",
            "unconditional_Poisson_only_if": "Var(mu_B)=0 or an equivalent degenerate mixing law",
            "infer_unconditional_Poisson_from_conditional_kernel": False,
        },
        "nonpositive_intensity_negative_control": {
            "mutation": "replace q0*exp(phi) by an unrestricted signed q",
            "rejected": True,
            "reason": "a negative set intensity cannot define a probability measure",
        },
        "attribution_negative_control": {
            "mutation": "claim the conditional kernel is derived by the paper, QED, or action",
            "rejected": True,
        },
    }


def _validate_result(result: Mapping[str, Any]) -> None:
    if result.get("schema_version") != RESULT_SCHEMA:
        raise ValueError("conditional Poisson-kernel result schema changed")
    if result.get("decision_counts") != {"pass": 0, "reject": 0, "blocked": 2}:
        raise ValueError("conditional Poisson-kernel candidate partition changed")
    if result.get("gate_counts") != {
        "candidate_actions": 2,
        "compiler_authored_conditional_kernels": 2,
        "conditional_Laplace_selector_pass": 2,
        "conditional_independent_increment_pass": 2,
        "conditional_Mecke_identity_pass": 2,
        "diffeomorphism_covariance_pass": 2,
        "stationary_Poisson_PMF_recovery_pass": 2,
        "deterministic_action_derivation_pass": 0,
        "paper_or_QED_actualization_derivation_pass": 0,
        "candidate_action_reject": 0,
        "theory_ontology_observational_pass": 0,
    }:
        raise ValueError("conditional Poisson-kernel gate counts changed")
    contract = result.get("conditional_Poisson_kernel_contract", {})
    if (
        contract.get("Laplace_functional", {}).get("uniquely_selects_conditional_Poisson_law")
        is not True
        or contract.get("joint_disjoint_set_PGF", {}).get("independent_increment_factorization")
        is not True
        or contract.get("Mecke_identity", {}).get("characterizes_same_conditional_Poisson_law")
        is not True
        or contract.get("diffeomorphism_covariance", {}).get("pass") is not True
    ):
        raise ValueError("conditional Poisson-kernel selector contract lost")
    records = result.get("candidate_records", [])
    if len(records) != 2 or any(record.get("paper_or_QED_derived") for record in records):
        raise ValueError("conditional Poisson-kernel attribution changed")
    if any(record.get("action_derived") for record in records):
        raise ValueError("conditional Poisson-kernel action derivation overclaim")
    if any(record.get("candidate_action_rejection_authorized") for record in records):
        raise ValueError("conditional Poisson-kernel gate overreached to action rejection")
    if result.get("first_blocker") != FIRST_BLOCKER:
        raise ValueError("conditional Poisson-kernel first blocker changed")
    if any(result.get("claim_seals", {}).values()) or any(result.get("data_seals", {}).values()):
        raise ValueError("conditional Poisson-kernel seal opened")
    if (
        result.get("content_sha256") is not None
        and _content_sha(result) != result["content_sha256"]
    ):
        raise ValueError("conditional Poisson-kernel content hash mismatch")


def build_gate(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    root = config_path.parents[1]
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    predecessor = _bound_predecessor(root, config["predecessor"])
    predecessor_records = _validate_predecessor(predecessor)
    source_path = Path(__file__).resolve()
    test_path = root / "tests/test_kastner_schlatter_conditional_poisson_kernel_completion_gate.py"
    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA,
        "campaign_id": config["campaign_id"],
        "scope": (
            "candidate-bound compiler-authored conditional Poisson random-measure completion "
            "for each positive scalar-intensity branch"
        ),
        "source_bindings": {
            "Poisson_selector_predecessor": config["predecessor"],
            "point_process_measure_gate": predecessor["source_bindings"][
                "point_process_measure_gate"
            ],
            "candidate_action_completion": predecessor["source_bindings"][
                "candidate_action_completion"
            ],
            "equation_graph": predecessor["source_bindings"]["equation_graph"],
            "primary_pdf_sha256": predecessor["source_bindings"]["primary_pdf_sha256"],
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
        "conditional_kernel_domain": config["conditional_kernel_domain"],
        "conditional_Poisson_kernel_contract": _kernel_contract(),
        "candidate_records": [_branch_record(record) for record in predecessor_records],
        "deterministic_controls": _controls(),
        "gate_counts": {
            "candidate_actions": 2,
            "compiler_authored_conditional_kernels": 2,
            "conditional_Laplace_selector_pass": 2,
            "conditional_independent_increment_pass": 2,
            "conditional_Mecke_identity_pass": 2,
            "diffeomorphism_covariance_pass": 2,
            "stationary_Poisson_PMF_recovery_pass": 2,
            "deterministic_action_derivation_pass": 0,
            "paper_or_QED_actualization_derivation_pass": 0,
            "candidate_action_reject": 0,
            "theory_ontology_observational_pass": 0,
        },
        "decision_counts": {"pass": 0, "reject": 0, "blocked": 2},
        "decision": "conditional_Poisson_kernel_mathematically_closed_physical_selection_blocked",
        "first_blocker": FIRST_BLOCKER,
        "secondary_blockers": [
            "no_QED_actualization_history_sigma_algebra_or_count_map",
            "no_deterministic_action_to_stochastic_kernel_derivation",
            "no_backreaction_or_joint_dynamics_for_events_and_classical_fields",
            "no_observational_transaction_event_or_exposure_registration",
        ],
        "claim_seals": {
            "paper_action_derived": False,
            "QED_actualization_kernel_derived": False,
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
