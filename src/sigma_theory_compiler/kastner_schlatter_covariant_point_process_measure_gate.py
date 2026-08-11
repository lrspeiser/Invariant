"""Covariant point-process measure identifiability gate for intensity actions."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

CONFIG_SCHEMA = "sigma-kastner-schlatter-covariant-point-process-measure-gate-config-1.0"
RESULT_SCHEMA = "sigma-kastner-schlatter-covariant-point-process-measure-gate-1.0"
FIRST_BLOCKER = (
    "no_registered_stochastic_generating_functional_or_QED_event_kernel_to_select_"
    "Poisson_over_a_covariant_Cox_competitor"
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
        raise ValueError("point-process measure path escapes repository") from error
    return path


def _bound_predecessor(root: Path, binding: Mapping[str, Any]) -> dict[str, Any]:
    if set(binding) != {"path", "file_sha256", "content_sha256"}:
        raise ValueError("point-process predecessor binding shape changed")
    path = _inside(root, str(binding["path"]))
    if not path.is_file() or _file_sha(path) != binding["file_sha256"]:
        raise ValueError("point-process predecessor file hash mismatch")
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or value.get("content_sha256") != binding["content_sha256"]
        or _content_sha(value) != binding["content_sha256"]
    ):
        raise ValueError("point-process predecessor content hash mismatch")
    return value


def _validate_config(config: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "campaign_id",
        "output_path",
        "predecessor",
        "measure_domain",
        "admission_policy",
        "seals",
    }
    if set(config) != required or config.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("point-process measure config shape changed")
    if config.get("measure_domain") != {
        "spacetime": "oriented Lorentzian manifold (M,g)",
        "test_sets": "relatively compact measurable B with finite mu_q(B)",
        "positive_intensity": "q=q0*exp(phi)>0 for finite real phi",
        "intensity_measure": "mu_q(B)=Integral_B q*dVol_g",
        "test_functions": "nonnegative compactly supported measurable f",
    }:
        raise ValueError("point-process measure domain changed")
    if config.get("admission_policy") != {
        "first_moment_is_a_complete_probability_law": False,
        "poisson_independent_increments_must_be_derived_or_explicitly_postulated": True,
        "covariant_competing_measure_witness_is_a_derivation_no_go": True,
        "candidate_action_rejection_allowed": False,
        "paper_QED_ontology_observation_inference_allowed": False,
    }:
        raise ValueError("point-process measure admission policy changed")
    if any(config.get("seals", {}).values()):
        raise ValueError("point-process measure seal opened")


def _validate_predecessor(value: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    counts = value.get("gate_counts", {})
    if (
        value.get("decision_counts") != {"pass": 0, "reject": 0, "blocked": 2}
        or counts.get("exact_positive_field_diffeomorphism_pass") != 2
        or counts.get("exact_reparameterized_action_pass") != 2
        or counts.get("action_derived_point_process_measure_pass") != 0
        or counts.get("candidate_action_reject") != 0
        or any(value.get("claim_seals", {}).values())
        or any(value.get("data_seals", {}).values())
    ):
        raise ValueError("positive-reparameterization predecessor boundary changed")
    records = value.get("candidate_records", [])
    if (
        [record.get("branch_id") for record in records] != ["eq35_middle_h", "eq35_printed_planck"]
        or any(record.get("paper_or_QED_derived") for record in records)
        or any(record.get("candidate_action_rejection_authorized") for record in records)
    ):
        raise ValueError("positive-reparameterization candidate boundary changed")
    return records


def _minimal_measure_contract() -> dict[str, Any]:
    return {
        "deterministic_intensity_measure": {
            "formula": "mu_q(B)=Integral_B q*dVol_g",
            "requirements": ["q measurable", "q>=0", "mu_q finite on compact sets"],
            "status_from_predecessor": "closed_for_regular_finite_phi_configurations",
        },
        "random_measure": {
            "object": "N:Omega -> locally finite integer-valued measures on M",
            "requires_probability_space": True,
            "provided_by_deterministic_action": False,
        },
        "poisson_Laplace_functional": {
            "formula": "L_P[f]=exp(-Integral_M (1-exp(-f))*dmu_q)",
            "scope": "f>=0 compactly supported measurable",
            "uniquely_determines_probability_law": True,
            "provided_by_deterministic_action": False,
        },
        "equivalent_count_contract": {
            "marginal": "N(B)~Poisson(mu_q(B))",
            "joint": "counts on pairwise disjoint sets are independent",
            "provided_by_deterministic_action": False,
        },
        "covariance_contract": {
            "formula": "P_{psi_*g,psi_*q}=psi_*P_{g,q}",
            "status": "satisfied_by_both_nonidentifiability_witnesses",
        },
        "physical_event_bridge": {
            "requires": "measurable QED actualization history -> counting measure N",
            "provided_by_paper_bound_action_artifacts": False,
        },
    }


def _nonidentifiability_witness() -> dict[str, Any]:
    return {
        "shared_input": "the same positive scalar q, metric g, and mu_q",
        "model_P": {
            "law": "Poisson random measure with intensity mu_q",
            "Laplace": "L_P[f]=exp(-Integral(1-exp(-f))*dmu_q)",
            "mean_count": "E_P[N(B)]=mu",
            "variance": "Var_P[N(B)]=mu",
            "diffeomorphism_covariant": True,
        },
        "model_C": {
            "law": "global Cox mixture: Z=1-epsilon or 1+epsilon with probability 1/2 each",
            "domain": "0<epsilon<1; conditional on Z, N is Poisson with intensity Z*mu_q",
            "Laplace": (
                "L_C[f]=(1/2)*exp(-(1-epsilon)*I_f)+(1/2)*exp(-(1+epsilon)*I_f), "
                "I_f=Integral(1-exp(-f))*dmu_q"
            ),
            "mean_count": "E_C[N(B)]=E[Z]*mu=mu",
            "variance": "Var_C[N(B)]=mu+epsilon^2*mu^2",
            "diffeomorphism_covariant": True,
        },
        "exact_separation": {
            "void_probability": (
                "P_C[N(B)=0]=(exp(-(1-epsilon)*mu)+exp(-(1+epsilon)*mu))/2"
                ">exp(-mu)=P_P[N(B)=0] for epsilon>0 and mu>0"
            ),
            "strict_inequality_reason": "strict convexity of exp(-z*mu) in z",
            "second_factorial_moment_P": "mu^2",
            "second_factorial_moment_C": "(1+epsilon^2)*mu^2",
            "same_first_moment": True,
            "different_probability_laws": True,
        },
        "conclusion": (
            "q*dVol_g and the deterministic action equations do not identify the Poisson law; "
            "a generating functional, independent-increment axiom, or event kernel is additional data"
        ),
    }


def _record(predecessor: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "branch_id": predecessor["branch_id"],
        "beta": predecessor["beta"],
        "compiler_authored_action": True,
        "paper_or_QED_derived": False,
        "positive_intensity_coordinate_available": True,
        "covariant_intensity_measure_construction": "pass",
        "action_output_inventory": {
            "metric_and_scalar_EL_equations": True,
            "positive_scalar_intensity_on_regular_restricted_sector": True,
            "probability_space": False,
            "random_counting_measure": False,
            "Laplace_or_probability_generating_functional": False,
            "independent_increment_axiom": False,
            "QED_event_kernel": False,
        },
        "gate_ledger": {
            "locally_finite_covariant_intensity_measure": "pass",
            "Poisson_probability_measure_from_action_alone": "reject",
            "Poisson_probability_measure_as_explicit_external_postulate": "pass",
            "Poisson_probability_measure_as_paper_or_QED_derivation": "blocked",
            "QED_actualization_to_counting_measure_bridge": "blocked",
            "candidate_action_rejection": "blocked",
        },
        "derivation_no_go": (
            "Poisson and covariant Cox laws share every deterministic action input and first "
            "intensity moment but have distinct Laplace functionals"
        ),
        "external_poisson_postulate_is_mathematically_well_posed": True,
        "external_poisson_postulate_is_action_derived": False,
        "candidate_action_rejection_authorized": False,
        "candidate_decision": "blocked",
        "first_blocker": FIRST_BLOCKER,
    }


def _controls() -> dict[str, Any]:
    return {
        "exact_moment_separation_control": {
            "mu": "2",
            "epsilon": "1/2",
            "Poisson_mean": "2",
            "Cox_mean": "2",
            "Poisson_variance": "2",
            "Cox_variance": "3",
            "Poisson_second_factorial_moment": "4",
            "Cox_second_factorial_moment": "5",
            "separates_laws_exactly": True,
        },
        "zero_mixing_positive_control": {
            "epsilon": "0",
            "Cox_reduces_to_Poisson": True,
        },
        "mean_only_negative_control": {
            "mutation": "infer Poisson law from E[N(B)]=mu_q(B)",
            "rejected": True,
            "counterexample": "covariant Cox model_C",
        },
        "covariance_only_negative_control": {
            "mutation": "infer uniqueness from diffeomorphism covariance",
            "rejected": True,
            "reason": "both model_P and model_C are covariant",
        },
        "action_overclaim_negative_control": {
            "mutation": "reject either action because stochastic data are absent",
            "rejected": True,
        },
    }


def _validate_result(result: Mapping[str, Any]) -> None:
    if result.get("schema_version") != RESULT_SCHEMA:
        raise ValueError("point-process measure result schema changed")
    if result.get("decision_counts") != {"pass": 0, "reject": 0, "blocked": 2}:
        raise ValueError("point-process measure candidate partition changed")
    if result.get("gate_counts") != {
        "candidate_actions": 2,
        "covariant_intensity_measure_pass": 2,
        "minimal_probability_measure_contracts_registered": 1,
        "exact_covariant_nonidentifiability_witnesses": 2,
        "action_only_Poisson_derivation_pass": 0,
        "action_only_Poisson_derivation_reject": 2,
        "external_Poisson_postulate_well_posed": 2,
        "paper_or_QED_Poisson_derivation_pass": 0,
        "candidate_action_reject": 0,
        "paper_QED_ontology_observational_pass": 0,
    }:
        raise ValueError("point-process measure gate counts changed")
    records = result.get("candidate_records", [])
    if len(records) != 2 or any(record.get("paper_or_QED_derived") for record in records):
        raise ValueError("point-process measure attribution changed")
    if any(record.get("candidate_action_rejection_authorized") for record in records):
        raise ValueError("point-process measure gate overreached to action rejection")
    witness = result.get("exact_nonidentifiability_witness", {})
    if (
        not witness.get("exact_separation", {}).get("same_first_moment")
        or not witness.get("exact_separation", {}).get("different_probability_laws")
        or not witness.get("model_P", {}).get("diffeomorphism_covariant")
        or not witness.get("model_C", {}).get("diffeomorphism_covariant")
    ):
        raise ValueError("point-process measure no-go witness lost")
    if result.get("first_blocker") != FIRST_BLOCKER:
        raise ValueError("point-process measure first blocker changed")
    if any(result.get("claim_seals", {}).values()) or any(result.get("data_seals", {}).values()):
        raise ValueError("point-process measure seal opened")
    if (
        result.get("content_sha256") is not None
        and _content_sha(result) != result["content_sha256"]
    ):
        raise ValueError("point-process measure content hash mismatch")


def build_gate(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    root = config_path.parents[1]
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    predecessor = _bound_predecessor(root, config["predecessor"])
    predecessor_records = _validate_predecessor(predecessor)
    source_path = Path(__file__).resolve()
    test_path = root / "tests/test_kastner_schlatter_covariant_point_process_measure_gate.py"
    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA,
        "campaign_id": config["campaign_id"],
        "scope": (
            "minimal covariant probability-measure contract and exact nonidentifiability of a "
            "Poisson law from either compiler-authored deterministic action"
        ),
        "source_bindings": {
            "positive_reparameterization_predecessor": config["predecessor"],
            "positive_intensity_predecessor": predecessor["source_bindings"][
                "positive_intensity_predecessor"
            ],
            "candidate_action_completion": predecessor["source_bindings"][
                "candidate_action_completion"
            ],
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
        "measure_domain": config["measure_domain"],
        "minimal_covariant_probability_measure_contract": _minimal_measure_contract(),
        "exact_nonidentifiability_witness": _nonidentifiability_witness(),
        "candidate_records": [_record(record) for record in predecessor_records],
        "deterministic_controls": _controls(),
        "gate_counts": {
            "candidate_actions": 2,
            "covariant_intensity_measure_pass": 2,
            "minimal_probability_measure_contracts_registered": 1,
            "exact_covariant_nonidentifiability_witnesses": 2,
            "action_only_Poisson_derivation_pass": 0,
            "action_only_Poisson_derivation_reject": 2,
            "external_Poisson_postulate_well_posed": 2,
            "paper_or_QED_Poisson_derivation_pass": 0,
            "candidate_action_reject": 0,
            "paper_QED_ontology_observational_pass": 0,
        },
        "decision_counts": {"pass": 0, "reject": 0, "blocked": 2},
        "decision": "covariant_intensity_closed_action_only_Poisson_derivation_rejected",
        "first_blocker": FIRST_BLOCKER,
        "secondary_blockers": [
            "no_QED_actualization_history_to_random_counting_measure_kernel",
            "no_independent_increment_or_Laplace_functional_derivation",
            "no_global_regular_solution_or_boundary_completion_theorem",
            "no_observational_transaction_event_or_exposure_registration",
        ],
        "claim_seals": {
            "paper_action_derived": False,
            "QED_actualization_action_derived": False,
            "Poisson_law_action_derived": False,
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
