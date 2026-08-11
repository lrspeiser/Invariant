"""Exact Poisson compatibility audit for compiler-authored transaction actions."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

CONFIG_SCHEMA = "sigma-kastner-schlatter-poisson-action-compatibility-config-1.0"
RESULT_SCHEMA = "sigma-kastner-schlatter-poisson-action-compatibility-1.0"
FIRST_BLOCKER = "no_action_derived_covariant_point_process_measure_or_positive_intensity_dynamics"


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _content_sha(value: Mapping[str, Any]) -> str:
    return _sha({key: item for key, item in value.items() if key != "content_sha256"})


def _load_bound(root: Path, binding: Mapping[str, Any]) -> dict[str, Any]:
    if set(binding) != {"path", "file_sha256", "content_sha256"}:
        raise ValueError("predecessor binding shape changed")
    path = (root / str(binding["path"])).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError("predecessor path escapes repository") from error
    if not path.is_file() or _file_sha(path) != binding["file_sha256"]:
        raise ValueError("predecessor file hash mismatch")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("predecessor must be an object")
    if value.get("content_sha256") != binding["content_sha256"]:
        raise ValueError("predecessor declared content hash mismatch")
    if _content_sha(value) != binding["content_sha256"]:
        raise ValueError("predecessor content hash mismatch")
    return value


def _validate_config(config: Mapping[str, Any]) -> None:
    if set(config) != {
        "schema_version",
        "campaign_id",
        "output_path",
        "predecessor",
        "seals",
    } or config.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("Poisson compatibility config shape changed")
    if config.get("seals") != {
        "action_derived_probability_measure_claim_allowed": False,
        "observations_opened": False,
        "paid_llm_calls": False,
        "paper_or_qed_derivation_claim_allowed": False,
        "theory_or_ontology_pass_allowed": False,
    }:
        raise ValueError("Poisson compatibility seals changed")


def _validate_predecessor(value: Mapping[str, Any]) -> None:
    if value.get("counts") != {
        "complete_local_deterministic_action_hypotheses": 2,
        "conditional_exact_eq35_branch_matches": 2,
        "normalization_branches": 2,
        "normalization_branches_selected_as_fact": 0,
        "observational_or_theory_passes": 0,
        "paper_derived_actions": 0,
    }:
        raise ValueError("candidate-action predecessor counts changed")
    branches = value.get("completion_hypotheses", [])
    if [item.get("beta") for item in branches] != ["1/2", "1/4"]:
        raise ValueError("candidate-action branches changed")
    for item in branches:
        action = item.get("candidate_action", {})
        stochastic = item.get("conditional_stochastic_completion", {})
        if action.get("stochastic_law_derived_by_action") is not False:
            raise ValueError("predecessor stochastic scope changed")
        if stochastic.get("law") != "N(B)|q ~ Poisson(mu_B), mu_B=Integral_B q*dVol_g":
            raise ValueError("predecessor conditional point-process law changed")
        if stochastic.get("derived_from_local_action") is not False:
            raise ValueError("conditional law was promoted to an action result")


def _branch_certificate(branch: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "branch_id": branch["branch_id"],
        "beta": branch["beta"],
        "deterministic_intensity_euler_lagrange": "B_q*Box(q)-A_q*(q-q0)=0",
        "stationary_background": "q=q0>0",
        "external_conditional_law": "N(B)|q ~ Poisson(mu_B), mu_B=Integral_B q*dVol_g",
        "stationary_conditional_law": "N(B)|q0 ~ Poisson(q0*Vol_g(B))",
        "stationary_homogeneous_poisson_interface_closes": True,
        "conditional_mean": "E[N(B)|q]=mu_B",
        "conditional_variance": "Var(N(B)|q)=mu_B",
        "marginal_mean": "E[N(B)]=E[mu_B]",
        "marginal_variance": "Var(N(B))=E[mu_B]+Var(mu_B)",
        "second_factorial_cumulant": "kappa_2_factorial(N(B))=Var(mu_B)",
        "homogeneous_poisson_requires": "Var(mu_B)=0 for every registered region B",
        "beta_enters_count_law": False,
        "probability_measure_derived_from_action": False,
        "qed_actualization_map_derived": False,
        "positive_intensity_preserved_by_scalar_dynamics": False,
        "paper_or_theory_pass": False,
    }


def _validate_result(result: Mapping[str, Any]) -> None:
    if result.get("schema_version") != RESULT_SCHEMA:
        raise ValueError("Poisson compatibility result schema changed")
    certificates = result.get("branch_certificates", [])
    if len(certificates) != 2 or [item.get("beta") for item in certificates] != ["1/2", "1/4"]:
        raise ValueError("Poisson compatibility branch partition changed")
    for item in certificates:
        if item.get("marginal_variance") != "Var(N(B))=E[mu_B]+Var(mu_B)":
            raise ValueError("mixed-Poisson variance identity changed")
        if item.get("stationary_homogeneous_poisson_interface_closes") is not True:
            raise ValueError("stationary conditional interface lost")
        if item.get("probability_measure_derived_from_action") is not False:
            raise ValueError("action-derived probability measure overclaimed")
        if item.get("positive_intensity_preserved_by_scalar_dynamics") is not False:
            raise ValueError("positive-intensity dynamics overclaimed")
        if item.get("qed_actualization_map_derived") is not False:
            raise ValueError("QED actualization overclaimed")
    if result.get("exact_mixed_poisson_control") != {
        "intensity_support": ["1", "3"],
        "probabilities": ["1/2", "1/2"],
        "E_mu": "2",
        "Var_mu": "1",
        "E_N": "2",
        "Var_N": "3",
        "Fano_factor": "3/2",
        "homogeneous_poisson_rejected_for_fluctuating_intensity": True,
    }:
        raise ValueError("mixed-Poisson control changed")
    if result.get("counts") != {
        "candidate_action_branches": 2,
        "conditional_covariant_point_process_interfaces": 2,
        "stationary_homogeneous_poisson_matches": 2,
        "action_derived_point_process_measures": 0,
        "positive_intensity_preservation_theorems": 0,
        "qed_actualization_derivations": 0,
        "fluctuating_intensity_homogeneous_poisson_closures": 0,
        "observational_or_theory_passes": 0,
    }:
        raise ValueError("Poisson compatibility counts changed")
    if result.get("first_blocker") != FIRST_BLOCKER:
        raise ValueError("Poisson compatibility blocker changed")
    if any(result.get("claim_seals", {}).values()) or any(result.get("data_seals", {}).values()):
        raise ValueError("Poisson compatibility seal opened")
    declared = result.get("content_sha256")
    if declared is not None and _content_sha(result) != declared:
        raise ValueError("Poisson compatibility content hash mismatch")


def build_audit(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    root = config_path.parents[1]
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    predecessor = _load_bound(root, config["predecessor"])
    _validate_predecessor(predecessor)
    source_path = Path(__file__).resolve()
    test_path = root / "tests/test_kastner_schlatter_poisson_action_compatibility.py"
    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA,
        "campaign_id": config["campaign_id"],
        "decision": "stationary_conditional_poisson_interface_closed_dynamic_derivation_blocked",
        "scope": (
            "exact probability-law compatibility audit of compiler-authored deterministic actions; "
            "the conditional point-process law remains an external hypothesis"
        ),
        "source_bindings": {
            "candidate_action": dict(config["predecessor"]),
            "config": {
                "path": str(config_path.relative_to(root)).replace("\\", "/"),
                "file_sha256": _file_sha(config_path),
            },
            "source": {
                "path": str(source_path.relative_to(root)).replace("\\", "/"),
                "file_sha256": _file_sha(source_path),
            },
            "test": {
                "path": str(test_path.relative_to(root)).replace("\\", "/"),
                "file_sha256": _file_sha(test_path),
            },
        },
        "covariant_point_process_contract": {
            "counting_measure": "N(B) for measurable spacetime region B",
            "invariant_exposure": "dVol_g=sqrt(abs(g))*d4x",
            "conditional_intensity_measure": "mu_B=Integral_B q*dVol_g",
            "required_domain": "q(x)>=0 almost everywhere and locally integrable",
            "disjoint_region_independence": "part of the external conditional Poisson law",
            "provided_by_deterministic_euler_lagrange_action": False,
        },
        "branch_certificates": [
            _branch_certificate(item) for item in predecessor["completion_hypotheses"]
        ],
        "mixed_poisson_theorem": {
            "law_of_total_expectation": "E[N(B)]=E[mu_B]",
            "law_of_total_variance": "Var(N(B))=E[mu_B]+Var(mu_B)",
            "consequence": (
                "random integrated intensity generically produces an overdispersed Cox/mixed-Poisson "
                "count, not the homogeneous Poisson law"
            ),
            "exact_homogeneous_poisson_condition": "Var(mu_B)=0 for every registered B",
        },
        "exact_mixed_poisson_control": {
            "intensity_support": ["1", "3"],
            "probabilities": ["1/2", "1/2"],
            "E_mu": "2",
            "Var_mu": "1",
            "E_N": "2",
            "Var_N": "3",
            "Fano_factor": "3/2",
            "homogeneous_poisson_rejected_for_fluctuating_intensity": True,
        },
        "counts": {
            "candidate_action_branches": 2,
            "conditional_covariant_point_process_interfaces": 2,
            "stationary_homogeneous_poisson_matches": 2,
            "action_derived_point_process_measures": 0,
            "positive_intensity_preservation_theorems": 0,
            "qed_actualization_derivations": 0,
            "fluctuating_intensity_homogeneous_poisson_closures": 0,
            "observational_or_theory_passes": 0,
        },
        "first_blocker": FIRST_BLOCKER,
        "secondary_blockers": [
            "no_QED_actualization_to_operational_transaction_event_map",
            "no_action_derived_probability_path_measure_for_q",
            "no_global_q_nonnegativity_preservation_or_positive_field_parameterization",
            "no_observational_transaction_event_and_exposure_registration",
        ],
        "claim_seals": {
            "action_derived_stochastic_law_claimed": False,
            "qed_actualization_derivation_claimed": False,
            "paper_transaction_ontology_validated": False,
            "theory_validity_claimed": False,
            "observational_pass": False,
            "dark_sector_elimination_proven": False,
        },
        "data_seals": {
            "observations_opened": False,
            "transaction_event_data_opened": False,
            "dark_matter_or_halo_data_opened": False,
            "redshift_or_cosmology_data_opened": False,
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
    result = build_audit(config_path)
    output = (
        Path(args.output).resolve()
        if args.output
        else config_path.parents[1] / json.loads(config_path.read_text(encoding="utf-8"))["output_path"]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
