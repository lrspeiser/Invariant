"""Exact dynamic-class audit for compiler-authored Kastner--Schlatter actions."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

CONFIG_SCHEMA = "sigma-kastner-schlatter-action-equivalence-config-1.0"
RESULT_SCHEMA = "sigma-kastner-schlatter-action-equivalence-1.0"
FIRST_BLOCKER = "no_paper_derivation_of_candidate_action_or_transaction_intensity_dynamics"


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _content_sha(value: Mapping[str, Any]) -> str:
    return _sha({key: item for key, item in value.items() if key != "content_sha256"})


def _load_bound(root: Path, binding: Mapping[str, Any], label: str) -> dict[str, Any]:
    if set(binding) != {"path", "file_sha256", "content_sha256"}:
        raise ValueError(f"{label} binding shape changed")
    path = (root / str(binding["path"])).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"{label} path escapes repository") from error
    if not path.is_file() or _file_sha(path) != binding["file_sha256"]:
        raise ValueError(f"{label} file hash mismatch")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be an object")
    if value.get("content_sha256") != binding["content_sha256"]:
        raise ValueError(f"{label} declared content hash mismatch")
    recomputed = _sha(value["canonical"]) if "canonical" in value else _content_sha(value)
    if recomputed != binding["content_sha256"]:
        raise ValueError(f"{label} content hash mismatch")
    return value


def _validate_config(config: Mapping[str, Any]) -> None:
    if (
        set(config)
        != {
            "schema_version",
            "campaign_id",
            "output_path",
            "predecessors",
            "seals",
        }
        or config.get("schema_version") != CONFIG_SCHEMA
    ):
        raise ValueError("equivalence config shape changed")
    if config.get("seals") != {
        "literature_novelty_claim_allowed": False,
        "paper_derivation_claim_allowed": False,
        "theory_equivalence_claim_allowed": False,
        "observational_pass_allowed": False,
        "dark_sector_elimination_claim_allowed": False,
        "observations_opened": False,
        "paid_llm_calls": False,
    }:
        raise ValueError("equivalence seals changed")


def _validate_sources(completion: Mapping[str, Any], scalar: Mapping[str, Any]) -> None:
    if completion.get("counts") != {
        "complete_local_deterministic_action_hypotheses": 2,
        "conditional_exact_eq35_branch_matches": 2,
        "normalization_branches": 2,
        "normalization_branches_selected_as_fact": 0,
        "observational_or_theory_passes": 0,
        "paper_derived_actions": 0,
    }:
        raise ValueError("candidate-action predecessor changed")
    branches = completion.get("completion_hypotheses", [])
    if [item.get("beta") for item in branches] != ["1/2", "1/4"]:
        raise ValueError("candidate-action branches changed")
    terms = scalar.get("canonical", {}).get("terms", [])
    term_ids = [term.get("id") for term in terms]
    if term_ids != ["EH_R", "SCALAR_MASS", "SCALAR_X"]:
        raise ValueError("canonical scalar operator basis changed")
    if scalar.get("canonical", {}).get("source_role") != "known_answer_control":
        raise ValueError("canonical scalar role changed")


def _mapping(beta: str, branch_id: str) -> dict[str, Any]:
    return {
        "branch_id": branch_id,
        "beta": beta,
        "field_redefinition": "varphi=sqrt(B_q)*(q-q0)/Lambda_phi**2",
        "inverse_field_redefinition": "q=q0+Lambda_phi**2*varphi/sqrt(B_q)",
        "parameter_map": {
            "M_Pl**2": "c**4/(8*pi*G)",
            "m_phi**2": "A_q*Lambda_phi**4/B_q",
            "Lambda_phi": "arbitrary positive normalization scale",
        },
        "operator_equalities": {
            "kinetic": "Lambda_phi**4*X_varphi=-(B_q/2)*(grad(q))**2",
            "mass": "-(m_phi**2/2)*varphi**2=-(A_q/2)*(q-q0)**2",
            "einstein_hilbert": "(M_Pl**2/2)*R=c**4*R/(16*pi*G)",
            "euler_residual": (
                "B_q*Box(q)-A_q*(q-q0)="
                "sqrt(B_q)/Lambda_phi**2*[Lambda_phi**4*Box(varphi)-m_phi**2*varphi]"
            ),
        },
        "unmatched_constant": f"-{beta}*c*h*q0",
        "propagating_dynamic_operator_equivalent": True,
        "full_action_equal_to_constant_free_control": False,
        "constant_vacuum_energy_is_the_only_unmatched_bulk_term": True,
        "invertibility_domain": "A_q>0, B_q>0, Lambda_phi>0",
        "paper_or_qed_derivation_inferred": False,
        "literature_novelty_inferred": False,
    }


def _validate_result(result: Mapping[str, Any]) -> None:
    if result.get("schema_version") != RESULT_SCHEMA:
        raise ValueError("equivalence result schema changed")
    mappings = result.get("equivalence_certificates", [])
    if len(mappings) != 2 or [item.get("beta") for item in mappings] != ["1/2", "1/4"]:
        raise ValueError("equivalence certificate partition changed")
    if not all(item.get("propagating_dynamic_operator_equivalent") for item in mappings):
        raise ValueError("dynamic equivalence certificate missing")
    if any(item.get("full_action_equal_to_constant_free_control") for item in mappings):
        raise ValueError("constant vacuum term was erased")
    if result.get("counts") != {
        "candidate_action_branches": 2,
        "canonical_dynamic_class_matches": 2,
        "full_action_equalities_to_constant_free_control": 0,
        "new_propagating_gravity_operator_classes": 0,
        "distinct_vacuum_normalization_branches": 2,
        "paper_or_qed_derived_actions": 0,
        "literature_novelty_claims": 0,
        "observational_or_theory_passes": 0,
    }:
        raise ValueError("equivalence counts changed")
    if result.get("first_blocker") != FIRST_BLOCKER:
        raise ValueError("equivalence blocker changed")
    if any(result.get("claim_seals", {}).values()) or any(result.get("data_seals", {}).values()):
        raise ValueError("equivalence seal opened")
    declared = result.get("content_sha256")
    if declared is not None and _content_sha(result) != declared:
        raise ValueError("equivalence content hash mismatch")


def build_audit(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    root = config_path.parents[1]
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    completion = _load_bound(root, config["predecessors"]["candidate_action"], "candidate action")
    scalar = _load_bound(root, config["predecessors"]["canonical_scalar"], "canonical scalar")
    _validate_sources(completion, scalar)
    source_path = Path(__file__).resolve()
    test_path = root / "tests/test_kastner_schlatter_action_equivalence_audit.py"
    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA,
        "campaign_id": config["campaign_id"],
        "decision": "canonical_dynamic_class_identified_paper_and_physics_claims_blocked",
        "scope": (
            "exact local field-redefinition audit of compiler-authored actions; "
            "not an equivalence proof for the Kastner--Schlatter ontology or paper"
        ),
        "source_bindings": {
            "candidate_action": dict(config["predecessors"]["candidate_action"]),
            "canonical_scalar": dict(config["predecessors"]["canonical_scalar"]),
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
        "canonical_control_contract": {
            "source_role": "known_answer_control",
            "operator_basis": ["EH_R", "SCALAR_MASS", "SCALAR_X"],
            "x_convention": "X_varphi=-(1/2)*g^mn*grad_m(varphi)*grad_n(varphi)",
            "constant_vacuum_term_present_in_control": False,
        },
        "equivalence_certificates": [
            _mapping("1/2", "eq35_middle_h"),
            _mapping("1/4", "eq35_printed_planck"),
        ],
        "branch_comparison": {
            "same_propagating_operator_class": True,
            "same_mass_ratio_A_q_over_B_q": True,
            "same_constant_vacuum_energy": False,
            "vacuum_energy_ratio_beta_half_over_beta_quarter": "2",
            "normalization_selected_as_fact": False,
        },
        "counts": {
            "candidate_action_branches": 2,
            "canonical_dynamic_class_matches": 2,
            "full_action_equalities_to_constant_free_control": 0,
            "new_propagating_gravity_operator_classes": 0,
            "distinct_vacuum_normalization_branches": 2,
            "paper_or_qed_derived_actions": 0,
            "literature_novelty_claims": 0,
            "observational_or_theory_passes": 0,
        },
        "first_blocker": FIRST_BLOCKER,
        "secondary_blockers": [
            "equation_35_h_vs_hbar_factor_normalization_clarification",
            "no_QED_actualization_to_covariant_Poisson_process_derivation",
            "no_transaction_derived_covariant_action_or_intensity_dynamics",
        ],
        "claim_seals": {
            "paper_action_claimed": False,
            "paper_ontology_equivalence_proven": False,
            "transactional_qed_derivation_proven": False,
            "literature_novelty_claimed": False,
            "theory_validity_claimed": False,
            "observational_pass": False,
            "dark_matter_elimination_proven": False,
            "dark_energy_elimination_proven": False,
        },
        "data_seals": {
            "observations_opened": False,
            "dark_matter_or_halo_data_opened": False,
            "redshift_or_cosmology_data_opened": False,
            "solar_system_data_opened": False,
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
        else config_path.parents[1]
        / json.loads(config_path.read_text(encoding="utf-8"))["output_path"]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
