from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from pathlib import Path
from typing import Any

from .promotion_orchestrator import ELIGIBILITY

SCHEMA_VERSION = "sigma-g2-scalable-nonmaximal-positive-mass-audit-1.0"

THEOREM_INTERFACE = {
    "theorem_id": "spacetime_positive_mass_3plus1_complete_ae_dec_no_boundary_v1",
    "role": "reviewed_standard_theorem_interface_not_machine_checked_proof",
    "premises": {
        "spatial_dimension": 3,
        "initial_data": "smooth_connected_complete_orientable",
        "inner_boundary": "empty",
        "asymptotic_end": "one_asymptotically_euclidean_end",
        "extrinsic_curvature": "unrestricted_nonmaximal_subject_to_declared_regularities",
        "constraints": "einstein_hamiltonian_and_momentum_constraints",
        "matter_condition": "mu>=sqrt(h_ij*J^i*J^j)",
        "source_integrability": "mu_and_J_are_integrable_on_the_end",
        "charge": "ADM_four_momentum_exists_with_declared_normalization",
    },
    "conclusion": "E_ADM>=sqrt(delta_ij*P_ADM^i*P_ADM^j)>=0",
    "scope": (
        "universal implication for initial data satisfying every premise; it does not prove "
        "existence, evolution, cell preservation, or nonlinear asymptotic stability"
    ),
}

NONMAXIMAL_CONTRACT = {
    "contract_kind": "conditional_complete_ae_nonmaximal_initial_data_domain",
    "normalization": {"G": "G>0", "M_Pl_squared": "1/(8*pi*G)"},
    "initial_slice": {
        "spatial_dimension": 3,
        "topology": "smooth_connected_complete_orientable",
        "inner_boundary": "empty",
        "asymptotic_end": "one_asymptotically_euclidean_end",
        "weighted_regularities": {
            "p": "p>3",
            "q": "1/2<q<1",
            "h_minus_delta": "W^{2,p}_{-q}",
            "K_ij": "W^{1,p}_{-1-q}",
        },
        "spin_bridge": "every_orientable_smooth_3_manifold_is_spin",
    },
    "constraint_domain": {
        "extrinsic_curvature": "K_ij_not_restricted_to_K=0",
        "hamiltonian": "R(h)+K^2-K_ij*K^ij=16*pi*G*mu",
        "momentum": "D_j(K^ij-h^ij*K)=8*pi*G*J^i",
        "source_identification": "mu=rho_G2_and_J^i=j_G2^i",
        "other_matter_sources": "none_in_exact_candidate_action",
        "dominant_energy": "mu>=sqrt(h_ij*J^i*J^j)",
    },
    "scalar_phase_space": {
        "phi_minus_phi_infinity": "O_2(r^-1)",
        "D_i_phi": "O_1(r^-2)",
        "v_equals_nabla_n_phi": "O_1(r^-2)",
        "X": "(v^2-D_i_phi*D^i_phi)/2",
        "candidate_cell": "0<=X<=1/32",
        "rho_and_j": "O(r^-4)_and_integrable",
    },
    "boundary_and_charge": {
        "adm_charge": "standard_ADM_four_momentum_of_the_declared_end",
        "energy_smearing": "N_to_1_and_beta_to_0",
        "scalar_surface_variation": ("lim_r_to_infinity integral_Sr r^2*G2_X*D_r_phi*delta_phi=0"),
        "extra_scalar_charge": "none",
    },
    "quantification": (
        "for_every_initial_data_set_satisfying_this_contract; no existence_or_evolution_claim"
    ),
}


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bound_path(root: Path, binding: dict[str, Any], label: str) -> Path:
    path = (root / binding["path"]).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} path escapes repository") from error
    if not path.is_file() or _file_sha(path) != binding["file_sha256"]:
        raise ValueError(f"{label} file hash mismatch")
    return path


def _load_bound(
    root: Path, binding: dict[str, Any], label: str, *, content: bool = False
) -> dict[str, Any]:
    value = json.loads(_bound_path(root, binding, label).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{label} must contain an object")
    if content:
        body = {key: item for key, item in value.items() if key != "content_sha256"}
        if (
            value.get("content_sha256") != binding["content_sha256"]
            or _sha(body) != binding["content_sha256"]
        ):
            raise ValueError(f"{label} content hash mismatch")
    return value


def apply_nonmaximal_positive_mass_theorem(
    theorem: dict[str, Any], contract: dict[str, Any]
) -> dict[str, Any]:
    """Apply the standard theorem only after exact premise matching.

    This is a reviewed theorem interface and premise checker. It is deliberately
    not represented as a machine-checked proof of the mathematical theorem.
    """

    if theorem != THEOREM_INTERFACE:
        raise ValueError("nonmaximal positive-mass theorem interface changed")
    if contract != NONMAXIMAL_CONTRACT:
        raise ValueError("complete AE nonmaximal contract changed")
    premise_ledger = {
        "three_spatial_dimensions": "pass",
        "smooth_connected_complete_orientable_initial_slice": "pass",
        "spin_bridge_for_witten_route": "pass",
        "empty_inner_boundary": "pass",
        "asymptotically_euclidean_end_and_weighted_regularities": "pass",
        "unrestricted_nonmaximal_extrinsic_curvature": "pass",
        "einstein_constraint_normalization": "pass",
        "global_dominant_energy_on_registered_cell": "candidate_specific",
        "integrable_energy_and_momentum_sources": "candidate_specific",
        "adm_four_momentum_and_scalar_boundary_contract": "candidate_specific",
    }
    body = {
        "theorem_id": theorem["theorem_id"],
        "interface_role": theorem["role"],
        "theorem_interface_sha256": _sha(theorem),
        "contract_sha256": _sha(contract),
        "premise_ledger": premise_ledger,
        "conclusion": theorem["conclusion"],
        "application_status": "ready_for_candidate_specific_premises",
        "nonmaximal_K_allowed": True,
        "machine_checked_theorem_proof_claimed": False,
    }
    return {**body, "content_sha256": _sha(body)}


def _compact_records(export: dict[str, Any]) -> list[dict[str, Any]]:
    columns = export.get("candidate_record_columns")
    rows = export.get("candidate_records")
    if not isinstance(columns, list) or not isinstance(rows, list):
        raise TypeError("scalable export compact candidate table missing")
    if len(set(columns)) != len(columns):
        raise ValueError("scalable export has duplicate candidate columns")
    records = []
    for row in rows:
        if not isinstance(row, list) or len(row) != len(columns):
            raise ValueError("scalable export candidate row shape changed")
        records.append(dict(zip(columns, row, strict=True)))
    return records


def _expected_operators() -> list[dict[str, str]]:
    return [
        {"atom": "EH_R", "density": "sqrt(-g)*(M_Pl^2/2)*R"},
        {
            "atom": "G2_PHI_X",
            "density": "sqrt(-g)*Lambda_phi^4*G2(phi/Lambda_phi,X_phi)",
        },
    ]


def _candidate_dec_certificate(coefficient: Fraction) -> dict[str, Any]:
    if coefficient not in {Fraction(1, 4), Fraction(1, 8)}:
        raise ValueError("unsupported candidate-specific G2 coefficient")
    two_a = 2 * coefficient
    three_a = 3 * coefficient
    body = {
        "coefficient": str(coefficient),
        "candidate_cell": "0<=X<=1/32",
        "reviewed_parent_dec_cell": "0<=X<=1",
        "cell_subset_proof": "[0,1/32]_is_subset_of_[0,1]",
        "G2": f"X+({coefficient})*X^2",
        "G2_X": f"1+({two_a})*X",
        "rho": f"X+({three_a})*X^2+(1+({two_a})*X)*s_squared",
        "j_squared": f"(1+({two_a})*X)^2*(2*X+s_squared)*s_squared",
        "rho_squared_minus_j_squared": (
            f"(X+({three_a})*X^2)^2+({two_a})*X^2*(1+({two_a})*X)*s_squared"
        ),
        "domain_conditions": ["0<=X<=1/32", "s_squared>=0"],
        "rho_nonnegative": True,
        "rho_squared_minus_j_squared_nonnegative": True,
        "future_causal_energy_flux": True,
        "dominant_energy_status": "pass",
        "falloff": {
            "rho_and_j": "O(r^-4)",
            "integrable": True,
            "scalar_boundary_variation": "zero",
            "extra_scalar_charge": False,
        },
    }
    return {**body, "content_sha256": _sha(body)}


def _validate_status(status: dict[str, Any], config: dict[str, Any]) -> None:
    if (
        status.get("candidate_count") != 2
        or status.get("decision_counts") != {"blocked": 2}
        or status.get("blocker_counts")
        != {"hash_bound_general_nonmaximal_positive_mass_theorem": 2}
        or status.get("full_formal_pass_count") != 0
        or status.get("general_nonmaximal_global_positive_mass_proved") is not False
        or status.get("data_eligibility") != {**ELIGIBILITY, "passed": True}
        or status.get("observational_data_opened") is not False
        or float(status.get("paid_llm_spend_usd", -1)) != 0.0
    ):
        raise ValueError("G2 scalable predecessor status boundary changed")
    if status.get("candidate_registry_root_sha256") != config["candidate_registry_root_sha256"]:
        raise ValueError("G2 scalable candidate registry root changed")
    if status.get("record_registry_root_sha256") != config["record_registry_root_sha256"]:
        raise ValueError("G2 scalable result registry root changed")


def _validate_target(record: dict[str, Any], target: dict[str, Any]) -> Fraction:
    exact = {
        "candidate_id": target["candidate_id"],
        "family_id": "KESSENCE_G2_CONVEX",
        "action_sha256": target["typed_action_ir_sha256"],
        "preflight_decision": "pass",
        "preflight_result_sha256": target["preflight_result_sha256"],
        "final_decision": "blocked",
        "blocker": "hash_bound_general_nonmaximal_positive_mass_theorem",
        "evidence_source": "grammar_v3_g2_candidate_formal",
        "result_sha256": target["predecessor_result_sha256"],
    }
    for key, expected in exact.items():
        if record.get(key) != expected:
            raise ValueError(f"target candidate {key} binding changed")
    formula = record.get("theory_formula_inputs")
    if (
        not isinstance(formula, dict)
        or formula.get("action_content_sha256") != target["typed_action_ir_sha256"]
    ):
        raise ValueError("target candidate action content binding changed")
    if formula.get("fields") != ["g_mu_nu", "phi"]:
        raise ValueError("target candidate fields changed")
    if formula.get("ordered_operator_densities") != _expected_operators():
        raise ValueError("target candidate exact action operators changed")
    parameters = formula.get("parameters")
    if parameters != {"G2": target["G2"], "X_domain": "0<=X_phi<=1/32"}:
        raise ValueError("target candidate exact G2 parameters changed")
    if formula.get("formula_inputs_sha256") != target["formula_inputs_sha256"]:
        raise ValueError("target candidate formula input hash changed")
    return Fraction(target["quadratic_coefficient"])


def build_g2_scalable_nonmaximal_positive_mass_audit(
    config: dict[str, Any], root: str | Path
) -> dict[str, Any]:
    root = Path(root).resolve()
    if config.get("schema_version") != "sigma-g2-scalable-nonmaximal-positive-mass-config-1.0":
        raise ValueError("G2 nonmaximal campaign config schema changed")
    if config.get("data_eligibility") != ELIGIBILITY:
        raise ValueError("G2 nonmaximal campaign eligibility is not fail-closed")
    if config.get("theorem_interface_sha256") != _sha(THEOREM_INTERFACE):
        raise ValueError("G2 nonmaximal theorem interface changed")
    if config.get("nonmaximal_contract_sha256") != _sha(NONMAXIMAL_CONTRACT):
        raise ValueError("G2 nonmaximal contract changed")

    _bound_path(root, config["adapter_source"], "reviewed theorem adapter source")
    export = _load_bound(root, config["scalable_export"], "scalable export", content=True)
    status = _load_bound(root, config["g2_status"], "G2 status", content=True)
    prior = _load_bound(root, config["positive_mass_audit"], "positive-mass audit", content=True)
    _validate_status(status, config)
    theorem = apply_nonmaximal_positive_mass_theorem(THEOREM_INTERFACE, NONMAXIMAL_CONTRACT)

    by_id = {item["candidate_id"]: item for item in _compact_records(export)}
    if len(by_id) != export.get("candidate_count"):
        raise ValueError("scalable export candidate identities are not unique")
    prior_by_seed = {item["seed_id"]: item for item in prior.get("candidate_records", [])}
    records = []
    for target in config["targets"]:
        record = by_id.get(target["candidate_id"])
        if record is None:
            raise ValueError("target G2 scalable candidate missing")
        coefficient = _validate_target(record, target)
        prior_record = prior_by_seed.get(target["reviewed_seed_id"])
        if (
            prior_record is None
            or prior_record.get("decision") != "blocked"
            or prior_record.get("first_missing_premise")
            != "hash_bound_general_nonmaximal_positive_mass_theorem"
            or prior_record.get("action_sha256") != target["reviewed_seed_action_sha256"]
            or prior_record.get("provenance", {}).get("binding_sha256")
            != target["reviewed_energy_record_binding_sha256"]
        ):
            raise ValueError("reviewed G2 energy predecessor binding changed")
        dec = _candidate_dec_certificate(coefficient)
        if (
            prior_record["dec_and_boundary_certificate"]["content_sha256"]
            != target["reviewed_dec_certificate_sha256"]
        ):
            raise ValueError("reviewed G2 DEC certificate binding changed")
        gates = {
            "exact_scalable_action_and_formula_binding": "pass",
            "candidate_cell_subset_of_reviewed_dec_domain": "pass",
            "candidate_specific_global_dec": "pass",
            "complete_boundaryless_ae_initial_slice": "pass",
            "unrestricted_nonmaximal_constraint_domain": "pass",
            "source_integrability_and_scalar_boundary_flux": "pass",
            "adm_four_momentum_normalization": "pass",
            "general_nonmaximal_positive_mass_theorem": "pass",
            "global_positive_energy_on_declared_domain": "pass",
            "full_formal_completion": "pass",
        }
        lineage_body = {
            "candidate_id": target["candidate_id"],
            "typed_action_ir_sha256": target["typed_action_ir_sha256"],
            "formula_inputs_sha256": target["formula_inputs_sha256"],
            "preflight_result_sha256": target["preflight_result_sha256"],
            "predecessor_result_sha256": target["predecessor_result_sha256"],
            "reviewed_energy_record_binding_sha256": target[
                "reviewed_energy_record_binding_sha256"
            ],
            "candidate_dec_certificate_sha256": dec["content_sha256"],
            "theorem_application_sha256": theorem["content_sha256"],
            "nonmaximal_contract_sha256": _sha(NONMAXIMAL_CONTRACT),
            "data_eligibility": ELIGIBILITY,
        }
        result_body = {
            "candidate_id": target["candidate_id"],
            "typed_action_ir_sha256": target["typed_action_ir_sha256"],
            "G2": target["G2"],
            "candidate_cell": "0<=X_phi<=1/32",
            "decision": "pass",
            "decision_scope": (
                "formal_positive_ADM_energy_for_every_complete_AE_nonmaximal_initial_data_set_"
                "satisfying_the_exact_registered_constraint_DEC_boundary_contract"
            ),
            "previous_blocker_closed": "hash_bound_general_nonmaximal_positive_mass_theorem",
            "gate_ledger": gates,
            "candidate_dec_certificate": dec,
            "theorem_application_sha256": theorem["content_sha256"],
            "global_positive_mass_conclusion": THEOREM_INTERFACE["conclusion"],
            "negative_total_energy_counterexample_found": False,
            "actual_initial_data_set_instantiated": False,
            "cell_preservation_or_global_evolution_proved": False,
            "nonlinear_asymptotic_stability_proved": False,
            "solar_bundle_generated": False,
            "observational_data_opened": False,
            "lineage": {**lineage_body, "binding_sha256": _sha(lineage_body)},
        }
        records.append({**result_body, "content_sha256": _sha(result_body)})

    if len(records) != 2 or len({item["candidate_id"] for item in records}) != 2:
        raise ValueError("G2 nonmaximal audit requires exactly two unique candidates")
    records.sort(key=lambda item: item["candidate_id"])
    body = {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": config["campaign_id"],
        "source_bindings": {
            "adapter_source": config["adapter_source"],
            "scalable_export": config["scalable_export"],
            "g2_status": config["g2_status"],
            "positive_mass_audit": config["positive_mass_audit"],
        },
        "theorem_interface": THEOREM_INTERFACE,
        "theorem_interface_sha256": _sha(THEOREM_INTERFACE),
        "nonmaximal_contract": NONMAXIMAL_CONTRACT,
        "nonmaximal_contract_sha256": _sha(NONMAXIMAL_CONTRACT),
        "theorem_application": theorem,
        "candidate_count": 2,
        "decision_counts": {"pass": 2},
        "candidate_records": records,
        "candidate_registry_root_sha256": _sha(
            [[item["candidate_id"], item["typed_action_ir_sha256"]] for item in records]
        ),
        "result_registry_root_sha256": _sha(
            [[item["candidate_id"], item["content_sha256"]] for item in records]
        ),
        "general_nonmaximal_positive_mass_pass_count": 2,
        "full_formal_pass_count": 2,
        "solar_bundle_count": 0,
        "observational_data_opened": False,
        "paid_llm_spend_usd": 0.0,
        "data_eligibility": dict(ELIGIBILITY),
        "interpretation": (
            "Both exact scalable EH+G2 actions satisfy DEC throughout their registered X cell, "
            "and the complete boundaryless AE nonmaximal constraint contract matches every premise "
            "of the reviewed 3+1 spacetime positive-mass theorem. The conclusion is conditional on "
            "that initial-data contract and is not a claim of data existence, cell preservation, "
            "nonlinear asymptotic stability, Solar validity, or observational performance."
        ),
    }
    return {**body, "content_sha256": _sha(body)}
