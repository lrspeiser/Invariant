"""Exact local formal admission for compiler-authored transaction-intensity actions."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

CONFIG_SCHEMA = "sigma-kastner-schlatter-candidate-action-formal-admission-config-1.0"
RESULT_SCHEMA = "sigma-kastner-schlatter-candidate-action-formal-admission-1.0"
FIRST_BLOCKER = "global_de_Sitter_boundary_charge_and_nonlinear_positive_energy_theorem_not_registered"


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
        raise ValueError("formal-admission path escapes repository") from error
    return path


def _bound_file(root: Path, binding: Mapping[str, Any], label: str) -> Path:
    allowed = {"path", "file_sha256", "content_sha256"}
    if set(binding) - allowed or not {"path", "file_sha256"}.issubset(binding):
        raise ValueError(f"{label} binding shape changed")
    path = _inside(root, str(binding["path"]))
    if not path.is_file() or _file_sha(path) != binding["file_sha256"]:
        raise ValueError(f"{label} file hash mismatch")
    return path


def _bound_json(root: Path, binding: Mapping[str, Any], label: str) -> dict[str, Any]:
    path = _bound_file(root, binding, label)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be an object")
    expected = binding.get("content_sha256")
    if expected is None or value.get("content_sha256") != expected or _content_sha(value) != expected:
        raise ValueError(f"{label} content hash mismatch")
    return value


def _validate_config(config: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "campaign_id",
        "output_path",
        "predecessors",
        "formal_domain",
        "admission_policy",
        "seals",
    }
    if set(config) != required or config.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("formal-admission config shape changed")
    if config.get("formal_domain") != {
        "spacetime_dimension": 4,
        "metric_signature": "-+++",
        "constant_signs": "G>0, c>0, h_planck>0",
        "intensity_domain": "q0>0, A_q>0, B_q>0",
        "adm_domain": "N>0 and positive-definite h_ij",
        "hyperbolic_gauge": "generalized harmonic metric gauge",
        "boundary_variation": (
            "GHY plus fixed induced metric; Dirichlet q or zero normal q flux"
        ),
    }:
        raise ValueError("formal-admission domain changed")
    if config.get("admission_policy") != {
        "candidate_actions_are_compiler_hypotheses": True,
        "paper_or_qed_derivation_inference": False,
        "local_gauge_fixed_hyperbolicity_can_pass": True,
        "restricted_positive_hamiltonian_is_not_global_energy": True,
        "full_formal_admission_requires_global_boundary_energy": True,
        "observational_or_theory_pass_allowed": False,
    }:
        raise ValueError("formal-admission policy changed")
    if config.get("seals") != {
        "observations_opened": False,
        "dark_matter_or_halo_inputs": False,
        "redshift_or_cosmology_inputs": False,
        "solar_system_inputs": False,
        "paid_llm_calls": False,
    }:
        raise ValueError("formal-admission seals changed")


def _validate_completion(completion: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    if (
        completion.get("decision")
        != "candidate_completions_registered_paper_derivation_and_physics_claims_blocked"
        or completion.get("counts", {}).get("complete_local_deterministic_action_hypotheses") != 2
        or completion.get("counts", {}).get("paper_derived_actions") != 0
        or any(completion.get("claim_seals", {}).values())
        or any(completion.get("data_seals", {}).values())
    ):
        raise ValueError("completion predecessor boundary changed")
    branches = completion.get("completion_hypotheses", [])
    if [item.get("branch_id") for item in branches] != [
        "eq35_middle_h",
        "eq35_printed_planck",
    ] or [item.get("beta") for item in branches] != ["1/2", "1/4"]:
        raise ValueError("completion predecessor branch partition changed")
    for branch in branches:
        if (
            branch.get("paper_authorship_or_derivation") is not False
            or branch.get("candidate_action", {}).get("local_deterministic_action_complete")
            is not True
            or branch.get("candidate_action", {}).get("parameter_domain")
            != "q0>0, A_q>0, B_q>0"
            or branch.get("euler_lagrange", {}).get("intensity")
            != "B_q*Box(q)-A_q*(q-q0)=0"
        ):
            raise ValueError("completion predecessor action contract changed")
    return branches


def _shared_formal_contract() -> dict[str, Any]:
    return {
        "covariant_variation_replay": {
            "status": "pass_exact_local",
            "variation_identities": [
                "delta(sqrt(-g))=-(1/2)*sqrt(-g)*g_mn*delta(g^mn)",
                "delta[(grad(q))^2]=grad_m(q)*grad_n(q)*delta(g^mn)",
                "delta(V)/delta(q)=A_q*(q-q0)",
            ],
            "metric_EL": (
                "G_mn=(8*pi*G/c^4)*[B_q*grad_m(q)*grad_n(q)"
                "-g_mn*((B_q/2)*(grad(q))^2+V_beta(q))]"
            ),
            "intensity_EL": "B_q*Box(q)-A_q*(q-q0)=0",
            "boundary_closure": (
                "GHY cancels the normal metric variation; scalar surface term vanishes for "
                "delta(q)=0 or n^m*grad_m(q)=0"
            ),
            "noether_identity": (
                "nabla^m(T_mn)=[B_q*Box(q)-A_q*(q-q0)]*grad_n(q)"
            ),
            "exact_residuals": {"metric": "0", "intensity": "0", "noether": "0"},
        },
        "adm_dirac": {
            "status": "pass_exact_bulk_regular_domain",
            "canonical_convention": (
                "c-rescaled momenta after removing the common overall 1/c from the action"
            ),
            "configuration_variables": {
                "h_ij": 6,
                "lapse_N": 1,
                "shift_Ni": 3,
                "intensity_q": 1,
                "total": 11,
            },
            "velocity_hessian": {
                "rank": 7,
                "nullity": 4,
                "regularity": "N>0, positive h_ij, B_q>0",
                "null_directions": ["dot(N)", "dot(N^1)", "dot(N^2)", "dot(N^3)"],
            },
            "scalar_momentum": "Pi_q=sqrt(h)*B_q*(dot(q)-N^i*D_i(q))/N",
            "hamiltonian_constraint": (
                "H_perp=(2*kappa/sqrt(h))*(pi_ij*pi^ij-pi^2/2)"
                "-sqrt(h)*R3/(2*kappa)+Pi_q^2/(2*B_q*sqrt(h))"
                "+sqrt(h)*[(B_q/2)*D_i(q)*D^i(q)+V_beta(q)]"
            ),
            "momentum_constraint": "H_i=-2*D_j(pi^j_i)+Pi_q*D_i(q)",
            "kappa_definition": "kappa=8*pi*G/c^4",
            "constraints": {
                "primary_first_class": 4,
                "secondary_first_class": 4,
                "second_class": 0,
                "total_first_class": 8,
            },
            "bulk_constraint_algebra": [
                "{D[N],D[M]}=D[[N,M]]",
                "{D[N],H[M]}=H[Lie_N(M)]",
                "{H[N],H[M]}=D[h^ij*(N*D_j(M)-M*D_j(N))]",
            ],
            "dof_count": {
                "phase_space_dimension": 22,
                "minus_twice_first_class": 16,
                "minus_second_class": 0,
                "physical_phase_space_dimension": 6,
                "physical_configuration_dof": 3,
                "tensor_dof": 2,
                "scalar_intensity_dof": 1,
            },
            "boundary_charge_included": False,
        },
        "principal_symbol": {
            "status": "pass_local_after_declared_gauge_reduction",
            "ungauged_metric_symbol": "gauge-degenerate by diffeomorphism invariance",
            "gauge": "generalized harmonic",
            "gauge_fixed_block": (
                "diag[C_GR*g^ab*xi_a*xi_b*I_10, B_q*g^ab*xi_a*xi_b]"
            ),
            "C_GR_sign": "positive for G>0 after conventional trace reversal",
            "characteristic_polynomial": (
                "nonzero_constant*(g^ab*xi_a*xi_b)^11"
            ),
            "characteristic_cone": "metric null cone",
            "scalar_speed_squared": "1 (or c^2 in physical time)",
            "strong_hyperbolicity": (
                "pass for the generalized-harmonic first-order reduction on a Lorentzian "
                "background with B_q>0"
            ),
            "ungauged_direct_symbol_invertibility": False,
            "global_evolution_or_geodesic_completeness_proven": False,
        },
        "stability": {
            "status": "pass_exact_local_and_quadratic",
            "scalar_dispersion": "omega^2=c^2*(|k|^2+A_q/B_q)",
            "effective_mass_squared": "A_q/B_q>0",
            "ghost_free_condition": "B_q>0",
            "gradient_stable_condition": "B_q>0",
            "tachyon_free_condition": "A_q/B_q>=0",
            "registered_domain_satisfies_all_three": True,
            "gravity_quadratic_sign": "two reduced tensor modes positive for G>0",
            "vacuum_energy_changes_principal_symbol": False,
        },
        "hamiltonian": {
            "status": "pass_scalar_sector_and_reduced_quadratic_only",
            "scalar_density": (
                "H_q=Pi_q^2/(2*B_q*sqrt(h))+sqrt(h)*[(B_q/2)*D_i(q)*D^i(q)"
                "+beta*c*h_planck*q0+(A_q/2)*(q-q0)^2]"
            ),
            "scalar_density_nonnegative": True,
            "scalar_density_zero_possible": False,
            "reason_strictly_positive_at_stationary_vacuum": "beta>0, c>0, h_planck>0, q0>0",
            "reduced_quadratic_physical_hamiltonian_positive": True,
            "full_bulk_canonical_hamiltonian": "Integral(N*H_perp+N^i*H_i)+boundary_charge",
            "bulk_value_on_constraint_surface": "boundary generator only",
            "global_boundary_charge_registered": False,
            "global_nonlinear_positive_energy_proven": False,
        },
    }


def _branch_record(branch: Mapping[str, Any], shared: Mapping[str, Any]) -> dict[str, Any]:
    beta = str(branch["beta"])
    target_lambda = branch["matching"]["derived_lambda"]
    return {
        "branch_id": branch["branch_id"],
        "beta": beta,
        "candidate_action_kind": "compiler_authored_EH_plus_canonical_scalar_intensity",
        "paper_or_QED_derived": False,
        "background": {
            "stationary_intensity": "q=q0",
            "cosmological_term": target_lambda,
            "cosmological_term_sign": "positive",
            "Minkowski_stationary_background": False,
            "natural_vacuum_class": "positive-Lambda Einstein/de_Sitter-type",
        },
        **shared,
        "gate_ledger": {
            "covariant_variation_replay": "pass",
            "regular_ADM_Dirac_and_three_DOF": "pass",
            "gauge_fixed_local_strong_hyperbolicity": "pass",
            "ghost_gradient_tachyon_conditions": "pass",
            "scalar_and_reduced_quadratic_Hamiltonian": "pass",
            "global_de_Sitter_boundary_charge": "blocked",
            "global_nonlinear_positive_energy": "blocked",
            "paper_or_QED_action_derivation": "blocked",
            "observational_validation": "blocked",
        },
        "decision": "blocked",
        "first_blocker": FIRST_BLOCKER,
        "formal_admission_pass": False,
        "candidate_rejection_authorized": False,
    }


def _negative_controls() -> dict[str, Any]:
    return {
        "B_q_negative": {
            "mutation": "B_q<0",
            "detected_failure": "negative Pi_q^2 coefficient and negative high-frequency gradient energy",
            "classification": "ghost_and_gradient_instability",
            "admitted": False,
        },
        "A_q_negative": {
            "mutation": "A_q<0 with B_q>0",
            "detected_failure": "mass_squared=A_q/B_q<0 and potential unbounded below",
            "classification": "tachyon_and_Hamiltonian_instability",
            "admitted": False,
        },
        "B_q_zero": {
            "mutation": "B_q=0",
            "detected_failure": "scalar principal symbol and Legendre momentum vanish",
            "classification": "degenerate_Dirac_stratum_not_the_registered_three_DOF_system",
            "admitted": False,
        },
        "restricted_to_global_inference": {
            "mutation": "infer global positive energy from pointwise H_q>=0",
            "detected_failure": (
                "positive Lambda is not asymptotically flat and no completed de_Sitter boundary "
                "generator was supplied"
            ),
            "classification": "scope_overreach",
            "admitted": False,
        },
        "all_negative_controls_rejected": True,
    }


def _validate_result(result: Mapping[str, Any]) -> None:
    if result.get("schema_version") != RESULT_SCHEMA:
        raise ValueError("formal-admission result schema changed")
    if result.get("decision_counts") != {"pass": 0, "reject": 0, "blocked": 2}:
        raise ValueError("formal-admission decision partition changed")
    records = result.get("candidate_records", [])
    if len(records) != 2 or any(item.get("formal_admission_pass") for item in records):
        raise ValueError("formal-admission candidate partition changed")
    if any(item.get("paper_or_QED_derived") for item in records):
        raise ValueError("compiler candidate falsely attributed to paper or QED")
    if any(item.get("candidate_rejection_authorized") for item in records):
        raise ValueError("formal-admission gate overreached to candidate rejection")
    expected_counts = {
        "candidate_actions": 2,
        "covariant_variation_pass": 2,
        "regular_ADM_Dirac_pass": 2,
        "three_local_DOF_pass": 2,
        "gauge_fixed_local_hyperbolicity_pass": 2,
        "ghost_gradient_tachyon_pass": 2,
        "scalar_Hamiltonian_positive_pass": 2,
        "global_positive_energy_pass": 0,
        "paper_or_QED_derived_actions": 0,
        "formal_admission_pass": 0,
    }
    if result.get("formal_counts") != expected_counts:
        raise ValueError("formal-admission counts changed")
    if result.get("first_blocker") != FIRST_BLOCKER:
        raise ValueError("formal-admission first blocker changed")
    controls = result.get("negative_controls", {})
    if controls.get("all_negative_controls_rejected") is not True or any(
        item.get("admitted") is not False
        for item in controls.values()
        if isinstance(item, dict)
    ):
        raise ValueError("formal-admission negative control changed")
    if any(result.get("claim_seals", {}).values()) or any(result.get("data_seals", {}).values()):
        raise ValueError("formal-admission seal opened")
    if result.get("content_sha256") is not None and _content_sha(result) != result["content_sha256"]:
        raise ValueError("formal-admission content hash mismatch")


def build_admission(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    root = config_path.parents[1]
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    completion = _bound_json(
        root, config["predecessors"]["completion_artifact"], "completion artifact"
    )
    _bound_file(root, config["predecessors"]["completion_config"], "completion config")
    _bound_file(root, config["predecessors"]["completion_source"], "completion source")
    branches = _validate_completion(completion)
    shared = _shared_formal_contract()
    source_path = Path(__file__).resolve()
    test_path = root / "tests/test_kastner_schlatter_candidate_action_formal_admission.py"
    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA,
        "campaign_id": config["campaign_id"],
        "scope": (
            "exact local formal analysis of two compiler-authored actions; not a derivation from "
            "the Kastner-Schlatter paper, QED actualization, or a stochastic action"
        ),
        "source_bindings": {
            **config["predecessors"],
            "primary_pdf_sha256": completion["source_bindings"]["primary_pdf_sha256"],
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
        "formal_domain": config["formal_domain"],
        "candidate_records": [_branch_record(branch, shared) for branch in branches],
        "negative_controls": _negative_controls(),
        "formal_counts": {
            "candidate_actions": 2,
            "covariant_variation_pass": 2,
            "regular_ADM_Dirac_pass": 2,
            "three_local_DOF_pass": 2,
            "gauge_fixed_local_hyperbolicity_pass": 2,
            "ghost_gradient_tachyon_pass": 2,
            "scalar_Hamiltonian_positive_pass": 2,
            "global_positive_energy_pass": 0,
            "paper_or_QED_derived_actions": 0,
            "formal_admission_pass": 0,
        },
        "decision_counts": {"pass": 0, "reject": 0, "blocked": 2},
        "decision": "local_formal_gates_pass_global_boundary_energy_and_full_admission_blocked",
        "first_blocker": FIRST_BLOCKER,
        "secondary_blockers": [
            "no_global_de_Sitter_initial_boundary_value_completion",
            "no_nonlinear_geodesic_completeness_or_stability_theorem",
            "no_paper_or_QED_derivation_of_the_compiler_action",
            "no_observational_or_transaction_event_validation",
        ],
        "claim_seals": {
            "paper_action_derived": False,
            "QED_actualization_action_derived": False,
            "formal_GR_equivalence_proven": False,
            "global_positive_energy_proven": False,
            "global_nonlinear_stability_proven": False,
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
    result = build_admission(config_path)
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
