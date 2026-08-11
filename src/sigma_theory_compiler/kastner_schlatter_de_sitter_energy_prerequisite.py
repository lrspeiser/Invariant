"""de Sitter charge prerequisites for compiler-authored EH/intensity actions."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

CONFIG_SCHEMA = "sigma-kastner-schlatter-de-sitter-energy-prerequisite-config-1.0"
RESULT_SCHEMA = "sigma-kastner-schlatter-de-sitter-energy-prerequisite-1.0"
PREDECESSOR_BLOCKER = (
    "global_de_Sitter_boundary_charge_and_nonlinear_positive_energy_theorem_not_registered"
)
FIRST_BLOCKER = (
    "candidate_bound_de_Sitter_boundary_conditions_zero_symplectic_flux_and_"
    "integrable_coupled_charge_not_registered"
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
        raise ValueError("de Sitter prerequisite path escapes repository") from error
    return path


def _bound_file(root: Path, binding: Mapping[str, Any], label: str) -> Path:
    if set(binding) - {"path", "file_sha256", "content_sha256"} or not {
        "path",
        "file_sha256",
    }.issubset(binding):
        raise ValueError(f"{label} binding shape changed")
    path = _inside(root, str(binding["path"]))
    if not path.is_file() or _file_sha(path) != binding["file_sha256"]:
        raise ValueError(f"{label} file hash mismatch")
    return path


def _bound_json(root: Path, binding: Mapping[str, Any], label: str) -> dict[str, Any]:
    value = json.loads(_bound_file(root, binding, label).read_text(encoding="utf-8"))
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
        "declared_charge_framework",
        "admission_policy",
        "seals",
    }
    if set(config) != required or config.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("de Sitter prerequisite config shape changed")
    if config.get("declared_charge_framework") != {
        "background": "four-dimensional de Sitter Einstein background with q=q0",
        "charge_formalism": (
            "Einstein-Hilbert covariant phase space / Iyer-Wald surface variation"
        ),
        "reference_normalization": "H_xi[de_Sitter,q0]=0",
        "fixed_background_energy_domain": (
            "static patch 0<=r<L with finite-energy scalar perturbations"
        ),
        "fixed_background_flux_condition": (
            "zero scalar energy flux through every patch boundary component"
        ),
        "nonlinear_boundary_phase_space_registered": False,
    }:
        raise ValueError("de Sitter charge framework changed")
    if config.get("admission_policy") != {
        "compiler_action_attribution_only": True,
        "closed_slice_zero_charge_is_not_positive_energy": True,
        "fixed_background_scalar_energy_is_not_coupled_nonlinear_energy": True,
        "charge_integrability_requires_zero_symplectic_flux": True,
        "paper_qed_ontology_observation_inference_allowed": False,
    }:
        raise ValueError("de Sitter prerequisite admission policy changed")
    if config.get("seals") != {
        "observations_opened": False,
        "dark_matter_or_halo_inputs": False,
        "redshift_or_cosmology_inputs": False,
        "solar_system_inputs": False,
        "QED_actualization_derivation_opened": False,
        "transaction_ontology_validated": False,
        "paid_llm_calls": False,
    }:
        raise ValueError("de Sitter prerequisite seals changed")


def _validate_predecessors(
    formal: Mapping[str, Any], completion: Mapping[str, Any]
) -> list[Mapping[str, Any]]:
    if (
        formal.get("decision_counts") != {"pass": 0, "reject": 0, "blocked": 2}
        or formal.get("first_blocker") != PREDECESSOR_BLOCKER
        or formal.get("formal_counts", {}).get("global_positive_energy_pass") != 0
        or formal.get("formal_counts", {}).get("scalar_Hamiltonian_positive_pass") != 2
        or any(formal.get("claim_seals", {}).values())
        or any(formal.get("data_seals", {}).values())
    ):
        raise ValueError("formal-admission predecessor boundary changed")
    records = formal.get("candidate_records", [])
    if [item.get("branch_id") for item in records] != [
        "eq35_middle_h",
        "eq35_printed_planck",
    ] or any(item.get("paper_or_QED_derived") for item in records):
        raise ValueError("formal-admission predecessor branch boundary changed")
    if (
        completion.get("counts", {}).get("paper_derived_actions") != 0
        or completion.get("counts", {}).get("complete_local_deterministic_action_hypotheses") != 2
        or any(completion.get("claim_seals", {}).values())
    ):
        raise ValueError("completion predecessor attribution boundary changed")
    return records


def _charge_interface() -> dict[str, Any]:
    return {
        "status": "pass_exact_variational_interface_only",
        "background_equations": [
            "bar_R_mn=Lambda_beta*bar_g_mn",
            "bar_R=4*Lambda_beta",
            "bar_q=q0",
            "grad_m(bar_q)=0",
        ],
        "linearized_equation": "E_L_mn=delta(G_mn)+Lambda_beta*h_mn",
        "linearized_bianchi": "bar_nabla^m(E_L_mn)=0",
        "einstein_hilbert_noether_charge_2form": (
            "Q_xi_mn=-(c^3/(16*pi*G))*epsilon_mnrs*bar_nabla^r(xi^s)"
        ),
        "metric_symplectic_potential_3form": (
            "theta_g_mnr=(c^3/(16*pi*G))*epsilon_smnr*"
            "(bar_nabla_l(h^sl)-bar_nabla^s(h))"
        ),
        "scalar_symplectic_potential_3form": (
            "theta_q_mnr=-(B_q/c)*epsilon_smnr*bar_nabla^s(q)*delta(q)"
        ),
        "surface_variation": "delta(H_xi)=Integral_boundary_Sigma[delta(Q_xi)-i_xi(theta_g+theta_q)]",
        "stationary_scalar_linear_surface_term": "theta_q[bar_q=q0]=0",
        "integrability_condition": (
            "Integral_boundary_region omega(delta_1,delta_2)=0 for the declared phase space"
        ),
        "finiteness_condition": "surface variation has a finite boundary limit",
        "reference": "H_xi[de_Sitter,q0]=0",
        "nontrivial_integrable_charge_value_registered": False,
    }


def _static_scalar_energy() -> dict[str, Any]:
    return {
        "status": "pass_exact_fixed_background_conditional_zero_flux",
        "field": "phi=q-q0",
        "equation": "B_q*bar_Box(phi)-A_q*phi=0",
        "excitation_stress": (
            "tau_mn=B_q*grad_m(phi)*grad_n(phi)-bar_g_mn*"
            "[(B_q/2)*(grad(phi))^2+(A_q/2)*phi^2]"
        ),
        "conservation": "bar_nabla^m(tau_mn)=0 on the scalar equation",
        "static_patch": "ds^2=-N^2*c^2*dt^2+h_ij*dx^i*dx^j, N=sqrt(1-r^2/L^2)",
        "energy": (
            "E_phi=(1/2)*Integral_Sigma N*sqrt(h)*[B_q*N^-2*c^-2*(partial_t(phi))^2"
            "+B_q*h^ij*D_i(phi)*D_j(phi)+A_q*phi^2]"
        ),
        "positivity_conditions": "A_q>0, B_q>0, N>0, positive h_ij",
        "finite_energy_required": True,
        "time_derivative": "d(E_phi)/dt=-Flux_phi[boundary_Sigma]",
        "conserved_if": "Flux_phi[boundary_Sigma]=0",
        "nonnegative": True,
        "strict_except_zero_field": True,
        "metric_backreaction_included": False,
        "gravitational_charge_positivity_inferred": False,
    }


def _branch_record(formal_record: Mapping[str, Any]) -> dict[str, Any]:
    branch = str(formal_record["branch_id"])
    if branch == "eq35_middle_h":
        radius = "L^2=3*c^3/(4*pi*G*h_planck*q0)"
        ratio = "1"
    elif branch == "eq35_printed_planck":
        radius = "L^2=3*c^3/(2*pi*G*h_planck*q0)"
        ratio = "2"
    else:
        raise ValueError("unexpected de Sitter branch")
    return {
        "branch_id": branch,
        "beta": formal_record["beta"],
        "compiler_authored_action": True,
        "paper_or_QED_derived": False,
        "de_Sitter_background": {
            "Lambda_beta": formal_record["background"]["cosmological_term"],
            "radius_relation": "Lambda_beta=3/L^2",
            "exact_radius": radius,
            "radius_squared_relative_to_middle_branch": ratio,
            "static_factor": "N^2=1-r^2/L^2",
            "static_domain": "0<=r<L",
            "cosmological_horizon": "r=L",
        },
        "covariant_charge_interface": _charge_interface(),
        "closed_global_slice_control": {
            "slice_topology": "S^3",
            "boundary": "empty",
            "surface_charge_variation": "0",
            "normalized_charge": "0",
            "positive_energy_theorem": False,
            "reason": "a trivial empty-boundary charge does not order nontrivial excitations",
        },
        "static_patch_scalar_energy": _static_scalar_energy(),
        "global_Killing_obstruction": {
            "de_Sitter_has_everywhere_globally_timelike_Killing_field": False,
            "static_patch_Killing_field": "timelike for r<L and null at r=L",
            "global_positive_generator_from_static_Killing_field": False,
        },
        "gate_ledger": {
            "exact_positive_Lambda_background_and_radius": "pass",
            "covariant_surface_charge_variation_interface": "pass",
            "closed_slice_empty_boundary_control": "pass",
            "fixed_background_scalar_static_patch_energy": "pass_conditional_zero_flux",
            "candidate_bound_boundary_phase_space_and_falloff": "blocked",
            "zero_gravitational_symplectic_flux": "blocked",
            "finite_integrable_nontrivial_coupled_charge": "blocked",
            "nonlinear_coupled_positive_energy": "blocked",
        },
        "decision": "blocked",
        "first_blocker": FIRST_BLOCKER,
        "candidate_rejection_authorized": False,
        "full_formal_admission_pass": False,
    }


def _negative_controls() -> dict[str, Any]:
    return {
        "ADM_substitution": {
            "mutation": "apply asymptotically-flat ADM charge to Lambda_beta>0 stationary data",
            "rejected_because": "the stationary reference is de Sitter-type, not asymptotically flat",
            "admitted": False,
        },
        "empty_boundary_positivity": {
            "mutation": "infer nonlinear positive energy from H_xi=0 on closed S^3 slices",
            "rejected_because": "the empty-boundary charge is identically trivial",
            "admitted": False,
        },
        "scalar_to_coupled_inference": {
            "mutation": "infer coupled gravitational positivity from E_phi>=0 on fixed de Sitter",
            "rejected_because": "metric backreaction and gravitational boundary charge are absent",
            "admitted": False,
        },
        "nonzero_flux_conservation": {
            "mutation": "claim static-patch conservation with nonzero horizon flux",
            "rejected_because": "dE_phi/dt=-Flux_phi is then nonzero",
            "admitted": False,
        },
        "all_negative_controls_rejected": True,
    }


def _validate_result(result: Mapping[str, Any]) -> None:
    if result.get("schema_version") != RESULT_SCHEMA:
        raise ValueError("de Sitter prerequisite result schema changed")
    if result.get("decision_counts") != {"pass": 0, "reject": 0, "blocked": 2}:
        raise ValueError("de Sitter prerequisite decision partition changed")
    if result.get("prerequisite_counts") != {
        "candidate_actions": 2,
        "exact_de_Sitter_background_radius_pass": 2,
        "covariant_charge_interface_pass": 2,
        "closed_slice_empty_boundary_control_pass": 2,
        "fixed_background_scalar_positive_energy_pass": 2,
        "nontrivial_integrable_coupled_charge_pass": 0,
        "nonlinear_coupled_positive_energy_pass": 0,
        "paper_or_QED_derived_actions": 0,
        "full_formal_admission_pass": 0,
    }:
        raise ValueError("de Sitter prerequisite counts changed")
    records = result.get("candidate_records", [])
    if len(records) != 2 or any(item.get("paper_or_QED_derived") for item in records):
        raise ValueError("de Sitter prerequisite attribution changed")
    if any(item.get("candidate_rejection_authorized") for item in records):
        raise ValueError("de Sitter prerequisite overreached to candidate rejection")
    if any(item.get("full_formal_admission_pass") for item in records):
        raise ValueError("de Sitter prerequisite overreached to full admission")
    if result.get("first_blocker") != FIRST_BLOCKER:
        raise ValueError("de Sitter prerequisite first blocker changed")
    controls = result.get("negative_controls", {})
    if controls.get("all_negative_controls_rejected") is not True or any(
        value.get("admitted") is not False
        for value in controls.values()
        if isinstance(value, dict)
    ):
        raise ValueError("de Sitter prerequisite negative control changed")
    if any(result.get("claim_seals", {}).values()) or any(result.get("data_seals", {}).values()):
        raise ValueError("de Sitter prerequisite seal opened")
    if result.get("content_sha256") is not None and _content_sha(result) != result["content_sha256"]:
        raise ValueError("de Sitter prerequisite content hash mismatch")


def build_prerequisite(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    root = config_path.parents[1]
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    formal = _bound_json(
        root, config["predecessors"]["formal_admission_artifact"], "formal admission artifact"
    )
    completion = _bound_json(
        root, config["predecessors"]["completion_artifact"], "completion artifact"
    )
    _bound_file(
        root, config["predecessors"]["formal_admission_config"], "formal admission config"
    )
    _bound_file(
        root, config["predecessors"]["formal_admission_source"], "formal admission source"
    )
    formal_records = _validate_predecessors(formal, completion)
    source_path = Path(__file__).resolve()
    test_path = root / "tests/test_kastner_schlatter_de_sitter_energy_prerequisite.py"
    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA,
        "campaign_id": config["campaign_id"],
        "scope": (
            "exact de Sitter charge and fixed-background energy prerequisites for two "
            "compiler-authored actions; no paper, QED, ontology, or observation inference"
        ),
        "source_bindings": {
            **config["predecessors"],
            "primary_pdf_sha256": formal["source_bindings"]["primary_pdf_sha256"],
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
        "declared_charge_framework": config["declared_charge_framework"],
        "candidate_records": [_branch_record(record) for record in formal_records],
        "negative_controls": _negative_controls(),
        "prerequisite_counts": {
            "candidate_actions": 2,
            "exact_de_Sitter_background_radius_pass": 2,
            "covariant_charge_interface_pass": 2,
            "closed_slice_empty_boundary_control_pass": 2,
            "fixed_background_scalar_positive_energy_pass": 2,
            "nontrivial_integrable_coupled_charge_pass": 0,
            "nonlinear_coupled_positive_energy_pass": 0,
            "paper_or_QED_derived_actions": 0,
            "full_formal_admission_pass": 0,
        },
        "decision_counts": {"pass": 0, "reject": 0, "blocked": 2},
        "decision": "charge_interface_and_fixed_scalar_energy_pass_coupled_global_energy_blocked",
        "first_blocker": FIRST_BLOCKER,
        "secondary_blockers": [
            "no_candidate_bound_de_Sitter_falloff_or_horizon_regular_boundary_phase_space",
            "no_zero_gravitational_symplectic_flux_certificate",
            "no_finite_integrable_nontrivial_coupled_charge",
            "no_nonlinear_coupled_positive_energy_or_stability_theorem",
        ],
        "claim_seals": {
            "paper_action_derived": False,
            "QED_actualization_action_derived": False,
            "transaction_ontology_validated": False,
            "nontrivial_coupled_charge_registered": False,
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
    result = build_prerequisite(config_path)
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
