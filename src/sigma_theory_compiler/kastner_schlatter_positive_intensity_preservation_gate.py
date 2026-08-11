"""Exact positivity-preservation obstruction for compiler-authored intensity fields."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

CONFIG_SCHEMA = "sigma-kastner-schlatter-positive-intensity-preservation-gate-config-1.0"
RESULT_SCHEMA = "sigma-kastner-schlatter-positive-intensity-preservation-gate-1.0"
FIRST_BLOCKER = (
    "no_candidate_bound_positive_intensity_reparameterization_or_proven_invariant_"
    "nonnegative_initial_data_cone"
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
        raise ValueError("positive-intensity path escapes repository") from error
    return path


def _bound_json(root: Path, binding: Mapping[str, Any], label: str) -> dict[str, Any]:
    if set(binding) != {"path", "file_sha256", "content_sha256"}:
        raise ValueError(f"{label} binding shape changed")
    path = _inside(root, str(binding["path"]))
    if not path.is_file() or _file_sha(path) != binding["file_sha256"]:
        raise ValueError(f"{label} file hash mismatch")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be an object")
    expected = binding["content_sha256"]
    if value.get("content_sha256") != expected or _content_sha(value) != expected:
        raise ValueError(f"{label} content hash mismatch")
    return value


def _validate_config(config: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "campaign_id",
        "output_path",
        "predecessors",
        "witness_domain",
        "admission_policy",
        "seals",
    }
    if set(config) != required or config.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("positive-intensity config shape changed")
    if config.get("witness_domain") != {
        "background": (
            "fully coupled spatially flat FLRW Einstein-scalar data with compact T3 spatial quotient"
        ),
        "metric": "ds^2=-dTau^2+a(Tau)^2*delta_ij*dx^i*dx^j",
        "time_coordinate": "Tau=x0=c*t",
        "homogeneous_mode": "q=q(Tau), H=a'/a",
        "initial_intensity": "q(0)=q0>0",
        "initial_branch": "contracting Friedmann branch H(0)<0",
        "parameter_domain": "q0>0, A_q>0, B_q>0, G>0, beta>0",
        "finite_volume": "finite positive comoving T3 volume",
    }:
        raise ValueError("positive-intensity witness domain changed")
    if config.get("admission_policy") != {
        "unrestricted_phase_space_positivity_must_hold_for_every_nonnegative_initial_q": True,
        "crossing_rejects_intensity_preservation_not_the_candidate_action": True,
        "stationary_interface_remains_conditional": True,
        "invent_positive_reparameterization": False,
        "paper_qed_ontology_observation_inference_allowed": False,
    }:
        raise ValueError("positive-intensity admission policy changed")
    if config.get("seals") != {
        "observations_opened": False,
        "dark_matter_or_halo_inputs": False,
        "redshift_or_cosmology_inputs": False,
        "solar_system_inputs": False,
        "QED_actualization_derivation_opened": False,
        "transaction_ontology_validated": False,
        "paper_action_attribution_allowed": False,
        "paid_llm_calls": False,
    }:
        raise ValueError("positive-intensity seals changed")


def _validate_predecessors(
    poisson: Mapping[str, Any],
    de_sitter: Mapping[str, Any],
    completion: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    if (
        poisson.get("decision")
        != "stationary_conditional_poisson_interface_closed_dynamic_derivation_blocked"
        or poisson.get("counts", {}).get("positive_intensity_preservation_theorems") != 0
        or poisson.get("counts", {}).get("stationary_homogeneous_poisson_matches") != 2
        or poisson.get("counts", {}).get("action_derived_point_process_measures") != 0
        or any(poisson.get("claim_seals", {}).values())
        or any(poisson.get("data_seals", {}).values())
    ):
        raise ValueError("Poisson/action predecessor boundary changed")
    if (
        de_sitter.get("decision_counts") != {"pass": 0, "reject": 0, "blocked": 2}
        or de_sitter.get("prerequisite_counts", {}).get(
            "exact_de_Sitter_background_radius_pass"
        )
        != 2
        or any(de_sitter.get("claim_seals", {}).values())
        or any(de_sitter.get("data_seals", {}).values())
    ):
        raise ValueError("de Sitter predecessor boundary changed")
    branches = completion.get("completion_hypotheses", [])
    if (
        [item.get("branch_id") for item in branches]
        != ["eq35_middle_h", "eq35_printed_planck"]
        or [item.get("beta") for item in branches] != ["1/2", "1/4"]
        or any(item.get("paper_authorship_or_derivation") for item in branches)
        or any(completion.get("claim_seals", {}).values())
    ):
        raise ValueError("candidate-action predecessor boundary changed")
    return branches


def _coupled_crossing_contract(vacuum_energy: str) -> dict[str, Any]:
    return {
        "potential": f"V(q)={vacuum_energy}+(A_q/2)*(q-q0)^2",
        "mass_squared": "m^2=A_q/B_q>0",
        "coupled_equations": {
            "Friedmann": "3*H^2=kappa*[(B_q/2)*(q')^2+V(q)]",
            "Raychaudhuri": "H'=-(kappa*B_q/2)*(q')^2",
            "scalar": "q''+3*H*q'+m^2*(q-q0)=0",
            "kappa": "kappa=8*pi*G/c^4",
        },
        "initial_data": {
            "q": "q(0)=q0>0",
            "velocity": "q'(0)=-v<0",
            "scale_factor": "a(0)=1",
            "contracting_constraint_root": (
                "H(0)=-sqrt[(kappa/3)*((B_q/2)*v^2+V(q0))]"
            ),
            "Hamiltonian_constraint_residual": "0",
            "momentum_constraint_residual": "0",
        },
        "comparison_definitions": {
            "y": "y=-q'>0",
            "C": "C=3*sqrt(kappa*B_q/6)>0",
            "velocity_threshold": "v^2>m^2*q0/C",
        },
        "comparison_on_zero_to_q0_interval": [
            "H'<=0 and H(0)<0 imply H<0",
            "Friedmann implies abs(H)>=sqrt(kappa*B_q/6)*y",
            "y'=3*abs(H)*y-m^2*(q0-q)",
            "y'>=C*y^2-m^2*q0",
            "v^2>m^2*q0/C implies y'>0 and y>=v until q=0",
        ],
        "crossing_time_bound": "0<Tau_cross<=q0/v",
        "crossing_exists": True,
        "full_metric_backreaction_included": True,
        "Einstein_constraints_satisfied": True,
        "finite_energy_on_compact_slice": True,
    }


def _branch_record(branch: Mapping[str, Any]) -> dict[str, Any]:
    if branch["branch_id"] == "eq35_middle_h":
        vacuum_energy = "(1/2)*c*h_planck*q0"
        stationary_hubble = "H_vac^2=4*pi*G*h_planck*q0/(3*c^3)"
    elif branch["branch_id"] == "eq35_printed_planck":
        vacuum_energy = "(1/4)*c*h_planck*q0"
        stationary_hubble = "H_vac^2=2*pi*G*h_planck*q0/(3*c^3)"
    else:
        raise ValueError("unexpected positive-intensity branch")
    return {
        "branch_id": branch["branch_id"],
        "beta": branch["beta"],
        "compiler_authored_action": True,
        "paper_or_QED_derived": False,
        "candidate_background_binding": {
            "stationary_Hubble_relation": stationary_hubble,
            "vacuum_energy": vacuum_energy,
            "Lambda_relation": "H_vac^2=Lambda_beta/3",
            "intensity_equation": "B_q*Box(q)-A_q*(q-q0)=0",
        },
        "exact_crossing_witness": _coupled_crossing_contract(vacuum_energy),
        "gate_ledger": {
            "candidate_bound_homogeneous_Einstein_scalar_reduction": "pass",
            "Friedmann_and_momentum_constraint_satisfaction": "pass",
            "finite_energy_coupled_positive_to_negative_crossing_witness": "pass",
            "unrestricted_nonnegative_intensity_phase_space_invariant": "reject",
            "stationary_q_equals_q0_conditional_Poisson_interface": "pass",
            "restricted_invariant_nonnegative_initial_data_cone": "blocked",
            "positive_field_reparameterized_candidate_action": "blocked",
            "action_derived_point_process_measure": "blocked",
        },
        "unrestricted_positive_intensity_preservation": False,
        "stationary_conditional_interface_preserved": True,
        "candidate_action_rejection_authorized": False,
        "decision": "blocked",
        "first_blocker": FIRST_BLOCKER,
    }


def _controls() -> dict[str, Any]:
    return {
        "stationary_positive_control": {
            "data": "q(0)=q0, q'(0)=0, H^2=kappa*V(q0)/3",
            "solution": "q(Tau)=q0>0",
            "crossing": False,
        },
        "constraint_omission_negative_control": {
            "mutation": "set H(0)^2=kappa*V(q0)/3 while retaining q'(0)=-v!=0",
            "rejected": True,
            "Hamiltonian_constraint_residual": "-(kappa*B_q/2)*v^2!=0",
        },
        "expanding_branch_comparison_negative_control": {
            "mutation": "reuse H<0 comparison inequalities on H(0)>0 data",
            "rejected": True,
            "reason": "the friction term changes sign in the y comparison",
        },
        "action_rejection_negative_control": {
            "mutation": "infer candidate-action inconsistency from q crossing zero",
            "rejected": True,
            "reason": "the canonical scalar action is regular; only its unrestricted intensity interpretation fails",
        },
    }


def _validate_result(result: Mapping[str, Any]) -> None:
    if result.get("schema_version") != RESULT_SCHEMA:
        raise ValueError("positive-intensity result schema changed")
    if result.get("decision_counts") != {"pass": 0, "reject": 0, "blocked": 2}:
        raise ValueError("positive-intensity candidate partition changed")
    if result.get("gate_counts") != {
        "candidate_actions": 2,
        "exact_crossing_witnesses": 2,
        "fully_coupled_constraint_satisfying_witnesses": 2,
        "unrestricted_positive_intensity_preservation_pass": 0,
        "unrestricted_positive_intensity_preservation_reject": 2,
        "stationary_conditional_Poisson_interface_pass": 2,
        "restricted_invariant_nonnegative_cone_pass": 0,
        "positive_reparameterized_action_pass": 0,
        "action_derived_point_process_measure_pass": 0,
        "candidate_action_reject": 0,
        "paper_QED_ontology_observational_pass": 0,
    }:
        raise ValueError("positive-intensity gate counts changed")
    records = result.get("candidate_records", [])
    if len(records) != 2 or any(item.get("paper_or_QED_derived") for item in records):
        raise ValueError("positive-intensity attribution changed")
    if any(item.get("candidate_action_rejection_authorized") for item in records):
        raise ValueError("positive-intensity gate overreached to action rejection")
    if any(item.get("unrestricted_positive_intensity_preservation") for item in records):
        raise ValueError("positive-intensity crossing obstruction lost")
    if result.get("first_blocker") != FIRST_BLOCKER:
        raise ValueError("positive-intensity first blocker changed")
    controls = result.get("deterministic_controls", {})
    if (
        controls.get("stationary_positive_control", {}).get("crossing") is not False
        or controls.get("constraint_omission_negative_control", {}).get("rejected") is not True
        or controls.get("expanding_branch_comparison_negative_control", {}).get("rejected")
        is not True
        or controls.get("action_rejection_negative_control", {}).get("rejected") is not True
    ):
        raise ValueError("positive-intensity deterministic control changed")
    if any(result.get("claim_seals", {}).values()) or any(result.get("data_seals", {}).values()):
        raise ValueError("positive-intensity seal opened")
    if result.get("content_sha256") is not None and _content_sha(result) != result["content_sha256"]:
        raise ValueError("positive-intensity content hash mismatch")


def build_gate(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    root = config_path.parents[1]
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    poisson = _bound_json(
        root, config["predecessors"]["poisson_action_compatibility"], "Poisson compatibility"
    )
    de_sitter = _bound_json(
        root, config["predecessors"]["de_sitter_energy_prerequisite"], "de Sitter prerequisite"
    )
    completion = _bound_json(
        root, config["predecessors"]["candidate_action_completion"], "candidate completion"
    )
    branches = _validate_predecessors(poisson, de_sitter, completion)
    source_path = Path(__file__).resolve()
    test_path = root / "tests/test_kastner_schlatter_positive_intensity_preservation_gate.py"
    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA,
        "campaign_id": config["campaign_id"],
        "scope": (
            "candidate-bound exact obstruction to interpreting the unrestricted canonical scalar "
            "phase space as a nonnegative point-process intensity"
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
        "witness_domain": config["witness_domain"],
        "candidate_records": [_branch_record(branch) for branch in branches],
        "deterministic_controls": _controls(),
        "gate_counts": {
            "candidate_actions": 2,
            "exact_crossing_witnesses": 2,
            "fully_coupled_constraint_satisfying_witnesses": 2,
            "unrestricted_positive_intensity_preservation_pass": 0,
            "unrestricted_positive_intensity_preservation_reject": 2,
            "stationary_conditional_Poisson_interface_pass": 2,
            "restricted_invariant_nonnegative_cone_pass": 0,
            "positive_reparameterized_action_pass": 0,
            "action_derived_point_process_measure_pass": 0,
            "candidate_action_reject": 0,
            "paper_QED_ontology_observational_pass": 0,
        },
        "decision_counts": {"pass": 0, "reject": 0, "blocked": 2},
        "decision": "unrestricted_intensity_positivity_rejected_actions_and_stationary_interfaces_blocked",
        "first_blocker": FIRST_BLOCKER,
        "secondary_blockers": [
            "no_action_derived_covariant_point_process_probability_measure",
            "no_QED_actualization_to_transaction_event_map",
            "no_candidate_bound_positive_field_reparameterization",
            "no_observational_transaction_event_or_exposure_registration",
        ],
        "claim_seals": {
            "paper_action_derived": False,
            "QED_actualization_action_derived": False,
            "transaction_ontology_validated": False,
            "candidate_action_rejected": False,
            "positive_reparameterized_action_registered": False,
            "action_derived_point_process_measure_registered": False,
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
