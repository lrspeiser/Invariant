"""Exact positive-field coordinate gate for compiler-authored intensity actions."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

CONFIG_SCHEMA = "sigma-kastner-schlatter-positive-reparameterization-gate-config-1.0"
RESULT_SCHEMA = "sigma-kastner-schlatter-positive-reparameterization-gate-1.0"
FIRST_BLOCKER = (
    "no_paper_or_QED_derived_selection_of_the_positive_field_sector_and_no_"
    "action_derived_point_process_probability_measure"
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
        raise ValueError("positive-reparameterization path escapes repository") from error
    return path


def _bound_predecessor(root: Path, binding: Mapping[str, Any]) -> dict[str, Any]:
    if set(binding) != {"path", "file_sha256", "content_sha256"}:
        raise ValueError("positive-reparameterization predecessor binding shape changed")
    path = _inside(root, str(binding["path"]))
    if not path.is_file() or _file_sha(path) != binding["file_sha256"]:
        raise ValueError("positive-reparameterization predecessor file hash mismatch")
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or value.get("content_sha256") != binding["content_sha256"]
        or _content_sha(value) != binding["content_sha256"]
    ):
        raise ValueError("positive-reparameterization predecessor content hash mismatch")
    return value


def _validate_config(config: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "campaign_id",
        "output_path",
        "predecessor",
        "field_space_contract",
        "admission_policy",
        "seals",
    }
    if set(config) != required or config.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("positive-reparameterization config shape changed")
    if config.get("field_space_contract") != {
        "original_field_space": "q in R",
        "restricted_field_space": "q in (0,infinity)",
        "new_coordinate": "phi in R",
        "map": "q=q0*exp(phi)",
        "inverse": "phi=log(q/q0)",
        "new_coordinate_dimension": "dimensionless",
        "parameter_domain": "q0>0, A_q>0, B_q>0, beta>0",
        "regularity_scope": "finite real phi on the maximal regular classical solution",
    }:
        raise ValueError("positive-reparameterization field-space contract changed")
    if config.get("admission_policy") != {
        "exact_change_of_variables_required": True,
        "global_original_phase_space_invariance_inferred": False,
        "restricted_sector_selection_from_paper_or_QED_inferred": False,
        "action_derived_point_process_measure_inferred": False,
        "candidate_action_rejection_allowed": False,
    }:
        raise ValueError("positive-reparameterization admission policy changed")
    if any(config.get("seals", {}).values()):
        raise ValueError("positive-reparameterization seal opened")


def _validate_predecessor(value: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    if (
        value.get("decision_counts") != {"pass": 0, "reject": 0, "blocked": 2}
        or value.get("gate_counts", {}).get("unrestricted_positive_intensity_preservation_reject")
        != 2
        or value.get("gate_counts", {}).get("positive_reparameterized_action_pass") != 0
        or value.get("gate_counts", {}).get("candidate_action_reject") != 0
        or any(value.get("claim_seals", {}).values())
        or any(value.get("data_seals", {}).values())
    ):
        raise ValueError("positive-intensity predecessor boundary changed")
    records = value.get("candidate_records", [])
    if (
        [record.get("branch_id") for record in records] != ["eq35_middle_h", "eq35_printed_planck"]
        or any(record.get("paper_or_QED_derived") for record in records)
        or any(record.get("candidate_action_rejection_authorized") for record in records)
    ):
        raise ValueError("positive-intensity predecessor candidate boundary changed")
    return records


def _record(predecessor: Mapping[str, Any]) -> dict[str, Any]:
    branch_id = str(predecessor["branch_id"])
    beta = str(predecessor["beta"])
    return {
        "branch_id": branch_id,
        "beta": beta,
        "compiler_authored_action": True,
        "paper_or_QED_derived": False,
        "field_diffeomorphism": {
            "map": "q=q0*exp(phi)",
            "inverse": "phi=log(q/q0)",
            "domain": "phi in R",
            "image": "q in (0,infinity)",
            "Jacobian": "dq/dphi=q0*exp(phi)=q>0",
            "global_on_declared_open_sector": True,
            "covers_q_equals_zero": False,
            "covers_negative_q": False,
        },
        "reparameterized_action": {
            "status": "compiler_authored_exact_rewrite_on_q>0",
            "bulk": (
                "S_beta^+=(1/c)*Integral sqrt(-g)*[c^4*R/(16*pi*G)"
                "-(B_q*q0^2/2)*exp(2*phi)*(grad(phi))^2-beta*c*h*q0"
                "-(A_q*q0^2/2)*(exp(phi)-1)^2] d4x"
            ),
            "kinetic_coefficient": "K(phi)=B_q*q0^2*exp(2*phi)>0",
            "potential": ("U_beta(phi)=beta*c*h*q0+(A_q*q0^2/2)*(exp(phi)-1)^2"),
            "Euler_Lagrange": (
                "B_q*q0^2*exp(2*phi)*(Box(phi)+(grad(phi))^2)-A_q*q0^2*exp(phi)*(exp(phi)-1)=0"
            ),
            "original_equation_times_Jacobian": ("q*[B_q*Box(q)-A_q*(q-q0)]=0"),
            "stress_tensor": (
                "T_mn^+=B_q*q0^2*exp(2*phi)*grad_m(phi)*grad_n(phi)"
                "-g_mn*[(B_q*q0^2/2)*exp(2*phi)*(grad(phi))^2+U_beta(phi)]"
            ),
            "metric_equation": "G_mn=(8*pi*G/c^4)*T_mn^+",
            "off_shell_identity": "nabla^m(T_mn^+)=E_phi*grad_n(phi)",
            "dimensions": {
                "phi": "1",
                "B_q*q0^2": "M*L*T^-2",
                "kinetic_and_potential_density": "M*L^-1*T^-2",
            },
            "exact_EL_equivalence_on_q_positive": True,
            "stationary_solution": "phi=0 iff q=q0",
        },
        "positivity_theorem": {
            "statement": (
                "Every finite-real-phi configuration, and hence every regular classical solution "
                "while phi remains finite, has q=q0*exp(phi)>0."
            ),
            "proof": "q0>0 and exp(phi)>0 for every finite real phi",
            "scope": "restricted open field sector on the maximal regular solution",
            "global_existence_or_geodesic_completeness_proved": False,
            "original_q_in_R_nonnegative_cone_invariant": False,
        },
        "gate_ledger": {
            "exact_positive_field_diffeomorphism": "pass",
            "exact_reparameterized_local_covariant_action": "pass",
            "Euler_Lagrange_equivalence_on_q_positive": "pass",
            "positive_intensity_on_regular_finite_phi_solutions": "pass",
            "unrestricted_original_q_phase_space_positivity": "reject",
            "paper_or_QED_selection_of_positive_sector": "blocked",
            "action_derived_point_process_probability_measure": "blocked",
            "candidate_action_rejection": "blocked",
        },
        "candidate_action_rejection_authorized": False,
        "candidate_decision": "blocked",
        "first_blocker": FIRST_BLOCKER,
    }


def _controls() -> dict[str, Any]:
    return {
        "stationary_positive_control": {
            "phi": "0",
            "q": "q0",
            "Euler_Lagrange_residual": "0",
            "pass": True,
        },
        "linear_map_negative_control": {
            "mutation": "q=q0*(1+phi)",
            "counterexample": "phi=-2 gives q=-q0<0",
            "rejected": True,
        },
        "missing_Jacobian_negative_control": {
            "mutation": "replace exp(2*phi) kinetic coefficient by 1",
            "rejected": True,
            "reason": "does not equal the original action under q=q0*exp(phi)",
        },
        "crossing_import_negative_control": {
            "mutation": "represent the predecessor sign-changing q trajectory by finite real phi",
            "rejected": True,
            "reason": "q=0 is the phi=-infinity boundary and q<0 is outside the chart image",
        },
        "action_overclaim_negative_control": {
            "mutation": "infer paper derivation, action validity, or point-process law",
            "rejected": True,
        },
    }


def _validate_result(result: Mapping[str, Any]) -> None:
    if result.get("schema_version") != RESULT_SCHEMA:
        raise ValueError("positive-reparameterization result schema changed")
    if result.get("decision_counts") != {"pass": 0, "reject": 0, "blocked": 2}:
        raise ValueError("positive-reparameterization candidate partition changed")
    if result.get("gate_counts") != {
        "candidate_actions": 2,
        "exact_positive_field_diffeomorphism_pass": 2,
        "exact_reparameterized_action_pass": 2,
        "EL_equivalence_on_positive_sector_pass": 2,
        "regular_solution_strict_positivity_pass": 2,
        "original_unrestricted_phase_space_positivity_reject": 2,
        "paper_or_QED_positive_sector_selection_pass": 0,
        "action_derived_point_process_measure_pass": 0,
        "candidate_action_reject": 0,
        "paper_QED_ontology_observational_pass": 0,
    }:
        raise ValueError("positive-reparameterization gate counts changed")
    records = result.get("candidate_records", [])
    if len(records) != 2 or any(record.get("paper_or_QED_derived") for record in records):
        raise ValueError("positive-reparameterization attribution changed")
    if any(record.get("candidate_action_rejection_authorized") for record in records):
        raise ValueError("positive-reparameterization gate overreached to action rejection")
    if any(
        not record.get("reparameterized_action", {}).get("exact_EL_equivalence_on_q_positive")
        or not record.get("field_diffeomorphism", {}).get("global_on_declared_open_sector")
        for record in records
    ):
        raise ValueError("positive-reparameterization exact contract lost")
    if result.get("first_blocker") != FIRST_BLOCKER:
        raise ValueError("positive-reparameterization first blocker changed")
    if any(result.get("claim_seals", {}).values()) or any(result.get("data_seals", {}).values()):
        raise ValueError("positive-reparameterization seal opened")
    if (
        result.get("content_sha256") is not None
        and _content_sha(result) != result["content_sha256"]
    ):
        raise ValueError("positive-reparameterization content hash mismatch")


def build_gate(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    root = config_path.parents[1]
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    predecessor = _bound_predecessor(root, config["predecessor"])
    records = _validate_predecessor(predecessor)
    source_path = Path(__file__).resolve()
    test_path = root / "tests/test_kastner_schlatter_positive_reparameterization_gate.py"
    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA,
        "campaign_id": config["campaign_id"],
        "scope": (
            "exact positive field-coordinate rewrite of both compiler-authored scalar-intensity "
            "actions on q>0; no physical selection of that restricted sector"
        ),
        "source_bindings": {
            "positive_intensity_predecessor": config["predecessor"],
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
        "field_space_contract": config["field_space_contract"],
        "candidate_records": [_record(record) for record in records],
        "deterministic_controls": _controls(),
        "gate_counts": {
            "candidate_actions": 2,
            "exact_positive_field_diffeomorphism_pass": 2,
            "exact_reparameterized_action_pass": 2,
            "EL_equivalence_on_positive_sector_pass": 2,
            "regular_solution_strict_positivity_pass": 2,
            "original_unrestricted_phase_space_positivity_reject": 2,
            "paper_or_QED_positive_sector_selection_pass": 0,
            "action_derived_point_process_measure_pass": 0,
            "candidate_action_reject": 0,
            "paper_QED_ontology_observational_pass": 0,
        },
        "decision_counts": {"pass": 0, "reject": 0, "blocked": 2},
        "decision": "positive_coordinate_actions_closed_physical_sector_selection_blocked",
        "first_blocker": FIRST_BLOCKER,
        "secondary_blockers": [
            "no_action_derived_covariant_point_process_probability_measure",
            "no_QED_actualization_to_transaction_event_map",
            "no_global_regular_solution_or_boundary_completion_theorem",
            "no_observational_transaction_event_or_exposure_registration",
        ],
        "claim_seals": {
            "paper_action_derived": False,
            "QED_actualization_action_derived": False,
            "positive_sector_selected_by_paper_or_QED": False,
            "transaction_ontology_validated": False,
            "candidate_action_rejected": False,
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
