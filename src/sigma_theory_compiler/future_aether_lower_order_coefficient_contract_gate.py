"""Lower-order coefficient contract gate for the sole elliptic Aether seed."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Any

from .future_aether_finite_tilt_york_symbol_gate import YORK_SHELL_BLOCKER
from .future_aether_nonlinear_lift_characteristic_gate import CHARACTERISTIC_BLOCKER
from .future_aether_principal_inverse_fredholm_gate import (
    BLOCKER as SOURCE_ELLIPTIC_BLOCKER,
)
from .future_aether_principal_inverse_fredholm_gate import (
    build_future_aether_principal_inverse_fredholm_gate,
)
from .promotion_orchestrator import ELIGIBILITY

CONFIG_SCHEMA = "sigma-future-aether-lower-order-coefficient-contract-gate-config-1.0"
RESULT_SCHEMA = "sigma-future-aether-lower-order-coefficient-contract-gate-1.0"
TARGET_FAMILY = "AETHER_K1234_PARAMETER_CELL"
BLOCKER = (
    "candidate_bound_full_canonical_seed_point_including_pi_and_p_A_and_"
    "distributed_H_D_coefficient_DAG"
)


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path.name} must contain an object")
    return value


def _bound_path(root: Path, binding: dict[str, Any], label: str) -> Path:
    if set(binding) - {"path", "file_sha256", "content_sha256"}:
        raise ValueError(f"{label} binding fields are invalid")
    path = (root / binding["path"]).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} path escapes repository") from error
    if not path.is_file() or _file_sha(path) != binding["file_sha256"]:
        raise ValueError(f"{label} file hash mismatch")
    return path


def _bound_json(
    root: Path, binding: dict[str, Any], label: str, *, content: bool = False
) -> dict[str, Any]:
    value = _load(_bound_path(root, binding, label))
    if content:
        body = {key: item for key, item in value.items() if key != "content_sha256"}
        if (
            value.get("content_sha256") != binding.get("content_sha256")
            or _sha(body) != binding["content_sha256"]
        ):
            raise ValueError(f"{label} content hash mismatch")
    return value


def _validate_config(config: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "campaign_id",
        "campaign_implementation",
        "source_principal_inverse_artifact",
        "source_principal_inverse_config",
        "source_principal_inverse_implementation",
        "source_compact_seed_artifact",
        "coefficient_contract",
        "budget",
        "observational_authorization",
        "data_eligibility",
        "external_paid_llm_calls",
    }
    if set(config) != required or config.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("future Aether lower-order contract config is invalid")
    if config.get("data_eligibility") != ELIGIBILITY:
        raise ValueError("future Aether lower-order contract eligibility is open")
    if config.get("observational_authorization") is not False:
        raise ValueError("future Aether lower-order contract opened observations")
    if config.get("external_paid_llm_calls") is not False:
        raise ValueError("future Aether lower-order contract enabled paid LLM calls")
    if config.get("coefficient_contract") != {
        "operator_form": "L u=A^ij(x)*partial_i partial_j u+B^i(x)*partial_i u+C(x)*u",
        "domain": "H^2_-1/2(R3;scalar+vector)",
        "codomain": "L^2_-5/2(R3;scalar+vector)",
        "matrix_dimension": 4,
        "required_background_jet_order": 3,
    }:
        raise ValueError("future Aether lower-order coefficient contract is not exact")
    if config.get("budget") != {
        "maximum_candidates": 14,
        "maximum_coefficient_contract_candidates": 1,
        "maximum_seed_jet_order": 3,
        "maximum_paid_llm_spend_usd": 0.0,
    }:
        raise ValueError("future Aether lower-order contract budget is not exact")


def exact_compact_profile_jet_control(amplitude_squared: Fraction) -> dict[str, Any]:
    """Bound the compact radial Aether profile through its registered C3 jet."""

    if amplitude_squared <= 0:
        raise ValueError("compact profile amplitude must be positive")
    gradient_squared = Fraction(2985984, 823543) * amplitude_squared
    hessian_component_squared = 3136 * amplitude_squared
    third_component_squared = 112896 * amplitude_squared
    grad_y_squared = 4 * amplitude_squared * gradient_squared
    body = {
        "profile": "A_i=a*(1-r^2)^4_+*delta_i1",
        "amplitude_squared": str(amplitude_squared),
        "support": "closed_unit_ball",
        "regularity": "C3_compact_support",
        "sup_A_squared": str(amplitude_squared),
        "sup_gradient_A_squared": str(gradient_squared),
        "gradient_maximizer_radius_squared": "1/7",
        "gradient_formula": "|grad A|=8*a*r*(1-r^2)^3",
        "sup_component_Hessian_A_squared_upper": str(hessian_component_squared),
        "Hessian_component_bound": "8*a+48*a=56*a",
        "sup_component_third_derivative_A_squared_upper": str(third_component_squared),
        "third_derivative_component_bound": "48*a+48*a+48*a+192*a=336*a",
        "sup_gradient_tilt_squared": str(grad_y_squared),
        "unit_branch_chi_range": f"1<=chi<=sqrt(1+({amplitude_squared}))",
        "weighted_local_bounds_available": True,
    }
    return {**body, "content_sha256": _sha(body)}


def exact_profile_regularity_negative_control() -> dict[str, Any]:
    """Show that lowering the cutoff exponent loses the declared C3 interface."""

    body = {
        "mutated_profile": "A_i=a*(1-r^2)^3_+*delta_i1",
        "mutated_regularity": "C2_compact_support",
        "required_regularity": "C3_compact_support",
        "coefficient_contract_admissible": False,
        "control_status": "reject_mutated_profile_not_candidate",
    }
    return {**body, "content_sha256": _sha(body)}


def _candidate_certificate(source: dict[str, Any], seed_artifact: dict[str, Any]) -> dict[str, Any]:
    prior = source["principal_inverse_fredholm_certificate"]
    if prior is None or prior.get("uniform_principal_symbol_inverse_bound_proven") is not True:
        raise ValueError("lower-order contract candidate lost principal inverse")
    symbolic = seed_artifact["symbolic_finite_amplitude_control"]
    if symbolic["compact_seed"] != {
        "inside_unit_ball": "A_i=10*(1-r^2)^4*delta_i1",
        "outside_unit_ball": "A_i=0",
        "regularity": "C^3_compact_support",
        "maximum_tilt_squared": "100",
        "asymptotic_Aether": "unit_normal_outside_compact_support",
    }:
        raise ValueError("compact seed symbolic profile changed")
    amplitude = Fraction(prior["principal_inverse_control"]["registered_tilt_upper"])
    jets = exact_compact_profile_jet_control(amplitude)
    negative = exact_profile_regularity_negative_control()
    body = {
        "candidate_id": source["candidate_id"],
        "typed_action_ir_sha256": source["typed_action_ir_sha256"],
        "source_principal_inverse_record_sha256": source["content_sha256"],
        "source_compact_seed_symbolic_control_sha256": symbolic["content_sha256"],
        "compact_profile_weighted_jet_control": jets,
        "profile_regularity_negative_control": negative,
        "declared_lower_order_coefficient_interface": {
            "operator_form": ("L u=A^ij(x)*partial_i partial_j u+B^i(x)*partial_i u+C(x)*u"),
            "unknown_order": ["phi", "X1", "X2", "X3"],
            "equation_order": ["H", "D1", "D2", "D3"],
            "A_principal_inverse_bound_sha256": prior["principal_inverse_control"][
                "content_sha256"
            ],
            "required_B_tensor_shape": [3, 4, 4],
            "required_C_tensor_shape": [4, 4],
            "required_exact_or_interval_outputs": [
                "compact_support_or_AE_decay",
                "weighted_multiplier_bounds",
                "relative_bound_against_principal_inverse",
            ],
        },
        "canonical_background_point_audit": {
            "q_ij": "delta_ij_only_in_frozen_source_ansatz",
            "A_i": "registered_compact_profile",
            "pi^ij": "not_registered_after_finite_tilt_Legendre_transform",
            "p_A^i": "not_registered_after_finite_tilt_Legendre_transform",
            "constraint_satisfying_background": False,
            "complete": False,
        },
        "distributed_constraint_DAG_audit": {
            "H_core_density": "not_registered",
            "D_i_cotangent_lift_density": "generic_structure_only_not_candidate_expansion",
            "linearized_B_order_one_coefficients": "not_registered",
            "linearized_C_order_zero_coefficients": "not_registered",
            "complete": False,
        },
        "compact_profile_C3_weighted_jet_bounds_proven": True,
        "lower_order_coefficient_contract_declared": True,
        "full_canonical_background_point_registered": False,
        "distributed_lower_order_coefficient_registry_complete": False,
        "weighted_relative_lower_order_bound_proven": False,
        "weighted_Fredholm_isomorphism_proven": False,
        "full_operator_inverse_norm_proven": False,
        "nonlinear_remainder_majorant_proven": False,
        "completed_boundary_sign_persistence_proven": False,
        "candidate_rejection_authorized": False,
    }
    return {**body, "content_sha256": _sha(body)}


def build_future_aether_lower_order_coefficient_contract_gate(
    config: dict[str, Any], root: str | Path
) -> dict[str, Any]:
    root = Path(root).resolve()
    _validate_config(config)
    _bound_path(root, config["campaign_implementation"], "lower-order contract implementation")
    source_implementation = _bound_path(
        root,
        config["source_principal_inverse_implementation"],
        "principal inverse implementation",
    )
    source_config = _bound_json(
        root, config["source_principal_inverse_config"], "principal inverse config"
    )
    source_artifact = _bound_json(
        root,
        config["source_principal_inverse_artifact"],
        "principal inverse artifact",
        content=True,
    )
    seed_artifact = _bound_json(
        root, config["source_compact_seed_artifact"], "compact seed artifact", content=True
    )
    callback_source = Path(
        inspect.getsourcefile(build_future_aether_principal_inverse_fredholm_gate) or ""
    ).resolve()
    if callback_source != source_implementation:
        raise ValueError("principal inverse implementation entrypoint changed")
    if build_future_aether_principal_inverse_fredholm_gate(source_config, root) != source_artifact:
        raise ValueError("principal inverse artifact no longer replays")
    if (
        source_artifact.get("candidate_count") != 14
        or source_artifact.get("decision_counts") != {"blocked": 14}
        or source_artifact.get("first_blocker_counts")
        != {
            CHARACTERISTIC_BLOCKER: 11,
            YORK_SHELL_BLOCKER: 2,
            SOURCE_ELLIPTIC_BLOCKER: 1,
        }
        or source_artifact.get("uniform_principal_symbol_inverse_bound_pass_count") != 1
        or source_artifact.get("candidate_rejection_authorized_count") != 0
        or seed_artifact.get("candidate_count") != 14
    ):
        raise ValueError("lower-order contract source scope changed")

    records = []
    blockers: Counter[str] = Counter()
    contract_count = 0
    for source in source_artifact["candidate_records"]:
        source_body = {key: value for key, value in source.items() if key != "content_sha256"}
        if source.get("content_sha256") != _sha(source_body):
            raise ValueError("lower-order contract source record changed")
        if source["first_blocker"] == SOURCE_ELLIPTIC_BLOCKER:
            certificate = _candidate_certificate(source, seed_artifact)
            blocker = BLOCKER
            status = "pass"
            contract_count += 1
        else:
            certificate = None
            blocker = source["first_blocker"]
            status = "not_reached"
        blockers[blocker] += 1
        provenance_body = {
            "candidate_id": source["candidate_id"],
            "typed_action_ir_sha256": source["typed_action_ir_sha256"],
            "source_principal_inverse_record_sha256": source["content_sha256"],
            "lower_order_contract_certificate_sha256": (
                certificate["content_sha256"] if certificate is not None else None
            ),
            "data_eligibility": ELIGIBILITY,
        }
        body = {
            "ordinal": source["ordinal"],
            "candidate_id": source["candidate_id"],
            "family_id": TARGET_FAMILY,
            "parameter_cell_id": source["parameter_cell_id"],
            "parameter_cell_lineage_sha256": source["parameter_cell_lineage_sha256"],
            "typed_action_ir_sha256": source["typed_action_ir_sha256"],
            "action_density_equivalence_sha256": source["action_density_equivalence_sha256"],
            "compilation_receipt_sha256": source["compilation_receipt_sha256"],
            "source_principal_inverse_record_sha256": source["content_sha256"],
            "parameters": source["parameters"],
            "exact_specialization": source["exact_specialization"],
            "lower_order_coefficient_contract_certificate": certificate,
            "gate_ledger": {
                "source_action_and_predecessor_binding": {"status": "pass"},
                "compact_profile_weighted_C3_jet_bounds": {"status": status},
                "lower_order_coefficient_interface": {"status": status},
                "full_canonical_background_point": {"status": "blocked"},
                "distributed_H_D_coefficient_DAG": {"status": "blocked"},
                "weighted_Fredholm_isomorphism_and_full_inverse_norm": {"status": "blocked"},
                "nonlinear_remainder": {"status": "blocked"},
                "completed_boundary_sign_persistence": {"status": "blocked"},
                "observational_data_seal": {"status": "pass"},
            },
            "decision": "blocked",
            "first_blocker": blocker,
            "formal_pass": False,
            "candidate_rejection_authorized": False,
            "constraint_satisfying_negative_total_energy_datum_proven": False,
            "full_formal_completion_claimed": False,
            "automatic_downstream_enqueue_performed": False,
            "solar_bundle_generated": False,
            "observational_data_opened": False,
            "data_eligibility": dict(ELIGIBILITY),
            "provenance": {**provenance_body, "binding_sha256": _sha(provenance_body)},
        }
        records.append({**body, "content_sha256": _sha(body)})
    expected_blockers = {
        CHARACTERISTIC_BLOCKER: 11,
        YORK_SHELL_BLOCKER: 2,
        BLOCKER: 1,
    }
    if contract_count != 1 or dict(blockers) != expected_blockers:
        raise ValueError("future Aether lower-order contract partition changed")
    record_root = _sha(
        [
            [item["candidate_id"], item["typed_action_ir_sha256"], item["content_sha256"]]
            for item in records
        ]
    )
    provenance_body = {
        "source_principal_inverse_content_sha256": source_artifact["content_sha256"],
        "source_principal_inverse_record_registry_root_sha256": source_artifact[
            "candidate_record_registry_root_sha256"
        ],
        "source_compact_seed_content_sha256": seed_artifact["content_sha256"],
        "candidate_record_registry_root_sha256": record_root,
        "data_eligibility": ELIGIBILITY,
    }
    body = {
        "schema_version": RESULT_SCHEMA,
        "campaign_id": config["campaign_id"],
        "immutable_config_sha256": _sha(config),
        "source_principal_inverse_binding": config["source_principal_inverse_artifact"],
        "source_compact_seed_binding": config["source_compact_seed_artifact"],
        "candidate_count": 14,
        "decision_counts": {"blocked": 14},
        "first_blocker_counts": dict(sorted(blockers.items())),
        "formal_pass_count": 0,
        "candidate_rejection_authorized_count": 0,
        "constraint_satisfying_negative_total_energy_datum_count": 0,
        "forced_characteristic_candidate_count": 11,
        "York_symbol_shell_candidate_count": 2,
        "uniformly_elliptic_candidate_count": 1,
        "compact_profile_C3_weighted_jet_bound_pass_count": 1,
        "lower_order_coefficient_contract_declared_count": 1,
        "full_canonical_background_point_registered_count": 0,
        "distributed_lower_order_coefficient_registry_complete_count": 0,
        "weighted_relative_lower_order_bound_pass_count": 0,
        "weighted_Fredholm_isomorphism_pass_count": 0,
        "full_operator_inverse_norm_pass_count": 0,
        "nonlinear_remainder_bound_pass_count": 0,
        "completed_boundary_sign_persistence_count": 0,
        "candidate_record_registry_root_sha256": record_root,
        "candidate_records": records,
        "bounded_lower_order_coefficient_contract_gate_completed": True,
        "full_candidate_specific_formal_completion_claimed": False,
        "automatic_downstream_enqueue_performed": False,
        "solar_bundle_count": 0,
        "observational_data_opened": False,
        "dark_matter_or_halo_inputs": False,
        "redshift_distance_inputs": False,
        "paid_llm_spend_usd": 0.0,
        "data_eligibility": dict(ELIGIBILITY),
        "provenance": {**provenance_body, "binding_sha256": _sha(provenance_body)},
        "interpretation": (
            "The sole elliptic candidate now has exact compact weighted Aether-profile jet bounds "
            "through order three and a typed 4-by-4 lower-order coefficient interface. Current "
            "evidence does not register the finite-tilt canonical pi and p_A background profiles "
            "or the distributed Hamiltonian/momentum coefficient DAG, so the B and C tensors "
            "cannot be derived or bounded. No weighted Fredholm inverse, nonlinear remainder, "
            "boundary-sign persistence, constraint solution, or theory rejection is claimed."
        ),
    }
    return {**body, "content_sha256": _sha(body)}


def publish_future_aether_lower_order_coefficient_contract_gate(
    config: dict[str, Any], root: str | Path, target: str | Path
) -> dict[str, Any]:
    artifact = build_future_aether_lower_order_coefficient_contract_gate(config, root)
    target = Path(target).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if _load(target) != artifact:
            raise ValueError("refusing to replace divergent lower-order contract artifact")
        return artifact
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("xb") as handle:
        handle.write((_canonical(artifact) + "\n").encode())
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    artifact = publish_future_aether_lower_order_coefficient_contract_gate(
        _load(Path(arguments.config)), arguments.root, arguments.output
    )
    print(_canonical({key: value for key, value in artifact.items() if key != "candidate_records"}))


if __name__ == "__main__":
    main()
