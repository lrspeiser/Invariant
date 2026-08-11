"""Typed weighted-IFT contract gate for the three regular future Aether seeds."""

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

from .future_aether_nonlinear_lift_characteristic_gate import CHARACTERISTIC_BLOCKER
from .future_aether_regular_adm_inverse_margin_gate import (
    BLOCKER as SOURCE_REGULAR_BLOCKER,
)
from .future_aether_regular_adm_inverse_margin_gate import (
    build_future_aether_regular_adm_inverse_margin_gate,
)
from .future_aether_weak_field_ae_constraint_gate import (
    build_future_aether_weak_field_ae_constraint_gate,
)
from .promotion_orchestrator import ELIGIBILITY

CONFIG_SCHEMA = "sigma-future-aether-weighted-ift-contract-gate-config-1.0"
RESULT_SCHEMA = "sigma-future-aether-weighted-ift-contract-gate-1.0"
TARGET_FAMILY = "AETHER_K1234_PARAMETER_CELL"
BLOCKER = (
    "candidate_bound_gauge_fixed_weighted_constraint_operator_norm_contract_with_"
    "nonlinear_remainder_and_completed_boundary_majorants"
)
REQUIRED_CONTRACT_FIELDS = (
    "weight_delta",
    "domain_space",
    "codomain_space",
    "gauge_fixing",
    "full_linearized_constraint_map",
    "reference_inverse_norm",
    "operator_perturbation_norm",
    "seed_nonlinear_constraint_residual_norm",
    "nonlinear_second_derivative_majorant",
    "completed_boundary_first_derivative_bound",
    "completed_boundary_second_derivative_bound",
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
        "source_inverse_margin_artifact",
        "source_inverse_margin_config",
        "source_inverse_margin_implementation",
        "source_weak_field_artifact",
        "source_weak_field_config",
        "source_weak_field_implementation",
        "budget",
        "observational_authorization",
        "data_eligibility",
        "external_paid_llm_calls",
    }
    if set(config) != required or config.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("future Aether weighted IFT config is invalid")
    if config.get("data_eligibility") != ELIGIBILITY:
        raise ValueError("future Aether weighted IFT eligibility is open")
    if config.get("observational_authorization") is not False:
        raise ValueError("future Aether weighted IFT opened observations")
    if config.get("external_paid_llm_calls") is not False:
        raise ValueError("future Aether weighted IFT enabled paid LLM calls")
    if config.get("budget") != {
        "maximum_candidates": 14,
        "maximum_regular_adm_candidates": 3,
        "maximum_contract_fields": 11,
        "maximum_paid_llm_spend_usd": 0.0,
    }:
        raise ValueError("future Aether weighted IFT budget is not exact")


def _fraction(contract: dict[str, str], key: str) -> Fraction:
    value = Fraction(contract[key])
    if value < 0:
        raise ValueError(f"{key} must be nonnegative")
    return value


def evaluate_quantitative_ift_contract(contract: dict[str, str]) -> dict[str, Any]:
    """Evaluate exact Neumann, Newton-Kantorovich, and boundary-sign inequalities."""

    required = {
        "reference_inverse_norm",
        "operator_perturbation_norm",
        "seed_nonlinear_constraint_residual_norm",
        "nonlinear_second_derivative_majorant",
        "completed_boundary_first_derivative_bound",
        "completed_boundary_second_derivative_bound",
        "negative_boundary_energy_margin",
    }
    if set(contract) != required:
        raise ValueError("quantitative IFT contract fields are invalid")
    c0 = _fraction(contract, "reference_inverse_norm")
    perturbation = _fraction(contract, "operator_perturbation_norm")
    residual = _fraction(contract, "seed_nonlinear_constraint_residual_norm")
    second = _fraction(contract, "nonlinear_second_derivative_majorant")
    boundary_first = _fraction(contract, "completed_boundary_first_derivative_bound")
    boundary_second = _fraction(contract, "completed_boundary_second_derivative_bound")
    energy_margin = _fraction(contract, "negative_boundary_energy_margin")
    if c0 == 0 or energy_margin == 0:
        raise ValueError("inverse norm and energy margin must be positive")
    neumann_product = c0 * perturbation
    neumann_pass = neumann_product < 1
    if not neumann_pass:
        body = {
            "neumann_product": str(neumann_product),
            "neumann_inverse_pass": False,
            "full_inverse_bound": None,
            "kantorovich_product": None,
            "kantorovich_pass": False,
            "solution_radius_bound": None,
            "boundary_correction_bound": None,
            "boundary_sign_persistence_pass": False,
            "all_conditions_pass": False,
        }
        return {**body, "content_sha256": _sha(body)}
    inverse = c0 / (1 - neumann_product)
    kantorovich_product = 2 * inverse * inverse * second * residual
    kantorovich_pass = kantorovich_product <= 1
    radius = 2 * inverse * residual
    boundary_correction = boundary_first * radius + boundary_second * radius * radius
    boundary_pass = kantorovich_pass and boundary_correction < energy_margin
    body = {
        "neumann_product": str(neumann_product),
        "neumann_inverse_pass": True,
        "full_inverse_bound": str(inverse),
        "kantorovich_product": str(kantorovich_product),
        "kantorovich_pass": kantorovich_pass,
        "solution_radius_bound": str(radius),
        "boundary_correction_bound": str(boundary_correction),
        "boundary_sign_persistence_pass": boundary_pass,
        "all_conditions_pass": kantorovich_pass and boundary_pass,
    }
    return {**body, "content_sha256": _sha(body)}


def _contract_certificate(prior: dict[str, Any], weak: dict[str, Any]) -> dict[str, Any]:
    inverse = prior["regular_ADM_inverse_margin_certificate"]
    weak_certificate = weak["weak_field_AE_constraint_certificate"]
    if inverse is None:
        raise ValueError("regular contract certificate lacks inverse margin")
    if (
        weak_certificate.get("linearized_Hamiltonian_constraint_completed") is not True
        or weak_certificate.get("linearized_momentum_constraint_completed") is not True
    ):
        raise ValueError("weak-field conformal/York reference completion changed")
    missing = {key: "not_registered" for key in REQUIRED_CONTRACT_FIELDS}
    body = {
        "candidate_id": prior["candidate_id"],
        "typed_action_ir_sha256": prior["typed_action_ir_sha256"],
        "source_inverse_margin_record_sha256": prior["content_sha256"],
        "source_weak_field_record_sha256": weak["content_sha256"],
        "available_exact_controls": {
            "scalar_conformal_reference_solution_exists": True,
            "vector_York_reference_solution_exists": True,
            "uniform_Aether_Legendre_block_inverse_bound": inverse["kinetic_block_inverse_bound"],
            "strict_negative_source_energy_margin_over_pi": inverse[
                "strict_negative_source_margin"
            ],
        },
        "typed_weighted_operator_contract": {
            "required_fields": list(REQUIRED_CONTRACT_FIELDS),
            "missing_fields": missing,
            "complete": False,
        },
        "exact_sufficient_conditions": {
            "neumann_inverse": "C0*epsilon<1; C<=C0/(1-C0*epsilon)",
            "newton_kantorovich": "2*C^2*K*eta<=1; solution_radius<=2*C*eta",
            "completed_boundary_sign": "b1*r+b2*r^2<mu",
            "symbols": {
                "C0": "reference_full_weighted_inverse_norm",
                "epsilon": "full_operator_perturbation_norm",
                "K": "nonlinear_second_derivative_majorant",
                "eta": "seed_nonlinear_constraint_residual_norm",
                "r": "solution_radius_bound",
                "b1": "completed_boundary_first_derivative_bound",
                "b2": "completed_boundary_second_derivative_bound",
                "mu": "strict_negative_source_energy_margin",
            },
        },
        "reference_principal_and_solution_controls_present": True,
        "full_weighted_operator_isomorphism_proven": False,
        "nonlinear_remainder_bound_proven": False,
        "completed_boundary_sign_persistence_proven": False,
        "candidate_rejection_authorized": False,
    }
    return {**body, "content_sha256": _sha(body)}


def build_future_aether_weighted_ift_contract_gate(
    config: dict[str, Any], root: str | Path
) -> dict[str, Any]:
    root = Path(root).resolve()
    _validate_config(config)
    _bound_path(root, config["campaign_implementation"], "weighted IFT implementation")
    inverse_implementation = _bound_path(
        root, config["source_inverse_margin_implementation"], "inverse-margin implementation"
    )
    inverse_config = _bound_json(
        root, config["source_inverse_margin_config"], "inverse-margin config"
    )
    inverse_artifact = _bound_json(
        root,
        config["source_inverse_margin_artifact"],
        "inverse-margin artifact",
        content=True,
    )
    weak_implementation = _bound_path(
        root, config["source_weak_field_implementation"], "weak-field implementation"
    )
    weak_config = _bound_json(root, config["source_weak_field_config"], "weak-field config")
    weak_artifact = _bound_json(
        root, config["source_weak_field_artifact"], "weak-field artifact", content=True
    )
    if (
        Path(
            inspect.getsourcefile(build_future_aether_regular_adm_inverse_margin_gate) or ""
        ).resolve()
        != inverse_implementation
    ):
        raise ValueError("inverse-margin implementation entrypoint changed")
    if (
        Path(
            inspect.getsourcefile(build_future_aether_weak_field_ae_constraint_gate) or ""
        ).resolve()
        != weak_implementation
    ):
        raise ValueError("weak-field implementation entrypoint changed")
    if (
        build_future_aether_regular_adm_inverse_margin_gate(inverse_config, root)
        != inverse_artifact
    ):
        raise ValueError("inverse-margin artifact no longer replays")
    if build_future_aether_weak_field_ae_constraint_gate(weak_config, root) != weak_artifact:
        raise ValueError("weak-field artifact no longer replays")
    if (
        inverse_artifact.get("candidate_count") != 14
        or inverse_artifact.get("decision_counts") != {"blocked": 14}
        or inverse_artifact.get("first_blocker_counts")
        != {CHARACTERISTIC_BLOCKER: 11, SOURCE_REGULAR_BLOCKER: 3}
        or inverse_artifact.get("uniform_Aether_Legendre_block_inverse_pass_count") != 3
        or inverse_artifact.get("candidate_rejection_authorized_count") != 0
        or weak_artifact.get("weak_field_linearized_constraint_completion_count") != 14
    ):
        raise ValueError("weighted IFT source scope changed")

    weak_records = {item["candidate_id"]: item for item in weak_artifact["candidate_records"]}
    records = []
    blockers: Counter[str] = Counter()
    regular_count = 0
    for source in inverse_artifact["candidate_records"]:
        source_body = {key: value for key, value in source.items() if key != "content_sha256"}
        weak = weak_records.get(source["candidate_id"])
        if source.get("content_sha256") != _sha(source_body) or weak is None:
            raise ValueError("weighted IFT source candidate binding changed")
        weak_body = {key: value for key, value in weak.items() if key != "content_sha256"}
        if (
            weak.get("content_sha256") != _sha(weak_body)
            or weak.get("typed_action_ir_sha256") != source["typed_action_ir_sha256"]
            or weak.get("parameters") != source["parameters"]
        ):
            raise ValueError("weak-field candidate action binding changed")
        if source["first_blocker"] == SOURCE_REGULAR_BLOCKER:
            certificate = _contract_certificate(source, weak)
            blocker = BLOCKER
            status = "pass_reference_only"
            regular_count += 1
        else:
            certificate = None
            blocker = CHARACTERISTIC_BLOCKER
            status = "not_reached"
        blockers[blocker] += 1
        provenance_body = {
            "candidate_id": source["candidate_id"],
            "typed_action_ir_sha256": source["typed_action_ir_sha256"],
            "source_inverse_margin_record_sha256": source["content_sha256"],
            "source_weak_field_record_sha256": weak["content_sha256"],
            "weighted_ift_contract_certificate_sha256": (
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
            "source_inverse_margin_record_sha256": source["content_sha256"],
            "source_weak_field_record_sha256": weak["content_sha256"],
            "parameters": source["parameters"],
            "exact_specialization": source["exact_specialization"],
            "weighted_ift_contract_certificate": certificate,
            "gate_ledger": {
                "source_action_and_predecessor_binding": {"status": "pass"},
                "reference_conformal_York_and_Aether_blocks": {"status": status},
                "typed_weighted_operator_contract": {"status": "blocked"},
                "full_weighted_operator_isomorphism": {"status": "blocked"},
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
    if regular_count != 3 or dict(blockers) != {CHARACTERISTIC_BLOCKER: 11, BLOCKER: 3}:
        raise ValueError("future Aether weighted IFT partition changed")
    record_root = _sha(
        [
            [item["candidate_id"], item["typed_action_ir_sha256"], item["content_sha256"]]
            for item in records
        ]
    )
    provenance_body = {
        "source_inverse_margin_content_sha256": inverse_artifact["content_sha256"],
        "source_inverse_margin_record_registry_root_sha256": inverse_artifact[
            "candidate_record_registry_root_sha256"
        ],
        "source_weak_field_content_sha256": weak_artifact["content_sha256"],
        "source_weak_field_record_registry_root_sha256": weak_artifact[
            "candidate_record_registry_root_sha256"
        ],
        "candidate_record_registry_root_sha256": record_root,
        "data_eligibility": ELIGIBILITY,
    }
    body = {
        "schema_version": RESULT_SCHEMA,
        "campaign_id": config["campaign_id"],
        "immutable_config_sha256": _sha(config),
        "source_inverse_margin_binding": config["source_inverse_margin_artifact"],
        "source_weak_field_binding": config["source_weak_field_artifact"],
        "candidate_count": 14,
        "decision_counts": {"blocked": 14},
        "first_blocker_counts": dict(sorted(blockers.items())),
        "formal_pass_count": 0,
        "candidate_rejection_authorized_count": 0,
        "constraint_satisfying_negative_total_energy_datum_count": 0,
        "forced_characteristic_candidate_count": 11,
        "regular_ADM_candidate_count": 3,
        "reference_conformal_York_Aether_block_control_count": 3,
        "typed_weighted_operator_contract_complete_count": 0,
        "full_weighted_operator_isomorphism_pass_count": 0,
        "nonlinear_remainder_bound_pass_count": 0,
        "completed_boundary_sign_persistence_count": 0,
        "missing_contract_field_counts": {key: 3 for key in REQUIRED_CONTRACT_FIELDS},
        "candidate_record_registry_root_sha256": record_root,
        "candidate_records": records,
        "bounded_weighted_ift_contract_gate_completed": True,
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
            "The three regular candidates have exact conformal/York reference solutions, Aether "
            "Legendre inverse bounds, and negative source-energy margins. This gate sharply "
            "fail-closes the nonlinear lift because no candidate-bound gauge-fixed weighted "
            "domain/codomain contract, full operator perturbation norm, nonlinear residual and "
            "second-derivative majorant, or completed-boundary derivative bounds are registered. "
            "The artifact records exact Neumann, Newton-Kantorovich, and boundary-sign sufficient "
            "conditions, but does not evaluate missing quantities or authorize rejection."
        ),
    }
    return {**body, "content_sha256": _sha(body)}


def publish_future_aether_weighted_ift_contract_gate(
    config: dict[str, Any], root: str | Path, target: str | Path
) -> dict[str, Any]:
    artifact = build_future_aether_weighted_ift_contract_gate(config, root)
    target = Path(target).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if _load(target) != artifact:
            raise ValueError("refusing to replace divergent weighted IFT artifact")
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
    artifact = publish_future_aether_weighted_ift_contract_gate(
        _load(Path(arguments.config)), arguments.root, arguments.output
    )
    print(_canonical({key: value for key, value in artifact.items() if key != "candidate_records"}))


if __name__ == "__main__":
    main()
