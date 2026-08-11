"""Principal constraint-variable gate for regular future Aether free data."""

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

from .adm_aether import einstein_aether_lapse_shift_constraint_seed_control
from .future_aether_nonlinear_lift_characteristic_gate import CHARACTERISTIC_BLOCKER
from .future_aether_weighted_reference_operator_gate import (
    BLOCKER as SOURCE_REGULAR_BLOCKER,
)
from .future_aether_weighted_reference_operator_gate import (
    build_future_aether_weighted_reference_operator_gate,
    exact_reference_symbol,
)
from .promotion_orchestrator import ELIGIBILITY

CONFIG_SCHEMA = "sigma-future-aether-fixed-free-data-principal-gate-config-1.0"
RESULT_SCHEMA = "sigma-future-aether-fixed-free-data-principal-gate-1.0"
TARGET_FAMILY = "AETHER_K1234_PARAMETER_CELL"
BLOCKER = (
    "candidate_bound_finite_tilt_metric_momentum_to_York_principal_symbol_from_"
    "spatially_distributed_Legendre_map"
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
    if set(binding) - {"path", "file_sha256", "content_sha256", "callable"}:
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
        "source_weighted_reference_artifact",
        "source_weighted_reference_config",
        "source_weighted_reference_implementation",
        "reviewed_adm_constraint_control",
        "budget",
        "observational_authorization",
        "data_eligibility",
        "external_paid_llm_calls",
    }
    if set(config) != required or config.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("future Aether fixed-free-data config is invalid")
    if config.get("data_eligibility") != ELIGIBILITY:
        raise ValueError("future Aether fixed-free-data eligibility is open")
    if config.get("observational_authorization") is not False:
        raise ValueError("future Aether fixed-free-data opened observations")
    if config.get("external_paid_llm_calls") is not False:
        raise ValueError("future Aether fixed-free-data enabled paid LLM calls")
    control = config.get("reviewed_adm_constraint_control", {})
    if control.get("callable") != (
        "sigma_theory_compiler.adm_aether:einstein_aether_lapse_shift_constraint_seed_control"
    ) or control.get("content_sha256") != (
        "d8ed883d59f2fd5db01fb340104b077bf8f697278a5225e0975f993387ae7f49"
    ):
        raise ValueError("future Aether reviewed ADM control changed")
    if config.get("budget") != {
        "maximum_candidates": 14,
        "maximum_regular_adm_candidates": 3,
        "maximum_augmented_symbol_columns": 7,
        "maximum_paid_llm_spend_usd": 0.0,
    }:
        raise ValueError("future Aether fixed-free-data budget is not exact")


def exact_augmented_symbol_control() -> dict[str, Any]:
    """Show why adding the three Aether free-data components cannot form an elliptic block."""

    reference = exact_reference_symbol((Fraction(1), Fraction(0), Fraction(0)))
    matrix = [
        ["4", "0", "0", "0", "0", "0", "0"],
        ["0", "8/3", "0", "0", "0", "0", "0"],
        ["0", "0", "2", "0", "0", "0", "0"],
        ["0", "0", "0", "2", "0", "0", "0"],
    ]
    body = {
        "unit_covector": ["1", "0", "0"],
        "column_order": ["phi", "X1", "X2", "X3", "delta_A1", "delta_A2", "delta_A3"],
        "row_order": ["Hamiltonian", "momentum_1", "momentum_2", "momentum_3"],
        "second_order_symbol_matrix": matrix,
        "metric_reference_symbol_sha256": reference["content_sha256"],
        "rank": 4,
        "column_count": 7,
        "right_kernel_dimension": 3,
        "Aether_second_order_columns_zero": True,
        "augmented_square_isomorphism_possible": False,
        "interpretation": (
            "The positive-unit-branch A_i are freely specified canonical data, not three extra "
            "secondary constraint variables. Their occurrence in H and D is at spatial order at "
            "most one, so adjoining delta_A to the second-order metric solve creates three zero "
            "principal columns rather than an Aether elliptic diagonal block."
        ),
    }
    return {**body, "content_sha256": _sha(body)}


def _principal_certificate(source: dict[str, Any], adm_control_sha256: str) -> dict[str, Any]:
    prior = source["weighted_reference_operator_certificate"]
    if prior is None:
        raise ValueError("regular candidate lacks weighted reference certificate")
    if (
        prior["declared_metric_weighted_contract"].get("weight_delta") != "-1/2"
        or prior["principal_symbol_certificates"].get("ellipticity_margin") != "2"
    ):
        raise ValueError("weighted reference contract changed")
    augmented = exact_augmented_symbol_control()
    body = {
        "candidate_id": source["candidate_id"],
        "typed_action_ir_sha256": source["typed_action_ir_sha256"],
        "parameters": source["parameters"],
        "source_weighted_reference_record_sha256": source["content_sha256"],
        "reviewed_generic_ADM_constraint_control_sha256": adm_control_sha256,
        "reduced_positive_unit_branch_constraint_variables": {
            "elliptic_solve_variables": ["phi", "X_i"],
            "elliptic_solve_variable_count": 4,
            "Aether_seed_variables": ["A_i", "p_A^i"],
            "Aether_role": "prescribed_free_data_for_the_four_secondary_constraints",
            "independent_Aether_secondary_constraint_variable_count": 0,
            "secondary_constraints": ["H_A+H_GR=0", "D_i^A+D_i^GR=0"],
        },
        "reviewed_spatial_jet_order_derivation": {
            "S_ij": "D_i A_j+chi K_ij",
            "P_i": "D_i chi+K_ij A^j",
            "lapse_constraint_Aether_divergence": "D_i(chi p_W^i)",
            "shift_constraint_Aether_part": "canonical_spatial_cotangent_lift",
            "maximum_Aether_free_data_spatial_derivative_order_in_constraints": 1,
            "Aether_second_order_principal_column": "zero",
            "metric_momentum_to_K_or_York_coefficient": (
                "finite_tilt_spatially_distributed_Legendre_map_not_registered"
            ),
        },
        "augmented_Aether_unknown_negative_control": augmented,
        "fixed_Aether_free_data_reference_positive_control": {
            "metric_reference_symbol_spectrum": ["2", "2", "8/3", "4"],
            "metric_reference_ellipticity_margin": "2",
            "status": "pass_reference_only",
        },
        "Aether_constraint_variable_diagonal_second_order_block_derived": True,
        "Aether_constraint_variable_diagonal_second_order_block": "zero_dimensional",
        "Aether_free_data_off_diagonal_second_order_columns_derived": True,
        "Aether_free_data_off_diagonal_second_order_columns": "zero_4_by_3",
        "finite_tilt_metric_York_principal_block_derived": False,
        "candidate_fixed_free_data_full_principal_symbol_proven_elliptic": False,
        "lower_order_coefficient_bounds_proven": False,
        "weighted_Fredholm_isomorphism_proven": False,
        "computable_inverse_norm_proven": False,
        "nonlinear_remainder_majorant_proven": False,
        "completed_boundary_sign_persistence_proven": False,
        "candidate_rejection_authorized": False,
    }
    return {**body, "content_sha256": _sha(body)}


def build_future_aether_fixed_free_data_principal_gate(
    config: dict[str, Any], root: str | Path
) -> dict[str, Any]:
    root = Path(root).resolve()
    _validate_config(config)
    _bound_path(root, config["campaign_implementation"], "fixed-free-data implementation")
    source_implementation = _bound_path(
        root,
        config["source_weighted_reference_implementation"],
        "weighted reference implementation",
    )
    source_config = _bound_json(
        root, config["source_weighted_reference_config"], "weighted reference config"
    )
    source_artifact = _bound_json(
        root,
        config["source_weighted_reference_artifact"],
        "weighted reference artifact",
        content=True,
    )
    adm_path = _bound_path(root, config["reviewed_adm_constraint_control"], "reviewed ADM control")
    callback_source = Path(
        inspect.getsourcefile(build_future_aether_weighted_reference_operator_gate) or ""
    ).resolve()
    adm_callback_source = Path(
        inspect.getsourcefile(einstein_aether_lapse_shift_constraint_seed_control) or ""
    ).resolve()
    if callback_source != source_implementation:
        raise ValueError("weighted reference implementation entrypoint changed")
    if adm_callback_source != adm_path:
        raise ValueError("reviewed ADM implementation entrypoint changed")
    if build_future_aether_weighted_reference_operator_gate(source_config, root) != source_artifact:
        raise ValueError("weighted reference artifact no longer replays")
    adm_control = einstein_aether_lapse_shift_constraint_seed_control()
    adm_control_sha256 = _sha(adm_control)
    if (
        adm_control_sha256 != config["reviewed_adm_constraint_control"].get("content_sha256")
        or adm_control.get("passed") is not True
        or adm_control.get("canonical_configuration")
        != ["q_ij (6)", "A_i (3, positive unit branch)", "N (1)", "N^i (3)"]
        or adm_control.get("verified_constraint_generations")
        != {"primary": 4, "secondary_seeds": 4}
    ):
        raise ValueError("reviewed ADM constraint control changed")
    if (
        source_artifact.get("candidate_count") != 14
        or source_artifact.get("decision_counts") != {"blocked": 14}
        or source_artifact.get("first_blocker_counts")
        != {CHARACTERISTIC_BLOCKER: 11, SOURCE_REGULAR_BLOCKER: 3}
        or source_artifact.get("declared_metric_weighted_contract_count") != 3
        or source_artifact.get("candidate_rejection_authorized_count") != 0
    ):
        raise ValueError("fixed-free-data source scope changed")

    records = []
    blockers: Counter[str] = Counter()
    regular_count = 0
    for source in source_artifact["candidate_records"]:
        source_body = {key: value for key, value in source.items() if key != "content_sha256"}
        if source.get("content_sha256") != _sha(source_body):
            raise ValueError("fixed-free-data source record changed")
        if source["first_blocker"] == SOURCE_REGULAR_BLOCKER:
            certificate = _principal_certificate(source, adm_control_sha256)
            blocker = BLOCKER
            status = "pass"
            regular_count += 1
        else:
            certificate = None
            blocker = CHARACTERISTIC_BLOCKER
            status = "not_reached"
        blockers[blocker] += 1
        provenance_body = {
            "candidate_id": source["candidate_id"],
            "typed_action_ir_sha256": source["typed_action_ir_sha256"],
            "source_weighted_reference_record_sha256": source["content_sha256"],
            "fixed_free_data_principal_certificate_sha256": (
                certificate["content_sha256"] if certificate is not None else None
            ),
            "reviewed_ADM_constraint_control_sha256": adm_control_sha256,
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
            "source_weighted_reference_record_sha256": source["content_sha256"],
            "parameters": source["parameters"],
            "exact_specialization": source["exact_specialization"],
            "fixed_free_data_principal_certificate": certificate,
            "gate_ledger": {
                "source_action_and_predecessor_binding": {"status": "pass"},
                "positive_unit_branch_constraint_variable_classification": {"status": status},
                "Aether_second_order_diagonal_and_off_diagonal_columns": {"status": status},
                "augmented_Aether_unknown_negative_control": {"status": status},
                "finite_tilt_metric_York_principal_symbol": {"status": "blocked"},
                "weighted_Fredholm_isomorphism_and_norm": {"status": "blocked"},
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
        raise ValueError("future Aether fixed-free-data partition changed")
    record_root = _sha(
        [
            [item["candidate_id"], item["typed_action_ir_sha256"], item["content_sha256"]]
            for item in records
        ]
    )
    provenance_body = {
        "source_weighted_reference_content_sha256": source_artifact["content_sha256"],
        "source_weighted_reference_record_registry_root_sha256": source_artifact[
            "candidate_record_registry_root_sha256"
        ],
        "reviewed_ADM_constraint_control_sha256": adm_control_sha256,
        "candidate_record_registry_root_sha256": record_root,
        "data_eligibility": ELIGIBILITY,
    }
    body = {
        "schema_version": RESULT_SCHEMA,
        "campaign_id": config["campaign_id"],
        "immutable_config_sha256": _sha(config),
        "source_weighted_reference_binding": config["source_weighted_reference_artifact"],
        "reviewed_ADM_constraint_control_binding": config["reviewed_adm_constraint_control"],
        "candidate_count": 14,
        "decision_counts": {"blocked": 14},
        "first_blocker_counts": dict(sorted(blockers.items())),
        "formal_pass_count": 0,
        "candidate_rejection_authorized_count": 0,
        "constraint_satisfying_negative_total_energy_datum_count": 0,
        "forced_characteristic_candidate_count": 11,
        "regular_ADM_candidate_count": 3,
        "positive_unit_branch_constraint_variable_classification_count": 3,
        "zero_dimensional_Aether_constraint_diagonal_block_count": 3,
        "zero_Aether_second_order_off_diagonal_columns_count": 3,
        "augmented_Aether_unknown_nonelliptic_negative_control_count": 3,
        "finite_tilt_metric_York_principal_symbol_pass_count": 0,
        "fixed_free_data_full_principal_ellipticity_pass_count": 0,
        "lower_order_coefficient_bound_pass_count": 0,
        "weighted_Fredholm_isomorphism_pass_count": 0,
        "computable_full_inverse_norm_count": 0,
        "nonlinear_remainder_bound_pass_count": 0,
        "completed_boundary_sign_persistence_count": 0,
        "candidate_record_registry_root_sha256": record_root,
        "candidate_records": records,
        "bounded_fixed_free_data_principal_gate_completed": True,
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
            "On the reduced positive-unit branch the Aether variables are prescribed free data "
            "for the four Hamiltonian/momentum constraints, not extra elliptic constraint "
            "variables. The reviewed K1-K4 spatial jets put Aether free-data derivatives at order "
            "at most one, so their second-order diagonal/off-diagonal columns are exactly absent; "
            "augmenting the four metric solve variables by delta A creates a rank-4, seven-column "
            "symbol with a three-dimensional kernel. The next premise is the actual finite-tilt "
            "metric-momentum-to-York principal block from the spatially distributed Legendre map. "
            "No Fredholm, nonlinear, boundary-sign, or theory-rejection claim follows."
        ),
    }
    return {**body, "content_sha256": _sha(body)}


def publish_future_aether_fixed_free_data_principal_gate(
    config: dict[str, Any], root: str | Path, target: str | Path
) -> dict[str, Any]:
    artifact = build_future_aether_fixed_free_data_principal_gate(config, root)
    target = Path(target).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if _load(target) != artifact:
            raise ValueError("refusing to replace divergent fixed-free-data artifact")
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
    artifact = publish_future_aether_fixed_free_data_principal_gate(
        _load(Path(arguments.config)), arguments.root, arguments.output
    )
    print(_canonical({key: value for key, value in artifact.items() if key != "candidate_records"}))


if __name__ == "__main__":
    main()
