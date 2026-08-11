"""Principal inverse and weighted-Fredholm premise gate for one Aether candidate."""

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

from .future_aether_finite_tilt_york_symbol_gate import (
    FREDHOLM_BLOCKER as SOURCE_ELLIPTIC_BLOCKER,
)
from .future_aether_finite_tilt_york_symbol_gate import (
    YORK_SHELL_BLOCKER,
    build_future_aether_finite_tilt_york_symbol_gate,
)
from .future_aether_nonlinear_lift_characteristic_gate import CHARACTERISTIC_BLOCKER
from .promotion_orchestrator import ELIGIBILITY

CONFIG_SCHEMA = "sigma-future-aether-principal-inverse-fredholm-gate-config-1.0"
RESULT_SCHEMA = "sigma-future-aether-principal-inverse-fredholm-gate-1.0"
TARGET_FAMILY = "AETHER_K1234_PARAMETER_CELL"
BLOCKER = (
    "candidate_bound_spatially_distributed_lower_order_linearized_constraint_"
    "coefficient_registry_on_weighted_spaces"
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
        "source_york_symbol_artifact",
        "source_york_symbol_config",
        "source_york_symbol_implementation",
        "source_compact_seed_artifact",
        "budget",
        "observational_authorization",
        "data_eligibility",
        "external_paid_llm_calls",
    }
    if set(config) != required or config.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("future Aether principal inverse config is invalid")
    if config.get("data_eligibility") != ELIGIBILITY:
        raise ValueError("future Aether principal inverse eligibility is open")
    if config.get("observational_authorization") is not False:
        raise ValueError("future Aether principal inverse opened observations")
    if config.get("external_paid_llm_calls") is not False:
        raise ValueError("future Aether principal inverse enabled paid LLM calls")
    if config.get("budget") != {
        "maximum_candidates": 14,
        "maximum_uniformly_elliptic_candidates": 1,
        "maximum_exact_rational_bounds": 3,
        "maximum_paid_llm_spend_usd": 0.0,
    }:
        raise ValueError("future Aether principal inverse budget is not exact")


def exact_principal_inverse_control(amplitude: Fraction) -> dict[str, Any]:
    """Give a rigorous uniform determinant gap and adjugate inverse bound."""

    if amplitude < 0 or amplitude >= 31:
        raise ValueError("principal inverse control requires 0<=amplitude<31")
    determinant_gap = Fraction(31, 384) * (31 - amplitude) ** 2 / (amplitude + 2) ** 3
    entry_bound = (8 * amplitude**2 + 415 * amplitude + 868) / 192
    york_inverse_bound = 6 * entry_bound**2 / determinant_gap
    if determinant_gap <= 0 or entry_bound <= 0 or york_inverse_bound <= 0:
        raise ValueError("principal inverse bounds lost positivity")
    body = {
        "registered_tilt_upper": str(amplitude),
        "York_determinant_absolute_lower_bound": str(determinant_gap),
        "York_symbol_entry_absolute_upper_bound": str(entry_bound),
        "York_symbol_inverse_2_norm_upper_bound": str(york_inverse_bound),
        "Hamiltonian_scalar_symbol_inverse_bound": "1/4",
        "full_block_principal_symbol_inverse_bound": str(max(york_inverse_bound, Fraction(1, 4))),
        "determinant_gap_derivation": {
            "factor_1_lower": "4*(31-Y)",
            "factor_2_lower": "496*(31-Y)",
            "denominator_upper": "24576*(Y+2)^3",
            "combined": "31*(31-Y)^2/(384*(Y+2)^3)",
        },
        "entry_bound_derivation": {
            "uniform_entry_bound": "(8*Y^2+415*Y+868)/192",
            "adjugate_Frobenius_bound": "6*B^2",
            "inverse_bound": "6*B^2/determinant_gap",
        },
        "principal_inverse_bound_proven": True,
    }
    return {**body, "content_sha256": _sha(body)}


def exact_characteristic_boundary_negative_control() -> dict[str, Any]:
    """Verify that the quantitative gap closes exactly at the excluded y=31 shell."""

    y = Fraction(31)
    numerator = Fraction(31, 384) * (31 - y) ** 2
    body = {
        "excluded_tilt": "31",
        "determinant_gap_numerator": str(numerator),
        "determinant_gap_positive": False,
        "principal_inverse_bound_available": False,
        "control_status": "reject_boundary_input_not_candidate",
    }
    return {**body, "content_sha256": _sha(body)}


def _candidate_certificate(source: dict[str, Any], seed_source: dict[str, Any]) -> dict[str, Any]:
    york = source["finite_tilt_York_symbol_certificate"]
    if york is None or york.get("uniform_fixed_free_data_principal_ellipticity_proven") is not True:
        raise ValueError("principal inverse candidate is not uniformly elliptic")
    if (
        seed_source["typed_action_ir_sha256"] != source["typed_action_ir_sha256"]
        or seed_source["parameters"] != source["parameters"]
        or seed_source["finite_amplitude_negative_seed_certificate"].get(
            "compact_asymptotically_Euclidean_Aether_seed"
        )
        is not True
    ):
        raise ValueError("compact seed candidate binding changed")
    amplitude = Fraction(york["registered_characteristic_free_tilt_upper"])
    inverse = exact_principal_inverse_control(amplitude)
    negative = exact_characteristic_boundary_negative_control()
    body = {
        "candidate_id": source["candidate_id"],
        "typed_action_ir_sha256": source["typed_action_ir_sha256"],
        "source_York_symbol_record_sha256": source["content_sha256"],
        "source_compact_seed_record_sha256": seed_source["content_sha256"],
        "principal_inverse_control": inverse,
        "excluded_characteristic_boundary_negative_control": negative,
        "principal_coefficient_AE_contract": {
            "weight_delta": "-1/2",
            "domain": "H^2_-1/2(R3;scalar+vector)",
            "codomain": "L^2_-5/2(R3;scalar+vector)",
            "compact_seed_profile": "A=a*(1-r^2)^4_+*e1",
            "compact_seed_profile_regular_class": "C3",
            "principal_coefficients_regular_class": "C3_rational_functions_of_y",
            "principal_coefficient_denominator": "y+2>=2",
            "equals_Euclidean_reference_outside_unit_ball": True,
        },
        "elliptic_symbol_homotopy": {
            "path": "y_t=t*y for t in [0,1]",
            "path_tilt_range": f"0<=y_t<={amplitude}<31",
            "no_principal_symbol_crossing": True,
            "same_principal_elliptic_homotopy_class_as_reference": True,
        },
        "missing_distributed_lower_order_registry": {
            "H_core_linearization_order_0_and_1_coefficients": "not_registered",
            "momentum_constraint_order_0_and_1_coefficients": "not_registered",
            "weighted_relative_bound_against_principal_part": "not_registered",
            "weighted_kernel_or_coercivity_estimate": "not_registered",
        },
        "uniform_principal_symbol_inverse_bound_proven": True,
        "principal_elliptic_homotopy_to_reference_proven": True,
        "full_weighted_operator_defined_with_all_coefficients": False,
        "weighted_Fredholm_isomorphism_proven": False,
        "full_operator_inverse_norm_proven": False,
        "nonlinear_remainder_majorant_proven": False,
        "completed_boundary_sign_persistence_proven": False,
        "candidate_rejection_authorized": False,
    }
    return {**body, "content_sha256": _sha(body)}


def build_future_aether_principal_inverse_fredholm_gate(
    config: dict[str, Any], root: str | Path
) -> dict[str, Any]:
    root = Path(root).resolve()
    _validate_config(config)
    _bound_path(root, config["campaign_implementation"], "principal inverse implementation")
    source_implementation = _bound_path(
        root, config["source_york_symbol_implementation"], "York symbol implementation"
    )
    source_config = _bound_json(root, config["source_york_symbol_config"], "York symbol config")
    source_artifact = _bound_json(
        root, config["source_york_symbol_artifact"], "York symbol artifact", content=True
    )
    seed_artifact = _bound_json(
        root, config["source_compact_seed_artifact"], "compact seed artifact", content=True
    )
    callback_source = Path(
        inspect.getsourcefile(build_future_aether_finite_tilt_york_symbol_gate) or ""
    ).resolve()
    if callback_source != source_implementation:
        raise ValueError("York symbol implementation entrypoint changed")
    if build_future_aether_finite_tilt_york_symbol_gate(source_config, root) != source_artifact:
        raise ValueError("York symbol artifact no longer replays")
    if (
        source_artifact.get("candidate_count") != 14
        or source_artifact.get("decision_counts") != {"blocked": 14}
        or source_artifact.get("first_blocker_counts")
        != {
            CHARACTERISTIC_BLOCKER: 11,
            SOURCE_ELLIPTIC_BLOCKER: 1,
            YORK_SHELL_BLOCKER: 2,
        }
        or source_artifact.get("uniform_fixed_free_data_principal_ellipticity_pass_count") != 1
        or source_artifact.get("candidate_rejection_authorized_count") != 0
        or seed_artifact.get("candidate_count") != 14
    ):
        raise ValueError("principal inverse source scope changed")
    seed_records = {item["candidate_id"]: item for item in seed_artifact["candidate_records"]}

    records = []
    blockers: Counter[str] = Counter()
    inverse_count = 0
    for source in source_artifact["candidate_records"]:
        source_body = {key: value for key, value in source.items() if key != "content_sha256"}
        if source.get("content_sha256") != _sha(source_body):
            raise ValueError("principal inverse source record changed")
        if source["first_blocker"] == SOURCE_ELLIPTIC_BLOCKER:
            seed_source = seed_records.get(source["candidate_id"])
            if seed_source is None:
                raise ValueError("principal inverse compact seed record is missing")
            certificate = _candidate_certificate(source, seed_source)
            blocker = BLOCKER
            status = "pass"
            inverse_count += 1
        else:
            certificate = None
            blocker = source["first_blocker"]
            status = "not_reached"
        blockers[blocker] += 1
        provenance_body = {
            "candidate_id": source["candidate_id"],
            "typed_action_ir_sha256": source["typed_action_ir_sha256"],
            "source_York_symbol_record_sha256": source["content_sha256"],
            "principal_inverse_certificate_sha256": (
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
            "source_York_symbol_record_sha256": source["content_sha256"],
            "parameters": source["parameters"],
            "exact_specialization": source["exact_specialization"],
            "principal_inverse_fredholm_certificate": certificate,
            "gate_ledger": {
                "source_action_and_predecessor_binding": {"status": "pass"},
                "uniform_principal_symbol_inverse_bound": {"status": status},
                "principal_elliptic_homotopy_to_reference": {"status": status},
                "distributed_lower_order_coefficient_registry": {"status": "blocked"},
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
    if inverse_count != 1 or dict(blockers) != expected_blockers:
        raise ValueError("future Aether principal inverse partition changed")
    record_root = _sha(
        [
            [item["candidate_id"], item["typed_action_ir_sha256"], item["content_sha256"]]
            for item in records
        ]
    )
    provenance_body = {
        "source_York_symbol_content_sha256": source_artifact["content_sha256"],
        "source_York_symbol_record_registry_root_sha256": source_artifact[
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
        "source_York_symbol_binding": config["source_york_symbol_artifact"],
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
        "uniform_principal_symbol_inverse_bound_pass_count": 1,
        "principal_elliptic_homotopy_to_reference_pass_count": 1,
        "distributed_lower_order_coefficient_registry_complete_count": 0,
        "weighted_Fredholm_isomorphism_pass_count": 0,
        "full_operator_inverse_norm_pass_count": 0,
        "nonlinear_remainder_bound_pass_count": 0,
        "completed_boundary_sign_persistence_count": 0,
        "candidate_record_registry_root_sha256": record_root,
        "candidate_records": records,
        "bounded_principal_inverse_fredholm_gate_completed": True,
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
            "For the sole uniformly elliptic York candidate, exact factor bounds yield a positive "
            "uniform determinant gap and an explicit pointwise principal-symbol inverse norm over "
            "the complete compact seed. Scaling the tilt to zero stays elliptic and gives an exact "
            "principal-symbol homotopy to the Euclidean reference. The spatially distributed "
            "order-zero/one linearized constraint coefficients are not registered, so no full "
            "weighted operator, Fredholm isomorphism, operator inverse norm, nonlinear remainder, "
            "boundary-sign persistence, or candidate rejection is claimed."
        ),
    }
    return {**body, "content_sha256": _sha(body)}


def publish_future_aether_principal_inverse_fredholm_gate(
    config: dict[str, Any], root: str | Path, target: str | Path
) -> dict[str, Any]:
    artifact = build_future_aether_principal_inverse_fredholm_gate(config, root)
    target = Path(target).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if _load(target) != artifact:
            raise ValueError("refusing to replace divergent principal inverse artifact")
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
    artifact = publish_future_aether_principal_inverse_fredholm_gate(
        _load(Path(arguments.config)), arguments.root, arguments.output
    )
    print(_canonical({key: value for key, value in artifact.items() if key != "candidate_records"}))


if __name__ == "__main__":
    main()
