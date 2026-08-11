"""Characteristic obstruction gate for nonlinear future-Aether seed lifts."""

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

from .aether_parameter_cell_formal_gate_campaign import _specialize
from .future_aether_finite_amplitude_negative_seed_gate import (
    build_future_aether_finite_amplitude_negative_seed_gate,
)
from .promotion_orchestrator import ELIGIBILITY

CONFIG_SCHEMA = "sigma-future-aether-nonlinear-lift-characteristic-gate-config-1.0"
RESULT_SCHEMA = "sigma-future-aether-nonlinear-lift-characteristic-gate-1.0"
TARGET_FAMILY = "AETHER_K1234_PARAMETER_CELL"
SOURCE_BLOCKER = (
    "nonlinear_Einstein_Aether_constraint_lift_of_explicit_compact_negative_source_seed_"
    "with_sign_preserving_completed_boundary_energy"
)
CHARACTERISTIC_BLOCKER = (
    "noncharacteristic_foliation_or_compact_negative_seed_avoiding_forced_ADM_"
    "Legendre_characteristic_crossing"
)
IFT_BLOCKER = (
    "candidate_bound_weighted_nonlinear_Einstein_Aether_constraint_map_inverse_and_"
    "remainder_bound_with_completed_boundary_sign_persistence"
)

I_GRAD = Fraction(262144, 255255)
I_AXIS = Fraction(262144, 765765)
I_ACCELERATION = Fraction(8589934592, 148767396525)


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
        "source_negative_seed_artifact",
        "source_negative_seed_config",
        "source_negative_seed_implementation",
        "budget",
        "observational_authorization",
        "data_eligibility",
        "external_paid_llm_calls",
    }
    if set(config) != required or config.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("future Aether characteristic-gate config is invalid")
    if config.get("data_eligibility") != ELIGIBILITY:
        raise ValueError("future Aether characteristic-gate eligibility is open")
    if config.get("observational_authorization") is not False:
        raise ValueError("future Aether characteristic-gate opened observations")
    if config.get("external_paid_llm_calls") is not False:
        raise ValueError("future Aether characteristic-gate enabled paid LLM calls")
    if config.get("budget") != {
        "maximum_candidates": 14,
        "registered_seed_amplitude_squared": 100,
        "maximum_characteristic_surfaces_per_candidate": 2,
        "maximum_paid_llm_spend_usd": 0.0,
    }:
        raise ValueError("future Aether characteristic-gate budget is not exact")


def _candidate_certificate(parameters: dict[str, str], specialization: dict[str, Any]) -> dict:
    c1 = Fraction(parameters["c1"])
    c23 = Fraction(parameters["c2"]) + Fraction(parameters["c3"])
    c4 = Fraction(parameters["c4"])
    negativity_threshold = (c1 * I_GRAD + c23 * I_AXIS) / (c4 * I_ACCELERATION)
    strata = specialization["global_unit_tilt_legendre_strata"]
    thresholds = {
        key: Fraction(value) for key, value in strata["finite_characteristic_tilt_squared"].items()
    }
    if len(thresholds) > 2 or negativity_threshold >= 100:
        raise ValueError("candidate characteristic or negativity threshold changed")
    registered_crossings = {
        mode: {
            "tilt_squared": str(value),
            "exact_radius_squared": f"1-(({value})/100)^(1/8)",
        }
        for mode, value in sorted(thresholds.items())
        if 0 < value < 100
    }
    first_characteristic = min(thresholds.values()) if thresholds else None
    forced = first_characteristic is not None and first_characteristic <= negativity_threshold
    alternative = first_characteristic is not None and negativity_threshold < first_characteristic
    globally_noncharacteristic = (
        not thresholds and strata["globally_noncharacteristic_for_finite_unit_tilt"]
    )
    if alternative:
        adjusted_amplitude = (negativity_threshold + first_characteristic) / 2
        blocker = IFT_BLOCKER
        lift_ready = True
    elif globally_noncharacteristic:
        adjusted_amplitude = Fraction(100)
        blocker = IFT_BLOCKER
        lift_ready = True
    else:
        adjusted_amplitude = None
        blocker = CHARACTERISTIC_BLOCKER
        lift_ready = False
    body = {
        "parameters": parameters,
        "negative_source_amplitude_threshold_squared": str(negativity_threshold),
        "finite_characteristic_tilt_squared": {
            key: str(value) for key, value in sorted(thresholds.items())
        },
        "registered_seed_characteristic_crossings": registered_crossings,
        "registered_seed_crosses_characteristic_surface": bool(registered_crossings),
        "negative_source_family_forces_characteristic_crossing": forced,
        "certified_negative_characteristic_free_amplitude_window_exists": alternative,
        "globally_noncharacteristic_for_finite_unit_tilt": globally_noncharacteristic,
        "adjusted_characteristic_free_amplitude_squared": (
            str(adjusted_amplitude) if adjusted_amplitude is not None else None
        ),
        "regular_ADM_implicit_lift_prerequisite_pass": lift_ready,
        "full_nonlinear_constraint_solution_proven": False,
        "completed_boundary_sign_persistence_proven": False,
        "candidate_rejection_authorized": False,
        "first_blocker": blocker,
    }
    return {**body, "content_sha256": _sha(body)}


def build_future_aether_nonlinear_lift_characteristic_gate(
    config: dict[str, Any], root: str | Path
) -> dict[str, Any]:
    root = Path(root).resolve()
    _validate_config(config)
    _bound_path(root, config["campaign_implementation"], "characteristic implementation")
    source_implementation = _bound_path(
        root,
        config["source_negative_seed_implementation"],
        "source negative-seed implementation",
    )
    source_config = _bound_json(
        root, config["source_negative_seed_config"], "source negative-seed config"
    )
    source_artifact = _bound_json(
        root,
        config["source_negative_seed_artifact"],
        "source negative-seed artifact",
        content=True,
    )
    callback_source = Path(
        inspect.getsourcefile(build_future_aether_finite_amplitude_negative_seed_gate) or ""
    ).resolve()
    if callback_source != source_implementation:
        raise ValueError("source negative-seed implementation entrypoint changed")
    if (
        build_future_aether_finite_amplitude_negative_seed_gate(source_config, root)
        != source_artifact
    ):
        raise ValueError("source negative-seed artifact no longer replays")
    if (
        source_artifact.get("candidate_count") != 14
        or source_artifact.get("decision_counts") != {"blocked": 14}
        or source_artifact.get("first_blocker_counts") != {SOURCE_BLOCKER: 14}
        or source_artifact.get("exact_negative_static_source_monopole_count") != 14
        or source_artifact.get("full_nonlinear_constraint_completion_count") != 0
        or source_artifact.get("candidate_rejection_authorized_count") != 0
    ):
        raise ValueError("source negative-seed decision scope changed")

    records = []
    blockers: Counter[str] = Counter()
    for source in source_artifact["candidate_records"]:
        source_body = {key: value for key, value in source.items() if key != "content_sha256"}
        specialization = _specialize(source["parameters"])
        if (
            source.get("content_sha256") != _sha(source_body)
            or source.get("family_id") != TARGET_FAMILY
            or source.get("decision") != "blocked"
            or source.get("first_blocker") != SOURCE_BLOCKER
            or source.get("exact_specialization") != specialization
        ):
            raise ValueError("source negative-seed candidate record changed")
        certificate = _candidate_certificate(source["parameters"], specialization)
        blocker = certificate["first_blocker"]
        blockers[blocker] += 1
        gates = {
            "source_action_and_negative_seed_binding": {"status": "pass"},
            "registered_seed_characteristic_shell_audit": {
                "status": (
                    "blocked"
                    if certificate["registered_seed_crosses_characteristic_surface"]
                    else "pass"
                )
            },
            "negative_seed_family_noncharacteristic_overlap": {
                "status": (
                    "blocked"
                    if certificate["negative_source_family_forces_characteristic_crossing"]
                    else "pass"
                )
            },
            "regular_ADM_implicit_lift_prerequisite": {
                "status": "pass"
                if certificate["regular_ADM_implicit_lift_prerequisite_pass"]
                else "blocked"
            },
            "weighted_nonlinear_constraint_inverse_and_remainder": {"status": "blocked"},
            "completed_boundary_sign_persistence": {"status": "blocked"},
            "observational_data_seal": {"status": "pass"},
        }
        provenance_body = {
            "candidate_id": source["candidate_id"],
            "typed_action_ir_sha256": source["typed_action_ir_sha256"],
            "source_negative_seed_record_sha256": source["content_sha256"],
            "source_exact_specialization_sha256": specialization["content_sha256"],
            "characteristic_certificate_sha256": certificate["content_sha256"],
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
            "source_negative_seed_record_sha256": source["content_sha256"],
            "parameters": source["parameters"],
            "exact_specialization": specialization,
            "nonlinear_lift_characteristic_certificate": certificate,
            "gate_ledger": gates,
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
    if len(records) != 14 or dict(blockers) != {CHARACTERISTIC_BLOCKER: 11, IFT_BLOCKER: 3}:
        raise ValueError("future Aether characteristic partition changed")
    record_root = _sha(
        [
            [item["candidate_id"], item["typed_action_ir_sha256"], item["content_sha256"]]
            for item in records
        ]
    )
    body = {
        "schema_version": RESULT_SCHEMA,
        "campaign_id": config["campaign_id"],
        "immutable_config_sha256": _sha(config),
        "source_negative_seed_binding": config["source_negative_seed_artifact"],
        "candidate_count": 14,
        "decision_counts": {"blocked": 14},
        "first_blocker_counts": dict(sorted(blockers.items())),
        "formal_pass_count": 0,
        "candidate_rejection_authorized_count": 0,
        "constraint_satisfying_negative_total_energy_datum_count": 0,
        "registered_seed_characteristic_crossing_count": 13,
        "negative_source_family_forced_characteristic_crossing_count": 11,
        "certified_negative_characteristic_free_amplitude_window_count": 2,
        "globally_noncharacteristic_candidate_count": 1,
        "regular_ADM_implicit_lift_prerequisite_pass_count": 3,
        "full_nonlinear_constraint_completion_count": 0,
        "completed_boundary_sign_persistence_count": 0,
        "candidate_record_registry_root_sha256": record_root,
        "candidate_records": records,
        "bounded_nonlinear_lift_characteristic_gate_completed": True,
        "full_candidate_specific_formal_completion_claimed": False,
        "automatic_downstream_enqueue_performed": False,
        "solar_bundle_count": 0,
        "observational_data_opened": False,
        "dark_matter_or_halo_inputs": False,
        "redshift_distance_inputs": False,
        "paid_llm_spend_usd": 0.0,
        "data_eligibility": dict(ELIGIBILITY),
        "provenance": {
            "source_negative_seed_content_sha256": source_artifact["content_sha256"],
            "source_negative_seed_record_registry_root_sha256": source_artifact[
                "candidate_record_registry_root_sha256"
            ],
            "candidate_record_registry_root_sha256": record_root,
            "data_eligibility": ELIGIBILITY,
        },
        "interpretation": (
            "The registered amplitude-100 compact negative seed crosses a finite ADM Legendre-"
            "characteristic shell for thirteen candidates. For eleven, the exact negative-source "
            "amplitude threshold already lies beyond the first characteristic shell, so this seed "
            "family cannot enter a regular ADM implicit-function lift without a different foliation "
            "or profile. Two candidates have exact lower-amplitude negative, characteristic-free "
            "windows and one is globally noncharacteristic; those three advance to the weighted "
            "nonlinear inverse/remainder and boundary-sign estimate. Characteristic slicing is not "
            "a theory rejection, and no nonlinear constraint solution is claimed."
        ),
    }
    provenance_body = body["provenance"]
    body["provenance"] = {**provenance_body, "binding_sha256": _sha(provenance_body)}
    return {**body, "content_sha256": _sha(body)}


def publish_future_aether_nonlinear_lift_characteristic_gate(
    config: dict[str, Any], root: str | Path, target: str | Path
) -> dict[str, Any]:
    artifact = build_future_aether_nonlinear_lift_characteristic_gate(config, root)
    target = Path(target).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if _load(target) != artifact:
            raise ValueError("refusing to replace divergent characteristic artifact")
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
    artifact = publish_future_aether_nonlinear_lift_characteristic_gate(
        _load(Path(arguments.config)), arguments.root, arguments.output
    )
    print(_canonical({key: value for key, value in artifact.items() if key != "candidate_records"}))


if __name__ == "__main__":
    main()
