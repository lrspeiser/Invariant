"""Uniform regular-ADM inverse-margin gate for future Aether seed lifts."""

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
from .future_aether_nonlinear_lift_characteristic_gate import (
    CHARACTERISTIC_BLOCKER,
    IFT_BLOCKER,
    build_future_aether_nonlinear_lift_characteristic_gate,
)
from .promotion_orchestrator import ELIGIBILITY

CONFIG_SCHEMA = "sigma-future-aether-regular-adm-inverse-margin-gate-config-1.0"
RESULT_SCHEMA = "sigma-future-aether-regular-adm-inverse-margin-gate-1.0"
TARGET_FAMILY = "AETHER_K1234_PARAMETER_CELL"
BLOCKER = (
    "candidate_bound_weighted_elliptic_Einstein_Aether_constraint_operator_isomorphism_"
    "and_nonlinear_remainder_bound_with_completed_boundary_sign_persistence"
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
        "source_characteristic_artifact",
        "source_characteristic_config",
        "source_characteristic_implementation",
        "budget",
        "observational_authorization",
        "data_eligibility",
        "external_paid_llm_calls",
    }
    if set(config) != required or config.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("future Aether inverse-margin config is invalid")
    if config.get("data_eligibility") != ELIGIBILITY:
        raise ValueError("future Aether inverse-margin eligibility is open")
    if config.get("observational_authorization") is not False:
        raise ValueError("future Aether inverse-margin opened observations")
    if config.get("external_paid_llm_calls") is not False:
        raise ValueError("future Aether inverse-margin enabled paid LLM calls")
    if config.get("budget") != {
        "maximum_candidates": 14,
        "maximum_regular_adm_candidates": 3,
        "maximum_paid_llm_spend_usd": 0.0,
    }:
        raise ValueError("future Aether inverse-margin budget is not exact")


def _regular_certificate(parameters: dict[str, str], prior: dict[str, Any]) -> dict[str, Any]:
    amplitude = Fraction(prior["adjusted_characteristic_free_amplitude_squared"])
    specialization = _specialize(parameters)
    speeds = {
        key: Fraction(value) for key, value in specialization["principal_speed_squared"].items()
    }
    thresholds = {
        key: Fraction(value) for key, value in prior["finite_characteristic_tilt_squared"].items()
    }
    factors = {}
    for sector, speed in sorted(speeds.items()):
        if speed > 1:
            threshold = thresholds[sector]
            factor = 1 - amplitude / threshold
        else:
            factor = Fraction(1)
        if factor <= 0:
            raise ValueError("regular ADM characteristic margin is not positive")
        factors[sector] = factor
    margin = min(factors.values())
    c1 = Fraction(parameters["c1"])
    c23 = Fraction(parameters["c2"]) + Fraction(parameters["c3"])
    c4 = Fraction(parameters["c4"])
    energy_upper = amplitude / 2 * (c1 * I_GRAD + c23 * I_AXIS - c4 * amplitude * I_ACCELERATION)
    if energy_upper >= 0:
        raise ValueError("regular ADM seed lost negative source-energy margin")
    body = {
        "parameters": parameters,
        "characteristic_free_seed_amplitude_squared": str(amplitude),
        "normalized_Legendre_sector_margins": {key: str(value) for key, value in factors.items()},
        "uniform_normalized_Legendre_margin": str(margin),
        "kinetic_block_inverse_bound": str(1 / margin),
        "static_source_energy_upper_bound_over_pi": str(energy_upper),
        "strict_negative_source_margin": str(-energy_upper),
        "uniform_Aether_Legendre_block_inverse_proven": True,
        "weighted_full_constraint_operator_isomorphism_proven": False,
        "nonlinear_Frechet_remainder_bound_proven": False,
        "completed_boundary_sign_persistence_proven": False,
        "candidate_rejection_authorized": False,
    }
    return {**body, "content_sha256": _sha(body)}


def build_future_aether_regular_adm_inverse_margin_gate(
    config: dict[str, Any], root: str | Path
) -> dict[str, Any]:
    root = Path(root).resolve()
    _validate_config(config)
    _bound_path(root, config["campaign_implementation"], "inverse-margin implementation")
    source_implementation = _bound_path(
        root,
        config["source_characteristic_implementation"],
        "source characteristic implementation",
    )
    source_config = _bound_json(
        root, config["source_characteristic_config"], "source characteristic config"
    )
    source_artifact = _bound_json(
        root,
        config["source_characteristic_artifact"],
        "source characteristic artifact",
        content=True,
    )
    callback_source = Path(
        inspect.getsourcefile(build_future_aether_nonlinear_lift_characteristic_gate) or ""
    ).resolve()
    if callback_source != source_implementation:
        raise ValueError("source characteristic implementation entrypoint changed")
    if (
        build_future_aether_nonlinear_lift_characteristic_gate(source_config, root)
        != source_artifact
    ):
        raise ValueError("source characteristic artifact no longer replays")
    if (
        source_artifact.get("candidate_count") != 14
        or source_artifact.get("decision_counts") != {"blocked": 14}
        or source_artifact.get("first_blocker_counts")
        != {CHARACTERISTIC_BLOCKER: 11, IFT_BLOCKER: 3}
        or source_artifact.get("regular_ADM_implicit_lift_prerequisite_pass_count") != 3
        or source_artifact.get("candidate_rejection_authorized_count") != 0
    ):
        raise ValueError("source characteristic decision scope changed")

    records = []
    blockers: Counter[str] = Counter()
    inverse_bounds: Counter[str] = Counter()
    energy_margins: Counter[str] = Counter()
    regular_count = 0
    for source in source_artifact["candidate_records"]:
        source_body = {key: value for key, value in source.items() if key != "content_sha256"}
        specialization = _specialize(source["parameters"])
        if (
            source.get("content_sha256") != _sha(source_body)
            or source.get("family_id") != TARGET_FAMILY
            or source.get("decision") != "blocked"
            or source.get("exact_specialization") != specialization
        ):
            raise ValueError("source characteristic candidate record changed")
        prior = source["nonlinear_lift_characteristic_certificate"]
        if source["first_blocker"] == IFT_BLOCKER:
            certificate = _regular_certificate(source["parameters"], prior)
            blocker = BLOCKER
            regular_count += 1
            inverse_bounds[certificate["kinetic_block_inverse_bound"]] += 1
            energy_margins[certificate["strict_negative_source_margin"]] += 1
            regular_status = "pass"
        else:
            certificate = None
            blocker = CHARACTERISTIC_BLOCKER
            regular_status = "not_reached"
        blockers[blocker] += 1
        gates = {
            "source_action_and_characteristic_binding": {"status": "pass"},
            "regular_ADM_characteristic_free_seed": {"status": regular_status},
            "uniform_Aether_Legendre_block_inverse": {"status": regular_status},
            "strict_negative_source_energy_margin": {"status": regular_status},
            "weighted_full_constraint_operator_isomorphism": {"status": "blocked"},
            "nonlinear_Frechet_remainder": {"status": "blocked"},
            "completed_boundary_sign_persistence": {"status": "blocked"},
            "observational_data_seal": {"status": "pass"},
        }
        provenance_body = {
            "candidate_id": source["candidate_id"],
            "typed_action_ir_sha256": source["typed_action_ir_sha256"],
            "source_characteristic_record_sha256": source["content_sha256"],
            "source_exact_specialization_sha256": specialization["content_sha256"],
            "regular_inverse_certificate_sha256": (
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
            "source_characteristic_record_sha256": source["content_sha256"],
            "parameters": source["parameters"],
            "exact_specialization": specialization,
            "regular_ADM_inverse_margin_certificate": certificate,
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
    if regular_count != 3 or dict(blockers) != {CHARACTERISTIC_BLOCKER: 11, BLOCKER: 3}:
        raise ValueError("future Aether regular inverse partition changed")
    record_root = _sha(
        [
            [item["candidate_id"], item["typed_action_ir_sha256"], item["content_sha256"]]
            for item in records
        ]
    )
    provenance_body = {
        "source_characteristic_content_sha256": source_artifact["content_sha256"],
        "source_characteristic_record_registry_root_sha256": source_artifact[
            "candidate_record_registry_root_sha256"
        ],
        "candidate_record_registry_root_sha256": record_root,
        "data_eligibility": ELIGIBILITY,
    }
    body = {
        "schema_version": RESULT_SCHEMA,
        "campaign_id": config["campaign_id"],
        "immutable_config_sha256": _sha(config),
        "source_characteristic_binding": config["source_characteristic_artifact"],
        "candidate_count": 14,
        "decision_counts": {"blocked": 14},
        "first_blocker_counts": dict(sorted(blockers.items())),
        "formal_pass_count": 0,
        "candidate_rejection_authorized_count": 0,
        "constraint_satisfying_negative_total_energy_datum_count": 0,
        "forced_characteristic_candidate_count": 11,
        "regular_ADM_candidate_count": 3,
        "uniform_Aether_Legendre_block_inverse_pass_count": 3,
        "strict_negative_source_margin_pass_count": 3,
        "weighted_full_constraint_operator_isomorphism_pass_count": 0,
        "nonlinear_Frechet_remainder_bound_pass_count": 0,
        "completed_boundary_sign_persistence_count": 0,
        "kinetic_block_inverse_bound_counts": dict(sorted(inverse_bounds.items())),
        "strict_negative_source_margin_counts": dict(sorted(energy_margins.items())),
        "candidate_record_registry_root_sha256": record_root,
        "candidate_records": records,
        "bounded_regular_ADM_inverse_margin_gate_completed": True,
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
            "The eleven forced-characteristic candidates remain blocked on foliation or seed "
            "profile. For the other three, exact characteristic-free negative amplitudes give "
            "strictly positive uniform Aether Legendre margins and explicit kinetic-block inverse "
            "bounds while retaining strictly negative source-energy margins. The first missing "
            "premise is no longer the Aether kinetic block: it is a candidate-bound weighted "
            "isomorphism for the complete elliptic constraint operator, a nonlinear Frechet "
            "remainder bound, and persistence of the negative completed-boundary sign."
        ),
    }
    return {**body, "content_sha256": _sha(body)}


def publish_future_aether_regular_adm_inverse_margin_gate(
    config: dict[str, Any], root: str | Path, target: str | Path
) -> dict[str, Any]:
    artifact = build_future_aether_regular_adm_inverse_margin_gate(config, root)
    target = Path(target).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if _load(target) != artifact:
            raise ValueError("refusing to replace divergent regular-ADM artifact")
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
    artifact = publish_future_aether_regular_adm_inverse_margin_gate(
        _load(Path(arguments.config)), arguments.root, arguments.output
    )
    print(_canonical({key: value for key, value in artifact.items() if key != "candidate_records"}))


if __name__ == "__main__":
    main()
