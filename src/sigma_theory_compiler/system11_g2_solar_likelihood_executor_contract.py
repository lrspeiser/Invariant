"""Action-bound likelihood executor contract for a future authorized System 11 opening."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

from sigma_theory_compiler.sigma_core import canonical_json_bytes, canonical_sha256

CONFIG_PATH = "configs/system11_g2_solar_likelihood_executor_contract.json"
SOURCE_PATH = "src/sigma_theory_compiler/system11_g2_solar_likelihood_executor_contract.py"
TEST_PATH = "tests/test_system11_g2_solar_likelihood_executor_contract.py"
OUTPUT_PATH = "runs/math/system11-g2-solar-likelihood-executor-contract/receipt.json"

CONFIG_SCHEMA = "sigma-system11-g2-solar-likelihood-executor-config-1.0"
PACKET_SCHEMA = "sigma-system11-authorized-calibrated-record-packet-1.0"
RESULT_SCHEMA = "sigma-system11-g2-solar-likelihood-result-1.0"
RECEIPT_SCHEMA = "sigma-system11-g2-solar-likelihood-executor-receipt-1.0"
CONTRACT_ID = "system11-g2-solar-likelihood-executor-001"

PRODUCTION_PACKET = "future_independently_authorized_calibrated_records"
SYNTHETIC_PACKET = "synthetic_known_answer_control"

_SHA256 = re.compile(r"[0-9a-f]{64}")
_CANDIDATES = {
    "G3A-2f8983c88f504150381064f2": {
        "action_sha256": "19f36a7c814ca11ace6de1270802a542872c35c27c7e64542eea672e16cbae88",
        "prediction_bundle_content_sha256": (
            "a060b82bd0c377e0b1a4186119efc0c3791208f99516fdd029192dcea5d68c2b"
        ),
        "source_contract_sha256": (
            "cd4097d96255d6f1ba15ea062d202021ed4324715869def8d0aa8cb2faa2fd95"
        ),
    },
    "G3A-58e59412e5fe77cd54caf863": {
        "action_sha256": "9457ba1ff99ecfdabc08200dda3ff15b8656b025d106fe2c2cd4abd77a01c3b5",
        "prediction_bundle_content_sha256": (
            "60b926ea6fd15041f71fd6366b3b9facb8fd0e27cf1882bdb486a5d2d7383938"
        ),
        "source_contract_sha256": (
            "84ca7bc3d02c30fd90f5731a01bfd84811cce9215c1f90a7db524d09c8e12825"
        ),
    },
}
_OBLIGATIONS = (
    "source_branch_domain_instantiation_sha256",
    "held_out_split_commitment_sha256",
    "selected_primary_record_roots_sha256",
    "observation_opening_authorization_sha256",
)


class System11LikelihoodExecutorError(ValueError):
    """A sealed executor authority or packet binding failed closed."""


def _load_json(path: Path, *, limit: int = 4_000_000) -> dict[str, Any]:
    try:
        if not path.is_file() or path.stat().st_size > limit:
            raise System11LikelihoodExecutorError(f"JSON missing or oversized: {path.name}")
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise System11LikelihoodExecutorError(f"cannot read JSON: {path.name}") from error
    if not isinstance(value, dict):
        raise System11LikelihoodExecutorError("JSON root must be an object")
    return value


def _semantic_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def _text_sha256(path: Path) -> str:
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _require_sha(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise System11LikelihoodExecutorError(f"invalid SHA-256: {label}")
    return value


def _resolve(root: Path, relative: object) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise System11LikelihoodExecutorError("bound path must be portable and relative")
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise System11LikelihoodExecutorError("bound path escapes root") from error
    return path


def _validate_config(config: Mapping[str, Any]) -> None:
    if set(config) != {
        "schema_version",
        "contract_id",
        "source_bindings",
        "candidates",
        "schema_bindings",
        "implementation_bindings",
        "opening_obligations",
        "score_contract",
    }:
        raise System11LikelihoodExecutorError("config keys changed")
    if config.get("schema_version") != CONFIG_SCHEMA or config.get("contract_id") != CONTRACT_ID:
        raise System11LikelihoodExecutorError("config identity changed")
    expected_candidates = [
        {"candidate_id": candidate_id, **bindings} for candidate_id, bindings in _CANDIDATES.items()
    ]
    if config.get("candidates") != expected_candidates:
        raise System11LikelihoodExecutorError("candidate/action freeze changed")
    if config.get("opening_obligations") != list(_OBLIGATIONS):
        raise System11LikelihoodExecutorError("opening obligations changed")
    schemas = config.get("schema_bindings")
    if (
        not isinstance(schemas, Mapping)
        or schemas.get("authorized_calibrated_record_packet") != PACKET_SCHEMA
    ):
        raise System11LikelihoodExecutorError("record packet schema changed")
    score = config.get("score_contract")
    if not isinstance(score, Mapping) or score != {
        "name": "exact_relative_gaussian_log_likelihood_without_candidate_independent_normalization",
        "accepted_channels": [
            "two_way_round_trip_light_time",
            "coherent_carrier_frequency_or_phase_ratio",
            "relative_angular_separation",
        ],
        "accepted_quantity_class": "calibrated",
        "accept_if_chi_square_lte": "2/1",
        "arithmetic": "exact_rational",
        "record_order": "lexicographic_record_id",
    }:
        raise System11LikelihoodExecutorError("score contract changed")
    bindings = config.get("source_bindings")
    if not isinstance(bindings, Mapping) or len(bindings) != 8:
        raise System11LikelihoodExecutorError("source-binding inventory changed")
    for role, descriptor in bindings.items():
        if not isinstance(descriptor, Mapping) or set(descriptor) != {
            "path",
            "semantic_sha256",
        }:
            raise System11LikelihoodExecutorError(f"source binding changed: {role}")
        _require_sha(descriptor.get("semantic_sha256"), f"source binding {role}")


def _load_authority(root: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    config = _load_json(root / CONFIG_PATH)
    _validate_config(config)
    values: dict[str, dict[str, Any]] = {}
    for role, descriptor in sorted(config["source_bindings"].items()):
        value = _load_json(_resolve(root, descriptor["path"]))
        if _semantic_sha256(value) != descriptor["semantic_sha256"]:
            raise System11LikelihoodExecutorError(f"semantic source changed: {role}")
        values[role] = value
    _validate_bound_authority(config, values)
    return config, values


def _validate_bound_authority(
    config: Mapping[str, Any], values: Mapping[str, Mapping[str, Any]]
) -> None:
    authorization = values["authorization_receipt"]
    one_shot = values["one_shot_receipt"]
    parser = values["parser_readiness"]
    calibration = values["calibration_readiness"]
    if (
        authorization.get("decision") != "block"
        or authorization.get("counts", {}).get("missing_external_opening_obligations") != 8
        or authorization.get("counts", {}).get("real_data_evaluations") != 0
        or authorization.get("data_boundary", {}).get("primary_record_payloads_read") is not False
        or one_shot.get("decision") != "block"
        or one_shot.get("counts", {}).get("missing_opening_obligations") != 8
        or one_shot.get("counts", {}).get("real_data_evaluations") != 0
        or one_shot.get("data_boundary", {}).get("observational_data_opened") is not False
    ):
        raise System11LikelihoodExecutorError("external opening boundary changed")
    implementation = config["implementation_bindings"]
    parser_fields = parser.get("filled_registration_fields", {})
    calibration_fields = calibration.get("filled_registration_fields", {})
    if (
        parser.get("schema_version") != config["schema_bindings"]["parser_readiness"]
        or parser.get("status") != "parser_ready_labels_selected_primary_records_sealed"
        or parser.get("observational_authorization") is not False
        or parser.get("metadata_selection", {}).get("primary_record_access_count") != 0
        or parser_fields.get("verified_ATDF_TDF_parser_sha256")
        != implementation["verified_ATDF_TDF_parser_sha256"]
        or parser_fields.get("verified_RSR_parser_sha256")
        != implementation["verified_RSR_parser_sha256"]
    ):
        raise System11LikelihoodExecutorError("parser authority changed")
    if (
        calibration.get("schema_version") != config["schema_bindings"]["calibration_readiness"]
        or calibration.get("status") != "calibration_implementation_ready_primary_records_sealed"
        or calibration.get("observational_authorization") is not False
        or calibration.get("primary_record_access_count") != 0
        or calibration_fields.get(
            "raw_to_calibrated_transform_and_covariance_implementation_sha256"
        )
        != implementation["raw_to_calibrated_transform_and_covariance_implementation_sha256"]
        or calibration.get("transformation_contract_sha256")
        != implementation["transformation_contract_sha256"]
    ):
        raise System11LikelihoodExecutorError("calibration authority changed")
    for role, candidate_id in (
        ("bundle_G3A_2f8983", "G3A-2f8983c88f504150381064f2"),
        ("bundle_G3A_58e594", "G3A-58e59412e5fe77cd54caf863"),
    ):
        bundle = values[role]
        expected = _CANDIDATES[candidate_id]
        if (
            bundle.get("candidate_id") != candidate_id
            or bundle.get("action_sha256") != expected["action_sha256"]
            or bundle.get("content_sha256") != expected["prediction_bundle_content_sha256"]
            or bundle.get("descriptor", {}).get("source_contract_sha256")
            != expected["source_contract_sha256"]
            or bundle.get("held_out_targets_opened") is not False
        ):
            raise System11LikelihoodExecutorError(f"action-bound bundle changed: {candidate_id}")


def _descriptor(value: object, label: str) -> tuple[dict[str, Any], str]:
    if not isinstance(value, Mapping) or set(value) != {"document", "semantic_sha256"}:
        raise System11LikelihoodExecutorError(f"descriptor changed: {label}")
    document = value.get("document")
    if not isinstance(document, dict):
        raise System11LikelihoodExecutorError(f"descriptor document changed: {label}")
    expected = _require_sha(value.get("semantic_sha256"), label)
    actual = _semantic_sha256(document)
    if actual != expected:
        raise System11LikelihoodExecutorError(f"descriptor hash changed: {label}")
    return document, actual


def _fraction(value: object, label: str) -> Fraction:
    if not isinstance(value, Mapping) or set(value) != {"numerator", "denominator"}:
        raise System11LikelihoodExecutorError(f"rational changed: {label}")
    numerator = value.get("numerator")
    denominator = value.get("denominator")
    if (
        isinstance(numerator, bool)
        or isinstance(denominator, bool)
        or not isinstance(numerator, int)
        or not isinstance(denominator, int)
        or denominator <= 0
    ):
        raise System11LikelihoodExecutorError(f"invalid rational: {label}")
    return Fraction(numerator, denominator)


def _fraction_text(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def execute_packet(
    root: Path, packet: Mapping[str, Any], *, allow_synthetic: bool = False
) -> dict[str, Any]:
    """Validate every authority edge, then score calibrated records exactly."""
    config, _ = _load_authority(root.resolve())
    expected_keys = {
        "schema_version",
        "packet_class",
        "candidate_id",
        "action_sha256",
        "prediction_bundle_content_sha256",
        "source_domain",
        "split_commitment",
        "primary_record_roots",
        "independent_authorization",
        "parser_contract",
        "calibration_contract",
        "calibrated_records",
    }
    if set(packet) != expected_keys or packet.get("schema_version") != PACKET_SCHEMA:
        raise System11LikelihoodExecutorError("authorized record packet schema changed")
    packet_class = packet.get("packet_class")
    if packet_class not in {PRODUCTION_PACKET, SYNTHETIC_PACKET}:
        raise System11LikelihoodExecutorError("record packet class changed")
    if packet_class == SYNTHETIC_PACKET and not allow_synthetic:
        raise System11LikelihoodExecutorError("synthetic control is not a production record packet")
    candidate_id = packet.get("candidate_id")
    candidate = _CANDIDATES.get(str(candidate_id))
    if candidate is None or packet.get("action_sha256") != candidate["action_sha256"]:
        raise System11LikelihoodExecutorError("candidate/action binding changed")
    if (
        packet.get("prediction_bundle_content_sha256")
        != candidate["prediction_bundle_content_sha256"]
    ):
        raise System11LikelihoodExecutorError("action-bound prediction root changed")

    domain, domain_hash = _descriptor(packet["source_domain"], "source domain")
    split, split_hash = _descriptor(packet["split_commitment"], "split commitment")
    roots, roots_hash = _descriptor(packet["primary_record_roots"], "primary record roots")
    authorization, authorization_hash = _descriptor(
        packet["independent_authorization"], "independent authorization"
    )
    schemas = config["schema_bindings"]
    if (
        set(domain)
        != {
            "schema_version",
            "candidate_id",
            "action_sha256",
            "source_contract_sha256",
            "domain_id",
            "domain_class",
            "status",
            "target_values_accessed",
        }
        or domain.get("schema_version") != schemas["source_domain"]
        or domain.get("candidate_id") != candidate_id
        or domain.get("action_sha256") != candidate["action_sha256"]
        or domain.get("source_contract_sha256") != candidate["source_contract_sha256"]
        or domain.get("domain_class") != "registered_static_connected_regular_horizonless_source"
        or domain.get("status") != "instantiated_before_target_access"
        or domain.get("target_values_accessed") is not False
    ):
        raise System11LikelihoodExecutorError("source domain is not authorized")
    if (
        set(split)
        != {
            "schema_version",
            "split_id",
            "status",
            "target_values_accessed",
            "candidate_ids",
            "heldout_role",
        }
        or split.get("schema_version") != schemas["split_commitment"]
        or split.get("status") != "committed_before_target_access"
        or split.get("target_values_accessed") is not False
        or split.get("candidate_ids") != list(_CANDIDATES)
        or split.get("heldout_role") != "untouched_target_blind_test"
    ):
        raise System11LikelihoodExecutorError("held-out split is not authorized")
    record_roots = roots.get("record_roots_sha256")
    if (
        set(roots)
        != {
            "schema_version",
            "status",
            "target_values_accessed",
            "primary_record_payloads_opened_before_authorization",
            "record_roots_sha256",
        }
        or roots.get("schema_version") != schemas["primary_record_roots"]
        or roots.get("status") != "selected_and_hash_bound_before_target_access"
        or roots.get("target_values_accessed") is not False
        or roots.get("primary_record_payloads_opened_before_authorization") is not False
        or not isinstance(record_roots, list)
        or not record_roots
        or len(set(record_roots)) != len(record_roots)
        or any(_SHA256.fullmatch(str(item)) is None for item in record_roots)
    ):
        raise System11LikelihoodExecutorError("primary record roots are not authorized")

    parser_contract = packet.get("parser_contract")
    implementation = config["implementation_bindings"]
    if not isinstance(parser_contract, Mapping) or parser_contract != {
        "schema_version": schemas["parser_readiness"],
        "artifact_semantic_sha256": config["source_bindings"]["parser_readiness"][
            "semantic_sha256"
        ],
        "verified_ATDF_TDF_parser_sha256": implementation["verified_ATDF_TDF_parser_sha256"],
        "verified_RSR_parser_sha256": implementation["verified_RSR_parser_sha256"],
    }:
        raise System11LikelihoodExecutorError("parser contract changed")
    calibration_contract = packet.get("calibration_contract")
    if not isinstance(calibration_contract, Mapping) or calibration_contract != {
        "schema_version": schemas["calibration_readiness"],
        "artifact_semantic_sha256": config["source_bindings"]["calibration_readiness"][
            "semantic_sha256"
        ],
        "raw_to_calibrated_transform_and_covariance_implementation_sha256": implementation[
            "raw_to_calibrated_transform_and_covariance_implementation_sha256"
        ],
        "transformation_contract_sha256": implementation["transformation_contract_sha256"],
    }:
        raise System11LikelihoodExecutorError("calibration contract changed")

    records = packet.get("calibrated_records")
    if (
        not isinstance(records, list)
        or not records
        or any(not isinstance(record, Mapping) for record in records)
    ):
        raise System11LikelihoodExecutorError("calibrated record inventory is empty")
    calibrated_records_hash = _semantic_sha256({"records": records})
    prediction_records_hash = _semantic_sha256(
        {
            "predictions": sorted(
                [
                    {
                        "record_id": record.get("record_id"),
                        "predicted": record.get("predicted"),
                    }
                    for record in records
                ],
                key=lambda item: str(item["record_id"]),
            )
        }
    )
    required_authorization = {
        "schema_version": schemas["independent_authorization"],
        "status": (
            "synthetic_control_only"
            if packet_class == SYNTHETIC_PACKET
            else "independently_authorized_future_opening"
        ),
        "independent_of_candidate_generation": True,
        "candidate_id": candidate_id,
        "action_sha256": candidate["action_sha256"],
        "prediction_bundle_content_sha256": candidate["prediction_bundle_content_sha256"],
        "source_branch_domain_instantiation_sha256": domain_hash,
        "held_out_split_commitment_sha256": split_hash,
        "selected_primary_record_roots_sha256": roots_hash,
        "parser_contract_semantic_sha256": _semantic_sha256(parser_contract),
        "calibration_contract_semantic_sha256": _semantic_sha256(calibration_contract),
        "calibrated_records_semantic_sha256": calibrated_records_hash,
        "action_bound_predictions_semantic_sha256": prediction_records_hash,
        "atomic_open_batches": 1,
        "candidate_evaluations": 2,
        "refits_after_open": 0,
        "promotion_actions": 0,
    }
    if authorization != required_authorization:
        raise System11LikelihoodExecutorError("independent authorization binding changed")

    accepted_channels = set(config["score_contract"]["accepted_channels"])
    seen: set[str] = set()
    contributions: list[dict[str, str]] = []
    chi_square = Fraction(0)
    for record in sorted(records, key=lambda item: str(item.get("record_id", ""))):
        if not isinstance(record, Mapping) or set(record) != {
            "record_id",
            "channel",
            "quantity_class",
            "split_role",
            "primary_record_root_sha256",
            "observed",
            "predicted",
            "variance",
        }:
            raise System11LikelihoodExecutorError("calibrated record schema changed")
        record_id = record.get("record_id")
        if not isinstance(record_id, str) or not record_id or record_id in seen:
            raise System11LikelihoodExecutorError("record identity changed")
        seen.add(record_id)
        if (
            record.get("channel") not in accepted_channels
            or record.get("quantity_class") != "calibrated"
            or record.get("split_role") != "untouched_target_blind_test"
            or record.get("primary_record_root_sha256") not in record_roots
        ):
            raise System11LikelihoodExecutorError("record channel, split, or root changed")
        observed = _fraction(record.get("observed"), f"{record_id} observed")
        predicted = _fraction(record.get("predicted"), f"{record_id} predicted")
        variance = _fraction(record.get("variance"), f"{record_id} variance")
        if variance <= 0:
            raise System11LikelihoodExecutorError("record variance must be positive")
        contribution = (observed - predicted) ** 2 / variance
        chi_square += contribution
        contributions.append(
            {"record_id": record_id, "chi_square_contribution": _fraction_text(contribution)}
        )
    threshold = Fraction(config["score_contract"]["accept_if_chi_square_lte"])
    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA,
        "contract_id": CONTRACT_ID,
        "packet_class": packet_class,
        "candidate_id": candidate_id,
        "action_sha256": candidate["action_sha256"],
        "prediction_bundle_content_sha256": candidate["prediction_bundle_content_sha256"],
        "decision": "pass" if chi_square <= threshold else "reject",
        "score": {
            "chi_square": _fraction_text(chi_square),
            "relative_log_likelihood": _fraction_text(-chi_square / 2),
            "accept_if_chi_square_lte": _fraction_text(threshold),
            "candidate_independent_normalization_included": False,
            "arithmetic": "exact_rational",
        },
        "record_contributions": contributions,
        "record_count": len(records),
        "authority_roots": {
            "source_branch_domain_instantiation_sha256": domain_hash,
            "held_out_split_commitment_sha256": split_hash,
            "selected_primary_record_roots_sha256": roots_hash,
            "observation_opening_authorization_sha256": authorization_hash,
            "calibrated_records_semantic_sha256": calibrated_records_hash,
            "action_bound_predictions_semantic_sha256": prediction_records_hash,
        },
        "claims": {
            "synthetic_known_answer_only": packet_class == SYNTHETIC_PACKET,
            "real_observation_opened_by_committed_receipt": False,
            "promotion_authorized": False,
            "truth_established": False,
        },
    }
    result["content_sha256"] = canonical_sha256(result)
    return result


def _described(document: dict[str, Any]) -> dict[str, Any]:
    return {"document": document, "semantic_sha256": _semantic_sha256(document)}


def build_synthetic_packet(root: Path, candidate_id: str, *, perturbed: bool) -> dict[str, Any]:
    """Build an in-memory known-answer packet; it is never production-admissible."""
    config, _ = _load_authority(root.resolve())
    candidate = _CANDIDATES[candidate_id]
    domain = {
        "schema_version": config["schema_bindings"]["source_domain"],
        "candidate_id": candidate_id,
        "action_sha256": candidate["action_sha256"],
        "source_contract_sha256": candidate["source_contract_sha256"],
        "domain_id": f"synthetic-known-answer-domain-{candidate_id}",
        "domain_class": "registered_static_connected_regular_horizonless_source",
        "status": "instantiated_before_target_access",
        "target_values_accessed": False,
    }
    split = {
        "schema_version": config["schema_bindings"]["split_commitment"],
        "split_id": "synthetic-known-answer-heldout-split",
        "status": "committed_before_target_access",
        "target_values_accessed": False,
        "candidate_ids": list(_CANDIDATES),
        "heldout_role": "untouched_target_blind_test",
    }
    record_root = ("1" if candidate_id.endswith("863") else "2") * 64
    roots = {
        "schema_version": config["schema_bindings"]["primary_record_roots"],
        "status": "selected_and_hash_bound_before_target_access",
        "target_values_accessed": False,
        "primary_record_payloads_opened_before_authorization": False,
        "record_roots_sha256": [record_root],
    }
    parser_contract = {
        "schema_version": config["schema_bindings"]["parser_readiness"],
        "artifact_semantic_sha256": config["source_bindings"]["parser_readiness"][
            "semantic_sha256"
        ],
        "verified_ATDF_TDF_parser_sha256": config["implementation_bindings"][
            "verified_ATDF_TDF_parser_sha256"
        ],
        "verified_RSR_parser_sha256": config["implementation_bindings"][
            "verified_RSR_parser_sha256"
        ],
    }
    calibration_contract = {
        "schema_version": config["schema_bindings"]["calibration_readiness"],
        "artifact_semantic_sha256": config["source_bindings"]["calibration_readiness"][
            "semantic_sha256"
        ],
        "raw_to_calibrated_transform_and_covariance_implementation_sha256": config[
            "implementation_bindings"
        ]["raw_to_calibrated_transform_and_covariance_implementation_sha256"],
        "transformation_contract_sha256": config["implementation_bindings"][
            "transformation_contract_sha256"
        ],
    }
    domain_descriptor = _described(domain)
    split_descriptor = _described(split)
    roots_descriptor = _described(roots)
    observed = [10, 20, 30] if not perturbed else [11, 22, 33]
    predicted = [10, 20, 30]
    variances = [1, 4, 9]
    channels = config["score_contract"]["accepted_channels"]
    records = [
        {
            "record_id": f"synthetic-{index + 1}",
            "channel": channels[index],
            "quantity_class": "calibrated",
            "split_role": "untouched_target_blind_test",
            "primary_record_root_sha256": record_root,
            "observed": {"numerator": observed[index], "denominator": 1},
            "predicted": {"numerator": predicted[index], "denominator": 1},
            "variance": {"numerator": variances[index], "denominator": 1},
        }
        for index in range(3)
    ]
    authorization = {
        "schema_version": config["schema_bindings"]["independent_authorization"],
        "status": "synthetic_control_only",
        "independent_of_candidate_generation": True,
        "candidate_id": candidate_id,
        "action_sha256": candidate["action_sha256"],
        "prediction_bundle_content_sha256": candidate["prediction_bundle_content_sha256"],
        "source_branch_domain_instantiation_sha256": domain_descriptor["semantic_sha256"],
        "held_out_split_commitment_sha256": split_descriptor["semantic_sha256"],
        "selected_primary_record_roots_sha256": roots_descriptor["semantic_sha256"],
        "parser_contract_semantic_sha256": _semantic_sha256(parser_contract),
        "calibration_contract_semantic_sha256": _semantic_sha256(calibration_contract),
        "calibrated_records_semantic_sha256": _semantic_sha256({"records": records}),
        "action_bound_predictions_semantic_sha256": _semantic_sha256(
            {
                "predictions": [
                    {
                        "record_id": record["record_id"],
                        "predicted": record["predicted"],
                    }
                    for record in records
                ]
            }
        ),
        "atomic_open_batches": 1,
        "candidate_evaluations": 2,
        "refits_after_open": 0,
        "promotion_actions": 0,
    }
    return {
        "schema_version": PACKET_SCHEMA,
        "packet_class": SYNTHETIC_PACKET,
        "candidate_id": candidate_id,
        "action_sha256": candidate["action_sha256"],
        "prediction_bundle_content_sha256": candidate["prediction_bundle_content_sha256"],
        "source_domain": domain_descriptor,
        "split_commitment": split_descriptor,
        "primary_record_roots": roots_descriptor,
        "independent_authorization": _described(authorization),
        "parser_contract": parser_contract,
        "calibration_contract": calibration_contract,
        "calibrated_records": records,
    }


def build_receipt(root: Path) -> dict[str, Any]:
    """Register the executor while preserving the unopened eight-obligation block."""
    root = root.resolve()
    config, _ = _load_authority(root)
    controls: list[dict[str, Any]] = []
    for candidate_id in _CANDIDATES:
        for perturbed in (False, True):
            result = execute_packet(
                root,
                build_synthetic_packet(root, candidate_id, perturbed=perturbed),
                allow_synthetic=True,
            )
            controls.append(
                {
                    "candidate_id": candidate_id,
                    "control": "perturbed_reject" if perturbed else "exact_match_pass",
                    "expected_decision": "reject" if perturbed else "pass",
                    "expected_chi_square": "3/1" if perturbed else "0/1",
                    "observed_decision": result["decision"],
                    "observed_chi_square": result["score"]["chi_square"],
                    "result_content_sha256": result["content_sha256"],
                }
            )
    if any(
        control["expected_decision"] != control["observed_decision"]
        or control["expected_chi_square"] != control["observed_chi_square"]
        for control in controls
    ):
        raise System11LikelihoodExecutorError("synthetic known-answer control changed")
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "contract_id": CONTRACT_ID,
        "decision": "block",
        "executor_status": "registered_and_synthetic_verified_real_records_sealed",
        "first_blocker": "independent_observation_opening_authorization_absent",
        "candidate_actions": [
            {"candidate_id": candidate_id, **bindings}
            for candidate_id, bindings in _CANDIDATES.items()
        ],
        "bound_schemas": config["schema_bindings"],
        "implementation_bindings": config["implementation_bindings"],
        "score_contract": config["score_contract"],
        "synthetic_known_answer_controls": controls,
        "missing_obligations_by_candidate": {
            candidate_id: list(_OBLIGATIONS) for candidate_id in _CANDIDATES
        },
        "counts": {
            "candidates": 2,
            "synthetic_controls": 4,
            "synthetic_pass_controls": 2,
            "synthetic_reject_controls": 2,
            "unique_missing_external_opening_fields": 4,
            "missing_external_opening_obligations": 8,
            "primary_record_accesses": 0,
            "real_data_evaluations": 0,
        },
        "data_boundary": {
            "target_data_read": False,
            "primary_record_payloads_read": False,
            "observational_data_opened": False,
            "synthetic_known_answer_records_scored": 12,
            "real_data_evaluation_count": 0,
        },
        "production_contract": {
            "accepted_packet_class": PRODUCTION_PACKET,
            "accepted_packet_schema": PACKET_SCHEMA,
            "requires_all_four_opening_roots_per_candidate": True,
            "wrong_action_domain_split_or_root_fails_closed": True,
            "refits_after_open": 0,
            "promotion_actions": 0,
            "command": (
                "python -m sigma_theory_compiler.system11_g2_solar_likelihood_executor_contract "
                "--root . --packet <authorized-calibrated-record-packet.json> "
                "--output <likelihood-result.json>"
            ),
        },
        "claims": {
            "observation_opened": False,
            "observational_result_exists": False,
            "candidate_supported_by_data": False,
            "candidate_rejected_by_data": False,
            "promotion_authorized": False,
            "truth_established": False,
        },
        "source_bindings": config["source_bindings"],
        "lane_bindings": {
            "config": {
                "path": CONFIG_PATH,
                "semantic_sha256": _semantic_sha256(config),
            },
            "source": {
                "path": SOURCE_PATH,
                "normalized_text_sha256": _text_sha256(root / SOURCE_PATH),
            },
            "test": {"path": TEST_PATH, "normalized_text_sha256": _text_sha256(root / TEST_PATH)},
        },
    }
    receipt["content_sha256"] = canonical_sha256(receipt)
    return receipt


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--packet", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.packet is None:
        if args.output is not None:
            _write_json(args.output, build_receipt(root))
        else:
            _write_json(root / OUTPUT_PATH, build_receipt(root))
        return 0
    if args.output is None:
        raise System11LikelihoodExecutorError("packet execution requires --output")
    packet = _load_json(args.packet.resolve())
    _write_json(args.output.resolve(), execute_packet(root, packet))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
