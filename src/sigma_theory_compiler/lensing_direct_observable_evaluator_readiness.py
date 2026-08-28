"""Hash-bound readiness contract for direct-observable gravitational lensing evidence."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "sigma-lensing-direct-observable-evaluator-readiness-1.0"
CONFIG_SCHEMA = "sigma-lensing-direct-observable-evaluator-readiness-config-1.0"
PACKET_SCHEMA = "sigma-lensing-direct-observable-source-packet-1.0"
DECISION = "blocked_missing_authorized_real_lensing_source_packet"
CONFIG_REL = "configs/lensing_direct_observable_evaluator_readiness.json"
SOURCE_REL = "src/sigma_theory_compiler/lensing_direct_observable_evaluator_readiness.py"
TEST_REL = "tests/test_lensing_direct_observable_evaluator_readiness.py"
ARTIFACT_REL = "runs/engine/lensing-direct-observable-evaluator-readiness.json"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")

ALLOWED_DATA_CLASSES = ("calibrated_direct_observable", "raw_direct_observable")
ALLOWED_CHANNELS = (
    "calibrated_image_pixels",
    "directly_measured_time_delay",
    "image_parity_and_topology",
    "raw_detector_counts",
    "relative_arc_positions",
    "relative_multiple_image_positions",
)
REQUIRED_CALIBRATION_ROLES = (
    "background_provenance",
    "detector_or_instrument_calibration",
    "point_spread_function",
    "uncertainty_or_covariance",
)
FORBIDDEN_FIELDS = frozenset(
    {
        "abundance_matched_mass",
        "cosmological_distance",
        "dark_matter_label",
        "dark_matter_map",
        "distance_modulus",
        "environment_from_redshift",
        "formula_selection_target",
        "gr_derived_convergence",
        "gr_derived_mass_map",
        "halo_concentration",
        "halo_mass",
        "halo_profile",
        "halo_radius",
        "latent_gravitating_component",
        "lensing_only_gravity_parameter",
        "nfw_fit",
        "object_specific_gravity_parameter",
        "redshift_derived_distance",
        "supernova_standardization",
    }
)
_EMPTY_ELIGIBILITY = {
    "dark_matter_or_halo_inputs": False,
    "redshift_distance_inputs": False,
    "supernova_inference_inputs": False,
    "model_dependent_targets": False,
    "latent_component_inputs": False,
    "observational_data_opened": False,
}


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _file_sha(path: Path) -> str:
    # Git may materialize committed text with CRLF on Windows. These readiness
    # bindings seal canonical repository text, so line-ending conversion alone
    # must not look like a scientific-policy change.
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _sealed(value: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(value)
    body["content_sha256"] = _sha(body)
    return body


def _validate_sealed(value: Mapping[str, Any], label: str) -> None:
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    if value.get("content_sha256") != _sha(body):
        raise ValueError(f"{label} content hash mismatch")


def _load_bound(root: Path, binding: Mapping[str, Any]) -> dict[str, Any]:
    if set(binding) != {"path", "file_sha256", "content_sha256"}:
        raise ValueError("lensing readiness source binding keys mismatch")
    path = (root / binding["path"]).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError("lensing readiness source binding escaped root") from error
    if not path.is_file() or _file_sha(path) != binding["file_sha256"]:
        raise ValueError(f"bound lensing readiness source changed: {binding['path']}")
    value = json.loads(path.read_text(encoding="utf-8"))
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    actual = _sha(body) if "content_sha256" in value else _sha(value)
    if (
        actual != binding["content_sha256"]
        or value.get("content_sha256", binding["content_sha256"]) != binding["content_sha256"]
    ):
        raise ValueError(f"bound lensing readiness content changed: {binding['path']}")
    return value


def load_config(root: Path, path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if set(config) != {
        "schema_version",
        "campaign_id",
        "evidence_policy",
        "galaxy_protocol",
        "authorized_real_source_packets",
        "observational_authorization",
        "seals",
    }:
        raise ValueError("lensing readiness config keys mismatch")
    expected_seals = {
        "observational_data_opened": False,
        "source_packet_opened": False,
        "network_access": False,
        "secret_access": False,
        "runtime_access": False,
        "live_campaign_SQLite_access": False,
        "gpu_or_cuda_access": False,
        "external_process_signals": False,
        "leaderboard_or_rank_writes": False,
    }
    if (
        config["schema_version"] != CONFIG_SCHEMA
        or config["campaign_id"] != "lensing-direct-observable-evaluator-readiness-2026-08-12"
        or config["authorized_real_source_packets"] != []
        or config["observational_authorization"] is not False
        or config["seals"] != expected_seals
    ):
        raise ValueError("lensing readiness config contract mismatch")
    return config


def _validate_policy(policy: Mapping[str, Any], protocol: Mapping[str, Any]) -> None:
    if (
        policy.get("schema_version") != "sigma-observational-evidence-policy-1.0"
        or policy.get("status") != "frozen"
        or "angular positions, separations, shapes, and image topology"
        not in policy.get("allowed_primary_observables", [])
        or "time delays" not in policy.get("allowed_primary_observables", [])
        or policy.get("supernovae", {}).get("default_status") != "excluded"
        or policy.get("unobserved_components", {}).get("default_status")
        != "prohibited_as_truth_or_rescue"
        or "treating redshift as a distance"
        not in policy.get("redshift", {}).get("not_allowed_by_default", [])
        or protocol.get("schema_version") != "sigma-galaxy-observable-protocol-1.0"
        or protocol.get("status") != "sealed"
        or protocol.get("data_opened") is not False
        or protocol.get("independent_lensing_falsification", {}).get("formula_selection_use")
        != "prohibited"
        or "dark-matter map"
        not in protocol.get("independent_lensing_falsification", {}).get("forbidden_targets", [])
        or "redshift-derived distance" not in protocol.get("prohibited_truth_or_rescue", [])
        or "supernova distance modulus" not in protocol.get("prohibited_truth_or_rescue", [])
    ):
        raise ValueError("lensing readiness policy or protocol boundary changed")


def _walk_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            keys.add(str(key))
            keys.update(_walk_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.update(_walk_keys(child))
    return keys


def validate_direct_observable_packet(packet: Mapping[str, Any]) -> None:
    """Validate metadata-only packet structure; this function does not open a data path."""
    required = {
        "schema_version",
        "packet_id",
        "data_class",
        "observable_channel",
        "source_manifest_sha256",
        "payload_file_sha256",
        "calibration_bindings",
        "transformation",
        "formula_selection_use",
        "target_role",
        "eligibility",
        "content_sha256",
    }
    if set(packet) != required:
        raise ValueError("lensing direct-observable packet keys mismatch")
    forbidden = sorted(_walk_keys(packet) & FORBIDDEN_FIELDS)
    if forbidden:
        raise ValueError(f"forbidden lensing field present: {forbidden[0]}")
    body = {key: item for key, item in packet.items() if key != "content_sha256"}
    if (
        packet["schema_version"] != PACKET_SCHEMA
        or packet["data_class"] not in ALLOWED_DATA_CLASSES
        or packet["observable_channel"] not in ALLOWED_CHANNELS
        or packet["formula_selection_use"] is not False
        or packet["target_role"] != "post_freeze_independent_falsification"
        or packet["eligibility"] != _EMPTY_ELIGIBILITY
        or packet["content_sha256"] != _sha(body)
    ):
        raise ValueError("lensing direct-observable packet contract mismatch")
    for name in ("source_manifest_sha256", "payload_file_sha256"):
        if not isinstance(packet[name], str) or _SHA256.fullmatch(packet[name]) is None:
            raise ValueError(f"lensing packet hash invalid: {name}")
    packet_id = packet["packet_id"]
    if not isinstance(packet_id, str) or not packet_id.startswith("lensing-packet-"):
        raise ValueError("lensing packet identifier changed")
    calibration = packet["calibration_bindings"]
    if not isinstance(calibration, list) or [row.get("role") for row in calibration] != list(
        REQUIRED_CALIBRATION_ROLES
    ):
        raise ValueError("lensing packet calibration roles incomplete or unordered")
    for row in calibration:
        if (
            set(row) != {"role", "content_sha256"}
            or _SHA256.fullmatch(str(row["content_sha256"])) is None
        ):
            raise ValueError("lensing packet calibration binding changed")
    transformation = packet["transformation"]
    if set(transformation) != {
        "kind",
        "raw_input_root_sha256",
        "implementation_sha256",
        "uncertainty_propagation_sha256",
        "assumptions_sha256",
    }:
        raise ValueError("lensing packet transformation keys mismatch")
    if packet["data_class"] == "raw_direct_observable":
        if transformation["kind"] != "identity_raw_counts_no_derived_truth":
            raise ValueError("raw lensing packet must use the identity transformation")
    elif transformation["kind"] != "audited_raw_to_calibrated_direct_observable":
        raise ValueError("calibrated lensing packet lacks an audited direct transformation")
    for name, value in transformation.items():
        if name.endswith("_sha256") and _SHA256.fullmatch(str(value)) is None:
            raise ValueError(f"lensing packet transformation hash invalid: {name}")


def _synthetic_packet(data_class: str, channel: str) -> dict[str, Any]:
    transformation_kind = (
        "identity_raw_counts_no_derived_truth"
        if data_class == "raw_direct_observable"
        else "audited_raw_to_calibrated_direct_observable"
    )
    body = {
        "schema_version": PACKET_SCHEMA,
        "packet_id": f"lensing-packet-synthetic-{data_class}",
        "data_class": data_class,
        "observable_channel": channel,
        "source_manifest_sha256": _sha({"synthetic": "manifest", "class": data_class}),
        "payload_file_sha256": _sha({"synthetic": "payload", "class": data_class}),
        "calibration_bindings": [
            {"role": role, "content_sha256": _sha({"synthetic": role})}
            for role in REQUIRED_CALIBRATION_ROLES
        ],
        "transformation": {
            "kind": transformation_kind,
            "raw_input_root_sha256": _sha({"synthetic": "raw", "class": data_class}),
            "implementation_sha256": _sha({"synthetic": "implementation"}),
            "uncertainty_propagation_sha256": _sha({"synthetic": "uncertainty"}),
            "assumptions_sha256": _sha({"synthetic": "assumptions"}),
        },
        "formula_selection_use": False,
        "target_role": "post_freeze_independent_falsification",
        "eligibility": dict(_EMPTY_ELIGIBILITY),
    }
    return {**body, "content_sha256": _sha(body)}


def _positive_schema_controls() -> list[dict[str, Any]]:
    controls = [
        _synthetic_packet("raw_direct_observable", "raw_detector_counts"),
        _synthetic_packet("calibrated_direct_observable", "relative_arc_positions"),
    ]
    for control in controls:
        validate_direct_observable_packet(control)
    return [
        {
            "control_id": f"positive_{control['data_class']}",
            "decision": "schema_pass",
            "packet_content_sha256": control["content_sha256"],
            "synthetic_only": True,
            "scientific_pass": False,
            "observational_data_opened": False,
        }
        for control in controls
    ]


def build_readiness(config: Mapping[str, Any], root: str | Path) -> dict[str, Any]:
    root = Path(root).resolve()
    policy = _load_bound(root, config["evidence_policy"])
    protocol = _load_bound(root, config["galaxy_protocol"])
    _validate_policy(policy, protocol)
    controls = _positive_schema_controls()
    blocker_ledger = [
        {
            "gate_id": "authorized_real_source_packet",
            "decision": "blocked",
            "blocker": "no_authorized_real_lensing_source_packet_registered",
        },
        {
            "gate_id": "observational_opening_authorization",
            "decision": "blocked",
            "blocker": "observational_authorization_is_false",
        },
    ]
    body = {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": config["campaign_id"],
        "decision": DECISION,
        "source_bindings": {
            "evidence_policy": config["evidence_policy"],
            "galaxy_protocol": config["galaxy_protocol"],
        },
        "contract": {
            "allowed_data_classes": list(ALLOWED_DATA_CLASSES),
            "allowed_observable_channels": list(ALLOWED_CHANNELS),
            "required_calibration_roles": list(REQUIRED_CALIBRATION_ROLES),
            "forbidden_fields": sorted(FORBIDDEN_FIELDS),
            "formula_selection_use": False,
            "target_role": "post_freeze_independent_falsification",
        },
        "authorized_real_source_packet_bindings": [],
        "positive_schema_controls": controls,
        "blocker_ledger": blocker_ledger,
        "counts": {
            "allowed_data_classes": len(ALLOWED_DATA_CLASSES),
            "allowed_observable_channels": len(ALLOWED_CHANNELS),
            "required_calibration_roles": len(REQUIRED_CALIBRATION_ROLES),
            "forbidden_fields": len(FORBIDDEN_FIELDS),
            "positive_schema_controls": len(controls),
            "positive_schema_control_passes": len(controls),
            "authorized_real_source_packets": 0,
            "real_source_packets_opened": 0,
            "scientific_passes": 0,
            "scientific_rejects": 0,
            "scientific_blocks": 1,
        },
        "first_blocker": "no_authorized_real_lensing_source_packet_registered",
        "observational_authorization": False,
        "observational_data_opened": False,
        "source_packet_opened": False,
        "candidate_use_authorized": False,
        "scientific_pass_claimed": False,
        "complete_comparable_evidence": False,
        "data_eligibility": dict(_EMPTY_ELIGIBILITY),
        "bindings": {
            label: {"path": path, "file_sha256": _file_sha(root / path)}
            for label, path in (
                ("config", CONFIG_REL),
                ("source", SOURCE_REL),
                ("test", TEST_REL),
            )
        },
        "seals": config["seals"],
        "interpretation": (
            "Two synthetic packets pass structural schema controls only. No authorized real "
            "lensing source packet exists or opens, so this readiness contract records zero "
            "scientific passes and no candidate conclusion."
        ),
    }
    return _sealed(body)


def validate_readiness(
    value: Mapping[str, Any], config: Mapping[str, Any], root: str | Path
) -> None:
    _validate_sealed(value, "lensing direct-observable readiness")
    expected = build_readiness(config, root)
    if (
        dict(value) != expected
        or value["decision"] != DECISION
        or value["authorized_real_source_packet_bindings"] != []
        or value["counts"]["authorized_real_source_packets"] != 0
        or value["counts"]["real_source_packets_opened"] != 0
        or value["counts"]["scientific_passes"] != 0
        or value["counts"]["scientific_rejects"] != 0
        or value["counts"]["scientific_blocks"] != 1
        or value["observational_authorization"] is not False
        or value["observational_data_opened"] is not False
        or value["source_packet_opened"] is not False
        or value["candidate_use_authorized"] is not False
        or value["scientific_pass_claimed"] is not False
        or value["complete_comparable_evidence"] is not False
        or any(value["data_eligibility"].values())
        or any(value["seals"].values())
    ):
        raise ValueError("lensing direct-observable readiness contract mismatch")


def load_and_build_readiness(
    root: str | Path, config_path: str | Path = CONFIG_REL
) -> dict[str, Any]:
    root = Path(root).resolve()
    config = load_config(root, (root / config_path).resolve())
    return build_readiness(config, root)
