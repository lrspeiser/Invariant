"""Sealed readiness contract for cluster direct-observable evidence."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "sigma-cluster-direct-observable-evaluator-readiness-1.0"
CONFIG_SCHEMA = "sigma-cluster-direct-observable-evaluator-readiness-config-1.0"
PACKET_SCHEMA = "sigma-cluster-direct-observable-source-packet-1.0"
DECISION = "blocked_missing_authorized_real_cluster_source_packet"
CONFIG_REL = "configs/cluster_direct_observable_evaluator_readiness.json"
SOURCE_REL = "src/sigma_theory_compiler/cluster_direct_observable_evaluator_readiness.py"
TEST_REL = "tests/test_cluster_direct_observable_evaluator_readiness.py"
ARTIFACT_REL = "runs/engine/cluster-direct-observable-evaluator-readiness.json"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")

ALLOWED_DATA_CLASSES = ("calibrated_direct_observable", "raw_direct_observable")
ALLOWED_CHANNELS = (
    "angular_positions_separations_and_shapes",
    "calibrated_member_light_or_spectra",
    "calibrated_sunyaev_zeldovich_intensity_or_temperature_decrement",
    "calibrated_xray_surface_brightness_or_spectrum",
    "directly_measured_spectral_wavelength_ratios",
    "raw_optical_or_infrared_detector_counts",
    "raw_sunyaev_zeldovich_detector_counts",
    "raw_xray_detector_counts",
)
REQUIRED_CALIBRATION_ROLES = (
    "background_provenance",
    "detector_or_instrument_calibration",
    "point_spread_or_beam_response",
    "selection_or_mask_provenance",
    "spectral_or_bandpass_response",
)
REQUIRED_COVARIANCE_ROLES = (
    "calibration_covariance",
    "measurement_uncertainty_or_covariance",
)
FORBIDDEN_FIELDS = frozenset(
    {
        "abundance_matched_mass",
        "cluster_halo_concentration",
        "cluster_halo_mass",
        "cluster_halo_profile",
        "cluster_halo_radius",
        "cosmological_distance",
        "dark_matter_fraction",
        "dark_matter_label",
        "dark_matter_map",
        "distance_modulus",
        "environment_from_redshift",
        "formula_selection_target",
        "gr_derived_dynamical_mass",
        "gr_derived_lensing_mass",
        "hydrostatic_equilibrium_mass",
        "hydrostatic_mass",
        "latent_gravitating_component",
        "latent_nonthermal_pressure",
        "mass_bias_calibration_target",
        "nfw_fit",
        "object_specific_gravity_parameter",
        "redshift_derived_distance",
        "supernova_standardization",
        "total_mass_profile",
    }
)
_EMPTY_ELIGIBILITY = {
    "dark_matter_or_halo_inputs": False,
    "hydrostatic_or_gr_derived_mass_truth": False,
    "redshift_distance_inputs": False,
    "supernova_inference_inputs": False,
    "model_dependent_targets": False,
    "latent_component_inputs": False,
    "formula_selection_inputs": False,
    "observational_data_opened": False,
}


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sealed(value: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(value)
    body["content_sha256"] = _sha(body)
    return body


def _validate_sealed(value: Mapping[str, Any], label: str) -> None:
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    if value.get("content_sha256") != _sha(body):
        raise ValueError(f"{label} content hash mismatch")


def _binding_core(binding: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "path": binding["path"],
        "file_sha256": binding["file_sha256"],
        "content_sha256": binding["content_sha256"],
    }


def _load_bound(root: Path, binding: Mapping[str, Any]) -> dict[str, Any]:
    core = _binding_core(binding)
    path = (root / core["path"]).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError("cluster readiness source binding escaped root") from error
    if not path.is_file() or _file_sha(path) != core["file_sha256"]:
        raise ValueError(f"bound cluster readiness source changed: {core['path']}")
    value = json.loads(path.read_text(encoding="utf-8"))
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    actual = _sha(body) if "content_sha256" in value else _sha(value)
    if (
        actual != core["content_sha256"]
        or value.get("content_sha256", core["content_sha256"]) != core["content_sha256"]
    ):
        raise ValueError(f"bound cluster readiness content changed: {core['path']}")
    return value


def load_config(root: Path, path: Path) -> dict[str, Any]:
    del root
    config = json.loads(path.read_text(encoding="utf-8"))
    if set(config) != {
        "schema_version",
        "campaign_id",
        "evidence_policy",
        "shared_galaxy_policy",
        "authorized_real_source_packets",
        "observational_authorization",
        "seals",
    }:
        raise ValueError("cluster readiness config keys mismatch")
    if set(config["evidence_policy"]) != {
        "path",
        "file_sha256",
        "content_sha256",
    } or set(config["shared_galaxy_policy"]) != {
        "path",
        "file_sha256",
        "content_sha256",
        "applicability",
    }:
        raise ValueError("cluster readiness source binding keys mismatch")
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
        or config["campaign_id"] != "cluster-direct-observable-evaluator-readiness-2026-08-12"
        or config["shared_galaxy_policy"]["applicability"]
        != "shared observational exclusions only; no galaxy split or lensing target is imported"
        or config["authorized_real_source_packets"] != []
        or config["observational_authorization"] is not False
        or config["seals"] != expected_seals
    ):
        raise ValueError("cluster readiness config contract mismatch")
    return config


def _validate_policy(policy: Mapping[str, Any], galaxy: Mapping[str, Any]) -> None:
    if (
        policy.get("schema_version") != "sigma-observational-evidence-policy-1.0"
        or policy.get("status") != "frozen"
        or "raw detector counts with calibration and background provenance"
        not in policy.get("allowed_primary_observables", [])
        or "calibrated spectra and wavelength ratios"
        not in policy.get("allowed_primary_observables", [])
        or "measured baryonic light and gas tracers with explicit uncertainty"
        not in policy.get("allowed_primary_observables", [])
        or policy.get("supernovae", {}).get("default_status") != "excluded"
        or policy.get("unobserved_components", {}).get("default_status")
        != "prohibited_as_truth_or_rescue"
        or "treating redshift as a distance"
        not in policy.get("redshift", {}).get("not_allowed_by_default", [])
        or galaxy.get("schema_version") != "sigma-galaxy-observable-protocol-1.0"
        or galaxy.get("status") != "sealed"
        or galaxy.get("data_opened") is not False
        or "dark-matter halo mass, concentration, radius, profile, or abundance-matching label"
        not in galaxy.get("discovery_channel", {}).get("forbidden_formula_inputs", [])
        or "per-galaxy acceleration scale or gravitational coupling"
        not in galaxy.get("discovery_channel", {}).get("forbidden_formula_inputs", [])
        or "redshift-derived distance" not in galaxy.get("prohibited_truth_or_rescue", [])
        or "supernova distance modulus" not in galaxy.get("prohibited_truth_or_rescue", [])
    ):
        raise ValueError("cluster readiness policy boundary changed")


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


def _validate_role_bindings(bindings: Any, required_roles: tuple[str, ...], label: str) -> None:
    if not isinstance(bindings, list) or [row.get("role") for row in bindings] != list(
        required_roles
    ):
        raise ValueError(f"cluster packet {label} roles incomplete or unordered")
    for row in bindings:
        if (
            set(row) != {"role", "content_sha256"}
            or _SHA256.fullmatch(str(row["content_sha256"])) is None
        ):
            raise ValueError(f"cluster packet {label} binding changed")


def validate_direct_observable_packet(packet: Mapping[str, Any]) -> None:
    """Validate packet metadata only; never open its payload or manifest path."""
    required = {
        "schema_version",
        "packet_id",
        "data_class",
        "observable_channel",
        "source_manifest_sha256",
        "payload_file_sha256",
        "calibration_bindings",
        "covariance_bindings",
        "transformation",
        "formula_selection_use",
        "target_role",
        "eligibility",
        "content_sha256",
    }
    if set(packet) != required:
        raise ValueError("cluster direct-observable packet keys mismatch")
    forbidden = sorted(_walk_keys(packet) & FORBIDDEN_FIELDS)
    if forbidden:
        raise ValueError(f"forbidden cluster field present: {forbidden[0]}")
    body = {key: item for key, item in packet.items() if key != "content_sha256"}
    if (
        packet["schema_version"] != PACKET_SCHEMA
        or packet["data_class"] not in ALLOWED_DATA_CLASSES
        or packet["observable_channel"] not in ALLOWED_CHANNELS
        or packet["formula_selection_use"] is not False
        or packet["target_role"] != "post_freeze_independent_cluster_falsification"
        or packet["eligibility"] != _EMPTY_ELIGIBILITY
        or packet["content_sha256"] != _sha(body)
    ):
        raise ValueError("cluster direct-observable packet contract mismatch")
    for name in ("source_manifest_sha256", "payload_file_sha256"):
        if not isinstance(packet[name], str) or _SHA256.fullmatch(packet[name]) is None:
            raise ValueError(f"cluster packet hash invalid: {name}")
    packet_id = packet["packet_id"]
    if not isinstance(packet_id, str) or not packet_id.startswith("cluster-packet-"):
        raise ValueError("cluster packet identifier changed")
    _validate_role_bindings(
        packet["calibration_bindings"], REQUIRED_CALIBRATION_ROLES, "calibration"
    )
    _validate_role_bindings(packet["covariance_bindings"], REQUIRED_COVARIANCE_ROLES, "covariance")
    transformation = packet["transformation"]
    if set(transformation) != {
        "kind",
        "raw_input_root_sha256",
        "implementation_sha256",
        "uncertainty_propagation_sha256",
        "assumptions_sha256",
    }:
        raise ValueError("cluster packet transformation keys mismatch")
    if packet["data_class"] == "raw_direct_observable":
        if transformation["kind"] != "identity_raw_measurement_no_derived_mass_truth":
            raise ValueError("raw cluster packet must use the identity transformation")
    elif transformation["kind"] != "audited_raw_to_calibrated_direct_observable":
        raise ValueError("calibrated cluster packet lacks an audited direct transformation")
    for name, value in transformation.items():
        if name.endswith("_sha256") and _SHA256.fullmatch(str(value)) is None:
            raise ValueError(f"cluster packet transformation hash invalid: {name}")


def _synthetic_packet(data_class: str, channel: str) -> dict[str, Any]:
    transformation_kind = (
        "identity_raw_measurement_no_derived_mass_truth"
        if data_class == "raw_direct_observable"
        else "audited_raw_to_calibrated_direct_observable"
    )
    body = {
        "schema_version": PACKET_SCHEMA,
        "packet_id": f"cluster-packet-synthetic-{data_class}",
        "data_class": data_class,
        "observable_channel": channel,
        "source_manifest_sha256": _sha({"synthetic": "manifest", "class": data_class}),
        "payload_file_sha256": _sha({"synthetic": "payload", "class": data_class}),
        "calibration_bindings": [
            {"role": role, "content_sha256": _sha({"synthetic": role})}
            for role in REQUIRED_CALIBRATION_ROLES
        ],
        "covariance_bindings": [
            {"role": role, "content_sha256": _sha({"synthetic": role})}
            for role in REQUIRED_COVARIANCE_ROLES
        ],
        "transformation": {
            "kind": transformation_kind,
            "raw_input_root_sha256": _sha({"synthetic": "raw", "class": data_class}),
            "implementation_sha256": _sha({"synthetic": "implementation"}),
            "uncertainty_propagation_sha256": _sha({"synthetic": "uncertainty"}),
            "assumptions_sha256": _sha({"synthetic": "assumptions"}),
        },
        "formula_selection_use": False,
        "target_role": "post_freeze_independent_cluster_falsification",
        "eligibility": dict(_EMPTY_ELIGIBILITY),
    }
    return {**body, "content_sha256": _sha(body)}


def _positive_schema_controls() -> list[dict[str, Any]]:
    controls = [
        _synthetic_packet("raw_direct_observable", "raw_xray_detector_counts"),
        _synthetic_packet(
            "calibrated_direct_observable",
            "calibrated_sunyaev_zeldovich_intensity_or_temperature_decrement",
        ),
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
    galaxy = _load_bound(root, config["shared_galaxy_policy"])
    _validate_policy(policy, galaxy)
    controls = _positive_schema_controls()
    blocker_ledger = [
        {
            "gate_id": "authorized_real_source_packet",
            "decision": "blocked",
            "blocker": "no_authorized_real_cluster_source_packet_registered",
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
            "shared_galaxy_policy": config["shared_galaxy_policy"],
        },
        "contract": {
            "allowed_data_classes": list(ALLOWED_DATA_CLASSES),
            "allowed_observable_channels": list(ALLOWED_CHANNELS),
            "required_calibration_roles": list(REQUIRED_CALIBRATION_ROLES),
            "required_covariance_roles": list(REQUIRED_COVARIANCE_ROLES),
            "forbidden_fields": sorted(FORBIDDEN_FIELDS),
            "formula_selection_use": False,
            "target_role": "post_freeze_independent_cluster_falsification",
            "hydrostatic_or_gr_derived_mass_is_raw_truth": False,
        },
        "authorized_real_source_packet_bindings": [],
        "positive_schema_controls": controls,
        "blocker_ledger": blocker_ledger,
        "counts": {
            "allowed_data_classes": len(ALLOWED_DATA_CLASSES),
            "allowed_observable_channels": len(ALLOWED_CHANNELS),
            "required_calibration_roles": len(REQUIRED_CALIBRATION_ROLES),
            "required_covariance_roles": len(REQUIRED_COVARIANCE_ROLES),
            "forbidden_fields": len(FORBIDDEN_FIELDS),
            "positive_schema_controls": len(controls),
            "positive_schema_control_passes": len(controls),
            "authorized_real_source_packets": 0,
            "real_source_packets_opened": 0,
            "scientific_passes": 0,
            "scientific_rejects": 0,
            "scientific_blocks": 1,
            "rank_writes": 0,
        },
        "first_blocker": "no_authorized_real_cluster_source_packet_registered",
        "observational_authorization": False,
        "observational_data_opened": False,
        "source_packet_opened": False,
        "candidate_use_authorized": False,
        "scientific_pass_claimed": False,
        "scientific_reject_claimed": False,
        "rank_claimed": False,
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
            "cluster source packet exists or opens; no hydrostatic, GR-derived, halo, or latent "
            "mass is accepted as raw truth, and no scientific conclusion or rank is emitted."
        ),
    }
    return _sealed(body)


def validate_readiness(
    value: Mapping[str, Any], config: Mapping[str, Any], root: str | Path
) -> None:
    _validate_sealed(value, "cluster direct-observable readiness")
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
        or value["counts"]["rank_writes"] != 0
        or value["observational_authorization"] is not False
        or value["observational_data_opened"] is not False
        or value["source_packet_opened"] is not False
        or value["candidate_use_authorized"] is not False
        or value["scientific_pass_claimed"] is not False
        or value["scientific_reject_claimed"] is not False
        or value["rank_claimed"] is not False
        or value["complete_comparable_evidence"] is not False
        or any(value["data_eligibility"].values())
        or any(value["seals"].values())
    ):
        raise ValueError("cluster direct-observable readiness contract mismatch")


def load_and_build_readiness(
    root: str | Path, config_path: str | Path = CONFIG_REL
) -> dict[str, Any]:
    root = Path(root).resolve()
    config = load_config(root, (root / config_path).resolve())
    return build_readiness(config, root)
