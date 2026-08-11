"""Fail-closed observational-readiness contract for Kastner--Schlatter equations."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

CONFIG_SCHEMA = "sigma-kastner-schlatter-observational-readiness-config-1.0"
ARTIFACT_SCHEMA = "sigma-kastner-schlatter-observational-readiness-1.0"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_STATUSES = {"source_registered", "missing_required", "forbidden", "source_blocked"}
_CLASSES = {"raw", "calibrated", "model_dependent", "latent", "derived", "metadata"}
_CATEGORIES = {"observable", "source_parameter", "nuisance", "likelihood", "split", "bundle"}


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _content_sha(value: dict[str, Any]) -> str:
    return _sha({key: item for key, item in value.items() if key != "content_sha256"})


def _load_bound(root: Path, binding: dict[str, Any], label: str) -> dict[str, Any]:
    path = (root / binding["path"]).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} path escapes repository") from error
    if not path.is_file() or _file_sha(path) != binding["file_sha256"]:
        raise ValueError(f"{label} file hash mismatch")
    value = json.loads(path.read_text(encoding="utf-8"))
    expected = binding.get("content_sha256")
    if expected is not None and (
        value.get("content_sha256") != expected or _content_sha(value) != expected
    ):
        raise ValueError(f"{label} content hash mismatch")
    return value


def _validate_config(config: dict[str, Any], root: Path) -> None:
    expected = {
        "schema_version",
        "campaign_id",
        "adapter_source",
        "test_source",
        "output_path",
        "predecessors",
        "field_registry",
        "policy",
    }
    if set(config) != expected or config.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("observational readiness config shape changed")
    for source_key in ("adapter_source", "test_source"):
        source = config[source_key]
        if not _SHA256.fullmatch(source.get("file_sha256", "")):
            raise ValueError(f"{source_key} hash is invalid")
        if _file_sha(root / source["path"]) != source["file_sha256"]:
            raise ValueError(f"{source_key} hash mismatch")
    policy = config["policy"]
    if policy != {
        "synthetic_only": True,
        "observations_opened": False,
        "dark_matter_or_halo_inputs_allowed": False,
        "redshift_or_cosmology_inputs_allowed": False,
        "transaction_event_observable_invention_allowed": False,
        "action_or_ontology_evidence_allowed": False,
        "real_data_pass_allowed": False,
    }:
        raise ValueError("fail-closed policy changed")
    fields = config["field_registry"]
    field_ids = [item.get("field_id") for item in fields]
    if len(field_ids) != len(set(field_ids)) or not fields:
        raise ValueError("field identifiers must be unique and nonempty")
    required = {
        "field_id",
        "lane_id",
        "category",
        "quantity_class",
        "status",
        "source_node_ids",
        "reason",
    }
    for field in fields:
        if set(field) != required:
            raise ValueError(f"field shape changed: {field.get('field_id')}")
        if field["status"] not in _STATUSES or field["quantity_class"] not in _CLASSES:
            raise ValueError(f"invalid field classification: {field['field_id']}")
        if field["category"] not in _CATEGORIES:
            raise ValueError(f"invalid field category: {field['field_id']}")
        if field["status"] == "source_registered" and not field["source_node_ids"]:
            raise ValueError(f"registered field lacks source node: {field['field_id']}")


def _validate_predecessors(config: dict[str, Any], root: Path) -> dict[str, dict[str, Any]]:
    if set(config["predecessors"]) != {"source_intake", "equation_graph", "cuda_consequence"}:
        raise ValueError("predecessor set changed")
    loaded = {
        name: _load_bound(root, binding, name)
        for name, binding in config["predecessors"].items()
    }
    graph = loaded["equation_graph"]
    cuda = loaded["cuda_consequence"]
    intake_hash = config["predecessors"]["source_intake"]["content_sha256"]
    if graph.get("source_lineage", {}).get("source_intake_content_sha256") != intake_hash:
        raise ValueError("equation graph is not bound to intake")
    if graph.get("admission_contract") != {
        "kind": "equation_universe_compatible_typed_knowledge_graph",
        "equation_universe_schema": "sigma-equation-universe-1.0",
        "equation_only": True,
        "fundamental_action": None,
        "variational_edges_present": False,
        "theory_equivalence_edges_present": False,
        "observational_edges_present": False,
    }:
        raise ValueError("equation-only admission boundary changed")
    if graph.get("graph_sha256") != config["predecessors"]["equation_graph"].get("graph_sha256"):
        raise ValueError("equation graph hash mismatch")
    if (
        cuda.get("predecessor_binding", {}).get("content_sha256") != intake_hash
        or cuda.get("synthetic_only") is not True
        or cuda.get("observations_opened") is not False
        or cuda.get("dark_matter_or_halo_inputs") is not False
        or cuda.get("redshift_or_cosmology_inputs") is not False
    ):
        raise ValueError("CUDA consequence predecessor boundary changed")
    return loaded


def _field_counts(fields: list[dict[str, Any]]) -> dict[str, Any]:
    def counts(key: str) -> dict[str, int]:
        return dict(sorted(Counter(item[key] for item in fields).items()))

    return {
        "total_fields": len(fields),
        "by_status": counts("status"),
        "by_category": counts("category"),
        "by_quantity_class": counts("quantity_class"),
        "by_lane": counts("lane_id"),
        "missing_field_count": sum(item["status"] == "missing_required" for item in fields),
        "forbidden_field_count": sum(item["status"] == "forbidden" for item in fields),
        "source_blocked_field_count": sum(item["status"] == "source_blocked" for item in fields),
        "source_registered_field_count": sum(
            item["status"] == "source_registered" for item in fields
        ),
    }


def _validate_source_bindings(fields: list[dict[str, Any]], graph: dict[str, Any]) -> None:
    nodes = {item["node_id"]: item for item in graph["knowledge_graph"]["nodes"]}
    ambiguous = "EQ-KS-35-LAMBDA-PLANCK"
    for field in fields:
        for node_id in field["source_node_ids"]:
            if node_id not in nodes:
                raise ValueError(f"unknown equation-graph node: {node_id}")
        if field["status"] == "source_registered" and ambiguous in field["source_node_ids"]:
            raise ValueError("ambiguous equation 35 Planck normalization was registered")


def _positive_control() -> dict[str, Any]:
    # An equation-only point-mass fixture; it is not a galaxy-data surrogate.
    gravitational_constant = 1.0
    central_mass = 81.0
    acceleration_scale = 1.0 / 16.0
    radius = 12.0
    acceleration = math.sqrt(gravitational_constant * central_mass * acceleration_scale) / radius
    speed_squared = math.sqrt(gravitational_constant * central_mass * acceleration_scale)
    speed = math.sqrt(speed_squared)
    return {
        "control_id": "synthetic-point-mass-mond-equation-control",
        "synthetic_only": True,
        "equation_node_ids": ["EQ-KS-68-DEEP-ACCELERATION", "EQ-KS-69-VELOCITY"],
        "inputs": {"G": gravitational_constant, "M": central_mass, "a0": acceleration_scale, "r": radius},
        "derived": {"abar": acceleration, "v": speed},
        "absolute_residuals": {
            "eq68": abs(acceleration * radius - speed_squared),
            "eq69": abs(speed**2 - speed_squared),
        },
        "schema_control_pass": abs(acceleration * radius - speed_squared) < 1e-15
        and abs(speed**2 - speed_squared) < 1e-15,
        "real_data_eligibility": False,
        "observational_pass": False,
    }


def _negative_controls(config: dict[str, Any], root: Path) -> dict[str, bool]:
    fields = config["field_registry"]
    forbidden = {item["field_id"] for item in fields if item["status"] == "forbidden"}
    registered = {item["field_id"] for item in fields if item["status"] == "source_registered"}
    transaction_sources = {
        node
        for item in fields
        if item["lane_id"] == "transaction_poisson"
        for node in item["source_node_ids"]
    }
    train_ids = {"synthetic-galaxy-a", "synthetic-galaxy-b"}
    heldout_ids = {"synthetic-galaxy-b", "synthetic-galaxy-c"}
    predecessor = config["predecessors"]["source_intake"]
    actual_predecessor_hash = _file_sha(root / predecessor["path"])
    first = "0" if predecessor["file_sha256"][0] != "0" else "1"
    tampered_predecessor_hash = first + predecessor["file_sha256"][1:]
    return {
        "dark_matter_halo_injection_rejected": {
            "mond.halo_mass", "mond.halo_concentration"
        }.issubset(forbidden),
        "redshift_cosmology_injection_rejected": {
            "lambda.redshift_catalog", "lambda.cosmology_fit"
        }.issubset(forbidden),
        "invented_transaction_detector_rejected": (
            "transaction.detector_protocol" not in registered
            and not any("DETECTOR" in node for node in transaction_sources)
        ),
        "galaxy_group_split_leakage_rejected": bool(train_ids & heldout_ids),
        "equation35_normalization_promotion_rejected": (
            "lambda.eq35_planck_normalization" not in registered
        ),
        "predecessor_hash_tamper_rejected": (
            predecessor["file_sha256"] == actual_predecessor_hash
            and tampered_predecessor_hash != actual_predecessor_hash
        ),
    }


def build_artifact(config_path: Path, root: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config, root)
    loaded = _validate_predecessors(config, root)
    fields = config["field_registry"]
    _validate_source_bindings(fields, loaded["equation_graph"])
    controls = _negative_controls(config, root)
    if not all(controls.values()):
        raise ValueError("one or more negative controls failed")
    positive = _positive_control()
    if not positive["schema_control_pass"]:
        raise ValueError("positive synthetic equation control failed")
    artifact: dict[str, Any] = {
        "schema_version": ARTIFACT_SCHEMA,
        "campaign_id": config["campaign_id"],
        "source_lineage": {
            "primary_pdf_sha256": loaded["source_intake"]["primary_source"]["pdf_sha256"],
            "source_intake_file_sha256": config["predecessors"]["source_intake"]["file_sha256"],
            "source_intake_content_sha256": config["predecessors"]["source_intake"]["content_sha256"],
            "equation_graph_file_sha256": config["predecessors"]["equation_graph"]["file_sha256"],
            "equation_graph_content_sha256": config["predecessors"]["equation_graph"]["content_sha256"],
            "equation_graph_sha256": config["predecessors"]["equation_graph"]["graph_sha256"],
            "cuda_consequence_file_sha256": config["predecessors"]["cuda_consequence"]["file_sha256"],
            "cuda_consequence_content_sha256": config["predecessors"]["cuda_consequence"]["content_sha256"],
            "config_file_sha256": _file_sha(config_path),
            "adapter_source_sha256": config["adapter_source"]["file_sha256"],
            "test_source_sha256": config["test_source"]["file_sha256"],
        },
        "contract_scope": {
            "kind": "observational_and_falsification_registration_only",
            "equation_only": True,
            "action_registered": False,
            "ontology_observable_registered": False,
            "transaction_event_detector_equivalence_assumed": False,
            "extended_galaxy_mass_operator_invented": False,
            "equation35_normalization_resolved": False,
        },
        "field_registry": fields,
        "registration_counts": _field_counts(fields),
        "lane_decisions": {
            "transaction_poisson": "blocked_no_operational_transaction_event_or_exposure_definition",
            "lambda_relation": "blocked_equation35_normalization_and_cosmology_authorization",
            "sds_clock_acceleration": "blocked_no_operational_clock_acceleration_bundle_and_lambda_input",
            "mond_btfr": "blocked_no_extended_baryonic_geometry_operator_or_observational_bundle",
        },
        "synthetic_positive_control": positive,
        "negative_controls": controls,
        "decision": "blocked_registration_incomplete_observations_sealed",
        "observational_access_count": 0,
        "real_data_bundle_count": 0,
        "real_data_pass_count": 0,
        "theory_or_ontology_pass_count": 0,
        "data_seals": {
            "synthetic_only": True,
            "observations_opened": False,
            "dark_matter_or_halo_inputs_opened": False,
            "redshift_or_cosmology_inputs_opened": False,
            "transaction_event_observations_opened": False,
            "paid_llm_calls": False,
        },
    }
    artifact["content_sha256"] = _content_sha(artifact)
    return artifact


def validate_artifact(artifact: dict[str, Any], config_path: Path, root: Path) -> None:
    rebuilt = build_artifact(config_path, root)
    if artifact != rebuilt or artifact.get("content_sha256") != _content_sha(artifact):
        raise ValueError("artifact does not exactly match deterministic rebuild")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.root.resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config
    artifact = build_artifact(config_path, root)
    output = root / json.loads(config_path.read_text(encoding="utf-8"))["output_path"]
    output.write_text(_canonical(artifact) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
