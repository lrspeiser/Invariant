from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.kastner_schlatter_observational_readiness_contract import (
    _validate_config,
    _validate_source_bindings,
    build_artifact,
    validate_artifact,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/kastner_schlatter_observational_readiness_contract.json"
ARTIFACT = ROOT / "runs/engine/kastner-schlatter-observational-readiness-contract.json"
GRAPH = ROOT / "runs/engine/kastner-schlatter-equation-graph-admission.json"


def test_exact_rebuild_and_fail_closed_counts() -> None:
    stored = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    rebuilt = build_artifact(CONFIG, ROOT)
    assert stored == rebuilt
    validate_artifact(stored, CONFIG, ROOT)
    assert rebuilt["registration_counts"] == {
        "total_fields": 88,
        "by_status": {
            "forbidden": 7,
            "missing_required": 58,
            "source_blocked": 4,
            "source_registered": 19,
        },
        "by_category": {
            "bundle": 11,
            "likelihood": 16,
            "nuisance": 14,
            "observable": 19,
            "source_parameter": 18,
            "split": 10,
        },
        "by_quantity_class": {
            "calibrated": 12,
            "derived": 12,
            "latent": 13,
            "metadata": 27,
            "model_dependent": 16,
            "raw": 8,
        },
        "by_lane": {
            "lambda_relation": 10,
            "mond_btfr": 34,
            "sds_clock_acceleration": 22,
            "transaction_poisson": 22,
        },
        "missing_field_count": 58,
        "forbidden_field_count": 7,
        "source_blocked_field_count": 4,
        "source_registered_field_count": 19,
    }
    assert rebuilt["observational_access_count"] == 0
    assert rebuilt["real_data_bundle_count"] == 0
    assert rebuilt["real_data_pass_count"] == 0
    assert rebuilt["theory_or_ontology_pass_count"] == 0
    assert all(rebuilt["negative_controls"].values())
    assert rebuilt["synthetic_positive_control"]["schema_control_pass"] is True
    assert rebuilt["synthetic_positive_control"]["real_data_eligibility"] is False


def test_policy_tamper_is_rejected() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["policy"]["observations_opened"] = True
    with pytest.raises(ValueError, match="fail-closed policy"):
        _validate_config(config, ROOT)


def test_predecessor_hash_tamper_is_rejected(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    original = config["predecessors"]["source_intake"]["file_sha256"]
    config["predecessors"]["source_intake"]["file_sha256"] = (
        ("0" if original[0] != "0" else "1") + original[1:]
    )
    tampered = tmp_path / "tampered-config.json"
    tampered.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="source_intake file hash mismatch"):
        build_artifact(tampered, ROOT)


def test_registered_field_requires_real_graph_node() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    fields = copy.deepcopy(config["field_registry"])
    fields[0]["source_node_ids"] = ["INVENTED-TRANSACTION-DETECTOR"]
    graph = json.loads(GRAPH.read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="unknown equation-graph node"):
        _validate_source_bindings(fields, graph)


def test_eq35_ambiguous_planck_form_cannot_be_registered() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    fields = copy.deepcopy(config["field_registry"])
    target = next(item for item in fields if item["field_id"] == "lambda.eq35_planck_normalization")
    target["status"] = "source_registered"
    graph = json.loads(GRAPH.read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="ambiguous equation 35"):
        _validate_source_bindings(fields, graph)


def test_forbidden_halo_and_redshift_fields_are_explicit() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    forbidden = {
        item["field_id"] for item in config["field_registry"] if item["status"] == "forbidden"
    }
    assert {"mond.halo_mass", "mond.halo_concentration", "mond.halo_profile"} <= forbidden
    assert {"mond.redshift_distance", "lambda.redshift_catalog", "lambda.cosmology_fit"} <= forbidden


def test_galaxy_split_contract_is_group_level_and_missing() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    fields = {item["field_id"]: item for item in config["field_registry"]}
    for field_id in (
        "mond.galaxy_group_train_ids",
        "mond.galaxy_group_calibration_ids",
        "mond.galaxy_group_heldout_ids",
        "mond.split_seed_and_manifest_sha256",
    ):
        assert fields[field_id]["status"] == "missing_required"
    assert "radius-row splitting is leakage" in fields["mond.galaxy_group_heldout_ids"]["reason"]
