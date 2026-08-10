from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from sigma_theory_compiler.composite_covariant_lift_campaign import (
    run_composite_covariant_lift_campaign,
)
from sigma_theory_compiler.composite_negative_local_kinetic_campaign import (
    run_composite_negative_local_kinetic_campaign,
)
from sigma_theory_compiler.composite_positive_qx_tilt_campaign import (
    run_composite_positive_qx_tilt_campaign,
)
from sigma_theory_compiler.composite_promotion_overlay import (
    CONFIG_SCHEMA,
    CompositePromotionOverlay,
)
from sigma_theory_compiler.composite_q_degenerate_formal_campaign import (
    run_composite_q_degenerate_formal_campaign,
)
from sigma_theory_compiler.high_throughput import (
    build_basis,
    candidate_id,
    correction_expression,
    decode_ordinal,
)
from sigma_theory_compiler.promotion_dossier import build_promotion_dossiers
from sigma_theory_compiler.promotion_orchestrator import (
    ELIGIBILITY,
    EVIDENCE_SCHEMA,
    PromotionOrchestrator,
)

ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "configs" / "generator_v2_billion.json"
GRAMMAR = ROOT / "configs" / "covariant_action_grammar.json"
FIELD_CONTRACT = ROOT / "configs" / "covariant_field_contract.json"
STATIC_DICTIONARY = (
    ROOT / "runs" / "static-lift" / "einstein-aether-static-dictionary-ir.json"
)
PIPELINE = ROOT / "configs" / "promotion_pipeline_fail_closed.json"


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _candidate(ordinal: int) -> dict:
    generator = _load(GENERATOR)
    decoded = decode_ordinal(
        generator["basis_count"], generator["max_action_terms"], ordinal
    )
    return {
        "candidate_id": candidate_id(generator["protocol_version"], decoded),
        "ordinal": ordinal,
        "term_ids": list(decoded["term_ids"]),
        "signs": list(decoded["signs"]),
        "correction_expression": correction_expression(
            decoded, build_basis(generator["basis_count"])
        ),
        "source_manifest_sha256": "b" * 64,
        "data_eligibility": dict(ELIGIBILITY),
    }


def _fixture(tmp_path: Path) -> dict[str, object]:
    database = tmp_path / "source-promotion.sqlite"
    orchestrator = PromotionOrchestrator(database, _load(PIPELINE))
    for ordinal in (7, 677, 693, 3008915, 24000575, 40084194):
        candidate = _candidate(ordinal)
        evidence = {
            "schema_version": EVIDENCE_SCHEMA,
            "candidate_id": candidate["candidate_id"],
            "ordinal": ordinal,
            "status": "pass",
            "source_result_sha256": "c" * 64,
            "status_root_sha256": "d" * 64,
            "data_eligibility": dict(ELIGIBILITY),
        }
        orchestrator.register_candidate(candidate, evidence)

    dossier = _write(tmp_path / "source-dossier.json", build_promotion_dossiers(database))
    database_sha = _file_sha(database)
    source_summary_sha = "a" * 64
    shared = {
        "database_file_sha256": database_sha,
        "generator_file_sha256": _file_sha(GENERATOR),
        "grammar_file_sha256": _file_sha(GRAMMAR),
        "field_contract_file_sha256": _file_sha(FIELD_CONTRACT),
        "static_dictionary_file_sha256": _file_sha(STATIC_DICTIONARY),
        "static_dictionary_content_sha256": _load(STATIC_DICTIONARY)["content_sha256"],
        "source_summary_file_sha256": source_summary_sha,
        "data_eligibility": dict(ELIGIBILITY),
    }
    lift_config = {
        "schema_version": "sigma-composite-covariant-lift-campaign-config-1.0",
        **shared,
        "expected_candidate_count": 6,
        "output": "unused.json",
    }
    formal_config = {
        "schema_version": "sigma-composite-q-degenerate-formal-campaign-config-1.0",
        **shared,
        "expected_production_candidate_count": 6,
        "expected_composite_candidate_count": 5,
        "output": "unused.json",
    }
    lift_config_path = _write(tmp_path / "lift-config.json", lift_config)
    formal_config_path = _write(tmp_path / "formal-config.json", formal_config)
    lift_artifact_path = _write(
        tmp_path / "lift-artifact.json",
        run_composite_covariant_lift_campaign(
            lift_config,
            database,
            GENERATOR,
            GRAMMAR,
            FIELD_CONTRACT,
            STATIC_DICTIONARY,
        ),
    )
    formal_artifact_path = _write(
        tmp_path / "formal-artifact.json",
        run_composite_q_degenerate_formal_campaign(
            formal_config,
            database,
            GENERATOR,
            GRAMMAR,
            FIELD_CONTRACT,
            STATIC_DICTIONARY,
        ),
    )
    negative_config = {
        "schema_version": "sigma-composite-negative-local-kinetic-campaign-config-1.0",
        **shared,
        "prior_campaign_file_sha256": _file_sha(formal_artifact_path),
        "prior_campaign_content_sha256": _load(formal_artifact_path)["content_sha256"],
        "expected_input_candidate_count": 3,
        "output": "unused.json",
    }
    negative_config_path = _write(tmp_path / "negative-config.json", negative_config)
    negative_artifact_path = _write(
        tmp_path / "negative-artifact.json",
        run_composite_negative_local_kinetic_campaign(
            negative_config,
            database,
            GENERATOR,
            GRAMMAR,
            FIELD_CONTRACT,
            STATIC_DICTIONARY,
            formal_artifact_path,
        ),
    )
    positive_config = {
        "schema_version": "sigma-composite-positive-qx-tilt-campaign-config-1.0",
        **shared,
        "prior_campaign_file_sha256": _file_sha(negative_artifact_path),
        "prior_campaign_content_sha256": _load(negative_artifact_path)["content_sha256"],
        "expected_input_candidate_count": 1,
        "output": "unused.json",
    }
    positive_config_path = _write(tmp_path / "positive-config.json", positive_config)
    positive_artifact_path = _write(
        tmp_path / "positive-artifact.json",
        run_composite_positive_qx_tilt_campaign(
            positive_config,
            database,
            GENERATOR,
            GRAMMAR,
            FIELD_CONTRACT,
            STATIC_DICTIONARY,
            negative_artifact_path,
        ),
    )
    paths = {
        "source_database": database,
        "source_dossier": dossier,
        "lift_campaign_config": lift_config_path,
        "lift_campaign_artifact": lift_artifact_path,
        "formal_campaign_config": formal_config_path,
        "formal_campaign_artifact": formal_artifact_path,
        "negative_formal_campaign_config": negative_config_path,
        "negative_formal_campaign_artifact": negative_artifact_path,
        "positive_tilt_campaign_config": positive_config_path,
        "positive_tilt_campaign_artifact": positive_artifact_path,
        "generator": GENERATOR,
        "grammar": GRAMMAR,
        "field_contract": FIELD_CONTRACT,
        "static_dictionary": STATIC_DICTIONARY,
    }
    config = {
        "schema_version": CONFIG_SCHEMA,
        "expected_source_candidate_count": 6,
        "expected_overlay_candidate_count": 6,
        "expected_formal_rejected_count": 6,
        "expected_remaining_formal_blocked_count": 0,
        "expected_formal_passed_count": 0,
        "maximum_candidates": 6,
        "maximum_disk_bytes": 10_000_000,
        "maximum_wall_seconds": 60,
        "external_paid_llm_calls": False,
        "data_eligibility": dict(ELIGIBILITY),
        "file_sha256": {name: _file_sha(path) for name, path in paths.items()},
    }
    return {"paths": paths, "config": config}


def _overlay(tmp_path: Path, fixture: dict[str, object]) -> CompositePromotionOverlay:
    return CompositePromotionOverlay(
        tmp_path / "overlay.sqlite",
        fixture["config"],
        **fixture["paths"],
    )


def test_overlay_is_restart_safe_idempotent_and_keeps_remaining_family_closed(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    source = fixture["paths"]["source_database"]
    dossier = fixture["paths"]["source_dossier"]
    source_before = Path(source).read_bytes()
    dossier_before = Path(dossier).read_bytes()

    partial = _overlay(tmp_path, fixture).apply(maximum_new_records=1)
    assert partial["state"] == "building"
    assert partial["overlay_candidate_count"] == 1

    completed = _overlay(tmp_path, fixture).apply()
    assert completed["state"] == "completed"
    assert completed["upstream_terminal_candidate_count"] == 0
    assert completed["overlay_candidate_count"] == 6
    assert completed["lift_passed_count"] == 6
    assert completed["formal_rejected_count"] == 6
    assert completed["remaining_formal_blocked_count"] == 0
    assert completed["formal_passed_count"] == 0
    assert completed["formal_layer_one_counts"] == {"blocked": 3, "rejected": 3}
    assert completed["formal_layer_two_counts"] == {
        "blocked": 1,
        "not_run": 3,
        "rejected": 2,
    }
    assert completed["formal_layer_three_counts"] == {
        "not_run": 5,
        "rejected": 1,
    }
    assert completed["solar_opened_count"] == completed["galaxy_opened_count"] == 0
    assert completed["data_eligibility"] == {**ELIGIBILITY, "passed": True}
    assert completed["paid_llm_spend_usd"] == 0.0
    assert len(completed["overlay_root_sha256"]) == 64

    replay = _overlay(tmp_path, fixture).apply()
    assert replay["inserted_this_run"] == 0
    assert replay["replayed_this_run"] == 6
    assert replay["overlay_root_sha256"] == completed["overlay_root_sha256"]
    assert Path(source).read_bytes() == source_before
    assert Path(dossier).read_bytes() == dossier_before

    exported = _overlay(tmp_path, fixture).export()
    assert exported["candidate_count"] == 6
    assert exported["overlay_root_sha256"] == completed["overlay_root_sha256"]
    assert exported["data_eligibility"] == {**ELIGIBILITY, "passed": True}
    assert exported["content_sha256"] == _sha(
        {key: value for key, value in exported.items() if key != "content_sha256"}
    )

    with sqlite3.connect(tmp_path / "overlay.sqlite") as connection:
        rows = connection.execute(
            "SELECT formal_decision,solar_state,galaxy_state,lift_input_lineage_sha256,"
            "lift_output_lineage_sha256,formal_layer_one_input_lineage_sha256,"
            "formal_layer_one_output_lineage_sha256,formal_output_lineage_sha256 "
            "FROM candidate_overlay ORDER BY ordinal"
        ).fetchall()
    assert [row[0] for row in rows] == [
        "rejected",
        "rejected",
        "rejected",
        "rejected",
        "rejected",
        "rejected",
    ]
    assert all(row[1:3] == ("blocked", "blocked") for row in rows)
    assert all(all(len(value) == 64 for value in row[3:]) for row in rows)
    with sqlite3.connect(tmp_path / "overlay.sqlite") as connection:
        third = connection.execute(
            "SELECT formal_layer_three_decision,formal_layer_three_input_lineage_sha256,"
            "formal_layer_three_output_lineage_sha256 FROM candidate_overlay ORDER BY ordinal"
        ).fetchall()
    assert [row[0] for row in third].count("rejected") == 1
    assert [row[0] for row in third].count("not_run") == 5
    assert sum(row[1] is not None and row[2] is not None for row in third) == 1


def test_overlay_rejects_artifact_tampering_and_changed_resume_config(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    _overlay(tmp_path, fixture).apply(maximum_new_records=1)

    changed = json.loads(json.dumps(fixture["config"]))
    changed["maximum_wall_seconds"] = 30
    with pytest.raises(ValueError, match="changed composite overlay config"):
        CompositePromotionOverlay(
            tmp_path / "overlay.sqlite", changed, **fixture["paths"]
        )

    positive_artifact = Path(fixture["paths"]["positive_tilt_campaign_artifact"])
    positive_original = positive_artifact.read_bytes()
    positive_tampered = _load(positive_artifact)
    positive_tampered["remaining_formal_blocked_count"] += 1
    _write(positive_artifact, positive_tampered)
    with pytest.raises(ValueError, match="positive_tilt_campaign_artifact file mismatch"):
        CompositePromotionOverlay(
            tmp_path / "positive-tampered.sqlite", fixture["config"], **fixture["paths"]
        )
    positive_artifact.write_bytes(positive_original)

    artifact = Path(fixture["paths"]["formal_campaign_artifact"])
    tampered = _load(artifact)
    tampered["remaining_formal_blocked_count"] += 1
    _write(artifact, tampered)
    with pytest.raises(ValueError, match="formal_campaign_artifact file mismatch"):
        CompositePromotionOverlay(
            tmp_path / "tampered.sqlite", fixture["config"], **fixture["paths"]
        )


def test_production_overlay_config_is_exactly_hash_bound_and_fail_closed() -> None:
    config = _load(ROOT / "configs" / "composite_promotion_overlay_fail_closed.json")
    assert config["expected_source_candidate_count"] == 5855
    assert config["expected_overlay_candidate_count"] == 70
    assert config["expected_formal_rejected_count"] == 70
    assert config["expected_remaining_formal_blocked_count"] == 0
    assert config["expected_formal_passed_count"] == 0
    assert config["external_paid_llm_calls"] is False
    assert config["data_eligibility"] == ELIGIBILITY
    assert config["file_sha256"]["lift_campaign_artifact"] == _file_sha(
        ROOT / "runs" / "engine" / "composite-covariant-lift-campaign.json"
    )
    assert config["file_sha256"]["formal_campaign_artifact"] == _file_sha(
        ROOT / "runs" / "engine" / "composite-q-degenerate-formal-campaign.json"
    )
    assert config["file_sha256"]["negative_formal_campaign_artifact"] == _file_sha(
        ROOT / "runs" / "engine" / "composite-negative-local-kinetic-campaign.json"
    )
    assert config["file_sha256"]["positive_tilt_campaign_artifact"] == _file_sha(
        ROOT / "runs" / "engine" / "composite-positive-qx-tilt-campaign.json"
    )
    assert len(_sha({"source_database": config["file_sha256"]["source_database"]})) == 64

    status = _load(
        ROOT / "runs" / "engine" / "composite-promotion-overlay-production-status.json"
    )
    body = {key: value for key, value in status.items() if key != "content_sha256"}
    assert status["content_sha256"] == _sha(body)
    assert status["source_database_file_sha256"] == config["file_sha256"][
        "source_database"
    ]
    assert status["source_dossier_file_sha256"] == config["file_sha256"][
        "source_dossier"
    ]
    assert status["lift_passed_count"] == 70
    assert status["formal_rejected_count"] == 70
    assert status["remaining_formal_blocked_count"] == 0
    assert status["formal_passed_count"] == 0
    assert status["solar_opened_count"] == status["galaxy_opened_count"] == 0
    assert status["data_eligibility"] == {**ELIGIBILITY, "passed": True}
