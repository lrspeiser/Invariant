from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path

import pytest

from sigma_theory_compiler.grammar_v3_g4_solar_promotion_service import (
    DESCRIPTOR_SCHEMA,
    EVALUATOR_DESCRIPTOR_SCHEMA,
    MISSING_DESCRIPTOR,
    OUTPUT_CHANNELS,
    PREDICTION_BUNDLE_SCHEMA,
    GrammarV3G4SolarPromotionService,
    _sha,
)
from sigma_theory_compiler.promotion_orchestrator import ELIGIBILITY

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "grammar_v3_g4_solar_promotion_service.json"
CONTRACT = (
    ROOT
    / "configs"
    / "grammar_v3_g4_solar_prediction_bundle_descriptor_contract.json"
)
PORTABLE = (
    ROOT / "runs" / "engine" / "grammar-v3-g4-solar-promotion-service-status.json"
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _service(directory: Path, config: dict, root: Path = ROOT):
    return GrammarV3G4SolarPromotionService(directory, config, root)


def test_missing_candidate_bundle_remains_deferred_and_gr_is_calibration_only(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path / "service", _load(CONFIG))
    status = service.start()
    assert status["lifecycle"] == "waiting_for_prediction_bundle"
    assert status["work_state_counts"] == {
        "deferred_missing_prediction_bundle_descriptor": 1
    }
    assert status["formal_pass_verified"] is True
    assert status["reviewed_prediction_audit_binding"] == {
        "audit_content_sha256": "0c0b7ee5fac3c1cc6986648013e3f28da8dd37cba9975a42bcab8f811ebd64c3",
        "candidate_provenance_sha256": "15d7d7995038b081b173845c9a686d24429d150e931f001a16e3fa52801ea1b1",
        "analytic_bundle_count": 1,
        "real_bundle_count": 0,
        "decision": "blocked",
    }
    assert status["prediction_bundle_descriptor_registered"] is False
    assert status["reviewed_solar_evaluator_invoked"] is False
    assert status["solar_evaluator_opened"] is False
    assert status["observational_data_opened"] is False
    assert status["paid_llm_spend_usd"] == 0.0
    assert status["data_eligibility"] == {**ELIGIBILITY, "passed": True}

    leaderboard = status["category_leaderboard"]
    assert leaderboard["scalar_truth_score"] is None
    assert len(leaderboard["ranked"]) == 1
    gr = leaderboard["ranked"][0]
    assert gr["candidate_id"] == "KNOWN-ANSWER-EINSTEIN-HILBERT"
    assert gr["role"] == "calibration_only_control"
    assert gr["promotion_eligible"] is False
    assert gr["exact_metrics"] == {
        "passed_control_count": 5,
        "total_control_count": 5,
    }
    candidate = leaderboard["blocked_or_untested"][0]
    assert candidate["candidate_id"] == "G3-f9c598b70a77ea54009d8f18"
    assert candidate["rank"] is None
    assert candidate["exact_metrics"] is None
    assert candidate["evidence_status"] == "untested"
    assert candidate["blocker"] == MISSING_DESCRIPTOR
    assert candidate["promotion_eligible"] is False


def test_restart_stop_resume_is_idempotent_and_bounded(tmp_path: Path) -> None:
    directory = tmp_path / "service"
    config = _load(CONFIG)
    service = _service(directory, config)
    first = service.start()
    restarted = _service(directory, config)
    assert restarted.status()["content_sha256"] == first["content_sha256"]
    stopped = restarted.stop()
    assert stopped["lifecycle"] == "stopped"
    resumed = _service(directory, config).resume()
    assert resumed["lifecycle"] == "waiting_for_prediction_bundle"
    assert resumed["cycle_count"] == 2
    assert resumed["task_count"] == 1
    assert resumed["queue_root_sha256"] == first["queue_root_sha256"]
    assert resumed["work_record_root_sha256"] == first["work_record_root_sha256"]


def test_formal_source_and_seal_tamper_fail_before_service_creation(
    tmp_path: Path,
) -> None:
    config = _load(CONFIG)
    config["formal_pass"]["audit"]["content_sha256"] = "0" * 64
    directory = tmp_path / "formal-tamper"
    with pytest.raises(ValueError, match="content changed"):
        _service(directory, config)
    assert not directory.exists()

    config = _load(CONFIG)
    config["data_eligibility"]["dark_matter_or_halo_inputs"] = True
    with pytest.raises(ValueError, match="forbidden input"):
        _service(tmp_path / "seal-tamper", config)

    config = _load(CONFIG)
    config["gr_calibration"]["promotion_eligible"] = True
    with pytest.raises(ValueError, match="leaked"):
        _service(tmp_path / "gr-leak", config)


def _copy_inputs(root: Path, config: dict) -> None:
    bindings = [
        config["formal_pass"]["audit"],
        config["formal_pass"]["queue_status"],
        config["formal_pass"]["service_status"],
        config["solar_prediction_audit"],
        config["solar_protocol"],
        config["gr_calibration"]["status_artifact"],
        config["prediction_bundle_descriptor_contract"],
    ]
    for binding in bindings:
        source = ROOT / binding["path"]
        target = root / binding["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)


def _register_fixture_descriptor(root: Path, config: dict) -> dict:
    source = root / "src" / "reviewed_candidate_solar_fixture.py"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("def reviewed_candidate_solar(candidate, context):\n    return {}\n")
    evaluator = {
        "schema_version": EVALUATOR_DESCRIPTOR_SCHEMA,
        "evaluator_id": "reviewed-g4-solar-fixture",
        "candidate_id": config["candidate"]["candidate_id"],
        "action_sha256": config["candidate"]["action_sha256"],
        "callback": "reviewed_candidate_solar_fixture:reviewed_candidate_solar",
        "artifact_path": "src/reviewed_candidate_solar_fixture.py",
        "artifact_sha256": _file_sha(source),
        "data_eligibility": ELIGIBILITY,
    }
    evaluator_path = root / "configs" / "reviewed_g4_solar_fixture.json"
    evaluator_path.parent.mkdir(parents=True, exist_ok=True)
    evaluator_path.write_text(json.dumps(evaluator, indent=2) + "\n")

    components = {
        "weak_field_solution_sha256": "1" * 64,
        "state_estimation_contract_sha256": "2" * 64,
        "instrument_calibration_contract_sha256": "3" * 64,
        "covariance_contract_sha256": "4" * 64,
        "likelihood_contract_sha256": "5" * 64,
        "split_commitment_sha256": "6" * 64,
        "stopping_rule_sha256": "7" * 64,
    }
    bundle_body = {
        "schema_version": PREDICTION_BUNDLE_SCHEMA,
        "candidate_id": config["candidate"]["candidate_id"],
        "action_sha256": config["candidate"]["action_sha256"],
        "output_channels": OUTPUT_CHANNELS,
        "universal_parameter_count": 3,
        "object_specific_gravity_parameter_count": 0,
        **components,
        "data_eligibility": ELIGIBILITY,
        "observational_data_opened": False,
    }
    bundle = {**bundle_body, "content_sha256": _sha(bundle_body)}
    bundle_path = root / "runs" / "engine" / "g4-solar-prediction-fixture.json"
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    bundle_path.write_text(json.dumps(bundle, indent=2) + "\n")

    formal = config["formal_pass"]
    descriptor = {
        "schema_version": DESCRIPTOR_SCHEMA,
        "descriptor_id": "reviewed-g4-solar-prediction-fixture",
        "candidate": config["candidate"],
        "formal_pass_binding": {
            "audit_content_sha256": formal["audit"]["content_sha256"],
            "candidate_provenance_sha256": formal["audit"][
                "candidate_provenance_sha256"
            ],
            "completed_work_records_root_sha256": formal["queue_status"][
                "completed_work_records_root_sha256"
            ],
            "queue_registry_root_sha256": formal["queue_status"][
                "queue_registry_root_sha256"
            ],
        },
        "prediction_audit_binding": {
            "audit_content_sha256": config["solar_prediction_audit"][
                "content_sha256"
            ],
            "candidate_provenance_sha256": config["solar_prediction_audit"][
                "candidate_provenance_sha256"
            ],
            "analytic_bundle_count": 1,
            "real_bundle_count": 0,
            "decision": "blocked",
        },
        "solar_protocol_binding": config["solar_protocol"],
        "prediction_bundle": {
            "path": "runs/engine/g4-solar-prediction-fixture.json",
            "file_sha256": _file_sha(bundle_path),
            "content_sha256": bundle["content_sha256"],
            "schema_version": PREDICTION_BUNDLE_SCHEMA,
            "output_channels": OUTPUT_CHANNELS,
            "universal_parameter_count": 3,
            "object_specific_gravity_parameter_count": 0,
            **components,
        },
        "reviewed_evaluator": {
            "descriptor_path": "configs/reviewed_g4_solar_fixture.json",
            "descriptor_file_sha256": _file_sha(evaluator_path),
            "evaluator_binding_sha256": _sha(evaluator),
            "callback": evaluator["callback"],
        },
        "observational_opening": {
            "authorized": False,
            "requires_independent_dataset_manifest_audit": True,
            "requires_preregistered_session_split": True,
        },
        "data_eligibility": ELIGIBILITY,
    }
    descriptor_path = root / "configs" / "g4-solar-prediction-fixture-descriptor.json"
    descriptor_path.write_text(json.dumps(descriptor, indent=2) + "\n")
    return {
        "path": "configs/g4-solar-prediction-fixture-descriptor.json",
        "file_sha256": _file_sha(descriptor_path),
    }


def test_exact_descriptor_epoch_becomes_ready_but_does_not_open_solar(
    tmp_path: Path,
) -> None:
    root = tmp_path / "project"
    config = _load(CONFIG)
    _copy_inputs(root, config)
    directory = tmp_path / "service"
    service = _service(directory, config, root)
    service.start()

    advanced = copy.deepcopy(config)
    advanced["candidate_prediction_bundle_descriptor"] = _register_fixture_descriptor(
        root, advanced
    )
    epoch = _service(directory, advanced, root)
    status = epoch.resume()
    assert status["lifecycle"] == "ready_for_reviewed_solar_evaluator"
    assert status["prediction_bundle_descriptor_registered"] is True
    assert status["work_state_counts"] == {"ready_for_reviewed_solar_evaluator": 1}
    assert status["reviewed_solar_evaluator_invoked"] is False
    assert status["solar_evaluator_opened"] is False
    assert status["observational_data_opened"] is False
    candidate = status["category_leaderboard"]["blocked_or_untested"][0]
    assert candidate["evidence_status"] == "ready_for_reviewed_evaluator"
    assert candidate["promotion_eligible"] is False


def test_descriptor_tamper_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "project"
    config = _load(CONFIG)
    _copy_inputs(root, config)
    config["candidate_prediction_bundle_descriptor"] = _register_fixture_descriptor(
        root, config
    )
    descriptor_path = root / config["candidate_prediction_bundle_descriptor"]["path"]
    descriptor = _load(descriptor_path)
    descriptor["prediction_bundle"]["object_specific_gravity_parameter_count"] = 1
    descriptor_path.write_text(json.dumps(descriptor, indent=2) + "\n")
    config["candidate_prediction_bundle_descriptor"]["file_sha256"] = _file_sha(
        descriptor_path
    )
    with pytest.raises(ValueError, match="exact contract"):
        _service(tmp_path / "tampered", config, root)


def test_portable_status_is_exact(tmp_path: Path) -> None:
    service = _service(tmp_path / "portable", _load(CONFIG))
    service.start()
    assert service.export() == _load(PORTABLE)


def test_machine_readable_descriptor_contract_is_exact() -> None:
    contract = _load(CONTRACT)
    assert contract["additionalProperties"] is False
    assert contract["properties"]["schema_version"]["const"] == DESCRIPTOR_SCHEMA
    assert contract["properties"]["candidate"]["properties"]["candidate_id"][
        "const"
    ] == "G3-f9c598b70a77ea54009d8f18"
    assert contract["properties"]["observational_opening"]["const"][
        "authorized"
    ] is False
    assert contract["properties"]["data_eligibility"]["const"] == ELIGIBILITY
