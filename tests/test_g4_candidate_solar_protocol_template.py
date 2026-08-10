from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.g4_candidate_solar_protocol_template import (
    ARTIFACT_SCHEMA,
    DESCRIPTOR_SCHEMA,
    OUTPUT_CHANNELS,
    REMAINING_FIELDS,
    G4CandidateSolarTemplateRegistry,
    _sha,
    build_g4_candidate_solar_protocol_template,
)
from sigma_theory_compiler.promotion_orchestrator import ELIGIBILITY

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "g4_candidate_solar_protocol_template.json"
ARTIFACT = ROOT / "runs" / "engine" / "g4-candidate-solar-protocol-template.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def rebuilt() -> dict:
    return build_g4_candidate_solar_protocol_template(_load(CONFIG), ROOT)


def test_artifact_is_an_exact_deterministic_rebuild(rebuilt: dict) -> None:
    artifact = _load(ARTIFACT)
    assert artifact == rebuilt
    body = {key: item for key, item in artifact.items() if key != "content_sha256"}
    assert artifact["content_sha256"] == _sha(body)
    assert artifact["schema_version"] == ARTIFACT_SCHEMA
    assert artifact["status"] == "frozen_template_unregistered_ineligible"


def test_direct_signal_and_quantity_classes_are_frozen_without_model_targets(
    rebuilt: dict,
) -> None:
    assert rebuilt["direct_signal_channels"] == OUTPUT_CHANNELS
    classes = rebuilt["frozen_contracts"]["quantity_classes"]
    assert set(classes) == {"raw", "calibrated", "derived", "model_dependent", "latent"}
    assert classes["raw"]["allowed"] is True
    assert classes["calibrated"]["allowed"] is True
    assert classes["derived"]["allowed"] is True
    assert classes["model_dependent"]["allowed_as_input_or_target"] is False
    assert classes["latent"]["allowed_as_input_or_target"] is False
    assert rebuilt["observational_authorization"] is False
    assert rebuilt["observational_data_opened"] is False
    assert rebuilt["data_eligibility"] == ELIGIBILITY


def test_file_selection_parser_and_calibration_remain_pre_access_blocked(
    rebuilt: dict,
) -> None:
    source = rebuilt["source_registration"]
    assert source["primary_files_selected"] == 0
    assert source["primary_files_downloaded"] == 0
    assert source["target_values_accessed"] is False
    parser = rebuilt["frozen_contracts"]["parser_verification"]
    assert parser["status"] == "blocked_unimplemented"
    assert parser["ATDF_TDF"]["implementation_sha256"] is None
    assert parser["RSR"]["implementation_sha256"] is None
    calibration = rebuilt["frozen_contracts"]["calibration"]
    assert calibration["raw_to_calibrated_transform_sha256"] is None
    assert calibration["calibration_file_root_sha256"] is None
    assert calibration["held_out_target_use"] == "forbidden"


def test_source_class_theorem_is_bound_but_real_source_instantiation_is_missing(
    rebuilt: dict,
) -> None:
    physics = rebuilt["frozen_contracts"]["source_physics"]
    theorem = physics["source_class_theorem"]
    assert theorem["status"] == "pass"
    assert theorem["source_class_sha256"] == (
        "916a03069b800099de4319f07d4f544ed663db6ef49a9c19f4994cf09f1866c3"
    )
    assert theorem["coercivity_sha256"] == (
        "2cb1a4241ea280730a0ca6a40edef6a027e12d12ae035e7395649140a6b2d977"
    )
    instantiation = physics["real_source_interval_instantiation"]
    assert instantiation["status"] == "missing"
    assert instantiation["content_sha256"] is None
    assert set(instantiation["required_fields"]) == {
        "source_support_radius_upper",
        "total_mass_and_compactness",
        "trace_density_or_concentration_upper",
        "pressure_trace_sign",
        "static_geometry_intervals",
        "scalar_boundary_and_topology",
    }
    assert physics["candidate_use_status"] == (
        "blocked_real_source_interval_instantiation_not_bound"
    )


def test_state_nuisance_covariance_likelihood_split_and_stopping_are_frozen(
    rebuilt: dict,
) -> None:
    contracts = rebuilt["frozen_contracts"]
    initial = contracts["initial_state_inference"]
    assert initial["fit_role"] == "training_sessions_only"
    assert initial["validation_or_test_refit"] is False
    assert initial["checkpoint_sha256"] is None
    nuisance = contracts["nuisance_model"]
    assert nuisance["gravity_nuisances"] == []
    assert nuisance["object_specific_gravity_parameter_count"] == 0
    assert nuisance["post_hoc_rescue"] == "forbidden"
    covariance = contracts["covariance"]
    assert covariance["implementation_sha256"] is None
    assert len(covariance["required_components"]) == 5
    likelihood = contracts["likelihood"]
    assert likelihood["channels"] == OUTPUT_CHANNELS
    assert likelihood["model_dependent_residual_targets"] is False
    assert likelihood["implementation_sha256"] is None
    split = contracts["session_split"]
    assert split["unit"] == "tracking pass or observing session"
    assert split["group_leakage_forbidden"] is True
    assert split["selection_before_target_access"] is True
    assert split["commitment_sha256"] is None
    stopping = contracts["stopping_rule"]
    assert stopping["maximum_candidate_actions"] == 1
    assert stopping["maximum_test_openings"] == 1
    assert stopping["test_failure_rescue_iterations"] == 0


def test_descriptor_shape_is_complete_but_required_registration_values_are_unset(
    rebuilt: dict,
) -> None:
    descriptor = rebuilt["descriptor_template"]
    assert descriptor["schema_version"] == DESCRIPTOR_SCHEMA
    assert set(descriptor) == {
        "schema_version",
        "descriptor_id",
        "candidate",
        "formal_pass_binding",
        "prediction_audit_binding",
        "solar_protocol_binding",
        "prediction_bundle",
        "reviewed_evaluator",
        "observational_opening",
        "data_eligibility",
    }
    assert rebuilt["descriptor_shape_status"] == "all_required_fields_present"
    assert rebuilt["descriptor_registration_status"] == "blocked_required_values_unset"
    assert rebuilt["remaining_registration_fields"] == REMAINING_FIELDS
    assert descriptor["prediction_bundle"]["path"] is None
    assert descriptor["prediction_bundle"]["weak_field_solution_sha256"] is None
    assert descriptor["prediction_bundle"]["split_commitment_sha256"] is None
    assert set(descriptor["reviewed_evaluator"].values()) == {None}
    assert descriptor["observational_opening"]["authorized"] is False
    assert rebuilt["candidate_use_authorized"] is False
    assert rebuilt["formula_search_authorized"] is False


def test_restart_registry_is_idempotent_and_rejects_changed_replay(
    tmp_path: Path, rebuilt: dict
) -> None:
    database = tmp_path / "registry" / "template.sqlite"
    registry = G4CandidateSolarTemplateRegistry(database)
    first = registry.checkpoint(rebuilt)
    assert first["replay"] is False
    restarted = G4CandidateSolarTemplateRegistry(database)
    second = restarted.checkpoint(rebuilt)
    assert second["replay"] is True
    status = restarted.status()
    assert status["state"] == "unregistered_ineligible"
    assert status["checkpoint_count"] == 1
    assert status["content_sha256"] == rebuilt["content_sha256"]
    assert status["remaining_registration_field_count"] == len(REMAINING_FIELDS)
    assert status["observational_data_opened"] is False

    tampered = copy.deepcopy(rebuilt)
    tampered["candidate"]["action_sha256"] = "0" * 64
    tampered_body = {
        key: item for key, item in tampered.items() if key != "content_sha256"
    }
    tampered["content_sha256"] = _sha(tampered_body)
    with pytest.raises(ValueError, match="replay changed"):
        restarted.checkpoint(tampered)


def test_config_cannot_select_files_or_authorize_observations_early(tmp_path: Path) -> None:
    config = _load(CONFIG)
    config["observational_authorization"] = True
    with pytest.raises(ValueError, match="opened forbidden data"):
        build_g4_candidate_solar_protocol_template(config, ROOT)

    config = _load(CONFIG)
    config["source_selection"]["selected_primary_files"] = [
        {"path": "target.dat", "sha256": "0" * 64}
    ]
    with pytest.raises(ValueError, match="selection is not sealed"):
        build_g4_candidate_solar_protocol_template(config, ROOT)

    with pytest.raises(ValueError, match="live campaign"):
        G4CandidateSolarTemplateRegistry(
            tmp_path / "campaign-v1-live.sqlite" / "template.sqlite"
        )


def test_artifact_file_hash_is_stable() -> None:
    assert hashlib.sha256(ARTIFACT.read_bytes()).hexdigest() == (
        "8425d6491676b3e90e93b6b403c1f54d667a7032b87bbc16d711c430faee8a4f"
    )
