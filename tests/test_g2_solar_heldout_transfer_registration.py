from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.g2_solar_heldout_transfer_registration import (
    BASE_MISSING_FIELDS,
    FILLED_FIELDS,
    REMAINING_FIELDS,
    _file_bytes,
    _sha,
    build_g2_solar_heldout_transfer_registration,
    evaluate_registration,
)
from sigma_theory_compiler.promotion_orchestrator import ELIGIBILITY

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "g2_solar_heldout_transfer_registration.json"
BUNDLE_DIR = ROOT / "runs" / "engine" / "g2-solar-action-bound-prediction-bundles"
ARTIFACT = ROOT / "runs" / "engine" / "g2-solar-heldout-transfer-registration.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def rebuilt() -> tuple[dict, dict]:
    return build_g2_solar_heldout_transfer_registration(_load(CONFIG), ROOT)


def test_committed_artifacts_are_exact_rebuilds(rebuilt: tuple[dict, dict]) -> None:
    bundle_files, artifact = rebuilt
    for relative, bundle in bundle_files.items():
        path = ROOT / relative
        assert bundle == _load(path)
        body = {key: item for key, item in bundle.items() if key != "content_sha256"}
        assert bundle["content_sha256"] == _sha(body)
        assert path.read_bytes() == _file_bytes(bundle)
    assert artifact == _load(ARTIFACT)
    body = {key: item for key, item in artifact.items() if key != "content_sha256"}
    assert artifact["content_sha256"] == _sha(body)
    assert ARTIFACT.read_bytes() == _file_bytes(artifact)
    expected_bundle_hashes = {
        "G3A-2f8983c88f504150381064f2.json": (
            "104e9fa344ff58aaa3fb055e341787ebb1a4499694b62e4d1fbfb132293ec8b0"
        ),
        "G3A-58e59412e5fe77cd54caf863.json": (
            "b7c1ddb06f04b1620ffa1b8bf64ba143f0e5b4510f16fe7eb9ad4cc5cbd82254"
        ),
    }
    assert {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in BUNDLE_DIR.glob("*.json")
    } == expected_bundle_hashes
    assert hashlib.sha256(ARTIFACT.read_bytes()).hexdigest() == (
        "e17db85a4b1864000907228defb8acc3382e9213671b629fa57d0ee8f51545d6"
    )


def test_registration_ledger_advances_six_fields_but_stays_blocked(
    rebuilt: tuple[dict, dict],
) -> None:
    _, artifact = rebuilt
    assert artifact["registration_advance_per_candidate"] == {
        "before_missing_field_count": 10,
        "filled_field_count": 6,
        "after_missing_field_count": 4,
        "filled_fields": FILLED_FIELDS,
        "remaining_fields": REMAINING_FIELDS,
    }
    assert artifact["decision_counts"] == {"blocked": 2}
    for registration in artifact["candidate_registrations"]:
        assert registration["base_missing_registration_fields"] == BASE_MISSING_FIELDS
        assert registration["filled_registration_fields"] == FILLED_FIELDS
        assert registration["remaining_registration_fields"] == REMAINING_FIELDS
        assert registration["evaluator_result"]["decision"] == "blocked"
        assert registration["evaluator_result"]["real_data_pass"] is False


def test_every_contract_and_bundle_is_bound_to_the_exact_candidate_action(
    rebuilt: tuple[dict, dict],
) -> None:
    bundle_files, artifact = rebuilt
    expected = {
        "G3A-2f8983c88f504150381064f2": (
            "19f36a7c814ca11ace6de1270802a542872c35c27c7e64542eea672e16cbae88"
        ),
        "G3A-58e59412e5fe77cd54caf863": (
            "9457ba1ff99ecfdabc08200dda3ff15b8656b025d106fe2c2cd4abd77a01c3b5"
        ),
    }
    assert {
        item["candidate_id"]: item["action_sha256"] for item in bundle_files.values()
    } == expected
    for registration in artifact["candidate_registrations"]:
        candidate_id = registration["candidate_id"]
        action = expected[candidate_id]
        assert registration["action_sha256"] == action
        for name in (
            "source_contract",
            "training_only_initial_state",
            "nuisance_likelihood_stopping",
            "evaluator_descriptor",
        ):
            assert registration["contracts"][name]["candidate_id"] == candidate_id
            assert registration["contracts"][name]["action_sha256"] == action


def test_quantity_classes_and_forbidden_inputs_are_fail_closed(
    rebuilt: tuple[dict, dict],
) -> None:
    _, artifact = rebuilt
    for registration in artifact["candidate_registrations"]:
        classes = registration["contracts"]["source_contract"]["quantity_classes"]
        assert set(classes) == {"raw", "calibrated", "derived", "model_dependent", "latent"}
        assert classes["raw"]["allowed"] is True
        assert classes["calibrated"]["allowed"] is True
        assert classes["derived"]["allowed"] is True
        assert classes["model_dependent"]["allowed_as_input_or_target"] is False
        assert classes["latent"]["allowed_as_input_or_target"] is False
        nuisance = registration["contracts"]["nuisance_likelihood_stopping"]["nuisance"]
        assert nuisance["gravity_nuisances"] == []
        assert nuisance["object_specific_gravity_parameter_count"] == 0
        assert nuisance["post_hoc_rescue"] == "forbidden"
    assert artifact["dark_matter_or_halo_inputs"] is False
    assert artifact["redshift_distance_inputs"] is False
    assert artifact["data_eligibility"] == ELIGIBILITY


def test_real_source_split_roots_and_authorization_are_not_fabricated(
    rebuilt: tuple[dict, dict],
) -> None:
    bundle_files, artifact = rebuilt
    for registration in artifact["candidate_registrations"]:
        hashes = registration["registration_hashes"]
        assert all(
            isinstance(hashes[name], str) and len(hashes[name]) == 64 for name in FILLED_FIELDS
        )
        assert all(hashes[name] is None for name in REMAINING_FIELDS)
        split = registration["contracts"]["split_commitment_template"]
        assert split["selected_session_ids"] == []
        assert split["commitment_sha256"] is None
        assert split["selection_before_target_access"] is True
    assert all(
        bundle["real_source_prediction_generated"] is False for bundle in bundle_files.values()
    )
    assert all(bundle["held_out_targets_opened"] is False for bundle in bundle_files.values())
    assert artifact["primary_record_access_count"] == 0
    assert artifact["held_out_target_access_count"] == 0
    assert artifact["candidate_use_authorized"] is False
    assert artifact["observational_authorization"] is False
    assert artifact["observational_data_opened"] is False
    assert artifact["real_data_pass_count"] == 0


def test_prediction_bundles_are_analytic_controls_not_real_data(
    rebuilt: tuple[dict, dict],
) -> None:
    bundle_files, artifact = rebuilt
    assert len(artifact["prediction_bundles"]) == 2
    assert len(bundle_files) == 2
    for relative, bundle in bundle_files.items():
        spec = next(item for item in artifact["prediction_bundles"] if item["path"] == relative)
        assert spec["file_sha256"] == hashlib.sha256(_file_bytes(bundle)).hexdigest()
        assert spec["content_sha256"] == bundle["content_sha256"]
        model = bundle["analytic_forward_model"]
        assert model["PPN_prediction"]["gamma"] == "1"
        assert model["PPN_prediction"]["beta"] == "1"
        assert model["role"] == "action_bound_analytic_forward_model_not_observational_evidence"
        assert bundle["real_source_prediction_generated"] is False
        assert bundle["held_out_targets_opened"] is False
        assert bundle["decision"] == "blocked"
    assert artifact["synthetic_controls"]["analytic_identity"]["decision"] == "pass"
    assert artifact["synthetic_controls"]["fail_closed_evaluator"]["real_data_pass_count"] == 0


def test_registration_tampering_fails_closed(rebuilt: tuple[dict, dict]) -> None:
    _, artifact = rebuilt
    registration = artifact["candidate_registrations"][0]
    wrong_action = copy.deepcopy(registration)
    wrong_action["registration_hashes"][FILLED_FIELDS[0]] = "bad"
    with pytest.raises(ValueError, match="invalid filled hash"):
        evaluate_registration(wrong_action)

    premature = copy.deepcopy(registration)
    premature["registration_hashes"][REMAINING_FIELDS[0]] = "0" * 64
    with pytest.raises(ValueError, match="cannot be filled"):
        evaluate_registration(premature)


def test_config_authorization_binding_and_path_escape_fail_closed(tmp_path: Path) -> None:
    config = _load(CONFIG)
    opened = copy.deepcopy(config)
    opened["observational_authorization"] = True
    with pytest.raises(ValueError, match="opened forbidden data"):
        build_g2_solar_heldout_transfer_registration(opened, ROOT)

    changed = copy.deepcopy(config)
    changed["bindings"]["g2_readiness"]["file_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="file hash mismatch"):
        build_g2_solar_heldout_transfer_registration(changed, ROOT)

    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    escaped = copy.deepcopy(config)
    escaped["bindings"]["g2_readiness"] = {
        "path": str(outside),
        "file_sha256": hashlib.sha256(outside.read_bytes()).hexdigest(),
    }
    with pytest.raises(ValueError, match="path escapes repository"):
        build_g2_solar_heldout_transfer_registration(escaped, ROOT)
