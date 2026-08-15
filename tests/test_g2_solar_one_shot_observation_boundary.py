from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path

import pytest

from sigma_theory_compiler.g2_solar_one_shot_observation_boundary import (
    CONFIG_PATH,
    OUTPUT_PATH,
    SOURCE_PATH,
    TEST_PATH,
    SolarOneShotBoundaryError,
    build_boundary,
    validate_boundary,
)
from sigma_theory_compiler.sigma_core import canonical_json_bytes, canonical_sha256

ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _write_exact(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def _write_legacy(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _reseal_legacy(value: dict[str, object]) -> None:
    value.pop("content_sha256", None)
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    value["content_sha256"] = hashlib.sha256(encoded).hexdigest()


def _snapshot_sha(config: dict[str, object]) -> str:
    return canonical_sha256(
        {
            key: config[key]
            for key in (
                "source_bindings",
                "candidate_ids",
                "required_opening_fields",
                "execution_contract",
                "policies",
            )
        }
    )


def _copy_root(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    config = _load(ROOT / CONFIG_PATH)
    bindings = config["source_bindings"]
    paths = {CONFIG_PATH, SOURCE_PATH, TEST_PATH}
    paths.update(str(item["path"]) for item in bindings.values())
    for relative in paths:
        source = ROOT / relative
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    return tmp_path, config


def _update_artifact(
    root: Path,
    config: dict[str, object],
    role: str,
    artifact: dict[str, object],
) -> None:
    descriptor = config["source_bindings"][role]
    path = root / descriptor["path"]
    _write_legacy(path, artifact)
    descriptor["file_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    descriptor["content_sha256"] = artifact["content_sha256"]
    config["snapshot_sha256"] = _snapshot_sha(config)
    _write_exact(root / CONFIG_PATH, config)


@pytest.fixture(scope="module")
def result() -> dict[str, object]:
    return build_boundary(root=ROOT)


def test_current_boundary_is_strong_block(result: dict[str, object]) -> None:
    assert result["decision"] == "block"
    assert result["first_blocker"] == "independent_observation_opening_authorization_absent"
    assert result["counts"] == {
        "candidates": 2,
        "candidate_blocks": 2,
        "candidate_passes": 0,
        "opening_obligations_per_candidate": 4,
        "missing_opening_obligations": 8,
        "unique_missing_opening_fields": 4,
        "registered_catalog_metadata_files": 16,
        "primary_record_accesses": 0,
        "held_out_target_accesses": 0,
        "real_data_evaluations": 0,
    }


def test_exact_four_opening_fields_are_missing_for_each_candidate(
    result: dict[str, object],
) -> None:
    expected = [
        "source_branch_domain_instantiation_sha256",
        "held_out_split_commitment_sha256",
        "selected_primary_record_roots_sha256",
        "observation_opening_authorization_sha256",
    ]
    assert result["missing_opening_fields"] == sorted(expected)
    for candidate in result["candidate_results"]:
        assert [item["field"] for item in candidate["opening_obligations"]] == expected
        assert all(item["present"] is False for item in candidate["opening_obligations"])
        assert candidate["decision"] == "block"


def test_action_bound_contracts_are_preserved(result: dict[str, object]) -> None:
    assert [item["candidate_id"] for item in result["candidate_results"]] == [
        "G3A-2f8983c88f504150381064f2",
        "G3A-58e59412e5fe77cd54caf863",
    ]
    for candidate in result["candidate_results"]:
        for key in (
            "action_sha256",
            "bundle_content_sha256",
            "evaluator_descriptor_sha256",
            "initial_state_contract_sha256",
            "nuisance_likelihood_stopping_sha256",
            "source_contract_sha256",
        ):
            assert len(candidate[key]) == 64


def test_no_observation_or_primary_record_was_opened(result: dict[str, object]) -> None:
    boundary = result["data_boundary"]
    assert boundary["observational_data_opened"] is False
    assert boundary["primary_record_access_count"] == 0
    assert boundary["held_out_target_access_count"] == 0
    assert boundary["real_data_evaluation_count"] == 0
    assert boundary["candidate_use_authorized"] is False
    assert all(item["real_data_result"] is None for item in result["candidate_results"])


def test_quantity_classes_remain_distinct(result: dict[str, object]) -> None:
    contract = result["one_shot_execution_contract"]
    assert contract["allowed_quantity_classes"] == ["raw", "calibrated", "derived"]
    assert contract["forbidden_quantity_classes"] == ["latent", "model_dependent"]
    assert result["data_boundary"]["raw_records"] == "sealed"
    assert result["data_boundary"]["derived_records"] == "not_computed"
    assert result["data_boundary"]["model_dependent_records_used_as_truth"] is False


def test_dark_matter_and_redshift_are_excluded(result: dict[str, object]) -> None:
    assert result["data_boundary"]["dark_matter_or_halo_inputs"] is False
    assert result["data_boundary"]["redshift_distance_inputs"] is False
    assert result["readiness_checks"]["dark_matter_and_redshift_exclusion"] == "pass"


def test_execution_contract_is_deterministic_and_not_yet_executable(
    result: dict[str, object],
) -> None:
    contract = result["one_shot_execution_contract"]
    assert contract["atomic_target_open_batches"] == 1
    assert contract["candidate_evaluations"] == 2
    assert contract["refits_after_open"] == 0
    assert contract["promotion_actions"] == 0
    assert contract["status"] == ("sealed_not_executable_until_all_opening_obligations_are_bound")
    assert contract["evaluation_order"] == [
        "G3A-2f8983c88f504150381064f2",
        "G3A-58e59412e5fe77cd54caf863",
    ]


def test_supporting_parser_and_calibration_gaps_are_recorded(
    result: dict[str, object],
) -> None:
    supporting = result["supporting_missing_fields"]
    assert "selected_primary_file_root_sha256" in supporting["parser"]
    assert "tracking_session_split_commitment_sha256" in supporting["parser"]
    assert "selected_primary_file_root_sha256" in supporting["calibration"]
    assert "reviewed_candidate_solar_evaluator_descriptor_sha256" in supporting["calibration"]


def test_claims_do_not_convert_block_into_data_result(result: dict[str, object]) -> None:
    assert result["claims"] == {
        "one_shot_execution_ready": False,
        "observation_opening_authorized": False,
        "observational_result_exists": False,
        "candidate_rejected_by_data": False,
        "candidate_supported_by_data": False,
        "truth_established": False,
        "promotion_authorized": False,
    }


def test_live_validation_replays_all_bound_evidence(result: dict[str, object]) -> None:
    validate_boundary(result, root=ROOT)


def test_checked_receipt_matches_live_replay(result: dict[str, object]) -> None:
    checked = _load(ROOT / OUTPUT_PATH)
    assert canonical_json_bytes(checked) == canonical_json_bytes(result)


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("decision",), "pass"),
        (("counts", "real_data_evaluations"), 1),
        (("claims", "promotion_authorized"), True),
        (("data_boundary", "observational_data_opened"), True),
    ],
)
def test_resealed_result_tampering_fails_live_replay(
    result: dict[str, object], path: tuple[object, ...], replacement: object
) -> None:
    tampered = copy.deepcopy(result)
    cursor: object = tampered
    for key in path[:-1]:
        cursor = cursor[key]  # type: ignore[index]
    cursor[path[-1]] = replacement  # type: ignore[index]
    body = {key: value for key, value in tampered.items() if key != "content_sha256"}
    tampered["content_sha256"] = canonical_sha256(body)
    with pytest.raises(SolarOneShotBoundaryError, match="exact live replay"):
        validate_boundary(tampered, root=ROOT)


def test_unknown_config_key_fails_closed(tmp_path: Path) -> None:
    root, config = _copy_root(tmp_path)
    config["unknown"] = True
    _write_exact(root / CONFIG_PATH, config)
    with pytest.raises(SolarOneShotBoundaryError, match="config keys changed"):
        build_boundary(root=root)


def test_required_opening_field_cannot_be_removed(tmp_path: Path) -> None:
    root, config = _copy_root(tmp_path)
    config["required_opening_fields"].pop()
    config["snapshot_sha256"] = _snapshot_sha(config)
    _write_exact(root / CONFIG_PATH, config)
    with pytest.raises(SolarOneShotBoundaryError, match="opening field registry changed"):
        build_boundary(root=root)


def test_one_shot_budget_cannot_be_expanded(tmp_path: Path) -> None:
    root, config = _copy_root(tmp_path)
    config["execution_contract"]["atomic_target_open_batches"] = 2
    config["snapshot_sha256"] = _snapshot_sha(config)
    _write_exact(root / CONFIG_PATH, config)
    with pytest.raises(SolarOneShotBoundaryError, match="execution contract changed"):
        build_boundary(root=root)


def test_source_hash_tamper_fails_before_audit(tmp_path: Path) -> None:
    root, _ = _copy_root(tmp_path)
    path = root / "configs/observational_evidence_policy.json"
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(SolarOneShotBoundaryError, match="bound source changed"):
        build_boundary(root=root)


def test_resealed_false_authorization_tamper_fails_closed(tmp_path: Path) -> None:
    root, config = _copy_root(tmp_path)
    transfer = _load(root / "runs/engine/g2-solar-heldout-transfer-registration.json")
    transfer["observational_authorization"] = True
    _reseal_legacy(transfer)
    _update_artifact(root, config, "transfer_artifact", transfer)
    with pytest.raises(SolarOneShotBoundaryError, match="transfer/readiness boundary changed"):
        build_boundary(root=root)


def test_resealed_bundle_root_tamper_fails_unexpected_authorization(tmp_path: Path) -> None:
    root, config = _copy_root(tmp_path)
    bundle = _load(
        root
        / "runs/engine/g2-solar-action-bound-prediction-bundles/G3A-2f8983c88f504150381064f2.json"
    )
    bundle["descriptor"]["selected_primary_record_roots_sha256"] = "a" * 64
    _reseal_legacy(bundle["descriptor"])
    _reseal_legacy(bundle)
    _update_artifact(root, config, "bundle_G3A_2f8983", bundle)
    with pytest.raises(SolarOneShotBoundaryError, match="unexpectedly authorizes opening"):
        build_boundary(root=root)


def test_no_network_runtime_sqlite_secret_or_d2_dependency(result: dict[str, object]) -> None:
    policies = result["source_bindings"]["evidence"]
    paths = [item["path"] for item in policies.values()]
    assert all(not path.endswith((".sqlite", ".sqlite-wal", ".sqlite-shm")) for path in paths)
    assert all("quartic_registered_direction" not in path for path in paths)
    assert result["one_shot_execution_contract"]["no_refit_after_open"] is True
    assert result["one_shot_execution_contract"]["no_promotion_side_effect"] is True
