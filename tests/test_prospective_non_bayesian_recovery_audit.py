from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path

import pytest

from sigma_theory_compiler.prospective_non_bayesian_recovery_audit import (
    CONFIG_PATH,
    OUTPUT_PATH,
    SOURCE_PATH,
    TEST_PATH,
    NonBayesianRecoveryError,
    build_recovery_audit,
    validate_recovery_audit,
)
from sigma_theory_compiler.sigma_core import canonical_json_bytes, canonical_sha256

ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _write(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def _reseal(value: dict[str, object]) -> None:
    value.pop("content_sha256", None)
    value["content_sha256"] = canonical_sha256(value)


def _snapshot_sha(config: dict[str, object]) -> str:
    return canonical_sha256(
        {
            key: config[key]
            for key in (
                "source_bindings",
                "excluded_families",
                "repair_families",
                "repair_contract",
                "frozen_worlds",
            )
        }
    )


def _copy_root(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    config = _load(ROOT / CONFIG_PATH)
    bindings = config["source_bindings"]
    assert isinstance(bindings, dict)
    paths = {CONFIG_PATH, SOURCE_PATH, TEST_PATH}
    paths.update(str(item["path"]) for item in bindings.values())
    for relative in paths:
        source = ROOT / relative
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    return tmp_path, config


def _update_bound_artifact(
    root: Path,
    config: dict[str, object],
    role: str,
    artifact: dict[str, object],
) -> None:
    bindings = config["source_bindings"]
    descriptor = bindings[role]
    path = root / descriptor["path"]
    _write(path, artifact)
    descriptor["file_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    descriptor["content_sha256"] = artifact["content_sha256"]
    config["snapshot_sha256"] = _snapshot_sha(config)
    _write(root / CONFIG_PATH, config)


@pytest.fixture(scope="module")
def result() -> dict[str, object]:
    return build_recovery_audit(root=ROOT)


def test_audit_has_honest_terminal_decision(result: dict[str, object]) -> None:
    assert result["decision"] in {
        "pass_at_least_one_non_bayesian_family_recovered_a_frozen_pass_world",
        "reject_no_non_bayesian_recovery_within_fixed_repair_budget",
    }
    assert result["counts"]["worlds"] == 2
    assert result["counts"]["non_bayesian_families"] == 6
    assert result["counts"]["original_failed_family_worlds"] == 12


def test_all_original_non_bayesian_failures_are_diagnosed(
    result: dict[str, object],
) -> None:
    diagnoses = result["original_failure_diagnoses"]
    assert len(diagnoses) == 12
    assert all(item["hard_exact"] == "pass" for item in diagnoses)
    assert all(item["hard_holdout"] == "reject" for item in diagnoses)
    assert all("not an incapacity" in item["diagnosis_scope"] for item in diagnoses)


def test_bayesian_is_excluded_from_every_repair_surface(result: dict[str, object]) -> None:
    assert result["preregistration"]["excluded_families"] == ["bayesian"]
    assert result["claims"]["bayesian_excluded"] is True
    phase = result["target_blind_generation"]
    for world in phase["worlds"]:
        assert [item["family"] for item in world["families"]] == [
            "cross_domain",
            "egraph",
            "evolutionary",
            "grammar",
            "llm",
            "symbolic",
        ]


def test_phase_a_is_target_blind_and_sealed(result: dict[str, object]) -> None:
    phase = result["target_blind_generation"]
    assert phase["generation_inputs"] == [
        "family",
        "frozen_candidate_ref",
        "ordinal",
        "public_world_sha256",
        "repair_id",
    ]
    assert phase["target_fields_read"] == []
    body = {key: value for key, value in phase.items() if key != "content_sha256"}
    assert phase["content_sha256"] == canonical_sha256(body)
    serialized = canonical_json_bytes(phase)
    assert b"a2464946dbc725622418a0871e03b4acb88da9737c3f1f4e70a9250c9f367ba7" not in serialized
    assert b"6dc10af7a574e74b07051e2c87975fae9abcba277fb7825aa620dd74c2358709" not in serialized


def test_fixed_budget_accounting_is_exact(result: dict[str, object]) -> None:
    assert result["counts"]["repair_candidates"] == 48
    assert result["counts"]["gate_checks"] == 96
    assert result["target_blind_generation"]["counts"] == {
        "worlds": 2,
        "families": 6,
        "repair_candidates": 48,
    }
    for world in result["target_blind_generation"]["worlds"]:
        assert all(item["candidate_count"] == 4 for item in world["families"])


def test_repaired_candidates_have_sigma_core_lineage(result: dict[str, object]) -> None:
    for world in result["target_blind_generation"]["worlds"]:
        for family in world["families"]:
            frozen = family["frozen_candidate"]
            for candidate in family["repair_candidates"]:
                assert candidate["provenance"]["inputs"] == [frozen]
                assert candidate["representation"]["frozen_candidate"] == frozen
                assert candidate["representation"]["target_fields_read"] == []


def test_family_world_results_are_exact_pass_reject_or_block(
    result: dict[str, object],
) -> None:
    decisions = []
    for world in result["world_results"]:
        for family in world["family_results"]:
            decisions.append(family["decision"])
            assert family["decision"] in {"pass", "reject", "block"}
            assert sum(family["counts"][key] for key in ("pass", "reject", "block")) == 4
    assert len(decisions) == 12


def test_preregistered_repair_recovers_both_worlds_without_bayesian(
    result: dict[str, object],
) -> None:
    assert result["decision"] == (
        "pass_at_least_one_non_bayesian_family_recovered_a_frozen_pass_world"
    )
    assert result["counts"]["family_world_passes"] == 3
    assert result["counts"]["family_world_rejects"] == 9
    assert result["counts"]["family_world_blocks"] == 0
    assert result["counts"]["recovered_worlds"] == 2
    assert result["recovered_world_ids"] == [
        "prospective.graph_parity",
        "prospective.modular_affine",
    ]
    assert result["recovering_families"] == ["egraph", "grammar", "llm"]
    passes = {
        (world["world_id"], family["family"])
        for world in result["world_results"]
        for family in world["family_results"]
        if family["decision"] == "pass"
    }
    assert passes == {
        ("prospective.graph_parity", "grammar"),
        ("prospective.graph_parity", "llm"),
        ("prospective.modular_affine", "egraph"),
    }


def test_recovery_claims_remain_conservative(result: dict[str, object]) -> None:
    assert result["claims"] == {
        "bayesian_excluded": True,
        "all_repairs_generated_before_frozen_target_replay": True,
        "repair_generation_target_fields_read": [],
        "new_target_access_performed": False,
        "post_design_tuning_performed": False,
        "prospective_success_established": False,
        "truth_established": False,
        "novelty_established": False,
        "promotion_authorized": False,
    }
    assert "already-unsealed" in result["scope"]


def test_live_validation_replays_all_bindings(result: dict[str, object]) -> None:
    validate_recovery_audit(result, root=ROOT)


def test_checked_receipt_matches_live_replay(result: dict[str, object]) -> None:
    checked = _load(ROOT / OUTPUT_PATH)
    assert canonical_json_bytes(checked) == canonical_json_bytes(result)


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("decision",), "block_ambiguous_repair_contract"),
        (("counts", "repair_candidates"), 47),
        (("claims", "truth_established"), True),
        (("target_blind_generation", "target_fields_read"), ["hypothesis"]),
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
    with pytest.raises(NonBayesianRecoveryError, match="exact live replay"):
        validate_recovery_audit(tampered, root=ROOT)


def test_unknown_config_key_fails_closed(tmp_path: Path) -> None:
    root, config = _copy_root(tmp_path)
    config["unknown"] = True
    _write(root / CONFIG_PATH, config)
    with pytest.raises(NonBayesianRecoveryError, match="config keys changed"):
        build_recovery_audit(root=root)


def test_bayesian_cannot_be_added_to_repair_registry(tmp_path: Path) -> None:
    root, config = _copy_root(tmp_path)
    config["repair_families"]["bayesian"] = "sha256_counter_inventory_breadth_v1"
    config["snapshot_sha256"] = _snapshot_sha(config)
    _write(root / CONFIG_PATH, config)
    with pytest.raises(NonBayesianRecoveryError, match="repair family registry changed"):
        build_recovery_audit(root=root)


def test_target_field_cannot_be_added_to_generation_contract(tmp_path: Path) -> None:
    root, config = _copy_root(tmp_path)
    config["repair_contract"]["target_fields"] = ["hypothesis"]
    config["snapshot_sha256"] = _snapshot_sha(config)
    _write(root / CONFIG_PATH, config)
    with pytest.raises(NonBayesianRecoveryError, match="repair contract changed"):
        build_recovery_audit(root=root)


def test_budget_or_ordinal_change_fails_closed(tmp_path: Path) -> None:
    root, config = _copy_root(tmp_path)
    config["repair_contract"]["ordinals"].append(4)
    config["snapshot_sha256"] = _snapshot_sha(config)
    _write(root / CONFIG_PATH, config)
    with pytest.raises(NonBayesianRecoveryError, match="repair contract changed"):
        build_recovery_audit(root=root)


def test_source_hash_tampering_fails_before_repair(tmp_path: Path) -> None:
    root, _ = _copy_root(tmp_path)
    path = root / "configs/prospective_tournament_robustness_ablation.json"
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(NonBayesianRecoveryError, match="bound source changed"):
        build_recovery_audit(root=root)


def test_resealed_candidate_tamper_fails_frozen_identity(tmp_path: Path) -> None:
    root, config = _copy_root(tmp_path)
    world = next(
        item for item in config["frozen_worlds"] if item["world_id"] == "prospective.graph_parity"
    )
    candidate = next(item for item in world["candidates"] if item["family"] == "grammar")
    candidate["hypothesis"] = 8
    config["snapshot_sha256"] = _snapshot_sha(config)
    _write(root / CONFIG_PATH, config)
    with pytest.raises(NonBayesianRecoveryError, match="candidate identities changed"):
        build_recovery_audit(root=root)


def test_resealed_target_tamper_fails_target_binding(tmp_path: Path) -> None:
    root, config = _copy_root(tmp_path)
    world = next(
        item for item in config["frozen_worlds"] if item["world_id"] == "prospective.graph_parity"
    )
    world["sealed_target_sha256"] = canonical_sha256(
        {"hypothesis": 5, "world_id": "prospective.graph_parity"}
    )
    config["snapshot_sha256"] = _snapshot_sha(config)
    _write(root / CONFIG_PATH, config)
    with pytest.raises(NonBayesianRecoveryError, match="frozen target binding changed"):
        build_recovery_audit(root=root)


def test_ablation_bayesian_dependence_tamper_fails_closed(tmp_path: Path) -> None:
    root, config = _copy_root(tmp_path)
    ablation = _load(root / "runs/math/prospective-tournament-robustness-ablation/campaign.json")
    record = next(item for item in ablation["ablations"] if item["removed_family"] == "bayesian")
    record["world_pass_to_reject_count"] = 1
    _reseal(ablation)
    _update_bound_artifact(root, config, "ablation_artifact", ablation)
    with pytest.raises(NonBayesianRecoveryError, match="Bayesian dependence premise changed"):
        build_recovery_audit(root=root)


def test_no_runtime_sqlite_network_or_d2_dependencies(result: dict[str, object]) -> None:
    bindings = result["source_bindings"]["frozen_inputs"]
    paths = [item["path"] for item in bindings.values()]
    assert all(not path.endswith((".sqlite", ".sqlite-wal", ".sqlite-shm")) for path in paths)
    assert all("quartic_registered_direction" not in path for path in paths)
    assert result["preregistration"]["policies"]["network_access"] == "forbidden"
    assert result["preregistration"]["policies"]["runtime_process_control"] == "forbidden"
