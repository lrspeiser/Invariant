import copy
import hashlib
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.covariant_grammar_v3_seed_campaign import (
    build_covariant_grammar_v3_seed_manifest,
    evaluate_pre_generation_constraints,
    iter_scalable_seed_specs,
)
from sigma_theory_compiler.promotion_orchestrator import ELIGIBILITY

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "covariant_grammar_v3_seed_campaign.json"
ARTIFACT = ROOT / "runs" / "engine" / "covariant-grammar-v3-seed-manifest.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_old_generic_unit_vector_q_shapes_are_rejected_before_generation() -> None:
    config = _load(CONFIG)
    for control in config["negative_controls"][:4]:
        result = evaluate_pre_generation_constraints(
            control["seed_spec"], config["pre_generation_constraints"]
        )
        assert result["decision"] == "reject"
        assert "generic_unit_vector_q_operator_excluded_by_70_reject_taxonomy" in result[
            "reasons"
        ]


def test_enabled_families_have_adapters_and_no_q_or_forbidden_matter_atoms() -> None:
    config = _load(CONFIG)
    enabled = [item for item in config["typed_family_seeds"] if item["enabled_for_generation"]]
    assert len(enabled) == 4
    for family in enabled:
        assert "Q_a_u" not in family["invariants"]
        assert "z_b" not in family["invariants"]
        assert family["hard_blockers"] == []
        assert any(adapter["available"] for adapter in family["formal_adapters"])


def test_enabled_seed_without_an_adapter_fails_closed() -> None:
    config = _load(CONFIG)
    seed = copy.deepcopy(config["typed_family_seeds"][0])
    seed["formal_adapters"] = []
    result = evaluate_pre_generation_constraints(seed, config["pre_generation_constraints"])
    assert result["decision"] == "reject"
    assert result["reasons"] == ["enabled_seed_has_no_available_formal_adapter"]


def test_artifact_is_exact_and_scalable_hook_is_deterministic() -> None:
    artifact = _load(ARTIFACT)
    body = {key: value for key, value in artifact.items() if key != "content_sha256"}
    assert artifact["content_sha256"] == hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert artifact["rejection_taxonomy"] == {
        "production_candidate_count": 70,
        "formal_reject_count": 70,
        "formal_pass_count": 0,
        "remaining_blocked_count": 0,
        "generation_rule": (
            "Q_a_u is excluded from the generic unit-vector generator; old F(X_a_u,Q_a_u) "
            "shapes cannot be emitted by grammar-v3"
        ),
    }
    assert artifact["known_answer_control_counts"] == {
        "certified_viable_known_answer": 3,
        "known_answer_blocked_negative_control": 1,
        "promotion_pass_known_answer_with_declared_health_gap": 1,
    }
    assert artifact["typed_family_counts"] == {"disabled": 3, "enabled": 4}
    assert artifact["negative_control_counts"] == {"reject": 5}
    seeds = list(iter_scalable_seed_specs(artifact))
    assert len(seeds) == 6
    assert [item["seed_id"] for item in seeds] == sorted(
        item["seed_id"] for item in seeds
    )
    assert len({item["seed_lineage_sha256"] for item in seeds}) == 6
    assert all(item["data_eligibility"] == ELIGIBILITY for item in seeds)


def test_manifest_rebuilds_exactly_from_hash_bound_inputs() -> None:
    assert build_covariant_grammar_v3_seed_manifest(_load(CONFIG), ROOT) == _load(ARTIFACT)


def test_tampered_known_answer_artifact_hash_is_rejected() -> None:
    config = _load(CONFIG)
    config["known_answer_controls"][0]["action_artifact"]["file_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="artifact file hash mismatch"):
        build_covariant_grammar_v3_seed_manifest(config, ROOT)


def test_declared_available_adapter_must_resolve_to_a_callable() -> None:
    config = _load(CONFIG)
    config["typed_family_seeds"][0]["formal_adapters"][0]["entrypoint"] = (
        "sigma_theory_compiler.action_health:not_a_real_adapter"
    )
    with pytest.raises(TypeError, match="not callable"):
        build_covariant_grammar_v3_seed_manifest(config, ROOT)
