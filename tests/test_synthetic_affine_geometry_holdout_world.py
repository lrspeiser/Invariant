from __future__ import annotations

import copy
import hashlib
import inspect
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.synthetic_affine_geometry_holdout_world import (
    BENCHMARK_ID,
    CONFIG_PATH,
    OUTPUT_PATH,
    _discover,
    _evaluate_bound,
    _load_config,
    _select_order,
    _write_immutable,
    run,
    validate_campaign,
)

ROOT = Path(__file__).resolve().parents[1]


def _canonical_sha(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def _reseal(value: dict[str, object]) -> None:
    value["content_sha256"] = _canonical_sha(
        {key: item for key, item in value.items() if key != "content_sha256"}
    )


@pytest.fixture(scope="module")
def checked() -> dict[str, object]:
    value = json.loads((ROOT / OUTPUT_PATH).read_text(encoding="utf-8"))
    validate_campaign(value, ROOT)
    assert value == run(ROOT)
    return value


def test_seeded_affine_world_has_closed_incidence_inventory(checked: dict[str, object]) -> None:
    config = _load_config(ROOT)
    assert _select_order(config) == 5
    world = checked["pre_unseal"]["public_input"]["world"]
    assert world["order"] == 5
    assert len(world["points"]) == 25
    assert len({tuple(point) for point in world["points"]}) == 25
    assert len(world["lines"]) == 30
    assert len({line["line_id"] for line in world["lines"]}) == 30
    assert all(len(line["points"]) == 5 for line in world["lines"])
    assert all(len({tuple(point) for point in line["points"]}) == 5 for line in world["lines"])
    assert world["candidate_intersection_bounds"] == list(range(6))


def test_public_only_discovery_exhausts_bounds_and_all_line_pairs(
    checked: dict[str, object],
) -> None:
    assert list(inspect.signature(_discover).parameters) == ["public_world"]
    pre = checked["pre_unseal"]
    assert pre["reference_payload_supplied_to_discovery"] is False
    assert pre["reference_theorem_id_absent"] is True
    assert pre["reference_seal_absent"] is True
    assert pre["discovery_callable_parameters"] == ["public_world"]
    discovery = pre["discovery"]
    assert discovery["grammar"] == {
        "candidate_kind": "nonnegative_integer_intersection_bound",
        "minimum": 0,
        "maximum": 5,
        "candidate_count": 6,
    }
    assert discovery["passing_bounds"] == [1, 2, 3, 4, 5]
    assert discovery["winner"]["bound"] == 1
    assert discovery["winner"]["line_pairs_checked"] == 435
    assert discovery["winner"]["passed"] is True
    assert all(row["intersection_size"] in {0, 1} for row in discovery["winner"]["rows"])
    assert _evaluate_bound(pre["public_input"]["world"], 0)["passed"] is False


def test_exact_pair_proof_chronology_and_post_unseal_match(checked: dict[str, object]) -> None:
    assert checked["proof"] == {
        "method": "exhaustive_unordered_distinct_line_pair_replay",
        "line_count": 30,
        "unordered_distinct_line_pairs": 435,
        "candidate_bounds_checked": 6,
        "total_pair_bound_obligations": 2610,
        "winning_bound": 1,
        "winning_pair_counterexamples": 0,
        "minimality_counterexamples": [{"bound": 0, "first_failing_pair": ["s0-b0", "s1-b0"]}],
    }
    assert [row["ordinal"] for row in checked["chronology"]] == list(range(1, 7))
    assert [row["phase"] for row in checked["chronology"]] == [
        "public_affine_world_generated",
        "reference_incidence_theorem_sealed",
        "public_discovery_input_sealed",
        "bounded_intersection_grammar_enumerated",
        "winner_and_pairwise_proof_sealed",
        "reference_unsealed_and_compared",
    ]
    comparison = checked["post_unseal"]["comparison"]
    assert comparison["performed_after_winner_seal"] is True
    assert comparison["reference_bound"] == comparison["rediscovered_bound"] == 1
    assert comparison["exact_match"] is True


def test_negative_controls_reject_wrong_or_incomplete_evidence(checked: dict[str, object]) -> None:
    controls = {row["control_id"]: row for row in checked["negative_controls"]}
    assert set(controls) == {
        "zero_bound_rejected",
        "parallel_only_truncation_rejected",
        "repeated_line_identity_rejected",
    }
    assert controls["zero_bound_rejected"]["rejected"] is True
    truncated = controls["parallel_only_truncation_rejected"]
    assert truncated["truncated_pair_count"] == 50
    assert truncated["truncated_check_would_pass_zero_bound"] is True
    assert truncated["full_replay_rejected_zero_bound"] is True
    assert controls["repeated_line_identity_rejected"]["rejected"] is True


def test_claims_scope_and_provenance_remain_bounded(checked: dict[str, object]) -> None:
    assert checked["benchmark_id"] == BENCHMARK_ID
    assert checked["decision_counts"] == {"pass": 1, "reject": 0, "blocked": 0}
    for claim in (
        "general_geometry_completeness_established",
        "unbounded_geometry_discovery_established",
        "historical_novelty_established",
        "formal_proof_assistant_kernel_checked",
        "hostile_process_isolation_established",
        "external_mathematical_significance_established",
    ):
        assert checked["claims"][claim] is False
    assert "one deterministic anonymous affine incidence plane of prime order 5" in checked["scope"]
    assert set(checked["source_bindings"]) == {"config", "source", "test"}
    for binding in checked["source_bindings"].values():
        path = ROOT / binding["path"]
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == binding["file_sha256"]


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("proof", "unordered_distinct_line_pairs"), 434),
        (("pre_unseal", "reference_theorem_id_absent"), False),
        (("post_unseal", "comparison", "exact_match"), False),
        (("claims", "historical_novelty_established"), True),
        (("source_bindings", "test", "path"), "tests/forged.py"),
        (("negative_controls", 0, "rejected"), False),
    ],
)
def test_resealed_semantic_mutations_fail_replay(
    checked: dict[str, object], path: tuple[object, ...], replacement: object
) -> None:
    forged = copy.deepcopy(checked)
    target = forged
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = replacement
    _reseal(forged)
    with pytest.raises(ValueError, match="contract changed|immutable replay mismatch"):
        validate_campaign(forged, ROOT)


def test_unknown_key_config_mutation_and_writer_fail_closed(
    checked: dict[str, object], tmp_path: Path
) -> None:
    forged = copy.deepcopy(checked)
    forged["unsupported"] = True
    _reseal(forged)
    with pytest.raises(ValueError, match="result keys changed"):
        validate_campaign(forged, ROOT)

    config = json.loads((ROOT / CONFIG_PATH).read_text(encoding="utf-8"))
    config["world_generator"]["prime_order_candidates"].append(11)
    config_path = tmp_path / CONFIG_PATH
    config_path.parent.mkdir(parents=True)
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="config contract changed"):
        _load_config(tmp_path)

    artifact = tmp_path / "campaign.json"
    _write_immutable(artifact, {"state": "bounded_pass"})
    before = artifact.read_bytes()
    _write_immutable(artifact, {"state": "bounded_pass"})
    assert artifact.read_bytes() == before
    with pytest.raises(FileExistsError, match="differs"):
        _write_immutable(artifact, {"state": "false_success"})
