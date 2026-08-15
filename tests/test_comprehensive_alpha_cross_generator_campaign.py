from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

from sigma_theory_compiler.candidate_pareto_explanations import (
    MetricReceipt,
    ParetoLimits,
    validate_pareto_replay,
)
from sigma_theory_compiler.comprehensive_alpha_cross_generator_campaign import (
    CAMPAIGN_ID,
    DESCRIPTOR,
    FAMILIES,
    OUTPUT_PATH,
    _generate_all,
    _load_config,
    _write_immutable,
    run,
    validate_campaign,
)
from sigma_theory_compiler.sigma_core import CandidateArtifact, GateOutcome

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


def test_all_native_generator_families_emit_one_common_pack_candidate(
    checked: dict[str, object],
) -> None:
    config = _load_config(ROOT)
    assert tuple(config["generator_families"]) == FAMILIES
    assert [row["family"] for row in checked["candidate_family_bindings"]] == list(FAMILIES)
    candidates = [CandidateArtifact.from_dict(row) for row in checked["selected_candidates"]]
    assert len(candidates) == len({row.artifact_id for row in candidates}) == 7
    assert all(row.provenance.domain_pack == DESCRIPTOR.ref for row in candidates)
    assert set(checked["generator_receipts"]) == set(FAMILIES)
    replayed, receipts = _generate_all(ROOT)
    assert [replayed[family].to_dict() for family in FAMILIES] == checked["selected_candidates"]
    assert receipts == checked["generator_receipts"]


def test_two_stage_two_gate_coverage_preserves_exact_status_distribution(
    checked: dict[str, object],
) -> None:
    bindings = {
        row["candidate"]["artifact_id"]: row["family"]
        for row in checked["candidate_family_bindings"]
    }
    stages = checked["stage_outcomes"]
    assert len(stages) == 14
    assert Counter(row["stage_id"] for row in stages) == {"typed": 7, "exact": 7}
    assert {row["status"] for row in stages} == {"pass"}
    gates = [GateOutcome.from_dict(row) for row in checked["gate_outcomes"]]
    by_family = {
        family: {
            row.gate_id: row.status.value
            for row in gates
            if bindings[row.artifact.artifact_id] == family
        }
        for family in FAMILIES
    }
    assert by_family == _load_config(ROOT)["preregistered_gate_outcomes"]
    candidate_dispositions = Counter(
        "pass"
        if all(status == "pass" for status in statuses.values())
        else next(status for status in statuses.values() if status != "pass")
        for statuses in by_family.values()
    )
    assert candidate_dispositions == {"pass": 4, "block": 1, "reject": 1, "error": 1}
    assert checked["hard_gate_status_counts"] == {
        "block": 1,
        "error": 1,
        "pass": 11,
        "reject": 1,
    }


def test_only_all_pass_candidates_receive_metrics_fronts_and_explanations(
    checked: dict[str, object],
) -> None:
    assert checked["pareto_eligible_families"] == [
        "bayesian",
        "evolutionary",
        "grammar",
        "symbolic",
    ]
    eligible_ids = {
        row["candidate"]["artifact_id"]
        for row in checked["candidate_family_bindings"]
        if row["family"] in checked["pareto_eligible_families"]
    }
    metric_ids = {row["candidate"]["artifact_id"] for row in checked["metric_receipts"]}
    front_ids = {
        row["artifact_id"] for front in checked["pareto"]["pareto_fronts"] for row in front
    }
    assert metric_ids == front_ids == eligible_ids
    assert len(checked["metric_receipts"]) == 8
    excluded = checked["hard_gate_exclusion_explanations"]
    assert [row["family"] for row in excluded] == ["cross_domain", "egraph", "llm"]
    assert all(
        row["metric_receipts"] == []
        and row["pareto_front"] is None
        and row["truth_established"] is False
        and row["promotion_authorized"] is False
        for row in excluded
    )
    assert checked["pareto"]["counts"]["candidates"] == 4
    assert checked["pareto"]["counts"]["hard_gate_ineligible"] == 0


def test_exact_pareto_explanation_replay_is_independently_checkable(
    checked: dict[str, object],
) -> None:
    candidates = tuple(CandidateArtifact.from_dict(row) for row in checked["pareto"]["candidates"])
    gates = tuple(GateOutcome.from_dict(row) for row in checked["pareto"]["gate_outcomes"])
    metrics = tuple(MetricReceipt.from_dict(row) for row in checked["pareto"]["metric_receipts"])
    validate_pareto_replay(
        checked["pareto"],
        candidates,
        gates,
        metrics,
        required_gate_ids=("hard_exact", "hard_structure"),
        metric_directions={
            "provenance_inputs": "maximize",
            "representation_bytes": "minimize",
        },
        limits=ParetoLimits(7, 2, 2, 128),
    )
    assert checked["pareto"]["claims"] == {
        "truth_established": False,
        "equivalence_established": False,
        "novelty_established": False,
        "absence_established": False,
        "promotion_authorized": False,
    }


def test_campaign_claims_counts_and_source_provenance_fail_closed(
    checked: dict[str, object],
) -> None:
    assert checked["campaign_id"] == CAMPAIGN_ID
    assert checked["counts"] == {
        "generator_families": 7,
        "selected_candidates": 7,
        "stage_outcomes": 14,
        "gate_outcomes": 14,
        "pareto_eligible_candidates": 4,
        "hard_gate_excluded_candidates": 3,
        "metric_receipts": 8,
    }
    for claim in (
        "generator_output_establishes_truth",
        "heuristic_score_establishes_truth",
        "pareto_rank_establishes_truth",
        "novelty_established",
        "promotion_authorized",
        "external_benchmark_success_established",
    ):
        assert checked["claims"][claim] is False
    assert set(checked["source_bindings"]) == {"config", "source", "test"}
    for binding in checked["source_bindings"].values():
        path = ROOT / binding["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == binding["file_sha256"]


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("counts", "pareto_eligible_candidates"), 5),
        (("claims", "promotion_authorized"), True),
        (("hard_gate_exclusion_explanations", 0, "pareto_front"), 1),
        (("hard_gate_exclusion_explanations", 0, "metric_receipts"), ["forged"]),
        (("gate_outcomes", 0, "status"), "reject"),
        (("source_bindings", "test", "path"), "tests/forged.py"),
    ],
)
def test_resealed_semantic_mutations_fail_immutable_replay(
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


def test_writer_is_idempotent_and_refuses_replacement(tmp_path: Path) -> None:
    path = tmp_path / "campaign.json"
    _write_immutable(path, {"state": "integration_only"})
    before = path.read_bytes()
    _write_immutable(path, {"state": "integration_only"})
    assert path.read_bytes() == before
    with pytest.raises(FileExistsError, match="differs"):
        _write_immutable(path, {"state": "truth_claim"})
