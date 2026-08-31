from __future__ import annotations

import copy
from collections import Counter
from pathlib import Path

import pytest

from sigma_theory_compiler import open_gravity_full3d_roadmap_adjudication_v1 as roadmap


@pytest.fixture(scope="module")
def packet() -> tuple[dict, dict]:
    config = roadmap.load_config()
    return config, roadmap.run_suite(config)


def test_exact_evidence_chain(packet: tuple[dict, dict]) -> None:
    config, _suite = packet
    assert len(config["evidence"]) == 11
    assert len({row["id"] for row in config["evidence"]}) == 11
    assert all(len(row["commit"]) == 40 for row in config["evidence"])
    assert all(len(row["sha256"]) == 64 for row in config["evidence"])


def test_exact_33_task_ledger_and_status_counts(packet: tuple[dict, dict]) -> None:
    config, suite = packet
    assert [row["task"] for row in config["tasks"]] == list(range(1, 34))
    assert suite["task_status_counts"] == dict(
        sorted(
            Counter(
                {
                    "COMPLETE": 10,
                    "PARTIAL": 13,
                    "BLOCKED": 3,
                    "COMPLETE_ZERO_ELIGIBLE": 1,
                    "NOT_RUN_CONDITIONAL_GATE": 5,
                    "COMPLETE_STOP": 1,
                }
            ).items()
        )
    )


def test_prior_radial_result_is_not_erased(packet: tuple[dict, dict]) -> None:
    _config, suite = packet
    metrics = suite["checks"]["PRIOR_NEGATIVE_RESULT_PRESERVED"]["metrics"]
    assert metrics["galaxies"] == 139
    assert metrics["clusters"] == 8
    assert metrics["dashboards"] == 147
    assert metrics["galaxy_invalid_cells"] == 8
    assert metrics["cluster_invalid_cells"] == 18
    assert metrics["cross_domain_survivors"] == 0


def test_theory_matrix_is_exactly_420_by_25(packet: tuple[dict, dict]) -> None:
    _config, suite = packet
    metrics = suite["checks"]["THEORY_MATRIX_420_BY_25"]["metrics"]
    assert metrics == {
        "mechanisms": 420,
        "gates": 25,
        "rows": 10_500,
        "pass_target_free": 4,
        "required_unrun": 7,
    }


def test_six_executable_fields_are_synthetically_falsifiable(
    packet: tuple[dict, dict],
) -> None:
    _config, suite = packet
    metrics = suite["checks"]["SIX_EXECUTABLE_FIELDS_SYNTHETICALLY_FALSIFIABLE"]["metrics"]
    assert metrics["mechanisms"] == 6
    assert metrics["recovery_null_falsification_gates"] == 11
    assert metrics["real_response_scoring_eligible"] is False


def test_multisector_closure_remains_partial_or_blocked(packet: tuple[dict, dict]) -> None:
    _config, suite = packet
    metrics = suite["checks"]["MULTISECTOR_PARTIAL_BLOCKED_BOUNDARY"]["metrics"]
    assert metrics == {
        "sectors": 11,
        "BLOCKED": 5,
        "PARTIAL": 6,
        "observational_authority": False,
    }


def test_real_full3d_campaign_stops_at_source_gate(packet: tuple[dict, dict]) -> None:
    config, suite = packet
    assert suite["checks"]["ZERO_REAL_FULL3D_SOURCE_READY"]["metrics"] == {
        "objects": 147,
        "full_3d_ready": 0,
        "campaign_ready": False,
    }
    tasks = {row["task"]: row for row in config["tasks"]}
    assert tasks[27]["status"] == "COMPLETE_ZERO_ELIGIBLE"
    assert all(tasks[index]["status"] == "NOT_RUN_CONDITIONAL_GATE" for index in range(28, 33))
    assert tasks[33]["status"] == "COMPLETE_STOP"


def test_all_required_checks_pass_in_frozen_order(packet: tuple[dict, dict]) -> None:
    config, suite = packet
    assert list(suite["checks"]) == config["required_checks"]
    assert suite["passed"] == 9
    assert suite["failed"] == 0
    assert all(row["passed"] is True for row in suite["checks"].values())


@pytest.mark.parametrize(
    "section",
    (
        "purpose",
        "evidence",
        "tasks",
        "required_checks",
        "next_decision",
        "access_contract",
        "claim_boundary",
    ),
)
def test_every_semantic_section_is_hard_pinned(packet: tuple[dict, dict], section: str) -> None:
    config, _suite = packet
    changed = copy.deepcopy(config)
    changed[section] = None
    with pytest.raises(roadmap.RoadmapAdjudicationError, match="config semantics changed"):
        roadmap.validate_config(changed)


def test_noncanonical_receipt_path_rejected_before_read(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    reads = 0

    def forbidden(*_args: object, **_kwargs: object) -> dict:
        nonlocal reads
        reads += 1
        return {}

    monkeypatch.setattr(roadmap, "OUTPUT_PATH", tmp_path / "private-response.json")
    monkeypatch.setattr(roadmap, "_read_json", forbidden)
    with pytest.raises(roadmap.RoadmapAdjudicationError, match="output path changed"):
        roadmap.validate_receipt()
    assert reads == 0


def test_receipt_rebuild_and_coherent_forgery_rejection(packet: tuple[dict, dict]) -> None:
    _config, _suite = packet
    receipt = roadmap.build_receipt()
    roadmap.validate_receipt_payload(receipt)
    forged = copy.deepcopy(receipt)
    forged["adjudication"]["campaign_ready"] = True
    body = {key: value for key, value in forged.items() if key != "content_sha256"}
    forged["content_sha256"] = roadmap.content_sha256(body)
    with pytest.raises(roadmap.RoadmapAdjudicationError, match="not reproducible"):
        roadmap.validate_receipt_payload(forged)


def test_zero_access_and_honest_claim_boundary(packet: tuple[dict, dict]) -> None:
    config, _suite = packet
    receipt = roadmap.build_receipt()
    assert all(value == 0 for value in receipt["access_accounting"].values())
    assert "a successful new gravity theory" in config["claim_boundary"]["does_not_establish"]
    assert "real full-3D observational testing" in config["claim_boundary"]["does_not_establish"]
    assert config["next_decision"]["campaign_ready"] is False
