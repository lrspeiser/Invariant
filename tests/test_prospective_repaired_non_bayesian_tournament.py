from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.prospective_repaired_non_bayesian_tournament import (
    CLAIMS,
    CONFIG_PATH,
    FAMILIES,
    OUTPUT_PATH,
    TARGET_PATH,
    ProspectiveRepairError,
    build_campaign,
    validate_campaign,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def live_campaign() -> dict:
    return build_campaign(ROOT)


@pytest.fixture(scope="module")
def checked_campaign() -> dict:
    return json.loads((ROOT / OUTPUT_PATH).read_text(encoding="utf-8"))


def test_checked_receipt_is_exact_live_replay(live_campaign: dict, checked_campaign: dict) -> None:
    assert checked_campaign == live_campaign
    validate_campaign(checked_campaign, root=ROOT)


def test_generation_is_complete_before_one_atomic_unseal(checked_campaign: dict) -> None:
    assert [row["phase"] for row in checked_campaign["chronology"]] == [
        "preregistered_config_loaded",
        "six_native_families_generated",
        "seventy_two_repaired_candidates_sealed",
        "atomic_three_target_unseal",
        "common_ladder_and_pareto_replayed",
    ]
    assert [row["target_fixture_reads"] for row in checked_campaign["chronology"]] == [
        0,
        0,
        0,
        1,
        1,
    ]
    assert checked_campaign["phase_a"]["target_fixture_reads"] == 0
    assert checked_campaign["phase_a"]["target_fields_read"] == []
    assert checked_campaign["phase_a"]["pre_unseal_io"] == {
        "enforcement_scope": (
            "owned_single_threaded_python_builtins_io_and_pathlib_read_surfaces_not_os_sandbox"
        ),
        "attempted_target_reads": 1,
        "denied_target_reads": 1,
        "denied_content_bytes_exposed": 0,
        "successful_target_reads": 0,
    }
    assert checked_campaign["phase_a"]["counts"] == {
        "worlds": 3,
        "families_per_world": 6,
        "candidates": 72,
    }


def test_all_six_non_bayesian_families_receive_four_candidates(checked_campaign: dict) -> None:
    assert checked_campaign["counts"]["repaired_candidates"] == 72
    assert checked_campaign["counts"]["gate_checks"] == 144
    for world in checked_campaign["world_results"]:
        families = [candidate["representation"]["family"] for candidate in world["candidates"]]
        assert set(families) == set(FAMILIES)
        assert {family: families.count(family) for family in FAMILIES} == {
            family: 4 for family in FAMILIES
        }
        assert all(
            candidate["representation"]["target_fields_read"] == []
            for candidate in world["candidates"]
        )


def test_outcomes_are_honest_and_only_passes_receive_metrics(checked_campaign: dict) -> None:
    assert checked_campaign["counts"]["world_block"] == 0
    for world in checked_campaign["world_results"]:
        assert world["counts"]["pass"] + world["counts"]["reject"] == 24
        assert world["counts"]["block"] == 0
        assert len(world["metric_receipts"]) == 2 * world["counts"]["pass"]
        assert (world["pareto"] is not None) == bool(world["counts"]["pass"])


def test_claims_preserve_scientific_boundary(checked_campaign: dict) -> None:
    assert checked_campaign["claims"] == CLAIMS
    assert CLAIMS["all_repaired_candidates_generated_before_target_fixture_read"] is True
    assert CLAIMS["retrospective_repair_generalizes_universally"] is False
    assert CLAIMS["generator_output_establishes_truth"] is False
    assert CLAIMS["corpus_absence_establishes_novelty"] is False
    assert CLAIMS["promotion_authorized"] is False


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("counts", "target_fixture_reads_pre_unseal"), 1),
        (("claims", "promotion_authorized"), True),
        (("phase_a", "target_fields_read"), ["hypothesis"]),
        (("phase_a", "pre_unseal_io", "denied_target_reads"), 0),
        (("world_results", 0, "unsealed_target", "hypothesis"), 10),
    ],
)
def test_resealed_semantic_tampering_fails_closed(
    checked_campaign: dict, path: tuple[object, ...], replacement: object
) -> None:
    tampered = copy.deepcopy(checked_campaign)
    cursor = tampered
    for key in path[:-1]:
        cursor = cursor[key]  # type: ignore[index]
    cursor[path[-1]] = replacement  # type: ignore[index]
    body = {key: value for key, value in tampered.items() if key != "content_sha256"}
    from sigma_theory_compiler.sigma_core import canonical_sha256

    tampered["content_sha256"] = canonical_sha256(body)
    with pytest.raises(ProspectiveRepairError, match="replay mismatch"):
        validate_campaign(tampered, root=ROOT)


def test_preregistration_and_target_fixture_are_separate_files() -> None:
    config = json.loads((ROOT / CONFIG_PATH).read_text(encoding="utf-8"))
    target = json.loads((ROOT / TARGET_PATH).read_text(encoding="utf-8"))
    assert config["repair_contract"]["target_fields"] == []
    assert config["policies"]["bayesian_generation"] == "excluded"
    assert "targets" not in config
    assert target["schema_version"] == "sigma-prospective-repaired-target-fixture-1.0"
    assert len(target["targets"]) == 3
