from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.quartic_tc2_d4_rational_chart_determining_gate import (
    FALSE_CLAIMS,
    TRUE_CLAIMS,
    RationalChartDeterminingError,
    validate_campaign,
)
from sigma_theory_compiler.quartic_tc2_diagonal_third_jet_campaign import _content_hash

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = (
    ROOT / "runs/physics-language/quartic-tc2-d4-rational-chart-determining-gate/campaign.json"
)


@pytest.fixture(scope="module")
def artifact() -> dict:
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def test_exact_atlas_and_regular_counterexample(artifact: dict) -> None:
    gate = artifact["exact_gate"]
    assert gate["atlas"]["union_covers_real_S2"] is True
    assert gate["atlas"]["real_singular_strata"] == 0
    assert gate["counterexample_selector"]["chart_coordinates"] == ["2/5", "1/5"]
    assert gate["counterexample_selector"]["direction"] == ["2/3", "2/3", "1/3"]


def test_exact_obstruction_disproves_current_full_sphere(artifact: dict) -> None:
    result = artifact["exact_gate"]["exact_rational_obstruction"]
    assert result["candidate_compatibilities"] == 0
    assert result["candidate_obstructions"] == 12
    assert result["current_full_sphere_D4_compatibility_disproved"] is True
    assert len(result["candidate_records"]) == 12
    assert all(
        row["zero_speed_cleared_numerator"]["numerator_rank"] == 4
        for row in result["candidate_records"]
    )


def test_symbolic_reduction_terminates_on_counterexample(artifact: dict) -> None:
    reduction = artifact["exact_gate"]["symbolic_chart_reduction"]
    assert reduction["terminal_reason"] == "exact_regular_rational_counterexample_found"
    assert reduction["full_polynomial_identity_reduction_required_after_counterexample"] is False
    assert reduction["singular_stratum_invoked"] is False


def test_closed_world_claims(artifact: dict) -> None:
    assert set(artifact["claims"]) == TRUE_CLAIMS | FALSE_CLAIMS
    assert all(artifact["claims"][key] is True for key in TRUE_CLAIMS)
    assert all(artifact["claims"][key] is False for key in FALSE_CLAIMS)


def test_negative_controls(artifact: dict) -> None:
    assert len(artifact["negative_controls"]) == 8
    assert all(value["rejected"] for value in artifact["negative_controls"].values())


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("counts", "candidate_obstructions"), 11),
        (("exact_gate", "atlas", "union_covers_real_S2"), False),
        (("exact_gate", "counterexample_selector", "direction"), ["1", "0", "0"]),
        (("exact_gate", "full_recurrence", "base_D4_RHS_sha256"), "0" * 64),
        (("exact_gate", "exact_rational_obstruction", "candidate_compatibilities"), 1),
        (("claims", "full_direction_sphere_D4_compatibility_proved"), True),
    ],
)
def test_resealed_semantic_tamper_rejected(
    artifact: dict, path: tuple[str, ...], value: object
) -> None:
    mutated = copy.deepcopy(artifact)
    cursor = mutated
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value
    body = {key: item for key, item in mutated.items() if key != "content_sha256"}
    mutated["content_sha256"] = _content_hash(body)
    with pytest.raises(RationalChartDeterminingError):
        validate_campaign(mutated)


def test_unknown_claim_rejected(artifact: dict) -> None:
    mutated = copy.deepcopy(artifact)
    mutated["claims"]["unknown"] = False
    body = {key: item for key, item in mutated.items() if key != "content_sha256"}
    mutated["content_sha256"] = _content_hash(body)
    with pytest.raises(RationalChartDeterminingError):
        validate_campaign(mutated)
