from pathlib import Path

from sigma_theory_compiler.gravity_item39_holographic_boundary_result import (
    build,
    check,
)

ROOT = Path(__file__).resolve().parents[1]


def test_item39_aggregate_result_keeps_positive_and_negative_evidence() -> None:
    result = build(ROOT, write=False)
    assert result["decision"] == ("NONPROMOTED_ITEM39_HOLOGRAPHIC_BOUNDARY_MIXED_DIAGNOSTIC")
    assert result["wallaby_dynamics"]["quality_passing_galaxies"] == 23
    assert result["wallaby_dynamics"]["improvement_vs_strongest_percent"] < 0.0
    lensing = result["unchanged_swells_lensing_diagnostic"]
    assert lensing["evaluable_lenses"] == 6
    assert lensing["improvement_vs_strongest_percent"] > 0.0
    assert result["gates"]["lensing_all_frozen_systematics_improve"] is False
    assert result["gates"]["promotion_passes"] is False
    assert result["wallaby_dynamics"]["confirmation_response_rows"] == 0
    assert result["compute"]["paid_api_calls"] == 0


def test_item39_aggregate_result_replays() -> None:
    result = check(ROOT)
    assert result["status"] == "ITEM39_AGGREGATE_RESULT_VALID"
    assert result["promotion_passes"] is False
    assert result["confirmation_response_rows"] == 0
    assert result["paid_model_calls"] == 0
