from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.quartic_tc2_d4_axis2_base_rhs_campaign import (
    QuarticTC2D4Axis2BaseRHSError,
    build_campaign,
    validate_campaign,
)
from sigma_theory_compiler.quartic_tc2_diagonal_third_jet_campaign import (
    _content_hash,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/backgrounds/quartic_tc2_d4_axis2_base_rhs_campaign.json"
ARTIFACT = ROOT / "runs/physics-language/quartic-tc2-d4-axis2-base-rhs-campaign/campaign.json"


@pytest.fixture(scope="module")
def artifact() -> dict:
    return build_campaign(ROOT, CONFIG)


def _rehash(document: dict) -> dict:
    document.pop("content_sha256", None)
    document["content_sha256"] = _content_hash(document)
    return document


def test_committed_artifact_replays_exactly(artifact: dict) -> None:
    assert artifact == json.loads(ARTIFACT.read_text(encoding="utf-8"))


def test_complete_axis2_polarization_is_replayed_exactly(artifact: dict) -> None:
    validate_campaign(artifact)
    exact = artifact["exact_axis2_base_D4_audit"]
    assert exact["directional_evaluations"] == 15
    assert exact["selector_record"]["selector_offset"] == 244
    assert exact["selector_record"]["active_indices"] == [0, 2, 3, 9]
    base = exact["polarized_base_D4"]
    assert base["RHS_base_nonzero_entries"] == 0
    assert base["RHS_base_free_parameters"] == []
    assert base["nonzero_equal_eigenspace_compressions"] == {}
    assert len(base["RHS_base_sha256"]) == 64
    assert len(base["D4P55_sha256"]) == 64
    assert len(base["D4K55_sha256"]) == 64
    assert len(base["D4TC2_sha256"]) == 64


def test_all_candidates_receive_exact_companion_comparison(artifact: dict) -> None:
    rows = artifact["exact_axis2_base_D4_audit"]["candidate_comparison"]
    assert len(rows) == 12
    assert all(len(row["base_zero_compression_sha256"]) == 64 for row in rows)
    assert all(len(row["required_negative_companion_sha256"]) == 64 for row in rows)
    assert (
        sum(row["zero_speed_cancellation_exact"] for row in rows)
        == artifact["counts"]["zero_speed_cancellations_exact"]
    )
    assert (
        sum(row["corrected_axis2_D4_Sylvester_solvable"] for row in rows)
        == artifact["counts"]["corrected_axis2_D4_compatibilities"]
    )
    assert all(not row["zero_speed_cancellation_exact"] for row in rows)
    assert all(not row["corrected_axis2_D4_Sylvester_solvable"] for row in rows)
    assert all(not row["wrong_sign_axis2_D4_Sylvester_solvable"] for row in rows)


def test_complete_all_eigenspace_result_is_not_inferred_from_zero_speed_only(
    artifact: dict,
) -> None:
    rows = artifact["exact_axis2_base_D4_audit"]["candidate_comparison"]
    assert all(
        isinstance(row["corrected_nonzero_equal_eigenspace_compressions"], dict) for row in rows
    )
    assert (
        artifact["counts"]["corrected_axis2_D4_compatibilities"]
        + artifact["counts"]["corrected_axis2_D4_obstructions"]
        == 12
    )
    assert artifact["counts"]["corrected_axis2_D4_compatibilities"] == 0
    assert artifact["counts"]["corrected_axis2_D4_obstructions"] == 12
    assert all(set(row["corrected_nonzero_equal_eigenspace_compressions"]) == {"0"} for row in rows)


def test_global_claims_remain_fail_closed(artifact: dict) -> None:
    claims = artifact["claims"]
    assert claims["full_axis2_base_D4_RHS_evaluated"] is True
    assert claims["all_12_axis2_D4_compatibilities_proved"] is False
    assert claims["all_12_axis2_D4_obstructions_proved"] is True
    assert claims["fixed_chart_curl_completion_axis2_D4_rejected"] is True
    for key in (
        "spatially_covariant_tensor_completion_proved",
        "all_spatial_direction_compatibility_proved",
        "corrected_candidate_family_registered",
        "remaining_D4_selector_closed",
        "full_tube_Sylvester_identity",
        "CK1_closed",
        "CK3_closed",
        "TC2_closed",
        "B7_closed",
        "global_H7_closed",
        "lifespan_proved",
    ):
        assert claims[key] is False


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("counts", "directional_evaluations"), 14),
        (("counts", "candidate_conditions_checked"), 11),
        (("claims", "full_axis2_base_D4_RHS_evaluated"), False),
        (("claims", "all_12_axis2_D4_obstructions_proved"), False),
        (
            ("exact_axis2_base_D4_audit", "polarized_base_D4", "RHS_base_sha256"),
            "0" * 64,
        ),
        (
            (
                "exact_axis2_base_D4_audit",
                "candidate_comparison",
                0,
                "zero_speed_cancellation_exact",
            ),
            True,
        ),
        (("claims", "TC2_closed"), True),
        (("negative_controls", "omit_complete_base_D4_RHS", "rejected"), False),
    ],
)
def test_validator_rejects_rehashed_tampering(
    artifact: dict, path: tuple[str | int, ...], value: object
) -> None:
    mutated = copy.deepcopy(artifact)
    cursor = mutated
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value
    with pytest.raises(QuarticTC2D4Axis2BaseRHSError):
        validate_campaign(_rehash(mutated))


def test_companion_binding_tamper_fails_before_replay(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["companion_range"]["content_sha256"] = "0" * 64
    config.pop("content_sha256")
    config["content_sha256"] = _content_hash(config)
    path = tmp_path / "tampered-config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(QuarticTC2D4Axis2BaseRHSError):
        build_campaign(ROOT, path)
