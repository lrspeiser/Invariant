from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.quartic_tc2_d4_parity_cubic_generic_direction_campaign import (
    QuarticTC2D4ParityCubicGenericDirectionError,
    build_campaign,
    validate_campaign,
)
from sigma_theory_compiler.quartic_tc2_mixed_third_jet_continuation_service import _with_hash

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/backgrounds/quartic_tc2_d4_parity_cubic_generic_direction_campaign.json"
ARTIFACT = (
    ROOT
    / "runs/physics-language/quartic-tc2-d4-parity-cubic-generic-direction-campaign/campaign.json"
)


@pytest.fixture(scope="module")
def artifact() -> dict:
    value = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    validate_campaign(value)
    return value


def test_checked_artifact_replays_exactly(artifact: dict) -> None:
    assert build_campaign(ROOT, CONFIG) == artifact


def test_selector_is_deterministic_and_fail_fast(artifact: dict) -> None:
    selector = artifact["exact_generic_direction_audit"]["selector"]
    assert selector["declared_rational_frame_names"] == [
        "xy_3_4_5",
        "xz_3_4_5",
        "xyz_1_2_2",
    ]
    assert selector["declared_frame_count"] == 3
    assert selector["frames_evaluated"] == 1
    assert selector["frames_unevaluated_after_stop"] == 2
    assert selector["stop_reason"] == "first_exact_generic_direction_obstruction"
    assert selector["not_an_interpolation_basis_for_the_full_sphere"] is True


def test_full_lower_and_fourth_recurrence_was_evaluated(artifact: dict) -> None:
    record = artifact["exact_generic_direction_audit"]["direction_records"][0]
    assert record["direction"] == ["3/5", "4/5", "0"]
    assert record["unit_norm"] is True
    assert record["directional_evaluations"] == 15
    assert record["all_seven_eigenspaces_checked_per_candidate"] is True
    assert len(record["base_D4_RHS_sha256"]) == 64
    assert record["cubic_correction_block_rank"] > 0


def test_first_generic_frame_obstructs_all_candidates(artifact: dict) -> None:
    record = artifact["exact_generic_direction_audit"]["direction_records"][0]
    assert record["candidate_compatibilities"] == 0
    assert record["candidate_obstructions"] == 12
    assert len(record["candidate_records"]) == 12
    assert all(not row["D4_Sylvester_solvable"] for row in record["candidate_records"])
    assert all(row["nonzero_equal_eigenspace_compressions"] for row in record["candidate_records"])


def test_result_rejects_only_the_declared_cubic_completion(artifact: dict) -> None:
    result = artifact["exact_generic_direction_audit"]["result"]
    assert result["candidate_direction_systems_evaluated"] == 12
    assert result["candidate_direction_compatibilities"] == 0
    assert result["candidate_direction_obstructions"] == 12
    assert result["cubic_escape_all_direction_completion_rejected"] is True
    assert result["full_generic_direction_sphere_classified"] is False


def test_every_downstream_claim_stays_fail_closed(artifact: dict) -> None:
    claims = artifact["claims"]
    assert claims["parity_cubic_all_direction_completion_rejected"] is True
    for key in (
        "full_generic_direction_sphere_classified",
        "generic_direction_D4_compatibility_proved",
        "local_differential_operator_origin_proved",
        "covariant_action_origin_proved",
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
    assert artifact["counts"]["inferred_global_passes"] == 0


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("counts", "frames_evaluated"), 2),
        (("counts", "candidate_direction_obstructions"), 11),
        (("exact_generic_direction_audit", "selector", "stop_reason"), "finite_basis_complete"),
        (
            (
                "exact_generic_direction_audit",
                "selector",
                "not_an_interpolation_basis_for_the_full_sphere",
            ),
            False,
        ),
        (
            (
                "exact_generic_direction_audit",
                "direction_records",
                "0",
                "all_seven_eigenspaces_checked_per_candidate",
            ),
            False,
        ),
        (("claims", "full_generic_direction_sphere_classified"), True),
        (("claims", "TC2_closed"), True),
        (("negative_controls", "continue_after_first_exact_obstruction", "rejected"), False),
    ],
)
def test_rehashed_semantic_tampering_is_rejected(
    artifact: dict, path: tuple[str, ...], replacement: object
) -> None:
    tampered = copy.deepcopy(artifact)
    cursor: object = tampered
    for key in path[:-1]:
        cursor = cursor[int(key)] if isinstance(cursor, list) else cursor[key]
    if isinstance(cursor, list):
        cursor[int(path[-1])] = replacement
    else:
        cursor[path[-1]] = replacement
    tampered = _with_hash(
        {key: value for key, value in tampered.items() if key != "content_sha256"}
    )
    with pytest.raises(QuarticTC2D4ParityCubicGenericDirectionError):
        validate_campaign(tampered)


def test_tampered_cubic_predecessor_fails_closed(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["parity_cubic_escape"]["content_sha256"] = "0" * 64
    config = _with_hash({key: value for key, value in config.items() if key != "content_sha256"})
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(QuarticTC2D4ParityCubicGenericDirectionError):
        build_campaign(ROOT, path)


def test_config_source_and_test_are_hash_bound() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    for key in ("campaign_source", "campaign_test"):
        path = ROOT / config[key]["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == config[key]["file_sha256"]
