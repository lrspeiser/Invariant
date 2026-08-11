from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.quartic_tc2_d4_minimal_tc2_escape_campaign import (
    QuarticTC2D4MinimalTC2EscapeCampaignError,
    build_campaign,
    validate_campaign,
)
from sigma_theory_compiler.quartic_tc2_diagonal_third_jet_campaign import _content_hash

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/backgrounds/quartic_tc2_d4_minimal_tc2_escape_campaign.json"


@pytest.fixture(scope="module")
def artifact() -> dict:
    return build_campaign(ROOT, CONFIG)


def _rehash(document: dict) -> dict:
    document.pop("content_sha256", None)
    document["content_sha256"] = _content_hash(document)
    return document


def test_rank_one_tc2_basis_spans_the_obstruction_line(artifact: dict) -> None:
    validate_campaign(artifact)
    exact = artifact["exact_escape"]
    ansatz = exact["correction_ansatz"]
    induced = exact["induced_cokernel_map"]
    assert ansatz["V_rank"] == 1
    assert ansatz["V_nonzero_entries"] == 6
    assert ansatz["W_rank"] == 2
    assert induced["rank"] == 1
    assert induced["canonical_obstruction_in_image"] is True
    assert induced["unique_solvability_condition"] == "eta=-(34816/15)*alpha^5"


def test_all_candidate_specializations_solve_after_unique_tuning(
    artifact: dict,
) -> None:
    rows = artifact["exact_escape"]["candidate_classification"]
    assert len(rows) == 12
    assert {row["eta_unique_tuning"] for row in rows} == {
        "-34816/15",
        "-1088/15",
        "1088/15",
        "34816/15",
    }
    assert all(row["corrected_D4_Sylvester_solvable"] for row in rows)
    assert all(row["corrected_deltaK_Hermitian"] for row in rows)
    assert all(row["corrected_D4_Sylvester_residual_zero"] for row in rows)
    assert all(row["covariant_operator_origin_proved"] is False for row in rows)


def test_minimality_and_universal_parameter_negative(artifact: dict) -> None:
    minimality = artifact["exact_escape"]["minimality"]
    assert minimality["minimum_parameter_dimension"] == 1
    assert minimality["minimum_block_rank"] == 1
    assert minimality["rank_one_block_sufficient"] is True
    universal = artifact["negative_controls"][
        "one_universal_eta_for_all_candidates"
    ]
    assert universal["single_eta_closes_all_12"] is False
    assert len(universal["distinct_required_values"]) == 4


def test_physical_and_downstream_claims_remain_fail_closed(artifact: dict) -> None:
    claims = artifact["claims"]
    assert claims["obligation_244_minimal_algebraic_TC2_escape_constructed"] is True
    assert claims["candidate_specific_tuned_D4_compatibility_count"] == 12
    for key in (
        "single_universal_eta_closes_all_12",
        "correction_covariant_or_action_derived",
        "correction_gauge_constraint_compatible",
        "corrected_candidate_family_registered",
        "all_3060_fourth_jet_obligations_evaluated",
        "full_fourth_jet_range_closed",
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
        (("exact_escape", "correction_ansatz", "V_rank"), 2),
        (("exact_escape", "correction_ansatz", "V_sha256"), "0" * 64),
        (("exact_escape", "induced_cokernel_map", "rank"), 0),
        (
            ("exact_escape", "induced_cokernel_map", "canonical_obstruction_in_image"),
            False,
        ),
        (
            (
                "exact_escape",
                "candidate_classification",
                0,
                "corrected_D4_Sylvester_solvable",
            ),
            False,
        ),
        (("claims", "correction_covariant_or_action_derived"), True),
        (("claims", "TC2_closed"), True),
        (("negative_controls", "wrong_sign", "rejected"), False),
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
    with pytest.raises(QuarticTC2D4MinimalTC2EscapeCampaignError):
        validate_campaign(_rehash(mutated))


def test_predecessor_binding_tamper_fails_before_replay(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["homogeneous_freedom_reduction"]["file_sha256"] = "0" * 64
    config.pop("content_sha256")
    config["content_sha256"] = _content_hash(config)
    path = tmp_path / "tampered-config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(QuarticTC2D4MinimalTC2EscapeCampaignError):
        build_campaign(ROOT, path)
