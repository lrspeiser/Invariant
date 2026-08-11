from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.quartic_tc2_d4_homogeneous_freedom_reduction import (
    QuarticTC2D4HomogeneousFreedomReductionError,
    build_reduction,
    validate_reduction,
)
from sigma_theory_compiler.quartic_tc2_diagonal_third_jet_campaign import _content_hash

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/backgrounds/quartic_tc2_d4_homogeneous_freedom_reduction.json"


@pytest.fixture(scope="module")
def artifact() -> dict:
    return build_reduction(ROOT, CONFIG)


def _rehash(document: dict) -> dict:
    document.pop("content_sha256", None)
    document["content_sha256"] = _content_hash(document)
    return document


def test_all_lower_homogeneous_freedom_has_zero_d4_image(artifact: dict) -> None:
    validate_reduction(artifact)
    reduction = artifact["homogeneous_freedom_reduction"]
    assert reduction["exact_identity"] == (
        "R0(Y)^T F_H(Y) R0(Y)=0 for every matrix H(Y)"
    )
    assert reduction["total_lower_jet_reference_kernel_slots_before_cross_order_constraints"] == 20842
    assert reduction["induced_D4_zero_eigenspace_map_rank"] == 0
    assert reduction["canonical_rank_two_witness_in_image"] is False
    assert artifact["counts"]["candidate_obstructions_invariant"] == 12


def test_all_fifteen_polarization_projectors_close_through_order_four(
    artifact: dict,
) -> None:
    audit = artifact["exact_zero_projector_audit"]
    assert audit["polarization_directions_checked"] == 15
    assert len(audit["records"]) == 15
    assert audit["record_chain_tip_sha256"] == audit["records"][-1]["record_sha256"]
    for record in audit["records"]:
        assert record["stationary_top_rows_zero_by_order"] == [True] * 5
        assert record["companion_inverse_residual_zero_by_order"] == [True] * 5
        assert record["P_times_R0_zero_by_order"] == [True] * 5
        assert record["R0_idempotent_by_order"] == [True] * 5


def test_reference_dimension_accounting_is_exact(artifact: dict) -> None:
    space = artifact["reference_sylvester_space"]
    assert space["eigenspace_ranks"] == {
        "0": 33,
        "1": 3,
        "-1": 3,
        "1/2": 4,
        "-1/2": 4,
        "1/3": 4,
        "-1/3": 4,
    }
    assert space["homogeneous_kernel_dimension_per_jet_coefficient"] == 613
    assert space["Sylvester_range_dimension"] == 927
    assert space["equal_eigenspace_cokernel_dimension"] == 558
    assert space["zero_eigenspace_skew_cokernel_dimension"] == 528


def test_downstream_claims_remain_fail_closed(artifact: dict) -> None:
    claims = artifact["claims"]
    assert claims[
        "alternative_lower_jet_homogeneous_completion_ruled_out_for_obligation_244"
    ] is True
    assert claims["all_12_registered_candidates_D4_obstructed_at_obligation_244"] is True
    for key in (
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
        (
            ("homogeneous_freedom_reduction", "induced_D4_zero_eigenspace_map_rank"),
            1,
        ),
        (
            ("homogeneous_freedom_reduction", "canonical_rank_two_witness_in_image"),
            True,
        ),
        (
            (
                "exact_zero_projector_audit",
                "records",
                0,
                "P_times_R0_zero_by_order",
            ),
            [True, True, True, True, False],
        ),
        (("candidate_classification", 0, "cancellation_possible"), True),
        (("claims", "TC2_closed"), True),
        (("negative_controls", "drop_lower_recurrence_hypothesis", "rejected"), False),
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
    with pytest.raises(QuarticTC2D4HomogeneousFreedomReductionError):
        validate_reduction(_rehash(mutated))


def test_bound_obstruction_tamper_fails_before_symbolic_replay(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["obstruction_certificate"]["file_sha256"] = "0" * 64
    config.pop("content_sha256")
    config["content_sha256"] = _content_hash(config)
    path = tmp_path / "tampered-config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(QuarticTC2D4HomogeneousFreedomReductionError):
        build_reduction(ROOT, path)
