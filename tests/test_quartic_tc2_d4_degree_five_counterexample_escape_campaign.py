from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.quartic_tc2_d4_degree_five_counterexample_escape_campaign import (
    FALSE_CLAIMS,
    TRUE_CLAIMS,
    DegreeFiveCounterexampleEscapeError,
    validate_campaign,
)
from sigma_theory_compiler.quartic_tc2_diagonal_third_jet_campaign import _content_hash

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = (
    ROOT
    / "runs/physics-language/quartic-tc2-d4-degree-five-counterexample-escape-campaign/campaign.json"
)


@pytest.fixture(scope="module")
def artifact() -> dict:
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def test_minimal_sparse_preserving_envelope(artifact: dict) -> None:
    envelope = artifact["exact_completion"]["minimal_preserving_envelope"]
    assert envelope["minimal_even_homogeneous_degree"] == 4
    assert envelope["degree_zero"]["prior_zero_nullity"] == 0
    assert envelope["degree_two"]["prior_zero_nullity"] == 0
    assert envelope["degree_four"]["prior_zero_nullity"] == 9
    assert envelope["degree_four_normalized_affine_dimension"] == 8
    assert envelope["one_monomial_supports_feasible"] == 0
    assert envelope["two_monomial_supports_checked"] == 105
    assert envelope["two_monomial_supports_feasible"] == 1
    assert envelope["sparsest_envelope"] == (
        "a7(n)=(81/14)*n2*n3*(2*n1*n2-n3^2)"
    )


def test_exact_range_and_degree_five_extension(artifact: dict) -> None:
    exact = artifact["exact_completion"]
    range_result = exact["exact_range_classification"]
    completion = exact["minimal_rank_two_completion"]
    extension = exact["degree_five_angular_extension"]
    assert range_result["normalized_target_rank"] == 4
    assert range_result["transverse_selector_rank"] == 22
    assert range_result["target_in_full_transverse_curl_range"] is True
    assert completion["constructed_completion_rank"] == 2
    assert completion["elementary_curl_channels"] == 2
    assert extension["minimal_total_extension_degree"] == 5
    assert extension["antipodally_odd"] is True
    assert extension["all_six_prior_direction_extensions_zero"] is True
    assert extension["physical_gradient_lift_annihilated_identically"] is True


def test_full_recurrence_closes_former_counterexample(artifact: dict) -> None:
    predecessor = artifact["exact_completion"]["predecessor_obstruction_replay"]
    corrected = artifact["exact_completion"]["corrected_result"]
    assert predecessor["directional_recurrence_evaluations"] == 15
    assert predecessor["candidate_compatibilities"] == 0
    assert predecessor["candidate_obstructions"] == 12
    assert corrected["candidate_conditions_checked"] == 12
    assert corrected["candidate_compatibilities"] == 12
    assert corrected["candidate_obstructions"] == 0
    assert all(row["D4_Sylvester_solvable"] for row in corrected["candidate_records"])
    assert all(
        row["nonzero_equal_eigenspace_compressions"] == {}
        for row in corrected["candidate_records"]
    )


def test_closed_world_claims_and_validator(artifact: dict) -> None:
    validate_campaign(artifact)
    assert set(artifact["claims"]) == TRUE_CLAIMS | FALSE_CLAIMS
    assert all(artifact["claims"][key] is True for key in TRUE_CLAIMS)
    assert all(artifact["claims"][key] is False for key in FALSE_CLAIMS)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (
            ("exact_completion", "minimal_preserving_envelope", "sparsest_support_size"),
            1,
        ),
        (
            (
                "exact_completion",
                "minimal_rank_two_completion",
                "aligned_block_sha256",
            ),
            "0" * 64,
        ),
        (
            (
                "exact_completion",
                "corrected_result",
                "candidate_records",
                0,
                "D4_Sylvester_solvable",
            ),
            False,
        ),
        (("claims", "full_direction_sphere_D4_compatibility_proved"), True),
    ],
)
def test_resealed_semantic_tamper_rejected(
    artifact: dict, path: tuple[object, ...], value: object
) -> None:
    mutated = copy.deepcopy(artifact)
    cursor = mutated
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value
    mutated["content_sha256"] = _content_hash(
        {key: item for key, item in mutated.items() if key != "content_sha256"}
    )
    with pytest.raises(DegreeFiveCounterexampleEscapeError):
        validate_campaign(mutated)


def test_unknown_claim_and_negative_control_rejected(artifact: dict) -> None:
    for mutation in ("claim", "control"):
        mutated = copy.deepcopy(artifact)
        if mutation == "claim":
            mutated["claims"]["unknown"] = False
        else:
            mutated["negative_controls"]["invented"] = {"rejected": True}
        mutated["content_sha256"] = _content_hash(
            {key: item for key, item in mutated.items() if key != "content_sha256"}
        )
        with pytest.raises(DegreeFiveCounterexampleEscapeError):
            validate_campaign(mutated)
