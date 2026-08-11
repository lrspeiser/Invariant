from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.quartic_tc2_d4_parity_cubic_angular_escape_campaign import (
    QuarticTC2D4ParityCubicAngularEscapeError,
    build_campaign,
    validate_campaign,
)
from sigma_theory_compiler.quartic_tc2_mixed_third_jet_continuation_service import _with_hash

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/backgrounds/quartic_tc2_d4_parity_cubic_angular_escape_campaign.json"
ARTIFACT = (
    ROOT / "runs/physics-language/quartic-tc2-d4-parity-cubic-angular-escape-campaign/campaign.json"
)


@pytest.fixture(scope="module")
def artifact() -> dict:
    value = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    validate_campaign(value)
    return value


def test_checked_artifact_replays_exactly(artifact: dict) -> None:
    assert build_campaign(ROOT, CONFIG) == artifact


def test_minimal_parity_preserving_degree_is_exact(artifact: dict) -> None:
    minimality = artifact["exact_escape"]["minimality"]
    assert minimality["constant_even_multiplier_impossible"] is True
    assert minimality["degree_one_multiplier_rejected_by_antipodal_parity"] is True
    assert minimality["minimal_nonconstant_even_multiplier_degree"] == 2
    assert minimality["minimal_total_angular_polynomial_degree"] == 3
    assert minimality["canonical_multiplier"] == "a(n)=n1^2"


def test_symbol_is_odd_bounded_and_preserves_only_claimed_axes(artifact: dict) -> None:
    symbol = artifact["exact_escape"]["exact_symbol"]
    assert symbol["nonzero_polynomial_coefficient_blocks"] == 2
    assert symbol["antipodal_odd"] is True
    assert symbol["sphere_multiplier_interval"] == "0<=n1^2<=1"
    assert symbol["minus_e1_is_negative_reference"] is True
    assert symbol["e2_block_zero"] is True
    assert symbol["minus_e2_block_zero"] is True
    assert symbol["e3_block_zero"] is True


def test_full_gradient_lift_annihilation_is_exact(artifact: dict) -> None:
    equivalence = artifact["exact_escape"]["physical_gradient_lift_equivalence"]
    assert equivalence["residual_zero"] is True
    assert equivalence["constraint_surface_principal_operator_zero"] is True
    assert len(equivalence["residual_sha256"]) == 64


def test_all_candidates_gain_only_the_axis2_result(artifact: dict) -> None:
    result = artifact["exact_escape"]["two_axis_D4_consequence"]
    assert result["reference_e1_solutions_inherited"] == 12
    assert result["axis2_base_D4_RHS_identically_zero"] is True
    assert result["axis2_companion_blocks_after_multiplier"] == 0
    assert result["axis2_D4_compatibilities"] == 12
    assert result["axis2_D4_obstructions"] == 0
    assert result["all_direction_D4_compatibility_proved"] is False
    assert len(result["candidate_records"]) == 12
    assert all(row["e1_D4_Sylvester_solvable_inherited"] for row in result["candidate_records"])
    assert all(row["e2_D4_Sylvester_solvable"] for row in result["candidate_records"])
    assert all(
        not row["all_direction_D4_Sylvester_solvable"] for row in result["candidate_records"]
    )


def test_nonlocal_origin_and_first_blocker_remain_explicit(artifact: dict) -> None:
    admission = artifact["exact_escape"]["pseudodifferential_constraint_admission"]
    assert admission["M1_fourier_symbol"] == "xi1^2/|xi|^2=n1^2"
    assert admission["periodic_or_Schwartz_constraint_surface_invariant"] is True
    assert admission["boundary_domain_realization_proved"] is False
    assert admission["local_differential_operator_realization_proved"] is False
    assert admission["covariant_action_origin_proved"] is False
    assert (
        artifact["exact_escape"]["first_blocker"]["name"]
        == "generic_direction_D4_and_nonlocal_variable_coefficient_admission"
    )


def test_every_global_claim_stays_fail_closed(artifact: dict) -> None:
    claims = artifact["claims"]
    for key in (
        "generic_direction_D4_compatibility_proved",
        "local_differential_operator_origin_proved",
        "covariant_action_origin_proved",
        "spatially_covariant_tensor_completion_proved",
        "variable_coefficient_pseudodifferential_energy_calculus_proved",
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
        (("counts", "new_axis2_D4_compatibilities"), 11),
        (("exact_escape", "minimality", "minimal_total_angular_polynomial_degree"), 2),
        (("exact_escape", "exact_symbol", "antipodal_odd"), False),
        (("exact_escape", "exact_symbol", "e2_block_zero"), False),
        (("exact_escape", "physical_gradient_lift_equivalence", "residual_zero"), False),
        (("exact_escape", "two_axis_D4_consequence", "axis2_D4_obstructions"), 1),
        (("claims", "generic_direction_D4_compatibility_proved"), True),
        (("claims", "TC2_closed"), True),
        (("negative_controls", "linear_multiplier_n1", "rejected"), False),
    ],
)
def test_rehashed_semantic_tampering_is_rejected(
    artifact: dict, path: tuple[str, ...], replacement: object
) -> None:
    tampered = copy.deepcopy(artifact)
    cursor = tampered
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = replacement
    tampered = _with_hash(
        {key: value for key, value in tampered.items() if key != "content_sha256"}
    )
    with pytest.raises(QuarticTC2D4ParityCubicAngularEscapeError):
        validate_campaign(tampered)


def test_tampered_predecessor_binding_fails_closed(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["full_linear_no_go"]["content_sha256"] = "0" * 64
    config = _with_hash({key: value for key, value in config.items() if key != "content_sha256"})
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(QuarticTC2D4ParityCubicAngularEscapeError):
        build_campaign(ROOT, path)


def test_config_source_and_test_are_hash_bound() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    for key in ("campaign_source", "campaign_test"):
        path = ROOT / config[key]["path"]
        import hashlib

        assert hashlib.sha256(path.read_bytes()).hexdigest() == config[key]["file_sha256"]
