from __future__ import annotations

import copy
import json
from fractions import Fraction
from pathlib import Path

import pytest

from sigma_theory_compiler.quartic_tc2_d4_coordinate_free_sphere_normal_form_reducer_registration import (
    SphereNormalFormRegistrationError,
    _content_hash,
    build_campaign,
    coefficient_vector,
    odd_sphere_modes,
    reconstruct,
    reduce_mod_sphere,
    validate_campaign,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / (
    "configs/backgrounds/"
    "quartic_tc2_d4_coordinate_free_sphere_normal_form_reducer_registration.json"
)
ARTIFACT = ROOT / (
    "runs/physics-language/"
    "quartic-tc2-d4-coordinate-free-sphere-normal-form-reducer-registration/campaign.json"
)


@pytest.fixture(scope="module")
def artifact() -> dict:
    document = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    validate_campaign(document, ROOT)
    return document


def test_mode_basis_is_exactly_210_odd_canonical_monomials(artifact: dict) -> None:
    modes = odd_sphere_modes()
    assert len(modes) == len(set(modes)) == 210
    assert all(a in (0, 1) and (a + b + c) % 2 == 1 and a + b + c <= 19 for a, b, c in modes)
    certificate = artifact["registered_sphere_normal_form_reducer"]
    assert [tuple(row["exponents"]) for row in certificate["mode_ordering"]] == modes
    assert certificate["basis_unit_replays"] == 210


def test_exact_quotient_witness_reconstructs_nontrivial_polynomial() -> None:
    polynomial = {
        (9, 2, 0): Fraction(3, 5),
        (3, 0, 4): Fraction(-7, 3),
        (1, 2, 2): Fraction(11),
    }
    quotient, remainder = reduce_mod_sphere(polynomial)
    assert quotient
    assert all(exponent[0] < 2 for exponent in remainder)
    assert reconstruct(quotient, remainder) == polynomial


def test_extractor_has_unit_vectors_and_rejects_boundary_inputs() -> None:
    modes = odd_sphere_modes()
    for index in (0, 71, 209):
        vector = coefficient_vector({modes[index]: Fraction(1)})
        assert vector[index] == "1"
        assert sum(value != "0" for value in vector) == 1
    with pytest.raises(SphereNormalFormRegistrationError, match="odd polynomials"):
        coefficient_vector({(0, 0, 2): Fraction(1)})
    with pytest.raises(SphereNormalFormRegistrationError, match="odd polynomials"):
        coefficient_vector({(0, 0, 21): Fraction(1)})


def test_manifest_advances_to_19_and_leaves_only_285_matrix_packets(artifact: dict) -> None:
    counts = artifact["counts"]
    assert counts["predecessor_registered_symbolic_input_packets"] == 18
    assert counts["new_sphere_reducer_packets_registered"] == 1
    assert counts["registered_symbolic_input_packets"] == 19
    assert counts["missing_symbolic_input_packets"] == 285
    assert counts["sphere_generator_multiple_replays"] == 615
    assert counts["nonzero_replay_remainders"] == 0
    manifest = {row["input_id"]: row for row in artifact["required_symbolic_input_manifest"]}
    assert manifest["sphere_mode_normal_form_reducer"]["registered_packets"] == 1
    missing = {
        row["input_id"]: row["missing_packets"] for row in artifact["remaining_missing_inputs"]
    }
    assert missing == {
        "polarized_P55_Taylor_packets": 75,
        "polarized_K55_Taylor_packets": 75,
        "polarized_TC2_Taylor_packets": 75,
        "lower_Sylvester_correction_recurrence": 60,
    }


def test_rows_phase_two_and_broad_claims_remain_blocked(artifact: dict) -> None:
    assert artifact["bounded_emitter_checkpoint"]["emitted_output_rows"] == 0
    assert artifact["phase_two"]["decision"] == "BLOCK"
    assert artifact["phase_two"]["blocker"] == (
        "285 required symbolic input packets remain unregistered"
    )
    for claim in (
        "complete_coordinate_free_coefficient_map_emitted",
        "full_direction_sphere_D4_compatibility_proved",
        "TC2_closed",
        "global_H7_closed",
        "nonlinear_PDE_closure_proved",
        "lifespan_proved",
        "theory_candidate_rejected",
    ):
        assert artifact["claims"][claim] is False


def test_artifact_replays_and_semantic_tamper_fails_closed(artifact: dict) -> None:
    assert build_campaign(ROOT, CONFIG) == artifact
    assert artifact["content_sha256"] == _content_hash(artifact)
    assert all(value == {"rejected": True} for value in artifact["negative_controls"].values())
    tampered = copy.deepcopy(artifact)
    tampered["counts"]["odd_sphere_remainder_modes"] = 209
    tampered["content_sha256"] = _content_hash(tampered)
    with pytest.raises(SphereNormalFormRegistrationError, match="campaign replay mismatch"):
        validate_campaign(tampered, ROOT)
