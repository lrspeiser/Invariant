from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.quartic_tc2_d4_coordinate_free_k0_directional_lift import (
    CoordinateFreeK0DirectionalLiftError,
    _content_hash,
    build_campaign,
    validate_campaign,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / ("configs/backgrounds/quartic_tc2_d4_coordinate_free_k0_directional_lift.json")
ARTIFACT = ROOT / (
    "runs/physics-language/quartic-tc2-d4-coordinate-free-k0-directional-lift/campaign.json"
)


@pytest.fixture(scope="module")
def artifact() -> dict:
    document = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    validate_campaign(document, ROOT)
    return document


def test_basis_free_formula_is_exact_and_closed(artifact: dict) -> None:
    formula = artifact["coordinate_free_K0_formula"]
    assert formula["content_sha256"] == _content_hash(formula)
    assert formula["domain"] == "n1^2+n2^2+n3^2=1"
    assert formula["block_theorem"]["F_C_equals_L_transpose_G"] is True
    assert formula["block_theorem"]["G_C_minus_C_transpose_G_zero"] is True


def test_e1_reference_K0_is_reproduced_entrywise(artifact: dict) -> None:
    assert artifact["counts"]["e1_reference_matrix_entries_compared"] == 3025
    assert artifact["counts"]["e1_reference_matrix_mismatches"] == 0
    assert artifact["claims"]["e1_reference_K0_reproduced_exactly"] is True


def test_all_six_exact_direction_controls_pass(artifact: dict) -> None:
    controls = artifact["exact_direction_controls"]
    assert len(controls) == 6
    assert all(row["P55_symmetrizer_residual_nonzero_entries"] == 0 for row in controls)
    assert [row["direction"] for row in controls[3:]] == [
        ["3/5", "4/5", "0"],
        ["1/3", "2/3", "2/3"],
        ["0", "3/5", "4/5"],
    ]


def test_expansion_and_broad_claims_remain_open(artifact: dict) -> None:
    assert artifact["polynomial_serialization_boundary"]["expanded_packet_emitted"] is False
    for claim in (
        "expanded_55x55_polynomial_K0_packet_emitted",
        "K55_Taylor_order_one_registered",
        "full_direction_sphere_D4_compatibility_proved",
        "global_H7_closed",
        "nonlinear_PDE_closure_proved",
        "lifespan_proved",
    ):
        assert artifact["claims"][claim] is False


def test_exact_replay_and_resealed_tamper_fail_closed(artifact: dict) -> None:
    assert build_campaign(ROOT, CONFIG) == artifact
    assert artifact["content_sha256"] == _content_hash(artifact)
    tampered = copy.deepcopy(artifact)
    tampered["counts"]["e1_reference_matrix_mismatches"] = 1
    tampered["content_sha256"] = _content_hash(tampered)
    with pytest.raises(CoordinateFreeK0DirectionalLiftError, match="campaign replay mismatch"):
        validate_campaign(tampered, ROOT)
