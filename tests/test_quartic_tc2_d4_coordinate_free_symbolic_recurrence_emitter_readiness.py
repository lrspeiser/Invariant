from __future__ import annotations

import copy
import json
from fractions import Fraction
from pathlib import Path

import pytest

from sigma_theory_compiler.quartic_tc2_d4_coordinate_free_symbolic_recurrence_emitter_readiness import (
    REQUIRED_ROWS,
    SPECTRUM,
    SymbolicRecurrenceEmitterReadinessError,
    _content_hash,
    build_campaign,
    lagrange_projector_recipes,
    validate_campaign,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs"
    / "backgrounds"
    / "quartic_tc2_d4_coordinate_free_symbolic_recurrence_emitter_readiness.json"
)
ARTIFACT = (
    ROOT
    / "runs"
    / "physics-language"
    / "quartic-tc2-d4-coordinate-free-symbolic-recurrence-emitter-readiness"
    / "campaign.json"
)


@pytest.fixture(scope="module")
def artifact() -> dict:
    document = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    validate_campaign(document, ROOT)
    return document


def test_all_exact_lagrange_projector_recipes_are_registered(artifact: dict) -> None:
    projector = artifact["exact_Lagrange_projector_recipes"]
    recipes = projector["recipes"]
    assert projector["degree"] == 6
    assert len(recipes) == 7
    assert sum(len(recipe["coefficients_low_to_high"]) for recipe in recipes) == 49
    assert projector["recipes_verified_by_exact_Kronecker_evaluation"] is True
    assert projector["matrix_projectors_evaluated"] is False
    for index, recipe in enumerate(recipes):
        expected = ["0"] * 7
        expected[index] = "1"
        assert recipe["Kronecker_evaluations_in_declared_spectrum_order"] == expected


def test_projector_recipe_duplicate_spectrum_fails_closed() -> None:
    with pytest.raises(SymbolicRecurrenceEmitterReadinessError, match="not distinct"):
        lagrange_projector_recipes([Fraction(0), Fraction(0)])


def test_exact_gradient_lift_coefficients_are_emitted(artifact: dict) -> None:
    lift = artifact["exact_gradient_lift_pencil"]
    assert lift["registered"] is True
    assert lift["field_columns"] == 11
    assert lift["spatial_gradient_rows"] == 33
    matrices = lift["coefficient_matrices"]
    assert [record["pencil_component"] for record in matrices] == ["L_1", "L_2", "L_3"]
    assert [record["nonzero_entries"][0]["row"] for record in matrices] == [44, 11, 22]
    assert sum(len(record["nonzero_entries"]) for record in matrices) == 33


def test_required_input_manifest_is_closed_without_zero_inference(artifact: dict) -> None:
    records = {
        record["input_id"]: record for record in artifact["required_symbolic_input_manifest"]
    }
    assert len(records) == 8
    assert records["physical_spatial_pencil_coefficients"]["required_packets"] == 3
    assert records["physical_spatial_pencil_coefficients"]["registered_packets"] == 0
    assert records["physical_gradient_lift_coefficients"]["registered_packets"] == 3
    assert records["polarized_P55_Taylor_packets"]["required_packets"] == 75
    assert records["polarized_K55_Taylor_packets"]["required_packets"] == 75
    assert records["polarized_TC2_Taylor_packets"]["required_packets"] == 75
    assert records["lower_Sylvester_correction_recurrence"]["required_packets"] == 60
    assert records["candidate_normalization_table"]["required_packets"] == 12
    assert records["sphere_mode_normal_form_reducer"]["required_packets"] == 1
    assert artifact["counts"]["required_symbolic_input_packets"] == 304
    assert artifact["counts"]["registered_symbolic_input_packets"] == 3
    assert artifact["counts"]["missing_symbolic_input_packets"] == 301


def test_sparse_checkpoint_stays_at_first_missing_row(artifact: dict) -> None:
    checkpoint = artifact["bounded_sparse_emission_checkpoint"]
    assert checkpoint["complete"] is False
    assert checkpoint["first_missing_input"] == "physical_spatial_pencil_coefficients"
    assert checkpoint["row_emission_cursor"] == {
        "next_cokernel_coordinate": 0,
        "next_flat_row_offset": 0,
        "next_odd_sphere_mode": 0,
    }
    assert checkpoint["counts"]["required_output_rows"] == REQUIRED_ROWS
    assert checkpoint["counts"]["emitted_output_rows"] == 0
    assert checkpoint["counts"]["missing_output_rows"] == REQUIRED_ROWS
    assert checkpoint["emitted_rows"] == []
    assert checkpoint["emitted_rhs_rows"] == []
    assert checkpoint["content_sha256"] == _content_hash(checkpoint)


def test_small_exact_pass_and_obstruction_controls(artifact: dict) -> None:
    controls = artifact["small_exact_controls"]
    passed = controls["exact_solvable_Sylvester_control"]
    obstructed = controls["exact_equal_eigenspace_obstruction_control"]
    assert passed["decision"] == "PASS"
    assert passed["residual"] == [["0", "0"], ["0", "0"]]
    assert obstructed["decision"] == "OBSTRUCTED_CLASS"
    assert obstructed["zero_eigenspace_dimension"] == 2
    assert obstructed["compressed_rhs"] == [["0", "1"], ["-1", "0"]]
    assert obstructed["witness_entry"] == {"row": 0, "column": 1, "value": "1"}


def test_phase_two_is_not_admitted(artifact: dict) -> None:
    phase = artifact["phase_two"]
    assert phase["decision"] == "BLOCK"
    assert phase["admitted"] is False
    assert phase["attempted"] is False
    assert phase["PASS"] is False
    assert phase["OBSTRUCTED_CLASS"] is False
    assert phase["BLOCK"] is True
    assert artifact["counts"]["phase_two_solve_attempts"] == 0


def test_broad_claims_remain_false(artifact: dict) -> None:
    claims = artifact["claims"]
    assert claims["all_seven_exact_Lagrange_projector_recipes_registered"] is True
    assert claims["exact_gradient_lift_pencil_registered"] is True
    for claim in (
        "complete_coordinate_free_coefficient_map_emitted",
        "complete_coordinate_free_rhs_emitted",
        "phase_two_exact_solve_admitted",
        "full_direction_sphere_D4_compatibility_proved",
        "complete_D2F_tensor_registered",
        "full_high_atom_identity_proved",
        "TC2_closed",
        "global_H7_closed",
        "full_tube_Sylvester_identity_proved",
        "nonlinear_PDE_closure_proved",
        "lifespan_proved",
        "theory_candidate_rejected",
        "unregistered_symbolic_inputs_treated_as_zero",
    ):
        assert claims[claim] is False


def test_negative_controls_all_reject(artifact: dict) -> None:
    assert len(artifact["negative_controls"]) == 8
    assert all(control == {"rejected": True} for control in artifact["negative_controls"].values())


def test_artifact_replays_deterministically(artifact: dict) -> None:
    assert build_campaign(ROOT, CONFIG) == artifact
    assert artifact["content_sha256"] == _content_hash(artifact)


def test_semantic_tamper_fails_closed(artifact: dict) -> None:
    tampered = copy.deepcopy(artifact)
    tampered["counts"]["emitted_output_rows"] = 1
    tampered["content_sha256"] = _content_hash(tampered)
    with pytest.raises(SymbolicRecurrenceEmitterReadinessError, match="campaign replay mismatch"):
        validate_campaign(tampered, ROOT)


def test_upstream_binding_tamper_fails_closed(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["upstream_bindings"]["reference_Sylvester_space"]["content_sha256"] = "0" * 64
    config["content_sha256"] = _content_hash(config)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(SymbolicRecurrenceEmitterReadinessError, match="config upstream hash"):
        build_campaign(ROOT, path)


def test_resource_cap_tamper_fails_closed(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["resource_caps"]["maximum_projector_recipe_terms"] = 48
    config["content_sha256"] = _content_hash(config)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(SymbolicRecurrenceEmitterReadinessError, match="invalid emitter"):
        build_campaign(ROOT, path)


def test_declared_spectrum_recipes_are_stable() -> None:
    recipes = lagrange_projector_recipes(SPECTRUM)
    assert [recipe["eigenvalue"] for recipe in recipes] == [
        "0",
        "1",
        "-1",
        "1/2",
        "-1/2",
        "1/3",
        "-1/3",
    ]
