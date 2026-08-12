from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from sigma_theory_compiler.anonymous_massive_vector_chronological_rediscovery import (
    CONFIG_PATH,
    OUTPUT_PATH,
    SOURCE_PATH,
    TEST_PATH,
    _canonical,
    _canonical_signature,
    _derive,
    build_benchmark,
    validate_result,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / CONFIG_PATH
ARTIFACT = ROOT / OUTPUT_PATH


@pytest.fixture(scope="module")
def result() -> dict[str, object]:
    value = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    validate_result(value, root=ROOT)
    assert build_benchmark(CONFIG) == value
    return value


def test_generation_closure_is_anonymous_and_precedes_unseal(result: dict[str, object]) -> None:
    audit = result["pre_unseal_leakage_audit"]
    assert audit["pre_unseal_dependency_paths"] == [CONFIG_PATH, SOURCE_PATH]
    assert TEST_PATH not in audit["pre_unseal_dependency_paths"]
    assert audit["forbidden_concept_count"] == 0
    assert audit["forbidden_concept_matches"] == []
    assert audit["unexpected_import_roots"] == []
    assert audit["passed"] is True
    io_contract = result["pre_unseal_phase_io_contract"]
    assert io_contract["enforcement_surfaces"] == [
        "builtins.open",
        "io.open",
        "pathlib.Path.open",
    ]
    assert io_contract["enforcement_scope"] == (
        "owned_single_threaded_python_file_read_surfaces_with_static_import_allowlist_"
        "not_an_operating_system_sandbox"
    )
    assert io_contract["allowed_paths"] == [CONFIG_PATH, SOURCE_PATH]
    assert io_contract["attempted_access_count"] == 9
    assert io_contract["allowed_access_count"] == 8
    assert io_contract["denied_access_count"] == 1
    assert io_contract["denied_paths"] == [TEST_PATH]
    assert io_contract["denied_content_bytes_exposed"] == 0
    assert [row["sequence"] for row in io_contract["attempted_accesses"]] == list(range(9))
    assert io_contract["attempted_accesses"][-1] == {
        "sequence": 8,
        "path": TEST_PATH,
        "mode": "rb",
        "surface": "pathlib.Path.open",
        "decision": "denied",
    }
    assert [row["sequence"] for row in result["chronology"]] == list(range(8))
    assert result["chronology"][1]["phase"] == "pre_unseal_file_reads_enforced"
    assert result["chronology"][6]["phase"] == "blinded_pareto_ranking_sealed"
    assert result["chronology"][7]["phase"] == "selected_structure_unsealed"


def test_enumeration_is_complete_deterministic_and_quotiented(result: dict[str, object]) -> None:
    enumeration = result["enumeration"]
    assert enumeration["raw_cartesian_candidates"] == 625
    assert enumeration["raw_orbit_multiplicity_sum"] == 625
    assert enumeration["canonical_equivalence_classes"] > 0
    assert enumeration["integration_by_parts_reduction"] == "q1_plus_q2_to_qcross"
    assert enumeration["normalization"] == "greatest_common_divisor_positive_scale_only"
    assert enumeration["field_relabel_orbit_size"] == 1
    assert build_benchmark(CONFIG) == build_benchmark(CONFIG)


def test_only_declared_integration_by_parts_and_positive_normalization_are_quotiented() -> None:
    assert _canonical_signature((-2, 1, 1, -2)) == (-1, 1, -1)
    assert _canonical_signature((-1, 0, 1, -1)) == (-1, 1, -1)
    assert _canonical_signature((1, -1, 0, 1)) == (1, -1, 1)
    assert _canonical_signature((1, -1, 0, 1)) != _canonical_signature((-1, 1, 0, -1))


def test_blinded_pareto_rank_is_sealed_before_coefficients(result: dict[str, object]) -> None:
    ranking = result["blinded_pareto_ranking"]
    assert ranking["coefficient_visibility_before_unseal"] is False
    assert ranking["pareto_front_size"] == 1
    assert len(ranking["pareto_front"]) == 1
    assert set(ranking["pareto_front"][0]) == {"candidate_id", "objectives"}
    assert ranking["selected_candidate_id"] == result["unsealed_result"]["selected_candidate_id"]
    phase = result["chronology"][6]
    assert phase["root_sha256"] == result["blinded_pre_unseal_root_sha256"]


def test_independent_euler_divergence_constraint_and_dof_derivation(
    result: dict[str, object],
) -> None:
    unsealed = result["unsealed_result"]
    assert unsealed["canonical_representative"] == {"q0": -1, "qcross": 1, "qm": -1}
    assert unsealed["independent_euler_derivation"] == {
        "box_rank_one_field": 1,
        "gradient_of_divergence": -1,
        "algebraic_rank_one_field": -1,
    }
    assert unsealed["independent_divergence_derivation"] == {
        "box_of_divergence": 0,
        "algebraic_divergence": -1,
    }
    assert unsealed["velocity_hessian_diagonal"] == [0, 1, 1, 1]
    assert unsealed["velocity_hessian_rank"] == 3
    assert unsealed["canonical_hamiltonian_derivation"] == {
        "canonical_momenta": {
            "pi0_velocity_coefficient": 0,
            "pi0_spatial_divergence_coefficient": -1,
            "pi_spatial_velocity_coefficient": 1,
        },
        "primary_constraint": {
            "exists": True,
            "pi0_coefficient": 1,
            "spatial_divergence_coefficient": 1,
        },
        "secondary_constraint": {
            "exists": True,
            "spatial_momentum_divergence_coefficient": -1,
            "laplacian_time_component_coefficient": 1,
            "algebraic_time_component_coefficient": -1,
        },
        "primary_secondary_poisson_bracket": {
            "laplacian_delta_coefficient": 0,
            "algebraic_delta_coefficient": 1,
            "two_constraint_matrix_determinant": 1,
            "nonzero": True,
        },
        "velocity_hessian_diagonal": [0, 1, 1, 1],
        "velocity_hessian_rank": 3,
        "constraint_class": "second_class_pair",
        "first_class_constraint_count": 0,
        "second_class_constraint_count": 2,
        "phase_space_dimension": 8,
        "physical_degrees_of_freedom": 3,
    }
    assert unsealed["constraint_class"] == "second_class_pair"
    assert unsealed["first_class_constraint_count"] == 0
    assert unsealed["second_class_constraint_count"] == 2
    assert (unsealed["phase_space_dimension"] - unsealed["second_class_constraint_count"]) // 2 == 3
    assert unsealed["physical_degrees_of_freedom"] == 3
    assert unsealed["mass_squared"] == {"numerator": 1, "denominator": 1}
    assert unsealed["discovered_structure"] == {
        "first_jet_coefficients_sum_to_zero": True,
        "algebraic_scale_nonzero": True,
        "positive_spatial_kinetic": True,
        "positive_mass_squared": True,
    }


def test_all_three_positive_mass_representatives_are_exposed_after_unseal(
    result: dict[str, object],
) -> None:
    unsealed = result["unsealed_result"]
    assert unsealed["eligible_mass_squared_values"] == [
        {"numerator": 1, "denominator": 2},
        {"numerator": 1, "denominator": 1},
        {"numerator": 2, "denominator": 1},
    ]
    rows = unsealed["eligible_positive_mass_representatives"]
    assert [row["canonical_representative"] for row in rows] == [
        {"q0": -2, "qcross": 2, "qm": -1},
        {"q0": -1, "qcross": 1, "qm": -1},
        {"q0": -1, "qcross": 1, "qm": -2},
    ]
    assert [row["selected_by_pareto"] for row in rows] == [False, True, False]
    assert unsealed["selection_statement"] == (
        "unit_mass_representative_selected_by_declared_simplicity_objectives_from_three_"
        "eligible_positive_mass_ratios_not_unique_theory_or_free_mass_family"
    )


def test_post_unseal_matches_proca_equivalence_class(result: dict[str, object]) -> None:
    representative = result["unsealed_result"]["canonical_representative"]
    # Expanding -1/4 F_{mu nu} F^{mu nu} - 1/2 m^2 A_mu A^mu in the
    # benchmark's 1/2-normalized primitive basis gives this post-unseal signature.
    assert representative == {"q0": -1, "qcross": 1, "qm": -1}
    assert result["claims"]["post_unseal_reference_equivalence_check_defined"] is True


def test_wrong_sign_massless_longitudinal_and_extra_derivative_negatives(
    result: dict[str, object],
) -> None:
    controls = {row["control_id"]: row for row in result["negative_controls"]}
    assert set(controls) == {
        "kinetic_sign_reversal",
        "zero_algebraic_scale",
        "propagating_longitudinal_mode",
        "four_derivative_intrusion",
    }
    assert all(row["eligible"] is False for row in controls.values())
    assert "positive_spatial_kinetic" in controls["kinetic_sign_reversal"]["violated_constraints"]
    assert controls["zero_algebraic_scale"]["physical_degrees_of_freedom"] == 2
    assert controls["zero_algebraic_scale"]["constraint_class"] == "first_class_pair"
    assert controls["zero_algebraic_scale"]["primary_secondary_poisson_bracket_determinant"] == 0
    assert "algebraic_divergence" in controls["zero_algebraic_scale"]["violated_constraints"]
    assert controls["propagating_longitudinal_mode"]["physical_degrees_of_freedom"] == 4
    assert controls["propagating_longitudinal_mode"]["primary_constraint_exists"] is False
    assert (
        controls["propagating_longitudinal_mode"]["primary_secondary_poisson_bracket_determinant"]
        is None
    )
    assert controls["four_derivative_intrusion"]["maximum_action_derivative_count"] == 4
    assert "derivative_bound" in controls["four_derivative_intrusion"]["violated_constraints"]


def test_positive_normalization_does_not_hide_wrong_sign() -> None:
    healthy = _derive((-1, 1, -1))
    reversed_sign = _derive((1, -1, 1))
    assert healthy["eligible"] is True
    assert reversed_sign["eligible"] is False
    assert reversed_sign["violations"]["positive_spatial_kinetic"] is True


@pytest.mark.parametrize(
    ("path_key", "mutation"),
    [
        ("config", lambda value: value["discovery_constraints"].__setitem__("massive", False)),
        ("source", lambda value: value.__setitem__("decision", "forged")),
        ("test", lambda value: value["claims"].__setitem__("observational_support", True)),
    ],
)
def test_local_binding_and_resealed_semantic_tamper_fail_closed(
    tmp_path: Path, path_key: str, mutation: object
) -> None:
    for relative in (CONFIG_PATH, SOURCE_PATH, TEST_PATH, OUTPUT_PATH):
        source = ROOT / relative
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    artifact_path = tmp_path / OUTPUT_PATH
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    if path_key == "config":
        config_path = tmp_path / CONFIG_PATH
        config = json.loads(config_path.read_text(encoding="utf-8"))
        mutation(config)
        config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    else:
        mutation(artifact)
        body = {key: value for key, value in artifact.items() if key != "content_sha256"}
        artifact["content_sha256"] = hashlib.sha256(_canonical(body)).hexdigest()
        if path_key == "test":
            artifact["source_bindings"]["test"]["file_sha256"] = "0" * 64
            body = {key: value for key, value in artifact.items() if key != "content_sha256"}
            artifact["content_sha256"] = hashlib.sha256(_canonical(body)).hexdigest()
        artifact_path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        validate_result(json.loads(artifact_path.read_text(encoding="utf-8")), root=tmp_path)


def test_unknown_top_level_key_fails_closed(tmp_path: Path) -> None:
    for relative in (CONFIG_PATH, SOURCE_PATH, TEST_PATH, OUTPUT_PATH):
        source = ROOT / relative
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    artifact_path = tmp_path / OUTPUT_PATH
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    artifact["unknown"] = False
    body = {key: value for key, value in artifact.items() if key != "content_sha256"}
    artifact["content_sha256"] = hashlib.sha256(_canonical(body)).hexdigest()
    artifact_path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="result keys changed"):
        validate_result(artifact, root=tmp_path)


def test_resealed_io_certificate_and_constraint_determinant_tamper_fail_closed(
    tmp_path: Path,
) -> None:
    for relative in (CONFIG_PATH, SOURCE_PATH, TEST_PATH, OUTPUT_PATH):
        source = ROOT / relative
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    artifact_path = tmp_path / OUTPUT_PATH
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    artifact["pre_unseal_phase_io_contract"]["denied_access_count"] = 0
    artifact["unsealed_result"]["canonical_hamiltonian_derivation"][
        "primary_secondary_poisson_bracket"
    ]["two_constraint_matrix_determinant"] = 0
    body = {key: value for key, value in artifact.items() if key != "content_sha256"}
    artifact["content_sha256"] = hashlib.sha256(_canonical(body)).hexdigest()
    with pytest.raises(ValueError, match="result boundary changed"):
        validate_result(artifact, root=tmp_path)


@pytest.mark.parametrize("forbidden_text", ["proca", "known_answer", "target_coefficients"])
def test_rebound_pre_unseal_dependency_leakage_fails_closed(
    tmp_path: Path, forbidden_text: str
) -> None:
    for relative in (CONFIG_PATH, SOURCE_PATH, TEST_PATH, OUTPUT_PATH):
        source = ROOT / relative
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    source_path = tmp_path / SOURCE_PATH
    source_path.write_text(
        source_path.read_text(encoding="utf-8") + f"\n# {forbidden_text}\n", encoding="utf-8"
    )
    artifact_path = tmp_path / OUTPUT_PATH
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    rebound_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
    artifact["source_bindings"]["source"]["file_sha256"] = rebound_sha
    artifact["pre_unseal_input_bindings"][1]["file_sha256"] = rebound_sha
    body = {key: value for key, value in artifact.items() if key != "content_sha256"}
    artifact["content_sha256"] = hashlib.sha256(_canonical(body)).hexdigest()
    with pytest.raises(ValueError, match="leakage audit failed"):
        validate_result(artifact, root=tmp_path)


def test_claim_boundary_remains_conservative(result: dict[str, object]) -> None:
    assert result["claims"] == {
        "complete_for_declared_finite_coefficient_box": True,
        "unique_blinded_pareto_winner": True,
        "pre_unseal_file_reads_enforced": True,
        "three_positive_mass_representatives_exposed": True,
        "flat_integration_by_parts_only": True,
        "post_unseal_reference_equivalence_check_defined": True,
        "unique_massive_equivalence_class_proved": False,
        "free_mass_family_proved": False,
        "unbounded_coefficient_space_exhausted": False,
        "interacting_vector_theories_classified": False,
        "curvature_coupling_classes_classified": False,
        "novel_theory_discovered": False,
        "observational_support": False,
    }
    assert result["first_remaining_blocker"].startswith("extend_beyond_quadratic")
