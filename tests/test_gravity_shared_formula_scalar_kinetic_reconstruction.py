from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler import gravity_shared_formula_scalar_kinetic_reconstruction as kinetic

ROOT = Path(__file__).resolve().parents[1]


def test_config_and_predecessor_commits_are_exact() -> None:
    config = kinetic.load_config(ROOT)
    results = kinetic.validate_predecessors(ROOT, config)
    assert [item["binding_id"] for item in results] == [
        "ben_synthetic_registry",
        "ben_development_executor_v4",
        "covariant_field_equations",
        "deep_aqual_transition_tradeoff",
    ]
    assert [item["artifact_count"] for item in results] == [1, 4, 1, 1]
    assert all(item["all_current_and_commit_hashes_match"] for item in results)


def test_all_60_classes_are_structurally_classified_without_target_rows() -> None:
    config = kinetic.load_config(ROOT)
    result = kinetic.classify_registry(ROOT, config)
    assert result["canonical_classes"] == 60
    assert result["source_only_classes"] == 3
    assert result["auxiliary_dependent_classes"] == 57
    assert result["variable_signature_counts"] == {
        "x_source": 3,
        "x_geometry+x_source": 3,
        "x_radial+x_source": 6,
        "x_radial+x_source+x_state": 12,
        "x_geometry+x_radial+x_source": 18,
        "x_geometry+x_radial+x_source+x_state": 18,
    }
    source_only = result["source_only_classes_detail"]
    assert [item["class_id"] for item in source_only] == [
        "ben.a3da343620d23c63b16a",
        "ben.cfe53a02a87ccf24af9c",
        "ben.f9a69717841da3b4e1cc",
    ]
    assert all(item["variables"] == ["x_source"] for item in source_only)
    assert all(item["minimal_single_scalar_reconstruction_eligible"] for item in source_only)


def test_symbolic_formula_to_kinetic_derivation_passes() -> None:
    checks, formulas = kinetic.symbolic_checks()
    assert len(checks) == 16
    assert all(item["passed"] for item in checks)
    assert tuple(item["check_id"] for item in checks) == kinetic.SYMBOLIC_CHECK_IDS
    assert formulas["general_C"] == "alpha^2*x/s"
    assert formulas["general_K_source_only"] == "alpha^2/(ds/dx)"
    assert formulas["quadrature_x_of_s"] == "s^2/(1-2*s)"


def test_numeric_quadrature_branch_preserves_cone_and_endpoint_warning() -> None:
    result = kinetic.numeric_checks(kinetic.load_config(ROOT))
    probes = result["quadrature_probes"]
    assert len(probes) == 5
    assert all(item["passed"] for item in probes)
    assert all(item["C_over_alpha_squared"] > 0 for item in probes)
    assert all(item["K_over_alpha_squared"] > 0 for item in probes)
    assert all(item["K_over_C"] > 2 for item in probes)
    assert probes[-1]["s"] == pytest.approx(0.5, abs=2e-7)
    assert probes[-1]["K_over_C"] > 4_000_000


def test_numeric_rar_branch_is_multivalued_and_changes_K_sign() -> None:
    result = kinetic.numeric_checks(kinetic.load_config(ROOT))
    turnover = result["rar_turnover"]
    assert turnover["r"] == pytest.approx(1.59362426004004)
    assert turnover["x"] == pytest.approx(2.53963828218817)
    assert turnover["s"] == pytest.approx(0.647610237891915)
    witness = result["rar_same_excess_witness"]
    assert [item["s"] for item in witness] == [0.5, 0.5]
    assert witness[0]["x"] == pytest.approx(0.548859359305037)
    assert witness[1]["x"] == pytest.approx(8.0807921327933)
    assert witness[0]["C_over_alpha_squared"] != pytest.approx(witness[1]["C_over_alpha_squared"])
    assert result["rar_excess_derivative_below_turnover"] > 0
    assert result["rar_excess_derivative_above_turnover"] < 0
    assert result["rar_multivalued_and_K_sign_change_confirmed"] is True


def test_receipt_is_partial_target_independent_and_zero_access() -> None:
    receipt = kinetic.build_receipt(ROOT)
    assert receipt["decision"] == kinetic.DECISION
    assert receipt["counts"]["canonical_formula_classes"] == 60
    assert receipt["counts"]["source_only_formula_classes"] == 3
    assert receipt["counts"]["auxiliary_dependent_formula_classes"] == 57
    assert receipt["counts"]["symbolic_checks_passed"] == 16
    assert receipt["adjudication"]["formula_to_minimal_kinetic_map_derived"] is True
    assert receipt["adjudication"]["quadrature_minimal_map_single_valued_and_locally_positive"]
    assert (
        receipt["adjudication"]["quadrature_minimal_map_causal_relative_to_conformal_matter_cone"]
        is False
    )
    assert receipt["adjudication"]["rar_like_minimal_map_single_valued_globally"] is False
    assert receipt["adjudication"]["full_covariant_formula_bridge_derived"] is False
    assert receipt["claim_boundary"]["surviving_physical_candidate_selected"] is False
    assert receipt["claim_boundary"]["publication_readiness_changed"] is False
    assert set(receipt["zero_access_and_compute"].values()) == {0}


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["registry_structural_contract"].__setitem__(
            "source_only_classes_expected", 4
        ),
        lambda value: value["source_only_adjudication_contract"]["quadrature_class"].__setitem__(
            "adjudication", "PASS_HEALTHY"
        ),
        lambda value: value["adjudication"].__setitem__("healthy_action", True),
        lambda value: value["claim_boundary"].__setitem__(
            "scientific_observational_claim_allowed", True
        ),
        lambda value: value["zero_access_and_compute"].__setitem__("network_calls", 1),
    ],
)
def test_config_mutations_fail_closed(mutation: object) -> None:
    changed = copy.deepcopy(kinetic.load_config(ROOT))
    mutation(changed)  # type: ignore[operator]
    with pytest.raises(kinetic.FormulaKineticReconstructionError, match="content changed"):
        kinetic.validate_config(changed)


def test_stored_receipt_rebuilds_exactly() -> None:
    stored = json.loads((ROOT / kinetic.OUTPUT_PATH).read_text(encoding="utf-8"))
    kinetic.validate_receipt(stored, ROOT)
    assert stored == kinetic.build_receipt(ROOT)


def test_atomic_writer_refuses_different_existing_bytes(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    assert kinetic._atomic_no_clobber(path, b"first\n") == "CREATED"
    assert kinetic._atomic_no_clobber(path, b"first\n") == "EXISTING_IDENTICAL"
    assert kinetic._atomic_no_clobber(path, b"second\n") == "EXISTING_DIFFERENT"
    assert path.read_bytes() == b"first\n"
