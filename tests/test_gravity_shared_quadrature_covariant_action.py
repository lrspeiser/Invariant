from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler import gravity_shared_quadrature_covariant_action as action

ROOT = Path(__file__).resolve().parents[1]


def test_config_and_all_predecessor_commits_are_exact() -> None:
    config = action.load_config(ROOT)
    results = action.validate_predecessors(ROOT, config)
    assert [item["binding_id"] for item in results] == [
        "shared_formula_scalar_kinetic_reconstruction",
        "universal_conformal_source",
        "covariant_field_equations",
        "deep_aqual_transition_tradeoff",
    ]
    assert [item["artifact_count"] for item in results] == [4, 4, 4, 1]
    assert all(item["all_current_and_commit_hashes_match"] for item in results)


def test_symbolic_action_motion_stress_and_lensing_checks_pass() -> None:
    checks, formulas = action.symbolic_checks()
    assert len(checks) == 24
    assert all(item["passed"] for item in checks)
    assert tuple(item["check_id"] for item in checks) == action.SYMBOLIC_CHECK_IDS
    assert formulas["dimensionless_kinetic_density"] == "s^2/4+s/4+ln(1-2*s)/8"
    assert formulas["p_Xbar"] == "alpha^2*s/(1-2*s)"
    assert formulas["motion_law"] == "y=sqrt(x^2+x)"


def test_numeric_branch_is_locally_positive_but_everywhere_superluminal() -> None:
    probes = action.numeric_checks(action.load_config(ROOT))
    assert len(probes) == 4
    assert all(item["passed"] for item in probes)
    assert all(item["C_over_alpha_squared"] > 0 for item in probes)
    assert all(item["K_over_alpha_squared"] > 0 for item in probes)
    assert all(item["K_over_C"] > 2 for item in probes)
    assert all(item["normalized_energy_density"] > 0 for item in probes)
    assert all(item["normalized_radial_NEC"] > 0 for item in probes)
    assert all(item["normalized_tangential_NEC"] == pytest.approx(0) for item in probes)
    assert probes[-1]["K_over_C"] == pytest.approx(51)


def test_action_contract_uses_one_universal_metric_and_no_photon_adjustment() -> None:
    config = action.load_config(ROOT)
    contract = config["restricted_action_contract"]
    assert "S_m[exp(2 alpha phi/Mpl) g,psi]" in contract["einstein_frame_action"]
    assert "every massive and photon field" in contract["matter_metric"]
    assert config["adjudication"]["universal_massive_matter_and_photon_metric_defined"]
    assert config["adjudication"]["separate_photon_adjustment_present"] is False


def test_receipt_preserves_exact_successes_and_open_physics_gates() -> None:
    receipt = action.build_receipt(ROOT)
    assert receipt["decision"] == action.DECISION
    assert receipt["counts"]["symbolic_checks_passed"] == 24
    assert receipt["counts"]["numeric_branch_probes_passed"] == 4
    assert receipt["adjudication"]["restricted_spacelike_covariant_action_defined"] is True
    assert receipt["adjudication"]["quadrature_motion_law_recovered_exactly"] is True
    assert receipt["adjudication"]["scalar_stress_tensor_derived"] is True
    assert receipt["adjudication"]["direct_conformal_lensing_shift_cancels"] is True
    assert receipt["adjudication"]["same_action_quantitative_lensing_solution_derived"] is False
    assert receipt["adjudication"]["scalar_cone_causal_relative_to_conformal_matter_cone"] is False
    assert receipt["adjudication"]["low_gradient_transition_nondegenerate"] is False
    assert receipt["adjudication"]["finite_gradient_endpoint_regular"] is False
    assert receipt["adjudication"]["timelike_cosmological_branch_defined"] is False
    assert receipt["adjudication"]["CP11_1_complete"] is False
    assert receipt["adjudication"]["CP11_4_complete"] is False
    assert receipt["adjudication"]["CP11_8_complete"] is False
    assert receipt["claim_boundary"]["scientific_observational_claim_allowed"] is False
    assert set(receipt["zero_access_and_compute"].values()) == {0}


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["restricted_action_contract"].__setitem__(
            "scope_limit", "global healthy action"
        ),
        lambda value: value["stress_and_lensing_contract"].__setitem__(
            "unsolved_requirement", "none"
        ),
        lambda value: value["adjudication"].__setitem__(
            "same_action_quantitative_lensing_solution_derived", True
        ),
        lambda value: value["adjudication"].__setitem__(
            "scalar_cone_causal_relative_to_conformal_matter_cone", True
        ),
        lambda value: value["claim_boundary"].__setitem__(
            "scientific_observational_claim_allowed", True
        ),
        lambda value: value["zero_access_and_compute"].__setitem__("network_calls", 1),
    ],
)
def test_config_mutations_fail_closed(mutation: object) -> None:
    changed = copy.deepcopy(action.load_config(ROOT))
    mutation(changed)  # type: ignore[operator]
    with pytest.raises(action.QuadratureActionError, match="content changed"):
        action.validate_config(changed)


def test_stored_receipt_rebuilds_exactly() -> None:
    stored = json.loads((ROOT / action.OUTPUT_PATH).read_text(encoding="utf-8"))
    action.validate_receipt(stored, ROOT)
    assert stored == action.build_receipt(ROOT)


def test_atomic_writer_refuses_different_existing_bytes(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    assert action._atomic_no_clobber(path, b"first\n") == "CREATED"
    assert action._atomic_no_clobber(path, b"first\n") == "EXISTING_IDENTICAL"
    assert action._atomic_no_clobber(path, b"second\n") == "EXISTING_DIFFERENT"
    assert path.read_bytes() == b"first\n"
