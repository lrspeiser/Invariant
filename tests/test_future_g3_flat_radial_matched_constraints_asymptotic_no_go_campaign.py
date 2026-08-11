from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.future_g3_flat_radial_matched_constraints_asymptotic_no_go_campaign import (
    FIRST_BLOCKER,
    _sha,
    _symbolic_leading_control,
    build_future_g3_flat_radial_matched_constraints_asymptotic_no_go_campaign,
)
from sigma_theory_compiler.promotion_orchestrator import ELIGIBILITY

ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT / "configs" / "future_g3_flat_radial_matched_constraints_asymptotic_no_go_campaign.json"
)
ARTIFACT = (
    ROOT
    / "runs"
    / "engine"
    / "future-g3-flat-radial-matched-constraints-asymptotic-no-go-campaign.json"
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def rebuilt() -> dict:
    return build_future_g3_flat_radial_matched_constraints_asymptotic_no_go_campaign(
        _load(CONFIG), ROOT
    )


def test_committed_artifact_is_exact_rebuild(rebuilt: dict) -> None:
    committed = _load(ARTIFACT)
    assert committed == rebuilt
    body = {key: item for key, item in committed.items() if key != "content_sha256"}
    assert committed["content_sha256"] == _sha(body)
    assert committed["content_sha256"] == (
        "9eeb346ddf1f57a65b5b0f352d2373bd1bf5438fd049ef33cc3264f1e61326aa"
    )
    assert hashlib.sha256(ARTIFACT.read_bytes()).hexdigest() == (
        "d3a10c1820acf026142844f391b0e9ecfee32646383816261fb4a53a61213189"
    )


def test_exact_joint_leading_obstruction_has_no_real_root(rebuilt: dict) -> None:
    symbolic = _symbolic_leading_control()
    assert rebuilt["symbolic_leading_order_control"] == symbolic
    assert symbolic["momentum_r_minus_3_coefficient"] == "alpha + 2*k"
    assert symbolic["Hamiltonian_r_minus_4_coefficient"] == ("2*alpha**2/3 - 2*k**2/3 + 1")
    assert symbolic["momentum_condition"] == "alpha=-2*k"
    assert symbolic["joint_reduced_Hamiltonian_coefficient"] == "2*k**2 + 1"
    assert symbolic["joint_real_solution_exists"] is False
    assert symbolic["formal_complex_k_roots"] == [
        "-sqrt(2)*I/2",
        "sqrt(2)*I/2",
    ]


def test_candidate_actions_are_bound_and_beta_is_only_subleading(rebuilt: dict) -> None:
    expected = {
        "G3A-8555e529226d13e2e9dacad5": (
            "4f31eb8efc25f3b28fd56d7d6dc6518461b1624f69f3b51ad8f05e6e7374e8eb",
            "33/4000",
        ),
        "G3A-8ec243e6dd285fd92e7b8e0c": (
            "08f435f45ff8f2333451d5cdad37bf201dfe58d254548ba6ccf5814564b98df0",
            "17/2000",
        ),
        "G3A-8d3cce39bcb13ba5061eb78b": (
            "181f6837c6934ff9ffbdba5b32383271704e67781a46b01d86fa31571442da98",
            "9/1000",
        ),
    }
    for record in rebuilt["candidate_records"]:
        action, beta = expected[record["candidate_id"]]
        certificate = record["flat_radial_matched_constraint_asymptotic_certificate"]
        asymptotics = certificate["exact_candidate_asymptotics"]
        assert record["action_sha256"] == action
        assert record["beta"] == beta
        assert certificate["action_sha256"] == action
        assert certificate["beta"] == beta
        assert certificate["direct_action_binding"] is True
        assert certificate["family_label_used_as_constraint_evidence"] is False
        assert asymptotics["matter_momentum_source_leading"].endswith("=O(r^-7)")
        assert asymptotics["cubic_Hamiltonian_term_leading"].endswith("=O(r^-8)")
        assert asymptotics["candidate_beta_absent_from_obstructing_orders"] is True


def test_one_constraint_and_complex_negative_controls_never_promote(
    rebuilt: dict,
) -> None:
    for record in rebuilt["candidate_records"]:
        certificate = record["flat_radial_matched_constraint_asymptotic_certificate"]
        momentum = certificate["momentum_only_negative_control"]
        hamiltonian = certificate["Hamiltonian_only_negative_control"]
        complex_control = certificate["complex_root_control"]
        assert momentum == {
            "k": "0",
            "alpha": "0",
            "momentum_coefficient": "0",
            "Hamiltonian_coefficient": "1",
            "joint_constraints_pass": False,
        }
        assert hamiltonian["Hamiltonian_coefficient"] == "0"
        assert hamiltonian["momentum_coefficient"] == "sqrt(6)"
        assert hamiltonian["joint_constraints_pass"] is False
        assert complex_control["role"] == "algebraic_negative_control_only"
        assert complex_control["rejected_by_real_initial_data_contract"] is True


def test_scope_counts_and_seals_remain_fail_closed(rebuilt: dict) -> None:
    assert rebuilt["candidate_count"] == 3
    assert rebuilt["decision_counts"] == {"blocked": 3}
    assert rebuilt["radial_momentum_leading_order_pass_count"] == 3
    assert rebuilt["flat_Hamiltonian_leading_order_pass_count"] == 3
    assert rebuilt["joint_real_asymptotic_coefficient_solution_count"] == 0
    assert rebuilt["flat_radial_matched_constraint_class_reject_count"] == 3
    assert rebuilt["registered_AF_metric_York_datum_pass_count"] == 0
    assert rebuilt["candidate_nontrivial_AF_Einstein_constraint_solution_pass_count"] == 0
    assert rebuilt["theory_reject_count"] == 0
    assert rebuilt["global_hamiltonian_energy_pass_count"] == 0
    assert rebuilt["full_formal_pass_count"] == 0
    assert rebuilt["first_blocker_counts"] == {FIRST_BLOCKER: 3}
    assert rebuilt["observational_data_opened"] is False
    assert rebuilt["dark_matter_or_halo_inputs"] is False
    assert rebuilt["redshift_distance_inputs"] is False
    assert rebuilt["paid_llm_spend_usd"] == 0.0
    assert rebuilt["data_eligibility"] == ELIGIBILITY
    assert rebuilt["synthetic_fixture_role"] == ("deterministic_symbolic_negative_controls_only")
    for record in rebuilt["candidate_records"]:
        certificate = record["flat_radial_matched_constraint_asymptotic_certificate"]
        assert certificate["decision"] == ("reject_flat_radial_r_minus_2_matched_constraint_class")
        assert certificate["theory_rejected"] is False
        assert record["candidate_nontrivial_AF_Einstein_constraint_solution_available"] is False
        assert record["global_energy_pass"] is False
        assert record["full_formal_pass"] is False


def test_action_contract_predecessor_source_and_eligibility_tamper_fail_closed() -> None:
    config = _load(CONFIG)

    action = copy.deepcopy(config)
    action["targets"][0]["action_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="target binding changed"):
        build_future_g3_flat_radial_matched_constraints_asymptotic_no_go_campaign(action, ROOT)

    contract = copy.deepcopy(config)
    contract["asymptotic_contract"]["spatial_metric"] = "unregistered"
    with pytest.raises(ValueError, match="asymptotic contract changed"):
        build_future_g3_flat_radial_matched_constraints_asymptotic_no_go_campaign(contract, ROOT)

    predecessor = copy.deepcopy(config)
    predecessor["bindings"]["predecessor"]["content_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="bound content hash mismatch"):
        build_future_g3_flat_radial_matched_constraints_asymptotic_no_go_campaign(predecessor, ROOT)

    source = copy.deepcopy(config)
    source["adapter_source"]["file_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="campaign source hash mismatch"):
        build_future_g3_flat_radial_matched_constraints_asymptotic_no_go_campaign(source, ROOT)

    eligibility = copy.deepcopy(config)
    eligibility["data_eligibility"]["observational_data_opened"] = True
    with pytest.raises(ValueError, match="eligibility is not fail-closed"):
        build_future_g3_flat_radial_matched_constraints_asymptotic_no_go_campaign(eligibility, ROOT)
