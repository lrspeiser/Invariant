from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.g2_scalable_nonmaximal_positive_mass_audit import (
    NONMAXIMAL_CONTRACT,
    THEOREM_INTERFACE,
    _sha,
    apply_nonmaximal_positive_mass_theorem,
    build_g2_scalable_nonmaximal_positive_mass_audit,
)
from sigma_theory_compiler.promotion_orchestrator import ELIGIBILITY

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "g2_scalable_nonmaximal_positive_mass_audit.json"
ARTIFACT = ROOT / "runs" / "engine" / "g2-scalable-nonmaximal-positive-mass-audit.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def rebuilt() -> dict:
    return build_g2_scalable_nonmaximal_positive_mass_audit(_load(CONFIG), ROOT)


def test_committed_artifact_is_exact_deterministic_rebuild(rebuilt: dict) -> None:
    committed = _load(ARTIFACT)
    assert committed == rebuilt
    body = {key: value for key, value in committed.items() if key != "content_sha256"}
    assert committed["content_sha256"] == _sha(body)
    assert hashlib.sha256(ARTIFACT.read_bytes()).hexdigest() == (
        "075748654a48442219123a689df61d6f3b9dd83563105de5c43522224763be9c"
    )


def test_standard_theorem_interface_matches_every_nonmaximal_contract_premise(
    rebuilt: dict,
) -> None:
    application = rebuilt["theorem_application"]
    assert application["theorem_interface_sha256"] == _sha(THEOREM_INTERFACE)
    assert application["contract_sha256"] == _sha(NONMAXIMAL_CONTRACT)
    assert application["nonmaximal_K_allowed"] is True
    assert application["conclusion"] == ("E_ADM>=sqrt(delta_ij*P_ADM^i*P_ADM^j)>=0")
    assert application["machine_checked_theorem_proof_claimed"] is False
    assert NONMAXIMAL_CONTRACT["initial_slice"]["topology"] == (
        "smooth_connected_complete_orientable"
    )
    assert NONMAXIMAL_CONTRACT["initial_slice"]["inner_boundary"] == "empty"
    assert NONMAXIMAL_CONTRACT["constraint_domain"]["extrinsic_curvature"] == (
        "K_ij_not_restricted_to_K=0"
    )
    assert NONMAXIMAL_CONTRACT["constraint_domain"]["dominant_energy"] == ("mu>=sqrt(h_ij*J^i*J^j)")


def test_exact_two_scalable_candidates_close_the_only_predecessor_blocker(
    rebuilt: dict,
) -> None:
    assert rebuilt["candidate_count"] == 2
    assert rebuilt["decision_counts"] == {"pass": 2}
    assert rebuilt["general_nonmaximal_positive_mass_pass_count"] == 2
    assert rebuilt["full_formal_pass_count"] == 2
    expected = {
        "G3A-2f8983c88f504150381064f2": (
            "19f36a7c814ca11ace6de1270802a542872c35c27c7e64542eea672e16cbae88",
            "X_phi+(1/4)*X_phi^2",
        ),
        "G3A-58e59412e5fe77cd54caf863": (
            "9457ba1ff99ecfdabc08200dda3ff15b8656b025d106fe2c2cd4abd77a01c3b5",
            "X_phi+(1/8)*X_phi^2",
        ),
    }
    for record in rebuilt["candidate_records"]:
        assert (record["typed_action_ir_sha256"], record["G2"]) == expected[record["candidate_id"]]
        assert record["decision"] == "pass"
        assert record["previous_blocker_closed"] == (
            "hash_bound_general_nonmaximal_positive_mass_theorem"
        )
        assert set(record["gate_ledger"].values()) == {"pass"}
        assert record["candidate_dec_certificate"]["cell_subset_proof"] == (
            "[0,1/32]_is_subset_of_[0,1]"
        )
        assert record["candidate_dec_certificate"]["dominant_energy_status"] == "pass"


def test_pass_scope_does_not_overclaim_existence_evolution_stability_or_solar(
    rebuilt: dict,
) -> None:
    assert rebuilt["solar_bundle_count"] == 0
    for record in rebuilt["candidate_records"]:
        assert record["actual_initial_data_set_instantiated"] is False
        assert record["cell_preservation_or_global_evolution_proved"] is False
        assert record["nonlinear_asymptotic_stability_proved"] is False
        assert record["negative_total_energy_counterexample_found"] is False
        assert record["solar_bundle_generated"] is False
        assert (
            "satisfying_the_exact_registered_constraint_DEC_boundary_contract"
            in record["decision_scope"]
        )


def test_observation_dark_matter_redshift_and_paid_llm_seals_hold(rebuilt: dict) -> None:
    assert rebuilt["data_eligibility"] == ELIGIBILITY
    assert rebuilt["observational_data_opened"] is False
    assert rebuilt["paid_llm_spend_usd"] == 0.0
    assert all(not record["observational_data_opened"] for record in rebuilt["candidate_records"])


def test_theorem_and_complete_contract_tamper_fail_closed() -> None:
    theorem = copy.deepcopy(THEOREM_INTERFACE)
    theorem["premises"]["matter_condition"] = "local_DEC_only"
    with pytest.raises(ValueError, match="theorem interface changed"):
        apply_nonmaximal_positive_mass_theorem(theorem, NONMAXIMAL_CONTRACT)

    contract = copy.deepcopy(NONMAXIMAL_CONTRACT)
    contract["initial_slice"]["inner_boundary"] = "unspecified"
    with pytest.raises(ValueError, match="contract changed"):
        apply_nonmaximal_positive_mass_theorem(THEOREM_INTERFACE, contract)


def test_family_label_or_action_hash_cannot_substitute_for_exact_action() -> None:
    config = _load(CONFIG)
    family_only = copy.deepcopy(config)
    family_only["targets"][0]["typed_action_ir_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="action_sha256 binding changed"):
        build_g2_scalable_nonmaximal_positive_mass_audit(family_only, ROOT)

    wrong_formula = copy.deepcopy(config)
    wrong_formula["targets"][1]["G2"] = "X_phi"
    with pytest.raises(ValueError, match="exact G2 parameters changed"):
        build_g2_scalable_nonmaximal_positive_mass_audit(wrong_formula, ROOT)


def test_predecessor_hash_and_repository_path_escape_fail_closed(tmp_path: Path) -> None:
    config = _load(CONFIG)
    bad_source = copy.deepcopy(config)
    bad_source["adapter_source"]["file_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="adapter source file hash mismatch"):
        build_g2_scalable_nonmaximal_positive_mass_audit(bad_source, ROOT)

    bad_hash = copy.deepcopy(config)
    bad_hash["positive_mass_audit"]["content_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="content hash mismatch"):
        build_g2_scalable_nonmaximal_positive_mass_audit(bad_hash, ROOT)

    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    escaped = copy.deepcopy(config)
    escaped["scalable_export"] = {
        "path": str(outside),
        "file_sha256": hashlib.sha256(outside.read_bytes()).hexdigest(),
        "content_sha256": "0" * 64,
    }
    with pytest.raises(ValueError, match="path escapes repository"):
        build_g2_scalable_nonmaximal_positive_mass_audit(escaped, ROOT)
