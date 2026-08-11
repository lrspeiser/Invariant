from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.kastner_schlatter_conditional_poisson_kernel_completion_gate import (
    FIRST_BLOCKER,
    _validate_result,
    build_gate,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/kastner_schlatter_conditional_poisson_kernel_completion_gate.json"
ARTIFACT = ROOT / "runs/engine/kastner-schlatter-conditional-poisson-kernel-completion-gate.json"


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def rebuilt() -> dict[str, object]:
    return build_gate(CONFIG)


def test_exact_rebuild_partition_and_counts(rebuilt: dict[str, object]) -> None:
    artifact = _load(ARTIFACT)
    assert rebuilt == artifact
    _validate_result(artifact)
    assert artifact["decision_counts"] == {"pass": 0, "reject": 0, "blocked": 2}
    assert artifact["gate_counts"] == {
        "candidate_actions": 2,
        "compiler_authored_conditional_kernels": 2,
        "conditional_Laplace_selector_pass": 2,
        "conditional_independent_increment_pass": 2,
        "conditional_Mecke_identity_pass": 2,
        "diffeomorphism_covariance_pass": 2,
        "stationary_Poisson_PMF_recovery_pass": 2,
        "deterministic_action_derivation_pass": 0,
        "paper_or_QED_actualization_derivation_pass": 0,
        "candidate_action_reject": 0,
        "theory_ontology_observational_pass": 0,
    }
    assert artifact["first_blocker"] == FIRST_BLOCKER


def test_conditional_kernel_closes_all_three_selector_routes(rebuilt: dict[str, object]) -> None:
    contract = rebuilt["conditional_Poisson_kernel_contract"]
    assert contract["kernel"] == "K[(g,phi),dN]=PRM(mu_g_phi)(dN)"
    assert contract["Laplace_functional"]["uniquely_selects_conditional_Poisson_law"] is True
    assert contract["joint_disjoint_set_PGF"]["independent_increment_factorization"] is True
    assert contract["Mecke_identity"]["characterizes_same_conditional_Poisson_law"] is True
    assert contract["diffeomorphism_covariance"]["pass"] is True
    assert (
        contract["existence_uniqueness_scope"]["global_singular_spacetime_or_divergent_phi_covered"]
        is False
    )


def test_each_branch_is_bound_but_not_attributed(rebuilt: dict[str, object]) -> None:
    assert [record["branch_id"] for record in rebuilt["candidate_records"]] == [
        "eq35_middle_h",
        "eq35_printed_planck",
    ]
    for record in rebuilt["candidate_records"]:
        ledger = record["gate_ledger"]
        assert record["compiler_authored_conditional_kernel"] is True
        assert record["paper_or_QED_derived"] is False
        assert record["action_derived"] is False
        assert ledger["conditional_Laplace_functional_selector"] == "pass"
        assert ledger["conditional_independent_increment_family"] == "pass"
        assert ledger["conditional_Mecke_identity"] == "pass"
        assert ledger["derivation_from_deterministic_candidate_action"] == "blocked"
        assert ledger["paper_or_QED_actualization_derivation"] == "blocked"
        assert record["candidate_action_rejection_authorized"] is False


def test_exact_positive_and_negative_controls(rebuilt: dict[str, object]) -> None:
    controls = rebuilt["deterministic_controls"]
    assert controls["stationary_PMF_positive_control"]["pass"] is True
    factorized = controls["two_cell_factorization_positive_control"]
    assert factorized["joint_PGF"] == factorized["product_of_marginal_PGFs"] == "exp(-3)"
    dependent = controls["dependent_equal_cell_negative_control"]
    assert dependent["joint_PGF_at_half_half"] == "exp(-3/2)"
    assert dependent["required_independent_PGF_at_half_half"] == "exp(-2)"
    assert dependent["rejected"] is True
    cox = controls["Cox_same_mean_negative_control"]
    assert cox["Poisson_second_factorial_moment"] == "4"
    assert cox["Cox_second_factorial_moment"] == "5"
    assert cox["rejected_by_Laplace_Mecke_selector"] is True
    assert (
        controls["random_field_marginalization_negative_control"][
            "infer_unconditional_Poisson_from_conditional_kernel"
        ]
        is False
    )
    assert controls["nonpositive_intensity_negative_control"]["rejected"] is True
    assert controls["attribution_negative_control"]["rejected"] is True


def test_attribution_and_data_seals_remain_closed(rebuilt: dict[str, object]) -> None:
    assert all(value is False for value in rebuilt["claim_seals"].values())
    assert all(value is False for value in rebuilt["data_seals"].values())


def test_lineage_config_and_result_tampering_fail_closed(
    tmp_path: Path, rebuilt: dict[str, object]
) -> None:
    config = _load(CONFIG)
    tampered = copy.deepcopy(config)
    tampered["predecessor"]["content_sha256"] = "0" * 64
    path = tmp_path / "configs" / CONFIG.name
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="file hash mismatch|content hash mismatch"):
        build_gate(path)

    opened = copy.deepcopy(config)
    opened["seals"]["observations_opened"] = True
    path.write_text(json.dumps(opened), encoding="utf-8")
    with pytest.raises(ValueError, match="seal opened"):
        build_gate(path)

    attributed = copy.deepcopy(rebuilt)
    attributed["candidate_records"][0]["paper_or_QED_derived"] = True
    attributed.pop("content_sha256")
    with pytest.raises(ValueError, match="attribution changed"):
        _validate_result(attributed)

    action_overclaim = copy.deepcopy(rebuilt)
    action_overclaim["candidate_records"][0]["action_derived"] = True
    action_overclaim.pop("content_sha256")
    with pytest.raises(ValueError, match="action derivation overclaim"):
        _validate_result(action_overclaim)

    rejected = copy.deepcopy(rebuilt)
    rejected["candidate_records"][0]["candidate_action_rejection_authorized"] = True
    rejected.pop("content_sha256")
    with pytest.raises(ValueError, match="overreached"):
        _validate_result(rejected)


def test_source_bindings_are_exact_and_portable(rebuilt: dict[str, object]) -> None:
    bindings = rebuilt["source_bindings"]
    for label in ("config", "source", "test"):
        binding = bindings[label]
        assert _file_sha(ROOT / binding["path"]) == binding["file_sha256"]
        assert not Path(binding["path"]).is_absolute()
    assert all("C:/" not in str(value) and "C:\\" not in str(value) for value in bindings.values())
