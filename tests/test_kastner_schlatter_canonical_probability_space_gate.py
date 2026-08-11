from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.kastner_schlatter_canonical_probability_space_gate import (
    EXPECTED_COUNTS,
    FIRST_BLOCKER,
    _validate_result,
    build_gate,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/kastner_schlatter_canonical_probability_space_gate.json"
ARTIFACT = ROOT / "runs/engine/kastner-schlatter-canonical-probability-space-gate.json"


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def rebuilt() -> dict[str, object]:
    return build_gate(CONFIG)


def test_exact_rebuild_partition_counts_and_blocker(rebuilt: dict[str, object]) -> None:
    artifact = _load(ARTIFACT)
    assert rebuilt == artifact
    _validate_result(artifact)
    assert artifact["decision_counts"] == {"pass": 0, "reject": 0, "blocked": 2}
    assert artifact["gate_counts"] == EXPECTED_COUNTS
    assert artifact["first_blocker"] == FIRST_BLOCKER


def test_canonical_probability_space_and_filtration_are_explicit(
    rebuilt: dict[str, object],
) -> None:
    construction = rebuilt["canonical_conditional_construction"]
    probability = construction["probability_space"]
    filtration = construction["causal_filtration"]
    assert probability["sample_space"] == "Omega=N_lf(W)"
    assert "eta->eta(B)" in probability["sigma_algebra"]
    assert probability["coordinate_counting_measure"] == "N(eta,B)=eta(B)"
    assert filtration["right_continuous"] is True
    assert filtration["complete"] is True
    assert filtration["coordinate_measure_adapted"] is True
    assert construction["paper_or_QED_supplies_this_probability_space"] is False
    assert construction["candidate_action_selects_this_probability_space"] is False


def test_compensator_martingale_and_projective_contract_are_exact(
    rebuilt: dict[str, object],
) -> None:
    construction = rebuilt["canonical_conditional_construction"]
    martingale = construction["compensator_martingale"]
    projective = construction["projective_extension"]
    assert martingale["deterministic_compensator"] == "A(B)=mu_g_phi(B)"
    assert martingale["martingale"] == "M_t(h)=Integral_{W_t} h*(dN-dmu)"
    assert "independent increments" in martingale["proof_interface"]
    assert projective["consistency"] == "partition coarsening maps sums of component counts"
    assert "finite-partition laws" in projective["uniqueness"]


def test_exact_positive_and_same_action_negative_controls(rebuilt: dict[str, object]) -> None:
    controls = rebuilt["exact_controls"]
    assert controls["finite_partition_PGF_positive_control"] == {
        "cell_means": ["2", "3"],
        "PGF_points": ["1/2", "1/3"],
        "joint_exponent": "-3",
        "factorized_exponent": "-3",
        "pass": True,
    }
    assert controls["centered_increment_martingale_positive_control"][
        "conditional_centered_increment"
    ] == "0"
    assert controls["exponential_martingale_positive_control"]["conditional_product"] == "1"
    cox = controls["same_action_Cox_nonidentifiability"]
    assert cox["E_Z"] == "1"
    assert cox["E_Z_squared"] == "5/4"
    assert cox["same_conditional_mean_given_g_phi"] is True
    assert cox["Poisson_variance_at_mu_2"] == "2"
    assert cox["Cox_variance_at_mu_2"] == "3"
    assert cox["Poisson_second_factorial_moment_at_mu_2"] == "4"
    assert cox["Cox_second_factorial_moment_at_mu_2"] == "5"
    assert cox["deterministic_mu_compensator_identity_holds_for_Cox"] is False
    assert cox["action_selects_between_completions"] is False


def test_candidate_records_advance_math_only_and_remain_blocked(
    rebuilt: dict[str, object],
) -> None:
    assert [record["branch_id"] for record in rebuilt["candidate_records"]] == [
        "eq35_middle_h",
        "eq35_printed_planck",
    ]
    for record in rebuilt["candidate_records"]:
        assert record["compiler_canonical_probability_space"] == "pass"
        assert record["compiler_causal_filtration"] == "pass"
        assert record["compiler_compensator_martingale_identity"] == "pass"
        assert record["paper_or_QED_probability_space_derived"] is False
        assert record["candidate_action_selects_Poisson_completion"] is False
        assert record["same_action_Cox_completion_witness"] is True
        assert record["candidate_action_rejection_authorized"] is False
        assert record["candidate_decision"] == "blocked"


def test_lineage_attribution_and_result_tampering_fail_closed(
    tmp_path: Path, rebuilt: dict[str, object]
) -> None:
    config = _load(CONFIG)
    tampered = copy.deepcopy(config)
    tampered["predecessors"]["deterministic_compensator"]["content_sha256"] = "0" * 64
    path = tmp_path / "configs" / CONFIG.name
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="file hash mismatch|content hash mismatch"):
        build_gate(path)

    opened = copy.deepcopy(config)
    opened["seals"]["QED_actualization_derivation_opened"] = True
    path.write_text(json.dumps(opened), encoding="utf-8")
    with pytest.raises(ValueError, match="seal opened"):
        build_gate(path)

    attributed = copy.deepcopy(rebuilt)
    attributed["canonical_conditional_construction"][
        "paper_or_QED_supplies_this_probability_space"
    ] = True
    attributed.pop("content_sha256")
    with pytest.raises(ValueError, match="attribution or filtration overclaim"):
        _validate_result(attributed)

    selected = copy.deepcopy(rebuilt)
    selected["candidate_records"][0]["candidate_action_selects_Poisson_completion"] = True
    selected.pop("content_sha256")
    with pytest.raises(ValueError, match="candidate boundary changed"):
        _validate_result(selected)

    rejected = copy.deepcopy(rebuilt)
    rejected["candidate_records"][0]["candidate_action_rejection_authorized"] = True
    rejected.pop("content_sha256")
    with pytest.raises(ValueError, match="candidate boundary changed"):
        _validate_result(rejected)


def test_source_bindings_are_exact_portable_and_sealed(rebuilt: dict[str, object]) -> None:
    bindings = rebuilt["source_bindings"]
    for label in (
        "deterministic_compensator",
        "conditional_poisson_kernel",
        "actualization_history_map",
        "qed_actualization_audit",
        "positive_reparameterization",
        "candidate_action_completion",
        "config",
        "source",
        "test",
    ):
        binding = bindings[label]
        assert _file_sha(ROOT / binding["path"]) == binding["file_sha256"]
        assert not Path(binding["path"]).is_absolute()
    assert (
        bindings["primary_pdf_sha256"]
        == "c2f671293d07b21397e745da00a3ce1a2193c00da647a2ebf4147612b76c1780"
    )
    assert all("C:/" not in str(value) and "C:\\" not in str(value) for value in bindings.values())
    assert all(value is False for value in rebuilt["claim_seals"].values())
    assert all(value is False for value in rebuilt["data_seals"].values())
