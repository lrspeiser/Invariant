from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.kastner_schlatter_candidate_action_completion import (
    FIRST_BLOCKER,
    _validate_result,
    build_completion,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/kastner_schlatter_candidate_action_completion.json"
ARTIFACT = ROOT / "runs/engine/kastner-schlatter-candidate-action-completion.json"


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def rebuilt() -> dict[str, object]:
    return build_completion(CONFIG)


def test_exact_rebuild_and_partition(rebuilt: dict[str, object]) -> None:
    artifact = _load(ARTIFACT)
    assert rebuilt == artifact
    _validate_result(artifact)
    assert artifact["counts"] == {
        "normalization_branches": 2,
        "complete_local_deterministic_action_hypotheses": 2,
        "conditional_exact_eq35_branch_matches": 2,
        "paper_derived_actions": 0,
        "normalization_branches_selected_as_fact": 0,
        "observational_or_theory_passes": 0,
    }
    assert artifact["first_blocker"] == FIRST_BLOCKER


def test_exact_eq35_normalization_branches(rebuilt: dict[str, object]) -> None:
    middle, printed = rebuilt["completion_hypotheses"]
    assert middle["beta"] == "1/2"
    assert middle["matching"]["derived_lambda"] == "4*pi*G*h*q0/c**3"
    assert middle["matching"]["derived_lambda_planck_units"] == "8*pi**2*l_P**2*q0"
    assert printed["beta"] == "1/4"
    assert printed["matching"]["derived_lambda"] == "2*pi*G*h*q0/c**3"
    assert printed["matching"]["derived_lambda_planck_units"] == "4*pi**2*l_P**2*q0"
    assert all(item["matching"]["exact_coefficient_match"] for item in (middle, printed))
    assert not any(item["matching"]["normalization_selected_as_fact"] for item in (middle, printed))


def test_variational_noether_and_dimension_contracts(rebuilt: dict[str, object]) -> None:
    assert rebuilt["dimensions"]["all_declared_terms_dimensionally_closed"] is True
    for branch in rebuilt["completion_hypotheses"]:
        action = branch["candidate_action"]
        assert action["local_deterministic_action_complete"] is True
        assert "S_GHY" in action["boundary"]
        assert branch["euler_lagrange"]["intensity"] == "B_q*Box(q)-A_q*(q-q0)=0"
        assert branch["noether_bianchi"]["on_shell_covariant_conservation"] is True
        assert branch["noether_bianchi"]["stationary_solution_consistent"] is True


def test_deterministic_positive_and_negative_controls(rebuilt: dict[str, object]) -> None:
    controls = rebuilt["deterministic_controls"]
    assert controls["middle_branch_coefficient_residual"] == "0"
    assert controls["printed_branch_coefficient_residual"] == "0"
    assert controls["full_pressure_beta_one_vs_middle_residual"] != "0"
    assert controls["full_pressure_beta_one_vs_printed_residual"] != "0"
    assert controls["off_shell_nonconstant_intensity_bianchi_residual_generically_zero"] is False
    assert controls["negative_controls_rejected"] is True


def test_stochastic_law_and_paper_attribution_fail_closed(rebuilt: dict[str, object]) -> None:
    for branch in rebuilt["completion_hypotheses"]:
        assert branch["paper_authorship_or_derivation"] is False
        stochastic = branch["conditional_stochastic_completion"]
        assert stochastic["derived_from_local_action"] is False
        assert stochastic["derived_from_QED_actualization"] is False
        assert "Poisson" in stochastic["law"]
    assert all(value is False for value in rebuilt["claim_seals"].values())
    assert all(value is False for value in rebuilt["data_seals"].values())


def test_predecessor_and_config_tamper_controls(tmp_path: Path) -> None:
    config = _load(CONFIG)
    tampered = copy.deepcopy(config)
    tampered["predecessors"]["equation_graph"]["content_sha256"] = "0" * 64
    path = tmp_path / "configs" / CONFIG.name
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="file hash mismatch|content hash mismatch"):
        build_completion(path)

    tampered = copy.deepcopy(config)
    tampered["seals"]["paper_action_claim_allowed"] = True
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="seals changed"):
        build_completion(path)

    tampered = copy.deepcopy(config)
    tampered["branches"][0]["target_lambda"] = "8*pi*G*h*q0/c**3"
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="branch coefficients changed"):
        build_completion(path)


def test_result_tamper_controls(rebuilt: dict[str, object]) -> None:
    attributed = copy.deepcopy(rebuilt)
    attributed["completion_hypotheses"][0]["paper_authorship_or_derivation"] = True
    attributed.pop("content_sha256")
    with pytest.raises(ValueError, match="falsely attributed"):
        _validate_result(attributed)

    incomplete = copy.deepcopy(rebuilt)
    incomplete["completion_hypotheses"][0]["candidate_action"][
        "local_deterministic_action_complete"
    ] = False
    incomplete.pop("content_sha256")
    with pytest.raises(ValueError, match="incomplete local"):
        _validate_result(incomplete)


def test_source_bindings_are_portable_and_exact(rebuilt: dict[str, object]) -> None:
    bindings = rebuilt["source_bindings"]
    assert bindings["config"]["file_sha256"] == _file_sha(CONFIG)
    assert bindings["source"]["file_sha256"] == _file_sha(
        ROOT / bindings["source"]["path"]
    )
    assert bindings["test"]["file_sha256"] == _file_sha(ROOT / bindings["test"]["path"])
    assert all("C:/" not in str(item) and "C:\\" not in str(item) for item in bindings.values())
