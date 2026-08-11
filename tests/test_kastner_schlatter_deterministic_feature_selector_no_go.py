from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.kastner_schlatter_deterministic_feature_selector_no_go import (
    EXPECTED_CLAIM_SEALS,
    EXPECTED_COUNTS,
    FIRST_BLOCKER,
    _validate_result,
    build_gate,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/kastner_schlatter_deterministic_feature_selector_no_go.json"
ARTIFACT = ROOT / "runs/engine/kastner-schlatter-deterministic-feature-selector-no-go.json"


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def rebuilt() -> dict[str, object]:
    return build_gate(CONFIG)


def test_exact_rebuild_counts_and_first_blocker(rebuilt: dict[str, object]) -> None:
    artifact = _load(ARTIFACT)
    assert rebuilt == artifact
    _validate_result(artifact)
    assert artifact["decision_counts"] == {"pass": 0, "reject": 0, "blocked": 2}
    assert artifact["gate_counts"] == EXPECTED_COUNTS
    assert artifact["first_blocker"] == FIRST_BLOCKER


def test_factorization_no_go_is_exact_and_narrow(rebuilt: dict[str, object]) -> None:
    theorem = rebuilt["factorization_no_go"]
    assert theorem["theorem_name"] == "registered_deterministic_feature_factorization_no_go"
    assert "S(P_Poisson)=s(D(P_Poisson))" in theorem["proof"]
    assert "only selectors factoring through D" in theorem["scope_limit"]
    assert "new QED stochastic kernel" in theorem["scope_limit"]


def test_same_feature_poisson_cox_witness_is_exact(rebuilt: dict[str, object]) -> None:
    witness = rebuilt["exact_controls"]["same_feature_distinct_law_witness"]
    assert witness["mixing_moments"] == {"E_Z": "1", "E_Z_squared": "5/4"}
    assert witness["shared_registered_deterministic_features"] == {
        "candidate_action": True,
        "Euler_Lagrange_equations": True,
        "intensity_field_mu_g_phi": True,
        "conditional_mean_measure_given_g_phi": "2",
    }
    separation = witness["law_separation"]
    assert separation["Poisson_variance"] == "2"
    assert separation["Cox_variance"] == "3"
    assert separation["Poisson_second_factorial_moment"] == "4"
    assert separation["Cox_second_factorial_moment"] == "5"
    assert separation["Poisson_second_factorial_cumulant"] == "0"
    assert separation["Cox_second_factorial_cumulant"] == "1"
    assert separation["Poisson_void_probability"] == "exp(-2)"
    assert separation["Cox_void_probability"] == "exp(-2)*cosh(1)"
    assert separation["laws_are_distinct"] is True


def test_candidate_actions_advance_no_go_only_and_remain_blocked(
    rebuilt: dict[str, object],
) -> None:
    assert [(row["branch_id"], row["beta"]) for row in rebuilt["candidate_records"]] == [
        ("eq35_middle_h", "1/2"),
        ("eq35_printed_planck", "1/4"),
    ]
    for record in rebuilt["candidate_records"]:
        assert record["deterministic_feature_fiber_no_go"] == "pass"
        assert record["exact_same_feature_Poisson_Cox_pair"] is True
        assert record["registered_stochastic_feature_outside_fiber"] is False
        assert record["paper_or_QED_selector_derived"] is False
        assert record["candidate_action_selects_Poisson"] is False
        assert record["candidate_action_rejection_authorized"] is False
        assert record["candidate_decision"] == "blocked"


def test_escape_contract_distinguishes_pair_separation_from_characterization(
    rebuilt: dict[str, object],
) -> None:
    contract = rebuilt["minimal_escape_contract"]
    assert (
        "zero versus one separates"
        in contract["witness_separating_but_not_universally_characterizing_option"]
    )
    assert (
        "does not characterize"
        in contract["witness_separating_but_not_universally_characterizing_option"]
    )
    assert len(contract["universally_sufficient_options"]) == 4
    assert contract["registered_options_closed"] == 0
    assert contract["compiler_canonical_probability_space_is_not_source_selection"] is True


def test_lineage_and_claim_tampering_fail_closed(
    tmp_path: Path, rebuilt: dict[str, object]
) -> None:
    config = _load(CONFIG)
    tampered = copy.deepcopy(config)
    tampered["predecessors"]["canonical_probability_space"]["content_sha256"] = "0" * 64
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

    selected = copy.deepcopy(rebuilt)
    selected["candidate_records"][0]["candidate_action_selects_Poisson"] = True
    selected["content_sha256"] = None
    with pytest.raises(ValueError, match="candidate boundary changed"):
        _validate_result(selected)

    overreached = copy.deepcopy(rebuilt)
    overreached["claim_seals"]["single_cell_witness_claimed_to_characterize_Poisson"] = True
    overreached["content_sha256"] = None
    with pytest.raises(ValueError, match="seal changed"):
        _validate_result(overreached)

    rejected = copy.deepcopy(rebuilt)
    rejected["candidate_records"][0]["candidate_action_rejection_authorized"] = True
    rejected["content_sha256"] = None
    with pytest.raises(ValueError, match="candidate boundary changed"):
        _validate_result(rejected)


def test_source_bindings_are_exact_portable_and_all_seals_closed(
    rebuilt: dict[str, object],
) -> None:
    bindings = rebuilt["source_bindings"]
    for label in (
        "canonical_probability_space",
        "deterministic_compensator",
        "candidate_action_completion",
        "equation_graph",
        "qed_actualization_audit",
        "actualization_history_map",
        "config",
        "source",
        "test",
    ):
        binding = bindings[label]
        assert _file_sha(ROOT / binding["path"]) == binding["file_sha256"]
        assert not Path(binding["path"]).is_absolute()
    assert bindings["primary_pdf_sha256"] == (
        "c2f671293d07b21397e745da00a3ce1a2193c00da647a2ebf4147612b76c1780"
    )
    assert rebuilt["claim_seals"] == EXPECTED_CLAIM_SEALS
    assert all(value is False for value in rebuilt["claim_seals"].values())
    assert all(value is False for value in rebuilt["data_seals"].values())
    assert all("C:/" not in str(value) and "C:\\" not in str(value) for value in bindings.values())


def test_source_has_no_runtime_data_or_process_surface() -> None:
    source = (
        ROOT / "src/sigma_theory_compiler/kastner_schlatter_deterministic_feature_selector_no_go.py"
    ).read_text(encoding="utf-8")
    lowered = source.lower()
    for forbidden in (
        "sqlite",
        "subprocess",
        "socket",
        "requests",
        "urllib",
        "cupy",
        "torch",
        "os.kill",
        "popen",
    ):
        assert forbidden not in lowered
