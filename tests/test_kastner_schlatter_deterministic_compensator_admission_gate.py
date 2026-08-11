from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.kastner_schlatter_deterministic_compensator_admission_gate import (
    FIRST_BLOCKER,
    SECOND_BLOCKER,
    _validate_result,
    build_gate,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/kastner_schlatter_deterministic_compensator_admission_gate.json"
ARTIFACT = ROOT / "runs/engine/kastner-schlatter-deterministic-compensator-admission-gate.json"


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
        "evidence_obligations": 10,
        "evidence_closed_by_compiler_hypotheses": 2,
        "evidence_absent": 8,
        "positive_candidate_mean_measures": 2,
        "compiler_compensator_theorem_interfaces": 2,
        "registered_causal_filtrations": 0,
        "action_or_QED_compensator_identities": 0,
        "exact_same_action_alternative_law_witnesses": 2,
        "paper_or_QED_Poisson_derivation_pass": 0,
        "candidate_action_reject": 0,
        "theory_ontology_observational_pass": 0,
    }
    assert artifact["first_blocker"] == FIRST_BLOCKER
    assert artifact["secondary_blockers"][0] == SECOND_BLOCKER


def test_compensator_characterization_is_exact_and_conditional(rebuilt: dict[str, object]) -> None:
    theorem = rebuilt["deterministic_compensator_Poisson_characterization"]
    assert theorem["premises"]["deterministic_measure"] == (
        "mu(dx)=q0*exp(phi(x))*dVol_g(x), conditional on fixed (g,phi)"
    )
    assert theorem["exponential_martingale"].startswith("Z_t(f)=exp")
    assert theorem["consequences"][0].startswith("conditional Poisson counts")
    assert "independent increments" in theorem["consequences"][1]
    assert theorem["candidate_action_or_paper_supplies_compensator_identity"] is False


def test_evidence_ledger_separates_mean_and_probability_law(rebuilt: dict[str, object]) -> None:
    ledger = rebuilt["evidence_gap_ledger"]
    statuses = [item["status"] for item in ledger]
    assert statuses.count("closed_by_compiler_action") == 1
    assert statuses.count("closed_by_compiler_kernel") == 1
    assert statuses.count("absent") == 8
    obligations = {item["obligation"]: item["status"] for item in ledger}
    assert obligations["positive_candidate_mean_measure"] == "closed_by_compiler_action"
    assert obligations["QED_predictable_compensator_identity"] == "absent"


def test_same_action_poisson_cox_nonidentifiability_is_exact(rebuilt: dict[str, object]) -> None:
    controls = rebuilt["exact_controls"]
    laplace = controls["two_cell_Laplace_positive_control"]
    assert laplace["Laplace_functional"] == laplace["factorized_cell_product"] == "exp(-3)"
    witness = controls["same_action_Poisson_Cox_nonidentifiability"]
    assert witness["same_unconditional_mean_measure"] is True
    assert witness["Poisson_variance_on_B"] == "mu_B"
    assert witness["Cox_variance_on_B"] == "mu_B+mu_B^2/4"
    assert witness["same_deterministic_candidate_action"] is True
    assert witness["action_selects_between_completions"] is False
    random = controls["random_compensator_negative_control"]
    assert random["deterministic_compensator_claim"] is False


def test_candidate_bindings_and_claim_seals_remain_closed(rebuilt: dict[str, object]) -> None:
    controls = rebuilt["exact_controls"]
    assert (
        controls["action_variable_type_negative_control"][
            "variational_derivation_of_compensator_identity"
        ]
        is False
    )
    assert controls["paper_rate_negative_control"]["rejected"] is True
    assert [record["branch_id"] for record in rebuilt["candidate_records"]] == [
        "eq35_middle_h",
        "eq35_printed_planck",
    ]
    for record in rebuilt["candidate_records"]:
        assert record["positive_mean_measure_on_regular_finite_phi_patch"] is True
        assert record["compiler_compensator_theorem_interface"] == "pass"
        assert record["action_registered_event_variables"] is False
        assert record["action_or_QED_compensator_identity_derived"] is False
        assert record["same_action_alternative_law_witness"] is True
        assert record["candidate_action_rejection_authorized"] is False


def test_lineage_config_and_result_tampering_fail_closed(
    tmp_path: Path, rebuilt: dict[str, object]
) -> None:
    config = _load(CONFIG)
    tampered = copy.deepcopy(config)
    tampered["predecessors"]["candidate_action_completion"]["content_sha256"] = "0" * 64
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

    variables = copy.deepcopy(rebuilt)
    variables["candidate_records"][0]["action_registered_event_variables"] = True
    variables.pop("content_sha256")
    with pytest.raises(ValueError, match="action-variable overclaim"):
        _validate_result(variables)

    derived = copy.deepcopy(rebuilt)
    derived["candidate_records"][0]["action_or_QED_compensator_identity_derived"] = True
    derived.pop("content_sha256")
    with pytest.raises(ValueError, match="derivation overclaim"):
        _validate_result(derived)

    rejected = copy.deepcopy(rebuilt)
    rejected["candidate_records"][0]["candidate_action_rejection_authorized"] = True
    rejected.pop("content_sha256")
    with pytest.raises(ValueError, match="overreached"):
        _validate_result(rejected)


def test_source_bindings_are_exact_portable_and_sealed(rebuilt: dict[str, object]) -> None:
    bindings = rebuilt["source_bindings"]
    for label in (
        "qed_actualization_audit",
        "positive_reparameterization",
        "candidate_action_completion",
        "candidate_formal_admission",
        "conditional_poisson_kernel",
        "paper_intake",
        "equation_graph",
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
