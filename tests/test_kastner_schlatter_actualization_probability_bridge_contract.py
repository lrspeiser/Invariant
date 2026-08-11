from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.kastner_schlatter_actualization_probability_bridge_contract import (
    EXPECTED_CLAIM_SEALS,
    EXPECTED_COUNTS,
    FIRST_BLOCKER,
    _validate_result,
    build_gate,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/kastner_schlatter_actualization_probability_bridge_contract.json"
ARTIFACT = ROOT / "runs/engine/kastner-schlatter-actualization-probability-bridge-contract.json"


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def rebuilt() -> dict[str, object]:
    return build_gate(CONFIG)


def test_exact_rebuild_counts_and_blocker(rebuilt: dict[str, object]) -> None:
    artifact = _load(ARTIFACT)
    assert rebuilt == artifact
    _validate_result(artifact)
    assert artifact["decision_counts"] == {"pass": 0, "reject": 0, "blocked": 2}
    assert artifact["gate_counts"] == EXPECTED_COUNTS
    assert artifact["first_blocker"] == FIRST_BLOCKER


def test_minimal_interface_has_three_primitive_source_registrations(
    rebuilt: dict[str, object],
) -> None:
    primitives = rebuilt["primitive_interface"]
    assert [row["primitive"] for row in primitives] == [
        "candidate_conditioned_history_probability_kernel",
        "measurable_locally_finite_count_map",
        "source_bound_countable_core_identity",
    ]
    assert [row["source_status"] for row in primitives] == [
        "absent",
        "partial_semantics_only",
        "absent",
    ]
    assert primitives[0]["signature"] == "Q:(g,phi)->Prob(H,Sigma_H)"
    assert primitives[1]["signature"] == "C:(H,Sigma_H)->(N_lf(W),Sigma_eval)"


def test_pushforward_probability_and_expectation_are_derived(
    rebuilt: dict[str, object],
) -> None:
    derived = rebuilt["compiler_derived_objects"]
    assert derived == [
        {
            "object": "pushforward_probability_law",
            "formula": "P_g_phi=C_*Q_g_phi on N_lf(W)",
            "derivation": "Q plus measurable C",
        },
        {
            "object": "counting_measure_expectation",
            "formula": "E_P[F(N)]=Integral_H F(C(h))*Q_g_phi(dh)",
            "derivation": "pushforward change of variables",
        },
    ]


def test_composition_theorem_is_sufficient_and_route_relative(
    rebuilt: dict[str, object],
) -> None:
    theorem = rebuilt["composition_theorem"]
    assert theorem["theorem_name"] == "minimal_actualization_probability_bridge_composition"
    assert len(theorem["premises"]) == 3
    assert len(theorem["derived_steps"]) == 3
    assert "jointly sufficient" in theorem["conclusion"]
    assert theorem["minimality"] == {
        "without_Q": "history expectation and pushforward probability are undefined",
        "without_C": "N(A), local finiteness, and pushforward law are undefined",
        "without_core_identity": "the law is defined but Poisson versus Cox remains unresolved",
    }
    assert "relative to the registered history-pushforward route" in theorem["scope_limit"]
    assert "could bypass histories but is also absent" in theorem["scope_limit"]


def test_identity_fixture_proves_schema_satisfiable_without_attribution(
    rebuilt: dict[str, object],
) -> None:
    fixture = rebuilt["exact_controls"]["compiler_identity_fixture_positive_control"]
    assert fixture["history_space"] == "H=N_lf(W)"
    assert fixture["history_kernel"] == "Q_g_phi=PRM(mu_g_phi)"
    assert fixture["count_map"] == "C=identity"
    assert fixture["countable_Laplace_core"] == "pass"
    assert fixture["source_or_QED_attribution"] is False
    assert fixture["purpose"] == "schema satisfiability only"


def test_each_missing_primitive_fails_at_its_exact_type_boundary(
    rebuilt: dict[str, object],
) -> None:
    controls = rebuilt["exact_controls"]
    assert controls["missing_Q_negative_control"]["failure"] == (
        "expectation and pushforward probability undefined"
    )
    assert controls["missing_C_negative_control"]["failure"] == (
        "set-indexed counts and pushforward law undefined"
    )
    assert controls["missing_certificate_negative_control"]["failure"] == (
        "same-mean Poisson and Cox pushforwards remain nonidentified"
    )
    assert controls["semantic_promotion_negative_control"]["rejected"] is True
    assert all(
        controls[key]["rejected"] is True
        for key in (
            "missing_Q_negative_control",
            "missing_C_negative_control",
            "missing_certificate_negative_control",
            "semantic_promotion_negative_control",
        )
    )


def test_both_candidates_have_interface_only_and_remain_blocked(
    rebuilt: dict[str, object],
) -> None:
    assert [(row["branch_id"], row["beta"]) for row in rebuilt["candidate_records"]] == [
        ("eq35_middle_h", "1/2"),
        ("eq35_printed_planck", "1/4"),
    ]
    for record in rebuilt["candidate_records"]:
        assert record["minimal_bridge_interface_registered_by_compiler"] is True
        assert record["primitive_source_registrations_complete"] == 0
        assert record["compiler_identity_fixture_pass"] is True
        assert record["paper_or_QED_bridge_complete"] is False
        assert record["candidate_action_selects_Poisson"] is False
        assert record["candidate_action_rejection_authorized"] is False
        assert record["candidate_decision"] == "blocked"


def test_lineage_and_overclaim_tampering_fail_closed(
    tmp_path: Path, rebuilt: dict[str, object]
) -> None:
    config = _load(CONFIG)
    tampered = copy.deepcopy(config)
    tampered["predecessors"]["source_type_audit"]["content_sha256"] = "0" * 64
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

    attributed = copy.deepcopy(rebuilt)
    attributed["claim_seals"]["compiler_identity_fixture_attributed_to_source"] = True
    attributed["content_sha256"] = None
    with pytest.raises(ValueError, match="seal changed"):
        _validate_result(attributed)

    rejected = copy.deepcopy(rebuilt)
    rejected["candidate_records"][0]["candidate_action_rejection_authorized"] = True
    rejected["content_sha256"] = None
    with pytest.raises(ValueError, match="candidate boundary changed"):
        _validate_result(rejected)


def test_source_bindings_are_exact_portable_and_sealed(rebuilt: dict[str, object]) -> None:
    bindings = rebuilt["source_bindings"]
    for label in (
        "source_type_audit",
        "countable_full_law_admission",
        "actualization_history_map",
        "canonical_probability_space",
        "candidate_action_completion",
        "qed_actualization_audit",
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
        ROOT
        / "src/sigma_theory_compiler/kastner_schlatter_actualization_probability_bridge_contract.py"
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
