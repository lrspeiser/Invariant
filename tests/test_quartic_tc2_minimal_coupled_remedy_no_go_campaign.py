import hashlib
import json
from pathlib import Path

from sigma_theory_compiler.quartic_tc2_minimal_coupled_remedy_no_go_campaign import (
    generic_tc2_minimal_coupled_remedy_no_go_control,
    run_quartic_tc2_minimal_coupled_remedy_no_go_campaign,
)

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "physics-language"
PATHS = (
    RUNS / "quartic-tc2-symmetrizer-no-go-campaign" / "campaign.json",
    RUNS / "quartic-two-channel-induced-operator-campaign" / "campaign.json",
    RUNS / "quartic-full-symmetrizer-moser-campaign" / "campaign.json",
    RUNS / "quartic-component-jacobian-contract-campaign" / "campaign.json",
)
CONFIG = (
    ROOT
    / "configs"
    / "backgrounds"
    / "quartic_tc2_minimal_coupled_remedy_no_go_campaign.json"
)
ARTIFACT = (
    RUNS / "quartic-tc2-minimal-coupled-remedy-no-go-campaign" / "campaign.json"
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _inputs() -> list[dict]:
    return [_load(path) for path in PATHS]


def _rehash(campaign: dict) -> None:
    body = {key: value for key, value in campaign.items() if key != "content_sha256"}
    campaign["content_sha256"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


def test_linearized_equation_range_constraints_and_negatives_are_exact() -> None:
    passed, control = generic_tc2_minimal_coupled_remedy_no_go_control()
    assert passed
    assert control["linearized_symmetrizer_equation"][
        "no_solution_in_declared_deltaK_class"
    ]
    assert control["Hermiticity_and_positivity"]["preserved_lower_margin"] == (
        "lambda_K55/2"
    )
    assert control["reciprocal_block_range_obstruction"][
        "right_covector_span_dimension"
    ] == 2
    assert not control["state_to_jet_constraint_obstruction"][
        "constraint_preserving_members_with_nonzero_TC2"
    ]
    assert all(item["rejected"] for item in control["negative_controls"].values())


def test_all_candidates_eliminate_minimal_coupled_class_but_not_tc2() -> None:
    result = run_quartic_tc2_minimal_coupled_remedy_no_go_campaign(
        *_inputs(), _load(CONFIG)
    )
    assert result["status"] == (
        "pass_all_12_minimal_coupled_deltaK_same_high_state_no_gos_"
        "TC2_global_H7_fail_closed"
    )
    assert result["counts"] == {
        "selected": 12,
        "spectral_deltaK_no_gos": 12,
        "same_high_state_reciprocal_no_gos": 12,
        "fixed_q_w_constraint_no_gos": 12,
        "legitimate_coupled_remedies": 0,
        "TC2_closures": 0,
        "B7_branches_closed": 0,
        "global_H7_closures": 0,
        "lifespans_proved": 0,
        "rejected": 0,
    }
    first = result["certificates"][0]
    assert not first["linearized_symmetrizer_audit"]["solution_in_class"]
    assert not first["additional_state_correction_audit"][
        "reciprocal_block_realizable_in_class"
    ]
    assert not first["state_to_jet_constraint_audit"]["constraint_preserved"]
    assert first["connection_to_B7_global_H7"][
        "minimal_coupled_ansatz_eliminated"
    ]
    assert not first["closure_ledger"]["TC2_closed"]
    assert result == _load(ARTIFACT)


def test_hash_tamper_expanded_ansatz_and_false_promotions_reject() -> None:
    inputs = _inputs()
    config = _load(CONFIG)

    corrupt = json.loads(json.dumps(inputs[0]))
    corrupt["upstream_sha256"]["induced_operator"] = "0" * 64
    _rehash(corrupt)
    result = run_quartic_tc2_minimal_coupled_remedy_no_go_campaign(
        corrupt, *inputs[1:], config
    )
    assert result["status"] == "reject"
    assert "upstream provenance mismatch" in result["errors"][0]

    expanded = dict(config)
    expanded["additional_state_class"] = "two_high_covectors_with_q_w_lift"
    result = run_quartic_tc2_minimal_coupled_remedy_no_go_campaign(
        *inputs, expanded
    )
    assert result["status"] == "reject"
    assert "unsupported minimal coupled contract" in result["errors"][0]

    for policy in ("constraint_policy", "global_H7_policy", "lifespan_policy"):
        false_promotion = dict(config)
        false_promotion[policy] = "pass"
        result = run_quartic_tc2_minimal_coupled_remedy_no_go_campaign(
            *inputs, false_promotion
        )
        assert result["status"] == "reject"
