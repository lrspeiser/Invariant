import hashlib
import json
from pathlib import Path

from sigma_theory_compiler.quartic_h7_resonant_remedy_campaign import (
    generic_h7_resonant_remedy_control,
    run_quartic_h7_resonant_remedy_campaign,
)

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "physics-language"
PATHS = (
    RUNS / "quartic-h7-paracomposition-topology-campaign" / "campaign.json",
    RUNS / "quartic-solved-source-c9-extension-campaign" / "campaign.json",
    RUNS / "quartic-full-source-jacobian-arithmetic-campaign" / "campaign.json",
    RUNS / "quartic-dyadic-localization-campaign" / "campaign.json",
    RUNS / "quartic-global-h7-energy-campaign" / "campaign.json",
)
CONFIG = (
    ROOT
    / "configs"
    / "backgrounds"
    / "quartic_h7_resonant_remedy_campaign.json"
)
ARTIFACT = RUNS / "quartic-h7-resonant-remedy-campaign" / "campaign.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _inputs() -> list[dict]:
    return [_load(path) for path in PATHS]


def _rehash(campaign: dict) -> None:
    body = {key: value for key, value in campaign.items() if key != "content_sha256"}
    campaign["content_sha256"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


def test_resonant_constant_h8_topology_and_negatives_are_exact() -> None:
    passed, control = generic_h7_resonant_remedy_control()
    assert passed
    resonant = control["resonant_Poincare_Plancherel_Young_bound"]
    assert resonant["balanced_partners"] == "7"
    assert resonant["Bernstein_L2_to_Linfinity_constant"] == (
        "2*sqrt(3)/(3*pi)"
    )
    assert resonant["instantiated"]
    h8 = control["conditional_H8_to_H7_topology"]
    assert h8["highest_required_solved_source_Frechet_order"] == 8
    assert h8["all_partition_topologies_compatible"]
    assert control["H8_not_controlled_by_H7"]["H7_scaling"] == "1"
    assert control["H8_not_controlled_by_H7"]["H8_scaling"] == "N"
    assert all(item["rejected"] for item in control["negative_controls"].values())


def test_resonance_closes_but_actual_high_low_and_global_h7_stay_open() -> None:
    result = run_quartic_h7_resonant_remedy_campaign(*_inputs(), _load(CONFIG))
    assert result["status"] == (
        "pass_all_12_resonant_H6xH7_operators_and_conditional_H8_"
        "remedies_actual_high_low_cancellation_fail_closed"
    )
    assert result["counts"] == {
        "selected": 12,
        "resonant_Fourier_operator_constants_instantiated": 12,
        "resonant_branches_closed": 12,
        "actual_high_low_cancellations_proved": 0,
        "conditional_H8_to_H7_remedies_proved": 12,
        "autonomous_H7_closures": 0,
        "autonomous_H8_closures": 0,
        "global_H7_closures": 0,
        "lifespans_proved": 0,
        "rejected": 0,
    }
    first = result["certificates"][0]
    assert first["resonant_branch"]["instantiated"]
    assert first["actual_high_low_cancellation_audit"]["status"] == (
        "unproved_fail_closed"
    )
    assert first["minimal_honest_remedy"]["proved_conditionally"]
    assert not first["minimal_honest_remedy"]["autonomous_H7_closure"]
    assert first["connection_to_B7_global_H7"]["resonant_branch_removed_from_B7"]
    assert not first["connection_to_B7_global_H7"]["B7_fully_replaced"]
    assert result == _load(ARTIFACT)


def test_provenance_and_false_cancellation_or_global_promotions_reject() -> None:
    inputs = _inputs()
    config = _load(CONFIG)

    corrupt = json.loads(json.dumps(inputs[0]))
    corrupt["upstream_sha256"]["dyadic_localization"] = "0" * 64
    _rehash(corrupt)
    result = run_quartic_h7_resonant_remedy_campaign(
        corrupt, *inputs[1:], config
    )
    assert result["status"] == "reject"
    assert "topology provenance mismatch" in result["errors"][0]

    wrong_order = dict(config)
    wrong_order["conditional_state_order"] = 7
    result = run_quartic_h7_resonant_remedy_campaign(*inputs, wrong_order)
    assert result["status"] == "reject"
    assert "unsupported resonant/remedy contract" in result["errors"][0]

    for policy in (
        "actual_high_low_cancellation_policy",
        "global_H7_policy",
        "lifespan_policy",
    ):
        false_promotion = dict(config)
        false_promotion[policy] = "pass"
        result = run_quartic_h7_resonant_remedy_campaign(
            *inputs, false_promotion
        )
        assert result["status"] == "reject"
