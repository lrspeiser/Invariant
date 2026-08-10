import hashlib
import json
from pathlib import Path

from sigma_theory_compiler.quartic_solved_source_c9_extension_campaign import (
    generic_c9_extension_control,
    run_quartic_solved_source_c9_extension_campaign,
)

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "physics-language"
PATHS = (
    RUNS / "quartic-full-source-jacobian-arithmetic-campaign" / "campaign.json",
    RUNS / "quartic-solved-source-moser-campaign" / "campaign.json",
    RUNS / "quartic-global-h7-energy-campaign" / "campaign.json",
    RUNS / "quartic-full-nonlinear-operator-remainder-campaign" / "campaign.json",
)
CONFIG = (
    ROOT
    / "configs"
    / "backgrounds"
    / "quartic_solved_source_c9_extension_campaign.json"
)
ARTIFACT = RUNS / "quartic-solved-source-c9-extension-campaign" / "campaign.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _inputs() -> list[dict]:
    return [_load(path) for path in PATHS]


def _rehash(campaign: dict) -> None:
    body = {key: value for key, value in campaign.items() if key != "content_sha256"}
    campaign["content_sha256"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


def test_c9_rational_jet_inverse_chain_and_negatives_are_exact() -> None:
    passed, control = generic_c9_extension_control()
    assert passed
    assert all(
        int(item["upper_square_minus_integer"].split("/")[0]) > 0
        for item in control["strict_radical_uppers"].values()
    )
    assert control["inverse_product_coefficients"] == ["1", *("0" for _ in range(9))]
    assert set(
        control["quadratic_time_block_chain_residuals_orders_1_to_9"].values()
    ) == {"0"}
    assert all(item["rejected"] for item in control["negative_controls"].values())


def test_all_candidates_extend_solved_source_operator_envelopes_through_C9() -> None:
    result = run_quartic_solved_source_c9_extension_campaign(
        *_inputs(), _load(CONFIG)
    )
    assert result["status"] == (
        "pass_all_12_solved_source_C9_operator_envelopes_H7_topology_fail_closed"
    )
    assert result["counts"] == {
        "selected": 12,
        "C5_C9_solved_source_operator_extensions": 12,
        "operator_order_gaps_closed": 12,
        "variable_coefficient_H7_paracomposition_theorems_closed": 0,
        "global_H7_closures": 0,
        "lifespans_proved": 0,
        "rejected": 0,
    }
    first = result["certificates"][0]
    assert first["orders_newly_closed"] == [5, 6, 7, 8, 9]
    assert first["orders_cumulatively_closed"] == list(range(10))
    assert all(item["dominates"] for item in first["C4_backward_dominance"].values())
    assert first["minimal_direct_coefficient_order_for_H7"][
        "operator_order_gap_closed"
    ]
    assert not first["variable_coefficient_H7_paracomposition_theorem"]["closed"]
    assert not first["global_H7_differential_inequality_closed"]
    assert result == _load(ARTIFACT)


def test_provenance_order_and_false_H7_promotion_reject() -> None:
    inputs = _inputs()
    config = _load(CONFIG)

    corrupt = json.loads(json.dumps(inputs[3]))
    corrupt["upstream_sha256"]["solved_source_C4"] = "0" * 64
    _rehash(corrupt)
    result = run_quartic_solved_source_c9_extension_campaign(
        *inputs[:3], corrupt, config
    )
    assert result["status"] == "reject"
    assert "prior C4 provenance mismatch" in result["errors"][0]

    wrong_order = dict(config)
    wrong_order["required_Frechet_order"] = 8
    result = run_quartic_solved_source_c9_extension_campaign(*inputs, wrong_order)
    assert result["status"] == "reject"
    assert "unsupported C9 extension contract" in result["errors"][0]

    for policy in ("paracomposition_policy", "global_H7_policy", "lifespan_policy"):
        false_promotion = dict(config)
        false_promotion[policy] = "pass"
        result = run_quartic_solved_source_c9_extension_campaign(
            *inputs, false_promotion
        )
        assert result["status"] == "reject"
