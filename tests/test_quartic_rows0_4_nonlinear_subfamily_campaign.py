import hashlib
import json
from pathlib import Path

from sigma_theory_compiler.quartic_rows0_4_nonlinear_subfamily_campaign import (
    generic_rows0_4_nonlinear_control,
    run_quartic_rows0_4_nonlinear_subfamily_campaign,
)

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "physics-language"
NAMES = (
    "quartic-row0-component-remainder-campaign",
    "quartic-row0-arithmetic-expansion-campaign",
    "quartic-row1-arithmetic-expansion-campaign",
    "quartic-row2-arithmetic-expansion-campaign",
    "quartic-row3-arithmetic-expansion-campaign",
    "quartic-row4-arithmetic-expansion-campaign",
    "quartic-solved-source-moser-campaign",
    "quartic-coordinate-jet-tube-campaign",
    "quartic-global-h7-energy-campaign",
)
PATHS = tuple(RUNS / name / "campaign.json" for name in NAMES)
CONFIG = (
    ROOT
    / "configs"
    / "backgrounds"
    / "quartic_rows0_4_nonlinear_subfamily_campaign.json"
)
ARTIFACT = (
    RUNS / "quartic-rows0-4-nonlinear-subfamily-campaign" / "campaign.json"
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _inputs() -> list[dict]:
    return [_load(path) for path in PATHS]


def _rehash(campaign: dict) -> None:
    body = {key: value for key, value in campaign.items() if key != "content_sha256"}
    campaign["content_sha256"] = hashlib.sha256(
        json.dumps(
            body, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode()
    ).hexdigest()


def test_inverse_determinant_h7_algebra_and_negatives_are_exact() -> None:
    passed, control = generic_rows0_4_nonlinear_control()
    assert passed
    assert control["time_block"]["sigma_min_lower"] == "1/5"
    assert control["time_block"]["determinant_absolute_lower"] == (
        "1/48828125"
    )
    assert control["H7_product_algebra"]["algebra_constant"] == (
        "2*sqrt(21)/sqrt(pi)"
    )
    assert control["dyadic_conversion"]["total_bilinear_Q7_factor"] == (
        "8388608*sqrt(21)/sqrt(pi)"
    )
    assert control["energy_absorption"]["residual"] == "0"
    assert all(item["rejected"] for item in control["negative_controls"].values())


def test_rows0_4_frozen_mixed_CB_is_quantitative_but_full_H7_is_open() -> None:
    result = run_quartic_rows0_4_nonlinear_subfamily_campaign(
        *_inputs(), _load(CONFIG)
    )
    assert result["status"] == (
        "pass_all_12_rows0_4_frozen_mixed_bilinear_subfamilies_"
        "full_nonlinear_H7_fail_closed"
    )
    assert result["counts"] == {
        "selected": 12,
        "sigma_min_bounds_certified": 12,
        "determinant_lower_bounds_certified": 12,
        "selected_mixed_roots_quantitatively_bounded_per_candidate": 30,
        "frozen_mixed_C_B_contributions_certified": 12,
        "full_row0_nonlinear_remainders_closed": 0,
        "global_H7_closures": 0,
        "lifespans_proved": 0,
        "rejected": 0,
    }
    assert [item["output_row"] for item in result["row_packet_audits"]] == (
        list(range(5))
    )
    first = result["certificates"][0]
    assert first["quantitative_time_block"]["sigma_min_lower"] == "1/5"
    assert first["quantitative_time_block"]["determinant_absolute_lower"] == (
        "1/48828125"
    )
    assert first["reachable_family_bounds"][
        "solved_source_F0_to_F4_component_upper_ceilings"
    ]["2"] == 2233882418
    assert first["frozen_mixed_bilinear_subfamily"][
        "selected_mixed_roots_quantitatively_bounded"
    ] == 30
    assert all(
        item["frozen_mixed_bilinear_subfamily"]["certified"]
        and not item["full_row0_nonlinear_remainder_closed"]
        and not item["rows0_4_full_nonlinear_remainder_closed"]
        and not item["full_11_row_remainder_closed"]
        and not item["global_H7_differential_inequality_closed"]
        and not item["nonlinear_lifespan_proved"]
        for item in result["certificates"]
    )
    assert result == _load(ARTIFACT)


def test_omission_corruption_inverse_and_false_closure_policies_reject() -> None:
    inputs = _inputs()
    config = _load(CONFIG)

    omitted = list(inputs)
    omitted[4] = {}
    result = run_quartic_rows0_4_nonlinear_subfamily_campaign(*omitted, config)
    assert result["status"] == "reject"
    assert "campaign prerequisite status mismatch" in result["errors"][0]

    corrupt = json.loads(json.dumps(inputs[4]))
    corrupt["upstream_sha256"]["row3_arithmetic"] = "0" * 64
    _rehash(corrupt)
    changed = list(inputs)
    changed[4] = corrupt
    result = run_quartic_rows0_4_nonlinear_subfamily_campaign(*changed, config)
    assert result["status"] == "reject"
    assert "row4 arithmetic provenance mismatch" in result["errors"][0]

    corrupt_inverse = json.loads(json.dumps(inputs[6]))
    corrupt_inverse["certificates"][0]["inverse_time_block_2_norm_upper"] = "6"
    _rehash(corrupt_inverse)
    changed = list(inputs)
    changed[6] = corrupt_inverse
    result = run_quartic_rows0_4_nonlinear_subfamily_campaign(*changed, config)
    assert result["status"] == "reject"
    assert "quantitative inverse time-block bound is absent" in result["errors"][0]

    for policy in (
        "variable_coefficient_policy",
        "full_row0_remainder_policy",
        "global_H7_policy",
        "lifespan_policy",
    ):
        false_closure = dict(config)
        false_closure[policy] = "pass"
        result = run_quartic_rows0_4_nonlinear_subfamily_campaign(
            *inputs, false_closure
        )
        assert result["status"] == "reject"
