import hashlib
import json
from pathlib import Path

from sigma_theory_compiler.quartic_full_nonlinear_operator_remainder_campaign import (
    generic_full_operator_remainder_control,
    run_quartic_full_nonlinear_operator_remainder_campaign,
)

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "physics-language"
PATHS = (
    RUNS / "quartic-full-source-jacobian-arithmetic-campaign" / "campaign.json",
    RUNS / "quartic-solved-source-moser-campaign" / "campaign.json",
    RUNS / "quartic-global-h7-energy-campaign" / "campaign.json",
)
CONFIG = (
    ROOT
    / "configs"
    / "backgrounds"
    / "quartic_full_nonlinear_operator_remainder_campaign.json"
)
ARTIFACT = (
    RUNS / "quartic-full-nonlinear-operator-remainder-campaign" / "campaign.json"
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


def test_operator_taylor_bony_h7_and_c4_obstruction_controls_are_exact() -> None:
    passed, control = generic_full_operator_remainder_control()
    assert passed
    assert control["operator_Taylor_identities"]["quadratic_scalar_residual"] == "0"
    assert set(
        control["operator_Taylor_identities"][
            "orders_1_to_3_scalar_residuals"
        ].values()
    ) == {"0"}
    assert control["H7_product_algebra"]["algebra_constant"] == (
        "2*sqrt(21)/sqrt(pi)"
    )
    assert control["H7_product_algebra"]["total_bilinear_Q7_factor"] == (
        "8388608*sqrt(21)/sqrt(pi)"
    )
    assert control["Bony_partition"]["partition_exact"]
    assert control["C4_to_H7_obstruction"]["fifth_derivative_at_zero"] == "N"
    assert all(item["rejected"] for item in control["negative_controls"].values())


def test_full_operator_bounds_bypass_tensor_enumeration_but_keep_H7_fail_closed() -> None:
    result = run_quartic_full_nonlinear_operator_remainder_campaign(
        *_inputs(), _load(CONFIG)
    )
    assert result["status"] == (
        "pass_all_12_full_pointwise_C4_and_frozen_H7_operator_"
        "remainders_variable_H7_fail_closed"
    )
    assert result["counts"] == {
        "selected": 12,
        "full_DF_entrywise_arithmetic_certificates": 12,
        "full_pointwise_C4_operator_remainders_closed": 12,
        "full_frozen_H7_all_direction_remainders_closed": 12,
        "component_D2_D4_enumerations_bypassed_for_norm_bounds": 12,
        "variable_coefficient_H7_remainders_closed": 0,
        "global_H7_closures": 0,
        "lifespans_proved": 0,
        "rejected": 0,
    }
    first = result["certificates"][0]
    assert first["source_identity_and_basis_provenance"][
        "entrywise_DF_arithmetic_roots"
    ] == 1683
    assert first["global_operator_envelopes"]["D2_bilinear_operator_upper"] == (
        "2233882418"
    )
    assert first["full_variable_pointwise_remainder"]["closed"]
    assert first["full_frozen_H7_quadratic_remainder"]["closed"]
    assert first["full_frozen_H7_quadratic_remainder"][
        "all_direction_operator_bound_bypasses_component_tensor_enumeration"
    ]
    assert not first["global_variable_coefficient_H7_remainder"]["closed"]
    assert first["global_variable_coefficient_H7_remainder"][
        "minimum_unavailable_operator_orders"
    ] == [5, 6, 7]
    assert result == _load(ARTIFACT)


def test_hash_provenance_and_false_H7_promotion_reject() -> None:
    inputs = _inputs()
    config = _load(CONFIG)

    corrupt = json.loads(json.dumps(inputs[0]))
    corrupt["certificates"][0]["total_entries_entrywise_arithmetic"] = 1682
    result = run_quartic_full_nonlinear_operator_remainder_campaign(
        corrupt, inputs[1], inputs[2], config
    )
    assert result["status"] == "reject"
    assert "content hash mismatch" in result["errors"][0]

    wrong_link = json.loads(json.dumps(inputs[2]))
    wrong_link["upstream_sha256"]["source_jacobian"] = "0" * 64
    _rehash(wrong_link)
    result = run_quartic_full_nonlinear_operator_remainder_campaign(
        inputs[0], inputs[1], wrong_link, config
    )
    assert result["status"] == "reject"
    assert "provenance mismatch" in result["errors"][0]

    for policy in (
        "variable_coefficient_H7_policy",
        "global_H7_policy",
        "lifespan_policy",
    ):
        false_promotion = dict(config)
        false_promotion[policy] = "pass"
        result = run_quartic_full_nonlinear_operator_remainder_campaign(
            *inputs, false_promotion
        )
        assert result["status"] == "reject"
        assert "unsupported full operator-remainder contract" in result["errors"][0]
