import hashlib
import json
from pathlib import Path

import sympy as sp

from sigma_theory_compiler.quartic_finite_low_operator_campaign import (
    generic_finite_low_operator_control,
    run_quartic_finite_low_operator_campaign,
)

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "physics-language"
PATHS = (
    RUNS / "quartic-frequency-localized-evolution-campaign" / "campaign.json",
    RUNS / "quartic-dyadic-localization-campaign" / "campaign.json",
    RUNS / "quartic-evolution-symbol-campaign" / "campaign.json",
    RUNS / "quartic-low-frequency-symbol-extension-campaign" / "campaign.json",
    RUNS / "quartic-positive-quantization-campaign" / "campaign.json",
    RUNS / "quartic-bounded-frequency-defect-campaign" / "campaign.json",
    RUNS / "quartic-solved-source-moser-campaign" / "campaign.json",
    RUNS / "quartic-component-jacobian-contract-campaign" / "campaign.json",
    RUNS / "quartic-unspecialized-source-jacobian-campaign" / "campaign.json",
    RUNS / "quartic-lower-source-remainder-campaign" / "campaign.json",
)
CONFIG = ROOT / "configs" / "backgrounds" / "quartic_finite_low_operator_campaign.json"
ARTIFACT = RUNS / "quartic-finite-low-operator-campaign" / "campaign.json"


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


def test_direct_low_operator_and_omission_witnesses_are_exact() -> None:
    passed, control = generic_finite_low_operator_control()
    assert passed
    assert control["coefficient_Cauchy_control"]["sum_of_squares_residual"] == "0"
    template = sp.sympify(
        control["sandwiched_principal_operator"]["exact_template"],
        locals={
            name: sp.Symbol(name, positive=True, finite=True)
            for name in ("A0", "Lambda_op", "R_low")
        },
    )
    assert template.is_positive
    assert all(item["rejected"] for item in control["negative_controls"].values())


def test_all_candidates_close_low_principal_but_not_source_or_global_sum() -> None:
    result = run_quartic_finite_low_operator_campaign(*_inputs(), _load(CONFIG))
    assert result["status"] == (
        "pass_all_12_finite_low_anti_wick_principal_operators_"
        "lower_sources_fail_closed"
    )
    assert result["counts"] == {
        "selected": 12,
        "finite_low_principal_operators_closed": 12,
        "localized_lower_sources_closed": 0,
        "complete_low_energy_inequalities_closed": 0,
        "global_H7_sums_applied": 0,
        "lifespans_proved": 0,
        "rejected": 0,
    }
    assert all(
        item["finite_low_principal_operator_closed"]
        and item["finite_low_anti_wick_principal"][
            "previous_low_composition_gate_closed"
        ]
        and item["finite_low_anti_wick_principal"][
            "compact_pointwise_defect_subsumed_not_added"
        ]
        and not item["finite_low_complete_energy_inequality_closed"]
        and not item["localized_lower_source_audit"][
            "whole_space_L2_source_bound_certified"
        ]
        and not item["global_H7_dyadic_sum_applied"]
        and not item["nonlinear_lifespan_proved"]
        for item in result["certificates"]
    )
    assert result == _load(ARTIFACT)


def test_omitted_corrupt_provenance_and_false_source_policy_reject() -> None:
    inputs = _inputs()
    config = _load(CONFIG)

    omitted = list(inputs)
    omitted[0] = {}
    result = run_quartic_finite_low_operator_campaign(*omitted, config)
    assert result["status"] == "reject"
    assert "frequency prerequisite status mismatch" in result["errors"][0]

    false_source = dict(config)
    false_source["lower_source_policy"] = "pass"
    result = run_quartic_finite_low_operator_campaign(*inputs, false_source)
    assert result["status"] == "reject"

    corrupt = json.loads(json.dumps(inputs[9]))
    corrupt["upstream_sha256"]["unspecialized_source_jacobian"] = "0" * 64
    _rehash(corrupt)
    inputs[9] = corrupt
    result = run_quartic_finite_low_operator_campaign(*inputs, config)
    assert result["status"] == "reject"
    assert "lower-source-to-source_jacobian provenance mismatch" in result["errors"][0]
