import hashlib
import json
from pathlib import Path

from sigma_theory_compiler.quartic_frequency_localized_evolution_campaign import (
    generic_frequency_localized_evolution_control,
    run_quartic_frequency_localized_evolution_campaign,
)

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "physics-language"
PATHS = (
    RUNS / "quartic-annular-k55-c6-campaign" / "campaign.json",
    RUNS / "quartic-bounded-frequency-defect-campaign" / "campaign.json",
    RUNS / "quartic-dyadic-localization-campaign" / "campaign.json",
    RUNS / "quartic-time-atom-budget-campaign" / "campaign.json",
    RUNS / "quartic-first-order-reduction-campaign" / "campaign.json",
    RUNS / "quartic-evolution-symbol-campaign" / "campaign.json",
    RUNS / "quartic-positive-quantization-campaign" / "campaign.json",
    RUNS / "quartic-anti-wick-composition-campaign" / "campaign.json",
)
CONFIG = ROOT / "configs" / "backgrounds" / "quartic_frequency_localized_evolution_campaign.json"
ARTIFACT = RUNS / "quartic-frequency-localized-evolution-campaign" / "campaign.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _inputs() -> list[dict]:
    return [_load(path) for path in PATHS]


def test_energy_young_low_projector_and_omission_controls_are_exact() -> None:
    passed, control = generic_frequency_localized_evolution_control()
    assert passed
    assert control["energy_differentiation"]["scalar_residual"] == "0"
    assert control["neighbor_coupling_Young"]["positive_residual"] == "(a - b)**2"
    assert sp_positive(control["fixed_low_projector_commutator"]["explicit_kappa_low"])
    assert all(
        item["rejected"] for item in control["negative_controls"].values()
    )


def sp_positive(expression: str) -> bool:
    import sympy as sp

    return bool(sp.sympify(expression).is_positive)


def test_all_candidates_receive_coupled_high_and_partial_low_inequalities() -> None:
    result = run_quartic_frequency_localized_evolution_campaign(
        *_inputs(), _load(CONFIG)
    )
    assert result["status"] == (
        "pass_all_12_frequency_localized_principal_shell_inequalities_"
        "sources_and_global_sum_fail_closed"
    )
    assert result["counts"] == {
        "selected": 12,
        "high_shell_principal_time_projection_inequalities_passed": 12,
        "finite_low_partial_inequalities_passed": 12,
        "complete_shell_inequalities_closed": 0,
        "global_H7_dyadic_sums_applied": 0,
        "rejected": 0,
    }
    assert all(
        item["per_shell_principal_time_projection_inequality_certified"]
        and item["finite_low_principal_partial_inequality_certified"]
        and not item["complete_shell_inequality_closed"]
        and not item["global_H7_dyadic_sum_applied"]
        and item["high_shell_j_ge_7"]["time_K_included"]
        and item["high_shell_j_ge_7"]["principal_composition_included"]
        and item["high_shell_j_ge_7"]["projection_commutator_included"]
        and not item["finite_physical_low_frequencies"][
            "low_anti_wick_composition_remainder_closed"
        ]
        for item in result["certificates"]
    )
    assert result == _load(ARTIFACT)


def test_omitted_or_corrupt_inputs_and_false_sum_policy_reject() -> None:
    inputs = _inputs()
    config = _load(CONFIG)
    omitted = list(inputs)
    omitted[0] = {}
    result = run_quartic_frequency_localized_evolution_campaign(
        *omitted, config
    )
    assert result["status"] == "reject"
    assert "annular_C6 prerequisite status mismatch" in result["errors"][0]

    false_sum = dict(config)
    false_sum["global_dyadic_sum_policy"] = "pass"
    result = run_quartic_frequency_localized_evolution_campaign(
        *inputs, false_sum
    )
    assert result["status"] == "reject"

    corrupt = json.loads(json.dumps(inputs[2]))
    corrupt["upstream_sha256"]["first_order"] = "0" * 64
    body = {key: value for key, value in corrupt.items() if key != "content_sha256"}
    corrupt["content_sha256"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()
    inputs[2] = corrupt
    result = run_quartic_frequency_localized_evolution_campaign(
        *inputs, config
    )
    assert result["status"] == "reject"
    assert "dyadic physical-pencil provenance mismatch" in result["errors"][0]
